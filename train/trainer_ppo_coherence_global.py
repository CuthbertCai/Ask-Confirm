"""
PPO algorithm
Reference: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/
"""
import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn.functional as F
from time import time
import logging

from utils.utils import *
from models.environment_coherence_global_model import EnvCoherenceGlobalModel
from models.policy_model import MLPActorCritic


class PPOBuffer(object):
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantage of state-action pairs
    """

    def __init__(self, config, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.cfg = config
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.logit_buf = np.zeros(combined_shape(size, self.cfg.n_categories), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, logit):
        """
        Append one timestep of agent-environment interaction to the buffer
        :param obs:
        :param act:
        :param rew:
        :param val:
        :param logp:
        :return:
        """
        assert self.ptr < self.max_size
        if self.cfg.cuda:
            obs = obs.cpu().numpy()
        else:
            obs = obs.numpy()
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.logit_buf[self.ptr] = logit
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        :param last_val:
        :return:
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        print('rew: ', rews)
        print('vals: ', vals)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        print('adv_buf: ', self.adv_buf[path_slice])

        # the next line computes reward-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        print('ret_buf: ', self.ret_buf[path_slice])

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        :return:
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf,
                    act=self.act_buf,
                    ret=self.ret_buf,
                    adv=self.adv_buf,
                    logp=self.logp_buf,
                    logit=self.logit_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class PPOTrainer(object):
    def __init__(self, config, split='train'):
        self.cfg = config

        # init env and actor-critic
        self.env = EnvCoherenceGlobalModel(config, split)
        self.ac = MLPActorCritic(config)
        if self.cfg.cuda:
            self.ac = self.ac.cuda()

        self.word_stat = np.load(self.cfg.data_dir + '/caches/vg_word_stat.npy')

        # count parameters
        pi_counts = count_var(self.ac.pi)
        v_counts = count_var(self.ac.v)
        logging.info('Number of Actor-Critic parameters: {}'.format(pi_counts + v_counts))

        # init PPOBuffer
        self.buf = PPOBuffer(config, self.cfg.n_feature_dim + self.cfg.n_categories, self.cfg.ppo_num_actions,
                             self.cfg.ppo_update_steps, self.cfg.ppo_gamma, self.cfg.ppo_lambda)

        # init criterion
        self.criterion = nn.BCEWithLogitsLoss()

        # init optimizer
        self.sup_optimizer = optim.Adam(self.ac.pi.parameters(), lr=self.cfg.ppo_sup_lr)
        self.pi_optimizer = optim.Adam(self.ac.pi.parameters(), lr=self.cfg.ppo_pi_lr)
        self.v_optimizer = optim.Adam(self.ac.v.parameters(), lr=self.cfg.ppo_v_lr)
        if self.cfg.ppo_pretrained is not None:
            self.load_pretrained_net(self.cfg.ppo_pretrained)

    def compute_loss_sup(self, data):
        obs, logits = data['obs'], data['logit']
        if self.cfg.cuda:
            obs, logits = obs.cuda(), logits.cuda()
        prob = self.ac.pi.get_prob(obs)
        criterion = nn.MSELoss()
        loss = criterion(prob, logits)
        return self.cfg.ppo_coef_logit * loss

    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        act = act.type(torch.int32)
        if self.cfg.cuda:
            obs, act, adv, logp_old = obs.cuda(), act.cuda(), adv.cuda(), logp_old.cuda()

        pi, logp = self.ac.pi(obs, act)
        logp = logp.transpose(0, 1)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.cfg.ppo_clip_ratio, 1 + self.cfg.ppo_clip_ratio) * adv.unsqueeze(
            1).repeat(1, ratio.shape[1])
        ent = pi.entropy().mean()
        loss_pi = - self.cfg.ppo_coef_ratio * (torch.min(ratio * adv.unsqueeze(1).repeat(1, ratio.shape[1]),
                                                         clip_adv)).mean() - self.cfg.ppo_coef_ent * ent
        # useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = ent.item()
        clipped = ratio.gt(1 + self.cfg.ppo_clip_ratio) | ratio.lt(1 - self.cfg.ppo_clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        if self.cfg.cuda:
            obs, ret = obs.cuda(), ret.cuda()
        value = self.ac.v(obs)
        return self.cfg.ppo_coef_value * ((value - ret) ** 2).mean()

    def train_epoch(self):
        data = self.buf.get()
        pi_loss_old, pi_info_old = self.compute_loss_pi(data)
        pi_loss_old = pi_loss_old.item()
        v_loss_old = self.compute_loss_v(data).item()

        # train policy with multiple steps of supervised learning
        for i in range(self.cfg.ppo_train_sup_iters):
            self.sup_optimizer.zero_grad()
            sup_loss = self.compute_loss_sup(data)
            sup_loss.backward()
            self.sup_optimizer.step()
            if i % 10 == 0:
                logging.info('PPO PI update at step {} Sup Loss: {:.6f}'.format(i, sup_loss.item()))

        # train policy with multiple steps of gradient descent
        for i in range(self.cfg.ppo_train_pi_iters):
            self.pi_optimizer.zero_grad()
            pi_loss, pi_info = self.compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * self.cfg.ppo_target_kl:
                logging.info('Early stopping at step {} due to reaching max kl.'.format(i))
                break
            pi_loss.backward()
            self.pi_optimizer.step()
            if i % 10 == 0:
                logging.info('PPO PI update at step {} PI Loss: {:.6f}, KL: {:.6f}'.format(i, pi_loss.item(), kl))

        # train value with multiple steps of gradient descent
        for i in range(self.cfg.ppo_train_v_iters):
            self.v_optimizer.zero_grad()
            v_loss = self.compute_loss_v(data)
            v_loss.backward()
            self.v_optimizer.step()
            if i % 10 == 0:
                logging.info('PPO Value update at step {} Value Loss: {:.6f}'.format(i, v_loss.item()))

        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info_old['cf']
        logging.info(
            'Loss PI: {:.6f}, Loss V: {:.6f}, KL: {:.6f}, Entropy: {:.6f}, ClipFrac: {:.6f} '
            'DeltaLossPi: {:.6f}, DeltaLossV: {:.6f}'.format(pi_loss_old, v_loss_old, kl, ent, cf,
                                                             pi_loss.item() - pi_loss_old,
                                                             v_loss.item() - v_loss_old))

    def get_start_obs(self):
        index = random.randint(0, len(self.env.loaddb) - 1)
        scene = self.env.db.scenedb[index]
        logging.info('Image %s' % scene['image_index'])
        all_meta_regions = [scene['regions'][x] for x in sorted(list(scene['regions'].keys()))]
        all_captions = [x['caption'] for x in all_meta_regions]
        all_captions = np.random.permutation(all_captions)
        # cap_idx = random.randint(0, len(all_captions) - 1)
        # txt = all_captions[cap_idx]
        txt = list(all_captions[:self.cfg.train_turns])
        self.env.reset(txt, index)
        if self.cfg.vg_img_feature is not None and self.cfg.vg_img_logits is not None:
            r, inds, rank = self.env.retrieve_precomp(*self.env.tokenize())
        else:
            r, inds, rank = self.env.retrieve(*self.env.tokenize())
        logits_dis = torch.from_numpy(self.env.vg_all_logits_dis).unsqueeze(0)
        o = torch.cat([torch.mean(self.env.txt_feat, dim=0, keepdim=True),logits_dis.type_as(self.env.txt_feat)], dim=1)
        return r, o, inds

    def train(self):
        start_time = time()
        timestep = 0

        # main loop: collect experience in env and update/log each epoch
        for epoch in range(self.cfg.ppo_epochs):
            ep_ret, ep_len = 0, 0
            _, o, _ = self.get_start_obs()
            # r_previous = 1 / self.cfg.n_categories
            for t in range(self.cfg.max_turns):
                if self.cfg.cuda:
                    o = o.cuda()
                a, v, logp = self.ac.step(o)
                # _, filter_a = filter_actions(a, self.env.logit)

                next_o, r, d, _, _ = self.env.step(a)
                logging.info(
                    'reward now: {:.6f}'.format(r))
                ep_ret += r
                ep_len += 1

                txt = self.env.txt
                word_inds = []
                for tx in txt:
                    tokens = [w for w in word_tokenize(tx)]
                    word_inds.extend([self.env.db.lang_vocab(w) for w in tokens])
                word_inds = list(set(word_inds))
                logit = np.sum(self.word_stat[word_inds], axis=0)
                logit = logit / np.sum(logit)

                # save and log
                self.buf.store(o, a, r, v, logp, logit)
                timestep += 1

                # update obs
                o = next_o

                timeout = t == self.cfg.max_turns - 1
                logging.info('turn: {}'.format(t))

                if timeout:
                    # if self.cfg.cuda:
                    #     o = o.cuda()
                    # a, v, logp = self.ac.step(o)
                    # dis = self.ac.pi.get_prob(o).squeeze(1).detach()
                    # _, filter_a = filter_actions(a, self.env.logit)
                    # _, r, d, _ = self.env.step(a, dis)
                    # logging.info(
                    #     'reward previous: {:.6f}, reward now: {:.6f}'.format(r_previous, r))
                    # r_delta = r - r_previous
                    # self.buf.store(o, a, r_delta, v, logp)
                    # ep_ret += r_delta
                    # ep_len += 1
                    # timestep += 1
                    _, v, _ = self.ac.step(o)
                    logging.info('Start a finish path')
                    self.buf.finish_path(v)

                start_update = timestep == self.cfg.ppo_update_steps
                if start_update:
                    timestep = 0
                    # self.buf.finish_path(v)
                    logging.info('Start a update')
                    self.train_epoch()
                    logging.info(
                        'Value: {:.3f}, NowEpisodeReward: {:.3f}, NowEpisodeLen: {}'.format(v, ep_ret, ep_len))
                # if d:
                #     break

                # _, o, _, logit = self.get_start_obs()
                # self.ac.pi.init_hidden()

            if (epoch % self.cfg.ppo_save_freq == 0) or (epoch == self.cfg.ppo_epochs - 1):
                self.save_checkpoint(epoch, self.cfg.exp_name)

            logging.info('Episode: {}, TotalInteract: {}, Time: {:.3f}'.format(epoch, (epoch + 1) * self.cfg.max_turns,
                                                                               time() - start_time))

    def test(self):
        self.ac.eval()
        all_retrieve_inds = []
        db_length = len(self.env.db.scenedb)
        # db_length = 500
        for idx in range(db_length):
            retrieve_inds = []
            # get test text
            test_scene = self.env.db.scenedb[idx]
            logging.info('Image %s' % test_scene['image_index'])
            all_meta_regions = [test_scene['regions'][x] for x in sorted(list(test_scene['regions'].keys()))]
            all_captions = [x['caption'] for x in all_meta_regions]
            txt = all_captions[:self.cfg.test_turns]
            self.env.reset(txt, idx)
            if self.cfg.vg_img_feature is not None and self.cfg.vg_img_logits is not None:
                _, inds, rank = self.env.retrieve_precomp(*self.env.tokenize())
            else:
                _, inds, rank = self.env.retrieve(*self.env.tokenize())
            logits_dis = torch.from_numpy(self.env.vg_all_logits_dis).unsqueeze(0)
            o = torch.cat([torch.mean(self.env.txt_feat, dim=0, keepdim=True), logits_dis.type_as(self.env.txt_feat)], dim=1)
            retrieve_inds.append(inds)
            for t in range(self.cfg.max_turns):
                if rank[0] < self.cfg.ppo_stop_rank:
                    break
                if self.cfg.cuda:
                    o = o.cuda()
                # a, v, logp = self.ac.step(o)
                dis = self.ac.pi.get_prob(o).squeeze(1).detach()

                a = torch.argsort(dis, descending=True).squeeze().cpu().numpy()[t * self.cfg.ppo_num_actions: (t+1) * self.cfg.ppo_num_actions]
                # _, filter_a = filter_actions(a, self.env.logit)

                next_o, r, d, inds, logit = self.env.step(a)
                retrieve_inds.append(inds)

                if d:
                    break

                # update obs
                o = next_o
            all_retrieve_inds.append(retrieve_inds)
            logging.info('Get %d th retrieve result' % idx)

        ranks = []
        for idx in range(len(all_retrieve_inds)):
            inds = np.array(all_retrieve_inds[idx])
            rank = np.where(inds == idx)[1]
            ranks.append(rank)

        for t in range(self.cfg.max_turns + 1):
            logging.info("Get %d th turn result" % t)
            res = []
            for idx, rank in enumerate(ranks):
                if t < len(rank):
                    res.append(rank[t])
                else:
                    res.append(rank[-1])
            res = np.array(res)
            r1 = 100.0 * len(np.where(res < 1)[0]) / len(res)
            r5 = 100.0 * len(np.where(res < 5)[0]) / len(res)
            r10 = 100.0 * len(np.where(res < 10)[0]) / len(res)
            r20 = 100.0 * len(np.where(res < 20)[0]) / len(res)
            r50 = 100.0 * len(np.where(res < 50)[0]) / len(res)
            r100 = 100.0 * len(np.where(res < 100)[0]) / len(res)
            medr = np.floor(np.median(res)) + 1
            meanr = res.mean() + 1

            logging.info(
                "Text to image: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f" % (r1, r5, r10, r20, r50, r100, medr, meanr))

        # all_retrieve_inds = np.array(all_retrieve_inds)
        #
        # ranks = np.zeros((len(all_retrieve_inds), self.cfg.max_turns + 1))
        # for idx in range(len(all_retrieve_inds)):
        #     inds = all_retrieve_inds[idx]
        #     rank = np.where(inds == idx)[1]
        #     ranks[idx] = rank
        #
        # for t in range(self.cfg.max_turns + 1):
        #     logging.info("Get %d th turn result" % t)
        #     rank = ranks[:, t]
        #     r1 = 100.0 * len(np.where(rank < 1)[0]) / len(rank)
        #     r5 = 100.0 * len(np.where(rank < 5)[0]) / len(rank)
        #     r10 = 100.0 * len(np.where(rank < 10)[0]) / len(rank)
        #     r20 = 100.0 * len(np.where(rank < 20)[0]) / len(rank)
        #     r100 = 100.0 * len(np.where(rank < 100)[0]) / len(rank)
        #     medr = np.floor(np.median(rank)) + 1
        #     meanr = rank.mean() + 1
        #
        #     logging.info("Text to image: %.1f %.1f %.1f %.1f %.1f %.1f %.1f" % (r1, r5, r10, r20, r100, medr, meanr))

    def save_checkpoint(self, epoch, model_name):
        print('Saving checkpoint...')
        checkpoint_dir = osp.join(self.cfg.model_dir, 'snapshots')
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        states = {
            'epoch': epoch,
            'state_dict': self.ac.state_dict(),
            'pi_optimizer': self.pi_optimizer.state_dict(),
            'v_optimizer': self.v_optimizer.state_dict()
        }
        torch.save(states, osp.join(checkpoint_dir, str(epoch) + '.pkl'))

    def load_pretrained_net(self, pretrained_path):
        assert (osp.exists(pretrained_path))
        states = torch.load(pretrained_path)
        self.ac.load_state_dict(states['state_dict'], strict=True)
        self.pi_optimizer.load_state_dict(states['pi_optimizer'])
        self.v_optimizer.load_state_dict(states['v_optimizer'])
        self.epoch = states['epoch']