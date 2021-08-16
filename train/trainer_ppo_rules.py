import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn.functional as F
from time import time
from torch.distributions.categorical import Categorical
from scipy.special import softmax
import logging
from collections import defaultdict

from utils.utils import *
from models.environment_rules_model import EnvRulesModel


class PPOTrainer(object):
    def __init__(self, config, split='train'):
        self.cfg = config

        # init env and actor-critic
        self.env = EnvRulesModel(config, split)

        # init optimizer
        if self.cfg.ppo_pretrained is not None:
            self.load_pretrained_net(self.cfg.ppo_pretrained)

        if self.cfg.ppo_rule == 'coherence':
            self.stat_coherence()
        if self.cfg.ppo_rule == 'word_coherence':
            self.word_stat = np.load(self.cfg.data_dir + '/caches/vg_word_stat.npy')

    def stat_coherence(self):
        vg_img_logits = self.env.vg_img_logits
        obj_pred_row, obj_pred_col = np.where(vg_img_logits > 0.9)
        logits_coherence = np.zeros((self.cfg.n_categories, self.cfg.n_categories))

        idx2objs = {}
        for i, idx in enumerate(obj_pred_row):
            if idx not in idx2objs.keys():
                idx2objs[idx] = [obj_pred_col[i]]
            else:
                idx2objs[idx].append(obj_pred_col[i])

        for _, objs in idx2objs.items():
            for obj in objs:
                other = list(set(objs) - {obj})
                logits_coherence[obj][other] += 1

        self.logits_coherence = logits_coherence / (np.sum(logits_coherence, axis=-1) + 1e-6)

    def get_action(self, turn):
        if self.cfg.ppo_rule == 'attr_freq':
            # top_img_logits = self.env.vg_img_logits[inds[:100]]
            # top_img_logits_dis = softmax(np.sum(top_img_logits, axis=0))
            logits_dis = self.env.vg_all_logits_dis
            dis = Categorical(logits=torch.from_numpy(logits_dis))
        elif self.cfg.ppo_rule == 'random':
            logits = torch.ones(self.cfg.n_categories)
            dis = Categorical(logits=logits)
        elif self.cfg.ppo_rule == 'query_attr_sim':
            attr_sim = self.env.attr_sim
            dis = Categorical(logits=attr_sim)
        elif self.cfg.ppo_rule == 'coherence':
            attr_sim = self.env.attr_sim
            top_attr = torch.argmax(attr_sim, dim=-1).item()
            logit_coherence = self.logits_coherence[top_attr]
            dis = Categorical(logits=torch.from_numpy(logit_coherence))
        elif self.cfg.ppo_rule == 'word_coherence':
            txt = self.env.txt
            word_inds = []
            for t in txt:
                tokens = [w for w in word_tokenize(t)]
                word_inds.extend([self.env.db.lang_vocab(w) for w in tokens])
            word_inds = list(set(word_inds))
            logits = np.sum(self.word_stat[word_inds], axis=0)
            logits = logits / np.sum(logits)
            dis = Categorical(logits=torch.from_numpy(logits))
        else:
            raise Exception('Please choose a right rule. Now is {}'.format(self.cfg.ppo_rule))
        if self.cfg.ppo_rule == 'random':
            a = dis.sample([self.cfg.ppo_num_actions])
        else:
            a = torch.argsort(dis.probs, descending=True)[turn * self.cfg.ppo_num_actions:(turn + 1) * self.cfg.ppo_num_actions]
        if self.cfg.cuda:
            a = a.squeeze().cpu()
        return a.numpy()

    def test(self):
        all_retrieve_inds = []
        db_length = len(self.env.db.scenedb)
        # db_length = 500
        for idx in range(db_length):
            retrieve_inds = []
            # get test text
            test_scene = self.env.db.scenedb[idx]
            all_meta_regions = [test_scene['regions'][x] for x in sorted(list(test_scene['regions'].keys()))]
            all_captions = [x['caption'] for x in all_meta_regions]
            txt = all_captions[:self.cfg.test_turns]
            self.env.reset(txt, idx)
            if self.cfg.vg_img_feature is not None and self.cfg.vg_img_logits is not None:
                _, inds, rank = self.env.retrieve_precomp(*self.env.tokenize())
            else:
                _, inds, rank = self.env.retrieve(*self.env.tokenize())
            retrieve_inds.append(inds)
            for t in range(self.cfg.max_turns):
                if rank[0] < self.cfg.ppo_stop_rank:
                    break
                a = self.get_action(t)
                next_o, r, d, inds, logit = self.env.step(a)
                retrieve_inds.append(inds)

                if d:
                    break

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

    def demo(self):
        action_stats = np.zeros((1601))
        all_retrieve_inds = []
        # db_length = len(self.env.db.scenedb)
        # db_length = 500
        db_length = 50
        start_time = time()
        for idx in range(db_length):
            retrieve_inds = []
            # get test text
            test_scene = self.env.db.scenedb[idx]
            logging.info('Image %s' % test_scene['image_index'])
            # all_meta_regions = [test_scene['regions'][x] for x in sorted(list(test_scene['regions'].keys()))]
            # all_captions = [x['caption'] for x in all_meta_regions]
            # txt = all_captions[:self.cfg.test_turns]
            txt = [input('Please input a query: ')]
            self.env.reset(txt, idx)
            if self.cfg.vg_img_feature is not None and self.cfg.vg_img_logits is not None:
                _, inds, rank = self.env.retrieve_precomp(*self.env.tokenize())
            else:
                _, inds, rank = self.env.retrieve(*self.env.tokenize())
            retrieve_inds.append(inds)
            for t in range(self.cfg.max_turns):
                if rank[0] < self.cfg.ppo_stop_rank:
                    break
                a = self.get_action(t)
                for a_ in a:
                    action_stats[a_] += 1
                next_o, r, d, inds, logit = self.env.demo_step(a)
                retrieve_inds.append(inds)

                if d:
                    break

            all_retrieve_inds.append(retrieve_inds)
            logging.info('Get %d th retrieve result' % idx)
        end_time = time()
        during = end_time - start_time
        logging.info('Test time: ' + str(during))
        np.save(self.cfg.ppo_rule + '_action_dis.npy', action_stats)
        logging.info('Save action distribution.')

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

    def save_checkpoint(self, epoch, model_name):
        print('Saving checkpoint...')
        checkpoint_dir = osp.join(self.cfg.model_dir, 'snapshots')
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        states = {
            'epoch': epoch,
        }
        torch.save(states, osp.join(checkpoint_dir, str(epoch) + '.pkl'))

    def load_pretrained_net(self, pretrained_path):
        assert (osp.exists(pretrained_path))
        states = torch.load(pretrained_path)
        self.epoch = states['epoch']
