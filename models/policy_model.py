import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        """
        Produce action distributions for given observations and
        optionlly compute the log-likelihood of given action under
        those distributions
        :param obs:
        :param act:
        :return:
        """
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class ContextEncoder(nn.Module):

    def __init__(self, config):
        super(ContextEncoder, self).__init__()
        self.cfg = config
        self.encoder = nn.GRU(self.cfg.n_categories, self.cfg.n_categories, 1, batch_first=True, bidirectional=True)

    def forward(self, input_feats):
        """
        Context a sequence observations to a state
        :param input_feats: [batchsize, instance_dim, n_categories], tensor
        :return:
        """
        out_feats, _ = self.encoder(input_feats)
        out_feats = (out_feats[:, :, :out_feats.shape[2] // 2] +
                     out_feats[:, :, out_feats.shape[2] // 2:]) / 2
        length = input_feats.shape[1]
        length = torch.tensor([length])
        I = length.view(-1, 1, 1)
        I = I.expand(input_feats.shape[0], 1, out_feats.shape[2]) - 1
        if self.cfg.cuda:
            I = I.cuda()
        out_feat = torch.gather(out_feats, 1, I)

        return out_feats, out_feat


class LinearContextEncoder(nn.Module):

    def __init__(self, config):
        super(LinearContextEncoder, self).__init__()
        self.cfg = config
        self.encoder = nn.Linear(self.cfg.instance_dim, 1)

    def forward(self, input_feats):
        """
        Context a sequence observations to a state
        :param input_feats: [buffer_len, instance_dim, n_categories], tensor
        :return:
        """
        input_feats = input_feats.transpose(1, 2)
        out_feats = self.encoder(input_feats).squeeze(-1)
        return out_feats


class ConcatContextEncoder(nn.Module):

    def __init__(self, config):
        super(ConcatContextEncoder, self).__init__()
        self.cfg = config
        self.encoder = nn.Linear((self.cfg.n_feature_dim + self.cfg.n_categories), self.cfg.n_feature_dim)

    def forward(self, input_feats):
        """
        Context a observation to a state
        :param input_feats: [buffer_len, n_feature_dim + n_categories]
        :return: tensor, [buffer_len, n_feature_dim]
        """
        out_feats = self.encoder(input_feats)
        return out_feats


class ConcatUpdater(nn.Module):

    def __init__(self, config):
        super(ConcatUpdater, self).__init__()
        self.cfg = config
        self.updater = nn.GRU(self.cfg.n_feature_dim, self.cfg.n_feature_dim, 1, batch_first=True)

    def init_hidden(self, bsize):
        vhs = torch.zeros(bsize, self.cfg.n_feature_dim)
        if self.cfg.cuda:
            vhs = vhs.cuda()
        return vhs

    def forward(self, state, hidden):
        """
        update state
        :param state: [batchsize, sequence_len, n_feature_dim], tensor
        :param hidden: [1, batchsize, n_feature_dim], tensor
        :return: [batchsize, sequence_len, n_feature_dim], tensor
        """
        self.updater.flatten_parameters()
        new_state, new_hidden = self.updater(state, hidden)
        return new_state, new_hidden


class Updater(nn.Module):

    def __init__(self, config):
        super(Updater, self).__init__()
        self.cfg = config
        self.updater = nn.GRU(self.cfg.n_categories, self.cfg.n_categories, 1, batch_first=True)

    def init_hidden(self, bsize):
        vhs = torch.zeros(bsize, self.cfg.n_categories)
        if self.cfg.cuda:
            vhs = vhs.cuda()
        return vhs

    def forward(self, state, hidden):
        """
        update state
        :param state: [batchsize, 1, n_categories], tensor
        :param hidden: [1, bathcsize, n_categories], tensor
        :return:
        """
        self.updater.flatten_parameters()
        new_state, new_hidden = self.updater(state, hidden)
        return new_state, new_hidden


class ConcatPolicyNet(Actor):
    def __init__(self, config):
        super(ConcatPolicyNet, self).__init__()
        self.cfg = config
        self.context_encoder = ConcatContextEncoder(config)
        self.updater = ConcatUpdater(config)
        self.logits_net = nn.Sequential(
            nn.Linear(self.cfg.n_feature_dim, self.cfg.n_feature_dim),
            nn.Tanh(),
            nn.Linear(self.cfg.n_feature_dim, self.cfg.n_categories),
            nn.Softmax(dim=-1)
        )
        if self.cfg.cuda:
            self.mask = torch.ones(self.cfg.n_categories).type(torch.uint8).cuda()
        else:
            self.mask = torch.ones(self.cfg.n_categories).type(torch.uint8)

    def init_hidden(self, bsize):
        self.hidden = self.updater.init_hidden(bsize).unsqueeze(0)

    def reset_mask(self):
        if self.cfg.cuda:
            self.mask = torch.ones(self.cfg.n_categories).type(torch.uint8).cuda()
        else:
            self.mask = torch.ones(self.cfg.n_categories).type(torch.uint8)

    def set_mask(self, actions):
        """
        set attributes in actions to 0
        :param actions: list of int
        :return:
        """
        self.mask[actions] = 0

    def get_mask(self):
        return self.mask

    def refine_logit(self, logit, mask=None):
        """
        refine logit
        :param logit:
        :return:
        """
        if mask is not None:
            logit = logit * mask.type_as(logit)
        else:
            logit = logit * self.mask.type_as(logit)
        logit = logit / torch.sum(logit, dim=-1, keepdim=True)
        return logit

    def get_state(self, obs):
        """
        compute state
        if buffer_len > 1
        batchsize = buffer_len / max_turns
        :param obs: [buffer_len, n_feature_dim + n_categories], tensor
        :return: state [batchsize, max_turns, n_feature_dim], tensor
        else:
        :param obs: [1, n_feature_dim + n_categories], tensor
        :return: state [1, 1, n_feature_dim], tensor
        """
        # -> (buffer_len, n_feature_dim) or (1, n_feature_dim)
        out_feats = self.context_encoder(obs)
        if out_feats.shape[0] > 1:
            # -> (batchsize, max_turns, n_feature_dim]
            out_feats = out_feats.view(out_feats.shape[0] // self.cfg.max_turns, -1, out_feats.shape[-1])
            self.init_hidden(out_feats.shape[0])
            # -> (batchsize, maxturns, n_feature_dim]
            state, _ = self.updater(out_feats, self.hidden)
        else:
            out_feats = out_feats.unsqueeze(0)
            state, next_hidden = self.updater(out_feats, self.hidden)
            # -> [1, 1, n_feature_dim]
            self.hidden = next_hidden.detach()
        return state

    def get_logits(self, state, mask=None):
        """
        compute logits
        if buffer_len > 1:
        batchsize = buffer_len / max_turns
        :param state: [batchsize, max_turns, n_feature_dim], tensor
        :param mask: [batchsize, max_turns, n_categories], tensor
        else:
        :param state: [1, 1, n_categories], tensor
        :return:
        """
        if mask is not None:
            mask = mask.view(mask.shape[0] // self.cfg.max_turns, -1, mask.shape[-1])
        logits = self.logits_net(state)
        logits = self.refine_logit(logits, mask)
        return Categorical(logits=logits)

    def get_prob(self, obs, mask=None):
        """
        compute prob of given obs
        :param obs: [buffer_len, n_feature_dim + n_categories], tensor
        :return:
        """
        if mask is not None:
            mask = mask.view(mask.shape[0] // self.cfg.max_turns, -1, mask.shape[-1])
        state = self.get_state(obs)
        prob = self.logits_net(state)
        prob = self.refine_logit(prob, mask)
        return prob

    def _distribution(self, obs, mask=None):
        """
        compute distribution of actions
        :param obs: [buffer_len, n_feature_dim + n_categories], tensor
        :return:
        """
        # -> (batchsize, max_turns, n_feature_dim)
        state = self.get_state(obs)
        return self.get_logits(state, mask)

    def _log_prob_from_distribution(self, pi, act):
        """
        compute log prob of action
        :param pi: torch distribution
        :param act: [buffer_len, ], actions
        :return:
        """
        act = act.view(torch.Size([self.cfg.ppo_num_actions]) + pi.probs.shape[:-1])
        return pi.log_prob(act)


class LinearPolicyNet(Actor):
    def __init__(self, config):
        super(LinearPolicyNet, self).__init__()
        self.cfg = config
        self.context_encoder = LinearContextEncoder(config)
        self.updater = Updater(config)
        self.logits_net = nn.Sequential(
            nn.Linear(self.cfg.n_categories, self.cfg.n_categories),
            nn.Tanh(),
            nn.Linear(self.cfg.n_categories, self.cfg.n_categories),
            nn.Tanh(),
            nn.Linear(self.cfg.n_categories, self.cfg.n_categories),
            nn.Softmax(dim=-1)
        )
        if self.cfg.cuda:
            self.mask = torch.ones(self.cfg.n_categories).type(torch.uint8).cuda()
        else:
            self.mask = torch.ones(self.cfg.n_categories).type(torch.uint8)

    def init_hidden(self, bsize):
        self.hidden = self.updater.init_hidden(bsize).unsqueeze(0)

    def reset_mask(self):
        if self.cfg.cuda:
            self.mask = torch.ones(self.cfg.n_categories).type(torch.uint8).cuda()
        else:
            self.mask = torch.ones(self.cfg.n_categories).type(torch.uint8)

    def set_mask(self, actions):
        """
        set attributes in actions to 0
        :param actions: list of int
        :return:
        """
        self.mask[actions] = 0

    def get_mask(self):
        return self.mask

    def refine_logit(self, logit, mask=None):
        """
        refine logit
        :param logit:
        :return:
        """
        if mask is not None:
            logit = logit * mask.type_as(logit)
        else:
            logit = logit * self.mask.type_as(logit)
        logit = logit / torch.sum(logit, dim=-1, keepdim=True)
        return logit

    def get_state(self, obs):
        """
        compute state
        if buffer_len > 1
        batchsize = buffer_len / max_turns
        :param obs: [buffer_len, instance_dim, n_categories], tensor
        :return: state [batchsize, max_turns, n_categories], tensor
        else:
        :param obs: [1, instance_dim, n_categories], tensor
        :return: state [1, 1, n_categories], tensor
        """
        # -> (buffer_len, n_categories)
        out_feats = self.context_encoder(obs)
        if out_feats.shape[0] > 1:
            # -> (batchsize, max_turns, n_categories]
            out_feats = out_feats.view(out_feats.shape[0] // self.cfg.max_turns, -1, out_feats.shape[-1])
            self.init_hidden(out_feats.shape[0])
            # -> (batchsize, maxturns, n_categories]
            state, _ = self.updater(out_feats, self.hidden)
        else:
            out_feats = out_feats.unsqueeze(0)
            state, next_hidden = self.updater(out_feats, self.hidden)
            # -> [1, 1, n_categories]
            self.hidden = next_hidden.detach()
        return state

    def get_logits(self, state, mask=None):
        """
        compute logits
        if buffer_len > 1:
        batchsize = buffer_len / max_turns
        :param state: [batchsize, max_turns, n_categories], tensor
        :param mask: [batchsize, max_turns, n_categories], tensor
        else:
        :param state: [1, 1, n_categories], tensor
        :return:
        """
        if mask is not None:
            mask = mask.view(mask.shape[0] // self.cfg.max_turns, -1, mask.shape[-1])
        logits = self.logits_net(state)
        before = logits
        logits = self.refine_logit(logits, mask)
        after = logits
        return Categorical(logits=logits)

    def get_prob(self, obs):
        """
        compute prob of given obs
        :param obs: [buffer_len, instance_dim, n_categories], tensor
        :return:
        """
        state = self.get_state(obs)
        prob = self.logits_net(state)
        return prob

    def _distribution(self, obs, mask=None):
        """
        compute distribution of actions
        :param obs: [buffer_len, instance_dim, n_categories], tensor
        :return:
        """
        # -> (batchsize, max_turns, n_categories)
        state = self.get_state(obs)
        return self.get_logits(state, mask)

    def _log_prob_from_distribution(self, pi, act):
        """
        compute log prob of action
        :param pi: torch distribution
        :param act: [buffer_len, ], actions
        :return:
        """
        act = act.view(torch.Size([self.cfg.ppo_num_actions]) + pi.probs.shape[:-1])
        return pi.log_prob(act)


class MLPPolicyNet(Actor):
    def __init__(self, config):
        super(MLPPolicyNet, self).__init__()
        self.cfg = config
        self.net = nn.Sequential(
            nn.Linear(self.cfg.n_feature_dim + self.cfg.n_categories, self.cfg.n_categories),
            nn.Tanh(),
            nn.Linear(self.cfg.n_categories, self.cfg.n_categories),
            nn.Tanh(),
            nn.Linear(self.cfg.n_categories, self.cfg.n_categories),
            nn.Softmax(dim=-1)
        )

    def get_logit(self, obs):
        """
        compute logit of given obs
        :param obs: [buffer_len, n_featuren_dim + n_categories], tensor
        :return:
        """
        logit = self.net(obs)
        logit = Categorical(probs=logit)
        return logit

    def get_prob(self, obs):
        """
        compute prob of given obs
        :param obs: [buffer_len, n_feature_dim + n_categories], tensor
        :return:
        """
        prob = self.net(obs)
        return prob

    def _distribution(self, obs):
        """
        compute distribution of actions
        :param obs: [buffer_len, n_feature_dim + n_categories], tensor
        :return:
        """
        return self.get_logit(obs)

    def _log_prob_from_distribution(self, pi, act):
        """
        compute log prob of action
        :param pi: torch distribution
        :param act: [buffer_len, num_actions], actions
        :return:
        """
        # -> (num_actions, buffer_len)
        act = act.transpose(0, 1)
        return pi.log_prob(act)

class MLPBonusPolicyNet(Actor):
    def __init__(self, config):
        super(MLPBonusPolicyNet, self).__init__()
        self.cfg = config
        self.net = nn.Sequential(
            nn.Linear(self.cfg.n_feature_dim + self.cfg.n_categories * 3, self.cfg.n_categories),
            nn.Tanh(),
            nn.Linear(self.cfg.n_categories, self.cfg.n_categories),
            nn.Tanh(),
            nn.Linear(self.cfg.n_categories, self.cfg.n_categories),
            nn.Softmax(dim=-1)
        )

    def get_logit(self, obs):
        """
        compute logit of given obs
        :param obs: [buffer_len, n_featuren_dim + n_categories], tensor
        :return:
        """
        logit = self.net(obs)
        logit = Categorical(probs=logit)
        return logit

    def get_prob(self, obs):
        """
        compute prob of given obs
        :param obs: [buffer_len, n_feature_dim + n_categories], tensor
        :return:
        """
        prob = self.net(obs)
        return prob

    def _distribution(self, obs):
        """
        compute distribution of actions
        :param obs: [buffer_len, n_feature_dim + n_categories], tensor
        :return:
        """
        return self.get_logit(obs)

    def _log_prob_from_distribution(self, pi, act):
        """
        compute log prob of action
        :param pi: torch distribution
        :param act: [buffer_len, num_actions], actions
        :return:
        """
        # -> (num_actions, buffer_len)
        act = act.transpose(0, 1)
        return pi.log_prob(act)


class PolicyNet(Actor):
    def __init__(self, config):
        super(PolicyNet, self).__init__()
        self.cfg = config
        self.context_encoder = ContextEncoder(config)
        self.updater = Updater(config)
        self.logits_net = nn.Sequential(
            nn.Linear(self.cfg.n_categories, self.cfg.n_categories),
            nn.Tanh(),
            nn.Linear(self.cfg.n_categories, self.cfg.n_categories),
            nn.Tanh(),
            nn.Linear(self.cfg.n_categories, self.cfg.n_categories)
        )

    def init_hidden(self):
        self.hidden = self.updater.init_hidden(self.cfg.ppo_batchsize).unsqueeze(0)

    def get_state(self, obs):
        """
        compute state
        :param obs: [batchsize, instance_dim, n_categories], tensor
        :return:
        """
        _, obs = self.context_encoder(obs)
        obs = obs.transpose(0, 1)
        state, next_hidden = self.updater(obs, self.hidden)
        state = state[:, -1, :].unsqueeze(1)
        self.hidden = next_hidden.detach()
        return state

    def get_logits(self, state):
        """
        compute logits
        :param state: [batchsize, 1, n_categories], tensor
        :return:
        """
        logits = self.logits_net(state.squeeze(1))
        return Categorical(logits=logits)

    def get_prob(self, obs):
        """
        compute prob of given obs
        :param obs: [batchsize, instance_dim, n_categories], tensor
        :return:
        """
        state = self.get_state(obs)
        prob = self.logits_net(state.squeeze(1))
        return prob

    def _distribution(self, obs, mask=None):
        """
        compute distribution of actions
        :param obs: [batchsize, instance_dim, n_categories], tensor
        :return:
        """
        # -> (batchsize, n_categories)
        state = self.get_state(obs)
        logits = self.logits_net(state.squeeze(1))
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        """
        compute log prob of action
        :param pi: torch distribution
        :param act: [batchsize, ], actions
        :return:
        """
        return pi.log_prob(act)


class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        self.cfg = config
        self.v_net = nn.Sequential(
            nn.Linear(self.cfg.n_categories, self.cfg.n_feature_dim),
            nn.Tanh(),
            nn.Linear(self.cfg.n_feature_dim, self.cfg.n_feature_dim),
            nn.Tanh(),
            nn.Linear(self.cfg.n_feature_dim, 1)
        )

    def forward(self, obs):
        """
        predict value of observation
        :param obs: [batchsize, n_categories]
        :return:
        """
        return self.v_net(obs).squeeze()

class MLPCritic(nn.Module):
    def __init__(self, config):
        super(MLPCritic, self).__init__()
        self.cfg = config
        self.v_net = nn.Sequential(
            nn.Linear(self.cfg.n_feature_dim + self.cfg.n_categories, self.cfg.n_feature_dim),
            nn.Tanh(),
            nn.Linear(self.cfg.n_feature_dim, self.cfg.n_feature_dim),
            nn.Tanh(),
            nn.Linear(self.cfg.n_feature_dim, 1)
        )

    def forward(self, obs):
        """
        predict value of observation
        :param obs: [batchszie, n_feature_dim + n_categories], tensor
        :return:
        """
        return self.v_net(obs).squeeze()


class ConcatCritic(nn.Module):
    def __init__(self, config):
        super(ConcatCritic, self).__init__()
        self.cfg = config
        self.v_net = nn.Sequential(
            nn.Linear(self.cfg.n_feature_dim, self.cfg.n_feature_dim),
            nn.Tanh(),
            nn.Linear(self.cfg.n_feature_dim, 1)
        )

    def forward(self, obs):
        """
        predict value of observation
        :param obs: [batchsize, n_feature_dim]
        :return:  [batchsize, ]
        """
        return self.v_net(obs).squeeze()

class MLPBonusCritic(nn.Module):
    def __init__(self, config):
        super(MLPBonusCritic, self).__init__()
        self.cfg = config
        self.v_net = nn.Sequential(
            nn.Linear(self.cfg.n_feature_dim + self.cfg.n_categories * 3, self.cfg.n_feature_dim),
            nn.Tanh(),
            nn.Linear(self.cfg.n_feature_dim, self.cfg.n_feature_dim),
            nn.Tanh(),
            nn.Linear(self.cfg.n_feature_dim, 1)
        )

    def forward(self, obs):
        """
        predict value of observation
        :param obs: [batchszie, n_feature_dim + n_categories], tensor
        :return:
        """
        return self.v_net(obs).squeeze()


class ActorCritic(nn.Module):
    def __init__(self, config):
        super(ActorCritic, self).__init__()
        self.cfg = config
        self.pi = PolicyNet(config)
        self.v = Critic(config)

    def step(self, obs):
        with torch.no_grad():
            state = self.pi.get_state(obs)
            pi = self.pi.get_logits(state)
            a = pi.sample()
            log_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(state)
            if self.cfg.cuda:
                a, v, log_a = a.cpu(), v.cpu(), log_a.cpu()
        return a.numpy(), v.numpy(), log_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class ConcatActorCritic(nn.Module):
    def __init__(self, config):
        super(ConcatActorCritic, self).__init__()
        self.cfg = config
        self.pi = ConcatPolicyNet(config)
        self.v = ConcatCritic(config)

    def step(self, obs):
        """
        :param obs: [buffer_len, n_feature_dim + n_categories], tensor
        :return:
        """
        with torch.no_grad():
            state = self.pi.get_state(obs)
            pi = self.pi.get_logits((state))
            a = pi.sample([self.cfg.ppo_num_actions])
            log_a = self.pi._log_prob_from_distribution(pi, a)
            mask = self.pi.get_mask()
            v = self.v(state)
            if self.cfg.cuda:
                a, v, log_a, mask = a.squeeze().cpu(), v.cpu(), log_a.squeeze().cpu(), mask.squeeze().cpu()
            else:
                a, v, log_a, mask = a.squeeze(), v.squeeze(), log_a.squeeze(), mask.squeeze()
            return a.numpy(), v.numpy(), log_a.numpy(), mask.numpy()

    def act(self, obs):
        """
        :param obs: [buffer_len, n_feature_dim + n_categories], tensor
        :return:
        """
        return self.step(obs)[0]


class LinearActorCritic(nn.Module):
    def __init__(self, config):
        super(LinearActorCritic, self).__init__()
        self.cfg = config
        self.pi = LinearPolicyNet(config)
        self.v = Critic(config)

    def step(self, obs):
        """
        :param obs: [buffer_len, instance_dim, n_categories], tensor
        :return:
        """
        with torch.no_grad():
            state = self.pi.get_state(obs)
            pi = self.pi.get_logits(state)
            a = pi.sample([self.cfg.ppo_num_actions])
            log_a = self.pi._log_prob_from_distribution(pi, a)
            mask = self.pi.get_mask()
            v = self.v(state)
            if self.cfg.cuda:
                a, v, log_a, mask = a.squeeze().cpu(), v.cpu(), log_a.squeeze().cpu(), mask.squeeze().cpu()
        return a.numpy(), v.numpy(), log_a.numpy(), mask.numpy()

    def act(self, obs):
        """
        :param obs: [buffer_len, instance_dim, n_categories], tensor
        :return:
        """
        return self.step(obs)[0]

class MLPActorCritic(nn.Module):
    def __init__(self, config):
        super(MLPActorCritic, self).__init__()
        self.cfg = config
        self.pi = MLPPolicyNet(config)
        self.v = MLPCritic(config)

    def step(self, obs):
        """

        :param obs:  [buffer_len, n_feature_dim + n_categories], tensor
        :return:
        """
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample([self.cfg.ppo_num_actions])
            log_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
            if self.cfg.cuda:
                a, v, log_a = a.squeeze().cpu(), v.cpu(), log_a.squeeze().cpu()
        return a.numpy(), v.numpy(), log_a.numpy()

    def act(self, obs):
        """

        :param obs: [buffer_len, n_feature_dim + n_categories], tensor
        :return:
        """
        return self.step(obs)[0]

class MLPBonusActorCritic(nn.Module):
    def __init__(self, config):
        super(MLPBonusActorCritic, self).__init__()
        self.cfg = config
        self.pi = MLPBonusPolicyNet(config)
        self.v = MLPBonusCritic(config)

    def step(self, obs):
        """

        :param obs:  [buffer_len, n_feature_dim + n_categories], tensor
        :return:
        """
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample([self.cfg.ppo_num_actions])
            log_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
            if self.cfg.cuda:
                a, v, log_a = a.squeeze().cpu(), v.cpu(), log_a.squeeze().cpu()
        return a.numpy(), v.numpy(), log_a.numpy()

    def act(self, obs):
        """

        :param obs: [buffer_len, n_feature_dim + n_categories], tensor
        :return:
        """
        return self.step(obs)[0]
