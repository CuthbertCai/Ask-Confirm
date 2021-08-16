import torch
import torch.nn as nn

from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from utils.utils import *

class TextImageMatchingGlobalModel(nn.Module):
    def __init__(self, config):
        super(TextImageMatchingGlobalModel, self).__init__()
        self.cfg = config
        self.img_enc = ImageEncoder(config)
        self.txt_enc = TextEncoder(config)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif param.dim() < 2:
                nn.init.uniform_(param)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, sent_inds, sent_msks, region_feats):
        # encode image feature
        img_feats = self.img_enc(region_feats)
        img_feats = torch.mean(img_feats, dim=1, keepdim=True)
        if self.cfg.l2_norm:
            img_feats = l2norm(img_feats)

        # encode text feature
        bsize, nturns, nwords = sent_inds.size()
        _, lang_feats, _ = self.txt_enc(sent_inds.view(-1, nwords), sent_msks.view(-1, nwords))
        lang_feats = lang_feats.view(bsize, nturns, self.cfg.n_feature_dim)
        lang_masks =lang_feats.new_ones(bsize, nturns)
        if self.cfg.l2_norm:
            lang_feats = l2norm(lang_feats)

        return img_feats, lang_feats

    def triplet_loss(self, img_feats, lang_feats):
        sim = self.compute_sim(img_feats, lang_feats)
        diagonal = sim.diag().view(img_feats.size(0), 1)
        d1 = diagonal.expand_as(sim)
        d2 = diagonal.t().expand_as(sim)

        # compare every diagonal score to scores in its column
        # image retrieval
        cost_i = (self.cfg.margin + sim - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # caption retrieval
        cost_l = (self.cfg.margin + sim - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(sim.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_i = cost_i.masked_fill_(I, 0)
        cost_l = cost_l.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.cfg.max_violation:
            cost_i = cost_i.max(1)[0]
            cost_l = cost_l.max(0)[0]
        return cost_i.sum() + cost_l.sum()

    def compute_sim(self, img_feats, lang_feats):
        """

        :param img_feats: tensor, [batchsize, 1, feature_dim]
        :param lang_feats: tensor, [batchsize, num_turn, feature_dim]
        :return: similarity: tensor, [batchsize, batchsize]
        """
        similarity = []
        for i in range(lang_feats.shape[0]):
            # -> (1, num_turn, feature_dim)
            lang_feat = lang_feats[i]
            # -> (batchsize, num_turn, feature_dim)
            lang_feat_expand = lang_feat.repeat(lang_feats.shape[0], 1, 1)
            # ->(batchsize, num_turn, 1)
            raw_sim = torch.bmm(lang_feat_expand, img_feats.transpose(1, 2))
            img_feats_norm = torch.norm(img_feats, 2, dim=-1, keepdim=True)
            lang_feats_norm = torch.norm(lang_feats, 2,dim=-1, keepdim=True)
            raw_sim_norm = lang_feats_norm * img_feats_norm.transpose(1, 2)
            sim = raw_sim / raw_sim_norm
            # -> (batchsize)
            sim = torch.mean(torch.mean(sim, dim=-1, keepdim=True), dim=-2, keepdim=True)
            similarity.append(sim.squeeze(-1))
        # -> (batchsize, batchsize)
        similarity = torch.cat(similarity, dim=1).t()
        return similarity

    def compute_triplet_loss(self, sent_inds, sent_msks, region_feats):
        img_feats, lang_feats = self.forward(sent_inds, sent_msks, region_feats)
        loss = self.triplet_loss(img_feats, lang_feats)
        return loss