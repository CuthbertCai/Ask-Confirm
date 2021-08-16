import sys

sys.path.append('../')
import os.path as osp
import numpy as np
import pickle
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from nltk.tokenize import word_tokenize
from scipy.special import softmax

from models.text_image_matching_model import TextImageMatchingModel
from models.image_attribute_model import ImageAttributeModel
from train.trainer_txt_img_matching import TextImageMatchingTrainer
from train.trainer_img_attr import ImageAttributeTrainer
from utils.utils import *
from utils.config import get_test_config
from utils.vocab import Vocabulary
from datasets.vg import vg
from datasets.loader import region_loader, region_collate_fn


class EnvRulesModel(nn.Module):
    def __init__(self, config, split='train'):
        super(EnvRulesModel, self).__init__()
        self.cfg = config
        self.txt_img_matching_trainer = TextImageMatchingTrainer(config)
        self.set_no_grad(self.txt_img_matching_trainer.net)
        if self.cfg.vg_img_feature is None or self.cfg.vg_img_logits is None:
            self.img_attr_trainer = ImageAttributeTrainer(config)
            self.set_no_grad(self.img_attr_trainer.net)
        self.txt = None
        self.index = None
        self.mask = None
        self.txt_feat = None
        self.logit = None
        self.logit_return = None
        self.cache_dir = osp.abspath(osp.join(config.data_dir, 'caches'))
        if split == 'train':
            self.db = vg(config, 'train')
        elif split == 'val':
            self.db = vg(config, 'val')
        elif split == 'test':
            self.db = vg(config, 'test')
        else:
            raise Exception('Please choose a right split.')

        self.loaddb = region_loader(self.db)
        self.loader = DataLoader(self.loaddb, batch_size=self.cfg.batch_size, shuffle=False,
                                 num_workers=self.cfg.num_workers, collate_fn=region_collate_fn)
        if self.cfg.vg_img_feature is not None:
            self.vg_img_feature = np.load(self.cfg.vg_img_feature)
        if self.cfg.vg_img_logits is not None:
            self.vg_img_logits = np.load(self.cfg.vg_img_logits)
            obj_pred = np.where(self.vg_img_logits > 0.9)[1]
            self.vg_all_logits_dis = np.zeros(self.cfg.n_categories)
            for obj in obj_pred:
                self.vg_all_logits_dis[obj] += 1
            self.vg_all_logits_dis = self.vg_all_logits_dis / np.sum(self.vg_all_logits_dis)
        if self.cfg.vg_attr_feature is not None:
            self.vg_attr_feature = np.load(self.cfg.vg_attr_feature)

        self.logit_gts = []
        max_len = 0
        for scene in self.loaddb:
            logit_gt = scene[6]
            if len(logit_gt) > max_len:
                max_len = len(logit_gt)
            self.logit_gts.append(logit_gt)
        self.logits = np.zeros((len(self.loaddb), max_len), dtype=np.int) - 1
        for i, logit in enumerate(self.logit_gts):
            le = len(self.logit_gts[i])
            self.logits[i, :le] = self.logit_gts[i]

    def set_no_grad(self, model):
        for param in model.parameters():
            param.required_grad = False

    def demo_step(self, action, dis=None):
        """
        compute next observation, reward and termination signal
        :param action:
        :param dis: distribution of actions, [1, n_categories]
        :return:
        """
        # update text
        logging.info('Suggest objects: ')
        for a in action:
            logging.info(self.db.classes[a][0])
        input_idx = input('Choose actions: ')
        input_logit_idx = [int(i) for i in list(input_idx)]
        input_logit = [action[i] for i in input_logit_idx]
        unfilter_a, filter_a = filter_actions(action, input_logit)
        logging.info('False objects:')
        for a in filter_a:
            logging.info(self.db.classes[a][0])

        # add objects to the tail of each query
        # for i in unfilter_a:
        #     attr = self.db.classes[i]
        #     for i in range(len(self.txt)):
        #         self.txt[i] += (' ' + attr[0])

        # add objects to new querys
        for i in unfilter_a:
            attr = self.db.classes[i]
            self.txt.append(attr[0])
            self.logit_return[i] = 0.0

        word_inds, word_msks = self.tokenize()
        for t in self.txt:
            logging.info(t)
        # print(self.txt)

        # mask some image using gt
        # for a in filter_a:
        #     filter_img = np.where(self.logits == a)
        #     filter_img = list(set(filter_img[0]))
        #     self.mask[filter_img] = 0

        # masm some images using threshold
        for a in filter_a:
            filter_img = np.where(self.vg_img_logits[:, a] > 0.9)
            self.mask[filter_img] *= 0.9

        # generate next obs
        if self.cfg.cuda:
            word_inds = word_inds.cuda(non_blocking=True)
            word_msks = word_msks.cuda(non_blocking=True)
        bsize, nturns, nwords = word_inds.size()
        _, txt_feat, _ = self.txt_img_matching_trainer.net.txt_enc(word_inds.view(-1, nwords),
                                                                   word_msks.view(-1, nwords))
        if bsize == 1:
            txt_feat = txt_feat.view(nturns, self.cfg.n_feature_dim)
        else:
            raise Exception('Batchsize should be 1.')
        if self.cfg.l2_norm:
            txt_feat = l2norm(txt_feat)
        txt_feat = txt_feat.detach()

        vg_attr_feature = torch.from_numpy(self.vg_attr_feature).type_as(txt_feat)
        self.attr_sim = self.compute_attr_similarity(txt_feat, vg_attr_feature)

        if self.cfg.vg_img_feature is not None and self.cfg.vg_img_logits is not None:
            reward, inds, rank = self.retrieve_precomp(word_inds, word_msks)
        else:
            reward, inds, rank = self.retrieve(word_inds, word_msks)

        # get top10 image
        top_inds = inds[:10]
        img_inds = [self.db.scenedb[idx]['image_index'] for idx in top_inds]
        logging.info('Top image: ')
        for i in img_inds:
            logging.info(i)

        if rank[0] < self.cfg.ppo_stop_rank:
            return None, None, True, inds, self.logit_return
        else:
            return None, None, False, inds, self.logit_return

    def step(self, action, dis=None):
        """
        compute next observation, reward and termination signal
        :param action:
        :param dis: distribution of actions, [1, n_categories]
        :return:
        """
        # update text
        unfilter_a, filter_a = filter_actions(action, self.logit)
        logging.info('False objects:')
        for a in filter_a:
            logging.info(self.db.classes[a][0])

        # add objects to the tail of each query
        # for i in unfilter_a:
        #     attr = self.db.classes[i]
        #     for i in range(len(self.txt)):
        #         self.txt[i] += (' ' + attr[0])

        # add objects to new querys
        for i in unfilter_a:
            attr = self.db.classes[i]
            self.txt.append(attr[0])
            self.logit_return[i] = 0.0

        word_inds, word_msks = self.tokenize()
        for t in self.txt:
            logging.info(t)
        # print(self.txt)

        # mask some image using gt
        # for a in filter_a:
        #     filter_img = np.where(self.logits == a)
        #     filter_img = list(set(filter_img[0]))
        #     self.mask[filter_img] = 0

        # masm some images using threshold
        for a in filter_a:
            filter_img = np.where(self.vg_img_logits[:, a] > 0.9)
            self.mask[filter_img] *= 0.9

        # generate next obs
        if self.cfg.cuda:
            word_inds = word_inds.cuda(non_blocking=True)
            word_msks = word_msks.cuda(non_blocking=True)
        bsize, nturns, nwords = word_inds.size()
        _, txt_feat, _ = self.txt_img_matching_trainer.net.txt_enc(word_inds.view(-1, nwords),
                                                                   word_msks.view(-1, nwords))
        if bsize == 1:
            txt_feat = txt_feat.view(nturns, self.cfg.n_feature_dim)
        else:
            raise Exception('Batchsize should be 1.')
        if self.cfg.l2_norm:
            txt_feat = l2norm(txt_feat)
        txt_feat = txt_feat.detach()

        vg_attr_feature = torch.from_numpy(self.vg_attr_feature).type_as(txt_feat)
        self.attr_sim = self.compute_attr_similarity(txt_feat, vg_attr_feature)

        if self.cfg.vg_img_feature is not None and self.cfg.vg_img_logits is not None:
            reward, inds, rank = self.retrieve_precomp(word_inds, word_msks)
        else:
            reward, inds, rank = self.retrieve(word_inds, word_msks)

        # get top10 image
        top_inds = inds[:10]
        img_inds = [self.db.scenedb[idx]['image_index'] for idx in top_inds]
        logging.info('Top image: ')
        for i in img_inds:
            logging.info(i)

        if rank[0] < self.cfg.ppo_stop_rank:
            return None, None, True, inds, self.logit_return
        else:
            return None, None, False, inds, self.logit_return

    def reset(self, txt, index):
        """
        reset environment
        :param txt: a str
        :return
        """
        self.txt = txt
        # self.txt = ""
        # for tx in txt:
        #     for t in tx:
        #         self.txt += t
        self.index = index
        word_inds, word_msks = self.tokenize()
        logging.info('Index: {}'.format(index))

        scene = self.db.scenedb[index]
        obj_to_ind = self.db.class_to_ind
        objs = [scene['objects'][obj]['name'] for obj in scene['objects'] if
                scene['objects'][obj]['name'] in obj_to_ind.keys()]
        obj_inds = list(set([obj_to_ind[obj] for obj in objs]))
        self.logit = obj_inds

        self.logit_return = np.zeros(self.cfg.n_categories)
        self.logit_return[self.logit] = 1.0

        # compute text feature
        if self.cfg.cuda:
            word_inds = word_inds.cuda(non_blocking=True)
            word_msks = word_msks.cuda(non_blocking=True)
        bsize, nturns, nwords = word_inds.size()
        _, txt_feat, _ = self.txt_img_matching_trainer.net.txt_enc(word_inds.view(-1, nwords),
                                                                   word_msks.view(-1, nwords))
        if bsize == 1:
            txt_feat = txt_feat.view(nturns, self.cfg.n_feature_dim)
        else:
            raise Exception('Batchsize should be 1.')
        if self.cfg.l2_norm:
            txt_feat = l2norm(txt_feat)
        self.txt_feat = txt_feat.detach()

        vg_attr_feature = torch.from_numpy(self.vg_attr_feature).type_as(txt_feat)
        self.attr_sim = self.compute_attr_similarity(self.txt_feat, vg_attr_feature)

        if self.vg_img_logits is not None:
            self.mask = np.ones(len(self.vg_img_logits))

    def tokenize(self):
        # tokens = [w for w in word_tokenize(self.txt)]
        # word_inds = [self.db.lang_vocab(w) for w in tokens]
        # word_inds.append(self.cfg.EOS_idx)
        #
        # return torch.tensor(word_inds)
        word_inds = []
        lengths = []
        for t in self.txt:
            tokens = [w for w in word_tokenize(t)]
            word_ind = [self.db.lang_vocab(w) for w in tokens]
            word_ind.append(self.cfg.EOS_idx)
            lengths.append(len(word_ind))
            word_inds.append(torch.Tensor(word_ind))
        max_length = max(lengths)
        new_word_inds = torch.zeros(len(word_inds), max_length).long()
        new_word_msks = torch.zeros(len(word_inds), max_length).long()
        for i in range(len(lengths)):
            new_word_inds[i, :lengths[i]] = word_inds[i]
            new_word_msks[i, :lengths[i]] = 1.0
        return new_word_inds.unsqueeze(0), new_word_msks.unsqueeze(0)

    def compute_similarity(self, txt_feat, all_img_feats):
        # compute similarity
        raw_sim = torch.matmul(txt_feat, all_img_feats.transpose(-1, -2))
        txt_feat_norm = torch.norm(txt_feat, 2, dim=-1, keepdim=True)
        all_img_feats_norm = torch.norm(all_img_feats, 2, dim=-1, keepdim=True)
        raw_sim_norm = txt_feat_norm.unsqueeze(0) * all_img_feats_norm.transpose(-1, -2)
        sim = raw_sim / raw_sim_norm

        attn = torch.softmax(sim * self.cfg.lse_lambda, dim=-1)
        sim = torch.mean(torch.mean(attn * sim, dim=-1, keepdim=True), dim=-2, keepdim=True).squeeze()

        # rank
        sim = sim.detach().cpu().numpy()
        sim = sim * self.mask
        return sim

    def compute_attr_similarity(self, txt_feat, all_attr_feats):
        # compute similarity
        raw_sim = torch.matmul(txt_feat, all_attr_feats.transpose(0, 1))
        txt_feat_norm = torch.norm(txt_feat, 2, dim=-1, keepdim=True)
        all_attr_feats_norm = torch.norm(all_attr_feats, 2, dim=-1, keepdim=True)
        raw_sim_norm = txt_feat_norm * all_attr_feats_norm.transpose(0, 1)
        sim = raw_sim / raw_sim_norm

        attn = torch.softmax(sim * self.cfg.lse_lambda, dim=-1)
        sim = torch.mean(attn * sim, dim=-2, keepdim=True).squeeze()
        return sim

    def retrieve_precomp(self, word_inds, word_msks):
        """
        rank image accroding to a caption
        :param word_inds: [1, nturns, word num], LongTensor
        :return:
        """
        # compute text feature
        if self.cfg.cuda:
            word_inds = word_inds.cuda(non_blocking=True)
            word_msks = word_msks.cuda(non_blocking=True)
        bsize, nturns, nwords = word_inds.size()
        _, txt_feat, _ = self.txt_img_matching_trainer.net.txt_enc(word_inds.view(-1, nwords),
                                                                   word_msks.view(-1, nwords))
        if bsize == 1:
            txt_feat = txt_feat.view(nturns, self.cfg.n_feature_dim)
        else:
            raise Exception('Batchsize should be 1.')
        if self.cfg.l2_norm:
            txt_feat = l2norm(txt_feat)

        all_img_feats = torch.from_numpy(self.vg_img_feature)
        all_img_attrs = torch.from_numpy(self.vg_img_logits)
        if self.cfg.cuda:
            all_img_feats, all_img_attrs = all_img_feats.cuda(), all_img_attrs.cuda()
        # logging.info('Get all image features and logits.')

        sim = self.compute_similarity(txt_feat, all_img_feats)
        sim_mean = np.mean(sim)
        # reward = np.log(sim[self.index] + 1e-10) - np.log(sim_mean + 1e-10)
        # reward = min(np.exp(sim[self.index] - sim_mean), 3)
        # reward = np.tanh(np.exp((sim[self.index] - sim_mean)))
        reward = (sim[self.index] - sim_mean) * 100.0
        inds = np.argsort(sim)[::-1]
        rank = np.where(inds == self.index)
        logging.info('Now rank {}'.format(rank))

        return reward, inds, rank

    def retrieve(self, word_inds, word_msks):
        """
        rank image accroding to a caption
        :param word_inds: [1, nturns, word num], LongTensor
        :return:
        """
        # compute text feature
        if self.cfg.cuda:
            word_inds = word_inds.cuda(non_blocking=True)
            word_msks = word_msks.cuda(non_blocking=True)
        bsize, nturns, nwords = word_inds.size()
        _, txt_feat, _ = self.txt_img_matching_trainer.net.txt_enc(word_inds.view(-1, nwords),
                                                                   word_msks.view(-1, nwords))
        if bsize == 1:
            txt_feat = txt_feat.view(nturns, self.cfg.n_feature_dim)
        else:
            raise Exception('Batchsize should be 1.')
        if self.cfg.l2_norm:
            txt_feat = l2norm(txt_feat)

        # compute image feature and image attribute logits
        all_img_feats = []
        all_img_attrs = []
        self.txt_img_matching_trainer.net.eval()
        self.img_attr_trainer.net.eval()
        with torch.no_grad():
            for cnt, batched in enumerate(self.loader):
                sent_inds, sent_msks, region_feats = self.txt_img_matching_trainer.batch_data(batched)
                img_feats, _ = self.txt_img_matching_trainer.net(sent_inds, sent_msks, region_feats)
                logits = self.img_attr_trainer.net(region_feats)
                all_img_feats.append(img_feats)
                all_img_attrs.append(logits)

            all_img_feats = torch.cat(all_img_feats, 0)
            all_img_attrs = torch.cat(all_img_attrs, 0)
            # logging.info('Get all image features and logits.')

        # rank
        sim = self.compute_similarity(txt_feat, all_img_feats)
        sim_mean = np.mean(sim)
        reward = np.log(sim[self.index] + 1e-10) - np.log(sim_mean + 1e-10)
        inds = np.argsort(sim)[::-1]
        rank = np.where(inds == self.index)
        logging.info('Now rank {}'.format(rank))

        return reward, inds, rank


def demo_test(config):
    env = EnvRulesModel(config, 'test')
    # index = random.randint(0, len(env.loaddb) - 1)
    index = 2458
    scene = env.db.scenedb[index]
    obj_to_ind = env.db.class_to_ind
    objs = [scene['objects'][obj]['name'] for obj in scene['objects'] if
            scene['objects'][obj]['name'] in obj_to_ind.keys()]
    print(objs)
    obj_inds = torch.tensor(list(set([obj_to_ind[obj] for obj in objs]))).long()
    obj_inds = F.one_hot(obj_inds, num_classes=config.n_categories)
    logit = torch.sum(obj_inds, dim=0).float()
    logit[0] = 0.0
    all_meta_regions = [scene['regions'][x] for x in sorted(list(scene['regions'].keys()))]
    all_captions = [x['caption'] for x in all_meta_regions]
    cap_idx = random.randint(0, len(all_captions) - 1)
    txt = all_captions[cap_idx]
    # env.reset(txt, index)
    env.reset("the eyes of the girl pizza", 2458)

    r, topk_attrs, inds = env.retrieve_precomp(env.tokenize().unsqueeze(0))
    rank = np.where(inds == index)
    print(env.txt, env.index, r, rank)
    # attrs = topk_attrs.mean(dim=1).cpu().numpy()
    # inds = np.argsort(attrs)[::-1]
    # top30 = inds[:30]
    # attrs = []
    # for i in range(30):
    #     attrs.append(env.db.classes[top30[i]])
    # print(attrs)

    attrs = env.vg_img_logits[index]
    inds = np.argsort(attrs)[::-1]
    top30 = inds[:30]
    attrs = []
    for i in range(30):
        attrs.append(env.db.classes[top30[i]])
    print(attrs)


if __name__ == "__main__":
    config, unparsed = get_test_config()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed_all(config.seed)
    demo_test(config)
