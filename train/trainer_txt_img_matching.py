import os
import numpy as np
from time import time
import pickle
import os.path as osp
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.utils import *
from utils.optim import Optimizer
from utils.vocab import Vocabulary

from datasets.loader import region_loader, region_collate_fn
from models.text_image_matching_model import TextImageMatchingModel


class TextImageMatchingTrainer(object):
    def __init__(self, config):
        self.cfg = config
        self.net = TextImageMatchingModel(self.cfg)
        if self.cfg.cuda:
            self.net = self.net.cuda()
        params = filter(lambda p: p.requires_grad, self.net.parameters())
        raw_optimizer = optim.Adam(params, lr=self.cfg.lr)
        optimizer = Optimizer(raw_optimizer, max_grad_norm=self.cfg.grad_norm_clipping)
        scheduler = optim.lr_scheduler.StepLR(raw_optimizer, step_size=20, gamma=0.1)
        optimizer.set_scheduler(scheduler)
        self.optimizer = optimizer
        self.epoch = 0
        print(self.cfg.txt_img_matching_pretrained)
        if self.cfg.txt_img_matching_pretrained is not None:
            self.load_pretrained_net(self.cfg.txt_img_matching_pretrained)

        print('-------------------')
        print('All parameters')
        for name, param in self.net.named_parameters():
            print(name, param.size())
        print('-------------------')
        print('Trainable parameters')
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                print(name, param.size())

    def batch_data(self, entry):
        sent_inds = entry['sent_inds'].long()
        sent_msks = entry['sent_msks'].float()
        region_feats = entry['region_feats'].float()

        if self.cfg.cuda:
            sent_inds = sent_inds.cuda(non_blocking=True)
            sent_msks = sent_msks.cuda(non_blocking=True)
            region_feats = region_feats.cuda(non_blocking=True)

        return sent_inds, sent_msks, region_feats

    def train(self, train_db, val_db, test_db):
        start = time()
        train_loaddb = region_loader(train_db)
        val_loaddb = region_loader(val_db)

        train_loader = DataLoader(train_loaddb, batch_size=self.cfg.batch_size, shuffle=True,
                                  num_workers=self.cfg.num_workers, collate_fn=region_collate_fn)
        val_loader = DataLoader(val_loaddb, batch_size=self.cfg.batch_size, shuffle=False,
                                num_workers=self.cfg.num_workers, collate_fn=region_collate_fn)

        for epoch in range(self.epoch, self.cfg.n_epochs):
            torch.cuda.empty_cache()
            train_losses = self.train_epoch(train_loaddb, train_loader, epoch)

            val_losses = self.validate_epoch(val_loaddb, val_loader, epoch)

            current_val_loss = np.mean(val_losses)
            self.optimizer.update(current_val_loss, epoch)
            logging.info(
                'Time: {:.3f}, Epoch: {}, TrainAverageLoss: {:.3f}, ValAverageLoss: {:.3f}'.format(
                    time() - start, epoch,
                    np.mean(train_losses),
                    current_val_loss))
            self.save_checkpoint(epoch, self.cfg.exp_name)

    def test(self, test_db):
        all_img_feats, all_txt_feats, losses = [], [], []
        test_loaddb = region_loader(test_db)
        test_loader = DataLoader(test_loaddb, batch_size=self.cfg.batch_size, shuffle=False,
                                 num_workers=self.cfg.num_workers, collate_fn=region_collate_fn)
        self.net.eval()
        with torch.no_grad():
            for cnt, batched in enumerate(test_loader):
                sent_inds, sent_msks, region_feats = self.batch_data(batched)
                img_feats, txt_feats = self.net(sent_inds, sent_msks, region_feats)
                loss = self.net.triplet_loss(img_feats, txt_feats)
                loss = torch.mean(loss)
                losses.append(loss.cpu().item())
                all_img_feats.append(img_feats)
                all_txt_feats.append(txt_feats)

        all_img_feats = torch.cat(all_img_feats, 0)
        all_txt_feats = torch.cat(all_txt_feats, 0)
        ##################################################################
        ## Evaluation
        ##################################################################
        self.evaluate(all_img_feats.cpu().numpy(), all_txt_feats.cpu().numpy())

    def train_epoch(self, train_db, train_loader, epoch):
        losses = []
        self.net.train()
        for cnt, batched in enumerate(train_loader):
            sent_inds, sent_msks, region_feats = self.batch_data(batched)
            loss = self.net.compute_triplet_loss(sent_inds, sent_msks, region_feats)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.cpu().item())

            if cnt % self.cfg.log_per_steps == 0:
                logging.info('Epoch: {}, Iter: {}, Loss: {:.3f}'.format(epoch, cnt, loss.item()))
        return losses

    def validate_epoch(self, val_db, val_loader, epoch):
        all_img_feats, all_txt_feats, losses = [], [], []
        self.net.eval()
        for cnt, batched in enumerate(val_loader):
            sent_inds, sent_msks, region_feats = self.batch_data(batched)

            with torch.no_grad():
                img_feats, txt_feats = self.net(sent_inds, sent_msks, region_feats)
                batch_loss = self.net.triplet_loss(img_feats, txt_feats)
                loss = torch.mean(batch_loss)
            losses.append(loss.cpu().item())
            all_img_feats.append(img_feats)
            all_txt_feats.append(txt_feats)

            if cnt % self.cfg.log_per_steps == 0:
                print('Val Epoch %03d, iter %07d:' % (epoch, cnt))
                tmp_losses = np.stack(losses, 0)
                print('mean loss: ', np.mean(tmp_losses))
                print('-------------------------')

        losses = np.array(losses)
        all_img_feats = torch.cat(all_img_feats, 0)
        all_txt_feats = torch.cat(all_txt_feats, 0)
        ##################################################################
        ## Evaluation
        ##################################################################
        self.evaluate(all_img_feats.cpu().numpy(), all_txt_feats.cpu().numpy())

        return losses

    def evaluate(self, all_img_feats, all_txt_feats):
        start = time()
        sims = self.shard_sim(all_img_feats, all_txt_feats)
        end = time()
        logging.info('compute similarity time: {:.3f}'.format(end - start))
        num_sample = all_img_feats.shape[0]
        r, rt = self.i2t(num_sample, sims)
        ri, rti = self.t2i(num_sample, sims)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        logging.info("rsum: %.1f" % rsum)
        logging.info("Average i2t Recall: %.1f" % ar)
        logging.info("Image to text: %.1f %.1f %.1f %.1f %.1f %.1f %.1f" % r)
        logging.info("Average t2i Recall: %.1f" % ari)
        logging.info("Text to image: %.1f %.1f %.1f %.1f %.1f %.1f, %.1f" % ri)

    def shard_sim(self, all_img_feats, all_txt_feats, shard_size=128):
        n_img_shard = (all_img_feats.shape[0] - 1) // shard_size
        n_txt_shard = (all_txt_feats.shape[0] - 1) // shard_size
        d = np.zeros((all_img_feats.shape[0], all_txt_feats.shape[0]))

        for i in range(n_img_shard):
            im_start, im_end = shard_size * i, min(shard_size * (i + 1), all_img_feats.shape[0])
            for j in range(n_txt_shard):
                sys.stdout.write('\r>> shard sim batch (%d, %d)' % (i, j))
                txt_start, txt_end = shard_size * j, min(shard_size * (j + 1), all_txt_feats.shape[0])
                with torch.no_grad():
                    img_feat = torch.from_numpy(all_img_feats[im_start:im_end]).cuda()
                    txt_feat = torch.from_numpy(all_txt_feats[txt_start:txt_end]).cuda()
                    sim = self.net.compute_sim(img_feat, txt_feat)
                    d[im_start:im_end, txt_start:txt_end] = sim.t().cpu().numpy()
        return d

    def i2t(self, num_imgs, sims):
        ranks = np.zeros(num_imgs)
        top1 = np.zeros(num_imgs)
        for index in range(num_imgs):
            inds = np.argsort(sims[index])[::-1]
            rank = np.where(inds == index)[0][0]
            ranks[index] = rank
            top1[index] = inds[0]

        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        r20 = 100.0 * len(np.where(ranks < 20)[0]) / len(ranks)
        r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
        medr = np.floor(np.median(ranks)) + 1
        meanr = ranks.mean() + 1
        return (r1, r5, r10, r20, r100, medr, meanr), (ranks, top1)

    def t2i(self, num_txts, sims):
        sims = sims.T
        ranks = np.zeros(num_txts)
        top1 = np.zeros(num_txts)
        for index in range(num_txts):
            inds = np.argsort(sims[index])[::-1]
            rank = np.where(inds == index)[0][0]
            ranks[index] = rank
            top1[index] = inds[0]

        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        r20 = 100.0 * len(np.where(ranks < 20)[0]) / len(ranks)
        r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
        medr = np.floor(np.median(ranks)) + 1
        meanr = ranks.mean() + 1
        return (r1, r5, r10, r20, r100, medr, meanr), (ranks, top1)

    def load_pretrained_net(self, pretrained_path):
        assert (osp.exists(pretrained_path))
        states = torch.load(pretrained_path)
        self.net.load_state_dict(states['state_dict'], strict=True)
        self.optimizer.optimizer.load_state_dict(states['optimizer'])
        self.epoch = states['epoch']

    def save_checkpoint(self, epoch, model_name):
        print("Saving checkpoint...")
        checkpoint_dir = osp.join(self.cfg.model_dir, 'snapshots')
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        states = {
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.optimizer.state_dict()
        }
        torch.save(states, osp.join(checkpoint_dir, str(epoch) + '.pkl'))
