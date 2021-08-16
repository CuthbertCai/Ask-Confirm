import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score

from utils.optim import Optimizer
from utils.utils import *
from utils.loss import FocalLoss

from datasets.loader import region_loader, region_collate_fn
from models.image_attribute_model import ImageAttributeModel
from utils.vocab import Vocabulary


class ImageAttributeTrainer(object):
    def __init__(self, config):
        self.cfg = config
        self.net = ImageAttributeModel(self.cfg)
        if self.cfg.cuda:
            self.net = self.net.cuda()
        params = filter(lambda p: p.requires_grad, self.net.parameters())
        raw_optimizer = optim.Adam(params, lr=self.cfg.lr)
        optimizer = Optimizer(raw_optimizer, max_grad_norm=self.cfg.grad_norm_clipping)
        scheduler = optim.lr_scheduler.StepLR(raw_optimizer, step_size=20, gamma=0.1)
        optimizer.set_scheduler(scheduler)
        self.optimizer = optimizer
        self.epoch = 0
        if self.cfg.img_attr_pretrained is not None:
            self.load_pretrained_net(self.cfg.img_attr_pretrained)
        if self.cfg.focal_loss:
            self.criterion = FocalLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

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
        region_feats = entry['region_feats'].float()
        region_masks = entry['region_masks'].float()
        region_clses = entry['region_clses'].long()
        obj_inds = entry['objs_inds'].long()
        obj_inds = F.one_hot(obj_inds, num_classes=self.cfg.n_categories)
        obj_inds = torch.sum(obj_inds, dim=1).float()
        obj_inds[:, 0] = 0.0

        if self.cfg.cuda:
            region_feats = region_feats.cuda(non_blocking=True)
            region_masks = region_masks.cuda(non_blocking=True)
            region_clses = region_clses.cuda(non_blocking=True)
            obj_inds = obj_inds.cuda(non_blocking=True)
        return region_feats, region_masks, region_clses, obj_inds

    def train(self, train_db, val_db, test_db):
        start = time()
        min_val_loss = 1000.0
        max_val_recall = -1.0
        train_loaddb = region_loader(train_db)
        val_loaddb = region_loader(val_db)

        train_loader = DataLoader(train_loaddb, batch_size=self.cfg.batch_size, shuffle=True,
                                  num_workers=self.cfg.num_workers, collate_fn=region_collate_fn)
        val_loader = DataLoader(val_loaddb, batch_size=self.cfg.batch_size, shuffle=False,
                                num_workers=self.cfg.num_workers, collate_fn=region_collate_fn)

        for epoch in range(self.epoch, self.cfg.n_epochs):
            torch.cuda.empty_cache()
            train_losses = self.train_epoch(train_loaddb, train_loader, epoch)

            val_losses, val_acc = self.validate_epoch(val_loaddb, val_loader, epoch)

            current_val_loss = np.mean(val_losses)
            self.optimizer.update(current_val_loss, epoch)
            logging.info(
                'Time: {:.3f}, Epoch: {}, TrainAverageLoss: {:.3f}, ValAverageLoss: {:.3f}, ACC: {:.3f}'.format(
                    time() - start, epoch,
                    np.mean(train_losses),
                    current_val_loss, val_acc))
            self.save_checkpoint(epoch, self.cfg.exp_name)

    def test(self, test_db):
        start = time()
        losses = []
        corrects = 0.0
        precision = 0.0
        test_loaddb = region_loader(test_db)
        test_loader = DataLoader(test_loaddb, batch_size=self.cfg.batch_size, shuffle=False,
                                 num_workers=self.cfg.num_workers, collate_fn=region_collate_fn)
        self.net.eval()
        with torch.no_grad():
            for cnt, batched in enumerate(test_loader):
                region_feats, _, _, obj_inds = self.batch_data(batched)
                logits = self.net(region_feats)
                loss = self.criterion(logits, obj_inds)
                preds = torch.sigmoid(logits) > 0.8
                preds = preds.float()

                losses.append(loss.cpu().item())
                corrects += f1_score(obj_inds.cpu().numpy(), preds.cpu().numpy(), average='samples') * \
                            region_feats.shape[0]
                precision += precision_score(obj_inds.cpu().numpy(), preds.cpu().numpy(), average='samples') * \
                            region_feats.shape[0]
            epoch_acc = corrects / len(test_loader.dataset)
            epoch_precision = precision / len(test_loader.dataset)

        logging.info('Time: {:.3f}, ACC: {:.3f}, Precision: {:.3f}'.format(time() - start, epoch_acc, epoch_precision))

    def train_epoch(self, train_db, train_loader, epoch):
        losses = []
        self.net.train()
        for cnt, batched in enumerate(train_loader):
            region_feats, _, _, obj_inds = self.batch_data(batched)
            logits = self.net(region_feats)
            loss = self.criterion(logits, obj_inds)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.cpu().item())

            if cnt % self.cfg.log_per_steps == 0:
                logging.info('Epoch: {}, Iter: {}, Loss: {:.3f}'.format(epoch, cnt, loss.item()))
        return losses

    def validate_epoch(self, val_loaddb, val_loader, epoch):
        losses = []
        corrects = 0.0
        self.net.eval()
        with torch.no_grad():
            for cnt, batched in enumerate(val_loader):
                region_feats, _, _, obj_inds = self.batch_data(batched)
                logits = self.net(region_feats)
                loss = self.criterion(logits, obj_inds)
                preds = torch.sigmoid(logits) > 0.5
                preds = preds.float()

                losses.append(loss.cpu().item())
                corrects += f1_score(obj_inds.cpu().numpy(), preds.cpu().numpy(), average='samples') * \
                            region_feats.shape[0]

            epoch_acc = corrects / len(val_loader.dataset)
        return losses, epoch_acc

    def load_pretrained_net(self, pretrained_path):
        # cache_dir = osp.join(self.cfg.log_dir, model_name)
        # pretrained_path = osp.join(cache_dir, 'snapshots', str(epoch) + '.pkl')
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
