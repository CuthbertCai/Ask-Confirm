import torch
import torch.nn as nn

from utils.utils import *

class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()
        self.cfg = config
        self.project = nn.Sequential(nn.Linear(2048, self.cfg.n_feature_dim))
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, region_feats):
        img_feats = self.project(region_feats)
        return img_feats