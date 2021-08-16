import torch
import torch.nn as nn
import torch.nn.functional as F

from models.image_encoder import ImageEncoder

class ImageAttributeModel(nn.Module):
    def __init__(self, config):
        super(ImageAttributeModel, self).__init__()
        self.cfg = config
        self.img_enc = ImageEncoder(self.cfg)
        # self.fc1 = nn.Linear(self.cfg.n_feature_dim, 36)
        self.fc1 = nn.Linear(self.cfg.n_feature_dim, 256)
        # self.fc2 = nn.Linear(36 * 36, self.cfg.n_categories)
        # self.fc2 = nn.Linear(36 * 128, self.cfg.n_categories)
        self.fc2 = nn.Linear(36 * 256, 256)
        self.fc3 = nn.Linear(256, self.cfg.n_categories)

    def forward(self, x):
        x = self.img_enc(x)
        x = self.fc1(x)
        x = x.view(x.shape[0], -1)
        x = self.fc2(x)
        x = self.fc3(x)
        return x