import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        logits = torch.sigmoid(logits)
        pos_sub = 1.0 - logits
        pos_sub = pos_sub * target

        neg_sub = logits * (1.0 - target)

        loss = -self.alpha * (pos_sub ** self.gamma) * torch.log(logits + 1e-8) - (1 - self.alpha) * (
                    neg_sub ** self.gamma) * torch.log(1.0 - logits + 1e-8)

        return loss.mean()
