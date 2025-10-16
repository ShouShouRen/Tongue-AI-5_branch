# losses_asl.py - Asymmetric Loss for Multi-Label Classification
import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8, reduction="mean"):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, targets):
        prob = torch.sigmoid(logits)
        prob_pos = prob
        prob_neg = 1 - prob

        if self.clip and self.clip > 0:
            prob_neg = (prob_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(prob_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(prob_neg.clamp(min=self.eps))

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt_pos = prob_pos * targets
            pt_neg = prob_neg * (1 - targets)
            one_sided_w = (1 - pt_pos) ** self.gamma_pos * targets + \
                          (1 - pt_neg) ** self.gamma_neg * (1 - targets)
            loss = one_sided_w * (loss_pos + loss_neg)
        else:
            loss = loss_pos + loss_neg

        loss = -loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
