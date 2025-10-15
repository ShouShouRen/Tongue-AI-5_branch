# losses_asl.py - Asymmetric Loss for Multi-Label Classification
import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-label classification
    Ref: https://arxiv.org/abs/2009.14119

    - 適合長尾、多標籤任務
    - gamma_pos: 聚焦難分類的正樣本 (通常 0~2)
    - gamma_neg: 聚焦難分類的負樣本 (通常 2~4)
    - clip: 負樣本概率下限 (避免太小)
    """
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8, reduction="mean"):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [B, C] raw output
        targets: [B, C] multi-hot labels
        """
        prob = torch.sigmoid(logits)
        prob_pos = prob
        prob_neg = 1 - prob

        # clip 負樣本，避免梯度消失
        if self.clip and self.clip > 0:
            prob_neg = (prob_neg + self.clip).clamp(max=1.0)

        # Cross Entropy
        loss_pos = targets * torch.log(prob_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(prob_neg.clamp(min=self.eps))

        # Focal 部分
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt_pos = prob_pos * targets
            pt_neg = prob_neg * (1 - targets)
            one_sided_w = (1 - pt_pos) ** self.gamma_pos * targets + \
                          (1 - pt_neg) ** self.gamma_neg * (1 - targets)
            loss = one_sided_w * (loss_pos + loss_neg)
        else:
            loss = loss_pos + loss_neg

        loss = -loss  # cross-entropy 是負的 log likelihood

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
