# losses.py — ClassBalancedBCELoss (pos/neg 分開加權，長尾友善)
import torch
import torch.nn as nn
import torch.nn.functional as F

def _effective_num_weights(n, beta=0.9999, eps=1e-8):
    """
    n: [C] 每類正/負樣本數
    回傳: [C] 對應的 class weight，n 越小，權重越大。
    """
    n = torch.clamp(n.float(), min=1.0)
    beta = torch.tensor(beta, dtype=torch.float32, device=n.device)
    weights = (1.0 - beta) / (1.0 - torch.pow(beta, n))
    weights = weights / (weights.mean() + eps)  # normalize
    return weights

class ClassBalancedBCELoss(nn.Module):
    """
    對多標籤 BCE：
      - 對每個類別的「正樣本」與「負樣本」分別用 effective number 產生權重
      - 避免只強調正樣本導致 precision 掉光
    參數：
      n_pos: [C] 每類別訓練集正樣本數
      n_neg: [C] 每類別訓練集負樣本數
      beta:  CB 的 beta，越靠近 1 越強調尾部
    """
    def __init__(self, n_pos, n_neg, beta=0.9999, reduction='mean'):
        super().__init__()
        assert n_pos.shape == n_neg.shape
        self.register_buffer('w_pos', _effective_num_weights(n_pos, beta))
        self.register_buffer('w_neg', _effective_num_weights(n_neg, beta))
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits, targets: [B, C]
        # 分別計算 pos/neg BCE，再以 per-class 權重線性合成
        loss_pos = F.binary_cross_entropy_with_logits(
            logits, torch.ones_like(targets), reduction='none'
        )  # 當作 y=1 的損失
        loss_neg = F.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(targets), reduction='none'
        )  # 當作 y=0 的損失

        # 權重展開成 [B, C]
        w_pos = self.w_pos.view(1, -1)
        w_neg = self.w_neg.view(1, -1)

        loss = targets * (w_pos * loss_pos) + (1.0 - targets) * (w_neg * loss_neg)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
