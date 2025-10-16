# losses.py — ClassBalancedBCELoss (pos/neg 分開加權)
import torch
import torch.nn as nn
import torch.nn.functional as F

def _effective_num_weights(n, beta=0.9999, eps=1e-8):
    n = torch.clamp(n.float(), min=1.0)
    beta = torch.tensor(beta, dtype=torch.float32, device=n.device)
    weights = (1.0 - beta) / (1.0 - torch.pow(beta, n))
    weights = weights / (weights.mean() + eps)
    return weights

class ClassBalancedBCELoss(nn.Module):
    def __init__(self, n_pos, n_neg, beta=0.9999, reduction='mean'):
        super().__init__()
        assert n_pos.shape == n_neg.shape
        self.register_buffer('w_pos', _effective_num_weights(n_pos, beta))
        self.register_buffer('w_neg', _effective_num_weights(n_neg, beta))
        self.reduction = reduction

    def forward(self, logits, targets):
        loss_pos = F.binary_cross_entropy_with_logits(
            logits, torch.ones_like(targets), reduction='none'
        )
        loss_neg = F.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(targets), reduction='none'
        )
        w_pos = self.w_pos.view(1, -1)
        w_neg = self.w_neg.view(1, -1)
        loss = targets * (w_pos * loss_pos) + (1.0 - targets) * (w_neg * loss_neg)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
