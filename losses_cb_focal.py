# losses_cb_focal.py - Class-Balanced Focal Loss
import torch
import torch.nn as nn

def _effective_num_weights(n, beta=0.9999, eps=1e-8):
    n = torch.clamp(n.float(), min=1.0)
    beta = torch.tensor(beta, dtype=torch.float32, device=n.device)
    weights = (1.0 - beta) / (1.0 - torch.pow(beta, n))
    weights = weights / (weights.mean() + eps)
    return weights

class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, n_pos, n_neg, beta=0.9999, gamma=2.0, reduction="mean"):
        super().__init__()
        assert n_pos.shape == n_neg.shape
        self.register_buffer("w_pos", _effective_num_weights(n_pos, beta))
        self.register_buffer("w_neg", _effective_num_weights(n_neg, beta))
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        prob = torch.sigmoid(logits)
        prob_pos = prob
        prob_neg = 1 - prob

        focal_pos = (1 - prob_pos) ** self.gamma
        focal_neg = (prob_pos) ** self.gamma

        loss_pos = -targets * self.w_pos.view(1,-1) * focal_pos * torch.log(prob_pos.clamp(min=1e-8))
        loss_neg = -(1 - targets) * self.w_neg.view(1,-1) * focal_neg * torch.log(prob_neg.clamp(min=1e-8))
        loss = loss_pos + loss_neg

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
