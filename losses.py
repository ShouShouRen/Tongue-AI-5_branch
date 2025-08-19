# losses.py — Class-Balanced BCE for multi-label, numerically stable
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassBalancedBCELoss(nn.Module):
    """
    Implements Class-Balanced Loss (Cui et al., 2019) for multi-label BCE.
    For each class c, compute effective numbers for positives/negatives separately:
        E_pos[c] = (1 - beta) / (1 - beta**n_pos[c])
        E_neg[c] = (1 - beta) / (1 - beta**n_neg[c])
    We weight the per-class BCE terms proportionally to E_* and rescale to keep magnitudes reasonable.
    Args:
        n_pos (Tensor[C]): count of positives per class
        n_neg (Tensor[C]): count of negatives per class
        beta (float): typically 0.999 or 0.9999 depending on dataset size
        eps (float): numerical stability
    """
    def __init__(self, n_pos: torch.Tensor, n_neg: torch.Tensor, beta: float = 0.9999, eps: float = 1e-8):
        super().__init__()
        assert n_pos.ndim == 1 and n_neg.ndim == 1 and n_pos.shape == n_neg.shape
        self.register_buffer('n_pos', n_pos.float().clamp(min=1.0))
        self.register_buffer('n_neg', n_neg.float().clamp(min=1.0))
        self.beta = beta
        self.eps = eps
        # Precompute class weights
        self.register_buffer('w_pos', self._effective_weight(self.n_pos))
        self.register_buffer('w_neg', self._effective_weight(self.n_neg))
        # Normalize weights so the mean weight per class ~ 1.0
        mean_w = (self.w_pos + self.w_neg).mean() / 2.0
        self.w_pos = self.w_pos / (mean_w + eps)
        self.w_neg = self.w_neg / (mean_w + eps)

    def _effective_weight(self, n: torch.Tensor) -> torch.Tensor:
        beta = self.beta
        return (1.0 - beta) / (1.0 - torch.pow(beta, n))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # logits, targets: [B, C]
        logits = logits.clamp(min=-10.0, max=10.0)
        loss_pos = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # [B,C]
        # Split weights per element according to target label
        w = targets * self.w_pos + (1.0 - targets) * self.w_neg  # [C] broadcast to [B,C]
        loss = (loss_pos * w).mean()
        return loss

class FocalBCELoss(nn.Module):
    """ Optional multi-label Focal BCE with logits. """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, eps: float = 1e-8):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        logits = logits.clamp(min=-10.0, max=10.0)
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - p_t).pow(self.gamma) * ce
        return focal.mean()
