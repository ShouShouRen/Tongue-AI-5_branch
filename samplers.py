# samplers.py — WeightedRandomSampler for multi-label long-tail
import torch
import pandas as pd
import numpy as np
from torch.utils.data import WeightedRandomSampler


def build_multilabel_weighted_sampler(csv_file: str, label_cols: list, power: float = 1.0):
    """
    Compute per-sample weights from per-class positive/negative frequencies.
    For each class c, get n_pos[c], n_neg[c]. The weight of a sample i is the
    average of inverse-frequencies of its labels: w_i = mean_c( 1/n_pos[c] if y_ic=1 else 1/n_neg[c] ).
    Optionally raise to `power` to control strength (power<1 softens; >1 strengthens).
    Returns: WeightedRandomSampler and the raw weights tensor.
    """
    df = pd.read_csv(csv_file)
    N = len(df)
    Y = df[label_cols].values.astype(np.float32)  # [N, C]
    n_pos = Y.sum(axis=0)  # [C]
    n_neg = N - n_pos      # [C]
    inv_pos = 1.0 / np.clip(n_pos, 1.0, None)
    inv_neg = 1.0 / np.clip(n_neg, 1.0, None)
    # per-sample weights
    W = (Y * inv_pos + (1.0 - Y) * inv_neg).mean(axis=1)  # [N]
    W = np.power(W, power)
    # normalize to mean=1 for stability
    W = W / (W.mean() + 1e-12)
    weights = torch.as_tensor(W, dtype=torch.double)
    sampler = WeightedRandomSampler(weights=weights, num_samples=N, replacement=True)
    # also return counts for loss
    n_pos_t = torch.as_tensor(n_pos, dtype=torch.float32)
    n_neg_t = torch.as_tensor(n_neg, dtype=torch.float32)
    return sampler, weights, n_pos_t, n_neg_t

