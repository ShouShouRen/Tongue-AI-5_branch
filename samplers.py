# samplers.py — WeightedRandomSampler for multilabel long-tail
import torch
import pandas as pd
import numpy as np
from torch.utils.data import WeightedRandomSampler

def _effective_num(n, beta=0.9999):
    n = np.clip(n.astype(np.float64), 1.0, None)
    weights = (1.0 - beta) / (1.0 - np.power(beta, n))
    weights = weights / weights.mean()
    return weights  # n 小 → 權重大

def build_multilabel_weighted_sampler(train_csv, label_cols, power=1.0, beta=0.9999):
    """
    讀取 training CSV，依多標籤稀有度給每張圖 sample weight。
    sample_weight(i) = sum_c [ w_c * y_ic ]，若全 0，給最小權重。
    回傳：sampler, n_pos_t, n_neg_t（供 loss 使用）
    """
    df = pd.read_csv(train_csv)
    Y = df[label_cols].values.astype(np.float32)  # [N, C]
    N, C = Y.shape

    n_pos = Y.sum(axis=0)                      # [C]
    n_neg = N - n_pos
    w_c = _effective_num(n_pos, beta=beta)     # [C] 小眾類較大

    sample_w = np.matmul(Y, w_c.reshape(-1, 1)).squeeze(1)  # [N]
    min_nonzero = np.minimum(sample_w[sample_w > 0].min() if (sample_w > 0).any() else 1.0, 1.0)
    sample_w = np.where(sample_w > 0, sample_w, min_nonzero * 0.1)

    sample_w = np.power(sample_w, power)
    sample_w = sample_w / sample_w.mean()

    sampler = WeightedRandomSampler(weights=torch.tensor(sample_w, dtype=torch.float32),
                                    num_samples=N, replacement=True)

    n_pos_t = torch.tensor(n_pos, dtype=torch.float32)
    n_neg_t = torch.tensor(n_neg, dtype=torch.float32)
    return sampler, n_pos_t, n_neg_t
