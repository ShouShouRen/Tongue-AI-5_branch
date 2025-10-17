# train.py — 五分支 + AMP + WeightedSampler + per-class thresholds
# --- 已修改版：在5-Fold後進行最終訓練，並為最終模型產生平均最佳門檻 ---

import os, gc, json, warnings
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, jaccard_score, accuracy_score

from dataset import TongueDataset
from model import SignOrientedNetwork
from samplers import build_multilabel_weighted_sampler
from torch.cuda.amp import autocast, GradScaler

# 可切換 LOSS：'ASL' / 'CB_BCE' / 'CB_FOCAL'
LOSS_TYPE = os.environ.get("LOSS_TYPE", "ASL")

if LOSS_TYPE == "ASL":
    # 假設您有 losses_asl.py
    from losses_asl import AsymmetricLoss
elif LOSS_TYPE == "CB_BCE":
    from losses import ClassBalancedBCELoss
elif LOSS_TYPE == "CB_FOCAL":
    # 假設您有 losses_cb_focal.py
    from losses_cb_focal import ClassBalancedFocalLoss
else:
    raise ValueError("LOSS_TYPE must be one of: ASL, CB_BCE, CB_FOCAL")

warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f} GB allocated / {reserved:.2f} GB reserved")

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

@torch.no_grad()
def evaluate_model_and_find_thresholds(model, data_loader, device, label_cols, threshold_grid=None):
    if threshold_grid is None:
        threshold_grid = np.concatenate([np.linspace(0.1, 0.9, 17), np.array([0.95])])

    model.eval()
    all_probs, all_labels = [], []

    for (x_whole, x_root, x_center, x_side, x_tip), labels in data_loader:
        x_whole  = x_whole.to(device)
        x_root   = x_root.to(device)
        x_center = x_center.to(device)
        x_side   = x_side.to(device)
        x_tip    = x_tip.to(device)
        labels   = labels.to(device)

        with autocast():
            logits = model(x_whole, x_root, x_center, x_side, x_tip)
            probs = torch.sigmoid(logits).float().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    if not all_probs:
        C = len(label_cols)
        zeros = [0.0] * C
        return (
            dict(acc=0.0, f1_micro=0.0, f1_macro=0.0, f1_individual=zeros, jaccard_individual=zeros, jaccard_avg=0.0),
            dict(acc=0.0, f1_micro=0.0, f1_macro=0.0, f1_individual=zeros, jaccard_individual=zeros, jaccard_avg=0.0),
            np.array([0.5]*C)
        )

    P = np.vstack(all_probs)
    Y = np.vstack(all_labels)
    C = Y.shape[1]

    preds_05 = (P > 0.5).astype(int)
    acc_05 = accuracy_score(Y, preds_05)

    best_thresholds = np.zeros(C, dtype=np.float32)
    best_f1_per_class = np.zeros(C, dtype=np.float32)
    best_jac_per_class = np.zeros(C, dtype=np.float32)

    for i in range(C):
        if Y[:, i].sum() == 0:
            best_thresholds[i], best_f1_per_class[i], best_jac_per_class[i] = 0.5, 0.0, 0.0
            continue
        best_f1, best_j, best_th = -1.0, 0.0, 0.5
        for th in threshold_grid:
            pred = (P[:, i] > th).astype(int)
            f1 = f1_score(Y[:, i], pred, zero_division=0)
            j = jaccard_score(Y[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_j, best_th = f1, j, th
        best_thresholds[i], best_f1_per_class[i], best_jac_per_class[i] = best_th, best_f1, best_j

    preds_best = np.array([P[:, i] > best_thresholds[i] for i in range(C)]).T
    acc_best = accuracy_score(Y, preds_best)

    metrics_at_best = dict(
        acc=acc_best, f1_micro=f1_score(Y, preds_best, average='micro', zero_division=0),
        f1_macro=np.mean(best_f1_per_class).item(), f1_individual=best_f1_per_class.tolist(),
        jaccard_individual=best_jac_per_class.tolist(), jaccard_avg=np.mean(best_jac_per_class).item(),
    )
    metrics_at_05 = dict(
        acc=acc_05, f1_micro=f1_score(Y, preds_05, average='micro', zero_division=0),
        f1_macro=f1_score(Y, preds_05, average='macro', zero_division=0),
        f1_individual=[f1_score(Y[:, i], preds_05[:, i], zero_division=0) for i in range(C)],
        jaccard_individual=[jaccard_score(Y[:, i], preds_05[:, i], zero_division=0) for i in range(C)],
        jaccard_avg=jaccard_score(Y, preds_05, average='samples', zero_division=0),
    )
    return metrics_at_best, metrics_at_05, best_thresholds

def train_model(train_csv, val_csv, image_root, label_cols, num_epochs, batch_size, lr, model_path, **kwargs):
    clear_gpu_memory()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_set = TongueDataset(train_csv, image_root, label_cols, is_train=True)
    sampler, n_pos_t, n_neg_t = build_multilabel_weighted_sampler(train_csv, label_cols, power=kwargs.get('sampler_power', 1.0))
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=False, drop_last=True)
    
    val_loader = None
    if val_csv:
        val_set = TongueDataset(val_csv, image_root, label_cols, is_train=False)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model = SignOrientedNetwork(num_classes=len(label_cols), backbone=kwargs.get('backbone'), feature_dim=kwargs.get('feature_dim')).to(device)

    if LOSS_TYPE == "ASL": criterion = AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05)
    else: criterion = ClassBalancedBCELoss(n_pos=n_pos_t.to(device), n_neg=n_neg_t.to(device), beta=kwargs.get('beta_cb'))

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler()
    best_macro_f1, patience, patience_counter = -1.0, 8, 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False)
        
        for b, ((x_whole, x_root, x_center, x_side, x_tip), labels) in enumerate(loop):
            # ... training step logic ...
            with torch.cuda.amp.autocast():
                logits = model(x_whole.to(device), x_root.to(device), x_center.to(device), x_side.to(device), x_tip.to(device))
                loss = criterion(logits, labels.to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()
            n_batches += 1
            loop.set_postfix(loss=total_loss / n_batches)

        scheduler.step()

        if val_loader:
            metrics_best, _, best_thresholds = evaluate_model_and_find_thresholds(model, val_loader, device, label_cols)
            print(f"\nEpoch {epoch+1}/{num_epochs} | Train Loss: {total_loss/n_batches:.4f} | Val F1 Macro: {metrics_best['f1_macro']:.4f}")

            if metrics_best['f1_macro'] > best_macro_f1:
                best_macro_f1 = metrics_best['f1_macro']
                state = {'model_state_dict': model.state_dict(), 'best_thresholds': best_thresholds.tolist(), 'label_cols': label_cols}
                torch.save(state, model_path)
                th_path = os.path.splitext(model_path)[0] + "_best_thresholds.json"
                with open(th_path, "w", encoding="utf-8") as f:
                    json.dump({label: float(th) for label, th in zip(label_cols, best_thresholds)}, f, indent=2)
                print(f"✅ Saved best model to {model_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    if not val_loader:
        torch.save({'model_state_dict': model.state_dict(), 'label_cols': label_cols}, model_path)
        print(f"✅ Saved final model to {model_path}")

    return best_macro_f1

def combine_kfold_csvs(k, output_path):
    all_dfs = [pd.read_csv(f) for i in range(1, k + 1) for f in [f'train_fold{i}.csv', f'val_fold{i}.csv'] if os.path.exists(f)]
    if not all_dfs: return None
    pd.concat(all_dfs).drop_duplicates().reset_index(drop=True).to_csv(output_path, index=False)
    print(f"✔ Combined CSV saved to {output_path}")
    return output_path

if __name__ == '__main__':
    PARAMS = {
        'label_cols': ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis', 'Crack', 'Toothmark', 'FurThick', 'FurYellow'],
        'num_epochs': 30, 'batch_size': 12, 'lr': 5e-5, 'beta_cb': 0.9999, 'sampler_power': 1.0,
        'backbone': 'swin_base_patch4_window7_224', 'feature_dim': 512, 'image_root': 'images'
    }
    NUM_FOLDS = 5

    print("="*20 + "\n  PHASE 1: K-Fold Cross-Validation\n" + "="*20)
    for i in range(1, NUM_FOLDS + 1):
        if not all(os.path.exists(f) for f in [f'train_fold{i}.csv', f'val_fold{i}.csv']): continue
        print(f"\n====== Training Fold {i} ======")
        train_model(f'train_fold{i}.csv', f'val_fold{i}.csv', model_path=f'signnet_best_fold{i}.pth', **PARAMS)

    print("\n" + "="*20 + "\n  PHASE 2: Final Training on Full Data\n" + "="*20)
    full_train_csv = combine_kfold_csvs(NUM_FOLDS, 'train_full.csv')
    final_model_path = 'signnet_final_model.pth'
    if full_train_csv:
        train_model(full_train_csv, None, model_path=final_model_path, **PARAMS)

    print("\n" + "="*20 + "\n  PHASE 3: Averaging Thresholds for Final Model\n" + "="*20)
    all_thresholds = []
    for i in range(1, NUM_FOLDS + 1):
        th_path = f'signnet_best_fold{i}_best_thresholds.json'
        if os.path.exists(th_path):
            with open(th_path, 'r', encoding='utf-8') as f:
                fold_ths = json.load(f)
                all_thresholds.append([fold_ths.get(label, 0.5) for label in PARAMS['label_cols']])
                print(f"Loaded thresholds from {th_path}")
    
    if all_thresholds:
        avg_thresholds = {label: float(th) for label, th in zip(PARAMS['label_cols'], np.mean(all_thresholds, axis=0))}
        final_th_path = os.path.splitext(final_model_path)[0] + "_best_thresholds.json"
        with open(final_th_path, 'w', encoding='utf-8') as f:
            json.dump(avg_thresholds, f, indent=2)
        print(f"\n✅ Averaged thresholds saved to: {final_th_path}")
        for label, th in avg_thresholds.items():
            print(f"  - {label:<15}: {th:.4f}")
