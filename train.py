# train.py — 五分支 + AMP + WeightedSampler + per-class thresholds
import os, gc, json, warnings
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, jaccard_score

from dataset import TongueDataset
from model import SignOrientedNetwork
from samplers import build_multilabel_weighted_sampler
from torch.cuda.amp import autocast, GradScaler

# 可切換 LOSS：'ASL' / 'CB_BCE' / 'CB_FOCAL'
LOSS_TYPE = os.environ.get("LOSS_TYPE", "ASL")

if LOSS_TYPE == "ASL":
    from losses_asl import AsymmetricLoss
elif LOSS_TYPE == "CB_BCE":
    from losses import ClassBalancedBCELoss
elif LOSS_TYPE == "CB_FOCAL":
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
            logits = model(x_whole, x_root, x_center, x_side, x_tip)  # [B, C]
            probs = torch.sigmoid(logits).float().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    if not all_probs:
        C = len(label_cols)
        zeros = [0.0] * C
        return (
            dict(f1_micro=0.0, f1_macro=0.0, f1_individual=zeros, jaccard_individual=zeros, jaccard_avg=0.0),
            dict(f1_micro=0.0, f1_macro=0.0, f1_individual=zeros, jaccard_individual=zeros, jaccard_avg=0.0),
            np.array([0.5]*C)
        )

    P = np.vstack(all_probs)  # [N, C]
    Y = np.vstack(all_labels) # [N, C]
    C = Y.shape[1]

    preds_05 = (P > 0.5).astype(int)
    f1_micro_05 = f1_score(Y, preds_05, average='micro', zero_division=0)

    best_thresholds = np.zeros(C, dtype=np.float32)
    best_f1_per_class = np.zeros(C, dtype=np.float32)
    best_jac_per_class = np.zeros(C, dtype=np.float32)

    for i in range(C):
        if Y[:, i].sum() == 0:
            best_thresholds[i] = 0.5
            best_f1_per_class[i] = 0.0
            best_jac_per_class[i] = 0.0
            continue
        best_f1 = -1.0
        best_j = 0.0
        best_th = 0.5
        for th in threshold_grid:
            pred = (P[:, i] > th).astype(int)
            f1 = f1_score(Y[:, i], pred, zero_division=0)
            j = jaccard_score(Y[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_j, best_th = f1, j, th
        best_thresholds[i] = best_th
        best_f1_per_class[i] = best_f1
        best_jac_per_class[i] = best_j

    preds_best = np.zeros_like(P, dtype=int)
    for i in range(C):
        preds_best[:, i] = (P[:, i] > best_thresholds[i]).astype(int)

    f1_micro_best = f1_score(Y, preds_best, average='micro', zero_division=0)
    f1_macro_best = float(np.mean(best_f1_per_class))
    jaccard_avg_best = float(np.mean(best_jac_per_class))

    f1_macro_05 = float(np.mean([f1_score(Y[:, i], preds_05[:, i], zero_division=0) for i in range(C)]))
    jaccard_avg_05 = float(np.mean([jaccard_score(Y[:, i], preds_05[:, i], zero_division=0) for i in range(C)]))

    metrics_at_best = dict(
        f1_micro=f1_micro_best,
        f1_macro=f1_macro_best,
        f1_individual=best_f1_per_class.tolist(),
        jaccard_individual=best_jac_per_class.tolist(),
        jaccard_avg=jaccard_avg_best,
    )
    metrics_at_05 = dict(
        f1_micro=f1_micro_05,
        f1_macro=f1_macro_05,
        f1_individual=[f1_score(Y[:, i], preds_05[:, i], zero_division=0) for i in range(C)],
        jaccard_individual=[jaccard_score(Y[:, i], preds_05[:, i], zero_division=0) for i in range(C)],
        jaccard_avg=jaccard_avg_05,
    )
    return metrics_at_best, metrics_at_05, best_thresholds

def train_model(train_csv, val_csv, image_root, label_cols,
                num_epochs=30, batch_size=12, lr=5e-5, model_path='signnet_best.pth',
                beta_cb=0.9999, sampler_power=1.0, backbone='swin_base_patch4_window7_224',
                feature_dim=512):
    clear_gpu_memory()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print_gpu_memory()

    # Dataset & Loader
    train_set = TongueDataset(train_csv, image_root, label_cols, is_train=True)
    val_set   = TongueDataset(val_csv,   image_root, label_cols, is_train=False)

    sampler, n_pos_t, n_neg_t = build_multilabel_weighted_sampler(train_csv, label_cols, power=sampler_power)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False, drop_last=False)

    # Model
    model = SignOrientedNetwork(num_classes=len(label_cols), backbone=backbone, feature_dim=feature_dim)
    model.to(device)

    # Loss
    if LOSS_TYPE == "ASL":
        criterion = AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05)
    elif LOSS_TYPE == "CB_BCE":
        criterion = ClassBalancedBCELoss(n_pos=n_pos_t.to(device), n_neg=n_neg_t.to(device), beta=beta_cb)
    elif LOSS_TYPE == "CB_FOCAL":
        criterion = ClassBalancedFocalLoss(n_pos=n_pos_t.to(device), n_neg=n_neg_t.to(device),
                                           beta=beta_cb, gamma=2.0)
    else:
        raise ValueError("Unknown LOSS_TYPE")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler()

    accumulation_steps = 2
    best_macro_f1 = -1.0
    patience, patience_counter = 8, 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False)
        optimizer.zero_grad(set_to_none=True)

        for b, ((x_whole, x_root, x_center, x_side, x_tip), labels) in enumerate(loop):
            x_whole  = x_whole.to(device, non_blocking=True)
            x_root   = x_root.to(device, non_blocking=True)
            x_center = x_center.to(device, non_blocking=True)
            x_side   = x_side.to(device, non_blocking=True)
            x_tip    = x_tip.to(device, non_blocking=True)
            labels   = labels.to(device, non_blocking=True)

            with autocast():
                logits = model(x_whole, x_root, x_center, x_side, x_tip)  # [B, C]
                loss = criterion(logits, labels) / accumulation_steps

            scaler.scale(loss).backward()

            if (b + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * accumulation_steps
            n_batches += 1
            loop.set_postfix(loss=total_loss / max(1, n_batches))

        scheduler.step()

        # Eval + 門檻
        clear_gpu_memory()
        metrics_best, metrics_05, best_thresholds = evaluate_model_and_find_thresholds(
            model, val_loader, device, label_cols
        )
        avg_train_loss = total_loss / max(1, n_batches)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"[Best-TH] Val F1 Micro: {metrics_best['f1_micro']:.4f} | Macro: {metrics_best['f1_macro']:.4f} | Jaccard: {metrics_best['jaccard_avg']:.4f}")
        print(f"[TH=0.5] Val F1 Micro: {metrics_05['f1_micro']:.4f} | Macro: {metrics_05['f1_macro']:.4f} | Jaccard: {metrics_05['jaccard_avg']:.4f}")
        for i, lab in enumerate(label_cols):
            print(f"{lab}: (BestTH F1={metrics_best['f1_individual'][i]:.3f} Jac={metrics_best['jaccard_individual'][i]:.3f})"
                  f" | (0.5 F1={metrics_05['f1_individual'][i]:.3f})  TH*={best_thresholds[i]:.2f}")

        # Save best (macro-F1@BestTH)
        if metrics_best['f1_macro'] > best_macro_f1:
            best_macro_f1 = metrics_best['f1_macro']
            state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_macro_f1': best_macro_f1,
                'metrics_best': metrics_best,
                'metrics_05': metrics_05,
                'label_cols': label_cols,
                'best_thresholds': best_thresholds.tolist(),
            }
            torch.save(state, model_path)
            with open(os.path.splitext(model_path)[0] + "_best_thresholds.json", "w", encoding="utf-8") as f:
                json.dump({label_cols[i]: float(best_thresholds[i]) for i in range(len(label_cols))}, f, ensure_ascii=False, indent=2)
            print(f"✅ Saved best model + thresholds to {model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        clear_gpu_memory()

    return best_macro_f1

if __name__ == '__main__':
    label_cols = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis',
                  'Crack', 'Toothmark', 'FurThick', 'FurYellow']
    for i in range(1, 5 + 1):
        train_csv = f'train_fold{i}.csv'
        val_csv   = f'val_fold{i}.csv'
        model_path = f'signnet_best_fold{i}.pth'
        print(f"\n====== Training Fold {i} ======")
        try:
            best_f1 = train_model(
                train_csv, val_csv, 'images', label_cols,
                num_epochs=30, batch_size=12, lr=5e-5,
                model_path=model_path, beta_cb=0.9999, sampler_power=1.0,
                backbone='swin_base_patch4_window7_224', feature_dim=512
            )
            print(f"Fold {i} Best Macro-F1@BestTH: {best_f1:.4f}")
        except Exception as e:
            print(f"Fold {i} 訓練失敗: {str(e)}")
            print_gpu_memory()
            clear_gpu_memory()
            continue
