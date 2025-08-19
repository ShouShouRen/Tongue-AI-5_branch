# train.py — integrates ClassBalancedBCELoss + WeightedRandomSampler, keeps AMP
import os, gc, warnings
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch import optim
from tqdm import tqdm
import numpy as np
import pandas as pd

from dataset import TongueDataset
from model import SignOrientedNetwork
from losses import ClassBalancedBCELoss
from samplers import build_multilabel_weighted_sampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, jaccard_score

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
def evaluate_model(model, data_loader, device, label_cols, threshold_grid=(0.3,0.4,0.5,0.6,0.7)):
    model.eval()
    all_probs, all_labels, all_preds_default = [], [], []
    for (x_whole, x_body, x_edge), labels in data_loader:
        x_whole, x_body, x_edge = x_whole.to(device), x_body.to(device), x_edge.to(device)
        labels = labels.to(device)
        with autocast():
            out = model(x_whole, x_body, x_edge)['final']
            probs = torch.sigmoid(out).float().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())
        all_preds_default.append((probs>0.5).astype(int))
    if not all_probs:
        C = len(label_cols)
        return dict(f1_micro=0.0, f1_macro=0.0, f1_individual=[0.0]*C, jaccard_individual=[0.0]*C, jaccard_avg=0.0)
    P = np.vstack(all_probs); Y = np.vstack(all_labels); P05 = np.vstack(all_preds_default)

    f1_ind, jac_ind = [], []
    for i in range(len(label_cols)):
        if Y[:, i].sum() == 0:
            f1_ind.append(0.0); jac_ind.append(0.0); continue
        best_f1, best_j = 0.0, 0.0
        for th in threshold_grid:
            pred = (P[:, i] > th).astype(int)
            if pred.sum() == 0 and Y[:, i].sum() == 0:
                continue
            f1 = f1_score(Y[:, i], pred, zero_division=0)
            j = jaccard_score(Y[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_j = f1, j
        f1_ind.append(best_f1); jac_ind.append(best_j)

    f1_micro = f1_score(Y, P05, average='micro', zero_division=0)
    f1_macro = float(np.mean(f1_ind)); jaccard_avg = float(np.mean(jac_ind))
    return dict(f1_micro=f1_micro, f1_macro=f1_macro, f1_individual=f1_ind, jaccard_individual=jac_ind, jaccard_avg=jaccard_avg)


def train_model(train_csv, val_csv, image_root, label_cols, num_epochs=30, batch_size=12, lr=5e-5, model_path='signnet_best.pth', beta_cb=0.9999):
    clear_gpu_memory()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print_gpu_memory()

    # Datasets
    train_set = TongueDataset(train_csv, image_root, label_cols, is_train=True)
    val_set   = TongueDataset(val_csv,   image_root, label_cols, is_train=False)

    # Sampler for long-tail
    sampler, weights, n_pos_t, n_neg_t = build_multilabel_weighted_sampler(train_csv, label_cols, power=1.0)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=False, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,  num_workers=0, pin_memory=False, drop_last=False)

    # Model
    model = SignOrientedNetwork(num_classes=len(label_cols), backbone='swin_base_patch4_window7_224', feature_dim=512)
    model.to(device)

    # Class-Balanced BCE Loss
    criterion = ClassBalancedBCELoss(n_pos=n_pos_t.to(device), n_neg=n_neg_t.to(device), beta=beta_cb)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler()

    accumulation_steps = 2
    best_f1 = 0.0
    patience, patience_counter = 8, 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False)
        optimizer.zero_grad(set_to_none=True)

        for b, ((x_whole, x_body, x_edge), labels) in enumerate(loop):
            x_whole, x_body, x_edge = x_whole.to(device, non_blocking=True), x_body.to(device, non_blocking=True), x_edge.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast():
                out = model(x_whole, x_body, x_edge)['final']
                loss = criterion(out, labels) / accumulation_steps
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

        # Eval
        clear_gpu_memory()
        metrics = evaluate_model(model, val_loader, device, label_cols)
        avg_train_loss = total_loss / max(1, n_batches)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val F1 Micro: {metrics['f1_micro']:.4f}")
        print(f"Val F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"Val Jaccard Avg: {metrics['jaccard_avg']:.4f}")
        for i, lab in enumerate(label_cols):
            print(f"{lab}: F1={metrics['f1_individual'][i]:.3f}, Jaccard={metrics['jaccard_individual'][i]:.3f}")

        # Save best
        if metrics['f1_macro'] > best_f1:
            best_f1 = metrics['f1_macro']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_f1': best_f1,
                'metrics': metrics
            }, model_path)
            print(f"✅ Saved best model to {model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        clear_gpu_memory()

    return best_f1

if __name__ == '__main__':
    label_cols = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis', 'Crack', 'Toothmark', 'FurThick', 'FurYellow']
    for i in range(1, 5+1):
        train_csv = f'train_fold{i}.csv'
        val_csv   = f'val_fold{i}.csv'
        model_path = f'signnet_best_fold{i}.pth'
        print(f"\n====== Training Fold {i} ======")
        try:
            best_f1 = train_model(train_csv, val_csv, 'images', label_cols, num_epochs=30, batch_size=12, lr=5e-5, model_path=model_path, beta_cb=0.9999)
            print(f"Fold {i} Best F1: {best_f1:.4f}")
        except Exception as e:
            print(f"Fold {i} 訓練失敗: {str(e)}")
            print_gpu_memory(); clear_gpu_memory()
            continue
