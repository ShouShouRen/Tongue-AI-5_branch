# train.py — single-encoder ROI branches + CB-BCE + WeightedRandomSampler + AMP + EMA + per-class thresholds + TTA
import os, gc, json, warnings
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, jaccard_score

from dataset import TongueDataset
from model import SignOrientedNetwork
from losses import ClassBalancedBCELoss
from samplers import build_multilabel_weighted_sampler
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


# ---------------- EMA ----------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def copy_to(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

    # with ema.average_parameters(model): ...
    def average_parameters(self, model):
        class _Ctx:
            def __init__(self, ema, model): self.ema, self.model = ema, model
            def __enter__(self): self.ema.copy_to(self.model)
            def __exit__(self, exc_type, exc, tb): self.ema.restore(self.model)
        return _Ctx(self, model)


def print_gpu_memory():
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {a:.2f} GB allocated / {r:.2f} GB reserved")


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# -------------- Eval + Thresholds --------------
@torch.no_grad()
def evaluate_model_and_find_thresholds(model, data_loader, device, label_cols, threshold_grid=None, use_tta=True):
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
            logits0 = model(x_whole, x_root, x_center, x_side, x_tip)
            p0 = torch.sigmoid(logits0)

            if use_tta:
                x_whole_f  = torch.flip(x_whole,  dims=[3])
                x_root_f   = torch.flip(x_root,   dims=[3])
                x_center_f = torch.flip(x_center, dims=[3])
                x_side_f   = torch.flip(x_side,   dims=[3])
                x_tip_f    = torch.flip(x_tip,    dims=[3])
                logits1 = model(x_whole_f, x_root_f, x_center_f, x_side_f, x_tip_f)
                p1 = torch.sigmoid(logits1)
                probs = ((p0 + p1) * 0.5).float().cpu().numpy()
            else:
                probs = p0.float().cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    if not all_probs:
        C = len(label_cols); zeros = [0.0]*C
        return (
            dict(f1_micro=0.0, f1_macro=0.0, f1_individual=zeros, jaccard_individual=zeros, jaccard_avg=0.0),
            dict(f1_micro=0.0, f1_macro=0.0, f1_individual=zeros, jaccard_individual=zeros, jaccard_avg=0.0),
            np.array([0.5]*C)
        )

    P = np.vstack(all_probs)  # [N,C]
    Y = np.vstack(all_labels) # [N,C]
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
        best_f1, best_j, best_th = -1.0, 0.0, 0.5
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


# -------------- Train --------------
def train_model(train_csv, val_csv, image_root, label_cols,
                num_epochs=30, batch_size=12,
                lr_head=1e-4, lr_backbone=1e-5,
                model_path='signnet_best.pth',
                beta_cb=0.9999, sampler_power=0.7,
                warmup_epochs=1):
    clear_gpu_memory()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print_gpu_memory()

    # datasets
    train_set = TongueDataset(train_csv, image_root, label_cols, is_train=True, rotate_upright_enabled=False)
    val_set   = TongueDataset(val_csv,   image_root, label_cols, is_train=False, rotate_upright_enabled=False)

    # sampler
    sampler, n_pos_t, n_neg_t = build_multilabel_weighted_sampler(train_csv, label_cols, power=sampler_power)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False, drop_last=False)

    # model (注意：這版 model 只接受以下參數，且 forward 只回傳 logits)
    model = SignOrientedNetwork(
        num_classes=len(label_cols),
        backbone='convnext_base',   # 原本 'swin_base_patch4_window7_224'
        feature_dim=768,            # 建議放大容量；保守可用 512
        dropout=0.1
    ).to(device)
    ema = EMA(model, decay=0.999)

    # warmup: freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    # loss
    criterion = ClassBalancedBCELoss(n_pos=n_pos_t.to(device), n_neg=n_neg_t.to(device), beta=beta_cb)

    # optimizer: head first
    head_params = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
    optimizer = optim.AdamW(head_params, lr=lr_head, weight_decay=1e-4, eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
    scaler = GradScaler()

    accumulation_steps = 2
    best_macro_f1 = -1.0
    patience, patience_counter = 8, 0

    for epoch in range(num_epochs):
        # unfreeze after warmup
        if epoch == warmup_epochs:
            for p in model.encoder.parameters():
                p.requires_grad = True
            backbone_params = [p for n, p in model.named_parameters() if n.startswith('encoder.')]
            head_params = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
            optimizer = optim.AdamW(
                [{'params': backbone_params, 'lr': lr_backbone},
                 {'params': head_params,    'lr': lr_head}],
                weight_decay=1e-4, eps=1e-8
            )
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

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
                logits = model(x_whole, x_root, x_center, x_side, x_tip)  # 只回傳最終 logits
                loss = criterion(logits, labels) / accumulation_steps

            scaler.scale(loss).backward()

            if (b + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)

            total_loss += float(loss.item()) * accumulation_steps
            n_batches += 1
            loop.set_postfix(loss=total_loss / max(1, n_batches))

        scheduler.step()

        # eval (+ EMA weights + TTA)
        clear_gpu_memory()
        with ema.average_parameters(model):
            metrics_best, metrics_05, best_thresholds = evaluate_model_and_find_thresholds(
                model, val_loader, device, label_cols, use_tta=True
            )
        avg_train_loss = total_loss / max(1, n_batches)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"[Best-TH] Val F1 Micro: {metrics_best['f1_micro']:.4f} | Macro: {metrics_best['f1_macro']:.4f} | Jaccard: {metrics_best['jaccard_avg']:.4f}")
        print(f"[TH=0.5] Val F1 Micro: {metrics_05['f1_micro']:.4f} | Macro: {metrics_05['f1_macro']:.4f} | Jaccard: {metrics_05['jaccard_avg']:.4f}")
        for i, lab in enumerate(label_cols):
            print(f"{lab}: (BestTH F1={metrics_best['f1_individual'][i]:.3f} Jac={metrics_best['jaccard_individual'][i]:.3f})"
                  f" | (0.5 F1={metrics_05['f1_individual'][i]:.3f})  TH*={best_thresholds[i]:.2f}")

        # save best by macro-F1@BestTH
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
                num_epochs=30, batch_size=12,
                lr_head=1e-4, lr_backbone=1e-5,
                model_path=model_path, beta_cb=0.9999, sampler_power=0.7, warmup_epochs=1
            )
            print(f"Fold {i} Best Macro-F1@BestTH: {best_f1:.4f}")
        except Exception as e:
            print(f"Fold {i} 訓練失敗: {str(e)}")
            print_gpu_memory()
            clear_gpu_memory()
            continue
