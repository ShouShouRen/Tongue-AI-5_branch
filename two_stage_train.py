# train_two_stage.py — Two-Stage (cRT) for your SignOrientedNetwork
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

# --------------- Utils ---------------
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

    for (x_whole, x_body, x_edge), labels in data_loader:
        x_whole, x_body, x_edge = x_whole.to(device), x_body.to(device), x_edge.to(device)
        labels = labels.to(device)
        with autocast():
            logits = model(x_whole, x_body, x_edge)['final']
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

# --------------- Stage helpers ---------------
def make_loaders(train_csv, val_csv, image_root, label_cols, batch_size, sampler_power):
    train_set = TongueDataset(train_csv, image_root, label_cols, is_train=True)
    val_set   = TongueDataset(val_csv,   image_root, label_cols, is_train=False)
    sampler, n_pos_t, n_neg_t = build_multilabel_weighted_sampler(train_csv, label_cols, power=sampler_power)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False, drop_last=False)
    return train_loader, val_loader, n_pos_t, n_neg_t

def freeze_backbones(model, freeze_feature_proj=True):
    """
    凍結三個分支的 encoder（以及可選的 feature_proj），保留注意力與各個 classifier 可訓練。
    """
    branches = [model.whole_branch, model.body_branch, model.edge_branch]
    for br in branches:
        for p in br.encoder.parameters():
            p.requires_grad = False
        if freeze_feature_proj:
            for p in br.feature_proj.parameters():
                p.requires_grad = False

def classifier_parameters(model):
    """
    僅回傳需訓練的 head/attention 參數。
    """
    params = []
    for m in [model.attention_body, model.attention_edge,
              model.body_specific, model.edge_specific,
              model.global_classifier, model.color_classifier, model.fur_classifier]:
        params += list(m.parameters())
    # 若選擇不凍結 feature_proj，也把它們加入
    for br in [model.whole_branch, model.body_branch, model.edge_branch]:
        for p in br.feature_proj.parameters():
            if p.requires_grad:
                params.append(p)
    return params

def reinit_classifier_last_layers(model):
    """
    可選：重新初始化分類器最後的線性層以減少 Stage1 的偏置。
    """
    def reset_linear(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    for head in [model.body_specific, model.edge_specific,
                 model.global_classifier, model.color_classifier, model.fur_classifier]:
        # 只重設最後一層 Linear
        if isinstance(head[-1], torch.nn.Linear):
            reset_linear(head[-1])

# --------------- Training loops ---------------
def run_stage(train_loader, val_loader, model, criterion, optimizer, scheduler, device,
              label_cols, num_epochs, scaler, save_path, early_patience=8, tag='Stage'):
    accumulation_steps = 2
    best_macro_f1 = -1.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        loop = tqdm(train_loader, desc=f"{tag} Epoch {epoch+1}/{num_epochs}", leave=False)
        optimizer.zero_grad(set_to_none=True)

        for b, ((x_whole, x_body, x_edge), labels) in enumerate(loop):
            x_whole = x_whole.to(device, non_blocking=True)
            x_body  = x_body.to(device, non_blocking=True)
            x_edge  = x_edge.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)

            with autocast():
                logits = model(x_whole, x_body, x_edge)['final']
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

        if scheduler is not None:
            scheduler.step()

        clear_gpu_memory()
        metrics_best, metrics_05, best_thresholds = evaluate_model_and_find_thresholds(
            model, val_loader, device, label_cols
        )
        avg_train_loss = total_loss / max(1, n_batches)

        print(f"\n{tag} Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"[Best-TH] Val F1 Micro: {metrics_best['f1_micro']:.4f} | Macro: {metrics_best['f1_macro']:.4f} | Jaccard: {metrics_best['jaccard_avg']:.4f}")
        print(f"[TH=0.5] Val F1 Micro: {metrics_05['f1_micro']:.4f} | Macro: {metrics_05['f1_macro']:.4f} | Jaccard: {metrics_05['jaccard_avg']:.4f}")
        for i, lab in enumerate(label_cols):
            print(f"{lab}: (BestTH F1={metrics_best['f1_individual'][i]:.3f} Jac={metrics_best['jaccard_individual'][i]:.3f})"
                  f" | (0.5 F1={metrics_05['f1_individual'][i]:.3f})  TH*={best_thresholds[i]:.2f}")

        if metrics_best['f1_macro'] > best_macro_f1:
            best_macro_f1 = metrics_best['f1_macro']
            state = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_macro_f1': best_macro_f1,
                'metrics_best': metrics_best,
                'metrics_05': metrics_05,
                'label_cols': label_cols,
                'best_thresholds': best_thresholds.tolist(),
            }
            torch.save(state, save_path)
            with open(os.path.splitext(save_path)[0] + "_best_thresholds.json", "w", encoding="utf-8") as f:
                json.dump({label_cols[i]: float(best_thresholds[i]) for i in range(len(label_cols))},
                          f, ensure_ascii=False, indent=2)
            print(f"✅ Saved best model + thresholds to {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        clear_gpu_memory()

    return best_macro_f1

# --------------- Stages ---------------
def stage1(train_csv, val_csv, image_root, label_cols,
           num_epochs=30, batch_size=12, lr=5e-5, model_path='signnet_stage1.pth',
           beta_cb=0.9999, sampler_power=1.0, backbone='swin_base_patch4_window7_224', feature_dim=512):

    clear_gpu_memory()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Stage1] Using device: {device}")
    print_gpu_memory()

    train_loader, val_loader, n_pos_t, n_neg_t = make_loaders(
        train_csv, val_csv, image_root, label_cols, batch_size, sampler_power
    )

    model = SignOrientedNetwork(num_classes=len(label_cols), backbone=backbone, feature_dim=feature_dim).to(device)

    criterion = ClassBalancedBCELoss(n_pos=n_pos_t.to(device), n_neg=n_neg_t.to(device), beta=beta_cb)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler()

    best_f1 = run_stage(train_loader, val_loader, model, criterion, optimizer, scheduler, device,
                        label_cols, num_epochs, scaler, model_path, tag='Stage1')
    return best_f1

def stage2(train_csv, val_csv, image_root, label_cols,
           num_epochs=15, batch_size=12, lr=3e-4, model_path_stage1='signnet_stage1.pth',
           model_path_out='signnet_stage2.pth', beta_cb=0.9999, sampler_power=1.25,
           backbone='swin_base_patch4_window7_224', feature_dim=512, reinit_last=True, unfreeze_feature_proj=False):

    clear_gpu_memory()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Stage2] Using device: {device}")
    print_gpu_memory()

    # 更「平衡」的取樣（sampler_power 拉高）
    train_loader, val_loader, n_pos_t, n_neg_t = make_loaders(
        train_csv, val_csv, image_root, label_cols, batch_size, sampler_power
    )

    # 載入 Stage1 權重
    model = SignOrientedNetwork(num_classes=len(label_cols), backbone=backbone, feature_dim=feature_dim).to(device)
    ckpt = torch.load(model_path_stage1, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt)

    # 凍結 backbones
    freeze_backbones(model, freeze_feature_proj=not unfreeze_feature_proj)

    # 可選：重新初始化分類器最後一層
    if reinit_last:
        reinit_classifier_last_layers(model)

    # 只訓練 heads/attention（以及可選 feature_proj）
    params = classifier_parameters(model)
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4, eps=1e-8)
    # 注意：Stage2 通常不需要太複雜的學習率日程，可選擇較短的 Cosine 或常數 LR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler()

    # 用同一個 CB-BCE（以 Stage2 使用到的訓練集統計）
    criterion = ClassBalancedBCELoss(n_pos=n_pos_t.to(device), n_neg=n_neg_t.to(device), beta=beta_cb)

    best_f1 = run_stage(train_loader, val_loader, model, criterion, optimizer, scheduler, device,
                        label_cols, num_epochs, scaler, model_path_out, tag='Stage2')
    return best_f1

# --------------- Main (5-fold example) ---------------
if __name__ == '__main__':
    label_cols = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis',
                  'Crack', 'Toothmark', 'FurThick', 'FurYellow']
    backbone = 'swin_base_patch4_window7_224'
    feature_dim = 512

    for i in range(1, 5 + 1):
        train_csv = f'train_fold{i}.csv'
        val_csv   = f'val_fold{i}.csv'
        s1_path   = f'signnet_stage1_fold{i}.pth'
        s2_path   = f'signnet_stage2_fold{i}.pth'

        print(f"\n====== Fold {i}: Stage 1 (Representation) ======")
        try:
            best_f1_s1 = stage1(
                train_csv, val_csv, 'images', label_cols,
                num_epochs=30, batch_size=12, lr=5e-5,
                model_path=s1_path, beta_cb=0.9999, sampler_power=1.0,
                backbone=backbone, feature_dim=feature_dim
            )
            print(f"Fold {i} Stage1 Best Macro-F1@BestTH: {best_f1_s1:.4f}")
        except Exception as e:
            print(f"Fold {i} Stage1 失敗: {str(e)}")
            print_gpu_memory()
            clear_gpu_memory()
            continue

        print(f"\n====== Fold {i}: Stage 2 (Classifier Re-training) ======")
        try:
            best_f1_s2 = stage2(
                train_csv, val_csv, 'images', label_cols,
                num_epochs=15, batch_size=12, lr=3e-4,
                model_path_stage1=s1_path, model_path_out=s2_path,
                beta_cb=0.9999, sampler_power=1.25,
                backbone=backbone, feature_dim=feature_dim,
                reinit_last=True,            # 建議開
                unfreeze_feature_proj=False  # 先鎖；若容量不夠再開
            )
            print(f"Fold {i} Stage2 Best Macro-F1@BestTH: {best_f1_s2:.4f}")
        except Exception as e:
            print(f"Fold {i} Stage2 失敗: {str(e)}")
            print_gpu_memory()
            clear_gpu_memory()
            continue
