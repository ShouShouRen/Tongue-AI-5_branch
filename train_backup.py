# train.py - 靈活的模型訓練工具
# --- 已同步更新，可透過指令行參數選擇模型、骨幹與超參數 ---

import os
import gc
import json
import warnings
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

# 從模型庫匯入所有模型
from model import SignOrientedNetwork, SimpleTimmModel, SignOrientedAttentionNetwork
from dataset import TongueDataset
from samplers import build_multilabel_weighted_sampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, accuracy_score

# --- 可切換的損失函數 ---
# 為了簡潔，直接在程式碼中定義，避免額外檔案
from losses import ClassBalancedBCELoss

warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def get_model(args):
    """根據 args 建立對應的模型"""
    model_class = None
    if args.model == 'SignOriented':
        model_class = SignOrientedNetwork
    elif args.model == 'Simple':
        model_class = SimpleTimmModel
    elif args.model == 'SignOrientedAttention':
        model_class = SignOrientedAttentionNetwork
    else:
        raise ValueError(f"未知的模型架構: {args.model}")
        
    # SimpleTimmModel 不需要 feature_dim
    if args.model == 'Simple':
        return model_class(
            num_classes=len(args.label_cols),
            backbone=args.backbone
        )
    else:
        return model_class(
            num_classes=len(args.label_cols),
            backbone=args.backbone,
            feature_dim=args.feature_dim
        )

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

@torch.no_grad()
def evaluate_model_and_find_thresholds(model, data_loader, device, label_cols, threshold_grid=None):
    if threshold_grid is None:
        threshold_grid = np.arange(0.1, 0.95, 0.05)

    model.eval()
    all_probs, all_labels = [], []
    loop = tqdm(data_loader, desc="Validating", leave=False)
    for (x_whole, x_root, x_center, x_side, x_tip), labels in loop:
        logits = model(x_whole.to(device), x_root.to(device), x_center.to(device), x_side.to(device), x_tip.to(device))
        probs = torch.sigmoid(logits).float().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    if not all_probs:
        return None, None

    P = np.vstack(all_probs)
    Y = np.vstack(all_labels)
    C = Y.shape[1]

    best_thresholds = np.zeros(C, dtype=np.float32)
    best_f1_per_class = np.zeros(C, dtype=np.float32)

    for i in range(C):
        if Y[:, i].sum() == 0:
            best_thresholds[i], best_f1_per_class[i] = 0.5, 0.0
            continue
        best_f1, best_th = -1.0, 0.5
        for th in threshold_grid:
            pred = (P[:, i] > th).astype(int)
            f1 = f1_score(Y[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_th = f1, th
        best_thresholds[i], best_f1_per_class[i] = best_th, best_f1

    preds_best = np.array([P[:, i] > best_thresholds[i] for i in range(C)]).T
    
    metrics_at_best = {
        'f1_macro': np.mean(best_f1_per_class).item(),
        'subset_acc': accuracy_score(Y, preds_best)
    }
    return metrics_at_best, best_thresholds


def train_model(args, train_csv, val_csv, model_path):
    clear_gpu_memory()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_set = TongueDataset(train_csv, args.image_root, args.label_cols, is_train=True)
    sampler, n_pos_t, n_neg_t = build_multilabel_weighted_sampler(train_csv, args.label_cols)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    val_loader = None
    if val_csv:
        val_set = TongueDataset(val_csv, args.image_root, args.label_cols, is_train=False)
        val_loader = DataLoader(val_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = get_model(args).to(device)
    
    criterion = ClassBalancedBCELoss(n_pos=n_pos_t.to(device), n_neg=n_neg_t.to(device), beta=0.9999)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    best_f1_macro, patience_counter = -1.0, 0

    for epoch in range(args.epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training", leave=False)
        for (x_whole, x_root, x_center, x_side, x_tip), labels in loop:
            with autocast():
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
            metrics, best_thresholds = evaluate_model_and_find_thresholds(model, val_loader, device, args.label_cols)
            if metrics:
                print(f"\nEpoch {epoch+1}/{args.epochs} | Train Loss: {total_loss/n_batches:.4f} | Val F1-Macro: {metrics['f1_macro']:.4f} | Val Subset-Acc: {metrics['subset_acc']:.4f}")
                
                if metrics['f1_macro'] > best_f1_macro:
                    best_f1_macro = metrics['f1_macro']
                    state = {'model_state_dict': model.state_dict()}
                    torch.save(state, model_path)
                    th_path = os.path.splitext(model_path)[0] + "_best_thresholds.json"
                    with open(th_path, "w", encoding="utf-8") as f:
                        json.dump({label: float(th) for label, th in zip(args.label_cols, best_thresholds)}, f, indent=2)
                    print(f"✅ Saved best model to {os.path.basename(model_path)}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
        else:
            print(f"\nEpoch {epoch+1}/{args.epochs} | Train Loss: {total_loss/n_batches:.4f}")

    if not val_loader:
        torch.save({'model_state_dict': model.state_dict()}, model_path)
        print(f"✅ Saved final model to {os.path.basename(model_path)}")
    
    return best_f1_macro

def combine_kfold_csvs(k, output_path):
    all_dfs = [pd.read_csv(f) for i in range(1, k + 1) for f in [f'train_fold{i}.csv', f'val_fold{i}.csv'] if os.path.exists(f)]
    if not all_dfs: return None
    pd.concat(all_dfs).drop_duplicates().reset_index(drop=True).to_csv(output_path, index=False)
    print(f"✔ Combined CSV saved to {output_path} with {len(pd.concat(all_dfs))} rows.")
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='訓練舌象辨識模型')
    parser.add_argument('--model', type=str, required=True, choices=['Simple', 'SignOriented', 'SignOrientedAttention'], help='要使用的模型架構')
    parser.add_argument('--backbone', type=str, required=True, help='模型使用的骨幹網路, e.g., "convnext_base"')
    parser.add_argument('--epochs', type=int, default=30, help='訓練的 Epoch 總數')
    parser.add_argument('--batch_size', type=int, default=16, help='訓練時的批次大小')
    parser.add_argument('--lr', type=float, default=1e-5, help='學習率')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping 的耐心值')
    parser.add_argument('--feature_dim', type=int, default=512, help='特徵維度 (僅用於 SignOriented 系列模型)')
    parser.add_argument('--image_root', type=str, default='images', help='圖片根目錄')
    parser.add_argument('--num_workers', type=int, default=4, help='資料載入器的工作線程數')
    args = parser.parse_args()

    args.label_cols = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis',
                       'Crack', 'Toothmark', 'FurThick', 'FurYellow']
    NUM_FOLDS = 5

    print("="*20 + "\n  PHASE 1: K-Fold Cross-Validation\n" + "="*20)
    for i in range(1, NUM_FOLDS + 1):
        train_csv, val_csv = f'train_fold{i}.csv', f'val_fold{i}.csv'
        if not all(os.path.exists(f) for f in [train_csv, val_csv]):
            print(f"Skipping Fold {i}: CSV file not found.")
            continue
        model_path = f'{args.model}_{args.backbone}_fold{i}.pth'
        print(f"\n====== Training Fold {i} | Model: {args.model} | Backbone: {args.backbone} ======")
        train_model(args, train_csv, val_csv, model_path)

    print("\n" + "="*20 + "\n  PHASE 2: Final Training on Full Data\n" + "="*20)
    full_train_csv = combine_kfold_csvs(NUM_FOLDS, 'train_full.csv')
    final_model_path = f'{args.model}_{args.backbone}_final.pth'
    if full_train_csv:
        print(f"\n====== Training Final Model | Model: {args.model} | Backbone: {args.backbone} ======")
        train_model(args, full_train_csv, None, final_model_path)

    print("\n" + "="*20 + "\n  PHASE 3: Averaging Thresholds for Final Model\n" + "="*20)
    all_thresholds = []
    for i in range(1, NUM_FOLDS + 1):
        th_path = f'{args.model}_{args.backbone}_fold{i}_best_thresholds.json'
        if os.path.exists(th_path):
            with open(th_path, 'r', encoding='utf-8') as f:
                fold_ths = json.load(f)
                all_thresholds.append([fold_ths.get(label, 0.5) for label in args.label_cols])
                print(f"Loaded thresholds from {os.path.basename(th_path)}")
    
    if all_thresholds:
        avg_thresholds = {label: float(th) for label, th in zip(args.label_cols, np.mean(all_thresholds, axis=0))}
        final_th_path = os.path.splitext(final_model_path)[0] + "_best_thresholds.json"
        with open(final_th_path, 'w', encoding='utf-8') as f:
            json.dump(avg_thresholds, f, indent=2)
        print(f"\n✅ Averaged thresholds saved to: {os.path.basename(final_th_path)}")
        for label, th in avg_thresholds.items():
            print(f"  - {label:<15}: {th:.4f}")
