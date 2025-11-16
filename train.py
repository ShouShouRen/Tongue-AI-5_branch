# train.py - K-Fold 交叉驗證標準流程
# --- 移除了 Phase 2 (Final Training)，只保留 K-Fold 最佳模型 ---

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
from model import (
    SignOrientedNetwork, 
    SimpleTimmModel, 
    SignOrientedAttentionNetwork,
    SignOrientedMambaVisionNetwork,
    SimpleMambaVision,
    MambaVisionWithMambaFusion
)
from dataset import TongueDataset
from samplers import build_multilabel_weighted_sampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, accuracy_score

# 嘗試載入自定義損失函數
try:
    from losses import ClassBalancedBCELoss
except ImportError:
    print("⚠️ ClassBalancedBCELoss not found, using standard BCEWithLogitsLoss")
    ClassBalancedBCELoss = None

warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def get_model(args):
    """根據 args 建立對應的模型"""
    
    # SimpleTimmModel - 單分支 CNN
    if args.model == 'Simple':
        return SimpleTimmModel(
            num_classes=len(args.label_cols),
            backbone=args.backbone,
            img_size=args.img_size  # 確保 img_size 被傳遞
        )
    
    # SignOriented - 原始五分支模型
    elif args.model == 'SignOriented':
        return SignOrientedNetwork(
            num_classes=len(args.label_cols),
            backbone=args.backbone,
            feature_dim=args.feature_dim
        )
    
    # SignOrientedAttention - Transformer Attention 融合
    elif args.model == 'SignOrientedAttention':
        return SignOrientedAttentionNetwork(
            num_classes=len(args.label_cols),
            backbone=args.backbone,
            feature_dim=args.feature_dim
        )
    elif args.model == 'MambaVision':
        return SignOrientedMambaVisionNetwork(
            num_classes=len(args.label_cols),
            mamba_vision_model=args.mamba_vision_model,
            feature_dim=args.feature_dim,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone
        )
    
    elif args.model == 'SimpleMambaVision':
        return SimpleMambaVision(
            num_classes=len(args.label_cols),
            mamba_vision_model=args.mamba_vision_model,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone
        )
    
    elif args.model == 'MambaVisionFusion':
        return MambaVisionWithMambaFusion(
            num_classes=len(args.label_cols),
            mamba_vision_model=args.mamba_vision_model,
            feature_dim=args.feature_dim,
            d_state=args.d_state,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone
        )
    
    else:
        raise ValueError(f"未知的模型架構: {args.model}")


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
        logits = model(
            x_whole.to(device), 
            x_root.to(device), 
            x_center.to(device), 
            x_side.to(device), 
            x_tip.to(device)
        )
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
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=True
    )
    
    val_loader = None
    if val_csv:
        val_set = TongueDataset(val_csv, args.image_root, args.label_cols, is_train=False)
        val_loader = DataLoader(
            val_set, 
            batch_size=args.batch_size * 2, 
            shuffle=False, 
            num_workers=args.num_workers, 
            pin_memory=True
        )

    model = get_model(args).to(device)
    
    # 計算模型參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # 損失函數
    if ClassBalancedBCELoss is not None:
        criterion = ClassBalancedBCELoss(
            n_pos=n_pos_t.to(device), 
            n_neg=n_neg_t.to(device), 
            beta=0.9999
        )
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,      # 每 10 個 epoch 重啟一次
            T_mult=2,    # 每次重啟後，週期加倍（10 → 20 → 40）
            eta_min=1e-7 # 最小學習率
        )
    scaler = GradScaler()
    
    best_f1_macro, patience_counter = -1.0, 0

    for epoch in range(args.epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training", leave=False)
        
        for (x_whole, x_root, x_center, x_side, x_tip), labels in loop:
            with autocast():
                logits = model(
                    x_whole.to(device), 
                    x_root.to(device), 
                    x_center.to(device), 
                    x_side.to(device), 
                    x_tip.to(device)
                )
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
            metrics, best_thresholds = evaluate_model_and_find_thresholds(
                model, val_loader, device, args.label_cols
            )
            if metrics:
                print(f"\nEpoch {epoch+1}/{args.epochs} | Train Loss: {total_loss/n_batches:.4f} | "
                      f"Val F1-Macro: {metrics['f1_macro']:.4f} | Val Subset-Acc: {metrics['subset_acc']:.4f}")
                
                if metrics['f1_macro'] > best_f1_macro:
                    best_f1_macro = metrics['f1_macro']
                    state = {'model_state_dict': model.state_dict()}
                    torch.save(state, model_path)
                    
                    th_path = os.path.splitext(model_path)[0] + "_best_thresholds.json"
                    with open(th_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {label: float(th) for label, th in zip(args.label_cols, best_thresholds)}, 
                            f, indent=2
                        )
                    print(f"✅ Saved best model to {os.path.basename(model_path)}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
        else:
            # 這種情況現在只會在 val_csv=None 時發生，
            # 也就是在被我們移除的 Phase 2 中。
            # 為了保險起見，我們仍然保留這段程式碼，
            # 儘管在 K-Fold 流程中它不應該被執行。
            print(f"\nEpoch {epoch+1}/{args.epochs} | Train Loss: {total_loss/n_batches:.4f}")

    # 移除了 if not val_loader: ... 的區塊，
    # 因為我們只關心在有 val_loader 時儲存的最佳模型。
    
    return best_f1_macro


def combine_kfold_csvs(k, output_path):
    # 這個函式現在不會被呼叫，但可以保留
    all_dfs = []
    for i in range(1, k + 1):
        for f in [f'train_fold{i}.csv', f'val_fold{i}.csv']:
            if os.path.exists(f):
                all_dfs.append(pd.read_csv(f))
    
    if not all_dfs:
        return None
    
    combined = pd.concat(all_dfs).drop_duplicates().reset_index(drop=True)
    combined.to_csv(output_path, index=False)
    print(f"✔ Combined CSV saved to {output_path} with {len(combined)} rows.")
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='訓練舌象辨識模型（支援 Mamba）')
    
    # 模型選擇
    parser.add_argument('--model', type=str, required=True,
                        choices=[
                            'Simple',
                            'SignOriented',
                            'SignOrientedAttention',
                            'MambaVision',
                            'SimpleMambaVision',
                            'MambaVisionFusion'
                        ],
                        help='模型架構')
    
    parser.add_argument('--backbone', type=str, default='convnext_base',
                        help='CNN 骨幹網路（CNN 模型用）')
    
    # 訓練參數
    parser.add_argument('--epochs', type=int, default=30, help='訓練 Epoch 數')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-5, help='學習率')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping 耐心值')
    
    # 模型架構參數（通用）
    parser.add_argument('--feature_dim', type=int, default=512, 
                        help='特徵維度')
    
    # Mamba 專用參數
    parser.add_argument('--d_state', type=int, default=16, 
                        help='Mamba 狀態空間維度')
    parser.add_argument('--num_mamba_layers', type=int, default=2, 
                        help='Mamba 層數（DeepMamba 用）')
    
    # Mamba Backbone 專用參數
    parser.add_argument('--img_size', type=int, default=224, 
                        help='輸入圖片大小')
    parser.add_argument('--patch_size', type=int, default=16, 
                        help='Patch 大小（Mamba Backbone 用）')
    parser.add_argument('--embed_dim', type=int, default=512, 
                        help='Mamba embedding 維度')
    parser.add_argument('--mamba_depth', type=int, default=6, 
                        help='Mamba Backbone 深度')
                        
    # <-- 🔥 3b. 在這裡加入新的權重路徑參數 -->

    
    # 資料路徑
    parser.add_argument('--image_root', type=str, default='images', 
                        help='圖片根目錄')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='資料載入線程數')
    parser.add_argument('--mamba_vision_model', type=str, 
                        default='mamba_vision_T',
                        choices=['mamba_vision_T', 'mamba_vision_T2', 
                                 'mamba_vision_S', 'mamba_vision_B', 'mamba_vision_L'],
                        help='(MambaVision 用) MambaVision 模型大小')
    
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='(MambaVision 用) 是否使用預訓練權重')
    
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help='(MambaVision 用) 是否凍結 backbone，只訓練分類頭')
    
    args = parser.parse_args()

    # 標籤欄位
    args.label_cols = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis',
                       'Crack', 'Toothmark', 'FurThick', 'FurYellow']
    NUM_FOLDS = 5

    # --- 🔥 關鍵修改：建立唯一的實驗名稱 ---
    if args.model in ['Simple', 'SignOriented', 'SignOrientedAttention']:
        experiment_name = f'{args.model}_{args.backbone}'
    elif args.model in ['MambaVision', 'SimpleMambaVision', 'MambaVisionFusion']:
        experiment_name = f'{args.model}_{args.mamba_vision_model}'
    else:
        experiment_name = f'{args.model}_{args.backbone}'
    
    print(f"📦 實驗名稱 (將用於存檔): {experiment_name}")
    # --- 結束修改 ---


    # 顯示配置
    print("="*70)
    print("   訓練配置")
    print("="*70)
    print(f"   模型: {args.model}")
    
    if args.model in ['MambaBackbone', 'SimpleMamba']:
        print(f"   Mamba 設置:")
        print(f"     - Embedding 維度: {args.embed_dim}")
        print(f"     - Mamba 深度: {args.mamba_depth}")
        print(f"     - 狀態維度: {args.d_state}")
        print(f"     - Patch 大小: {args.patch_size}")
    else:
        print(f"   骨幹: {args.backbone}")
        print(f"   特徵維度: {args.feature_dim}")
        if 'Mamba' in args.model:
            print(f"   Mamba 狀態維度: {args.d_state}")
            if args.model == 'SignOrientedDeepMamba':
                print(f"   Mamba 層數: {args.num_mamba_layers}")
    
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Learning Rate: {args.lr}")
    print("="*70)

    # Phase 1: K-Fold Cross-Validation
    print("\n" + "="*30 + "\n   PHASE 1: K-Fold CV\n" + "="*30)
    for i in range(1, NUM_FOLDS + 1):
        train_csv = f'train_fold{i}.csv'
        val_csv = f'val_fold{i}.csv'
        
        if not all(os.path.exists(f) for f in [train_csv, val_csv]):
            print(f"⚠️   Skipping Fold {i}: CSV files not found")
            continue
        
        model_path = f'{experiment_name}_fold{i}.pth'
        
        print(f"\n{'='*60}")
        print(f"   Training Fold {i}/{NUM_FOLDS}")
        print(f"{'='*60}")
        train_model(args, train_csv, val_csv, model_path)

    # --- 🔥 關鍵修改：移除了 Phase 2 (Final Training) ---
    # print("\n" + "="*30 + "\n   PHASE 2: Final Training\n" + "="*30)
    # full_train_csv = combine_kfold_csvs(NUM_FOLDS, 'train_full.csv')
    # final_model_path = f'{experiment_name}_final.pth'
    # 
    # if full_train_csv:
    #     print(f"\n{'='*60}")
    #     print(f"   Training Final Model")
    #     print(f"{'='*60}")
    #     train_model(args, full_train_csv, None, final_model_path)
    # --- 結束修改 ---


    # Phase 3: Average Thresholds
    print("\n" + "="*30 + "\n   PHASE 3: Average Thresholds\n" + "="*30)
    all_thresholds = []
    
    for i in range(1, NUM_FOLDS + 1):
        th_path = f'{experiment_name}_fold{i}_best_thresholds.json'
        
        if os.path.exists(th_path):
            with open(th_path, 'r', encoding='utf-8') as f:
                fold_ths = json.load(f)
                all_thresholds.append([fold_ths.get(label, 0.5) for label in args.label_cols])
                print(f"✓ Loaded {os.path.basename(th_path)}")
    
    if all_thresholds:
        avg_thresholds = {
            label: float(th) 
            for label, th in zip(args.label_cols, np.mean(all_thresholds, axis=0))
        }
        
        # 這個 "final" 檔案現在代表的是 K-Fold 的平均值
        final_th_path = f'{experiment_name}_final_best_thresholds.json'
        
        with open(final_th_path, 'w', encoding='utf-8') as f:
            json.dump(avg_thresholds, f, indent=2)
        
        print(f"\n✅ Averaged thresholds saved to: {final_th_path}")
        for label, th in avg_thresholds.items():
            print(f"   - {label:<15}: {th:.4f}")
    
    print("\n" + "="*70)
    print("   ✅ K-Fold 訓練完成！")
    print("="*70)
