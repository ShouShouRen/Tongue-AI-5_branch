# train_distill.py - 知識蒸餾專用訓練腳本
# --- 訓練一個輕量級 Student 模型來模仿 Teacher 模型 ---

import os
import gc
import json
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import copy

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

# --- 複製自 train.py ---
# 為了讓此腳本獨立運作，我們完整複製 get_model
def get_model(args):
    """根據 args 建立對應的模型"""
    
    # SimpleTimmModel - 單分支 CNN
    if args.model == 'Simple':
        return SimpleTimmModel(
            num_classes=len(args.label_cols),
            backbone=args.backbone,
            img_size=args.img_size
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

# 🔥 --- 把這段遺漏的程式碼貼在這裡 --- 🔥
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# --- 複製自 train.py ---
@torch.no_grad()
def evaluate_model_and_find_thresholds(model, data_loader, device, label_cols, threshold_grid=None):
    if threshold_grid is None:
        threshold_grid = np.arange(0.1, 0.95, 0.05)
    model.eval()
    all_probs, all_labels = [], []
    loop = tqdm(data_loader, desc="Validating", leave=False)
    for (x_whole, x_root, x_center, x_side, x_tip), labels in loop:
        logits = model(
            x_whole.to(device), x_root.to(device), x_center.to(device), 
            x_side.to(device), x_tip.to(device)
        )
        probs = torch.sigmoid(logits).float().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())
    if not all_probs: return None, None
    P = np.vstack(all_probs); Y = np.vstack(all_labels); C = Y.shape[1]
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
            if f1 > best_f1: best_f1, best_th = f1, th
        best_thresholds[i], best_f1_per_class[i] = best_th, best_f1
    preds_best = np.array([P[:, i] > best_thresholds[i] for i in range(C)]).T
    metrics_at_best = {
        'f1_macro': np.mean(best_f1_per_class).item(),
        'subset_acc': accuracy_score(Y, preds_best)
    }
    return metrics_at_best, best_thresholds

# -------------------------------------------------------------------------
# 🔥 核心修改：train_model 函式
# -------------------------------------------------------------------------

def train_model(student_args, current_fold, train_csv, val_csv, student_model_path):
    clear_gpu_memory()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- 1. 載入並凍結 Teacher (老師) 模型 ---
    print("="*60)
    print(f"👨‍🏫 載入 Teacher 模型 (Fold {current_fold})...")
    
    # 建立 teacher 參數
    teacher_args = copy.deepcopy(student_args)
    teacher_args.model = student_args.teacher_model
    teacher_args.backbone = student_args.teacher_backbone
    
    try:
        teacher_model = get_model(teacher_args).to(device)
        
        # 建立老師模型的路徑
        teacher_exp_name = f"{teacher_args.model}_{teacher_args.backbone}"
        teacher_path = f"{teacher_exp_name}_fold{current_fold}.pth"
        
        if not os.path.exists(teacher_path):
            print(f"❌ 錯誤：找不到 Teacher 模型權重: {teacher_path}")
            return -1
            
        teacher_model.load_state_dict(torch.load(teacher_path, map_location=device)['model_state_dict'])
        teacher_model.eval() # 設為評估模式
        for param in teacher_model.parameters(): # 凍結所有參數
            param.requires_grad = False
            
        print(f"  ✓ 成功載入 Teacher: {teacher_path}")
    except Exception as e:
        print(f"❌ 載入 Teacher 模型失敗: {e}")
        return -1

    # --- 2. 建立 Student (學生) 模型 ---
    print(f"🧑‍🎓 建立 Student 模型 (Fold {current_fold})...")
    # student_args 就是我們的主 args
    student_model = get_model(student_args).to(device)
    
    # 計算模型參數量 (學生的)
    total_params = sum(p.numel() for p in student_model.parameters())
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f"  📊 Student Model Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # --- 3. 準備資料 ---
    train_set = TongueDataset(train_csv, student_args.image_root, student_args.label_cols, is_train=True)
    sampler, n_pos_t, n_neg_t = build_multilabel_weighted_sampler(train_csv, student_args.label_cols)
    train_loader = DataLoader(
        train_set, batch_size=student_args.batch_size, sampler=sampler, 
        num_workers=student_args.num_workers, pin_memory=True, drop_last=True
    )
    
    val_set = TongueDataset(val_csv, student_args.image_root, student_args.label_cols, is_train=False)
    val_loader = DataLoader(
        val_set, batch_size=student_args.batch_size * 2, shuffle=False, 
        num_workers=student_args.num_workers, pin_memory=True
    )

    # --- 4. 定義 Loss 函式 ---
    
    # A. Hard Loss (學生 vs. 標準答案)
    if ClassBalancedBCELoss is not None:
        criterion_hard = ClassBalancedBCELoss(
            n_pos=n_pos_t.to(device), n_neg=n_neg_t.to(device), beta=0.9999
        )
    else:
        criterion_hard = torch.nn.BCEWithLogitsLoss()
        
    # B. Soft Loss (學生 vs. 老師)
    criterion_soft = nn.KLDivLoss(reduction='batchmean')
    
    # 讀取 KD 超參數
    T = student_args.temperature
    alpha = student_args.alpha
    print(f"  🔥 蒸餾參數: Alpha={alpha} (學老師的權重), Temperature={T}")
    print("="*60)
        
    # --- 5. 設定優化器 (只優化學生) ---
    optimizer = optim.AdamW(student_model.parameters(), lr=student_args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )
    scaler = GradScaler()
    
    best_f1_macro, patience_counter = -1.0, 0

    # --- 6. 訓練迴圈 ---
    for epoch in range(student_args.epochs):
        student_model.train() # 確保學生是訓練模式
        total_loss, total_loss_hard, total_loss_soft = 0.0, 0.0, 0.0
        n_batches = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{student_args.epochs} Training", leave=False)
        
        for (x_whole, x_root, x_center, x_side, x_tip), labels in loop:
            # 準備輸入 (全部上 device)
            x_whole_d = x_whole.to(device)
            x_root_d = x_root.to(device)
            x_center_d = x_center.to(device)
            x_side_d = x_side.to(device)
            x_tip_d = x_tip.to(device)
            labels_d = labels.to(device)
            
            # A. 取得 Teacher 的 Logits (不計算梯度)
            with torch.no_grad():
                teacher_logits = teacher_model(
                    x_whole_d, x_root_d, x_center_d, x_side_d, x_tip_d
                )
            
            # B. 取得 Student 的 Logits (計算梯度)
            with autocast():
                student_logits = student_model(
                    x_whole_d, x_root_d, x_center_d, x_side_d, x_tip_d
                )
                
                # C. 計算 Hard Loss (學生 vs 答案)
                loss_hard = criterion_hard(student_logits, labels_d)
                
                # D. 計算 Soft Loss (學生 vs 老師)
                loss_soft = criterion_soft(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1)
                ) * (T * T) # 乘 T^2 來還原梯度尺度
                
                # E. 計算總 Loss
                loss = (alpha * loss_soft) + ((1.0 - alpha) * loss_hard)
            
            # F. 反向傳播 (只更新學生)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            total_loss_hard += loss_hard.item()
            total_loss_soft += loss_soft.item()
            n_batches += 1
            loop.set_postfix(
                Loss=total_loss/n_batches, 
                Hard=total_loss_hard/n_batches, 
                Soft=total_loss_soft/n_batches
            )
        
        scheduler.step()

        # --- 7. 驗證迴圈 (只驗證學生) ---
        metrics, best_thresholds = evaluate_model_and_find_thresholds(
            student_model, val_loader, device, student_args.label_cols
        )
        if metrics:
            print(f"\nEpoch {epoch+1}/{student_args.epochs} | Train Loss: {total_loss/n_batches:.4f} | "
                  f"Val F1-Macro: {metrics['f1_macro']:.4f} | Val Subset-Acc: {metrics['subset_acc']:.4f}")
            
            if metrics['f1_macro'] > best_f1_macro:
                best_f1_macro = metrics['f1_macro']
                state = {'model_state_dict': student_model.state_dict()}
                torch.save(state, student_model_path) # 儲存學生模型
                
                th_path = os.path.splitext(student_model_path)[0] + "_best_thresholds.json"
                with open(th_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {label: float(th) for label, th in zip(student_args.label_cols, best_thresholds)}, 
                        f, indent=2
                    )
                print(f"✅ Saved best STUDENT model to {os.path.basename(student_model_path)}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= student_args.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    return best_f1_macro


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='知識蒸餾 K-Fold 訓練腳本')
    
    # --- 學生 (Student) 模型參數 ---
    parser.add_argument('--model', type=str, required=True,
                        choices=[
                            'Simple', 'SignOriented', 'SignOrientedAttention',
                            'MambaVision', 'SimpleMambaVision', 'MambaVisionFusion'
                        ],
                        help='(學生) 要訓練的輕量級模型架構')
    
    parser.add_argument('--backbone', type=str, default='mobilenetv3_large_100',
                        help='(學生) 輕量級 CNN 骨幹網路')
    
    # --- 老師 (Teacher) 模型參數 ---
    parser.add_argument('--teacher_model', type=str, required=True,
                        choices=[
                            'Simple', 'SignOriented', 'SignOrientedAttention',
                            'MambaVision', 'SimpleMambaVision', 'MambaVisionFusion'
                        ],
                        help='(老師) 預訓練好的大型模型架構')
    
    parser.add_argument('--teacher_backbone', type=str, required=True,
                        help='(老師) 大型 CNN 骨幹網路 (例如 swin_base_patch4_window7_224)')

    # --- 蒸餾 (Distillation) 參數 ---
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='蒸餾損失(Soft Loss)的權重, alpha=0.7 代表 70% 學老師, 30% 學答案')
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='蒸餾溫度 T, 用於平滑 logits (T > 1)')
    
    # --- 訓練參數 (與 train.py 相同) ---
    parser.add_argument('--epochs', type=int, default=30, help='訓練 Epoch 數')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-5, help='學習率')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping 耐心值')
    parser.add_argument('--feature_dim', type=int, default=512, help='特徵維度')
    parser.add_argument('--d_state', type=int, default=16, help='Mamba 狀態空間維度')
    parser.add_argument('--num_mamba_layers', type=int, default=2, help='Mamba 層數')
    parser.add_argument('--img_size', type=int, default=224, help='輸入圖片大小')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch 大小')
    parser.add_argument('--embed_dim', type=int, default=512, help='Mamba embedding 維度')
    parser.add_argument('--mamba_depth', type=int, default=6, help='Mamba Backbone 深度')
    parser.add_argument('--image_root', type=str, default='images', help='圖片根目錄')
    parser.add_argument('--num_workers', type=int, default=4, help='資料載入線程數')
    parser.add_argument('--mamba_vision_model', type=str, default='mamba_vision_T',
                        choices=['mamba_vision_T', 'mamba_vision_T2', 
                                 'mamba_vision_S', 'mamba_vision_B', 'mamba_vision_L'],
                        help='(學生) MambaVision 模型大小')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--freeze_backbone', action='store_true', default=False)
    
    args = parser.parse_args()

    # 標籤欄位
    args.label_cols = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis',
                       'Crack', 'Toothmark', 'FurThick', 'FurYellow']
    NUM_FOLDS = 5

    # --- 🔥 建立學生的實驗名稱 ---
    # 學生模型 (我們主要關心的)
    if args.model in ['Simple', 'SignOriented', 'SignOrientedAttention']:
        student_base_name = f'{args.model}_{args.backbone}'
    elif args.model in ['MambaVision', 'SimpleMambaVision', 'MambaVisionFusion']:
        student_base_name = f'{args.model}_{args.mamba_vision_model}'
    else:
        student_base_name = f'{args.model}_{args.backbone}'
    
    # 加上蒸餾標記
    experiment_name = f"{student_base_name}_KD" # 例如: SignOrientedAttention_mobilenetv3_large_100_KD
    
    print(f"📦 實驗名稱 (將用於存檔): {experiment_name}")
    print(f"👨‍🏫 Teacher: {args.teacher_model}_{args.teacher_backbone}")
    print(f"🧑‍🎓 Student: {student_base_name}")
    print("="*70)

    # --- K-Fold 迴圈 (與 train.py 相同) ---
    
    # Phase 1: K-Fold Cross-Validation
    print("\n" + "="*30 + "\n   PHASE 1: K-Fold CV (Distillation)\n" + "="*30)
    for i in range(1, NUM_FOLDS + 1):
        train_csv = f'train_fold{i}.csv'
        val_csv = f'val_fold{i}.csv'
        
        if not all(os.path.exists(f) for f in [train_csv, val_csv]):
            print(f"⚠️   Skipping Fold {i}: CSV files not found")
            continue
        
        model_path = f'{experiment_name}_fold{i}.pth'
        
        print(f"\n{'='*60}")
        print(f"   Training Fold {i}/{NUM_FOLDS}")
        print(f"{'='*66}")
        
        # 傳入所有 args，以及當前的 fold
        train_model(args, i, train_csv, val_csv, model_path)

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
        
        final_th_path = f'{experiment_name}_final_best_thresholds.json'
        
        with open(final_th_path, 'w', encoding='utf-8') as f:
            json.dump(avg_thresholds, f, indent=2)
        
        print(f"\n✅ Averaged thresholds saved to: {final_th_path}")
        for label, th in avg_thresholds.items():
            print(f"   - {label:<15}: {th:.4f}")
    
    print("\n" + "="*70)
    print("   ✅ 知識蒸餾 K-Fold 訓練完成！")
    print("="*70)
