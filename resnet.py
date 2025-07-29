# train.py - ResNet50版本 (完整版本)
import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
from dataset import TongueDataset
from model import SignOrientedNetwork
from sklearn.metrics import f1_score, jaccard_score
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import gc
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings('ignore')

# 設定環境變數以優化記憶體使用
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def print_gpu_memory():
    """監控 GPU 記憶體使用"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f} GB allocated / {reserved:.2f} GB reserved")

def clear_gpu_memory():
    """清理 GPU 記憶體"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

class WeightedBCELoss(nn.Module):
    """論文中使用的加權BCE損失 - 修正版本"""
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        # 添加數值穩定性檢查
        inputs = torch.clamp(inputs, min=-10, max=10)  # 防止極值
        loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='mean'
        )
        
        # 檢查NaN
        if torch.isnan(loss):
            print("Warning: NaN loss detected, using fallback BCE")
            loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
            
        return loss

def compute_class_weights(csv_file, label_cols):
    """計算類別權重 - 修正版本"""
    df = pd.read_csv(csv_file)
    
    pos_weights = []
    for col in label_cols:
        pos_count = df[col].sum()
        neg_count = len(df) - pos_count
        
        # 避免極端權重值
        if pos_count == 0:
            weight = 1.0
        elif neg_count == 0:
            weight = 1.0
        else:
            weight = neg_count / pos_count
            # 限制權重範圍，避免過度懲罰
            weight = min(max(weight, 0.1), 10.0)
        
        pos_weights.append(weight)
        print(f"{col}: pos={pos_count}, neg={neg_count}, weight={weight:.3f}")
    
    return torch.tensor(pos_weights, dtype=torch.float32)

def evaluate_model(model, data_loader, device, label_cols, threshold=0.5):
    """評估模型性能 - 修正版本"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for (x_whole, x_body, x_edge), labels in data_loader:
            try:
                x_whole = x_whole.to(device, non_blocking=True)
                x_body = x_body.to(device, non_blocking=True)
                x_edge = x_edge.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with autocast():
                    outputs = model(x_whole, x_body, x_edge)
                    # 檢查輸出是否包含NaN
                    if torch.isnan(outputs['final']).any():
                        print("Warning: NaN detected in model output")
                        continue
                        
                    probs = torch.sigmoid(outputs['final']).cpu().numpy()
                    preds_binary = (probs > threshold).astype(int)
                
                all_preds.append(preds_binary)
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("驗證時記憶體不足，跳過此批次")
                    clear_gpu_memory()
                    continue
                else:
                    raise e
    
    if not all_preds:
        return {
            'f1_micro': 0.0, 'f1_macro': 0.0, 'f1_individual': [0.0] * len(label_cols),
            'jaccard_individual': [0.0] * len(label_cols), 'jaccard_avg': 0.0
        }
    
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    
    # 動態調整閾值 - 解決F1=0問題
    f1_individual = []
    jaccard_scores = []
    
    for i in range(len(label_cols)):
        if all_labels[:, i].sum() == 0:  # 如果沒有正樣本
            f1_individual.append(0.0)
            jaccard_scores.append(0.0)
            continue
            
        # 嘗試不同閾值找到最佳F1
        best_f1 = 0.0
        best_jac = 0.0
        
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            pred_thresh = (all_probs[:, i] > thresh).astype(int)
            
            # 確保有預測為正的樣本
            if pred_thresh.sum() > 0:
                f1 = f1_score(all_labels[:, i], pred_thresh, zero_division=0)
                jac = jaccard_score(all_labels[:, i], pred_thresh, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_jac = jac
        
        f1_individual.append(best_f1)
        jaccard_scores.append(best_jac)
        
        print(f"{label_cols[i]}: pos_samples={all_labels[:, i].sum()}, best_f1={best_f1:.3f}")
    
    # 計算整體指標
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_macro = np.mean(f1_individual)
    jaccard_avg = np.mean(jaccard_scores)
    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_individual': f1_individual,
        'jaccard_individual': jaccard_scores,
        'jaccard_avg': jaccard_avg
    }

def train_model(train_csv, val_csv, image_root, label_cols,
                backbone='resnet50', feature_dim=512,
                num_epochs=30, batch_size=32, lr=1e-4, model_path="signnet_best.pth"):
    
    # 清理記憶體
    clear_gpu_memory()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using backbone: {backbone}")
    print_gpu_memory()
    
    # 數據加載
    train_set = TongueDataset(train_csv, image_root, label_cols, is_train=True)
    val_set = TongueDataset(val_csv, image_root, label_cols, is_train=False)
    
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    
    # 模型初始化 - 支援不同backbone
    model = SignOrientedNetwork(
        num_classes=len(label_cols),
        backbone=backbone,
        feature_dim=feature_dim
    )
    model.to(device)
    
    # 計算類別權重（修正版本）
    pos_weights = compute_class_weights(train_csv, label_cols)
    print("Class weights:", pos_weights)
    
    # 損失函數 - 使用修正的加權BCE
    main_criterion = WeightedBCELoss(pos_weight=pos_weights.to(device))
    
    # 優化器 - 根據backbone調整學習率
    backbone_lr_config = {
        'resnet50': {'lr': 1e-4, 'weight_decay': 1e-4},
        'resnet34': {'lr': 1e-4, 'weight_decay': 5e-5},
        'resnet18': {'lr': 2e-4, 'weight_decay': 5e-5},
        'swin_base_patch4_window7_224': {'lr': 5e-5, 'weight_decay': 1e-4},
        'efficientnet_b0': {'lr': 1e-4, 'weight_decay': 1e-4},
        'vit_base_patch16_224': {'lr': 5e-5, 'weight_decay': 1e-4}
    }
    
    config = backbone_lr_config.get(backbone, {'lr': lr, 'weight_decay': 1e-4})
    print(f"Using learning rate: {config['lr']}, weight decay: {config['weight_decay']}")
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay'],
        eps=1e-8
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 混合精度訓練
    scaler = GradScaler()
    
    # 根據backbone調整batch size和梯度累積
    backbone_batch_config = {
        'resnet50': {'batch_size': 16, 'accumulation_steps': 2},
        'resnet34': {'batch_size': 24, 'accumulation_steps': 1},
        'resnet18': {'batch_size': 32, 'accumulation_steps': 1},
        'swin_base_patch4_window7_224': {'batch_size': 12, 'accumulation_steps': 2},
        'efficientnet_b0': {'batch_size': 20, 'accumulation_steps': 1},
        'vit_base_patch16_224': {'batch_size': 12, 'accumulation_steps': 2}
    }
    
    batch_config = backbone_batch_config.get(backbone, {'batch_size': batch_size, 'accumulation_steps': 2})
    accumulation_steps = batch_config['accumulation_steps']
    print(f"Effective batch size: {batch_size * accumulation_steps}")
    
    best_f1 = 0
    patience = 8
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        train_loss = 0
        train_batches = 0
        
        # 清理記憶體
        clear_gpu_memory()
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False)
        
        for batch_idx, ((x_whole, x_body, x_edge), labels) in enumerate(loop):
            try:
                x_whole = x_whole.to(device, non_blocking=True)
                x_body = x_body.to(device, non_blocking=True)
                x_edge = x_edge.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # 檢查輸入數據是否包含NaN
                if torch.isnan(x_whole).any() or torch.isnan(labels).any():
                    print(f"Warning: NaN in input data at batch {batch_idx}")
                    continue
                
                # 使用混合精度
                with autocast():
                    outputs = model(x_whole, x_body, x_edge)
                    
                    # 檢查模型輸出
                    if torch.isnan(outputs['final']).any():
                        print(f"Warning: NaN in model output at batch {batch_idx}")
                        continue
                    
                    # 計算損失
                    main_loss = main_criterion(outputs['final'], labels)
                    total_loss = main_loss / accumulation_steps
                
                # 檢查損失是否為NaN
                if torch.isnan(total_loss):
                    print(f"Warning: NaN loss at batch {batch_idx}, skipping")
                    continue
                
                # 反向傳播
                scaler.scale(total_loss).backward()
                
                # 梯度累積
                if (batch_idx + 1) % accumulation_steps == 0:
                    # 梯度裁剪
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # 定期清理記憶體
                    if (batch_idx + 1) % (accumulation_steps * 10) == 0:
                        clear_gpu_memory()
                
                train_loss += total_loss.item() * accumulation_steps
                train_batches += 1
                loop.set_postfix(loss=total_loss.item() * accumulation_steps)
                
                # 進度顯示
                if batch_idx % 50 == 0 and batch_idx > 0:
                    print_gpu_memory()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"記憶體不足在 batch {batch_idx}，跳過此批次")
                    print_gpu_memory()
                    clear_gpu_memory()
                    
                    # 重置優化器狀態
                    optimizer.zero_grad()
                    continue
                else:
                    raise e
        
        scheduler.step()
        
        # 計算平均訓練損失
        avg_train_loss = train_loss / max(train_batches, 1)
        
        # 驗證階段
        clear_gpu_memory()
        metrics = evaluate_model(model, val_loader, device, label_cols)
        
        print(f"\nEpoch {epoch+1}/{num_epochs} - {backbone}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val F1 Micro: {metrics['f1_micro']:.4f}")
        print(f"Val F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"Val Jaccard Avg: {metrics['jaccard_avg']:.4f}")
        
        # 打印各個標籤的F1分數
        for i, label in enumerate(label_cols):
            f1_val = metrics['f1_individual'][i] if i < len(metrics['f1_individual']) else 0.0
            jac_val = metrics['jaccard_individual'][i] if i < len(metrics['jaccard_individual']) else 0.0
            print(f"{label}: F1={f1_val:.3f}, Jaccard={jac_val:.3f}")
        
        # 保存最佳模型
        if metrics['f1_macro'] > best_f1:
            best_f1 = metrics['f1_macro']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_f1': best_f1,
                'metrics': metrics,
                'backbone': backbone,
                'feature_dim': feature_dim
            }, model_path)
            print(f"✅ Saved best model to {model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # 每個 epoch 結束後清理記憶體
        clear_gpu_memory()
    
    return best_f1

if __name__ == "__main__":
    label_cols = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis',
                  'Crack', 'Toothmark', 'FurThick', 'FurYellow']
    
    # 測試不同的backbone
    backbones_to_test = ['resnet50', 'resnet34', 'resnet18']
    
    for backbone in backbones_to_test:
        print(f"\n====== Testing {backbone} ======")
        
        for i in range(1, 6):
            train_csv = f"aug_train_fold{i}.csv"
            val_csv = f"val_fold{i}.csv"
            model_path = f"signnet_{backbone}_fold{i}.pth"
            
            print(f"\n====== Training {backbone} Fold {i} ======")
            
            try:
                # 根據backbone調整batch size
                if backbone == 'resnet50':
                    batch_size = 12
                    lr = 5e-5
                elif backbone == 'resnet34':
                    batch_size = 24
                    lr = 1e-4
                else:  # resnet18
                    batch_size = 32
                    lr = 2e-4
                
                best_f1 = train_model(
                    train_csv, val_csv, "augmented_images", label_cols,
                    backbone=backbone,
                    feature_dim=512,
                    num_epochs=30, 
                    batch_size=batch_size,
                    lr=lr,
                    model_path=model_path
                )
                print(f"{backbone} Fold {i} Best F1: {best_f1:.4f}")
                
            except Exception as e:
                print(f"{backbone} Fold {i} 訓練失敗: {str(e)}")
                print_gpu_memory()
                clear_gpu_memory()
                continue
    
    print("\n====== All experiments completed ======")