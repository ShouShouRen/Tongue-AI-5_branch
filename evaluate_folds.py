# evaluate_folds.py (v7 - 修正 SyntaxError)

import torch
import torch.nn as nn
import argparse
import json
import os
from PIL import Image
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

# --- 關鍵：從你的專案匯入 ---
from model import (
    SignOrientedNetwork, 
    SimpleTimmModel, 
    SignOrientedAttentionNetwork,
    SignOrientedMambaVisionNetwork,
    SimpleMambaVision,
    MambaVisionWithMambaFusion
)
from dataset import TongueDataset # 假設你的 TongueDataset 在 dataset.py
from torch.utils.data import DataLoader

# --- 匯入評估指標 ---
from sklearn.metrics import f1_score, jaccard_score, accuracy_score

warnings.filterwarnings('ignore')

# --- 關鍵：從 train.py 完整複製 get_model 函式 ---
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

@torch.no_grad()
def evaluate_fold(model, val_loader, device, thresholds):
    """在單一 fold 上進行評估"""
    model.eval()
    all_probs, all_labels = [], []
    
    for (x_whole, x_root, x_center, x_side, x_tip), labels in tqdm(val_loader, desc="Evaluating", leave=False):
        logits = model(
            x_whole.to(device), 
            x_root.to(device), 
            x_center.to(device), 
            x_side.to(device), 
            x_tip.to(device)
        )
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
    P = np.vstack(all_probs) # 預測機率
    Y = np.vstack(all_labels) # 真實標籤
    
    # 使用傳入的 thresholds 進行二值化
    T = np.array(thresholds)
    Preds = (P > T).astype(int)
    
    # 計算指標
    per_class_f1 = f1_score(Y, Preds, average=None, zero_division=0)
    avg_f1_macro = np.mean(per_class_f1)
    jaccard = jaccard_score(Y, Preds, average='samples', zero_division=0)
    subset_acc = accuracy_score(Y, Preds)
    per_class_acc = (Y == Preds).astype(float).mean(axis=0) # (8,) 
    avg_acc_label_based = np.mean(per_class_acc)
    
    return per_class_f1, avg_f1_macro, jaccard, subset_acc, avg_acc_label_based, per_class_acc


# --- 主程式 ---
def main():
    parser = argparse.ArgumentParser(description='K-Fold 交叉驗證評估 (K-Fold Evaluation)')
    
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='實驗名稱 (例如: Simple_mobilenetv3_large_100_KD)')
    parser.add_argument('--model_dir', type=str, default=".",
                        help='存放 .pth 權重檔和 .json 檔案的目錄')
    parser.add_argument('--image_root', type=str, default='images', 
                        help='圖片根目錄 (dataset.py 需要)')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='K-Fold 的 K 值 (等於模型數量)')
    parser.add_argument('--img_size', type=int, default=224, help='輸入圖片大小')
    parser.add_argument('--feature_dim', type=int, default=512, help='特徵維度')
    parser.add_argument('--d_state', type=int, default=16, help='Mamba 狀態空間維度')
    parser.add_argument('--mamba_vision_model', type=str, default='mamba_vision_T', help='MambaVision 模型大小')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--freeze_backbone', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--batch_size', type=int, default=32) 
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. "HACK": 解析 get_model 所需的參數
    thresh_path = os.path.join(args.model_dir, f"{args.experiment_name}_final_best_thresholds.json")
    if not os.path.exists(thresh_path):
        print(f"❌ 錯誤：找不到 {thresh_path}。無法獲取標籤順序。")
        return
    with open(thresh_path, 'r', encoding='utf-8') as f:
        thresholds_data = json.load(f)
    
    labels = list(thresholds_data.keys())
    args.label_cols = labels
    
    try:
        parse_name = args.experiment_name
        if parse_name.endswith('_KD'):
            parse_name = parse_name[:-3] 

        parts = parse_name.split('_', 1) 
        args.model = parts[0]
        model_specific_name = parts[1]
        
        if 'mamba' in args.model.lower():
            args.mamba_vision_model = model_specific_name
            print(f"  -> 自動解析: Model = {args.model}, Mamba = {args.mamba_vision_model}")
        else:
            args.backbone = model_specific_name 
            print(f"  -> 自動解析: Model = {args.model}, Backbone = {args.backbone}")
            
    except Exception as e:
        print(f"❌ 錯誤：無法從 '{args.experiment_name}' 解析模型名稱。錯誤: {e}")
        return

    # 2. 🔥 計算模型參數與大小
    print("  -> 計算模型參數與大小...")
    total_params = 0
    try:
        rep_model = get_model(args) 
        total_params = sum(p.numel() for p in rep_model.parameters())
        print(f"  -> 模型參數 (Params): {total_params:,}")
        del rep_model 
    except Exception as e:
        print(f"  -> 警告：計算模型參數失敗: {e}")

    model_path_1 = os.path.join(args.model_dir, f"{args.experiment_name}_fold1.pth")
    model_size_mb = 0.0
    if os.path.exists(model_path_1):
        model_size_mb = os.path.getsize(model_path_1) / (1024 * 1024)
        print(f"  -> 模型大小 (Size): {model_size_mb:.2f} MB")
    else:
        print(f"  -> 警告: 找不到 {model_path_1}，無法計算模型大小。")

    # 3. 儲存 5 個 Fold 的所有指標
    all_fold_metrics = [] 
    for i in range(1, args.num_folds + 1):
        print("\n" + "="*50)
        print(f"  Processing Fold {i}/{args.num_folds}")
        print("="*50)
        
        # A. 載入該 Fold 的驗證集 CSV
        val_csv_path = f'val_fold{i}.csv'
        if not os.path.exists(val_csv_path):
            print(f"❌ 錯誤：找不到 {val_csv_path}")
            continue
        
        val_set = TongueDataset(val_csv_path, args.image_root, args.label_cols, is_train=False)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, 
                              num_workers=args.num_workers, pin_memory=True)
        print(f"  ✓ 載入 {len(val_set)} 筆驗證資料 from {val_csv_path}")

        # B. 載入該 Fold 的模型
        model_path = os.path.join(args.model_dir, f"{args.experiment_name}_fold{i}.pth")
        if not os.path.exists(model_path):
            print(f"❌ 錯誤：找不到 {model_path}")
            continue
            
        model = get_model(args) 
        model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
        model.to(device)
        print(f"  ✓ 載入模型 from {model_path}")
        
        # C. 載入該 Fold 的 *專屬* Thresholds
        fold_thresh_path = os.path.join(args.model_dir, f"{args.experiment_name}_fold{i}_best_thresholds.json")
        if not os.path.exists(fold_thresh_path):
            print(f"❌ 錯誤：找不到 {fold_thresh_path}")
            continue
            
        with open(fold_thresh_path, 'r', encoding='utf-8') as f:
            fold_thresh_data = json.load(f)
        
        fold_thresholds = [fold_thresh_data[label] for label in labels]
        print(f"  ✓ 載入 Fold {i} 專屬 thresholds")

        # D. 執行評估
        per_class_f1, avg_f1_macro, jaccard, subset_acc, avg_acc_label_based, per_class_acc = evaluate_fold(
            model, val_loader, device, fold_thresholds
        )
        
        # 儲存結果
        fold_results = {
            'per_class_f1': per_class_f1,
            'Average': avg_f1_macro,
            'Jaccard': jaccard,
            'SubsetAcc': subset_acc,
            'AvgAccLabel': avg_acc_label_based, 
            'per_class_acc': per_class_acc 
        }
        all_fold_metrics.append(fold_results)

    # 4. 迴圈結束，計算 mean ± std 並印出完整表格
    print("\n" + "="*120) 
    print(f"  {args.experiment_name} 的 5-Fold 交叉驗證結果 (mean% ± std%)")
    print("="*120)
    
    header = f"{'Model':<25} | {'Avg F1 (Macro)':<15} | {'Avg Acc (Label)':<15} | {'Subset Acc':<15} | {'Jaccard':<15} | {'Params (M)':<10} | {'Size (MB)':<10}"
    print(header)
    print("-" * len(header))
    
    # 提取所有指標
    avg_f1s = [m['Average'] for m in all_fold_metrics]
    jaccards = [m['Jaccard'] for m in all_fold_metrics]
    subset_accs = [m['SubsetAcc'] for m in all_fold_metrics]
    avg_acc_labels = [m['AvgAccLabel'] for m in all_fold_metrics] 
    
    # --- 計算 Mean ± Std (並轉為百分比) ---
    def format_metric(values):
        mean = np.mean(values) * 100
        std = np.std(values) * 100
        return f"{mean:.2f} ± {std:.2f}"
    
    avg_f1_str = format_metric(avg_f1s)
    avg_acc_label_str = format_metric(avg_acc_labels) 
    subset_acc_str = format_metric(subset_accs)
    jaccard_str = format_metric(jaccards)

    # --- 格式化 Params 和 Size ---
    params_m_str = f"{total_params / 1_000_000:.2f} M"
    size_mb_str = f"{model_size_mb:.2f} MB"
    
    model_name_short = args.experiment_name[:25] 
    data_row = f"{model_name_short:<25} | {avg_f1_str:<15} | {avg_acc_label_str:<15} | {subset_acc_str:<15} | {jaccard_str:<15} | {params_m_str:<10} | {size_mb_str:<10}"
    print(data_row)

    # 5. 印出 F1-Score 細項表格
    print("\n" + "-"*120)
    print("  F1-Score (mean% ± std%) 細節")
    
    col_width = max(max(len(label) for label in labels), 10) 
    header_parts = [f"{label:<{col_width}}" for label in labels]
    table_width = len("  " + " | ".join(header_parts))
    print("  " + "-"*(table_width-2))
    print("  " + " | ".join(header_parts))
    
    all_per_class_f1s = np.array([m['per_class_f1'] for m in all_fold_metrics])
    f1_means = np.mean(all_per_class_f1s, axis=0) * 100
    f1_stds = np.std(all_per_class_f1s, axis=0) * 100
    
    mean_parts = [f"{m:>{col_width}.2f}" for m in f1_means]
    std_parts  = [f"{s:>{col_width-1}.2f}" for s in f1_stds] 
    
    print("  " + " | ".join(mean_parts))
    print("  " + " | ".join([f"±{s}" for s in std_parts]))
    
    # 6. 印出 Accuracy 細項表格
    print("\n" + "-"*120)
    print("  Accuracy (mean% ± std%) 細節")
    print("  " + "-"*(table_width-2))
    
    # 🔥 --- 關鍵修改 (v7) ---
    print("  " + " | ".join(header_parts)) # 移除了多餘的 'S'
    # --- 修改結束 ---
    
    all_per_class_accs = np.array([m['per_class_acc'] for m in all_fold_metrics])
    acc_means = np.mean(all_per_class_accs, axis=0) * 100
    acc_stds = np.std(all_per_class_accs, axis=0) * 100
    
    mean_parts_acc = [f"{m:>{col_width}.2f}" for m in acc_means]
    std_parts_acc  = [f"{s:>{col_width-1}.2f}" for s in acc_stds]
    
    print("  " + " | ".join(mean_parts_acc))
    print("  " + " | ".join([f"±{s}" for s in std_parts_acc]))
    print("="*120)

if __name__ == '__main__':
    main()
