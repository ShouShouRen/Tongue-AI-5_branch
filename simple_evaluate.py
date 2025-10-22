# simple_evaluate.py - 靈活的模型評估工具
# --- 已同步更新，可透過指令行參數選擇模型 ---

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys, os, json, argparse

# 從模型庫匯入所有模型
from model import SignOrientedNetwork, SimpleTimmModel, SignOrientedAttentionNetwork
from dataset import TongueDataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

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
        
    return model_class(
        num_classes=len(args.label_cols),
        backbone=args.backbone,
        feature_dim=args.feature_dim
    )

def evaluate_model(args):
    """評估模型指標 (自動使用同名 threshold.json)"""

    # --- 載入對應的 thresholds ---
    th_path = os.path.splitext(args.model_path)[0] + "_best_thresholds.json"
    if os.path.exists(th_path):
        with open(th_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)
        print(f"✅ 使用 thresholds 檔案: {th_path}")
    else:
        print(f"⚠ 沒找到對應的 threshold 檔案 ({os.path.basename(th_path)})，全部類別閾值預設 0.5")
        thresholds = {lab: 0.5 for lab in args.label_cols}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 動態建立模型 ---
    model = get_model(args)

    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()

    print(f"載入模型: {os.path.basename(args.model_path)} (架構: {args.model}, 骨幹: {args.backbone})")

    val_set = TongueDataset(args.val_csv, args.image_root, args.label_cols, is_train=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"驗證樣本數: {len(val_set)}")

    all_preds, all_labels = [], []
    loop = tqdm(val_loader, desc="Evaluating")
    with torch.no_grad():
        for (x_whole, x_root, x_center, x_side, x_tip), labels_batch in loop:
            logits = model(x_whole.to(device), x_root.to(device), x_center.to(device), x_side.to(device), x_tip.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()

            preds = np.zeros_like(probs, dtype=int)
            for i, lab in enumerate(args.label_cols):
                preds[:, i] = (probs[:, i] > thresholds.get(lab, 0.5)).astype(int)

            all_preds.append(preds)
            all_labels.append(labels_batch.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # --- 預先計算所有類別的指標 ---
    per_class_acc = [accuracy_score(all_labels[:, i], all_preds[:, i]) for i in range(len(args.label_cols))]
    per_class_f1 = [f1_score(all_labels[:, i], all_preds[:, i], zero_division=0) for i in range(len(args.label_cols))]
    per_class_prec = [precision_score(all_labels[:, i], all_preds[:, i], zero_division=0) for i in range(len(args.label_cols))]
    per_class_rec = [recall_score(all_labels[:, i], all_preds[:, i], zero_division=0) for i in range(len(args.label_cols))]

    # --- 計算整體指標 ---
    print("\n" + "="*25 + " 總體評估結果 " + "="*25)
    subset_acc = accuracy_score(all_labels, all_preds)
    avg_acc = np.mean(per_class_acc)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"Subset Accuracy (完全匹配率)         : {subset_acc:.4f}")
    print(f"Average Accuracy (各類別準確率平均)      : {avg_acc:.4f}")
    print(f"F1-Score (Macro)                   : {f1_macro:.4f}")
    print(f"F1-Score (Micro)                   : {f1_micro:.4f}")

    # --- 顯示各類別詳細指標 ---
    print("\n" + "="*25 + " 各類別詳細指標 " + "="*25)
    print(f"{'Label':<15} {'Acc':<7} {'F1':<7} {'Prec':<7} {'Rec':<7} {'Support':<8} {'TH*':<6}")
    print("-" * 65)

    for i, label in enumerate(args.label_cols):
        support = int(all_labels[:, i].sum())
        th = thresholds.get(label, 0.5)
        print(f"{label:<15} {per_class_acc[i]:<7.3f} {per_class_f1[i]:<7.3f} {per_class_prec[i]:<7.3f} {per_class_rec[i]:<7.3f} {support:<8d} {th:<6.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='評估舌象辨識模型')
    parser.add_argument('model_path', type=str, help='要評估的模型權重檔案路徑 (.pth)')
    parser.add_argument('val_csv', type=str, help='驗證資料的 CSV 檔案路徑')
    parser.add_argument('--model', type=str, required=True, choices=['Simple', 'SignOriented', 'SignOrientedAttention'], help='要使用的模型架構')
    parser.add_argument('--backbone', type=str, required=True, help='模型使用的骨幹網路, e.g., "convnext_base"')
    
    # --- 和 train.py 保持一致的非必要參數 ---
    parser.add_argument('--feature_dim', type=int, default=512, help='特徵維度 (僅用於 SignOriented 系列模型)')
    parser.add_argument('--image_root', type=str, default='images', help='圖片根目錄')
    parser.add_argument('--batch_size', type=int, default=32, help='評估時的批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='資料載入器的工作線程數')
    
    args = parser.parse_args()
    
    # 將固定的標籤列表加入 args
    args.label_cols = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis',
                       'Crack', 'Toothmark', 'FurThick', 'FurYellow']

    evaluate_model(args)


### 如何使用

#現在，您的評估指令會和訓練指令完美對應：


# 評估 "Simple" 模型
#python simple_evaluate.py Simple_convnext_base_final.pth test.csv --model Simple --backbone convnext_base

# 評估舊的 "SignOriented" 模型
#python simple_evaluate.py SignOriented_swin_base_patch4_window7_224_final.pth test.csv --model SignOriented --backbone swin_base_patch4_window7_224

# 評估新的 "SignOrientedAttention" 模型
#python simple_evaluate.py SignOrientedAttention_convnext_base_final.pth test.csv --model SignOrientedAttention --backbone convnext_base

