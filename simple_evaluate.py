# simple_evaluate.py - 簡單測試指標
# --- 已修改版：支援指令行參數切換模型與骨幹 ---

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import TongueDataset
# vvv 從 model_zoo 匯入模型 vvv
from model import SignOrientedNetwork, SimpleTimmModel
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score, accuracy_score
import numpy as np
import sys, os, json, argparse

def evaluate_model(args):
    """根據指令行參數評估模型"""
    model_path = args.model_path
    val_csv = args.val_csv
    
    # --- 載入對應的 thresholds ---
    th_path = os.path.splitext(model_path)[0] + "_best_thresholds.json"
    if os.path.exists(th_path):
        with open(th_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)
        print(f"✅ 使用 thresholds 檔案: {th_path}")
    else:
        print(f"⚠ 沒找到對應的 threshold 檔案 ({os.path.basename(th_path)})，全部類別閾值預設 0.5")
        thresholds = {lab: 0.5 for lab in args.label_cols}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # vvv 根據 --model 參數選擇模型 vvv
    if args.model == 'SignOriented':
        model = SignOrientedNetwork(num_classes=len(args.label_cols), backbone=args.backbone, feature_dim=args.feature_dim)
    elif args.model == 'Simple':
        model = SimpleTimmModel(num_classes=len(args.label_cols), backbone=args.backbone)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"載入模型: {model_path} (架構: {args.model}, 骨幹: {args.backbone})")
    val_set = TongueDataset(val_csv, args.image_root, args.label_cols, is_train=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"驗證樣本數: {len(val_set)}")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for (x_whole, x_root, x_center, x_side, x_tip), labels_batch in tqdm(val_loader, desc="Evaluating"):
            x_whole, x_root, x_center, x_side, x_tip = x_whole.to(device), x_root.to(device), x_center.to(device), x_side.to(device), x_tip.to(device)
            logits = model(x_whole, x_root, x_center, x_side, x_tip)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = np.zeros_like(probs, dtype=int)
            for i, lab in enumerate(args.label_cols):
                preds[:, i] = (probs[:, i] > thresholds.get(lab, 0.5)).astype(int)
            all_preds.append(preds)
            all_labels.append(labels_batch.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # --- 計算並顯示指標 ---
    subset_acc = accuracy_score(all_labels, all_preds)
    per_class_acc = [accuracy_score(all_labels[:, i], all_preds[:, i]) for i in range(len(args.label_cols))]
    
    print("\n" + "="*25 + " 總體評估結果 " + "="*25)
    print(f"{'Subset Accuracy (完全匹配率)':<35}: {subset_acc:.4f}")
    print(f"{'Average Accuracy (各類別準確率平均)':<35}: {np.mean(per_class_acc):.4f}")
    print(f"{'F1-Score (Macro)':<35}: {f1_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")
    print(f"{'F1-Score (Micro)':<35}: {f1_score(all_labels, all_preds, average='micro', zero_division=0):.4f}")
    
    print("\n" + "="*25 + " 各類別詳細指標 " + "="*25)
    print(f"{'Label':<15} {'Acc':<7} {'F1':<7} {'Prec':<7} {'Rec':<7} {'Support':<8} {'TH*':<6}")
    print("-" * 65)
    for i, label in enumerate(args.label_cols):
        f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        prec = precision_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        rec = recall_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        support = int(all_labels[:, i].sum())
        th = thresholds.get(label, 0.5)
        print(f"{label:<15} {per_class_acc[i]:<7.3f} {f1:<7.3f} {prec:<7.3f} {rec:<7.3f} {support:<8d} {th:<6.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument('model_path', type=str, help='Path to the trained model (.pth file).')
    parser.add_argument('val_csv', type=str, help='Path to the validation/test CSV file.')
    parser.add_argument('--model', type=str, default='Simple', choices=['SignOriented', 'Simple'], help='Model architecture used for training.')
    parser.add_argument('--backbone', type=str, default='convnext_base', help='Backbone model from timm library.')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--image-root', type=str, default='images')
    parser.add_argument('--feature-dim', type=int, default=512, help='Feature dimension for SignOriented model.')
    
    args = parser.parse_args()
    args.label_cols = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis', 'Crack', 'Toothmark', 'FurThick', 'FurYellow']
    evaluate_model(args)

