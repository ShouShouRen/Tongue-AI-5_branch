# simple_evaluate.py - 簡單測試指標 (五分支 + 自動載入同名 threshold.json)
# --- 已修改版：加入 Average Accuracy 指標 ---

import torch
from torch.utils.data import DataLoader
from dataset import TongueDataset
from model import SignOrientedNetwork
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score, accuracy_score
import numpy as np
import sys, os, json

def evaluate_model(model_path, val_csv, image_root="images"):
    """評估模型指標 (自動使用同名 threshold.json)"""
    labels = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis',
              'Crack', 'Toothmark', 'FurThick', 'FurYellow']

    # --- 載入對應的 thresholds ---
    th_path = os.path.splitext(model_path)[0] + "_best_thresholds.json"
    if os.path.exists(th_path):
        with open(th_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)
        print(f"✅ 使用 thresholds 檔案: {th_path}")
    else:
        print(f"⚠ 沒找到對應的 threshold 檔案 ({th_path})，全部類別閾值預設 0.5")
        thresholds = {lab: 0.5 for lab in labels}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignOrientedNetwork(num_classes=len(labels), backbone='swin_base_patch4_window7_224', feature_dim=512)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()

    print(f"載入模型: {model_path}")
    print(f"驗證數據: {val_csv}")

    val_set = TongueDataset(val_csv, image_root, labels, is_train=False)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)

    print(f"驗證樣本數: {len(val_set)}")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for (x_whole, x_root, x_center, x_side, x_tip), labels_batch in val_loader:
            x_whole  = x_whole.to(device)
            x_root   = x_root.to(device)
            x_center = x_center.to(device)
            x_side   = x_side.to(device)
            x_tip    = x_tip.to(device)

            logits = model(x_whole, x_root, x_center, x_side, x_tip)
            probs = torch.sigmoid(logits).cpu().numpy()

            preds = np.zeros_like(probs, dtype=int)
            for i, lab in enumerate(labels):
                preds[:, i] = (probs[:, i] > thresholds.get(lab, 0.5)).astype(int)

            all_preds.append(preds)
            all_labels.append(labels_batch.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # --- 預先計算所有類別的指標 ---
    per_class_acc = [accuracy_score(all_labels[:, i], all_preds[:, i]) for i in range(len(labels))]
    per_class_f1 = [f1_score(all_labels[:, i], all_preds[:, i], zero_division=0) for i in range(len(labels))]
    per_class_prec = [precision_score(all_labels[:, i], all_preds[:, i], zero_division=0) for i in range(len(labels))]
    per_class_rec = [recall_score(all_labels[:, i], all_preds[:, i], zero_division=0) for i in range(len(labels))]
    per_class_jac = [jaccard_score(all_labels[:, i], all_preds[:, i], zero_division=0) for i in range(len(labels))]

    # --- 計算整體指標 ---
    print("\n評估結果")
    print("=" * 75)
    subset_acc = accuracy_score(all_labels, all_preds)
    avg_acc = np.mean(per_class_acc)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"Subset Accuracy:  {subset_acc:.4f}  (完全匹配率)")
    print(f"Average Accuracy: {avg_acc:.4f}  (各類別準確率平均)")
    print(f"F1-Score (Micro): {f1_micro:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")

    # --- 顯示各類別詳細指標 ---
    print(f"\n各類別詳細指標")
    print("-" * 82)
    print(f"{'Label':<15} {'Acc':<7} {'F1':<7} {'Prec':<7} {'Rec':<7} {'Jac':<7} {'Support':<8} {'TH*':<6}")
    print("-" * 82)

    for i, label in enumerate(labels):
        support = int(all_labels[:, i].sum())
        th = thresholds.get(label, 0.5)
        print(f"{label:<15} {per_class_acc[i]:<7.3f} {per_class_f1[i]:<7.3f} {per_class_prec[i]:<7.3f} {per_class_rec[i]:<7.3f} {per_class_jac[i]:<7.3f} {support:<8d} {th:<6.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使用方法: python simple_evaluate.py <模型路徑> <驗證CSV>")
        print("例如 (Fold 1): python simple_evaluate.py signnet_best_fold1.pth val_fold1.csv")
        print("例如 (Final):  python simple_evaluate.py signnet_final_model.pth your_test_set.csv")
    else:
        evaluate_model(sys.argv[1], sys.argv[2])
