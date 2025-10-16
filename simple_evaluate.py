# simple_evaluate.py - 簡單測試指標 (五分支 + 自動載入同名 threshold.json)
import torch
from torch.utils.data import DataLoader
from dataset import TongueDataset
from model import SignOrientedNetwork
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
import numpy as np
import sys, os, json

def evaluate_model(model_path, val_csv, image_root="images"):
    """評估模型指標 (自動使用同名 threshold.json)"""
    labels = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis',
              'Crack', 'Toothmark', 'FurThick', 'FurYellow']

    # 同名 thresholds
    th_path = os.path.splitext(model_path)[0] + "_best_thresholds.json"
    if os.path.exists(th_path):
        with open(th_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)
        print(f"✅ 使用 thresholds 檔案: {th_path}")
    else:
        print("⚠ 沒找到 threshold 檔案，全部類別閾值預設 0.5")
        thresholds = {lab: 0.5 for lab in labels}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignOrientedNetwork(num_classes=len(labels), backbone='swin_base_patch4_window7_224', feature_dim=512)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"載入模型: {model_path}")
    print(f"驗證數據: {val_csv}")

    val_set = TongueDataset(val_csv, image_root, labels, is_train=False)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)

    print(f"驗證樣本數: {len(val_set)}")

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for (x_whole, x_root, x_center, x_side, x_tip), labels_batch in val_loader:
            x_whole  = x_whole.to(device)
            x_root   = x_root.to(device)
            x_center = x_center.to(device)
            x_side   = x_side.to(device)
            x_tip    = x_tip.to(device)
            labels_batch = labels_batch.to(device)

            logits = model(x_whole, x_root, x_center, x_side, x_tip)
            probs = torch.sigmoid(logits).cpu().numpy()

            preds = np.zeros_like(probs, dtype=int)
            for i, lab in enumerate(labels):
                preds[:, i] = (probs[:, i] > thresholds.get(lab, 0.5)).astype(int)

            all_preds.append(preds)
            all_labels.append(labels_batch.cpu().numpy())
            all_probs.append(probs)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)

    print("\n評估結果")
    print("=" * 75)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    jaccard_macro = jaccard_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"F1-Score (Micro): {f1_micro:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"Precision (Macro): {precision_macro:.4f}")
    print(f"Recall (Macro): {recall_macro:.4f}")
    print(f"Jaccard (Macro): {jaccard_macro:.4f}")

    print(f"\n各類別詳細指標")
    print("-" * 75)
    print(f"{'Label':<15} {'F1':<6} {'Prec':<6} {'Rec':<6} {'Jac':<6} {'Support':<8} {'TH*':<6}")
    print("-" * 75)

    for i, label in enumerate(labels):
        f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        precision = precision_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        recall = recall_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        jaccard = jaccard_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        support = int(all_labels[:, i].sum())
        th = thresholds.get(label, 0.5)
        print(f"{label:<15} {f1:<6.3f} {precision:<6.3f} {recall:<6.3f} {jaccard:<6.3f} {support:<8d} {th:<6.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使用方法: python simple_evaluate.py <模型路徑> <驗證CSV>")
        print("例如: python simple_evaluate.py signnet_best_fold1.pth val_fold1.csv")
    else:
        evaluate_model(sys.argv[1], sys.argv[2])
