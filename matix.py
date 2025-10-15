# simple_evaluate.py — 多標籤評估 + 三種8×8矩陣
import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import TongueDataset
from model import SignOrientedNetwork

def _safe_div(a, b):
    b = np.maximum(b, 1e-9)
    return a / b

def plot_multilabel_matrices(y_true, y_pred, labels, prefix):
    """
    產出三張 8×8 圖：
      1) cooc: 共現率        P(pred j | true i)
      2) mis:  誤歸類率      P(pred j & true j=0 | true i)
      3) bias: 過度預測偏向  P(pred j|true i) - P(pred j|true j=0)
    """
    C = len(labels)
    supp_i = y_true.sum(axis=0)                          # [C]
    notj = (1 - y_true)                                  # [N,C]

    # 共現：true i=1 且 pred j=1
    cooc_cnt = (y_true[:, :, None] * y_pred[:, None, :]).sum(axis=0)   # [C,C]
    cooc_rate = _safe_div(cooc_cnt, supp_i[:, None])

    # 誤歸類：true i=1 且 pred j=1 且 true j=0
    mis_cnt = (y_true[:, :, None] * y_pred[:, None, :] * notj[:, None, :]).sum(axis=0)
    mis_rate = _safe_div(mis_cnt, supp_i[:, None])

    # 過度預測偏向
    p_pred_given_truei = cooc_rate.copy()                # [C,C]
    denom_notj = notj.sum(axis=0)                        # [C]
    p_pred_given_notj = _safe_div((notj * y_pred).sum(axis=0), denom_notj)  # [C]
    bias = p_pred_given_truei - p_pred_given_notj[None, :]

    def _heat(mat, title, fname, fmt=".2f"):
        plt.figure(figsize=(9, 7))
        sns.heatmap(mat, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted j")
        plt.ylabel("True i")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"saved: {fname}")

    _heat(cooc_rate, "Co-occurrence rate  P(pred j | true i)", f"{prefix}_cooc.png")
    _heat(mis_rate,  "Misclassification rate  P(pred j & true j=0 | true i)", f"{prefix}_mis.png")
    _heat(bias,      "Over-prediction bias  P(pred j|true i) - P(pred j|true j=0)", f"{prefix}_bias.png")

def evaluate_model(model_path, val_csv, image_root="images"):
    labels = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis',
              'Crack', 'Toothmark', 'FurThick', 'FurYellow']

    # thresholds：同名 json（若無則 0.5）
    th_path = os.path.splitext(model_path)[0] + ".json"
    if os.path.exists(th_path):
        with open(th_path, "r", encoding="utf-8") as f:
            th_dict = json.load(f)
        thresholds = np.array([th_dict.get(lab, 0.5) for lab in labels], dtype=np.float32)
        print(f"使用 thresholds: {th_path}")
    else:
        thresholds = np.full(len(labels), 0.5, dtype=np.float32)
        print("未找到 thresholds，使用 0.5")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignOrientedNetwork(num_classes=len(labels),
                                backbone='swin_base_patch4_window7_224',
                                feature_dim=512)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    val_set = TongueDataset(val_csv, image_root, labels, is_train=False)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)
    print(f"驗證樣本數: {len(val_set)}")

    all_probs, all_labels = [], []
    with torch.no_grad():
        for (x_whole, x_root, x_center, x_side, x_tip), y in val_loader:
            x_whole  = x_whole.to(device)
            x_root   = x_root.to(device)
            x_center = x_center.to(device)
            x_side   = x_side.to(device)
            x_tip    = x_tip.to(device)
            y = y.to(device)

            logits = model(x_whole, x_root, x_center, x_side, x_tip)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    preds = (all_probs > thresholds[None, :]).astype(int)

    # 指標
    print("\n評估結果")
    print("=" * 60)
    f1_micro = f1_score(all_labels, preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, preds, average='macro', zero_division=0)
    prec_m = precision_score(all_labels, preds, average='macro', zero_division=0)
    rec_m = recall_score(all_labels, preds, average='macro', zero_division=0)
    jac_m = jaccard_score(all_labels, preds, average='macro', zero_division=0)
    print(f"F1-Score (Micro): {f1_micro:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"Precision (Macro): {prec_m:.4f}")
    print(f"Recall (Macro): {rec_m:.4f}")
    print(f"Jaccard (Macro): {jac_m:.4f}")

    print("\n各類別詳細指標")
    print("-" * 60)
    print(f"{'Label':<15} {'F1':<6} {'Prec':<6} {'Rec':<6} {'Jac':<6} {'Support':<8} {'TH*':<6}")
    print("-" * 60)
    for i, lab in enumerate(labels):
        f1 = f1_score(all_labels[:, i], preds[:, i], zero_division=0)
        pr = precision_score(all_labels[:, i], preds[:, i], zero_division=0)
        rc = recall_score(all_labels[:, i], preds[:, i], zero_division=0)
        jc = jaccard_score(all_labels[:, i], preds[:, i], zero_division=0)
        supp = int(all_labels[:, i].sum())
        print(f"{lab:<15} {f1:<6.3f} {pr:<6.3f} {rc:<6.3f} {jc:<6.3f} {supp:<8d} {thresholds[i]:<6.2f}")

    # 三種矩陣
    prefix = os.path.splitext(model_path)[0] + "_confmat"
    plot_multilabel_matrices(all_labels, preds, labels, prefix)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使用方法: python simple_evaluate.py <模型路徑> <驗證CSV>")
        print("例如: python simple_evaluate.py signnet_best_fold1.pth val_fold1.csv")
    else:
        evaluate_model(sys.argv[1], sys.argv[2])
