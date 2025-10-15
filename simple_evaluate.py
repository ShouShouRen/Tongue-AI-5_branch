# simple_evaluate.py — 5-branch，支援 Best-TH 載入/求解/列印/存檔
import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

from dataset import TongueDataset
from model import SignOrientedNetwork

LABELS = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis',
          'Crack', 'Toothmark', 'FurThick', 'FurYellow']

def find_best_thresholds(P, Y, grid=None):
    if grid is None:
        grid = np.concatenate([np.linspace(0.1, 0.9, 17), np.array([0.95])])
    C = Y.shape[1]
    best_th = np.zeros(C, dtype=np.float32)
    for i in range(C):
        if Y[:, i].sum() == 0:
            best_th[i] = 0.5
            continue
        best_f1, best_t = -1.0, 0.5
        y = Y[:, i]
        pi = P[:, i]
        for t in grid:
            f1 = f1_score(y, (pi > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_th[i] = best_t
    return best_th

def eval_with_thresholds(P, Y, th):
    preds = (P > th.reshape(1, -1)).astype(int)
    res = dict(
        f1_micro=f1_score(Y, preds, average='micro', zero_division=0),
        f1_macro=f1_score(Y, preds, average='macro', zero_division=0),
        p_macro =precision_score(Y, preds, average='macro', zero_division=0),
        r_macro =recall_score(Y, preds, average='macro', zero_division=0),
        j_macro=jaccard_score(Y, preds, average='macro', zero_division=0),
        per_cls=[(
            f1_score(Y[:, i], preds[:, i], zero_division=0),
            precision_score(Y[:, i], preds[:, i], zero_division=0),
            recall_score(Y[:, i], preds[:, i], zero_division=0),
            jaccard_score(Y[:, i], preds[:, i], zero_division=0),
            int(Y[:, i].sum())
        ) for i in range(Y.shape[1])]
    )
    return res

def evaluate_model(model_path, val_csv, image_root="images", batch_size=16, save_if_missing=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignOrientedNetwork(num_classes=8, backbone='swin_base_patch4_window7_224', feature_dim=512).to(device)

    ckpt = torch.load(model_path, map_location=device)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    th_path = os.path.splitext(model_path)[0] + "_best_thresholds.json"
    per_class_th = None
    if os.path.exists(th_path):
        with open(th_path, "r", encoding="utf-8") as f:
            th_dict = json.load(f)
        per_class_th = np.array([float(th_dict[k]) for k in LABELS], dtype=np.float32)
        print(f"使用最佳門檻: {th_path}")
    else:
        print("未找到最佳門檻 JSON，稍後自動尋找 Best-TH")

    val_set = TongueDataset(val_csv, image_root, LABELS, is_train=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"載入模型: {model_path}")
    print(f"驗證數據: {val_csv}")
    print(f"驗證樣本數: {len(val_set)}")

    all_probs, all_labels = [], []
    with torch.no_grad():
        for (x_whole, x_root, x_center, x_side, x_tip), labels_batch in val_loader:
            x_whole  = x_whole.to(device); x_root   = x_root.to(device)
            x_center = x_center.to(device); x_side   = x_side.to(device)
            x_tip    = x_tip.to(device)
            logits = model(x_whole, x_root, x_center, x_side, x_tip)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels_batch.numpy())

    P = np.vstack(all_probs)   # [N,8]
    Y = np.vstack(all_labels)  # [N,8]

    # 求或載入 Best-TH
    if per_class_th is None:
        per_class_th = find_best_thresholds(P, Y)
        print("\n自動尋找 Best-TH 完成")
        if save_if_missing:
            with open(th_path, "w", encoding="utf-8") as f:
                json.dump({LABELS[i]: float(per_class_th[i]) for i in range(len(LABELS))}, f, ensure_ascii=False, indent=2)
            print(f"已存檔: {th_path}")

    # 列印 Best-TH 表
    print("\nBest-TH（逐類）")
    print("-" * 60)
    for i, lb in enumerate(LABELS):
        print(f"{lb:<15} TH*={per_class_th[i]:.2f}")

    # 用 Best-TH 與 TH=0.5 各評一次
    res_best = eval_with_thresholds(P, Y, per_class_th)
    res_05   = eval_with_thresholds(P, Y, np.full(len(LABELS), 0.5, dtype=np.float32))

    def show_block(title, R):
        print(f"\n{title}")
        print("=" * 60)
        print(f"F1-Score (Micro): {R['f1_micro']:.4f}")
        print(f"F1-Score (Macro): {R['f1_macro']:.4f}")
        print(f"Precision (Macro): {R['p_macro']:.4f}")
        print(f"Recall (Macro): {R['r_macro']:.4f}")
        print(f"Jaccard (Macro): {R['j_macro']:.4f}")
        print(f"\n各類別詳細指標")
        print("-" * 60)
        print(f"{'Label':<15} {'F1':<6} {'Prec':<6} {'Rec':<6} {'Jac':<6} {'Support':<8}")
        print("-" * 60)
        for i, lb in enumerate(LABELS):
            f1, pr, rc, jc, sup = R['per_cls'][i]
            print(f"{lb:<15} {f1:<6.3f} {pr:<6.3f} {rc:<6.3f} {jc:<6.3f} {sup:<8d}")

    show_block("評估結果 @ Best-TH", res_best)
    show_block("評估結果 @ TH=0.5",   res_05)

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("用法: python simple_evaluate.py <模型路徑> [<驗證CSV>]；若省略則用 val_fold1.csv")
        sys.exit(1)
    model_path = sys.argv[1]
    val_csv = sys.argv[2] if len(sys.argv) == 3 else "val_fold1.csv"
    evaluate_model(model_path, val_csv)
