# debug_dataset.py
import os, json
import torch
import numpy as np
import cv2
from PIL import Image
from dataset import TongueDataset   # 直接用你現有的 dataset.py

# ==== 路徑設定 ====
CSV = "train_fold1.csv"
IMG_ROOT = "images"
LABEL_COLS = ['TonguePale','TipSideRed','Spot','Ecchymosis','Crack','Toothmark','FurThick','FurYellow']
N = 12   # 取前 N 筆做檢查

os.makedirs("debug_out", exist_ok=True)

def to_np_img(t):
    # tensor(CHW) [0,1] or uint8 → HWC uint8
    if isinstance(t, torch.Tensor):
        t = t.cpu().numpy()
        if t.ndim == 3: t = np.transpose(t, (1,2,0))
        t = (t * 255.0).clip(0,255).astype(np.uint8) if t.max()<=1.0 else t.astype(np.uint8)
    return t

def stack5(imgs, titles=None, out_path=None):
    # imgs: [whole, root, center, side, tip], 每張 224x224x3
    h, w = imgs[0].shape[:2]
    grid = np.zeros((h*1, w*5, 3), dtype=np.uint8)
    for i, im in enumerate(imgs):
        grid[:, i*w:(i+1)*w, :] = im
    if titles:
        # 簡單在上面畫字
        for i, t in enumerate(titles):
            cv2.putText(grid, t, (i*w+6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    if out_path:
        cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return grid

def nonzero_ratio(img):
    return float((img.sum(axis=2) > 0).mean())

if __name__ == "__main__":
    ds = TongueDataset(CSV, IMG_ROOT, LABEL_COLS, is_train=False)
    print(f"Loaded dataset size = {len(ds)}")

    stats = []
    for i in range(min(N, len(ds))):
        (img_whole, root, center, side, tip), labels = ds[i]

        # 還原到可視化（dataset 裡做了 Normalize → 這裡只做簡單反標準化）
        def denorm(t):
            # 反 Normalize: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
            mean = np.array([0.485,0.456,0.406]).reshape(1,1,3)
            std  = np.array([0.229,0.224,0.225]).reshape(1,1,3)
            x = to_np_img(t.float())
            x = x.astype(np.float32)/255.0
            x = (x * std + mean)
            x = (x*255.0).clip(0,255).astype(np.uint8)
            return x

        w = denorm(img_whole); r = denorm(root); c = denorm(center); s = denorm(side); t = denorm(tip)

        nz = {
            "root":   nonzero_ratio(r),
            "center": nonzero_ratio(c),
            "side":   nonzero_ratio(s),
            "tip":    nonzero_ratio(t),
        }
        titles = [
            "whole",
            f"root nz={nz['root']:.2f}",
            f"center nz={nz['center']:.2f}",
            f"side nz={nz['side']:.2f}",
            f"tip nz={nz['tip']:.2f}",
        ]
        out_path = os.path.join("debug_out", f"sample_{i:03d}.jpg")
        stack5([w,r,c,s,t], titles, out_path)

        stats.append({"idx": i, "nonzero": nz, "labels": labels.numpy().tolist(), "out": out_path})
        print(f"[#{i}] saved {out_path} | nz={nz} | labels={labels.numpy().tolist()}")

    with open("debug_out/_summary.json","w",encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("Done. 查看 debug_out/ 下的拼圖與 _summary.json。")
