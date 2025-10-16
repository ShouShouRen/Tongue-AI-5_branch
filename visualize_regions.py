# visualize_regions.py — 檢查是否正確分割五區（原圖疊框 + 五區拼圖）
import os, cv2, argparse, numpy as np, torch
from torchvision.utils import make_grid, save_image
from PIL import Image
import pandas as pd

# ====== 與 dataset.py 一致的分區邏輯 ======
def _safe_imread(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def _find_mask_path(img_path: str) -> str:
    base, name = os.path.split(img_path); stem, _ = os.path.splitext(name)
    for p in [os.path.join(base, stem + "_mask.png"),
              os.path.join(base, stem + "_mask.jpg"),
              os.path.join(base, "masks", stem + ".png"),
              os.path.join(base, "masks", stem + ".jpg")]:
        if os.path.exists(p): return p
    return ""

def _bbox_from_mask(mask: np.ndarray):
    th = (mask > 0).astype(np.uint8)
    cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        h, w = mask.shape[:2]
        return 0, 0, w, h
    cnt = max(cs, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, w, h

def five_regions(img: np.ndarray, mask: np.ndarray | None):
    H, W = img.shape[:2]
    if mask is not None:
        x, y, w, h = _bbox_from_mask(mask); roi = img[y:y+h, x:x+w]
    else:
        x, y, w, h = 0, 0, W, H; roi = img
    root_ratio, center_ratio, tip_ratio = 0.2, 0.5, 0.3
    side_x = (0.25, 0.75)
    top_h = int(root_ratio * h); mid_h = int(center_ratio * h); tip_h = int(tip_ratio * h)
    cx0, cx1 = int(side_x[0]*w), int(side_x[1]*w)
    boxes = {
        "root":   (0,           top_h,       0,   w),
        "center": (top_h,       top_h+mid_h, cx0, cx1),
        "sideL":  (top_h,       h,           0,   cx0),
        "sideR":  (top_h,       h,           cx1, w),
        "tip":    (h-tip_h,     h,           cx0, cx1),
        "whole":  (0,           h,           0,   w),
    }
    crops = {
        "whole": roi[boxes["whole"][0]:boxes["whole"][1], boxes["whole"][2]:boxes["whole"][3]],
        "root":  roi[boxes["root"][0]:boxes["root"][1],   boxes["root"][2]:boxes["root"][3]],
        "center":roi[boxes["center"][0]:boxes["center"][1], boxes["center"][2]:boxes["center"][3]],
        "side":  np.concatenate([
                    roi[boxes["sideL"][0]:boxes["sideL"][1], boxes["sideL"][2]:boxes["sideL"][3]],
                    roi[boxes["sideR"][0]:boxes["sideR"][1], boxes["sideR"][2]:boxes["sideR"][3]],
                 ], axis=1) if (cx0>0 and cx1<w) else roi[boxes["sideL"][0]:boxes["sideL"][1], boxes["sideL"][2]:boxes["sideL"][3]],
        "tip":   roi[boxes["tip"][0]:boxes["tip"][1],     boxes["tip"][2]:boxes["tip"][3]],
    }
    return crops, boxes, (x, y), (h, w)

# ====== 視覺化 ======
COLORS = {"root":(0,80,255),"center":(255,200,0),"sideL":(0,255,0),"sideR":(0,180,0),"tip":(255,0,180),"whole":(180,180,180)}

def draw_boxes_on_image(img_rgb: np.ndarray, boxes: dict, origin_xy, roi_hw):
    x0, y0 = origin_xy; h, w = roi_hw
    img = img_rgb.copy()
    for name in ["whole","root","center","sideL","sideR","tip"]:
        y1, y2, x1, x2 = boxes[name]
        p1 = (x0 + x1, y0 + y1); p2 = (x0 + x2, y0 + y2)
        cv2.rectangle(img, p1, p2, COLORS[name][::-1], 2)
        cv2.putText(img, name, (p1[0]+3, p1[1]+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[name][::-1], 2, cv2.LINE_AA)
    return img

def npimg_to_tensor(np_img: np.ndarray):
    if np_img.ndim == 2: np_img = np.repeat(np_img[...,None], 3, axis=2)
    return torch.from_numpy(np_img).permute(2,0,1).float()/255.0

def save_panel(img_rgb, crops, out_path, tile_size=224):
    left = cv2.resize(img_rgb, (tile_size*2, tile_size*2), interpolation=cv2.INTER_AREA)
    tiles = []
    for k in ["whole","root","center","side","tip"]:
        c = crops[k]
        c = np.zeros((tile_size, tile_size, 3), dtype=np.uint8) if c.size == 0 else cv2.resize(c, (tile_size, tile_size))
        tiles.append(npimg_to_tensor(c))
    right_grid = make_grid(torch.stack(tiles, 0), nrow=5)
    right_np = (right_grid.permute(1,2,0).numpy()*255).astype(np.uint8)
    right_np = cv2.resize(right_np, (tile_size*5, tile_size*2), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((tile_size*2, tile_size*7, 3), dtype=np.uint8)
    canvas[:, :tile_size*2] = left
    canvas[:, tile_size*2:] = right_np
    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

# ====== CSV 欄位自動偵測 ======
def resolve_image_field(df: pd.DataFrame) -> str:
    low = {c.lower(): c for c in df.columns}
    candidates = [
        "image","img","filename","file","path","filepath","img_path","image_path","imgpath"
    ]
    for k in candidates:
        if k in low:
            return low[k]
    raise KeyError(
        f"CSV 找不到影像欄位，請提供其中之一：{candidates}；實際欄位={list(df.columns)}"
    )
def main():
    ap = argparse.ArgumentParser("Visualize five-region splitting")
    ap.add_argument("csv", type=str, help="CSV 檔路徑")
    ap.add_argument("--image_root", type=str, default="images")
    ap.add_argument("--out_dir", type=str, default="region_vis")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--draw_boxes", action="store_true", help="輸出原圖疊框")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)
    img_col = resolve_image_field(df)
    n = len(df) if args.limit <= 0 else min(args.limit, len(df))
    print(f"Total {len(df)} rows, visualize {n}, using column '{img_col}'")

    for i in range(n):
        row = df.iloc[i]
        name = str(row[img_col]).strip()
        img_path = name if os.path.isabs(name) or os.path.exists(name) else os.path.join(args.image_root, name)
        img = _safe_imread(img_path)

        mask_path = _find_mask_path(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None

        crops, boxes, origin, roi_hw = five_regions(img, mask)

        if args.draw_boxes:
            boxed = draw_boxes_on_image(img, boxes, origin, roi_hw)
            out1 = os.path.join(args.out_dir, f"{os.path.splitext(os.path.basename(name))[0]}_boxes.jpg")
            cv2.imwrite(out1, cv2.cvtColor(boxed, cv2.COLOR_RGB2BGR))

        out2 = os.path.join(args.out_dir, f"{os.path.splitext(os.path.basename(name))[0]}_panel.jpg")
        save_panel(img, crops, out2)
        print(f"saved: {out2}")

    print("done.")

if __name__ == "__main__":
    main()
