# dataset.py — 五分支 Dataset，回傳 (whole, root, center, side, tip), labels
import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from typing import List, Tuple
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# --------- 工具 ---------
def _safe_imread(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def _find_mask_path(img_path: str) -> str:
    base, name = os.path.split(img_path)
    stem, _ = os.path.splitext(name)
    candidates = [
        os.path.join(base, stem + "_mask.png"),
        os.path.join(base, stem + "_mask.jpg"),
        os.path.join(base, stem + "_mask.jpeg"),
        os.path.join(base, "masks", stem + ".png"),
        os.path.join(base, "masks", stem + ".jpg"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""

def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = mask.shape[:2]
        return 0, 0, w, h
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, w, h

def _resolve_image_field(df: pd.DataFrame) -> str:
    low = {c.lower(): c for c in df.columns}
    for k in ["image","img","filename","file","path","filepath","img_path","image_path","imgpath"]:
        if k in low:
            return low[k]
    raise KeyError(f"找不到影像欄位，支援：image/img/filename/file/path/filepath/img_path/image_path/imgpath；實際欄位={list(df.columns)}")

def _five_regions(img: np.ndarray, mask: np.ndarray | None) -> list[np.ndarray]:
    """
    依固定比例分五區。若有 mask，先以 mask 外接框裁切 ROI 再分區。
      - root:   上 20% 全寬
      - center: 中間 50% 的中帶 (x 25%~75%)
      - side:   兩側 (下 80%) 合併
      - tip:    下 30% 的中帶
      - whole:  ROI 全區
    """
    H, W = img.shape[:2]
    if mask is not None:
        x, y, w, h = _bbox_from_mask(mask)
        roi = img[y:y+h, x:x+w]
        mroi = mask[y:y+h, x:x+w]
    else:
        x, y, w, h = 0, 0, W, H
        roi = img
        mroi = None

    root_ratio, center_ratio, tip_ratio = 0.2, 0.5, 0.3
    cx0, cx1 = int(0.25 * w), int(0.75 * w)
    top_h = int(root_ratio * h)
    mid_h = int(center_ratio * h)
    tip_h = int(tip_ratio * h)

    # 切區
    whole  = roi
    root   = roi[0:top_h, :]
    center = roi[top_h:top_h+mid_h, cx0:cx1]
    sideL  = roi[top_h:h, 0:cx0]
    sideR  = roi[top_h:h, cx1:w]
    side   = np.concatenate([sideL, sideR], axis=1) if sideL.size and sideR.size else (sideL if sideL.size else sideR)
    tip    = roi[h-tip_h:h, cx0:cx1]

    if mroi is not None:
        def _apply(a):
            if a.size == 0: return a
            m = cv2.resize(mroi, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
            return cv2.bitwise_and(a, a, mask=(m > 0).astype(np.uint8))
        whole  = _apply(whole)
        root   = _apply(root)
        center = _apply(center)
        side   = _apply(side)
        tip    = _apply(tip)

    return [whole, root, center, side, tip]

# --------- Dataset ---------
class TongueDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 image_root: str,
                 label_cols: List[str],
                 is_train: bool = True,
                 input_size: int = 224,
                 auto_rotate: bool = True):
        self.df = pd.read_csv(csv_path)
        self.img_col = _resolve_image_field(self.df)
        self.image_root = image_root
        self.label_cols = label_cols
        self.is_train = is_train
        self.input_size = input_size
        self.auto_rotate = auto_rotate

        train_tfms = T.Compose([
            T.Resize((input_size, input_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        val_tfms = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.tfms = train_tfms if is_train else val_tfms

        print(f"✔ 載入資料: {os.path.basename(csv_path)} 共 {len(self.df)} 筆 — 影像欄位: {self.img_col} — 自動旋轉: {'啟用' if self.auto_rotate else '關閉'}")

    def __len__(self):
        return len(self.df)

    def _maybe_rotate(self, img: np.ndarray) -> np.ndarray:
        if not self.auto_rotate or not self.is_train:
            return img
        k = random.choice([0, 0, 0, 1, 3])  # 多數不轉，偶爾 ±90°
        return img if k == 0 else np.rot90(img, k).copy()

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        name = str(row[self.img_col]).strip()
        img_path = name if os.path.isabs(name) or os.path.exists(name) else os.path.join(self.image_root, name)

        img = _safe_imread(img_path)
        mask_path = _find_mask_path(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None

        img = self._maybe_rotate(img)
        regions = _five_regions(img, mask)  # [whole, root, center, side, tip]

        tensors = []
        for reg in regions:
            if reg.size == 0:
                reg = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            pil = Image.fromarray(reg)
            tensors.append(self.tfms(pil))

        labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
        return (tensors[0], tensors[1], tensors[2], tensors[3], tensors[4]), labels
