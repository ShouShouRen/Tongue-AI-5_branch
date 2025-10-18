# dataset.py
# --- 已升級為混合式遮罩演算法 (HSV + OTSU + 最大連通元件) ---

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def create_hybrid_mask(img: np.ndarray) -> np.ndarray:
    """
    結合 HSV 顏色分割和 OTSU 灰階分割，並找出最大連通元件，產生更穩健的遮罩。
    """
    # --- 1. HSV 顏色分割 ---
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # 放寬顏色範圍以包含更多可能性
    lower_red1 = np.array([0, 40, 50])
    upper_red1 = np.array([20, 255, 255])
    mask_hsv1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([155, 40, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_hsv2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    hsv_mask = cv2.bitwise_or(mask_hsv1, mask_hsv2)

    # --- 2. OTSU 灰階分割 ---
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- 3. 合併兩種遮罩 (聯集) ---
    combined_mask = cv2.bitwise_or(hsv_mask, otsu_mask)

    # --- 4. 形態學後處理 ---
    # 使用一個較大的 kernel 來強力填補內部孔洞
    kernel_close = np.ones((15, 15), np.uint8)
    mask_closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)

    # --- 5. 找出最大連通元件 ---
    num_labels, labels_im = cv2.connectedComponents(mask_closed)
    
    if num_labels < 2: # 如果沒有找到前景物件
        return mask_closed

    max_size = 0
    largest_label = 0
    for i in range(1, num_labels): # 從 1 開始以忽略背景
        component_size = np.sum(labels_im == i)
        if component_size > max_size:
            max_size = component_size
            largest_label = i
            
    final_mask = (labels_im == largest_label).astype(np.uint8) * 255
    
    return final_mask

class TongueDataset(Dataset):
    """
    - 使用混合式遮罩 (HSV+OTSU+最大連通元件) 取外接框，resize 到 384
    - 五分區：root(上)、center(中段)、side(左右下段)、tip(下)；root/tip 有放寬與保底
    - 每區：區內保留彩色，區外=黑色；若有效像素太少則回退整圖
    """
    def __init__(self, csv_file, image_root, label_cols,
                 is_train=True, rotate_upright_enabled=False):
        self.data = pd.read_csv(csv_file)
        self.image_root = image_root
        self.label_cols = label_cols
        self.is_train = is_train
        self.rotate_upright_enabled = rotate_upright_enabled

        # 解析圖片欄位名稱
        self.img_col_name = self._resolve_image_field(self.data)

        print(f"✔ 載入資料: {os.path.basename(csv_file)} 共 {len(self.data)} 筆 — 影像欄位: {self.img_col_name} — 自動直立: {'啟用' if rotate_upright_enabled else '停用'}")

        if is_train:
            self.augment = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=10,
                                   border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
                A.HueSaturationValue(5, 10, 5, p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.CoarseDropout(max_holes=2, max_height=32, max_width=32,
                                min_holes=1, fill_value=0, p=0.3),
            ])
        else:
            self.augment = None

        self.normalize_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def _resolve_image_field(self, df: pd.DataFrame) -> str:
        """從 DataFrame 中找到圖片路徑欄位的實際名稱"""
        low = {c.lower(): c for c in df.columns}
        for k in ["image_path", "image", "img", "filename", "file", "path", "filepath", "img_path", "imgpath"]:
            if k in low:
                return low[k]
        raise KeyError(f"在 {os.path.basename(df)} 中找不到影像欄位。")

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _nonzero_ratio(rgb):
        # rgb: (H, W, 3) uint8
        return (rgb.sum(axis=2) > 0).mean()

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_root, row[self.img_col_name])
        img = np.array(Image.open(img_path).convert('RGB'))

        # 使用新的混合式遮罩演算法
        mask = create_hybrid_mask(img)

        # 外接框裁切 + resize 到 384
        x, y, w, h = cv2.boundingRect(mask)
        if w == 0 or h == 0: # 如果遮罩為空，使用整張圖避免錯誤
            x, y, w, h = 0, 0, img.shape[1], img.shape[0]
            
        cropped = img[y:y + h, x:x + w]
        mask_c = mask[y:y + h, x:x + w]

        resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LINEAR)
        mask_r = cv2.resize(mask_c, (224, 224), interpolation=cv2.INTER_NEAREST)

        # ---------- 分區（根在上、尖在下；左右為 side） ----------
        H, W = mask_r.shape
        root_ratio, center_ratio = 0.28, 0.50
        top_h, mid_h = int(root_ratio * H), int(center_ratio * H)
        bot_start = top_h + mid_h
        root_extra, root_min_frac = 0.06, 0.22
        tip_extra, tip_min_frac, tip_widen = 0.06, 0.18, 0.18

        zone_mask = np.zeros_like(mask_r, dtype=np.uint8)

        # root（上方整寬；高度放寬 + 保底）
        root_end = max(int(root_min_frac * H), min(H, top_h + int(root_extra * H)))
        zone_mask[0:root_end, :] = 1

        # center（以 root_end 為上界，維持中段高度，水平 25%~75%）
        center_top = root_end
        center_bot = min(H, center_top + mid_h)
        cx_l, cx_r = int(0.25 * W), int(0.75 * W)
        zone_mask[center_top:center_bot, cx_l:cx_r] = 2

        # side（從 center_top 以下到底，左右各 25%）
        zone_mask[center_top:H, 0:int(0.25 * W)] = 3
        zone_mask[center_top:H, int(0.75 * W):W] = 3

        # tip（底部中段；縱向放寬 + 保底；水平再放寬）
        tip_start = max(0, bot_start - int(tip_extra * H))
        tip_min_h = int(tip_min_frac * H)
        if (H - tip_start) < tip_min_h:
            tip_start = max(0, H - tip_min_h)
        tx_l = max(0, int((0.25 - tip_widen) * W))
        tx_r = min(W, int((0.75 + tip_widen) * W))
        zone_mask[tip_start:H, tx_l:tx_r] = 4

        zone_mask = cv2.bitwise_and(zone_mask, zone_mask, mask=mask_r)

        def cut(val: int):
            m = (zone_mask == val).astype(np.uint8)
            return cv2.bitwise_and(resized, resized, mask=m)

        img_whole = resized
        root_img = cut(1)
        center_img = cut(2)
        side_img = cut(3)
        tip_img = cut(4)

        if self._nonzero_ratio(root_img) < 0.10:   root_img = img_whole.copy()
        if self._nonzero_ratio(center_img) < 0.10: center_img = img_whole.copy()
        if self._nonzero_ratio(side_img) < 0.10:   side_img = img_whole.copy()
        if self._nonzero_ratio(tip_img) < 0.10:    tip_img = img_whole.copy()

        if self.is_train and self.augment is not None:
            img_whole = self.augment(image=img_whole)['image']
            root_img = self.augment(image=root_img)['image']
            center_img = self.augment(image=center_img)['image']
            side_img = self.augment(image=side_img)['image']
            tip_img = self.augment(image=tip_img)['image']

        img_whole = self.normalize_transform(image=img_whole)['image']
        root_img = self.normalize_transform(image=root_img)['image']
        center_img = self.normalize_transform(image=center_img)['image']
        side_img = self.normalize_transform(image=side_img)['image']
        tip_img = self.normalize_transform(image=tip_img)['image']

        labels = torch.tensor(row[self.label_cols].values.astype(float), dtype=torch.float32)
        return (img_whole, root_img, center_img, side_img, tip_img), labels

