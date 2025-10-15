# dataset.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TongueDataset(Dataset):
    """
    - 不旋轉（rotate_upright_enabled 預設 False）
    - 以原圖 OTSU 遮罩取外接框，resize 到 224
    - 五分區：root(上)、center(中段)、side(左右下段)、tip(下)；root/tip 有放寬與保底
    - 每區：區內保留彩色，區外=黑色；若有效像素太少則回退整圖
    """
    def __init__(self, csv_file, image_root, label_cols,
                 is_train=True, rotate_upright_enabled=False):
        self.data = pd.read_csv(csv_file)
        self.image_root = image_root
        self.label_cols = label_cols
        self.is_train = is_train
        self.rotate_upright_enabled = rotate_upright_enabled  # 這版不使用（固定不旋轉）

        print(f"✔ 載入資料: {csv_file} 共 {len(self.data)} 筆 — 自動直立: {'啟用' if rotate_upright_enabled else '停用'}")

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

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _nonzero_ratio(rgb):
        # rgb: (H, W, 3) uint8
        return (rgb.sum(axis=2) > 0).mean()

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_root, row['image_path'])
        img = np.array(Image.open(img_path).convert('RGB'))  # RGB uint8

        # 以灰階 OTSU 取得舌頭遮罩
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 外接框裁切 + resize 到 224
        x, y, w, h = cv2.boundingRect(mask)
        cropped = img[y:y + h, x:x + w]
        mask_c = mask[y:y + h, x:x + w]

        resized = cv2.resize(cropped, (384, 384), interpolation=cv2.INTER_LINEAR)
        mask_r = cv2.resize(mask_c, (384, 384), interpolation=cv2.INTER_NEAREST)

        # ---------- 分區（根在上、尖在下；左右為 side） ----------
        H, W = mask_r.shape
        # 基礎比例
        root_ratio = 0.28
        center_ratio = 0.50
        top_h = int(root_ratio * H)
        mid_h = int(center_ratio * H)
        bot_start = top_h + mid_h

        # 放寬/保底策略（可依資料再微調）
        root_extra = 0.06       # root 往下多吃 6% 高度
        root_min_frac = 0.22    # root 最少佔整高 22%

        tip_extra = 0.06        # tip 往上多吃 6% 高度
        tip_min_frac = 0.18     # tip 最少佔整高 18%
        tip_widen = 0.18        # tip 水平左右各再放寬 18%

        zone_mask = np.zeros_like(mask_r, dtype=np.uint8)

        # root（上方整寬；高度放寬 + 保底）
        root_end = min(H, top_h + int(root_extra * H))
        root_end = max(root_end, int(root_min_frac * H))
        zone_mask[0:root_end, :] = 1

        # center（以 root_end 為上界，維持中段高度，水平 25%~75%）
        center_top = root_end
        center_bot = min(H, center_top + mid_h)
        cx_l = int(0.25 * W)
        cx_r = int(0.75 * W)
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

        # 只保留舌區域（與 mask_r 交集）
        zone_mask = cv2.bitwise_and(zone_mask, zone_mask, mask=mask_r)

        # 取出各區的彩色圖（區外=黑色）
        def cut(val: int):
            m = (zone_mask == val).astype(np.uint8)
            return cv2.bitwise_and(resized, resized, mask=m)

        img_whole = resized
        root_img = cut(1)
        center_img = cut(2)
        side_img = cut(3)
        tip_img = cut(4)

        # 若區域有效像素太少，回退整圖（避免全黑訓練）
        if self._nonzero_ratio(root_img) < 0.10:   root_img = img_whole.copy()
        if self._nonzero_ratio(center_img) < 0.10: center_img = img_whole.copy()
        if self._nonzero_ratio(side_img) < 0.10:   side_img = img_whole.copy()
        if self._nonzero_ratio(tip_img) < 0.10:    tip_img = img_whole.copy()

        # 資料增強（對每個分支各自做，以維持分區語義）
        if self.is_train and self.augment is not None:
            img_whole = self.augment(image=img_whole)['image']
            root_img = self.augment(image=root_img)['image']
            center_img = self.augment(image=center_img)['image']
            side_img = self.augment(image=side_img)['image']
            tip_img = self.augment(image=tip_img)['image']

        # Normalize + ToTensor（float32）
        img_whole = self.normalize_transform(image=img_whole)['image']
        root_img = self.normalize_transform(image=root_img)['image']
        center_img = self.normalize_transform(image=center_img)['image']
        side_img = self.normalize_transform(image=side_img)['image']
        tip_img = self.normalize_transform(image=tip_img)['image']

        labels = torch.tensor(row[self.label_cols].values.astype(float), dtype=torch.float32)
        return (img_whole, root_img, center_img, side_img, tip_img), labels
