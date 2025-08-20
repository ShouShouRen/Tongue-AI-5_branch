# dataset.py — structure-safe augments + rotate_upright + mask partition
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def rotate_upright(image_np):
    """旋轉圖像，讓舌尖朝上（若無明顯輪廓則原樣返回）"""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_np

    cnt = max(contours, key=cv2.contourArea)
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

    dx = bottommost[0] - topmost[0]
    dy = bottommost[1] - topmost[1]
    angle = np.degrees(np.arctan2(dy, dx))

    h, w = image_np.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, -angle + 90, 1.0)
    rotated = cv2.warpAffine(image_np, rot_matrix, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated


class TongueDataset(Dataset):
    def __init__(self, csv_file, image_root, label_cols, is_train=True, rotate_upright_enabled=True):
        self.data = pd.read_csv(csv_file)
        self.image_root = image_root
        self.label_cols = label_cols
        self.is_train = is_train
        self.rotate_upright_enabled = rotate_upright_enabled

        print(f"✔ 載入資料: {csv_file} 共 {len(self.data)} 筆 — 自動旋轉: {'啟用' if rotate_upright_enabled else '停用'}")

        if is_train:
            self.augment = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=10,
                                   border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.CoarseDropout(max_holes=2, max_height=32, max_width=32, min_holes=1, fill_value=0, p=0.3),
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

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_root, row['image_path'])
        img_np = np.array(Image.open(img_path).convert('RGB'))

        if self.rotate_upright_enabled:
            img_np = rotate_upright(img_np)

        img_whole, body_img, edge_img = self.mask_based_partition(img_np)

        if self.is_train and self.augment is not None:
            img_whole = self.augment(image=img_whole)['image']
            body_img  = self.augment(image=body_img)['image']
            edge_img  = self.augment(image=edge_img)['image']

        img_whole = self.normalize_transform(image=img_whole)['image']
        body_img  = self.normalize_transform(image=body_img)['image']
        edge_img  = self.normalize_transform(image=edge_img)['image']

        labels = torch.tensor(row[self.label_cols].values.astype(float), dtype=torch.float32)
        return (img_whole, body_img, edge_img), labels

    def mask_based_partition(self, img_np):
        """
        以簡單閾值得到舌頭 mask，做裁切/縮放為 224，再以腐蝕得到 edge/body 分割。
        """
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        x, y, w, h = cv2.boundingRect(mask)
        cropped = img_np[y:y+h, x:x+w]
        resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LINEAR)

        mask_cropped = mask[y:y+h, x:x+w]
        mask_resized = cv2.resize(mask_cropped, (224, 224), interpolation=cv2.INTER_NEAREST)

        diag = int(np.sqrt(224**2 + 224**2))
        edge_width = int(diag * 0.191)
        kernel = np.ones((edge_width, edge_width), np.uint8)
        mask_eroded = cv2.erode(mask_resized, kernel, iterations=1)

        mask_edge = cv2.subtract(mask_resized, mask_eroded)
        mask_edge[:224 // 4, :] = 0  # 去掉上方 1/4 的 edge
        mask_body = cv2.subtract(mask_resized, mask_edge)

        body_img = cv2.bitwise_and(resized, resized, mask=mask_body)
        edge_img = cv2.bitwise_and(resized, resized, mask=mask_edge)

        return resized, body_img, edge_img
