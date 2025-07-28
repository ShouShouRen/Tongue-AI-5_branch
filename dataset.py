import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def rotate_upright(image_np):
    """旋轉圖像，讓舌尖朝上"""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_np  # 無輪廓，跳過

    cnt = max(contours, key=cv2.contourArea)
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

    # 假設舌尖是 topmost，舌根是 bottommost
    dx = bottommost[0] - topmost[0]
    dy = bottommost[1] - topmost[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # 計算旋轉矩陣 & 旋轉圖片
    h, w = image_np.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, -angle + 90, 1.0)  # 舌尖朝上（90度）
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

        print(f"✔ 載入資料: {csv_file} 共 {len(self.data)} 筆")
        print(f"✔ 自動旋轉：{'啟用' if self.rotate_upright_enabled else '停用'}")

        if is_train:
            self.transform = A.Compose([
                A.Resize(384, 384),
                A.HorizontalFlip(p=0.3),
                A.RandomBrightnessContrast(0.1, 0.1, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(384, 384),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 讀圖
        img_path = os.path.join(self.image_root, row['image_path'])
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)

        # ⬆️ 旋轉成舌尖朝上
        if self.rotate_upright_enabled:
            img_np = rotate_upright(img_np)

        # 分區
        img_whole = img_np.copy()
        img_body = self.crop_center_region(img_np, region="body")
        img_edge = self.crop_center_region(img_np, region="edge")

        # transform
        img_whole = self.transform(image=img_whole)['image']
        img_body = self.transform(image=img_body)['image']
        img_edge = self.transform(image=img_edge)['image']

        labels = torch.tensor(row[self.label_cols].values.astype(float), dtype=torch.float32)
        return (img_whole, img_body, img_edge), labels

    def crop_center_region(self, img_np, region="body"):
        h, w = img_np.shape[:2]
        if region == "body":
            x1, y1 = int(w*0.15), int(h*0.15)
            x2, y2 = int(w*0.85), int(h*0.85)
            return img_np[y1:y2, x1:x2]
        elif region == "edge":
            mask = np.ones((h, w), dtype=np.uint8) * 255
            inner = np.zeros((h, w), dtype=np.uint8)
            x1, y1 = int(w*0.2), int(h*0.2)
            x2, y2 = int(w*0.8), int(h*0.8)
            inner[y1:y2, x1:x2] = 255
            edge_mask = cv2.subtract(mask, inner)
            return cv2.bitwise_and(img_np, img_np, mask=edge_mask)
        return img_np
