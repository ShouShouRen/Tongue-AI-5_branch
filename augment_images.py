import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from albumentations import (
    HorizontalFlip, RandomBrightnessContrast, Rotate, GaussianBlur, HueSaturationValue,
    CLAHE, ShiftScaleRotate, Compose
)

# === 設定 ===
INPUT_CSV = "train_fold1.csv"
IMAGE_ROOT = "images"  # 含多個子資料夾
OUTPUT_DIR = "augmented_images"
OUTPUT_CSV = "aug_train_fold1.csv"
AUG_PER_IMAGE = 2
TARGET_LABELS = ['TonguePale', 'TipSideRed', 'Ecchymosis']

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(INPUT_CSV)

def get_strong_augmentations():
    return Compose([
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.5),
        Rotate(limit=25, p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        HueSaturationValue(p=0.4),
        CLAHE(p=0.3),
        GaussianBlur(p=0.3),
    ])

augment = get_strong_augmentations()
positive_df = df[df[TARGET_LABELS].any(axis=1)].copy()
augmented_rows = []

for _, row in tqdm(positive_df.iterrows(), total=len(positive_df), desc="生成增強圖像中"):
    rel_path = row["image_path"]
    abs_path = os.path.join(IMAGE_ROOT, rel_path)

    if not os.path.exists(abs_path):
        continue

    image = cv2.imread(abs_path)
    if image is None:
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    subfolder = os.path.dirname(rel_path)
    base_name = os.path.splitext(os.path.basename(rel_path))[0]

    # === 複製原圖到 augmented_images ===
    output_subdir = os.path.join(OUTPUT_DIR, subfolder)
    os.makedirs(output_subdir, exist_ok=True)

    orig_output_path = os.path.join(output_subdir, os.path.basename(rel_path))
    if not os.path.exists(orig_output_path):
        cv2.imwrite(orig_output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    for i in range(AUG_PER_IMAGE):
        aug_result = augment(image=image)
        aug_image = aug_result["image"]
        aug_image = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)

        aug_name = f"{base_name}_aug{i+1}.jpg"
        aug_rel_path = os.path.join(subfolder, aug_name)
        aug_abs_path = os.path.join(OUTPUT_DIR, aug_rel_path)
        cv2.imwrite(aug_abs_path, aug_image)

        new_row = row.copy()
        new_row["image_path"] = aug_rel_path
        augmented_rows.append(new_row)


# 合併並輸出新CSV
aug_df = pd.DataFrame(augmented_rows)
merged_df = pd.concat([df, aug_df], ignore_index=True)
merged_df.to_csv(OUTPUT_CSV, index=False)
print(f"✔ 增強完成，共產生 {len(augmented_rows)} 張新圖，已儲存到 {OUTPUT_CSV}")
