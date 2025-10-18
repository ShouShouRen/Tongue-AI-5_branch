# visualize_dataset.py - 用於視覺化檢查 Dataset 分區結果的工具 (詳細步驟分析版)
# --- 已升級為混合式遮罩演算法 (HSV + OTSU + 最大連通元件) ---

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pandas as pd
from PIL import Image

# --- 設定 ---
# vvv 請根據您的環境修改這些路徑 vvv
CSV_PATH = 'val_fold1.csv'       # 用於檢查的 CSV 檔案
IMAGE_ROOT = 'images'            # 圖片所在的根目錄
NUM_SAMPLES_TO_SHOW = 5          # 要顯示多少個樣本
INPUT_SIZE = 384                 # 資料集處理的圖片尺寸
# ^^^ 請根據您的環境修改這些路徑 ^^^

LABEL_COLS = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis',
              'Crack', 'Toothmark', 'FurThick', 'FurYellow']

def _resolve_image_field(df: pd.DataFrame) -> str:
    """從 DataFrame 中找到圖片路徑欄位的實際名稱"""
    low = {c.lower(): c for c in df.columns}
    for k in ["image_path", "image", "img", "filename", "file", "path", "filepath", "img_path", "imgpath"]:
        if k in low:
            return low[k]
    raise KeyError(f"在 {CSV_PATH} 中找不到影像欄位。")

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
    
    return final_mask, hsv_mask, otsu_mask, mask_closed


def visualize_sample_details(img_path, labels_str):
    """
    載入單張圖片，重現 dataset.py 中的分區邏輯，並詳細視覺化每一步。
    """
    img = np.array(Image.open(img_path).convert('RGB'))
    
    # 使用新的混合式方法建立遮罩，並取得中間過程的遮罩
    final_mask, hsv_mask, otsu_mask, combined_closed_mask = create_hybrid_mask(img)

    x, y, w, h = cv2.boundingRect(final_mask)
    if w == 0 or h == 0: x, y, w, h = 0, 0, img.shape[1], img.shape[0]

    # --- 接下來的邏輯與之前類似，但使用 final_mask ---
    cropped_roi = img[y:y + h, x:x + w]
    mask_c = final_mask[y:y + h, x:x + w]
    resized = cv2.resize(cropped_roi, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask_c, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)

    H, W = mask_r.shape
    root_ratio, center_ratio = 0.28, 0.50
    top_h, mid_h = int(root_ratio * H), int(center_ratio * H)
    bot_start = top_h + mid_h
    root_extra, root_min_frac = 0.06, 0.22
    tip_extra, tip_min_frac, tip_widen = 0.06, 0.18, 0.18

    geom_zone_mask = np.zeros_like(mask_r, dtype=np.uint8)
    
    # 1: root
    root_end = max(int(root_min_frac * H), min(H, top_h + int(root_extra * H)))
    geom_zone_mask[0:root_end, :] = 1
    # 2: center
    center_top = root_end; center_bot = min(H, center_top + mid_h)
    cx_l, cx_r = int(0.25 * W), int(0.75 * W)
    geom_zone_mask[center_top:center_bot, cx_l:cx_r] = 2
    # 3: side
    geom_zone_mask[center_top:H, 0:int(0.25 * W)] = 3
    geom_zone_mask[center_top:H, int(0.75 * W):W] = 3
    # 4: tip
    tip_start = max(0, bot_start - int(tip_extra * H))
    tip_min_h = int(tip_min_frac * H)
    if (H - tip_start) < tip_min_h: tip_start = max(0, H - tip_min_h)
    tx_l = max(0, int((0.25 - tip_widen) * W))
    tx_r = min(W, int((0.75 + tip_widen) * W))
    geom_zone_mask[tip_start:H, tx_l:tx_r] = 4

    final_zone_mask = cv2.bitwise_and(geom_zone_mask, geom_zone_mask, mask=mask_r)
    root_mask = (final_zone_mask == 1).astype(np.uint8)
    final_root_img = cv2.bitwise_and(resized, resized, mask=root_mask)

    # 繪圖
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"分區過程分析 (使用混合式演算法)\nLabels: {labels_str}", fontsize=16)

    axes[0, 0].imshow(resized); axes[0, 0].set_title('1. 縮放後舌像 (Resized)')
    axes[0, 1].imshow(hsv_mask, cmap='gray'); axes[0, 1].set_title('2a. HSV 遮罩')
    axes[0, 2].imshow(otsu_mask, cmap='gray'); axes[0, 2].set_title('2b. OTSU 遮罩')
    axes[1, 0].imshow(combined_closed_mask, cmap='gray'); axes[1, 0].set_title('3. 合併與填補後')
    axes[1, 1].imshow(mask_r, cmap='gray'); axes[1, 1].set_title('4. 最終遮罩 (最大連通元件)')
    axes[1, 2].imshow(final_root_img); axes[1, 2].set_title('5. 最終 Root 區')

    for ax in axes.flat: ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def main():
    if not os.path.exists(CSV_PATH):
        print(f"錯誤：找不到 CSV 檔案 '{CSV_PATH}'。請檢查路徑設定。")
        return
    if not os.path.exists(IMAGE_ROOT):
        print(f"錯誤：找不到圖片根目錄 '{IMAGE_ROOT}'。請檢查路徑設定。")
        return

    try:
        df = pd.read_csv(CSV_PATH)
        img_col_name = _resolve_image_field(df)
        samples = df.sample(NUM_SAMPLES_TO_SHOW)
    except Exception as e:
        print(f"讀取或處理 CSV 時發生錯誤: {e}")
        return

    print(f"將隨機顯示 {NUM_SAMPLES_TO_SHOW} 個樣本的分區過程...")
    for _, row in samples.iterrows():
        try:
            img_path = os.path.join(IMAGE_ROOT, row[img_col_name])
            labels = [LABEL_COLS[i] for i, l in enumerate(row[LABEL_COLS]) if l > 0.5]
            labels_str = ", ".join(labels) if labels else "無"
            visualize_sample_details(img_path, labels_str)
        except Exception as e:
            print(f"處理圖片 {row[img_col_name]} 時發生錯誤: {e}")
    
    print("視覺化檢查完成。")

if __name__ == '__main__':
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Heiti TC', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"無法設定中文字體，標題可能顯示為方塊: {e}")
    
    main()

