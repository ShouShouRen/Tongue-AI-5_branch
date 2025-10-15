# preview_paper_upright.py
import os, cv2, numpy as np, pandas as pd, math
from PIL import Image

# ===== 路徑與參數 =====
CSV = "train_fold1.csv"      # 需含 image_path；如有遮罩請加 mask_path
IMG_ROOT = "images"
OUT = "preview_parts_paper"
N = 30
IMG_SIZE = 224
ALPHA_DEG = 60
MARGIN_RATIO = 0.08
MIN_AREA = 200
MIN_CONF = 0.10             # 旋轉信心下限；低於就不轉
CURV_DIFF_T = 12.0          # 曲率差門檻(度)；不足則弱化曲率
PAD_RATIO = 0.5             # 旋轉用大畫布邊界

os.makedirs(OUT, exist_ok=True)

def to_bgr(x): return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

def load_image_keep_alpha(path):
    im = Image.open(path)
    return np.array(im) if im.mode == "RGBA" else np.array(im.convert("RGB"))

# ===== 遮罩（外部優先；否則穩健 fallback） =====
def get_mask(img_rgba_or_rgb, row=None):
    if img_rgba_or_rgb.ndim == 3 and img_rgba_or_rgb.shape[-1] == 4:
        alpha = img_rgba_or_rgb[..., 3]
        m = (alpha > 0).astype(np.uint8) * 255
        if m.max() > 0: return m

    if row is not None and 'mask_path' in row and isinstance(row['mask_path'], str) and len(str(row['mask_path'])) > 0:
        mp = os.path.join(IMG_ROOT, row['mask_path'])
        if os.path.exists(mp):
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is not None and m.max() > 0: return m

    rgb = img_rgba_or_rgb[..., :3] if (img_rgba_or_rgb.ndim == 3 and img_rgba_or_rgb.shape[-1] == 4) else img_rgba_or_rgb
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV); h, s, v = cv2.split(hsv)
    m1 = cv2.inRange(s, 25, 255) & cv2.inRange(v, 20, 255)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB); L, A, B = cv2.split(lab)
    m2 = cv2.inRange(A, 135, 255) | cv2.inRange(B, 135, 255)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    m3 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
    m = (m1 | m2 | m3)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return m
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    m_out = np.zeros_like(m); cv2.drawContours(m_out, [hull], -1, 255, -1)
    return m_out

def boost_root(mask, top_frac=0.14, grow=6):
    H, W = mask.shape; top_h = int(H * top_frac)
    if top_h <= 0: return mask
    if mask[:top_h].mean() < 70:
        seed = mask.copy(); seed[:int(H*0.35)] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))
        rec = cv2.dilate(seed, kernel, iterations=2)
        mask = cv2.bitwise_or(mask, rec)
    return mask

def _mask(img_rgb_or_rgba, row=None): return get_mask(img_rgb_or_rgba, row)

# ===== 幾何工具 =====
def pad_for_transform(img, pad_ratio=PAD_RATIO):
    H, W = img.shape[:2]; px, py = int(W*pad_ratio), int(H*pad_ratio)
    if img.ndim == 3 and img.shape[-1] == 4:
        return cv2.copyMakeBorder(img, py, py, px, px, cv2.BORDER_CONSTANT, value=(0,0,0,0)), (px, py)
    return cv2.copyMakeBorder(img, py, py, px, px, cv2.BORDER_CONSTANT, value=(0,0,0)), (px, py)

def ensure_inside(img, tx, ty):
    H, W = img.shape[:2]
    left = max(0, -int(tx)); right = max(0, int(tx))
    top  = max(0, -int(ty)); bottom = max(0, int(ty))
    if left or right or top or bottom:
        if img.ndim == 3 and img.shape[-1] == 4:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0,0))
        else:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
        tx += left; ty += top
    return img, tx, ty

def rotate_image(img, deg):
    H, W = img.shape[:2]
    bval = (0,0,0) if not (img.ndim==3 and img.shape[-1]==4) else (0,0,0,0)
    M = cv2.getRotationMatrix2D((W/2, H/2), deg, 1.0)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=bval)

# ===== 輔助：曲率與 PCA 軸端點 =====
def curvature_score_at(contour, idx, k=12):
    n = len(contour)
    a = contour[(idx - k) % n][0].astype(np.float32)
    b = contour[idx % n][0].astype(np.float32)
    c = contour[(idx + k) % n][0].astype(np.float32)
    v1 = a - b; v2 = c - b
    n1 = np.linalg.norm(v1)+1e-6; n2 = np.linalg.norm(v2)+1e-6
    cos = float(np.clip((v1 @ v2) / (n1*n2), -1.0, 1.0))
    ang = math.degrees(math.acos(cos))      # 0~180，小角=尖
    return 180.0 - ang                      # 尖=大分數

def pca_axis_endpoints(contour):
    pts = contour.reshape(-1,2).astype(np.float32)
    mean = pts.mean(0, keepdims=True)
    X = pts - mean
    cov = (X.T @ X) / max(1, len(X)-1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, np.argmax(eigvals)]
    v = v / (np.linalg.norm(v)+1e-6)
    proj = ((pts - mean) @ v.reshape(2,1)).ravel()
    i1 = int(np.argmin(proj)); i2 = int(np.argmax(proj))
    return i1, i2, v

def mean_curvature_in_band(contour, y0, y1):
    idxs = [i for i,p in enumerate(contour[:,0,:]) if y0 <= p[1] < y1]
    if len(idxs) < 10: return 0.0
    k = max(6, len(contour)//150)
    vals = [curvature_score_at(contour, i, k=k) for i in idxs]
    return float(np.mean(vals)) if vals else 0.0

# ===== 厚度 + 曲率 投票找 root/tip =====
def find_root_tip_voted(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None, None, 0.0
    contour = max(cnts, key=cv2.contourArea)
    i1, i2, _ = pca_axis_endpoints(contour)
    P1, P2 = contour[i1][0].astype(np.float32), contour[i2][0].astype(np.float32)

    dt = cv2.distanceTransform((mask>0).astype(np.uint8), cv2.DIST_L2, 3)
    H, W = mask.shape
    def patch_mean(p, r=18):
        x,y = int(p[0]), int(p[1])
        x1, y1 = max(0, x-r), max(0, y-r)
        x2, y2 = min(W, x+r+1), min(H, y+r+1)
        if x1>=x2 or y1>=y2: return 0.0
        return float(dt[y1:y2, x1:x2].mean())

    t1, t2 = patch_mean(P1), patch_mean(P2)                         # 厚度分數（厚=大）
    c1, c2 = curvature_score_at(contour, i1, 12), curvature_score_at(contour, i2, 12)  # 曲率（尖=大）

    tip_by_curv = 1 if c2 > c1 else 0
    root_by_thk = 0 if t1 > t2 else 1

    curv_conf = max(0.0, (abs(c2 - c1) - CURV_DIFF_T) / (40.0 - CURV_DIFF_T))
    curv_conf = float(np.clip(curv_conf, 0.0, 1.0))
    thk_conf = float(abs(t1 - t2) / (max(t1, t2) + 1e-6))
    thk_conf = float(np.clip(thk_conf, 0.0, 1.0))

    agree = (tip_by_curv == (1 - root_by_thk))
    conf = 0.6*curv_conf + 0.6*thk_conf if agree else 0.35*max(curv_conf, thk_conf)

    tip_idx = (1 if curv_conf >= thk_conf else 1 - root_by_thk)
    if tip_idx == 0:
        tip, root = P1, P2
    else:
        tip, root = P2, P1
    return root, tip, float(conf)

# ===== 直立評分：上寬下尖 + 重心在下 + 上帶填充 =====
def upright_score(mask):
    if mask.max()==0: return 0.0
    H, W = mask.shape
    ys, xs = np.where(mask>0)
    if len(xs) < 50: return 0.0

    rows = np.unique(ys); spans = {}
    for r in rows:
        cols = xs[ys==r]; spans[r] = cols.max()-cols.min()+1
    top_band = [spans[r] for r in rows if r < 0.25*H]
    bottom_band = [spans[r] for r in rows if r >= 0.75*H]
    if len(top_band)==0 or len(bottom_band)==0: return 0.0
    w_top = np.mean(top_band); w_bot = np.mean(bottom_band)
    s_width = np.clip((w_top/(w_bot+1e-6) - 1.0), 0.0, 1.0)

    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    c_top = mean_curvature_in_band(cnt, 0, int(0.25*H))
    c_bot = mean_curvature_in_band(cnt, int(0.75*H), H)
    s_curv = np.clip((c_bot - c_top)/30.0, 0.0, 1.0)

    cy = ys.mean()
    s_cy = np.clip((cy/H - 0.5)*2.0, 0.0, 1.0)

    s_topfill = np.clip((np.mean(mask[:int(0.15*H),:]>0) - 0.08)/0.22, 0.0, 1.0)

    return float(0.45*s_width + 0.30*s_curv + 0.15*s_cy + 0.10*s_topfill)

# ===== 根據投票方向旋轉；再驗極性 =====
def orient_with_root_up(img, mask, min_conf=MIN_CONF):
    root, tip, conf = find_root_tip_voted(mask)
    if root is None or tip is None or conf < min_conf:
        return img, 0.0, False
    dx = tip[0]-root[0]; dy = tip[1]-root[1]
    ang = math.degrees(math.atan2(dy, dx))
    rot_deg = 90.0 - ang
    rot = rotate_image(img, rot_deg)

    m2 = _mask(rot)
    r2, t2, _ = find_root_tip_voted(m2)
    if r2 is not None and t2 is not None and t2[1] < r2[1]:
        rot = rotate_image(rot, 180.0); rot_deg += 180.0
    return rot, rot_deg, True

# ===== 直立：候選角度打分選最優；低分就不轉 =====
def upright_by_paper(img, alpha_deg=ALPHA_DEG, margin_ratio=MARGIN_RATIO, target_xy=None, row=None):
    big, _ = pad_for_transform(img, pad_ratio=PAD_RATIO)
    m0 = _mask(big, row=row)
    if m0.max()==0: return img, {"rot_deg":0.0, "used":False}

    rot0, deg0, used0 = orient_with_root_up(big, m0, min_conf=MIN_CONF)
    base_list = [0.0, 180.0] + ([deg0 % 360.0, (deg0+180.0)%360.0] if used0 else [])
    cand_deg = []
    for b in base_list:
        for d in [b-60,b-30,b,b+30,b+60]:
            dd = ((d+360)%360.0)
            if all(abs(dd-x)>5 for x in cand_deg): cand_deg.append(dd)

    best = {"score": -1, "deg": 0.0, "img": big}
    for d in cand_deg:
        rot = rotate_image(big, d)
        sc = upright_score(_mask(rot, row=row))
        if sc > best["score"]:
            best = {"score": sc, "deg": d, "img": rot}

    if best["score"] < 0.28:
        rot, total_deg, used = (big, 0.0, False)
    else:
        rot, total_deg, used = (best["img"], best["deg"], True)

    H, W = rot.shape[:2]
    tx, ty = (W/2, 0.82*H) if target_xy is None else target_xy
    m1 = _mask(rot, row=row)
    r1, t1, _ = find_root_tip_voted(m1)
    if t1 is not None:
        dx, dy = tx - t1[0], ty - t1[1]
        rot, dx, dy = ensure_inside(rot, dx, dy)
        T = np.float32([[1,0,dx],[0,1,dy]])
        rot = cv2.warpAffine(rot, T, (rot.shape[1], rot.shape[0]), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0,0,0) if not (rot.ndim==3 and rot.shape[-1]==4) else (0,0,0,0))

    m2 = _mask(rot, row=row)
    if m2.max()>0:
        x0,y0,w0,h0 = cv2.boundingRect(m2)
        mx,my = int(margin_ratio*w0), int(margin_ratio*h0)
        x1,y1 = max(0,x0-mx), max(0,y0-my)
        x2,y2 = min(rot.shape[1], x0+w0+mx), min(rot.shape[0], y0+h0+my)
        rot = rot[y1:y2, x1:x2]

    return rot, {"rot_deg": total_deg, "used": int(used)}

# ===== 五分區切割 =====
def partition(img_np, row=None):
    mask = _mask(img_np, row=row)
    mask = boost_root(mask, top_frac=0.14, grow=6)

    x, y, w, h = cv2.boundingRect(mask)
    cropped = img_np[y:y+h, x:x+w]
    mask_c = mask[y:y+h, x:x+w]

    resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask_c, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    H, W = mask_r.shape

    root_ratio, center_ratio = 0.26, 0.48
    top_h = int(root_ratio * H); mid_h = int(center_ratio * H); bot_start = top_h + mid_h

    zone = np.zeros_like(mask_r, dtype=np.uint8)
    zone[0:top_h, :] = 1
    zone[top_h:top_h+mid_h, int(0.22*W):int(0.78*W)] = 2
    zone[top_h:H, 0:int(0.22*W)] = 3
    zone[top_h:H, int(0.78*W):W] = 3
    zone[bot_start:H, int(0.22*W):int(0.78*W)] = 4
    zone = cv2.bitwise_and(zone, zone, mask=mask_r)

    def cut(mval):
        m = (zone == mval).astype(np.uint8)
        return cv2.bitwise_and(resized, resized, mask=m), m

    root_img,   m1 = cut(1)
    center_img, m2 = cut(2)
    side_img,   m3 = cut(3)
    tip_img,    m4 = cut(4)
    return resized, (root_img, center_img, side_img, tip_img), (m1, m2, m3, m4), zone

def tile_row(images_rgb, titles=None, fontscale=0.5):
    H, W = images_rgb[0].shape[:2]
    canvas = np.zeros((H, W*len(images_rgb), 3), dtype=np.uint8)
    for i, im in enumerate(images_rgb):
        im3 = im if im.shape[-1]==3 else im[..., :3]
        canvas[:, i*W:(i+1)*W] = im3
        if titles:
            cv2.putText(canvas, titles[i], (i*W+5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0,255,0), 1, cv2.LINE_AA)
    return canvas

def ratio(mask01): return float(mask01.mean())

# ===== 主流程 =====
def main():
    df = pd.read_csv(CSV)
    for idx in range(min(N, len(df))):
        row = df.iloc[idx]
        p = os.path.join(IMG_ROOT, row['image_path'])
        img = load_image_keep_alpha(p)

        img_upright, info = upright_by_paper(img, alpha_deg=ALPHA_DEG, margin_ratio=MARGIN_RATIO, row=row)
        if img_upright.ndim == 3 and img_upright.shape[-1] == 4:
            img_upright = img_upright[..., :3]

        whole, parts, masks, zone = partition(img_upright, row=row)
        root_img, center_img, side_img, tip_img = parts
        m1, m2, m3, m4 = masks

        r = [ratio(m) for m in (m1, m2, m3, m4)]
        print(f"[{idx:02d}] rot={info['rot_deg']:.1f}° used={int(info['used'])}  "
              f"root={r[0]:.3f} center={r[1]:.3f} side={r[2]:.3f} tip={r[3]:.3f} | {row['image_path']}")

        edges = cv2.Canny((zone>0).astype(np.uint8)*255, 50, 150)
        overlay = whole.copy(); overlay[edges>0] = [255, 80, 200]

        strip = tile_row([overlay, root_img, center_img, side_img, tip_img],
                         titles=["whole+zones","root","center","side","tip"])
        out_path = os.path.join(OUT, f"sample_{idx:02d}.jpg")
        cv2.imwrite(out_path, to_bgr(strip))

    print(f"Saved previews to: {OUT}")

if __name__ == "__main__":
    main()
