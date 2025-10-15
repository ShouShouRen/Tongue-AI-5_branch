# scan_models.py
import os, glob, math, torch
from model import SignOrientedNetwork

CKPT_KEYS_HINT = ["best_macro_f1", "metrics_best", "metrics_05", "label_cols"]

CANDIDATES = [
    ("convnext_base", 768),
    ("swin_base_patch4_window7_224", 512),
    ("resnet50", 512),
    ("convnext_base", 512),
    ("swin_base_patch4_window7_224", 768),
]

def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, train

def match_ratio(model, sd):
    msd = model.state_dict()
    ok = 0
    for k, v in msd.items():
        if k in sd and tuple(sd[k].shape) == tuple(v.shape):
            ok += 1
    return ok / max(1, len(msd))

def try_recover_model(state_dict, num_classes):
    best = None
    best_r = -1.0
    for bb, fd in CANDIDATES:
        m = SignOrientedNetwork(num_classes=num_classes, backbone=bb, feature_dim=fd, dropout=0.1)
        r = match_ratio(m, state_dict)
        if r > best_r:
            best, best_r = (bb, fd, m), r
    return (*best, best_r)

def _pick(d, keys):
    for k in keys:
        if k in d: return d[k]
    return float('nan')

def _fmt_metrics(tag, md):
    f1_micro  = md.get("f1_micro", float('nan'))
    f1_macro  = md.get("f1_macro", float('nan'))
    # 常見 ACC 欄位名
    acc_micro = _pick(md, ["acc_micro", "accuracy_micro"])
    acc_macro = _pick(md, ["acc_macro", "accuracy_macro"])
    subset    = _pick(md, ["subset_acc", "subset_accuracy", "exact_match"])
    # 後備：單一 accuracy/acc
    acc_any   = _pick(md, ["accuracy", "acc"])

    parts = [
        f"{tag}: F1_micro={f1_micro:.4f}, F1_macro={f1_macro:.4f}",
        f"ACC_micro={acc_micro:.4f}" if not math.isnan(acc_micro) else "ACC_micro=N/A",
        f"ACC_macro={acc_macro:.4f}" if not math.isnan(acc_macro) else "ACC_macro=N/A",
        f"SubsetAcc={subset:.4f}"     if not math.isnan(subset)     else "SubsetAcc=N/A",
    ]
    # 若僅有單一 accuracy，額外列出
    if math.isnan(acc_micro) and math.isnan(acc_macro) and not math.isnan(acc_any):
        parts.append(f"ACC={acc_any:.4f}")
    return " | ".join(parts)

def describe_metrics(ckpt):
    if "metrics_best" in ckpt:
        base = f"best_macro_f1={ckpt.get('best_macro_f1', float('nan')):.4f}"
        best_line = _fmt_metrics("Best", ckpt["metrics_best"])
        extra = []
        if "metrics_05" in ckpt:
            extra.append(_fmt_metrics("Thr@0.5", ckpt["metrics_05"]))
        return " | ".join([base, best_line] + extra)
    if "metrics_05" in ckpt:
        base = f"best_macro_f1={ckpt.get('best_macro_f1', float('nan')):.4f}"
        return " | ".join([base, _fmt_metrics("Thr@0.5", ckpt["metrics_05"])])
    if "best_macro_f1" in ckpt:
        return f"best_macro_f1={ckpt['best_macro_f1']:.4f}"
    return "metrics: N/A"

def human(n):
    if n < 1e3: return f"{n}"
    if n < 1e6: return f"{n/1e3:.1f}K"
    if n < 1e9: return f"{n/1e6:.2f}M"
    return f"{n/1e9:.2f}B"

def main():
    pattern = "5_branch_baseline.pth"
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        print(f"No files matched: {pattern}")
        return

    for fp in files:
        sz = os.path.getsize(fp) / (1024**2)
        ckpt = torch.load(fp, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
            label_cols = ckpt.get("label_cols", None)
            num_classes = len(label_cols) if label_cols is not None else 8
        elif isinstance(ckpt, dict):
            sd = ckpt
            num_classes = 8
        else:
            print(f"[{fp}] unsupported checkpoint format")
            continue

        bb, fd, model, ratio = try_recover_model(sd, num_classes)
        strict = ratio > 0.95
        missing, unexpected = model.load_state_dict(sd, strict=strict)
        total, trainable = count_params(model)

        print("\n---", fp)
        print(f"size={sz:.2f} MB | classes={num_classes}")
        print(f"guess: backbone={bb}, feature_dim={fd}, match_ratio={ratio:.3f}, strict={strict}")
        if not strict:
            print(f"load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        print(f"params_total={human(total)} | params_trainable={human(trainable)}")
        if isinstance(ckpt, dict):
            print(describe_metrics(ckpt))

if __name__ == "__main__":
    main()
