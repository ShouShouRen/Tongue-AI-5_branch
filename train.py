# train.py — 已修改：支援指令行參數切換模型與骨幹
import os, gc, json, warnings, argparse
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, jaccard_score, accuracy_score

from dataset import TongueDataset
# vvv 從 model_zoo 匯入模型 vvv
from model import SignOrientedNetwork, SimpleTimmModel
from samplers import build_multilabel_weighted_sampler
from torch.cuda.amp import autocast, GradScaler

# --- 輔助函數 (無變動) ---
warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f} GB allocated / {reserved:.2f} GB reserved")

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

@torch.no_grad()
def evaluate_model_and_find_thresholds(model, data_loader, device, label_cols, threshold_grid=None):
    if threshold_grid is None:
        threshold_grid = np.concatenate([np.linspace(0.1, 0.9, 17), np.array([0.95])])
    model.eval()
    all_probs, all_labels = [], []
    for (x_whole, x_root, x_center, x_side, x_tip), labels in data_loader:
        x_whole, x_root, x_center, x_side, x_tip = x_whole.to(device), x_root.to(device), x_center.to(device), x_side.to(device), x_tip.to(device)
        with autocast():
            logits = model(x_whole, x_root, x_center, x_side, x_tip)
            probs = torch.sigmoid(logits).float().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())
    if not all_probs:
        C = len(label_cols)
        zeros = [0.0] * C
        return (dict(acc=0.0, f1_macro=0.0, jaccard_avg=0.0, f1_individual=zeros), np.array([0.5]*C))

    P, Y = np.vstack(all_probs), np.vstack(all_labels)
    C = Y.shape[1]
    best_thresholds, best_f1_per_class, best_jac_per_class = np.zeros(C), np.zeros(C), np.zeros(C)
    for i in range(C):
        if Y[:, i].sum() == 0:
            best_thresholds[i], best_f1_per_class[i], best_jac_per_class[i] = 0.5, 0.0, 0.0
            continue
        best_f1, best_j, best_th = -1.0, 0.0, 0.5
        for th in threshold_grid:
            pred = (P[:, i] > th).astype(int)
            f1, j = f1_score(Y[:, i], pred, zero_division=0), jaccard_score(Y[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_j, best_th = f1, j, th
        best_thresholds[i], best_f1_per_class[i], best_jac_per_class[i] = best_th, best_f1, best_j
    
    preds_best = np.array([P[:, i] > best_thresholds[i] for i in range(C)]).T
    metrics = {
        'acc': accuracy_score(Y, preds_best),
        'f1_macro': np.mean(best_f1_per_class).item(),
        'jaccard_avg': np.mean(best_jac_per_class).item(),
        'f1_individual': best_f1_per_class.tolist(),
    }
    return metrics, best_thresholds

def train_model(args, train_csv, val_csv, model_path):
    clear_gpu_memory()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training with args: {vars(args)} ---")

    # Dataset & Loader
    train_set = TongueDataset(train_csv, args.image_root, args.label_cols, is_train=True)
    sampler, n_pos_t, n_neg_t = build_multilabel_weighted_sampler(train_csv, args.label_cols)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    
    val_loader = None
    if val_csv:
        val_set = TongueDataset(val_csv, args.image_root, args.label_cols, is_train=False)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # vvv 根據 --model 參數選擇模型 vvv
    if args.model == 'SignOriented':
        model = SignOrientedNetwork(num_classes=len(args.label_cols), backbone=args.backbone, feature_dim=args.feature_dim)
    elif args.model == 'Simple':
        model = SimpleTimmModel(num_classes=len(args.label_cols), backbone=args.backbone)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    model.to(device)

    # Loss
    if args.loss == 'ASL':
        from losses_asl import AsymmetricLoss
        criterion = AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05)
    elif args.loss == 'CB_BCE':
        from losses import ClassBalancedBCELoss
        criterion = ClassBalancedBCELoss(n_pos=n_pos_t.to(device), n_neg=n_neg_t.to(device))
    else:
        raise ValueError(f"Unknown loss type: {args.loss}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    best_metric, patience_counter = -1.0, 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training", leave=False)
        for (x_whole, x_root, x_center, x_side, x_tip), labels in loop:
            x_whole, x_root, x_center, x_side, x_tip = x_whole.to(device), x_root.to(device), x_center.to(device), x_side.to(device), x_tip.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            with autocast():
                logits = model(x_whole, x_root, x_center, x_side, x_tip)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (len(loop)))
        
        scheduler.step()

        if val_loader:
            metrics, best_thresholds = evaluate_model_and_find_thresholds(model, val_loader, device, args.label_cols)
            current_metric = metrics[args.watch_metric]
            print(f"\nEpoch {epoch+1}/{args.epochs} | Train Loss: {total_loss/len(loop):.4f} | Val {args.watch_metric}: {current_metric:.4f}")

            if current_metric > best_metric:
                best_metric = current_metric
                state = {'model_state_dict': model.state_dict()}
                torch.save(state, model_path)
                th_path = os.path.splitext(model_path)[0] + "_best_thresholds.json"
                with open(th_path, "w") as f:
                    json.dump({label: th for label, th in zip(args.label_cols, best_thresholds.tolist())}, f, indent=2)
                print(f"✅ Saved best model to {model_path} (Val {args.watch_metric}: {best_metric:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    if not val_loader:
        torch.save({'model_state_dict': model.state_dict()}, model_path)
        print(f"✅ Saved final model to {model_path}")

def main(args):
    # PHASE 1: K-Fold CV
    print("="*25 + "\n  PHASE 1: K-Fold CV\n" + "="*25)
    for i in range(1, args.num_folds + 1):
        train_csv, val_csv = f'train_fold{i}.csv', f'val_fold{i}.csv'
        if not all(os.path.exists(f) for f in [train_csv, val_csv]):
            print(f"Skipping Fold {i}: CSV not found.")
            continue
        print(f"\n====== Training Fold {i} ======")
        model_path = f'{args.model}_{args.backbone}_fold{i}.pth'
        train_model(args, train_csv, val_csv, model_path)

    # PHASE 2: Final Training
    print("\n" + "="*25 + "\n  PHASE 2: Final Training\n" + "="*25)
    all_csvs = [pd.read_csv(f) for i in range(1, args.num_folds + 1) for f in [f'train_fold{i}.csv', f'val_fold{i}.csv'] if os.path.exists(f)]
    if not all_csvs:
        print("No CSVs found for final training. Exiting.")
        return
    full_df = pd.concat(all_csvs).drop_duplicates().reset_index(drop=True)
    full_train_csv = 'train_full.csv'
    full_df.to_csv(full_train_csv, index=False)
    
    final_model_path = f'{args.model}_{args.backbone}_final.pth'
    train_model(args, full_train_csv, None, final_model_path)

    # PHASE 3: Averaging Thresholds
    print("\n" + "="*25 + "\n  PHASE 3: Averaging Thresholds\n" + "="*25)
    all_ths = []
    for i in range(1, args.num_folds + 1):
        th_path = f'{args.model}_{args.backbone}_fold{i}_best_thresholds.json'
        if os.path.exists(th_path):
            with open(th_path, 'r') as f:
                all_ths.append(list(json.load(f).values()))
    if all_ths:
        avg_ths = np.mean(all_ths, axis=0)
        final_th_path = os.path.splitext(final_model_path)[0] + "_best_thresholds.json"
        with open(final_th_path, 'w') as f:
            json.dump({label: th for label, th in zip(args.label_cols, avg_ths.tolist())}, f, indent=2)
        print(f"✅ Averaged thresholds saved to {final_th_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Tongue Diagnosis Models")
    parser.add_argument('--model', type=str, default='Simple', choices=['SignOriented', 'Simple'], help='Model architecture to use.')
    parser.add_argument('--backbone', type=str, default='convnext_base', help='Backbone model from timm library.')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--loss', type=str, default='ASL', choices=['ASL', 'CB_BCE'])
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--image-root', type=str, default='images')
    parser.add_argument('--feature-dim', type=int, default=512, help='Feature dimension for SignOriented model.')
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--watch-metric', type=str, default='f1_macro', help='Metric to watch for saving best model.')
    
    args = parser.parse_args()
    args.label_cols = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis', 'Crack', 'Toothmark', 'FurThick', 'FurYellow']
    main(args)
