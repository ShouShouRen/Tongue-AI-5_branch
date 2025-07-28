# simple_evaluate.py - ResNet版本簡單測試指標
import torch
from torch.utils.data import DataLoader
from dataset import TongueDataset
from model import SignOrientedNetwork
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
import numpy as np
import sys

def evaluate_model(model_path, val_csv, image_root="images"):
    """評估模型指標"""
    
    # 標籤定義
    labels = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis',
              'Crack', 'Toothmark', 'FurThick', 'FurYellow']
    
    # 載入模型（用 ResNet50 作為 backbone）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignOrientedNetwork(num_classes=8, backbone='resnet50', feature_dim=512)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"載入模型: {model_path}")
    print(f"驗證數據: {val_csv}")
    
    # 載入數據
    val_set = TongueDataset(val_csv, image_root, labels, is_train=False)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"驗證樣本數: {len(val_set)}")
    
    # 預測
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for (x_whole, x_body, x_edge), labels_batch in val_loader:
            x_whole = x_whole.to(device)
            x_body = x_body.to(device)
            x_edge = x_edge.to(device)
            labels_batch = labels_batch.to(device)
            
            outputs = model(x_whole, x_body, x_edge)
            probs = torch.sigmoid(outputs['final']).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_preds.append(preds)
            all_labels.append(labels_batch.cpu().numpy())
            all_probs.append(probs)
    
    # 合併結果
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    
    # 計算指標
    print("\n評估結果")
    print("=" * 60)
    
    # 整體指標
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    jaccard_macro = jaccard_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"F1-Score (Micro): {f1_micro:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"Precision (Macro): {precision_macro:.4f}")
    print(f"Recall (Macro): {recall_macro:.4f}")
    print(f"Jaccard (Macro): {jaccard_macro:.4f}")
    
    # 各類別指標
    print(f"\n各類別詳細指標")
    print("-" * 60)
    print(f"{'Label':<15} {'F1':<6} {'Prec':<6} {'Rec':<6} {'Jac':<6} {'Support':<8}")
    print("-" * 60)
    
    for i, label in enumerate(labels):
        f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        precision = precision_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        recall = recall_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        jaccard = jaccard_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        support = all_labels[:, i].sum()
        
        print(f"{label:<15} {f1:<6.3f} {precision:<6.3f} {recall:<6.3f} {jaccard:<6.3f} {support:<8.0f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使用方法: python simple_evaluate.py <模型路徑> <驗證CSV>")
        print("例如: python simple_evaluate.py signnet_resnet50_fold1.pth val_fold1.csv")
    else:
        evaluate_model(sys.argv[1], sys.argv[2])
