# simple_predict.py - 簡單預測單張照片
import torch
from model import SignOrientedNetwork
from PIL import Image
import torchvision.transforms as transforms
import sys

def predict_image(image_path, model_path="signnet_best_fold1.pth"):
    """預測單張圖像"""
    
    # 標籤定義
    labels = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis',
              'Crack', 'Toothmark', 'FurThick', 'FurYellow']
    
    chinese_labels = ['舌淡白', '舌尖邊紅', '紅點', '瘀斑', '裂紋', '齒痕', '苔厚', '苔黃']
    
    # 載入模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignOrientedNetwork(num_classes=8, backbone='swin_base_patch4_window7_224', feature_dim=512)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 圖像預處理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 載入並處理圖像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 預測
    with torch.no_grad():
        outputs = model(image_tensor, image_tensor, image_tensor)
        probs = torch.sigmoid(outputs['final']).cpu().numpy()[0]
    
    # 打印結果
    print(f"\n預測結果 - {image_path}")
    print("=" * 50)
    
    for i, (eng, chi, prob) in enumerate(zip(labels, chinese_labels, probs)):
        status = "✓" if prob > 0.5 else "✗"
        print(f"{chi}({eng}): {prob:.3f} {status}")
    
    # 統計
    positive_count = sum(1 for p in probs if p > 0.5)
    print(f"\n檢測到 {positive_count} 個陽性症狀")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python simple_predict.py <圖像路徑>")
        print("例如: python simple_predict.py test.jpg")
    else:
        predict_image(sys.argv[1])