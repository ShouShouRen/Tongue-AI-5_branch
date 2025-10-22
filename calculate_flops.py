# calculate_flops.py - 計算 PyTorch 模型的 FLOPS 和參數數量

import torch
import argparse
from fvcore.nn import FlopCountAnalysis, parameter_count

# 從您的模型庫匯入模型
from model import SimpleTimmModel, SignOrientedNetwork, SignOrientedAttentionNetwork

def get_model(args):
    """根據 args 建立對應的模型"""
    model_class = None
    if args.model == 'SignOriented':
        model_class = SignOrientedNetwork
    elif args.model == 'Simple':
        model_class = SimpleTimmModel
    elif args.model == 'SignOrientedAttention':
        model_class = SignOrientedAttentionNetwork
    else:
        raise ValueError(f"未知的模型架構: {args.model}")

    print(f"初始化模型: {args.model} (骨幹: {args.backbone})")
    print("-" * 30)

    # SimpleTimmModel 不需要 feature_dim
    if args.model == 'Simple':
        model = model_class(
            num_classes=args.num_classes,
            backbone=args.backbone
        )
    else:
        model = model_class(
            num_classes=args.num_classes,
            backbone=args.backbone,
            feature_dim=args.feature_dim
        )
    return model.eval() # 設定為評估模式

def calculate_flops(args):
    """計算指定模型的 FLOPS 和參數"""
    model = get_model(args)
    img_size = args.img_size

    # 建立符合模型輸入的虛擬張量 (Batch size = 1)
    dummy_input_whole = torch.randn(1, 3, img_size, img_size)

    if args.model == 'Simple':
        # Simple 模型只需要一個輸入
        inputs = (dummy_input_whole,)
    else:
        # SignOriented* 模型需要五個輸入
        dummy_input_root = torch.randn(1, 3, img_size, img_size)
        dummy_input_center = torch.randn(1, 3, img_size, img_size)
        dummy_input_side = torch.randn(1, 3, img_size, img_size)
        dummy_input_tip = torch.randn(1, 3, img_size, img_size)
        inputs = (dummy_input_whole, dummy_input_root, dummy_input_center, dummy_input_side, dummy_input_tip)

    print(f"使用輸入尺寸: {img_size}x{img_size}")
    print("開始計算 FLOPS...")

    # 使用 fvcore 計算 FLOPS
    try:
        flops = FlopCountAnalysis(model, inputs)
        total_flops = flops.total()
        
        # 計算參數數量
        params_count_dict = parameter_count(model)
        total_params = params_count_dict[''] # 總參數

        gflops = total_flops / 1e9
        tflops = total_flops / 1e12
        mparams = total_params / 1e6

        print("-" * 30)
        print("計算完成!")
        print(f"模型架構      : {args.model}")
        print(f"骨幹網路      : {args.backbone}")
        print(f"輸入尺寸      : {img_size}x{img_size}")
        print("-" * 30)
        print(f"總參數 (Params) : {mparams:.2f} M")
        print(f"總計算量 (FLOPs): {gflops:.2f} GFLOPS")
        print(f"總計算量 (FLOPs): {tflops:.4f} TFLOPS")
        print(f"FLOPs  : {total_flops} FLOPs")
        print("-" * 30)
        # print("\n詳細分析:")
        # print(flops.by_module()) # 印出每個模組的詳細計算量 (可選)

    except Exception as e:
        print("\n計算 FLOPS 時發生錯誤:")
        print(e)
        print("\n可能原因:")
        print("- 模型定義錯誤或與輸入不匹配。")
        print("- fvcore 不支援模型中的某些操作 (較少見)。")
        print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='計算 PyTorch 模型的 FLOPS 和參數數量')
    parser.add_argument('--model', type=str, required=True, choices=['Simple', 'SignOriented', 'SignOrientedAttention'], help='要分析的模型架構')
    parser.add_argument('--backbone', type=str, required=True, help='模型使用的骨幹網路, e.g., "convnext_base", "resnet50", "vim_tiny_patch16_224"')
    parser.add_argument('--img_size', type=int, default=224, help='輸入圖片的尺寸 (預設: 224)')
    parser.add_argument('--num_classes', type=int, default=8, help='模型的輸出類別數量 (預設: 8)')
    parser.add_argument('--feature_dim', type=int, default=512, help='特徵維度 (僅用於 SignOriented 系列模型, 預設: 512)')

    args = parser.parse_args()

    calculate_flops(args)

### 如何使用

#1.  **儲存檔案**：將上面的程式碼儲存為 `calculate_flops.py`。
#2.  **執行計算**：
#    * **分析 `Simple` + `ConvNeXt` (預設 224x224)**：
#      ```bash
#     python calculate_flops.py --model Simple --backbone convnext_base
#      ```
#    * **分析 `SignOrientedAttention` + `ResNet50` (指定 384x384)**：
#      ```bash
#      python calculate_flops.py --model SignOrientedAttention --backbone resnet50 --img_size 384
#      ```
#    * **分析 `Simple` + `Vim` (建議 224x224)**：
#      ```bash
#      python calculate_flops.py --model Simple --backbone vim_tiny_patch16_224 --img_size 224
      
