# calculate_flops.py - 計算 PyTorch 模型的 FLOPS 和參數數量
# (已更新，包含 TFLOPs/s 和 FPS 效能測試)

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
    print("開始計算 FLOPS (理論計算量)...")

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

        print("理論計算量計算完成!")

        # --- 開始計算實際推論速度 (TFLOPs/s) ---
        print("\n開始計算實際推論速度 (TFLOPs/s)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        avg_time_sec = 0.0
        fps = 0.0
        effective_tflops_per_sec = 0.0

        if device.type == 'cpu':
            print("警告: 偵測到 CPU。TFLOPs/s 的計算在 GPU 上執行才有意義。")
            print("跳過實際效能測試。")
        else:
            print(f"使用裝置: {torch.cuda.get_device_name(device)}")
            model.to(device)
            # 將輸入張量也移動到 GPU
            inputs = tuple(inp.to(device) for inp in inputs)

            # 預熱 (Warm-up) - 讓 GPU 進入高效能狀態
            print("預熱中 (Warm-up)...")
            for _ in range(30):
                _ = model(*inputs)
            
            torch.cuda.synchronize() # 確保預熱完成

            # 準確計時
            num_runs = 100
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            print(f"執行 {num_runs} 次推論以取得平均時間...")
            
            start_event.record()
            for _ in range(num_runs):
                _ = model(*inputs)
            end_event.record()

            # 等待所有 CUDA 核心完成
            torch.cuda.synchronize()

            # 計算時間
            elapsed_time_ms = start_event.elapsed_time(end_event)
            avg_time_sec = (elapsed_time_ms / 1000) / num_runs # 每次推論的平均秒數
            fps = 1.0 / avg_time_sec # 每秒推論次數 (FPS)
            
            # TFLOPs/s = (每次推論的 FLOPs / 每次推論的秒數) / 1e12
            effective_tflops_per_sec = (total_flops / avg_time_sec) / 1e12
            print("效能測試完成!")

        print("-" * 30)
        print("計算總結")
        print(f"模型架構      : {args.model}")
        print(f"骨幹網路      : {args.backbone}")
        print(f"輸入尺寸      : {img_size}x{img_size}")
        print("-" * 30)
        print("模型複雜度 (理論):")
        print(f"總參數 (Params) : {mparams:.2f} M")
        print(f"總計算量 (FLOPs): {gflops:.2f} GFLOPS (理論單次)")
        print(f"總計算量 (FLOPs): {tflops:.4f} TFLOPS (理論單次)")
        print("-" * 30)
        print("實際硬體效能 (Batch Size=1):")
        if device.type == 'cuda':
            print(f"平均推論時間 : {avg_time_sec * 1000:.2f} ms")
            print(f"每秒推論次數(FPS): {fps:.2f} FPS")
            print(f"實際計算效能 : {effective_tflops_per_sec:.2f} TFLOPs/s")
        else:
            print(" (僅在 CUDA 裝置上計算)")
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
