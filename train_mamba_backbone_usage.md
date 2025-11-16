# Mamba 作為 Backbone 的使用指南

## 🎯 兩種 Mamba 使用方式對比

### 方案 A：Mamba 作為 Backbone（你想要的）

```
圖片 → Mamba Backbone → 特徵 → 分類
```

**優點**：Mamba 直接處理圖片，完全拋棄 CNN
**適合**：想探索 Mamba 在視覺任務的潛力

### 方案 B：CNN + Mamba 融合（我之前給的）

```
圖片 → CNN Backbone → 特徵 → Mamba 融合 → 分類
```

**優點**：結合 CNN 的視覺能力和 Mamba 的序列建模
**適合**：追求最佳性能

---

## 🚀 使用 Mamba Backbone 訓練

### 步驟 1: 更新 model_zoo.py

在你的 `model_zoo.py` 中添加上面三個新類別：

- `VisionMambaBackbone` - Mamba 視覺骨幹
- `SignOrientedMambaBackboneNetwork` - 五分支 Mamba
- `SimpleMambaBackbone` - 簡化版（測試用）

### 步驟 2: 更新 train.py 的 get_model 函數

```python
def get_model(args):
    """根據 args 建立對應的模型"""
    from model_zoo import (
        SignOrientedNetwork,
        SimpleTimmModel,
        SignOrientedAttentionNetwork,
        SignOrientedMambaBackboneNetwork,  # 新增：Mamba 作為 Backbone
        SignOrientedHybridNetwork,         # 新增：CNN + Mamba 融合
        SimpleMambaBackbone                # 新增：簡化版
    )

    model_map = {
        'SignOriented': SignOrientedNetwork,
        'Simple': SimpleTimmModel,
        'SignOrientedAttention': SignOrientedAttentionNetwork,
        'MambaBackbone': SignOrientedMambaBackboneNetwork,  # 🆕
        'MambaHybrid': SignOrientedHybridNetwork,           # 🆕
        'SimpleMamba': SimpleMambaBackbone                  # 🆕
    }

    if args.model not in model_map:
        raise ValueError(f"未知的模型: {args.model}")

    model_class = model_map[args.model]

    # SimpleTimmModel
    if args.model == 'Simple':
        return model_class(
            num_classes=len(args.label_cols),
            backbone=args.backbone
        )

    # Mamba Backbone 模型
    elif args.model == 'MambaBackbone':
        return model_class(
            num_classes=len(args.label_cols),
            img_size=getattr(args, 'img_size', 224),
            patch_size=getattr(args, 'patch_size', 16),
            embed_dim=getattr(args, 'embed_dim', 512),
            depth=getattr(args, 'mamba_depth', 6),
            d_state=getattr(args, 'd_state', 16),
            feature_dim=args.feature_dim
        )

    # SimpleMamba
    elif args.model == 'SimpleMamba':
        return model_class(
            num_classes=len(args.label_cols),
            img_size=getattr(args, 'img_size', 224),
            patch_size=getattr(args, 'patch_size', 16),
            embed_dim=getattr(args, 'embed_dim', 384),
            depth=getattr(args, 'mamba_depth', 4),
            d_state=getattr(args, 'd_state', 16)
        )

    # Mamba Hybrid (CNN + Mamba)
    elif args.model == 'MambaHybrid':
        return model_class(
            num_classes=len(args.label_cols),
            backbone=args.backbone,
            feature_dim=args.feature_dim,
            d_state=getattr(args, 'd_state', 16)
        )

    # 其他模型
    else:
        return model_class(
            num_classes=len(args.label_cols),
            backbone=args.backbone,
            feature_dim=args.feature_dim
        )
```

### 步驟 3: 添加命令列參數

在 train.py 的 argparse 部分添加：

```python
parser.add_argument('--model', type=str, required=True,
                    choices=['Simple', 'SignOriented', 'SignOrientedAttention',
                            'MambaBackbone', 'MambaHybrid', 'SimpleMamba'],
                    help='模型架構')

# Mamba Backbone 專用參數
parser.add_argument('--img_size', type=int, default=224,
                    help='輸入圖片大小')
parser.add_argument('--patch_size', type=int, default=16,
                    help='Patch 大小（Mamba Backbone 用）')
parser.add_argument('--embed_dim', type=int, default=512,
                    help='Mamba embedding 維度')
parser.add_argument('--mamba_depth', type=int, default=6,
                    help='Mamba 層數')
parser.add_argument('--d_state', type=int, default=16,
                    help='Mamba 狀態空間維度')
```

---

## 📝 訓練指令範例

### 🔥 配置 1: 簡單測試（推薦首次使用）

```bash
python train.py \
  --model SimpleMamba \
  --backbone dummy \
  --epochs 5 \
  --batch_size 16 \
  --lr 2e-4 \
  --embed_dim 256 \
  --mamba_depth 4 \
  --d_state 8 \
  --patch_size 16
```

### 🔥 配置 2: 完整 Mamba Backbone（五分支）

```bash
python train.py \
  --model MambaBackbone \
  --backbone dummy \
  --epochs 30 \
  --batch_size 12 \
  --lr 1e-4 \
  --feature_dim 512 \
  --embed_dim 512 \
  --mamba_depth 6 \
  --d_state 16 \
  --patch_size 16 \
  --img_size 224
```

### 🔥 配置 3: CNN + Mamba 混合（最強性能）

```bash
python train.py \
  --model MambaHybrid \
  --backbone convnext_base \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-5 \
  --feature_dim 512 \
  --d_state 16
```

### 🔥 配置 4: 小型 Mamba（顯存不足時）

```bash
python train.py \
  --model MambaBackbone \
  --backbone dummy \
  --epochs 30 \
  --batch_size 24 \
  --lr 2e-4 \
  --feature_dim 384 \
  --embed_dim 384 \
  --mamba_depth 4 \
  --d_state 12 \
  --patch_size 16
```

---

## 🎛️ 超參數調整指南

### Mamba Backbone 專用參數

| 參數          | 預設值 | 範圍    | 影響                              |
| ------------- | ------ | ------- | --------------------------------- |
| `embed_dim`   | 512    | 256-768 | Mamba 特徵維度，越大越強但越慢    |
| `mamba_depth` | 6      | 4-12    | Mamba 層數，類似 Transformer 深度 |
| `d_state`     | 16     | 8-32    | 狀態空間維度，控制記憶容量        |
| `patch_size`  | 16     | 8/16/32 | Patch 大小，越小序列越長          |

### 推薦配置組合

#### 輕量級（快速實驗）

```bash
--embed_dim 256 --mamba_depth 4 --d_state 8 --patch_size 16
```

#### 標準（平衡性能）

```bash
--embed_dim 512 --mamba_depth 6 --d_state 16 --patch_size 16
```

#### 重量級（追求極致）

```bash
--embed_dim 768 --mamba_depth 12 --d_state 24 --patch_size 8
```

---

## 📊 三種方案對比

| 方案              | Backbone | 融合方式   | 參數量 | 訓練速度 | 推薦場景      |
| ----------------- | -------- | ---------- | ------ | -------- | ------------- |
| **SimpleMamba**   | Mamba    | 單分支     | 最少   | 最快     | 快速驗證      |
| **MambaBackbone** | Mamba    | 拼接       | 中等   | 中等     | 純 Mamba 方案 |
| **MambaHybrid**   | CNN      | Mamba 融合 | 最多   | 較慢     | 追求性能      |

---

## 🔬 實驗建議

### 對比實驗 1: Backbone 對比

```bash
# A. 傳統 CNN
python train.py --model SignOriented --backbone convnext_base

# B. Mamba Backbone
python train.py --model MambaBackbone --embed_dim 512 --mamba_depth 6

# C. CNN + Mamba 混合
python train.py --model MambaHybrid --backbone convnext_base --d_state 16
```

### 對比實驗 2: Mamba 深度影響

```bash
# 淺層 Mamba
python train.py --model MambaBackbone --mamba_depth 4

# 中層 Mamba
python train.py --model MambaBackbone --mamba_depth 6

# 深層 Mamba
python train.py --model MambaBackbone --mamba_depth 12
```

### 對比實驗 3: Patch 大小影響

```bash
# 大 Patch (序列短，速度快)
python train.py --model MambaBackbone --patch_size 32

# 中 Patch (平衡)
python train.py --model MambaBackbone --patch_size 16

# 小 Patch (序列長，細節多)
python train.py --model MambaBackbone --patch_size 8
```

---

## ⚡ 效能優化

### 如果訓練太慢

1. **減少 patch 數量**

   ```bash
   --patch_size 32  # 從 16 改成 32
   ```

2. **減少 Mamba 深度**

   ```bash
   --mamba_depth 4  # 從 6 改成 4
   ```

3. **減少 embedding 維度**
   ```bash
   --embed_dim 384  # 從 512 改成 384
   ```

### 如果顯存不足

```bash
python train.py \
  --model SimpleMamba \
  --batch_size 8 \
  --embed_dim 256 \
  --mamba_depth 4 \
  --patch_size 16
```

---

## 🐛 常見問題

### Q1: `--backbone dummy` 是什麼意思？

A: 當使用 `MambaBackbone` 或 `SimpleMamba` 時，不需要 CNN backbone，但 argparse 可能要求這個參數，所以填 `dummy` 作為佔位符。

### Q2: Mamba Backbone 比 CNN 慢嗎？

A: 取決於配置：

- 小型 Mamba (`depth=4, embed_dim=256`) 比 CNN 快
- 大型 Mamba (`depth=12, embed_dim=768`) 比 CNN 慢

### Q3: 該選哪個模型？

A:

- **想快速驗證**: `SimpleMamba`
- **想純 Mamba**: `MambaBackbone`
- **想要最佳性能**: `MambaHybrid` (CNN + Mamba)

---

## 🎓 Mamba 作為 Backbone 的原理

### 為什麼 Mamba 可以當 Backbone？

1. **Patch 序列化**: 圖片切成 16x16 patches → 變成序列
2. **位置編碼**: 加上位置資訊
3. **Mamba 處理**: 用狀態空間模型處理這個序列
4. **全局池化**: 平均所有 patch 特徵

```
輸入圖片 (224x224x3)
    ↓
切成 Patches (196個 16x16 patches)
    ↓
Patch Embedding (196x512)
    ↓
+ 位置編碼
    ↓
Mamba Block 1
Mamba Block 2
...
Mamba Block N
    ↓
Global Pooling → (512維向量)
    ↓
分類器 → 8類標籤
```

這就像 Vision Transformer (ViT)，但把 Attention 換成 Mamba！

---

## ✅ 快速開始

```bash
# 1. 確保環境正確
conda activate mamba_stable
cd /mnt/c/Users/peter/Tongue-AI-V2

# 2. 快速測試（3-5分鐘）
python train.py \
  --model SimpleMamba \
  --backbone dummy \
  --epochs 3 \
  --batch_size 16 \
  --lr 2e-4

# 3. 如果成功，進行完整訓練
python train.py \
  --model MambaBackbone \
  --backbone dummy \
  --epochs 30 \
  --batch_size 12 \
  --lr 1e-4 \
  --embed_dim 512 \
  --mamba_depth 6
```

現在你真的可以把 Mamba 當作 Backbone 使用了！🚀
