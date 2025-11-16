# model.py — 模型庫：存放多個可選的模型架構
# --- 新增 Mamba 模型 ---
import sys
import os
# 將目前檔案 (model.py) 所在的目錄 (專案根目錄) 加入到 Python 的搜尋路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from timm import create_model
from mamba_ssm import Mamba

# --- 方案二(困難版) 所需的額外匯入 ---
import os
from collections import OrderedDict

# -----------------------------------------------------------------------------
# 模型一：您原本的五分支模型 (SignOrientedNetwork)
# -----------------------------------------------------------------------------
class SignOrientedNetwork(nn.Module):
    def __init__(self, num_classes=8, backbone='swin_base_patch4_window7_224', feature_dim=512):
        super().__init__()
        print(f"Initializing SignOrientedNetwork with backbone: {backbone}")

        self.encoder = create_model(backbone, pretrained=True, features_only=True)
        self.enc_dim = self.encoder.feature_info.channels()[-1]

        self.branch_hints = {
            "side":   [1,1,1,1,0,1,0,0], "tip":    [1,1,0,1,0,0,0,0],
            "center": [0,0,0,0,1,0,1,1], "root":   [0,1,0,0,1,0,0,0],
            "whole":  [1,1,1,1,1,1,1,1],
        }

        self.branch_proj = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.enc_dim + num_classes, feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ) for name in self.branch_hints.keys()
        })

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        fused_dim = feature_dim * 5
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)[-1]
        if feat.dim() == 4 and feat.shape[-1] == self.enc_dim:
            feat = feat.permute(0, 3, 1, 2).contiguous()
        return self.flatten(self.pool(feat)) if feat.dim() == 4 else feat

    def _add_hint(self, flat: torch.Tensor, branch_name: str, device) -> torch.Tensor:
        hint = torch.tensor(self.branch_hints[branch_name], device=device, dtype=torch.float32)
        hint = hint.unsqueeze(0).expand(flat.size(0), -1)
        return self.branch_proj[branch_name](torch.cat([flat, hint], dim=1))

    def forward(self, x_whole, x_root, x_center, x_side, x_tip):
        features = [
            self._add_hint(self._encode(x_whole),  "whole",  x_whole.device),
            self._add_hint(self._encode(x_root),   "root",   x_root.device),
            self._add_hint(self._encode(x_center), "center", x_center.device),
            self._add_hint(self._encode(x_side),   "side",   x_side.device),
            self._add_hint(self._encode(x_tip),    "tip",    x_tip.device)
        ]
        fused = torch.cat(features, dim=1)
        return self.classifier(fused)

# -----------------------------------------------------------------------------
# 模型二：簡潔的單一輸入模型 (SimpleTimmModel)
# -----------------------------------------------------------------------------
class SimpleTimmModel(nn.Module):
    def __init__(self, num_classes, backbone='convnext_base', feature_dim=512,img_size=224):
        super().__init__()
        print(f"Initializing SimpleTimmModel with backbone: {backbone}")
        # 1. 建立一個 kwargs 字典
        model_kwargs = {
            'pretrained': True,
            'num_classes': num_classes
        }

        # 2. 智慧判斷：只有 ViT/Swin/DINO 類型的模型才需要 img_size
        if 'vit' in backbone or 'swin' in backbone or 'dinov2' in backbone:
            print(f"  -> (ViT/Swin/DINO) 傳遞 img_size={img_size}")
            model_kwargs['img_size'] = img_size
        else:
            print(f"  -> (CNN) 不傳遞 img_size")

        # 3. 使用 **kwargs 語法來建立模型
        self.model = create_model(backbone, **model_kwargs)

    def forward(self, x_whole, x_root, x_center, x_side, x_tip):
        return self.model(x_whole)

# -----------------------------------------------------------------------------
# 模型三：優化版！使用 Attention 融合的複雜網路 (SignOrientedAttentionNetwork)
# -----------------------------------------------------------------------------
class SignOrientedAttentionNetwork(nn.Module):
    def __init__(self, num_classes=8, backbone='swin_base_patch4_window7_224', feature_dim=512):
        super().__init__()
        print(f"Initializing SignOrientedAttentionNetwork with backbone: {backbone}")

        self.encoder = create_model(backbone, pretrained=True, features_only=True)
        self.enc_dim = self.encoder.feature_info.channels()[-1]
        self.branch_hints = {
            "side":   [1,1,1,1,0,1,0,0], "tip":    [1,1,0,1,0,0,0,0],
            "center": [0,0,0,0,1,0,1,1], "root":   [0,1,0,0,1,0,0,0],
            "whole":  [1,1,1,1,1,1,1,1],
        }
        self.branch_proj = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.enc_dim + num_classes, feature_dim),
                nn.ReLU(inplace=True)
            ) for name in self.branch_hints.keys()
        })
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        self.fusion_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=feature_dim * 2,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_classes)
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)[-1]
        if feat.dim() == 4 and feat.shape[-1] == self.enc_dim:
            feat = feat.permute(0, 3, 1, 2).contiguous()
        return self.flatten(self.pool(feat)) if feat.dim() == 4 else feat

    def _add_hint(self, flat: torch.Tensor, branch_name: str, device) -> torch.Tensor:
        hint = torch.tensor(self.branch_hints[branch_name], device=device, dtype=torch.float32)
        hint = hint.unsqueeze(0).expand(flat.size(0), -1)
        return self.branch_proj[branch_name](torch.cat([flat, hint], dim=1))

    def forward(self, x_whole, x_root, x_center, x_side, x_tip):
        features = [
            self._add_hint(self._encode(x_whole),  "whole",  x_whole.device),
            self._add_hint(self._encode(x_root),   "root",   x_root.device),
            self._add_hint(self._encode(x_center), "center", x_center.device),
            self._add_hint(self._encode(x_side),   "side",   x_side.device),
            self._add_hint(self._encode(x_tip),    "tip",    x_tip.device)
        ]
        
        feature_sequence = torch.stack(features, dim=1)
        fused_sequence = self.fusion_layer(feature_sequence)
        fused_representation = fused_sequence[:, 0, :]
        
        return self.classifier(fused_representation)

# ============================================================================
# Vision Mamba Backbone: (手動搭建，從零訓練)
# ============================================================================

class PatchEmbedding(nn.Module):
    """將圖片切成 patches 並嵌入"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x



# -----------------------------------------------------------------------------
# 載入 MambaVision 預訓練模型
# -----------------------------------------------------------------------------
try:
    # 
    #  唯一的修改： 移除 mamba_vision 的底線
    # 
    from mambavision import create_model as create_mamba_vision_model
    MAMBA_VISION_AVAILABLE = True
except ImportError:
    print("="*50)
    # 
    #  (可選) 也可以順便修改這裡的警告訊息，保持一致
    # 
    print("警告：找不到 'mambavision'。") 
    print("請確認已在 MambaVision repo 目錄執行: pip install . --no-deps")
    print("'MambaVision' 系列模型將無法使用。")
    print("="*50)
    MAMBA_VISION_AVAILABLE = False

# ============================================================================
# (模型十) 🆕 使用 MambaVision 預訓練模型作為 Backbone（五分支）
# ============================================================================
class SignOrientedMambaVisionNetwork(nn.Module):
    """
    五分支模型，使用 MambaVision 預訓練模型作為 Backbone
    """
    def __init__(self, num_classes=8, mamba_vision_model='mamba_vision_T', 
                 feature_dim=512, pretrained=True, freeze_backbone=False):
        super().__init__()
        
        if not MAMBA_VISION_AVAILABLE:
            raise ImportError("請先安裝 mamba-vision: pip install mamba-vision")
        
        print(f"Initializing SignOrientedMambaVisionNetwork")
        print(f"  -> MambaVision 模型: {mamba_vision_model}")
        print(f"  -> 使用預訓練: {pretrained}")
        print(f"  -> 凍結 Backbone: {freeze_backbone}")
        
        # 載入 MambaVision 預訓練模型（設定 num_classes=0 來移除分類頭）
        self.encoder = create_mamba_vision_model(
            mamba_vision_model,
            pretrained=pretrained,
            num_classes=0  # 移除原始分類頭，只保留特徵提取部分
        )
        
        # 如果需要凍結 backbone
        if freeze_backbone:
            print("  -> 凍結 MambaVision Backbone 參數")
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # 獲取 MambaVision 的輸出特徵維度
        # 不同的 MambaVision 模型有不同的輸出維度
        model_dims = {
            'mamba_vision_T': 640,   # Tiny
            'mamba_vision_T2': 640,  # Tiny2
            'mamba_vision_S': 768,   # Small
            'mamba_vision_B': 1024,  # Base
            'mamba_vision_L': 1024,  # Large
        }
        self.enc_dim = model_dims.get(mamba_vision_model, 640)
        
        self.branch_hints = {
            "side":   [1,1,1,1,0,1,0,0], 
            "tip":    [1,1,0,1,0,0,0,0],
            "center": [0,0,0,0,1,0,1,1], 
            "root":   [0,1,0,0,1,0,0,0],
            "whole":  [1,1,1,1,1,1,1,1],
        }
        
        # 分支投影層：將 MambaVision 特徵 + hint 映射到統一維度
        self.branch_proj = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.enc_dim + num_classes, feature_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(feature_dim),
                nn.Dropout(0.1)
            ) for name in self.branch_hints.keys()
        })
        
        # 最終分類器
        fused_dim = feature_dim * 5
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def _add_hint(self, flat: torch.Tensor, branch_name: str, device) -> torch.Tensor:
        hint = torch.tensor(self.branch_hints[branch_name], device=device, dtype=torch.float32)
        hint = hint.unsqueeze(0).expand(flat.size(0), -1)
        return self.branch_proj[branch_name](torch.cat([flat, hint], dim=1))
    
    def forward(self, x_whole, x_root, x_center, x_side, x_tip):
        # 使用 MambaVision encoder 提取特徵
        features = [
            self._add_hint(self.encoder(x_whole),  "whole",  x_whole.device),
            self._add_hint(self.encoder(x_root),   "root",   x_root.device),
            self._add_hint(self.encoder(x_center), "center", x_center.device),
            self._add_hint(self.encoder(x_side),   "side",   x_side.device),
            self._add_hint(self.encoder(x_tip),    "tip",    x_tip.device)
        ]
        
        fused = torch.cat(features, dim=1)
        return self.classifier(fused)


# ============================================================================
# (模型十一) 🆕 簡化版：單分支 MambaVision（快速測試用）
# ============================================================================
class SimpleMambaVision(nn.Module):
    """
    最簡單的 MambaVision 模型 (只用 whole 分支)
    適合快速測試和對比實驗
    """
    def __init__(self, num_classes=8, mamba_vision_model='mamba_vision_T', 
                 pretrained=True, freeze_backbone=False):
        super().__init__()
        
        if not MAMBA_VISION_AVAILABLE:
            raise ImportError("請先安裝 mamba-vision: pip install mamba-vision")
        
        print(f"Initializing SimpleMambaVision")
        print(f"  -> MambaVision 模型: {mamba_vision_model}")
        print(f"  -> 使用預訓練: {pretrained}")
        print(f"  -> 凍結 Backbone: {freeze_backbone}")
        
        # 直接使用 MambaVision 的完整模型（包含分類頭）
        self.model = create_mamba_vision_model(
            mamba_vision_model,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # 如果需要凍結 backbone（保留分類頭可訓練）
        if freeze_backbone:
            print("  -> 凍結 MambaVision Backbone，只訓練分類頭")
            for name, param in self.model.named_parameters():
                if 'head' not in name.lower():  # 只凍結非 head 的參數
                    param.requires_grad = False
    
    def forward(self, x_whole, x_root, x_center, x_side, x_tip):
        # 只使用 whole 分支
        return self.model(x_whole)


# ============================================================================
# (模型十二) 🆕 MambaVision + Mamba Fusion（混合架構）
# ============================================================================
class MambaVisionWithMambaFusion(nn.Module):
    """
    使用 MambaVision 作為特徵提取器 + Mamba 模組進行特徵融合
    結合了兩種 Mamba 的優勢
    """
    def __init__(self, num_classes=8, mamba_vision_model='mamba_vision_T', 
                 feature_dim=512, d_state=16, pretrained=True, freeze_backbone=False):
        super().__init__()
        
        if not MAMBA_VISION_AVAILABLE:
            raise ImportError("請先安裝 mamba-vision: pip install mamba-vision")
        
        print(f"Initializing MambaVisionWithMambaFusion")
        print(f"  -> MambaVision 模型: {mamba_vision_model}")
        print(f"  -> Mamba 融合層 d_state: {d_state}")
        
        # MambaVision 特徵提取器
        self.encoder = create_mamba_vision_model(
            mamba_vision_model,
            pretrained=pretrained,
            num_classes=0
        )
        
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        model_dims = {
            'mamba_vision_T': 640, 'mamba_vision_T2': 640,
            'mamba_vision_S': 768, 'mamba_vision_B': 1024,
            'mamba_vision_L': 1024,
        }
        self.enc_dim = model_dims.get(mamba_vision_model, 640)
        
        self.branch_hints = {
            "side":   [1,1,1,1,0,1,0,0], "tip":    [1,1,0,1,0,0,0,0],
            "center": [0,0,0,0,1,0,1,1], "root":   [0,1,0,0,1,0,0,0],
            "whole":  [1,1,1,1,1,1,1,1],
        }
        
        self.branch_proj = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.enc_dim + num_classes, feature_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(feature_dim)
            ) for name in self.branch_hints.keys()
        })
        
        # 使用 Mamba 進行特徵融合
        self.mamba_fusion = Mamba(
            d_model=feature_dim,
            d_state=d_state,
            d_conv=4,
            expand=2
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes)
        )
    
    def _add_hint(self, flat: torch.Tensor, branch_name: str, device) -> torch.Tensor:
        hint = torch.tensor(self.branch_hints[branch_name], device=device, dtype=torch.float32)
        hint = hint.unsqueeze(0).expand(flat.size(0), -1)
        return self.branch_proj[branch_name](torch.cat([flat, hint], dim=1))
    
    def forward(self, x_whole, x_root, x_center, x_side, x_tip):
        features = [
            self._add_hint(self.encoder(x_whole),  "whole",  x_whole.device),
            self._add_hint(self.encoder(x_root),   "root",   x_root.device),
            self._add_hint(self.encoder(x_center), "center", x_center.device),
            self._add_hint(self.encoder(x_side),   "side",   x_side.device),
            self._add_hint(self.encoder(x_tip),    "tip",    x_tip.device)
        ]
        
        feature_sequence = torch.stack(features, dim=1)
        mamba_output = self.mamba_fusion(feature_sequence)
        fused_representation = mamba_output[:, -1, :]
        
        return self.classifier(fused_representation)
