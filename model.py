# model_zoo.py — 模型庫：存放多個可選的模型架構
# --- 已新增 Attention 融合模型 ---

import torch
import torch.nn as nn
from timm import create_model

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
    def __init__(self, num_classes, backbone='convnext_base', feature_dim=512):
        super().__init__()
        print(f"Initializing SimpleTimmModel with backbone: {backbone}")
        self.model = create_model(backbone, pretrained=True, num_classes=num_classes)

    def forward(self, x_whole, x_root, x_center, x_side, x_tip):
        return self.model(x_whole)

# -----------------------------------------------------------------------------
# 模型三：優化版！使用 Attention 融合的複雜網路 (SignOrientedAttentionNetwork)
# -----------------------------------------------------------------------------
class SignOrientedAttentionNetwork(nn.Module):
    def __init__(self, num_classes=8, backbone='swin_base_patch4_window7_224', feature_dim=512):
        super().__init__()
        print(f"Initializing SignOrientedAttentionNetwork with backbone: {backbone}")

        # --- 沿用舊版的特徵提取部分 ---
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
                nn.ReLU(inplace=True) # 移除 Dropout，讓 Attention 層來正則化
            ) for name in self.branch_hints.keys()
        })
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        # --- 核心優化：用 Transformer Attention Layer 取代 torch.cat ---
        self.fusion_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,               # 注意力頭的數量，8 是常用值
            dim_feedforward=feature_dim * 2, # 前饋網路維度
            dropout=0.2,           # Dropout 正則化
            activation='gelu',     # GELU 活化函數
            batch_first=True       # 輸入維度是 (Batch, Sequence, Features)
        )

        # --- 最終分類器 ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim), # 加入 LayerNorm 穩定訓練
            nn.Linear(feature_dim, num_classes)
        )

    # _encode 和 _add_hint 方法與舊版 SignOrientedNetwork 完全相同
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
        # 1. 像以前一樣，提取五個分支的特徵
        features = [
            self._add_hint(self._encode(x_whole),  "whole",  x_whole.device),
            self._add_hint(self._encode(x_root),   "root",   x_root.device),
            self._add_hint(self._encode(x_center), "center", x_center.device),
            self._add_hint(self._encode(x_side),   "side",   x_side.device),
            self._add_hint(self._encode(x_tip),    "tip",    x_tip.device)
        ]
        
        # 2. 將五個特徵堆疊成一個序列 (Batch, 5, feature_dim)
        # 這就像把五份報告交給一個委員會
        feature_sequence = torch.stack(features, dim=1)

        # 3. 讓 Transformer Attention 層 (委員會) 進行智慧融合
        # 委員會成員會互相討論，決定哪份報告最重要
        fused_sequence = self.fusion_layer(feature_sequence)

        # 4. 融合後，我們取第一個特徵 (對應 whole 分支) 作為代表
        # 這相當於聽取委員會主席 (看過全局資訊的 whole 分支) 的最終總結
        fused_representation = fused_sequence[:, 0, :]
        
        # 5. 將最終的融合特徵送入分類器
        return self.classifier(fused_representation)

