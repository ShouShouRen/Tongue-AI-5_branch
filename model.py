# model_zoo.py — 模型庫：存放多個可選的模型架構
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

        # 共用 backbone
        self.encoder = create_model(backbone, pretrained=True, features_only=True)
        self.enc_dim = self.encoder.feature_info.channels()[-1]

        # 分支提示 (固定 0/1 mask)
        self.branch_hints = {
            "side":   [1,1,1,1,0,1,0,0],
            "tip":    [1,1,0,1,0,0,0,0],
            "center": [0,0,0,0,1,0,1,1],
            "root":   [0,1,0,0,1,0,0,0],
            "whole":  [1,1,1,1,1,1,1,1],
        }

        # 每分支一個小投影
        self.branch_proj = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.enc_dim + num_classes, feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            )
            for name in self.branch_hints.keys()
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
        feats = self.encoder(x)
        feat = feats[-1]

        if feat.dim() == 4 and feat.shape[-1] == self.enc_dim:
            feat = feat.permute(0, 3, 1, 2).contiguous()

        if feat.dim() == 4:
            flat = self.flatten(self.pool(feat))
        else:
            flat = feat
        return flat

    def _add_hint(self, flat: torch.Tensor, branch_name: str, device) -> torch.Tensor:
        hint = torch.tensor(self.branch_hints[branch_name], device=device, dtype=torch.float32)
        hint = hint.unsqueeze(0).expand(flat.size(0), -1)
        out = torch.cat([flat, hint], dim=1)
        return self.branch_proj[branch_name](out)

    def forward(self, x_whole, x_root, x_center, x_side, x_tip):
        f_whole  = self._add_hint(self._encode(x_whole),  "whole",  x_whole.device)
        f_root   = self._add_hint(self._encode(x_root),   "root",   x_root.device)
        f_center = self._add_hint(self._encode(x_center), "center", x_center.device)
        f_side   = self._add_hint(self._encode(x_side),   "side",   x_side.device)
        f_tip    = self._add_hint(self._encode(x_tip),    "tip",    x_tip.device)

        fused = torch.cat([f_whole, f_root, f_center, f_side, f_tip], dim=1)
        return self.classifier(fused)

# -----------------------------------------------------------------------------
# 模型二：簡潔的單一輸入模型 (SimpleTimmModel)
# -----------------------------------------------------------------------------
class SimpleTimmModel(nn.Module):
    def __init__(self, num_classes, backbone='convnext_base', feature_dim=512):
        super().__init__()
        print(f"Initializing SimpleTimmModel with backbone: {backbone}")
        # 直接使用 timm 建立模型，並替換掉最後的分類頭
        self.model = create_model(backbone, pretrained=True, num_classes=num_classes)

    def forward(self, x_whole, x_root, x_center, x_side, x_tip):
        # 這個模型架構很簡單，只使用完整的舌頭影像
        # 我們仍然接收全部五個輸入，以和 Dataset 相容，但只使用第一個
        return self.model(x_whole)# model_zoo.py — 模型庫：存放多個可選的模型架構
