# model.py — 輕量五分支：共用 backbone + 分支 hint
import torch
import torch.nn as nn
from timm import create_model

class SignOrientedNetwork(nn.Module):
    def __init__(self, num_classes=8, backbone='swin_base_patch4_window7_224', feature_dim=512):
        super().__init__()
        print(f"Initializing Light SignOrientedNetwork with backbone: {backbone}")

        # 共用 backbone
        self.encoder = create_model(backbone, pretrained=True, features_only=True)
        self.enc_dim = self.encoder.feature_info.channels()[-1]

        # 分支提示 (固定 0/1 mask，依你指定)
        self.branch_hints = {
            "side":   [1,1,1,1,0,1,0,0],  # Pale, TipSideRed(=RedSpot對應你標籤第2位), Toothmark, Ecchymosis
            "tip":    [1,1,0,1,0,0,0,0],  # Pale, TipSideRed, Ecchymosis
            "center": [0,0,0,0,1,0,1,1],  # Crack, FurThick, FurYellow
            "root":   [0,1,0,0,1,0,0,0],  # TipSideRed(紅點/紅斑對應你Spot系統), Crack
            "whole":  [1,1,1,1,1,1,1,1],  # 全部
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
        feats = self.encoder(x)              # list of feature maps
        feat = feats[-1]                     # Swin: [B, H, W, C]；ResNet: [B, C, H, W]

        if feat.dim() == 4 and feat.shape[-1] == self.enc_dim:
            # Swin 輸出 [B,H,W,C] → [B,C,H,W]
            feat = feat.permute(0, 3, 1, 2).contiguous()

        if feat.dim() == 4:
            flat = self.flatten(self.pool(feat))       # [B,C]
        elif feat.dim() == 3:
            flat = torch.mean(feat, dim=1)             # [B,C]
        elif feat.dim() == 2:
            flat = feat
        else:
            raise ValueError(f"Unexpected feat shape: {feat.shape}")
        return flat  # [B, enc_dim]

    def _add_hint(self, flat: torch.Tensor, branch_name: str, device) -> torch.Tensor:
        hint = torch.tensor(self.branch_hints[branch_name], device=device, dtype=torch.float32)
        hint = hint.unsqueeze(0).expand(flat.size(0), -1)     # [B,num_classes]
        out = torch.cat([flat, hint], dim=1)                  # [B, enc_dim+num_classes]
        return self.branch_proj[branch_name](out)             # [B, feature_dim]

    def forward(self, x_whole, x_root, x_center, x_side, x_tip):
        f_whole  = self._add_hint(self._encode(x_whole),  "whole",  x_whole.device)
        f_root   = self._add_hint(self._encode(x_root),   "root",   x_root.device)
        f_center = self._add_hint(self._encode(x_center), "center", x_center.device)
        f_side   = self._add_hint(self._encode(x_side),   "side",   x_side.device)
        f_tip    = self._add_hint(self._encode(x_tip),    "tip",    x_tip.device)

        fused = torch.cat([f_whole, f_root, f_center, f_side, f_tip], dim=1)
        return self.classifier(fused)  # [B, num_classes]


if __name__ == "__main__":
    model = SignOrientedNetwork(num_classes=8, backbone='swin_base_patch4_window7_224', feature_dim=256)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    y = model(x, x, x, x, x)
    print("Output:", y.shape)
