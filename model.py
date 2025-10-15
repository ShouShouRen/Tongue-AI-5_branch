# model.py — Five-branch with whole→(color,fur) connections, concat→self-attn
# Optimized: single-pass encoder for 5 branches; exposes .encoder
import torch
import torch.nn as nn

try:
    from timm import create_model as timm_create_model
except Exception:
    timm_create_model = None
from torchvision import models as tv_models


# ------------------------------
# Backbone wrapper: always returns [B,C,H,W]
# ------------------------------
class FeatureEncoder(nn.Module):
    def __init__(self, backbone_name="swin_base_patch4_window7_224", pretrained=True):
        super().__init__()
        self.out_channels, self.kind = None, "none"

        if timm_create_model is not None:
            try:
                m = timm_create_model(backbone_name, pretrained=pretrained, features_only=True)
                self.impl = m
                self.out_channels = m.feature_info.channels()[-1]
                self.kind = "timm_features_only"
            except Exception:
                self.impl = None
        else:
            self.impl = None

        if self.impl is None:
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            m = tv_models.resnet50(weights=weights)
            self.impl = nn.Sequential(*list(m.children())[:-2])  # [B,2048,H,W]
            self.out_channels = 2048
            self.kind = "tv_resnet50"

    def forward(self, x):
        if self.kind == "timm_features_only":
            feat = self.impl(x)[-1]  # [B,H,W,C] or [B,C,H,W]
            if feat.shape[-1] == self.out_channels:  # channel-last -> channel-first
                feat = feat.permute(0, 3, 1, 2).contiguous()
            return feat
        return self.impl(x)


# ------------------------------
# Small MLP and Fusion
# ------------------------------
class MLP(nn.Module):
    def __init__(self, dim, hidden=None, out=None, p=0.1, act=True):
        super().__init__()
        hidden = hidden or dim * 2
        out = out or dim
        layers = [nn.Linear(dim, hidden)]
        if act: layers.append(nn.ReLU(inplace=True))
        layers += [nn.Dropout(p), nn.Linear(hidden, out)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class Fusion(nn.Module):
    """Fuse [f_color, f_fur, f_region] with self-attention → region vector."""
    def __init__(self, d, heads=4, ff=4):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=heads,
                                         dim_feedforward=ff*d,
                                         batch_first=True, norm_first=True)
        self.te = nn.TransformerEncoder(enc, num_layers=1)
        self.out = nn.Linear(d, d)
    def forward(self, f_color, f_fur, f_i):        # [B,d] ×3
        x = torch.stack([f_color, f_fur, f_i], 1)  # [B,3,d]
        y = self.te(x)[:, -1, :]                   # region slot
        return self.out(y)                         # [B,d]


# ------------------------------
# Main network
# ------------------------------
class SignOrientedNetwork(nn.Module):
    """
    五分支：whole/root/center/side/tip。
    whole 產生 f_color, f_fur；各區域與其 concat→self-attn 融合後走各自 head；
    依映射聚合為 8 類 logits。Encoder 單次前向處理五區域（提速）。
    """
    def __init__(self,
                 num_classes: int = 8,
                 backbone: str = "swin_base_patch4_window7_224",  # 可改 'convnext_base'
                 feature_dim: int = 512,                           # ConvNeXt-B 建議 768
                 dropout: float = 0.1,
                 attn_heads: int = 4):
        super().__init__()
        self.num_classes = num_classes

        # expose .encoder for freezing/unfreezing
        self.encoder = FeatureEncoder(backbone, pretrained=True)
        enc_dim = self.encoder.out_channels
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(enc_dim, feature_dim)

        # whole-derived semantic vectors
        self.color_vec = MLP(feature_dim, hidden=feature_dim*2, out=feature_dim, p=dropout)
        self.fur_vec   = MLP(feature_dim, hidden=feature_dim*2, out=feature_dim, p=dropout)

        # region fusion
        self.fuse = nn.ModuleDict({
            "root":   Fusion(feature_dim, heads=attn_heads),
            "center": Fusion(feature_dim, heads=attn_heads),
            "side":   Fusion(feature_dim, heads=attn_heads),
            "tip":    Fusion(feature_dim, heads=attn_heads),
        })

        # branch → classes mapping
        self.branch_map = {
            "whole":  [0,1,2,3,4,5,6,7],
            "root":   [2,4],          # Spot, Crack
            "center": [7,4,6],        # FurYellow, Crack, FurThick
            "side":   [0,2,5,3],      # TonguePale, Spot, Toothmark, Ecchymosis
            "tip":    [0,1,3],        # TonguePale, TipSideRed, Ecchymosis
        }

        # heads
        def head(nout):
            return nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, nout),
            )
        self.head_whole  = head(len(self.branch_map["whole"]))
        self.head_root   = head(len(self.branch_map["root"]))
        self.head_center = head(len(self.branch_map["center"]))
        self.head_side   = head(len(self.branch_map["side"]))
        self.head_tip    = head(len(self.branch_map["tip"]))

        # learnable gains (0..1 via sigmoid at aggregation)
        self.gain = nn.ParameterDict({
            "whole":  nn.Parameter(torch.zeros(1)),
            "root":   nn.Parameter(torch.zeros(1)),
            "center": nn.Parameter(torch.zeros(1)),
            "side":   nn.Parameter(torch.zeros(1)),
            "tip":    nn.Parameter(torch.zeros(1)),
        })

    # ---- helpers
    def _encode_vec(self, x):                 # [B,3,H,W] -> [B,D]
        f = self.encoder(x)                   # [B,C,H,W]
        v = self.gap(f).flatten(1)            # [B,C]
        return self.proj(v)                   # [B,D]

    def _encode_vec_5(self, xs):              # xs: list of 5 tensors [B,3,H,W]
        # single-pass encoder for speed
        x = torch.cat(xs, dim=0)              # [5B,3,H,W]
        f = self.encoder(x)                   # [5B,C,H,W]
        v = self.gap(f).flatten(1)            # [5B,C]
        v = self.proj(v)                      # [5B,D]
        B = xs[0].size(0)
        return v[0:B], v[B:2*B], v[2*B:3*B], v[3*B:4*B], v[4*B:5*B]

    # ---- forward
    def forward(self, x_whole, x_root, x_center, x_side, x_tip):
        # encode five regions in one pass
        f_whole, f_root, f_center, f_side, f_tip = self._encode_vec_5(
            [x_whole, x_root, x_center, x_side, x_tip]
        )

        # whole-derived semantics
        f_color = self.color_vec(f_whole)
        f_fur   = self.fur_vec(f_whole)

        # fusion per region
        z_root   = self.fuse["root"](f_color, f_fur, f_root)
        z_center = self.fuse["center"](f_color, f_fur, f_center)
        z_side   = self.fuse["side"](f_color, f_fur, f_side)
        z_tip    = self.fuse["tip"](f_color, f_fur, f_tip)

        # branch logits
        log_whole  = self.head_whole(f_whole)    # [B,8]
        log_root   = self.head_root(z_root)      # [B,2]
        log_center = self.head_center(z_center)  # [B,3]
        log_side   = self.head_side(z_side)      # [B,4]
        log_tip    = self.head_tip(z_tip)        # [B,3]

        # aggregate to final 8 classes
        B = f_whole.size(0)
        dev = f_whole.device
        final = torch.zeros(B, self.num_classes, device=dev, dtype=log_whole.dtype)

        final = final + torch.sigmoid(self.gain["whole"]) * log_whole

        def add_branch(name, logits):
            idx = torch.tensor(self.branch_map[name], device=dev, dtype=torch.long)
            final.index_add_(1, idx, torch.sigmoid(self.gain[name]) * logits)

        add_branch("root",   log_root)
        add_branch("center", log_center)
        add_branch("side",   log_side)
        add_branch("tip",    log_tip)

        return final  # [B,8]
