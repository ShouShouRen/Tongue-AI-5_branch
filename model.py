# model.py - 修正版本：支援多種backbone包含ResNet和Swin Transformer (支持384尺寸)
import torch
import torch.nn as nn
from timm import create_model
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """自注意力機制，用於融合不同分支的特徵 - 記憶體優化版本"""
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 添加維度 [batch_size, 1, d_model] 用於自注意力
        x = x.unsqueeze(1)
        
        q = self.q_linear(x).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 計算注意力分數
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 應用注意力權重
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        output = self.out(attn_output).squeeze(1)
        
        # 殘差連接和層正規化
        return self.layer_norm(x.squeeze(1) + output)

class FeatureBranch(nn.Module):
    """特徵提取分支 - 支援多種backbone（修正版本，支持Swin 384）"""
    def __init__(self, backbone_name='resnet50', feature_dim=512):
        super().__init__()
        
        # 支援的backbone配置 - 新增Swin 384支持
        self.backbone_configs = {
            'resnet50': {
                'model_name': 'resnet50',
                'output_dim': 2048,
                'use_features_only': True
            },
            'resnet34': {
                'model_name': 'resnet34', 
                'output_dim': 512,
                'use_features_only': True
            },
            'resnet18': {
                'model_name': 'resnet18',
                'output_dim': 512, 
                'use_features_only': True
            },
            'swin_base_patch4_window7_224': {
                'model_name': 'swin_base_patch4_window7_224',
                'output_dim': 1024,
                'use_features_only': True
            },
            'swin_base_patch4_window12_384': {
                'model_name': 'swin_base_patch4_window12_384',
                'output_dim': 1024,  # Swin Base 的特徵維度
                'use_features_only': True
            },
            'swin_tiny_patch4_window7_224': {
                'model_name': 'swin_tiny_patch4_window7_224',
                'output_dim': 768,
                'use_features_only': True
            },
            'swin_tiny_patch4_window12_384': {
                'model_name': 'swin_tiny_patch4_window12_384',
                'output_dim': 768,  # Swin Tiny 的特徵維度
                'use_features_only': True
            },
            'swin_small_patch4_window12_384': {
                'model_name': 'swin_small_patch4_window12_384',
                'output_dim': 768,  # Swin Small 的特徵維度
                'use_features_only': True
            },
            'efficientnet_b0': {
                'model_name': 'efficientnet_b0',
                'output_dim': 1280,
                'use_features_only': True
            },
            'vit_base_patch16_224': {
                'model_name': 'vit_base_patch16_224',
                'output_dim': 768,
                'use_features_only': False  # ViT不支援features_only
            }
        }
        
        if backbone_name not in self.backbone_configs:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Supported: {list(self.backbone_configs.keys())}")
        
        config = self.backbone_configs[backbone_name]
        self.backbone_name = backbone_name
        self.config = config
        
        # 創建backbone
        if config['use_features_only']:
            try:
                self.encoder = create_model(config['model_name'], pretrained=True, features_only=True)
                # 獲取實際的輸出維度
                encoder_dim = self.encoder.feature_info.channels()[-1]
                print(f"Successfully loaded {backbone_name} with features_only=True, encoder_dim: {encoder_dim}")
            except Exception as e:
                print(f"Warning: Failed to load {backbone_name} with features_only=True: {e}")
                print(f"Trying to load {backbone_name} without features_only...")
                # 如果features_only失敗，嘗試不用features_only
                self.encoder = create_model(config['model_name'], pretrained=True, num_classes=0)
                encoder_dim = config['output_dim']
                self.config['use_features_only'] = False
        else:
            # 對於不支援features_only的模型（如ViT）
            self.encoder = create_model(config['model_name'], pretrained=True, num_classes=0)
            encoder_dim = config['output_dim']
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        # 特徵投影層
        self.feature_proj = nn.Sequential(
            nn.Linear(encoder_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.output_dim = feature_dim
        print(f"Using backbone: {backbone_name}, encoder_dim: {encoder_dim}, feature_dim: {feature_dim}")
        
    def forward(self, x):
        if self.config['use_features_only']:
            # 使用features_only的模型（ResNet, Swin等）
            features = self.encoder(x)
            last_feat = features[-1]
            
            # 處理不同模型的輸出格式
            if 'swin' in self.backbone_name.lower():
                # Swin Transformer 特殊處理
                if last_feat.dim() == 3:
                    # [B, H*W, C] 格式 -> 使用全域平均池化
                    flattened = torch.mean(last_feat, dim=1)  # [B, H*W, C] -> [B, C]
                elif last_feat.dim() == 4:
                    # 4D張量，需要判斷格式
                    if last_feat.shape[-1] > last_feat.shape[1]:
                        # [B, H, W, C] -> [B, C, H, W]
                        last_feat = last_feat.permute(0, 3, 1, 2)
                    # 現在應該是 [B, C, H, W] 格式
                    pooled = self.pool(last_feat)
                    flattened = self.flatten(pooled)
                else:
                    raise ValueError(f"Unexpected Swin tensor dimension: {last_feat.dim()}, shape: {last_feat.shape}")
            else:
                # ResNet 等其他模型
                if last_feat.dim() == 4:
                    if last_feat.shape[-1] > last_feat.shape[1]:
                        # [B, H, W, C] -> [B, C, H, W]
                        last_feat = last_feat.permute(0, 3, 1, 2)
                    pooled = self.pool(last_feat)
                    flattened = self.flatten(pooled)
                elif last_feat.dim() == 3:
                    # 如果是3D，做平均池化
                    flattened = torch.mean(last_feat, dim=1)
                else:
                    raise ValueError(f"Unexpected tensor dimension: {last_feat.dim()}")
        else:
            # 不支援features_only的模型（如ViT）
            flattened = self.encoder(x)
        
        projected = self.feature_proj(flattened)
        return projected

class SignOrientedNetwork(nn.Module):
    """基於符號導向的多標籤檢測網絡 - 支援多種backbone"""
    def __init__(self, num_classes=8, backbone='resnet50', feature_dim=512):
        super().__init__()
        
        print(f"Initializing SignOrientedNetwork with backbone: {backbone}")
        
        # 三個分支：全舌、舌身、舌邊
        self.whole_branch = FeatureBranch(backbone, feature_dim)
        self.body_branch = FeatureBranch(backbone, feature_dim)
        self.edge_branch = FeatureBranch(backbone, feature_dim)
        
        # 顏色分類分支（論文中的4類顏色）
        self.color_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 4)  # White, Yellow, Black, Red
        )
        
        # 舌苔檢測分支
        self.fur_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # 有無舌苔
        )
        
        # 注意力機制用於融合特徵
        self.attention_body = SelfAttention(feature_dim * 2)
        self.attention_edge = SelfAttention(feature_dim * 2)
        
        # 區域特定的分類器
        self.body_specific = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 3)  # Crack, RedSpot, FurThick 主要在舌身
        )
        
        self.edge_specific = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 3)  # Pale, TipSideRed, Toothmark 主要在舌邊
        )
        
        # 全局分類器（處理FurYellow和Ecchymosis）
        self.global_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # FurYellow, Ecchymosis
        )
        
    def forward(self, x_whole, x_body, x_edge):
        # 提取各分支特徵
        f_whole = self.whole_branch(x_whole)
        f_body = self.body_branch(x_body)
        f_edge = self.edge_branch(x_edge)
        
        # 顏色和舌苔檢測（基於全舌）
        color_pred = self.color_classifier(f_whole)
        fur_pred = self.fur_classifier(f_whole)
        
        # 融合特徵
        body_combined = torch.cat([f_whole, f_body], dim=1)
        edge_combined = torch.cat([f_whole, f_edge], dim=1)
        
        # 應用注意力機制
        body_attended = self.attention_body(body_combined)
        edge_attended = self.attention_edge(edge_combined)
        
        # 區域特定預測
        body_pred = self.body_specific(body_attended)  # [Crack, RedSpot, FurThick]
        edge_pred = self.edge_specific(edge_attended)  # [Pale, TipSideRed, Toothmark]
        global_pred = self.global_classifier(f_whole)  # [FurYellow, Ecchymosis]
        
        # 組合所有預測
        # 按照標籤順序：['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis', 'Crack', 'Toothmark', 'FurThick', 'FurYellow']
        final_pred = torch.cat([
            edge_pred[:, 0:1],    # Pale
            edge_pred[:, 1:2],    # TipSideRed
            body_pred[:, 1:2],    # Spot (RedSpot)
            global_pred[:, 1:2],  # Ecchymosis
            body_pred[:, 0:1],    # Crack
            edge_pred[:, 2:3],    # Toothmark
            body_pred[:, 2:3],    # FurThick
            global_pred[:, 0:1]   # FurYellow
        ], dim=1)
        
        return {
            'final': final_pred,
            'color': color_pred,
            'fur': fur_pred
        }

# 測試不同backbone的函數
def test_backbone(backbone_name='resnet50', feature_dim=512, input_size=224):
    """測試不同backbone是否正常工作"""
    print(f"\n=== Testing {backbone_name} with input size {input_size} ===")
    
    try:
        model = SignOrientedNetwork(
            num_classes=8,
            backbone=backbone_name,
            feature_dim=feature_dim
        )
        
        # 創建測試輸入 - 根據模型要求調整尺寸
        batch_size = 2
        if '384' in backbone_name:
            input_size = 384
        x = torch.randn(batch_size, 3, input_size, input_size)
        
        # 前向傳播測試
        with torch.no_grad():
            outputs = model(x, x, x)
            
        print(f"✅ {backbone_name} works!")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {outputs['final'].shape}")
        print(f"   Color shape: {outputs['color'].shape}")
        print(f"   Fur shape: {outputs['fur'].shape}")
        
        # 檢查輸出是否包含NaN
        if torch.isnan(outputs['final']).any():
            print(f"⚠️  Warning: NaN detected in {backbone_name} output")
            
        return True
        
    except Exception as e:
        print(f"❌ {backbone_name} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 測試不同的backbone，特別是Swin 384
    backbones_to_test = [
        ('resnet50', 512, 224),
        ('resnet34', 512, 224), 
        ('swin_base_patch4_window7_224', 512, 224),
        ('swin_base_patch4_window12_384', 512, 384),  # 主要測試這個
        ('swin_tiny_patch4_window12_384', 512, 384),
        ('efficientnet_b0', 512, 224),
    ]
    
    print("Testing backbone compatibility for Swin 384...")
    
    for backbone, feature_dim, input_size in backbones_to_test:
        success = test_backbone(backbone, feature_dim, input_size)
        if not success:
            print(f"Skipping {backbone} due to errors")
        print("-" * 50)