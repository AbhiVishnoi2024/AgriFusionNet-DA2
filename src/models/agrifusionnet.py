from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    EfficientNet_B3_Weights,
    Swin_T_Weights,
    efficientnet_b3,
    swin_t,
)


class CNNBranch(nn.Module):
    """EfficientNet-B3 feature extractor."""

    def __init__(self, embed_dim: int = 256, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b3(weights=weights)
        self.features = backbone.features
        self.proj = nn.Conv2d(1536, embed_dim, kernel_size=1)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            self.features.eval()
            with torch.no_grad():
                x = self.features(x)
        else:
            x = self.features(x)
        return self.proj(x)


class TransformerBranch(nn.Module):
    """Swin-Tiny feature extractor."""

    def __init__(self, embed_dim: int = 256, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        backbone = swin_t(weights=weights)
        self.features = backbone.features
        self.proj = nn.Conv2d(768, embed_dim, kernel_size=1)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            self.features.eval()
            with torch.no_grad():
                x = self.features(x)
        else:
            x = self.features(x)
        # Torchvision Swin features are [B, H, W, C].
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.proj(x)


class CrossAttentionFusion(nn.Module):
    """Learned cross-attention between CNN and Transformer feature tokens."""

    def __init__(self, dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.norm_c = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        self.cnn_to_transformer = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.transformer_to_cnn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, cnn: torch.Tensor, transformer: torch.Tensor) -> torch.Tensor:
        target_size = (7, 7)
        cnn = F.adaptive_avg_pool2d(cnn, target_size)
        transformer = F.adaptive_avg_pool2d(transformer, target_size)

        cnn_tokens = cnn.flatten(2).transpose(1, 2)  # [B, 49, D]
        transformer_tokens = transformer.flatten(2).transpose(1, 2)

        c = self.norm_c(cnn_tokens)
        t = self.norm_t(transformer_tokens)

        c2, _ = self.cnn_to_transformer(c, t, t)
        t2, _ = self.transformer_to_cnn(t, c, c)
        fused = cnn_tokens + 0.5 * c2 + 0.5 * t2
        fused = fused + self.ffn(self.out_norm(fused))
        return fused


class AgriFusionNet(nn.Module):
    """DA2 version: CNN + Swin + cross-attention + classification head."""

    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 256,
        num_heads: int = 8,
        pretrained: bool = True,
        freeze_backbones: bool = True,
    ):
        super().__init__()
        self.cnn = CNNBranch(embed_dim, pretrained, freeze_backbones)
        self.transformer = TransformerBranch(embed_dim, pretrained, freeze_backbones)
        self.fusion = CrossAttentionFusion(embed_dim, num_heads)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        cnn_features = self.cnn(x)
        transformer_features = self.transformer(x)
        fused = self.fusion(cnn_features, transformer_features)
        pooled = fused.mean(dim=1)
        logits = self.classifier(pooled)

        if return_features:
            cnn_features.retain_grad()
            return logits, {"cnn_features": cnn_features, "fused_tokens": fused}
        return logits


if __name__ == "__main__":
    model = AgriFusionNet(pretrained=False, freeze_backbones=True)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("Input:", x.shape)
    print("Output:", y.shape)
