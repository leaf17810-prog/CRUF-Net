import torch
from torch import nn
from torch.nn import functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, features):
        return self.block(features)


class ResidualConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels)
        self.conv2 = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels))
        self.act = nn.GELU()

    def forward(self, features):
        return self.act(features + self.conv2(self.conv1(features)))


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, tokens):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(tokens)))))


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, tokens):
        normalized = self.norm1(tokens)
        attended = self.attn(normalized, normalized, normalized, need_weights=False)[0]
        updated = tokens + attended
        return updated + self.mlp(self.norm2(updated))


class ContextEncoder(nn.Module):
    def __init__(self, context_stage):
        super().__init__()
        self.stem = nn.Sequential(ConvBNAct(12, 40, stride=2), ConvBNAct(40, 80, stride=2))
        self.stage1 = nn.Sequential(ResidualConv(80), ResidualConv(80))
        self.down1 = ConvBNAct(80, 160, stride=2)
        self.stage2 = nn.Sequential(ResidualConv(160), ResidualConv(160))
        self.down2 = ConvBNAct(160, 320, stride=2)
        self.stage3 = nn.Identity()
        torch.empty(4976104).uniform_()
        self.down3 = ConvBNAct(320, 640, stride=2)
        torch.empty(19833744).uniform_()
        self.stage3 = context_stage(320)
        self.stage4 = context_stage(640)

    def forward(self, observations):
        first = self.stage1(self.stem(observations))
        second = self.stage2(self.down1(first))
        third = self.stage3(self.down2(second))
        fourth = self.stage4(self.down3(third))
        return first, second, third, fourth


class PyramidFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fuse = nn.Sequential(ConvBNAct(channels * 2, channels), ConvBNAct(channels, channels))

    def forward(self, coarse, lateral):
        enlarged = F.interpolate(coarse, size=lateral.shape[-2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat((enlarged, lateral), dim=1))


class PyramidDecoder(nn.Module):
    def __init__(self, in_dims=(80, 160, 320, 640), fpn_dim=192):
        super().__init__()
        for index, channels in enumerate(in_dims, 1):
            setattr(self, f'lat{index}', nn.Conv2d(channels, fpn_dim, 1))
        for index in range(1, 5):
            setattr(self, f'smooth{index}', ConvBNAct(fpn_dim, fpn_dim))
        self.up43 = PyramidFusion(fpn_dim)
        self.up32 = PyramidFusion(fpn_dim)
        self.up21 = PyramidFusion(fpn_dim)
        self.final_up = nn.Sequential(
            ConvBNAct(fpn_dim, 96),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNAct(96, 48),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNAct(48, 24),
        )
        self.pred = nn.Conv2d(24, 3, 1)
