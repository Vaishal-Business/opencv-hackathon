"""stage1_models.py — E_id (identity encoder) and P128 (128² painter)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """Per-channel scale/shift conditioned on a global vector. Init to identity."""
    def __init__(self, cond_dim: int, ch: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden, ch * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.mlp(cond)
        g, b = gb.chunk(2, dim=1)
        return x * (1.0 + g[:, :, None, None]) + b[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, ch: int, cond_dim: int | None = None):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.BatchNorm2d(ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.film = FiLM(cond_dim, ch) if cond_dim else None

    def forward(self, x, cond=None):
        h = self.act(self.n1(self.c1(x)))
        if self.film is not None:
            h = self.film(h, cond)
        h = self.n2(self.c2(h))
        return self.act(x + h)


class IdEncoder(nn.Module):
    """256×256 RGB → 256-d identity vector."""
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3,   32, 3, 2, 1), nn.BatchNorm2d(32),  nn.LeakyReLU(0.1, True),
            nn.Conv2d(32,  64, 3, 2, 1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1, True),
            nn.Conv2d(128,192, 3, 2, 1), nn.BatchNorm2d(192), nn.LeakyReLU(0.1, True),
            nn.Conv2d(192,256, 3, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1, True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.stem(x)).flatten(1))


class Painter128(nn.Module):
    """
    U-Net-lite painter. Input: cat[G128 (20), W128 (3)] = 23 ch + cond (FiLM).
    Output: c0 (3, sigmoid), g0 (1, sigmoid), f128 (16, linear handoff).
    """
    def __init__(self, in_ch: int = 23, cond_dim: int = 364, f_dim: int = 16):
        super().__init__()
        self.enc0 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, True),
            ResBlock(64),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.LeakyReLU(0.1, True),
            ResBlock(96),
        )
        self.d2_pre = nn.Sequential(
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1, True),
        )
        self.d2_r1 = ResBlock(128, cond_dim)
        self.d2_r2 = ResBlock(128, cond_dim)
        self.d3_pre = nn.Sequential(
            nn.Conv2d(128, 160, 3, stride=2, padding=1), nn.BatchNorm2d(160), nn.LeakyReLU(0.1, True),
        )
        self.d3_r1 = ResBlock(160, cond_dim)
        self.d3_r2 = ResBlock(160, cond_dim)

        self.up3 = nn.Sequential(
            nn.Conv2d(160 + 128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1, True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128 + 96, 96, 3, padding=1), nn.BatchNorm2d(96), nn.LeakyReLU(0.1, True),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(96 + 64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, True),
        )
        self.head = nn.Conv2d(64, 3 + 1 + f_dim, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        h0 = self.enc0(x)                       # 128², 64
        h1 = self.down1(h0)                     #  64², 96
        h2 = self.d2_pre(h1)                    #  32², 128
        h2 = self.d2_r2(self.d2_r1(h2, cond), cond)
        h3 = self.d3_pre(h2)                    #  16², 160
        h3 = self.d3_r2(self.d3_r1(h3, cond), cond)

        u3 = F.interpolate(h3, scale_factor=2, mode="bilinear", align_corners=False)
        u3 = self.up3(torch.cat([u3, h2], 1))
        u2 = F.interpolate(u3, scale_factor=2, mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, h1], 1))
        u1 = F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, h0], 1))

        out = self.head(u1)
        c0 = torch.sigmoid(out[:, :3])
        g0 = torch.sigmoid(out[:, 3:4])
        f128 = out[:, 4:]
        return c0, g0, f128
