"""stage1_losses.py — Charbonnier, LPIPS wrapper, f_id InfoNCE."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class Charbonnier(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, x, y):
        return torch.sqrt((x - y) ** 2 + self.eps2).mean()


def info_nce_id_loss(f_A: torch.Tensor, f_B: torch.Tensor, temperature: float = 0.1):
    """
    Within a batch, f_A[i] should be closest to f_B[i] among all f_B[j].
    Both (B, D); L2-normalized internally.
    """
    f_A = F.normalize(f_A, dim=-1)
    f_B = F.normalize(f_B, dim=-1)
    logits = f_A @ f_B.T / temperature
    labels = torch.arange(f_A.shape[0], device=f_A.device)
    return F.cross_entropy(logits, labels)
