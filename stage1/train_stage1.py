"""
train_stage1.py — Stage 1: train E_id + P128 on same-identity pairs.

Pass gates (Section 17):
  * I128 beats warp-only W128 by >= 20% LPIPS@128
  * mean g0 stays in [0.3, 0.9]  (no gate collapse)

Run:
    pip install -q lpips tensorboard
    python train_stage1.py \
        --meta_dir /content/drive/MyDrive/galnp/processed/meta \
        --crop_dir /content/drive/MyDrive/galnp/processed/crops \
        --basis    /content/drive/MyDrive/expression/expression_basis.npz \
        --out_dir  /content/stage1_out \
        --epochs 200 --batch 8 --lr 2e-4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import lpips

from stage1_data   import Stage1Dataset
from stage1_models import IdEncoder, Painter128
from stage1_losses import Charbonnier, info_nce_id_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--meta_dir", required=True)
    p.add_argument("--crop_dir", required=True)
    p.add_argument("--basis",    required=True)
    p.add_argument("--out_dir",  default="stage1_out")
    p.add_argument("--epochs",    type=int,   default=200)
    p.add_argument("--batch",     type=int,   default=8)
    p.add_argument("--lr",        type=float, default=2e-4)
    p.add_argument("--val_frac",  type=float, default=0.15)
    p.add_argument("--val_every", type=int,   default=5)
    p.add_argument("--log_every", type=int,   default=20)
    p.add_argument("--vis_every", type=int,   default=10)
    p.add_argument("--w_charb",   type=float, default=1.0)
    p.add_argument("--w_lpips",   type=float, default=0.2)
    p.add_argument("--w_id",      type=float, default=0.1)
    p.add_argument("--f_dim",     type=int,   default=256)
    p.add_argument("--f128_dim",  type=int,   default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed",      type=int,   default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------

def split_pairs(n: int, val_frac: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(round(n * val_frac)))
    return perm[n_val:], perm[:n_val]


def collate(batch_list):
    out = {}
    for k in batch_list[0]:
        if isinstance(batch_list[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch_list])
        else:
            out[k] = [b[k] for b in batch_list]
    return out


def make_cond(e_B, e_A, gaze, f_id):
    return torch.cat([e_B, e_B - e_A, gaze, f_id], dim=1)


@torch.no_grad()
def evaluate(model_id, painter, loader, lpips_fn, charb, device):
    model_id.eval(); painter.eval()
    n = 0
    lp_ours = lp_warp = chr_ours = g_sum = 0.0
    for b in loader:
        A     = b["A_rgb"].to(device)
        B     = b["B_rgb"].to(device)
        W512  = b["W512"].to(device)
        G128  = b["G128"].to(device)
        e_A   = b["e_A"].to(device)
        e_B   = b["e_B"].to(device)
        Bn    = A.shape[0]

        A_256 = F.avg_pool2d(A, 2)
        B_128 = F.avg_pool2d(B, 4)
        W_128 = F.avg_pool2d(W512, 4)

        f_id = model_id(A_256)
        gaze = torch.zeros(Bn, 4, device=device)
        c0, g0, _ = painter(torch.cat([G128, W_128], 1), make_cond(e_B, e_A, gaze, f_id))
        I128 = g0 * W_128 + (1 - g0) * c0

        lp_ours  += lpips_fn(I128, B_128, normalize=True).mean().item() * Bn
        lp_warp  += lpips_fn(W_128, B_128, normalize=True).mean().item() * Bn
        chr_ours += charb(I128, B_128).item() * Bn
        g_sum    += g0.mean().item() * Bn
        n += Bn
    return {
        "lpips_ours": lp_ours / n,
        "lpips_warp": lp_warp / n,
        "charb_ours": chr_ours / n,
        "mean_g":     g_sum / max(1, n),
    }


def _tb_image_strip(A_512, B_512, W_512, I128_up, g0_up):
    """Build a single [3, H, 5*W] strip for TensorBoard logging."""
    def _to3(x):
        return x.detach().clamp(0, 1).cpu()
    A  = _to3(A_512[0])
    B  = _to3(B_512[0])
    W  = _to3(W_512[0])
    I  = _to3(I128_up[0])
    g  = _to3(g0_up[0].expand(3, -1, -1))
    return torch.cat([A, B, W, I, g], dim=2)


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "tb"))

    ds = Stage1Dataset(args.meta_dir, args.crop_dir, args.basis)
    tr_idx, va_idx = split_pairs(len(ds), args.val_frac, args.seed)
    tr = Subset(ds, tr_idx); va = Subset(ds, va_idx)

    bs = min(args.batch, len(tr))
    tr_loader = DataLoader(tr, batch_size=bs, shuffle=True,
                           num_workers=args.num_workers, collate_fn=collate, drop_last=True)
    va_loader = DataLoader(va, batch_size=min(bs, max(1, len(va))), shuffle=False,
                           num_workers=args.num_workers, collate_fn=collate)
    print(f"[stage1] {len(tr)} train pairs, {len(va)} val pairs, batch={bs}")

    model_id = IdEncoder(out_dim=args.f_dim).to(device)
    cond_dim = 52 + 52 + 4 + args.f_dim       # 364
    painter  = Painter128(in_ch=23, cond_dim=cond_dim, f_dim=args.f128_dim).to(device)
    print(f"[stage1] E_id  params = {sum(p.numel() for p in model_id.parameters())/1e6:.2f} M")
    print(f"[stage1] P128  params = {sum(p.numel() for p in painter.parameters())/1e6:.2f} M")

    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False
    charb = Charbonnier()

    params = list(model_id.parameters()) + list(painter.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    total_steps = args.epochs * max(1, len(tr_loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=args.lr / 10)

    step = 0; t0 = time.time(); best = float("inf")

    for epoch in range(args.epochs):
        model_id.train(); painter.train()
        for b in tr_loader:
            A     = b["A_rgb"].to(device)
            B     = b["B_rgb"].to(device)
            W512  = b["W512"].to(device)
            G128  = b["G128"].to(device)
            e_A   = b["e_A"].to(device)
            e_B   = b["e_B"].to(device)
            Bn    = A.shape[0]

            A_256 = F.avg_pool2d(A, 2)
            B_128 = F.avg_pool2d(B, 4)
            W_128 = F.avg_pool2d(W512, 4)

            f_id_A = model_id(A_256)
            f_id_B = model_id(F.avg_pool2d(B, 2))
            gaze = torch.zeros(Bn, 4, device=device)

            c0, g0, _ = painter(torch.cat([G128, W_128], 1),
                                make_cond(e_B, e_A, gaze, f_id_A))
            I128 = g0 * W_128 + (1 - g0) * c0

            l_charb = charb(I128, B_128)
            l_lpips = lpips_fn(I128, B_128, normalize=True).mean()
            l_id    = info_nce_id_loss(f_id_A, f_id_B)
            loss = args.w_charb * l_charb + args.w_lpips * l_lpips + args.w_id * l_id

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1

            if step % args.log_every == 0:
                dt = time.time() - t0
                print(f"[ep {epoch:03d} st {step:06d}] "
                      f"charb={l_charb.item():.4f} lpips={l_lpips.item():.4f} "
                      f"id={l_id.item():.4f} g0={g0.mean().item():.3f} ({dt:.1f}s)")
                for k, v in [("train/charb", l_charb.item()),
                             ("train/lpips", l_lpips.item()),
                             ("train/id",    l_id.item()),
                             ("train/g0",    g0.mean().item()),
                             ("train/lr",    sched.get_last_lr()[0])]:
                    writer.add_scalar(k, v, step)
                t0 = time.time()

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            stats = evaluate(model_id, painter, va_loader, lpips_fn, charb, device)
            ratio = stats["lpips_ours"] / max(1e-6, stats["lpips_warp"])
            print(f"  [VAL ep {epoch:03d}] lpips_ours={stats['lpips_ours']:.4f} "
                  f"lpips_warp={stats['lpips_warp']:.4f} "
                  f"improvement={(1 - ratio) * 100:.1f}% mean_g={stats['mean_g']:.3f}")
            for k, v in [("val/lpips_ours", stats["lpips_ours"]),
                         ("val/lpips_warp", stats["lpips_warp"]),
                         ("val/charb_ours", stats["charb_ours"]),
                         ("val/mean_g",     stats["mean_g"])]:
                writer.add_scalar(k, v, step)

            if stats["lpips_ours"] < best:
                best = stats["lpips_ours"]
                torch.save({
                    "model_id": model_id.state_dict(),
                    "painter":  painter.state_dict(),
                    "opt":      opt.state_dict(),
                    "step":     step,
                    "args":     vars(args),
                    "stats":    stats,
                }, out_dir / "best.pt")
                print(f"  [VAL] new best -> {out_dir / 'best.pt'}")

        if (epoch + 1) % args.vis_every == 0:
            # Visual panel on one val batch
            model_id.eval(); painter.eval()
            with torch.no_grad():
                b = next(iter(va_loader))
                A = b["A_rgb"].to(device); B = b["B_rgb"].to(device)
                W512 = b["W512"].to(device); G128 = b["G128"].to(device)
                e_A = b["e_A"].to(device); e_B = b["e_B"].to(device)
                A_256 = F.avg_pool2d(A, 2)
                W_128 = F.avg_pool2d(W512, 4)
                f_id = model_id(A_256)
                gaze = torch.zeros(A.shape[0], 4, device=device)
                c0, g0, _ = painter(torch.cat([G128, W_128], 1),
                                    make_cond(e_B, e_A, gaze, f_id))
                I128 = g0 * W_128 + (1 - g0) * c0
                I128_up = F.interpolate(I128, scale_factor=4, mode="bilinear",
                                        align_corners=False)
                g0_up = F.interpolate(g0, scale_factor=4, mode="bilinear",
                                      align_corners=False)
                strip = _tb_image_strip(A, B, W512, I128_up, g0_up)
                writer.add_image("val/strip (A | B | W | I | g0)", strip, step)
            model_id.train(); painter.train()

        if (epoch + 1) % 20 == 0:
            torch.save({
                "model_id": model_id.state_dict(),
                "painter":  painter.state_dict(),
                "opt":      opt.state_dict(),
                "step":     step,
                "args":     vars(args),
            }, out_dir / f"ckpt_ep{epoch + 1:03d}.pt")

    writer.close()
    print("[stage1] done.")


if __name__ == "__main__":
    main()