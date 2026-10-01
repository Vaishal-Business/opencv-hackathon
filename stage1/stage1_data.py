"""
stage1_data.py — Stage 1 same-identity pair dataset.

Per sample yields:
    A_rgb     (3,512,512) float [0,1]   — A's aligned crop
    B_rgb     (3,512,512) float [0,1]   — B's crop, realigned to A's crop frame
    W512      (3,512,512) float [0,1]   — A warped to B's mesh shape
    G128      (20,128,128) float        — G-buffer for P128
    mask128   (128,128)    float 0/1    — mesh coverage
    A_lm, B_lm (478,3) float32          — landmarks in crop pixel coords
    e_A, e_B   (52,)   float32          — blendshape scores
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial import Delaunay
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Discovery (subject-prefix match; handles 001_03_neutral ↔ 001_08_smiling)
# ---------------------------------------------------------------------------

def _norm_id(raw) -> str:
    s = str(raw).strip()
    if s.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        s = Path(s).stem
    return "_".join(str(int(p)) if p.isdigit() else p for p in s.split("_"))


def _find_crop(crop_dir: Path, stem: str, expr: str):
    for ext in (".jpg", ".jpeg", ".png"):
        c = crop_dir / f"{stem}_{expr}{ext}"
        if c.exists():
            return c
    hits = sorted(crop_dir.glob(f"{stem}_{expr}.*"))
    return hits[0] if hits else None


def discover_pairs(meta_dir: Path, crop_dir: Path) -> List[Dict]:
    meta_dir, crop_dir = Path(meta_dir), Path(crop_dir)
    pairs: List[Dict] = []
    for npath in sorted(meta_dir.glob("*_neutral.npz")):
        nstem = npath.stem[:-len("_neutral")]
        subject = nstem.split("_")[0]
        subj_norm = _norm_id(subject)
        spath = next(
            (p for p in sorted(meta_dir.glob("*_smiling.npz"))
             if _norm_id(p.stem[:-len("_smiling")].split("_")[0]) == subj_norm),
            None,
        )
        if spath is None:
            continue
        sstem = spath.stem[:-len("_smiling")]
        n_img = _find_crop(crop_dir, nstem, "neutral")
        s_img = _find_crop(crop_dir, sstem, "smiling")
        if n_img is None or s_img is None:
            continue
        pairs.append({
            "subject": subject,
            "A_stem": nstem, "B_stem": sstem,
            "A_npz": npath, "B_npz": spath,
            "A_img": n_img, "B_img": s_img,
        })
    return pairs


def _read_rgb(path, affine_2x3: np.ndarray | None = None, out_size: int = 512) -> np.ndarray:
    """Read image, optionally warp by a 2×3 affine to out_size×out_size."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    if affine_2x3 is not None and not np.allclose(
        affine_2x3, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), atol=1e-4
    ):
        bgr = cv2.warpAffine(
            bgr, affine_2x3, (out_size, out_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Umeyama similarity (temporary B→A realignment; goes to ~identity once you
# switch prepare.py to the shared-alignment version)
# ---------------------------------------------------------------------------

def _umeyama_2d(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    n = src.shape[0]
    sm, dm = src.mean(0), dst.mean(0)
    sc, dc = src - sm, dst - dm
    cov = dc.T @ sc / n
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1.0
    rot = u @ np.diag(d) @ vt
    var = (sc ** 2).sum() / n
    scale = float((s * d).sum() / max(var, 1e-9))
    t = dm - scale * rot @ sm
    return np.hstack([scale * rot, t[:, None]]).astype(np.float32)


def _apply_affine_2d(pts: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Apply (2,3) affine to (N,3) points; scale z by sqrt(|det A_2x2|)."""
    xy = pts[:, :2] @ A[:, :2].T + A[:, 2]
    scale = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
    z = pts[:, 2:3] * scale
    return np.concatenate([xy, z], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Correspondence / warping
# ---------------------------------------------------------------------------

def delaunay_simplices(xy: np.ndarray) -> np.ndarray:
    return Delaunay(xy).simplices.astype(np.int32)


def build_sampling_grid(
    src_xy: np.ndarray, dst_xy: np.ndarray, simplices: np.ndarray,
    H: int, W: int,
) -> torch.Tensor:
    """
    F.grid_sample grid (H,W,2) mapping destination pixel -> source coord.
    Uncovered pixels get identity.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = xx.copy()
    map_y = yy.copy()

    for tri in simplices:
        d = dst_xy[tri]; s = src_xy[tri]
        x0 = max(0, int(np.floor(d[:, 0].min())))
        x1 = min(W, int(np.ceil(d[:, 0].max())) + 1)
        y0 = max(0, int(np.floor(d[:, 1].min())))
        y1 = min(H, int(np.ceil(d[:, 1].max())) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float32),
                             np.arange(y0, y1, dtype=np.float32))
        ax, ay = d[1, 0] - d[0, 0], d[1, 1] - d[0, 1]
        bx, by = d[2, 0] - d[0, 0], d[2, 1] - d[0, 1]
        den = ax * by - bx * ay
        if abs(den) < 1e-8:
            continue
        px, py = xs - d[0, 0], ys - d[0, 1]
        b1 = (px * by - py * bx) / den
        b2 = (ax * py - ay * px) / den
        b0 = 1.0 - b1 - b2
        inside = (b0 >= -1e-6) & (b1 >= -1e-6) & (b2 >= -1e-6)
        if not inside.any():
            continue
        xi = xs[inside].astype(np.int32); yi = ys[inside].astype(np.int32)
        b0i, b1i, b2i = b0[inside], b1[inside], b2[inside]
        map_x[yi, xi] = b0i * s[0, 0] + b1i * s[1, 0] + b2i * s[2, 0]
        map_y[yi, xi] = b0i * s[0, 1] + b1i * s[1, 1] + b2i * s[2, 1]

    gx = (map_x / (W - 1)) * 2.0 - 1.0
    gy = (map_y / (H - 1)) * 2.0 - 1.0
    return torch.from_numpy(np.stack([gx, gy], -1).astype(np.float32))


def warp_image(img: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    B = img.shape[0]
    g = grid.unsqueeze(0).expand(B, -1, -1, -1)
    return torch.nn.functional.grid_sample(
        img, g, mode="bilinear", padding_mode="zeros", align_corners=True,
    )


# ---------------------------------------------------------------------------
# G-buffer (20 ch)
# ---------------------------------------------------------------------------

_LIP_UPPER = [61,185,40,39,37,0,267,269,270,409,291]
_LIP_LOWER = [61,146,91,181,84,17,314,405,321,375,291]
_MOUTH_IN  = [78,95,88,178,87,14,317,402,318,324,308]
_EYE_L     = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,130,25,110,24,23,22,26,112]
_EYE_R     = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
_BROW_L    = [70,63,105,66,107,55,65,52,53,46]
_BROW_R    = [336,296,334,293,300,285,295,282,283,276]
_NOSE      = [1,2,3,4,5,6,45,51,48,115,131,134,102,49,220,305,275,440,281,363,360,279,294,278,344,439,122,196,3,236,198,420]


def assign_landmark_regions(N: int = 478) -> np.ndarray:
    r = np.zeros(N, dtype=np.int32)
    for i in _LIP_UPPER: r[i] = 1
    for i in _LIP_LOWER: r[i] = 2
    for i in _MOUTH_IN:  r[i] = 3
    for i in _EYE_L:     r[i] = 4
    for i in _EYE_R:     r[i] = 5
    for i in _BROW_L + _BROW_R: r[i] = 6
    for i in _NOSE:      r[i] = 7
    return r


def _vertex_normals(V: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    N = np.zeros_like(V, dtype=np.float64)
    for tri in simplices:
        p = V[tri]
        N[tri] += np.cross(p[1] - p[0], p[2] - p[0])
    lens = np.linalg.norm(N, axis=1, keepdims=True)
    return (N / np.where(lens > 1e-9, lens, 1.0)).astype(np.float32)


def build_gbuffer(L_A: np.ndarray, V_out: np.ndarray, simplices: np.ndarray,
                  H: int = 128, W: int = 128, crop_size: int = 512):
    """Rasterize landmark mesh to a 20-channel G-buffer at (H, W)."""
    scale = float(H) / float(crop_size)
    src = L_A[:, :2].astype(np.float64) * scale
    dst = V_out[:, :2].astype(np.float64) * scale
    vn = _vertex_normals(V_out, simplices)
    region = assign_landmark_regions(len(L_A))
    uv = (L_A[:, :2] / float(crop_size)).astype(np.float64)

    G = np.zeros((20, H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.float32)
    depth = np.zeros((H, W), dtype=np.float32)
    nrm = np.zeros((3, H, W), dtype=np.float32)
    reg = np.zeros((H, W), dtype=np.int64)
    uvm = np.zeros((2, H, W), dtype=np.float32)
    corrm = np.zeros((2, H, W), dtype=np.float32)

    for tri in simplices:
        d = dst[tri]; s = src[tri]
        z = V_out[tri, 2] * scale
        nt = vn[tri]; uvt = uv[tri]; rt = region[tri]
        rt_major = int(np.bincount(rt).argmax())

        x0 = max(0, int(np.floor(d[:, 0].min())))
        x1 = min(W, int(np.ceil(d[:, 0].max())) + 1)
        y0 = max(0, int(np.floor(d[:, 1].min())))
        y1 = min(H, int(np.ceil(d[:, 1].max())) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float32),
                             np.arange(y0, y1, dtype=np.float32))
        ax, ay = d[1, 0] - d[0, 0], d[1, 1] - d[0, 1]
        bx, by = d[2, 0] - d[0, 0], d[2, 1] - d[0, 1]
        den = ax * by - bx * ay
        if abs(den) < 1e-8:
            continue
        px, py = xs - d[0, 0], ys - d[0, 1]
        b1 = (px * by - py * bx) / den
        b2 = (ax * py - ay * px) / den
        b0 = 1.0 - b1 - b2
        inside = (b0 >= -1e-6) & (b1 >= -1e-6) & (b2 >= -1e-6)
        if not inside.any():
            continue
        xi = xs[inside].astype(np.int32); yi = ys[inside].astype(np.int32)
        b0i, b1i, b2i = b0[inside], b1[inside], b2[inside]

        depth[yi, xi] = b0i * z[0] + b1i * z[1] + b2i * z[2]
        for c in range(3):
            nrm[c, yi, xi] = b0i * nt[0, c] + b1i * nt[1, c] + b2i * nt[2, c]
        uvm[0, yi, xi] = b0i * uvt[0, 0] + b1i * uvt[1, 0] + b2i * uvt[2, 0]
        uvm[1, yi, xi] = b0i * uvt[0, 1] + b1i * uvt[1, 1] + b2i * uvt[2, 1]
        reg[yi, xi] = rt_major
        corrm[0, yi, xi] = b0i * s[0, 0] + b1i * s[1, 0] + b2i * s[2, 0]
        corrm[1, yi, xi] = b0i * s[0, 1] + b1i * s[1, 1] + b2i * s[2, 1]
        mask[yi, xi] = 1.0

    if mask.any():
        dv = depth[mask]
        dmin, dmax = float(dv.min()), float(dv.max())
        if dmax > dmin:
            depth = (depth - dmin) / (dmax - dmin) * mask

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    flw_x = (corrm[0] - xx) / 64.0
    flw_y = (corrm[1] - yy) / 64.0

    gx = np.gradient(corrm[0], axis=1); gy = np.gradient(corrm[0], axis=0)
    hx = np.gradient(corrm[1], axis=1); hy = np.gradient(corrm[1], axis=0)
    detJ = gx * hy - gy * hx
    stretch = np.clip(np.log(np.abs(detJ) + 1e-6), -3.0, 3.0) * mask
    conf = np.exp(-(stretch ** 2) / (2 * 0.8 ** 2)) * mask

    for r in range(8):
        G[5 + r] = (reg == r).astype(np.float32) * mask

    G[0] = mask
    G[1] = depth
    G[2:5] = nrm
    G[13:15] = uvm
    G[15] = flw_x; G[16] = flw_y
    G[17] = stretch
    G[18] = conf
    return G, mask.astype(bool)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Stage1Dataset(Dataset):
    def __init__(self, meta_dir, crop_dir, basis_path,
                 crop_size: int = 512, gbuffer_size: int = 128,
                 realign_B: bool = True):
        self.pairs = discover_pairs(Path(meta_dir), Path(crop_dir))
        if not self.pairs:
            raise RuntimeError(f"No pairs found under {meta_dir}/{crop_dir}")
        self.crop_size = crop_size
        self.gbuffer_size = gbuffer_size
        self.realign_B = realign_B

        with np.load(basis_path, allow_pickle=True) as z:
            self.delta_B = z["delta_B"].astype(np.float32)      # (52, V, 3)

        # One-time per-subject triangulation + per-pair B→A affine
        self.simplices_per_subject: Dict[str, np.ndarray] = {}
        self.b_align: Dict[int, np.ndarray] = {}
        for i, p in enumerate(self.pairs):
            subj = p["subject"]
            with np.load(p["A_npz"]) as zA:
                A_lm = zA["landmarks"].astype(np.float32)
            if subj not in self.simplices_per_subject:
                self.simplices_per_subject[subj] = delaunay_simplices(A_lm[:, :2])
            if realign_B:
                with np.load(p["B_npz"]) as zB:
                    B_lm = zB["landmarks"].astype(np.float32)
                self.b_align[i] = _umeyama_2d(B_lm[:, :2], A_lm[:, :2])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        with np.load(p["A_npz"]) as z:
            A_lm = z["landmarks"].astype(np.float32)
            e_A = z["blendshapes"].astype(np.float32)
        with np.load(p["B_npz"]) as z:
            B_lm = z["landmarks"].astype(np.float32)
            e_B = z["blendshapes"].astype(np.float32)

        b_aff = self.b_align.get(idx)
        if b_aff is not None:
            B_lm = _apply_affine_2d(B_lm, b_aff)

        simplices = self.simplices_per_subject[p["subject"]]

        # Source image A: unmodified.
        # Target image B: warp into A's crop frame with the SAME affine as landmarks.
        A_rgb = _read_rgb(p["A_img"])
        B_rgb = _read_rgb(p["B_img"], affine_2x3=b_aff, out_size=self.crop_size)

        # V_out = A_lm + Σ_k (e_B − e_A)_k · ΔB_k
        de = (e_B - e_A).astype(np.float32)
        V_out = A_lm + np.tensordot(de, self.delta_B, axes=(0, 0)).astype(np.float32)

        # Warp A to V_out's mesh shape
        grid_512 = build_sampling_grid(
            A_lm[:, :2], V_out[:, :2], simplices,
            H=self.crop_size, W=self.crop_size,
        )

        # G-buffer at 128
        G128, mask128 = build_gbuffer(
            A_lm, V_out, simplices,
            H=self.gbuffer_size, W=self.gbuffer_size,
            crop_size=self.crop_size,
        )

        A_t = torch.from_numpy(A_rgb).permute(2, 0, 1).float() / 255.0
        B_t = torch.from_numpy(B_rgb).permute(2, 0, 1).float() / 255.0
        W512 = warp_image(A_t.unsqueeze(0), grid_512).squeeze(0)

        return {
            "A_rgb":   A_t,
            "B_rgb":   B_t,
            "W512":    W512,
            "G128":    torch.from_numpy(G128),
            "mask128": torch.from_numpy(mask128.astype(np.float32)),
            "A_lm":    torch.from_numpy(A_lm),
            "B_lm":    torch.from_numpy(B_lm),
            "e_A":     torch.from_numpy(e_A),
            "e_B":     torch.from_numpy(e_B),
            "subject": p["subject"],
            "A_stem":  p["A_stem"],
            "B_stem":  p["B_stem"],
        }
