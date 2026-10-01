"""
geometry/gbuffer.py

Build a 20-channel G-buffer (512x512) from rasterizer output + original photo
for GALNP Stage 0.

Channels (indexing 1..20 in the user's spec, zero-based here):
0.  coverage mask (1)
1.  depth (normalized) (1)
2-4. normals (3)
5-12. region one-hot (8) [skin, upper_lip, lower_lip, mouth_cavity, eye_L, eye_R, brows, nose_other]
13-14. canonical UV (2)
15-16. flow / correspondence offset (corr - pixel) / 64 (2)
17.    stretch = log det J(corr) (1)
18.    confidence (1)
19.    padding / extra (1)

Also returns a warped photo W = grid_sample(original_photo, correspondence_coords).

Notes
-----
- This module uses PyTorch for tensor ops and grid_sample.
- It expects `raster_output` to be the RasterResult returned by geometry.rasterizer.rasterize/rasterize_mesh.
- If per-pixel correspondence ("corr") is not already present in raster_output.attrs,
  you may pass per-vertex projected positions `V_A_projected` (shape (V,2) or (B,V,2))
  and `faces` (F,3) so the function can interpolate per-pixel correspondence using
  raster_output.pix_to_face and raster_output.barycentric.
- Depth normalization, stretch and confidence are computed per-image with safe fallbacks.
- The function is written to be efficient and vectorized; it supports batch processing.

Return
------
A dict with keys:
- "G": torch.Tensor (B, 20, H, W) the G-buffer
- "W": torch.Tensor (B, 3, H, W) warped photo (RGB float32 in [0,1])
- "corr": torch.Tensor (B, H, W, 2) correspondence in pixel coordinates
- "components": dict of intermediate tensors (mask, depth, normals, regions, uv, flow, stretch, confidence)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np


def _ensure_batch(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has batch dim as first dimension."""
    if t.ndim == 3:
        return t.unsqueeze(0)
    return t


def _one_hot_regions(region_map: torch.Tensor, num_regions: int = 8) -> torch.Tensor:
    """
    region_map: (B, H, W) int64
    returns: (B, num_regions, H, W) float32 one-hot
    """
    B, H, W = region_map.shape
    # clamp negative (background) to 0
    region_clamped = torch.clamp(region_map, min=0)
    oh = F.one_hot(region_clamped.long(), num_classes=num_regions)  # (B,H,W,num_regions)
    oh = oh.permute(0, 3, 1, 2).to(dtype=torch.float32)  # (B,num_regions,H,W)
    return oh


def _normalize_depth(depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Normalize depth to [0,1] per image using valid pixels only.
    depth: (B,H,W)
    mask: (B,H,W) boolean
    """
    B = depth.shape[0]
    out = torch.zeros_like(depth)
    for i in range(B):
        valid = mask[i]
        if valid.any():
            d = depth[i][valid]
            dmin = float(d.min())
            dmax = float(d.max())
            if dmax > dmin:
                out[i][valid] = (depth[i][valid] - dmin) / (dmax - dmin)
            else:
                out[i][valid] = 0.0
    return out


def _compute_jacobian_and_stretch(corr: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute approximate Jacobian J of the correspondence field corr(x,y) -> (u,v)
    using finite differences and return:
      - detJ (B,H,W) and stretch = log(|detJ| + eps) (B,H,W)
    corr: (B,H,W,2) pixel coords
    mask: (B,H,W) boolean valid pixels
    """
    # corr: (B,H,W,2) -> permute to (B,2,H,W)
    corr_t = corr.permute(0, 3, 1, 2)  # (B,2,H,W)
    # Compute spatial gradients using Sobel-like kernels (finite differences)
    # kernels for dx, dy
    kernel_dx = torch.tensor([[-0.5, 0.0, 0.5]], dtype=corr_t.dtype, device=corr_t.device).reshape(1, 1, 1, 3)
    kernel_dy = kernel_dx.permute(0, 1, 3, 2)
    # pad to keep same size
    pad = (1, 1, 1, 1)
    # compute gradients for each channel u and v
    du_dx = F.conv2d(F.pad(corr_t[:, 0:1], pad, mode="replicate"), kernel_dx)
    du_dy = F.conv2d(F.pad(corr_t[:, 0:1], pad, mode="replicate"), kernel_dy)
    dv_dx = F.conv2d(F.pad(corr_t[:, 1:2], pad, mode="replicate"), kernel_dx)
    dv_dy = F.conv2d(F.pad(corr_t[:, 1:2], pad, mode="replicate"), kernel_dy)
    # du_dx etc shapes: (B,1,H,W)
    # compute detJ = du_dx * dv_dy - du_dy * dv_dx
    detJ = (du_dx * dv_dy - du_dy * dv_dx).squeeze(1)  # (B,H,W)
    # mask out invalid pixels
    detJ = detJ * mask.to(detJ.dtype)
    stretch = torch.log(torch.abs(detJ) + eps)
    # For invalid pixels set stretch to 0
    stretch = stretch * mask.to(stretch.dtype)
    return detJ, stretch


def _compute_confidence(stretch: torch.Tensor, normals: Optional[torch.Tensor], mask: torch.Tensor, mouth_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Heuristic confidence combining:
      - penalize extreme stretch (large |stretch|)
      - favor normals facing camera (normal_z)
      - penalize mouth cavity region slightly
    stretch: (B,H,W)
    normals: (B,3,H,W) or None
    mask: (B,H,W)
    mouth_mask: (B,H,W) boolean where mouth cavity is present
    Returns confidence in [0,1] (B,H,W)
    """
    B, H, W = mask.shape
    conf = torch.zeros_like(stretch)
    # stretch contribution: map stretch to [0,1] via gaussian-like falloff
    # smaller absolute stretch -> higher score
    s = stretch
    s_abs = torch.abs(s)
    # scale parameter
    sigma = 1.0
    stretch_score = torch.exp(- (s_abs ** 2) / (2 * sigma ** 2))
    conf = stretch_score

    # normal contribution
    if normals is not None:
        # normals: (B,3,H,W)
        nz = normals[:, 2:3, :, :].squeeze(1)  # (B,H,W)
        # map nz in [-1,1] to [0,1] favoring >0 (facing camera)
        normal_score = (nz + 1.0) / 2.0
        conf = conf * 0.6 + normal_score * 0.4

    # mouth penalty
    if mouth_mask is not None:
        conf = conf * (~mouth_mask).to(conf.dtype) + conf * 0.6 * mouth_mask.to(conf.dtype)

    # mask invalid pixels
    conf = conf * mask.to(conf.dtype)
    # clamp
    conf = torch.clamp(conf, 0.0, 1.0)
    return conf


def build_gbuffer(
    raster_output,
    original_image: torch.Tensor,
    V_A_projected: Optional[torch.Tensor] = None,
    faces: Optional[torch.Tensor] = None,
    resolution: Tuple[int, int] = (512, 512),
    flow_scale: float = 64.0,
) -> Dict[str, torch.Tensor]:
    """
    Build the G-buffer and warped photo.

    Parameters
    ----------
    raster_output : object
        The RasterResult returned by geometry.rasterizer.rasterize/rasterize_mesh.
        Expected attributes:
          - pix_to_face: (B,H,W) int tensor (-1 background)
          - barycentric: (B,H,W,3) float tensor
          - depth: (B,H,W) float tensor
          - attrs: dict of interpolated attributes (e.g., 'uv', 'normal', 'region', 'corr')
    original_image : torch.Tensor
        (B, 3, H, W) float32 in [0,1] or (3,H,W) -> will be expanded.
    V_A_projected : Optional[torch.Tensor]
        Per-vertex projected positions in pixel coords (V,2) or (B,V,2). Used if raster_output.attrs lacks 'corr'.
    faces : Optional[torch.Tensor]
        (F,3) int tensor required if V_A_projected is provided and raster_output.attrs lacks 'corr'.
    resolution : (H, W)
    flow_scale : float
        Divisor for flow normalization (corr - pixel) / flow_scale

    Returns
    -------
    dict with keys:
      - "G": (B,20,H,W) float32
      - "W": (B,3,H,W) warped photo
      - "corr": (B,H,W,2) correspondence in pixel coords
      - "components": dict of intermediate tensors
    """
    device = original_image.device
    H, W = resolution
    # Ensure batch dims
    pix_to_face = _ensure_batch(raster_output.pix_to_face)  # (B,H,W)
    bary = _ensure_batch(raster_output.barycentric)  # (B,H,W,3)
    depth = _ensure_batch(raster_output.depth)  # (B,H,W)
    attrs = getattr(raster_output, "attrs", {})

    B = pix_to_face.shape[0]

    # Ensure original_image shape (B,3,H,W)
    if original_image.ndim == 3:
        original_image = original_image.unsqueeze(0)
    if original_image.shape[2] != H or original_image.shape[3] != W:
        # If original image resolution differs, we will still grid_sample using corr coords in pixel space.
        # But it's recommended to pass aligned 512x512 crops.
        pass

    # coverage mask
    mask = (pix_to_face >= 0)  # (B,H,W) bool

    # depth normalized
    depth_norm = _normalize_depth(depth, mask)  # (B,H,W)

    # normals: try raster_output.attrs['normal'] expected shape (B,H,W,3) or (B,3,H,W)
    normals = None
    if "normal" in attrs:
        n = attrs["normal"]
        if n.ndim == 4 and n.shape[-1] == 3:
            normals = n.permute(0, 3, 1, 2)  # (B,3,H,W)
        elif n.ndim == 4 and n.shape[1] == 3:
            normals = n  # (B,3,H,W)
        else:
            # try to reshape
            try:
                normals = n.permute(0, 3, 1, 2)
            except Exception:
                normals = None
    # If normals missing, set default facing camera (0,0,1)
    if normals is None:
        normals = torch.zeros((B, 3, H, W), dtype=torch.float32, device=device)
        normals[:, 2, :, :] = 1.0

    # regions: expect attrs['region'] (B,H,W) int or float
    if "region" in attrs:
        region_map = attrs["region"]
        if region_map.ndim == 4 and region_map.shape[-1] == 1:
            region_map = region_map[..., 0]
        if region_map.ndim == 3:
            region_map = region_map
        else:
            # try permute
            try:
                region_map = region_map.squeeze(-1)
            except Exception:
                region_map = torch.zeros((B, H, W), dtype=torch.int64, device=device)
    else:
        # default all skin (region 0)
        region_map = torch.zeros((B, H, W), dtype=torch.int64, device=device)

    # region one-hot (8 channels)
    region_onehot = _one_hot_regions(region_map, num_regions=8)  # (B,8,H,W)

    # canonical UVs
    if "uv" in attrs:
        uv = attrs["uv"]  # (B,H,W,2) or (B,2,H,W)
        if uv.ndim == 4 and uv.shape[-1] == 2:
            uv_pixels = uv.permute(0, 3, 1, 2)  # (B,2,H,W)
        elif uv.ndim == 4 and uv.shape[1] == 2:
            uv_pixels = uv
        else:
            uv_pixels = torch.zeros((B, 2, H, W), dtype=torch.float32, device=device)
    else:
        uv_pixels = torch.zeros((B, 2, H, W), dtype=torch.float32, device=device)

    # correspondence corr: prefer raster_output.attrs['corr'] else interpolate from V_A_projected + faces
    corr = None  # (B,H,W,2)
    if "corr" in attrs:
        c = attrs["corr"]
        if c.ndim == 4 and c.shape[-1] == 2:
            corr = c  # (B,H,W,2)
        elif c.ndim == 4 and c.shape[1] == 2:
            corr = c.permute(0, 2, 3, 1)
        else:
            corr = None

    if corr is None and V_A_projected is not None and faces is not None:
        # Interpolate per-pixel correspondence using pix_to_face and barycentric
        # V_A_projected: (V,2) or (B,V,2)
        Vproj = V_A_projected
        if isinstance(Vproj, torch.Tensor):
            Vproj_t = Vproj.to(device)
        else:
            Vproj_t = torch.as_tensor(np.asarray(Vproj), dtype=torch.float32, device=device)
        if Vproj_t.ndim == 2:
            Vproj_t = Vproj_t.unsqueeze(0).expand(B, -1, -1)  # (B,V,2)
        # faces: (F,3)
        faces_t = faces.to(device=device, dtype=torch.long)
        # Build per-pixel corr by indexing faces
        # pix_to_face: (B,H,W) with -1 for background
        # bary: (B,H,W,3)
        # We'll create corr by gathering vertex coords for each face index
        Bp, Hp, Wp = pix_to_face.shape
        corr_out = torch.zeros((B, Hp, Wp, 2), dtype=torch.float32, device=device)
        # For efficiency, vectorize: create face_idx_flat and bary_flat
        face_idx_flat = pix_to_face.reshape(Bp, -1)  # (B, HW)
        bary_flat = bary.reshape(Bp, -1, 3)  # (B, HW, 3)
        for b in range(Bp):
            face_idx_b = face_idx_flat[b]  # (HW,)
            bary_b = bary_flat[b]  # (HW,3)
            valid_mask = face_idx_b >= 0
            if valid_mask.any():
                valid_faces = face_idx_b[valid_mask]  # indices into faces
                # gather vertex indices
                face_vertices = faces_t[valid_faces]  # (N_valid, 3)
                # gather per-vertex projected coords
                v0 = Vproj_t[b, face_vertices[:, 0], :]  # (N_valid,2)
                v1 = Vproj_t[b, face_vertices[:, 1], :]
                v2 = Vproj_t[b, face_vertices[:, 2], :]
                bary_vals = bary_b[valid_mask]  # (N_valid,3)
                corr_vals = v0 * bary_vals[:, 0:1] + v1 * bary_vals[:, 1:2] + v2 * bary_vals[:, 2:3]  # (N_valid,2)
                corr_out[b].view(-1, 2)[valid_mask] = corr_vals
        corr = corr_out  # (B,H,W,2)

    # If still None, set corr to pixel centers (identity)
    if corr is None:
        # pixel coordinates: x in [0,W-1], y in [0,H-1]
        xs = torch.linspace(0.0, W - 1.0, W, device=device)
        ys = torch.linspace(0.0, H - 1.0, H, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        corr = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1).to(dtype=torch.float32)

    # flow = (corr - pixel_coords) / flow_scale
    xs = torch.linspace(0.0, W - 1.0, W, device=device)
    ys = torch.linspace(0.0, H - 1.0, H, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pixel_coords = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # (B,H,W,2)
    flow = (corr - pixel_coords) / float(flow_scale)  # (B,H,W,2)

    # stretch: compute Jacobian det and log det
    detJ, stretch = _compute_jacobian_and_stretch(corr, mask)

    # confidence
    normals_for_conf = normals  # (B,3,H,W)
    mouth_mask = None
    # mouth_mask from region_map where region index corresponds to mouth cavity; find index if present
    # assume region mapping: 0 skin, 1 upper_lip, 2 lower_lip, 3 mouth_cavity, 4 left_eye, 5 right_eye, 6 brows, 7 nose_other
    mouth_mask = (region_map == 3)
    confidence = _compute_confidence(stretch, normals_for_conf, mask, mouth_mask)

    # Prepare channels
    # 0 coverage mask (float)
    coverage = mask.to(dtype=torch.float32).unsqueeze(1)  # (B,1,H,W)
    # 1 depth_norm (float)
    depth_ch = depth_norm.unsqueeze(1)  # (B,1,H,W)
    # 2-4 normals (B,3,H,W)
    normals_ch = normals  # already (B,3,H,W)
    # 5-12 region one-hot (B,8,H,W)
    regions_ch = region_onehot  # (B,8,H,W)
    # 13-14 canonical UV (B,2,H,W)
    uv_ch = uv_pixels  # (B,2,H,W)
    # 15-16 flow (B,2,H,W)
    flow_ch = flow.permute(0, 3, 1, 2)  # (B,2,H,W)
    # 17 stretch (B,1,H,W)
    stretch_ch = stretch.unsqueeze(1)
    # 18 confidence (B,1,H,W)
    conf_ch = confidence.unsqueeze(1)
    # 19 padding zeros
    pad_ch = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)

    # Concatenate in order to form G (B,20,H,W)
    G = torch.cat([coverage, depth_ch, normals_ch, regions_ch, uv_ch, flow_ch, stretch_ch, conf_ch, pad_ch], dim=1)
    assert G.shape[1] == 20, f"G-buffer channel mismatch: got {G.shape[1]} channels"

    # Warped photo W: grid_sample expects normalized coords in [-1,1] with (x,y) order
    # Build sampling grid from corr (pixel coords) to normalized [-1,1]
    # original_image: (B,3,H_img,W_img) may be same resolution; assume same H,W for aligned crops
    B_img = original_image.shape[0]
    if B_img != B:
        # broadcast original image if needed
        if B_img == 1:
            original_image = original_image.expand(B, -1, -1, -1)
        else:
            raise ValueError("Batch size of original_image must match raster_output or be 1")

    # Normalize corr to [-1,1] for grid_sample: x -> -1..1, y -> -1..1
    # Note grid_sample expects grid in shape (B,H,W,2) with last dim (x_norm, y_norm)
    x_norm = (corr[..., 0] / (W - 1)) * 2.0 - 1.0
    y_norm = (corr[..., 1] / (H - 1)) * 2.0 - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)  # (B,H,W,2)
    # grid_sample expects float32
    grid = grid.to(dtype=original_image.dtype)
    # sample
    # align_corners True to match pixel mapping precisely
    W_warp = F.grid_sample(original_image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    components = {
        "mask": mask,
        "depth": depth_norm,
        "normals": normals,
        "regions_onehot": region_onehot,
        "uv": uv_pixels,
        "flow": flow,
        "stretch": stretch,
        "confidence": confidence,
        "pixel_coords": pixel_coords,
    }

    return {
        "G": G,
        "W": W_warp,
        "corr": corr,
        "components": components,
    }
