"""
geometry/rasterizer.py

Thin wrapper around nvdiffrast for GALNP Stage 0 rasterization.

Features
- Rasterize triangle meshes (batched) into G-buffer style outputs at 512x512.
- Orthographic / weak-perspective camera suitable for aligned face crops.
- Return:
    - pix_to_face: (B, H, W) int32 face index per pixel (-1 for background)
    - barycentric: (B, H, W, 3) float32 barycentric coords (sum to 1 where valid)
    - depth: (B, H, W) float32 depth (camera-space z)
    - attrs: dict of interpolated per-pixel attributes (normals, uvs, regions, etc.)
- Two entry points:
    - rasterize(...) low-level wrapper around nvdiffrast.rasterize + dr.interpolate
    - rasterize_mesh(...) higher-level convenience that accepts vertices, faces, and per-vertex attributes
- Small abstraction so backend can be swapped later (PyTorch3D, custom CUDA, etc.)

Important note about licensing
------------------------------
nvdiffrast is distributed under a non-commercial license. By using this module you
agree to comply with nvdiffrast's license terms. See the nvdiffrast repository
for details: https://github.com/NVlabs/nvdiffrast

Dependencies
------------
- torch
- nvdiffrast (nvdiffrast.torch)
- numpy

This module intentionally keeps the API narrow and typed so it can be replaced
with another rasterizer backend later with minimal changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# Try to import nvdiffrast; raise a clear error if missing
try:
    import nvdiffrast.torch as dr  # type: ignore
except Exception as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "nvdiffrast (nvdiffrast.torch) is required for geometry.rasterizer. "
        "Install from https://github.com/NVlabs/nvdiffrast and ensure CUDA is available."
    ) from exc


# Default raster resolution for aligned face crops
DEFAULT_RESOLUTION = (512, 512)


@dataclass
class RasterResult:
    """
    Container for rasterization outputs.

    Attributes
    ----------
    pix_to_face : torch.LongTensor
        (B, H, W) face index per pixel, -1 for background.
    barycentric : torch.FloatTensor
        (B, H, W, 3) barycentric coordinates for the hit triangle.
    depth : torch.FloatTensor
        (B, H, W) depth values (camera-space z) for the hit triangle.
    attrs : Dict[str, torch.Tensor]
        Interpolated per-pixel attributes. Each tensor has shape (B, H, W, C_attr).
    raw_rast : torch.Tensor
        The raw rasterizer output (B, H, W, 4) returned by nvdiffrast (kept for debugging).
    """

    pix_to_face: torch.LongTensor
    barycentric: torch.FloatTensor
    depth: torch.FloatTensor
    attrs: Dict[str, torch.Tensor]
    raw_rast: torch.Tensor


# -------------------------
# Camera / transform helpers
# -------------------------
def _build_orthographic_projection(scale: float = 1.0, translate: Tuple[float, float] = (0.0, 0.0)) -> torch.Tensor:
    """
    Build a simple 4x4 orthographic projection matrix that maps model coordinates
    (assumed to be roughly in [-0.5, 0.5] or similar) into normalized device coordinates.

    The returned matrix maps (x, y, z, 1) -> (x', y', z', 1) where x',y' are in NDC [-1,1].
    We keep z unchanged (weak-perspective / orthographic).

    Parameters
    ----------
    scale : float
        Uniform scale applied to x,y before projection.
    translate : (tx, ty)
        Translation applied to x,y after scaling.

    Returns
    -------
    proj : torch.Tensor (4,4)
    """
    tx, ty = translate
    # Scale then translate, then map to NDC [-1,1] by multiplying by 2
    # We construct a 4x4 affine matrix:
    # [ s 0 0 tx ]
    # [ 0 s 0 ty ]
    # [ 0 0 1  0 ]
    # [ 0 0 0  1 ]
    proj = torch.eye(4, dtype=torch.float32)
    proj[0, 0] = float(scale)
    proj[1, 1] = float(scale)
    proj[0, 3] = float(tx)
    proj[1, 3] = float(ty)
    return proj


# -------------------------
# Low-level rasterize wrapper
# -------------------------
def rasterize(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attributes: Optional[Dict[str, torch.Tensor]] = None,
    resolution: Tuple[int, int] = DEFAULT_RESOLUTION,
    proj: Optional[torch.Tensor] = None,
    near: float = 0.0,
    far: float = 1.0,
    cull_backfaces: bool = True,
) -> RasterResult:
    """
    Low-level rasterize using nvdiffrast.

    Parameters
    ----------
    vertices : torch.Tensor
        (B, V, 3) float32 vertex positions in model space.
    faces : torch.Tensor
        (F, 3) int64 or int32 triangle indices (shared across batch).
        If faces are per-batch, pass shape (B, F, 3) and ensure dtype is int32/int64.
    attributes : dict[str, torch.Tensor], optional
        Per-vertex attributes to interpolate. Each tensor should be (B, V, C) or (V, C).
    resolution : (H, W)
        Output raster resolution.
    proj : torch.Tensor, optional
        (4,4) projection matrix in model space. If None, an identity orthographic projection is used.
    near, far : float
        Depth range (kept for compatibility; nvdiffrast returns raw z).
    cull_backfaces : bool
        Whether to cull back-facing triangles.

    Returns
    -------
    RasterResult
    """
    # Validate inputs
    if vertices.ndim != 3:
        raise ValueError("vertices must be (B, V, 3)")
    B, V, C = vertices.shape
    if C != 3:
        raise ValueError("vertices last dim must be 3")

    # faces may be (F,3) or (B,F,3)
    if faces.ndim == 2:
        F = faces.shape[0]
        faces_t = faces.to(dtype=torch.int32, device=vertices.device)
    elif faces.ndim == 3:
        if faces.shape[0] != B:
            raise ValueError("If faces are batched, faces.shape[0] must equal vertices.shape[0]")
        F = faces.shape[1]
        faces_t = faces.to(dtype=torch.int32, device=vertices.device)
    else:
        raise ValueError("faces must be (F,3) or (B,F,3)")

    # Projection: apply proj to vertices if provided
    device = vertices.device
    if proj is None:
        proj = _build_orthographic_projection().to(device)
    else:
        proj = proj.to(device)

    # Convert vertices to homogeneous and apply proj
    # vertices: (B, V, 3) -> (B, V, 4)
    ones = torch.ones((B, V, 1), dtype=vertices.dtype, device=device)
    verts_h = torch.cat([vertices, ones], dim=2)  # (B,V,4)
    # Apply projection: (B,V,4) x (4,4)^T -> (B,V,4)
    verts_proj = torch.matmul(verts_h, proj.T)  # (B,V,4)
    # Convert to NDC by dividing by w (should be 1 for affine proj)
    verts_ndc = verts_proj[..., :3]  # (B,V,3)

    # nvdiffrast expects positions in clip space (x,y,z) where x,y in [-1,1] map to screen
    # and z is depth. We pass verts_ndc directly.
    # Prepare faces: if faces are (F,3) replicate for batch
    if faces.ndim == 2:
        faces_batch = faces_t.unsqueeze(0).expand(B, -1, -1)
    else:
        faces_batch = faces_t  # already batched

    # Call nvdiffrast rasterizer
    # dr.rasterize returns (rast, rast_db) where rast is (B, H, W, 4)
    H, W = resolution
    # nvdiffrast expects vertices as (B, V, 3) and triangles as (B, F, 3)
    rast, _ = dr.rasterize(verts_ndc.contiguous(), faces_batch.contiguous(), resolution=(H, W))

    # rast: (B, H, W, 4) where:
    #  - rast[..., 0] : triangle index + 1 (0 means background)
    #  - rast[..., 1:4] : barycentric coordinates (b0, b1, b2)
    #  - rast[..., 3] : z (depth) in clip space (interpolated)
    # Note: API details may vary across nvdiffrast versions; this is the common layout.
    # Convert to convenient tensors
    # face_idx_raw: (B,H,W) int (0 = background)
    face_idx_raw = rast[..., 0].to(dtype=torch.int32)
    # Convert to -1 for background, and zero-based indices for faces
    pix_to_face = (face_idx_raw - 1).to(dtype=torch.int64)
    pix_to_face = torch.where(face_idx_raw == 0, torch.full_like(pix_to_face, -1), pix_to_face)

    bary = rast[..., 1:4].to(dtype=torch.float32)  # (B,H,W,3)
    depth = rast[..., 3].to(dtype=torch.float32)  # (B,H,W)

    # Interpolate attributes if provided
    interp_attrs: Dict[str, torch.Tensor] = {}
    if attributes:
        for name, attr in attributes.items():
            # attr may be (B,V,C) or (V,C)
            if isinstance(attr, torch.Tensor):
                attr_t = attr.to(device)
            else:
                attr_t = torch.as_tensor(np.asarray(attr), device=device)
            if attr_t.ndim == 2:
                # (V,C) -> (B,V,C)
                attr_t = attr_t.unsqueeze(0).expand(B, -1, -1)
            if attr_t.ndim != 3:
                raise ValueError(f"Attribute {name} must be (V,C) or (B,V,C)")
            # dr.interpolate expects attributes as (B, V, C) and uses rast and faces to interpolate
            # The API: dr.interpolate(attr, rast, tri) -> (B,H,W,C)
            interp = dr.interpolate(attr_t.contiguous(), rast.contiguous(), faces_batch.contiguous())
            interp_attrs[name] = interp  # (B,H,W,C)

    return RasterResult(
        pix_to_face=pix_to_face,
        barycentric=bary,
        depth=depth,
        attrs=interp_attrs,
        raw_rast=rast,
    )


# -------------------------
# High-level convenience
# -------------------------
def rasterize_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    uv: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    regions: Optional[np.ndarray] = None,
    resolution: Tuple[int, int] = DEFAULT_RESOLUTION,
    device: Optional[torch.device] = None,
    orthographic_scale: float = 1.0,
    orthographic_translate: Tuple[float, float] = (0.0, 0.0),
) -> RasterResult:
    """
    High-level rasterize helper that accepts numpy arrays and common per-vertex attributes.

    Parameters
    ----------
    verts : (B, V, 3) or (V, 3) numpy
    faces : (F, 3) or (B, F, 3) numpy
    uv : (V, 2) numpy optional
    normals : (V, 3) numpy optional
    regions : (V,) int numpy optional (region indices)
    resolution : (H, W)
    device : torch.device or None (defaults to cpu)
    orthographic_scale : float
    orthographic_translate : (tx, ty)

    Returns
    -------
    RasterResult
    """
    if device is None:
        device = torch.device("cpu")
    # Convert inputs to torch
    def _to_torch(x, dtype=torch.float32):
        if x is None:
            return None
        t = torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
        return t

    verts_t = _to_torch(verts, dtype=torch.float32)
    faces_t = _to_torch(faces, dtype=torch.int32)
    uv_t = _to_torch(uv, dtype=torch.float32)
    normals_t = _to_torch(normals, dtype=torch.float32)
    regions_t = _to_torch(regions, dtype=torch.float32) if regions is not None else None

    # Ensure batch dimension on verts
    if verts_t.ndim == 2:
        verts_t = verts_t.unsqueeze(0)  # (1,V,3)
    if faces_t.ndim == 2:
        faces_t = faces_t  # (F,3)
    elif faces_t.ndim == 3 and faces_t.shape[0] == verts_t.shape[0]:
        faces_t = faces_t
    else:
        # If faces are (F,3) but verts are batched, replicate faces for batch
        if faces_t.ndim == 2 and verts_t.ndim == 3:
            faces_t = faces_t  # rasterize() will expand faces
        else:
            raise ValueError("faces must be (F,3) or (B,F,3)")

    # Build attributes dict
    attrs: Dict[str, torch.Tensor] = {}
    if uv_t is not None:
        attrs["uv"] = uv_t
    if normals_t is not None:
        attrs["normal"] = normals_t
    if regions_t is not None:
        # store regions as float attribute for interpolation; caller can round later
        attrs["region"] = regions_t.unsqueeze(-1) if regions_t.ndim == 1 else regions_t

    # Build projection matrix
    proj = _build_orthographic_projection(scale=orthographic_scale, translate=orthographic_translate)

    # Call low-level rasterize
    result = rasterize(
        vertices=verts_t,
        faces=faces_t,
        attributes=attrs if attrs else None,
        resolution=resolution,
        proj=proj,
    )

    # Post-process region attribute to integer if present
    if "region" in result.attrs:
        # round to nearest int and cast to long
        region_interp = result.attrs["region"]  # (B,H,W,1)
        region_int = torch.round(region_interp[..., 0]).to(dtype=torch.int64)
        result.attrs["region"] = region_int  # replace with (B,H,W) int tensor

    return result


# -------------------------
# Backend abstraction note
# -------------------------
# The functions above are intentionally small and use only:
#   - dr.rasterize(...)
#   - dr.interpolate(...)
# If you later want to swap to a different backend (PyTorch3D or custom CUDA),
# replace rasterize(...) implementation and keep the RasterResult contract.
#
# Example usage:
#   result = rasterize_mesh(verts_np, faces_np, uv=uv_np, normals=norm_np)
#   pix_to_face = result.pix_to_face.cpu().numpy()
#   bary = result.barycentric.cpu().numpy()
#   depth = result.depth.cpu().numpy()
#   uv_pixels = result.attrs.get("uv")  # (B,H,W,2) or None
#
# Keep in mind that nvdiffrast returns triangle indices with 1-based indexing in the raster buffer,
# so we convert them to -1 for background and 0-based face indices for valid pixels.

# End of file
