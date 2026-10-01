"""
geometry/mesh_model.py

MeshModel for GALNP Stage 0.

Responsibilities
- Load canonical template (from geometry.template.MeshTemplate) and expression basis
  (from geometry.expression_basis.ExpressionBasis).
- Provide a torch-based deform method:
      V_out = V_A + sum_k (e_B[k] - e_A[k]) * ΔB[k]
  supporting single-instance and batched inputs.
- Return a per-vertex correspondence attribute useful for later warping:
  - If the loaded template contains UVs and metadata.target_size, returns UV pixel
    coordinates (V,2) as the projected positions in the aligned crop.
  - Otherwise returns the input vertex positions V_A (identity mapping).
- Keep internals in torch for later differentiability; conversion helpers accept numpy.

Notes
- This module prefers PyTorch. If torch is not available, it raises ImportError.
- The class is intentionally small and focused; it does not perform rendering,
  camera projection, or image warping itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch import Tensor
except Exception as exc:  # pragma: no cover - environment dependent
    raise ImportError("geometry.mesh_model requires PyTorch (torch).") from exc

# Local imports (assumes geometry package is on PYTHONPATH)
from geometry.template import MeshTemplate, load_or_build_template  # type: ignore
from geometry.expression_basis import ExpressionBasis  # type: ignore


ArrayLike = Union[np.ndarray, Tensor]


@dataclass
class MeshModel:
    """
    MeshModel holds a template mesh and an expression basis and provides deformation.

    Typical usage
    -------------
    model = MeshModel()
    model.load_template(path_to_template_npz)   # loads MeshTemplate
    model.load_basis(path_to_basis_npz)         # loads ExpressionBasis
    V_out, corr = model.deform(V_A, e_A, e_B)

    Methods accept numpy arrays or torch tensors. Internally tensors are torch.float32
    on CPU. If you pass CUDA tensors, operations will follow the tensor device.
    """

    template: Optional[MeshTemplate] = None
    basis: Optional[ExpressionBasis] = None
    _delta_B_torch: Optional[Tensor] = None  # (K, V, 3) torch tensor
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    # -------------------------
    # Loading utilities
    # -------------------------
    def load_template(self, template_path: Optional[Path] = None, download_url: Optional[str] = None) -> MeshTemplate:
        """
        Load the canonical template. If template_path is None, uses default cache path
        and will raise if missing. Returns the loaded MeshTemplate and stores it.
        """
        if template_path is None:
            tpl = load_or_build_template(obj_path=None)  # will raise if no cache; kept for API parity
        else:
            tpl = load_or_build_template(obj_path=Path(template_path), cache_path=Path(template_path).with_suffix(".npz"))
        self.template = tpl
        return tpl

    def load_template_from_object(self, template: MeshTemplate) -> MeshTemplate:
        """
        Directly set the template from an existing MeshTemplate instance.
        """
        self.template = template
        return template

    def load_basis(self, basis_path: Optional[Path] = None, basis_obj: Optional[ExpressionBasis] = None) -> ExpressionBasis:
        """
        Load the expression basis from disk (.npz saved by ExpressionBasis.save) or
        accept an ExpressionBasis instance directly.
        """
        if basis_obj is not None:
            self.basis = basis_obj
        elif basis_path is not None:
            self.basis = ExpressionBasis.load(Path(basis_path))
        else:
            raise ValueError("Either basis_path or basis_obj must be provided")

        # Convert delta_B to torch tensor and store
        delta_B = np.asarray(self.basis.delta_B, dtype=np.float32)  # (K, V, 3)
        self._delta_B_torch = torch.from_numpy(delta_B).to(device=self.device, dtype=self.dtype)
        return self.basis

    def set_device(self, device: Union[str, torch.device]) -> None:
        """
        Move internal tensors to the specified device.
        """
        self.device = torch.device(device)
        if self._delta_B_torch is not None:
            self._delta_B_torch = self._delta_B_torch.to(self.device)

    # -------------------------
    # Helpers: conversions
    # -------------------------
    def _to_tensor(self, arr: ArrayLike, dtype: Optional[torch.dtype] = None) -> Tensor:
        """
        Convert numpy array or torch tensor to torch tensor on self.device.
        """
        if isinstance(arr, torch.Tensor):
            t = arr.to(device=self.device, dtype=(dtype or self.dtype))
        else:
            t = torch.as_tensor(np.asarray(arr), dtype=(dtype or self.dtype), device=self.device)
        return t

    def _ensure_basis_loaded(self) -> None:
        if self.basis is None or self._delta_B_torch is None:
            raise RuntimeError("Expression basis not loaded. Call load_basis() first.")

    # -------------------------
    # Core deformation API
    # -------------------------
    def deform(
        self,
        V_A: ArrayLike,
        e_A: ArrayLike,
        e_B: ArrayLike,
        return_correspondence: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Deform vertices V_A from expression e_A to expression e_B.

        Parameters
        ----------
        V_A : (V,3) or (B,V,3) numpy or torch
            Source vertices in the same vertex ordering as the basis/template.
        e_A : (K,) or (B,K) numpy or torch
            Source expression coefficients.
        e_B : (K,) or (B,K) numpy or torch
            Target expression coefficients.
        return_correspondence : bool
            If True, also return per-vertex correspondence attribute (V,2) or (B,V,2).

        Returns
        -------
        V_out : torch.Tensor
            Deformed vertices, shape (B,V,3) or (V,3) depending on input.
        correspondence : Optional[torch.Tensor]
            Per-vertex projected positions useful for warping (UV pixel coords if available),
            shape (B,V,2) or (V,2). None if return_correspondence=False.
        """
        self._ensure_basis_loaded()

        # Convert inputs to tensors
        Vt = self._to_tensor(V_A)  # (V,3) or (B,V,3)
        eA_t = self._to_tensor(e_A)
        eB_t = self._to_tensor(e_B)

        # Normalize shapes: promote to batch dimension if needed
        if Vt.ndim == 2:
            Vt = Vt.unsqueeze(0)  # (1,V,3)
            squeezed_V = True
        elif Vt.ndim == 3:
            squeezed_V = False
        else:
            raise ValueError("V_A must have shape (V,3) or (B,V,3)")

        B, V, C = Vt.shape
        if C != 3:
            raise ValueError("V_A must have 3 coordinates per vertex")

        K, V_basis, Cb = self._delta_B_torch.shape
        if V_basis != V:
            raise ValueError(f"Vertex count mismatch: basis has {V_basis} verts but V_A has {V}")

        # Ensure eA/eB shapes: (B,K)
        def _expand_coeffs(e_t: Tensor) -> Tensor:
            if e_t.ndim == 1:
                e_t = e_t.unsqueeze(0)  # (1,K)
            if e_t.ndim == 2:
                if e_t.shape[0] != B and e_t.shape[0] == 1:
                    # broadcast single coeffs to batch
                    e_t = e_t.expand(B, -1)
            if e_t.shape[0] != B:
                raise ValueError("Batch size of e_A/e_B must match V_A or be 1")
            return e_t

        eA_t = _expand_coeffs(eA_t)
        eB_t = _expand_coeffs(eB_t)

        if eA_t.shape[1] != K or eB_t.shape[1] != K:
            raise ValueError(f"Coefficient length mismatch: expected K={K}")

        # Compute delta coefficients (B,K)
        delta_e = eB_t - eA_t  # (B,K)

        # Compute weighted sum: sum_k delta_e[:,k] * delta_B[k] -> (B,V,3)
        # delta_B: (K,V,3) -> reshape to (K, V*3) or use broadcasting
        # Efficient computation: (B,K) @ (K, V, 3) -> (B,V,3)
        # Use einsum
        weighted = torch.einsum("bk,kvc->bvc", delta_e.to(self.device, self.dtype), self._delta_B_torch)  # (B,V,3)

        V_out = Vt + weighted  # (B,V,3)

        # Optionally compute correspondence attribute
        corr = None
        if return_correspondence:
            corr = self._compute_correspondence(Vt)

        # Squeeze batch dim if input was single
        if squeezed_V:
            V_out = V_out.squeeze(0)  # (V,3)
            if corr is not None:
                corr = corr.squeeze(0)  # (V,2)

        return V_out, corr

    # -------------------------
    # Correspondence helper
    # -------------------------
    def _compute_correspondence(self, Vt: Tensor) -> Optional[Tensor]:
        """
        Compute per-vertex correspondence attribute for warping.

        Strategy:
        - If template has UVs and metadata.target_size, return UV pixel coords:
            uv (V,2) in [0,1] -> uv * target_size -> (V,2) pixels
          This is returned as (B,V,2) matching batch size of Vt.
        - Otherwise return the input vertex positions Vt (B,V,3) projected to 2D by dropping Z.

        Returns None only if template is missing and Vt is invalid.
        """
        if self.template is None:
            # fallback: return XY of Vt
            return Vt[..., :2].to(self.device)

        uvs = getattr(self.template, "uvs", None)
        if uvs is not None and uvs.shape[0] == self.template.verts.shape[0]:
            # Use template metadata to determine pixel size if available
            target_size = None
            try:
                target_size = int(self.template.metadata.get("target_size", 0))
            except Exception:
                target_size = 0
            if target_size and target_size > 0:
                uv_pixels = torch.from_numpy(np.asarray(uvs, dtype=np.float32)).to(device=self.device, dtype=self.dtype) * float(target_size)
            else:
                # return normalized UVs in [0,1]
                uv_pixels = torch.from_numpy(np.asarray(uvs, dtype=np.float32)).to(device=self.device, dtype=self.dtype)
            # Broadcast to batch
            B = Vt.shape[0]
            uv_pixels = uv_pixels.unsqueeze(0).expand(B, -1, -1)  # (B,V,2)
            return uv_pixels
        else:
            # No UVs: return XY of Vt
            return Vt[..., :2].to(self.device)

    # -------------------------
    # Utility: save/load model state (template path + basis path)
    # -------------------------
    def save_state(self, path: Path) -> None:
        """
        Save a small JSON-like state describing loaded template and basis sources.
        This does not serialize tensors; it stores references for later reload.
        """
        import json

        state = {
            "template_source": getattr(self.template, "metadata", {}).get("source_obj", None) if self.template else None,
            "basis_source": getattr(self.basis, "metadata", {}).get("source", None) if self.basis else None,
            "device": str(self.device),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: Path) -> None:
        """
        Load state file produced by save_state. This only restores device setting.
        """
        import json

        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        dev = state.get("device", "cpu")
        self.set_device(dev)

