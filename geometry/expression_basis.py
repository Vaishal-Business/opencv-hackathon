"""
geometry/expression_basis.py

Build an expression basis ΔB from neutral <-> smiling pairs produced by Stage 0
(`prepare.py`) and provide utilities to save/load and apply the basis.

Overview
--------
- Scans a directory of processed .npz files (each expected to contain at least:
  'landmarks' (478,3) and 'blendshapes' (52,) and filename pattern "<face_id>_<expression>.npz"
  where expression is 'neutral' or 'smiling').
- Groups files by face_id and keeps pairs that have both neutral and smiling.
- For each pair:
    * Align neutral -> smiling using a 2D similarity (Umeyama) computed on x,y.
    * Apply the similarity to neutral landmarks (x,y) and scale z accordingly.
    * Compute landmark delta: smiling - aligned_neutral  (shape (N,3))
    * Compute blendshape delta: smiling_blendshapes - neutral_blendshapes (shape (52,))
- Solve a ridge regression (λ = 1e-2) to find ΔB of shape (52, N, 3) such that:
      for each sample i:  sum_k ΔB[k] * b_i[k] ≈ landmark_delta_i
  Closed-form solution used: B = Y X^T (X X^T + λ I)^{-1}
  where X is (52, M) blendshape deltas, Y is (N*3, M) stacked landmark deltas.
- Optionally lift the landmark basis to a full template mesh (V_template vertices)
  by nearest-vertex mapping (simple and fast).
- Save / load ΔB as compressed .npz (or .npz + metadata pickle).
- Provide apply_expression(verts, e_delta, delta_B) to deform vertices.

Dependencies
------------
- numpy
- scipy (for KDTree)
- typing, pathlib, dataclasses, pickle, logging

This module intentionally avoids PyTorch and MediaPipe dependencies.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

LOGGER = logging.getLogger("geometry.expression_basis")
LOGGER.addHandler(logging.NullHandler())

# Constants
NUM_LANDMARKS = 478
NUM_BLENDSHAPES = 52
DEFAULT_REGULARIZATION = 1e-2


# -------------------------
# Utility math: Umeyama similarity (2D)
# -------------------------
def estimate_similarity_transform_2d(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Estimate 2D similarity transform (scale * R, translation) mapping src -> dst.
    src, dst: (N,2) arrays. Returns (A, s) where A is (2,3) affine matrix such that:
        dst ≈ (src @ A[:, :2].T) + A[:, 2]
    and s is the uniform scale.
    Implementation follows Umeyama (1991) least-squares similarity.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 2

    n = src.shape[0]
    if n == 0:
        raise ValueError("Empty point sets for similarity estimation")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    cov = (dst_c.T @ src_c) / n
    U, S, VT = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(VT) < 0:
        D[-1, -1] = -1.0
    R = U @ D @ VT
    var_src = (src_c ** 2).sum() / n
    scale = float((S * np.diag(D)).sum() / var_src) if var_src > 0 else 1.0
    t = dst_mean - scale * (R @ src_mean)
    A = np.hstack([scale * R, t.reshape(2, 1)])  # (2,3)
    return A, scale


def apply_similarity_to_landmarks(landmarks: np.ndarray, A: np.ndarray, width: Optional[int] = None) -> np.ndarray:
    """
    Apply 2x3 affine similarity A to landmarks.
    landmarks: (N,3) normalized coords (x,y in [0,1]) OR pixel coords depending on caller.
    If landmarks are normalized and width is provided, x,y are scaled by width/height externally.
    This function expects landmarks in normalized coords and returns transformed landmarks in same scale.
    For our use: we will apply A in pixel coordinates, so caller should convert before calling.
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.ndim != 2 or lm.shape[1] < 2:
        raise ValueError("landmarks must be (N, >=2)")
    xy = lm[:, :2]
    xy_t = (xy @ A[:, :2].T) + A[:, 2]
    # z scaling: multiply original z by scale (approx)
    # Extract scale from A as sqrt(|det(A[:2,:2])|)
    scale = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
    z = lm[:, 2] * scale if lm.shape[1] > 2 else np.zeros((lm.shape[0],), dtype=np.float64)
    return np.concatenate([xy_t, z[:, None]], axis=1)


# -------------------------
# Data container for basis
# -------------------------
@dataclass
class ExpressionBasis:
    """
    Container for expression basis ΔB.

    Attributes
    ----------
    delta_B : np.ndarray
        (K, V, 3) float32 where K = 52 (blendshape dims), V = number of vertices (landmarks or template verts).
    vertex_indices : int
        Number of vertices V.
    source : str
        Description of how the basis was built (e.g., 'landmark_space' or 'lifted_to_template').
    metadata : dict
        Additional metadata (regularization, sample count, mapping info).
    """
    delta_B: np.ndarray
    vertex_count: int
    source: str
    metadata: Dict

    def save(self, path: Path) -> None:
        """
        Save the basis to a compressed .npz and a small pickle for metadata.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), delta_B=self.delta_B.astype(np.float32))
        meta_pkl = path.with_suffix(path.suffix + ".meta.pkl")
        with open(meta_pkl, "wb") as f:
            pickle.dump({"vertex_count": int(self.vertex_count), "source": self.source, "metadata": self.metadata}, f)

    @staticmethod
    def load(path: Path) -> "ExpressionBasis":
        """
        Load a basis saved with save().
        """
        path = Path(path)
        data = np.load(str(path), allow_pickle=False)
        delta_B = data["delta_B"].astype(np.float32)
        meta_pkl = path.with_suffix(path.suffix + ".meta.pkl")
        if meta_pkl.exists():
            with open(meta_pkl, "rb") as f:
                meta = pickle.load(f)
            vertex_count = int(meta.get("vertex_count", delta_B.shape[1]))
            source = meta.get("source", "unknown")
            metadata = meta.get("metadata", {})
        else:
            vertex_count = int(delta_B.shape[1])
            source = "unknown"
            metadata = {}
        return ExpressionBasis(delta_B=delta_B, vertex_count=vertex_count, source=source, metadata=metadata)


# -------------------------
# Core builder
# -------------------------
def _load_npz_landmark(npz_path: Path) -> Optional[Dict]:
    try:
        data = np.load(str(npz_path), allow_pickle=True)
    except Exception as exc:
        LOGGER.debug("Failed to load %s: %s", npz_path, exc)
        return None

    if "landmarks" not in data or "blendshapes" not in data:
        LOGGER.debug("Missing required arrays in %s", npz_path)
        return None

    landmarks = np.asarray(data["landmarks"], dtype=np.float32)
    blendshapes = np.asarray(data["blendshapes"], dtype=np.float32)

    # Robust filename parsing
    stem = npz_path.stem
    parts = stem.split("_")
    if len(parts) >= 2:
        expression = parts[-1].lower()
        if parts[0].isdigit():
            face_id = parts[0]
        else:
            face_id = "_".join(parts[:-1])
    else:
        face_id = stem
        expression = ""

    return {"face_id": face_id, "expression": expression, "landmarks": landmarks, "blendshapes": blendshapes, "path": str(npz_path)}

def build_expression_basis_from_dir(
    processed_dir: Path,
    regularization: float = DEFAULT_REGULARIZATION,
    lift_to_template_verts: Optional[np.ndarray] = None,
    lift_method: str = "nearest",
    save_path: Optional[Path] = None,
) -> ExpressionBasis:
    """
    Build expression basis ΔB from a directory of processed .npz files.

    Parameters
    ----------
    processed_dir : Path
        Directory containing .npz files produced by prepare.py
    regularization : float
        Ridge regression lambda (default 1e-2)
    lift_to_template_verts : Optional[np.ndarray]
        If provided, an array of template vertices (V_template, 3). The computed
        landmark-space basis (K, N, 3) will be lifted to (K, V_template, 3)
        by nearest-vertex mapping.
    lift_method : str
        'nearest' currently supported.
    save_path : Optional[Path]
        If provided, save the resulting basis to this path (.npz + .meta.pkl).

    Returns
    -------
    ExpressionBasis
    """
    processed_dir = Path(processed_dir)
    files = sorted(processed_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {processed_dir}")

    # Load all files
    records = []
    for p in files:
        rec = _load_npz_landmark(p)
        if rec is not None:
            records.append(rec)

    # Group by face_id
    groups: Dict[str, Dict[str, Dict]] = {}
    for r in records:
        fid = r["face_id"]
        expr = r["expression"].lower()
        groups.setdefault(fid, {})[expr] = r

    # Collect pairs
    pairs = []
    for fid, d in groups.items():
        if "neutral" in d and "smiling" in d:
            pairs.append((d["neutral"], d["smiling"]))
    if not pairs:
        raise RuntimeError("No neutral-smiling pairs found in the processed directory")

    LOGGER.info("Found %d neutral-smiling pairs", len(pairs))

    # For each pair compute aligned landmark delta and blendshape delta
    X_list = []  # blendshape deltas (52,) per sample
    Y_list = []  # landmark deltas flattened (N*3,) per sample

    for neutral_rec, smile_rec in pairs:
        lm_neu = np.asarray(neutral_rec["landmarks"], dtype=np.float64)  # (N,3)
        lm_smi = np.asarray(smile_rec["landmarks"], dtype=np.float64)
        bs_neu = np.asarray(neutral_rec["blendshapes"], dtype=np.float64)  # (52,)
        bs_smi = np.asarray(smile_rec["blendshapes"], dtype=np.float64)

        if lm_neu.shape[0] != NUM_LANDMARKS or lm_smi.shape[0] != NUM_LANDMARKS:
            LOGGER.debug("Skipping pair %s due to unexpected landmark count", neutral_rec["path"])
            continue

        # Convert normalized x,y to pixel-like coordinates for similarity estimation.
        # Use unit square scaling: multiply by 1.0 (we only need relative positions).
        src_xy = lm_neu[:, :2].copy()
        dst_xy = lm_smi[:, :2].copy()

        # Estimate similarity transform mapping neutral -> smiling
        try:
            A, scale = estimate_similarity_transform_2d(src_xy, dst_xy)
        except Exception as exc:
            LOGGER.debug("Similarity estimation failed for %s: %s", neutral_rec["path"], exc)
            continue

        # Apply transform to neutral landmarks (in same normalized coordinate space)
        lm_neu_aligned = apply_similarity_to_landmarks(lm_neu, A)  # (N,3)

        # Compute delta: smiling - aligned_neutral
        delta_lm = lm_smi - lm_neu_aligned  # (N,3)

        # Flatten to (N*3,)
        Y_list.append(delta_lm.reshape(-1))  # length N*3

        # Blendshape delta
        delta_bs = (bs_smi - bs_neu).reshape(-1)  # (52,)
        X_list.append(delta_bs)

    X = np.asarray(X_list, dtype=np.float64).T  # shape (52, M)
    Y = np.asarray(Y_list, dtype=np.float64).T  # shape (N*3, M)

    M = X.shape[1]
    if M == 0:
        raise RuntimeError("No valid pairs after filtering")

    LOGGER.info("Solving ridge regression with %d samples", M)

    # Closed-form ridge solution: B = Y X^T (X X^T + λ I)^{-1}
    Gram = X @ X.T  # (52,52)
    reg = regularization
    Gram_reg = Gram + reg * np.eye(Gram.shape[0], dtype=np.float64)
    # Solve linear system for (X X^T + λI)^{-1} via Cholesky or np.linalg.solve
    try:
        inv_term = np.linalg.inv(Gram_reg)
    except np.linalg.LinAlgError:
        inv_term = np.linalg.pinv(Gram_reg)

    B_mat = Y @ X.T @ inv_term  # shape (N*3, 52)
    N3 = B_mat.shape[0]
    N = N3 // 3
    delta_B_landmarks = B_mat.T.reshape((NUM_BLENDSHAPES, N, 3)).astype(np.float32)  # (52, N, 3)

    metadata = {
        "num_pairs": int(M),
        "regularization": float(regularization),
        "landmark_count": int(N),
    }

    source = "landmark_space"

    # Optionally lift to template verts
    if lift_to_template_verts is not None:
        tpl_verts = np.asarray(lift_to_template_verts, dtype=np.float64)
        V_tpl = tpl_verts.shape[0]
        LOGGER.info("Lifting basis from %d landmarks to %d template verts using %s mapping", N, V_tpl, lift_method)
        # Build KDTree on landmark positions (use neutral average positions across samples? We'll use the neutral positions from the first pair)
        # For a robust mapping, compute the canonical landmark positions as the mean neutral landmarks across pairs
        mean_neutral = np.zeros((N, 3), dtype=np.float64)
        count = 0
        for neutral_rec, _ in pairs:
            lm_neu = np.asarray(neutral_rec["landmarks"], dtype=np.float64)
            if lm_neu.shape[0] == N:
                mean_neutral += lm_neu
                count += 1
        if count > 0:
            mean_neutral /= float(count)
        else:
            raise RuntimeError("Cannot compute mean neutral landmarks for lifting")

        # Build KDTree on mean_neutral (use x,y,z)
        tree = cKDTree(mean_neutral)
        dists, idxs = tree.query(tpl_verts, k=1)  # idxs: (V_tpl,)
        # Create lifted delta_B: for each blendshape k, for each template vertex v, copy delta from nearest landmark idxs[v]
        delta_B_lifted = np.zeros((NUM_BLENDSHAPES, V_tpl, 3), dtype=np.float32)
        for k in range(NUM_BLENDSHAPES):
            delta_B_lifted[k] = delta_B_landmarks[k, idxs, :]
        delta_B = delta_B_lifted
        source = "lifted_to_template_nearest"
        metadata["lift_method"] = lift_method
        metadata["lift_mapping"] = {"template_vertex_to_landmark_idx": idxs.tolist()}
    else:
        delta_B = delta_B_landmarks

    basis = ExpressionBasis(delta_B=delta_B, vertex_count=int(delta_B.shape[1]), source=source, metadata=metadata)

    if save_path is not None:
        basis.save(Path(save_path))
        LOGGER.info("Saved expression basis to %s", save_path)

    return basis


# -------------------------
# Apply expression
# -------------------------
def apply_expression(verts: np.ndarray, e_delta: np.ndarray, delta_B: np.ndarray) -> np.ndarray:
    """
    Apply an expression coefficient vector e_delta (K,) to verts (V,3) using delta_B (K,V,3).

    Returns deformed verts (V,3).

    Parameters
    ----------
    verts : np.ndarray
        (V,3) float
    e_delta : np.ndarray
        (K,) float coefficients (blendshape delta magnitudes)
    delta_B : np.ndarray
        (K,V,3) float basis

    Notes
    -----
    - K must match delta_B.shape[0], V must match delta_B.shape[1].
    - This is a linear combination: verts + sum_k e_k * delta_B[k]
    """
    verts = np.asarray(verts, dtype=np.float64)
    e_delta = np.asarray(e_delta, dtype=np.float64)
    delta_B = np.asarray(delta_B, dtype=np.float64)

    if delta_B.ndim != 3:
        raise ValueError("delta_B must be (K, V, 3)")
    K, V, C = delta_B.shape
    if C != 3:
        raise ValueError("delta_B last dimension must be 3")
    if verts.shape[0] != V:
        raise ValueError(f"verts length {verts.shape[0]} does not match delta_B vertex count {V}")
    if e_delta.shape[0] != K:
        raise ValueError(f"e_delta length {e_delta.shape[0]} does not match basis K={K}")

    # Weighted sum across K
    # result = verts + (e_delta[:,None,None] * delta_B).sum(axis=0)
    weighted = (e_delta.reshape(K, 1, 1) * delta_B).sum(axis=0)
    return (verts + weighted).astype(np.float32)


# -------------------------
# Convenience CLI-like function
# -------------------------
def build_and_save_expression_basis(
    processed_dir: Path,
    out_path: Path,
    regularization: float = DEFAULT_REGULARIZATION,
    template_verts: Optional[np.ndarray] = None,
) -> ExpressionBasis:
    """
    Convenience wrapper: build basis and save to out_path (.npz + .meta.pkl).
    """
    basis = build_expression_basis_from_dir(processed_dir, regularization=regularization, lift_to_template_verts=template_verts, save_path=out_path)
    return basis


# -------------------------
# Example usage (not executed on import)
# -------------------------
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Build expression basis from processed .npz pairs")
    parser.add_argument("--processed_dir", type=Path, required=True, help="Directory with processed .npz files")
    parser.add_argument("--out", type=Path, default=Path("data/models/expression_basis.npz"), help="Output path for basis (.npz)")
    parser.add_argument("--reg", type=float, default=DEFAULT_REGULARIZATION, help="Ridge regularization lambda")
    parser.add_argument("--template_obj_verts", type=str, default="", help="Optional: path to template verts .npy (V,3) to lift basis")
    args = parser.parse_args()

    tpl_verts = None
    if args.template_obj_verts:
        try:
            tpl_verts = np.load(args.template_obj_verts)
        except Exception as exc:
            LOGGER.error("Failed to load template verts: %s", exc)
            sys.exit(1)

    basis = build_and_save_expression_basis(args.processed_dir, args.out, regularization=args.reg, template_verts=tpl_verts)
    print("Built basis:", basis.delta_B.shape, "source:", basis.source, "metadata:", basis.metadata)
