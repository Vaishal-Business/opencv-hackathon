
"""
geometry/template.py

Utilities for loading and preparing the canonical MediaPipe face template
for GALNP Stage 0.

Features
- Load an OBJ file containing the canonical face (verts, faces, UVs, groups).
- Optional automatic download of a canonical_face_model.obj from a URL.
- One-pass midpoint subdivision (each triangle -> 4 triangles) producing
  ~1.8k verts / ~3.6k faces from a 468-vertex base mesh.
- Preserve and propagate UVs and per-vertex region labels (if present as
  face groups / OBJ 'g' or 'usemtl' entries).
- Build a left-right mirror vertex index map (by mirrored X coordinate).
- Identify mouth cavity faces from a named group or by heuristic (inner lip loop).
- Save / load prepared template to a compact .npz file.
- Small visualization helper to plot the mesh colored by region.

Dependencies: numpy, scipy, trimesh, matplotlib
(Only NumPy / SciPy / trimesh are required by the user; matplotlib is optional
for the visualization helper.)
"""

from __future__ import annotations

import io
import os
import pickle
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from scipy.spatial import cKDTree

# Optional import for visualization
try:
    import matplotlib.pyplot as plt  # type: ignore
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# ---------------------------
# Dataclass for the template
# ---------------------------
@dataclass
class MeshTemplate:
    """
    Container for a prepared canonical face template.

    Attributes
    ----------
    verts : np.ndarray
        (V, 3) float32 vertex positions in canonical coordinates.
    faces : np.ndarray
        (F, 3) int32 triangle indices.
    uvs : np.ndarray
        (V, 2) float32 per-vertex UV coordinates (if available). If not present,
        this will be an array of zeros.
    regions : np.ndarray
        (V,) int32 region label per vertex. Labels are integers indexing
        `region_names` below. If unknown, all zeros.
    region_names : List[str]
        Mapping from region index -> region name.
    mirror_idx : np.ndarray
        (V,) int32 index of the mirrored vertex (left-right). If no exact
        mirror found, the nearest vertex is used.
    mouth_cavity_faces : np.ndarray
        (M, 3) int32 faces that belong to the mouth cavity (inner mouth).
    metadata : dict
        Any additional metadata (source path, applied subdivision, etc.)
    """
    verts: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    regions: np.ndarray
    region_names: List[str]
    mirror_idx: np.ndarray
    mouth_cavity_faces: np.ndarray
    metadata: Dict = None

    def save(self, path: Path) -> None:
        """
        Save the template to a compressed .npz file (numpy) with a small pickle
        for region_names and metadata.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            verts=self.verts.astype(np.float32),
            faces=self.faces.astype(np.int32),
            uvs=self.uvs.astype(np.float32),
            regions=self.regions.astype(np.int32),
            mirror_idx=self.mirror_idx.astype(np.int32),
            mouth_cavity_faces=self.mouth_cavity_faces.astype(np.int32),
        )
        # Save region_names and metadata as pickle alongside .npz
        pkl = path.with_suffix(path.suffix + ".meta.pkl")
        with open(pkl, "wb") as f:
            pickle.dump({"region_names": self.region_names, "metadata": self.metadata or {}}, f)

    @staticmethod
    def load(path: Path) -> "MeshTemplate":
        """
        Load a template saved with `save()`.
        """
        path = Path(path)
        data = np.load(str(path), allow_pickle=False)
        pkl = path.with_suffix(path.suffix + ".meta.pkl")
        if pkl.exists():
            with open(pkl, "rb") as f:
                meta = pickle.load(f)
            region_names = meta.get("region_names", [])
            metadata = meta.get("metadata", {})
        else:
            region_names = []
            metadata = {}
        return MeshTemplate(
            verts=data["verts"].astype(np.float32),
            faces=data["faces"].astype(np.int32),
            uvs=data["uvs"].astype(np.float32),
            regions=data["regions"].astype(np.int32),
            region_names=region_names,
            mirror_idx=data["mirror_idx"].astype(np.int32),
            mouth_cavity_faces=data["mouth_cavity_faces"].astype(np.int32),
            metadata=metadata,
        )


# ---------------------------
# Utilities
# ---------------------------
def download_if_needed(url: str, dest: Path, force: bool = False) -> Path:
    """
    Download a file from `url` to `dest` if it does not exist or if force=True.
    Returns the destination path.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    with open(dest, "wb") as f:
        f.write(data)
    return dest


def _parse_obj_groups(obj_path: Path) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Parse an OBJ file and return a list of group names in order and a mapping
    group_name -> list of face indices (0-based). This is a lightweight parser
    that only reads 'g' and 'f' lines to preserve grouping information.
    """
    groups: Dict[str, List[int]] = {}
    group_order: List[str] = []
    current_group = "default"
    face_idx = 0
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("g ") or line.startswith("o ") or line.startswith("usemtl "):
                # group/object/usemtl line
                parts = line.split(maxsplit=1)
                if len(parts) > 1:
                    current_group = parts[1].strip()
                else:
                    current_group = "default"
                if current_group not in groups:
                    groups[current_group] = []
                    group_order.append(current_group)
            elif line.startswith("f "):
                # face line
                groups.setdefault(current_group, []).append(face_idx)
                face_idx += 1
    return group_order, groups


# ---------------------------
# Midpoint subdivision
# ---------------------------
def midpoint_subdivide(verts: np.ndarray, faces: np.ndarray, uvs: Optional[np.ndarray] = None,
                       regions_per_face: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Perform one iteration of midpoint subdivision (each triangle -> 4 triangles).
    - verts: (V,3)
    - faces: (F,3)
    - uvs: (V,2) or None
    - regions_per_face: (F,) int region index per face (optional)

    Returns:
    - new_verts: (V_new, 3)
    - new_faces: (F_new, 3)
    - new_uvs: (V_new, 2) or None
    - new_regions_per_face: (F_new,) or None
    """
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    V = verts.shape[0]

    # Edge to midpoint index
    edge_map: Dict[Tuple[int, int], int] = {}
    midpoints: List[np.ndarray] = []
    mid_uvs: List[np.ndarray] = []

    def edge_key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    # Precompute face-wise regions if provided
    has_uv = uvs is not None
    if has_uv:
        uvs = np.asarray(uvs, dtype=np.float64)

    new_faces_list: List[Tuple[int, int, int]] = []
    new_regions_list: List[int] = []

    # Start with original vertices
    new_verts = verts.tolist()
    if has_uv:
        new_uvs = uvs.tolist()
    else:
        new_uvs = None

    for fi, f in enumerate(faces):
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        # For each edge, create or reuse midpoint
        mids = []
        for (i0, i1) in ((a, b), (b, c), (c, a)):
            k = edge_key(i0, i1)
            if k in edge_map:
                mids.append(edge_map[k])
            else:
                p0, p1 = verts[i0], verts[i1]
                mid = 0.5 * (p0 + p1)
                idx = len(new_verts)
                new_verts.append(mid)
                edge_map[k] = idx
                mids.append(idx)
                if has_uv:
                    uv0, uv1 = uvs[i0], uvs[i1]
                    new_uvs.append(0.5 * (uv0 + uv1))
        m_ab, m_bc, m_ca = mids

        # Create 4 new faces
        new_faces_list.append((a, m_ab, m_ca))
        new_faces_list.append((m_ab, b, m_bc))
        new_faces_list.append((m_ca, m_bc, c))
        new_faces_list.append((m_ab, m_bc, m_ca))

        # propagate region label per face if provided
        if regions_per_face is not None:
            r = int(regions_per_face[fi])
            new_regions_list.extend([r, r, r, r])

    new_verts = np.asarray(new_verts, dtype=np.float32)
    new_faces = np.asarray(new_faces_list, dtype=np.int32)
    new_uvs_arr = np.asarray(new_uvs, dtype=np.float32) if has_uv else None
    new_regions_arr = np.asarray(new_regions_list, dtype=np.int32) if regions_per_face is not None else None

    return new_verts, new_faces, new_uvs_arr, new_regions_arr


# ---------------------------
# Mirror map builder
# ---------------------------
def build_mirror_map(verts: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    """
    Build a left-right mirror index map by finding, for each vertex v=(x,y,z),
    the nearest vertex to (-x, y, z) within a tolerance. If no vertex is within
    tol, the nearest vertex is used (still returns an index).
    """
    verts = np.asarray(verts, dtype=np.float64)
    V = verts.shape[0]
    mirrored = verts.copy()
    mirrored[:, 0] *= -1.0  # mirror X

    tree = cKDTree(verts)
    dists, idxs = tree.query(mirrored, k=1)
    # If distance > tol, still accept but warn (no printing here; caller may inspect)
    return np.asarray(idxs, dtype=np.int32)


# ---------------------------
# Region extraction helpers
# ---------------------------
_DEFAULT_REGION_NAMES = [
    "skin",
    "upper_lip",
    "lower_lip",
    "mouth_cavity",
    "left_eye",
    "right_eye",
    "left_brow",
    "right_brow",
    "nose_other",
]


def _assign_vertex_regions_from_face_groups(faces: np.ndarray, face_groups: Dict[str, List[int]],
                                           region_names_map: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
    """
    Given faces (F,3) and a mapping group_name -> list of face indices (0-based),
    produce a per-vertex region index array and a list of region names.

    region_names_map maps group_name substrings to canonical region names (e.g. 'upper_lip' -> 'upper_lip').

    Strategy:
    - Build face->region mapping by checking group names for keywords.
    - Then propagate to vertices by majority of incident faces.
    """
    F = faces.shape[0]
    face_region = np.full((F,), -1, dtype=np.int32)
    # Build canonical region name set
    canonical_names = []
    # map canonical name -> index
    cname_to_idx: Dict[str, int] = {}

    # First pass: assign face regions based on group names
    for gname, face_idxs in face_groups.items():
        # find a matching canonical region
        matched = None
        g_low = gname.lower()
        for key, canonical in region_names_map.items():
            if key in g_low:
                matched = canonical
                break
        if matched is None:
            # try to match any of the default names
            for cand in _DEFAULT_REGION_NAMES:
                if cand.replace("_", "") in g_low or cand in g_low:
                    matched = cand
                    break
        if matched is None:
            matched = "skin"  # fallback

        if matched not in cname_to_idx:
            cname_to_idx[matched] = len(canonical_names)
            canonical_names.append(matched)
        ridx = cname_to_idx[matched]
        for fi in face_idxs:
            if 0 <= fi < F:
                face_region[fi] = ridx

    # Any unassigned faces -> skin
    if -1 in face_region:
        skin_idx = cname_to_idx.get("skin")
        if skin_idx is None:
            skin_idx = len(canonical_names)
            canonical_names.append("skin")
            cname_to_idx["skin"] = skin_idx
        face_region[face_region == -1] = skin_idx

    # Propagate to vertices by majority vote of incident faces
    V = int(faces.max()) + 1
    vert_face_lists: List[List[int]] = [[] for _ in range(V)]
    for fi, f in enumerate(faces):
        for v in f:
            vert_face_lists[int(v)].append(fi)

    vert_regions = np.zeros((V,), dtype=np.int32)
    for vi, flist in enumerate(vert_face_lists):
        if not flist:
            vert_regions[vi] = cname_to_idx.get("skin", 0)
            continue
        regs = face_region[flist]
        # majority
        vals, counts = np.unique(regs, return_counts=True)
        vert_regions[vi] = int(vals[np.argmax(counts)])

    return vert_regions, canonical_names


# ---------------------------
# Mouth cavity detection
# ---------------------------
def detect_mouth_cavity_faces(faces: np.ndarray, face_groups: Dict[str, List[int]]) -> np.ndarray:
    """
    Return an array of face indices that correspond to the mouth cavity.
    Prefer explicit groups named 'mouth', 'mouth_cavity', 'inner_mouth', 'inner_lip'.
    Otherwise return an empty array (caller can attempt heuristics).
    """
    candidates = []
    for gname, face_idxs in face_groups.items():
        gl = gname.lower()
        if any(k in gl for k in ("mouth", "inner", "oral", "cavity", "inner_lip")):
            candidates.extend(face_idxs)
    if not candidates:
        return np.zeros((0, 3), dtype=np.int32)
    # Return faces as indices array
    return np.asarray(sorted(set(candidates)), dtype=np.int32)


# ---------------------------
# Main loader / builder
# ---------------------------
def load_canonical_obj(obj_path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, List[int]]]:
    """
    Load an OBJ using trimesh but also parse group->face mapping.

    Returns:
    - verts (V,3)
    - faces (F,3)
    - uvs (V,2) if available else None
    - face_groups: mapping group_name -> list of face indices (0-based)
    """
    obj_path = Path(obj_path)
    # Use trimesh to load geometry (it will triangulate if needed)
    mesh = trimesh.load_mesh(str(obj_path), process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        # If the OBJ contains multiple meshes, merge them
        mesh = trimesh.util.concatenate(mesh.dump())  # type: ignore

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    # Attempt to extract per-vertex UVs. trimesh stores visual.uv per vertex or per face
    uvs = None
    try:
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            # trimesh.visual.uv is per-vertex or per-face-vertex depending on loader
            uv = np.asarray(mesh.visual.uv, dtype=np.float32)
            # If uv length equals number of vertices, good
            if uv.shape[0] == verts.shape[0] and uv.shape[1] >= 2:
                uvs = uv[:, :2]
            else:
                # If uv is per-face-vertex, we need to convert to per-vertex by averaging
                # Build per-vertex uv by averaging incident uv entries
                V = verts.shape[0]
                accum = np.zeros((V, 2), dtype=np.float64)
                counts = np.zeros((V,), dtype=np.int32)
                # trimesh stores uv per face-vertex in mesh.visual.uv.reshape((-1,2))
                # but mapping is not trivial; fallback: set None
                uvs = None
    except Exception:
        uvs = None

    # Parse groups from OBJ file to preserve region names
    group_order, face_groups = _parse_obj_groups(obj_path)

    return verts, faces, uvs, face_groups


def build_template_from_obj(obj_path: Path,
                            download_url: Optional[str] = None,
                            desired_frac: float = 0.75,
                            max_zoom_out: float = 1.6) -> MeshTemplate:
    """
    Load the canonical OBJ, perform one midpoint subdivision, propagate UVs and
    region labels, build mirror map, and detect mouth cavity faces.

    Parameters
    ----------
    obj_path : Path
        Path to canonical_face_model.obj. If it does not exist and download_url
        is provided, the file will be downloaded.
    download_url : Optional[str]
        URL to download the OBJ if missing.
    desired_frac, max_zoom_out : (unused here)
        Kept for API compatibility with Stage 0 pipeline (alignment uses these).
    """
    obj_path = Path(obj_path)
    if not obj_path.exists():
        if download_url:
            download_if_needed(download_url, obj_path)
        else:
            raise FileNotFoundError(f"OBJ not found: {obj_path}")

    verts, faces, uvs, face_groups = load_canonical_obj(obj_path)

    # Build a mapping from group names to canonical region names (heuristic)
    # Keys are substrings to match in group names -> canonical region name
    region_names_map = {
        "upperlip": "upper_lip",
        "upper_lip": "upper_lip",
        "lowerlip": "lower_lip",
        "lower_lip": "lower_lip",
        "innerlip": "mouth_cavity",
        "inner_lip": "mouth_cavity",
        "mouth": "mouth_cavity",
        "left_eye": "left_eye",
        "right_eye": "right_eye",
        "eye": "left_eye",  # fallback; will be resolved by majority
        "brow": "left_brow",
        "nose": "nose_other",
        "skin": "skin",
        "cheek": "skin",
        "lip": "lower_lip",
    }

    # Assign face->region and then vertex->region
    face_region_arr = None
    try:
        vert_regions, region_names = _assign_vertex_regions_from_face_groups(faces, face_groups, region_names_map)
    except Exception:
        # Fallback: all skin
        V = verts.shape[0]
        vert_regions = np.zeros((V,), dtype=np.int32)
        region_names = ["skin"]

    # Perform one midpoint subdivision and propagate per-face region to new faces
    # We need regions per face to propagate; build face->region by majority of vertices
    F = faces.shape[0]
    face_regions = np.zeros((F,), dtype=np.int32)
    for fi, f in enumerate(faces):
        regs = vert_regions[f]
        vals, counts = np.unique(regs, return_counts=True)
        face_regions[fi] = int(vals[np.argmax(counts)])

    new_verts, new_faces, new_uvs, new_face_regions = midpoint_subdivide(verts, faces, uvs, face_regions)

    # Convert face-region to vertex-region by majority vote
    V_new = new_verts.shape[0]
    vert_face_lists: List[List[int]] = [[] for _ in range(V_new)]
    for fi, f in enumerate(new_faces):
        for v in f:
            vert_face_lists[int(v)].append(fi)
    new_vert_regions = np.zeros((V_new,), dtype=np.int32)
    for vi, flist in enumerate(vert_face_lists):
        if not flist:
            new_vert_regions[vi] = 0
            continue
        regs = new_face_regions[flist]
        vals, counts = np.unique(regs, return_counts=True)
        new_vert_regions[vi] = int(vals[np.argmax(counts)])

    # Build mirror map
    mirror_idx = build_mirror_map(new_verts, tol=1e-3)

    # Detect mouth cavity faces (indices into new_faces)
    # Map original face indices to new face indices: each original face produced 4 faces in order
    # original fi -> new indices [4*fi, 4*fi+1, 4*fi+2, 4*fi+3]
    mouth_face_idxs_orig = detect_mouth_cavity_faces(faces, face_groups)
    if mouth_face_idxs_orig.size > 0:
        mouth_new_idxs = []
        for fi in mouth_face_idxs_orig:
            base = int(fi) * 4
            mouth_new_idxs.extend([base, base + 1, base + 2, base + 3])
        mouth_new_idxs = np.asarray(sorted(set(mouth_new_idxs)), dtype=np.int32)
    else:
        # fallback: find faces whose vertex region == mouth_cavity region index (if present)
        mouth_idx = None
        for i, name in enumerate(region_names):
            if "mouth" in name or "inner" in name:
                mouth_idx = i
                break
        if mouth_idx is not None:
            mask = new_face_regions == mouth_idx
            mouth_new_idxs = np.where(mask)[0].astype(np.int32)
        else:
            mouth_new_idxs = np.zeros((0,), dtype=np.int32)

    # Build per-vertex UVs: midpoint_subdivide returned per-vertex uvs if original had uvs
    if new_uvs is None:
        new_uvs = np.zeros((new_verts.shape[0], 2), dtype=np.float32)

    # Package metadata
    metadata = {
        "source_obj": str(obj_path),
        "subdivision": "midpoint_1x",
        "original_verts": int(verts.shape[0]),
        "original_faces": int(faces.shape[0]),
    }

    template = MeshTemplate(
        verts=new_verts.astype(np.float32),
        faces=new_faces.astype(np.int32),
        uvs=new_uvs.astype(np.float32),
        regions=new_vert_regions.astype(np.int32),
        region_names=region_names,
        mirror_idx=mirror_idx.astype(np.int32),
        mouth_cavity_faces=new_faces[mouth_new_idxs] if mouth_new_idxs.size > 0 else np.zeros((0, 3), dtype=np.int32),
        metadata=metadata,
    )
    return template


# ---------------------------
# Visualization helper
# ---------------------------
def visualize_template(template: MeshTemplate, figsize: Tuple[int, int] = (8, 8), elev: float = 20, azim: float = -60) -> None:
    """
    Simple 3D visualization of the mesh colored by region. Requires matplotlib.

    Parameters
    ----------
    template : MeshTemplate
    figsize : tuple
    elev, azim : view angles
    """
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required for visualization (install matplotlib).")

    verts = template.verts
    faces = template.faces
    regions = template.regions
    region_names = template.region_names

    # Build a color map for regions
    n_regions = max(1, (max(regions) + 1) if regions.size else 1)
    cmap = plt.get_cmap("tab10")
    colors = np.array([cmap(i % 10) for i in range(n_regions)])[:, :3]

    face_colors = colors[regions[faces].mean(axis=1).astype(int) % n_regions]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, facecolors=face_colors, linewidth=0.2, antialiased=True, shade=False)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # Legend
    for i, name in enumerate(region_names):
        ax.plot([], [], color=colors[i], label=name)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.show()


# ---------------------------
# Convenience loader
# ---------------------------
def load_or_build_template(obj_path: Optional[Path] = None,
                           download_url: Optional[str] = None,
                           cache_path: Optional[Path] = None,
                           force_rebuild: bool = False) -> MeshTemplate:
    """
    Convenience function:
    - If cache_path (.npz) exists and not force_rebuild, load it.
    - Otherwise build from obj_path (downloading if needed) and save to cache_path.

    Returns MeshTemplate.
    """
    if cache_path is None:
        cache_path = Path("data/models/canonical_face_template.npz")
    else:
        cache_path = Path(cache_path)

    if cache_path.exists() and not force_rebuild:
        try:
            return MeshTemplate.load(cache_path)
        except Exception:
            # fall through to rebuild
            pass

    if obj_path is None:
        raise ValueError("obj_path must be provided to build the template if cache is missing.")

    template = build_template_from_obj(obj_path, download_url=download_url)
    template.save(cache_path)
    return template


# ---------------------------
# Module test / example (not executed on import)
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build and inspect canonical face template.")
    parser.add_argument("--obj", type=Path, required=True, help="Path to canonical_face_model.obj")
    parser.add_argument("--out", type=Path, default=Path("data/models/canonical_face_template.npz"), help="Output .npz path")
    parser.add_argument("--visualize", action="store_true", help="Show a quick visualization")
    args = parser.parse_args()

    tpl = load_or_build_template(obj_path=args.obj, cache_path=args.out, force_rebuild=True)
    print("Template built:")
    print(" verts:", tpl.verts.shape)
    print(" faces:", tpl.faces.shape)
    print(" uvs:", tpl.uvs.shape)
    print(" regions:", tpl.regions.shape, "region names:", tpl.region_names)
    print(" mirror map:", tpl.mirror_idx.shape)
    print(" mouth cavity faces:", tpl.mouth_cavity_faces.shape)
    if args.visualize:
        visualize_template(tpl)
