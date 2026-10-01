#!/usr/bin/env python3
"""
prepare.py - GALNP Stage 0: face preprocessing (MediaPipe + OpenCV + NumPy only).

For every frontal image (neutral_front/, smiling_front/) this script:
  1. runs the MediaPipe Tasks Vision FaceLandmarker (478 landmarks, 52 blendshapes,
     4x4 facial transformation matrix),
  2. aligns the face to a canonical 512x512 template with a 5-point similarity transform,
     using a SINGLE SHARED ALIGNMENT PER IDENTITY (derived from the neutral image, or
     from the average of the identity's images if no neutral is available) so that all
     expressions of the same face live in the same coordinate frame — this is required
     for correspondence / flow / stretch / warping to be meaningful downstream,
  3. saves the aligned RGB crop (JPEG, quality 95) and a metadata .npz file.

Auto-zoom
---------
A per-image "zoom out" step makes the face (forehead -> chin) occupy a configurable
fraction of the vertical crop (recommended 0.70-0.80). Scaling is clamped and only
zoom-out is performed. When --consistent_alignment is on (default), the zoom scale is
computed once per identity from the neutral image and reused for every expression.
"""

from __future__ import annotations

import argparse
import csv
import logging
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tqdm import tqdm

LOGGER = logging.getLogger("prepare")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker_v2_with_blendshapes/float16/1/face_landmarker_v2_with_blendshapes.task"
)
DEFAULT_MODEL_PATH = Path("data/models/face_landmarker_v2_with_blendshapes.task")

NUM_LANDMARKS = 478
NUM_BLENDSHAPES = 52

EXPRESSION_DIRS: Dict[str, str] = {
    "neutral_front": "neutral",
    "smiling_front": "smiling",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# 5 MediaPipe landmark indices, ordered left-to-right as seen in the image.
FIVE_POINT_IDX: Tuple[int, ...] = (33, 263, 1, 61, 291)

FOREHEAD_TOP_IDX = 10
CHIN_IDX = 152

CANONICAL_TEMPLATE_NORM = np.array(
    [
        [0.305, 0.39],  # eye outer corner (image-left)
        [0.695, 0.39],  # eye outer corner (image-right)
        [0.500, 0.55],  # nose tip
        [0.381, 0.68],  # mouth corner (image-left)
        [0.619, 0.68],  # mouth corner (image-right)
    ],
    dtype=np.float64,
)


class CsvMeta(NamedTuple):
    age: float
    gender: str
    ethnicity: str


EMPTY_META = CsvMeta(age=float("nan"), gender="", ethnicity="")


class Outcome(NamedTuple):
    success: bool
    reason: str
    extent: Optional[Tuple[float, float]] = None


class FaceResult(NamedTuple):
    landmarks: np.ndarray
    blendshapes: np.ndarray
    pose: np.ndarray


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def ensure_model(model_path: Path) -> Path:
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading FaceLandmarker model to %s ...", model_path)
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as exc:  # noqa: BLE001
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(
            f"Could not download the model from {MODEL_URL}.\n"
            f"Download it manually and pass --model_path. Error: {exc}"
        ) from exc
    return model_path


def load_landmarker(model_path: Path, min_confidence: float) -> vision.FaceLandmarker:
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=min_confidence,
        min_face_presence_confidence=min_confidence,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


# --------------------------------------------------------------------------- #
# CSV metadata (robust to multi-part ids like "001_03")
# --------------------------------------------------------------------------- #
def _normalize_id(raw: str) -> str:
    """
    Canonicalize an id for lookup. Strips a trailing image extension and leading
    zeros from every numeric underscore-separated component.

    '001'      -> '1'
    '001_03'   -> '1_3'
    '003_04.png' -> '3_4'
    'abc_07'   -> 'abc_7'
    """
    s = str(raw).strip()
    if s.lower().endswith(tuple(IMAGE_EXTENSIONS)):
        s = Path(s).stem
    parts = s.split("_")
    return "_".join(str(int(p)) if p.isdigit() else p for p in parts)


def find_csv(input_dir: Path, csv_path: Optional[Path]) -> Optional[Path]:
    if csv_path is not None:
        return csv_path if csv_path.exists() else None
    preferred = input_dir / "london_faces_ratings.csv"
    if preferred.exists():
        return preferred
    candidates = sorted(input_dir.glob("*.csv"))
    return candidates[0] if candidates else None


def load_metadata(csv_path: Optional[Path]) -> Dict[str, CsvMeta]:
    if csv_path is None:
        LOGGER.warning("No CSV found; age/gender/ethnicity will be left empty.")
        return {}

    table: Dict[str, CsvMeta] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "face_id" not in reader.fieldnames:
            LOGGER.warning("CSV %s has no 'face_id' column; metadata ignored.", csv_path)
            return {}
        for row in reader:
            face_id = (row.get("face_id") or "").strip()
            if not face_id:
                continue
            try:
                age = float((row.get("face_age") or "").strip())
            except ValueError:
                age = float("nan")
            table[_normalize_id(face_id)] = CsvMeta(
                age=age,
                gender=(row.get("face_gender") or "").strip(),
                ethnicity=(row.get("face_eth") or "").strip(),
            )
    LOGGER.info("Loaded metadata for %d faces from %s", len(table), csv_path)
    return table


def lookup_meta(table: Dict[str, CsvMeta], face_id: str) -> Optional[CsvMeta]:
    """Try the normalized id, then a leading-numeric-component fallback."""
    if not table:
        return None
    key = _normalize_id(face_id)
    if key in table:
        return table[key]
    parts = str(face_id).split("_")
    if len(parts) > 1 and parts[0].isdigit():
        k2 = str(int(parts[0]))
        if k2 in table:
            return table[k2]
    return None


# --------------------------------------------------------------------------- #
# Landmark detection
# --------------------------------------------------------------------------- #
def detect_face(landmarker: vision.FaceLandmarker, rgb: np.ndarray) -> Optional[FaceResult]:
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None
    if not result.face_blendshapes or not result.facial_transformation_matrixes:
        return None

    lm = result.face_landmarks[0]
    if len(lm) < NUM_LANDMARKS:
        return None
    landmarks = np.array([[p.x, p.y, p.z] for p in lm[:NUM_LANDMARKS]], dtype=np.float32)

    cats = sorted(result.face_blendshapes[0], key=lambda c: c.index)
    blendshapes = np.array([c.score for c in cats], dtype=np.float32)
    if blendshapes.shape[0] != NUM_BLENDSHAPES:
        return None

    pose = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float32).reshape(4, 4)
    return FaceResult(landmarks, blendshapes, pose)


def compute_confidence(landmarks_norm: np.ndarray) -> float:
    xy = landmarks_norm[:, :2]
    inside = (xy[:, 0] >= 0.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= 0.0) & (xy[:, 1] <= 1.0)
    return float(inside.mean())


def get_5point_landmarks(landmarks_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = landmarks_norm[list(FIVE_POINT_IDX), :2].astype(np.float64)
    pts[:, 0] *= width
    pts[:, 1] *= height
    return pts


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def estimate_similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares 2D similarity (Umeyama, 1991): src -> dst, both (N,2). Returns (2,3)."""
    n = src.shape[0]
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean

    cov = dst_c.T @ src_c / n
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1.0
    rot = u @ np.diag(d) @ vt

    var_src = (src_c ** 2).sum() / n
    scale = float((s * d).sum() / var_src)
    trans = dst_mean - scale * rot @ src_mean
    return np.hstack([scale * rot, trans[:, None]])


def transform_landmarks(landmarks_norm: np.ndarray, matrix: np.ndarray,
                        width: int, height: int) -> np.ndarray:
    """Map normalized landmarks into aligned-crop pixel coordinates, (478, 3) float32."""
    xy = landmarks_norm[:, :2].astype(np.float64) * np.array([width, height])
    xy_crop = xy @ matrix[:, :2].T + matrix[:, 2]
    scale = float(np.sqrt(abs(np.linalg.det(matrix[:, :2]))))
    z_crop = landmarks_norm[:, 2].astype(np.float64) * width * scale
    return np.concatenate([xy_crop, z_crop[:, None]], axis=1).astype(np.float32)


def face_vertical_extent(landmarks_crop: np.ndarray, target_size: int) -> Tuple[float, float]:
    return (
        float(landmarks_crop[FOREHEAD_TOP_IDX, 1]) / target_size,
        float(landmarks_crop[CHIN_IDX, 1]) / target_size,
    )


def compute_zoom_scale_for_face_from_crop(
    landmarks_crop: np.ndarray,
    target_size: int,
    desired_frac: float = 0.75,
    min_scale: float = 1.0,
    max_scale: float = 2.0,
) -> float:
    forehead_y = float(landmarks_crop[FOREHEAD_TOP_IDX, 1])
    chin_y = float(landmarks_crop[CHIN_IDX, 1])
    current_frac = (chin_y - forehead_y) / float(target_size)
    if current_frac <= 0 or not np.isfinite(current_frac):
        return 1.0
    scale = current_frac / desired_frac
    if scale < min_scale:
        scale = min_scale
    if scale > max_scale:
        scale = max_scale
    return float(scale)


def compute_shared_alignment(
    landmarks_norm: np.ndarray,
    width: int,
    height: int,
    target_size: int,
    desired_frac: float = 0.75,
    max_zoom_out: float = 1.6,
) -> Tuple[np.ndarray, float]:
    """
    Given one image's normalized landmarks, compute a single (matrix, scale) that
    maps original-image pixels -> canonical crop pixels (512x512 by default).

    This is the function used to derive the SHARED per-identity alignment. The
    matrix and scale depend only on the reference image's landmarks.
    """
    src = get_5point_landmarks(landmarks_norm, width, height)
    dst = CANONICAL_TEMPLATE_NORM * target_size
    matrix0 = estimate_similarity_transform(src, dst)

    lm_crop = transform_landmarks(landmarks_norm, matrix0, width, height)
    scale = compute_zoom_scale_for_face_from_crop(
        lm_crop, target_size, desired_frac, min_scale=1.0, max_scale=max_zoom_out
    )

    if scale == 1.0:
        return matrix0, 1.0

    center = np.array([target_size / 2.0, target_size / 2.0], dtype=np.float64)
    dst_scaled = center + (dst - center) / scale
    matrix1 = estimate_similarity_transform(src, dst_scaled)
    return matrix1, scale


def warp_to_crop(image_rgb: np.ndarray, matrix: np.ndarray, target_size: int) -> np.ndarray:
    return cv2.warpAffine(
        image_rgb,
        matrix,
        (target_size, target_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


# --------------------------------------------------------------------------- #
# Detection-only pass
# --------------------------------------------------------------------------- #
def detect_only(
    image_path: Path,
    landmarker: vision.FaceLandmarker,
    min_confidence: float,
) -> Tuple[Optional[Dict], str]:
    """
    Read an image, run the landmarker, and return a detection dict (or None, reason).
    Does NOT write anything. Used in pass 1.
    """
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None, "unreadable_image"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    face = detect_face(landmarker, rgb)
    if face is None:
        return None, "no_face_detected"

    confidence = compute_confidence(face.landmarks)
    if confidence < min_confidence:
        return None, "low_confidence"

    five = get_5point_landmarks(face.landmarks, w, h)
    if not np.isfinite(five).all():
        return None, "invalid_landmarks"

    return {
        "landmarks": face.landmarks,
        "blendshapes": face.blendshapes,
        "pose": face.pose,
        "confidence": confidence,
        "width": w,
        "height": h,
    }, "ok"


# --------------------------------------------------------------------------- #
# Dataset traversal
# --------------------------------------------------------------------------- #
def collect_images(input_dir: Path) -> List[Tuple[Path, str, str]]:
    items: List[Tuple[Path, str, str]] = []
    for folder, expression in EXPRESSION_DIRS.items():
        folder_path = input_dir / folder
        if not folder_path.is_dir():
            LOGGER.warning("Folder not found, skipping: %s", folder_path)
            continue
        files = sorted(p for p in folder_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        LOGGER.info("Found %d images in %s", len(files), folder_path)
        items.extend((p, expression, p.stem) for p in files)
    return items


# --------------------------------------------------------------------------- #
# Main run
# --------------------------------------------------------------------------- #
def run(
    input_dir: Path,
    output_dir: Path,
    csv_path: Optional[Path],
    target_size: int,
    min_confidence: float,
    model_path: Path = DEFAULT_MODEL_PATH,
    overwrite: bool = False,
    desired_frac: float = 0.75,
    max_zoom_out: float = 1.6,
    consistent_alignment: bool = True,
) -> Counter:
    crops_dir = output_dir / "crops"
    meta_dir = output_dir / "meta"
    crops_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(find_csv(input_dir, csv_path))
    items = collect_images(input_dir)
    if not items:
        raise FileNotFoundError(
            f"No images found. Expected {list(EXPRESSION_DIRS)} inside {input_dir}"
        )

    # Group items by face_id so we can compute a shared alignment per identity.
    by_id: Dict[str, List[Tuple[Path, str]]] = {}
    for image_path, expression, face_id in items:
        by_id.setdefault(face_id, []).append((image_path, expression))

    landmarker = load_landmarker(ensure_model(model_path), min_confidence)
    stats: Counter = Counter()
    missing_csv = 0
    extents: List[Tuple[float, float]] = []
    scales: List[float] = []

    try:
        # -------------------------------------------------------------------
        # Pass 1: detect landmarks on all images that need processing.
        # -------------------------------------------------------------------
        detections: Dict[Tuple[str, str], object] = {}  # value: dict | {"error": str} | "skip"
        for face_id, entries in tqdm(by_id.items(), desc="Detecting", unit="id"):
            for image_path, expression in entries:
                stem = f"{face_id}_{expression}"
                if (
                    not overwrite
                    and (crops_dir / f"{stem}.jpg").exists()
                    and (meta_dir / f"{stem}.npz").exists()
                ):
                    detections[(face_id, expression)] = "skip"
                    continue
                det, reason = detect_only(image_path, landmarker, min_confidence)
                if det is None:
                    detections[(face_id, expression)] = {"error": reason}
                else:
                    detections[(face_id, expression)] = det

        # -------------------------------------------------------------------
        # Compute shared alignment per identity (prefer neutral, else first
        # available expression for that identity).
        # -------------------------------------------------------------------
        alignments: Dict[str, Tuple[Optional[np.ndarray], float]] = {}
        if consistent_alignment:
            for face_id, entries in by_id.items():
                neutral_det = None
                first_det = None
                for image_path, expression in entries:
                    d = detections.get((face_id, expression))
                    if not isinstance(d, dict) or "error" in d:
                        continue
                    if first_det is None:
                        first_det = d
                    if expression == "neutral":
                        neutral_det = d
                        break
                ref = neutral_det if neutral_det is not None else first_det
                if ref is not None:
                    matrix, scale = compute_shared_alignment(
                        ref["landmarks"], ref["width"], ref["height"],
                        target_size, desired_frac, max_zoom_out,
                    )
                    alignments[face_id] = (matrix, scale)
                    LOGGER.debug(
                        "Shared alignment for identity %s: scale=%.3f",
                        face_id, scale,
                    )

        # -------------------------------------------------------------------
        # Pass 2: crop, transform landmarks, save.
        # -------------------------------------------------------------------
        for face_id, entries in tqdm(by_id.items(), desc="Preprocessing", unit="id"):
            shared_matrix, shared_scale = alignments.get(face_id, (None, 1.0))

            for image_path, expression in entries:
                key = (face_id, expression)
                d = detections.get(key)

                if d == "skip":
                    stats["skipped_existing"] += 1
                    continue
                if not isinstance(d, dict) or "error" in d:
                    reason = d.get("error") if isinstance(d, dict) else "unknown"
                    stats["failed"] += 1
                    stats[f"failed_{reason}"] += 1
                    LOGGER.debug("Skipped %s (%s)", image_path.name, reason)
                    continue

                meta = lookup_meta(metadata, face_id)
                if meta is None:
                    meta = EMPTY_META
                    missing_csv += 1

                # Re-read image (only the ones we'll actually write).
                bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    stats["failed"] += 1
                    stats["failed_unreadable_image"] += 1
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                # Use the shared per-identity matrix when available; otherwise fall
                # back to a per-image alignment (still consistent within this run).
                if shared_matrix is not None:
                    matrix, applied_scale = shared_matrix, shared_scale
                else:
                    matrix, applied_scale = compute_shared_alignment(
                        d["landmarks"], d["width"], d["height"],
                        target_size, desired_frac, max_zoom_out,
                    )

                crop_rgb = warp_to_crop(rgb, matrix, target_size)
                landmarks_crop = transform_landmarks(
                    d["landmarks"], matrix, d["width"], d["height"]
                )

                stem = f"{face_id}_{expression}"
                ok = cv2.imwrite(
                    str(crops_dir / f"{stem}.jpg"),
                    cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                if not ok:
                    stats["failed"] += 1
                    stats["failed_write_failed"] += 1
                    continue

                np.savez_compressed(
                    meta_dir / f"{stem}.npz",
                    landmarks=landmarks_crop,
                    blendshapes=d["blendshapes"].astype(np.float32),
                    pose=d["pose"].astype(np.float32),
                    confidence=float(d["confidence"]),
                    expression=expression,
                    face_id=face_id,
                    age=float(meta.age),
                    gender=meta.gender,
                    ethnicity=meta.ethnicity,
                    align_matrix=matrix.astype(np.float32),
                    target_size=int(target_size),
                    applied_scale=float(applied_scale),
                )

                stats["success"] += 1
                extent = face_vertical_extent(landmarks_crop, target_size)
                extents.append(extent)
                scales.append(applied_scale)
                if extent[0] < 0.0 or extent[1] > 1.0:
                    stats["warn_face_clipped"] += 1
    finally:
        landmarker.close()

    # ------------------------------------------------------------------- #
    # Summary
    # ------------------------------------------------------------------- #
    total = len(items)
    LOGGER.info("=" * 50)
    LOGGER.info("Total images       : %d", total)
    LOGGER.info("Succeeded          : %d", stats["success"])
    LOGGER.info("Failed             : %d", stats["failed"])
    for key in sorted(k for k in stats if k.startswith("failed_")):
        LOGGER.info("  - %-22s: %d", key.replace("failed_", ""), stats[key])
    if extents:
        top, chin = np.mean(extents, axis=0)
        LOGGER.info(
            "Framing (mean)     : forehead-top y=%.2f, chin y=%.2f -> face height %.0f%%, "
            "space below chin %.0f%%",
            top, chin, 100 * (chin - top), 100 * (1 - chin),
        )
        if stats["warn_face_clipped"]:
            LOGGER.warning(
                "%d crops have the forehead top or chin outside the frame",
                stats["warn_face_clipped"],
            )
    if scales:
        LOGGER.info(
            "Applied scale (mean): %.3f (1.0 = no zoom-out; shared per identity: %s)",
            float(np.mean(scales)), "yes" if consistent_alignment else "no",
        )
    if stats["skipped_existing"]:
        LOGGER.info("Already processed  : %d (use --overwrite to redo)", stats["skipped_existing"])
    if metadata and missing_csv:
        LOGGER.warning(
            "%d images had no matching CSV row (metadata left empty)", missing_csv
        )
    LOGGER.info("Output             : %s", output_dir.resolve())
    return stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GALNP Stage 0: MediaPipe FaceLandmarker preprocessing and 5-point alignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_dir", type=Path, required=True,
                   help="Root folder containing neutral_front/ and smiling_front/")
    p.add_argument("--output_dir", type=Path, default=Path("data/processed"),
                   help="Where crops/ and meta/ are written")
    p.add_argument("--csv_path", type=Path, default=None,
                   help="Ratings CSV with face_id, face_age, face_gender, face_eth "
                        "(auto-detected in input_dir if omitted)")
    p.add_argument("--target_size", type=int, default=512, help="Aligned crop size in pixels")
    p.add_argument("--min_confidence", type=float, default=0.5,
                   help="Minimum detection/presence confidence (and in-frame landmark fraction)")
    p.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH,
                   help="FaceLandmarker .task file (downloaded automatically if missing)")
    p.add_argument("--overwrite", action="store_true",
                   help="Reprocess images that already have outputs")
    p.add_argument("--verbose", action="store_true", help="Log every skipped image")
    p.add_argument("--target_face_frac", type=float, default=0.75,
                   help="Desired fraction of crop height occupied by face (0.7-0.8 recommended)")
    p.add_argument("--max_zoom_out", type=float, default=1.6,
                   help="Maximum allowed zoom-out scale (safety cap)")
    p.add_argument(
        "--no_consistent_alignment", dest="consistent_alignment",
        action="store_false",
        help=("Disable shared per-identity alignment. Each image will be aligned "
              "independently, which breaks correspondence between neutral and smiling."),
    )
    p.set_defaults(consistent_alignment=True)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        csv_path=args.csv_path,
        target_size=args.target_size,
        min_confidence=args.min_confidence,
        model_path=args.model_path,
        overwrite=args.overwrite,
        desired_frac=args.target_face_frac,
        max_zoom_out=args.max_zoom_out,
        consistent_alignment=args.consistent_alignment,
    )


if __name__ == "__main__":
    # Examples:
    #   python prepare.py --input_dir /path/to/london_faces \
    #                     --output_dir data/processed
    #
    #   Colab:
    #   !python prepare.py --input_dir /content/drive/MyDrive/london_faces \
    #                      --output_dir /content/drive/MyDrive/galnp/processed
    main()