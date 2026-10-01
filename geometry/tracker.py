"""
geometry/tracker.py

Lightweight wrapper around MediaPipe FaceLandmarker (mediapipe.tasks.vision.FaceLandmarker)
for GALNP Stage 0.

Features
- Load the official `face_landmarker_v2_with_blendshapes.task` model (download if missing).
- Provide a simple, typed API to run detection on a single image or a batch of images.
- Return a clean dictionary per image:
    {
      "landmarks": (478, 3) float32,
      "blendshapes": (52,) float32,
      "pose": (4, 4) float32,
      "confidence": float,
      "success": bool
    }
- Convenience function to also return the 5 stable alignment points (pixel coords).
- Graceful handling of failures (returns success=False and sensible defaults).
- No dependency on the rest of the GALNP codebase.
- Works with CPU or GPU (MediaPipe chooses runtime based on environment).

Dependencies
------------
- numpy
- mediapipe (mediapipe.tasks)
- tqdm is NOT required here

Usage
-----
from geometry.tracker import FaceLandmarkerWrapper, FIVE_POINT_IDX

with FaceLandmarkerWrapper() as fl:
    result = fl.detect_single(rgb_image)  # rgb_image: np.ndarray HxWx3 uint8
    # or
    results = fl.detect_batch([rgb1, rgb2])
    # get 5-point pixel coordinates:
    pts5 = fl.landmarks_to_5pt(result["landmarks"], width, height)
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Import MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

LOGGER = logging.getLogger("geometry.tracker")

# Default model URL and path (official MediaPipe face_landmarker with blendshapes)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/"
    "face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path("data/models/face_landmarker.task")

# Constants
NUM_LANDMARKS = 478
NUM_BLENDSHAPES = 52

# 5 stable landmark indices used for alignment (MediaPipe indexing)
FIVE_POINT_IDX: Tuple[int, ...] = (33, 263, 1, 61, 291)


# -------------------------
# Utilities
# -------------------------
def ensure_model(model_path: Path = DEFAULT_MODEL_PATH, url: str = MODEL_URL) -> Path:
    """
    Ensure the FaceLandmarker .task model exists on disk. Download if missing.

    Returns the path to the model file.
    """
    model_path = Path(model_path)
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading FaceLandmarker model to %s ...", model_path)
    try:
        urllib.request.urlretrieve(url, model_path)
    except Exception as exc:  # pragma: no cover - network
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(f"Could not download the model from {url}. Error: {exc}") from exc
    return model_path


def compute_inframe_fraction(landmarks_norm: np.ndarray) -> float:
    """
    Compute the fraction of landmarks whose x,y are inside [0,1].
    landmarks_norm: (N,3) with x,y normalized in [0,1] (MediaPipe convention).
    """
    if landmarks_norm is None or landmarks_norm.size == 0:
        return 0.0
    xy = landmarks_norm[:, :2]
    inside = (xy[:, 0] >= 0.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= 0.0) & (xy[:, 1] <= 1.0)
    return float(np.mean(inside))


def landmarks_to_pixel_coords(landmarks_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Convert normalized landmarks (x,y in [0,1]) to pixel coordinates (width, height).
    Returns (N,2) float64.
    """
    if landmarks_norm is None or landmarks_norm.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    pts = landmarks_norm[:, :2].astype(np.float64)
    pts[:, 0] *= width
    pts[:, 1] *= height
    return pts


# -------------------------
# Wrapper class
# -------------------------
@dataclass
class FaceLandmarkerWrapper:
    """
    Wrapper around mediapipe.tasks.vision.FaceLandmarker.

    Example
    -------
    with FaceLandmarkerWrapper(model_path=Path("...")) as fl:
        out = fl.detect_single(rgb_image)
    """

    model_path: Path = DEFAULT_MODEL_PATH
    min_confidence: float = 0.5
    num_faces: int = 1

    # Internal (initialized in __enter__)
    _landmarker: Optional[vision.FaceLandmarker] = None

    def __post_init__(self) -> None:
        # Do not auto-load model here; load lazily in __enter__ or on first call
        self.model_path = Path(self.model_path)

    def _ensure_loaded(self) -> None:
        if self._landmarker is not None:
            return
        model_file = ensure_model(self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_file)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=self.num_faces,
            min_face_detection_confidence=self.min_confidence,
            min_face_presence_confidence=self.min_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        LOGGER.debug("FaceLandmarker loaded from %s", model_file)

    def close(self) -> None:
        """Close the underlying MediaPipe landmarker if loaded."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None

    # Context manager support
    def __enter__(self) -> "FaceLandmarkerWrapper":
        self._ensure_loaded()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -------------------------
    # Detection API
    # -------------------------
    def _process_mp_result(self, result) -> Dict:
        """
        Convert a mediapipe FaceLandmarker result (single face) into the clean dict.
        If result is None or missing fields, returns success=False.
        """
        if result is None:
            return {
                "landmarks": np.zeros((NUM_LANDMARKS, 3), dtype=np.float32),
                "blendshapes": np.zeros((NUM_BLENDSHAPES,), dtype=np.float32),
                "pose": np.zeros((4, 4), dtype=np.float32),
                "confidence": 0.0,
                "success": False,
            }

        # face_landmarks is a list; we use the first face
        if not result.face_landmarks:
            return {
                "landmarks": np.zeros((NUM_LANDMARKS, 3), dtype=np.float32),
                "blendshapes": np.zeros((NUM_BLENDSHAPES,), dtype=np.float32),
                "pose": np.zeros((4, 4), dtype=np.float32),
                "confidence": 0.0,
                "success": False,
            }

        lm = result.face_landmarks[0]
        if len(lm) < NUM_LANDMARKS:
            return {
                "landmarks": np.zeros((NUM_LANDMARKS, 3), dtype=np.float32),
                "blendshapes": np.zeros((NUM_BLENDSHAPES,), dtype=np.float32),
                "pose": np.zeros((4, 4), dtype=np.float32),
                "confidence": 0.0,
                "success": False,
            }

        landmarks = np.array([[p.x, p.y, p.z] for p in lm[:NUM_LANDMARKS]], dtype=np.float32)

        # Blendshapes: result.face_blendshapes is a list of lists; pick first face
        blendshapes = np.zeros((NUM_BLENDSHAPES,), dtype=np.float32)
        try:
            cats = sorted(result.face_blendshapes[0], key=lambda c: c.index)
            blendshapes = np.array([c.score for c in cats], dtype=np.float32)
        except Exception:
            # leave zeros if missing
            pass

        # Pose: facial_transformation_matrixes
        pose = np.zeros((4, 4), dtype=np.float32)
        try:
            pose = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float32).reshape(4, 4)
        except Exception:
            pass

        confidence = compute_inframe_fraction(landmarks)

        return {
            "landmarks": landmarks,
            "blendshapes": blendshapes,
            "pose": pose,
            "confidence": float(confidence),
            "success": True,
        }

    def detect_single(self, rgb: np.ndarray) -> Dict:
        """
        Run FaceLandmarker on a single RGB image (HxWx3 uint8 or float in [0,255]).
        Returns the result dictionary described above. Always returns a dict; on
        failure success=False.
        """
        if not isinstance(rgb, np.ndarray):
            raise TypeError("rgb must be a numpy.ndarray HxWx3 uint8")

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must have shape HxWx3")

        self._ensure_loaded()

        # Convert to MediaPipe Image
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = self._landmarker.detect(mp_image)
        except Exception as exc:
            LOGGER.debug("FaceLandmarker detection failed: %s", exc)
            return {
                "landmarks": np.zeros((NUM_LANDMARKS, 3), dtype=np.float32),
                "blendshapes": np.zeros((NUM_BLENDSHAPES,), dtype=np.float32),
                "pose": np.zeros((4, 4), dtype=np.float32),
                "confidence": 0.0,
                "success": False,
            }

        return self._process_mp_result(result)

    def detect_batch(self, rgbs: Sequence[np.ndarray]) -> List[Dict]:
        """
        Run FaceLandmarker on a batch of RGB images. Returns a list of result dicts
        in the same order as the input list. Each entry follows the same schema.
        """
        if not isinstance(rgbs, (list, tuple)):
            raise TypeError("rgbs must be a list or tuple of numpy arrays")

        self._ensure_loaded()
        results: List[Dict] = []
        for img in rgbs:
            try:
                res = self.detect_single(img)
            except Exception as exc:
                LOGGER.debug("Error processing image in batch: %s", exc)
                res = {
                    "landmarks": np.zeros((NUM_LANDMARKS, 3), dtype=np.float32),
                    "blendshapes": np.zeros((NUM_BLENDSHAPES,), dtype=np.float32),
                    "pose": np.zeros((4, 4), dtype=np.float32),
                    "confidence": 0.0,
                    "success": False,
                }
            results.append(res)
        return results

    # -------------------------
    # Convenience helpers
    # -------------------------
    @staticmethod
    def landmarks_to_5pt(landmarks_norm: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Given normalized landmarks (N,3), return the 5 stable alignment points
        in pixel coordinates (5,2) in the order defined by FIVE_POINT_IDX.

        If landmarks_norm is invalid, returns an empty array shape (0,2).
        """
        if landmarks_norm is None or landmarks_norm.shape[0] < max(FIVE_POINT_IDX) + 1:
            return np.zeros((0, 2), dtype=np.float64)
        pts = landmarks_norm[list(FIVE_POINT_IDX), :2].astype(np.float64)
        pts[:, 0] *= width
        pts[:, 1] *= height
        return pts

    @staticmethod
    def landmarks_to_pixel_landmarks(landmarks_norm: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Convert full normalized landmarks (N,3) to pixel coordinates (N,3),
        where x,y are in pixels and z is scaled similarly to x (approx).
        """
        if landmarks_norm is None or landmarks_norm.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        xy = landmarks_norm[:, :2].astype(np.float64) * np.array([width, height], dtype=np.float64)
        # approximate z scaling: MediaPipe z is relative depth scaled like x
        scale = float(width)
        z = landmarks_norm[:, 2].astype(np.float64) * scale
        out = np.concatenate([xy, z[:, None]], axis=1).astype(np.float32)
        return out


# -------------------------
# Convenience module-level functions
# -------------------------
def make_landmarker(model_path: Optional[Path] = None, min_confidence: float = 0.5, num_faces: int = 1) -> FaceLandmarkerWrapper:
    """
    Create a FaceLandmarkerWrapper instance. The wrapper is a context manager
    and should be closed when done (use `with` or call .close()).
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    return FaceLandmarkerWrapper(model_path=Path(model_path), min_confidence=min_confidence, num_faces=num_faces)


# -------------------------
# Example usage (not executed on import)
# -------------------------
if __name__ == "__main__":
    import argparse
    import cv2

    parser = argparse.ArgumentParser(description="Quick test for geometry.tracker FaceLandmarkerWrapper")
    parser.add_argument("--image", type=str, required=True, help="Path to an RGB image")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to face_landmarker .task")
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit("Could not read image")
    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with make_landmarker(Path(args.model)) as fl:
        out = fl.detect_single(img_rgb)
        print("success:", out["success"])
        print("confidence:", out["confidence"])
        print("landmarks shape:", out["landmarks"].shape)
        print("blendshapes shape:", out["blendshapes"].shape)
        print("pose shape:", out["pose"].shape)
        if out["success"]:
            pts5 = fl.landmarks_to_5pt(out["landmarks"], img_rgb.shape[1], img_rgb.shape[0])
            print("5-point pixel coords:\n", pts5)
