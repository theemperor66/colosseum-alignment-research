"""Observation-only predictors and a separately called, privileged segmentation labeler."""

from __future__ import annotations

import hashlib
import math

import numpy as np

from colosseum_assurance.control.perception import detect_vertical_structure
from colosseum_assurance.perception_eval.contracts import LabelEvidence, LabelSpec


def extract_features(
    rgb: np.ndarray | None, depth: np.ndarray | None = None, *, hfov_rad: float = math.pi / 2,
) -> dict[str, float] | None:
    """RGB/depth -> frozen continuous features. No identity, mask, scenario or label input exists.

    RGB is required for this RGB-frame visibility event. Detector negatives and black frames remain
    eligible; they are not silently filtered. Missing depth is represented with an explicit feature.
    """
    if rgb is None or np.asarray(rgb).size == 0:
        return None
    a = np.asarray(rgb)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("RGB must be uint8 HxWx3, not segmentation or a pre-labelled score")
    if not 0 < hfov_rad < math.pi:
        raise ValueError("camera horizontal field of view must be in (0, pi)")
    luminance = np.mean(a.astype(np.float64), axis=2) / 255.0
    gradients = [np.abs(np.diff(luminance, axis=axis)).ravel() for axis in (0, 1)]
    gradient = np.concatenate(gradients)
    features = {
        "rgb_mean": float(luminance.mean()), "rgb_std": float(luminance.std()),
        "rgb_gradient": float(gradient.mean()) if gradient.size else 0.0,
        "depth_available": 0.0, "depth_valid_fraction": 0.0, "depth_near_fraction": 0.0,
        "depth_median_scaled": 0.0, "structure_support": 0.0, "structure_width_scaled": 0.0,
    }
    if depth is None or np.asarray(depth).size == 0:
        return features
    d = np.asarray(depth)
    if d.ndim != 2 or d.dtype.kind not in "fiu":
        raise ValueError("depth must be a numeric HxW array of ray distances in metres")
    if d.shape != a.shape[:2]:
        raise ValueError("RGB/depth feature extraction requires matching image dimensions")
    finite = np.isfinite(d) & (d > 0)
    features["depth_available"] = 1.0
    features["depth_valid_fraction"] = float(finite.mean())
    features["depth_near_fraction"] = float((finite & (d <= 10.0)).mean())
    if finite.any():
        features["depth_median_scaled"] = float(np.clip(np.median(d[finite]) / 60.0, 0, 1))
        detection = detect_vertical_structure(d.astype(float), hfov_rad=hfov_rad, max_range_m=60.0)
        features["structure_support"] = float(detection.support_fraction)
        features["structure_width_scaled"] = float(np.clip((detection.width_m or 0.0) / 6.0, 0, 1))
    return features


def label_from_segmentation(
    segmentation: np.ndarray | None, *, asset_color_rgb: tuple[int, int, int] | None,
    identity_verified: bool, image_timestamp_ns: int | None, segmentation_timestamp_ns: int | None,
    image_camera_name: str, segmentation_camera_name: str, image_shape: tuple[int, int] | None,
    spec: LabelSpec,
) -> LabelEvidence:
    """Independent privileged label; the predictor never calls or receives this function's inputs.

    `identity_verified` must attest that the color denotes the designated asset, not just any structure.
    Segmentation palette colors must come from the simulator's verified ID/color mapping. No guessed
    palette, image classifier prediction, or missing mask can manufacture a negative label.
    """
    base = {"label_spec_hash": spec.spec_hash}

    def unknown(reason: str) -> LabelEvidence:
        return LabelEvidence(value=None, reason=reason, **base)

    if not identity_verified or asset_color_rgb is None:
        return unknown("asset_identity_unverified")
    if (len(asset_color_rgb) != 3
            or any(type(v) is not int or not 0 <= v <= 255 for v in asset_color_rgb)):
        return unknown("invalid_asset_palette_color")
    if segmentation is None:
        return unknown("segmentation_missing")
    a = np.asarray(segmentation)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3 or not a.size:
        return unknown("segmentation_invalid")
    if not image_camera_name or image_camera_name != segmentation_camera_name:
        return unknown("camera_mismatch")
    if image_shape is None or tuple(a.shape[:2]) != tuple(image_shape):
        return unknown("image_shape_missing_or_mismatched")
    if any(type(t) is not int or t <= 0 for t in (image_timestamp_ns, segmentation_timestamp_ns)):
        return unknown("capture_timestamp_missing")
    assert image_timestamp_ns is not None and segmentation_timestamp_ns is not None
    if abs(image_timestamp_ns - segmentation_timestamp_ns) > spec.max_sync_error_ns:
        return unknown("capture_not_synchronized")
    visible = int(np.all(a == np.asarray(asset_color_rgb, dtype=np.uint8), axis=2).sum())
    total = int(a.shape[0] * a.shape[1])
    present = visible >= spec.min_visible_pixels and visible / total >= spec.min_visible_fraction
    return LabelEvidence(
        value=int(present), reason="segmentation_pixel_threshold", visible_pixels=visible,
        total_pixels=total, segmentation_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
        image_timestamp_ns=image_timestamp_ns, segmentation_timestamp_ns=segmentation_timestamp_ns,
        identity_verified=True, asset_color_rgb=asset_color_rgb, camera_name=image_camera_name, **base,
    )
