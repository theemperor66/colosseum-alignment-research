"""Separate observation features and privileged labels for every scheduled camera frame."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from colosseum_assurance.interfaces import CapturedFrame
from colosseum_assurance.protocol.context_confidence import ContextConfidenceSpec
from colosseum_assurance.protocol.expanded import ExpandedStudySpec


def retain_camera_transformation(*, raw: dict[str, CapturedFrame],
                                 before: dict[str, CapturedFrame],
                                 delivered: dict[str, CapturedFrame] | None,
                                 report: dict[str, Any], prefix: Path, sidecar: Path,
                                 identity: dict[str, Any]) -> dict[str, CapturedFrame] | None:
    """Save every opt-in acquisition before inference; masks/sidecar remain evaluator-only.

    Array hashes bind pixels, file hashes bind the retained NPY. Equal stages share one file.
    Raw means adapter-returned pixels, before any legacy photometric transform or this fault.
    """
    prefix.parent.mkdir(parents=True, exist_ok=True)
    stages: dict[str, Any] = {}
    known: dict[tuple[str, str, str, tuple[int, ...]], dict[str, Any]] = {}
    result = None if delivered is None else dict(delivered)
    for stage, frames in (("raw", raw), ("pre_obscuration", before), ("delivered", delivered or {})):
        stages[stage] = {}
        for name, frame in sorted(frames.items()):
            ref = frame.ref.model_dump(mode="json")
            item: dict[str, Any] = {"ref": ref, "pixels": None}
            if frame.array is not None:
                pixels = frame.array
                pixel_hash = "sha256:" + hashlib.sha256(pixels.tobytes()).hexdigest()
                key = (name, pixel_hash, str(pixels.dtype), pixels.shape)
                saved = known.get(key)
                if saved is None:
                    path = Path(str(prefix) + f"_camera_{stage}_{name}.npy")
                    existing = Path(frame.ref.path) if frame.ref.path else None
                    if stage == "raw" and existing is not None and frame.ref.pixels_as == "npy":
                        retained = np.load(existing, allow_pickle=False)
                        if (retained.dtype == pixels.dtype and retained.shape == pixels.shape
                                and retained.tobytes() == pixels.tobytes()):
                            path = existing
                        else:
                            existing = None
                    else:
                        existing = None
                    if existing is None:
                        # Exclusive creation preserves partial acquisitions after a write failure.
                        with path.open("xb") as stream:
                            np.save(stream, pixels, allow_pickle=False)
                    saved = {"path": str(path.resolve()), "sha256": hashlib.sha256(
                        path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
                        "array_sha256": pixel_hash, "dtype": str(pixels.dtype),
                        "shape": list(pixels.shape)}
                    known[key] = saved
                frame_id = (identity["acquisition_id"] if stage == "delivered" and name == "rgb"
                            else f"{identity['acquisition_id']}:{stage}:{name}")
                ref.update(frame_id=frame_id, content_sha256=pixel_hash,
                           path=saved["path"], pixels_as="npy")
                item = {"ref": ref, "pixels": saved}
                if stage == "delivered" and result is not None:
                    result[name] = CapturedFrame(ref=frame.ref.model_copy(update=ref), array=pixels)
            stages[stage][name] = item
    record = {**identity, **report, "frames": stages,
              "timestamp_basis": "FrameRef episode-relative acquisition, not receipt time",
              "native_timestamp_reference": (
                  "adapter privileged_camera_poses.jsonl by save_prefix when available"),
              "save_prefix": str(prefix.resolve()),
              "raw_pixel_definition": "adapter-returned arrays before runtime photometric processing"}
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    return result


def record_perception_frame(*, frames: dict[str, CapturedFrame], spec: ExpandedStudySpec,
                            identity: dict[str, Any], scenario_group: str, arm_id: str,
                            frame_id: str, provenance: str, stratum: str,
                            out: Path, dropped: bool, hfov_rad: float,
                            context_policy: ContextConfidenceSpec | None = None) -> dict[str, Any]:
    from colosseum_assurance.perception_eval import (
        LabelSpec,
        PerceptionRow,
        extract_features,
        label_from_segmentation,
    )

    image = frames.get("rgb")
    depth = frames.get("depth")
    segmentation = frames.get("segmentation")
    rgb = None if image is None or dropped else image.array
    depth_pixels = None if depth is None or dropped else depth.array
    # Black pixels are a valid, difficult observation, not a missing camera.
    usable = (image is not None and image.ref.acquisition_time_known and image.ref.sim_time_s is not None
              and rgb is not None and rgb.size > 0 and image.ref.width > 0 and image.ref.height > 0
              and not dropped)
    if context_policy is not None and usable:
        usable = tuple(rgb.shape) == (image.ref.height, image.ref.width, 3)
    if depth is not None and image is not None:
        if (not depth.ref.acquisition_time_known or depth.ref.sim_time_s is None
                or image.ref.sim_time_s is None or depth.ref.camera_name != image.ref.camera_name
                or abs(depth.ref.sim_time_s - image.ref.sim_time_s) > LabelSpec().max_sync_error_ns / 1e9):
            depth_pixels = None
    features = extract_features(rgb, depth_pixels, hfov_rad=hfov_rad) if usable else None
    mask = None if segmentation is None else segmentation.array
    # PerturbationChannel applies the actual delivered occlusion to the otherwise
    # independent segmentation mask at the same time it transforms camera pixels.

    def stamp(frame: CapturedFrame | None) -> int | None:
        if frame is None or frame.ref.sim_time_s is None or not frame.ref.acquisition_time_known:
            return None
        return round(frame.ref.sim_time_s * 1e9)

    label = label_from_segmentation(
        mask, asset_color_rgb=spec.asset_segmentation_color,
        identity_verified=bool(identity.get("identity_verified")),
        image_timestamp_ns=stamp(image) if usable else None,
        segmentation_timestamp_ns=stamp(segmentation),
        image_camera_name=image.ref.camera_name if image else "missing",
        segmentation_camera_name=segmentation.ref.camera_name if segmentation else "missing",
        image_shape=None if rgb is None else rgb.shape[:2], spec=LabelSpec(),
    )
    split = spec.perception_split_by_group.get(scenario_group, spec.perception_split)
    if spec.perception_split_by_group and scenario_group not in spec.perception_split_by_group:
        raise ValueError("scenario group absent from frozen perception split mapping")
    row = PerceptionRow(frame_id=frame_id, scenario_id=scenario_group, arm_id=arm_id, split=split,
                        stratum=stratum, provenance=provenance, features=features,
                        missing_reason="missing, dropped, or unstamped image" if features is None else None,
                        label=label)
    # The sidecar is evaluator-owned: it contains labels. It is never attached to
    # ObservationPacket, supplied to the controller, or supplied to its monitor.
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as stream:
        stream.write(row.model_dump_json() + "\n")
    result: dict[str, Any] = {"frame_id": frame_id, "prediction": None, "model_status": "unfitted",
                              "sim_time_s": None if image is None else image.ref.sim_time_s}
    if spec.perception_model_path:
        from colosseum_assurance.perception_eval import load_model, predict_probability

        model = load_model(spec.perception_model_path)
        if model.model_hash != spec.perception_model_hash:
            raise ValueError("perception model does not match the frozen study protocol")
        if model.label_spec_hash != LabelSpec().spec_hash:
            raise ValueError("trained perception event does not match the frozen runtime label definition")
        if provenance != "fixture_fake" and model.evidence_class == "fixture_only":
            raise ValueError("a fixture-trained perception model cannot be used for a live study")
        # Only the observation-derived feature vector enters inference.
        result.update(prediction=predict_probability(model, features), model_status="frozen_model",
                      model_hash=model.model_hash)
        if context_policy is not None and features is not None and image is not None:
            from colosseum_assurance.perception_eval.contracts import content_hash
            from colosseum_assurance.schemas import PerceptionPredictionEvidence

            if (model.model_hash != context_policy.expected_model_hash
                    or model.label_spec_hash != context_policy.expected_label_spec_hash
                    or model.feature_version != context_policy.feature_version):
                raise ValueError("context policy differs from frozen model definition")
            rgb_hash = "sha256:" + hashlib.sha256(rgb.tobytes()).hexdigest()
            image.ref = image.ref.model_copy(update={"frame_id": frame_id, "content_sha256": rgb_hash})
            evidence = PerceptionPredictionEvidence(
                frame_id=frame_id, rgb_sha256=rgb_hash, camera_name=image.ref.camera_name,
                image_time_s=image.ref.sim_time_s, width=image.ref.width, height=image.ref.height,
                model_hash=model.model_hash, feature_version=model.feature_version,
                label_spec_hash=model.label_spec_hash, features_sha256=content_hash(features),
                features=features,
                depth_available=bool(features["depth_available"]),
                depth_valid_fraction=features["depth_valid_fraction"], rgb_std=features["rgb_std"],
                probability=result["prediction"],
            )
            result["evidence"] = evidence.model_dump(mode="json")
    return result
