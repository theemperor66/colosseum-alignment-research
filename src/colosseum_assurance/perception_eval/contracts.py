"""Frozen contracts for a bounded civilian asset-visibility probability experiment."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FEATURE_VERSION = "civilian_rgb_depth_v1"
FEATURE_NAMES = (
    "rgb_mean", "rgb_std", "rgb_gradient", "depth_available", "depth_valid_fraction",
    "depth_near_fraction", "depth_median_scaled", "structure_support", "structure_width_scaled",
)
Provenance = Literal["fixture_fake", "third_party_colosseum_build", "colosseum_build_verified"]
Split = Literal["train", "calibration", "test"]


def content_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


class FrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class LabelSpec(FrozenRecord):
    """Positive iff BOTH thresholds hold in an independently captured segmentation frame."""

    label_version: Literal["designated_civilian_asset_visible_v1"] = "designated_civilian_asset_visible_v1"
    min_visible_pixels: int = Field(default=8, ge=1)
    min_visible_fraction: float = Field(default=0.001, gt=0, le=1)
    max_sync_error_ns: int = Field(default=1_000_000, ge=0)

    @property
    def spec_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))


class LabelEvidence(FrozenRecord):
    value: Literal[0, 1] | None
    reason: str
    label_spec_hash: str
    visible_pixels: int | None = Field(default=None, ge=0)
    total_pixels: int | None = Field(default=None, ge=1)
    segmentation_sha256: str | None = None
    image_timestamp_ns: int | None = Field(default=None, ge=1)
    segmentation_timestamp_ns: int | None = Field(default=None, ge=1)
    identity_verified: bool = False
    asset_color_rgb: tuple[int, int, int] | None = None
    camera_name: str | None = None

    @field_validator("value", mode="before")
    @classmethod
    def binary_not_boolean(cls, value: object) -> object:
        if value is not None and (type(value) is not int or value not in (0, 1)):
            raise ValueError("label must be integer 0/1 or None")
        return value

    @model_validator(mode="after")
    def known_requires_evidence(self) -> LabelEvidence:
        if self.asset_color_rgb is not None and any(v < 0 or v > 255 for v in self.asset_color_rgb):
            raise ValueError("asset color components must be bytes")
        if self.value is not None:
            if not (self.identity_verified and self.segmentation_sha256 and self.image_timestamp_ns
                    and self.segmentation_timestamp_ns and self.camera_name
                    and self.asset_color_rgb is not None):
                raise ValueError("known labels require synchronized identity/mask evidence")
            if self.visible_pixels is None or self.total_pixels is None:
                raise ValueError("known labels require pixel counts")
            if self.visible_pixels > self.total_pixels:
                raise ValueError("visible pixels exceed frame size")
        return self


class PerceptionRow(FrozenRecord):
    """One scheduled frame, including missing images and independently undecidable labels."""

    frame_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    arm_id: str = Field(min_length=1)
    split: Split
    stratum: str = Field(default="all", min_length=1)
    provenance: Provenance
    feature_version: Literal["civilian_rgb_depth_v1"] = FEATURE_VERSION
    features: dict[str, float] | None
    missing_reason: str | None = None
    label: LabelEvidence

    @field_validator("features")
    @classmethod
    def check_features(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        if value is not None:
            validate_features(value)
        return value

    @model_validator(mode="after")
    def missing_has_reason(self) -> PerceptionRow:
        if self.features is None and not self.missing_reason:
            raise ValueError("missing observations require a reason, not probability zero")
        return self


def validate_features(features: dict[str, float]) -> None:
    if set(features) != set(FEATURE_NAMES):
        raise ValueError(f"features must be exactly {FEATURE_NAMES}; no label/truth fields are accepted")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in features.values()):
        raise ValueError("features must contain finite real numbers, never booleans")


def validate_rows(rows: list[PerceptionRow]) -> None:
    if not rows:
        raise ValueError("empty perception dataset")
    ids: set[str] = set()
    membership: dict[str, tuple[str, str]] = {}
    for row in rows:
        if row.frame_id in ids:
            raise ValueError(f"duplicate frame_id: {row.frame_id}")
        ids.add(row.frame_id)
        key = (row.split, row.stratum)
        if membership.setdefault(row.scenario_id, key) != key:
            raise ValueError(f"scenario {row.scenario_id} crosses split or stratum boundaries")
        if row.features is not None:
            validate_features(row.features)
    if len({r.label.label_spec_hash for r in rows}) != 1:
        raise ValueError("dataset mixes visibility-label definitions")
    if len({r.provenance == "fixture_fake" for r in rows}) != 1:
        raise ValueError("fixture and live data cannot be mixed")


def assign_scenario_splits(
    scenario_ids: list[str], *, seed: int = 7717, train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
) -> dict[str, Split]:
    """Assign WHOLE scenarios before looking at any image or label; persist the returned mapping."""
    ids = sorted(set(scenario_ids))
    if len(ids) < 3 or any(not x for x in ids):
        raise ValueError("at least three nonempty scenario IDs are needed for three disjoint splits")
    if not (0 < train_fraction < 1 and 0 < calibration_fraction < 1
            and train_fraction + calibration_fraction < 1):
        raise ValueError("positive train/calibration fractions must leave a test fraction")
    ordered = sorted(ids, key=lambda x: hashlib.sha256(f"{seed}|{x}".encode()).digest())
    n_train = max(1, min(len(ids) - 2, int(len(ids) * train_fraction)))
    n_cal = max(1, min(len(ids) - n_train - 1, int(len(ids) * calibration_fraction)))
    return {sid: ("train" if i < n_train else "calibration" if i < n_train + n_cal else "test")
            for i, sid in enumerate(ordered)}
