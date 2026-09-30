"""Explicit v4 civilian observation-confidence policy; no default experimental thresholds."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from colosseum_assurance.schemas import StrictModel


class ContextConfidenceSpec(StrictModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    semantics_version: Literal[
        "observed_asset_capture_gate_v1",
        "observable_quality_capture_gate_v1",
        "record_only_capture_evidence_v1",
    ] = "observed_asset_capture_gate_v1"
    probability_claim: Literal["asset_presence_in_delivered_rgb_only"] = (
        "asset_presence_in_delivered_rgb_only"
    )
    calibration_scope: Literal["unvalidated_on_new_closed_loop_distribution"] = (
        "unvalidated_on_new_closed_loop_distribution"
    )
    applies_to_arms: list[str] = Field(min_length=1)
    expected_model_hash: str
    expected_label_spec_hash: str
    feature_version: Literal["civilian_rgb_depth_v1"] = "civilian_rgb_depth_v1"
    camera_name: str = Field(min_length=1)
    clear_min_depth_fraction: float = Field(ge=0, le=1)
    clear_min_rgb_std: float = Field(ge=0, le=1)
    # Required even for quality-only policies: explicit null documents unused score thresholds.
    clear_threshold: float | None = Field(ge=0, le=1)
    degraded_threshold: float | None = Field(ge=0, le=1)
    max_evidence_age_s: float = Field(gt=0)
    max_capture_opportunities: int = Field(ge=1, le=1000)

    @field_validator("expected_model_hash", "expected_label_spec_hash")
    @classmethod
    def canonical_hash(cls, value: str) -> str:
        if (
            not value.startswith("sha256:")
            or len(value) != 71
            or any(c not in "0123456789abcdef" for c in value[7:])
        ):
            raise ValueError("expected identities must be canonical SHA-256 hashes")
        return value

    @model_validator(mode="after")
    def coherent(self):
        if len(self.applies_to_arms) != len(set(self.applies_to_arms)):
            raise ValueError("context gate arm IDs must be unique")
        if self.semantics_version in {
            "observable_quality_capture_gate_v1",
            "record_only_capture_evidence_v1",
        }:
            if self.clear_threshold is not None or self.degraded_threshold is not None:
                raise ValueError("quality/record-only capture requires explicit null probability thresholds")
        else:
            if self.clear_threshold is None or self.degraded_threshold is None:
                raise ValueError("probability capture requires numeric thresholds in both contexts")
            if self.degraded_threshold < self.clear_threshold:
                raise ValueError("degraded evidence cannot lower the capture confidence requirement")
        return self

    def content_hash(self) -> str:
        raw = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return "sha256:" + hashlib.sha256(raw).hexdigest()
