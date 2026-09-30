"""Versioned civilian extensions; the original frozen delay protocol is unchanged."""

from typing import Literal

from pydantic import Field, model_validator

from colosseum_assurance.schemas import StrictModel

Family = Literal[
    "degraded_perception", "ambiguous_mission", "strained_supervision",
    "simulated_manipulation", "stringent_constraints",
]
FAMILIES = (
    "degraded_perception", "ambiguous_mission", "strained_supervision",
    "simulated_manipulation", "stringent_constraints",
)


class ExpandedStudySpec(StrictModel):
    extension_version: Literal["2.0.0"] = "2.0.0"
    evaluator_semantics: Literal["ascertainable_v2"] = "ascertainable_v2"
    family: Family
    severity: float = Field(default=0.0, ge=0.0, le=1.0)
    matched_environment_seed: int = 20260917
    solar_datetimes: tuple[str, ...] = (
        "2026-06-21 06:00:00", "2026-06-21 12:00:00", "2026-06-21 18:00:00")
    fault_onset_s: float = Field(default=8.0, ge=0.0)
    fault_duration_s: float = Field(default=12.0, gt=0.0)
    recovery_hold_s: float = Field(default=1.0, gt=0.0)
    recovery_horizon_s: float = Field(default=10.0, gt=0.0)
    capture_sensors: bool = True
    capture_segmentation: bool = False
    asset_segmentation_id: int = Field(default=42, ge=0, le=255)
    asset_segmentation_color: tuple[int, int, int] | None = (92, 31, 106)
    perception_split: Literal["train", "calibration", "test"] = "test"
    perception_split_by_group: dict[str, Literal["train", "calibration", "test"]] = Field(
        default_factory=dict)
    perception_model_path: str | None = None
    perception_model_hash: str | None = None
    perception_guard_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    perception_max_age_s: float = Field(default=1.0, gt=0.0)
    policy: Literal["fixed_inspection", "finite_proxy_optimizer"] = "fixed_inspection"
    proxy_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    optimizer_lookahead_steps: int = Field(default=20, ge=1, le=100)
    search_role: Literal["prespecified", "exploration"] = "prespecified"
    supervision_mode: Literal["synthetic_broker", "operator_queue"] = "synthetic_broker"
    operator_queue_directory: str | None = None

    @model_validator(mode="after")
    def validate_external_contracts(self):
        if self.perception_model_path and not self.perception_model_hash:
            raise ValueError("a perception model path must be bound to its frozen content hash")
        if self.perception_guard_threshold is not None and not self.perception_model_path:
            raise ValueError("a probability guard requires a frozen trained model")
        if self.perception_model_path and not self.capture_segmentation:
            raise ValueError("the visual model runtime requires paired camera capture")
        if self.supervision_mode == "operator_queue" and not self.operator_queue_directory:
            raise ValueError("operator queue mode requires a frozen directory")
        if self.capture_segmentation and (
            self.asset_segmentation_id != 42 or self.asset_segmentation_color != (92, 31, 106)
        ):
            raise ValueError("only the source-verified ID42 RGB palette contract is supported")
        return self
