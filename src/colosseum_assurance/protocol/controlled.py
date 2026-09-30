"""Opt-in semantics for new controlled studies; never applied to historical protocols."""

from typing import Any, Literal

from pydantic import ConfigDict, Field, model_serializer

from colosseum_assurance.protocol.camera_obscuration import CameraObscurationSpec
from colosseum_assurance.schemas import StrictModel


class ControlledStudySpec(StrictModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)

    semantics_version: Literal["controlled_followup_v1"] = "controlled_followup_v1"
    arm_order: Literal["stratified_permutations_v1"] = "stratified_permutations_v1"
    order_seed: int = 20260927
    followup: Literal["common_control_window_v1"] = "common_control_window_v1"
    terminal_policy: Literal["hold_or_remain_disarmed_v1"] = "hold_or_remain_disarmed_v1"
    ground_mode: Literal["landed_disarmed_stationary_v1"] = "landed_disarmed_stationary_v1"
    ground_max_state_age_s: float = Field(default=.25, gt=0, le=1)
    ground_max_speed_mps: float = Field(default=.02, gt=0, le=.1)
    ground_max_angular_speed_rps: float = Field(default=.02, gt=0, le=.1)
    ground_position_slack_m: float = Field(default=.05, gt=0, le=.5)
    ground_height_tolerance_m: float = Field(default=.01, gt=0, le=.1)
    ground_confirmation_samples: int = Field(default=3, ge=3, le=20)
    require_feasible_inspection_geometry: bool = True
    camera_obscuration: CameraObscurationSpec | None = None

    @model_serializer(mode="wrap")
    def preserve_prior_protocols(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        if self.camera_obscuration is None:
            data.pop("camera_obscuration", None)
        return data
