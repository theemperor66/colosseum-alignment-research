"""Explicit synthetic camera fault, independent of the historical multi-modal F1 fault."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from colosseum_assurance.schemas import StrictModel


class CameraObscurationInterval(StrictModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)

    @model_validator(mode="after")
    def positive_duration(self):
        if self.end_s <= self.start_s:
            raise ValueError("camera interval must have positive duration")
        return self


class CameraObscurationSpec(StrictModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    semantics_version: Literal["scheduled_rgb_mask_obscuration_v1"] = "scheduled_rgb_mask_obscuration_v1"
    clock_basis: Literal["post_cruise_acquisition_time"] = "post_cruise_acquisition_time"
    transformation: Literal["zero_rgb_and_paired_mask_only"] = "zero_rgb_and_paired_mask_only"
    intervals: list[CameraObscurationInterval]
    max_pairing_gap_s: float = Field(default=0.001, ge=0, le=0.001)

    @model_validator(mode="after")
    def ordered_disjoint(self):
        for left, right in zip(self.intervals, self.intervals[1:], strict=False):
            if right.start_s < left.end_s:
                raise ValueError("camera intervals must be ordered and nonoverlapping")
        return self

    def interval_at(self, post_cruise_s: float) -> int | None:
        return next(
            (i for i, row in enumerate(self.intervals) if row.start_s <= post_cruise_s < row.end_s), None
        )
