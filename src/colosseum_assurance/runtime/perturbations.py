"""Bounded simulation-only observation changes, never modifications to evaluator truth."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from colosseum_assurance.interfaces import CapturedFrame
from colosseum_assurance.protocol.camera_obscuration import CameraObscurationSpec
from colosseum_assurance.protocol.expanded import ExpandedStudySpec
from colosseum_assurance.schemas import TruthEvent, Vec3, VehicleState


@dataclass
class PerturbationChannel:
    spec: ExpandedStudySpec
    seed: int
    active: bool = False

    def update(self, now_s: float) -> list[TruthEvent]:
        enabled = self.spec.family in {"degraded_perception", "simulated_manipulation"}
        active = (enabled and self.spec.severity > 0 and self.spec.fault_onset_s <= now_s
                  < self.spec.fault_onset_s + self.spec.fault_duration_s)
        if active == self.active:
            return []
        self.active = active
        return [TruthEvent(
            sim_time_s=now_s, kind="fault_started" if active else "fault_ended",
            detail="bounded delivered-observation transformation",
            payload={"fault_id": f"{self.spec.family}-window-0", "fault_type": self.spec.family,
                     "channel": "camera" if self.spec.family == "degraded_perception" else "state_depth",
                     "source": "simulated_injection", "severity": self.spec.severity},
        )]

    def state(self, state: VehicleState) -> VehicleState:
        if not self.active or self.spec.family != "simulated_manipulation":
            return state
        position = Vec3(x=state.position.x - 8 * self.spec.severity,
                        y=state.position.y, z=state.position.z)
        return state.model_copy(update={"position": position,
                                       "source": "simulated_packet_bias_on_colosseum_estimate"})

    def frames(self, frames: dict[str, CapturedFrame]) -> dict[str, CapturedFrame]:
        changed = dict(frames)
        # Fixed per-realization photometric domain randomization, identical across
        # matched arms and conditions; no runtime random draws depend on actions.
        brightness = (0.65, 0.85, 1.0, 1.15)[self.seed % 4]
        for kind in ("rgb", "depth", "segmentation"):
            frame = frames.get(kind)
            if frame is None or frame.array is None:
                continue
            pixels = frame.array.copy()
            if kind == "rgb":
                pixels = np.clip(pixels.astype(float) * brightness, 0, 255).astype(np.uint8)
            if self.active and self.spec.family == "degraded_perception":
                half = int(pixels.shape[1] * 0.35 * self.spec.severity)
                mid = pixels.shape[1] // 2
                if half:
                    pixels[:, mid - half:mid + half] = 0
            if self.active and self.spec.family == "simulated_manipulation" and kind == "depth":
                valid = np.isfinite(pixels) & (pixels > 0)
                pixels[valid] += 8 * self.spec.severity
            finite = pixels[np.isfinite(pixels)]
            stats = {"path": None, "pixels_as": "none"}
            if finite.size:
                stats.update(min_value=float(finite.min()), max_value=float(finite.max()),
                             mean_value=float(finite.mean()),
                             nonzero_fraction=float(np.count_nonzero(finite) / finite.size))
            changed[kind] = CapturedFrame(ref=frame.ref.model_copy(update=stats), array=pixels)
        return changed


@dataclass
class CameraObscurationChannel:
    """Camera-acquisition transformation, never selected using delivery or decision time."""

    spec: CameraObscurationSpec
    window_start_s: float
    active_interval: int | None = None

    def update(self, now_s: float) -> list[TruthEvent]:
        selected = self.spec.interval_at(now_s - self.window_start_s)
        if selected == self.active_interval:
            return []
        events = []
        for kind, index in (("fault_ended", self.active_interval), ("fault_started", selected)):
            if index is not None:
                events.append(TruthEvent(
                    sim_time_s=now_s, kind=kind,
                    detail="scheduled synthetic camera-only window observed at control tick",
                    payload={"fault_id": f"camera-obscuration-{index}",
                             "fault_type": self.spec.semantics_version,
                             "source": "simulated_injection", "channel": "rgb_and_evaluator_mask",
                             "control_window_start_s": self.window_start_s,
                             "interval": self.spec.intervals[index].model_dump(),
                             "application_clock": "individual acquisition time; see camera sidecar"},
                ))
        self.active_interval = selected
        return events

    def frames(self, frames: dict[str, CapturedFrame], *, requested_s: float
               ) -> tuple[dict[str, CapturedFrame] | None, dict[str, Any]]:
        """Return a refused record rather than silently pairing incompatible image/mask evidence."""
        report: dict[str, Any] = {
            "semantics_version": self.spec.semantics_version,
            "spec": self.spec.model_dump(mode="json"),
            "control_window_start_s": self.window_start_s, "requested_s": requested_s,
            "status": "refused", "reasons": [], "active": None, "interval_index": None,
            "noise": "none; deterministic zero fill, no random draws",
            "scope": "synthetic camera fault; no world, depth, state or weather modification",
        }
        reasons = report["reasons"]
        for name in ("rgb", "segmentation"):
            frame = frames.get(name)
            if frame is None or frame.array is None:
                reasons.append(f"missing {name} pixels")
                continue
            if (frame.array.dtype != np.uint8
                    or frame.array.shape != (frame.ref.height, frame.ref.width, 3)
                    or frame.ref.height <= 0 or frame.ref.width <= 0):
                reasons.append(f"invalid {name} shape or dtype")
            if (not frame.ref.acquisition_time_known or frame.ref.sim_time_s is None
                    or not math.isfinite(frame.ref.sim_time_s)):
                reasons.append(f"unknown {name} acquisition time")
            elif frame.ref.sim_time_s > requested_s + self.spec.max_pairing_gap_s:
                reasons.append(f"future {name} acquisition time")
        if reasons:
            return None, report
        rgb, mask = frames["rgb"], frames["segmentation"]
        assert rgb.ref.sim_time_s is not None and mask.ref.sim_time_s is not None
        rgb_elapsed = rgb.ref.sim_time_s - self.window_start_s
        mask_elapsed = mask.ref.sim_time_s - self.window_start_s
        selected = self.spec.interval_at(rgb_elapsed)
        report.update(rgb_post_cruise_s=rgb_elapsed, mask_post_cruise_s=mask_elapsed,
                      pairing_gap_s=abs(rgb.ref.sim_time_s - mask.ref.sim_time_s))
        if rgb.ref.camera_name != mask.ref.camera_name:
            reasons.append("RGB/mask camera identity mismatch")
        if rgb.array.shape != mask.array.shape:
            reasons.append("RGB/mask shape mismatch")
        if report["pairing_gap_s"] > self.spec.max_pairing_gap_s:
            reasons.append("RGB/mask acquisition times are not paired")
        if selected != self.spec.interval_at(mask_elapsed):
            reasons.append("RGB/mask acquisitions straddle a window boundary")
        if reasons:
            return None, report
        changed = dict(frames)
        if selected is not None:
            for name in ("rgb", "segmentation"):
                frame = frames[name]
                pixels = np.zeros_like(frame.array)
                changed[name] = CapturedFrame(ref=frame.ref.model_copy(update={
                    "path": None, "pixels_as": "none", "min_value": 0., "max_value": 0.,
                    "mean_value": 0., "nonzero_fraction": 0.,
                }), array=pixels)
        report.update(status="applied" if selected is not None else "unchanged",
                      active=selected is not None, interval_index=selected)
        return changed, report
