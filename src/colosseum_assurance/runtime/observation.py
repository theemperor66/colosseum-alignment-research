"""The delayed observation channel.

The controller and its guard never see the simulator directly. Samples enter a history buffer with the
simulator time at which they were *measured*, and each control step reads back the newest sample that is
already older than the scheduled delay. Dropouts remove a sample entirely, so the vehicle must act on
stale information. This is where the study's independent variable lives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from colosseum_assurance.scenario.manifest import ScheduleSet
from colosseum_assurance.schemas import (
    AuthorizationView,
    DepthSummary,
    FrameRef,
    ObservationPacket,
    SensorHealth,
    SupervisionView,
    VehicleState,
)


@dataclass(slots=True)
class _Timed:
    measured_at_s: float
    payload: object


@dataclass
class ObservationPipeline:
    """Builds :class:`ObservationPacket` values from measured samples and the exogenous schedule."""

    schedules: ScheduleSet
    declared_delay_bound_s: float
    dropout_window_steps: int = 6
    _states: list[_Timed] = field(default_factory=list)
    _depths: list[_Timed] = field(default_factory=list)
    _rgbs: list[_Timed] = field(default_factory=list)
    _sensors: dict[str, list[_Timed]] = field(default_factory=dict)
    _sensor_errors: dict[str, dict[str, Any]] = field(default_factory=dict)
    _predictions: list[_Timed] = field(default_factory=list)
    _recent_dropouts: list[bool] = field(default_factory=list)
    _last_valid_state_time_s: float | None = None

    # ------------------------------------------------------------------ ingest
    def push_state(self, state: VehicleState) -> None:
        self._states.append(_Timed(state.sim_time_s, state))

    def push_depth(self, summary: DepthSummary) -> None:
        self._depths.append(_Timed(summary.sim_time_s, summary))

    def push_rgb(self, ref: FrameRef) -> None:
        self._rgbs.append(_Timed(ref.sim_time_s, ref))

    def push_sensors(self, samples: dict[str, Any]) -> None:
        for name, result in samples.items():
            if result.get("available"):
                self._sensors.setdefault(name, []).append(_Timed(result["sample"]["sim_time_s"], result))
                self._sensor_errors.pop(name, None)
            else:
                self._sensor_errors[name] = result

    def push_prediction(self, prediction: dict[str, Any]) -> None:
        if prediction.get("sim_time_s") is not None:
            self._predictions.append(_Timed(prediction["sim_time_s"], prediction))

    # ------------------------------------------------------------------ build
    def build(
        self,
        step_index: int,
        now_s: float,
        supervision: SupervisionView,
        authorization: AuthorizationView,
        mission_phase_hint: str | None = None,
    ) -> ObservationPacket:
        """Return what the vehicle can know at ``now_s``, given the scheduled delay and dropouts."""
        delay_s = self.schedules.delay_at_step(step_index)
        cutoff = now_s - delay_s

        state_dropped = self.schedules.state_dropped(step_index)
        depth_dropped = self.schedules.depth_dropped(step_index)
        self._recent_dropouts.append(bool(state_dropped or depth_dropped))
        if len(self._recent_dropouts) > self.dropout_window_steps:
            self._recent_dropouts.pop(0)

        state_item = None if state_dropped else self._newest_at_or_before(self._states, cutoff)
        depth_item = None if depth_dropped else self._newest_at_or_before(self._depths, cutoff)
        rgb_item = None if depth_dropped else self._newest_at_or_before(self._rgbs, cutoff)
        prediction_item = None if depth_dropped else self._newest_at_or_before(self._predictions, cutoff)
        prediction = {} if prediction_item is None else prediction_item.payload

        state = None if state_item is None else state_item.payload
        depth = None if depth_item is None else depth_item.payload
        rgb = None if rgb_item is None else rgb_item.payload

        state_age = None if state is None else max(0.0, now_s - state.sim_time_s)  # type: ignore[union-attr]
        depth_age = None if depth is None else max(0.0, now_s - depth.sim_time_s)  # type: ignore[union-attr]
        if state is not None:
            self._last_valid_state_time_s = state.sim_time_s  # type: ignore[union-attr]

        health = SensorHealth(
            state_sample_available=state is not None,
            depth_available=depth is not None,
            rgb_available=rgb is not None,
            state_age_s=state_age,
            depth_age_s=depth_age,
            dropouts_in_window=sum(1 for v in self._recent_dropouts if v),
            notes="scheduled dropout" if (state_dropped or depth_dropped) else "",
        )
        return ObservationPacket(
            step_index=step_index,
            receive_sim_time_s=now_s,
            state=state,  # type: ignore[arg-type]
            state_age_s=state_age,
            depth=depth,  # type: ignore[arg-type]
            rgb=rgb,  # type: ignore[arg-type]
            supervision=supervision,
            authorization=authorization,
            sensor_health=health,
            perception_evidence=prediction.get("evidence"),
            asset_presence_probability=prediction.get("prediction"),
            perception_model_hash=prediction.get("model_hash"),
            perception_prediction_time_s=prediction.get("sim_time_s"),
            sensor_samples={
                name: ({"available": False, "error": "scheduled dropout or no sample old enough"}
                       if state_dropped or (item := self._newest_at_or_before(history, cutoff)) is None
                       else item.payload)
                for name, history in self._sensors.items()
            } | self._sensor_errors,
            mission_phase_hint=mission_phase_hint,
            declared_observation_delay_s=self.declared_delay_bound_s,
        )

    @staticmethod
    def _newest_at_or_before(items: list[_Timed], cutoff_s: float) -> _Timed | None:
        best: _Timed | None = None
        for item in items:
            if item.measured_at_s <= cutoff_s + 1e-9:
                if best is None or item.measured_at_s > best.measured_at_s:
                    best = item
        return best

    @property
    def last_valid_state_time_s(self) -> float | None:
        return self._last_valid_state_time_s
