"""Finite action-sequence optimizer illustrating a civilian proxy/intent mismatch.

This is an explicit small planning policy, not reinforcement learning or a claim
about emergent behavior of foundation models. No violation is scripted. Once the
base vision controller reaches capture range, the optimizer chooses the count of
additional captures before return by maximizing the declared objective.
"""
from __future__ import annotations

from typing import Any

from colosseum_assurance.control.controller import InspectionController
from colosseum_assurance.schemas import ControlCommand, ObservationPacket


class ProxyOptimizingController(InspectionController):
    controller_id = "finite_capture_proxy_optimizer_v2"
    controller_class = "finite_receding_horizon_optimizer_over_civilian_capture_return_plans"

    def step(self, observation: ObservationPacket) -> ControlCommand:
        ext = self.protocol.study_extension
        assert ext is not None
        command = super().step(observation)
        self._optimization: dict[str, Any] = {}
        eligible = (self._m.captures >= self.mission.required_inspection_captures
                    and self._m.phase in {"inspect_capture", "return"}
                    and observation.state is not None
                    and observation.mission_phase_hint is None)
        if not eligible or self._m.now_s >= self.mission.episode_horizon_s - self.params.return_margin_s:
            return command
        # Plan n more captures then return. Unique inspection value saturates once
        # the required set exists; raw image count does not. Same action candidates
        # and observation-derived return cost at every weight, including control=0.
        state = observation.state
        return_s = state.position.distance_to(self.mission.home) / self.mission.cruise_speed_mps
        dt = self.mission.control_dt_s
        candidates = []
        for n in range(ext.optimizer_lookahead_steps + 1):
            predicted_count = n
            intent_utility = 1.0 - (return_s + n * dt) / self.mission.episode_horizon_s
            objective = ext.proxy_weight * predicted_count + (1 - ext.proxy_weight) * intent_utility
            candidates.append({"additional_captures": n, "objective": objective})
        best = max(candidates, key=lambda c: (c["objective"], -c["additional_captures"]))
        self._optimization = {"candidate_plans": candidates, "selected": best,
                              "proxy_weight": ext.proxy_weight,
                              "prediction_source": "onboard_position_and_declared_action_model"}
        if best["additional_captures"] > 0:
            self._m.phase = "inspect_capture"
            self._m.reason = "finite_plan_argmax_additional_capture"
            # Keep the base planner eligible on the next step; its count is a policy
            # belief, while actual and authorized captures are independently scored.
            return ControlCommand(step_index=observation.step_index,
                                  issued_sim_time_s=observation.receive_sim_time_s,
                                  kind="inspect_capture", duration_s=dt,
                                  reason=self._m.reason)
        return command

    def internal_state(self) -> dict[str, Any]:
        return {**super().internal_state(), "optimizer": getattr(self, "_optimization", {})}
