"""New-study stationary mode evidence, entirely on the permitted onboard/actuator channel."""

from __future__ import annotations

import math

from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.schemas import ControlCommand, ObservationPacket


class GroundModeTracker:
    """Require fresh, consecutive mode observations; height alone can never establish this mode."""

    def __init__(self, protocol: ProtocolConfig):
        self.protocol = protocol
        self.count = 0
        self.last_time: float | None = None
        self.last_position = None

    def evaluate(self, packet: ObservationPacket, command: ControlCommand) -> tuple[bool, str]:
        spec = self.protocol.controlled_study
        if spec is None:
            return False, "legacy_airborne_contract"
        state = packet.state
        evidence = packet.sensor_samples.get("actuation_mode", {})
        sample = evidence.get("sample", {})
        reason = "stationary_disarmed_mode_not_established"
        eligible = False
        if state is not None and evidence.get("available"):
            try:
                age = packet.receive_sim_time_s-state.sim_time_s
                angular = float(sample["angular_speed_rps"])
                target_z = float(sample["declared_landing_z"])
                speed = math.dist(state.velocity.as_tuple(), (0, 0, 0))
                eligible = (
                    sample["contract"] == "reset_valid_disarm_ack_v1"
                    and sample["acknowledged_disarmed"] is True and sample["armed"] is False
                    and sample["api_control_enabled"] is True and state.api_control_enabled
                    and state.landed and command.kind in {"noop", "hold"}
                    and command.yaw_rad is None
                    and all(math.isfinite(v) for v in (age, angular, target_z, speed))
                    and 0 <= age <= spec.ground_max_state_age_s
                    and abs(float(sample["sim_time_s"])-state.sim_time_s) <= 1e-6
                    and speed <= spec.ground_max_speed_mps
                    and angular <= spec.ground_max_angular_speed_rps and angular >= 0
                    and abs(state.position.z-target_z) <= spec.ground_height_tolerance_m
                    and state.position.horizontal_distance_to(self.protocol.mission.home)
                    <= self.protocol.mission.return_tolerance_m
                )
                if eligible and self.last_time is not None and self.last_position is not None:
                    span = state.sim_time_s-self.last_time
                    eligible = (span > 0 and span <= 1.5*self.protocol.mission.control_dt_s
                                and state.position.distance_to(self.last_position)
                                <= spec.ground_position_slack_m+spec.ground_max_speed_mps*span)
                if sample.get("acknowledged_disarmed") and not eligible:
                    reason = "disarm_record_present_but_stationary_mode_evidence_invalid"
            except (KeyError, TypeError, ValueError):
                eligible = False
        self.count = self.count+1 if eligible else 0
        self.last_time = None if state is None else state.sim_time_s
        self.last_position = None if state is None else state.position
        confirmed = self.count >= spec.ground_confirmation_samples
        if confirmed:
            reason = "fresh_landed_disarmed_stationary_mode_confirmed"
        return confirmed, reason
