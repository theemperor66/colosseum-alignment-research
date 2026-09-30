"""The fixed, goal-directed, perception-based inspection controller.

This is NOT a learned policy. It is a hand-written finite state machine with a depth-servo inner loop.
Every conclusion drawn from this study is a conclusion about *this* controller under monitoring, and
that scope is recorded in ``internal_state()["controller_class"]`` so no downstream artifact can quietly
upgrade it into a claim about learned agents.

Information the controller may use
----------------------------------
Only an :class:`~colosseum_assurance.schemas.ObservationPacket` (delayed, sometimes missing) and a
:class:`~colosseum_assurance.interfaces.MissionBrief` (mission, frozen policy, home, *nominal* asset
position, declared delay bounds). The true asset position is jittered away from nominal and the scene
contains occluders, so the approach must be resolved from the depth-derived bearing and range. The
controller imports no scenario, truth, evaluation, or runtime module.

Three design choices that the experiment depends on
---------------------------------------------------
1. **No delay compensation.** The received state is used as if it were current. That is the ordinary
   engineering choice this study is about: under delay the vehicle acts on a position it has already
   left. Compensating here would hide the effect that the guards are supposed to catch.
2. **No policy enforcement.** The controller *requests* authorization before entering the authorized
   inspection radius and re-requests an expired token, but it never refuses to fly. If the supervisory
   response does not arrive within a bounded wait it continues the mission. Violations therefore emerge
   from ordinary interaction with delayed supervision, not from scripted misbehaviour. Enforcement is
   the guards' job, which is exactly what the three arms compare.
3. **It keeps acting when observation fails.** With no state this step it repeats the last commanded
   target at a decayed speed, and after ``max_blind_steps`` it holds. Freezing on the first dropout
   would remove the failure mode under study.

Determinism: no random number generator is constructed anywhere in this module. Given the same
observation sequence the command sequence is identical, which ``internal_state()["uses_rng"] is False``
records for the audit lane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from colosseum_assurance.control.perception import sector_bearings_rad
from colosseum_assurance.interfaces import MissionBrief
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.schemas import (
    AuthorizationStatus,
    ControlCommand,
    DepthSummary,
    ObservationPacket,
    Vec3,
    VehicleState,
)

__all__ = ["ControllerParams", "InspectionController", "PHASES"]

PHASES: tuple[str, ...] = (
    "init",
    "takeoff",
    "transit",
    "search_align",
    "request_authorization",
    "approach",
    "inspect_capture",
    "return",
    "land",
    "done",
)

GUARD_HINTS: frozenset[str] = frozenset({"inspection_suspended_by_guard", "returning_by_guard"})


@dataclass(frozen=True, slots=True)
class ControllerParams:
    """Frozen controller tuning. Part of the frozen experimental configuration."""

    camera_hfov_rad: float = math.pi / 2.0
    """Horizontal field of view assumed when converting sector indices back to bearings. Colosseum's
    default camera FOV is 90 degrees; the runner may override it through ``brief.extras``."""

    search_margin_m: float = 4.0
    """Transit stops this far outside the authorized inspection radius, so the authorization request
    happens before the boundary rather than after it."""

    align_tolerance_rad: float = 0.12
    max_search_steps: int = 24
    """After this many search steps without a detection the vehicle proceeds on its running estimate.
    A perception failure must degrade the mission, not deadlock it."""

    search_sweep_rad: float = 0.30
    search_strafe_m: float = 1.5

    auth_wait_steps: int = 20
    auth_retry_steps: int = 6
    """Bounded wait for the supervisory response, then the mission continues. With dt = 0.5 s this is
    10 s, which is longer than the nominal and moderate supervisory delays and shorter than the severe
    one: whether the vehicle enters the radius unauthorized is decided by the condition, not by us."""

    approach_step_m: float = 1.5
    emergency_stop_m: float = 1.5
    slow_down_range_m: float = 8.0
    min_speed_fraction: float = 0.25
    clearance_margin_m: float = 2.0
    """A hop is only flown into a sector that reports at least the hop plus this margin."""

    retreat_step_m: float = 1.5
    deviation_hysteresis_rad: float = 0.10
    """Small penalty for switching the side it deviates to, so the vehicle commits to one way round an
    obstacle instead of oscillating in front of it."""

    blind_hop_m: float = 1.0
    no_depth_speed_factor: float = 0.6
    fov_margin_rad: float = 0.15
    """A goal outside the camera's field of view is turned toward before the vehicle moves: the frame
    that justifies a move must cover the direction of that move. Obstacles beside and behind the
    vehicle stay invisible, which is a real limitation of a single forward camera and is left in."""

    asset_bearing_match_rad: float = 0.30
    asset_range_match_m: float = 2.5
    """An obstacle this close to the perceived target in bearing and range *is* the target, so the
    avoidance layer must not steer away from the structure the mission exists to inspect."""

    asset_estimate_gain: float = 0.5
    """Exponential update of the asset position estimate from each accepted detection."""

    asset_prior_radius_m: float = 6.0
    """A detection whose implied world position is further than this from the *nominal* asset position
    is not the inspection asset. The brief declares the surveyed asset location, so rejecting distant
    structures is ordinary mission knowledge, not privileged truth. Without this gate the vehicle
    happily inspects the first wall it sees; with it, perception still has to resolve the metre-scale
    displacement and the bearing that the nominal position cannot supply."""

    asset_prior_bearing_rad: float = 0.5
    """Same gate for a bearing-only (RGB fallback) detection, which has no range to place in the world."""

    max_blind_steps: int = 8
    blind_speed_decay: float = 0.7
    stale_state_margin_s: float = 0.5
    return_margin_s: float = 12.0
    takeoff_tolerance_m: float = 0.8
    land_tolerance_m: float = 0.6


@dataclass(slots=True)
class _Memory:
    """Everything the controller carries between steps. Reset per episode."""

    phase: str = "init"
    previous_phase: str = "init"
    reason: str = "reset"
    step_index: int = -1
    now_s: float = 0.0
    last_state: VehicleState | None = None
    last_state_receive_time_s: float | None = None
    last_state_age_s: float | None = None
    blind_steps: int = 0
    asset_estimate: Vec3 | None = None
    asset_estimate_source: str = "nominal"
    last_target: Vec3 | None = None
    last_speed_mps: float = 0.0
    last_kind: str = "noop"
    perception_used: bool = False
    target_accepted: bool = False
    target_bearing_rad: float | None = None
    target_range_m: float | None = None
    rejected_detections: int = 0
    detections: int = 0
    search_steps: int = 0
    search_side: float = 0.0
    deviation_side: float = 0.0
    auth_wait_steps: int = 0
    auth_requests: int = 0
    last_auth_request_step: int | None = None
    captures: int = 0
    capture_opportunities: int = 0
    capture_pending_step: int | None = None
    capture_execution_feedback: dict[str, Any] | None = None
    capture_unavailability_reason: str | None = None
    dwell_started_s: float | None = None
    guard_override: str | None = None
    notes: list[str] = field(default_factory=list)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _horizontal_distance(a: Vec3, b: Vec3) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _heading_to(origin: Vec3, target: Vec3) -> float:
    return math.atan2(target.y - origin.y, target.x - origin.x)


def _advance(origin: Vec3, heading_rad: float, distance_m: float, altitude_z: float) -> Vec3:
    return Vec3(
        x=origin.x + distance_m * math.cos(heading_rad),
        y=origin.y + distance_m * math.sin(heading_rad),
        z=altitude_z,
    )


def _wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class InspectionController:
    """Fixed goal-directed controller implementing :class:`~colosseum_assurance.interfaces.Controller`."""

    controller_id = "fixed_inspection_v1"
    controller_class = "hand_written_goal_directed_state_machine"

    def __init__(self, protocol: ProtocolConfig, params: ControllerParams | None = None) -> None:
        self.protocol = protocol
        self.mission = protocol.mission
        self.obligations = protocol.obligations
        self.params = params or ControllerParams()
        self._image_width = protocol.simulation.image_width
        self._max_command_duration_s = protocol.simulation.max_command_duration_s
        self._hfov_rad = self.params.camera_hfov_rad
        self.brief: MissionBrief | None = None
        self._m = _Memory()
        self._capture_ack_budget = None

    # -------------------------------------------------------------------- lifecycle
    def reset(self, brief: MissionBrief) -> None:
        """Start a new episode. Nothing survives from the previous one except the frozen tuning."""
        self.brief = brief
        self.mission = brief.mission
        self.obligations = brief.obligations
        hfov = brief.extras.get("camera_hfov_rad") if brief.extras else None
        self._hfov_rad = float(hfov) if hfov else self.params.camera_hfov_rad
        self._m = _Memory(
            asset_estimate=Vec3(
                x=brief.nominal_asset_position.x,
                y=brief.nominal_asset_position.y,
                z=brief.nominal_asset_position.z,
            )
        )

    def enable_capture_acknowledgments(self, max_opportunities: int) -> None:
        """V4 only: proposals are not evidence that the actuator performed a capture."""
        if self.protocol.context_confidence is None or max_opportunities != (
                self.protocol.context_confidence.max_capture_opportunities):
            raise ValueError("capture acknowledgment requires the frozen v4 opportunity budget")
        self._capture_ack_budget = max_opportunities

    def acknowledge_capture(self, proposed: ControlCommand, executed: ControlCommand,
                            rgb_acquired: bool) -> dict[str, Any]:
        m = self._m
        if (self._capture_ack_budget is None or proposed.kind != "inspect_capture"
                or m.capture_pending_step != proposed.step_index
                or executed.step_index != proposed.step_index):
            raise ValueError("capture acknowledgment must match one outstanding v4 proposal")
        performed = executed.kind == "inspect_capture" and rgb_acquired
        if performed:
            m.captures += 1
        m.capture_pending_step = None
        m.capture_execution_feedback = dict(step_index=proposed.step_index, performed=performed,
            executed_kind=executed.kind, rgb_acquired=rgb_acquired,
            acknowledged_captures=m.captures, capture_opportunities=m.capture_opportunities,
            semantics="onboard execution receipt, not independent target-content or safety judgment")
        return dict(m.capture_execution_feedback)

    # -------------------------------------------------------------------- properties
    @property
    def _cruise_z(self) -> float:
        assert self.brief is not None
        return self.brief.home.z - self.mission.cruise_altitude_m

    @property
    def _search_distance_m(self) -> float:
        return self.obligations.authorized_inspection_radius_m + self.params.search_margin_m

    # -------------------------------------------------------------------- main entry
    def step(self, observation: ObservationPacket) -> ControlCommand:
        """Return one bounded command for this observation."""
        if self.brief is None:
            raise RuntimeError("InspectionController.step called before reset(brief)")
        m = self._m
        m.step_index = observation.step_index
        m.now_s = observation.receive_sim_time_s
        m.perception_used = False

        self._route_guard_hint(observation)
        self._ingest_state(observation)
        depth = observation.depth
        self._ingest_perception(observation, depth)

        if m.last_state is None:
            return self._emit("hold", reason="no_state_received_yet")
        if observation.state is None:
            return self._blind_command()

        command = self._dispatch(observation, depth)
        m.last_kind = command.kind
        if command.target is not None and command.kind == "move_to":
            m.last_target = command.target
            m.last_speed_mps = command.speed_mps or 0.0
        return command

    def internal_state(self) -> dict[str, Any]:
        """Provenance-rich decision context. Must stay JSON-serialisable: it is written to evidence."""
        m = self._m
        return {
            "controller_id": self.controller_id,
            "controller_class": self.controller_class,
            "is_learned_policy": False,
            "uses_rng": False,
            "policy_version": self.brief.policy_version if self.brief else None,
            "phase": m.phase,
            "previous_phase": m.previous_phase,
            "reason": m.reason,
            "target": None if m.last_target is None else m.last_target.model_dump(mode="json"),
            "commanded_speed_mps": m.last_speed_mps,
            "last_command_kind": m.last_kind,
            "perception_used": m.perception_used,
            "perception_detections": m.detections,
            "asset_estimate": None if m.asset_estimate is None else m.asset_estimate.model_dump(mode="json"),
            "asset_estimate_source": m.asset_estimate_source,
            "last_valid_observation_sim_time_s": m.last_state_receive_time_s,
            "last_observed_state_age_s": m.last_state_age_s,
            "blind_steps": m.blind_steps,
            "authorization_requests": m.auth_requests,
            "authorization_wait_steps": m.auth_wait_steps,
            "captures_issued": m.captures if self._capture_ack_budget is None else m.capture_opportunities,
            **({"captures_acknowledged": m.captures, "capture_opportunities": m.capture_opportunities,
                "capture_pending_step": m.capture_pending_step,
                "capture_execution_feedback": m.capture_execution_feedback,
                "capture_unavailability_reason": m.capture_unavailability_reason}
               if self._capture_ack_budget is not None else {}),
            "guard_override": m.guard_override,
            "notes": list(m.notes[-4:]),
        }

    # -------------------------------------------------------------------- ingestion
    def _route_guard_hint(self, observation: ObservationPacket) -> None:
        """Do not fight the guard. A suspension or a guard return ends the inspection attempt.

        The controller gives up the inspection and flies home; it never re-enters the radius after a
        guard has intervened. The intervention burden that the analysis reports (suspension,
        abandonment, completion time) is therefore a real consequence of the guard, not of a controller
        that quietly resumed.
        """
        hint = observation.mission_phase_hint
        m = self._m
        if hint in GUARD_HINTS and m.phase not in {"return", "land", "done"}:
            m.guard_override = hint
            self._set_phase("return", f"guard_hint:{hint}")
        elif hint in GUARD_HINTS:
            m.guard_override = hint

    def _ingest_state(self, observation: ObservationPacket) -> None:
        m = self._m
        if observation.state is not None:
            m.last_state = observation.state
            m.last_state_receive_time_s = observation.receive_sim_time_s
            m.last_state_age_s = observation.state_age_s
            m.blind_steps = 0
        else:
            m.blind_steps += 1

    def _ingest_perception(self, observation: ObservationPacket, depth: DepthSummary | None) -> None:
        """Accept or reject this step's detection, then fold an accepted one into the asset estimate.

        The nominal asset position is only a starting point: the true asset is displaced from it, so an
        approach flown on the nominal position alone would miss the inspection tolerance. Perception
        supplies that displacement. A detection is rejected when the world position it implies is
        further from the *declared nominal* asset position than ``asset_prior_radius_m``; the detector
        otherwise locks onto the first wall or shed with tower-like proportions. The gate uses only
        mission-brief information.
        """
        m = self._m
        assert self.brief is not None
        m.target_accepted = False
        m.target_bearing_rad = None
        m.target_range_m = None
        if depth is None or not depth.target_visible or depth.target_bearing_rad is None:
            return
        state = observation.state or m.last_state
        if state is None:
            return
        nominal = self.brief.nominal_asset_position

        if depth.target_range_m is None:
            # Bearing-only (for example the RGB fallback): usable for alignment, not for position.
            prior_bearing = _wrap_pi(_heading_to(state.position, nominal) - state.yaw_rad)
            if abs(_wrap_pi(depth.target_bearing_rad - prior_bearing)) <= self.params.asset_prior_bearing_rad:
                m.target_accepted = True
                m.target_bearing_rad = depth.target_bearing_rad
                m.perception_used = True
            else:
                m.rejected_detections += 1
            return

        world_bearing = state.yaw_rad + depth.target_bearing_rad
        horizontal_range = (depth.target_horizontal_range_m if depth.target_horizontal_range_m is not None
                            else depth.target_range_m)
        seen = Vec3(
            x=state.position.x + horizontal_range * math.cos(world_bearing),
            y=state.position.y + horizontal_range * math.sin(world_bearing),
            z=m.asset_estimate.z if m.asset_estimate else state.position.z,
        )
        if _horizontal_distance(seen, nominal) > self.params.asset_prior_radius_m:
            m.rejected_detections += 1
            return
        gain = self.params.asset_estimate_gain
        prior = m.asset_estimate or seen
        m.asset_estimate = Vec3(
            x=(1.0 - gain) * prior.x + gain * seen.x,
            y=(1.0 - gain) * prior.y + gain * seen.y,
            z=prior.z,
        )
        m.asset_estimate_source = "depth_detection"
        m.detections += 1
        m.target_accepted = True
        m.target_bearing_rad = depth.target_bearing_rad
        m.target_range_m = depth.target_range_m
        m.perception_used = True

    # -------------------------------------------------------------------- command helpers
    def _emit(
        self,
        kind: str,
        *,
        target: Vec3 | None = None,
        speed: float | None = None,
        yaw: float | None = None,
        duration: float | None = None,
        reason: str = "",
    ) -> ControlCommand:
        """Build one bounded command. Speed and duration are always clamped, never open ended."""
        m = self._m
        dt = self.mission.control_dt_s
        bounded_duration = _clamp(duration if duration is not None else dt, 0.0, self._max_command_duration_s)
        bounded_speed = (
            None if speed is None else _clamp(speed, 0.0, self.mission.cruise_speed_mps)
        )
        m.reason = reason
        return ControlCommand(
            step_index=m.step_index,
            issued_sim_time_s=m.now_s,
            kind=kind,  # type: ignore[arg-type]
            target=target,
            speed_mps=bounded_speed,
            duration_s=bounded_duration,
            yaw_rad=None if yaw is None else _wrap_pi(yaw),
            reason=reason,
            issued_by="controller",
            controller_phase=m.phase,
        )

    def _set_phase(self, phase: str, reason: str) -> None:
        if phase not in PHASES and not (phase == "aborted" and self._capture_ack_budget is not None):
            raise ValueError(f"unknown phase {phase!r}")
        m = self._m
        if phase != m.phase:
            m.previous_phase = m.phase
            m.phase = phase
            m.notes.append(f"{m.now_s:.2f}:{m.previous_phase}->{phase}:{reason}")

    def _blind_command(self) -> ControlCommand:
        """No state arrived this step. Keep flying the last intent, slower, then hold.

        Documented behaviour, because it is the behaviour the study measures: the vehicle does not stop
        on the first missing packet, and it does not continue indefinitely either.
        """
        m = self._m
        if m.blind_steps > self.params.max_blind_steps or m.last_target is None or m.last_kind != "move_to":
            return self._emit("hold", reason=f"observation_missing_{m.blind_steps}_steps_holding")
        decayed = m.last_speed_mps * (self.params.blind_speed_decay**m.blind_steps)
        floor = self.params.min_speed_fraction * self.mission.approach_speed_mps
        return self._emit(
            "move_to",
            target=m.last_target,
            speed=max(decayed, floor),
            reason=f"observation_missing_{m.blind_steps}_steps_continuing_on_last_target",
        )

    # -------------------------------------------------------------------- perception-driven steering
    def _sector_bearings(self, n: int) -> list[float]:
        return [float(b) for b in sector_bearings_rad(n, self._hfov_rad, self._image_width)]

    def _obstacle_is_target(self, depth: DepthSummary) -> bool:
        """True when the nearest surface in the frame is the accepted inspection target itself."""
        m = self._m
        if not m.target_accepted or m.target_bearing_rad is None or depth.obstacle_bearing_rad is None:
            return False
        offset = abs(_wrap_pi(depth.obstacle_bearing_rad - m.target_bearing_rad))
        if offset > self.params.asset_bearing_match_rad:
            return False
        if m.target_range_m is None or depth.min_range_m is None:
            return False
        return abs(depth.min_range_m - m.target_range_m) <= self.params.asset_range_match_m

    def _free_heading(
        self, depth: DepthSummary, goal_rel_bearing: float, desired_step_m: float
    ) -> tuple[float | None, float, str]:
        """Pick the sector closest to the goal direction that is clear enough for the intended hop.

        This is the whole obstacle-avoidance rule: a hop is flown only into a sector whose reported
        range covers the hop plus ``clearance_margin_m``, and among those the one nearest the goal
        bearing wins. Steering toward free space is better defined than steering away from the nearest
        return, because a wall filling the middle of the frame has no unambiguous "away" direction.
        Returns ``(None, 0, "boxed_in")`` when nothing is clear enough, which the caller answers with a
        retreat rather than with an indefinite hold.
        """
        ranges = depth.sector_min_range_m
        if not ranges:
            return goal_rel_bearing, desired_step_m, "no_sectors"
        bearings = self._sector_bearings(len(ranges))
        margin = self.params.clearance_margin_m
        free = [i for i, r in enumerate(ranges) if r >= desired_step_m + margin]
        if free:
            last = self._m.deviation_side

            def cost(i: int) -> tuple[float, float]:
                offset = _wrap_pi(bearings[i] - goal_rel_bearing)
                switched = last != 0.0 and offset != 0.0 and math.copysign(1.0, offset) != last
                penalty = self.params.deviation_hysteresis_rad if switched else 0.0
                return (abs(offset) + penalty, abs(bearings[i]))

            best = min(free, key=cost)
            offset = _wrap_pi(bearings[best] - goal_rel_bearing)
            deviation = abs(offset)
            aligned = deviation <= self.params.align_tolerance_rad
            self._m.deviation_side = 0.0 if aligned else math.copysign(1.0, offset)
            note = "clear" if aligned else f"deviating_{offset:+.2f}rad"
            return bearings[best], min(desired_step_m, max(ranges[best] - margin, 0.0)), note
        widest = max(range(len(ranges)), key=lambda i: ranges[i])
        if ranges[widest] > margin:
            return bearings[widest], min(desired_step_m, ranges[widest] - margin), "constrained"
        return None, 0.0, "boxed_in"

    def _retreat(self, state: VehicleState, depth: DepthSummary | None, reason: str) -> ControlCommand:
        """Back away from the nearest surface, keeping the camera on it.

        Reversing is flown blind, which is why the step is short and slow. Holding instead would be
        worse: the vehicle would sit in front of an obstacle until the horizon expired.
        """
        bearing = 0.0
        if depth is not None and depth.obstacle_bearing_rad is not None:
            bearing = depth.obstacle_bearing_rad
        heading = state.yaw_rad + bearing + math.pi
        target = _advance(state.position, heading, self.params.retreat_step_m, state.position.z)
        return self._emit(
            "move_to",
            target=target,
            speed=self.mission.approach_speed_mps * self.params.min_speed_fraction * 2.0,
            yaw=state.yaw_rad,
            reason=reason,
        )

    def _too_close(self, depth: DepthSummary | None) -> bool:
        """True when a surface blocks the corridor straight ahead, or touches the vehicle envelope.

        The emergency test deliberately uses the free path rather than the frame-wide minimum: a thin
        mast one metre to the side is something to steer past, not something to back away from. An
        earlier version retreated from any near return and wedged the vehicle between two obstacles
        until the horizon expired.
        """
        if depth is None or not depth.valid:
            return False
        p = self.params
        if self._obstacle_is_target(depth):
            floor = max(p.emergency_stop_m * 0.5, 0.6)
            return depth.min_range_m is not None and depth.min_range_m < floor
        if depth.free_path_m is not None and depth.free_path_m < p.emergency_stop_m:
            return True
        return depth.min_range_m is not None and depth.min_range_m < p.emergency_stop_m * 0.6

    def _plan_move(
        self,
        goal: Vec3,
        base_speed: float,
        depth: DepthSummary | None,
        *,
        ignore_target_structure: bool,
        max_hop_m: float,
        reason: str,
    ) -> ControlCommand:
        """Turn a goal point into one bounded, depth-checked command."""
        m = self._m
        state = m.last_state
        assert state is not None
        p = self.params
        goal_heading = _heading_to(state.position, goal)
        rel = _wrap_pi(goal_heading - state.yaw_rad)
        if abs(rel) > self._hfov_rad / 2.0 - p.fov_margin_rad:
            return self._emit(
                "hold", target=state.position, yaw=goal_heading, reason=f"{reason}:turning_to_face_goal"
            )
        distance = _horizontal_distance(state.position, goal)
        desired = _clamp(distance, 0.0, max_hop_m)
        if desired < 1e-3:
            return self._emit("hold", target=state.position, yaw=goal_heading, reason=f"{reason}:at_goal")

        speed = base_speed
        if depth is None or not depth.valid:
            step = min(desired, p.blind_hop_m)
            return self._emit(
                "move_to",
                target=_advance(state.position, goal_heading, step, goal.z),
                speed=speed * p.no_depth_speed_factor,
                yaw=goal_heading,
                reason=f"{reason}:no_depth_cautious",
            )
        if depth.free_path_m is not None and depth.free_path_m < p.slow_down_range_m:
            speed *= _clamp(depth.free_path_m / p.slow_down_range_m, p.min_speed_fraction, 1.0)
        if ignore_target_structure and self._obstacle_is_target(depth):
            return self._emit(
                "move_to",
                target=_advance(state.position, goal_heading, desired, goal.z),
                speed=speed,
                yaw=goal_heading,
                reason=f"{reason}:closing_on_inspection_target",
            )
        if self._too_close(depth):
            return self._retreat(state, depth, f"{reason}:emergency_retreat")
        chosen, allowed, note = self._free_heading(depth, rel, desired)
        if chosen is None:
            return self._retreat(state, depth, f"{reason}:boxed_in")
        if allowed < 0.1:
            return self._emit(
                "hold", target=state.position, yaw=goal_heading, reason=f"{reason}:no_room:{note}"
            )
        heading = state.yaw_rad + chosen
        return self._emit(
            "move_to",
            target=_advance(state.position, heading, allowed, goal.z),
            speed=speed,
            yaw=heading,
            reason=f"{reason}:{note}",
        )

    def _dispatch(self, observation: ObservationPacket, depth: DepthSummary | None) -> ControlCommand:
        """Run phase handlers until one produces a command (bounded, so a transition cannot spin)."""
        self._check_return_budget(observation)
        for _ in range(len(PHASES)):
            handler = getattr(self, f"_phase_{self._m.phase}")
            command = handler(observation, depth)
            if command is not None:
                return command
        return self._emit("hold", reason="phase_machine_made_no_command")

    def _check_return_budget(self, observation: ObservationPacket) -> None:
        """Leave enough horizon to fly home. Running out of time is a mission failure, not a guard event."""
        m = self._m
        state = m.last_state
        if state is None or m.phase in {"return", "land", "done", "aborted"}:
            return
        assert self.brief is not None
        distance = _horizontal_distance(state.position, self.brief.home)
        travel_s = distance / max(self.mission.cruise_speed_mps, 0.1)
        needed_s = observation.receive_sim_time_s + travel_s + self.params.return_margin_s
        if needed_s >= self.mission.episode_horizon_s:
            self._set_phase("return", "return_budget_exhausted")

    def _phase_init(
        self, observation: ObservationPacket, depth: DepthSummary | None
    ) -> ControlCommand | None:
        self._set_phase("takeoff", "armed")
        return self._emit("arm", reason="arming_before_takeoff")

    def _phase_takeoff(
        self, observation: ObservationPacket, depth: DepthSummary | None
    ) -> ControlCommand | None:
        state = self._m.last_state
        assert state is not None and self.brief is not None
        if state.position.z <= self._cruise_z + self.params.takeoff_tolerance_m:
            self._set_phase("transit", "cruise_altitude_reached")
            return None
        return self._emit(
            "takeoff",
            target=Vec3(x=self.brief.home.x, y=self.brief.home.y, z=self._cruise_z),
            duration=min(self._max_command_duration_s, 5.0),
            reason="climbing_to_cruise_altitude",
        )

    def _phase_transit(
        self, observation: ObservationPacket, depth: DepthSummary | None
    ) -> ControlCommand | None:
        m = self._m
        state = m.last_state
        assert state is not None and m.asset_estimate is not None
        distance = _horizontal_distance(state.position, m.asset_estimate)
        if distance <= self._search_distance_m + 0.5:
            self._set_phase("search_align", "reached_search_standoff")
            return None
        heading = _heading_to(state.position, m.asset_estimate)
        hold_off = max(distance - self._search_distance_m, 0.5)
        goal = _advance(state.position, heading, hold_off, self._cruise_z)
        max_hop = self.mission.cruise_speed_mps * self.mission.control_dt_s * 4.0
        return self._plan_move(
            goal,
            self.mission.cruise_speed_mps,
            depth,
            ignore_target_structure=False,
            max_hop_m=max_hop,
            reason="transit_to_asset_estimate",
        )

    def _phase_search_align(
        self, observation: ObservationPacket, depth: DepthSummary | None
    ) -> ControlCommand | None:
        m = self._m
        state = m.last_state
        assert state is not None and m.asset_estimate is not None
        if m.target_accepted and m.target_range_m is not None and m.target_bearing_rad is not None:
            if abs(m.target_bearing_rad) <= self.params.align_tolerance_rad:
                self._set_phase("request_authorization", "target_acquired_and_aligned")
                return None
            heading = state.yaw_rad + m.target_bearing_rad
            m.search_steps += 1
            # Native moveToPosition may finish a short/reached path without applying its yaw.
            # A heading-bearing hold uses the runner's explicit bounded rotateToYaw command.
            return self._emit(
                "hold",
                target=state.position,
                yaw=heading,
                reason=f"yawing_onto_perceived_target_bearing_{m.target_bearing_rad:+.2f}rad",
            )
        m.search_steps += 1
        if m.search_steps > self.params.max_search_steps:
            self._set_phase("request_authorization", "target_not_acquired_proceeding_on_estimate")
            return None
        # Deterministic sweep plus a sidestep toward the clearer half of the frame: an occluder is
        # broken by moving around it, and the depth profile is the only evidence of which way is open.
        phase_in_sweep = m.search_steps % 4
        offsets = (0.0, self.params.search_sweep_rad, 0.0, -self.params.search_sweep_rad)
        heading = _heading_to(state.position, m.asset_estimate) + offsets[phase_in_sweep]
        if m.search_side == 0.0:
            m.search_side = self._clearer_side(depth)
        sidestep = heading + m.search_side * math.pi / 2.0
        goal = _advance(state.position, sidestep, self.params.search_strafe_m, self._cruise_z)
        return self._plan_move(
            goal,
            self.mission.approach_speed_mps,
            depth,
            ignore_target_structure=False,
            max_hop_m=self.params.search_strafe_m,
            reason=f"searching_for_target_sweep_{phase_in_sweep}",
        )

    def _clearer_side(self, depth: DepthSummary | None) -> float:
        """+1 to sidestep right, -1 to sidestep left. Chosen once, from the depth profile.

        Committing to one side matters: alternating sidesteps would leave the vehicle oscillating in
        front of an occluder until the search budget ran out.
        """
        if depth is None or not depth.valid or len(depth.sector_min_range_m) < 2:
            return 1.0
        ranges = depth.sector_min_range_m
        half = len(ranges) // 2
        left = sum(ranges[:half]) / max(half, 1)
        right = sum(ranges[len(ranges) - half:]) / max(half, 1)
        return 1.0 if right >= left else -1.0

    def _phase_request_authorization(
        self, observation: ObservationPacket, depth: DepthSummary | None
    ) -> ControlCommand | None:
        m = self._m
        state = m.last_state
        assert state is not None
        auth = observation.authorization
        now = observation.receive_sim_time_s
        valid = (
            auth.status is AuthorizationStatus.GRANTED
            and (auth.expires_at_s is None or auth.expires_at_s > now)
        )
        if valid:
            self._set_phase("approach", "authorization_granted")
            return None
        if auth.status is AuthorizationStatus.DENIED:
            self._set_phase("return", "authorization_denied")
            return None
        if m.auth_wait_steps > self.params.auth_wait_steps:
            self._set_phase(
                "approach",
                "authorization_wait_timeout_proceeding",
            )
            m.notes.append(f"{now:.2f}:entering_radius_without_valid_authorization")
            return None
        m.auth_wait_steps += 1
        due = (
            m.last_auth_request_step is None
            or (m.step_index - m.last_auth_request_step) >= self.params.auth_retry_steps
        )
        if due:
            m.last_auth_request_step = m.step_index
            m.auth_requests += 1
            return self._emit(
                "request_authorization",
                reason=f"requesting_inspection_authorization_attempt_{m.auth_requests}",
            )
        heading = _heading_to(state.position, m.asset_estimate) if m.asset_estimate else state.yaw_rad
        return self._emit(
            "hold", target=state.position, yaw=heading,
            reason=f"waiting_for_authorization_step_{m.auth_wait_steps}",
        )

    def _phase_approach(
        self, observation: ObservationPacket, depth: DepthSummary | None
    ) -> ControlCommand | None:
        m = self._m
        state = m.last_state
        assert state is not None and m.asset_estimate is not None
        standoff = self.mission.inspection_standoff_m
        rerequest = self._maybe_rerequest_authorization(observation)
        if rerequest is not None:
            return rerequest

        servo = m.target_accepted and m.target_range_m is not None and m.target_bearing_rad is not None
        if servo:
            assert m.target_range_m is not None and m.target_bearing_rad is not None
            error = m.target_range_m - standoff
            heading = state.yaw_rad + m.target_bearing_rad
            source = "depth_servo"
        else:
            error = _horizontal_distance(state.position, m.asset_estimate) - standoff
            heading = _heading_to(state.position, m.asset_estimate)
            source = "estimate_only"
        if abs(error) <= self.mission.inspection_tolerance_m:
            self._set_phase("inspect_capture", f"standoff_reached_{source}")
            return None
        if error < 0.0:
            # Too close. Back straight out while keeping the camera on the structure; turning around
            # here would lose the range measurement that the standoff servo needs.
            back = _advance(state.position, heading + math.pi, min(-error, self.params.retreat_step_m),
                            self._cruise_z)
            return self._emit(
                "move_to",
                target=back,
                speed=self.mission.approach_speed_mps * self.params.min_speed_fraction * 2.0,
                yaw=heading,
                reason=f"approach_{source}_backing_off_range_error_{error:+.2f}m",
            )
        stride = min(error, self.params.approach_step_m)
        goal = _advance(state.position, heading, stride, self._cruise_z)
        return self._plan_move(
            goal,
            self.mission.approach_speed_mps,
            depth,
            ignore_target_structure=True,
            max_hop_m=self.params.approach_step_m,
            reason=f"approach_{source}_range_error_{error:+.2f}m",
        )


    def _maybe_rerequest_authorization(self, observation: ObservationPacket) -> ControlCommand | None:
        """Re-request a token that the received view shows as expired. Requesting is not enforcing."""
        m = self._m
        auth = observation.authorization
        now = observation.receive_sim_time_s
        stale = auth.status is AuthorizationStatus.EXPIRED or (
            auth.expires_at_s is not None and auth.expires_at_s <= now
        )
        if not stale:
            return None
        if (
            m.last_auth_request_step is not None
            and (m.step_index - m.last_auth_request_step) < self.params.auth_retry_steps
        ):
            return None
        m.last_auth_request_step = m.step_index
        m.auth_requests += 1
        return self._emit(
            "request_authorization",
            reason=f"authorization_expired_rerequesting_attempt_{m.auth_requests}",
        )

    def _phase_inspect_capture(
        self, observation: ObservationPacket, depth: DepthSummary | None
    ) -> ControlCommand | None:
        m = self._m
        state = m.last_state
        assert state is not None and m.asset_estimate is not None
        now = observation.receive_sim_time_s
        if m.dwell_started_s is None:
            m.dwell_started_s = now
        dwell_done = (now - m.dwell_started_s) >= self.mission.inspection_dwell_s
        if m.captures >= self.mission.required_inspection_captures and dwell_done:
            self._set_phase("return", "inspection_complete")
            return None
        heading = (
            state.yaw_rad + m.target_bearing_rad
            if m.target_accepted and m.target_bearing_rad is not None
            else _heading_to(state.position, m.asset_estimate)
        )
        if m.captures < self.mission.required_inspection_captures:
            if self._capture_ack_budget is not None:
                if m.capture_pending_step is not None:
                    return self._emit("hold", reason="awaiting_capture_execution_acknowledgment")
                if m.capture_opportunities >= self._capture_ack_budget:
                    m.capture_unavailability_reason = "context_capture_opportunities_exhausted"
                    self._set_phase("return", m.capture_unavailability_reason)
                    return None
                m.capture_opportunities += 1
                m.capture_pending_step = m.step_index
            else:
                m.captures += 1
            return self._emit(
                "inspect_capture",
                target=state.position,
                yaw=heading,
                duration=min(
                    self.mission.inspection_dwell_s / self.mission.required_inspection_captures,
                    self._max_command_duration_s,
                ),
                reason=(f"inspection_capture_{m.captures}_of_{self.mission.required_inspection_captures}"
                        if self._capture_ack_budget is None else
                        f"capture_opportunity_{m.capture_opportunities};acknowledged_{m.captures}"),
            )
        return self._emit(
            "hold", target=state.position, yaw=heading,
            reason=f"dwelling_until_{self.mission.inspection_dwell_s:.1f}s_elapsed",
        )

    def _phase_return(
        self, observation: ObservationPacket, depth: DepthSummary | None
    ) -> ControlCommand | None:
        m = self._m
        state = m.last_state
        assert state is not None and self.brief is not None
        home = self.brief.home
        distance = _horizontal_distance(state.position, home)
        if distance <= self.mission.return_tolerance_m:
            self._set_phase("land", "home_reached")
            return None
        goal = Vec3(x=home.x, y=home.y, z=self._cruise_z)
        max_hop = self.mission.cruise_speed_mps * self.mission.control_dt_s * 4.0
        return self._plan_move(
            goal,
            self.mission.cruise_speed_mps,
            depth,
            ignore_target_structure=False,
            max_hop_m=max_hop,
            reason="returning_home",
        )

    def _phase_land(
        self, observation: ObservationPacket, depth: DepthSummary | None
    ) -> ControlCommand | None:
        m = self._m
        state = m.last_state
        assert state is not None and self.brief is not None
        # Being near the ground is not evidence of landing: stopping descent there can leave the
        # vehicle hovering above its support. Completion requires the onboard landed report.
        if state.landed:
            if (self._capture_ack_budget is not None
                    and m.captures < self.mission.required_inspection_captures):
                m.capture_unavailability_reason = (m.capture_unavailability_reason
                    or "required_capture_acknowledgments_unavailable")
                self._set_phase("aborted", m.capture_unavailability_reason)
                return self._emit("noop", reason=m.capture_unavailability_reason)
            self._set_phase("done", "landed")
            return None
        return self._emit("land", duration=min(self._max_command_duration_s, 5.0), reason="landing_at_home")

    def _phase_done(
        self, observation: ObservationPacket, depth: DepthSummary | None
    ) -> ControlCommand | None:
        return self._emit("noop", reason="mission_finished")

    def _phase_aborted(self, observation: ObservationPacket, depth: DepthSummary | None) -> ControlCommand:
        return self._emit("noop", reason=self._m.capture_unavailability_reason or "mission_unavailable")
