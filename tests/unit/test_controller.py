"""The fixed controller is the same in every arm, so any drift in it changes every measured result.

WHY these tests exist
---------------------
Three properties carry the whole experiment and none of them is obvious from reading the class:

1. **Determinism.** The arms are compared pairwise on the same scenario realizations. If the controller
   were not a pure function of its observation sequence, a paired difference would partly measure noise
   in the controller. ``internal_state()["uses_rng"] is False`` claims this; the test proves it.
2. **Perception really drives the commands.** The brief carries only the NOMINAL asset position. If the
   controller flew on that alone, the study would not be a perception-based closed loop at all. The test
   holds the brief fixed, moves the TRUE asset, and requires different commands.
3. **The documented degraded behaviours.** Missing observations, stale observations and guard hints each
   have a behaviour written down in the module docstring. Those sentences are the specification the
   analysis quotes, so each one is asserted here in a test named after it.

The plant below is a deliberately trivial kinematic stand-in, and the depth summaries are built by hand
from known geometry. Neither is a simulator, and no result produced here is experimental evidence about
Colosseum; this file tests only the controller's decision logic. Depth summarisation itself is tested
from real pixel arrays in ``tests/unit/test_perception.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import pytest

from colosseum_assurance.control.controller import PHASES, ControllerParams, InspectionController
from colosseum_assurance.interfaces import MissionBrief
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.schemas import (
    AuthorizationStatus,
    AuthorizationView,
    ControlCommand,
    DepthSummary,
    ObservationPacket,
    SensorHealth,
    SupervisionView,
    Vec3,
    VehicleState,
)

PROTOCOL = ProtocolConfig()
MISSION = PROTOCOL.mission
PARAMS = ControllerParams()  # the frozen controller tuning, read from the module under test
DT = MISSION.control_dt_s
HFOV_RAD = math.radians(PROTOCOL.simulation.camera_hfov_deg)
SENSING_HORIZON_M = 60.0
N_SECTORS = 7

# The scenario generator displaces the true asset from the nominal position; the controller is told only
# the nominal one, exactly as in a real episode.
TRUE_ASSET = Vec3(x=28.6, y=1.2, z=-6.0)


def make_brief() -> MissionBrief:
    """A brief with the NOMINAL asset position only, as the runner builds it."""
    return MissionBrief(
        mission=MISSION,
        obligations=PROTOCOL.obligations,
        home=MISSION.home,
        nominal_asset_position=MISSION.asset_nominal_position,
        declared_observation_delay_s=0.0,
        declared_supervision_delay_s=1.0,
        policy_version=PROTOCOL.obligations.policy_version,
        camera_name=PROTOCOL.simulation.camera_name,
        extras={"camera_hfov_rad": HFOV_RAD},
    )


@dataclass
class FakeVehicle:
    """A first-order kinematic stand-in for the plant. Not a simulator, and never evidence.

    It exists only so the controller sees the consequences of its own commands and can therefore leave
    one phase and enter the next. Motion is the simplest thing that respects the commanded speed.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw_rad: float = 0.0
    landed: bool = True
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def apply(self, command: ControlCommand) -> None:
        before = (self.x, self.y, self.z)
        if command.yaw_rad is not None:
            self.yaw_rad = command.yaw_rad
        if command.kind == "takeoff" and command.target is not None:
            self.z = max(command.target.z, self.z - 2.0 * DT)
            self.landed = False
        elif command.kind == "move_to" and command.target is not None:
            speed = command.speed_mps or 0.0
            dx, dy, dz = command.target.x - self.x, command.target.y - self.y, command.target.z - self.z
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            hop = min(distance, speed * DT)
            if distance > 1e-9:
                self.x += dx / distance * hop
                self.y += dy / distance * hop
                self.z += dz / distance * hop
            self.landed = False
        elif command.kind == "land":
            self.z = min(0.0, self.z + 2.0 * DT)
            self.landed = self.z >= -0.6
        moved = ((self.x, self.y, self.z), before)
        self.velocity = tuple((now - was) / DT for now, was in zip(*moved, strict=True))

    def state(self, sim_time_s: float) -> VehicleState:
        return VehicleState(
            sim_time_s=sim_time_s,
            position=Vec3(x=self.x, y=self.y, z=self.z),
            velocity=Vec3(x=self.velocity[0], y=self.velocity[1], z=self.velocity[2]),
            yaw_rad=self.yaw_rad,
            landed=self.landed,
        )


def perceive(vehicle: FakeVehicle, asset: Vec3, sim_time_s: float) -> DepthSummary:
    """Depth features for an otherwise empty scene containing only the inspection asset.

    The bearing is camera-relative and positive to the right, which is the convention asserted in
    ``tests/unit/test_perception.py``. The asset is visible only while it is inside the field of view,
    so the controller has to yaw onto it rather than assume it.
    """
    dx, dy = asset.x - vehicle.x, asset.y - vehicle.y
    distance = math.hypot(dx, dy)
    bearing = math.atan2(
        math.sin(math.atan2(dy, dx) - vehicle.yaw_rad), math.cos(math.atan2(dy, dx) - vehicle.yaw_rad)
    )
    visible = abs(bearing) <= HFOV_RAD / 2.0 and distance <= SENSING_HORIZON_M and vehicle.z < -1.0
    return DepthSummary(
        sim_time_s=sim_time_s,
        camera_name=PROTOCOL.simulation.camera_name,
        valid=True,
        min_range_m=distance if visible else None,
        free_path_m=distance if (visible and abs(bearing) < 0.15) else SENSING_HORIZON_M,
        sector_min_range_m=[SENSING_HORIZON_M] * N_SECTORS,
        obstacle_bearing_rad=bearing if visible else None,
        target_visible=visible,
        target_bearing_rad=bearing if visible else None,
        target_range_m=distance if visible else None,
        coverage_fraction=1.0,
    )


def make_packet(
    step_index: int,
    now_s: float,
    state: VehicleState | None,
    depth: DepthSummary | None,
    authorization: AuthorizationView,
    *,
    mission_phase_hint: str | None = None,
    declared_delay_s: float = 0.0,
) -> ObservationPacket:
    """Build a packet whose timing fields are consistent, as ``schemas.py`` now requires."""
    age = None if state is None else now_s - state.sim_time_s
    return ObservationPacket(
        step_index=step_index,
        receive_sim_time_s=now_s,
        state=state,
        state_age_s=age,
        depth=depth,
        rgb=None,
        supervision=SupervisionView(
            sim_time_s=now_s,
            last_heartbeat_sim_time_s=now_s,
            last_heartbeat_received_at_s=now_s,
            heartbeat_age_s=0.0,
            link_state="nominal",
        ),
        authorization=authorization,
        sensor_health=SensorHealth(
            state_sample_available=state is not None,
            depth_available=depth is not None,
            rgb_available=False,
            state_age_s=age,
        ),
        mission_phase_hint=mission_phase_hint,
        declared_observation_delay_s=declared_delay_s,
    )


@dataclass
class FlightLog:
    """What one synthetic flight produced, kept in one object so tests can assert on any part of it."""

    commands: list[ControlCommand] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)
    internal: list[dict[str, Any]] = field(default_factory=list)
    controller: InspectionController | None = None
    vehicle: FakeVehicle | None = None

    @property
    def phase_sequence(self) -> list[str]:
        """Phases in order of first entry, with repeats collapsed."""
        out: list[str] = []
        for phase in self.phases:
            if not out or out[-1] != phase:
                out.append(phase)
        return out

    def fingerprint(self) -> list[tuple[str, str, tuple[float, float, float] | None, float | None]]:
        """A comparable summary of the command sequence, used for the determinism test."""
        return [
            (
                command.kind,
                command.reason,
                None if command.target is None else command.target.as_tuple(),
                command.speed_mps,
            )
            for command in self.commands
        ]


def fly(
    *,
    asset: Vec3 = TRUE_ASSET,
    grant_authorization: bool = True,
    authorization_delay_s: float = 1.0,
    hint: str | None = None,
    hint_from_step: int | None = None,
    stop_in_phase: str | None = None,
    max_steps: int = 200,
) -> FlightLog:
    """Fly the synthetic mission and record every command.

    Authorization is granted ``authorization_delay_s`` after the controller asks for it, which is how the
    supervisory channel behaves in the nominal condition.
    """
    controller = InspectionController(PROTOCOL)
    controller.reset(make_brief())
    vehicle = FakeVehicle()
    log = FlightLog(controller=controller, vehicle=vehicle)
    authorization = AuthorizationView()
    requested_at: float | None = None

    for step_index in range(max_steps):
        now_s = step_index * DT
        if (
            grant_authorization
            and requested_at is not None
            and now_s >= requested_at + authorization_delay_s
            and authorization.status is not AuthorizationStatus.GRANTED
        ):
            decided_at = requested_at + authorization_delay_s
            authorization = AuthorizationView(
                token_id="token-1",
                status=AuthorizationStatus.GRANTED,
                requested_at_s=requested_at,
                granted_at_s=decided_at,
                expires_at_s=decided_at + PROTOCOL.obligations.authorization_validity_s,
                received_at_s=decided_at,
                scope="inspection",
            )
        hint_now = hint if (hint_from_step is not None and step_index >= hint_from_step) else None
        packet = make_packet(
            step_index,
            now_s,
            vehicle.state(now_s),
            perceive(vehicle, asset, now_s),
            authorization,
            mission_phase_hint=hint_now,
        )
        command = controller.step(packet)
        state = controller.internal_state()
        log.commands.append(command)
        log.phases.append(state["phase"])
        log.internal.append(state)
        if command.kind == "request_authorization" and requested_at is None:
            requested_at = now_s
            authorization = AuthorizationView(
                token_id="token-1", status=AuthorizationStatus.PENDING, requested_at_s=now_s
            )
        vehicle.apply(command)
        if stop_in_phase is not None and state["phase"] == stop_in_phase:
            return log
        if state["phase"] == "done":
            break
    return log


# ----------------------------------------------------------------------------------------------
# 1. Determinism
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("height_m", [.5368, 0.0])
@pytest.mark.parametrize("descent_speed_mps", [0.0, .203])
def test_near_ground_hover_or_descent_is_not_reported_as_landed(height_m, descent_speed_mps):
    """Nominal06 stopped above its support because altitude substituted for an actual landed flag."""
    controller = InspectionController(PROTOCOL)
    controller.reset(make_brief())
    controller._set_phase("land", "replay_near_ground_not_landed")
    vehicle = FakeVehicle(x=0, y=0, z=-height_m, landed=False,
                          velocity=(0, 0, descent_speed_mps))
    packet = make_packet(0, 60, vehicle.state(60), None, AuthorizationView())
    command = controller.step(packet)
    assert command.kind == "land" and command.controller_phase == "land"
    assert controller.internal_state()["phase"] == "land"

    # Only the later onboard touchdown report permits the terminal transition.
    vehicle.landed = True
    vehicle.velocity = (0, 0, 0)
    packet = make_packet(1, 60.5, vehicle.state(60.5), None, AuthorizationView())
    command = controller.step(packet)
    assert command.kind == "noop" and command.controller_phase == "done"


def test_the_same_observation_sequence_twice_produces_identical_commands() -> None:
    """Paired arm comparisons are only meaningful if the controller adds no variation of its own."""
    first, second = fly(), fly()
    assert first.fingerprint() == second.fingerprint()
    assert len(first.commands) > 20  # a whole mission, not two trivially short runs
    assert first.phase_sequence == second.phase_sequence
    assert first.controller is not None
    assert first.controller.internal_state()["uses_rng"] is False
    assert first.controller.internal_state()["is_learned_policy"] is False


# ----------------------------------------------------------------------------------------------
# 2. Phase progression
# ----------------------------------------------------------------------------------------------
def test_the_mission_progresses_through_every_phase_to_done() -> None:
    """A controller that cannot finish a clean mission would make every guarded arm look good."""
    log = fly()
    assert log.phase_sequence == [
        "takeoff",
        "transit",
        "search_align",
        "request_authorization",
        "approach",
        "inspect_capture",
        "return",
        "land",
        "done",
    ]
    assert all(phase in PHASES for phase in log.phases)
    state = log.internal[-1]
    assert state["phase"] == "done"
    assert state["captures_issued"] == MISSION.required_inspection_captures
    assert state["authorization_requests"] >= 1
    kinds = {command.kind for command in log.commands}
    assert {"arm", "takeoff", "move_to", "request_authorization", "inspect_capture", "land"} <= kinds
    # The horizon is 120 s; a clean mission must finish well inside it.
    assert log.commands[-1].issued_sim_time_s < MISSION.episode_horizon_s


def test_the_controller_requests_authorization_but_never_refuses_to_fly() -> None:
    """Documented design choice 2: the guards enforce policy, the controller does not.

    With no supervisory answer at all the controller must wait a bounded number of steps, re-ask, and
    then continue into the inspection radius. That is the ordinary behaviour whose consequences the
    three arms compare; a controller that refused to fly would remove the failure mode under study.
    """
    log = fly(grant_authorization=False)
    assert "approach" in log.phase_sequence
    assert log.internal[-1]["captures_issued"] == MISSION.required_inspection_captures
    assert log.internal[-1]["authorization_requests"] > 1  # it re-asked rather than giving up silently
    entered_at = log.phases.index("approach")
    waited_steps = log.internal[entered_at]["authorization_wait_steps"]
    assert waited_steps > PARAMS.auth_wait_steps


# ----------------------------------------------------------------------------------------------
# 3. Perception really drives the commands
# ----------------------------------------------------------------------------------------------
def test_two_different_perceived_target_bearings_produce_different_commanded_targets() -> None:
    """The brief holds only the nominal asset position, so the difference can come from perception alone.

    Both flights are handed the identical ``MissionBrief``. Only the TRUE asset moves, and it moves by
    less than the prior radius so that both detections are accepted. Different commands therefore prove
    the controller resolved the displacement from the depth features.
    """
    nominal = MISSION.asset_nominal_position
    left = fly(asset=Vec3(x=28.6, y=-1.2, z=-6.0))
    right = fly(asset=Vec3(x=28.6, y=1.2, z=-6.0))

    assert left.fingerprint() != right.fingerprint()

    left_estimate = left.internal[-1]["asset_estimate"]
    right_estimate = right.internal[-1]["asset_estimate"]
    assert left.internal[-1]["asset_estimate_source"] == "depth_detection"
    assert right.internal[-1]["asset_estimate_source"] == "depth_detection"
    # Each estimate followed its own true asset, on opposite sides of the nominal position.
    assert left_estimate["y"] < nominal.y < right_estimate["y"]
    assert left_estimate["y"] == pytest.approx(-1.2, abs=0.5)
    assert right_estimate["y"] == pytest.approx(1.2, abs=0.5)

    # The commanded targets themselves differ, not only the internal estimate.
    left_targets = [c.target.as_tuple() for c in left.commands if c.kind == "move_to" and c.target]
    right_targets = [c.target.as_tuple() for c in right.commands if c.kind == "move_to" and c.target]
    assert left_targets and right_targets
    assert left_targets != right_targets
    assert any(abs(a[1] - b[1]) > 0.3 for a, b in zip(left_targets, right_targets, strict=False))


def test_perception_used_is_false_until_a_detection_is_accepted() -> None:
    """``perception_used`` is written into the evidence record, so it must mean what it says."""
    log = fly()
    assert log.internal[0]["perception_used"] is False  # still on the ground, nothing detected yet
    assert any(state["perception_used"] for state in log.internal)
    assert log.internal[-1]["perception_detections"] > 0


# ----------------------------------------------------------------------------------------------
# 4. Bounded commands
# ----------------------------------------------------------------------------------------------
def test_no_command_ever_exceeds_the_mission_speed_or_duration_limits() -> None:
    """An unbounded command would let the controller, not the protocol, decide the risk."""
    log = fly()
    assert log.commands
    for command in log.commands:
        if command.speed_mps is not None:
            assert 0.0 <= command.speed_mps <= MISSION.cruise_speed_mps
        assert command.duration_s is not None
        assert 0.0 <= command.duration_s <= PROTOCOL.simulation.max_command_duration_s
    assert max(c.speed_mps or 0.0 for c in log.commands) == pytest.approx(MISSION.cruise_speed_mps)


# ----------------------------------------------------------------------------------------------
# 5. Missing observations: the documented blind behaviour
# ----------------------------------------------------------------------------------------------
def test_missing_observations_continue_on_the_last_target_at_a_decaying_speed_then_hold() -> None:
    """Documented design choice 3, asserted exactly as the module docstring states it.

    "With no state this step it repeats the last commanded target at a decayed speed, and after
    ``max_blind_steps`` it holds." Freezing on the first dropout would delete the failure mode the
    study measures, so the decay schedule and the hold threshold are both pinned here.
    """
    log = fly(stop_in_phase="transit")
    controller = log.controller
    assert controller is not None
    last_move = log.commands[-1]
    assert last_move.kind == "move_to" and last_move.target is not None
    start_index = len(log.commands)
    base_speed = last_move.speed_mps or 0.0
    floor = PARAMS.min_speed_fraction * MISSION.approach_speed_mps

    blind: list[ControlCommand] = []
    for offset in range(12):
        step_index = start_index + offset
        now_s = step_index * DT
        packet = ObservationPacket(
            step_index=step_index,
            receive_sim_time_s=now_s,
            state=None,
            state_age_s=None,
            depth=None,
            rgb=None,
            supervision=SupervisionView(
                sim_time_s=now_s,
                last_heartbeat_sim_time_s=now_s,
                last_heartbeat_received_at_s=now_s,
                heartbeat_age_s=0.0,
                link_state="nominal",
            ),
            authorization=AuthorizationView(),
            sensor_health=SensorHealth(
                state_sample_available=False,
                depth_available=False,
                rgb_available=False,
                state_age_s=None,
                dropouts_in_window=offset + 1,
            ),
        )
        blind.append(controller.step(packet))

    max_blind = PARAMS.max_blind_steps
    for offset, command in enumerate(blind[:max_blind], start=1):
        assert command.kind == "move_to"
        assert command.target is not None
        assert command.target.as_tuple() == last_move.target.as_tuple()  # the LAST target, unchanged
        expected = max(base_speed * PARAMS.blind_speed_decay**offset, floor)
        assert command.speed_mps == pytest.approx(expected, abs=1e-9)
        assert f"observation_missing_{offset}_steps" in command.reason
    assert blind[max_blind - 1].speed_mps == pytest.approx(floor)  # the decay never reaches zero
    for offset, command in enumerate(blind[max_blind:], start=max_blind + 1):
        assert command.kind == "hold"  # after max_blind_steps it stops moving blind
        assert command.reason == f"observation_missing_{offset}_steps_holding"
    assert controller.internal_state()["blind_steps"] == len(blind)


def test_a_state_that_never_arrives_at_all_yields_a_hold_not_an_exception() -> None:
    """Episode start under a severe delay: nothing is old enough to have been received yet."""
    controller = InspectionController(PROTOCOL)
    controller.reset(make_brief())
    packet = make_packet(0, 0.0, None, None, AuthorizationView())
    command = controller.step(packet)
    assert command.kind == "hold"
    assert command.reason == "no_state_received_yet"
    assert controller.internal_state()["last_valid_observation_sim_time_s"] is None


# ----------------------------------------------------------------------------------------------
# 6. Stale observations: the documented no-compensation behaviour
# ----------------------------------------------------------------------------------------------
def test_a_stale_state_is_acted_on_as_if_it_were_current_because_delay_is_not_compensated() -> None:
    """Documented design choice 1, and the reason delay can hurt at all.

    "The received state is used as if it were current." Two packets carry the same measured position and
    velocity; one is fresh and one is 2.5 s old. If the controller compensated for the age it would aim
    further along the velocity vector, and the commands would differ. They must not.
    """

    Command = tuple[str, tuple[float, float, float] | None]

    def commands_for(state_age_s: float, velocity: Vec3) -> list[Command]:
        controller = InspectionController(PROTOCOL)
        controller.reset(make_brief())
        vehicle = FakeVehicle(x=10.0, y=0.0, z=-6.0, landed=False)
        out: list[Command] = []
        for step_index in range(4):
            now_s = step_index * DT + state_age_s
            state = VehicleState(
                sim_time_s=now_s - state_age_s,
                position=Vec3(x=vehicle.x, y=vehicle.y, z=vehicle.z),
                velocity=velocity,
                yaw_rad=vehicle.yaw_rad,
            )
            packet = make_packet(
                step_index,
                now_s,
                state,
                perceive(vehicle, TRUE_ASSET, now_s - state_age_s),
                AuthorizationView(),
                declared_delay_s=state_age_s,
            )
            command = controller.step(packet)
            out.append((command.kind, None if command.target is None else command.target.as_tuple()))
        return out

    moving = Vec3(x=MISSION.cruise_speed_mps, y=0.0, z=0.0)
    fresh = commands_for(0.0, moving)
    stale = commands_for(2.5, moving)
    assert stale == fresh  # identical commands: the 2.5 s age changed nothing

    # And the age is recorded even though it is not used, so the audit lane can see it.
    controller = InspectionController(PROTOCOL)
    controller.reset(make_brief())
    vehicle = FakeVehicle(x=10.0, y=0.0, z=-6.0, landed=False)
    packet = make_packet(0, 2.5, vehicle.state(0.0), perceive(vehicle, TRUE_ASSET, 0.0), AuthorizationView())
    controller.step(packet)
    assert controller.internal_state()["last_observed_state_age_s"] == pytest.approx(2.5)


# ----------------------------------------------------------------------------------------------
# 7. Guard hints
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("hint", ["inspection_suspended_by_guard", "returning_by_guard"])
def test_a_guard_hint_ends_the_inspection_attempt_and_sends_the_vehicle_home(hint: str) -> None:
    """The intervention burden reported by the analysis must be a real consequence of the guard.

    A controller that quietly resumed the inspection after an intervention would make the guarded arms
    look cheap. The hint must move the controller into ``return`` and it must not come back.
    """
    log = fly(stop_in_phase="approach")
    controller, vehicle = log.controller, log.vehicle
    assert controller is not None and vehicle is not None
    assert controller.internal_state()["phase"] == "approach"
    captures_before = controller.internal_state()["captures_issued"]

    start_index = len(log.commands)
    now_s = start_index * DT
    packet = make_packet(
        start_index,
        now_s,
        vehicle.state(now_s),
        perceive(vehicle, TRUE_ASSET, now_s),
        AuthorizationView(
            token_id="token-1",
            status=AuthorizationStatus.GRANTED,
            granted_at_s=0.0,
            expires_at_s=now_s + 45.0,
            received_at_s=0.0,
        ),
        mission_phase_hint=hint,
    )
    controller.step(packet)
    state = controller.internal_state()
    assert state["phase"] == "return"
    assert state["guard_override"] == hint
    assert state["reason"].startswith("returning_home")

    # The hint is not repeated afterwards; the controller must still not go back to the asset.
    phases_after: set[str] = set()
    for offset in range(1, 40):
        step_index = start_index + offset
        now_s = step_index * DT
        command = controller.step(
            make_packet(
                step_index,
                now_s,
                vehicle.state(now_s),
                perceive(vehicle, TRUE_ASSET, now_s),
                AuthorizationView(),
            )
        )
        vehicle.apply(command)
        phases_after.add(controller.internal_state()["phase"])
        if controller.internal_state()["phase"] == "done":
            break
    assert phases_after <= {"return", "land", "done"}
    assert controller.internal_state()["captures_issued"] == captures_before


# ----------------------------------------------------------------------------------------------
# 8. The evidence record
# ----------------------------------------------------------------------------------------------
def test_internal_state_exposes_the_fields_the_evidence_record_depends_on() -> None:
    """These four fields bound every claim the study makes about what the controller was doing."""
    log = fly(stop_in_phase="approach")
    controller = log.controller
    assert controller is not None
    state = controller.internal_state()

    assert state["phase"] == "approach"
    assert state["controller_class"] == "hand_written_goal_directed_state_machine"
    assert state["controller_id"] == "fixed_inspection_v1"
    assert isinstance(state["perception_used"], bool)
    assert state["perception_used"] is True  # the approach phase servos on the depth detection
    assert state["last_valid_observation_sim_time_s"] == pytest.approx((len(log.commands) - 1) * DT)
    assert state["policy_version"] == PROTOCOL.obligations.policy_version

    # The record is written to JSON evidence, so every value must survive serialisation unchanged.
    assert json.loads(json.dumps(state)) == state
