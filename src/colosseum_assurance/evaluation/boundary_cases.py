"""Hand-written, manually checkable boundary cases for the independent evaluator.

WHY THIS MODULE EXISTS
----------------------
research-acceptance.md section 3 requires the evaluator to be "validated against manually checkable
examples, including exact boundaries, stale or missing information, expired permission, late detections,
premature termination, and deadlines". A test that only re-runs the evaluator's own arithmetic proves
nothing. Every case below is therefore small enough for a person to check by hand: round numbers, one
obligation under test at a time, and a one-sentence comment saying why a human should expect the stated
verdict.

The six worked timelines ``timeline_t1_..`` to ``timeline_t6_..`` are the SHARED REFERENCE between the
monitor lane and the evaluator lane. They use exactly the numbers written in ``docs/timing-semantics.md``
section 6. The monitors are implemented separately and must measure the same obligations from the same
timelines; if the two lanes disagree on one of these six, the timing document decides and the code is a
bug.

HONESTY BOUNDARY
----------------
These ledgers are FIXTURES. ``SimulatorIdentity.provenance`` is ``fixture_fake`` and ``run_class`` is
``fixture``, so nothing here can be mistaken for, or written into, experimental evidence. No Colosseum
server produced any number in this file. Fixtures test software; they are never study results.

WHAT A CASE IS
--------------
Each case is a zero-argument function returning
``(PrivilegedLedger, ScenarioManifest, EpisodeRecord | None, ExpectedOutcome)``. The record is ``None``
when the case only exercises ``assess_obligations``, which by design never sees an episode record. A
record is supplied when the case is about monitor-relative outcomes (acceptance, false assurance, missed
detection) or about episode completeness.

All numbers quoted in the comments come from the frozen default :class:`ProtocolConfig`:

* geofence ``x`` in [-12, 42] m, tolerance 0.50 m;
* inspection radius 8.00 m around the true asset, authorization validity 45.0 s;
* heartbeat period 2.0 s, loss threshold 6.0 s, reaction deadline 5.0 s, grace 0.0 s;
* safe-response speed 0.60 m/s, truth sampling 0.1 s, maximum permitted truth gap 0.25 s;
* home at the origin with a 3.0 m return tolerance.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.scenario.manifest import ObstacleSpec, ScenarioManifest, ScheduleSet
from colosseum_assurance.schemas import (
    AuthorizationView,
    ControlCommand,
    EpisodeRecord,
    MonitorReport,
    ObservationPacket,
    PrivilegedLedger,
    SensorHealth,
    SimulatorIdentity,
    StepRecord,
    SupervisionView,
    TerminationRecord,
    TruthEvent,
    TruthSample,
    Vec3,
    VehicleState,
    Verdict,
)

# --------------------------------------------------------------------------------------
# Scene and timing constants shared by every case
# --------------------------------------------------------------------------------------
TRUTH_INTERVAL_S: Final = 0.1
CONTROL_DT_S: Final = 0.5

#: True asset position used by every case. It equals ``MissionSpec.asset_nominal_position`` so a reader
#: can check distances mentally: the inspection radius is 8.00 m around this point.
ASSET_X: Final = 28.0
ASSET_Y: Final = 0.0
ASSET_Z: Final = -6.0

#: Transit lane used whenever a case is NOT about the inspection obligation. At ``y = 12.0`` the vehicle
#: is at least 12.0 m from the asset for any ``x``, which is comfortably outside the 8.00 m radius, so
#: the inspection obligation stays ``not_applicable`` and cannot mask the obligation under test.
PATROL_LANE_Y: Final = 12.0

#: Comparison slack for the hand-written times in this module. It is far below one truth interval, so it
#: never moves a boundary; it only absorbs binary representation noise in values such as ``t - 6.0``.
EPS: Final = 1e-9

_PROTOCOL: Final = ProtocolConfig()
_PROTOCOL_HASH: Final = _PROTOCOL.content_hash()

_SCENARIO_ID: Final = "fixture-boundary-cases-r000"
_WALL_CLOCK: Final = "1970-01-01T00:00:00+00:00"


def boundary_protocol() -> ProtocolConfig:
    """Return the frozen default protocol every case is written against.

    Cases quote its thresholds in their comments, so the test must evaluate against this exact object.
    """
    return _PROTOCOL


def v(x: float, y: float, z: float) -> Vec3:
    """Build a fresh :class:`Vec3`. Fresh instances stop two samples sharing one mutable point."""
    return Vec3(x=float(x), y=float(y), z=float(z))


# --------------------------------------------------------------------------------------
# Expectation record
# --------------------------------------------------------------------------------------
class _Unchecked:
    """Sentinel meaning "this case makes no claim about that field"."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return "UNCHECKED"


UNCHECKED: Final = _Unchecked()


@dataclass(frozen=True, slots=True)
class ExpectedOutcome:
    """What a human should expect from one boundary case, and why.

    ``why`` is printed by the test on failure. It is the sentence a reviewer checks against the numbers
    in the case, so a disagreement between this module and the evaluator is a readable statement rather
    than a bare assertion error.

    Fields left at :data:`UNCHECKED` are not asserted. A case claims only what it was written to prove,
    so an unrelated change elsewhere does not produce a misleading failure here.
    """

    why: str
    obligation_verdicts: dict[str, Verdict]
    episode_verdict: Verdict
    #: obligation_id -> exact expected ``first_violation_sim_time_s``. Only for cases that fix a time.
    first_violation_times_s: dict[str, float] = field(default_factory=dict)
    #: obligation_id -> {measurement key: expected value}. Checked with a 1e-6 absolute tolerance.
    obligation_measurements: dict[str, dict[str, float]] = field(default_factory=dict)
    #: obligation_id -> substring that must appear in ``unknown_reason`` (lower-cased comparison).
    unknown_reason_contains: dict[str, str] = field(default_factory=dict)
    physical_verdict: Verdict | _Unchecked = UNCHECKED
    procedural_verdict: Verdict | _Unchecked = UNCHECKED
    episode_first_violation_sim_time_s: float | None | _Unchecked = UNCHECKED
    completeness: Literal["complete", "incomplete"] | _Unchecked = UNCHECKED
    accepted_by_monitor: bool | None | _Unchecked = UNCHECKED
    false_assurance: bool | None | _Unchecked = UNCHECKED
    missed_detection: bool | None | _Unchecked = UNCHECKED
    detection_delay_s: float | None | _Unchecked = UNCHECKED
    #: The mission oracle's two distinct claims: the job was done, and it was done without a violation.
    mission_completed: bool | _Unchecked = UNCHECKED
    mission_completed_safely: bool | _Unchecked = UNCHECKED
    completion_time_s: float | None | _Unchecked = UNCHECKED
    truth_coverage_fraction: float | _Unchecked = UNCHECKED


CaseResult = tuple[PrivilegedLedger, ScenarioManifest, EpisodeRecord | None, ExpectedOutcome]


# --------------------------------------------------------------------------------------
# Small builders: scene, truth track, ledger, episode record
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TruthState:
    """The privileged world state a case asserts at one instant.

    Position and velocity are supplied together and are kept physically consistent inside each case
    (a stated speed really does move the stated position), because a fixture that contradicts itself
    cannot support a claim about what the evaluator measured.
    """

    position: Vec3
    velocity: Vec3
    landed: bool = False
    collision_active: bool = False
    collision_count: int = 0
    collision_object: str | None = None


def _fixture_identity() -> SimulatorIdentity:
    """Provenance of every fixture in this module: not a simulator, and labelled as such."""
    return SimulatorIdentity(
        provenance="fixture_fake",
        endpoint_label="hand_written_boundary_case",
        simulator_name="none",
        notes="Hand-written evaluator boundary case. No simulator produced these numbers.",
    )


def boundary_manifest(scenario_id: str = _SCENARIO_ID) -> ScenarioManifest:
    """A minimal scene whose only evaluator-relevant fact is the true asset position.

    ``ScheduleSet`` is filled with a single step because no case replays a schedule: the cases write the
    resulting privileged truth directly, which is what the evaluator reads.
    """
    schedules = ScheduleSet(
        steps=1,
        dt_s=CONTROL_DT_S,
        observation_delay_s=[0.0],
        state_dropout=[False],
        depth_dropout=[False],
        heartbeat_times_s=[],
        supervision_outages_s=[],
        authorization_response_delay_s=[2.0],
        authorization_decision=["granted"],
        authorization_validity_s=_PROTOCOL.obligations.authorization_validity_s,
        visibility="clear",
        schedule_seed=0,
    )
    return ScenarioManifest(
        scenario_id=scenario_id,
        protocol_hash=_PROTOCOL_HASH,
        run_class="fixture",
        cell_id="obs_nominal__sup_nominal",
        observation_delay_level="obs_nominal",
        supervision_delay_level="sup_nominal",
        layout_variant="open",
        visibility="clear",
        realization_index=0,
        seed=0,
        start_position=v(0.0, 0.0, 0.0),
        start_yaw_rad=0.0,
        asset_position=v(ASSET_X, ASSET_Y, ASSET_Z),
        inspection_viewpoint=v(ASSET_X - 4.0, ASSET_Y, ASSET_Z),
        obstacles=[
            ObstacleSpec(
                name="inspection_tower",
                kind="inspection_asset",
                center=v(ASSET_X, ASSET_Y, -7.0),
                extent=v(1.2, 1.2, 7.0),
            ),
            ObstacleSpec(
                name="occluding_wall",
                kind="wall",
                center=v(19.0, 1.0, -4.0),
                extent=v(0.6, 5.0, 4.0),
            ),
            ObstacleSpec(
                name="ground_plane",
                kind="ground_plane",
                center=v(0.0, 0.0, 0.5),
                extent=v(200.0, 200.0, 0.5),
            ),
        ],
        schedules=schedules,
        notes="Hand-written boundary-case scene; asset at the nominal position so distances stay round.",
    )


def tick_times(end_s: float, interval: float = TRUTH_INTERVAL_S) -> list[float]:
    """Sample times ``0, interval, ..., end_s`` built from integers so 17.5 really is 17.5."""
    count = int(round(end_s / interval))
    return [round(i * interval, 9) for i in range(count + 1)]


def truth_track(
    end_s: float,
    state_at: Callable[[float], TruthState],
    *,
    interval: float = TRUTH_INTERVAL_S,
    skip: Callable[[float], bool] | None = None,
) -> list[TruthSample]:
    """Densify a hand-written state function into truth samples at the frozen truth interval.

    The hand-written part is ``state_at``: a few lines of round numbers. Densification is mechanical and
    exists only because the frozen oracle samples every 0.1 s and refuses gaps above 0.25 s. ``skip``
    deliberately drops samples to build an under-sampled ledger.
    """
    samples: list[TruthSample] = []
    for t in tick_times(end_s, interval):
        if skip is not None and skip(t):
            continue
        state = state_at(t)
        samples.append(
            TruthSample(
                sim_time_s=t,
                position=state.position,
                velocity=state.velocity,
                yaw_rad=0.0,
                collision_active=state.collision_active,
                collision_count=state.collision_count,
                collision_object=state.collision_object,
                landed=state.landed,
                source="fixture_fake_ground_truth",
            )
        )
    return samples


def heartbeat_times(
    end_s: float,
    *,
    period: float = 2.0,
    outage: tuple[float, float] | None = None,
) -> list[float]:
    """Scheduled heartbeat times that were actually DELIVERED.

    Outage windows are half-open ``[start, end)`` (docs/timing-semantics.md section 3): a heartbeat due
    exactly at ``start`` is lost, one due exactly at ``end`` is delivered.
    """
    times: list[float] = []
    for index in range(int(end_s / period) + 1):
        t = round(index * period, 9)
        if t > end_s + EPS:
            break
        if outage is not None and outage[0] - EPS <= t < outage[1] - EPS:
            continue
        times.append(t)
    return times


def heartbeat_events(times: Sequence[float]) -> list[TruthEvent]:
    """Privileged heartbeat deliveries. Production and receipt coincide while the link is up."""
    return [
        TruthEvent(sim_time_s=t, kind="supervision_heartbeat", payload={"received_at_s": t})
        for t in times
    ]


def privileged_ledger(
    *,
    episode_id: str,
    samples: Sequence[TruthSample],
    events: Sequence[TruthEvent],
    termination: TerminationRecord,
    declared_end_s: float,
    scenario_id: str = _SCENARIO_ID,
    arm_id: str = "A1_policy_only",
    interval: float = TRUTH_INTERVAL_S,
    expected_sample_count: int | None = None,
    notes: str = "hand-written boundary case",
) -> PrivilegedLedger:
    """Assemble a ledger whose coverage metadata is MEASURED, never asserted.

    ``expected_sample_count`` defaults to the samples the episode should have produced over its declared
    duration, so dropping samples lowers the reported coverage instead of hiding the loss.
    """
    expected = (
        expected_sample_count
        if expected_sample_count is not None
        else int(round(declared_end_s / interval)) + 1
    )
    coverage = 0.0 if not samples else round(min(1.0, len(samples) / expected), 6)
    return PrivilegedLedger(
        episode_id=episode_id,
        scenario_id=scenario_id,
        arm_id=arm_id,
        run_class="fixture",
        protocol_hash=_PROTOCOL_HASH,
        simulator_identity=_fixture_identity(),
        sample_interval_s=interval,
        samples=list(samples),
        events=list(events),
        termination=termination,
        truth_coverage_fraction=coverage,
        expected_sample_count=expected,
        notes=notes,
    )


def affirmative_pass_report(step_index: int, sim_time_s: float, monitor_id: str) -> MonitorReport:
    """A monitor report claiming everything is fine: affirmative PASS on every frozen obligation."""
    return MonitorReport(
        step_index=step_index,
        sim_time_s=sim_time_s,
        monitor_id=monitor_id,
        verdict=Verdict.PASS,
        obligation_verdicts=dict.fromkeys(_PROTOCOL.obligations.obligation_ids, Verdict.PASS),
        intervention="none",
        rationale="fixture monitor: received evidence looked compliant",
        affirmative=True,
    )


def episode_record(
    *,
    episode_id: str,
    end_s: float,
    state_at: Callable[[float], TruthState],
    termination: TerminationRecord,
    scenario_id: str = _SCENARIO_ID,
    arm_id: str = "A1_policy_only",
    monitor_id: str | None = "policy_only_v1",
    observation_delay_s: float = 0.0,
    heartbeats_s: Sequence[float] = (),
    report_at: Callable[[int, float], MonitorReport | None] | None = None,
    dt_s: float = CONTROL_DT_S,
) -> EpisodeRecord:
    """Build the EXPOSED record for a case: delayed observations plus recorded monitor verdicts.

    The record carries only what the vehicle could observe. ``observation_delay_s`` shifts the ACQUISITION
    time of the state behind the RECEIPT time, which is the mechanism timeline T6 needs: a stale but
    in-bounds position can hide a true geofence breach from the monitor without hiding it from the
    evaluator.
    """
    steps: list[StepRecord] = []
    for index, t in enumerate(tick_times(end_s, dt_s)):
        acquired = round(max(0.0, t - observation_delay_s), 9)
        state = state_at(acquired)
        delivered = [h for h in heartbeats_s if h <= t + EPS]
        last_heartbeat = delivered[-1] if delivered else None
        observation = ObservationPacket(
            step_index=index,
            receive_sim_time_s=t,
            state=VehicleState(
                sim_time_s=acquired,
                position=state.position,
                velocity=state.velocity,
                yaw_rad=0.0,
                landed=state.landed,
                source="fixture_fake_state",
            ),
            state_age_s=round(t - acquired, 9),
            depth=None,
            rgb=None,
            supervision=SupervisionView(
                sim_time_s=t,
                last_heartbeat_sim_time_s=last_heartbeat,
                last_heartbeat_received_at_s=last_heartbeat,
                heartbeat_age_s=None if last_heartbeat is None else round(t - last_heartbeat, 9),
                link_state="nominal" if last_heartbeat is not None else "unknown",
            ),
            authorization=AuthorizationView(),
            sensor_health=SensorHealth(
                state_sample_available=True,
                depth_available=False,
                rgb_available=False,
                state_age_s=round(t - acquired, 9),
                notes="fixture: no camera payload is needed for evaluator boundary cases",
            ),
        )
        command = ControlCommand(
            step_index=index,
            issued_sim_time_s=t,
            kind="move_to",
            target=state.position,
            reason="fixture command; the case fixes the trajectory in privileged truth",
        )
        if report_at is not None:
            report = report_at(index, t)
        elif monitor_id is not None:
            report = affirmative_pass_report(index, t, monitor_id)
        else:
            report = None
        steps.append(
            StepRecord(
                step_index=index,
                sim_time_s=t,
                wall_clock_s=t,
                observation=observation,
                command=command,
                monitor_report=report,
            )
        )
    return EpisodeRecord(
        episode_id=episode_id,
        scenario_id=scenario_id,
        arm_id=arm_id,
        run_class="fixture",
        protocol_hash=_PROTOCOL_HASH,
        policy_version=_PROTOCOL.obligations.policy_version,
        simulator_identity=_fixture_identity(),
        started_wall_clock=_WALL_CLOCK,
        dt_s=dt_s,
        steps=steps,
        termination=termination,
        monitor_id=monitor_id,
        notes="hand-written boundary case; wall-clock fields are placeholders and never enter a verdict",
    )


def termination_record(
    reason: str,
    sim_time_s: float,
    *,
    dt_s: float = CONTROL_DT_S,
    completed_mission: bool = False,
) -> TerminationRecord:
    """Termination record at ``sim_time_s``. Error reasons are forced non-terminal by the schema.

    Public because ``tests/unit/test_evaluator_semantics.py`` composes its own episodes from the same
    builders, so a semantics test and a boundary case cannot drift apart in how an episode is ended.
    """
    return TerminationRecord(
        reason=reason,  # type: ignore[arg-type]
        detail="hand-written boundary case",
        step_index=int(round(sim_time_s / dt_s)),
        sim_time_s=sim_time_s,
        completed_mission=completed_mission,
    )


def _hold(x: float, y: float, z: float) -> TruthState:
    """Station keeping at one point: zero velocity, so the safe-response speed test is unambiguous."""
    return TruthState(position=v(x, y, z), velocity=v(0.0, 0.0, 0.0))


def _constant_speed_track(x0: float, speed_mps: float) -> Callable[[float], TruthState]:
    """A straight transit along +x in the patrol lane at one constant speed.

    Used by cases whose subject is the supervisory link, not the trajectory. The speed is above the
    0.60 m/s safe-response limit in every such case, so a PASS can only come from restoration, never
    from the vehicle accidentally being slow.
    """

    def state_at(t: float) -> TruthState:
        return TruthState(
            position=v(x0 + speed_mps * t, PATROL_LANE_Y, -6.0),
            velocity=v(speed_mps, 0.0, 0.0),
        )

    return state_at


# --------------------------------------------------------------------------------------
# Geofence boundary: exactly at the tolerance, and one centimetre past it
# --------------------------------------------------------------------------------------
def geofence_exactly_at_tolerance() -> CaseResult:
    """Hold station exactly 0.50 m outside the geofence face, which is exactly the tolerance.

    WHY A HUMAN EXPECTS PASS: the fence face is ``x = 42.00`` and the tolerance is 0.50 m, so the
    allowed limit is ``x = 42.50``. The rule is "outside when exceedance is GREATER than the tolerance",
    so a vehicle sitting on 42.50 is still compliant. Every other obligation is deliberately inert.
    """
    end_s = 10.0
    state_at = lambda t: _hold(42.5, PATROL_LANE_Y, -6.0)  # noqa: E731 - one-line hand-written track
    beats = heartbeat_times(end_s)
    ledger = privileged_ledger(
        episode_id="bc-geofence-at-tolerance",
        samples=truth_track(end_s, state_at),
        events=heartbeat_events(beats),
        termination=termination_record("horizon_reached", end_s),
        declared_end_s=end_s,
    )
    expected = ExpectedOutcome(
        why="exceedance 0.500 m equals the 0.500 m tolerance, and the rule needs a strictly larger value",
        obligation_verdicts={
            "geofence": Verdict.PASS,
            "collision": Verdict.PASS,
            "inspection_authorization": Verdict.NOT_APPLICABLE,
            "loss_of_supervision_response": Verdict.NOT_APPLICABLE,
        },
        obligation_measurements={"geofence": {"max_exceedance_m": 0.5, "samples_outside": 0.0}},
        episode_verdict=Verdict.PASS,
        physical_verdict=Verdict.PASS,
        procedural_verdict=Verdict.NOT_APPLICABLE,
        episode_first_violation_sim_time_s=None,
    )
    return ledger, boundary_manifest(), None, expected


def geofence_one_centimetre_beyond_tolerance() -> CaseResult:
    """Sit on the tolerance, then drift 1 cm further out at 0.1 m/s during the step ending at t = 6.1 s.

    WHY A HUMAN EXPECTS VIOLATION AT 6.1 s: at t = 6.0 s the exceedance is exactly 0.50 m and still
    compliant; one truth sample later the exceedance is 0.51 m, which is greater than the tolerance. The
    first OUTSIDE truth sample is the timestamp of the violation, so the answer is 6.1 s, not 6.0 s.
    """
    end_s = 10.0

    def state_at(t: float) -> TruthState:
        outside = t >= 6.1 - EPS
        # 0.1 m/s for exactly one 0.1 s truth interval moves the vehicle exactly 0.01 m.
        drifting = abs(t - 6.1) < EPS
        return TruthState(
            position=v(42.51 if outside else 42.5, PATROL_LANE_Y, -6.0),
            velocity=v(0.1 if drifting else 0.0, 0.0, 0.0),
        )

    beats = heartbeat_times(end_s)
    ledger = privileged_ledger(
        episode_id="bc-geofence-one-cm-beyond",
        samples=truth_track(end_s, state_at),
        events=heartbeat_events(beats),
        termination=termination_record("horizon_reached", end_s),
        declared_end_s=end_s,
    )
    expected = ExpectedOutcome(
        why="exceedance 0.510 m exceeds the 0.500 m tolerance from t = 6.1 s onward",
        obligation_verdicts={
            "geofence": Verdict.VIOLATION,
            "collision": Verdict.PASS,
            "inspection_authorization": Verdict.NOT_APPLICABLE,
            "loss_of_supervision_response": Verdict.NOT_APPLICABLE,
        },
        first_violation_times_s={"geofence": 6.1},
        obligation_measurements={"geofence": {"first_outside_exceedance_m": 0.51}},
        episode_verdict=Verdict.VIOLATION,
        physical_verdict=Verdict.VIOLATION,
        procedural_verdict=Verdict.NOT_APPLICABLE,
        episode_first_violation_sim_time_s=6.1,
    )
    return ledger, boundary_manifest(), None, expected


# --------------------------------------------------------------------------------------
# Collision: a real mid-flight contact, and the exempt ground contact of a landing
# --------------------------------------------------------------------------------------
def collision_mid_flight() -> CaseResult:
    """Cruise at 3.0 m/s and strike the occluding wall at t = 5.0 s.

    WHY A HUMAN EXPECTS VIOLATION AT 5.0 s: the struck object is a scene wall, not the ground plane, so
    no landing exemption applies. Contact is recorded in the truth sample at 5.0 s, which is therefore
    the violation time.
    """
    end_s = 8.0

    def state_at(t: float) -> TruthState:
        hit = abs(t - 5.0) < EPS
        return TruthState(
            position=v(3.0 * t, PATROL_LANE_Y, -6.0),
            velocity=v(3.0, 0.0, 0.0),
            collision_active=hit,
            collision_count=1 if t >= 5.0 - EPS else 0,
            collision_object="occluding_wall" if hit else None,
        )

    beats = heartbeat_times(end_s)
    ledger = privileged_ledger(
        episode_id="bc-collision-mid-flight",
        samples=truth_track(end_s, state_at),
        events=heartbeat_events(beats),
        termination=termination_record("guard_abort", end_s),
        declared_end_s=end_s,
    )
    expected = ExpectedOutcome(
        why="a 3.0 m/s contact with a wall 15 m from home cannot be a landing, so no exemption applies",
        obligation_verdicts={
            "geofence": Verdict.PASS,
            "collision": Verdict.VIOLATION,
            "inspection_authorization": Verdict.NOT_APPLICABLE,
            "loss_of_supervision_response": Verdict.NOT_APPLICABLE,
        },
        first_violation_times_s={"collision": 5.0},
        obligation_measurements={"collision": {"collision_occurrences": 1.0, "exempt_occurrences": 0.0}},
        episode_verdict=Verdict.VIOLATION,
        physical_verdict=Verdict.VIOLATION,
        procedural_verdict=Verdict.NOT_APPLICABLE,
        episode_first_violation_sim_time_s=5.0,
    )
    return ledger, boundary_manifest(), None, expected


def ground_contact_while_landed_near_home() -> CaseResult:
    """Descend at 0.5 m/s and touch down 1.0 m from home at t = 4.0 s.

    WHY A HUMAN EXPECTS PASS: touching the ground plane at 0.0 m/s, reported as landed, 1.0 m from home
    and inside the 3.0 m return tolerance, is the expected end of a flight. All four exemption conditions
    hold, so the contact is recorded but not scored as a collision.
    """
    end_s = 6.0

    def state_at(t: float) -> TruthState:
        touchdown = abs(t - 4.0) < EPS
        if t < 4.0 - EPS:
            return TruthState(
                position=v(1.0, 0.0, -2.0 + 0.5 * t),
                velocity=v(0.0, 0.0, 0.5),
            )
        return TruthState(
            position=v(1.0, 0.0, 0.0),
            velocity=v(0.0, 0.0, 0.0),
            landed=True,
            collision_active=touchdown,
            collision_count=1,
            collision_object="Ground" if touchdown else None,
        )

    beats = heartbeat_times(end_s)
    ledger = privileged_ledger(
        episode_id="bc-ground-contact-landed",
        samples=truth_track(end_s, state_at),
        events=heartbeat_events(beats),
        termination=termination_record("landed_early", end_s),
        declared_end_s=end_s,
    )
    expected = ExpectedOutcome(
        why="ground contact at 0.0 m/s, landed, 1.0 m from home is an exempt landing, not a collision",
        obligation_verdicts={
            "geofence": Verdict.PASS,
            "collision": Verdict.PASS,
            "inspection_authorization": Verdict.NOT_APPLICABLE,
            "loss_of_supervision_response": Verdict.NOT_APPLICABLE,
        },
        obligation_measurements={"collision": {"collision_occurrences": 1.0, "exempt_occurrences": 1.0}},
        episode_verdict=Verdict.PASS,
        physical_verdict=Verdict.PASS,
        procedural_verdict=Verdict.NOT_APPLICABLE,
        episode_first_violation_sim_time_s=None,
    )
    return ledger, boundary_manifest(), None, expected


# --------------------------------------------------------------------------------------
# Inspection authorization: inside the radius with a valid token, an expired token, and no token
# --------------------------------------------------------------------------------------
def _grant(sim_time_s: float, token_id: str = "auth-1") -> TruthEvent:
    """A privileged authorization grant whose recorded scope names the inspection task."""
    return TruthEvent(
        sim_time_s=sim_time_s,
        kind="authorization_granted",
        detail="fixture grant",
        payload={"token_id": token_id, "scope": "inspect_tower"},
    )


def inspection_one_centimetre_inside_radius_with_valid_token() -> CaseResult:
    """Creep to 7.99 m from the asset, 0.01 m inside the 8.00 m radius, holding a valid grant.

    WHY A HUMAN EXPECTS PASS: the vehicle performs the inspection step (distance at or below the radius)
    from t = 6.9 s, and the grant issued at t = 3.0 s is valid on [3.0, 48.0) s, which covers every
    inside-radius sample up to the t = 12.0 s end of the episode.
    """
    end_s = 12.0

    def state_at(t: float) -> TruthState:
        if t <= 2.0 + EPS:  # 8.49 m from the asset: outside the radius
            return _hold(19.51, 0.0, ASSET_Z)
        if t <= 7.0 + EPS:  # 0.1 m/s creep; crosses the 8.00 m radius at t = 6.9 s
            return TruthState(position=v(19.51 + 0.1 * (t - 2.0), 0.0, ASSET_Z), velocity=v(0.1, 0.0, 0.0))
        return _hold(20.01, 0.0, ASSET_Z)  # 7.99 m from the asset: 0.01 m inside the radius

    beats = heartbeat_times(end_s)
    events = [
        TruthEvent(sim_time_s=1.0, kind="authorization_requested", payload={"request_ordinal": 0}),
        _grant(3.0),
        *heartbeat_events(beats),
    ]
    ledger = privileged_ledger(
        episode_id="bc-inspection-inside-with-token",
        samples=truth_track(end_s, state_at),
        events=events,
        termination=termination_record("horizon_reached", end_s),
        declared_end_s=end_s,
    )
    expected = ExpectedOutcome(
        why="every inside-radius sample from 6.9 s lies inside the grant window [3.0, 48.0) s",
        obligation_verdicts={
            "geofence": Verdict.PASS,
            "collision": Verdict.PASS,
            "inspection_authorization": Verdict.PASS,
            "loss_of_supervision_response": Verdict.NOT_APPLICABLE,
        },
        obligation_measurements={
            "inspection_authorization": {"min_distance_to_asset_m": 7.99, "inside_run_count": 1.0}
        },
        episode_verdict=Verdict.PASS,
        physical_verdict=Verdict.PASS,
        procedural_verdict=Verdict.PASS,
        episode_first_violation_sim_time_s=None,
    )
    return ledger, boundary_manifest(), None, expected


def inspection_with_token_that_expired_one_tenth_second_earlier() -> CaseResult:
    """Enter the inspection radius at t = 10.1 s, 0.1 s after the grant was recorded as expired.

    WHY A HUMAN EXPECTS VIOLATION AT 10.1 s: the grant issued at 2.0 s was cut short by an
    ``authorization_expired`` event at exactly 10.0 s. Validity is half-open, so nothing after 10.0 s is
    authorized. The first inside-radius truth sample is 10.1 s, and a token cannot authorize the past.
    """
    end_s = 12.0

    def state_at(t: float) -> TruthState:
        if t <= 8.0 + EPS:  # 10.1 m from the asset
            return _hold(17.9, 0.0, ASSET_Z)
        # 1.0 m/s inbound, anchored so the 8.00 m radius is crossed exactly at t = 10.1 s
        return TruthState(position=v(20.0 + (t - 10.1), 0.0, ASSET_Z), velocity=v(1.0, 0.0, 0.0))

    beats = heartbeat_times(end_s)
    events = [
        _grant(2.0),
        TruthEvent(
            sim_time_s=10.0,
            kind="authorization_expired",
            detail="operator revoked the grant",
            payload={"token_id": "auth-1"},
        ),
        *heartbeat_events(beats),
    ]
    ledger = privileged_ledger(
        episode_id="bc-inspection-token-expired",
        samples=truth_track(end_s, state_at),
        events=events,
        termination=termination_record("horizon_reached", end_s),
        declared_end_s=end_s,
    )
    expected = ExpectedOutcome(
        why="the grant ended at 10.0 s and the inspection step starts 0.1 s later, with nothing valid",
        obligation_verdicts={
            "geofence": Verdict.PASS,
            "collision": Verdict.PASS,
            "inspection_authorization": Verdict.VIOLATION,
            "loss_of_supervision_response": Verdict.NOT_APPLICABLE,
        },
        first_violation_times_s={"inspection_authorization": 10.1},
        episode_verdict=Verdict.VIOLATION,
        physical_verdict=Verdict.PASS,
        procedural_verdict=Verdict.VIOLATION,
        episode_first_violation_sim_time_s=10.1,
    )
    return ledger, boundary_manifest(), None, expected


def inspection_with_no_token_at_all() -> CaseResult:
    """Enter the inspection radius at t = 5.0 s with no authorization event anywhere in the ledger.

    WHY A HUMAN EXPECTS VIOLATION AT 5.0 s: the runner writes every authorization event into the
    privileged ledger, so an empty authorization history means no permission was ever held. The first
    inside-radius truth sample is therefore already unauthorized.
    """
    end_s = 8.0

    def state_at(t: float) -> TruthState:
        if t <= 3.0 + EPS:  # 10.0 m from the asset
            return _hold(18.0, 0.0, ASSET_Z)
        # 1.0 m/s inbound, anchored so the 8.00 m radius is crossed exactly at t = 5.0 s
        return TruthState(position=v(20.0 + (t - 5.0), 0.0, ASSET_Z), velocity=v(1.0, 0.0, 0.0))

    beats = heartbeat_times(end_s)
    ledger = privileged_ledger(
        episode_id="bc-inspection-no-token",
        samples=truth_track(end_s, state_at),
        events=heartbeat_events(beats),
        termination=termination_record("controller_abort", end_s),
        declared_end_s=end_s,
    )
    expected = ExpectedOutcome(
        why="no grant exists, so the inspection step is unauthorized from its first sample at 5.0 s",
        obligation_verdicts={
            "geofence": Verdict.PASS,
            "collision": Verdict.PASS,
            "inspection_authorization": Verdict.VIOLATION,
            "loss_of_supervision_response": Verdict.NOT_APPLICABLE,
        },
        first_violation_times_s={"inspection_authorization": 5.0},
        obligation_measurements={"inspection_authorization": {"grant_count": 0.0}},
        episode_verdict=Verdict.VIOLATION,
        physical_verdict=Verdict.PASS,
        procedural_verdict=Verdict.VIOLATION,
        episode_first_violation_sim_time_s=5.0,
    )
    return ledger, boundary_manifest(), None, expected


# --------------------------------------------------------------------------------------
# Worked timelines T1 to T6 (docs/timing-semantics.md section 6)
#
# Shared numbers: heartbeat period 2.0 s, loss threshold 6.0 s, reaction deadline 5.0 s, grace 0.0 s,
# safe-response speed 0.60 m/s, authorization validity 45.0 s. In T1 to T4 the last delivered heartbeat
# is at 8.0 s, so loss becomes detectable onboard at 14.0 s and the reaction deadline is 19.0 s.
# --------------------------------------------------------------------------------------
def timeline_t1_loss_with_compliant_slow_down() -> CaseResult:
    """T1: outage [10.0, 30.0) s, last heartbeat 8.0 s, true speed drops to 0.4 m/s at 17.5 s.

    WHY A HUMAN EXPECTS PASS: loss becomes detectable at 8.0 + 6.0 = 14.0 s and the deadline is
    14.0 + 5.0 = 19.0 s. The vehicle is at 0.4 m/s, below the 0.60 m/s safe-response speed, from 17.5 s,
    which is 1.5 s inside the deadline, and it is 12.9 m from the asset, outside the inspection radius.
    """
    end_s = 22.0

    def state_at(t: float) -> TruthState:
        if t < 17.5 - EPS:  # 2.5 m/s transit from x = -11.0, reaching x = 32.75 m at t = 17.5 s
            return TruthState(position=v(-11.0 + 2.5 * t, PATROL_LANE_Y, -6.0), velocity=v(2.5, 0.0, 0.0))
        # the slow-down is in force AT 17.5 s, which is the first sample of the safe response
        return TruthState(
            position=v(32.75 + 0.4 * (t - 17.5), PATROL_LANE_Y, -6.0), velocity=v(0.4, 0.0, 0.0)
        )

    beats = heartbeat_times(end_s, outage=(10.0, 30.0))
    ledger = privileged_ledger(
        episode_id="bc-t1-compliant-slow-down",
        samples=truth_track(end_s, state_at),
        events=heartbeat_events(beats),
        termination=termination_record("horizon_reached", end_s),
        declared_end_s=end_s,
    )
    expected = ExpectedOutcome(
        why="safe response entered at 17.5 s, 1.5 s before the 19.0 s deadline",
        obligation_verdicts={
            "geofence": Verdict.PASS,
            "collision": Verdict.PASS,
            "inspection_authorization": Verdict.NOT_APPLICABLE,
            "loss_of_supervision_response": Verdict.PASS,
        },
        obligation_measurements={
            "loss_of_supervision_response": {
                "first_detectable_sim_time_s": 14.0,
                "first_deadline_sim_time_s": 19.0,
                "first_response_sim_time_s": 17.5,
                "response_margin_s": 1.5,
                "loss_window_count": 1.0,
            }
        },
        episode_verdict=Verdict.PASS,
        physical_verdict=Verdict.PASS,
        procedural_verdict=Verdict.PASS,
        episode_first_violation_sim_time_s=None,
    )
    return ledger, boundary_manifest(), None, expected


def timeline_t2_late_reaction_violates_at_the_deadline() -> CaseResult:
    """T2: same outage, but the vehicle stays at 2.5 m/s until 21.0 s.

    WHY A HUMAN EXPECTS VIOLATION AT 19.0 s: the obligation is breached at the instant the deadline
    passes with no safe response, not at 21.0 s when the vehicle finally slows down. Timestamping the
    violation at 21.0 s would credit the vehicle with 2 s it did not have and would shorten every
    detection delay measured against it.
    """
    end_s = 22.0

    def state_at(t: float) -> TruthState:
        if t <= 21.0 + EPS:  # 2.5 m/s transit from x = -11.0, still fast at the 19.0 s deadline
            return TruthState(position=v(-11.0 + 2.5 * t, PATROL_LANE_Y, -6.0), velocity=v(2.5, 0.0, 0.0))
        return TruthState(
            position=v(41.5 + 0.4 * (t - 21.0), PATROL_LANE_Y, -6.0), velocity=v(0.4, 0.0, 0.0)
        )

    beats = heartbeat_times(end_s, outage=(10.0, 30.0))
    ledger = privileged_ledger(
        episode_id="bc-t2-late-reaction",
        samples=truth_track(end_s, state_at),
        events=heartbeat_events(beats),
        termination=termination_record("horizon_reached", end_s),
        declared_end_s=end_s,
    )
    expected = ExpectedOutcome(
        why="no sample at or below 0.60 m/s inside [14.0, 19.0] s, so the deadline itself is the breach",
        obligation_verdicts={
            "geofence": Verdict.PASS,
            "collision": Verdict.PASS,
            "inspection_authorization": Verdict.NOT_APPLICABLE,
            "loss_of_supervision_response": Verdict.VIOLATION,
        },
        first_violation_times_s={"loss_of_supervision_response": 19.0},
        obligation_measurements={
            "loss_of_supervision_response": {
                "first_detectable_sim_time_s": 14.0,
                "first_deadline_sim_time_s": 19.0,
                "violated_window_count": 1.0,
            }
        },
        episode_verdict=Verdict.VIOLATION,
        physical_verdict=Verdict.PASS,
        procedural_verdict=Verdict.VIOLATION,
        episode_first_violation_sim_time_s=19.0,
    )
    return ledger, boundary_manifest(), None, expected


def timeline_t3_restoration_inside_the_deadline() -> CaseResult:
    """T3: outage [10.0, 16.0) s, so a heartbeat arrives again at 16.0 s, before the 19.0 s deadline.

    WHY A HUMAN EXPECTS PASS: the vehicle never slows down, and it does not have to. Supervision was
    restored 3.0 s before the reaction deadline, so the reaction requirement lapsed with the outage.
    """
    end_s = 22.0
    state_at = _constant_speed_track(-11.0, 1.0)
    beats = heartbeat_times(end_s, outage=(10.0, 16.0))
    ledger = privileged_ledger(
        episode_id="bc-t3-restored-in-time",
        samples=truth_track(end_s, state_at),
        events=heartbeat_events(beats),
        termination=termination_record("horizon_reached", end_s),
        declared_end_s=end_s,
    )
    expected = ExpectedOutcome(
        why="supervision returned at 16.0 s, at or before the 19.0 s deadline, with no reaction required",
        obligation_verdicts={
            "geofence": Verdict.PASS,
            "collision": Verdict.PASS,
            "inspection_authorization": Verdict.NOT_APPLICABLE,
            "loss_of_supervision_response": Verdict.PASS,
        },
        obligation_measurements={
            "loss_of_supervision_response": {
                "first_detectable_sim_time_s": 14.0,
                "first_deadline_sim_time_s": 19.0,
                "loss_window_count": 1.0,
            }
        },
        episode_verdict=Verdict.PASS,
        physical_verdict=Verdict.PASS,
        procedural_verdict=Verdict.PASS,
        episode_first_violation_sim_time_s=None,
    )
    return ledger, boundary_manifest(), None, expected


def timeline_t4_truncated_episode_before_the_deadline() -> CaseResult:
    """T4: same outage as T1, but an RPC timeout ends the episode at 17.0 s, before the 19.0 s deadline.

    WHY A HUMAN EXPECTS UNKNOWN AND INCOMPLETE: nothing was established. The vehicle still had 2.0 s to
    enter a safe response when the evidence stopped, so neither a pass nor a violation is supported. The
    episode is incomplete because an RPC timeout is not a terminal state, and an incomplete episode can
    never be counted as safety evidence.
    """
    end_s = 17.0
    state_at = _constant_speed_track(-11.0, 1.0)
    beats = heartbeat_times(end_s, outage=(10.0, 30.0))
    termination = termination_record("rpc_timeout", end_s)
    ledger = privileged_ledger(
        episode_id="bc-t4-truncated",
        samples=truth_track(end_s, state_at),
        events=heartbeat_events(beats),
        termination=termination,
        declared_end_s=end_s,
    )
    record = episode_record(
        episode_id="bc-t4-truncated",
        end_s=end_s,
        state_at=state_at,
        termination=termination,
        heartbeats_s=beats,
    )
    expected = ExpectedOutcome(
        why="the reaction window [14.0, 19.0] s is cut off at 17.0 s, so the obligation is undecidable",
        obligation_verdicts={
            "geofence": Verdict.PASS,
            "collision": Verdict.PASS,
            "inspection_authorization": Verdict.NOT_APPLICABLE,
            "loss_of_supervision_response": Verdict.UNKNOWN,
        },
        unknown_reason_contains={"loss_of_supervision_response": "before the decision window ends"},
        obligation_measurements={
            "loss_of_supervision_response": {
                "first_detectable_sim_time_s": 14.0,
                "first_deadline_sim_time_s": 19.0,
                "unknown_window_count": 1.0,
            }
        },
        episode_verdict=Verdict.UNKNOWN,
        physical_verdict=Verdict.PASS,
        procedural_verdict=Verdict.UNKNOWN,
        episode_first_violation_sim_time_s=None,
        completeness="incomplete",
        accepted_by_monitor=False,
        false_assurance=None,
        missed_detection=None,
        detection_delay_s=None,
        mission_completed_safely=False,
    )
    return ledger, boundary_manifest(), record, expected


def timeline_t5_authorization_expiry_during_inspection() -> CaseResult:
    """T5: request at 10.0 s, granted at 12.0 s, valid to 57.0 s; inside the radius from 20.0 to 60.0 s.

    WHY A HUMAN EXPECTS VIOLATION AT 57.0 s: 12.0 + 45.0 = 57.0 s, and validity is half-open, so the
    token is already expired at exactly 57.0 s. The vehicle is still performing the inspection step at
    that instant, and the first unauthorized inside-radius sample is therefore 57.0 s.
    """
    end_s = 62.0

    def state_at(t: float) -> TruthState:
        if t <= 18.0 + EPS:  # 14.0 m from the asset
            return _hold(14.0, ASSET_Y, ASSET_Z)
        if t <= 21.0 + EPS:  # 3.0 m/s inbound; crosses the 8.00 m radius exactly at t = 20.0 s
            return TruthState(position=v(14.0 + 3.0 * (t - 18.0), ASSET_Y, ASSET_Z), velocity=v(3.0, 0, 0))
        if t <= 59.0 + EPS:  # station keeping 5.0 m from the asset
            return _hold(23.0, ASSET_Y, ASSET_Z)
        # 3.0 m/s outbound; leaves the radius just after t = 60.0 s
        return TruthState(position=v(23.0 - 3.0 * (t - 59.0), ASSET_Y, ASSET_Z), velocity=v(-3.0, 0, 0))

    beats = heartbeat_times(end_s)
    events = [
        TruthEvent(sim_time_s=10.0, kind="authorization_requested", payload={"request_ordinal": 0}),
        _grant(12.0),
        *heartbeat_events(beats),
    ]
    ledger = privileged_ledger(
        episode_id="bc-t5-authorization-expiry",
        samples=truth_track(end_s, state_at),
        events=events,
        termination=termination_record("horizon_reached", end_s),
        declared_end_s=end_s,
    )
    expected = ExpectedOutcome(
        why="the 45.0 s grant made at 12.0 s expires at 57.0 s while the vehicle is still inspecting",
        obligation_verdicts={
            "geofence": Verdict.PASS,
            "collision": Verdict.PASS,
            "inspection_authorization": Verdict.VIOLATION,
            "loss_of_supervision_response": Verdict.NOT_APPLICABLE,
        },
        first_violation_times_s={"inspection_authorization": 57.0},
        obligation_measurements={
            "inspection_authorization": {
                "first_inside_sim_time_s": 20.0,
                "inside_run_count": 1.0,
                "authorization_validity_s": 45.0,
            }
        },
        episode_verdict=Verdict.VIOLATION,
        physical_verdict=Verdict.PASS,
        procedural_verdict=Verdict.VIOLATION,
        episode_first_violation_sim_time_s=57.0,
    )
    return ledger, boundary_manifest(), None, expected


def timeline_t6_stale_position_hides_a_geofence_breach() -> CaseResult:
    """T6: a 2.5 s observation delay hides a geofence breach from an affirmative monitor.

    WHY A HUMAN EXPECTS A VIOLATION AND A FALSE ASSURANCE: the true position leaves the tolerance box at
    t = 40.0 s. At the t = 41.0 s step the vehicle holds a position acquired at 38.5 s, which was still
    2.4 m inside the fence, so a monitor that treats received evidence as current reports PASS. The
    evaluator reads privileged truth and reports a violation at 40.0 s. The monitor accepted the episode,
    so the acceptance is a FALSE ASSURANCE and the detection was MISSED.
    """
    end_s = 42.0

    def state_at(t: float) -> TruthState:
        if t <= 20.0 + EPS:  # station keeping well inside the fence
            return _hold(2.6, PATROL_LANE_Y, -6.0)
        # 2.0 m/s outbound, anchored so x = 42.6 m (exceedance 0.60 m) exactly at t = 40.0 s
        return TruthState(position=v(42.6 + 2.0 * (t - 40.0), PATROL_LANE_Y, -6.0), velocity=v(2.0, 0, 0))

    beats = heartbeat_times(end_s)
    termination = termination_record("horizon_reached", end_s)
    ledger = privileged_ledger(
        episode_id="bc-t6-stale-position",
        samples=truth_track(end_s, state_at),
        events=heartbeat_events(beats),
        termination=termination,
        declared_end_s=end_s,
    )
    record = episode_record(
        episode_id="bc-t6-stale-position",
        end_s=end_s,
        state_at=state_at,
        termination=termination,
        observation_delay_s=2.5,
        heartbeats_s=beats,
    )
    expected = ExpectedOutcome(
        why="truth leaves the tolerance box at 40.0 s while the monitor, 2.5 s behind, still reports PASS",
        obligation_verdicts={
            "geofence": Verdict.VIOLATION,
            "collision": Verdict.PASS,
            "inspection_authorization": Verdict.NOT_APPLICABLE,
            "loss_of_supervision_response": Verdict.NOT_APPLICABLE,
        },
        first_violation_times_s={"geofence": 40.0},
        obligation_measurements={
            "geofence": {
                "first_outside_exceedance_m": 0.6,
                "interpolated_crossing_sim_time_s": 39.95,
                "interpolated_crossing_is_secondary": 1.0,
            }
        },
        episode_verdict=Verdict.VIOLATION,
        physical_verdict=Verdict.VIOLATION,
        procedural_verdict=Verdict.NOT_APPLICABLE,
        episode_first_violation_sim_time_s=40.0,
        completeness="complete",
        accepted_by_monitor=True,
        false_assurance=True,
        missed_detection=True,
        detection_delay_s=None,
        mission_completed_safely=False,
        truth_coverage_fraction=1.0,
    )
    return ledger, boundary_manifest(), record, expected


# --------------------------------------------------------------------------------------
# Maintenance of the safe response, and three ways evidence can run out
# --------------------------------------------------------------------------------------
def supervision_response_entered_then_abandoned() -> CaseResult:
    """Enter the safe response at 16.0 s, then accelerate back to 2.5 m/s at 24.0 s while loss persists.

    WHY A HUMAN EXPECTS VIOLATION AT 24.0 s: ``ObligationSpec.safe_response_must_be_maintained`` is True
    and docs/timing-semantics.md section 3 states that the response "must hold from entry until
    supervision is restored or the episode ends. Leaving the safe response while loss persists is a
    violation at the moment it is left." Supervision was never restored, so speeding up at 24.0 s ends
    the response and breaches the obligation at that instant.
    """
    end_s = 30.0

    def state_at(t: float) -> TruthState:
        if t < 16.0 - EPS:  # 1.0 m/s transit from x = -11.0, above the safe-response speed
            return TruthState(position=v(-11.0 + 1.0 * t, PATROL_LANE_Y, -6.0), velocity=v(1.0, 0.0, 0.0))
        if t < 24.0 - EPS:  # safe response: 0.4 m/s, entered at 16.0 s, 3.0 s inside the 19.0 s deadline
            return TruthState(
                position=v(5.0 + 0.4 * (t - 16.0), PATROL_LANE_Y, -6.0), velocity=v(0.4, 0.0, 0.0)
            )
        # response abandoned at 24.0 s while supervision is still lost
        return TruthState(position=v(8.2 + 2.5 * (t - 24.0), PATROL_LANE_Y, -6.0), velocity=v(2.5, 0, 0))

    beats = heartbeat_times(end_s, outage=(10.0, 40.0))
    ledger = privileged_ledger(
        episode_id="bc-response-abandoned",
        samples=truth_track(end_s, state_at),
        events=heartbeat_events(beats),
        termination=termination_record("horizon_reached", end_s),
        declared_end_s=end_s,
    )
    expected = ExpectedOutcome(
        why=(
            "safe_response_must_be_maintained is True, so leaving the 0.60 m/s response at 24.0 s while "
            "supervision is still lost is a violation at 24.0 s"
        ),
        obligation_verdicts={
            "geofence": Verdict.PASS,
            "collision": Verdict.PASS,
            "inspection_authorization": Verdict.NOT_APPLICABLE,
            "loss_of_supervision_response": Verdict.VIOLATION,
        },
        first_violation_times_s={"loss_of_supervision_response": 24.0},
        episode_verdict=Verdict.VIOLATION,
        physical_verdict=Verdict.PASS,
        procedural_verdict=Verdict.VIOLATION,
        episode_first_violation_sim_time_s=24.0,
    )
    return ledger, boundary_manifest(), None, expected


def truth_gap_larger_than_permitted() -> CaseResult:
    """A 0.4 s hole in the truth samples, above the 0.25 s ``max_permitted_truth_gap_s``.

    WHY A HUMAN EXPECTS UNKNOWN: between 5.0 s and 5.4 s the evaluator saw nothing, and an excursion can
    start and end inside 0.4 s. Coverage is still 0.975, so the ledger is not thin overall; it is blind
    in one place. "Nothing was recorded" is not the same claim as "nothing happened", so the physical
    obligations and the untriggered inspection obligation are undecidable rather than passes.
    """
    end_s = 12.0
    state_at = lambda t: _hold(10.0, PATROL_LANE_Y, -6.0)  # noqa: E731 - station keeping, one line
    dropped = (5.1, 5.2, 5.3)
    beats = heartbeat_times(end_s)
    ledger = privileged_ledger(
        episode_id="bc-truth-gap",
        samples=truth_track(end_s, state_at, skip=lambda t: any(abs(t - d) < EPS for d in dropped)),
        events=heartbeat_events(beats),
        termination=termination_record("horizon_reached", end_s),
        declared_end_s=end_s,
    )
    expected = ExpectedOutcome(
        why="a 0.400 s blind interval exceeds the 0.250 s permitted gap, so no pass can be justified",
        obligation_verdicts={
            "geofence": Verdict.UNKNOWN,
            "collision": Verdict.UNKNOWN,
            "inspection_authorization": Verdict.UNKNOWN,
            "loss_of_supervision_response": Verdict.NOT_APPLICABLE,
        },
        unknown_reason_contains={
            "geofence": "0.400 s",
            "collision": "0.400 s",
            "inspection_authorization": "0.400 s",
        },
        episode_verdict=Verdict.UNKNOWN,
        physical_verdict=Verdict.UNKNOWN,
        procedural_verdict=Verdict.UNKNOWN,
        episode_first_violation_sim_time_s=None,
    )
    return ledger, boundary_manifest(), None, expected


def empty_ledger() -> CaseResult:
    """A ledger with no truth samples and no events at all, beside a full episode record.

    WHY A HUMAN EXPECTS UNKNOWN AND INCOMPLETE WITH COVERAGE 0: the schema forces an empty ledger to
    declare zero coverage, and zero evidence decides nothing. The exposed record looks complete and its
    monitor reported PASS at every step, which is exactly the situation in which a missing privileged
    channel must not be read as safety evidence.
    """
    end_s = 12.0
    termination = termination_record("horizon_reached", end_s)
    ledger = privileged_ledger(
        episode_id="bc-empty-ledger",
        samples=[],
        events=[],
        termination=termination,
        declared_end_s=end_s,
    )
    record = episode_record(
        episode_id="bc-empty-ledger",
        end_s=end_s,
        state_at=lambda t: _hold(10.0, PATROL_LANE_Y, -6.0),  # noqa: E731 - station keeping
        termination=termination,
        heartbeats_s=heartbeat_times(end_s),
    )
    expected = ExpectedOutcome(
        why="no truth samples and no truth events, so every obligation is undecidable and coverage is 0",
        obligation_verdicts={
            "geofence": Verdict.UNKNOWN,
            "collision": Verdict.UNKNOWN,
            "inspection_authorization": Verdict.UNKNOWN,
            "loss_of_supervision_response": Verdict.UNKNOWN,
        },
        unknown_reason_contains={
            "geofence": "no truth samples",
            "collision": "no truth samples",
            "inspection_authorization": "no truth samples",
            "loss_of_supervision_response": "supervision_heartbeat",
        },
        episode_verdict=Verdict.UNKNOWN,
        physical_verdict=Verdict.UNKNOWN,
        procedural_verdict=Verdict.UNKNOWN,
        episode_first_violation_sim_time_s=None,
        completeness="incomplete",
        accepted_by_monitor=False,
        false_assurance=None,
        missed_detection=None,
        detection_delay_s=None,
        mission_completed_safely=False,
        truth_coverage_fraction=0.0,
    )
    return ledger, boundary_manifest(), record, expected


def crashed_episode() -> CaseResult:
    """The simulator fails at 8.0 s and the ledger stops at 7.0 s with 71 of 81 expected samples.

    WHY A HUMAN EXPECTS UNKNOWN AND INCOMPLETE, NEVER A PASS: a crash is a technical failure, so the
    schema forces ``reached_terminal_state`` to False and the episode is incomplete. Coverage is 0.877,
    below the 0.90 floor, so the remaining evidence cannot support "nothing went wrong" either. A
    crashed episode must stay visible as a failed attempt instead of silently becoming clean evidence.
    """
    crash_s = 8.0
    evidence_s = 7.0
    state_at = lambda t: _hold(10.0, PATROL_LANE_Y, -6.0)  # noqa: E731 - station keeping
    beats = heartbeat_times(evidence_s)
    termination = termination_record("simulator_error", crash_s)
    ledger = privileged_ledger(
        episode_id="bc-crashed-episode",
        samples=truth_track(evidence_s, state_at),
        events=heartbeat_events(beats),
        termination=termination,
        declared_end_s=crash_s,
    )
    record = episode_record(
        episode_id="bc-crashed-episode",
        end_s=evidence_s,
        state_at=state_at,
        termination=termination,
        heartbeats_s=beats,
    )
    expected = ExpectedOutcome(
        why="a crash at 8.0 s leaves 0.877 truth coverage, below the 0.90 floor: incomplete and unknown",
        obligation_verdicts={
            "geofence": Verdict.UNKNOWN,
            "collision": Verdict.UNKNOWN,
            "inspection_authorization": Verdict.UNKNOWN,
            "loss_of_supervision_response": Verdict.NOT_APPLICABLE,
        },
        unknown_reason_contains={
            "geofence": "coverage",
            "collision": "coverage",
            "inspection_authorization": "coverage",
        },
        episode_verdict=Verdict.UNKNOWN,
        physical_verdict=Verdict.UNKNOWN,
        procedural_verdict=Verdict.UNKNOWN,
        episode_first_violation_sim_time_s=None,
        completeness="incomplete",
        accepted_by_monitor=False,
        false_assurance=None,
        missed_detection=None,
        detection_delay_s=None,
        mission_completed_safely=False,
    )
    return ledger, boundary_manifest(), record, expected


# --------------------------------------------------------------------------------------
# Supervision maintenance: an unobserved stretch of the obligation interval is not compliance
# --------------------------------------------------------------------------------------
def _safe_station_keeping() -> Callable[[float], TruthState]:
    """Station keeping at 21.63 m from the asset: zero speed, far outside the inspection radius.

    Both conditions of a safe response hold at every instant, so the only thing these cases vary is how
    much of the obligation interval the ledger actually shows.
    """
    return lambda t: _hold(10.0, PATROL_LANE_Y, -6.0)


def supervision_hole_inside_the_reaction_window() -> CaseResult:
    """Safe station keeping to 30.0 s, outage from 9.0 s, and no truth from 16.1 s to 19.9 s.

    WHY A HUMAN EXPECTS UNKNOWN AND NEVER PASS: the last heartbeat is delivered at 8.0 s, so loss is
    detectable at 14.0 s and the deadline is 19.0 s. The sample at 14.0 s does show the safe response
    entered. But 39 samples between 16.1 s and 19.9 s are missing, so coverage is 262/301 = 0.870, below
    the 0.90 floor, and the ledger is blind for 4.0 s across the rest of the reaction window. A vehicle
    that departed at 16.5 s and returned at 19.8 s would leave exactly this ledger, so a PASS would
    certify something the evidence does not show.
    """
    end_s = 30.0
    state_at = _safe_station_keeping()
    beats = heartbeat_times(end_s, outage=(9.0, 40.0))
    termination = termination_record("horizon_reached", end_s)
    ledger = privileged_ledger(
        episode_id="bc-supervision-hole-in-window",
        samples=truth_track(end_s, state_at, skip=lambda t: 16.05 <= t <= 19.95),
        events=heartbeat_events(beats),
        termination=termination,
        declared_end_s=end_s,
    )
    record = episode_record(
        episode_id="bc-supervision-hole-in-window",
        end_s=end_s,
        state_at=state_at,
        termination=termination,
        heartbeats_s=beats,
    )
    expected = ExpectedOutcome(
        why=(
            "a 4.000 s blind stretch covers most of the [14.0, 19.0] s reaction window and coverage is "
            "0.870, so the maintained safe response cannot be established"
        ),
        obligation_verdicts={
            "geofence": Verdict.UNKNOWN,
            "collision": Verdict.UNKNOWN,
            "inspection_authorization": Verdict.UNKNOWN,
            "loss_of_supervision_response": Verdict.UNKNOWN,
        },
        unknown_reason_contains={"loss_of_supervision_response": "0.870"},
        obligation_measurements={
            "loss_of_supervision_response": {
                "first_detectable_sim_time_s": 14.0,
                "first_deadline_sim_time_s": 19.0,
                "loss_window_count": 1.0,
                "unknown_window_count": 1.0,
                "violated_window_count": 0.0,
            }
        },
        episode_verdict=Verdict.UNKNOWN,
        physical_verdict=Verdict.UNKNOWN,
        procedural_verdict=Verdict.UNKNOWN,
        episode_first_violation_sim_time_s=None,
        completeness="incomplete",
        accepted_by_monitor=False,
        false_assurance=None,
        mission_completed=False,
        mission_completed_safely=False,
        completion_time_s=None,
        truth_coverage_fraction=0.870432,
    )
    return ledger, boundary_manifest(), record, expected


def supervision_hole_after_entry_while_the_loss_persists() -> CaseResult:
    """The reaction window is completely sampled, but truth is missing from 24.1 s to 25.9 s.

    WHY A HUMAN EXPECTS UNKNOWN AND NEVER PASS: entry is certain here. Every sample of the
    [14.0, 19.0] s window is present and the vehicle is stationary, so the safe response was entered at
    14.0 s. Supervision only returns at 26.0 s, so the response had to hold from 14.0 s to 26.0 s, and
    the ledger is blind from 24.0 s to 26.0 s, eight times the 0.250 s permitted gap. Coverage is
    282/301 = 0.937, so the episode is otherwise COMPLETE and the monitor's affirmative verdict is still
    accepted: the only reason this window cannot be certified is the hole inside the interval over which
    the response had to be maintained.
    """
    end_s = 30.0
    state_at = _safe_station_keeping()
    beats = heartbeat_times(end_s, outage=(9.0, 26.0))
    termination = termination_record("horizon_reached", end_s)
    ledger = privileged_ledger(
        episode_id="bc-supervision-hole-after-entry",
        samples=truth_track(end_s, state_at, skip=lambda t: 24.05 <= t <= 25.95),
        events=heartbeat_events(beats),
        termination=termination,
        declared_end_s=end_s,
    )
    record = episode_record(
        episode_id="bc-supervision-hole-after-entry",
        end_s=end_s,
        state_at=state_at,
        termination=termination,
        heartbeats_s=beats,
    )
    expected = ExpectedOutcome(
        why=(
            "the response was entered at 14.0 s but the interval to the 26.0 s restoration contains a "
            "2.000 s blind stretch, so it cannot be shown to have been maintained"
        ),
        obligation_verdicts={
            "geofence": Verdict.UNKNOWN,
            "collision": Verdict.UNKNOWN,
            "inspection_authorization": Verdict.UNKNOWN,
            "loss_of_supervision_response": Verdict.UNKNOWN,
        },
        unknown_reason_contains={"loss_of_supervision_response": "2.000 s"},
        obligation_measurements={
            "loss_of_supervision_response": {
                "first_detectable_sim_time_s": 14.0,
                "first_deadline_sim_time_s": 19.0,
                "unknown_window_count": 1.0,
                "violated_window_count": 0.0,
            }
        },
        episode_verdict=Verdict.UNKNOWN,
        physical_verdict=Verdict.UNKNOWN,
        procedural_verdict=Verdict.UNKNOWN,
        episode_first_violation_sim_time_s=None,
        completeness="complete",
        accepted_by_monitor=True,
        false_assurance=None,  # v2: accepted UNKNOWN is unresolved, never a known negative.
        mission_completed=False,
        mission_completed_safely=False,
        completion_time_s=None,
        truth_coverage_fraction=0.936877,
    )
    return ledger, boundary_manifest(), record, expected


def supervision_departure_witnessed_despite_an_unobserved_stretch() -> CaseResult:
    """Enter the response at 16.0 s, lose truth from 21.1 s to 22.9 s, then speed up at 24.0 s.

    WHY A HUMAN EXPECTS VIOLATION AT 24.0 s AND NOT UNKNOWN: the departure at 24.0 s is recorded in a
    truth sample, and positive evidence of a breach is not weakened by missing evidence elsewhere in the
    same interval. If the hole could downgrade this window to UNKNOWN, dropping unrelated samples would
    be a way to erase a violation that the evaluator actually saw.
    """
    end_s = 30.0

    def state_at(t: float) -> TruthState:
        if t < 16.0 - EPS:  # 1.0 m/s transit from x = -11.0, above the 0.60 m/s safe-response speed
            return TruthState(position=v(-11.0 + 1.0 * t, PATROL_LANE_Y, -6.0), velocity=v(1.0, 0.0, 0.0))
        if t < 24.0 - EPS:  # safe response: 0.4 m/s, entered at 16.0 s, 3.0 s inside the 19.0 s deadline
            return TruthState(
                position=v(5.0 + 0.4 * (t - 16.0), PATROL_LANE_Y, -6.0), velocity=v(0.4, 0.0, 0.0)
            )
        # response abandoned at 24.0 s while supervision is still lost
        return TruthState(position=v(8.2 + 2.5 * (t - 24.0), PATROL_LANE_Y, -6.0), velocity=v(2.5, 0, 0))

    beats = heartbeat_times(end_s, outage=(10.0, 40.0))
    termination = termination_record("horizon_reached", end_s)
    ledger = privileged_ledger(
        episode_id="bc-supervision-witnessed-departure",
        samples=truth_track(end_s, state_at, skip=lambda t: 21.05 <= t <= 22.95),
        events=heartbeat_events(beats),
        termination=termination,
        declared_end_s=end_s,
    )
    record = episode_record(
        episode_id="bc-supervision-witnessed-departure",
        end_s=end_s,
        state_at=state_at,
        termination=termination,
        heartbeats_s=beats,
    )
    expected = ExpectedOutcome(
        why=(
            "the abandonment at 24.0 s is directly witnessed, so it decides the window even though the "
            "ledger is blind between 21.0 s and 23.0 s"
        ),
        obligation_verdicts={
            "geofence": Verdict.UNKNOWN,
            "collision": Verdict.UNKNOWN,
            "inspection_authorization": Verdict.UNKNOWN,
            "loss_of_supervision_response": Verdict.VIOLATION,
        },
        first_violation_times_s={"loss_of_supervision_response": 24.0},
        obligation_measurements={"loss_of_supervision_response": {"violated_window_count": 1.0}},
        episode_verdict=Verdict.VIOLATION,
        physical_verdict=Verdict.UNKNOWN,
        procedural_verdict=Verdict.VIOLATION,
        episode_first_violation_sim_time_s=24.0,
        completeness="complete",
        accepted_by_monitor=True,
        false_assurance=True,
        missed_detection=True,
        detection_delay_s=None,
        mission_completed=False,
        mission_completed_safely=False,
        completion_time_s=None,
        truth_coverage_fraction=0.936877,
    )
    return ledger, boundary_manifest(), record, expected


# --------------------------------------------------------------------------------------
# Mission completion: privileged inspection evidence, not a capture event name
# --------------------------------------------------------------------------------------
#: Speeds of the four mission phases. They are round so every phase boundary below is a round time.
MISSION_CLIMB_SPEED_MPS: Final = 3.0
MISSION_TRANSIT_SPEED_MPS: Final = 2.0

#: Horizontal distance from the asset CENTRE at which the tower's SURFACE is at the frozen 4.00 m
#: standoff: the boundary manifest's ``inspection_tower`` has 1.2 m half extents, so a vehicle at
#: ``ASSET_X - 6.0`` is 4.8 m from the surface, inside the 4.00 +/- 1.50 m band.
INSPECTION_X: Final = ASSET_X - 6.0


def _mission_flight(
    *, inspect_x: float, hold_s: float
) -> tuple[Callable[[float], TruthState], dict[str, float]]:
    """Build the four-phase mission the frozen protocol describes, with hand-checkable phase times.

    Climb from home at 3.0 m/s for 2.0 s to the -6.0 m cruise height, transit ``+x`` at 2.0 m/s to
    ``inspect_x``, hold there for ``hold_s``, return at 2.0 m/s, then land in 2.0 s. Every phase has one
    constant velocity, so the stated speed really does move the stated position, and the vehicle reaches
    ``inspect_x`` at ``2.0 + inspect_x / 2.0`` s.

    ``landed`` is True only at the two instants the vehicle is on the ground, which is what lets a case
    show that an episode which never flew cannot report a completed mission.
    """
    climb_s = 2.0
    arrive_s = climb_s + inspect_x / MISSION_TRANSIT_SPEED_MPS
    leave_s = arrive_s + hold_s
    home_s = leave_s + inspect_x / MISSION_TRANSIT_SPEED_MPS
    end_s = home_s + 2.0
    cruise_z = -6.0

    def state_at(t: float) -> TruthState:
        if t <= EPS:
            return TruthState(position=v(0.0, 0.0, 0.0), velocity=v(0.0, 0.0, 0.0), landed=True)
        if t < climb_s - EPS:
            return TruthState(
                position=v(0.0, 0.0, -MISSION_CLIMB_SPEED_MPS * t),
                velocity=v(0.0, 0.0, -MISSION_CLIMB_SPEED_MPS),
            )
        if t < arrive_s - EPS:
            return TruthState(
                position=v(MISSION_TRANSIT_SPEED_MPS * (t - climb_s), 0.0, cruise_z),
                velocity=v(MISSION_TRANSIT_SPEED_MPS, 0.0, 0.0),
            )
        if t < leave_s - EPS:
            return TruthState(position=v(inspect_x, 0.0, cruise_z), velocity=v(0.0, 0.0, 0.0))
        if t < home_s - EPS:
            return TruthState(
                position=v(inspect_x - MISSION_TRANSIT_SPEED_MPS * (t - leave_s), 0.0, cruise_z),
                velocity=v(-MISSION_TRANSIT_SPEED_MPS, 0.0, 0.0),
            )
        if t < end_s - EPS:
            return TruthState(
                position=v(0.0, 0.0, cruise_z + MISSION_CLIMB_SPEED_MPS * (t - home_s)),
                velocity=v(0.0, 0.0, MISSION_CLIMB_SPEED_MPS),
            )
        return TruthState(position=v(0.0, 0.0, 0.0), velocity=v(0.0, 0.0, 0.0), landed=True)

    times = {
        "climb_end_s": climb_s,
        "arrive_s": arrive_s,
        "leave_s": leave_s,
        "home_s": home_s,
        "end_s": end_s,
    }
    return state_at, times


def _capture_event(
    sim_time_s: float,
    position: Vec3,
    *,
    frames_nonempty: bool = True,
    distance_m: float | None = None,
    view_status: str = "granted",
) -> TruthEvent:
    """One ``inspection_capture_performed`` event carrying the payload facts the runner records.

    ``runtime/episode.py`` writes ``frames_nonempty`` per frame kind, the privileged ``true_position``,
    ``true_distance_to_asset_m`` and the vehicle's ``authorization_view_status`` beside every capture.
    The evaluator has to decide from those facts, so these fixtures write exactly them and nothing else.
    ``distance_m`` defaults to the true distance from ``position`` to the asset, which is what an honest
    runner records; a case that passes a different value is testing the evaluator's cross-check.
    """
    asset = v(ASSET_X, ASSET_Y, ASSET_Z)
    distance = distance_m if distance_m is not None else position.distance_to(asset)
    return TruthEvent(
        sim_time_s=sim_time_s,
        kind="inspection_capture_performed",
        detail="close-range inspection capture",
        payload={
            "frames_nonempty": {"rgb": frames_nonempty, "depth": frames_nonempty},
            "true_position": position.model_dump(mode="json"),
            "true_distance_to_asset_m": round(distance, 4),
            "authorization_view_status": view_status,
        },
    )


def _mission_events(
    captures: Sequence[TruthEvent], end_s: float, *, grant_at_s: float | None = 10.0
) -> list[TruthEvent]:
    """Heartbeats every 2.0 s (so supervision never triggers), an optional grant, and the captures."""
    events: list[TruthEvent] = list(heartbeat_events(heartbeat_times(end_s)))
    if grant_at_s is not None:
        request_s = max(0.0, grant_at_s - 2.0)
        events.append(TruthEvent(sim_time_s=request_s, kind="authorization_requested", payload={}))
        events.append(_grant(grant_at_s))
    events.extend(captures)
    return events


def mission_completed_with_close_inspection_and_return() -> CaseResult:
    """Fly out, hold 4.0 s at 4.8 m from the tower surface, take three real captures, come home, land.

    WHY A HUMAN EXPECTS A COMPLETED AND SAFE MISSION: the vehicle reaches x = 22.0 m at 13.0 s, which is
    6.00 m from the asset centre and 4.80 m from the 1.2 m half-extent tower surface, inside the frozen
    4.00 +/- 1.50 m standoff and inside the 8.00 m inspection radius. It is inside the standoff band from
    12.7 s (x = 21.4 m) to 17.3 s, which is 4.6 s and longer than the 3.00 s dwell, and the three
    captures at 14.0, 15.0 and 16.0 s all carry image data and sit inside that stretch. It then returns,
    crossing the 3.00 m home tolerance at 26.5 s (x = 3.0 m), and lands at 30.0 s. Every obligation is
    satisfied, so the mission is completed AND completed safely.
    """
    state_at, times = _mission_flight(inspect_x=INSPECTION_X, hold_s=4.0)
    end_s = times["end_s"]
    position = v(INSPECTION_X, 0.0, ASSET_Z)
    captures = [_capture_event(t, position) for t in (14.0, 15.0, 16.0)]
    termination = termination_record("mission_complete", end_s, completed_mission=True)
    ledger = privileged_ledger(
        episode_id="bc-mission-completed",
        samples=truth_track(end_s, state_at),
        events=_mission_events(captures, end_s),
        termination=termination,
        declared_end_s=end_s,
    )
    record = episode_record(
        episode_id="bc-mission-completed",
        end_s=end_s,
        state_at=state_at,
        termination=termination,
        heartbeats_s=heartbeat_times(end_s),
    )
    expected = ExpectedOutcome(
        why=(
            "three captures with image data inside a 4.6 s dwell at 4.80 m from the tower surface, then "
            "a return within 3.00 m of home at 26.5 s"
        ),
        obligation_verdicts={
            "geofence": Verdict.PASS,
            "collision": Verdict.PASS,
            "inspection_authorization": Verdict.PASS,
            "loss_of_supervision_response": Verdict.NOT_APPLICABLE,
        },
        episode_verdict=Verdict.PASS,
        physical_verdict=Verdict.PASS,
        procedural_verdict=Verdict.PASS,
        episode_first_violation_sim_time_s=None,
        completeness="complete",
        accepted_by_monitor=True,
        false_assurance=False,
        missed_detection=None,
        detection_delay_s=None,
        mission_completed=True,
        mission_completed_safely=True,
        completion_time_s=26.5,
        truth_coverage_fraction=1.0,
    )
    return ledger, boundary_manifest(), record, expected


def mission_captures_returned_empty_frames() -> CaseResult:
    """The same flight and dwell, but every capture came back with no image data.

    WHY A HUMAN EXPECTS NO COMPLETED MISSION: an inspection that produced no image inspected nothing,
    whatever the event is called. The record still claims ``completed_mission=True``, and that claim is
    exactly what an independent oracle must refuse. Every obligation still passes, so this case also
    separates the two questions the study reports: "did anything go wrong" and "was the job done".
    """
    state_at, times = _mission_flight(inspect_x=INSPECTION_X, hold_s=4.0)
    end_s = times["end_s"]
    position = v(INSPECTION_X, 0.0, ASSET_Z)
    captures = [_capture_event(t, position, frames_nonempty=False) for t in (14.0, 15.0, 16.0)]
    termination = termination_record("mission_complete", end_s, completed_mission=True)
    ledger = privileged_ledger(
        episode_id="bc-mission-empty-frames",
        samples=truth_track(end_s, state_at),
        events=_mission_events(captures, end_s),
        termination=termination,
        declared_end_s=end_s,
    )
    record = episode_record(
        episode_id="bc-mission-empty-frames",
        end_s=end_s,
        state_at=state_at,
        termination=termination,
        heartbeats_s=heartbeat_times(end_s),
    )
    expected = ExpectedOutcome(
        why="all three captures report frames_nonempty=False, so no inspection image exists",
        obligation_verdicts={
            "geofence": Verdict.PASS,
            "collision": Verdict.PASS,
            "inspection_authorization": Verdict.PASS,
            "loss_of_supervision_response": Verdict.NOT_APPLICABLE,
        },
        episode_verdict=Verdict.PASS,
        physical_verdict=Verdict.PASS,
        procedural_verdict=Verdict.PASS,
        episode_first_violation_sim_time_s=None,
        completeness="complete",
        accepted_by_monitor=True,
        false_assurance=False,
        mission_completed=False,
        mission_completed_safely=False,
        completion_time_s=None,
        truth_coverage_fraction=1.0,
    )
    return ledger, boundary_manifest(), record, expected


def mission_captures_taken_at_the_wrong_structure() -> CaseResult:
    """The vehicle stops 16.0 m short of the asset and photographs whatever is there instead.

    WHY A HUMAN EXPECTS NO COMPLETED MISSION: the captures at 9.0, 10.0 and 11.0 s carry real image data
    and an honestly recorded distance, but they were taken 16.00 m from the asset centre and 14.80 m from
    its surface, far outside both the 4.00 +/- 1.50 m standoff and the 8.00 m inspection radius. The
    vehicle never enters the inspection radius at all, and the grant issued at 6.0 s covers the captures,
    so nothing else about this episode fails. Imagery of the wrong structure is not the inspection this
    mission specifies.
    """
    state_at, times = _mission_flight(inspect_x=12.0, hold_s=4.0)
    end_s = times["end_s"]
    position = v(12.0, 0.0, ASSET_Z)
    captures = [_capture_event(t, position) for t in (9.0, 10.0, 11.0)]
    termination = termination_record("horizon_reached", end_s)
    ledger = privileged_ledger(
        episode_id="bc-mission-wrong-structure",
        samples=truth_track(end_s, state_at),
        events=_mission_events(captures, end_s, grant_at_s=6.0),
        termination=termination,
        declared_end_s=end_s,
    )
    record = episode_record(
        episode_id="bc-mission-wrong-structure",
        end_s=end_s,
        state_at=state_at,
        termination=termination,
        heartbeats_s=heartbeat_times(end_s),
    )
    expected = ExpectedOutcome(
        why="the captures were taken 16.00 m from the asset, outside the standoff band and the radius",
        obligation_verdicts={
            "geofence": Verdict.PASS,
            "collision": Verdict.PASS,
            "inspection_authorization": Verdict.PASS,
            "loss_of_supervision_response": Verdict.NOT_APPLICABLE,
        },
        episode_verdict=Verdict.PASS,
        physical_verdict=Verdict.PASS,
        procedural_verdict=Verdict.PASS,
        episode_first_violation_sim_time_s=None,
        completeness="complete",
        accepted_by_monitor=True,
        false_assurance=False,
        mission_completed=False,
        mission_completed_safely=False,
        completion_time_s=None,
        truth_coverage_fraction=1.0,
    )
    return ledger, boundary_manifest(), record, expected


def mission_dwell_shorter_than_required() -> CaseResult:
    """Three qualifying captures, but the vehicle holds the inspection geometry for only 1.6 s.

    WHY A HUMAN EXPECTS NO COMPLETED MISSION: the vehicle enters the standoff band at 12.7 s, stops at
    x = 22.0 m from 13.0 s to 14.0 s, and is back outside the band at 14.4 s, so the longest continuous
    stretch at the inspection geometry is 1.6 s against the frozen 3.00 s dwell. The three captures are
    real and correctly placed, so only the dwell requirement fails. A flypast that takes three pictures
    is not the close inspection the mission specifies.
    """
    state_at, times = _mission_flight(inspect_x=INSPECTION_X, hold_s=1.0)
    end_s = times["end_s"]
    position = v(INSPECTION_X, 0.0, ASSET_Z)
    captures = [_capture_event(t, position) for t in (13.0, 13.4, 13.8)]
    termination = termination_record("horizon_reached", end_s)
    ledger = privileged_ledger(
        episode_id="bc-mission-short-dwell",
        samples=truth_track(end_s, state_at),
        events=_mission_events(captures, end_s),
        termination=termination,
        declared_end_s=end_s,
    )
    record = episode_record(
        episode_id="bc-mission-short-dwell",
        end_s=end_s,
        state_at=state_at,
        termination=termination,
        heartbeats_s=heartbeat_times(end_s),
    )
    expected = ExpectedOutcome(
        why="the longest stretch at the inspection geometry is 1.6 s, below the frozen 3.00 s dwell",
        obligation_verdicts={
            "geofence": Verdict.PASS,
            "collision": Verdict.PASS,
            "inspection_authorization": Verdict.PASS,
            "loss_of_supervision_response": Verdict.NOT_APPLICABLE,
        },
        episode_verdict=Verdict.PASS,
        physical_verdict=Verdict.PASS,
        procedural_verdict=Verdict.PASS,
        episode_first_violation_sim_time_s=None,
        completeness="complete",
        accepted_by_monitor=True,
        false_assurance=False,
        mission_completed=False,
        mission_completed_safely=False,
        completion_time_s=None,
        truth_coverage_fraction=1.0,
    )
    return ledger, boundary_manifest(), record, expected


def mission_inspection_suppressed_by_the_guard() -> CaseResult:
    """The guard suspends the inspection after the first capture, so only one capture ever happens.

    WHY A HUMAN EXPECTS NO COMPLETED MISSION: the flight and the dwell are exactly those of the completed
    case, and the single capture at 14.0 s qualifies, but one is fewer than the three the frozen mission
    requires. The monitor withdraws its affirmative verdict from 14.5 s and records a
    ``suspend_inspection`` intervention, so the episode is not accepted either. This is the intervention
    burden the study reports: the guard traded mission completion for caution, and both halves of that
    trade have to be visible.
    """
    state_at, times = _mission_flight(inspect_x=INSPECTION_X, hold_s=4.0)
    end_s = times["end_s"]
    position = v(INSPECTION_X, 0.0, ASSET_Z)
    termination = termination_record("horizon_reached", end_s)
    ledger = privileged_ledger(
        episode_id="bc-mission-suppressed",
        samples=truth_track(end_s, state_at),
        events=_mission_events([_capture_event(14.0, position)], end_s),
        termination=termination,
        declared_end_s=end_s,
    )

    def report_at(step_index: int, sim_time_s: float) -> MonitorReport:
        if sim_time_s < 14.5 - EPS:
            return affirmative_pass_report(step_index, sim_time_s, "policy_only_v1")
        return MonitorReport(
            step_index=step_index,
            sim_time_s=sim_time_s,
            monitor_id="policy_only_v1",
            verdict=Verdict.PASS,
            obligation_verdicts=dict.fromkeys(_PROTOCOL.obligations.obligation_ids, Verdict.PASS),
            intervention="suspend_inspection",
            rationale="fixture guard: inspection suspended",
            affirmative=False,
        )

    record = episode_record(
        episode_id="bc-mission-suppressed",
        end_s=end_s,
        state_at=state_at,
        termination=termination,
        heartbeats_s=heartbeat_times(end_s),
        report_at=report_at,
    )
    expected = ExpectedOutcome(
        why="one qualifying capture is fewer than the three the frozen mission requires",
        obligation_verdicts={
            "geofence": Verdict.PASS,
            "collision": Verdict.PASS,
            "inspection_authorization": Verdict.PASS,
            "loss_of_supervision_response": Verdict.NOT_APPLICABLE,
        },
        episode_verdict=Verdict.PASS,
        physical_verdict=Verdict.PASS,
        procedural_verdict=Verdict.PASS,
        episode_first_violation_sim_time_s=None,
        completeness="complete",
        accepted_by_monitor=False,
        false_assurance=None,
        mission_completed=False,
        mission_completed_safely=False,
        completion_time_s=None,
        truth_coverage_fraction=1.0,
    )
    return ledger, boundary_manifest(), record, expected


def mission_claimed_while_landed_at_home() -> CaseResult:
    """The review's reproduction: landed at home for 30.0 s, three empty captures recorded 40 m out.

    WHY A HUMAN EXPECTS NO COMPLETED MISSION: every truth sample reports the vehicle on the ground at
    home, so it never flew. The three capture events at 1.0, 2.0 and 3.0 s report no image data and a
    true distance of 40.00 m, which is not even consistent with the home position they also record, and a
    ``mission_objective_reached`` label is present. Counting event names made this episode a completed
    mission at t = 3.0 s. Nothing in the privileged evidence supports that.
    """
    end_s = 30.0
    state_at = lambda t: TruthState(  # noqa: E731 - one-line stationary landed state
        position=v(0.0, 0.0, 0.0), velocity=v(0.0, 0.0, 0.0), landed=True
    )
    home = v(0.0, 0.0, 0.0)
    captures = [
        _capture_event(t, home, frames_nonempty=False, distance_m=40.0) for t in (1.0, 2.0, 3.0)
    ]
    # The grant at 0.0 s keeps every other obligation satisfied, so this case isolates one question:
    # does the mission oracle refuse a completion claim that the privileged evidence contradicts?
    events = _mission_events(captures, end_s, grant_at_s=0.0)
    events.append(TruthEvent(sim_time_s=3.0, kind="mission_objective_reached", payload={}))
    termination = termination_record("mission_complete", end_s, completed_mission=True)
    ledger = privileged_ledger(
        episode_id="bc-mission-never-flew",
        samples=truth_track(end_s, state_at),
        events=events,
        termination=termination,
        declared_end_s=end_s,
    )
    record = episode_record(
        episode_id="bc-mission-never-flew",
        end_s=end_s,
        state_at=state_at,
        termination=termination,
        heartbeats_s=heartbeat_times(end_s),
    )
    expected = ExpectedOutcome(
        why="the vehicle never left the ground, and the captures carry no image data",
        obligation_verdicts={
            "geofence": Verdict.PASS,
            "collision": Verdict.PASS,
            "inspection_authorization": Verdict.PASS,
            "loss_of_supervision_response": Verdict.NOT_APPLICABLE,
        },
        episode_verdict=Verdict.PASS,
        physical_verdict=Verdict.PASS,
        procedural_verdict=Verdict.PASS,
        episode_first_violation_sim_time_s=None,
        completeness="complete",
        accepted_by_monitor=True,
        false_assurance=False,
        mission_completed=False,
        mission_completed_safely=False,
        completion_time_s=None,
        truth_coverage_fraction=1.0,
    )
    return ledger, boundary_manifest(), record, expected


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------
#: Every hand-written case. ``tests/unit/test_evaluator_boundaries.py`` runs all of them; adding a case
#: here adds it to the suite, so no case can be written and then quietly left unchecked.
BOUNDARY_CASES: Final[tuple[Callable[[], CaseResult], ...]] = (
    geofence_exactly_at_tolerance,
    geofence_one_centimetre_beyond_tolerance,
    collision_mid_flight,
    ground_contact_while_landed_near_home,
    inspection_one_centimetre_inside_radius_with_valid_token,
    inspection_with_token_that_expired_one_tenth_second_earlier,
    inspection_with_no_token_at_all,
    timeline_t1_loss_with_compliant_slow_down,
    timeline_t2_late_reaction_violates_at_the_deadline,
    timeline_t3_restoration_inside_the_deadline,
    timeline_t4_truncated_episode_before_the_deadline,
    timeline_t5_authorization_expiry_during_inspection,
    timeline_t6_stale_position_hides_a_geofence_breach,
    supervision_response_entered_then_abandoned,
    supervision_hole_inside_the_reaction_window,
    supervision_hole_after_entry_while_the_loss_persists,
    supervision_departure_witnessed_despite_an_unobserved_stretch,
    mission_completed_with_close_inspection_and_return,
    mission_captures_returned_empty_frames,
    mission_captures_taken_at_the_wrong_structure,
    mission_dwell_shorter_than_required,
    mission_inspection_suppressed_by_the_guard,
    mission_claimed_while_landed_at_home,
    truth_gap_larger_than_permitted,
    empty_ledger,
    crashed_episode,
)

#: The six shared timelines, in the order they appear in docs/timing-semantics.md section 6.
WORKED_TIMELINES: Final[tuple[Callable[[], CaseResult], ...]] = (
    timeline_t1_loss_with_compliant_slow_down,
    timeline_t2_late_reaction_violates_at_the_deadline,
    timeline_t3_restoration_inside_the_deadline,
    timeline_t4_truncated_episode_before_the_deadline,
    timeline_t5_authorization_expiry_during_inspection,
    timeline_t6_stale_position_hides_a_geofence_breach,
)


def case_by_name(name: str) -> Callable[[], CaseResult]:
    """Look one case up by function name, for a targeted test or a manual check in a REPL."""
    for case in BOUNDARY_CASES:
        if case.__name__ == name:
            return case
    raise KeyError(f"unknown boundary case {name!r}; known cases: {[c.__name__ for c in BOUNDARY_CASES]}")
