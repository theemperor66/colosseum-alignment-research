"""``policy_only_v1`` is the diagnostic baseline, so its LIMITATION is a measured quantity.

WHY these tests exist
---------------------
This guard exists to answer "what does it cost when a runtime guard simply believes what it is told?".
Two things therefore have to be pinned, not one:

* the policy predicates themselves, which it shares with the assumption-aware guard. Sharing them is a
  precondition of the comparison, but it does NOT reduce the two arms to one differing check: the
  configurations also differ in uncertainty propagation, authorization-boundary handling, stale-view
  handling and unknown-triggered escalation. The paired difference is attributable to the whole
  configuration, not to a single mechanism (see docs/comparator-grounding.md);
* the documented limitation - it NEVER emits UNKNOWN, it holds its previous verdict when data is
  missing, and it reads every received timestamp at face value.

If the limitation were quietly fixed, the A1 arm would stop being the baseline the research plan
describes and the headline comparison would measure something else. So the tests below assert the
limitation as a contract, with the reason written next to each one.

The supervision timelines use the numbers of docs/timing-semantics.md section 6 (T1, T2, T3) and the
authorization timeline uses T5, so monitor and evaluator are checked against one set of numbers.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from colosseum_assurance.interfaces import MissionBrief
from colosseum_assurance.monitors.policy_only import PolicyOnlyMonitor
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.schemas import (
    AuthorizationStatus,
    AuthorizationView,
    ControlCommand,
    DepthSummary,
    MonitorReport,
    ObservationPacket,
    SensorHealth,
    SupervisionView,
    Vec3,
    VehicleState,
    Verdict,
)

PROTOCOL = ProtocolConfig()
OBLIGATIONS = PROTOCOL.obligations
DT = PROTOCOL.mission.control_dt_s

# 16 m from the nominal asset and far inside the geofence: neither the inspection radius nor the fence
# is engaged, so a test can isolate one obligation at a time.
FAR_FROM_ASSET = (12.0, 0.0, -6.0)
# 3.5 m from the nominal asset, i.e. certainly inside the 8 m authorized inspection radius.
INSIDE_RADIUS = (24.5, 0.0, -6.0)
# 2.5 m beyond the geofence tolerance boundary (x_max 42.0 plus 0.5 m tolerance).
OUTSIDE_GEOFENCE = (45.0, 0.0, -6.0)


def make_brief(*, observation_delay_s: float = 0.0, supervision_delay_s: float = 1.0) -> MissionBrief:
    """A brief carrying the NOMINAL asset position, which is all a guard is allowed to know."""
    return MissionBrief(
        mission=PROTOCOL.mission,
        obligations=OBLIGATIONS,
        home=PROTOCOL.mission.home,
        nominal_asset_position=PROTOCOL.mission.asset_nominal_position,
        declared_observation_delay_s=observation_delay_s,
        declared_supervision_delay_s=supervision_delay_s,
        policy_version=OBLIGATIONS.policy_version,
    )


def make_monitor(**brief_kwargs: float) -> PolicyOnlyMonitor:
    monitor = PolicyOnlyMonitor(PROTOCOL)
    monitor.reset(make_brief(**brief_kwargs))  # type: ignore[arg-type]
    return monitor


# ----------------------------------------------------------------------------------------------
# Packet construction
#
# ``schemas.py`` now validates timing: ``state_age_s`` must equal receive - acquisition, the
# ``SensorHealth`` availability flags must match the attached payloads, and
# ``SupervisionView.heartbeat_age_s`` must equal ``sim_time_s - last_heartbeat_received_at_s``. Hand
# written packets are therefore easy to get wrong, so one helper derives every dependent field from the
# few numbers a test actually cares about. A test that needs an inconsistent packet must build it
# explicitly, which makes the inconsistency visible in the diff.
# ----------------------------------------------------------------------------------------------
def clear_depth(sim_time_s: float, *, min_range_m: float | None = 25.0, coverage: float = 1.0,
                valid: bool = True, degraded_reason: str | None = None) -> DepthSummary:
    """A usable depth frame with a single obstacle distance, or an explicitly unusable one."""
    return DepthSummary(
        sim_time_s=sim_time_s,
        camera_name="front_center",
        valid=valid,
        min_range_m=min_range_m,
        free_path_m=min_range_m,
        sector_min_range_m=[] if min_range_m is None else [min_range_m] * 7,
        obstacle_bearing_rad=None if min_range_m is None else 0.0,
        coverage_fraction=coverage,
        degraded_reason=degraded_reason,
    )


def make_packet(
    step_index: int,
    now_s: float,
    position: tuple[float, float, float],
    *,
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    yaw_rad: float = 0.0,
    state_age_s: float = 0.0,
    has_state: bool = True,
    depth: DepthSummary | None | str = "clear",
    depth_min_range_m: float | None = 25.0,
    depth_coverage: float = 1.0,
    heartbeat_received_at_s: float | None = None,
    supervision_view_time_s: float | None = None,
    link_state: str = "nominal",
    authorization: AuthorizationView | None = None,
    dropouts_in_window: int = 0,
    declared_delay_s: float | None = None,
    depth_age_s: float | None = None,
) -> ObservationPacket:
    """Build one internally consistent :class:`ObservationPacket`."""
    acquired_at = now_s - state_age_s
    state = None
    if has_state:
        state = VehicleState(
            sim_time_s=acquired_at,
            position=Vec3(x=position[0], y=position[1], z=position[2]),
            velocity=Vec3(x=velocity[0], y=velocity[1], z=velocity[2]),
            yaw_rad=yaw_rad,
        )
    summary = (
        clear_depth(acquired_at, min_range_m=depth_min_range_m, coverage=depth_coverage)
        if depth == "clear"
        else depth
    )
    view_time = now_s if supervision_view_time_s is None else supervision_view_time_s
    heartbeat_at = view_time if heartbeat_received_at_s is None else heartbeat_received_at_s
    supervision = SupervisionView(
        sim_time_s=view_time,
        last_heartbeat_sim_time_s=heartbeat_at,
        last_heartbeat_received_at_s=heartbeat_at,
        heartbeat_age_s=view_time - heartbeat_at,
        link_state=link_state,  # type: ignore[arg-type]
    )
    return ObservationPacket(
        step_index=step_index,
        receive_sim_time_s=now_s,
        state=state,
        state_age_s=state_age_s if has_state else None,
        depth=summary,  # type: ignore[arg-type]
        rgb=None,
        supervision=supervision,
        authorization=authorization or AuthorizationView(),
        sensor_health=SensorHealth(
            state_sample_available=state is not None,
            depth_available=summary is not None,
            rgb_available=False,
            state_age_s=state_age_s if has_state else None,
            depth_age_s=depth_age_s,
            dropouts_in_window=dropouts_in_window,
        ),
        declared_observation_delay_s=declared_delay_s,
    )


def make_command(
    step_index: int,
    now_s: float,
    *,
    kind: str = "move_to",
    target: tuple[float, float, float] | None = None,
    speed_mps: float = 1.5,
) -> ControlCommand:
    """The command the guard is asked to judge alongside the observation."""
    return ControlCommand(
        step_index=step_index,
        issued_sim_time_s=now_s,
        kind=kind,  # type: ignore[arg-type]
        target=None if target is None else Vec3(x=target[0], y=target[1], z=target[2]),
        speed_mps=speed_mps,
        duration_s=DT,
        reason="unit test",
    )



def granted_token(*, granted_at_s: float, validity_s: float | None = None) -> AuthorizationView:
    """A token as the vehicle holds it: granted, with a half-open validity window."""
    validity = OBLIGATIONS.authorization_validity_s if validity_s is None else validity_s
    return AuthorizationView(
        token_id="token-1",
        status=AuthorizationStatus.GRANTED,
        requested_at_s=granted_at_s - 2.0,
        granted_at_s=granted_at_s,
        expires_at_s=granted_at_s + validity,
        received_at_s=granted_at_s,
        scope="inspection",
    )


# ----------------------------------------------------------------------------------------------
# The helper itself
# ----------------------------------------------------------------------------------------------
def test_the_packet_helper_produces_schema_consistent_timing() -> None:
    """Every later assertion depends on these derived fields, so check them once, explicitly."""
    packet = make_packet(3, 10.0, FAR_FROM_ASSET, state_age_s=2.5, heartbeat_received_at_s=6.0)
    assert packet.state is not None
    assert packet.state.sim_time_s == pytest.approx(7.5)
    assert packet.state_age_s == pytest.approx(2.5)
    assert packet.sensor_health.state_age_s == pytest.approx(2.5)
    assert packet.sensor_health.state_sample_available is True
    assert packet.sensor_health.depth_available is True
    assert packet.supervision.heartbeat_age_s == pytest.approx(4.0)

    blind = make_packet(4, 10.5, FAR_FROM_ASSET, has_state=False, depth=None)
    assert blind.state is None and blind.state_age_s is None
    assert blind.sensor_health.state_sample_available is False
    assert blind.sensor_health.depth_available is False


# ----------------------------------------------------------------------------------------------
# The shared policy predicates
# ----------------------------------------------------------------------------------------------
def test_a_clean_nominal_step_is_an_affirmative_pass() -> None:
    """The acceptance rule counts affirmative passes, so a nominal step must actually produce one."""
    monitor = make_monitor()
    report = monitor.evaluate(
        make_packet(0, 5.0, FAR_FROM_ASSET, declared_delay_s=0.0),
        make_command(0, 5.0, target=(13.5, 0.0, -6.0)),
    )
    assert report.verdict is Verdict.PASS
    assert report.affirmative is True
    assert report.monitor_id == "policy_only_v1"
    assert report.obligation_verdicts["geofence"] is Verdict.PASS
    assert report.obligation_verdicts["collision"] is Verdict.PASS
    # Neither procedural obligation is engaged this far from the asset with a live link.
    assert report.obligation_verdicts["inspection_authorization"] is Verdict.NOT_APPLICABLE
    assert report.obligation_verdicts["loss_of_supervision_response"] is Verdict.NOT_APPLICABLE
    assert report.intervention == "none"
    assert report.evidence_age_s == pytest.approx(0.0)


def test_a_geofence_breach_in_the_received_state_is_a_violation() -> None:
    """The one case this guard does catch immediately: the breach is visible in the data it holds."""
    monitor = make_monitor()
    report = monitor.evaluate(
        make_packet(1, 6.0, OUTSIDE_GEOFENCE, declared_delay_s=0.0), make_command(1, 6.0)
    )
    assert report.verdict is Verdict.VIOLATION
    assert report.affirmative is False
    assert report.obligation_verdicts["geofence"] is Verdict.VIOLATION
    assert report.intervention == "return_to_launch"
    assert "outside the geofence" in report.rationale


def test_entering_the_inspection_radius_without_a_token_violates_authorization() -> None:
    """Performing the inspection step unauthorized is the procedural failure the study counts."""
    monitor = make_monitor()
    report = monitor.evaluate(
        make_packet(2, 7.0, INSIDE_RADIUS, declared_delay_s=0.0),
        make_command(2, 7.0, target=(25.0, 0.0, -6.0)),
    )
    assert report.obligation_verdicts["inspection_authorization"] is Verdict.VIOLATION
    assert report.verdict is Verdict.VIOLATION
    assert report.intervention == "suspend_inspection"
    assert "no granted authorization" in report.rationale


def test_a_command_that_would_enter_the_radius_engages_the_obligation_before_arrival() -> None:
    """A guard that only reacted after entry could not prevent the entry it is supposed to prevent."""
    monitor = make_monitor()
    report = monitor.evaluate(
        make_packet(3, 8.0, FAR_FROM_ASSET, declared_delay_s=0.0),
        make_command(3, 8.0, target=(28.0, 0.0, -6.0)),  # the command drives into the radius
    )
    assert report.obligation_verdicts["inspection_authorization"] is Verdict.VIOLATION
    assert "commanded 0.00 m" in report.rationale


def test_t5_a_token_stops_authorizing_exactly_at_its_expiry() -> None:
    """docs/timing-semantics.md T5: granted at 12.0, valid on [12.0, 57.0); the violation is at 57.0.

    Validity is half-open, so the vehicle is authorized at 56.5 and unauthorized at 57.0. The guard
    reads the record's own ``expires_at_s`` against the vehicle clock, which is exactly what T5 asks.
    """
    token = granted_token(granted_at_s=12.0)  # expires at 57.0 with the frozen 45 s validity
    assert token.expires_at_s == pytest.approx(57.0)

    monitor = make_monitor()
    still_valid = monitor.evaluate(
        make_packet(0, 56.5, INSIDE_RADIUS, authorization=token, declared_delay_s=0.0),
        make_command(0, 56.5),
    )
    assert still_valid.obligation_verdicts["inspection_authorization"] is Verdict.PASS
    assert still_valid.verdict is Verdict.PASS

    expired = monitor.evaluate(
        make_packet(1, 57.0, INSIDE_RADIUS, authorization=token, declared_delay_s=0.0),
        make_command(1, 57.0),
    )
    assert expired.obligation_verdicts["inspection_authorization"] is Verdict.VIOLATION
    assert expired.intervention == "suspend_inspection"
    assert "expired" in expired.rationale


def test_the_collision_obligation_is_a_depth_proximity_proxy() -> None:
    """No onboard guard has contact truth; the proxy must be scored and labelled as a proxy."""
    monitor = make_monitor()
    close = monitor.evaluate(
        make_packet(0, 5.0, FAR_FROM_ASSET, depth_min_range_m=0.5, declared_delay_s=0.0),
        make_command(0, 5.0),
    )
    assert close.obligation_verdicts["collision"] is Verdict.VIOLATION
    assert close.intervention == "hold"
    assert "proximity proxy" in close.rationale
    assert "depth proximity proxy" in monitor.describe()["limitations"][-1]


# ----------------------------------------------------------------------------------------------
# The documented limitation
# ----------------------------------------------------------------------------------------------
def test_stale_evidence_is_believed_because_this_guard_ignores_evidence_age() -> None:
    """The defining limitation, stated as a contract: 2.5 s of age changes nothing in the verdict.

    The same position is reported fresh and 2.5 s old. Both produce an affirmative pass. This is what
    makes a delayed breach invisible to this arm, which is the effect the study measures.
    """
    monitor = make_monitor(observation_delay_s=2.5)
    fresh = monitor.evaluate(
        make_packet(0, 5.0, FAR_FROM_ASSET, declared_delay_s=2.5), make_command(0, 5.0)
    )
    stale = monitor.evaluate(
        make_packet(1, 7.5, FAR_FROM_ASSET, state_age_s=2.5, declared_delay_s=2.5), make_command(1, 7.5)
    )
    assert fresh.verdict is stale.verdict is Verdict.PASS
    assert fresh.affirmative is stale.affirmative is True
    assert stale.evidence_age_s == pytest.approx(2.5)  # the age is recorded, and then ignored
    assert monitor.describe()["timestamp_handling"] == "received_values_treated_as_current"


def test_a_sensing_dropout_burst_is_invisible_to_this_guard() -> None:
    """It has no sensing-validity check, so a burst of lost frames does not change its verdict."""
    monitor = make_monitor()
    quiet = monitor.evaluate(
        make_packet(0, 5.0, FAR_FROM_ASSET, declared_delay_s=0.0), make_command(0, 5.0)
    )
    burst = monitor.evaluate(
        make_packet(1, 5.5, FAR_FROM_ASSET, dropouts_in_window=9, declared_delay_s=0.0),
        make_command(1, 5.5),
    )
    assert quiet.verdict is burst.verdict is Verdict.PASS
    assert burst.affirmative is True
    assert monitor.describe()["assumption_checks"] == []


def test_missing_data_holds_the_previous_verdict_instead_of_reporting_unknown() -> None:
    """"Hold the previous verdict" is the mechanism by which missing evidence becomes assurance."""
    monitor = make_monitor()
    breach = monitor.evaluate(
        make_packet(0, 5.0, OUTSIDE_GEOFENCE, declared_delay_s=0.0), make_command(0, 5.0)
    )
    assert breach.obligation_verdicts["geofence"] is Verdict.VIOLATION

    blind = monitor.evaluate(
        make_packet(1, 5.5, OUTSIDE_GEOFENCE, has_state=False, depth=None, declared_delay_s=0.0),
        make_command(1, 5.5),
    )
    assert blind.obligation_verdicts["geofence"] is Verdict.VIOLATION  # held, not re-established
    assert "previous verdicts held" in blind.rationale
    assert monitor.describe()["missing_data_behaviour"] == "hold_previous_verdict"


def test_an_episode_that_starts_blind_is_affirmed_having_seen_nothing() -> None:
    """The silent conversion of missing evidence into a pass, implemented on purpose and measured here."""
    monitor = make_monitor()
    report = monitor.evaluate(
        make_packet(0, 0.0, FAR_FROM_ASSET, has_state=False, depth=None, declared_delay_s=0.0),
        make_command(0, 0.0, kind="hold", target=None),
    )
    assert report.verdict is Verdict.PASS
    assert report.affirmative is True  # affirmed with no state and no depth ever received
    assert all(v is not Verdict.UNKNOWN for v in report.obligation_verdicts.values())


def test_it_never_emits_unknown_over_a_long_degraded_sequence() -> None:
    """The contract ``emits_unknown is False`` must hold for every step, not only for clean ones."""
    monitor = make_monitor(observation_delay_s=1.0)
    verdicts: set[Verdict] = set()
    for step_index in range(24):
        now_s = 5.0 + step_index * DT
        if step_index % 3 == 0:
            packet = make_packet(
                step_index, now_s, FAR_FROM_ASSET, has_state=False, depth=None,
                dropouts_in_window=step_index, declared_delay_s=1.0,
            )
        elif step_index % 3 == 1:
            packet = make_packet(
                step_index, now_s, OUTSIDE_GEOFENCE, state_age_s=3.0, declared_delay_s=1.0
            )
        else:
            packet = make_packet(
                step_index, now_s, FAR_FROM_ASSET,
                depth=clear_depth(now_s, min_range_m=None, valid=False, degraded_reason="dropout"),
                declared_delay_s=1.0,
            )
        report = monitor.evaluate(packet, make_command(step_index, now_s))
        verdicts.add(report.verdict)
        assert all(v is not Verdict.UNKNOWN for v in report.obligation_verdicts.values())
        assert report.assumption_verdicts == {}
    assert verdicts <= {Verdict.PASS, Verdict.VIOLATION}
    assert monitor.describe()["emits_unknown"] is False


# ----------------------------------------------------------------------------------------------
# Supervision timelines T1, T2, T3 of docs/timing-semantics.md
# ----------------------------------------------------------------------------------------------
LAST_HEARTBEAT_S = 8.0
DETECTABLE_AT_S = LAST_HEARTBEAT_S + 6.0   # t_det = 14.0 with the frozen 6.0 s threshold
DEADLINE_S = DETECTABLE_AT_S + 5.0         # t_dl = 19.0 with the frozen 5.0 s reaction deadline


def run_supervision_timeline(
    monitor: PolicyOnlyMonitor,
    *,
    heartbeat_at: Callable[[float], float],
    speed_at: Callable[[float], float],
    first_s: float = 8.0,
    last_s: float = 22.0,
) -> dict[float, MonitorReport]:
    """Step a supervision timeline on the control grid and keep every report, keyed by time."""
    reports: dict[float, MonitorReport] = {}
    step_index = 0
    now_s = first_s
    while now_s <= last_s + 1e-9:
        speed = speed_at(now_s)
        packet = make_packet(
            step_index, now_s, FAR_FROM_ASSET, velocity=(speed, 0.0, 0.0),
            heartbeat_received_at_s=heartbeat_at(now_s), declared_delay_s=0.0,
        )
        reports[round(now_s, 3)] = monitor.evaluate(
            packet, make_command(step_index, now_s, kind="hold", target=None, speed_mps=speed)
        )
        step_index += 1
        now_s = round(now_s + DT, 3)
    return reports


def heartbeats_lost_from_ten() -> Callable[[float], float]:
    """Outage [10.0, 30.0): the newest heartbeat received stays 8.0 for the whole window."""
    return lambda t: LAST_HEARTBEAT_S if t >= 10.0 else t - (t % OBLIGATIONS.supervision_heartbeat_period_s)


def test_t1_loss_of_supervision_with_a_compliant_slow_down_is_a_pass() -> None:
    """T1: last heartbeat 8.0, loss holds for t > 14.0, safe response entered at 17.5, deadline 19.0."""
    monitor = make_monitor()
    reports = run_supervision_timeline(
        monitor, heartbeat_at=heartbeats_lost_from_ten(), speed_at=lambda t: 0.4 if t >= 17.5 else 2.5
    )
    supervision = {t: r.obligation_verdicts["loss_of_supervision_response"] for t, r in reports.items()}

    # Loss is strictly greater than the threshold: at t = 14.0 the gap is exactly 6.0 s and is not a loss.
    assert supervision[14.0] is Verdict.NOT_APPLICABLE
    assert supervision[14.5] is Verdict.PASS
    assert supervision[17.5] is Verdict.PASS  # the safe response is entered here
    assert "safe response held" in reports[17.5].rationale
    assert all(v is not Verdict.VIOLATION for v in supervision.values())
    assert all(r.verdict is not Verdict.VIOLATION for r in reports.values())


def test_t2_late_reaction_is_a_violation_at_the_first_step_past_the_reaction_deadline() -> None:
    """T2: the vehicle never slows, so the deadline passes unmet.

    docs/timing-semantics.md gives the evaluator's first violation time as 19.0, the deadline itself.
    The guard uses an inclusive ``now >= deadline`` test, so when the control grid lands exactly on the
    deadline the guard and the evaluator agree on the instant. A strict ``>`` here would make every
    guard one control step late by construction and would inflate the measured detection delay.
    """
    monitor = make_monitor()
    reports = run_supervision_timeline(
        monitor, heartbeat_at=heartbeats_lost_from_ten(), speed_at=lambda t: 2.5
    )
    supervision = {t: r.obligation_verdicts["loss_of_supervision_response"] for t, r in reports.items()}

    assert DEADLINE_S == pytest.approx(19.0)
    assert supervision[14.0] is Verdict.NOT_APPLICABLE
    assert supervision[14.5] is Verdict.PASS  # inside the reaction window, not yet late
    assert supervision[18.5] is Verdict.PASS  # the last step before the deadline
    assert supervision[DEADLINE_S] is Verdict.VIOLATION  # the deadline instant itself
    assert reports[DEADLINE_S].verdict is Verdict.VIOLATION
    assert reports[DEADLINE_S].intervention == "return_to_launch"
    assert "reaction deadline 19.0 s passed" in reports[DEADLINE_S].rationale
    first_violation = min(t for t, v in supervision.items() if v is Verdict.VIOLATION)
    assert first_violation == pytest.approx(DEADLINE_S), (
        "guard and evaluator must agree on the deadline instant, so detection delay is not inflated"
    )


def test_t3_restoration_before_the_deadline_closes_the_obligation() -> None:
    """T3: outage [10.0, 16.0); a heartbeat received at 16.0 ends the loss with no reaction required."""
    monitor = make_monitor()

    def heartbeat_at(t: float) -> float:
        if 10.0 <= t < 16.0:
            return LAST_HEARTBEAT_S
        if t < 10.0:
            return t - (t % OBLIGATIONS.supervision_heartbeat_period_s)
        return 16.0 if t < 18.0 else t - (t % OBLIGATIONS.supervision_heartbeat_period_s)

    reports = run_supervision_timeline(monitor, heartbeat_at=heartbeat_at, speed_at=lambda t: 2.5)
    supervision = {t: r.obligation_verdicts["loss_of_supervision_response"] for t, r in reports.items()}

    assert supervision[15.5] is Verdict.PASS          # loss held, still inside the reaction window
    assert supervision[16.0] is Verdict.NOT_APPLICABLE  # restored before the 19.0 s deadline
    assert supervision[19.5] is Verdict.NOT_APPLICABLE  # and the deadline never applies
    assert all(v is not Verdict.VIOLATION for v in supervision.values())
    assert reports[16.0].intervention == "none"


# ----------------------------------------------------------------------------------------------
# Interventions
# ----------------------------------------------------------------------------------------------
def test_the_required_response_is_commanded_before_the_deadline_turns_it_into_a_violation() -> None:
    """A guard that only reported the violation afterwards could not prevent anything."""
    monitor = make_monitor()
    reports = run_supervision_timeline(
        monitor, heartbeat_at=heartbeats_lost_from_ten(), speed_at=lambda t: 2.5, last_s=19.0
    )
    assert reports[14.0].intervention == "none"        # not yet a loss
    assert reports[14.5].verdict is Verdict.PASS       # still compliant
    assert reports[14.5].intervention == "return_to_launch"  # and already commanded home
    assert reports[DEADLINE_S].intervention == "return_to_launch"


def test_intervention_escalation_puts_physical_and_supervisory_failures_above_procedural_ones() -> None:
    """Escalation rank decides what the runner executes, so the ordering is a contract."""
    monitor = make_monitor()
    both = monitor.evaluate(
        make_packet(0, 6.0, OUTSIDE_GEOFENCE, declared_delay_s=0.0),
        make_command(0, 6.0, target=(28.0, 0.0, -6.0)),  # unauthorized entry AND a fence breach
    )
    assert both.obligation_verdicts["geofence"] is Verdict.VIOLATION
    assert both.obligation_verdicts["inspection_authorization"] is Verdict.VIOLATION
    assert both.intervention == "return_to_launch"  # the higher rank wins

    only_procedural = make_monitor().evaluate(
        make_packet(1, 6.0, INSIDE_RADIUS, declared_delay_s=0.0), make_command(1, 6.0)
    )
    assert only_procedural.intervention == "suspend_inspection"

    only_proximity = make_monitor().evaluate(
        make_packet(2, 6.0, FAR_FROM_ASSET, depth_min_range_m=0.5, declared_delay_s=0.0),
        make_command(2, 6.0),
    )
    assert only_proximity.intervention == "hold"


def test_describe_states_the_limitations_the_analysis_must_report() -> None:
    """The comparator documentation is generated from ``describe()``, so it must stay truthful."""
    info = make_monitor().describe()
    assert info["monitor_id"] == "policy_only_v1"
    assert info["monitor_class"] == "policy_predicates_on_received_state"
    assert info["emits_unknown"] is False
    assert info["policy_version"] == OBLIGATIONS.policy_version
    assert set(info["obligation_ids"]) == set(OBLIGATIONS.obligation_ids)
    assert info["asset_position_reference"] == "mission_brief_nominal_asset_position"
    joined = " ".join(info["limitations"])
    assert "never reports unknown" in joined
    assert "ignores the age" in joined


def test_evaluate_before_reset_is_an_error_not_a_verdict() -> None:
    """A guard with no brief has no geofence and no asset; it must refuse rather than guess."""
    monitor = PolicyOnlyMonitor(PROTOCOL)
    with pytest.raises(RuntimeError):
        monitor.evaluate(make_packet(0, 1.0, FAR_FROM_ASSET), make_command(0, 1.0))
