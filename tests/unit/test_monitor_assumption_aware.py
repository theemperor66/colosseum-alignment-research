"""``assumption_aware_v1`` must be informative AND must refuse to affirm what it cannot decide.

WHY these tests exist
---------------------
The comparator is only interesting if it fails in the right direction. Two opposite mistakes would
both destroy the study:

1. **Always abstain.** A guard that answered UNKNOWN everywhere would score a perfect false-assurance
   rate while accepting nothing. The analysis reports assurance coverage next to every conditional
   rate for exactly this reason, and ``test_a_clean_nominal_step_is_an_affirmative_pass`` below proves
   the guard is not that: on clean evidence it says PASS, affirmatively.
2. **Silently pass on missing evidence.** That is the baseline's behaviour, and reproducing it here
   would collapse the comparison. Every check that cannot be decided must produce UNKNOWN, and UNKNOWN
   must never be affirmative.

The policy predicates and their numeric thresholds are shared with ``policy_only_v1``; the arms differ
in assumption handling and in nothing else. The supervision timelines therefore use the same T1, T2 and
T3 numbers from docs/timing-semantics.md section 6, and must produce the same obligation verdicts when
the evidence is clean. Where this guard differs, the difference is asserted with its reason.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from colosseum_assurance.interfaces import MissionBrief
from colosseum_assurance.monitors.assumption_aware import (
    CHECK_IDS,
    AssumptionAwareMonitor,
    AssumptionThresholds,
)
from colosseum_assurance.monitors.base import ASSET_POSITION_UNCERTAINTY_M
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
THRESHOLDS = AssumptionThresholds()
DT = PROTOCOL.mission.control_dt_s

# With a fresh sample the reachability envelope is cruise_speed * (age + dt + clock margin) = 2.25 m,
# and the radius test adds the declared 2.0 m asset survey uncertainty on top of it.
NOMINAL_ENVELOPE_M = PROTOCOL.mission.cruise_speed_mps * (0.0 + DT + THRESHOLDS.clock_margin_s)
RADIUS_SLACK_M = NOMINAL_ENVELOPE_M + ASSET_POSITION_UNCERTAINTY_M

FAR_FROM_ASSET = (12.0, 0.0, -6.0)     # 16 m away: the radius cannot be engaged even with the slack
INSIDE_RADIUS = (24.5, 0.0, -6.0)      # 3.5 m away: inside the 8 m radius even after the slack
OUTSIDE_GEOFENCE = (45.0, 0.0, -6.0)   # 2.5 m beyond the fence tolerance


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


def make_monitor(**brief_kwargs: float) -> AssumptionAwareMonitor:
    monitor = AssumptionAwareMonitor(PROTOCOL)
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



def granted_token(*, granted_at_s: float, now_s: float, validity_s: float | None = None) -> AuthorizationView:
    """A granted token that was received recently, so the record-age margin stays small."""
    validity = OBLIGATIONS.authorization_validity_s if validity_s is None else validity_s
    return AuthorizationView(
        token_id="token-1",
        status=AuthorizationStatus.GRANTED,
        requested_at_s=granted_at_s - 2.0,
        granted_at_s=granted_at_s,
        expires_at_s=granted_at_s + validity,
        received_at_s=min(granted_at_s, now_s),
        scope="inspection",
    )


# ----------------------------------------------------------------------------------------------
# It is not an always-abstain monitor
# ----------------------------------------------------------------------------------------------
def test_a_clean_nominal_step_is_an_affirmative_pass() -> None:
    """The headline anti-abstention property, pinned exactly as the module docstring promises.

    Fresh state, usable depth, live heartbeat, far from the asset and far inside the geofence: every
    assumption is decidable and every obligation is either satisfied or not engaged. The guard must
    therefore affirm, because the acceptance rule only counts affirmative passes.
    """
    monitor = make_monitor()
    report = monitor.evaluate(
        make_packet(0, 5.0, FAR_FROM_ASSET, declared_delay_s=0.0),
        make_command(0, 5.0, target=(13.5, 0.0, -6.0)),
    )
    assert report.verdict is Verdict.PASS
    assert report.affirmative is True
    assert report.monitor_id == "assumption_aware_v1"
    assert set(report.assumption_verdicts) == set(CHECK_IDS)
    assert report.assumption_verdicts["a_evidence_age"] is Verdict.PASS
    assert report.assumption_verdicts["b_sensing_validity"] is Verdict.PASS
    assert report.assumption_verdicts["c_capability_consistency"] is Verdict.PASS
    assert report.assumption_verdicts["e_supervision_liveness"] is Verdict.PASS
    assert report.obligation_verdicts["geofence"] is Verdict.PASS
    assert report.obligation_verdicts["collision"] is Verdict.PASS
    assert report.intervention == "none"
    assert f"{NOMINAL_ENVELOPE_M:.2f} m" in report.rationale


def test_a_clean_nominal_sequence_affirms_at_every_step() -> None:
    """One affirmative step could be luck; an accepted episode needs an affirmative pass at EVERY step."""
    monitor = make_monitor()
    reports = []
    for step_index in range(20):
        now_s = 5.0 + step_index * DT
        x = 2.0 + 0.5 * step_index  # outbound cruise, still far from both the asset and the fence
        # The commanded segment must stay clear of the radius plus the slack, or the authorization
        # obligation would engage and the step would legitimately become undecidable.
        assert (PROTOCOL.mission.asset_nominal_position.x - (x + 0.75)) - RADIUS_SLACK_M > (
            OBLIGATIONS.authorized_inspection_radius_m
        )
        reports.append(
            monitor.evaluate(
                make_packet(step_index, now_s, (x, 0.0, -6.0), velocity=(1.0, 0.0, 0.0),
                            declared_delay_s=0.0),
                make_command(step_index, now_s, target=(x + 0.75, 0.0, -6.0)),
            )
        )
    assert all(report.verdict is Verdict.PASS for report in reports)
    assert all(report.affirmative for report in reports)
    assert all(report.intervention == "none" for report in reports)


def test_a_dropped_depth_frame_with_a_wide_clearance_is_still_affirmed() -> None:
    """Bounds, not certainty: a 25 m clearance stays a clearance even after 2.25 m of possible motion.

    This is the rule that stops the guard abstaining whenever anything is missing. The opposite case,
    where the extrapolated clearance no longer excludes contact, must go to UNKNOWN.
    """
    wide = make_monitor()
    wide.evaluate(make_packet(0, 5.0, FAR_FROM_ASSET, depth_min_range_m=25.0, declared_delay_s=0.0),
                  make_command(0, 5.0))
    report = wide.evaluate(
        make_packet(1, 5.5, FAR_FROM_ASSET, depth=None, dropouts_in_window=1, declared_delay_s=0.0),
        make_command(1, 5.5),
    )
    assert report.verdict is Verdict.PASS
    assert report.affirmative is True
    assert report.assumption_verdicts["b_sensing_validity"] is Verdict.PASS

    narrow = make_monitor()
    narrow.evaluate(make_packet(0, 5.0, FAR_FROM_ASSET, depth_min_range_m=2.0, declared_delay_s=0.0),
                    make_command(0, 5.0))
    report = narrow.evaluate(
        make_packet(1, 5.5, FAR_FROM_ASSET, depth=None, dropouts_in_window=1, declared_delay_s=0.0),
        make_command(1, 5.5),
    )
    assert report.verdict is Verdict.UNKNOWN
    assert report.affirmative is False
    assert report.assumption_verdicts["b_sensing_validity"] is Verdict.UNKNOWN
    assert report.obligation_verdicts["collision"] is Verdict.UNKNOWN


# ----------------------------------------------------------------------------------------------
# The shared policy predicates: same verdicts as the baseline when the evidence is clean
# ----------------------------------------------------------------------------------------------
def test_a_geofence_breach_in_the_received_state_is_a_violation() -> None:
    """An exceedance that is already visible needs no envelope reasoning: it is simply a violation."""
    report = make_monitor().evaluate(
        make_packet(0, 6.0, OUTSIDE_GEOFENCE, declared_delay_s=0.0), make_command(0, 6.0)
    )
    assert report.obligation_verdicts["geofence"] is Verdict.VIOLATION
    assert report.verdict is Verdict.VIOLATION
    assert report.affirmative is False
    assert report.intervention == "return_to_launch"


def test_entering_the_inspection_radius_without_a_token_violates_authorization() -> None:
    """Certain engagement plus a decidably invalid token is a violation, not an abstention."""
    report = make_monitor().evaluate(
        make_packet(0, 7.0, INSIDE_RADIUS, declared_delay_s=0.0),
        make_command(0, 7.0, target=(25.0, 0.0, -6.0)),
    )
    assert report.obligation_verdicts["inspection_authorization"] is Verdict.VIOLATION
    assert report.intervention == "suspend_inspection"
    assert "certainly engaged" in report.rationale
    # The distance plus the slack must really be inside the radius, or the test would prove nothing.
    assert 3.5 + RADIUS_SLACK_M <= OBLIGATIONS.authorized_inspection_radius_m


def test_t5_a_token_stops_authorizing_exactly_at_its_expiry() -> None:
    """T5 again, from the same numbers: granted at 12.0, valid on [12.0, 57.0), violation at 57.0."""
    monitor = make_monitor()
    token = granted_token(granted_at_s=12.0, now_s=12.0)
    assert token.expires_at_s == pytest.approx(57.0)
    report = monitor.evaluate(
        make_packet(0, 57.0, INSIDE_RADIUS, authorization=token, declared_delay_s=0.0),
        make_command(0, 57.0),
    )
    assert report.obligation_verdicts["inspection_authorization"] is Verdict.VIOLATION
    assert report.intervention == "suspend_inspection"
    assert "expired" in report.rationale


def test_an_old_authorization_record_that_is_about_to_expire_is_undecidable_not_valid() -> None:
    """Time of use, not time of issue: a 44.5 s old record cannot prove 0.5 s of remaining validity.

    The clock margin grows with the age of the record, so a token that the baseline still reads as
    valid becomes undecidable here. That is the intended difference, and it is one of the interventions
    the analysis counts as the guard's cost.
    """
    monitor = make_monitor()
    token = granted_token(granted_at_s=12.0, now_s=12.0)  # received at 12.0, expires at 57.0
    report = monitor.evaluate(
        make_packet(0, 56.5, INSIDE_RADIUS, authorization=token, declared_delay_s=0.0),
        make_command(0, 56.5),
    )
    assert report.obligation_verdicts["inspection_authorization"] is Verdict.UNKNOWN
    assert report.assumption_verdicts["d_authorization_validity"] is Verdict.UNKNOWN
    assert report.verdict is Verdict.UNKNOWN
    assert report.affirmative is False
    assert report.intervention == "suspend_inspection"  # undecidable authorization stops the inspection
    assert "record-age margin" in report.rationale


def test_a_pending_authorization_inside_the_radius_is_undecidable() -> None:
    """A request with no answer yet is not a denial and not a grant; it is not decidable."""
    pending = AuthorizationView(
        token_id="token-1", status=AuthorizationStatus.PENDING, requested_at_s=6.0
    )
    report = make_monitor().evaluate(
        make_packet(0, 7.0, INSIDE_RADIUS, authorization=pending, declared_delay_s=0.0),
        make_command(0, 7.0),
    )
    assert report.obligation_verdicts["inspection_authorization"] is Verdict.UNKNOWN
    assert report.verdict is Verdict.UNKNOWN
    assert "still pending" in report.rationale


def test_the_collision_obligation_uses_the_same_depth_proximity_proxy_as_the_baseline() -> None:
    """Both arms score contact from the same proxy, so a difference here would confound the comparison."""
    report = make_monitor().evaluate(
        make_packet(0, 5.0, FAR_FROM_ASSET, depth_min_range_m=0.5, declared_delay_s=0.0),
        make_command(0, 5.0),
    )
    assert report.obligation_verdicts["collision"] is Verdict.VIOLATION
    assert report.intervention == "hold"
    assert "proximity proxy" in report.rationale


# ----------------------------------------------------------------------------------------------
# Assumption checks
# ----------------------------------------------------------------------------------------------
def test_evidence_older_than_the_declared_bound_breaks_the_evidence_age_assumption() -> None:
    """The declared delay bound is an assumption; when the evidence breaks it, nothing else can stand."""
    monitor = make_monitor(observation_delay_s=1.0)
    report = monitor.evaluate(
        make_packet(0, 8.0, FAR_FROM_ASSET, state_age_s=3.0, declared_delay_s=1.0),
        make_command(0, 8.0),
    )
    bound = 1.0 + 0.5 * DT
    assert report.assumption_verdicts["a_evidence_age"] is Verdict.VIOLATION
    assert report.verdict is Verdict.UNKNOWN  # a broken assumption is not an obligation violation
    assert report.affirmative is False
    assert f"exceeds the declared bound {bound:.2f} s" in report.rationale
    assert report.evidence_age_s == pytest.approx(3.0)


def test_a_sensing_dropout_burst_breaks_the_sensing_validity_assumption() -> None:
    """Beyond a bounded burst the vehicle has no sensing evidence to reason with, and must say so."""
    monitor = make_monitor()
    inside_burst = monitor.evaluate(
        make_packet(0, 5.0, FAR_FROM_ASSET, dropouts_in_window=THRESHOLDS.max_dropout_burst,
                    declared_delay_s=0.0),
        make_command(0, 5.0),
    )
    assert inside_burst.assumption_verdicts["b_sensing_validity"] is Verdict.PASS
    assert inside_burst.verdict is Verdict.PASS

    over_burst = monitor.evaluate(
        make_packet(1, 5.5, FAR_FROM_ASSET, dropouts_in_window=THRESHOLDS.max_dropout_burst + 1,
                    declared_delay_s=0.0),
        make_command(1, 5.5),
    )
    assert over_burst.assumption_verdicts["b_sensing_validity"] is Verdict.VIOLATION
    assert over_burst.verdict is Verdict.UNKNOWN
    assert over_burst.affirmative is False
    assert "dropout burst" in over_burst.rationale


def test_low_depth_coverage_breaks_the_sensing_validity_assumption() -> None:
    """A frame the perception layer calls valid can still be too sparse to bound obstacle proximity."""
    thin = clear_depth(5.0, min_range_m=25.0, coverage=0.10)
    report = make_monitor().evaluate(
        make_packet(0, 5.0, FAR_FROM_ASSET, depth=thin, declared_delay_s=0.0), make_command(0, 5.0)
    )
    assert report.assumption_verdicts["b_sensing_validity"] is Verdict.VIOLATION
    assert report.verdict is Verdict.UNKNOWN
    assert "coverage" in report.rationale


def test_impossible_received_motion_breaks_the_capability_assumption() -> None:
    """Two samples that imply a speed the vehicle cannot reach mean the state channel is not trustworthy."""
    monitor = make_monitor()
    monitor.evaluate(make_packet(0, 5.0, (12.0, 0.0, -6.0), declared_delay_s=0.0), make_command(0, 5.0))
    report = monitor.evaluate(
        make_packet(1, 5.5, (40.0, 0.0, -6.0), declared_delay_s=0.0), make_command(1, 5.5)
    )
    assert report.assumption_verdicts["c_capability_consistency"] is Verdict.VIOLATION
    assert report.verdict is Verdict.UNKNOWN
    assert "capability bound" in report.rationale


def test_a_state_that_never_arrived_leaves_every_position_obligation_undecidable() -> None:
    """With no position sample at all the guard must abstain, where the baseline affirms."""
    report = make_monitor().evaluate(
        make_packet(0, 0.0, FAR_FROM_ASSET, has_state=False, depth=None, declared_delay_s=0.0),
        make_command(0, 0.0, kind="hold", target=None),
    )
    assert report.verdict is Verdict.UNKNOWN
    assert report.affirmative is False
    assert report.obligation_verdicts["geofence"] is Verdict.UNKNOWN
    assert report.obligation_verdicts["collision"] is Verdict.UNKNOWN
    assert report.assumption_verdicts["a_evidence_age"] is Verdict.UNKNOWN
    assert report.assumption_verdicts["c_capability_consistency"] is Verdict.UNKNOWN


# ----------------------------------------------------------------------------------------------
# Supervision timelines T1, T2, T3 of docs/timing-semantics.md
# ----------------------------------------------------------------------------------------------
LAST_HEARTBEAT_S = 8.0
DETECTABLE_AT_S = LAST_HEARTBEAT_S + OBLIGATIONS.loss_of_supervision_threshold_s   # 14.0
DEADLINE_S = DETECTABLE_AT_S + OBLIGATIONS.loss_of_supervision_reaction_deadline_s  # 19.0


def run_supervision_timeline(
    monitor: AssumptionAwareMonitor,
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
    """T1 must produce the same obligation verdicts here as in the baseline: the predicate is shared.

    The vehicle is 16 m from the nominal asset, so "outside the inspection zone" is certain even after
    the envelope and the declared survey uncertainty are subtracted. That is what lets this guard
    CONFIRM the safe response instead of abstaining.
    """
    monitor = make_monitor()
    reports = run_supervision_timeline(
        monitor, heartbeat_at=heartbeats_lost_from_ten(), speed_at=lambda t: 0.4 if t >= 17.5 else 2.5
    )
    supervision = {t: r.obligation_verdicts["loss_of_supervision_response"] for t, r in reports.items()}
    assert supervision[14.0] is Verdict.NOT_APPLICABLE  # a gap of exactly 6.0 s is not yet a loss
    assert supervision[14.5] is Verdict.PASS
    assert supervision[17.5] is Verdict.PASS
    assert "safe response confirmed" in reports[17.5].rationale
    assert all(v is not Verdict.VIOLATION for v in supervision.values())
    assert all(v is not Verdict.UNKNOWN for v in supervision.values())


def test_t2_late_reaction_is_a_violation_at_the_first_step_past_the_reaction_deadline() -> None:
    """T2, asserted in both guards so the arms stay comparable.

    docs/timing-semantics.md gives the evaluator's first violation time as the deadline itself, 19.0.
    Both guards use the inclusive ``now >= deadline`` test, so both report the violation at 19.0 when
    the control grid lands on it. Asserting it in both files keeps the boundary identical across the
    arms; a change in one guard alone would break the comparison.
    """
    monitor = make_monitor()
    reports = run_supervision_timeline(
        monitor, heartbeat_at=heartbeats_lost_from_ten(), speed_at=lambda t: 2.5
    )
    supervision = {t: r.obligation_verdicts["loss_of_supervision_response"] for t, r in reports.items()}
    assert DEADLINE_S == pytest.approx(19.0)
    assert supervision[18.5] is Verdict.PASS
    assert supervision[DEADLINE_S] is Verdict.VIOLATION
    assert reports[DEADLINE_S].intervention == "return_to_launch"
    assert "without a confirmed safe response" in reports[DEADLINE_S].rationale
    first_violation = min(t for t, v in supervision.items() if v is Verdict.VIOLATION)
    assert first_violation == pytest.approx(DEADLINE_S)


def test_t3_restoration_before_the_deadline_closes_the_obligation() -> None:
    """T3: a heartbeat received at 16.0 ends the loss before the 19.0 s deadline, with no reaction."""
    monitor = make_monitor()

    def heartbeat_at(t: float) -> float:
        if 10.0 <= t < 16.0:
            return LAST_HEARTBEAT_S
        if t < 10.0:
            return t - (t % OBLIGATIONS.supervision_heartbeat_period_s)
        return 16.0 if t < 18.0 else t - (t % OBLIGATIONS.supervision_heartbeat_period_s)

    reports = run_supervision_timeline(monitor, heartbeat_at=heartbeat_at, speed_at=lambda t: 2.5)
    supervision = {t: r.obligation_verdicts["loss_of_supervision_response"] for t, r in reports.items()}
    assert supervision[15.5] is Verdict.PASS
    assert supervision[16.0] is Verdict.NOT_APPLICABLE
    assert supervision[19.5] is Verdict.NOT_APPLICABLE
    assert all(v is not Verdict.VIOLATION for v in supervision.values())


def test_a_supervision_view_that_is_not_current_makes_the_link_state_undecidable() -> None:
    """The difference from the baseline: this guard asks how old its own supervision view is.

    A view lagging by more than one heartbeat period could be hiding a heartbeat that has already
    arrived, so "lost" cannot be established from it. The baseline reads the same view as current and
    reports a comfortable NOT_APPLICABLE.
    """
    report = make_monitor().evaluate(
        make_packet(0, 20.0, FAR_FROM_ASSET, heartbeat_received_at_s=14.0,
                    supervision_view_time_s=17.0, declared_delay_s=0.0),
        make_command(0, 20.0, kind="hold", target=None),
    )
    lag = 20.0 - 17.0
    assert lag > OBLIGATIONS.supervision_heartbeat_period_s
    assert report.assumption_verdicts["e_supervision_liveness"] is Verdict.UNKNOWN
    assert report.obligation_verdicts["loss_of_supervision_response"] is Verdict.UNKNOWN
    assert report.verdict is Verdict.UNKNOWN
    assert "supervision view is not current" in report.rationale


# ----------------------------------------------------------------------------------------------
# Escalation
# ----------------------------------------------------------------------------------------------
def lagging_view_packet(step_index: int, now_s: float) -> ObservationPacket:
    """A packet whose ONLY defect is a stale supervision view, so the streak ladder can be isolated."""
    return make_packet(
        step_index,
        now_s,
        FAR_FROM_ASSET,
        heartbeat_received_at_s=now_s - 4.0,
        supervision_view_time_s=now_s - 3.0,
        authorization=granted_token(granted_at_s=now_s - 0.2, now_s=now_s),
        declared_delay_s=0.0,
    )


def test_a_run_of_unknown_verdicts_escalates_step_by_step() -> None:
    """Escalation must be graded: a single undecidable step may not ground the aircraft.

    The documented ladder is: nothing for ``unknown_grace_steps``, then hold, then suspend_inspection
    after ``unknown_hold_steps``, then return_to_launch after ``unknown_rtl_steps``. The interventions
    the analysis reports as the guard's cost come straight from this ladder.
    """
    monitor = make_monitor(supervision_delay_s=1.0)
    interventions: list[str] = []
    for step_index in range(26):
        now_s = 20.0 + step_index * DT
        report = monitor.evaluate(
            lagging_view_packet(step_index, now_s), make_command(step_index, now_s, kind="hold", target=None)
        )
        assert report.verdict is Verdict.UNKNOWN
        assert report.affirmative is False
        interventions.append(report.intervention)

    assert interventions[: THRESHOLDS.unknown_grace_steps] == ["none"] * THRESHOLDS.unknown_grace_steps
    assert interventions[THRESHOLDS.unknown_grace_steps] == "hold"
    assert interventions[THRESHOLDS.unknown_hold_steps - 1] == "hold"
    assert interventions[THRESHOLDS.unknown_hold_steps] == "suspend_inspection"
    assert interventions[THRESHOLDS.unknown_rtl_steps - 1] == "suspend_inspection"
    assert interventions[THRESHOLDS.unknown_rtl_steps] == "return_to_launch"
    assert interventions[-1] == "return_to_launch"


def test_escalation_starts_earlier_when_no_supervisor_could_answer_in_time() -> None:
    """If the declared supervisory delay exceeds the reaction deadline, waiting cannot help.

    The guard then climbs the ladder ``unknown_grace_steps`` earlier. This is the one place where the
    declared supervisory response delay of the condition changes guard behaviour, so it is asserted
    against the same step sequence as the feasible case.
    """
    severe = OBLIGATIONS.loss_of_supervision_reaction_deadline_s + 9.0  # the severe supervisory level
    monitor = make_monitor(supervision_delay_s=severe)
    interventions: list[str] = []
    for step_index in range(26):
        now_s = 20.0 + step_index * DT
        report = monitor.evaluate(
            lagging_view_packet(step_index, now_s), make_command(step_index, now_s, kind="hold", target=None)
        )
        interventions.append(report.intervention)

    assert interventions[0] == "hold"  # no grace at all: the first undecidable step already holds
    assert interventions[THRESHOLDS.unknown_hold_steps - THRESHOLDS.unknown_grace_steps] == (
        "suspend_inspection"
    )
    assert interventions[THRESHOLDS.unknown_rtl_steps - THRESHOLDS.unknown_grace_steps] == (
        "return_to_launch"
    )


def test_a_pass_resets_the_unknown_streak() -> None:
    """Escalation measures a persistent loss of decidability, not a total over the whole episode."""
    monitor = make_monitor()
    for step_index in range(5):
        now_s = 20.0 + step_index * DT
        monitor.evaluate(
            lagging_view_packet(step_index, now_s), make_command(step_index, now_s, kind="hold", target=None)
        )
    recovered = monitor.evaluate(
        make_packet(5, 22.5, FAR_FROM_ASSET, declared_delay_s=0.0), make_command(5, 22.5)
    )
    assert recovered.verdict is Verdict.PASS
    assert recovered.intervention == "none"

    again = monitor.evaluate(lagging_view_packet(6, 23.0), make_command(6, 23.0, kind="hold", target=None))
    assert again.verdict is Verdict.UNKNOWN
    assert again.intervention == "none"  # the ladder restarted from the bottom


def test_describe_declares_the_grounding_and_the_limits_of_the_claim() -> None:
    """The comparator documentation is generated from this, and it must not overclaim reproduction."""
    info = make_monitor().describe()
    assert info["monitor_id"] == "assumption_aware_v1"
    assert info["emits_unknown"] is True
    assert info["grounding"] == "inspired_implementation"
    assert list(info["assumption_checks"]) == list(CHECK_IDS)
    assert info["policy_version"] == OBLIGATIONS.policy_version
    assert set(info["obligation_ids"]) == set(OBLIGATIONS.obligation_ids)
    assert info["thresholds"]["asset_position_uncertainty_m"] == ASSET_POSITION_UNCERTAINTY_M
    joined = " ".join(info["limitations"])
    assert "not a reproduction" in joined
    assert "not truth" in joined
    assert len(info["grounding_sources"]) == 3


def test_evaluate_before_reset_is_an_error_not_a_verdict() -> None:
    """A guard with no brief has no geofence and no asset; it must refuse rather than guess."""
    monitor = AssumptionAwareMonitor(PROTOCOL)
    with pytest.raises(RuntimeError):
        monitor.evaluate(make_packet(0, 1.0, FAR_FROM_ASSET), make_command(0, 1.0))
