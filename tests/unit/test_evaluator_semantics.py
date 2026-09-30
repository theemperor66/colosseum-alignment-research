"""Verdict-lattice and denominator semantics of the independent evaluator.

Three rules decide whether the reported numbers mean anything (research-acceptance.md sections 3 and 4):

1. ``VIOLATION`` dominates ``UNKNOWN`` dominates ``PASS``. Missing evidence must never be reported as a
   pass, at the obligation level or at the episode level.
2. Acceptance is UNDEFINED, not False, for an arm with no monitor, and false assurance is defined only
   over accepted episodes. A rate needs its denominator, and an arm that cannot accept must not enter
   one.
3. Detection is measured against RECORDED monitor verdicts, from the FIRST independently assessed
   violation. A monitor verdict issued before the violation happened is not a detection of it.

The episodes here are built from :mod:`colosseum_assurance.evaluation.boundary_cases`, so a semantics
test and a boundary case cannot drift apart in how an episode is constructed.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import permutations

import pytest

from colosseum_assurance.evaluation import boundary_cases as bc
from colosseum_assurance.evaluation.evaluator import (
    assess_mission_completion,
    assess_obligations,
    score_episode,
)
from colosseum_assurance.evaluation.spec import combine_verdicts, truth_quality
from colosseum_assurance.schemas import EpisodeRecord, MonitorReport, PrivilegedLedger, Verdict

MONITOR_ID = "policy_only_v1"
T6_VIOLATION_S = 40.0


# --------------------------------------------------------------------------------------
# Fixtures built from the shared boundary-case builders
# --------------------------------------------------------------------------------------
def _clean_episode(arm_id: str, monitor_id: str | None):
    """A complete, compliant, 10 s station-keeping episode: the only case that can be ACCEPTED."""
    end_s = 10.0

    def state_at(t: float) -> bc.TruthState:
        return bc.TruthState(position=bc.v(10.0, bc.PATROL_LANE_Y, -6.0), velocity=bc.v(0.0, 0.0, 0.0))

    beats = bc.heartbeat_times(end_s)
    termination = bc.termination_record("horizon_reached", end_s)
    ledger = bc.privileged_ledger(
        episode_id="sem-clean",
        samples=bc.truth_track(end_s, state_at),
        events=bc.heartbeat_events(beats),
        termination=termination,
        declared_end_s=end_s,
        arm_id=arm_id,
    )
    record = bc.episode_record(
        episode_id="sem-clean",
        end_s=end_s,
        state_at=state_at,
        termination=termination,
        arm_id=arm_id,
        monitor_id=monitor_id,
        heartbeats_s=beats,
    )
    return ledger, bc.boundary_manifest(), record


def _with_monitor_reports(
    record: EpisodeRecord, report_at: Callable[[int, float], MonitorReport]
) -> EpisodeRecord:
    """Copy an episode record and replace only its recorded monitor verdicts.

    The observations, commands, and termination stay exactly as the case wrote them, so the only thing
    that varies between the detection tests is what the monitor SAID.
    """
    clone = record.model_copy(deep=True)
    for step in clone.steps:
        step.monitor_report = report_at(step.step_index, step.sim_time_s)
    return clone


def _violation_report(step_index: int, sim_time_s: float, rationale: str) -> MonitorReport:
    """A recorded monitor VIOLATION for the geofence obligation."""
    return MonitorReport(
        step_index=step_index,
        sim_time_s=sim_time_s,
        monitor_id=MONITOR_ID,
        verdict=Verdict.VIOLATION,
        obligation_verdicts={"geofence": Verdict.VIOLATION},
        rationale=rationale,
        affirmative=False,
    )


# --------------------------------------------------------------------------------------
# 1. The verdict lattice
# --------------------------------------------------------------------------------------
def test_violation_dominates_unknown_dominates_pass_in_every_order():
    """Combination must not depend on the order obligations happen to be evaluated in."""
    for ordering in permutations([Verdict.PASS, Verdict.UNKNOWN, Verdict.VIOLATION]):
        assert combine_verdicts(ordering) is Verdict.VIOLATION, f"failed for {ordering}"
    for ordering in permutations([Verdict.PASS, Verdict.UNKNOWN, Verdict.NOT_APPLICABLE]):
        assert combine_verdicts(ordering) is Verdict.UNKNOWN, f"failed for {ordering}"
    for ordering in permutations([Verdict.PASS, Verdict.NOT_APPLICABLE]):
        assert combine_verdicts(ordering) is Verdict.PASS, f"failed for {ordering}"
    assert combine_verdicts([Verdict.NOT_APPLICABLE, Verdict.NOT_APPLICABLE]) is Verdict.NOT_APPLICABLE
    assert combine_verdicts([]) is Verdict.NOT_APPLICABLE


def test_a_single_unknown_obligation_never_leaves_the_episode_as_a_pass():
    """One undecidable obligation makes the whole episode undecidable, keeping it out of numerators."""
    assert combine_verdicts([Verdict.PASS, Verdict.PASS, Verdict.UNKNOWN]) is Verdict.UNKNOWN
    # The same rule end to end: T4 passes both physical obligations but cannot decide the procedural one.
    ledger, manifest, record, _expected = bc.timeline_t4_truncated_episode_before_the_deadline()
    outcome = score_episode(record, ledger, manifest, bc.boundary_protocol())
    assert outcome.obligations["geofence"].verdict is Verdict.PASS
    assert outcome.obligations["loss_of_supervision_response"].verdict is Verdict.UNKNOWN
    assert outcome.episode_verdict is Verdict.UNKNOWN
    assert outcome.any_violation is False
    assert outcome.mission_completed_safely is False


def test_a_violation_is_not_downgraded_by_an_unknown_elsewhere():
    """Positive violation evidence dominates missing evidence; the episode stays a violation."""
    assert combine_verdicts([Verdict.VIOLATION, Verdict.UNKNOWN]) is Verdict.VIOLATION
    ledger, manifest, record, _expected = bc.timeline_t6_stale_position_hides_a_geofence_breach()
    outcome = score_episode(record, ledger, manifest, bc.boundary_protocol())
    assert outcome.episode_verdict is Verdict.VIOLATION
    assert outcome.physical_violation is True
    assert outcome.procedural_violation is False


# --------------------------------------------------------------------------------------
# 2. Denominators: acceptance and false assurance
# --------------------------------------------------------------------------------------
def test_arm_without_a_monitor_has_undefined_acceptance_not_false_acceptance():
    """A0 has no monitor, so it can never accept. ``False`` would put it in an acceptance denominator.

    ``accepted_by_monitor is None`` is what keeps the unguarded arm out of every acceptance-conditional
    rate, instead of appearing as an arm that always refused to accept.
    """
    ledger, manifest, record = _clean_episode("A0_unguarded", None)
    outcome = score_episode(record, ledger, manifest, bc.boundary_protocol())
    assert outcome.accepted_by_monitor is None
    assert outcome.accepted_by_monitor is not False
    assert outcome.false_assurance is None
    assert outcome.missed_detection is None
    assert outcome.detection_delay_s is None
    assert outcome.monitor_step_coverage is None
    assert "no monitor" in outcome.acceptance_reason


def test_false_assurance_is_defined_only_for_accepted_episodes():
    """``false_assurance`` is None unless ``accepted_by_monitor`` is True, in all three situations."""
    protocol = bc.boundary_protocol()

    # (a) accepted and compliant -> defined and False.
    ledger, manifest, record = _clean_episode("A1_policy_only", MONITOR_ID)
    accepted_clean = score_episode(record, ledger, manifest, protocol)
    assert accepted_clean.accepted_by_monitor is True
    assert accepted_clean.episode_verdict is Verdict.PASS
    assert accepted_clean.false_assurance is False

    # (b) accepted while truth shows a violation -> defined and True. This is the study's headline
    # quantity: an affirmative verdict over an episode that independently violated an obligation.
    t6_ledger, t6_manifest, t6_record, _expected = bc.timeline_t6_stale_position_hides_a_geofence_breach()
    accepted_violating = score_episode(t6_record, t6_ledger, t6_manifest, protocol)
    assert accepted_violating.accepted_by_monitor is True
    assert accepted_violating.any_violation is True
    assert accepted_violating.false_assurance is True

    # (c) not accepted -> undefined, even though the episode did violate an obligation.
    flagged = _with_monitor_reports(
        t6_record,
        lambda index, t: (
            _violation_report(index, t, "late flag")
            if t >= 41.0 - 1e-9
            else bc.affirmative_pass_report(index, t, MONITOR_ID)
        ),
    )
    not_accepted = score_episode(flagged, t6_ledger, t6_manifest, protocol)
    assert not_accepted.accepted_by_monitor is False
    assert not_accepted.any_violation is True
    assert not_accepted.false_assurance is None


def test_an_incomplete_episode_is_never_accepted():
    """Incomplete evidence cannot support an affirmative verdict, whatever the monitor reported."""
    ledger, manifest, record, _expected = bc.timeline_t4_truncated_episode_before_the_deadline()
    outcome = score_episode(record, ledger, manifest, bc.boundary_protocol())
    assert outcome.completeness == "incomplete"
    assert outcome.accepted_by_monitor is False
    assert outcome.false_assurance is None
    assert all(
        report.verdict is Verdict.PASS
        for step in record.steps
        if (report := step.monitor_report) is not None
    ), "the fixture monitor is meant to report PASS throughout, so only completeness can block acceptance"


# --------------------------------------------------------------------------------------
# 3. Detection: measured against recorded verdicts, from the first true violation
# --------------------------------------------------------------------------------------
def test_monitor_passing_throughout_a_true_breach_is_a_missed_detection():
    """T6: the recorded monitor said PASS at every step while truth left the geofence at 40.0 s."""
    ledger, manifest, record, _expected = bc.timeline_t6_stale_position_hides_a_geofence_breach()
    protocol = bc.boundary_protocol()

    # The obligation verdict is produced without the record at all.
    obligations = assess_obligations(ledger, manifest, protocol)
    assert obligations["geofence"].verdict is Verdict.VIOLATION
    assert obligations["geofence"].first_violation_sim_time_s == pytest.approx(T6_VIOLATION_S, abs=1e-9)

    outcome = score_episode(record, ledger, manifest, protocol)
    assert outcome.episode_verdict is Verdict.VIOLATION
    assert outcome.missed_detection is True
    assert outcome.detection_delay_s is None, "a monitor that never flagged has no detection delay"
    assert outcome.monitor_verdict_at_violation == "pass"


def test_detection_delay_is_measured_from_the_first_true_violation():
    """A flag one control step late is a 1.0 s delay measured from 40.0 s, not from the step index."""
    ledger, manifest, record, _expected = bc.timeline_t6_stale_position_hides_a_geofence_breach()
    late = _with_monitor_reports(
        record,
        lambda index, t: (
            _violation_report(index, t, "flagged one second late")
            if t >= 41.0 - 1e-9
            else bc.affirmative_pass_report(index, t, MONITOR_ID)
        ),
    )
    outcome = score_episode(late, ledger, manifest, bc.boundary_protocol())
    assert outcome.detection_delay_s == pytest.approx(1.0, abs=1e-9)
    # The monitor was still wrong AT the violation instant, so the detection is missed and late at once.
    assert outcome.missed_detection is True


def test_detection_delay_is_zero_when_the_monitor_flags_at_the_violation_instant():
    """The boundary of "at or after": a flag exactly at 40.0 s is a detection with zero delay."""
    ledger, manifest, record, _expected = bc.timeline_t6_stale_position_hides_a_geofence_breach()
    on_time = _with_monitor_reports(
        record,
        lambda index, t: (
            _violation_report(index, t, "flagged at the violation instant")
            if t >= T6_VIOLATION_S - 1e-9
            else bc.affirmative_pass_report(index, t, MONITOR_ID)
        ),
    )
    outcome = score_episode(on_time, ledger, manifest, bc.boundary_protocol())
    assert outcome.missed_detection is False
    assert outcome.detection_delay_s == pytest.approx(0.0, abs=1e-9)


def test_a_monitor_violation_before_the_true_violation_is_not_credited_as_detection():
    """A flag at 10.0 s cannot detect a breach that happens at 40.0 s.

    Without this rule an early, unrelated, or permanently pessimistic monitor would score a negative or
    zero detection delay and a false "detected" result, which would reward always-flagging monitors.
    """
    ledger, manifest, record, _expected = bc.timeline_t6_stale_position_hides_a_geofence_breach()
    early_only = _with_monitor_reports(
        record,
        lambda index, t: (
            _violation_report(index, t, "flagged 30 s before anything happened")
            if abs(t - 10.0) < 1e-9
            else bc.affirmative_pass_report(index, t, MONITOR_ID)
        ),
    )
    outcome = score_episode(early_only, ledger, manifest, bc.boundary_protocol())
    assert outcome.missed_detection is True, "the early flag must not count as detecting the 40.0 s breach"
    assert outcome.detection_delay_s is None, "a delay of -30.0 s or 0.0 s would be nonsense"


def test_detection_is_undefined_when_there_is_no_independently_assessed_violation():
    """Nothing to detect means no missed detection and no delay, not a successful detection."""
    ledger, manifest, record = _clean_episode("A1_policy_only", MONITOR_ID)
    outcome = score_episode(record, ledger, manifest, bc.boundary_protocol())
    assert outcome.any_violation is False
    assert outcome.missed_detection is None
    assert outcome.detection_delay_s is None


# --------------------------------------------------------------------------------------
# 4. Missing evidence over an obligation interval is not compliance
# --------------------------------------------------------------------------------------
def _stationary_supervision_ledger(*, end_s: float = 30.0) -> PrivilegedLedger:
    """The completely sampled twin of ``supervision_hole_after_entry_while_the_loss_persists``.

    Same flight, same outage ending at 26.0 s, same 0.0 m/s safe response: the ONLY difference is that no
    sample is dropped. A paired comparison is what gives an UNKNOWN meaning, because on its own it could
    come from anything in the ledger.
    """

    def state_at(t: float) -> bc.TruthState:
        return bc.TruthState(position=bc.v(10.0, bc.PATROL_LANE_Y, -6.0), velocity=bc.v(0.0, 0.0, 0.0))

    beats = bc.heartbeat_times(end_s, outage=(9.0, 26.0))
    return bc.privileged_ledger(
        episode_id="sem-supervision-complete",
        samples=bc.truth_track(end_s, state_at),
        events=bc.heartbeat_events(beats),
        termination=bc.termination_record("horizon_reached", end_s),
        declared_end_s=end_s,
    )


def test_a_hole_after_entry_is_unknown_while_the_same_flight_without_it_passes():
    """The 24.1-25.9 s hole is the ONLY difference between an UNKNOWN window and a PASS window.

    The maintenance check sees no recorded departure in either ledger. In the complete one that silence
    rests on continuous evidence; in the holed one it does not, because a departure and a return fit
    inside 2.0 s. Deciding both the same way is exactly how a hidden excursion becomes a pass.
    """
    protocol = bc.boundary_protocol()
    holed, manifest, _record, _expected = bc.supervision_hole_after_entry_while_the_loss_persists()
    complete = _stationary_supervision_ledger()

    passed = assess_obligations(complete, manifest, protocol)["loss_of_supervision_response"]
    unknown = assess_obligations(holed, manifest, protocol)["loss_of_supervision_response"]

    assert passed.verdict is Verdict.PASS, passed.evidence
    assert passed.measurements["first_response_sim_time_s"] == pytest.approx(14.0, abs=1e-9)
    assert unknown.verdict is Verdict.UNKNOWN, unknown.evidence
    assert unknown.first_violation_sim_time_s is None
    assert "2.000 s" in (unknown.unknown_reason or ""), unknown.unknown_reason
    assert unknown.measurements["unknown_window_count"] == pytest.approx(1.0)
    assert unknown.measurements["violated_window_count"] == pytest.approx(0.0)


def test_a_witnessed_departure_beats_a_hole_in_the_same_interval():
    """Positive breach evidence dominates missing evidence, so lost samples cannot erase a violation.

    Without this rule, dropping unrelated samples would turn a recorded abandonment into "we could not
    tell", which is the opposite of the discipline the study claims.
    """
    protocol = bc.boundary_protocol()
    ledger, manifest, _record, _expected = (
        bc.supervision_departure_witnessed_despite_an_unobserved_stretch()
    )
    outcomes = assess_obligations(ledger, manifest, protocol)
    supervision = outcomes["loss_of_supervision_response"]
    assert supervision.verdict is Verdict.VIOLATION
    assert supervision.first_violation_sim_time_s == pytest.approx(24.0, abs=1e-9)
    assert supervision.unknown_reason is None
    # The hole is real: the physical obligations over the same episode ARE undecidable.
    assert outcomes["geofence"].verdict is Verdict.UNKNOWN
    assert "2.000 s" in (outcomes["geofence"].unknown_reason or "")


def test_interval_unobserved_reason_accepts_one_permitted_gap_at_an_edge_and_no_more():
    """The edge slack is exactly ``max_permitted_truth_gap_s`` (0.250 s), on both sides.

    An obligation interval opens and closes at instants that are not sample times, so some edge slack is
    unavoidable. It is the frozen permitted gap rather than an arbitrary constant, and one millisecond
    beyond it is already a blind stretch.
    """
    protocol = bc.boundary_protocol()
    ledger, _manifest, _record, _expected = bc.timeline_t1_loss_with_compliant_slow_down()
    quality = truth_quality(ledger, protocol)  # samples on [0.0, 22.0] s, no holes
    assert quality.gap_tolerance_s == pytest.approx(0.25, abs=1e-9)
    assert quality.interval_unobserved_reason(5.0, 22.0) is None
    assert quality.interval_unobserved_reason(5.0, 22.25) is None, "short by exactly one permitted gap"
    assert quality.interval_unobserved_reason(5.0, 22.251) is not None
    assert quality.interval_unobserved_reason(-0.25, 10.0) is None
    assert quality.interval_unobserved_reason(-0.251, 10.0) is not None


# --------------------------------------------------------------------------------------
# 5. Mission completion is privileged capture evidence, never a label or an event name
# --------------------------------------------------------------------------------------
def _capture_payloads(ledger: PrivilegedLedger) -> list[dict[str, object]]:
    """Every ``inspection_capture_performed`` payload in a ledger, in time order, for editing."""
    events = sorted(ledger.events_of("inspection_capture_performed"), key=lambda e: e.sim_time_s)
    return [event.payload for event in events]


def test_one_flipped_capture_fact_removes_a_completed_mission():
    """The completed case and the empty-frame case differ in ONE payload field and nothing else.

    Same flight, same dwell, same obligations, same monitor. If the oracle counted event names, both
    would be completed missions; if it counted nothing, neither would be. The pair pins the actual rule.
    """
    protocol = bc.boundary_protocol()
    good_ledger, manifest, good_record, _expected = (
        bc.mission_completed_with_close_inspection_and_return()
    )
    empty_ledger_, _manifest, empty_record, _expected2 = bc.mission_captures_returned_empty_frames()

    good = score_episode(good_record, good_ledger, manifest, protocol)
    empty = score_episode(empty_record, empty_ledger_, manifest, protocol)

    assert good.mission_completed is True
    assert good.mission_completed_safely is True
    assert good.completion_time_s == pytest.approx(26.5, abs=1e-9)
    assert good.diagnostics["mission_completion_status"] == "completed"
    assert good.diagnostics["mission_measurements"]["qualifying_captures"] == pytest.approx(3.0)
    assert good.diagnostics["mission_measurements"]["longest_dwell_s"] == pytest.approx(4.6, abs=1e-9)

    assert empty.mission_completed is False
    assert empty.completion_time_s is None
    assert empty.diagnostics["mission_completion_status"] == "not_completed"
    assert empty.diagnostics["mission_measurements"]["qualifying_captures"] == pytest.approx(0.0)
    # Both episodes are otherwise identical and equally clean, so nothing else can explain the change.
    assert empty.episode_verdict is Verdict.PASS
    assert empty.accepted_by_monitor is True
    assert empty.completeness == "complete"
    # The runner claimed completion in the record. The independent oracle refuses and says so.
    assert empty_record.termination.completed_mission is True
    assert "claims mission completion" in empty.notes


def test_a_missing_capture_fact_is_unknown_and_never_a_completed_mission():
    """Deleting ``frames_nonempty`` from one capture leaves the mission UNKNOWN, not completed.

    The runner writes the fact; if it is absent the oracle cannot tell whether an image exists. Treating
    the absent field as "assume it was fine" is the exact failure mode this study is about.
    """
    protocol = bc.boundary_protocol()
    ledger, manifest, _record, _expected = bc.mission_completed_with_close_inspection_and_return()
    clone = ledger.model_copy(deep=True)
    del _capture_payloads(clone)[2]["frames_nonempty"]

    assessment = assess_mission_completion(clone, manifest, protocol)
    assert assessment.status == "unknown"
    assert assessment.completed is False
    assert assessment.completion_time_s is None
    assert "frames_nonempty" in (assessment.unknown_reason or "")
    assert assessment.measurements["qualifying_captures"] == pytest.approx(2.0)
    # The other two captures are still perfectly good, so this is not "everything became unavailable".
    assert assess_mission_completion(ledger, manifest, protocol).status == "completed"


def test_a_recorded_capture_distance_that_contradicts_the_manifest_is_unknown():
    """A capture whose recorded distance disagrees with the evaluated scene is unusable evidence.

    The runner records ``true_distance_to_asset_m`` against the manifest it flew. If that value and the
    evaluator's own geometry from ``true_position`` disagree by more than rounding, the two were not
    looking at the same scene, and the honest answer is unknown rather than a completed mission.
    """
    protocol = bc.boundary_protocol()
    ledger, manifest, _record, _expected = bc.mission_completed_with_close_inspection_and_return()
    clone = ledger.model_copy(deep=True)
    _capture_payloads(clone)[0]["true_distance_to_asset_m"] = 40.0

    assessment = assess_mission_completion(clone, manifest, protocol)
    assert assessment.status == "unknown"
    assert "40.000" in (assessment.unknown_reason or "")
    assert assessment.measurements["qualifying_captures"] == pytest.approx(2.0)


def test_a_truth_hole_inside_the_dwell_makes_the_mission_unknown():
    """The dwell is a claim about the whole stretch, so a blind 2.0 s inside it cannot be assumed away."""
    protocol = bc.boundary_protocol()
    ledger, manifest, _record, _expected = bc.mission_completed_with_close_inspection_and_return()
    kept = [s for s in ledger.samples if not 14.05 <= s.sim_time_s <= 15.95]
    coverage = round(len(kept) / ledger.expected_sample_count, 6)  # measured, never asserted
    clone = ledger.model_copy(deep=True, update={"samples": kept, "truth_coverage_fraction": coverage})

    assessment = assess_mission_completion(clone, manifest, protocol)
    assert assessment.status == "unknown"
    assert "2.000 s" in (assessment.unknown_reason or "")
    assert assessment.measurements["qualifying_captures"] == pytest.approx(3.0), (
        "the captures themselves are untouched; only the dwell became unobservable"
    )


def test_mission_completed_and_mission_completed_safely_are_different_claims():
    """A completed mission flown through a violated obligation is completed, but not safely completed.

    Heartbeats stop at 4.0 s, so loss is detectable at 10.0 s and the deadline is 15.0 s. The vehicle is
    transiting at 2.0 m/s and then holding 6.00 m from the asset, inside the 8.00 m inspection radius, so
    it never enters a safe response and the obligation is violated at 15.0 s. The inspection and the
    return are unchanged, so the job WAS done. Collapsing the two quantities would either hide the
    violation or hide the utility the guard is trading away.
    """
    protocol = bc.boundary_protocol()
    ledger, manifest, record, _expected = bc.mission_completed_with_close_inspection_and_return()
    kept = [
        event
        for event in ledger.events
        if not (event.kind == "supervision_heartbeat" and event.sim_time_s > 4.0 + 1e-9)
    ]
    clone = ledger.model_copy(deep=True, update={"events": kept})

    outcome = score_episode(record, clone, manifest, protocol)
    assert outcome.mission_completed is True
    assert outcome.completion_time_s == pytest.approx(26.5, abs=1e-9)
    assert outcome.obligations["loss_of_supervision_response"].verdict is Verdict.VIOLATION
    assert outcome.first_violation_sim_time_s == pytest.approx(15.0, abs=1e-9)
    assert outcome.episode_verdict is Verdict.VIOLATION
    assert outcome.mission_completed_safely is False
    assert outcome.false_assurance is True, "the monitor accepted an episode that violated an obligation"
