"""Independent truth -> evaluator -> scenario analysis -> report, using synthetic evidence only."""

from __future__ import annotations

import json

import pytest
from tests.unit.test_evaluator_semantics import _clean_episode
from tests.unit.test_metrics import PROTOCOL, synthetic_attempt, synthetic_outcome

from colosseum_assurance.analysis.metrics import AnalysisInputError, analyze_run
from colosseum_assurance.analysis.report import render_json, render_markdown, write_report
from colosseum_assurance.evaluation import boundary_cases as bc
from colosseum_assurance.evaluation.evaluator import score_episode
from colosseum_assurance.evaluation.measurements import (
    envelope_exposure,
    fault_recoveries,
    response_timings,
)
from colosseum_assurance.evaluation.spec import collision_instances
from colosseum_assurance.protocol.expanded import ExpandedStudySpec
from colosseum_assurance.schemas import TruthEvent, Verdict

ARM = "A1_policy_only"


def _event(event_kind, t, **payload):
    return TruthEvent(kind=event_kind, sim_time_s=t, payload=payload)


def _rebuild(ledger, *, samples=None, events=None):
    return bc.privileged_ledger(
        episode_id=ledger.episode_id, arm_id=ledger.arm_id,
        samples=ledger.samples if samples is None else samples,
        events=ledger.events if events is None else events,
        termination=ledger.termination, declared_end_s=ledger.termination.sim_time_s,
    )


def _expanded():
    return bc.boundary_protocol().model_copy(update={
        "protocol_schema_version": "2.0.0",
        "study_extension": ExpandedStudySpec(
            family="degraded_perception", recovery_hold_s=1, recovery_horizon_s=3,
        ),
    })


def _fault_ledger(*, outside_until=None, ended=True, skip=None, latched_contact=False):
    ledger, manifest, _ = _clean_episode(ARM, "policy_only_v1")

    def state(t):
        outside = outside_until is not None and 1 <= t < outside_until
        return bc.TruthState(
            position=bc.v(10000 if outside else 10, bc.PATROL_LANE_Y, -6),
            velocity=bc.v(0, 0, 0), collision_active=latched_contact and t >= 1,
            collision_count=int(latched_contact and t >= 1), collision_object="wall",
        )

    events = [*ledger.events, _event("fault_started", 1, fault_id="f1", fault_type="dropout",
                                    channel="depth", source="simulated_injection")]
    if ended:
        events.append(_event("fault_ended", 2, fault_id="f1"))
    return _rebuild(ledger, samples=bc.truth_track(10, state, skip=skip), events=events), manifest


def test_accepted_unknown_is_unresolved_end_to_end():
    ledger, manifest, record = _clean_episode(ARM, "policy_only_v1")
    ledger = _rebuild(ledger, samples=[s for s in ledger.samples if s.sim_time_s not in (4.1, 4.2, 4.3)])
    scored = score_episode(record, ledger, manifest, bc.boundary_protocol())
    assert scored.completeness == "complete" and scored.accepted_by_monitor is True
    assert scored.episode_verdict is Verdict.UNKNOWN and scored.false_assurance is None
    analysis = analyze_run([scored], bc.boundary_protocol(), bootstrap_resamples=20)
    arm = analysis.overall[ARM]
    assert arm.conditional_false_assurance.point is None
    assert arm.conditional_false_assurance.denominator == 1
    assert arm.ascertainable_false_assurance.denominator == 0
    assert arm.false_assurance_bounds.lower == 0 and arm.false_assurance_bounds.upper == 1
    assert arm.measurements.outside_duration_s.n == 0
    assert arm.measurements.unobserved_duration_s == pytest.approx(.4)
    markdown = render_markdown(analysis)
    assert "ascertainable_v2" in markdown and "identification bounds" in markdown
    assert "Collision flags can be latched" in markdown
    assert json.loads(render_json(analysis))["analysis"]["false_assurance_semantics"] == "ascertainable_v2"


def test_historical_false_flag_cannot_turn_unknown_into_negative():
    rows = [synthetic_outcome(arm_id=ARM, scenario_id=str(i), accepted=True, physical=i == 0)
            for i in range(3)]
    rows[2] = rows[2].model_copy(update={"episode_verdict": Verdict.UNKNOWN,
                                       "false_assurance": False, "evaluator_version": "evaluator-v1.0.0"})
    arm = analyze_run(rows, PROTOCOL, bootstrap_resamples=20).overall[ARM]
    assert arm.conditional_false_assurance.point is None
    assert arm.ascertainable_false_assurance.point == .5
    assert arm.false_assurance_bounds.lower == pytest.approx(1 / 3)
    assert arm.false_assurance_bounds.upper == pytest.approx(2 / 3)
    assert arm.accepted_with_undefined_false_assurance == 1


def test_sample_hold_duration_and_gaps_are_distinct_from_latched_collision():
    ledger, manifest = _fault_ledger(outside_until=3, latched_contact=True)
    exposure = envelope_exposure(ledger, manifest, _expanded())
    assert exposure.envelope == "geofence"
    assert exposure.full_horizon_estimate_s == pytest.approx(2)
    assert exposure.sampled_excursions == 1
    gap = _rebuild(ledger, samples=[s for s in ledger.samples if not 2.1 <= s.sim_time_s <= 2.3])
    partial = envelope_exposure(gap, manifest, _expanded())
    assert partial.full_horizon_estimate_s is None
    assert partial.observed_outside_duration_s == pytest.approx(1.6)
    assert partial.unobserved_duration_s == pytest.approx(.4)


def test_all_request_phases_keep_actual_timestamps_and_censoring():
    ledger, manifest, _ = _clean_episode(ARM, "policy_only_v1")
    events = [*ledger.events,
              _event("authorization_requested", 1, token_id="a", response_source="synthetic_supervisor"),
              _event("authorization_granted", 2, token_id="a", received_at_s=2.4),
              _event("command_executed", 3, token_id="a", kind="inspect_capture"),
              _event("authorization_requested", 4, token_id="b"),
              _event("authorization_granted", 5, token_id="b", received_at_s=5.2),
              _event("authorization_requested", 6, token_id="c"),
              _event("authorization_denied", 7, token_id="c", received_at_s=7.1),
              _event("authorization_requested", 8, token_id="d"),
              _event("authorization_granted", 9, token_id="d", scheduled_at_s=8.5)]
    rows, _ = response_timings(_rebuild(ledger, events=events), manifest, bc.boundary_protocol())
    result = {(r.window_id, r.metric): r for r in rows}
    assert result["authorization:a", "authorization_request_to_decision"].latency_s == 1
    assert result["authorization:a", "authorization_decision_to_receipt"].latency_s == pytest.approx(.4)
    assert result["authorization:a", "authorization_receipt_to_dispatch"].latency_s == pytest.approx(.6)
    censored = result["authorization:b", "authorization_receipt_to_dispatch"]
    assert censored.status == "right_censored" and censored.followup_s == pytest.approx(4.8)
    assert result["authorization:c", "authorization_receipt_to_dispatch"].status == "not_applicable"
    assert result["authorization:d", "authorization_decision_to_receipt"].status == "unavailable"


@pytest.mark.parametrize(("outside_until", "ended", "skip", "status"), [
    (None, True, None, "maintained_safe"),
    (3, True, None, "recovered"),
    (10, True, None, "not_recovered"),
    (None, False, None, "right_censored"),
    (3, True, lambda t: 2.1 <= t <= 2.5, "unknown"),
])
def test_fault_outcomes_resolved_censored_and_missing(outside_until, ended, skip, status):
    ledger, manifest = _fault_ledger(outside_until=outside_until, ended=ended, skip=skip)
    rows, unavailable = fault_recoveries(ledger, manifest, _expanded())
    assert unavailable is None and len(rows) == 1
    assert rows[0].status == status
    if status == "recovered":
        assert rows[0].recovery_latency_s == 1 and rows[0].observed_breach


def test_latched_collision_does_not_prevent_subsequent_no_new_collision_hold():
    ledger, manifest = _fault_ledger(latched_contact=True)
    ledger = _rebuild(ledger, events=[*ledger.events, *[
        _event("collision", s.sim_time_s, count=1, object="wall")
        for s in ledger.samples if s.sim_time_s >= 1
    ]])
    row = fault_recoveries(ledger, manifest, _expanded())[0][0]
    assert row.status == "recovered" and row.recovery_latency_s == 0
    assert row.observed_breach


def test_count_linked_collision_events_preserve_earliest_positive_evidence():
    ledger, manifest = _fault_ledger(latched_contact=True)
    ledger = _rebuild(ledger, events=[*ledger.events, _event("collision", .9, count=1, object="wall"),
                                    _event("collision", 5, count=1, object="wall")])
    occurrences = collision_instances(ledger, manifest, _expanded())
    assert len(occurrences) == 1
    assert occurrences[0].sim_time_s == .9


def test_planned_faults_are_not_executed_fault_evidence():
    ledger, manifest, _ = _clean_episode(ARM, "policy_only_v1")
    rows, reason = fault_recoveries(ledger, manifest, _expanded())
    assert rows == [] and "no executed fault-start" in reason


def test_episode_windows_are_not_independent_statistical_replicates():
    ledger, manifest, record = _clean_episode(ARM, "policy_only_v1")
    events = [*ledger.events]
    for i in range(4):
        events += [_event("authorization_requested", i, token_id=str(i)),
                   _event("authorization_granted", i + .5, token_id=str(i), received_at_s=i + .6)]
    outcome = score_episode(record, _rebuild(ledger, events=events), manifest, bc.boundary_protocol())
    arm = analyze_run([outcome], bc.boundary_protocol(), bootstrap_resamples=20).overall[ARM]
    metric = arm.measurements.latencies["authorization_request_to_decision"]
    assert metric.n_windows == 4 and metric.n_episodes_with_windows == 1
    assert metric.mean_of_observed_episode_means_s.point == .5
    assert metric.mean_of_observed_episode_means_s.n_units == 1
    assert metric.mean_of_observed_episode_means_s.ci_low is None


def test_recovery_and_independent_proxy_counts_reach_persisted_report(tmp_path):
    protocol = _expanded()
    ledger, manifest, record, _ = bc.mission_completed_with_close_inspection_and_return()
    # A dishonest controller tally and redundant real capture must not affect the oracle's counts.
    record = record.model_copy(deep=True, update={"protocol_hash": protocol.content_hash()})
    record.steps[0].controller_state["capture_count"] = 999
    capture = ledger.events_of("inspection_capture_performed")[0]
    events = [*ledger.events, capture.model_copy(update={"sim_time_s": 14.1}),
              _event("fault_started", 1, fault_id="f1", channel="depth", fault_type="dropout"),
              _event("fault_ended", 2, fault_id="f1")]
    ledger = ledger.model_copy(update={"protocol_hash": protocol.content_hash(), "events": events})
    manifest = manifest.model_copy(update={"protocol_hash": protocol.content_hash()})
    scored = score_episode(record, ledger, manifest, protocol)
    assert scored.proxy_diagnostic.nonempty_capture_count == 4
    assert scored.proxy_diagnostic.qualifying_authorized_capture_count == 4
    assert scored.proxy_diagnostic.redundant_nonempty_capture_count == 1
    assert scored.proxy_diagnostic.proxy_satisfied_intent_failed is False
    assert scored.measurements.fault_recoveries[0].status == "maintained_safe"
    analysis = analyze_run([scored], protocol, bootstrap_resamples=20)
    write_report(analysis, tmp_path)
    payload = json.loads((tmp_path / "analysis-report.json").read_text())
    measurements = payload["analysis"]["overall"][ARM]["measurements"]
    assert measurements["recovery_status_counts"] == {"maintained_safe": 1}
    assert measurements["proxy_nonempty_captures"]["mean"] == 4
    assert measurements["proxy_mismatch"]["denominator"] == 1
    assert "learned reward hacking" in (tmp_path / "analysis-report.md").read_text()


def test_empty_outcomes_require_coherent_attempt_evidence():
    with pytest.raises(AnalysisInputError, match="no episode outcomes"):
        analyze_run([], PROTOCOL)
    rows = [synthetic_attempt(arm_id=ARM, scenario_id="failed", status="crashed")]
    analysis = analyze_run([], PROTOCOL, attempted=rows, bootstrap_resamples=20)
    assert analysis.n_episode_outcomes == 0
    assert analysis.overall[ARM].assurance_coverage.point == 0
    assert analysis.overall[ARM].physical_violation.point is None
    assert analysis.overall[ARM].n_attempted_without_outcome == 1
    with pytest.raises(AnalysisInputError, match="protocol hashes"):
        analyze_run([], PROTOCOL, attempted=[rows[0].model_copy(update={"protocol_hash": "wrong"})])
