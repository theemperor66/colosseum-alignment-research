"""Metric tests with hand-built outcome lists that pin every denominator.

The synthetic builders here exist only to exercise the software. They are never experimental evidence,
which is why every one of them is named ``synthetic_*`` and every record carries ``run_class="fixture"``.
"""

from __future__ import annotations

import json

import pytest

from colosseum_assurance.analysis.metrics import (
    REASON_ATTEMPTED_LEDGER_SHORTER,
    REASON_ATTEMPTED_MEMBERSHIP_UNKNOWN,
    REASON_NO_ACCEPTED,
    REASON_NO_MONITOR,
    REASON_NO_VIOLATIONS,
    AnalysisInputError,
    analyze_run,
    compute_run_analysis,
    load_outcomes,
)
from colosseum_assurance.analysis.stats import mcnemar_exact
from colosseum_assurance.evaluation.outcomes import EpisodeOutcome
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.schemas import AttemptedRun, Verdict

PROTOCOL = ProtocolConfig()
PROTOCOL_HASH = PROTOCOL.content_hash()
CELLS = [c["cell_id"] for c in PROTOCOL.cells()]


def synthetic_outcome(
    *,
    arm_id: str,
    scenario_id: str,
    cell_id: str = CELLS[0],
    physical: bool = False,
    procedural: bool = False,
    complete: bool = True,
    accepted: bool | None = None,
    false_assurance: bool | None = None,
    missed_detection: bool | None = None,
    detection_delay_s: float | None = None,
    completion_time_s: float | None = 62.0,
    interventions: int = 0,
    suspended: bool = False,
    abandoned: bool = False,
    protocol_hash: str = PROTOCOL_HASH,
    run_class: str = "fixture",
) -> EpisodeOutcome:
    """Build one evaluator outcome by hand so the expected denominators are obvious in the test."""
    violated = physical or procedural
    obs_level, sup_level = cell_id.split("__")
    return EpisodeOutcome(
        episode_id=f"{scenario_id}__{arm_id}",
        scenario_id=scenario_id,
        arm_id=arm_id,
        run_class=run_class,  # type: ignore[arg-type]
        protocol_hash=protocol_hash,
        cell_id=cell_id,
        observation_delay_level=obs_level,
        supervision_delay_level=sup_level,
        completeness="complete" if complete else "incomplete",
        incomplete_reason=None if complete else "rpc_timeout",
        termination_reason="mission_complete" if complete else "rpc_timeout",
        episode_verdict=Verdict.VIOLATION if violated else Verdict.PASS,
        physical_verdict=Verdict.VIOLATION if physical else Verdict.PASS,
        procedural_verdict=Verdict.VIOLATION if procedural else Verdict.PASS,
        physical_violation=physical,
        procedural_violation=procedural,
        any_violation=violated,
        first_violation_sim_time_s=20.0 if violated else None,
        mission_completed=complete,
        mission_completed_safely=complete and not violated,
        completion_time_s=completion_time_s if complete else None,
        accepted_by_monitor=accepted,
        false_assurance=false_assurance,
        missed_detection=missed_detection,
        detection_delay_s=detection_delay_s,
        interventions=interventions,
        suspended=suspended,
        abandoned=abandoned,
    )


def synthetic_attempt(
    *,
    arm_id: str,
    scenario_id: str,
    status: str,
    protocol_hash: str = PROTOCOL_HASH,
    run_class: str = "fixture",
) -> AttemptedRun:
    """One attempted-run ledger row. A failed attempt has no episode_id, which is the whole point."""
    return AttemptedRun(
        attempt_id=f"{arm_id}-{scenario_id}-{status}",
        episode_id=f"{scenario_id}__{arm_id}" if status == "completed" else None,
        scenario_id=scenario_id,
        arm_id=arm_id,
        run_class=run_class,  # type: ignore[arg-type]
        protocol_hash=protocol_hash,
        status=status,  # type: ignore[arg-type]
        started_wall_clock="2026-09-17T00:00:00.000+00:00",
        simulator_provenance="fixture_fake",
        error_type=None if status == "completed" else "RpcError",
    )


# --------------------------------------------------------------------------------------
# Denominators and undefined values
# --------------------------------------------------------------------------------------
def test_zero_accepted_episodes_gives_an_undefined_rate_not_zero() -> None:
    outcomes = [
        synthetic_outcome(arm_id="A1_policy_only", scenario_id=f"s{i}", accepted=False, physical=i < 2)
        for i in range(4)
    ]
    analysis = analyze_run(outcomes, PROTOCOL, bootstrap_resamples=200, shift_resamples=100)
    arm = analysis.arm("A1_policy_only")

    assert arm.n_accepted == 0
    rate = arm.conditional_false_assurance
    assert rate.point is None, "a conditional rate over an empty accepted set must be undefined"
    assert rate.denominator == 0
    assert rate.undefined_reason == REASON_NO_ACCEPTED
    # Coverage itself is defined and zero: the arm had 4 complete episodes and accepted none of them.
    assert arm.assurance_coverage.point == pytest.approx(0.0)
    assert arm.assurance_coverage.denominator == 4


def test_arm_without_a_monitor_has_undefined_acceptance_not_zero() -> None:
    outcomes = [synthetic_outcome(arm_id="A0_unguarded", scenario_id=f"s{i}") for i in range(3)]
    analysis = analyze_run(outcomes, PROTOCOL, bootstrap_resamples=200, shift_resamples=100)
    arm = analysis.arm("A0_unguarded")

    assert arm.has_monitor is False
    assert arm.assurance_coverage.point is None
    assert arm.assurance_coverage.undefined_reason == REASON_NO_MONITOR
    assert arm.conditional_false_assurance.point is None
    assert arm.conditional_false_assurance.undefined_reason == REASON_NO_MONITOR
    assert arm.missed_detection.point is None
    assert arm.missed_detection.undefined_reason == REASON_NO_MONITOR


def test_all_episode_denominator_keeps_incomplete_episodes() -> None:
    outcomes = [
        synthetic_outcome(arm_id="A1_policy_only", scenario_id="s0", physical=True, accepted=False),
        synthetic_outcome(arm_id="A1_policy_only", scenario_id="s1", accepted=True, false_assurance=False),
        synthetic_outcome(arm_id="A1_policy_only", scenario_id="s2", accepted=True, false_assurance=False),
        synthetic_outcome(arm_id="A1_policy_only", scenario_id="s3", complete=False, accepted=None),
    ]
    arm = analyze_run(outcomes, PROTOCOL, bootstrap_resamples=200, shift_resamples=100).arm("A1_policy_only")

    assert (arm.n_assessed, arm.n_complete, arm.n_incomplete) == (4, 3, 1)
    assert arm.physical_violation.denominator == 4
    assert arm.physical_violation.numerator == 1
    assert arm.physical_violation_complete_only.denominator == 3
    assert arm.incomplete_reasons == {"rpc_timeout": 1}
    assert arm.termination_reasons["rpc_timeout"] == 1
    # Headline assurance coverage counts EVERY attempted episode. An arm that failed to finish an
    # episode did not provide assurance for it either, so the incomplete episode stays in the
    # denominator (independent review: a 1-accepted, 9-incomplete arm must not read as 100% coverage).
    assert arm.assurance_coverage.denominator == 4
    assert arm.assurance_coverage.numerator == 2
    assert arm.assurance_coverage.point == pytest.approx(0.5)
    # The complete-conditional figure is reported beside it, with its own denominator and label.
    assert arm.assurance_coverage_complete_only.denominator == 3
    assert arm.assurance_coverage_complete_only.numerator == 2


def test_headline_coverage_is_not_inflated_by_incomplete_episodes() -> None:
    """The reviewer's example: one accepted episode and nine incomplete ones is 10% coverage."""
    outcomes = [synthetic_outcome(arm_id="A1_policy_only", scenario_id="s0", accepted=True,
                                  false_assurance=False)]
    outcomes += [
        synthetic_outcome(arm_id="A1_policy_only", scenario_id=f"s{i}", complete=False, accepted=None)
        for i in range(1, 10)
    ]
    arm = analyze_run(outcomes, PROTOCOL, bootstrap_resamples=200, shift_resamples=100).arm(
        "A1_policy_only"
    )
    assert arm.assurance_coverage.numerator == 1
    assert arm.assurance_coverage.denominator == 10
    assert arm.assurance_coverage.point == pytest.approx(0.1)
    assert arm.assurance_coverage_complete_only.point == pytest.approx(1.0)
    assert arm.assurance_coverage_complete_only.denominator == 1
    assert any("complete-conditional" in w for w in arm.warnings)


def test_missed_detection_has_its_own_denominator() -> None:
    outcomes = [
        synthetic_outcome(arm_id="A1_policy_only", scenario_id="s0", physical=True,
                          accepted=True, false_assurance=True, missed_detection=True),
        synthetic_outcome(arm_id="A1_policy_only", scenario_id="s1", physical=True,
                          accepted=False, missed_detection=False, detection_delay_s=2.0),
        synthetic_outcome(arm_id="A1_policy_only", scenario_id="s2", accepted=True,
                          false_assurance=False, missed_detection=False),
    ]
    arm = analyze_run(outcomes, PROTOCOL, bootstrap_resamples=200, shift_resamples=100).arm("A1_policy_only")

    assert arm.n_violation_episodes == 2
    assert arm.missed_detection.denominator == 2, "denominator is violation episodes, not all episodes"
    assert arm.missed_detection.numerator == 1
    # Only the detected violation contributes a delay; the missed one is counted as missing.
    assert arm.detection_delay_s.n == 1
    assert arm.detection_delay_s.n_missing == 1
    assert arm.detection_delay_s.median == pytest.approx(2.0)


def test_detection_delay_is_undefined_when_no_violation_episode_exists() -> None:
    outcomes = [
        synthetic_outcome(arm_id="A2_assumption_aware", scenario_id=f"s{i}", accepted=True,
                          false_assurance=False, missed_detection=False)
        for i in range(3)
    ]
    arm = analyze_run(outcomes, PROTOCOL, bootstrap_resamples=200,
                      shift_resamples=100).arm("A2_assumption_aware")
    assert arm.detection_delay_s.median is None
    assert arm.detection_delay_s.undefined_reason == REASON_NO_VIOLATIONS
    assert arm.missed_detection.point is None
    assert arm.missed_detection.undefined_reason == REASON_NO_VIOLATIONS


# --------------------------------------------------------------------------------------
# The always-abstain failure mode
# --------------------------------------------------------------------------------------
def synthetic_abstain_run() -> list[EpisodeOutcome]:
    """A1 accepts everything and is sometimes wrong; A2 accepts nothing and is actually worse."""
    outcomes: list[EpisodeOutcome] = []
    for i in range(6):
        outcomes.append(synthetic_outcome(
            arm_id="A1_policy_only", scenario_id=f"s{i}", cell_id=CELLS[i % 2],
            physical=i < 2, accepted=True, false_assurance=i < 2, missed_detection=i < 2,
        ))
        outcomes.append(synthetic_outcome(
            arm_id="A2_assumption_aware", scenario_id=f"s{i}", cell_id=CELLS[i % 2],
            physical=i < 4, accepted=False, missed_detection=True, suspended=True,
            interventions=3, completion_time_s=None, complete=True,
        ))
    return outcomes


def test_always_abstain_arm_cannot_look_better_on_a_conditional_rate() -> None:
    analysis = analyze_run(synthetic_abstain_run(), PROTOCOL, bootstrap_resamples=300,
                           shift_resamples=150)
    policy = analysis.arm("A1_policy_only")
    abstain = analysis.arm("A2_assumption_aware")

    # The abstaining arm has a strictly worse independently assessed violation fraction ...
    assert abstain.physical_violation.point == pytest.approx(4 / 6)
    assert policy.physical_violation.point == pytest.approx(2 / 6)
    assert abstain.physical_violation.point > policy.physical_violation.point
    # ... and it buys no conditional rate at all, rather than a flattering 0.0.
    assert abstain.conditional_false_assurance.point is None
    assert abstain.conditional_false_assurance.undefined_reason == REASON_NO_ACCEPTED
    assert policy.conditional_false_assurance.point == pytest.approx(2 / 6)
    # Its coverage is visibly lower and is reported next to the safety numbers.
    assert abstain.assurance_coverage.point == pytest.approx(0.0)
    assert policy.assurance_coverage.point == pytest.approx(1.0)
    assert abstain.assurance_coverage.point < policy.assurance_coverage.point
    assert any("A2_assumption_aware" in w and "0" in w for w in analysis.warnings)


def test_zero_episode_acceptance_warning_does_not_infer_monitor_abstention() -> None:
    # Nominal05 had PASS step verdicts but no accepted episode after crossing the horizon.
    outcome = synthetic_outcome(arm_id="A1_policy_only", scenario_id="horizon", complete=False,
                                accepted=False).model_copy(update={"termination_reason": "horizon_reached"})
    analysis = analyze_run([outcome], PROTOCOL, bootstrap_resamples=50, shift_resamples=50)
    warning = next(w for w in analysis.warnings if "accepted 0" in w)
    assert "UNDEFINED, not 0.0" in warning
    assert "no affirmative episode-level acceptance" in warning
    assert "termination reasons and step verdicts" in warning
    assert "abstaining everywhere" not in warning
    assert analysis.arm("A1_policy_only").conditional_false_assurance.point is None


def test_partial_abstention_is_flagged_with_its_coverage() -> None:
    """An arm that accepts one clean episode gets a perfect conditional rate; coverage must expose it."""
    outcomes: list[EpisodeOutcome] = []
    for i in range(6):
        outcomes.append(synthetic_outcome(
            arm_id="A2_assumption_aware", scenario_id=f"s{i}",
            physical=i < 3, accepted=(i == 5), false_assurance=False if i == 5 else None,
            missed_detection=False if i < 3 else None, detection_delay_s=1.0 if i < 3 else None,
        ))
    analysis = analyze_run(outcomes, PROTOCOL, bootstrap_resamples=300, shift_resamples=150)
    arm = analysis.arm("A2_assumption_aware")

    assert arm.conditional_false_assurance.point == pytest.approx(0.0)
    assert arm.conditional_false_assurance.denominator == 1, "the flattering rate rests on one episode"
    assert arm.assurance_coverage.point == pytest.approx(1 / 6)
    assert arm.any_violation.point == pytest.approx(0.5)
    coverage_warnings = [w for w in analysis.warnings if "coverage" in w and "A2_assumption_aware" in w]
    assert coverage_warnings, "a conditional rate on low coverage must be flagged"


# --------------------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------------------
def synthetic_paired_run(drop_from_second: set[str] | None = None) -> list[EpisodeOutcome]:
    drop = drop_from_second or set()
    outcomes: list[EpisodeOutcome] = []
    flips = {"s0": (True, False), "s1": (True, False), "s2": (False, True), "s3": (False, False)}
    for scenario, (in_first, in_second) in flips.items():
        outcomes.append(synthetic_outcome(
            arm_id="A1_policy_only", scenario_id=scenario, physical=in_first, accepted=True,
            false_assurance=in_first, missed_detection=in_first,
        ))
        if scenario in drop:
            continue
        outcomes.append(synthetic_outcome(
            arm_id="A2_assumption_aware", scenario_id=scenario, physical=in_second, accepted=True,
            false_assurance=in_second, missed_detection=in_second,
        ))
    return outcomes


def test_paired_comparison_uses_mcnemar_counts_that_match_the_hand_count() -> None:
    analysis = analyze_run(synthetic_paired_run(), PROTOCOL, bootstrap_resamples=400,
                           shift_resamples=150)
    comparison = analysis.comparison("A1_policy_only_vs_A2_assumption_aware__physical_violation")

    assert comparison.n_paired == 4
    assert (comparison.mcnemar.b, comparison.mcnemar.c) == (2, 1)
    assert comparison.mcnemar.concordant_pairs == 1
    assert comparison.mcnemar.p_value == pytest.approx(mcnemar_exact(2, 1).p_value)
    assert comparison.first_rate.point == pytest.approx(0.5)
    assert comparison.second_rate.point == pytest.approx(0.25)
    assert comparison.difference.point == pytest.approx(0.25)
    assert comparison.role == "primary"


def test_a_scenario_missing_from_one_arm_is_dropped_and_counted_as_unpaired() -> None:
    analysis = analyze_run(synthetic_paired_run(drop_from_second={"s3"}), PROTOCOL,
                           bootstrap_resamples=400, shift_resamples=150)
    comparison = analysis.comparison("A1_policy_only_vs_A2_assumption_aware__physical_violation")

    assert comparison.n_paired == 3, "the unmatched scenario must leave the paired test"
    assert comparison.n_first_only == 1
    assert comparison.n_second_only == 0
    assert comparison.unpaired_scenario_ids == ["s3"]
    assert comparison.first_rate.denominator == 3
    assert any("unpaired" in w or "only one arm" in w for w in analysis.warnings)
    # The per-arm all-episode metrics still see every assessed episode.
    assert analysis.arm("A1_policy_only").n_assessed == 4
    assert analysis.arm("A2_assumption_aware").n_assessed == 3


def test_primary_comparison_is_stratified_by_cell_when_several_cells_exist() -> None:
    outcomes: list[EpisodeOutcome] = []
    for i in range(6):
        for arm in ("A1_policy_only", "A2_assumption_aware"):
            outcomes.append(synthetic_outcome(
                arm_id=arm, scenario_id=f"s{i}", cell_id=CELLS[i % 3], physical=(i % 2 == 0),
                accepted=True, false_assurance=False, missed_detection=False,
            ))
    analysis = analyze_run(outcomes, PROTOCOL, bootstrap_resamples=300, shift_resamples=100)
    comparison = analysis.comparison("A1_policy_only_vs_A2_assumption_aware__physical_violation")
    assert comparison.stratified_by == "cell_id"
    assert comparison.difference.stratified is True


def test_context_comparisons_cover_the_unguarded_arm() -> None:
    outcomes = synthetic_paired_run()
    outcomes += [synthetic_outcome(arm_id="A0_unguarded", scenario_id=f"s{i}", physical=i < 3)
                 for i in range(4)]
    analysis = analyze_run(outcomes, PROTOCOL, bootstrap_resamples=300, shift_resamples=100)

    ids = {c.comparison_id for c in analysis.context_comparisons}
    assert "A0_unguarded_vs_A1_policy_only__physical_violation" in ids
    assert "A0_unguarded_vs_A2_assumption_aware__physical_violation" in ids
    for comparison in analysis.context_comparisons:
        assert comparison.role == "context"
        assert "Context only" in comparison.note


# --------------------------------------------------------------------------------------
# Strata, shifts, burden
# --------------------------------------------------------------------------------------
def test_per_stratum_rows_carry_their_own_denominators() -> None:
    outcomes: list[EpisodeOutcome] = []
    for i in range(4):
        outcomes.append(synthetic_outcome(arm_id="A1_policy_only", scenario_id=f"s{i}",
                                          cell_id=CELLS[i % 2], physical=(i % 2 == 0),
                                          accepted=True, false_assurance=(i % 2 == 0),
                                          missed_detection=(i % 2 == 0)))
    analysis = analyze_run(outcomes, PROTOCOL, bootstrap_resamples=200, shift_resamples=100)

    cell_rows = {m.stratum_key: m for m in analysis.by_stratum if m.stratum_kind == "cell"}
    assert set(cell_rows) == {CELLS[0], CELLS[1]}
    for row in cell_rows.values():
        assert row.n_assessed == 2
        assert row.physical_violation.denominator == 2
    assert cell_rows[CELLS[0]].physical_violation.numerator == 2
    assert cell_rows[CELLS[1]].physical_violation.numerator == 0

    obs_rows = [m for m in analysis.by_stratum if m.stratum_kind == "observation_delay_level"]
    assert obs_rows and sum(m.n_assessed for m in obs_rows) == 4


def test_intervention_burden_and_completion_time_are_reported() -> None:
    outcomes = [
        synthetic_outcome(arm_id="A2_assumption_aware", scenario_id="s0", interventions=2,
                          suspended=True, completion_time_s=80.0, accepted=False),
        synthetic_outcome(arm_id="A2_assumption_aware", scenario_id="s1", interventions=0,
                          completion_time_s=60.0, accepted=True, false_assurance=False),
        synthetic_outcome(arm_id="A2_assumption_aware", scenario_id="s2", interventions=4,
                          abandoned=True, complete=False, accepted=None),
    ]
    arm = analyze_run(outcomes, PROTOCOL, bootstrap_resamples=200,
                      shift_resamples=100).arm("A2_assumption_aware")

    assert arm.total_interventions == 6
    assert arm.interventions_per_episode.median == pytest.approx(2.0)
    assert arm.suspension_fraction.numerator == 1 and arm.suspension_fraction.denominator == 3
    assert arm.abandonment_fraction.numerator == 1
    assert arm.completion_time_s.n == 2 and arm.completion_time_s.n_missing == 1
    assert arm.completion_time_s.median == pytest.approx(70.0)


def test_paired_shift_comparisons_are_computed_for_continuous_outcomes() -> None:
    outcomes: list[EpisodeOutcome] = []
    for i in range(4):
        outcomes.append(synthetic_outcome(arm_id="A1_policy_only", scenario_id=f"s{i}",
                                          completion_time_s=60.0, accepted=True, false_assurance=False))
        outcomes.append(synthetic_outcome(arm_id="A2_assumption_aware", scenario_id=f"s{i}",
                                          completion_time_s=75.0, accepted=True, false_assurance=False))
    analysis = analyze_run(outcomes, PROTOCOL, bootstrap_resamples=200, shift_resamples=200)
    shift = next(s for s in analysis.shift_comparisons if s.outcome_id == "completion_time_s")

    assert shift.n_paired_scenarios == 4
    assert shift.n_paired_with_values == 4
    assert shift.shift.point == pytest.approx(-15.0)
    assert shift.first_distribution.median == pytest.approx(60.0)
    assert shift.second_distribution.median == pytest.approx(75.0)


# --------------------------------------------------------------------------------------
# Input guards and workflow entry points
# --------------------------------------------------------------------------------------
def test_mixing_run_classes_is_refused() -> None:
    outcomes = [
        synthetic_outcome(arm_id="A1_policy_only", scenario_id="s0"),
        synthetic_outcome(arm_id="A1_policy_only", scenario_id="s1", run_class="heldout"),
    ]
    with pytest.raises(AnalysisInputError, match="run classes"):
        analyze_run(outcomes, PROTOCOL)


def test_mixing_protocol_hashes_is_refused() -> None:
    outcomes = [
        synthetic_outcome(arm_id="A1_policy_only", scenario_id="s0"),
        synthetic_outcome(arm_id="A1_policy_only", scenario_id="s1", protocol_hash="sha256:deadbeef"),
    ]
    with pytest.raises(AnalysisInputError, match="protocol hashes"):
        analyze_run(outcomes, PROTOCOL)


def test_a_repeated_scenario_in_one_arm_is_refused() -> None:
    outcomes = [
        synthetic_outcome(arm_id="A1_policy_only", scenario_id="s0"),
        synthetic_outcome(arm_id="A1_policy_only", scenario_id="s0"),
    ]
    with pytest.raises(AnalysisInputError, match="more than once"):
        analyze_run(outcomes, PROTOCOL)


def test_empty_input_is_refused() -> None:
    with pytest.raises(AnalysisInputError):
        analyze_run([], PROTOCOL)


def test_compute_run_analysis_refuses_a_mislabelled_run() -> None:
    outcomes = [synthetic_outcome(arm_id="A1_policy_only", scenario_id="s0")]
    with pytest.raises(AnalysisInputError, match="run_class"):
        compute_run_analysis(outcomes, PROTOCOL, run_class="heldout", protocol_hash=PROTOCOL_HASH,
                             bootstrap_resamples=100, shift_resamples=50)
    with pytest.raises(AnalysisInputError, match="protocol_hash"):
        compute_run_analysis(outcomes, PROTOCOL, run_class="fixture", protocol_hash="sha256:nope",
                             bootstrap_resamples=100, shift_resamples=50)
    good = compute_run_analysis(outcomes, PROTOCOL, run_class="fixture", protocol_hash=PROTOCOL_HASH,
                                bootstrap_resamples=100, shift_resamples=50)
    assert good.run_class == "fixture"
    assert good.protocol_hash == PROTOCOL_HASH


def test_typed_attempted_ledger_makes_lost_episodes_visible() -> None:
    """Three scored episodes out of five attempts: the two crashes must stay in the denominator."""
    outcomes = [synthetic_outcome(arm_id="A1_policy_only", scenario_id=f"s{i}", accepted=True,
                                  false_assurance=False) for i in range(3)]
    ledger = [synthetic_attempt(arm_id="A1_policy_only", scenario_id=f"s{i}",
                                status="completed" if i < 3 else "crashed") for i in range(5)]
    analysis = compute_run_analysis(
        outcomes, PROTOCOL, run_class="fixture", protocol_hash=PROTOCOL_HASH,
        attempted=ledger, bootstrap_resamples=100, shift_resamples=50,
    )
    arm = analysis.arm("A1_policy_only")

    assert analysis.n_attempted_runs == 5
    assert arm.n_attempted == 5
    assert arm.attempted_denominator_source == "attempted_ledger"
    assert arm.n_assessed == 3
    assert arm.n_attempted_without_outcome == 2, "crashed attempts must stay visible"
    assert arm.n_accepted == 3
    assert arm.assurance_coverage.point == pytest.approx(0.6), "3 accepted over 5 attempted runs"
    assert arm.assurance_coverage.denominator == 5
    assert arm.assurance_coverage_complete_only.point == pytest.approx(1.0)
    assert analysis.attempts_summary["by_status"] == {"completed": 3, "crashed": 2}


def test_attempt_summary_dicts_are_refused_instead_of_silently_ignored() -> None:
    """The old summary shape produced a silently wrong denominator; it must now fail loudly."""
    outcomes = [synthetic_outcome(arm_id="A1_policy_only", scenario_id=f"s{i}") for i in range(3)]
    summary = {"attempts": 5, "by_arm": {"A1_policy_only": {"completed": 3, "crashed": 2}}}

    with pytest.raises(AnalysisInputError, match="attempts_summary is no longer an input"):
        compute_run_analysis(outcomes, PROTOCOL, run_class="fixture", protocol_hash=PROTOCOL_HASH,
                             attempts_summary=summary, bootstrap_resamples=50, shift_resamples=50)
    with pytest.raises(AnalysisInputError, match="typed attempted-run ledger"):
        analyze_run(outcomes, PROTOCOL, attempted=summary,  # type: ignore[arg-type]
                    bootstrap_resamples=50, shift_resamples=50)
    with pytest.raises(AnalysisInputError, match="AttemptedRun"):
        analyze_run(outcomes, PROTOCOL, attempted=[{"arm_id": "A1_policy_only"}],  # type: ignore[list-item]
                    bootstrap_resamples=50, shift_resamples=50)


def test_an_arm_whose_every_run_failed_stays_in_the_report_with_zero_coverage() -> None:
    """The review's probe: one accepted A1 episode, ten A1 attempts, ten failed A2 attempts."""
    outcomes = [synthetic_outcome(arm_id="A1_policy_only", scenario_id="s0", accepted=True,
                                  false_assurance=False)]
    ledger = [synthetic_attempt(arm_id="A1_policy_only", scenario_id=f"s{i}",
                                status="completed" if i == 0 else "crashed") for i in range(10)]
    ledger += [synthetic_attempt(arm_id="A2_assumption_aware", scenario_id=f"s{i}", status="crashed")
               for i in range(10)]
    strata = {f"s{i}": {"cell": CELLS[0], "observation_delay_level": CELLS[0].split("__")[0],
                        "supervision_delay_level": CELLS[0].split("__")[1]} for i in range(10)}

    analysis = analyze_run(outcomes, PROTOCOL, attempted=ledger, scenario_strata=strata,
                           bootstrap_resamples=100, shift_resamples=50)

    assert analysis.arm_ids == ["A1_policy_only", "A2_assumption_aware"]
    assert analysis.arms_without_outcomes == ["A2_assumption_aware"]
    a1, a2 = analysis.arm("A1_policy_only"), analysis.arm("A2_assumption_aware")
    assert (a1.n_attempted, a1.n_assessed) == (10, 1)
    assert a1.assurance_coverage.point == pytest.approx(0.1), "one accepted episode over ten attempts"
    assert (a2.n_attempted, a2.n_assessed, a2.n_accepted) == (10, 0, 0)
    assert a2.assurance_coverage.point == pytest.approx(0.0)
    assert a2.assurance_coverage.denominator == 10
    assert a2.conditional_false_assurance.point is None
    assert a2.conditional_false_assurance.undefined_reason == REASON_NO_ACCEPTED
    assert a2.physical_violation.point is None, "an unscored arm has no violation fraction"
    assert a2.missed_detection.undefined_reason == REASON_NO_VIOLATIONS

    cells = {(m.arm_id, m.stratum_key): m for m in analysis.by_stratum if m.stratum_kind == "cell"}
    a1_cell = cells[("A1_policy_only", CELLS[0])]
    assert a1_cell.n_attempted == 10 and a1_cell.n_assessed == 1
    assert a1_cell.assurance_coverage.point == pytest.approx(0.1), "per-cell coverage uses all attempts"
    a2_cell = cells[("A2_assumption_aware", CELLS[0])]
    assert a2_cell.n_attempted == 10 and a2_cell.assurance_coverage.point == pytest.approx(0.0)
    assert any("produced no assessed outcome" in w for w in analysis.warnings)


def test_unplaceable_attempts_make_stratum_denominators_unavailable_not_smaller() -> None:
    """No manifest for the failed scenarios: the per-cell attempted denominator must say so."""
    outcomes = [synthetic_outcome(arm_id="A1_policy_only", scenario_id="s0", accepted=True,
                                  false_assurance=False)]
    ledger = [synthetic_attempt(arm_id="A1_policy_only", scenario_id=f"s{i}",
                                status="completed" if i == 0 else "crashed") for i in range(4)]

    analysis = analyze_run(outcomes, PROTOCOL, attempted=ledger, bootstrap_resamples=100,
                           shift_resamples=50)
    arm = analysis.arm("A1_policy_only")
    assert arm.n_attempted == 4, "the arm total is still known: an attempt names its arm"

    cell = next(m for m in analysis.by_stratum if m.stratum_kind == "cell")
    assert cell.n_attempted is None
    assert cell.attempted_denominator_source == "unavailable"
    assert cell.attempted_denominator_reason == REASON_ATTEMPTED_MEMBERSHIP_UNKNOWN
    assert cell.assurance_coverage.point is None
    assert cell.assurance_coverage.undefined_reason == REASON_ATTEMPTED_MEMBERSHIP_UNKNOWN
    assert cell.assurance_coverage_complete_only.point == pytest.approx(1.0), (
        "the complete-conditional figure stays defined beside it, with its own denominator"
    )
    assert any("membership is unknown" in w for w in analysis.warnings)


def test_a_ledger_with_fewer_runs_than_outcomes_is_reported_as_unavailable() -> None:
    """A truncated ledger cannot silently produce a coverage above one, or a quiet fallback."""
    outcomes = [synthetic_outcome(arm_id="A1_policy_only", scenario_id=f"s{i}", accepted=True,
                                  false_assurance=False) for i in range(3)]
    ledger = [synthetic_attempt(arm_id="A1_policy_only", scenario_id="s0", status="completed")]

    analysis = analyze_run(outcomes, PROTOCOL, attempted=ledger, bootstrap_resamples=50,
                           shift_resamples=50)
    arm = analysis.arm("A1_policy_only")
    assert arm.n_attempted is None
    assert arm.attempted_denominator_reason == REASON_ATTEMPTED_LEDGER_SHORTER
    assert arm.assurance_coverage.point is None
    assert arm.n_attempted_without_outcome is None
    assert any("attempted denominator is reported as unavailable" in w for w in analysis.warnings)


def test_a_ledger_from_another_run_is_refused() -> None:
    outcomes = [synthetic_outcome(arm_id="A1_policy_only", scenario_id="s0")]
    foreign = synthetic_attempt(arm_id="A1_policy_only", scenario_id="s0", status="completed",
                                protocol_hash="sha256:not-this-run")
    with pytest.raises(AnalysisInputError, match="protocol hashes"):
        analyze_run(outcomes, PROTOCOL, attempted=[foreign], bootstrap_resamples=50, shift_resamples=50)


def test_load_outcomes_reads_a_jsonl_file(tmp_path) -> None:
    outcomes = [synthetic_outcome(arm_id="A1_policy_only", scenario_id=f"s{i}") for i in range(3)]
    path = tmp_path / "outcomes.jsonl"
    path.write_text("\n".join(o.model_dump_json() for o in outcomes) + "\n", encoding="utf-8")

    assert [o.episode_id for o in load_outcomes(path)] == [o.episode_id for o in outcomes]
    assert len(load_outcomes(tmp_path)) == 3  # a directory resolves to outcomes.jsonl

    path.write_text('{"not": "an outcome"}\n', encoding="utf-8")
    with pytest.raises(AnalysisInputError, match="outcomes.jsonl:1"):
        load_outcomes(path)


def test_analysis_is_deterministic_for_the_same_input() -> None:
    outcomes = synthetic_paired_run()
    one = analyze_run(outcomes, PROTOCOL, bootstrap_resamples=300, shift_resamples=100)
    two = analyze_run(list(reversed(outcomes)), PROTOCOL, bootstrap_resamples=300, shift_resamples=100)
    assert json.dumps(one.model_dump(mode="json"), sort_keys=True) == json.dumps(
        two.model_dump(mode="json"), sort_keys=True
    ), "analysis must not depend on the order the outcomes arrive in"
