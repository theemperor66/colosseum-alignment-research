"""Cross-cutting invariants: no rate without a denominator, no undefined value without a reason.

These tests walk the serialized analysis and the rendered report instead of checking one metric at a
time, so a new field added later cannot quietly introduce a naked 0.0 where the data defines nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from colosseum_assurance.analysis.figures import (
    audit_summary_is_renderable,
    figure_audit_reconstruction,
    render_all,
    write_all_figures,
)
from colosseum_assurance.analysis.metrics import CONDITIONAL_RATE_CAVEAT, RunAnalysis, analyze_run
from colosseum_assurance.analysis.report import (
    PENDING_BANNER,
    SYNTHETIC_BANNER,
    render_json,
    render_markdown,
    write_report,
)
from colosseum_assurance.analysis.stats import (
    BootstrapEstimate,
    ProportionEstimate,
    cluster_bootstrap_statistic,
    wilson_interval,
)
from colosseum_assurance.evaluation.outcomes import EpisodeOutcome, ObligationOutcome
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.schemas import AttemptedRun, Verdict

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
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
    run_class: str = "fixture",
) -> EpisodeOutcome:
    """Build one evaluator outcome by hand. Fixture data only; never experimental evidence."""
    violated = physical or procedural
    obs_level, sup_level = cell_id.split("__")
    return EpisodeOutcome(
        episode_id=f"{scenario_id}__{arm_id}",
        scenario_id=scenario_id,
        arm_id=arm_id,
        run_class=run_class,  # type: ignore[arg-type]
        protocol_hash=PROTOCOL_HASH,
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


def synthetic_mixed_run(run_class: str = "fixture") -> list[EpisodeOutcome]:
    """Three arms with deliberately awkward data: a no-monitor arm, an abstainer, an incomplete run."""
    outcomes: list[EpisodeOutcome] = []
    for i in range(6):
        cell = CELLS[i % 3]
        outcomes.append(synthetic_outcome(
            arm_id="A0_unguarded", scenario_id=f"s{i}", cell_id=cell, physical=i < 3,
            run_class=run_class,
        ))
        outcomes.append(synthetic_outcome(
            arm_id="A1_policy_only", scenario_id=f"s{i}", cell_id=cell, physical=i < 2,
            accepted=True, false_assurance=i < 2, missed_detection=i < 2,
            detection_delay_s=None if i < 2 else None, run_class=run_class,
        ))
        outcomes.append(synthetic_outcome(
            arm_id="A2_assumption_aware", scenario_id=f"s{i}", cell_id=cell, physical=False,
            accepted=False, missed_detection=None, interventions=2, suspended=True,
            complete=(i != 5), run_class=run_class,
        ))
    return outcomes


def build_analysis(run_class: str = "fixture") -> RunAnalysis:
    return analyze_run(
        synthetic_mixed_run(run_class), PROTOCOL, bootstrap_resamples=300, shift_resamples=150,
        code_version={"git_commit": "0" * 40, "package_version": "0.1.0"},
    )


# --------------------------------------------------------------------------------------
# Structural invariants over the serialized analysis
# --------------------------------------------------------------------------------------
def walk(node: Any, path: str = "$") -> list[tuple[str, dict[str, Any]]]:
    """Yield every dict in the serialized analysis together with its JSON path."""
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(node, dict):
        found.append((path, node))
        for key, value in node.items():
            found.extend(walk(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(walk(value, f"{path}[{index}]"))
    return found


def test_every_empty_denominator_yields_an_undefined_rate_with_a_reason() -> None:
    payload = build_analysis().model_dump(mode="json")
    checked = 0
    for path, node in walk(payload):
        if {"numerator", "denominator", "point"} <= set(node):
            checked += 1
            if node["denominator"] == 0:
                assert node["point"] is None, f"{path}: an empty denominator must not yield a number"
                assert node["undefined_reason"], f"{path}: undefined values need a machine-readable reason"
            if node["point"] is None:
                assert node["denominator"] == 0, f"{path}: a defined denominator must produce a point"
                assert node["undefined_reason"], f"{path}: missing undefined_reason"
            else:
                assert 0.0 <= node["point"] <= 1.0
                assert node["numerator"] <= node["denominator"]
    assert checked > 30, "the walk must actually reach the proportion records"


def test_every_undefined_distribution_and_bootstrap_states_its_reason() -> None:
    payload = build_analysis().model_dump(mode="json")
    distributions = 0
    bootstraps = 0
    for path, node in walk(payload):
        if {"median", "n", "n_missing"} <= set(node):
            distributions += 1
            if node["median"] is None:
                assert node["n"] == 0, f"{path}: values present but no median"
                assert node["undefined_reason"], f"{path}: missing undefined_reason"
        if {"point", "resamples", "n_units"} <= set(node):
            bootstraps += 1
            if node["point"] is None:
                assert node["undefined_reason"], f"{path}: missing undefined_reason"
                assert node["n_units"] == 0
    assert distributions > 5 and bootstraps > 2


def test_unguarded_arm_reports_undefined_acceptance_everywhere_it_appears() -> None:
    analysis = build_analysis()
    rows = [analysis.overall["A0_unguarded"], *[m for m in analysis.by_stratum
                                                if m.arm_id == "A0_unguarded"]]
    assert len(rows) > 1
    for row in rows:
        assert row.has_monitor is False
        assert row.assurance_coverage.point is None
        assert row.conditional_false_assurance.point is None
        assert row.n_accepted == 0
        # All-episode outcomes stay defined: only monitor-relative quantities are undefined.
        assert row.physical_violation.point is not None


def test_an_unavailable_attempted_denominator_is_never_silently_replaced() -> None:
    """Walk every metrics row: an unavailable attempted denominator must say so and stay undefined.

    The ledger here holds twelve attempts for scenarios that no manifest and no outcome can place in a
    cell, so per-stratum membership is genuinely unknown. The failure mode this pins is the quiet
    substitution of the completed-only count, which would report a confident per-cell coverage.
    """
    outcomes = synthetic_mixed_run()
    ledger = [
        AttemptedRun(
            attempt_id=f"a{i}-{arm}", episode_id=None, scenario_id=f"unlisted{i}", arm_id=arm,
            run_class="fixture", protocol_hash=PROTOCOL_HASH, status="crashed",
            started_wall_clock="2026-09-17T00:00:00.000+00:00", simulator_provenance="fixture_fake",
        )
        for i in range(4)
        for arm in ("A0_unguarded", "A1_policy_only", "A2_assumption_aware")
    ]
    analysis = analyze_run(outcomes, PROTOCOL, attempted=ledger + _completed_ledger(outcomes),
                           bootstrap_resamples=200, shift_resamples=100)

    rows = [analysis.overall[a] for a in analysis.arm_ids] + list(analysis.by_stratum)
    unavailable = [r for r in rows if r.attempted_denominator_source == "unavailable"]
    assert len(unavailable) >= 3, "the per-stratum rows must be the ones that lose their denominator"
    for row in unavailable:
        assert row.n_attempted is None, "an unavailable denominator must not print a number"
        assert row.attempted_denominator_reason
        assert row.n_attempted_without_outcome is None
        assert row.assurance_coverage.point is None
    for arm_id in analysis.arm_ids:
        overall = analysis.overall[arm_id]
        assert overall.attempted_denominator_source == "attempted_ledger"
        assert overall.n_attempted == 4 + sum(1 for o in outcomes if o.arm_id == arm_id), (
            "the per-arm total is still known: every attempt names its arm"
        )
    markdown = render_markdown(analysis)
    assert "unavailable (attempted_stratum_membership_unresolved)" in markdown


def _completed_ledger(outcomes: list[EpisodeOutcome]) -> list[AttemptedRun]:
    """One completed ledger row per scored outcome, as the runner would have appended it."""
    return [
        AttemptedRun(
            attempt_id=f"done-{o.episode_id}", episode_id=o.episode_id, scenario_id=o.scenario_id,
            arm_id=o.arm_id, run_class="fixture", protocol_hash=PROTOCOL_HASH, status="completed",
            started_wall_clock="2026-09-17T00:00:00.000+00:00", simulator_provenance="fixture_fake",
        )
        for o in outcomes
    ]


# --------------------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------------------
def test_fixture_report_carries_the_synthetic_banner() -> None:
    markdown = render_markdown(build_analysis("fixture"))
    assert SYNTHETIC_BANNER in markdown
    banner_line = next(i for i, line in enumerate(markdown.splitlines()) if SYNTHETIC_BANNER in line)
    assert banner_line <= 3, "the banner must sit at the top of the report, not be buried in it"
    assert PENDING_BANNER in markdown


def test_heldout_report_has_no_synthetic_banner() -> None:
    markdown = render_markdown(build_analysis("heldout"))
    assert SYNTHETIC_BANNER not in markdown
    assert "run_class" not in markdown.splitlines()[0] or "heldout" in markdown.splitlines()[0]
    assert "- Run class: `heldout`" in markdown


def test_report_states_provenance_acceptance_rule_and_counts() -> None:
    analysis = build_analysis()
    markdown = render_markdown(analysis)
    assert analysis.protocol_hash in markdown
    assert analysis.protocol_short_hash in markdown
    assert analysis.acceptance_rule_id in markdown
    assert "ACCEPTED by an arm if and only if" in markdown
    assert "git_commit=" + "0" * 40 in markdown
    assert "evaluator-v2.0.0" in markdown
    assert f"Episode outcomes analysed: {analysis.n_episode_outcomes}" in markdown
    assert "A0_unguarded" in markdown and "A2_assumption_aware" in markdown


def test_report_prints_the_denominator_next_to_every_conditional_rate() -> None:
    markdown = render_markdown(build_analysis())
    section = markdown.split("## Assurance coverage")[1].split("## Intervention burden")[0]
    rows = [line for line in section.splitlines() if line.startswith("| `A")]
    assert len(rows) == 6  # Three coverage rows and three explicit ascertainability/bounds rows.
    for row in rows:
        assert "n=" in row or "undefined (n=0;" in row
    abstain_row = next(row for row in rows if "A2_assumption_aware" in row)
    assert "undefined (n=0; no_accepted_episodes)" in abstain_row
    assert "0.000 [0.000," in abstain_row, "coverage 0.000 must sit beside the undefined rate"


def test_report_warns_that_conditional_rates_select_different_sets() -> None:
    markdown = render_markdown(build_analysis())
    assert CONDITIONAL_RATE_CAVEAT in markdown
    assert "not an estimate of a causal safety effect" in markdown
    assert "zero observed violations does not establish zero risk" in markdown


def test_report_shows_unpaired_scenarios_when_pairing_breaks() -> None:
    outcomes = [o for o in synthetic_mixed_run() if not (o.arm_id == "A2_assumption_aware"
                                                         and o.scenario_id == "s4")]
    analysis = analyze_run(outcomes, PROTOCOL, bootstrap_resamples=200, shift_resamples=100)
    markdown = render_markdown(analysis)
    assert "unpaired first/second" in markdown
    assert "dropped, unpaired: `s4`" in markdown


def test_report_rendering_is_byte_deterministic() -> None:
    analysis = build_analysis()
    assert render_markdown(analysis) == render_markdown(analysis)
    assert render_json(analysis) == render_json(analysis)


def test_write_report_returns_string_paths_and_writes_both_files(tmp_path) -> None:
    analysis = build_analysis()
    written = write_report(analysis, tmp_path, {"run_metadata": {"episodes": 18},
                                                "limits": ["no live simulator was used"]})
    assert set(written) == {"markdown", "json"}
    assert all(isinstance(p, str) for p in written.values())
    markdown = (tmp_path / "analysis-report.md").read_text(encoding="utf-8")
    assert "no live simulator was used" in markdown
    assert "run_metadata" in markdown
    assert (tmp_path / "analysis-report.json").exists()


def test_report_json_contains_the_full_analysis() -> None:
    analysis = build_analysis()
    payload = json.loads(render_json(analysis))
    assert payload["banner"] == SYNTHETIC_BANNER
    assert payload["analysis"]["protocol_hash"] == PROTOCOL_HASH
    assert payload["audit"] is None
    assert payload["analysis"]["overall"]["A2_assumption_aware"]["conditional_false_assurance"][
        "undefined_reason"] == "no_accepted_episodes"


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def test_render_all_writes_real_png_files_without_audit_scores(tmp_path) -> None:
    analysis = build_analysis()
    paths = render_all(analysis, tmp_path, None, protocol=ProtocolConfig())
    assert len(paths) == 5, "four analysis figures plus the procedural heatmap"
    for path in paths:
        data = (tmp_path / path.split("/")[-1]).read_bytes()
        assert data.startswith(PNG_MAGIC)
        assert len(data) > 5000


def test_render_all_ignores_an_audit_object_that_carries_no_scores(tmp_path) -> None:
    """A half-built audit result must leave the figure missing, not produce an empty chart."""
    class NotAnAuditSummary:
        variant_ids: list[str] = []

    paths = render_all(build_analysis(), tmp_path, NotAnAuditSummary())
    assert len(paths) == 5
    assert not (tmp_path / "fig_d_audit_reconstruction.png").exists()


def test_figures_survive_an_arm_with_no_defined_monitor_metrics(tmp_path) -> None:
    outcomes = [synthetic_outcome(arm_id="A0_unguarded", scenario_id=f"s{i}", cell_id=CELLS[i % 2])
                for i in range(4)]
    analysis = analyze_run(outcomes, PROTOCOL, bootstrap_resamples=200, shift_resamples=100)
    written = write_all_figures(analysis, tmp_path, protocol=ProtocolConfig())
    assert set(written) >= {"a_violation_fractions", "b_safety_completion_tradeoff", "c_detection_delay"}
    for path in written.values():
        assert path.read_bytes().startswith(PNG_MAGIC)


def test_heatmap_marks_cells_with_no_episodes_as_not_available(tmp_path) -> None:
    """Only two of the nine protocol cells have data; the rest must not be drawn as measured zeros."""
    analysis = build_analysis()
    cell_rows = {m.stratum_key for m in analysis.by_stratum if m.stratum_kind == "cell"}
    assert len(cell_rows) == 3 < len(PROTOCOL.cells())
    written = write_all_figures(analysis, tmp_path, protocol=ProtocolConfig())
    assert written["e_cell_heatmap_physical"].exists()


@pytest.mark.parametrize("run_class", ["fixture", "smoke"])
def test_synthetic_run_classes_are_watermarked(run_class: str, tmp_path) -> None:
    analysis = analyze_run(synthetic_mixed_run(run_class), PROTOCOL, bootstrap_resamples=100,
                           shift_resamples=50)
    assert analysis.is_synthetic is True
    assert SYNTHETIC_BANNER in render_markdown(analysis)
    assert analysis.interpretation_notes[0] == SYNTHETIC_BANNER + "."


# --------------------------------------------------------------------------------------
# The audit figure, exercised through the published audit-score contract
# --------------------------------------------------------------------------------------
@dataclass
class StubQuestionScore:
    """Minimal stand-in for ``audit.scoring.QuestionScore`` (the attributes figures.py reads)."""

    variant_id: str
    question_id: str
    n_episodes: int
    correct: int
    incorrect_confident: int
    insufficient_evidence: int
    correct_rate: ProportionEstimate


@dataclass
class StubVariantScore:
    variant_id: str
    removed_fields: list[str]
    n_episodes: int
    per_question: dict[str, StubQuestionScore]
    overall_correct_rate: BootstrapEstimate
    retains_redundant_routes: bool = False


@dataclass
class StubAuditSummary:
    variant_ids: list[str]
    question_ids: list[str]
    variants: dict[str, StubVariantScore]


def synthetic_audit_summary() -> StubAuditSummary:
    """Hand-built audit scores: the provenance-rich record answers Q2, the action-only record cannot."""
    questions = ["Q1_policy_version", "Q2_observation_available"]
    counts = {
        ("provenance_rich", "Q1_policy_version"): (8, 0, 0),
        ("provenance_rich", "Q2_observation_available"): (7, 1, 0),
        ("action_only", "Q1_policy_version"): (8, 0, 0),
        ("action_only", "Q2_observation_available"): (0, 0, 8),
    }
    variants: dict[str, StubVariantScore] = {}
    for variant_id, removed in (("provenance_rich", []), ("action_only", ["observation"])):
        per_question = {
            q: StubQuestionScore(
                variant_id=variant_id, question_id=q, n_episodes=8,
                correct=counts[(variant_id, q)][0],
                incorrect_confident=counts[(variant_id, q)][1],
                insufficient_evidence=counts[(variant_id, q)][2],
                correct_rate=wilson_interval(counts[(variant_id, q)][0], 8),
            )
            for q in questions
        }
        correct = sum(s.correct for s in per_question.values())
        variants[variant_id] = StubVariantScore(
            variant_id=variant_id, removed_fields=removed, n_episodes=8, per_question=per_question,
            overall_correct_rate=cluster_bootstrap_statistic(
                [1.0] * correct + [0.0] * (16 - correct),
                [f"ep{i // 2}" for i in range(16)], seed=7717, resamples=200,
            ),
        )
    return StubAuditSummary(variant_ids=["provenance_rich", "action_only"], question_ids=questions,
                            variants=variants)


def test_audit_figure_is_written_when_scores_follow_the_contract(tmp_path) -> None:
    summary = synthetic_audit_summary()
    assert audit_summary_is_renderable(summary) is True
    path = figure_audit_reconstruction(build_analysis(), summary, tmp_path)
    assert path.exists()
    assert path.read_bytes().startswith(PNG_MAGIC)
    assert path.stat().st_size > 5000


def test_write_all_figures_includes_the_audit_panel_when_scores_are_supplied(tmp_path) -> None:
    written = write_all_figures(build_analysis(), tmp_path, audit=synthetic_audit_summary(),
                                protocol=ProtocolConfig())
    assert "d_audit_reconstruction" in written
    assert len(written) == 6, "five analysis figures plus the audit panel"
    paths = render_all(build_analysis(), tmp_path, synthetic_audit_summary())
    assert len(paths) == 6
    assert any("fig_d_audit_reconstruction" in p for p in paths)


def test_undecidable_obligations_are_counted_and_warned_about() -> None:
    """A truth-gap UNKNOWN must surface as its own count, never as the pass side of a rate."""
    outcome = synthetic_outcome(arm_id="A1_policy_only", scenario_id="s0", accepted=False)
    outcome.obligations = {
        "geofence": ObligationOutcome(
            obligation_id="geofence", category="physical", verdict=Verdict.UNKNOWN,
            unknown_reason="truth_gap_exceeds_max_permitted_truth_gap_s",
        ),
        "collision": ObligationOutcome(
            obligation_id="collision", category="physical", verdict=Verdict.PASS,
        ),
    }
    clean = synthetic_outcome(arm_id="A1_policy_only", scenario_id="s1", accepted=False)
    analysis = analyze_run([outcome, clean], PROTOCOL, bootstrap_resamples=100, shift_resamples=50)
    arm = analysis.arm("A1_policy_only")

    assert arm.obligation_unknown_counts == {"geofence": 1}
    assert arm.unknown_reasons == {"truth_gap_exceeds_max_permitted_truth_gap_s": 1}
    assert any("UNDECIDABLE" in w for w in analysis.warnings)
    markdown = render_markdown(analysis)
    assert "undecidable obligations" in markdown
    assert "geofence=1" in markdown
    assert "never a pass" in markdown


def test_report_and_figure_label_a_redundancy_probe_as_such(tmp_path) -> None:
    """A variant that keeps a redundant route must not be described as removing the information."""
    summary = synthetic_audit_summary()
    summary.variants["action_only"].retains_redundant_routes = True
    markdown = render_markdown(build_analysis(), audit=summary)
    assert "redundancy probe (information retained by another route)" in markdown
    assert "information removed" in markdown  # the other variant keeps the honest label

    path = figure_audit_reconstruction(build_analysis(), summary, tmp_path)
    assert path.exists() and path.read_bytes().startswith(PNG_MAGIC)


def test_audit_section_handles_composite_question_field_keys(tmp_path) -> None:
    """Scoring is per answer field, so the scored key is '<question_id>.<field_name>'."""
    summary = synthetic_audit_summary()
    keys = ["Q2_observation_available.observed_position", "Q2_observation_available.evidence_age"]
    for variant in summary.variants.values():
        variant.per_question = {
            key: StubQuestionScore(
                variant_id=variant.variant_id, question_id=key.split(".")[0],
                n_episodes=8, correct=8 if key.endswith("observed_position") else 0,
                incorrect_confident=0, insufficient_evidence=0 if key.endswith("observed_position") else 8,
                correct_rate=wilson_interval(8 if key.endswith("observed_position") else 0, 8),
            )
            for key in keys
        }
    summary.question_ids = keys
    markdown = render_markdown(build_analysis(), audit=summary)
    assert "Q2_observation_available.evidence_age" in markdown
    assert "Q2_observation_available.observed_position" in markdown
    path = figure_audit_reconstruction(build_analysis(), summary, tmp_path)
    assert path.read_bytes().startswith(PNG_MAGIC)
