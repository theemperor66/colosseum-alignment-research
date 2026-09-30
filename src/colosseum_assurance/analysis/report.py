"""Deterministic Markdown + JSON rendering of a :class:`RunAnalysis`.

Two properties matter more than looks:

* **Determinism.** The same analysis object renders byte-identically, so a report can be diffed across
  code versions and regenerated from saved data as a reproduction check. Nothing here reads the clock;
  a wall-clock stamp appears only when the caller supplies one.
* **No naked rates.** Every proportion is printed with its numerator, denominator, and interval, and an
  undefined value prints the machine-readable reason instead of a zero. Fixture and smoke runs carry a
  banner on the first content line so a synthetic table can never be quoted as experimental evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from colosseum_assurance.analysis.metrics import ArmMetrics, RunAnalysis
from colosseum_assurance.analysis.stats import (
    BootstrapEstimate,
    DistributionSummary,
    ProportionEstimate,
)
from colosseum_assurance.schemas import ANCHORED_LIVE_PROVENANCES

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps report.py runnable without the audit lane
    from colosseum_assurance.audit.scoring import AuditScoreSummary

REPORT_SCHEMA_VERSION = "2.0.0"
_EXTRA_CONTROL_KEYS = {"audit", "extra_limits", "limits", "basename"}
SYNTHETIC_BANNER = "SYNTHETIC FIXTURE DATA - NOT EXPERIMENTAL EVIDENCE"
PENDING_BANNER = "RESULTS PENDING - no live-simulator episodes are included in this run"

__all__ = [
    "PENDING_BANNER",
    "REPORT_SCHEMA_VERSION",
    "SYNTHETIC_BANNER",
    "render_json",
    "render_markdown",
    "write_report",
]


# --------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------
def _json_safe(value: Any) -> Any:
    """Convert pydantic models and paths to JSON-safe values without losing them silently."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return repr(value)


def _audit_payload(audit: AuditScoreSummary | None) -> Any:
    if audit is None:
        return None
    return _json_safe(audit)


def _prop(p: ProportionEstimate) -> str:
    """Render a proportion so the denominator is never lost."""
    if p.point is None:
        return f"undefined (n={p.denominator}; {p.undefined_reason or 'no denominator'})"
    interval = (f" [{p.ci_low:.3f}, {p.ci_high:.3f}]"
                if p.ci_low is not None and p.ci_high is not None else " [interval unavailable]")
    clusters = f", scenarios={p.n_units}" if hasattr(p, "n_units") else ""
    return f"{p.point:.3f}{interval} (k={p.numerator}/n={p.denominator}{clusters})"


def _attempted(m: ArmMetrics) -> str:
    """Render the attempted denominator, or say why it could not be established.

    An unavailable attempted denominator is printed as such. Substituting the assessed or completed
    count here would turn "we do not know how many runs this cell had" into a confident denominator.
    """
    if m.n_attempted is None:
        return f"unavailable ({m.attempted_denominator_reason or 'no reason recorded'})"
    if m.attempted_denominator_source == "assessed_outcomes":
        return f"{m.n_attempted} (assessed episodes; no attempted-run ledger was supplied)"
    return str(m.n_attempted)


def _dist(d: DistributionSummary) -> str:
    if d.median is None:
        return f"undefined (n=0, missing={d.n_missing}; {d.undefined_reason or 'no values'})"
    return f"{d.median:.2f} [IQR {d.q1:.2f}-{d.q3:.2f}] (n={d.n}, missing={d.n_missing})"


def _boot(b: BootstrapEstimate) -> str:
    if b.point is None:
        return f"undefined ({b.undefined_reason or 'no units'})"
    if b.ci_low is None or b.ci_high is None:
        return f"{b.point:+.3f} (n={b.n_units}, no interval)"
    return f"{b.point:+.3f} [{b.ci_low:+.3f}, {b.ci_high:+.3f}] (n={b.n_units} {b.unit}s)"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    return out


# --------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------
def _provenance_section(a: RunAnalysis) -> list[str]:
    code = ", ".join(f"{k}={v}" for k, v in sorted(a.code_version.items())) or "unrecorded"
    lines = [
        "## Provenance",
        "",
        f"- Run class: `{a.run_class}`",
        f"- Protocol hash: `{a.protocol_hash}` (short `{a.protocol_short_hash}`, label `{a.protocol_label}`)",
        f"- Policy version: `{a.policy_version}`",
        f"- Evaluator versions: {', '.join(f'`{v}`' for v in a.evaluator_versions) or 'unrecorded'}",
        f"- Analysis version: `{a.analysis_version}` (schema `{a.schema_version}`)",
        f"- Code version: {code}",
        f"- Simulator provenance: {', '.join(f'`{p}`' for p in a.simulator_provenance) or 'unrecorded'}",
        f"- Statistical unit: `{a.statistical_unit}`",
        f"- Confidence level: {a.confidence_level:.2f}; bootstrap resamples: {a.bootstrap_resamples}; "
        f"seed: {a.bootstrap_seed}",
        f"- Strata: {', '.join(f'`{s}`' for s in a.strata_used) or 'none'}",
        f"- Episode outcomes analysed: {a.n_episode_outcomes} over {a.n_scenarios} scenario realizations "
        f"and {len(a.arm_ids)} arms",
    ]
    if a.n_attempted_runs is not None:
        lines.append(f"- Attempted runs in the ledger: {a.n_attempted_runs}")
    else:
        lines.append(
            "- Attempted runs in the ledger: **not supplied**. Denominators below count assessed "
            "episodes only, so runs that never produced an outcome are invisible in them."
        )
    lines += _protocol_source_lines(a)
    inventory = a.attempted_inventory
    if inventory is not None:
        lines.append(
            f"- Scenario realizations attempted: {len(inventory.scenario_ids)}, of which "
            f"{a.n_scenarios} produced at least one assessed outcome"
        )
    if inventory is not None and inventory.unresolved_scenario_ids:
        lines.append(
            f"- Attempted runs whose stratum membership could not be established: "
            f"{sum(inventory.unresolved_by_arm.values())} over "
            f"{len(inventory.unresolved_scenario_ids)} scenario realizations without a manifest or an "
            "outcome. Their per-stratum denominators are reported as unavailable, not as zero."
        )
    if a.arms_without_outcomes:
        lines.append(
            "- Arms attempted but never scored: "
            + ", ".join(f"`{arm}`" for arm in a.arms_without_outcomes)
            + ". They stay in every table below with zero coverage and undefined rates."
        )
    if a.generated_at_wall_clock:
        lines.append(f"- Generated at: {a.generated_at_wall_clock}")
    if a.shared_realizations_across_cells:
        lines.append("- Shared realizations across cells: yes (dependency cluster larger than one cell)")
    lines.append("")
    return lines


def _protocol_source_lines(a: RunAnalysis) -> list[str]:
    """Name the protocol that actually scored this run, including any fallback warning.

    ``colassure analyze`` used to fall back to current defaults without saying so, which let a report
    describe a specification the run never saw. The provenance record from
    ``runtime.evidence.load_run_protocol`` is therefore printed, not summarised away.
    """
    provenance = a.protocol_provenance
    if not provenance:
        return [
            "- Protocol source: **unrecorded**. This report cannot name the protocol file that scored "
            "the run; treat its protocol hash as the only evidence."
        ]
    lines = [f"- Protocol source: `{provenance.get('source', 'unrecorded')}`"]
    for key in ("path", "protocol_hash", "recorded_protocol_hash", "written_utc"):
        value = provenance.get(key)
        if value:
            lines.append(f"  - {key}: `{value}`")
    note = provenance.get("note")
    if note:
        lines.append(f"  - note: {note}")
    warning = provenance.get("warning")
    if warning:
        lines.append(f"  - **warning**: {warning}")
    return lines


def _counts_section(a: RunAnalysis) -> list[str]:
    rows = []
    for arm_id in a.arm_ids:
        m = a.overall[arm_id]
        incomplete = ", ".join(f"{k}={v}" for k, v in sorted(m.incomplete_reasons.items())) or "-"
        undecidable = ", ".join(f"{k}={v}" for k, v in sorted(m.obligation_unknown_counts.items())) or "-"
        without = (
            "unavailable" if m.n_attempted_without_outcome is None
            else str(m.n_attempted_without_outcome)
        )
        rows.append([
            f"`{arm_id}`", "yes" if m.has_monitor else "no", f"`{m.monitor_id or 'none'}`",
            _attempted(m), str(m.n_assessed), str(m.n_complete), str(m.n_incomplete),
            without, incomplete, undecidable,
        ])
    return [
        "## Episode counts",
        "",
        "Incomplete episodes stay in the denominator of the all-episode outcomes. They are not safety",
        "evidence and are never silently dropped. An UNDECIDABLE obligation (for example a truth trace",
        "with a gap larger than the permitted sampling gap) is an UNKNOWN verdict, never a pass.",
        "",
        "The attempted column comes from the attempted-run ledger, which is the only record of a run",
        "that crashed before it produced an outcome. An arm with attempts and no outcomes is listed",
        "here with zero assessed episodes; it is not removed from the study.",
        "",
        *_table(
            ["arm", "monitor", "monitor id", "attempted", "assessed", "complete", "incomplete",
             "attempted without outcome", "incomplete reasons", "undecidable obligations"],
            rows,
        ),
        "",
    ]


def _outcomes_section(a: RunAnalysis) -> list[str]:
    rows = []
    for arm_id in a.arm_ids:
        m = a.overall[arm_id]
        rows.append([
            f"`{arm_id}`", str(m.n_assessed), _prop(m.physical_violation), _prop(m.procedural_violation),
            _prop(m.any_violation), _prop(m.safe_mission_completion), _prop(m.mission_completion),
        ])
    sens = []
    for arm_id in a.arm_ids:
        m = a.overall[arm_id]
        sens.append([
            f"`{arm_id}`", str(m.n_complete), _prop(m.physical_violation_complete_only),
            _prop(m.procedural_violation_complete_only), _prop(m.unknown_verdict_fraction),
        ])
    return [
        "## All-episode outcomes (denominator: every assessed episode)",
        "",
        "Physical and procedural violations are reported separately, as the protocol requires.",
        "",
        *_table(
            ["arm", "n assessed", "physical violation", "procedural violation", "any violation",
             "safe mission completion", "mission completion"],
            rows,
        ),
        "",
        "### Sensitivity: complete episodes only",
        "",
        *_table(
            ["arm", "n complete", "physical violation", "procedural violation", "unknown episode verdict"],
            sens,
        ),
        "",
    ]


def _assurance_section(a: RunAnalysis) -> list[str]:
    rows = []
    ascertainment_rows = []
    for arm_id in a.arm_ids:
        m = a.overall[arm_id]
        rows.append([
            f"`{arm_id}`", _attempted(m), str(m.n_complete), str(m.n_accepted),
            _prop(m.assurance_coverage), _prop(m.assurance_coverage_complete_only),
            _prop(m.conditional_false_assurance), str(m.n_violation_episodes),
            _prop(m.missed_detection), _dist(m.detection_delay_s),
        ])
        bounds = m.false_assurance_bounds
        rendered_bounds = (
            f"[{bounds.lower:.3f}, {bounds.upper:.3f}]" if bounds.lower is not None else "undefined"
        )
        ascertainment_rows.append([
            f"`{arm_id}`", str(bounds.accepted), str(bounds.ascertainable), str(bounds.unresolved),
            _prop(m.ascertainable_false_assurance), rendered_bounds,
        ])
    return [
        "## Assurance coverage, conditional false assurance, missed detection",
        "",
        "### Acceptance rule",
        "",
        f"`{a.acceptance_rule_id}`",
        "",
        "> " + a.acceptance_rule_text.replace("\n", " "),
        "",
        *_table(
            ["arm", "n attempted", "n complete", "n accepted",
             "assurance coverage (all attempted)", "coverage (complete only)",
             "conditional false assurance", "n violation episodes", "missed detection",
             "detection delay (s)"],
            rows,
        ),
        "",
        "Headline assurance coverage uses **every attempted episode** as its denominator. An arm that",
        "fails to finish an episode did not provide assurance for it either, so a run with one accepted",
        "episode and nine crashed ones has 10 percent coverage, not 100 percent. The",
        "complete-conditional column is the same numerator over completed episodes only and is labelled",
        "as such.",
        "",
        "An arm without a monitor has **undefined** acceptance, not zero acceptance. A conditional rate",
        "over an empty accepted set is undefined and is printed as such.",
        "",
        f"False-assurance semantics: `{a.false_assurance_semantics}`. Monitor acceptance is retained when",
        "independent truth is UNKNOWN, but that episode is not counted as a known non-violation.",
        "The full accepted-set rate is undefined whenever accepted truth is unresolved. The following",
        "subset rate uses only ascertainable accepted episodes and may describe a selected population.",
        "Bounds assign every unresolved episode first to non-violation, then to violation; they are",
        "**identification bounds from missing evidence, not sampling confidence intervals**.",
        "",
        *_table(["arm", "all accepted", "ascertainable accepted", "unresolved accepted",
                 "false assurance in ascertainable subset", "full-set identification bounds"],
                ascertainment_rows),
        "",
    ]


def _burden_section(a: RunAnalysis) -> list[str]:
    rows = []
    for arm_id in a.arm_ids:
        m = a.overall[arm_id]
        rows.append([
            f"`{arm_id}`", _dist(m.interventions_per_episode), str(m.total_interventions),
            _prop(m.suspension_fraction), _prop(m.abandonment_fraction), _dist(m.completion_time_s),
        ])
    return [
        "## Intervention burden",
        "",
        "Availability cost of each guard. Read it beside the safety numbers: an arm that suspends or",
        "abandons everything cannot be called safer on a conditional rate alone.",
        "",
        *_table(
            ["arm", "interventions per episode", "total interventions", "suspension fraction",
             "abandonment fraction", "completion time (s)"],
            rows,
        ),
        "",
    ]


def _measurements_section(a: RunAnalysis) -> list[str]:
    exposure_rows, latency_rows, recovery_rows, proxy_rows = [], [], [], []
    for arm_id in a.arm_ids:
        m = a.overall[arm_id].measurements
        exposure_rows.append([
            f"`{arm_id}`", str(m.n_measured_episodes), str(m.n_episodes_without_measurements),
            _dist(m.outside_duration_s), _boot(m.mean_outside_duration_s),
            f"{m.unobserved_duration_s:.3f}",
        ])
        for metric, summary in m.latencies.items():
            latency_rows.append([
                f"`{arm_id}`", f"`{metric}`", str(summary.n_windows),
                json.dumps(summary.status_counts, sort_keys=True),
                _dist(summary.observed_episode_means_s),
                _boot(summary.mean_of_observed_episode_means_s),
                json.dumps(summary.source_counts, sort_keys=True),
            ])
        recovery_rows.append([
            f"`{arm_id}`", str(m.n_fault_episodes), json.dumps(m.recovery_status_counts, sort_keys=True),
            _boot(m.mean_recovery_episode_success_fraction), _dist(m.recovery_episode_mean_latency_s),
            json.dumps(m.recovery_unavailable_reasons, sort_keys=True),
        ])
        proxy_rows.append([
            f"`{arm_id}`", str(m.proxy_episodes), str(m.proxy_unresolved),
            _dist(m.proxy_nonempty_captures), _dist(m.proxy_qualifying_authorized_captures),
            _dist(m.proxy_redundant_captures), _prop(m.proxy_mismatch),
        ])
    return [
        "## Sampled envelope exposure, response timing, and fault recovery", "",
        "Geofence exposure is a left-held estimate over truth intervals within the frozen maximum gap.",
        "Collision flags can be latched, so they do not establish contact duration and are excluded.",
        "It is not continuous flight duration: between-sample excursions remain invisible. A complete",
        "episode estimate is unavailable when any horizon interval is unobserved. Partial observed",
        "exposure is retained in JSON; missing measurement records are never replaced by zero.", "",
        *_table(["arm", "measured episodes", "without measurement", "full-horizon estimate (s)",
                 "mean across scenarios (s)", "unobserved horizon time (s)"], exposure_rows), "",
        "Every request and loss window retains its observation/censoring status. Latency summaries",
        "first average observed windows within each episode, then treat scenario realizations as the",
        "replicates. They exclude censored/unavailable latency values and are not uncensored population",
        "means. Synthetic supervisor response is not a measurement of human reaction time.", "",
        *_table(["arm", "timing phase", "windows", "statuses", "observed episode means (s)",
                 "mean across scenarios (s)", "sources"], latency_rows), "",
        "Recovery requires geofence compliance at every sample and no new nonexempt collision over",
        "the frozen hold after a recorded fault end. It does not establish continuous contact cessation.",
        "Maintained safety, observed recovery after a breach, failure, censoring, and unknown evidence",
        "are separate outcomes. Success fractions use only resolved windows within each episode;",
        "unknown and censored windows remain in the status counts. This neither establishes sensor",
        "repair or mission resumption nor attributes an excursion causally to an injected fault.", "",
        *_table(["arm", "fault episodes", "window statuses", "mean resolved success fraction",
                 "observed recovery episode means (s)", "unavailable reasons"], recovery_rows), "",
        "### Bounded planning proxy versus inspection intent", "",
        "The proxy is the independently recorded count of nonempty inspection captures. Intent also",
        "requires qualifying geometry, dwell, return and conformance. A proxy-satisfied but failed",
        "intent episode is a descriptive mismatch in this bounded planner; it does not establish",
        "learned reward hacking. Unknown evidence remains unresolved and leaves the scored subset.", "",
        *_table(["arm", "proxy episodes", "unresolved", "raw nonempty captures",
                 "qualifying authorized captures", "redundant raw captures", "proxy/intent mismatch"],
                proxy_rows), "",
    ]


def _comparison_block(title: str, comparisons: list[Any]) -> list[str]:
    if not comparisons:
        return [f"## {title}", "", "Not computed for this run.", ""]
    rows = []
    for c in comparisons:
        mc = c.mcnemar
        p_text = "undefined" if mc.p_value is None else f"{mc.p_value:.4f}"
        rows.append([
            f"`{c.first_arm}` vs `{c.second_arm}`", f"`{c.outcome_id}`", c.role, str(c.n_paired),
            f"{c.n_first_only}/{c.n_second_only}", _prop(c.first_rate), _prop(c.second_rate),
            _boot(c.difference), f"b={mc.b}, c={mc.c}, p={p_text}",
        ])
    lines = [
        f"## {title}",
        "",
        "Only scenario realizations present in **both** arms enter the paired test. The "
        "`unpaired first/second` column counts realizations dropped because the other arm has no "
        "assessed outcome for them.",
        "",
        *_table(
            ["arms", "outcome", "role", "n paired", "unpaired first/second", "first rate", "second rate",
             "paired difference (first - second)", "exact McNemar"],
            rows,
        ),
        "",
    ]
    for c in comparisons:
        if c.note:
            lines.append(f"- `{c.comparison_id}`: {c.note}")
        if c.unpaired_scenario_ids:
            shown = ", ".join(f"`{s}`" for s in c.unpaired_scenario_ids[:10])
            more = "" if len(c.unpaired_scenario_ids) <= 10 else f" (+{len(c.unpaired_scenario_ids) - 10})"
            lines.append(f"  - dropped, unpaired: {shown}{more}")
    lines.append("")
    return lines


def _shift_section(a: RunAnalysis) -> list[str]:
    if not a.shift_comparisons:
        return []
    rows = []
    for s in a.shift_comparisons:
        rows.append([
            f"`{s.first_arm}` vs `{s.second_arm}`", f"`{s.outcome_id}`", str(s.n_paired_scenarios),
            str(s.n_paired_with_values), _dist(s.first_distribution), _dist(s.second_distribution),
            _boot(s.shift),
        ])
    return [
        "## Paired shifts for continuous outcomes",
        "",
        "Hodges-Lehmann paired shift with a percentile bootstrap interval over scenario realizations.",
        "",
        *_table(
            ["arms", "outcome", "n paired scenarios", "n pairs with both values", "first", "second",
             "shift (first - second)"],
            rows,
        ),
        "",
    ]


def _stratum_section(a: RunAnalysis) -> list[str]:
    if not a.by_stratum:
        return []
    lines = [
        "## Per-stratum outcomes",
        "",
        "Per-stratum assurance coverage uses the attempted runs of that stratum, taken from the ledger",
        "and placed by the run's scenario manifests. Where membership cannot be established the",
        "attempted denominator is printed as unavailable: the completed-only count is a different",
        "quantity and using it in its place would inflate the coverage of every failed cell.",
        "",
    ]
    for kind in ("cell", "observation_delay_level", "supervision_delay_level"):
        rows = [
            [f"`{m.stratum_key}`", f"`{m.arm_id}`", _attempted(m), str(m.n_assessed), str(m.n_complete),
             _prop(m.physical_violation), _prop(m.procedural_violation),
             _prop(m.assurance_coverage), _prop(m.conditional_false_assurance)]
            for m in a.by_stratum if m.stratum_kind == kind
        ]
        if not rows:
            continue
        lines += [
            f"### By `{kind}`",
            "",
            *_table(
                [kind, "arm", "n attempted", "n assessed", "n complete", "physical violation",
                 "procedural violation", "assurance coverage", "conditional false assurance"],
                rows,
            ),
            "",
        ]
    return lines


def _audit_section(audit: AuditScoreSummary | None) -> list[str]:
    if audit is None:
        return [
            "## Offline audit reconstruction",
            "",
            "Not included in this report (no audit scoring supplied).",
            "",
        ]
    rows: list[list[str]] = []
    for variant_id in audit.variant_ids:
        variant = audit.variants[variant_id]
        for question_id in audit.question_ids:
            q = variant.per_question.get(question_id)
            if q is None:
                continue
            rows.append([
                f"`{variant_id}`", f"`{question_id}`", str(q.n_episodes), str(q.correct),
                str(q.incorrect_confident), str(q.insufficient_evidence),
                str(getattr(q, "reference_unavailable", 0)), _prop(q.correct_rate),
            ])
    rows.sort()
    overall_rows = []
    for v in audit.variant_ids:
        variant = audit.variants[v]
        # A variant that keeps a redundant recovery route did not remove the information. Saying so is
        # the difference between an honest redundancy probe and a false ablation claim.
        kind = ("redundancy probe (information retained by another route)"
                if getattr(variant, "retains_redundant_routes", False) else "information removed")
        overall_rows.append([
            f"`{v}`", kind, str(variant.n_episodes),
            ", ".join(f"`{f}`" for f in variant.removed_fields) or "none (full record)",
            _boot(variant.overall_correct_rate),
        ])
    return [
        "## Offline audit reconstruction",
        "",
        "Reconstruction is scored per episode and question. Insufficient evidence is a third outcome, not",
        "an error. Offline record ablation cannot change flight safety; it changes only what an auditor",
        "can establish afterwards.",
        "",
        *_table(["record variant", "variant kind", "n episodes", "removed fields",
                 "overall correct (cluster bootstrap)"], overall_rows),
        "",
        *_table(
            ["record variant", "question", "n episodes", "correct", "incorrect confident",
             "insufficient evidence", "reference unavailable", "correct rate"],
            rows,
        ),
        "",
    ]


# --------------------------------------------------------------------------------------
# Public rendering
# --------------------------------------------------------------------------------------
def _extra_section(extra: dict[str, Any] | None) -> list[str]:
    """Render caller-supplied run context verbatim, so nothing is quietly dropped from the report."""
    shown = {k: v for k, v in sorted((extra or {}).items()) if k not in _EXTRA_CONTROL_KEYS}
    if not shown:
        return []
    lines = ["## Run context supplied by the runner", ""]
    for key, value in shown.items():
        rendered = json.dumps(value, sort_keys=True) if isinstance(value, dict | list) else str(value)
        lines.append(f"- `{key}`: {rendered}")
    lines.append("")
    return lines


def render_markdown(
    analysis: RunAnalysis,
    *,
    audit: AuditScoreSummary | None = None,
    extra_limits: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Render the full Markdown report. Deterministic for a given analysis object."""
    lines: list[str] = [
        f"# Run analysis - {analysis.run_class} - protocol {analysis.protocol_short_hash}",
        "",
    ]
    if analysis.is_synthetic:
        lines += [
            f"> **{SYNTHETIC_BANNER}**",
            ">",
            f"> This report was generated from `run_class={analysis.run_class}` records. They exist to "
            "exercise the software. They are not measurements of any simulated flight and must not be "
            "quoted as results.",
            "",
        ]
    if all(p not in ANCHORED_LIVE_PROVENANCES for p in analysis.simulator_provenance):
        lines += [f"> **{PENDING_BANNER}.** Recorded provenance: "
                  f"{', '.join(f'`{p}`' for p in analysis.simulator_provenance) or '`unrecorded`'}.", ""]

    lines += _provenance_section(analysis)
    lines += ["## Primary outcome", "", "> " + analysis.primary_outcome_text.replace("\n", " "), ""]
    lines += _counts_section(analysis)
    lines += _outcomes_section(analysis)
    lines += _assurance_section(analysis)
    lines += _burden_section(analysis)
    lines += _measurements_section(analysis)
    lines += _comparison_block("Primary paired comparison", list(analysis.primary_comparisons))
    lines += _comparison_block("Context comparisons (unguarded arm)", list(analysis.context_comparisons))
    lines += _shift_section(analysis)
    lines += _stratum_section(analysis)
    lines += _audit_section(audit)
    lines += _extra_section(extra)

    lines += ["## Warnings", ""]
    if analysis.warnings:
        lines += [f"- {w}" for w in analysis.warnings]
    else:
        lines.append("- none recorded")
    lines.append("")

    lines += ["## How to read these numbers", ""]
    lines += [f"- {n}" for n in analysis.interpretation_notes]
    for extra in extra_limits or []:
        lines.append(f"- {extra}")
    lines.append("")
    return "\n".join(lines)


def render_json(
    analysis: RunAnalysis,
    *,
    audit: AuditScoreSummary | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Render the machine-readable report. Sorted keys, stable indentation."""
    payload: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "banner": SYNTHETIC_BANNER if analysis.is_synthetic else None,
        "analysis": analysis.model_dump(mode="json"),
        "audit": _audit_payload(audit),
        "extra": _json_safe(extra) if extra else None,
    }
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def write_report(
    analysis: RunAnalysis,
    out_dir: Path | str,
    extra: dict[str, Any] | None = None,
    *,
    audit: AuditScoreSummary | None = None,
    basename: str = "analysis-report",
    extra_limits: list[str] | None = None,
) -> dict[str, str]:
    """Write ``<basename>.md`` and ``<basename>.json`` into ``out_dir``; return ``{kind: path}``.

    ``extra`` is free-form run context from the runner. Recognised control keys are ``audit`` (an
    audit score summary), ``limits`` / ``extra_limits`` (extra caveat lines), and ``basename``.
    Anything else is rendered verbatim in its own section instead of being discarded.
    """
    extra = dict(extra or {})
    audit = audit if audit is not None else extra.get("audit")
    limits = list(extra_limits or []) + list(extra.get("extra_limits") or extra.get("limits") or [])
    basename = str(extra.get("basename", basename))
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    md_path = directory / f"{basename}.md"
    json_path = directory / f"{basename}.json"
    md_path.write_text(
        render_markdown(analysis, audit=audit, extra_limits=limits, extra=extra), encoding="utf-8"
    )
    json_path.write_text(render_json(analysis, audit=audit, extra=extra), encoding="utf-8")
    return {"markdown": str(md_path), "json": str(json_path)}
