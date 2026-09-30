"""Episode-level metrics: every rate carries its denominator, every undefined value says why.

Design rules that this module enforces rather than documents (research-acceptance.md section 4):

* **The statistical unit is the scenario realization.** Outcomes arrive one per (arm, scenario). Steps,
  frames, and monitor decisions never enter a denominator here.
* **The primary comparison is paired.** A scenario contributes to the A1-vs-A2 comparison only when
  both arms produced an assessed outcome for it. Scenarios present in one arm only are dropped from
  the paired test and counted as unpaired, never quietly imputed.
* **A conditional rate over an empty accepted set is undefined.** ``None`` with an
  ``undefined_reason``, never ``0.0``. An arm with no monitor has *undefined* acceptance, not zero
  acceptance.
* **Coverage travels with every conditional rate.** An always-abstain or always-abort configuration
  therefore cannot look good: its conditional false-assurance rate is undefined or rests on a tiny
  accepted set, and its assurance coverage and mission completion are printed next to it.
* **Mixing run classes or protocol hashes is an error.** Fixture, pilot, and held-out records cannot be
  pooled by accident.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from colosseum_assurance.analysis.measurements import MeasurementSummary, summarize_measurements
from colosseum_assurance.analysis.stats import (
    BootstrapEstimate,
    DistributionSummary,
    McNemarResult,
    ProportionEstimate,
    mcnemar_exact,
    paired_bootstrap_difference,
    paired_shift_estimate,
    summarize_distribution,
    wilson_interval,
)
from colosseum_assurance.evaluation.outcomes import EpisodeOutcome
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.schemas import ANCHORED_LIVE_PROVENANCES, AttemptedRun, StrictModel, Verdict

ANALYSIS_VERSION = "analysis-v2.0.0"

# Machine-readable reasons for an undefined quantity. Report and figures switch on these strings.
REASON_NO_EPISODES = "no_episodes_in_group"
REASON_NO_COMPLETE = "no_complete_episodes"
REASON_NO_MONITOR = "arm_has_no_monitor_acceptance_undefined"
REASON_NO_ACCEPTED = "no_accepted_episodes"
REASON_UNKNOWN_ACCEPTED = "accepted_episodes_have_unascertainable_independent_truth"
REASON_NO_VIOLATIONS = "no_violation_episodes"
REASON_NO_PAIRS = "no_scenarios_present_in_both_arms"
REASON_NO_VALUES = "no_observed_values"
# Attempted-denominator reasons. An attempted denominator that cannot be established must say so; the
# completed-only count is a different quantity and silently substituting it inflates every coverage.
REASON_ATTEMPTED_MEMBERSHIP_UNKNOWN = "attempted_stratum_membership_unresolved"
REASON_ATTEMPTED_LEDGER_SHORTER = "attempted_ledger_holds_fewer_runs_than_assessed_outcomes"

# Stratum kind -> the ``EpisodeOutcome``/manifest attribute that carries its membership.
STRATUM_KINDS: tuple[str, ...] = ("cell", "observation_delay_level", "supervision_delay_level")
_OUTCOME_STRATUM_ATTR = {
    "cell": "cell_id",
    "observation_delay_level": "observation_delay_level",
    "supervision_delay_level": "supervision_delay_level",
}

CONDITIONAL_RATE_CAVEAT = (
    "Conditional false-assurance rates for two arms are computed over differently selected accepted "
    "sets. Their difference is not an estimate of a causal safety effect. The primary comparison is the "
    "paired all-episode violation difference; acceptance coverage is reported beside every conditional "
    "rate."
)

__all__ = [
    "ANALYSIS_VERSION",
    "CONDITIONAL_RATE_CAVEAT",
    "REASON_ATTEMPTED_LEDGER_SHORTER",
    "REASON_ATTEMPTED_MEMBERSHIP_UNKNOWN",
    "STRATUM_KINDS",
    "AnalysisInputError",
    "ArmMetrics",
    "AttemptedInventory",
    "PairedComparison",
    "PairedShiftComparison",
    "RunAnalysis",
    "analysis_to_dict",
    "analyze_run",
    "build_attempted_inventory",
    "compute_run_analysis",
    "load_outcomes",
    "load_scenario_strata",
]


class AnalysisInputError(ValueError):
    """Raised when the outcome set cannot be analysed as one coherent run."""


class FalseAssuranceBounds(StrictModel):
    """Identification bounds due to unresolved truth, NOT a sampling confidence interval."""

    accepted: int = 0
    ascertainable: int = 0
    observed_violations: int = 0
    unresolved: int = 0
    lower: float | None = None
    upper: float | None = None
    semantics: str = "ascertainable_v2"
    interpretation: str = (
        "Lower = observed violations / all accepted; upper adds every unresolved accepted episode. "
        "These are missing-evidence identification bounds, not confidence limits."
    )


# --------------------------------------------------------------------------------------
# Per-arm, per-stratum metrics
# --------------------------------------------------------------------------------------
class ArmMetrics(StrictModel):
    """All metrics for one arm inside one stratum. Counts first, then rates with denominators."""

    arm_id: str
    stratum_kind: Literal["all", "cell", "observation_delay_level", "supervision_delay_level"]
    stratum_key: str
    has_monitor: bool
    monitor_id: str | None = None

    n_attempted: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Attempted runs for this arm in this stratum. ``None`` means the attempted denominator "
            "could not be established here; ``attempted_denominator_reason`` says why. It is never the "
            "completed-only count in disguise."
        ),
    )
    attempted_denominator_source: Literal["attempted_ledger", "assessed_outcomes", "unavailable"] = Field(
        default="assessed_outcomes",
        description=(
            "``attempted_ledger``: counted from the typed attempted-run ledger. ``assessed_outcomes``: "
            "no ledger was supplied, so assessed episodes are the denominator and runs that never "
            "produced an outcome are invisible. ``unavailable``: the ledger exists but does not place "
            "these runs in this stratum."
        ),
    )
    attempted_denominator_reason: str | None = None
    n_assessed: int = Field(ge=0, description="Episodes with an independent evaluator outcome.")
    n_complete: int = Field(ge=0)
    n_incomplete: int = Field(ge=0)
    n_attempted_without_outcome: int | None = Field(default=0, ge=0)
    incomplete_reasons: dict[str, int] = Field(default_factory=dict)
    termination_reasons: dict[str, int] = Field(default_factory=dict)
    episode_verdicts: dict[str, int] = Field(default_factory=dict)

    # All-episode outcomes. Denominator is every assessed episode, complete or not: an incomplete
    # episode is not safety evidence and must not be dropped from the denominator silently.
    physical_violation: ProportionEstimate
    procedural_violation: ProportionEstimate
    any_violation: ProportionEstimate
    physical_violation_complete_only: ProportionEstimate
    procedural_violation_complete_only: ProportionEstimate

    mission_completion: ProportionEstimate
    safe_mission_completion: ProportionEstimate

    n_accepted: int = Field(default=0, ge=0)
    assurance_coverage: ProportionEstimate = Field(
        description=(
            "Headline coverage: accepted episodes over ALL ATTEMPTED episodes. An arm that crashes nine "
            "episodes and accepts the tenth has 10% coverage, not 100%."
        )
    )
    assurance_coverage_complete_only: ProportionEstimate = Field(
        description="Secondary, complete-conditional coverage: accepted over completed episodes."
    )
    conditional_false_assurance: ProportionEstimate
    ascertainable_false_assurance: ProportionEstimate = Field(
        default_factory=lambda: ProportionEstimate(numerator=0, denominator=0, method="none")
    )
    false_assurance_bounds: FalseAssuranceBounds = Field(default_factory=FalseAssuranceBounds)
    accepted_with_undefined_false_assurance: int = Field(default=0, ge=0)

    n_violation_episodes: int = Field(default=0, ge=0)
    missed_detection: ProportionEstimate
    detection_delay_s: DistributionSummary

    interventions_per_episode: DistributionSummary
    total_interventions: int = Field(default=0, ge=0)
    suspension_fraction: ProportionEstimate
    abandonment_fraction: ProportionEstimate
    completion_time_s: DistributionSummary

    unknown_verdict_fraction: ProportionEstimate
    obligation_unknown_counts: dict[str, int] = Field(
        default_factory=dict,
        description="obligation_id -> episodes whose independent verdict for it was UNKNOWN.",
    )
    unknown_reasons: dict[str, int] = Field(
        default_factory=dict,
        description="Machine-readable reasons the evaluator could not decide, with episode counts.",
    )
    truth_coverage: DistributionSummary
    measurements: MeasurementSummary = Field(default_factory=MeasurementSummary)
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# The attempted-run inventory: the only record of runs that produced no outcome
# --------------------------------------------------------------------------------------
class AttemptedInventory(StrictModel):
    """What the typed attempted-run ledger says, including where stratum membership is unknown.

    The ledger, not the outcome file, defines the attempted denominator: a crashed run leaves a ledger
    row and no outcome, so deriving arms or denominators from outcomes alone deletes exactly the
    failures the study must report. Stratum membership has to be looked up per scenario, because an
    ``AttemptedRun`` carries only ``scenario_id``. Where that lookup fails, the affected per-stratum
    denominators are marked unavailable rather than falling back to the completed-only count.
    """

    n_total: int = Field(ge=0)
    scenario_ids: list[str] = Field(
        default_factory=list,
        description="Every scenario realization that was attempted, whether or not it was ever scored.",
    )
    by_arm: dict[str, int] = Field(default_factory=dict)
    status_by_arm: dict[str, dict[str, int]] = Field(
        default_factory=dict, description="arm_id -> status -> attempts, so crashes stay named."
    )
    by_stratum: dict[str, dict[str, dict[str, int]]] = Field(
        default_factory=dict,
        description="stratum kind -> stratum key -> arm_id -> attempts with established membership.",
    )
    unresolved_by_arm: dict[str, int] = Field(
        default_factory=dict,
        description="Attempts whose scenario has neither a manifest nor an outcome, per arm.",
    )
    unresolved_scenario_ids: list[str] = Field(default_factory=list)
    membership_sources: dict[str, int] = Field(
        default_factory=dict, description="'manifest' / 'outcome' / 'unresolved' -> attempts."
    )
    conflicting_scenario_ids: list[str] = Field(
        default_factory=list,
        description="Scenarios whose manifest and outcomes disagree about stratum membership.",
    )

    def stratum_count(self, kind: str, key: str, arm_id: str) -> int:
        return self.by_stratum.get(kind, {}).get(key, {}).get(arm_id, 0)

    def stratum_keys(self, kind: str) -> set[str]:
        return set(self.by_stratum.get(kind, {}))


def _strata_from_outcomes(items: Sequence[EpisodeOutcome]) -> dict[str, dict[str, str]]:
    """Stratum membership implied by the outcomes themselves, for scenarios that produced one."""
    out: dict[str, dict[str, str]] = {}
    for outcome in items:
        out.setdefault(outcome.scenario_id, {
            kind: str(getattr(outcome, attr) or "unassigned")
            for kind, attr in _OUTCOME_STRATUM_ATTR.items()
        })
    return out


def load_scenario_strata(run_dir: Path | str) -> dict[str, dict[str, str]]:
    """Read stratum membership for every scenario realization from the run's own manifests.

    Manifests are written before an episode runs, so they are the only membership source that survives
    a crashed attempt. A malformed manifest raises instead of being skipped: a silently dropped
    manifest would turn a known cell into an unresolved one and quietly weaken every per-cell claim.
    """
    from colosseum_assurance.scenario.manifest import ScenarioManifest

    directory = Path(run_dir) / "manifests"
    if not directory.is_dir():
        return {}
    out: dict[str, dict[str, str]] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            manifest = ScenarioManifest.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - re-raised with the path, which the caller needs
            raise AnalysisInputError(f"{path} is not a valid scenario manifest: {exc}") from exc
        out[manifest.scenario_id] = {
            "cell": manifest.cell_id,
            "observation_delay_level": manifest.observation_delay_level,
            "supervision_delay_level": manifest.supervision_delay_level,
        }
    return out


def build_attempted_inventory(
    attempted: Sequence[AttemptedRun],
    items: Sequence[EpisodeOutcome],
    scenario_strata: Mapping[str, Mapping[str, str]] | None = None,
) -> AttemptedInventory:
    """Group the attempted ledger by arm and stratum, recording where membership could not be found.

    Membership is taken from the manifests first (they exist even when the episode never finished) and
    from the outcomes second. A scenario known to neither source stays unresolved and is counted, so
    its effect on the per-stratum denominators is visible instead of assumed away.
    """
    manifest_strata = dict(scenario_strata or {})
    outcome_strata = _strata_from_outcomes(items)
    conflicts = sorted(
        scenario_id
        for scenario_id, membership in outcome_strata.items()
        if scenario_id in manifest_strata
        and any(manifest_strata[scenario_id].get(k) != membership[k] for k in STRATUM_KINDS)
    )

    by_arm: Counter[str] = Counter()
    status_by_arm: dict[str, Counter[str]] = defaultdict(Counter)
    by_stratum: dict[str, dict[str, Counter[str]]] = {k: defaultdict(Counter) for k in STRATUM_KINDS}
    unresolved_by_arm: Counter[str] = Counter()
    unresolved_scenarios: set[str] = set()
    sources: Counter[str] = Counter()

    for run in attempted:
        by_arm[run.arm_id] += 1
        status_by_arm[run.arm_id][run.status] += 1
        membership = manifest_strata.get(run.scenario_id) or outcome_strata.get(run.scenario_id)
        if membership is None:
            unresolved_by_arm[run.arm_id] += 1
            unresolved_scenarios.add(run.scenario_id)
            sources["unresolved"] += 1
            continue
        sources["manifest" if run.scenario_id in manifest_strata else "outcome"] += 1
        for kind in STRATUM_KINDS:
            by_stratum[kind][str(membership.get(kind) or "unassigned")][run.arm_id] += 1

    return AttemptedInventory(
        n_total=len(attempted),
        scenario_ids=sorted({run.scenario_id for run in attempted}),
        by_arm=dict(sorted(by_arm.items())),
        status_by_arm={a: dict(sorted(c.items())) for a, c in sorted(status_by_arm.items())},
        by_stratum={
            kind: {key: dict(sorted(arms.items())) for key, arms in sorted(keys.items())}
            for kind, keys in by_stratum.items()
        },
        unresolved_by_arm=dict(sorted(unresolved_by_arm.items())),
        unresolved_scenario_ids=sorted(unresolved_scenarios),
        membership_sources=dict(sorted(sources.items())),
        conflicting_scenario_ids=conflicts,
    )


# --------------------------------------------------------------------------------------
# Paired comparisons
# --------------------------------------------------------------------------------------
class PairedComparison(StrictModel):
    """One paired binary comparison between two arms over shared scenario realizations."""

    comparison_id: str
    first_arm: str
    second_arm: str
    outcome_id: Literal[
        "physical_violation", "procedural_violation", "any_violation",
        "safe_mission_completion", "mission_completion",
    ]
    role: Literal["primary", "context", "secondary"]
    n_paired: int = Field(ge=0)
    n_first_only: int = Field(default=0, ge=0)
    n_second_only: int = Field(default=0, ge=0)
    unpaired_scenario_ids: list[str] = Field(default_factory=list)
    first_rate: ProportionEstimate
    second_rate: ProportionEstimate
    difference: BootstrapEstimate
    mcnemar: McNemarResult
    stratified_by: str | None = None
    note: str = ""


class PairedShiftComparison(StrictModel):
    """Paired Hodges-Lehmann shift for a continuous outcome (completion time, detection delay)."""

    comparison_id: str
    first_arm: str
    second_arm: str
    outcome_id: Literal["completion_time_s", "detection_delay_s"]
    n_paired_with_values: int = Field(ge=0)
    n_paired_scenarios: int = Field(ge=0)
    first_distribution: DistributionSummary
    second_distribution: DistributionSummary
    shift: BootstrapEstimate
    note: str = ""


# --------------------------------------------------------------------------------------
# Whole-run analysis
# --------------------------------------------------------------------------------------
class RunAnalysis(StrictModel):
    """The complete analysis of one run class. Serialized verbatim into the JSON report."""

    schema_version: str = "2.0.0"
    analysis_version: str = ANALYSIS_VERSION
    false_assurance_semantics: str = "ascertainable_v2"
    run_class: Literal["fixture", "pilot", "heldout", "smoke"]
    protocol_hash: str
    protocol_short_hash: str
    protocol_label: str = "draft"
    policy_version: str = ""
    evaluator_versions: list[str] = Field(default_factory=list)
    code_version: dict[str, str] = Field(default_factory=dict)
    simulator_provenance: list[str] = Field(default_factory=list)
    generated_at_wall_clock: str | None = Field(
        default=None, description="Left None by default so the report renders deterministically."
    )

    statistical_unit: Literal["scenario_realization"] = "scenario_realization"
    confidence_level: float = 0.95
    bootstrap_resamples: int = 0
    bootstrap_seed: int = 0
    acceptance_rule_id: str = ""
    acceptance_rule_text: str = ""
    primary_outcome_text: str = ""
    strata_used: list[str] = Field(default_factory=list)

    n_episode_outcomes: int = Field(default=0, ge=0)
    n_scenarios: int = Field(default=0, ge=0)
    n_attempted_runs: int | None = None
    attempts_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Status/arm counts derived from the attempted ledger, so crashes stay visible.",
    )
    attempted_inventory: AttemptedInventory | None = Field(
        default=None,
        description="Typed attempted-run inventory; None when no ledger was supplied to the analysis.",
    )
    protocol_provenance: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Where the protocol that scored this run came from (see "
            "``runtime.evidence.load_run_protocol``). A report must name the protocol that actually "
            "scored the run, including a fallback warning when one applies."
        ),
    )
    arms_without_outcomes: list[str] = Field(
        default_factory=list,
        description="Arms present in the attempted ledger that produced no assessed outcome at all.",
    )
    arm_ids: list[str] = Field(default_factory=list)
    cell_ids: list[str] = Field(default_factory=list)
    shared_realizations_across_cells: bool = False

    overall: dict[str, ArmMetrics] = Field(default_factory=dict)
    by_stratum: list[ArmMetrics] = Field(default_factory=list)
    primary_comparisons: list[PairedComparison] = Field(default_factory=list)
    context_comparisons: list[PairedComparison] = Field(default_factory=list)
    shift_comparisons: list[PairedShiftComparison] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
    interpretation_notes: list[str] = Field(default_factory=list)

    @property
    def is_synthetic(self) -> bool:
        """Fixture and smoke runs are engineering artefacts, never experimental evidence."""
        return self.run_class in {"fixture", "smoke"}

    def arm(self, arm_id: str) -> ArmMetrics:
        return self.overall[arm_id]

    def comparison(self, comparison_id: str) -> PairedComparison:
        for c in list(self.primary_comparisons) + list(self.context_comparisons):
            if c.comparison_id == comparison_id:
                return c
        raise KeyError(f"unknown comparison_id: {comparison_id}")


# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------
_OUTCOME_GETTERS = {
    "physical_violation": lambda o: bool(o.physical_violation),
    "procedural_violation": lambda o: bool(o.procedural_violation),
    "any_violation": lambda o: bool(o.any_violation),
    "safe_mission_completion": lambda o: bool(o.mission_completed_safely),
    "mission_completion": lambda o: bool(o.mission_completed),
}


def _validate_inputs(outcomes: Sequence[EpisodeOutcome], protocol: ProtocolConfig) -> str:
    """Refuse to pool records from different protocols or run classes."""
    if not outcomes:
        raise AnalysisInputError("no episode outcomes supplied; nothing can be analysed")
    hashes = sorted({o.protocol_hash for o in outcomes})
    if len(hashes) > 1:
        raise AnalysisInputError(
            f"episode outcomes carry {len(hashes)} different protocol hashes: {hashes}. "
            "Pilot, held-out, and fixture data must not be pooled."
        )
    classes = sorted({o.run_class for o in outcomes})
    if len(classes) > 1:
        raise AnalysisInputError(
            f"episode outcomes mix run classes {classes}. Analyse each run class separately."
        )
    expected = protocol.content_hash()
    if hashes[0] != expected:
        raise AnalysisInputError(
            f"episode outcomes were produced under protocol {hashes[0]} but the supplied protocol hashes "
            f"to {expected}. Analyse with the protocol that produced the run."
        )
    return classes[0]


def _check_duplicates(outcomes: Sequence[EpisodeOutcome]) -> None:
    seen: dict[tuple[str, str], list[str]] = defaultdict(list)
    for o in outcomes:
        seen[(o.arm_id, o.scenario_id)].append(o.episode_id)
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if dupes:
        listed = "; ".join(f"{arm}/{scen}: {sorted(eps)}" for (arm, scen), eps in sorted(dupes.items()))
        raise AnalysisInputError(
            "a scenario realization appears more than once in the same arm, which breaks the paired "
            f"statistical unit. Resolve the retry policy before analysing: {listed}"
        )


def _proportion(
    flags: Sequence[bool], confidence_level: float, *, reason: str
) -> ProportionEstimate:
    return wilson_interval(sum(1 for f in flags if f), len(flags), confidence_level,
                           undefined_reason=reason)


def _arm_metrics(
    arm_id: str,
    monitor_id: str | None,
    has_monitor: bool,
    outcomes: Sequence[EpisodeOutcome],
    *,
    stratum_kind: str,
    stratum_key: str,
    confidence_level: float,
    attempted_count: int | None,
    attempted_unavailable_reason: str | None = None,
    bootstrap_seed: int = 7717,
    bootstrap_resamples: int = 2000,
) -> ArmMetrics:
    """Compute one arm's metrics inside one stratum, refusing to invent a defined value.

    ``attempted_count`` is the number of attempted runs for this arm and stratum, taken from the typed
    ledger. ``None`` with an ``attempted_unavailable_reason`` means the ledger cannot place these runs
    here; the attempted denominator is then explicitly unavailable. ``None`` without a reason means no
    ledger was supplied at all, and the assessed episodes are used with that source recorded.
    """
    n = len(outcomes)
    complete = [o for o in outcomes if o.completeness == "complete"]
    incomplete = [o for o in outcomes if o.completeness != "complete"]
    warnings: list[str] = []

    # Acceptance. ``None`` means undefined; only an explicit True counts as accepted.
    accepted = [o for o in outcomes if o.accepted_by_monitor is True]
    undefined_acceptance = [o for o in outcomes if o.accepted_by_monitor is None]
    # Coverage denominator. The headline number counts every attempted episode, because an arm that
    # fails to produce an episode has not provided assurance for it either (independent review:
    # "headline coverage must use all attempted episodes"). The complete-conditional figure is kept
    # beside it with its own denominator and label.
    source: Literal["attempted_ledger", "assessed_outcomes", "unavailable"]
    attempted_reason = attempted_unavailable_reason
    if attempted_unavailable_reason is not None:
        attempted_denominator, source = None, "unavailable"
    elif attempted_count is None:
        attempted_denominator, source = n, "assessed_outcomes"
    elif attempted_count < n:
        # Fewer ledger rows than assessed outcomes means the ledger is not a complete inventory of this
        # group. Using it would produce a coverage above 1; using the outcomes instead would hide the
        # inconsistency. Refuse both and say so.
        attempted_denominator, source = None, "unavailable"
        attempted_reason = REASON_ATTEMPTED_LEDGER_SHORTER
        warnings.append(
            f"arm {arm_id} ({stratum_kind}={stratum_key}): the attempted ledger holds {attempted_count} "
            f"runs but {n} assessed outcomes exist here. The attempted denominator is reported as "
            "unavailable rather than guessed."
        )
    else:
        attempted_denominator, source = attempted_count, "attempted_ledger"

    if has_monitor and attempted_denominator is None:
        coverage = ProportionEstimate(
            numerator=0, denominator=0, point=None, method="none", confidence_level=confidence_level,
            undefined_reason=attempted_reason,
        )
        coverage_complete = wilson_interval(
            len(accepted), len(complete), confidence_level,
            undefined_reason=REASON_NO_COMPLETE if not complete else None,
        )
        warnings.append(
            f"arm {arm_id} ({stratum_kind}={stratum_key}): assurance coverage over attempted runs is "
            f"unavailable ({attempted_reason}). The complete-conditional coverage beside it has a "
            "different denominator and is not a substitute."
        )
    elif has_monitor:
        coverage = wilson_interval(
            len(accepted), attempted_denominator, confidence_level,
            undefined_reason=REASON_NO_EPISODES if not attempted_denominator else None,
        )
        coverage_complete = wilson_interval(
            len(accepted), len(complete), confidence_level,
            undefined_reason=REASON_NO_COMPLETE if not complete else None,
        )
        if attempted_denominator > len(complete):
            warnings.append(
                f"arm {arm_id}: assurance coverage denominator is {attempted_denominator} attempted "
                f"episodes, of which {len(complete)} completed. The complete-conditional coverage "
                "is reported separately and is the larger number."
            )
        if undefined_acceptance:
            warnings.append(
                f"{len(undefined_acceptance)} of {n} episodes in arm {arm_id} have undefined acceptance "
                "although the arm carries a monitor; they are excluded from the accepted set."
            )
    else:
        coverage = ProportionEstimate(
            numerator=0, denominator=0, point=None, method="none", confidence_level=confidence_level,
            undefined_reason=REASON_NO_MONITOR,
        )
        coverage_complete = coverage
        if accepted:
            warnings.append(
                f"arm {arm_id} has no monitor but {len(accepted)} episodes claim acceptance; "
                "check the runner and evaluator wiring."
            )

    # Re-derive ascertainability from independent verdicts, including historical records that stored
    # accepted+UNKNOWN as false_assurance=False. The v2 report explicitly identifies this reanalysis.
    ascertainable = [o for o in accepted if o.episode_verdict is not Verdict.UNKNOWN]
    undefined_fa = len(accepted) - len(ascertainable)
    false_count = sum(o.episode_verdict is Verdict.VIOLATION for o in ascertainable)
    if undefined_fa:
        warnings.append(
            f"{undefined_fa} accepted episodes in arm {arm_id} have unascertainable independent truth. "
            "The full accepted-set false-assurance rate is undefined. The ascertainable subset rate "
            "and missing-evidence identification bounds have separate denominators."
        )
    if not has_monitor:
        conditional_fa = ProportionEstimate(
            numerator=0, denominator=0, point=None, method="none", confidence_level=confidence_level,
            undefined_reason=REASON_NO_MONITOR,
        )
        ascertainable_fa = conditional_fa
    else:
        ascertainable_fa = wilson_interval(
            false_count, len(ascertainable), confidence_level,
            undefined_reason=REASON_UNKNOWN_ACCEPTED if accepted else REASON_NO_ACCEPTED,
        )
        conditional_fa = wilson_interval(
            false_count, len(accepted), confidence_level,
            undefined_reason=REASON_NO_ACCEPTED if not accepted else None,
        )
        if undefined_fa:
            conditional_fa = ProportionEstimate(
                numerator=false_count, denominator=len(accepted), point=None, method="none",
                confidence_level=confidence_level, undefined_reason=REASON_UNKNOWN_ACCEPTED,
            )
    fa_bounds = FalseAssuranceBounds(
        accepted=len(accepted), ascertainable=len(ascertainable), observed_violations=false_count,
        unresolved=undefined_fa,
        lower=false_count / len(accepted) if accepted and has_monitor else None,
        upper=(false_count + undefined_fa) / len(accepted) if accepted and has_monitor else None,
    )

    # Missed detection is conditional on an independently assessed violation having happened.
    violation_episodes = [o for o in outcomes if o.any_violation]
    detectable = [o for o in violation_episodes if o.missed_detection is not None]
    if not has_monitor:
        missed = ProportionEstimate(
            numerator=0, denominator=0, point=None, method="none", confidence_level=confidence_level,
            undefined_reason=REASON_NO_MONITOR,
        )
    else:
        missed = wilson_interval(
            sum(1 for o in detectable if o.missed_detection), len(detectable), confidence_level,
            undefined_reason=REASON_NO_VIOLATIONS if not detectable else None,
        )

    # UNKNOWN is a first-class outcome: a sparse truth trace makes an obligation undecidable, and that
    # must stay visible instead of being absorbed into the PASS side of a violation fraction.
    unknown_counts: dict[str, int] = {}
    unknown_reasons: dict[str, int] = {}
    for outcome in outcomes:
        for obligation_id, obligation in outcome.obligations.items():
            if obligation.verdict is Verdict.UNKNOWN:
                unknown_counts[obligation_id] = unknown_counts.get(obligation_id, 0) + 1
                reason = obligation.unknown_reason or "unspecified"
                unknown_reasons[reason] = unknown_reasons.get(reason, 0) + 1
    unknown_counts = dict(sorted(unknown_counts.items()))
    unknown_reasons = dict(sorted(unknown_reasons.items()))
    if unknown_counts:
        warnings.append(
            f"arm {arm_id}: obligations were UNDECIDABLE in some episodes "
            f"({unknown_counts}). An UNKNOWN verdict is not a pass and is excluded from neither the "
            "denominator nor the report."
        )

    if n == 0 and attempted_denominator:
        # An entirely failed arm or cell must stay in the report. Dropping it would turn a total
        # failure into an absence, which reads as "not studied" instead of "studied and never scored".
        warnings.append(
            f"arm {arm_id} ({stratum_kind}={stratum_key}): {attempted_denominator} attempted runs "
            "produced no assessed outcome at all. Its assurance coverage is 0 over those attempts and "
            "every conditional rate is undefined; it is not absent from the study."
        )

    empty_reason = REASON_NO_EPISODES
    metrics = ArmMetrics(
        arm_id=arm_id,
        stratum_kind=stratum_kind,  # type: ignore[arg-type]
        stratum_key=stratum_key,
        has_monitor=has_monitor,
        monitor_id=monitor_id,
        n_attempted=attempted_denominator,
        attempted_denominator_source=source,
        attempted_denominator_reason=attempted_reason,
        n_assessed=n,
        n_complete=len(complete),
        n_incomplete=len(incomplete),
        n_attempted_without_outcome=(
            None if attempted_denominator is None else max(0, attempted_denominator - n)
        ),
        incomplete_reasons=dict(Counter(o.incomplete_reason or "unspecified" for o in incomplete)),
        termination_reasons=dict(Counter(o.termination_reason for o in outcomes)),
        episode_verdicts=dict(Counter(
            o.episode_verdict.value if isinstance(o.episode_verdict, Verdict) else str(o.episode_verdict)
            for o in outcomes
        )),
        physical_violation=_proportion([o.physical_violation for o in outcomes], confidence_level,
                                       reason=empty_reason),
        procedural_violation=_proportion([o.procedural_violation for o in outcomes], confidence_level,
                                         reason=empty_reason),
        any_violation=_proportion([o.any_violation for o in outcomes], confidence_level,
                                  reason=empty_reason),
        physical_violation_complete_only=_proportion(
            [o.physical_violation for o in complete], confidence_level, reason=REASON_NO_COMPLETE),
        procedural_violation_complete_only=_proportion(
            [o.procedural_violation for o in complete], confidence_level, reason=REASON_NO_COMPLETE),
        mission_completion=_proportion([o.mission_completed for o in outcomes], confidence_level,
                                       reason=empty_reason),
        safe_mission_completion=_proportion([o.mission_completed_safely for o in outcomes],
                                            confidence_level, reason=empty_reason),
        n_accepted=len(accepted),
        assurance_coverage=coverage,
        assurance_coverage_complete_only=coverage_complete,
        conditional_false_assurance=conditional_fa,
        ascertainable_false_assurance=ascertainable_fa,
        false_assurance_bounds=fa_bounds,
        accepted_with_undefined_false_assurance=undefined_fa,
        n_violation_episodes=len(violation_episodes),
        missed_detection=missed,
        detection_delay_s=summarize_distribution(
            [o.detection_delay_s for o in violation_episodes],
            undefined_reason=REASON_NO_VIOLATIONS if not violation_episodes else REASON_NO_VALUES,
        ),
        interventions_per_episode=summarize_distribution(
            [float(o.interventions) for o in outcomes], undefined_reason=empty_reason),
        total_interventions=sum(o.interventions for o in outcomes),
        suspension_fraction=_proportion([o.suspended for o in outcomes], confidence_level,
                                        reason=empty_reason),
        abandonment_fraction=_proportion([o.abandoned for o in outcomes], confidence_level,
                                         reason=empty_reason),
        completion_time_s=summarize_distribution([o.completion_time_s for o in outcomes],
                                                 undefined_reason=empty_reason),
        unknown_verdict_fraction=_proportion(
            [o.episode_verdict is Verdict.UNKNOWN for o in outcomes], confidence_level,
            reason=empty_reason),
        obligation_unknown_counts=unknown_counts,
        unknown_reasons=unknown_reasons,
        truth_coverage=summarize_distribution([o.truth_coverage_fraction for o in outcomes],
                                              undefined_reason=empty_reason),
        measurements=summarize_measurements(
            outcomes, seed=bootstrap_seed, resamples=bootstrap_resamples, confidence_level=confidence_level,
        ),
        warnings=warnings,
    )
    return metrics


def _pair_outcomes(
    first: Sequence[EpisodeOutcome], second: Sequence[EpisodeOutcome]
) -> tuple[list[tuple[EpisodeOutcome, EpisodeOutcome]], list[str], list[str]]:
    """Match two arms on ``scenario_id``. Unmatched scenarios are returned, never imputed."""
    a = {o.scenario_id: o for o in first}
    b = {o.scenario_id: o for o in second}
    shared = sorted(set(a) & set(b))
    pairs = [(a[s], b[s]) for s in shared]
    return pairs, sorted(set(a) - set(b)), sorted(set(b) - set(a))


def _paired_comparison(
    first_arm: str,
    second_arm: str,
    outcome_id: str,
    role: str,
    first: Sequence[EpisodeOutcome],
    second: Sequence[EpisodeOutcome],
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
    stratify: bool,
    note: str = "",
) -> PairedComparison:
    getter = _OUTCOME_GETTERS[outcome_id]
    pairs, first_only, second_only = _pair_outcomes(first, second)
    x = [getter(p[0]) for p in pairs]
    y = [getter(p[1]) for p in pairs]
    strata = [p[0].cell_id or "unassigned" for p in pairs] if stratify else None
    if strata is not None and len({s for s in strata}) <= 1:
        strata = None  # a single stratum makes stratified and unstratified resampling identical
    b = sum(1 for xi, yi in zip(x, y, strict=True) if xi and not yi)
    c = sum(1 for xi, yi in zip(x, y, strict=True) if yi and not xi)
    concordant = len(pairs) - b - c
    difference = paired_bootstrap_difference(
        x, y, seed=seed, resamples=resamples, confidence_level=confidence_level, strata=strata,
    )
    return PairedComparison(
        comparison_id=f"{first_arm}_vs_{second_arm}__{outcome_id}",
        first_arm=first_arm,
        second_arm=second_arm,
        outcome_id=outcome_id,  # type: ignore[arg-type]
        role=role,  # type: ignore[arg-type]
        n_paired=len(pairs),
        n_first_only=len(first_only),
        n_second_only=len(second_only),
        unpaired_scenario_ids=sorted(set(first_only) | set(second_only)),
        first_rate=wilson_interval(sum(1 for v in x if v), len(x), confidence_level,
                                   undefined_reason=REASON_NO_PAIRS if not x else None),
        second_rate=wilson_interval(sum(1 for v in y if v), len(y), confidence_level,
                                    undefined_reason=REASON_NO_PAIRS if not y else None),
        difference=difference,
        mcnemar=mcnemar_exact(b, c, concordant_pairs=concordant),
        stratified_by="cell_id" if strata is not None else None,
        note=note,
    )


def _shift_comparison(
    first_arm: str,
    second_arm: str,
    outcome_id: str,
    first: Sequence[EpisodeOutcome],
    second: Sequence[EpisodeOutcome],
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> PairedShiftComparison:
    pairs, _, _ = _pair_outcomes(first, second)
    values = {
        "completion_time_s": lambda o: o.completion_time_s,
        "detection_delay_s": lambda o: o.detection_delay_s,
    }[outcome_id]
    xs = [values(p[0]) for p in pairs]
    ys = [values(p[1]) for p in pairs]
    usable = sum(1 for a, b in zip(xs, ys, strict=True) if a is not None and b is not None)
    return PairedShiftComparison(
        comparison_id=f"{first_arm}_vs_{second_arm}__{outcome_id}",
        first_arm=first_arm,
        second_arm=second_arm,
        outcome_id=outcome_id,  # type: ignore[arg-type]
        n_paired_with_values=usable,
        n_paired_scenarios=len(pairs),
        first_distribution=summarize_distribution(xs, undefined_reason=REASON_NO_VALUES),
        second_distribution=summarize_distribution(ys, undefined_reason=REASON_NO_VALUES),
        shift=paired_shift_estimate(xs, ys, seed=seed, resamples=resamples,
                                    confidence_level=confidence_level),
        note=("Pairs where either arm produced no value are dropped and counted; a shift over few pairs "
              "is a weak statement, not a null result."),
    )


def _coverage_warnings(overall: dict[str, ArmMetrics]) -> list[str]:
    """Expose missing episode acceptance without inferring the cause from coverage alone."""
    out: list[str] = []
    for arm_id in sorted(overall):
        m = overall[arm_id]
        if not m.has_monitor:
            continue
        cov = m.assurance_coverage
        if cov.point is None:
            out.append(
                f"arm {arm_id}: assurance coverage is undefined ({cov.undefined_reason}); its "
                "conditional false-assurance rate cannot be compared with any other arm."
            )
            continue
        if m.n_accepted == 0:
            out.append(
                f"arm {arm_id} accepted 0 of {cov.denominator} attempted episodes (coverage 0.000). Its "
                "conditional false-assurance rate is UNDEFINED, not 0.0. There is no affirmative "
                "episode-level acceptance; inspect termination reasons and step verdicts to "
                "distinguish incomplete episodes from monitor abstention or rejection."
            )
        elif cov.point < 0.5:
            out.append(
                f"arm {arm_id} accepted only {m.n_accepted} of {cov.denominator} attempted episodes "
                f"(coverage {cov.point:.3f}). Read its conditional false-assurance rate together with "
                "that coverage and with safe mission completion "
                f"({m.safe_mission_completion.describe()})."
            )
    return out


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def _typed_attempts(attempted: Any) -> list[AttemptedRun] | None:
    """Validate the attempted-run argument, rejecting the old summary-dict shape loudly.

    A summary dict cannot carry ``scenario_id``, so it can never place an attempt in a stratum, and the
    shape produced by ``runtime.evidence.summarize_attempts`` was silently unrecognised by the previous
    reader: real attempts without outcomes simply vanished from every denominator. Refusing the dict is
    the only way a caller finds out.
    """
    if attempted is None:
        return None
    if isinstance(attempted, Mapping):
        raise AnalysisInputError(
            "the analysis needs the typed attempted-run ledger, not an attempted-run summary dict. "
            "Pass `runtime.evidence.load_attempted_runs(run_dir / 'attempted_runs.jsonl')`. A summary "
            "carries no scenario_id, so it cannot place an attempt in a stratum, and an unrecognised "
            "summary shape silently drops every attempt that produced no outcome."
        )
    runs = list(attempted)
    bad = [type(r).__name__ for r in runs if not isinstance(r, AttemptedRun)]
    if bad:
        raise AnalysisInputError(
            f"attempted must be a sequence of schemas.AttemptedRun records, got {sorted(set(bad))}"
        )
    return runs


def _validate_attempted(
    attempted: Sequence[AttemptedRun], items: Sequence[EpisodeOutcome], protocol: ProtocolConfig,
) -> None:
    """Refuse an attempted ledger that belongs to a different run than the outcomes."""
    hashes = sorted({r.protocol_hash for r in attempted})
    outcome_hash = items[0].protocol_hash if items else protocol.content_hash()
    if hashes and hashes != [outcome_hash]:
        raise AnalysisInputError(
            f"the attempted-run ledger carries protocol hashes {hashes} but the outcomes carry "
            f"{outcome_hash!r}. A ledger from another run cannot supply this run's denominators."
        )
    classes = sorted({r.run_class for r in attempted})
    if len(classes) > 1 or (classes and items and classes != [items[0].run_class]):
        raise AnalysisInputError(
            f"the attempted-run ledger mixes run classes {classes} with outcomes from "
            f"{items[0].run_class if items else 'no assessed episodes'!r}."
        )


def _attempts_summary(ledger: Sequence[AttemptedRun] | None) -> dict[str, Any]:
    """Status and arm counts for the report, derived from the ledger the analysis actually used.

    The summary is recomputed here instead of being accepted from the caller, so the counts printed in
    the report cannot describe a different ledger than the one that produced the denominators.
    """
    if ledger is None:
        return {}
    from colosseum_assurance.runtime.evidence import summarize_attempts

    return summarize_attempts(list(ledger))


def _stratum_attempted(
    inventory: AttemptedInventory | None, kind: str, key: str, arm_id: str
) -> tuple[int | None, str | None]:
    """Attempted runs for one arm in one stratum, or ``None`` with the reason it cannot be established.

    An ``AttemptedRun`` names a scenario, not a cell. If any of this arm's attempts name a scenario
    whose membership is unknown, then every per-stratum count for that arm is only a lower bound, so
    the denominator is reported as unavailable instead of being quietly replaced by the scored count.
    """
    if inventory is None:
        return None, None
    if inventory.unresolved_by_arm.get(arm_id, 0):
        return None, REASON_ATTEMPTED_MEMBERSHIP_UNKNOWN
    return inventory.stratum_count(kind, key, arm_id), None


def load_outcomes(path: Path | str) -> list[EpisodeOutcome]:
    """Read episode outcomes from ``outcomes.jsonl`` (one JSON object per line).

    A directory is accepted and resolved to ``<dir>/outcomes.jsonl``. A malformed line names its line
    number: a silently skipped outcome would silently change every denominator below.
    """
    target = Path(path)
    if target.is_dir():
        target = target / "outcomes.jsonl"
    if not target.exists():
        raise FileNotFoundError(f"no outcome file at {target}")
    out: list[EpisodeOutcome] = []
    if target.suffix == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("outcomes", [])
        return [EpisodeOutcome.model_validate(row) for row in rows]
    for lineno, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            out.append(EpisodeOutcome.model_validate_json(text))
        except Exception as exc:  # noqa: BLE001 - re-raised with the line number for a usable message
            raise AnalysisInputError(f"{target}:{lineno} is not a valid EpisodeOutcome: {exc}") from exc
    return out


def compute_run_analysis(
    outcomes: list[EpisodeOutcome],
    protocol: ProtocolConfig,
    *,
    run_class: str,
    protocol_hash: str,
    attempted: Sequence[AttemptedRun] | None = None,
    scenario_strata: Mapping[str, Mapping[str, str]] | None = None,
    protocol_provenance: Mapping[str, Any] | None = None,
    code_version: dict[str, str] | None = None,
    confidence_level: float | None = None,
    bootstrap_resamples: int | None = None,
    shift_resamples: int = 2000,
    generated_at_wall_clock: str | None = None,
    attempts_summary: Any = None,
) -> RunAnalysis:
    """Workflow entry point: analyse one run and verify the caller's run class and protocol hash.

    ``run_class`` and ``protocol_hash`` are checked against the records rather than trusted, so a
    fixture analysis cannot be filed as held-out evidence by passing the wrong label. ``attempted`` is
    the typed attempted-run ledger; ``attempts_summary`` exists only to reject the old dict shape with
    a message instead of silently ignoring the attempts it describes.
    """
    if attempts_summary is not None:
        raise AnalysisInputError(
            "attempts_summary is no longer an input to the analysis: the summary shape carries no "
            "scenario_id and was silently ignored, so attempts without outcomes disappeared from the "
            "denominators. Pass attempted=load_attempted_runs(run_dir / 'attempted_runs.jsonl')."
        )
    analysis = analyze_run(
        outcomes, protocol, attempted=attempted, scenario_strata=scenario_strata,
        protocol_provenance=protocol_provenance, code_version=code_version,
        confidence_level=confidence_level, bootstrap_resamples=bootstrap_resamples,
        shift_resamples=shift_resamples, generated_at_wall_clock=generated_at_wall_clock,
    )
    if run_class != analysis.run_class:
        raise AnalysisInputError(
            f"caller declared run_class={run_class!r} but the episode outcomes carry "
            f"{analysis.run_class!r}. Refusing to mislabel a run."
        )
    if protocol_hash not in {analysis.protocol_hash, analysis.protocol_short_hash}:
        raise AnalysisInputError(
            f"caller declared protocol_hash={protocol_hash!r} but the episode outcomes carry "
            f"{analysis.protocol_hash!r}."
        )
    return analysis


def analyze_run(
    outcomes: Iterable[EpisodeOutcome],
    protocol: ProtocolConfig,
    *,
    attempted: Sequence[AttemptedRun] | None = None,
    scenario_strata: Mapping[str, Mapping[str, str]] | None = None,
    protocol_provenance: Mapping[str, Any] | None = None,
    code_version: dict[str, str] | None = None,
    confidence_level: float | None = None,
    bootstrap_resamples: int | None = None,
    shift_resamples: int = 2000,
    generated_at_wall_clock: str | None = None,
) -> RunAnalysis:
    """Turn evaluator outcomes into a :class:`RunAnalysis`.

    ``attempted`` is the typed attempted-run ledger. When it is supplied it defines the arm set and
    every attempted denominator, so an arm or cell whose runs all failed stays in the report with zero
    coverage instead of disappearing with the outcomes it never produced. ``scenario_strata`` maps
    ``scenario_id`` to its stratum membership (normally read from the run's manifests with
    :func:`load_scenario_strata`); without it, membership for a scenario that produced no outcome
    cannot be established and the affected per-stratum denominators are reported as unavailable.
    """
    items = list(outcomes)
    ledger = _typed_attempts(attempted)
    if ledger is not None:
        _validate_attempted(ledger, items, protocol)
    if items or not ledger:
        run_class = _validate_inputs(items, protocol)
    else:
        run_class = ledger[0].run_class
    _check_duplicates(items)

    spec = protocol.analysis
    conf = confidence_level if confidence_level is not None else spec.confidence_level
    resamples = bootstrap_resamples if bootstrap_resamples is not None else spec.bootstrap_resamples
    seed = spec.bootstrap_seed

    arm_specs = {a.arm_id: a for a in protocol.arms.arms}
    ledger_arms = {r.arm_id for r in ledger} if ledger is not None else set()
    unknown_arms = sorted(({o.arm_id for o in items} | ledger_arms) - set(arm_specs))
    if unknown_arms:
        raise AnalysisInputError(
            f"episode outcomes or attempted runs reference arms that the protocol does not define: "
            f"{unknown_arms}"
        )
    # The arm set is the union of scored arms and attempted arms. An arm whose every run failed has no
    # outcome to be derived from, and deriving arms from outcomes alone deleted it from the report.
    arm_ids = [a for a in protocol.arms.arm_ids
               if any(o.arm_id == a for o in items) or a in ledger_arms]

    by_arm: dict[str, list[EpisodeOutcome]] = {a: [] for a in arm_ids}
    for o in items:
        by_arm[o.arm_id].append(o)
    for a in arm_ids:
        by_arm[a].sort(key=lambda o: (o.scenario_id, o.episode_id))

    inventory = (
        build_attempted_inventory(ledger, items, scenario_strata) if ledger is not None else None
    )

    warnings: list[str] = []
    if inventory is not None and inventory.conflicting_scenario_ids:
        warnings.append(
            f"{len(inventory.conflicting_scenario_ids)} scenario realizations have manifests that "
            f"disagree with their outcomes about stratum membership (first: "
            f"{inventory.conflicting_scenario_ids[0]}). The manifest is used, because it was written "
            "before the episode ran."
        )
    if inventory is not None and inventory.unresolved_scenario_ids:
        warnings.append(
            f"{sum(inventory.unresolved_by_arm.values())} attempted runs name scenarios with no "
            f"manifest and no outcome (first: {inventory.unresolved_scenario_ids[0]}); their stratum "
            "membership is unknown, so per-stratum attempted denominators for the affected arms are "
            "reported as unavailable rather than silently reduced to the scored episodes."
        )

    # Shared realizations across cells would create a dependency cluster larger than one scenario.
    cells_per_scenario: dict[str, set[str]] = defaultdict(set)
    for o in items:
        cells_per_scenario[o.scenario_id].add(o.cell_id or "unassigned")
    shared = sorted(s for s, cs in cells_per_scenario.items() if len(cs) > 1)
    if shared:
        warnings.append(
            f"{len(shared)} scenario realizations appear in more than one condition cell "
            f"(first: {shared[0]}). The paired bootstrap already resamples whole realizations, so the "
            "dependency cluster is respected, but stratification by cell is ambiguous for them."
        )

    overall: dict[str, ArmMetrics] = {}
    for a in arm_ids:
        overall[a] = _arm_metrics(
            a, arm_specs[a].monitor_id, arm_specs[a].monitor_id is not None, by_arm[a],
            stratum_kind="all", stratum_key="all", confidence_level=conf,
            bootstrap_seed=seed, bootstrap_resamples=resamples,
            # The arm total is always known from the ledger: an attempt names its arm even when the
            # scenario it names cannot be placed in a stratum.
            attempted_count=inventory.by_arm.get(a, 0) if inventory is not None else None,
        )
        warnings.extend(overall[a].warnings)

    arms_without_outcomes = [a for a in arm_ids if not by_arm[a]]
    for a in arms_without_outcomes:
        statuses = (inventory.status_by_arm.get(a, {}) if inventory is not None else {})
        warnings.append(
            f"arm {a} produced no assessed outcome; its attempted runs ended as "
            f"{statuses or 'unrecorded statuses'}. It stays in the report with zero coverage."
        )

    by_stratum: list[ArmMetrics] = []
    for kind in STRATUM_KINDS:
        attr = _OUTCOME_STRATUM_ATTR[kind]
        outcome_keys = {str(getattr(o, attr) or "unassigned") for o in items}
        keys = sorted(outcome_keys | (inventory.stratum_keys(kind) if inventory is not None else set()))
        for key in keys:
            for a in arm_ids:
                subset = [o for o in by_arm[a] if str(getattr(o, attr) or "unassigned") == key]
                resolved = inventory.stratum_count(kind, key, a) if inventory is not None else 0
                attempted_count, unavailable = _stratum_attempted(inventory, kind, key, a)
                if not subset and not resolved:
                    # Nothing scored here and nothing attempted here that we can establish. Inventing a
                    # row would state a denominator the evidence does not support.
                    continue
                by_stratum.append(_arm_metrics(
                    a, arm_specs[a].monitor_id, arm_specs[a].monitor_id is not None, subset,
                    stratum_kind=kind, stratum_key=key, confidence_level=conf,
                    bootstrap_seed=seed, bootstrap_resamples=resamples,
                    attempted_count=attempted_count, attempted_unavailable_reason=unavailable,
                ))

    monitored = [a for a in arm_ids if arm_specs[a].monitor_id is not None]
    unguarded = [a for a in arm_ids if arm_specs[a].monitor_id is None]

    primary: list[PairedComparison] = []
    context: list[PairedComparison] = []
    shifts: list[PairedShiftComparison] = []

    if len(monitored) >= 2:
        first, second = monitored[0], monitored[1]
        for outcome_id, note in (
            ("physical_violation", "Primary outcome: paired all-episode physical violation difference."),
            ("procedural_violation", "Primary outcome, reported separately from physical violations."),
        ):
            primary.append(_paired_comparison(
                first, second, outcome_id, "primary", by_arm[first], by_arm[second],
                confidence_level=conf, resamples=resamples, seed=seed, stratify=True, note=note,
            ))
        for outcome_id in ("any_violation", "safe_mission_completion"):
            primary.append(_paired_comparison(
                first, second, outcome_id, "secondary", by_arm[first], by_arm[second],
                confidence_level=conf, resamples=resamples, seed=seed, stratify=True,
                note="Secondary outcome; interpret together with the primary comparison.",
            ))
        for outcome_id in ("completion_time_s", "detection_delay_s"):
            shifts.append(_shift_comparison(
                first, second, outcome_id, by_arm[first], by_arm[second],
                confidence_level=conf, resamples=shift_resamples, seed=seed,
            ))
    else:
        warnings.append(
            "fewer than two monitored arms are present; the primary paired comparison is not computed."
        )

    for base in unguarded:
        for a in monitored:
            for outcome_id in ("physical_violation", "procedural_violation", "safe_mission_completion"):
                context.append(_paired_comparison(
                    base, a, outcome_id, "context", by_arm[base], by_arm[a],
                    confidence_level=conf, resamples=resamples, seed=seed, stratify=True,
                    note=("Context only: the unguarded arm shows scenario difficulty. It has no monitor, "
                          "so its acceptance and missed detection are undefined, not zero."),
                ))

    for c in primary + context:
        if c.n_first_only or c.n_second_only:
            warnings.append(
                f"comparison {c.comparison_id}: {c.n_first_only + c.n_second_only} scenario "
                f"realizations were present in only one arm and were dropped from the paired test "
                f"(paired n={c.n_paired})."
            )

    warnings.extend(_coverage_warnings(overall))

    notes = [
        CONDITIONAL_RATE_CAVEAT,
        "The statistical unit is the scenario realization. Steps, frames, and monitor decisions from one "
        "episode are not independent replications.",
        "All-episode violation fractions use every assessed episode as denominator, including incomplete "
        "ones. An incomplete episode is not evidence of safety; complete-only fractions are reported "
        "beside them as a sensitivity check.",
        "Rates are conditional on the declared stress-test sampling distribution. They are not estimates "
        "of real-world incident rates, and zero observed violations does not establish zero risk.",
    ]
    if run_class in {"fixture", "smoke"}:
        notes.insert(0, "SYNTHETIC FIXTURE DATA - NOT EXPERIMENTAL EVIDENCE.")
    notes.append(
        "False-assurance analysis uses ascertainable_v2: UNKNOWN independent truth is unresolved, "
        "never an established negative. Historical inputs are explicitly reanalysed under v2; "
        "their frozen protocol and stored outcomes are not modified."
    )

    provenance = sorted({str(o.diagnostics.get("simulator_provenance", "unrecorded")) for o in items})
    if ledger is not None:
        provenance = sorted(set(provenance) | {r.simulator_provenance for r in ledger})
    if any(p not in ANCHORED_LIVE_PROVENANCES for p in provenance):
        notes.append(
            "Simulator provenance for this run is "
            f"{provenance}. Only records produced by a live Colosseum server support experimental claims."
        )

    return RunAnalysis(
        run_class=run_class,  # type: ignore[arg-type]
        protocol_hash=protocol.content_hash(),
        protocol_short_hash=protocol.short_hash,
        protocol_label=protocol.protocol_label,
        policy_version=protocol.obligations.policy_version,
        evaluator_versions=sorted({o.evaluator_version for o in items}),
        code_version=dict(code_version or {}),
        simulator_provenance=provenance,
        generated_at_wall_clock=generated_at_wall_clock,
        confidence_level=conf,
        bootstrap_resamples=resamples,
        bootstrap_seed=seed,
        acceptance_rule_id=spec.acceptance_rule.rule_id,
        acceptance_rule_text=spec.acceptance_rule.statement,
        primary_outcome_text=spec.primary_outcome,
        strata_used=list(spec.strata),
        n_episode_outcomes=len(items),
        n_scenarios=len({o.scenario_id for o in items}),
        n_attempted_runs=len(ledger) if ledger is not None else None,
        attempts_summary=_attempts_summary(ledger),
        attempted_inventory=inventory,
        protocol_provenance=dict(protocol_provenance or {}),
        arms_without_outcomes=arms_without_outcomes,
        arm_ids=arm_ids,
        cell_ids=sorted(
            {o.cell_id or "unassigned" for o in items}
            | (inventory.stratum_keys("cell") if inventory is not None else set())
        ),
        shared_realizations_across_cells=bool(shared),
        overall=overall,
        by_stratum=by_stratum,
        primary_comparisons=primary,
        context_comparisons=context,
        shift_comparisons=shifts,
        warnings=warnings,
        interpretation_notes=notes,
    )


def analysis_to_dict(analysis: RunAnalysis) -> dict[str, Any]:
    """JSON-safe dictionary of the analysis, used by the report writer and by tests."""
    return analysis.model_dump(mode="json")
