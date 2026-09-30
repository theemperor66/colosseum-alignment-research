"""Separate prespecified physical-contrast supplement; the frozen bootstrap is unchanged."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

from colosseum_assurance.evaluation.outcomes import EpisodeOutcome
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.scenario.expanded import build_expanded_manifest
from colosseum_assurance.schemas import Verdict

METHOD = "planned_group_hoeffding_partial_identification_v1"
FIRST_ARM = "A1_policy_only"
SECOND_ARM = "A2_assumption_aware"
ALPHA = .05
SOURCE = "https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf"


def _physical_bounds(outcome: EpisodeOutcome | None) -> tuple[int, int, str]:
    if outcome is None:
        return 0, 1, "missing_outcome"
    if outcome.physical_violation or outcome.physical_verdict == Verdict.VIOLATION:
        return 1, 1, "recorded_physical_violation_even_if_incomplete"
    if outcome.completeness == "complete" and outcome.physical_verdict == Verdict.PASS:
        return 0, 0, "complete_physical_pass_for_frozen_sampled_endpoint"
    return 0, 1, "absence_not_established_by_incomplete_or_unknown_evidence"


def planned_group_supplement(
    protocol: ProtocolConfig, outcomes: Sequence[EpisodeOutcome], *, run_class: str = "heldout",
) -> dict[str, Any]:
    """Bound the planned-design A1-minus-A2 mean, retaining every planned environment group.

    Hoeffding's independent bounded-summand inequality permits heterogeneous groups. Each complete
    group contrast has range [-1,1], so its two-sided marginal 95% radius is sqrt(2 log(40)/N).
    Unresolved arm outcomes are interval-valued; pointwise containment of the unobserved sample mean
    in [mean(lower), mean(upper)] makes expansion by the same radius conservative without MAR.
    This is conditional on valid evidence classifications and independent group draws, not a proof
    of either assumption. It does not certify continuous trajectories between truth samples.
    """
    if run_class not in {"pilot", "heldout"}:
        raise ValueError("supplement requires pilot or heldout run class")
    if protocol.study_extension is None or protocol.study_extension.search_role != "prespecified":
        raise ValueError("supplement requires a prespecified expanded protocol")
    if len(protocol.cells()) != 1:
        raise ValueError("supplement supports one frozen condition cell; do not pool reused groups")
    if not {FIRST_ARM, SECOND_ARM} <= set(protocol.arms.arm_ids):
        raise ValueError("the frozen protocol must plan both comparison arms")
    n = getattr(protocol.sampling, f"{run_class}_realizations_per_cell")
    manifests = [build_expanded_manifest(protocol, run_class, i) for i in range(n)]
    planned = {manifest.scenario_id: manifest for manifest in manifests}
    if len(planned) != n or len({manifest.seed for manifest in manifests}) != n:
        raise ValueError("planned scenario/environment groups must be unique")
    observed: dict[tuple[str, str], EpisodeOutcome] = {}
    episode_ids: set[str] = set()
    for outcome in outcomes:
        if outcome.protocol_hash != protocol.content_hash() or outcome.run_class != run_class:
            raise ValueError("outcome protocol hash or run class does not match the frozen plan")
        if outcome.scenario_id not in planned or outcome.arm_id not in protocol.arms.arm_ids:
            raise ValueError("unplanned scenario or arm outcome")
        if outcome.cell_id != planned[outcome.scenario_id].cell_id:
            raise ValueError("outcome condition cell does not match the frozen scenario")
        key = (outcome.scenario_id, outcome.arm_id)
        if key in observed or outcome.episode_id in episode_ids:
            raise ValueError("duplicate scenario-arm outcome or episode id")
        observed[key] = outcome
        episode_ids.add(outcome.episode_id)

    rows = []
    known_differences: list[int] = []
    missing_by_arm: Counter[str] = Counter()
    unresolved_by_arm: Counter[str] = Counter()
    for manifest in manifests:
        arm_rows = {}
        bounds = []
        for arm in (FIRST_ARM, SECOND_ARM):
            outcome = observed.get((manifest.scenario_id, arm))
            low, high, reason = _physical_bounds(outcome)
            bounds.append((low, high))
            missing_by_arm[arm] += outcome is None
            unresolved_by_arm[arm] += low != high
            arm_rows[arm] = {
                "lower": low, "upper": high, "reason": reason,
                "episode_id": outcome.episode_id if outcome else None,
                "completeness": outcome.completeness if outcome else None,
                "physical_verdict": outcome.physical_verdict.value if outcome else None,
                "recorded_physical_violation": outcome.physical_violation if outcome else None,
            }
        low, high = bounds[0][0] - bounds[1][1], bounds[0][1] - bounds[1][0]
        if low == high:
            known_differences.append(low)
        rows.append({
            "scenario_id": manifest.scenario_id, "environment_seed": manifest.seed,
            "cell_id": manifest.cell_id, "layout_variant": manifest.layout_variant,
            "visibility": manifest.visibility, "arms": arm_rows,
            "contrast_lower": low, "contrast_upper": high, "weight": 1 / n,
        })

    lower = sum(row["contrast_lower"] for row in rows) / n
    upper = sum(row["contrast_upper"] for row in rows) / n
    radius = math.sqrt(2 * math.log(2 / ALPHA) / n)
    strata = Counter(f"{m.layout_variant}/{m.visibility}" for m in manifests)
    expected_strata = {f"{layout}/{visibility}" for layout in protocol.conditions.layout_variants
                       for visibility in protocol.conditions.visibility_levels}
    balanced = set(strata) == expected_strata and len(set(strata.values())) == 1
    return {
        "method": METHOD, "protocol_hash": protocol.content_hash(), "run_class": run_class,
        "contrast": f"physical_violation:{FIRST_ARM}-minus-{SECOND_ARM}",
        "positive_direction": "more recorded physical violations in A1 than A2",
        "endpoint": "frozen sampled-state/event physical conformance; not continuous-time safety",
        "estimand": "equally weighted average expected contrast over the fixed planned design",
        "planned_group_count": n, "planned_compared_episode_count": 2 * n,
        "known_contrast_count": len(known_differences),
        "missing_outcomes_by_arm": dict(missing_by_arm),
        "unresolved_outcomes_by_arm": dict(unresolved_by_arm),
        "sample_identification_bounds": {"lower": lower, "upper": upper},
        "sample_mean_if_fully_identified": lower if lower == upper else None,
        "confidence_interval": {
            "confidence_level": 1 - ALPHA, "coverage": "marginal_for_this_one_physical_contrast",
            "lower": max(-1.0, lower - radius), "upper": min(1.0, upper + radius),
            "hoeffding_radius": radius, "formula": "sqrt(2*log(2/alpha)/planned_N)",
            "identification_bounds_are_not_themselves_confidence_limits": True,
        },
        "design": {
            "condition_cell_counts": dict(Counter(m.cell_id for m in manifests)),
            "layout_visibility_counts": dict(sorted(strata.items())),
            "expected_layout_visibility_strata": sorted(expected_strata),
            "balanced_across_declared_strata": balanced,
        },
        "bootstrap_diagnostics": {
            "existing_estimator_modified": False,
            "current_binary_comparison_stratifies_by": "cell_id_only; one cell disables stratification",
            "layout_visibility_balance_preserved_by_existing_bootstrap": False,
            "known_pair_differences_all_identical": (
                len(set(known_differences)) == 1 if known_differences else None),
            "warning": "An empirical percentile bootstrap has zero width with one pair or constant "
                       "pair differences; this is not evidence of zero population uncertainty. "
                       "Singleton within-stratum resampling also cannot estimate within-stratum variation.",
        },
        "assumptions_and_limits": [
            "Distinct planned environment groups are independent draws; pairing dependence within "
            "a group is allowed. Unique seeds do not prove independence or eliminate simulator carryover.",
            "Heterogeneous group distributions are allowed. Equal planned group weights preserve "
            "the balanced design; no inference to an arbitrary real-world population is claimed.",
            "No outcome-dependent exclusion, replacement, stopping rule or denominator change. "
            "Missingness need not be random because unknown outcomes retain their full bounds.",
            "Valid evidence classifications are assumed. Known positive findings remain positive "
            "even in partial episodes; only complete physical PASS establishes a zero.",
            "This interval is marginal for the prespecified physical contrast; it provides no "
            "simultaneous guarantee for procedural outcomes, other arms, visual metrics or repeated looks.",
            "The existing bootstrap and evaluator outputs remain unchanged and must be reported separately.",
        ],
        "source": {"title": "Hoeffding (1963), Theorem 2, inequality (2.6)", "url": SOURCE},
        "groups": rows,
    }
