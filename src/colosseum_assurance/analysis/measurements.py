"""Scenario-level summaries of independently measured exposure and response windows.

Multiple windows from one flight never become independent replications. We first calculate one mean
per episode, then summarize and bootstrap those means over scenario realizations within each arm.
Observed-only latencies are explicitly conditional: censored windows remain counted alongside them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

from pydantic import Field

from colosseum_assurance.analysis.stats import (
    BootstrapEstimate,
    DistributionSummary,
    ProportionEstimate,
    paired_bootstrap_difference,
    summarize_distribution,
    wilson_interval,
)
from colosseum_assurance.evaluation.outcomes import EpisodeOutcome
from colosseum_assurance.schemas import StrictModel


class LatencySummary(StrictModel):
    metric: str
    status_counts: dict[str, int] = Field(default_factory=dict)
    source_counts: dict[str, int] = Field(default_factory=dict)
    n_windows: int = 0
    n_episodes_with_windows: int = 0
    observed_episode_means_s: DistributionSummary = Field(default_factory=DistributionSummary)
    mean_of_observed_episode_means_s: BootstrapEstimate = Field(default_factory=BootstrapEstimate)
    censor_followup_s: DistributionSummary = Field(default_factory=DistributionSummary)
    interpretation: str = (
        "One mean per episode over observed windows only; censoring/unavailable counts are separate. "
        "Not an uncensored population mean or a measurement of human performance."
    )


class MeasurementSummary(StrictModel):
    version: str = "sampled-exposure-and-response-v2"
    n_measured_episodes: int = 0
    n_episodes_without_measurements: int = 0
    outside_duration_s: DistributionSummary = Field(default_factory=DistributionSummary)
    mean_outside_duration_s: BootstrapEstimate = Field(default_factory=BootstrapEstimate)
    partial_observed_outside_s: float = 0.0
    unobserved_duration_s: float = 0.0
    latencies: dict[str, LatencySummary] = Field(default_factory=dict)
    recovery_status_counts: dict[str, int] = Field(default_factory=dict)
    recovery_unavailable_reasons: dict[str, int] = Field(default_factory=dict)
    n_fault_episodes: int = 0
    recovery_episode_success_fractions: DistributionSummary = Field(default_factory=DistributionSummary)
    mean_recovery_episode_success_fraction: BootstrapEstimate = Field(default_factory=BootstrapEstimate)
    recovery_episode_mean_latency_s: DistributionSummary = Field(default_factory=DistributionSummary)
    mean_recovery_episode_latency_s: BootstrapEstimate = Field(default_factory=BootstrapEstimate)
    proxy_episodes: int = 0
    proxy_unresolved: int = 0
    proxy_nonempty_captures: DistributionSummary = Field(default_factory=DistributionSummary)
    proxy_qualifying_authorized_captures: DistributionSummary = Field(default_factory=DistributionSummary)
    proxy_redundant_captures: DistributionSummary = Field(default_factory=DistributionSummary)
    proxy_mismatch: ProportionEstimate = Field(
        default_factory=lambda: ProportionEstimate(numerator=0, denominator=0, point=None, method="none")
    )


def _mean_interval(
    values: Sequence[float], *, seed: int, resamples: int, confidence_level: float,
) -> BootstrapEstimate:
    if len(values) < 2:
        return BootstrapEstimate(
            point=sum(values) / len(values) if values else None, n_units=len(values),
            confidence_level=confidence_level, seed=seed, resamples=0, method="none",
            undefined_reason="fewer than two independent scenario realizations; interval unavailable",
        )
    return paired_bootstrap_difference(
        values, [0.0] * len(values), seed=seed, resamples=resamples,
        confidence_level=confidence_level,
    )


def summarize_measurements(
    outcomes: Sequence[EpisodeOutcome], *, seed: int, resamples: int, confidence_level: float,
) -> MeasurementSummary:
    measured = [o for o in outcomes if o.measurements is not None]
    exposure_values = [
        o.measurements.envelope_exposure.full_horizon_estimate_s if o.measurements else None
        for o in outcomes
    ]
    timings: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for outcome in measured:
        assert outcome.measurements is not None
        for row in outcome.measurements.latencies:
            timings[row.metric][outcome.episode_id].append(row)
    latency_summaries = {}
    for metric, episodes in sorted(timings.items()):
        rows = [r for group in episodes.values() for r in group]
        means: list[float | None] = []
        for group in episodes.values():
            observed = [r.latency_s for r in group if r.status == "observed" and r.latency_s is not None]
            means.append(sum(observed) / len(observed) if observed else None)
        latency_summaries[metric] = LatencySummary(
            metric=metric, n_windows=len(rows), n_episodes_with_windows=len(episodes),
            status_counts=dict(sorted(Counter(r.status for r in rows).items())),
            source_counts=dict(sorted(Counter(r.source for r in rows).items())),
            observed_episode_means_s=summarize_distribution(means),
            mean_of_observed_episode_means_s=_mean_interval(
                [v for v in means if v is not None], seed=seed, resamples=resamples,
                confidence_level=confidence_level,
            ),
            censor_followup_s=summarize_distribution(
                [r.followup_s for r in rows if r.status == "right_censored"]
            ),
        )
    statuses: Counter[str] = Counter()
    unavailable: Counter[str] = Counter()
    success: list[float | None] = []
    recovery_means: list[float | None] = []
    for outcome in measured:
        measurement = outcome.measurements
        assert measurement is not None
        if measurement.recovery_unavailable_reason:
            unavailable[measurement.recovery_unavailable_reason] += 1
        recovery = measurement.fault_recoveries
        statuses.update(r.status for r in recovery)
        if not recovery:
            continue
        resolved = [r for r in recovery if r.status in {"maintained_safe", "recovered", "not_recovered"}]
        success.append(
            sum(r.status != "not_recovered" for r in resolved) / len(resolved) if resolved else None
        )
        times = [r.recovery_latency_s for r in recovery if r.status == "recovered"
                 and r.recovery_latency_s is not None]
        recovery_means.append(sum(times) / len(times) if times else None)
    proxy = [o.proxy_diagnostic for o in outcomes if o.proxy_diagnostic is not None]
    resolved_proxy = [p.proxy_satisfied_intent_failed for p in proxy
                      if p.proxy_satisfied_intent_failed is not None]
    return MeasurementSummary(
        n_measured_episodes=len(measured), n_episodes_without_measurements=len(outcomes) - len(measured),
        outside_duration_s=summarize_distribution(exposure_values),
        mean_outside_duration_s=_mean_interval(
            [v for v in exposure_values if v is not None], seed=seed, resamples=resamples,
            confidence_level=confidence_level,
        ),
        partial_observed_outside_s=sum(o.measurements.envelope_exposure.observed_outside_duration_s
                                     for o in measured if o.measurements),
        unobserved_duration_s=sum(o.measurements.envelope_exposure.unobserved_duration_s
                                 for o in measured if o.measurements),
        latencies=latency_summaries, recovery_status_counts=dict(sorted(statuses.items())),
        recovery_unavailable_reasons=dict(sorted(unavailable.items())), n_fault_episodes=len(success),
        recovery_episode_success_fractions=summarize_distribution(success),
        mean_recovery_episode_success_fraction=_mean_interval(
            [v for v in success if v is not None], seed=seed, resamples=resamples,
            confidence_level=confidence_level,
        ),
        recovery_episode_mean_latency_s=summarize_distribution(recovery_means),
        mean_recovery_episode_latency_s=_mean_interval(
            [v for v in recovery_means if v is not None], seed=seed, resamples=resamples,
            confidence_level=confidence_level,
        ),
        proxy_episodes=len(proxy), proxy_unresolved=len(proxy) - len(resolved_proxy),
        proxy_nonempty_captures=summarize_distribution([float(p.nonempty_capture_count) for p in proxy]),
        proxy_qualifying_authorized_captures=summarize_distribution(
            [float(p.qualifying_authorized_capture_count) for p in proxy]),
        proxy_redundant_captures=summarize_distribution(
            [float(p.redundant_nonempty_capture_count) for p in proxy]),
        proxy_mismatch=wilson_interval(sum(resolved_proxy), len(resolved_proxy), confidence_level,
                                      undefined_reason="no ascertainable proxy-versus-intent episodes"),
    )
