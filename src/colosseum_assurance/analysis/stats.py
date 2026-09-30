"""Statistical primitives for episode-level analysis, implemented without SciPy.

Why this module exists at all:

* The statistical unit of this study is the **scenario realization**, never a frame, step, or
  monitor decision (research-plan.md, "Outcomes that make the results interpretable"). Every routine
  here therefore takes one value per realization, or an explicit cluster label when one realization
  contributes rows to several cells.
* The primary comparison is **paired** across arms on the same realization, so resampling must move
  whole realizations together with both arm outcomes. A plain two-sample bootstrap would destroy the
  pairing and understate the precision of the difference.
* Adding SciPy for a handful of closed-form quantities would add a large pinned dependency to a study
  whose main claim is reproducibility. All routines below are checked against hand-computable
  references in ``tests/unit/test_stats.py``.

Every estimate is returned as a small pydantic record carrying its own denominator and, when the
quantity is not defined, an explicit ``undefined_reason``. A rate without a denominator is exactly the
reporting mistake this study is about, so the types make it hard to drop one.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from pydantic import Field

from colosseum_assurance.schemas import StrictModel

__all__ = [
    "BootstrapEstimate",
    "DistributionSummary",
    "McNemarResult",
    "ProportionEstimate",
    "cluster_bootstrap_statistic",
    "hodges_lehmann_paired",
    "mcnemar_exact",
    "normal_cdf",
    "normal_quantile",
    "paired_bootstrap_difference",
    "paired_shift_estimate",
    "summarize_distribution",
    "wilson_interval",
]


# --------------------------------------------------------------------------------------
# Normal distribution helpers (replacing scipy.stats.norm)
# --------------------------------------------------------------------------------------
def normal_cdf(x: float) -> float:
    """Standard normal CDF via ``math.erfc`` (machine precision, no dependency)."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# Acklam's rational approximation coefficients; refined below with two Halley steps so the result is
# accurate to ~1e-15 and the reported confidence level is exactly the requested one.
_ACKLAM_A = (
    -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
    1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
)
_ACKLAM_B = (
    -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
    6.680131188771972e01, -1.328068155288572e01,
)
_ACKLAM_C = (
    -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
    -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
)
_ACKLAM_D = (
    7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00,
)
_ACKLAM_P_LOW = 0.02425


def normal_quantile(p: float) -> float:
    """Inverse standard normal CDF (probit) for ``0 < p < 1``.

    Used for Wilson intervals and for nothing else; the bootstrap routines are percentile based and
    make no normality assumption.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"normal_quantile requires 0 < p < 1, got {p!r}")
    if p < _ACKLAM_P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q + _ACKLAM_C[3]) * q
              + _ACKLAM_C[4]) * q + _ACKLAM_C[5]) / ((((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q
                                                      + _ACKLAM_D[2]) * q + _ACKLAM_D[3]) * q + 1.0)
    elif p <= 1.0 - _ACKLAM_P_LOW:
        q = p - 0.5
        r = q * q
        x = (((((_ACKLAM_A[0] * r + _ACKLAM_A[1]) * r + _ACKLAM_A[2]) * r + _ACKLAM_A[3]) * r
              + _ACKLAM_A[4]) * r + _ACKLAM_A[5]) * q / (((((_ACKLAM_B[0] * r + _ACKLAM_B[1]) * r
                                                           + _ACKLAM_B[2]) * r + _ACKLAM_B[3]) * r
                                                          + _ACKLAM_B[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q + _ACKLAM_C[3]) * q
               + _ACKLAM_C[4]) * q + _ACKLAM_C[5]) / ((((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q
                                                        + _ACKLAM_D[2]) * q + _ACKLAM_D[3]) * q + 1.0)
    # Halley refinement on f(x) = Phi(x) - p.
    for _ in range(2):
        err = normal_cdf(x) - p
        pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        if pdf <= 0.0:  # pragma: no cover - only reachable for |x| > 38
            break
        u = err / pdf
        x = x - u / (1.0 + 0.5 * x * u)
    return x


def _z_for(confidence_level: float) -> float:
    if not 0.5 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must lie in (0.5, 1.0), got {confidence_level!r}")
    return normal_quantile(1.0 - (1.0 - confidence_level) / 2.0)


# --------------------------------------------------------------------------------------
# Proportions
# --------------------------------------------------------------------------------------
class ProportionEstimate(StrictModel):
    """A proportion with the denominator it was computed over.

    ``point`` is ``None`` when ``denominator == 0``. A conditional rate over an empty accepted set is
    undefined, not zero (research-acceptance.md section 4).
    """

    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    point: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    confidence_level: float = 0.95
    method: Literal["wilson", "none"] = "wilson"
    undefined_reason: str | None = None

    @property
    def is_defined(self) -> bool:
        return self.point is not None

    def describe(self) -> str:
        """One-line human rendering that always shows the denominator."""
        if self.point is None:
            return f"undefined (n={self.denominator}; {self.undefined_reason or 'no denominator'})"
        ci = ""
        if self.ci_low is not None and self.ci_high is not None:
            ci = f" [{self.ci_low:.3f}, {self.ci_high:.3f}]"
        return f"{self.point:.3f}{ci} (n={self.denominator}, k={self.numerator})"


def wilson_interval(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
    *,
    undefined_reason: str | None = None,
) -> ProportionEstimate:
    """Wilson score interval for a binomial proportion.

    Wilson rather than Wald because several outcomes here are expected near 0 or 1 (for example zero
    observed violations in the nominal cell), where the Wald interval collapses to zero width and
    would falsely suggest certainty. Zero observed failures does not establish zero risk.
    """
    if trials < 0 or successes < 0:
        raise ValueError("successes and trials must be non-negative")
    if successes > trials:
        raise ValueError(f"successes ({successes}) cannot exceed trials ({trials})")
    if trials == 0:
        return ProportionEstimate(
            numerator=0, denominator=0, point=None, ci_low=None, ci_high=None,
            confidence_level=confidence_level, method="none",
            undefined_reason=undefined_reason or "empty denominator",
        )
    z = _z_for(confidence_level)
    n = float(trials)
    k = float(successes)
    z2 = z * z
    denom = n + z2
    centre = (k + z2 / 2.0) / denom
    half = (z / denom) * math.sqrt(k * (n - k) / n + z2 / 4.0)
    point = k / n
    # Clamp to [0, 1] and to the point estimate: the Wilson interval always contains p-hat, so a bound
    # on the wrong side of it can only be floating-point noise, and downstream error bars must not
    # receive a negative length because of it.
    return ProportionEstimate(
        numerator=successes,
        denominator=trials,
        point=point,
        ci_low=min(max(0.0, centre - half), point),
        ci_high=max(min(1.0, centre + half), point),
        confidence_level=confidence_level,
        method="wilson",
    )


# --------------------------------------------------------------------------------------
# Paired binary test
# --------------------------------------------------------------------------------------
class McNemarResult(StrictModel):
    """Exact (binomial) McNemar test for paired binary outcomes.

    ``b`` counts realizations where the first arm shows the outcome and the second does not; ``c``
    counts the reverse. Concordant pairs carry no information about the difference and are reported
    only for transparency.
    """

    b: int = Field(ge=0, description="First arm 1, second arm 0.")
    c: int = Field(ge=0, description="First arm 0, second arm 1.")
    concordant_pairs: int = Field(default=0, ge=0)
    n_pairs: int = Field(default=0, ge=0)
    p_value: float | None = None
    method: Literal["exact_binomial", "none"] = "exact_binomial"
    undefined_reason: str | None = None

    @property
    def n_discordant(self) -> int:
        return self.b + self.c


def _binom_cdf_half(k: int, n: int) -> float:
    """P(X <= k) for X ~ Binomial(n, 0.5), summed exactly in floating point."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    total = 0.0
    for i in range(k + 1):
        total += math.comb(n, i)
    return total / (2.0**n)


def mcnemar_exact(b: int, c: int, *, concordant_pairs: int = 0) -> McNemarResult:
    """Two-sided exact McNemar test.

    The exact binomial form is used instead of the chi-square approximation because the discordant
    count in this study is expected to be small (few realizations flip between the two monitored
    arms), which is exactly where the chi-square version is unreliable.

    With no discordant pairs the test carries no information; the p-value is reported as 1.0 with an
    explicit reason rather than as a significant or missing result.
    """
    if b < 0 or c < 0:
        raise ValueError("b and c must be non-negative")
    n = b + c
    if n == 0:
        return McNemarResult(
            b=0, c=0, concordant_pairs=concordant_pairs, n_pairs=concordant_pairs, p_value=1.0,
            method="none", undefined_reason="no discordant pairs: the paired test has no information",
        )
    p = 2.0 * _binom_cdf_half(min(b, c), n)
    return McNemarResult(
        b=b, c=c, concordant_pairs=concordant_pairs, n_pairs=n + concordant_pairs,
        p_value=min(1.0, p), method="exact_binomial",
    )


# --------------------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------------------
class BootstrapEstimate(StrictModel):
    """A point estimate with a percentile bootstrap interval and its resampling provenance."""

    point: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    confidence_level: float = 0.95
    resamples: int = 0
    seed: int = 0
    n_units: int = Field(default=0, ge=0, description="Resampled units (realizations or clusters).")
    unit: str = "scenario_realization"
    stratified: bool = False
    method: Literal["paired_percentile", "cluster_percentile", "none"] = "paired_percentile"
    undefined_reason: str | None = None

    @property
    def is_defined(self) -> bool:
        return self.point is not None

    def describe(self) -> str:
        if self.point is None:
            return f"undefined ({self.undefined_reason or 'no units'})"
        if self.ci_low is None or self.ci_high is None:
            return f"{self.point:+.3f} (n={self.n_units}, no interval)"
        return f"{self.point:+.3f} [{self.ci_low:+.3f}, {self.ci_high:+.3f}] (n={self.n_units})"


def _stratified_index_draws(
    strata: Sequence[str] | None, n: int, resamples: int, rng: np.random.Generator
) -> np.ndarray:
    """Return an ``(resamples, n)`` index matrix, resampling within strata when strata are given.

    Stratified resampling keeps the severity structure of the design: each bootstrap replicate has the
    same number of realizations per condition cell as the observed data, so the interval does not
    absorb variation caused by accidentally over-representing a severe cell.
    """
    if strata is None:
        return rng.integers(0, n, size=(resamples, n))
    if len(strata) != n:
        raise ValueError("strata must have one label per unit")
    out = np.empty((resamples, n), dtype=np.int64)
    order: list[str] = []
    seen: set[str] = set()
    for label in strata:
        if label not in seen:
            seen.add(label)
            order.append(label)
    cursor = 0
    for label in order:
        members = np.array([i for i, lab in enumerate(strata) if lab == label], dtype=np.int64)
        size = members.size
        picks = rng.integers(0, size, size=(resamples, size))
        out[:, cursor:cursor + size] = members[picks]
        cursor += size
    return out


def _percentile_ci(samples: np.ndarray, confidence_level: float) -> tuple[float, float]:
    alpha = 1.0 - confidence_level
    low, high = np.percentile(samples, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high)


def paired_bootstrap_difference(
    first: Sequence[bool | int | float],
    second: Sequence[bool | int | float],
    *,
    seed: int,
    resamples: int = 10000,
    confidence_level: float = 0.95,
    strata: Sequence[str] | None = None,
    unit: str = "scenario_realization",
) -> BootstrapEstimate:
    """Percentile bootstrap CI for ``mean(first) - mean(second)`` over paired units.

    ``first[i]`` and ``second[i]`` are the two arm outcomes for the *same* scenario realization. Each
    resample draws whole realizations with replacement and keeps both arm values together, which is
    what preserves the pairing. Optionally stratify by condition cell.
    """
    if len(first) != len(second):
        raise ValueError("paired sequences must have equal length")
    n = len(first)
    if n == 0:
        return BootstrapEstimate(
            point=None, resamples=0, seed=seed, n_units=0, unit=unit, method="none",
            confidence_level=confidence_level, stratified=strata is not None,
            undefined_reason="no paired units",
        )
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    point = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    idx = _stratified_index_draws(strata, n, resamples, rng)
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    low, high = _percentile_ci(diffs, confidence_level)
    return BootstrapEstimate(
        point=point, ci_low=low, ci_high=high, confidence_level=confidence_level,
        resamples=resamples, seed=seed, n_units=n, unit=unit, stratified=strata is not None,
        method="paired_percentile",
    )


def cluster_bootstrap_statistic(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    seed: int,
    resamples: int = 10000,
    confidence_level: float = 0.95,
    statistic: Callable[[np.ndarray], float] | None = None,
    unit: str = "cluster",
) -> BootstrapEstimate:
    """Percentile bootstrap over **clusters**, for rows that are not independent.

    One scenario realization can contribute several rows (for example the same realization scored for
    several audit questions, or reused across severity cells). Resampling rows would treat those rows
    as independent replications and shrink the interval dishonestly, so whole clusters are drawn with
    replacement and all their rows travel together.
    """
    if len(values) != len(clusters):
        raise ValueError("values and clusters must have equal length")
    stat = statistic or (lambda arr: float(arr.mean()))
    if not values:
        return BootstrapEstimate(
            point=None, resamples=0, seed=seed, n_units=0, unit=unit, method="none",
            confidence_level=confidence_level,
            undefined_reason="no rows to resample",
        )
    arr = np.asarray(values, dtype=float)
    groups: dict[str, list[int]] = {}
    for i, key in enumerate(clusters):
        groups.setdefault(key, []).append(i)
    keys = list(groups)
    members = [np.asarray(groups[k], dtype=np.int64) for k in keys]
    point = stat(arr)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(resamples, len(keys)))
    samples = np.empty(resamples, dtype=float)
    for r in range(resamples):
        picked = np.concatenate([members[j] for j in draws[r]])
        samples[r] = stat(arr[picked])
    low, high = _percentile_ci(samples, confidence_level)
    return BootstrapEstimate(
        point=point, ci_low=low, ci_high=high, confidence_level=confidence_level,
        resamples=resamples, seed=seed, n_units=len(keys), unit=unit, method="cluster_percentile",
    )


# --------------------------------------------------------------------------------------
# Continuous outcomes
# --------------------------------------------------------------------------------------
class DistributionSummary(StrictModel):
    """Median/IQR summary of a continuous outcome such as detection delay or completion time.

    Median and IQR rather than mean and standard deviation because these distributions are censored
    (an undetected violation has no detection delay at all) and skewed. The count of *missing* values
    is carried alongside so a short delay list cannot be mistaken for a fast monitor.
    """

    n: int = Field(default=0, ge=0)
    n_missing: int = Field(default=0, ge=0)
    median: float | None = None
    q1: float | None = None
    q3: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    undefined_reason: str | None = None

    @property
    def iqr(self) -> float | None:
        if self.q1 is None or self.q3 is None:
            return None
        return self.q3 - self.q1

    def describe(self) -> str:
        if self.median is None:
            return f"undefined (n=0 of {self.n + self.n_missing}; {self.undefined_reason or 'no values'})"
        return (f"median {self.median:.2f} (IQR {self.q1:.2f}-{self.q3:.2f}, n={self.n}, "
                f"missing={self.n_missing})")


def summarize_distribution(
    values: Sequence[float | None],
    *,
    undefined_reason: str | None = None,
) -> DistributionSummary:
    """Summarize a continuous outcome, counting ``None`` entries as explicitly missing."""
    present = [float(v) for v in values if v is not None]
    missing = len(values) - len(present)
    if not present:
        return DistributionSummary(
            n=0, n_missing=missing, undefined_reason=undefined_reason or "no observed values"
        )
    arr = np.asarray(present, dtype=float)
    q1, med, q3 = (float(x) for x in np.percentile(arr, [25.0, 50.0, 75.0]))
    return DistributionSummary(
        n=len(present), n_missing=missing, median=med, q1=q1, q3=q3,
        minimum=float(arr.min()), maximum=float(arr.max()), mean=float(arr.mean()),
    )


def hodges_lehmann_paired(differences: Sequence[float]) -> float | None:
    """One-sample Hodges-Lehmann estimate: the median of all Walsh averages of the differences.

    This is the location shift that the paired signed-rank procedure estimates. It is preferred to the
    mean difference for completion time and detection delay, which have heavy tails caused by a few
    aborted or very late episodes.
    """
    if not differences:
        return None
    d = np.asarray(differences, dtype=float)
    i, j = np.triu_indices(d.size, k=0)
    walsh = (d[i] + d[j]) / 2.0
    return float(np.median(walsh))


def paired_shift_estimate(
    first: Sequence[float | None],
    second: Sequence[float | None],
    *,
    seed: int,
    resamples: int = 2000,
    confidence_level: float = 0.95,
    unit: str = "scenario_realization",
) -> BootstrapEstimate:
    """Hodges-Lehmann paired shift (``first`` minus ``second``) with a percentile bootstrap CI.

    Pairs where either value is missing are dropped and counted, because a shift estimate can only be
    formed where both arms produced the outcome. ``resamples`` defaults below the proportion bootstrap
    count: the Walsh-average estimator is O(n^2) per replicate, and the extra replicates buy less than
    the honest width of this interval.
    """
    if len(first) != len(second):
        raise ValueError("paired sequences must have equal length")
    pairs = [(float(a), float(b)) for a, b in zip(first, second, strict=True)
             if a is not None and b is not None]
    if not pairs:
        return BootstrapEstimate(
            point=None, resamples=0, seed=seed, n_units=0, unit=unit, method="none",
            confidence_level=confidence_level,
            undefined_reason="no realization has the outcome in both arms",
        )
    d = np.asarray([a - b for a, b in pairs], dtype=float)
    point = hodges_lehmann_paired(d.tolist())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(resamples, d.size))
    samples = np.empty(resamples, dtype=float)
    for r in range(resamples):
        samples[r] = hodges_lehmann_paired(d[idx[r]].tolist()) or 0.0
    low, high = _percentile_ci(samples, confidence_level)
    return BootstrapEstimate(
        point=point, ci_low=low, ci_high=high, confidence_level=confidence_level,
        resamples=resamples, seed=seed, n_units=d.size, unit=unit, method="paired_percentile",
    )
