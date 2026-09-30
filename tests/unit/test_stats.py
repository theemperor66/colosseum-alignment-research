"""Statistics tests checked against hand-computable references, not against the implementation.

Every expected number below is either a published Wilson/McNemar value or a quantity that can be
derived by hand from the definition, so the tests fail if the implementation drifts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from colosseum_assurance.analysis.stats import (
    _stratified_index_draws,
    cluster_bootstrap_statistic,
    hodges_lehmann_paired,
    mcnemar_exact,
    normal_cdf,
    normal_quantile,
    paired_bootstrap_difference,
    paired_shift_estimate,
    summarize_distribution,
    wilson_interval,
)


# --------------------------------------------------------------------------------------
# Normal helpers
# --------------------------------------------------------------------------------------
def test_normal_quantile_matches_known_critical_values() -> None:
    assert normal_quantile(0.975) == pytest.approx(1.959963985, abs=1e-8)
    assert normal_quantile(0.995) == pytest.approx(2.575829304, abs=1e-8)
    assert normal_quantile(0.95) == pytest.approx(1.644853627, abs=1e-8)
    assert normal_quantile(0.5) == pytest.approx(0.0, abs=1e-12)


def test_normal_cdf_and_quantile_round_trip() -> None:
    for p in (0.001, 0.01, 0.2, 0.5, 0.8, 0.99, 0.999):
        assert normal_cdf(normal_quantile(p)) == pytest.approx(p, abs=1e-12)


def test_normal_quantile_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        normal_quantile(0.0)
    with pytest.raises(ValueError):
        normal_quantile(1.0)


# --------------------------------------------------------------------------------------
# Wilson interval
# --------------------------------------------------------------------------------------
def test_wilson_interval_for_5_of_10_matches_published_value() -> None:
    """Wilson 95% CI for 5/10 is (0.2366, 0.7634)."""
    est = wilson_interval(5, 10)
    assert est.point == pytest.approx(0.5)
    assert est.ci_low == pytest.approx(0.237, abs=5e-4)
    assert est.ci_high == pytest.approx(0.763, abs=5e-4)
    assert (est.numerator, est.denominator) == (5, 10)
    assert est.method == "wilson"
    assert est.undefined_reason is None


def test_wilson_interval_for_zero_of_10_has_exact_zero_lower_bound() -> None:
    """Wilson 95% CI for 0/10 is (0.000, 0.2775): the lower bound is analytically exactly zero."""
    est = wilson_interval(0, 10)
    assert est.point == 0.0
    assert est.ci_low == pytest.approx(0.0, abs=1e-12)
    assert est.ci_high == pytest.approx(0.278, abs=5e-4)
    # A zero-width interval here would falsely claim that zero observed failures proves zero risk.
    assert est.ci_high > 0.25


def test_wilson_interval_is_symmetric_at_the_other_extreme() -> None:
    low = wilson_interval(0, 10)
    high = wilson_interval(10, 10)
    assert high.ci_high == pytest.approx(1.0, abs=1e-12)
    assert high.ci_low == pytest.approx(1.0 - low.ci_high, abs=1e-12)


def test_wilson_interval_widens_at_99_percent() -> None:
    narrow = wilson_interval(5, 10, 0.95)
    wide = wilson_interval(5, 10, 0.99)
    assert wide.ci_high - wide.ci_low > narrow.ci_high - narrow.ci_low


def test_wilson_interval_with_empty_denominator_is_undefined_not_zero() -> None:
    est = wilson_interval(0, 0, undefined_reason="no_accepted_episodes")
    assert est.point is None
    assert est.ci_low is None and est.ci_high is None
    assert est.denominator == 0
    assert est.undefined_reason == "no_accepted_episodes"
    assert est.method == "none"
    assert "undefined" in est.describe()


def test_wilson_interval_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)


# --------------------------------------------------------------------------------------
# Exact McNemar
# --------------------------------------------------------------------------------------
def test_mcnemar_exact_b1_c9_matches_hand_computed_p_value() -> None:
    """2 * P(X <= 1) for X ~ Binomial(10, 0.5) = 2 * 11/1024 = 0.021484375."""
    result = mcnemar_exact(1, 9)
    assert result.p_value == pytest.approx(0.0215, abs=5e-5)
    assert result.p_value == pytest.approx(2 * 11 / 1024, abs=1e-15)
    assert (result.b, result.c, result.n_discordant) == (1, 9, 10)
    assert result.method == "exact_binomial"


def test_mcnemar_exact_is_symmetric_in_b_and_c() -> None:
    assert mcnemar_exact(1, 9).p_value == mcnemar_exact(9, 1).p_value


def test_mcnemar_exact_small_cases_match_the_binomial_definition() -> None:
    assert mcnemar_exact(0, 5).p_value == pytest.approx(2 * (1 / 32), abs=1e-15)
    assert mcnemar_exact(0, 1).p_value == pytest.approx(1.0, abs=1e-15)
    # b == c: 2 * P(X <= b) exceeds 1 and must be capped rather than reported above one.
    assert mcnemar_exact(3, 3).p_value == pytest.approx(1.0)


def test_mcnemar_without_discordant_pairs_is_uninformative_not_significant() -> None:
    result = mcnemar_exact(0, 0, concordant_pairs=40)
    assert result.p_value == 1.0
    assert result.method == "none"
    assert result.undefined_reason is not None
    assert result.n_pairs == 40


def test_mcnemar_counts_concordant_pairs_separately() -> None:
    result = mcnemar_exact(2, 1, concordant_pairs=17)
    assert result.n_discordant == 3
    assert result.n_pairs == 20


# --------------------------------------------------------------------------------------
# Paired bootstrap
# --------------------------------------------------------------------------------------
def test_paired_bootstrap_point_is_the_observed_difference() -> None:
    first = [True, True, False, False, True]
    second = [False, True, False, False, False]
    est = paired_bootstrap_difference(first, second, seed=7717, resamples=500)
    assert est.point == pytest.approx(3 / 5 - 1 / 5)
    assert est.n_units == 5
    assert est.unit == "scenario_realization"


def test_degenerate_paired_bootstrap_gives_a_zero_width_interval() -> None:
    """If every realization shows the same difference, resampling cannot produce any spread."""
    est = paired_bootstrap_difference([1, 1, 1, 1], [0, 0, 0, 0], seed=11, resamples=400)
    assert est.point == pytest.approx(1.0)
    assert est.ci_low == pytest.approx(1.0)
    assert est.ci_high == pytest.approx(1.0)
    assert est.ci_high - est.ci_low == pytest.approx(0.0, abs=1e-12)


def test_paired_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    a = [1, 0, 1, 1, 0, 0, 1, 0]
    b = [0, 0, 1, 0, 0, 1, 1, 0]
    one = paired_bootstrap_difference(a, b, seed=7717, resamples=800)
    two = paired_bootstrap_difference(a, b, seed=7717, resamples=800)
    other = paired_bootstrap_difference(a, b, seed=7718, resamples=800)
    assert (one.ci_low, one.ci_high) == (two.ci_low, two.ci_high)
    assert (one.ci_low, one.ci_high) != (other.ci_low, other.ci_high)


def test_paired_bootstrap_keeps_the_pair_together() -> None:
    """Perfectly matched arms must give a zero difference in every resample.

    If the resampler drew the two arms independently, the difference would fluctuate and the interval
    would not collapse, which is exactly the mistake this assertion catches.
    """
    values = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    est = paired_bootstrap_difference(values, values, seed=3, resamples=600)
    assert est.point == 0.0
    assert est.ci_low == pytest.approx(0.0)
    assert est.ci_high == pytest.approx(0.0)


def test_stratified_resampling_preserves_the_cell_sizes() -> None:
    strata = ["cell_a", "cell_a", "cell_a", "cell_b", "cell_b"]
    rng = np.random.default_rng(42)
    draws = _stratified_index_draws(strata, len(strata), 50, rng)
    assert draws.shape == (50, 5)
    for row in draws:
        assert sum(1 for i in row if strata[i] == "cell_a") == 3
        assert sum(1 for i in row if strata[i] == "cell_b") == 2


def test_stratified_bootstrap_is_reported_as_stratified() -> None:
    est = paired_bootstrap_difference(
        [1, 0, 1, 0], [0, 0, 1, 1], seed=5, resamples=300, strata=["a", "a", "b", "b"]
    )
    assert est.stratified is True
    assert est.n_units == 4


def test_paired_bootstrap_with_no_units_is_undefined() -> None:
    est = paired_bootstrap_difference([], [], seed=1, resamples=100)
    assert est.point is None
    assert est.undefined_reason is not None
    assert est.method == "none"


def test_paired_bootstrap_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_difference([1, 0], [1], seed=1, resamples=10)


# --------------------------------------------------------------------------------------
# Cluster bootstrap
# --------------------------------------------------------------------------------------
def test_cluster_bootstrap_counts_clusters_not_rows() -> None:
    values = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    clusters = ["ep1", "ep1", "ep2", "ep2", "ep3", "ep3"]
    est = cluster_bootstrap_statistic(values, clusters, seed=7717, resamples=400)
    assert est.n_units == 3
    assert est.point == pytest.approx(4 / 6)
    assert est.method == "cluster_percentile"


def test_cluster_bootstrap_is_wider_than_treating_rows_as_independent() -> None:
    """Rows inside one episode are perfectly correlated here, so ignoring clusters understates width."""
    values: list[float] = []
    clustered: list[str] = []
    independent: list[str] = []
    for episode in range(10):
        value = 1.0 if episode % 2 == 0 else 0.0
        for question in range(4):
            values.append(value)
            clustered.append(f"ep{episode}")
            independent.append(f"ep{episode}-q{question}")
    grouped = cluster_bootstrap_statistic(values, clustered, seed=7717, resamples=1500)
    ungrouped = cluster_bootstrap_statistic(values, independent, seed=7717, resamples=1500)
    assert grouped.n_units == 10
    assert ungrouped.n_units == 40
    assert (grouped.ci_high - grouped.ci_low) > (ungrouped.ci_high - ungrouped.ci_low)


def test_cluster_bootstrap_with_no_rows_is_undefined() -> None:
    est = cluster_bootstrap_statistic([], [], seed=1, resamples=10)
    assert est.point is None and est.undefined_reason is not None


def test_cluster_bootstrap_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        cluster_bootstrap_statistic([1.0, 2.0], ["a"], seed=1, resamples=10)


# --------------------------------------------------------------------------------------
# Continuous outcomes
# --------------------------------------------------------------------------------------
def test_summarize_distribution_matches_hand_computed_quartiles() -> None:
    summary = summarize_distribution([1.0, 2.0, 3.0, 4.0])
    assert summary.median == pytest.approx(2.5)
    assert summary.q1 == pytest.approx(1.75)
    assert summary.q3 == pytest.approx(3.25)
    assert summary.iqr == pytest.approx(1.5)
    assert (summary.minimum, summary.maximum) == (1.0, 4.0)
    assert (summary.n, summary.n_missing) == (4, 0)


def test_summarize_distribution_counts_missing_values_explicitly() -> None:
    summary = summarize_distribution([2.0, None, 4.0, None])
    assert summary.n == 2
    assert summary.n_missing == 2
    assert summary.median == pytest.approx(3.0)
    assert "missing=2" in summary.describe()


def test_summarize_distribution_with_no_values_is_undefined() -> None:
    summary = summarize_distribution([None, None], undefined_reason="no_violation_episodes")
    assert summary.median is None
    assert summary.iqr is None
    assert summary.n == 0 and summary.n_missing == 2
    assert summary.undefined_reason == "no_violation_episodes"


def test_hodges_lehmann_is_the_median_of_walsh_averages() -> None:
    """Walsh averages of [1, 2, 3] are [1, 1.5, 2, 2, 2.5, 3]; their median is 2.0."""
    assert hodges_lehmann_paired([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert hodges_lehmann_paired([5.0]) == pytest.approx(5.0)
    assert hodges_lehmann_paired([]) is None


def test_hodges_lehmann_shifts_with_the_data() -> None:
    base = [0.5, 1.5, 2.5, 9.0]
    shifted = [v + 3.0 for v in base]
    assert hodges_lehmann_paired(shifted) == pytest.approx(hodges_lehmann_paired(base) + 3.0)


def test_paired_shift_estimate_on_a_constant_difference_has_zero_width() -> None:
    first = [10.0, 12.0, 14.0]
    second = [8.0, 10.0, 12.0]
    est = paired_shift_estimate(first, second, seed=7717, resamples=300)
    assert est.point == pytest.approx(2.0)
    assert est.ci_low == pytest.approx(2.0)
    assert est.ci_high == pytest.approx(2.0)
    assert est.n_units == 3


def test_paired_shift_estimate_drops_pairs_with_a_missing_value() -> None:
    est = paired_shift_estimate([1.0, None, 3.0], [0.0, 0.0, 1.0], seed=7717, resamples=200)
    assert est.n_units == 2
    # Differences are [1, 2]; Walsh averages [1, 1.5, 2]; median 1.5.
    assert est.point == pytest.approx(1.5)


def test_paired_shift_estimate_without_usable_pairs_is_undefined() -> None:
    est = paired_shift_estimate([None, None], [1.0, 2.0], seed=1, resamples=50)
    assert est.point is None
    assert est.undefined_reason is not None
    assert est.n_units == 0


def test_walsh_average_count_is_the_triangular_number() -> None:
    """Guard the O(n^2) estimator against an off-by-one in the index generation."""
    values = [float(v) for v in range(6)]
    i, j = np.triu_indices(len(values), k=0)
    assert len(i) == len(values) * (len(values) + 1) // 2
    assert hodges_lehmann_paired(values) == pytest.approx(2.5)
    assert math.isclose(hodges_lehmann_paired([2.0, 2.0, 2.0]), 2.0)


def test_wilson_bounds_always_bracket_the_point_estimate() -> None:
    """Float noise must never put a bound on the wrong side of p-hat.

    A bound below the point estimate produces a negative error-bar length, which matplotlib rejects;
    this was a real crash in the tradeoff figure for a 0/5 coverage estimate.
    """
    for trials in range(1, 40):
        for successes in range(trials + 1):
            est = wilson_interval(successes, trials)
            assert est.ci_low <= est.point <= est.ci_high, f"{successes}/{trials} bracket broken"
            assert 0.0 <= est.ci_low
            assert est.ci_high <= 1.0
