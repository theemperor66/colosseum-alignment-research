"""Scientific figures for one run, drawn with matplotlib on the Agg backend.

Rules this module follows:

* **No figure invents a number.** Every value drawn comes from a :class:`RunAnalysis` field or an audit
  score record. Where the underlying quantity is undefined (no accepted episodes, no violation
  episodes, no monitor) the figure prints ``undefined (n=0)`` in place of a bar rather than drawing a
  zero, because a drawn zero is the exact misreading this study is about.
* **Every figure is self-identifying.** Title and caption carry the run class and the protocol short
  hash, so a figure cannot be separated from the protocol version that produced it.
* **Fixture and smoke runs are watermarked.** A diagonal ``SYNTHETIC FIXTURE`` watermark crosses the
  whole canvas so a screenshot of engineering data cannot be mistaken for evidence.
* Agg only, no seaborn, no network fonts: figures must render identically on a headless worker.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")  # headless workers have no display; set before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected first)
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from colosseum_assurance.analysis.metrics import ArmMetrics, RunAnalysis  # noqa: E402
from colosseum_assurance.analysis.stats import DistributionSummary, ProportionEstimate  # noqa: E402
from colosseum_assurance.protocol.spec import ProtocolConfig  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only
    from colosseum_assurance.audit.scoring import AuditScoreSummary

WATERMARK_TEXT = "SYNTHETIC FIXTURE"
ARM_COLOURS = {
    "A0_unguarded": "#6c6c6c",
    "A1_policy_only": "#c1553b",
    "A2_assumption_aware": "#2f6f9f",
}
_FALLBACK_COLOURS = ["#7a5195", "#ef5675", "#ffa600", "#003f5c"]

__all__ = [
    "WATERMARK_TEXT",
    "render_all",
    "figure_audit_reconstruction",
    "figure_cell_heatmap",
    "figure_detection_delay",
    "figure_safety_completion_tradeoff",
    "figure_violation_fractions",
    "write_all_figures",
]


# --------------------------------------------------------------------------------------
# Shared chrome
# --------------------------------------------------------------------------------------
def _arm_colour(arm_id: str, index: int) -> str:
    return ARM_COLOURS.get(arm_id, _FALLBACK_COLOURS[index % len(_FALLBACK_COLOURS)])


def _stamp(analysis: RunAnalysis) -> str:
    return f"run_class={analysis.run_class} | protocol {analysis.protocol_short_hash}"


def _finish(fig: Any, analysis: RunAnalysis, caption: str, path: Path, caption_y: float = 0.005) -> Path:
    """Add the provenance caption and the synthetic watermark, then save and close."""
    if analysis.is_synthetic:
        fig.text(
            0.5, 0.5, WATERMARK_TEXT, fontsize=42, color="#b00020", alpha=0.16,
            ha="center", va="center", rotation=28, zorder=10, fontweight="bold",
        )
    full_caption = f"{caption}\n{_stamp(analysis)} | analysis {analysis.analysis_version}"
    if analysis.is_synthetic:
        full_caption = f"SYNTHETIC FIXTURE DATA - NOT EXPERIMENTAL EVIDENCE. {full_caption}"
    fig.text(0.01, caption_y, full_caption, fontsize=7, color="#333333", ha="left", va="top", wrap=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _err_pair(estimate: ProportionEstimate) -> tuple[float, float]:
    """Return (below, above) error-bar lengths, never negative.

    A ``ci_low`` of exactly 0.0 is falsy, so this uses explicit ``is None`` checks: an ``or``-based
    fallback silently turns a genuine zero bound into the point estimate.
    """
    if estimate.point is None:
        return 0.0, 0.0
    low = estimate.ci_low if estimate.ci_low is not None else estimate.point
    high = estimate.ci_high if estimate.ci_high is not None else estimate.point
    return max(estimate.point - low, 0.0), max(high - estimate.point, 0.0)


def _prop_bar(ax: Any, x: float, estimate: ProportionEstimate, colour: str, width: float) -> None:
    """Draw one proportion with its interval, or an explicit 'undefined' marker."""
    if estimate.point is None:
        ax.text(x, 0.02, f"undefined\n(n={estimate.denominator})\n{estimate.undefined_reason or ''}",
                ha="center", va="bottom",
                fontsize=6.5, color="#b00020", rotation=90)
        return
    ax.bar(x, estimate.point, width=width, color=colour, edgecolor="black", linewidth=0.4)
    low, high = _err_pair(estimate)
    has_interval = estimate.ci_low is not None and estimate.ci_high is not None
    if has_interval:
        ax.errorbar(x, estimate.point, yerr=[[low], [high]], fmt="none",
                    ecolor="black", elinewidth=1.0, capsize=3)
    suffix = "" if has_interval else "\nCI unavailable"
    ax.text(x, min(1.0, estimate.point + high) + 0.03,
            f"{estimate.numerator}/{estimate.denominator}{suffix}",
            ha="center", va="bottom", fontsize=7)


# --------------------------------------------------------------------------------------
# (a) paired violation fractions by arm
# --------------------------------------------------------------------------------------
def figure_violation_fractions(analysis: RunAnalysis, out_dir: Path | str,
                               filename: str = "fig_a_violation_fractions.png") -> Path:
    """All-episode physical and procedural violation fractions per arm, with Wilson intervals.

    Physical and procedural obligations are drawn in separate panels because the protocol scores them
    separately; combining them would hide which obligation failed.
    """
    arms = analysis.arm_ids
    panels = [
        ("physical_violation", "Physical violation (geofence, collision)"),
        ("procedural_violation", "Procedural violation (authorization, supervision deadline)"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (field, subtitle) in zip(axes, panels, strict=True):
        for i, arm_id in enumerate(arms):
            metrics: ArmMetrics = analysis.overall[arm_id]
            _prop_bar(ax, i, getattr(metrics, field), _arm_colour(arm_id, i), 0.6)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([a.replace("_", "\n", 1) for a in arms], fontsize=8)
        ax.set_ylim(0.0, 1.15)
        ax.set_title(subtitle, fontsize=9)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("fraction of assessed episodes")
    paired = [c for c in analysis.primary_comparisons if c.role == "primary"]
    pair_text = "; ".join(
        f"{c.outcome_id}: paired diff {c.difference.describe()}, exact McNemar b={c.mcnemar.b} "
        f"c={c.mcnemar.c} p={'undefined' if c.mcnemar.p_value is None else f'{c.mcnemar.p_value:.4f}'}"
        for c in paired
    ) or "no paired comparison computed"
    fig.suptitle(f"All-episode violation fractions by arm ({_stamp(analysis)})", fontsize=11)
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    caption = (
        "Bars are all-episode fractions over every assessed episode; labels give successes/denominator. "
        "Error bars are Wilson 95% intervals. Paired statistics over shared scenario realizations: "
        f"{pair_text}."
    )
    return _finish(fig, analysis, caption, Path(out_dir) / filename)


# --------------------------------------------------------------------------------------
# (b) safety / completion tradeoff
# --------------------------------------------------------------------------------------
def figure_safety_completion_tradeoff(analysis: RunAnalysis, out_dir: Path | str,
                                      filename: str = "fig_b_safety_completion_tradeoff.png") -> Path:
    """Assurance coverage against violation fraction, annotated by arm.

    The point of this figure is to make an always-abstain configuration visibly bad: such an arm sits at
    coverage 0 no matter how clean its conditional rate looks, and its safe-completion annotation shows
    what the abstention cost.
    """
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    undefined: list[str] = []
    for i, arm_id in enumerate(analysis.arm_ids):
        m = analysis.overall[arm_id]
        viol = m.any_violation
        cov = m.assurance_coverage
        if viol.point is None:
            # An arm attempted but never scored has no violation fraction to plot. It is named here,
            # with its attempted count and coverage, so the figure cannot silently omit a failed arm.
            attempted = "unknown" if m.n_attempted is None else str(m.n_attempted)
            undefined.append(
                f"{arm_id}: violation fraction undefined ({viol.undefined_reason}); attempted "
                f"{attempted}, assessed {m.n_assessed}, coverage {cov.describe()}"
            )
            continue
        if cov.point is None:
            undefined.append(f"{arm_id}: assurance coverage undefined ({cov.undefined_reason}), "
                             f"violation fraction {viol.point:.3f} (n={viol.denominator})")
            continue
        colour = _arm_colour(arm_id, i)
        ax.scatter(cov.point, viol.point, s=110, color=colour, edgecolor="black", zorder=3)
        cov_low, cov_high = _err_pair(cov)
        viol_low, viol_high = _err_pair(viol)
        ax.errorbar(
            cov.point, viol.point, xerr=[[cov_low], [cov_high]], yerr=[[viol_low], [viol_high]],
            fmt="none", ecolor=colour, elinewidth=1.0, capsize=2, alpha=0.8, zorder=2,
        )
        safe = m.safe_mission_completion
        safe_text = "undefined" if safe.point is None else f"{safe.point:.2f}"
        ax.annotate(
            f"{arm_id}\ncoverage {cov.numerator}/{cov.denominator}\n"
            f"safe completion {safe_text}\naccepted n={m.n_accepted}",
            (cov.point, viol.point), textcoords="offset points", xytext=(10, 8), fontsize=7.5,
        )
    ax.set_xlabel("assurance coverage (accepted / attempted episodes)")
    ax.set_ylabel("any-violation fraction (all assessed episodes)")
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.10)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_title(f"Safety / availability tradeoff ({_stamp(analysis)})", fontsize=11)
    if undefined:
        ax.text(0.02, 1.02, "\n".join(undefined), fontsize=7, color="#b00020", va="top")
    fig.tight_layout()
    caption = (
        "One point per arm. A conditional false-assurance rate must be read together with the coverage "
        "on this axis: an arm that accepts nothing sits at coverage 0 and its conditional rate is "
        "undefined, not good. Arms without a monitor have undefined coverage and are listed as text."
    )
    return _finish(fig, analysis, caption, Path(out_dir) / filename)


# --------------------------------------------------------------------------------------
# (c) detection delay
# --------------------------------------------------------------------------------------
def _summary_box(ax: Any, x: float, summary: DistributionSummary, colour: str) -> None:
    if summary.median is None:
        ax.text(x, 0.0, f"no observed delays\n(n=0, missing={summary.n_missing})", ha="center",
                va="bottom", fontsize=7, color="#b00020", rotation=90)
        return
    q1 = summary.q1 if summary.q1 is not None else summary.median
    q3 = summary.q3 if summary.q3 is not None else summary.median
    lo = summary.minimum if summary.minimum is not None else q1
    hi = summary.maximum if summary.maximum is not None else q3
    ax.add_patch(plt.Rectangle((x - 0.22, q1), 0.44, max(q3 - q1, 1e-9), facecolor=colour, alpha=0.55,
                               edgecolor="black", linewidth=0.6, zorder=2))
    ax.plot([x - 0.22, x + 0.22], [summary.median, summary.median], color="black", linewidth=1.6, zorder=3)
    ax.plot([x, x], [lo, q1], color="black", linewidth=0.8, zorder=1)
    ax.plot([x, x], [q3, hi], color="black", linewidth=0.8, zorder=1)
    ax.text(x, hi, f"n={summary.n}\nmissing={summary.n_missing}", ha="center", va="bottom", fontsize=7)


def figure_detection_delay(analysis: RunAnalysis, out_dir: Path | str,
                           filename: str = "fig_c_detection_delay.png") -> Path:
    """Detection-delay distribution per arm, drawn from the median/IQR summary.

    ``missing`` counts violation episodes with no detection at all. A short delay box beside a large
    missing count means the monitor was fast on the few violations it saw, not that it was reliable.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, arm_id in enumerate(analysis.arm_ids):
        m = analysis.overall[arm_id]
        _summary_box(ax, i, m.detection_delay_s, _arm_colour(arm_id, i))
    ax.set_xticks(range(len(analysis.arm_ids)))
    ax.set_xticklabels([a.replace("_", "\n", 1) for a in analysis.arm_ids], fontsize=8)
    ax.set_ylabel("detection delay (s) after the independently assessed violation")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_title(f"Detection delay by arm ({_stamp(analysis)})", fontsize=11)
    ax.relim()
    ax.autoscale_view()
    fig.tight_layout()
    missed = "; ".join(
        f"{a}: missed detection {analysis.overall[a].missed_detection.describe()}"
        for a in analysis.arm_ids
    )
    caption = ("Box: interquartile range; line: median; whiskers: observed minimum and maximum. "
               f"Missed detections are a separate outcome - {missed}.")
    return _finish(fig, analysis, caption, Path(out_dir) / filename)


# --------------------------------------------------------------------------------------
# (d) audit reconstruction accuracy
# --------------------------------------------------------------------------------------
def figure_audit_reconstruction(analysis: RunAnalysis, audit: AuditScoreSummary, out_dir: Path | str,
                                filename: str = "fig_d_audit_reconstruction.png") -> Path:
    """Correct reconstruction with intervals over scenarios, preserving correlated arms and questions."""
    variants = list(audit.variant_ids)
    questions = list(audit.question_ids)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.7 * len(variants) * max(len(questions), 1)), 4.6))
    group_width = 0.8
    bar_width = group_width / max(len(variants), 1)
    handles: list[Any] = []
    for vi, variant_id in enumerate(variants):
        variant = audit.variants[variant_id]
        colour = _FALLBACK_COLOURS[vi % len(_FALLBACK_COLOURS)]
        for qi, question_id in enumerate(questions):
            score = variant.per_question.get(question_id)
            x = qi - group_width / 2 + bar_width * (vi + 0.5)
            if score is None:
                continue
            _prop_bar(ax, x, score.correct_rate, colour, bar_width * 0.92)
        # A proxy patch, because an empty bar container does not carry the variant colour into the
        # legend and would mislabel which record variant a bar belongs to. A variant that keeps a
        # redundant recovery route is labelled as a probe: it did not remove the information.
        probe = " [redundancy probe]" if getattr(variant, "retains_redundant_routes", False) else ""
        handles.append(Patch(facecolor=colour, edgecolor="black",
                             label=f"{variant_id}{probe} ({variant.n_episodes} episodes, "
                                   f"{getattr(variant, 'n_scenarios', 'unrecorded')} scenarios)"))
    ax.set_xticks(range(len(questions)))
    ax.set_xticklabels(questions, fontsize=8, rotation=12)
    ax.set_ylim(0.0, 1.2)
    ax.set_ylabel("fraction of episodes reconstructed correctly")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(handles=handles, fontsize=7, loc="upper right", ncol=1)
    ax.set_title(f"Audit reconstruction by record variant and question ({_stamp(analysis)})", fontsize=11)
    fig.tight_layout()
    overall = "; ".join(
        f"{v}: overall correct {audit.variants[v].overall_correct_rate.describe()}" for v in variants
    )
    probes = [v for v in variants if getattr(audit.variants[v], "retains_redundant_routes", False)]
    probe_note = (
        f" Variants marked [redundancy probe] ({', '.join(probes)}) keep a redundant recovery route and "
        "therefore do not remove the information; they test whether the procedure uses that route."
        if probes else ""
    )
    caption = (
        "Bars are per-episode correct-reconstruction fractions with scenario-cluster intervals; labels give "
        "correct/denominator. Insufficient evidence is scored as a third outcome, not as an error. "
        "Matched arms are resampled together; fewer than two scenarios gives no interval. "
        f"Cluster bootstrap over scenarios for the all-question rate - {overall}.{probe_note} Offline "
        "record ablation changes only what an auditor can establish; it cannot change flight safety."
    )
    return _finish(fig, analysis, caption, Path(out_dir) / filename)


# --------------------------------------------------------------------------------------
# (e) per-cell heatmap over the delay grid
# --------------------------------------------------------------------------------------
def _ordered_levels(keys: Sequence[str], protocol_levels: Sequence[str] | None) -> list[str]:
    """Order condition levels by the protocol when available, else nominal < moderate < severe."""
    present = list(dict.fromkeys(keys))
    if protocol_levels:
        ordered = [lv for lv in protocol_levels if lv in present]
        ordered += [k for k in present if k not in ordered]
        return ordered
    rank = {"nominal": 0, "moderate": 1, "severe": 2}
    return sorted(present, key=lambda k: (min((v for word, v in rank.items() if word in k), default=9), k))


def figure_cell_heatmap(analysis: RunAnalysis, out_dir: Path | str,
                        filename: str = "fig_e_cell_heatmap.png",
                        outcome_field: str = "physical_violation",
                        protocol: ProtocolConfig | None = None) -> Path:
    """Violation fraction per condition cell, one panel per arm, over the delay grid.

    Cells with no episodes print ``n/a`` instead of a colour, so an empty cell cannot be read as a
    measured zero.
    """
    cell_rows = [m for m in analysis.by_stratum if m.stratum_kind == "cell"]
    obs_order = _ordered_levels(
        [m.stratum_key.split("__")[0] for m in cell_rows],
        [lv.level_id for lv in protocol.conditions.observation_delay_levels] if protocol else None,
    )
    sup_order = _ordered_levels(
        [m.stratum_key.split("__")[-1] for m in cell_rows],
        [lv.level_id for lv in protocol.conditions.supervision_delay_levels] if protocol else None,
    )
    arms = analysis.arm_ids
    fig, axes = plt.subplots(1, max(len(arms), 1), figsize=(3.9 * max(len(arms), 1), 4.0), squeeze=False)
    lookup = {(m.arm_id, m.stratum_key): m for m in cell_rows}
    for ai, arm_id in enumerate(arms):
        ax = axes[0][ai]
        grid = np.full((len(obs_order), len(sup_order)), np.nan)
        for i, obs in enumerate(obs_order):
            for j, sup in enumerate(sup_order):
                m = lookup.get((arm_id, f"{obs}__{sup}"))
                if m is None:
                    continue
                est: ProportionEstimate = getattr(m, outcome_field)
                if est.point is None:
                    continue
                grid[i, j] = est.point
                ax.text(j, i, f"{est.point:.2f}\n{est.numerator}/{est.denominator}", ha="center",
                        va="center", fontsize=7,
                        color="white" if est.point > 0.55 else "black")
        for i in range(len(obs_order)):
            for j in range(len(sup_order)):
                if np.isnan(grid[i, j]):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="#b00020")
        ax.imshow(np.ma.masked_invalid(grid), cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(sup_order)))
        ax.set_xticklabels(sup_order, fontsize=7, rotation=20)
        ax.set_yticks(range(len(obs_order)))
        ax.set_yticklabels(obs_order if ai == 0 else [""] * len(obs_order), fontsize=7)
        ax.set_title(arm_id, fontsize=9)
        if ai == 0:
            ax.set_ylabel("observation delay level")
        ax.set_xlabel("supervision delay level")
    mappable = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(vmin=0.0, vmax=1.0))
    fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.85,
                 label=outcome_field.replace("_", " ") + " fraction")
    fig.suptitle(f"{outcome_field.replace('_', ' ')} by condition cell ({_stamp(analysis)})", fontsize=11)
    caption = (
        "Each tile shows the fraction and its successes/denominator for one delay cell. Empty cells are "
        "printed as n/a and are not coloured. Per-cell denominators are small by design; read the "
        "pooled comparison for the primary claim."
    )
    return _finish(fig, analysis, caption, Path(out_dir) / filename, caption_y=-0.09)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def audit_summary_is_renderable(audit: Any) -> bool:
    """True when an object exposes the audit-score attributes the reconstruction figure needs.

    Checked rather than assumed, so a partially built audit result leaves the figure missing with a
    clear reason instead of producing a half-empty chart.
    """
    if audit is None:
        return False
    required = ("variant_ids", "question_ids", "variants")
    if not all(hasattr(audit, name) for name in required):
        return False
    return all(
        hasattr(audit.variants.get(v), "per_question") for v in audit.variant_ids
    ) if audit.variant_ids else False


def figure_exposure_response(
    analysis: RunAnalysis, out_dir: Path | str,
    filename: str = "fig_f_exposure_response_recovery.png",
) -> Path:
    """Episode-level measurements with unavailable, censored and independent-unit counts visible."""
    arms = analysis.arm_ids
    measurements = {arm: analysis.overall[arm].measurements for arm in arms}
    metrics = sorted({name for measured in measurements.values() for name in measured.latencies})
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.3))

    def point(ax, x, estimate, colour, *, annotate=True):
        if estimate.point is None:
            if annotate:
                ax.text(x, 0, "unavailable", ha="center", va="bottom", rotation=90, fontsize=7)
            return
        ax.scatter([x], [estimate.point], color=colour, zorder=3)
        if estimate.ci_low is not None and estimate.ci_high is not None:
            ax.errorbar(x, estimate.point,
                        yerr=[[max(0, estimate.point - estimate.ci_low)],
                              [max(0, estimate.ci_high - estimate.point)]],
                        fmt="none", color=colour, capsize=3)
        if annotate:
            ax.annotate(f"n={estimate.n_units}" + ("; no CI" if estimate.ci_low is None else ""),
                        (x, estimate.point), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=6)

    counts = []
    for i, arm in enumerate(arms):
        measured = measurements[arm]
        colour = _arm_colour(arm, i)
        point(axes[0], i, measured.mean_outside_duration_s, colour)
        point(axes[2], i, measured.mean_recovery_episode_success_fraction, colour)
        offset = (i - (len(arms) - 1) / 2) * 0.22
        for j, metric in enumerate(metrics):
            timing = measured.latencies.get(metric)
            if timing is not None:
                estimate = timing.mean_of_observed_episode_means_s
                point(axes[1], j + offset, estimate, colour, annotate=False)
                interval = "no CI" if estimate.ci_low is None else "CI shown"
                counts.append(f"{arm} / {metric}: {timing.status_counts}; "
                              f"n={estimate.n_units}, {interval}")
        axes[1].scatter([], [], color=colour, label=arm.replace("_", " "))
        counts.append(
            f"{arm}: geofence unobserved={measured.unobserved_duration_s:.3f}s; "
            f"missing measurement episodes={measured.n_episodes_without_measurements}; "
            f"recovery windows={measured.recovery_status_counts}; "
            f"recovery unavailable={measured.recovery_unavailable_reasons}"
        )
    for ax in (axes[0], axes[2]):
        ax.set_xticks(range(len(arms)), [a.replace("_", "\n", 1) for a in arms], fontsize=8)
    axes[0].set(title="Sampled geofence exposure", ylabel="mean seconds per eligible episode")
    axes[1].set(title="Observed response latency", ylabel="mean of observed episode means (seconds)")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=6, loc="upper right")
    axes[1].set_xticks(range(len(metrics)), [m.replace("_", "\n") for m in metrics], fontsize=7)
    if not metrics:
        axes[1].text(0.5, 0.5, "No response windows available", transform=axes[1].transAxes,
                     ha="center", fontsize=9)
    axes[2].set(title="Observed fault recovery", ylabel="mean resolved success fraction per episode")
    axes[2].set_ylim(-0.05, 1.15)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Exposure, response and recovery ({_stamp(analysis)})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    caption = (
        "Points summarize episodes, with scenario-level uncertainty; n is the independent-unit count. "
        "Fewer than two scenarios gives no interval. Exposure uses sampled geofence occupancy, not "
        "continuous safety or collision-contact duration. Latency means use observed windows only; "
        "censored and unavailable windows stay counted below. Recovery excludes unresolved windows "
        "from the resolved-success denominator; these conditional rates are not unconditional safety.\n"
        + "\n".join(counts)
    )
    return _finish(fig, analysis, caption, Path(out_dir) / filename)


def render_all(
    analysis: RunAnalysis,
    out_dir: Path | str,
    audit: Any | None = None,
    *,
    protocol: ProtocolConfig | None = None,
) -> list[str]:
    """Workflow entry point: write every supported figure and return the paths as strings.

    ``audit`` may be ``None`` or an object that does not (yet) carry audit scores; the audit figure is
    then skipped rather than faked.
    """
    usable = audit if audit_summary_is_renderable(audit) else None
    written = write_all_figures(analysis, out_dir, audit=usable, protocol=protocol)
    return [str(written[key]) for key in sorted(written)]


def write_all_figures(
    analysis: RunAnalysis,
    out_dir: Path | str,
    *,
    audit: AuditScoreSummary | None = None,
    protocol: ProtocolConfig | None = None,
) -> dict[str, Path]:
    """Write every figure this run supports and return ``{figure_id: path}``.

    The audit figure is written only when audit scores are supplied: an absent audit experiment must
    leave a missing figure, not an empty one.
    """
    directory = Path(out_dir)
    written: dict[str, Path] = {
        "a_violation_fractions": figure_violation_fractions(analysis, directory),
        "b_safety_completion_tradeoff": figure_safety_completion_tradeoff(analysis, directory),
        "c_detection_delay": figure_detection_delay(analysis, directory),
        "e_cell_heatmap_physical": figure_cell_heatmap(analysis, directory, protocol=protocol),
    }
    written["e_cell_heatmap_procedural"] = figure_cell_heatmap(
        analysis, directory, filename="fig_e_cell_heatmap_procedural.png",
        outcome_field="procedural_violation", protocol=protocol,
    )
    if audit is not None:
        written["d_audit_reconstruction"] = figure_audit_reconstruction(analysis, audit, directory)
    if any(m.measurements.n_measured_episodes for m in analysis.overall.values()):
        written["f_exposure_response_recovery"] = figure_exposure_response(analysis, directory)
    return written
