"""Held-out frame estimates with uncertainty at the independent scenario level."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from colosseum_assurance.perception_eval.contracts import PerceptionRow, content_hash, validate_rows
from colosseum_assurance.perception_eval.model import ModelArtifact, predict_probability


def _rate(numerator: int, denominator: int) -> dict:
    return {"value": numerator / denominator if denominator else None,
            "numerator": numerator, "denominator": denominator,
            "undefined_reason": None if denominator else "zero_denominator"}


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    if not positives or not negatives:
        return None
    order = np.argsort(p, kind="stable")
    rank_sum, i = 0.0, 0
    while i < len(order):
        end = i + 1
        while end < len(order) and p[order[end]] == p[order[i]]:
            end += 1
        rank_sum += float(y[order[i:end]].sum()) * (i + 1 + end) / 2
        i = end
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def _summary(y_all: np.ndarray, p_all: np.ndarray, threshold: float, edges: tuple[float, ...]) -> dict:
    predicted, labelled = np.isfinite(p_all), np.isfinite(y_all)
    scored = predicted & labelled
    y, p = y_all[scored], p_all[scored]
    positive = p >= threshold
    tp = int(np.sum(positive & (y == 1)))
    fp = int(np.sum(positive & (y == 0)))
    fn = int(np.sum(~positive & (y == 1)))
    tn = int(np.sum(~positive & (y == 0)))
    bins = []
    for i, (lo, hi) in enumerate(zip(edges, edges[1:], strict=False)):
        member = (p >= lo) & ((p < hi) if i < len(edges) - 2 else (p <= hi))
        count = int(member.sum())
        mean = float(p[member].mean()) if count else None
        frequency = float(y[member].mean()) if count else None
        bins.append({"lower": lo, "upper": hi, "upper_inclusive": i == len(edges) - 2,
                     "count": count, "mean_probability": mean, "positive_frequency": frequency,
                     "absolute_gap": abs(mean - frequency) if count else None})
    rates = {
        "sensitivity": _rate(tp, tp + fn), "false_positive_rate": _rate(fp, fp + tn),
        "precision": _rate(tp, tp + fp), "specificity": _rate(tn, tn + fp),
        "accuracy": _rate(tp + tn, len(y)),
        "prediction_coverage": _rate(int(predicted.sum()), len(y_all)),
        "label_coverage": _rate(int(labelled.sum()), len(y_all)),
        "joint_coverage": _rate(int(scored.sum()), len(y_all)),
    }
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    scores = {
        "brier": float(np.mean((p - y) ** 2)) if len(y) else None,
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)))
        if len(y) else None,
        "auc": _auc(y, p),
        "ece": sum(b["count"] * (b["absolute_gap"] or 0) for b in bins) / len(y) if len(y) else None,
    }
    reasons = {name: None if value is not None else
               ("both_classes_required" if name == "auc" and len(y) else "no_jointly_observed_labels")
               for name, value in scores.items()}
    return {
        "counts": {"eligible_frames": len(y_all), "predicted_frames": int(predicted.sum()),
                   "labelled_frames": int(labelled.sum()), "scored_frames": len(y),
                   "positive_labels": tp + fn, "negative_labels": fp + tn,
                   "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn},
        "rates": rates, "scores": scores, "score_undefined_reasons": reasons, "reliability_bins": bins,
    }


def _values(summary: dict) -> dict[str, float | None]:
    return {**{key: value["value"] for key, value in summary["rates"].items()}, **summary["scores"]}


def _with_intervals(
    rows: list[PerceptionRow], labels: np.ndarray, probabilities: np.ndarray, model: ModelArtifact,
    *, bootstrap_resamples: int, seed: int, confidence_level: float,
) -> dict:
    summary = _summary(labels, probabilities, model.decision_threshold, model.bin_edges)
    scenarios = sorted({r.scenario_id for r in rows})
    strata = sorted({r.stratum for r in rows})
    by_stratum = {name: sorted({r.scenario_id for r in rows if r.stratum == name}) for name in strata}
    index = {sid: np.asarray([i for i, row in enumerate(rows) if row.scenario_id == sid], dtype=int)
             for sid in scenarios}
    samples: dict[str, list[float]] = {key: [] for key in _values(summary)}
    # A one-cluster stratum cannot supply empirical between-scenario variation. Do not issue
    # deceptively narrow intervals by repeatedly resampling its frames or treating its arms as IID.
    enough_clusters = all(len(ids) >= 2 for ids in by_stratum.values())
    if enough_clusters:
        rng = np.random.default_rng(seed)
        for _ in range(bootstrap_resamples):
            sampled_ids = [sid for ids in by_stratum.values() for sid in rng.choice(ids, size=len(ids))]
            indices = np.concatenate([index[sid] for sid in sampled_ids])
            values = _values(_summary(labels[indices], probabilities[indices], model.decision_threshold,
                                      model.bin_edges))
            for key, value in values.items():
                if value is not None:
                    samples[key].append(value)
    alpha = (1 - confidence_level) / 2
    intervals = {}
    for key, estimates in samples.items():
        enough = len(estimates) >= math.ceil(0.8 * bootstrap_resamples)
        point_defined = _values(summary)[key] is not None
        valid = enough and enough_clusters and point_defined
        bounds = np.quantile(estimates, [alpha, 1 - alpha]) if valid else (None, None)
        intervals[key] = {
            "lower": float(bounds[0]) if valid else None, "upper": float(bounds[1]) if valid else None,
            "confidence_level": confidence_level, "valid_resamples": len(estimates),
            "requested_resamples": bootstrap_resamples,
            "undefined_reason": None if valid else ("point_estimate_undefined" if not point_defined else
                                                     "fewer_than_two_scenarios_in_a_stratum"
                                                     if not enough_clusters else "too_few_defined_resamples"),
        }
    summary["scenario_count"] = len(scenarios)
    summary["intervals"] = intervals
    return summary


def evaluate_model(
    model: ModelArtifact, rows: list[PerceptionRow], *, bootstrap_resamples: int = 500,
    seed: int = 7717, confidence_level: float = 0.95,
) -> dict:
    """Evaluate only frozen test groups; all eligible frames contribute coverage denominators."""
    validate_rows(rows)
    if type(bootstrap_resamples) is not int or bootstrap_resamples < 20:
        raise ValueError("bootstrap_resamples must be an integer of at least 20")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie between zero and one")
    memberships = {"train": model.train_scenario_ids, "calibration": model.calibration_scenario_ids,
                   "test": model.test_scenario_ids}
    for row in rows:
        if row.scenario_id not in memberships[row.split]:
            raise ValueError(f"scenario {row.scenario_id} is outside its frozen model split")
        if row.label.label_spec_hash != model.label_spec_hash:
            raise ValueError("evaluation visibility-label definition differs from fitted model")
        if (row.provenance == "fixture_fake") != (model.evidence_class == "fixture_only"):
            raise ValueError("fixture-trained models and live evidence must not be mixed")
    test = sorted((r for r in rows if r.split == "test"), key=lambda r: r.frame_id)
    if not test:
        raise ValueError("no test frames provided")
    if {r.scenario_id for r in test} != set(model.test_scenario_ids):
        raise ValueError("evaluation must account for every frozen test scenario, including missing captures")
    labels = np.asarray([r.label.value if r.label.value is not None else np.nan for r in test], dtype=float)
    probabilities = np.asarray([predict_probability(model, r.features) if r.features is not None else np.nan
                                for r in test], dtype=float)
    kwargs = dict(bootstrap_resamples=bootstrap_resamples, seed=seed, confidence_level=confidence_level)
    overall = _with_intervals(test, labels, probabilities, model, **kwargs)
    baseline_p = np.where(np.isfinite(probabilities), model.training_prior, np.nan)
    baseline = _with_intervals(test, labels, baseline_p, model, **kwargs)
    strata = {}
    for stratum in sorted({r.stratum for r in test}):
        indices = np.asarray([i for i, row in enumerate(test) if row.stratum == stratum])
        strata[stratum] = _with_intervals([test[i] for i in indices], labels[indices], probabilities[indices],
                                          model, **kwargs)
    return {
        "report_version": "civilian_visibility_evaluation_v1", "model_hash": model.model_hash,
        "split_hash": model.split_hash, "label_spec_hash": model.label_spec_hash,
        "dataset_scope": model.dataset_scope, "inventory_hash": model.inventory_hash,
        "evaluation_data_hash": content_hash([r.model_dump(mode="json") for r in test]),
        "evidence_class": ("fixture_only" if model.evidence_class == "fixture_only"
                           else "live_held_out_evaluation"),
        "evaluation_provenance_counts": dict(sorted(Counter(r.provenance for r in test).items())),
        "claim": "designated civilian asset visibility above the frozen pixel threshold; not flight safety",
        "decision_threshold": model.decision_threshold, "bin_edges": list(model.bin_edges),
        "test_scenario_ids": sorted(model.test_scenario_ids),
        "unknown_label_reasons": dict(sorted(Counter(r.label.reason for r in test
                                                      if r.label.value is None).items())),
        "missing_prediction_reasons": dict(sorted(Counter(r.missing_reason for r in test
                                                          if r.features is None).items())),
        "overall": overall, "strata": strata, "training_prior_baseline": baseline,
        "training_prior": model.training_prior,
        "uncertainty": {"method": "stratified scenario-cluster percentile bootstrap",
                        "seed": seed, "resamples": bootstrap_resamples, "confidence_level": confidence_level,
                        "minimum_defined_fraction": 0.8,
                        "sampling_unit": "whole scenario, retaining all arms and frames",
                        "estimand": "frame-weighted metrics conditional on observed labels and predictions"},
        "limitations": ["Fixture results do not validate a rendered simulator or real-world perception.",
                        "Missing labels/images are excluded from conditional scores, not coverage counts.",
                        "ECE depends on the frozen bins and sample size; Brier also reflects discrimination.",
                        "Intervals omit training-set uncertainty and unobserved distribution shift.",
                        "Export all scheduled frames; hashes cannot establish unrecorded omissions."],
    }
