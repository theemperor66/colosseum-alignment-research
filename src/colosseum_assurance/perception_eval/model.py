"""Actually fitted regularized logistic visual model, with separate affine-logit calibration."""

from __future__ import annotations

import math
from collections import Counter
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from colosseum_assurance.perception_eval.contracts import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    FrozenRecord,
    PerceptionRow,
    content_hash,
    validate_features,
    validate_rows,
)


class ModelArtifact(FrozenRecord):
    model_version: Literal["civilian_visibility_logistic_v1"] = "civilian_visibility_logistic_v1"
    feature_version: str = FEATURE_VERSION
    feature_names: tuple[str, ...] = FEATURE_NAMES
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    calibration_slope: float = 1.0
    calibration_intercept: float = 0.0
    recalibrated: bool
    regularization: float = Field(gt=0)
    decision_threshold: float = Field(ge=0, le=1)
    bin_edges: tuple[float, ...]
    training_prior: float = Field(gt=0, lt=1)
    train_scenario_ids: tuple[str, ...]
    calibration_scenario_ids: tuple[str, ...]
    test_scenario_ids: tuple[str, ...]
    split_hash: str
    training_data_hash: str
    calibration_data_hash: str
    dataset_scope: Literal["declared_rows_only", "planned_group_inventory"] = "declared_rows_only"
    inventory_hash: str | None = None
    label_spec_hash: str
    training_provenance: tuple[str, ...]
    evidence_class: str
    training_rows: int = Field(ge=1)
    calibration_rows: int = Field(ge=0)
    training_iterations: int = Field(ge=1)
    calibration_iterations: int = Field(ge=0)

    @model_validator(mode="after")
    def coherent(self) -> ModelArtifact:
        n = len(FEATURE_NAMES)
        if self.feature_names != FEATURE_NAMES or self.feature_version != FEATURE_VERSION:
            raise ValueError("unsupported observation feature definition")
        if any(len(x) != n for x in (self.feature_mean, self.feature_scale, self.coefficients)):
            raise ValueError("model dimensions do not match the frozen feature schema")
        if any(x <= 0 for x in self.feature_scale):
            raise ValueError("feature scales must be positive")
        if (len(self.bin_edges) < 3 or self.bin_edges[0] != 0 or self.bin_edges[-1] != 1
                or any(a >= b for a, b in zip(self.bin_edges, self.bin_edges[1:], strict=False))):
            raise ValueError("reliability bins must strictly increase from zero to one")
        groups = [self.train_scenario_ids, self.calibration_scenario_ids, self.test_scenario_ids]
        if any(not g or len(set(g)) != len(g) for g in groups):
            raise ValueError("each frozen split requires unique scenario IDs")
        if len(set(sum((list(x) for x in groups), []))) != sum(map(len, groups)):
            raise ValueError("scenario leakage across frozen model splits")
        expected = content_hash(dict(zip(("train", "calibration", "test"), groups, strict=True)))
        if self.split_hash != expected:
            raise ValueError("frozen split hash mismatch")
        if (self.dataset_scope == "planned_group_inventory") != (self.inventory_hash is not None):
            raise ValueError("planned inventory scope requires its frozen content hash")
        if self.inventory_hash is not None and (
            not self.inventory_hash.startswith("sha256:") or len(self.inventory_hash) != 71
            or any(c not in "0123456789abcdef" for c in self.inventory_hash[7:])
        ):
            raise ValueError("inventory hash must be canonical SHA-256")
        fixture = "fixture_fake" in self.training_provenance
        allowed = {"fixture_fake", "third_party_colosseum_build", "colosseum_build_verified"}
        if not self.training_provenance or not set(self.training_provenance) <= allowed:
            raise ValueError("model requires recognized training provenance")
        if fixture and set(self.training_provenance) != {"fixture_fake"}:
            raise ValueError("model cannot mix fixture and live training data")
        if self.evidence_class != ("fixture_only" if fixture else "live_data_model_unvalidated"):
            raise ValueError("model evidence class conflicts with training provenance")
        return self

    @property
    def model_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -z))


def _fit_logistic(x: np.ndarray, y: np.ndarray, weights: np.ndarray, l2: float) -> tuple[np.ndarray, int]:
    """Convex weighted likelihood, damped Newton updates; intercept is not L2 penalized."""
    design = np.column_stack((np.ones(len(x)), x))
    w = np.zeros(design.shape[1])
    prior = float(weights @ y)
    w[0] = math.log(prior / (1 - prior))
    penalty = np.ones(len(w)) * l2
    penalty[0] = 0.0

    def objective(v: np.ndarray) -> float:
        z = design @ v
        return float(weights @ (np.logaddexp(0, z) - y * z) + 0.5 * np.sum(penalty * v * v))

    for iteration in range(1, 201):
        probabilities = _sigmoid(design @ w)
        gradient = design.T @ (weights * (probabilities - y)) + penalty * w
        if float(np.max(np.abs(gradient))) < 1e-9:
            return w, iteration
        curvature = weights * probabilities * (1 - probabilities)
        hessian = design.T @ (curvature[:, None] * design) + np.diag(penalty + 1e-12)
        step = np.linalg.solve(hessian, gradient)
        initial = objective(w)
        fraction = 1.0
        while fraction >= 2 ** -30:
            candidate = w - fraction * step
            if objective(candidate) <= initial - 1e-4 * fraction * float(gradient @ step):
                w = candidate
                break
            fraction /= 2
        else:
            if float(np.max(np.abs(gradient))) < 1e-7:
                return w, iteration
            raise ValueError("logistic fitting did not converge (line search)")
    raise ValueError("logistic fitting did not converge within 200 iterations")


def _usable(rows: list[PerceptionRow], split: str) -> list[PerceptionRow]:
    return [r for r in rows if r.split == split and r.features is not None and r.label.value is not None]


def _arrays(rows: list[PerceptionRow]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not rows or {r.label.value for r in rows} != {0, 1}:
        raise ValueError("each fitted split requires independently labelled positives AND negatives")
    x = np.asarray([[r.features[name] for name in FEATURE_NAMES] for r in rows], dtype=float)
    y = np.asarray([r.label.value for r in rows], dtype=float)
    counts = Counter(r.scenario_id for r in rows)
    weights = np.asarray([1 / counts[r.scenario_id] for r in rows], dtype=float)
    return x, y, weights / weights.sum()


def fit_model(
    rows: list[PerceptionRow], *, regularization: float = 0.01, decision_threshold: float = 0.5,
    bin_edges: tuple[float, ...] = tuple(i / 10 for i in range(11)), recalibrate: bool = True,
) -> ModelArtifact:
    """Train scaler/coefficients only on train, calibrate only on calibration, never inspect test values."""
    validate_rows(rows)
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive")
    groups = {split: tuple(sorted({r.scenario_id for r in rows if r.split == split}))
              for split in ("train", "calibration", "test")}
    if any(not group for group in groups.values()):
        raise ValueError("train, calibration and test scenario memberships must be frozen before fitting")
    train = sorted(_usable(rows, "train"), key=lambda r: r.frame_id)
    calibration = sorted(_usable(rows, "calibration"), key=lambda r: r.frame_id)
    x, y, weights = _arrays(train)
    mean = weights @ x
    scale = np.sqrt(weights @ ((x - mean) ** 2))
    scale[scale < 1e-12] = 1.0
    coefficients, iterations = _fit_logistic((x - mean) / scale, y, weights, regularization)
    cal_slope, cal_intercept, cal_iterations = 1.0, 0.0, 0
    if recalibrate:
        cx, cy, cw = _arrays(calibration)
        logits = coefficients[0] + ((cx - mean) / scale) @ coefficients[1:]
        cal, cal_iterations = _fit_logistic(logits[:, None], cy, cw, regularization)
        cal_intercept, cal_slope = float(cal[0]), float(cal[1])
    fitting_rows = train + (calibration if recalibrate else [])
    provenances = tuple(sorted({r.provenance for r in fitting_rows}))
    return ModelArtifact(
        feature_mean=tuple(mean), feature_scale=tuple(scale), coefficients=tuple(coefficients[1:]),
        intercept=float(coefficients[0]), calibration_slope=cal_slope, calibration_intercept=cal_intercept,
        recalibrated=recalibrate, regularization=regularization, decision_threshold=decision_threshold,
        bin_edges=bin_edges, training_prior=float(weights @ y),
        train_scenario_ids=groups["train"], calibration_scenario_ids=groups["calibration"],
        test_scenario_ids=groups["test"], split_hash=content_hash(groups),
        training_data_hash=content_hash([r.model_dump(mode="json") for r in train]),
        calibration_data_hash=content_hash([r.model_dump(mode="json") for r in calibration]),
        label_spec_hash=rows[0].label.label_spec_hash, training_provenance=provenances,
        evidence_class="fixture_only" if "fixture_fake" in provenances else "live_data_model_unvalidated",
        training_rows=len(train), calibration_rows=len(calibration) if recalibrate else 0,
        training_iterations=iterations, calibration_iterations=cal_iterations,
    )


def predict_probability(model: ModelArtifact | None, features: dict[str, float] | None) -> float | None:
    """Estimate asset visibility, never flight safety. No model or no image means abstention."""
    if model is None or features is None:
        return None
    validate_features(features)
    values = np.asarray([features[key] for key in FEATURE_NAMES])
    z = model.intercept + ((values - model.feature_mean) / model.feature_scale) @ model.coefficients
    return float(_sigmoid(np.asarray(model.calibration_intercept + model.calibration_slope * z)))
