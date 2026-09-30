"""Bounded civilian visual probabilities, independent mask labels, and held-out evaluation."""

from colosseum_assurance.perception_eval.contracts import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    LabelEvidence,
    LabelSpec,
    PerceptionRow,
    assign_scenario_splits,
)
from colosseum_assurance.perception_eval.features import extract_features, label_from_segmentation
from colosseum_assurance.perception_eval.io import (
    evaluate_dataset,
    fit_dataset,
    load_model,
    load_rows,
    save_model,
    write_rows,
)
from colosseum_assurance.perception_eval.metrics import evaluate_model
from colosseum_assurance.perception_eval.model import ModelArtifact, fit_model, predict_probability

__all__ = [
    "FEATURE_NAMES", "FEATURE_VERSION", "LabelEvidence", "LabelSpec", "ModelArtifact", "PerceptionRow",
    "assign_scenario_splits", "evaluate_dataset", "evaluate_model", "extract_features", "fit_dataset",
    "fit_model", "label_from_segmentation", "load_model", "load_rows", "predict_probability", "save_model",
    "write_rows",
]
