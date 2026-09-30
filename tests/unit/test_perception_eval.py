"""Statistical and data-boundary tests using explicitly synthetic pixels, never live evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from colosseum_assurance.perception_eval import (
    FEATURE_NAMES,
    LabelEvidence,
    LabelSpec,
    PerceptionRow,
    assign_scenario_splits,
    evaluate_dataset,
    evaluate_model,
    extract_features,
    fit_dataset,
    fit_model,
    label_from_segmentation,
    load_model,
    load_rows,
    predict_probability,
    save_model,
    write_rows,
)
from colosseum_assurance.perception_eval.contracts import validate_rows
from colosseum_assurance.perception_eval.metrics import _summary

COLOR = (92, 31, 106)
SPEC = LabelSpec(min_visible_pixels=8, min_visible_fraction=0.1, max_sync_error_ns=100)


def label(present: bool, **overrides) -> LabelEvidence:
    mask = np.zeros((10, 10, 3), dtype=np.uint8)
    if present:
        mask[2:8, 2:8] = COLOR
    options = dict(asset_color_rgb=COLOR, identity_verified=True, image_timestamp_ns=1000,
                   segmentation_timestamp_ns=1001, image_camera_name="front",
                   segmentation_camera_name="front",
                   image_shape=(10, 10), spec=SPEC)
    options.update(overrides)
    return label_from_segmentation(mask, **options)


def fixture_rows() -> list[PerceptionRow]:
    rows = []
    for split in ("train", "calibration", "test"):
        for group in range(4):
            for frame in range(8):
                # A simple learnable photographic feature; labels still come from separate masks.
                positive = frame % 2 == 1
                brightness = (170 if positive else 55) + group * 3 + frame
                rgb = np.full((10, 10, 3), brightness, dtype=np.uint8)
                rows.append(PerceptionRow(
                    frame_id=f"{split}/{group}/{frame}", scenario_id=f"{split}/{group}",
                    arm_id="unshielded" if frame < 4 else "shielded", split=split,
                    stratum="all", provenance="fixture_fake", features=extract_features(rgb),
                    label=label(positive),
                ))
    return rows


def test_masks_are_independent_and_negatives_are_known_only_with_evidence():
    assert label(True).value == 1
    negative = label(False)
    assert negative.value == 0 and negative.visible_pixels == 0
    assert negative.segmentation_sha256
    assert label(False, identity_verified=False).value is None
    assert label(False, asset_color_rgb=None).value is None
    assert label(False, image_timestamp_ns=None).reason == "capture_timestamp_missing"
    assert label(False, segmentation_timestamp_ns=1101).reason == "capture_not_synchronized"
    assert label(False, segmentation_camera_name="rear").reason == "camera_mismatch"
    assert label(False, image_shape=(9, 10)).reason == "image_shape_missing_or_mismatched"
    assert label(False, asset_color_rgb=(True, 31, 106)).value is None
    assert label(False, segmentation_timestamp_ns=1100).value == 0  # Inclusive sync bound.


def test_mask_threshold_requires_pixel_count_and_fraction():
    mask = np.zeros((10, 10, 3), dtype=np.uint8)
    mask[0, :9] = COLOR
    kwargs = dict(asset_color_rgb=COLOR, identity_verified=True, image_timestamp_ns=1,
                  segmentation_timestamp_ns=1, image_camera_name="front", segmentation_camera_name="front",
                  image_shape=(10, 10), spec=SPEC)
    assert label_from_segmentation(mask, **kwargs).value == 0  # 9 pixels pass count but fail fraction.
    mask[0, 9] = COLOR
    assert label_from_segmentation(mask, **kwargs).value == 1
    assert label_from_segmentation(None, **kwargs).value is None
    assert label_from_segmentation(mask.astype(float), **kwargs).reason == "segmentation_invalid"


def test_black_images_and_detector_negatives_are_observations_and_missing_is_abstention():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    features = extract_features(rgb)
    assert features is not None and set(features) == set(FEATURE_NAMES)
    assert features["rgb_mean"] == 0 and features["depth_available"] == 0
    invalid_depth = np.full((10, 10), np.nan)
    with_depth = extract_features(rgb, invalid_depth)
    assert with_depth["depth_available"] == 1 and with_depth["depth_valid_fraction"] == 0
    assert with_depth["structure_support"] == 0
    assert extract_features(None) is None
    with pytest.raises(ValueError, match="RGB/depth"):
        extract_features(rgb, np.ones((2, 2)))
    with pytest.raises(ValueError, match="RGB must"):
        extract_features(rgb.astype(float))
    assert predict_probability(None, features) is None


def test_group_assignment_is_deterministic_and_never_splits_arms():
    ids = [f"scene-{i}" for i in range(12)]
    assert assign_scenario_splits(ids) == assign_scenario_splits(list(reversed(ids)) + ids)
    assert set(assign_scenario_splits(ids).values()) == {"train", "calibration", "test"}
    rows = fixture_rows()
    rows[-1] = rows[-1].model_copy(update={"split": "train"})
    with pytest.raises(ValueError, match="crosses split"):
        fit_model(rows)
    with pytest.raises(ValueError, match="at least three"):
        assign_scenario_splits(["one", "two"])


def test_fits_observations_beats_prior_and_freezes_every_parameter_against_test_changes():
    rows = fixture_rows()
    model = fit_model(rows)
    negative = next(r for r in rows if r.label.value == 0)
    positive = next(r for r in rows if r.label.value == 1)
    assert predict_probability(model, negative.features) < 0.1
    assert predict_probability(model, positive.features) > 0.9
    assert predict_probability(model, None) is None
    modified = [r.model_copy(update={
        "features": {name: 999.0 for name in FEATURE_NAMES}, "label": label(r.label.value == 0),
    }) if r.split == "test" else r for r in rows]
    assert fit_model(modified).model_hash == model.model_hash
    report = evaluate_model(model, rows, bootstrap_resamples=20)
    assert report["evidence_class"] == "fixture_only"
    assert report["overall"]["scores"]["brier"] < report["training_prior_baseline"]["scores"]["brier"]
    assert report["overall"]["scores"]["auc"] == 1
    assert report["overall"]["counts"]["false_negative"] == 0
    assert report["overall"]["intervals"]["brier"]["valid_resamples"] == 20
    assert model.training_iterations > 1 and model.calibration_iterations > 1
    assert any(abs(v) > 0 for v in model.coefficients)


def test_calibration_changes_only_calibration_parameters_and_scaler_is_train_only():
    rows = fixture_rows()
    original = fit_model(rows)
    modified = [r.model_copy(update={"label": label(r.label.value == 0)})
                if r.split == "calibration" else r for r in rows]
    changed = fit_model(modified)
    assert changed.feature_mean == original.feature_mean
    assert changed.feature_scale == original.feature_scale
    assert changed.coefficients == original.coefficients
    assert changed.intercept == original.intercept
    assert changed.calibration_slope < 0 < original.calibration_slope
    assert changed.model_hash != original.model_hash
    assert changed.training_data_hash == original.training_data_hash


def test_scenario_balanced_fitting_is_not_distorted_by_duplicated_frames_in_one_scene():
    rows = fixture_rows()
    duplicates = [r.model_copy(update={"frame_id": r.frame_id + "/duplicate"}) for r in rows
                  if r.scenario_id == "train/0"]
    first, second = fit_model(rows), fit_model(rows + duplicates)
    assert second.feature_mean == pytest.approx(first.feature_mean)
    assert second.coefficients == pytest.approx(first.coefficients)


def test_missing_frames_unknown_masks_and_detector_negatives_keep_denominators():
    rows = fixture_rows()
    model = fit_model(rows)
    rows[-1] = rows[-1].model_copy(update={"features": None, "missing_reason": "capture_failed"})
    rows[-2] = rows[-2].model_copy(update={"label": label(False, identity_verified=False)})
    report = evaluate_model(model, rows, bootstrap_resamples=20)
    counts = report["overall"]["counts"]
    assert counts["eligible_frames"] == 32
    assert counts["predicted_frames"] == 31 and counts["labelled_frames"] == 31
    assert counts["scored_frames"] == 30
    assert report["overall"]["rates"]["joint_coverage"]["denominator"] == 32
    assert report["unknown_label_reasons"] == {"asset_identity_unverified": 1}
    assert report["missing_prediction_reasons"] == {"capture_failed": 1}


def test_known_analytic_confusion_scores_and_reliability_bins():
    summary = _summary(np.array([1, 0, 1, 0, np.nan]), np.array([0.8, 0.6, 0.4, 0.2, np.nan]),
                       0.5, (0.0, 0.5, 1.0))
    assert [summary["counts"][key] for key in ("true_positive", "false_positive", "false_negative",
                                             "true_negative")] == [1, 1, 1, 1]
    assert summary["rates"]["sensitivity"]["value"] == 0.5
    assert summary["rates"]["false_positive_rate"]["value"] == 0.5
    assert summary["scores"]["brier"] == pytest.approx(0.2)
    assert summary["scores"]["auc"] == pytest.approx(0.75)
    assert summary["scores"]["ece"] == pytest.approx(0.2)
    assert summary["reliability_bins"][0]["positive_frequency"] == 0.5
    tied = _summary(np.array([1, 0, 1, 0]), np.array([0.5, 0.5, 0.5, 0.5]), 0.5, (0.0, 0.5, 1.0))
    assert tied["scores"]["auc"] == 0.5
    assert tied["reliability_bins"][0]["count"] == 0
    assert tied["reliability_bins"][1]["count"] == 4  # Left inclusive at edge.


def test_single_class_and_no_predictions_leave_undefined_rates_and_scores():
    result = _summary(np.array([0, 0]), np.array([0.1, 0.2]), 0.5, (0.0, 0.5, 1.0))
    assert result["rates"]["precision"]["value"] is None
    assert result["rates"]["sensitivity"]["denominator"] == 0
    assert result["scores"]["auc"] is None
    assert result["score_undefined_reasons"]["auc"] == "both_classes_required"
    empty = _summary(np.array([0, 1]), np.array([np.nan, np.nan]), 0.5, (0.0, 0.5, 1.0))
    assert empty["scores"]["brier"] is None
    assert empty["rates"]["prediction_coverage"]["value"] == 0


def test_bootstrap_uses_independent_scenarios_not_frames_and_is_reproducible():
    rows = fixture_rows()
    for index, row in enumerate(rows):
        if row.split == "test":
            rows[index] = row.model_copy(update={"stratum": row.scenario_id})
    model = fit_model(rows)
    result = evaluate_model(model, rows, bootstrap_resamples=20)
    interval = result["overall"]["intervals"]["brier"]
    assert interval["lower"] is None
    assert interval["undefined_reason"] == "fewer_than_two_scenarios_in_a_stratum"
    original = fixture_rows()
    fitted = fit_model(original)
    assert evaluate_model(fitted, original, bootstrap_resamples=20) == \
        evaluate_model(fitted, original, bootstrap_resamples=20)


def test_repeated_correlated_frames_do_not_artificially_narrow_cluster_intervals():
    rows = fixture_rows()
    model = fit_model(rows)
    copies = [r.model_copy(update={"frame_id": f"{r.frame_id}/repeat-{repeat}"})
              for r in rows if r.split == "test" for repeat in range(5)]
    first = evaluate_model(model, rows, bootstrap_resamples=40)
    second = evaluate_model(model, rows + copies, bootstrap_resamples=40)
    one = first["overall"]["intervals"]["brier"]
    many = second["overall"]["intervals"]["brier"]
    assert one["upper"] > one["lower"]
    assert many["lower"] == pytest.approx(one["lower"], abs=1e-12)
    assert many["upper"] == pytest.approx(one["upper"], abs=1e-12)
    assert second["overall"]["counts"]["eligible_frames"] == 6 * first["overall"]["counts"]["eligible_frames"]


def test_rejects_leakage_incomplete_groups_label_drift_and_fixture_laundering():
    rows = fixture_rows()
    model = fit_model(rows)
    test = [r for r in rows if r.split == "test"]
    with pytest.raises(ValueError, match="every frozen test scenario"):
        evaluate_model(model, test[:8], bootstrap_resamples=20)
    with pytest.raises(ValueError, match="frozen model split"):
        evaluate_model(model, [r.model_copy(update={"split": "test"}) for r in rows if r.split == "train"])
    live = [r.model_copy(update={"provenance": "colosseum_build_verified"}) for r in test]
    with pytest.raises(ValueError, match="fixture-trained"):
        evaluate_model(model, live)
    changed_spec = LabelSpec(min_visible_pixels=9)
    drift = [r.model_copy(update={"label": label(bool(r.label.value), spec=changed_spec)}) for r in test]
    with pytest.raises(ValueError, match="label definition"):
        evaluate_model(model, drift)
    with pytest.raises(ValueError, match="fixture and live"):
        fit_model(rows + [live[0].model_copy(update={"frame_id": "extra"})])


def test_fitting_refuses_one_class_in_fitted_splits():
    rows = fixture_rows()
    changed = [r.model_copy(update={"label": label(False)}) if r.split == "calibration" else r for r in rows]
    with pytest.raises(ValueError, match="positives AND negatives"):
        fit_model(changed)
    assert fit_model(changed, recalibrate=False).calibration_iterations == 0


def test_strict_feature_and_label_boundaries():
    row = fixture_rows()[0]
    with pytest.raises(ValidationError, match="features must be exactly"):
        PerceptionRow(**{**row.model_dump(), "features": {**row.features, "ground_truth": 1.0}})
    with pytest.raises(ValidationError):
        PerceptionRow(**{**row.model_dump(), "features": {**row.features, "rgb_mean": float("nan")}})
    with pytest.raises(ValidationError, match="missing observations"):
        PerceptionRow(**{**row.model_dump(), "features": None})
    with pytest.raises(ValidationError, match="integer 0/1"):
        LabelEvidence(**{**row.label.model_dump(), "value": False})
    with pytest.raises(ValueError, match="duplicate frame"):
        validate_rows([row, row])


def test_artifacts_round_trip_detect_tampering_and_refuse_overwrite(tmp_path: Path):
    rows = fixture_rows()
    dataset = tmp_path / "rows.jsonl"
    model_path = tmp_path / "model.json"
    write_rows(dataset, rows)
    assert load_rows(dataset) == rows
    model = fit_dataset(dataset, model_path)
    assert load_model(model_path).model_hash == model.model_hash
    report = evaluate_dataset(dataset, model_path, tmp_path / "results", bootstrap_resamples=20)
    assert report["evidence_class"] == "fixture_only"
    assert "FIXTURE ONLY" in (tmp_path / "results/evaluation.md").read_text()
    assert (tmp_path / "results/reliability.png").read_bytes().startswith(b"\x89PNG")
    with pytest.raises(FileExistsError):
        write_rows(dataset, rows)
    with pytest.raises(FileExistsError):
        save_model(model_path, model)
    with pytest.raises(FileExistsError):
        evaluate_dataset(dataset, model_path, tmp_path / "results", bootstrap_resamples=20)
    envelope = json.loads(model_path.read_text())
    envelope["artifact"]["intercept"] += 0.1
    model_path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_model(model_path)


def write_inventory(root: Path, rows: list[PerceptionRow], **updates) -> Path:
    inventory = {"groups": {r.scenario_id: r.split for r in rows}, "run_class": "fixture",
                 "protocol_hash": "sha256:" + "f" * 64, **updates}
    path = root / "group-inventory.json"
    path.write_text(json.dumps(inventory))
    return path


@pytest.mark.parametrize("mutation,match", [
    ("missing", "missing planned scenario groups"),
    ("extra", "unplanned scenario groups"),
    ("reassigned", "differs from frozen group inventory"),
])
def test_planned_inventory_detects_wholly_missing_extra_or_reassigned_groups(tmp_path, mutation, match):
    rows = fixture_rows()
    write_inventory(tmp_path, rows)
    if mutation == "missing":
        rows = [r for r in rows if r.scenario_id != "train/0"]
    elif mutation == "extra":
        rows += [rows[0].model_copy(update={"scenario_id": "unplanned", "frame_id": "unplanned/0"})]
    else:
        rows = [r.model_copy(update={"split": "test"}) if r.scenario_id == "train/0" else r for r in rows]
    path = tmp_path / "rows.jsonl"
    write_rows(path, rows)
    with pytest.raises(ValueError, match=match):
        fit_dataset(path, tmp_path / "model.json")
    assert not (tmp_path / "model.json").exists()


def test_inventory_scope_and_hash_bind_fit_and_evaluation(tmp_path):
    rows = fixture_rows()
    path, model_path = tmp_path / "rows.jsonl", tmp_path / "model.json"
    write_rows(path, rows)
    assert fit_model(rows).dataset_scope == "declared_rows_only"
    assert fit_model(rows).inventory_hash is None
    inventory = write_inventory(tmp_path, rows)
    model = fit_dataset(path, model_path)
    assert model.dataset_scope == "planned_group_inventory" and model.inventory_hash.startswith("sha256:")
    report = evaluate_dataset(path, model_path, tmp_path / "evaluation", bootstrap_resamples=20)
    assert report["inventory_hash"] == model.inventory_hash
    inventory.unlink()
    with pytest.raises(ValueError, match="frozen dataset scope/hash"):
        evaluate_dataset(path, model_path, tmp_path / "missing-inventory")
    write_inventory(tmp_path, rows, protocol_hash="changed-protocol")
    with pytest.raises(ValueError, match="frozen dataset scope/hash"):
        evaluate_dataset(path, model_path, tmp_path / "changed-inventory")
    write_inventory(tmp_path, rows)
    path.write_text("\n".join(r.model_dump_json() for r in rows if r.scenario_id != "test/0"))
    with pytest.raises(ValueError, match="missing planned scenario groups"):
        evaluate_dataset(path, model_path, tmp_path / "missing-test-group")


def test_inventory_cannot_promote_a_fixture_run_class(tmp_path):
    rows = fixture_rows()
    path = tmp_path / "rows.jsonl"
    write_rows(path, rows)
    write_inventory(tmp_path, rows, run_class="heldout")
    with pytest.raises(ValueError, match="fixture/live row provenance"):
        fit_dataset(path, tmp_path / "invalid-model.json")
