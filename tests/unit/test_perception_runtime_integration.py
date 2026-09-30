"""Vision capture, split inventory, fitted model and delayed guard boundaries; fixture evidence only."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from tests.unit.test_monitor_policy_only import make_brief
from tests.unit.test_perception_eval import fixture_rows, label

from colosseum_assurance.config import AppConfig, PathsConfig
from colosseum_assurance.evaluation import boundary_cases as bc
from colosseum_assurance.interfaces import AdapterError, CapturedFrame
from colosseum_assurance.monitors.perception_guard import PerceptionConfidenceGuard
from colosseum_assurance.monitors.policy_only import PolicyOnlyMonitor
from colosseum_assurance.perception_eval import (
    LabelSpec,
    fit_dataset,
    fit_model,
    load_rows,
    save_model,
    write_rows,
)
from colosseum_assurance.protocol.expanded import ExpandedStudySpec
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.runtime.episode import EpisodeRunner
from colosseum_assurance.runtime.evidence import EvidenceWriter, load_attempted_runs
from colosseum_assurance.runtime.observation import ObservationPipeline
from colosseum_assurance.runtime.perception_capture import record_perception_frame
from colosseum_assurance.scenario.expanded import build_expanded_manifest
from colosseum_assurance.schemas import AuthorizationView, ControlCommand, FrameRef, SupervisionView, Verdict
from colosseum_assurance.sim import build_adapter, fixture_fake_server
from colosseum_assurance.workflows.perception import prepare_perception_protocol


def _frame(kind, pixels, *, at=9.5, camera="front"):
    return CapturedFrame(ref=FrameRef(
        kind=kind, camera_name=camera, sim_time_s=at, acquisition_time_known=at is not None,
        width=pixels.shape[1], height=pixels.shape[0], max_value=float(pixels.max()),
        nonzero_fraction=float(np.count_nonzero(pixels) / pixels.size),
    ), array=pixels)


def _frames(rgb=None, depth=None):
    return {
        "rgb": _frame("rgb", np.full((10, 10, 3), 200, dtype=np.uint8) if rgb is None else rgb),
        "depth": _frame("depth", np.ones((10, 10), dtype=np.float32)) if depth is None else depth,
        "segmentation": _frame("segmentation", np.tile(
            np.asarray([92, 31, 106], dtype=np.uint8), (10, 10, 1))),
    }


def _record(tmp_path, frames, *, spec=None):
    out = tmp_path / "rows.jsonl"
    prediction = record_perception_frame(
        frames=frames, spec=spec or ExpandedStudySpec(
            family="degraded_perception", capture_segmentation=True),
        identity={"identity_verified": True}, scenario_group="test-runtime", arm_id="A1_policy_only",
        frame_id="runtime:0", provenance="fixture_fake", stratum="all", out=out,
        dropped=False, hfov_rad=math.pi / 2,
    )
    return prediction, load_rows(out)[-1]


def test_timestamped_black_rgb_is_an_observation_and_keeps_independent_label(tmp_path):
    _, row = _record(tmp_path, _frames(rgb=np.zeros((10, 10, 3), dtype=np.uint8)))
    assert row.features is not None and row.features["rgb_mean"] == 0
    assert row.label.value == 1 and row.missing_reason is None


@pytest.mark.parametrize(("at", "camera"), [(12.0, "front"), (None, "front"), (9.5, "rear")])
def test_unstamped_future_or_different_camera_depth_is_not_backdated_into_rgb(tmp_path, at, camera):
    depth = _frame("depth", np.ones((10, 10), dtype=np.float32), at=at, camera=camera)
    prediction, row = _record(tmp_path, _frames(depth=depth))
    assert row.features is not None and row.features["depth_available"] == 0
    assert row.label.value == 1
    assert prediction["sim_time_s"] == 9.5


def test_real_camera_rpc_failure_preserves_attempted_frame_and_planned_groups(tmp_path):
    protocol = prepare_perception_protocol(run_class="fixture", realizations=6, horizon_s=40)
    paths = PathsConfig(results_root=tmp_path)
    writer = EvidenceWriter(paths, "fixture", protocol.content_hash(), protocol)
    with fixture_fake_server() as endpoint:
        config = AppConfig(endpoint=endpoint, paths=paths)
        adapter = build_adapter(config, protocol)

        def failure(*args, **kwargs):
            raise AdapterError("synthetic capture RPC failure")

        adapter.capture = failure
        try:
            result = EpisodeRunner(adapter, protocol, config, writer, save_frames=False).run(
                build_expanded_manifest(protocol, "fixture", 0), "A0_unguarded")
        finally:
            adapter.close()
    assert result.attempt.status in {"crashed", "partial"}
    attempts = load_attempted_runs(writer.root / "attempted_runs.jsonl")
    assert len(attempts) == 1 and "capture" in attempts[0].error_message
    rows = load_rows(writer.root / "perception" / "rows.jsonl")
    assert len(rows) == 1 and rows[0].features is None and rows[0].label.value is None
    assert rows[0].missing_reason
    inventory = json.loads((writer.root / "perception" / "group-inventory.json").read_text())
    assert inventory["groups"] == protocol.study_extension.perception_split_by_group
    assert inventory["protocol_hash"] == protocol.content_hash()


def test_fit_requires_every_planned_group_even_if_first_capture_never_arrived(tmp_path):
    rows = fixture_rows()
    groups = {r.scenario_id: r.split for r in rows}
    dataset = tmp_path / "rows.jsonl"
    write_rows(dataset, [r for r in rows if r.scenario_id != "test/0"])
    (tmp_path / "group-inventory.json").write_text(json.dumps({
        "groups": groups, "protocol_hash": "sha256:" + "0" * 64, "run_class": "fixture",
    }))
    with pytest.raises(ValueError):
        fit_dataset(dataset, tmp_path / "model.json")
    assert not (tmp_path / "model.json").exists()


def test_fitted_model_canonical_hash_reaches_guard_only_after_observation_delay(tmp_path):
    # Train and calibrate using independent default-policy masks, never fake constant probabilities.
    rows = [r.model_copy(update={"label": label(bool(r.label.value), spec=LabelSpec())})
            for r in fixture_rows()]
    model = fit_model(rows)
    path = tmp_path / "model.json"
    save_model(path, model)
    spec = ExpandedStudySpec(family="degraded_perception", capture_segmentation=True,
                             perception_model_path=str(path), perception_model_hash=model.model_hash)
    prediction, row = _record(tmp_path, _frames(), spec=spec)
    assert prediction["model_hash"] == model.model_hash and model.model_hash.startswith("sha256:")
    assert prediction["prediction"] is not None and row.label.value == 1
    schedules = bc.boundary_manifest().schedules.model_copy(update={"observation_delay_s": [.5]})
    pipeline = ObservationPipeline(schedules, declared_delay_bound_s=.5)
    pipeline.push_prediction(prediction)
    supervision = SupervisionView(sim_time_s=9.5, last_heartbeat_sim_time_s=9.5, heartbeat_age_s=0)
    before = pipeline.build(0, 9.5, supervision, AuthorizationView())
    after = pipeline.build(0, 10, supervision, AuthorizationView())
    assert before.asset_presence_probability is None
    assert after.asset_presence_probability == prediction["prediction"]
    guard = PerceptionConfidenceGuard(PolicyOnlyMonitor(ProtocolConfig()), threshold=0.0, max_age_s=1)
    guard.reset(make_brief())
    command = ControlCommand(step_index=0, issued_sim_time_s=10, kind="inspect_capture")
    result = guard.evaluate(after, command)
    assert result.assumption_verdicts["perception_capture_confidence"] is Verdict.PASS


def test_runtime_refuses_model_fitted_for_different_visibility_label_policy(tmp_path):
    model = fit_model(fixture_rows())  # Different fraction/synchronization policy in these test rows.
    path = tmp_path / "model.json"
    save_model(path, model)
    spec = ExpandedStudySpec(family="degraded_perception", capture_segmentation=True,
                             perception_model_path=str(path), perception_model_hash=model.model_hash)
    with pytest.raises(ValueError):
        _record(tmp_path, _frames(), spec=spec)


def test_independent_mask_changes_label_but_never_the_runtime_probability(tmp_path):
    rows = [r.model_copy(update={"label": label(bool(r.label.value), spec=LabelSpec())})
            for r in fixture_rows()]
    model = fit_model(rows)
    path = tmp_path / "model.json"
    save_model(path, model)
    spec = ExpandedStudySpec(family="degraded_perception", capture_segmentation=True,
                             perception_model_path=str(path), perception_model_hash=model.model_hash)
    visible = _frames()
    absent = _frames()
    absent["segmentation"] = _frame("segmentation", np.zeros((10, 10, 3), dtype=np.uint8))
    one, positive = _record(tmp_path / "visible", visible, spec=spec)
    two, negative = _record(tmp_path / "absent", absent, spec=spec)
    assert positive.label.value == 1 and negative.label.value == 0
    assert positive.features == negative.features
    assert one["prediction"] == two["prediction"] and one["prediction"] is not None
