"""Camera-only fault contracts and production-runner fixtures; no live evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from tests.unit.test_perception_runtime_integration import _frames

from colosseum_assurance.config import AppConfig, PathsConfig
from colosseum_assurance.protocol.camera_obscuration import CameraObscurationSpec
from colosseum_assurance.protocol.controlled import ControlledStudySpec
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.runtime.arms import Arm
from colosseum_assurance.runtime.episode import EpisodeRunner
from colosseum_assurance.runtime.evidence import EvidenceExists, EvidenceWriter
from colosseum_assurance.runtime.perturbations import CameraObscurationChannel, PerturbationChannel
from colosseum_assurance.scenario.expanded import build_expanded_manifest, expanded_protocol
from colosseum_assurance.schemas import ControlCommand
from colosseum_assurance.sim import build_adapter, fixture_fake_server


def spec(intervals=None):
    return CameraObscurationSpec(
        intervals=intervals
        if intervals is not None
        else [{"start_s": 0.5, "end_s": 1.0}, {"start_s": 2.0, "end_s": 2.5}]
    )


def at(time):
    frames = _frames()
    for frame in frames.values():
        frame.ref.sim_time_s = time
    return frames


@pytest.mark.parametrize(
    "elapsed,active",
    [
        (-0.1, False),
        (0.0, False),
        (0.499, False),
        (0.5, True),
        (0.999, True),
        (1.0, False),
        (2.0, True),
        (2.5, False),
    ],
)
def test_half_open_acquisition_windows_preserve_depth_raw_and_state_independence(elapsed, active):
    raw = at(10 + elapsed)
    copies = {k: v.array.copy() for k, v in raw.items()}
    channel = CameraObscurationChannel(spec(), 10.0)
    # Receipt/control time can be arbitrarily later; acquisition time controls the decision.
    changed, report = channel.frames(raw, requested_s=30.0)
    assert report["active"] is active
    assert changed["depth"] is raw["depth"]
    for kind in raw:
        assert np.array_equal(raw[kind].array, copies[kind])
    for kind in ("rgb", "segmentation"):
        assert np.array_equal(changed[kind].array, np.zeros_like(copies[kind]) if active else copies[kind])


@pytest.mark.parametrize(
    "intervals",
    [
        [{"start_s": 1.0, "end_s": 1.0}],
        [{"start_s": 2.0, "end_s": 1.0}],
        [{"start_s": 1.0, "end_s": 3.0}, {"start_s": 2.0, "end_s": 4.0}],
        [{"start_s": 3.0, "end_s": 4.0}, {"start_s": 0.0, "end_s": 1.0}],
        [{"start_s": float("nan"), "end_s": 1.0}],
        [{"start_s": 0.0, "end_s": float("inf")}],
        [{"start_s": True, "end_s": 1.0}],
    ],
)
def test_invalid_or_ambiguous_schedule_rejected(intervals):
    with pytest.raises(ValidationError):
        spec(intervals)


@pytest.mark.parametrize(
    "mutation", ["missing", "camera", "shape", "dtype", "unknown", "future", "pairing", "straddling"]
)
def test_ambiguous_mask_never_becomes_valid_transformed_label(mutation):
    frames = at(10.5)
    mask = frames["segmentation"]
    if mutation == "missing":
        frames.pop("segmentation")
    elif mutation == "camera":
        mask.ref.camera_name = "another"
    elif mutation == "shape":
        mask.array = mask.array[:1]
    elif mutation == "dtype":
        mask.array = mask.array.astype(float)
    elif mutation == "unknown":
        mask.ref = mask.ref.model_copy(update={"sim_time_s": None, "acquisition_time_known": False})
    elif mutation == "future":
        for frame in frames.values():
            frame.ref.sim_time_s = 12.0
    elif mutation == "pairing":
        mask.ref.sim_time_s = 10.6
    elif mutation == "straddling":
        mask.ref.sim_time_s = 10.4999
    changed, report = CameraObscurationChannel(spec(), 10.0).frames(frames, requested_s=10.5)
    assert changed is None and report["status"] == "refused" and report["active"] is None
    assert report["reasons"]


def protocol(capture_every=1):
    data = expanded_protocol("degraded_perception", 0, horizon_s=6).model_dump(mode="json")
    data.update(
        protocol_schema_version="3.0.0",
        controlled_study=ControlledStudySpec(camera_obscuration=spec()).model_dump(mode="json"),
    )
    data["study_extension"].update(capture_segmentation=True, capture_sensors=False)
    data["simulation"].update(capture_every_n_steps=capture_every, save_frames_every_n_steps=20)
    return ProtocolConfig.model_validate(data)


@pytest.mark.parametrize("change", ["no_extension", "no_mask", "no_rgb", "beyond", "legacy_f1", "legacy_f4"])
def test_opt_in_prerequisites_fail_before_execution(change):
    data = protocol().model_dump(mode="json")
    if change == "no_extension":
        data["study_extension"] = None
    elif change == "no_mask":
        data["study_extension"]["capture_segmentation"] = False
    elif change == "no_rgb":
        data["simulation"]["capture_rgb"] = False
    elif change == "beyond":
        data["controlled_study"]["camera_obscuration"]["intervals"][0]["end_s"] = 9.0
    else:
        data["study_extension"]["severity"] = 0.5
        data["study_extension"]["family"] = (
            "simulated_manipulation" if change == "legacy_f4" else "degraded_perception"
        )
    with pytest.raises(ValidationError):
        ProtocolConfig.model_validate(data)


class CaptureEachStep:
    controller_id = "synthetic_capture_command_source"

    def reset(self, brief):
        pass

    def step(self, observation):
        return ControlCommand(
            step_index=observation.step_index,
            issued_sim_time_s=observation.receive_sim_time_s,
            kind="inspect_capture",
        )

    def internal_state(self):
        return {"phase": "inspect_capture", "synthetic_test_command_source": True}


def run(protocol, root, *, mutate=None):
    manifest = build_expanded_manifest(protocol, "fixture", 0)
    manifest.schedules.observation_delay_s = [0.5] * len(manifest.schedules.observation_delay_s)
    paths = PathsConfig(results_root=root)
    writer = EvidenceWriter(paths, "fixture", protocol.content_hash(), protocol)
    with fixture_fake_server() as endpoint:
        config = AppConfig(endpoint=endpoint, paths=paths)
        adapter = build_adapter(config, protocol)
        if mutate:
            actual = adapter.capture

            def capture(*args, **kwargs):
                frames = actual(*args, **kwargs)
                mutate(frames)
                return frames

            adapter.capture = capture
        runner = EpisodeRunner(adapter, protocol, config, writer)
        try:
            result = runner.run(
                manifest, "A0_unguarded", arm=Arm(protocol.arms.get("A0_unguarded"), CaptureEachStep(), None)
            )
        finally:
            adapter.close()
    return result, writer, runner


def load_pixels(record, stage, kind):
    item = record["frames"][stage][kind]["pixels"]
    data = Path(item["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == item["sha256"]
    assert len(data) == item["bytes"]
    pixels = np.load(item["path"], allow_pickle=False)
    assert "sha256:" + hashlib.sha256(pixels.tobytes()).hexdigest() == item["array_sha256"]
    assert pixels.dtype.name == item["dtype"] and list(pixels.shape) == item["shape"]
    return pixels


def test_production_runner_after_cruise_delay_raw_custody_masks_and_execution(tmp_path):
    p = protocol()
    result, writer, runner = run(p, tmp_path)
    assert result.attempt.status == "completed", result.attempt.model_dump()
    sidecar = writer.root / "privileged_camera_transformations" / f"{result.record.episode_id}.jsonl"
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert len(rows) == len(result.record.steps) == 12
    assert runner._common_window["control_window_start_s"] > 1.0
    assert rows[0]["rgb_post_cruise_s"] == 0 and not rows[0]["active"]
    events = {e.payload["step_index"]: e for e in result.ledger.events_of("inspection_capture_performed")}
    for row in rows:
        k = row["step_index"]
        active = p.controlled_study.camera_obscuration.interval_at(row["rgb_post_cruise_s"]) is not None
        assert row["active"] is active
        assert row["protocol_hash"] == p.content_hash()
        for kind in ("rgb", "depth", "segmentation"):
            raw = load_pixels(row, "raw", kind)
            before = load_pixels(row, "pre_obscuration", kind)
            delivered = load_pixels(row, "delivered", kind)
            if kind == "depth":
                assert np.array_equal(raw, delivered, equal_nan=True)
            elif active:
                assert not np.any(delivered)
            else:
                assert np.array_equal(before, delivered)
        actual = events[k].payload["frames"]["rgb"]
        assert actual == row["frames"]["delivered"]["rgb"]["ref"]
        observed = result.record.steps[k].observation.rgb
        if k:
            assert observed.frame_id == rows[k - 1]["acquisition_id"]
            assert observed.sim_time_s < actual["sim_time_s"]
            assert (
                observed.content_sha256 == rows[k - 1]["frames"]["delivered"]["rgb"]["ref"]["content_sha256"]
            )
    # At onset, the currently executed dark capture differs from the clear delayed permission image.
    onset = next(row for row in rows if row["active"])
    k = onset["step_index"]
    assert (
        result.record.steps[k].observation.rgb.content_sha256
        != events[k].payload["frames"]["rgb"]["content_sha256"]
    )
    perception = [
        json.loads(line) for line in (writer.root / "perception/rows.jsonl").read_text().splitlines()
    ]
    for row, transformed in zip(perception, rows, strict=True):
        if transformed["active"]:
            assert row["features"]["rgb_mean"] == 0 and row["label"]["value"] == 0
            assert row["features"]["depth_available"] == 1
    state_pairs = writer.root / "privileged_state_transformations" / f"{result.record.episode_id}.jsonl"
    assert all(
        r["raw_state"] == r["delivered_state"] for r in map(json.loads, state_pairs.read_text().splitlines())
    )
    with pytest.raises(EvidenceExists):
        writer.assert_episode_not_recorded(result.record.episode_id)
    assert result.ledger.events_of("fault_started") and result.ledger.events_of("fault_ended")


def test_execution_between_observation_acquisitions_uses_same_camera_only_contract(tmp_path):
    result, writer, _ = run(protocol(capture_every=2), tmp_path)
    assert result.attempt.status == "completed", result.attempt.model_dump()
    rows = [
        json.loads(line)
        for line in (writer.root / "privileged_camera_transformations" / f"{result.record.episode_id}.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(rows) == 12
    assert sum(r["role"] == "execution_fallback" for r in rows) == 6
    first = rows[1]
    assert first["role"] == "execution_fallback" and first["active"]
    assert not np.any(load_pixels(first, "delivered", "rgb"))
    assert not np.any(load_pixels(first, "delivered", "segmentation"))
    assert np.array_equal(load_pixels(first, "raw", "depth"), load_pixels(first, "delivered", "depth"))
    event = result.ledger.events_of("inspection_capture_performed")[1]
    assert event.payload["frames"]["rgb"]["frame_id"] == first["acquisition_id"]


def test_production_pairing_refusal_retains_raw_and_never_writes_false_label(tmp_path):
    def unpaired(frames):
        frames["segmentation"].ref.camera_name = "mismatched"

    result, writer, _ = run(protocol(), tmp_path, mutate=unpaired)
    assert result.attempt.status == "crashed"
    assert "pairing refused" in result.attempt.error_message
    row = json.loads(
        (writer.root / "privileged_camera_transformations" / f"{result.record.episode_id}.jsonl").read_text()
    )
    assert row["status"] == "refused" and row["frames"]["delivered"] == {}
    assert load_pixels(row, "raw", "rgb").size > 0
    assert not (writer.root / "perception/rows.jsonl").exists()
    assert not result.ledger.events_of("inspection_capture_performed")


def test_opt_in_without_persistence_refuses_before_any_adapter_call(tmp_path):
    p = protocol()
    runner = EpisodeRunner(object(), p, AppConfig(), save_frames=False)
    with pytest.raises(ValueError, match="retained"):
        runner.run(build_expanded_manifest(p, "fixture", 0), "A0_unguarded")


def test_camera_sidecar_only_orphan_refuses_repeat_before_touching_adapter(tmp_path):
    p = protocol()
    paths = PathsConfig(results_root=tmp_path)
    writer = EvidenceWriter(paths, "fixture", p.content_hash(), p)
    manifest = build_expanded_manifest(p, "fixture", 0)
    episode_id = f"{manifest.scenario_id}__A0_unguarded"
    sidecar = writer.root / "privileged_camera_transformations" / f"{episode_id}.jsonl"
    sidecar.parent.mkdir()
    retained = b'{"status":"refused","frames":{"raw":{}}}\n'
    sidecar.write_bytes(retained)
    runner = EpisodeRunner(object(), p, AppConfig(paths=paths), writer)
    with pytest.raises(EvidenceExists, match="privileged_camera_transformations"):
        runner.run(manifest, "A0_unguarded")
    assert sidecar.read_bytes() == retained


def test_legacy_f1_still_masks_depth_and_serialization_omits_option():
    p = expanded_protocol("degraded_perception", 1.0)
    channel = PerturbationChannel(p.study_extension, 0, active=True)
    raw = _frames()
    altered = channel.frames(raw)
    assert np.any(altered["depth"].array == 0) and np.all(raw["depth"].array > 0)
    assert "camera_obscuration" not in ControlledStudySpec().model_dump()
    absent = ControlledStudySpec.model_validate(
        ControlledStudySpec().model_dump() | {"camera_obscuration": None}
    )
    assert absent.model_dump_json() == ControlledStudySpec().model_dump_json()


def test_v4_gate_acknowledges_acquired_black_pixels_without_claiming_target_presence(tmp_path, monkeypatch):
    from tests.unit import test_context_confidence_v4 as existing

    # Reuse the fitted-model recipe and real controller/gate/runner fixture. Its direct phase
    # entry is a software test, never evidence of navigation or scene qualification.
    p = existing.v4.__wrapped__(tmp_path)
    data = p.model_dump(mode="json")
    data["controlled_study"]["camera_obscuration"] = spec([{"start_s": 0.0, "end_s": 30.0}]).model_dump(
        mode="json"
    )
    data["context_confidence"].update(clear_threshold=0.0, degraded_threshold=0.0)
    p = ProtocolConfig.model_validate(data)

    class RetainingRunner(EpisodeRunner):
        def __init__(self, *args, **kwargs):
            kwargs["save_frames"] = True
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(existing, "EpisodeRunner", RetainingRunner)
    result = existing.run_fixture(p, tmp_path / "black-camera")
    assert result.attempt.status == "completed", result.attempt.model_dump()
    performed = result.ledger.events_of("inspection_capture_performed")
    assert len(performed) == p.mission.required_inspection_captures == 3
    for event in performed:
        assert event.payload["capture_execution_feedback"]["performed"]
        assert not event.payload["frames_nonempty"]["rgb"]  # Black is acquired, not a useful view.
        assert event.payload["context_confidence_permission"]["status"] == "pass"
        rgb = np.load(event.payload["frames"]["rgb"]["path"], allow_pickle=False)
        mask = np.load(event.payload["frames"]["segmentation"]["path"], allow_pickle=False)
        assert rgb.size > 0 and not np.any(rgb) and not np.any(mask)
        assert (
            event.payload["frames"]["rgb"]["frame_id"]
            != (event.payload["context_confidence_permission"]["observation_frame_id"])
        )
