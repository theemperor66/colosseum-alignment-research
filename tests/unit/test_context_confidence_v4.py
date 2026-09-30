"""MON-02 opt-in software contract tests; synthetic images/RPC fixtures are not live evidence."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from pydantic import ValidationError
from tests.unit.test_monitor_policy_only import FAR_FROM_ASSET, make_brief, make_packet
from tests.unit.test_perception_eval import fixture_rows, label

from colosseum_assurance.config import AppConfig, PathsConfig
from colosseum_assurance.control.controller import InspectionController
from colosseum_assurance.monitors.context_confidence import ContextConfidenceGuard
from colosseum_assurance.monitors.policy_only import PolicyOnlyMonitor
from colosseum_assurance.perception_eval import LabelSpec, extract_features, fit_model, save_model
from colosseum_assurance.perception_eval.contracts import content_hash
from colosseum_assurance.protocol.context_confidence import ContextConfidenceSpec
from colosseum_assurance.protocol.controlled import ControlledStudySpec
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.runtime.arms import build_arm
from colosseum_assurance.runtime.episode import EpisodeRunner
from colosseum_assurance.runtime.evidence import EvidenceWriter
from colosseum_assurance.scenario.expanded import build_expanded_manifest, expanded_protocol
from colosseum_assurance.schemas import (
    ControlCommand,
    FrameRef,
    MonitorReport,
    PerceptionPredictionEvidence,
    Verdict,
)
from colosseum_assurance.sim import build_adapter, fixture_fake_server

MODEL_HASH = "sha256:" + "a" * 64


def policy(**updates):
    values = dict(
        applies_to_arms=["A1_policy_only"],
        expected_model_hash=MODEL_HASH,
        expected_label_spec_hash=LabelSpec().spec_hash,
        camera_name="front_center",
        clear_min_depth_fraction=0.5,
        clear_min_rgb_std=0.0,
        clear_threshold=0.8,
        degraded_threshold=0.9,
        max_evidence_age_s=1.0,
        max_capture_opportunities=8,
    )
    return ContextConfidenceSpec.model_validate(values | updates)


def packet(*, probability=0.85, depth=True):
    rgb = np.full((10, 10, 3), 180, dtype=np.uint8)
    features = extract_features(rgb, np.full((10, 10), 10.0) if depth else None)
    fingerprint = "sha256:" + hashlib.sha256(rgb.tobytes()).hexdigest()
    obs = make_packet(4, 10.0, FAR_FROM_ASSET, heartbeat_received_at_s=10.0)
    evidence = PerceptionPredictionEvidence(
        frame_id="synthetic:frame-3",
        rgb_sha256=fingerprint,
        camera_name="front_center",
        image_time_s=9.5,
        width=10,
        height=10,
        model_hash=MODEL_HASH,
        feature_version="civilian_rgb_depth_v1",
        label_spec_hash=LabelSpec().spec_hash,
        features_sha256=content_hash(features),
        features=features,
        depth_available=depth,
        depth_valid_fraction=features["depth_valid_fraction"],
        rgb_std=features["rgb_std"],
        probability=probability,
    )
    return obs.model_copy(
        update=dict(
            rgb=FrameRef(
                kind="rgb",
                camera_name="front_center",
                sim_time_s=9.5,
                width=10,
                height=10,
                frame_id=evidence.frame_id,
                content_sha256=fingerprint,
            ),
            perception_evidence=evidence,
            asset_presence_probability=probability,
            perception_model_hash=MODEL_HASH,
            perception_prediction_time_s=9.5,
        )
    )


def monitor(config=None):
    guard = ContextConfidenceGuard(PolicyOnlyMonitor(ProtocolConfig()), config or policy())
    guard.reset(make_brief())
    return guard


def capture():
    return ControlCommand(step_index=4, issued_sim_time_s=10, kind="inspect_capture")


def test_context_quality_is_same_observation_not_privileged_visibility():
    clear = monitor().evaluate(packet(), capture())
    degraded = monitor().evaluate(packet(depth=False), capture())
    assert clear.verdict is Verdict.PASS and clear.affirmative
    assert clear.context_confidence_evidence["context"] == "clear_observation"
    assert clear.context_confidence_evidence["threshold"] == 0.8
    assert degraded.verdict is Verdict.UNKNOWN and not degraded.affirmative
    assert degraded.intervention == "hold"
    assert degraded.context_confidence_evidence["context"] == "degraded_observation"
    assert degraded.context_confidence_evidence["threshold"] == 0.9
    changed_truth = packet().model_copy(update={"sensor_samples": {"truth": {"asset_visible": False}}})
    assert monitor().evaluate(changed_truth, capture()) == clear


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "model",
        "label",
        "frame",
        "camera",
        "pixels",
        "features",
        "future",
        "stale",
        "probability",
        "no_rgb",
    ],
)
def test_unknown_or_mismatched_binding_never_affirms(mutation):
    obs = packet()
    if mutation == "missing":
        obs.perception_evidence = None
    elif mutation == "no_rgb":
        obs.rgb = None
    elif mutation == "model":
        obs.perception_evidence.model_hash = "sha256:" + "b" * 64
    elif mutation == "label":
        obs.perception_evidence.label_spec_hash = "sha256:" + "b" * 64
    elif mutation == "frame":
        obs.rgb.frame_id = "another frame"
    elif mutation == "camera":
        obs.rgb.camera_name = "another camera"
    elif mutation == "pixels":
        obs.rgb.content_sha256 = "sha256:" + "b" * 64
    elif mutation == "features":
        obs.perception_evidence.features["rgb_std"] += 0.1
    elif mutation in {"future", "stale"}:
        when = 10.1 if mutation == "future" else 8.9
        obs = obs.model_copy(update={"perception_prediction_time_s": when})
        obs.perception_evidence.image_time_s = when
        obs.rgb.sim_time_s = when
    elif mutation == "probability":
        obs.perception_evidence.probability = 0.99
    result = monitor().evaluate(obs, capture())
    assert result.verdict is Verdict.UNKNOWN and not result.affirmative
    assert result.intervention == "hold"
    base = PolicyOnlyMonitor(ProtocolConfig())
    base.reset(make_brief())
    assert result.obligation_verdicts == base.evaluate(obs, capture()).obligation_verdicts


@pytest.mark.parametrize("kind", ["hold", "return_to_launch", "abort", "suspend_inspection"])
def test_stronger_base_interventions_and_violations_are_retained(kind):
    class Base:
        monitor_id = "fixture_base"

        def evaluate(self, obs, command):
            return MonitorReport(
                step_index=obs.step_index,
                sim_time_s=obs.receive_sim_time_s,
                monitor_id=self.monitor_id,
                verdict=Verdict.VIOLATION,
                intervention=kind,
                obligation_verdicts={"geofence": Verdict.VIOLATION},
            )

    result = ContextConfidenceGuard(Base(), policy()).evaluate(packet(probability=0.1), capture())
    assert result.verdict is Verdict.VIOLATION and result.intervention == kind
    assert not result.affirmative


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_evidence_age_s", float("inf")),
        ("clear_threshold", float("nan")),
        ("clear_min_depth_fraction", True),
        ("max_capture_opportunities", 0),
    ],
)
def test_nonfinite_and_invalid_policy_values_rejected_before_hashing(field, value):
    with pytest.raises(ValidationError):
        policy(**{field: value})


def test_inclusive_age_threshold_and_non_capture_are_explicit():
    obs = packet(probability=0.8)
    assert monitor(policy(max_evidence_age_s=0.5)).evaluate(obs, capture()).verdict is Verdict.PASS
    command = capture().model_copy(update={"kind": "land"})
    result = monitor().evaluate(packet(probability=None), command)
    assert result.assumption_verdicts["context_asset_capture_confidence"] is Verdict.NOT_APPLICABLE
    assert result.context_confidence_evidence is None


@pytest.fixture
def v4(tmp_path):
    rows = [
        r.model_copy(update={"label": label(bool(r.label.value), spec=LabelSpec())}) for r in fixture_rows()
    ]
    model = fit_model(rows)
    model_path = tmp_path / "fixture-model.json"
    save_model(model_path, model)
    data = expanded_protocol("degraded_perception", 0, horizon_s=30).model_dump(mode="json")
    data.update(
        protocol_schema_version="4.0.0",
        controlled_study=ControlledStudySpec().model_dump(),
        context_confidence=policy(expected_model_hash=model.model_hash).model_dump(),
    )
    data["study_extension"].update(
        perception_model_path=str(model_path),
        perception_model_hash=model.model_hash,
        capture_segmentation=True,
        capture_sensors=False,
    )
    data["simulation"]["capture_every_n_steps"] = 1
    return ProtocolConfig.model_validate(data)


def test_v4_requires_explicit_contract_and_does_not_change_v1_v2_v3_serialization(v4):
    for protocol in (ProtocolConfig(), expanded_protocol("degraded_perception", 0)):
        data = protocol.model_dump(mode="json")
        assert "context_confidence" not in data
        rebuilt = ProtocolConfig.model_validate(data | {"context_confidence": None})
        assert protocol.canonical_json() == rebuilt.canonical_json()
    legacy = v4.model_dump(mode="json")
    legacy.pop("context_confidence")
    legacy["protocol_schema_version"] = "3.0.0"
    v3 = ProtocolConfig.model_validate(legacy)
    assert v3.model_dump(mode="json") == legacy
    assert build_arm(v3, "A1_policy_only").controller._capture_ack_budget is None
    for mutation in (
        {"protocol_schema_version": "3.0.0"},
        {"context_confidence": None},
        {"controlled_study": None},
    ):
        with pytest.raises(ValidationError):
            ProtocolConfig.model_validate(v4.model_dump(mode="json") | mutation)
    old_obs = make_packet(0, 0, FAR_FROM_ASSET)
    assert "perception_evidence" not in old_obs.model_dump(mode="json")
    assert (
        "frame_id"
        not in FrameRef(kind="rgb", camera_name="front", sim_time_s=0, width=1, height=1).model_dump()
    )
    assert (
        "context_confidence_evidence"
        not in MonitorReport(step_index=0, sim_time_s=0, monitor_id="old", verdict=Verdict.PASS).model_dump()
    )


def test_v4_acknowledgment_counts_only_executed_available_capture_and_refuses_duplicates(v4):
    controller = build_arm(v4, "A1_policy_only").controller
    controller.reset(make_brief())
    controller._m.capture_pending_step = 4
    controller._m.capture_opportunities = 1
    command = capture()
    receipt = controller.acknowledge_capture(command, command.model_copy(update={"kind": "hold"}), True)
    assert not receipt["performed"] and controller._m.captures == 0
    with pytest.raises(ValueError, match="outstanding"):
        controller.acknowledge_capture(command, command, True)
    controller._m.capture_pending_step = 4
    assert not controller.acknowledge_capture(command, command, False)["performed"]
    controller._m.capture_pending_step = 4
    assert controller.acknowledge_capture(command, command, True)["performed"]
    assert controller._m.captures == 1


def run_fixture(protocol, tmp_path, *, always_dark=False):
    """Enter inspection directly to isolate the production gate/execution loop, not flight competence."""

    class StartCapture(InspectionController):
        def reset(self, brief):
            super().reset(brief)
            self._set_phase("inspect_capture", "synthetic phase entry for MON02 execution regression")

    manifest = build_expanded_manifest(protocol, "fixture", 0)
    # Fixed one-step delay makes the accepted observation and executed image observably different.
    manifest.schedules.observation_delay_s = [0.5] * len(manifest.schedules.observation_delay_s)
    paths = PathsConfig(results_root=tmp_path)
    writer = EvidenceWriter(paths, "fixture", protocol.content_hash(), protocol)
    arm = build_arm(protocol, "A1_policy_only")
    arm.controller = StartCapture(protocol)
    arm.controller.enable_capture_acknowledgments(protocol.context_confidence.max_capture_opportunities)
    with fixture_fake_server() as endpoint:
        config = AppConfig(endpoint=endpoint, paths=paths)
        adapter = build_adapter(config, protocol)
        actual_capture = adapter.capture
        calls = 0

        def photographed(*args, **kwargs):
            nonlocal calls
            frames = actual_capture(*args, **kwargs)
            if "rgb" in frames:
                calls += 1
                frames["rgb"].array = np.full_like(
                    frames["rgb"].array, 20 if always_dark or calls <= 2 else 210 + calls % 20
                )
            return frames

        adapter.capture = photographed
        try:
            result = EpisodeRunner(adapter, protocol, config, writer, save_frames=False).run(
                manifest, "A1_policy_only", arm=arm
            )
        finally:
            adapter.close()
    return result


def test_production_runner_veto_reobserves_and_acknowledges_real_executions(v4, tmp_path):
    result = run_fixture(v4, tmp_path / "retry")
    assert result.attempt.status == "completed", result.attempt.model_dump()
    proposals = [s for s in result.record.steps if s.command.kind == "inspect_capture"]
    assert len(proposals) > v4.mission.required_inspection_captures
    vetoes = [s for s in proposals if s.executed_command.kind == "hold"]
    assert vetoes and all(not s.controller_state["capture_execution_feedback"]["performed"] for s in vetoes)
    actual = result.ledger.events_of("inspection_capture_performed")
    assert len(actual) == v4.mission.required_inspection_captures
    for event in actual:
        permission = event.payload["context_confidence_permission"]
        assert permission["status"] == "pass"
        executed = event.payload["frames"]["rgb"]
        assert executed["frame_id"] != permission["observation_frame_id"]
        assert executed["sim_time_s"] > permission["acquisition_time_s"]
        assert event.payload["capture_execution_feedback"]["performed"]
        assert "does not certify newer-image content" in event.payload["capture_identity_scope"]
    assert result.record.steps[-1].provenance["controlled_followup"]


def test_production_runner_exhaustion_records_zero_captures_and_unavailability(v4, tmp_path):
    result = run_fixture(v4, tmp_path / "exhaust", always_dark=True)
    assert result.attempt.status == "completed", result.attempt.model_dump()
    proposals = [s for s in result.record.steps if s.command.kind == "inspect_capture"]
    assert len(proposals) == v4.context_confidence.max_capture_opportunities
    assert not result.ledger.events_of("inspection_capture_performed")
    assert all(s.controller_state["captures_acknowledged"] == 0 for s in proposals)
    states = [s.controller_state for s in result.record.steps]
    assert any(
        s.get("capture_unavailability_reason") == "context_capture_opportunities_exhausted" for s in states
    )
    endings = [
        e
        for e in result.ledger.events_of("controller_phase_change")
        if "controlled_study_mission_terminal" in e.payload
    ]
    assert endings and endings[0].payload["controlled_study_mission_terminal"]["reason"] == "controller_abort"


def test_v1_v2_v3_canonical_hash_regressions_from_prechange_source():
    base = ProtocolConfig()
    expanded = expanded_protocol("degraded_perception", 0)
    v3_data = expanded.model_dump(mode="json") | {
        "protocol_schema_version": "3.0.0",
        "controlled_study": ControlledStudySpec().model_dump(),
    }
    assert base.content_hash() == "sha256:e7c50b015da96a86c7428c3981ed28126ae0dd4e41de47296efb9f6278644eb3"
    assert (
        expanded.content_hash() == "sha256:31735ddb278944b3b7fd0fca0b8b6e930d5855a596b16b9e1b0f8103920162cf"
    )
    assert ProtocolConfig.model_validate(v3_data).content_hash() == (
        "sha256:8fe6285cd8464b03812d15ec1e0717d6b39c8d1ea0fa316eaa1a7efa05655ea5"
    )


def test_producer_binds_pixels_but_mask_changes_never_change_prediction_evidence(v4, tmp_path):
    from tests.unit.test_perception_runtime_integration import _frames

    from colosseum_assurance.runtime.perception_capture import record_perception_frame

    one = _frames()
    two = _frames()
    two["segmentation"].array[:] = 0
    results = []
    labels = []
    for name, frames in (("positive-mask", one), ("negative-mask", two)):
        for frame in frames.values():
            frame.ref.camera_name = v4.context_confidence.camera_name
        out = tmp_path / name / "rows.jsonl"
        results.append(
            record_perception_frame(
                frames=frames,
                spec=v4.study_extension,
                identity={"identity_verified": True},
                scenario_group="synthetic-same-frame",
                arm_id="A1_policy_only",
                frame_id="same-frame:0",
                provenance="fixture_fake",
                stratum="synthetic",
                out=out,
                dropped=False,
                hfov_rad=np.pi / 2,
                context_policy=v4.context_confidence,
            )
        )
        labels.append(json.loads(out.read_text())["label"]["value"])
    assert labels == [1, 0]
    assert results[0]["evidence"] == results[1]["evidence"]
    assert one["rgb"].ref.content_sha256 == results[0]["evidence"]["rgb_sha256"]
    assert set(results[0]["evidence"]["features"]) == set(extract_features(one["rgb"].array))


def test_header_mismatch_is_unknown_and_evidence_rejects_nonfinite_time(v4, tmp_path):
    from tests.unit.test_perception_runtime_integration import _frames

    from colosseum_assurance.runtime.perception_capture import record_perception_frame

    frames = _frames()
    frames["rgb"].ref.width += 1
    result = record_perception_frame(
        frames=frames,
        spec=v4.study_extension,
        identity={"identity_verified": True},
        scenario_group="synthetic",
        arm_id="A1_policy_only",
        frame_id="synthetic:0",
        provenance="fixture_fake",
        stratum="synthetic",
        out=tmp_path / "bad-shape.jsonl",
        dropped=False,
        hfov_rad=np.pi / 2,
        context_policy=v4.context_confidence,
    )
    assert result["prediction"] is None and "evidence" not in result
    data = packet().perception_evidence.model_dump() | {"image_time_s": float("inf")}
    with pytest.raises(ValidationError):
        PerceptionPredictionEvidence.model_validate(data)
