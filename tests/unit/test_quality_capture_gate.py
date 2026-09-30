"""Availability-matched Q policy: software fixtures, not live vision/safety evidence."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from tests.unit.test_context_confidence_v4 import MODEL_HASH, capture, monitor, packet, policy
from tests.unit.test_context_confidence_v4 import v4 as v4_fixture

from colosseum_assurance.config import AppConfig, PathsConfig
from colosseum_assurance.control.controller import InspectionController
from colosseum_assurance.monitors.context_confidence import ContextConfidenceGuard
from colosseum_assurance.perception_eval.contracts import content_hash
from colosseum_assurance.protocol.context_confidence import ContextConfidenceSpec
from colosseum_assurance.protocol.controlled import ControlledStudySpec
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.runtime.arms import build_arm
from colosseum_assurance.runtime.episode import EpisodeRunner
from colosseum_assurance.runtime.evidence import EvidenceWriter
from colosseum_assurance.scenario.expanded import build_expanded_manifest, expanded_protocol
from colosseum_assurance.schemas import MonitorReport, Verdict
from colosseum_assurance.sim import build_adapter, fixture_fake_server

# Import the existing production-model fixture without duplicating its fitted artifact recipe.
v4 = v4_fixture
QUALITY = "observable_quality_capture_gate_v1"


def quality_policy(**updates):
    return policy(semantics_version=QUALITY, clear_threshold=None, degraded_threshold=None, **updates)


def quality_evidence(observation, *, depth_available=True, fraction=0.5, rgb_std=0.2):
    evidence = observation.perception_evidence
    features = dict(
        evidence.features,
        depth_available=float(depth_available),
        depth_valid_fraction=fraction,
        rgb_std=rgb_std,
    )
    observation.perception_evidence = evidence.model_copy(
        update={
            "features": features,
            "features_sha256": content_hash(features),
            "depth_available": depth_available,
            "depth_valid_fraction": fraction,
            "rgb_std": rgb_std,
        }
    )
    return observation


@pytest.mark.parametrize("probability", [0.0, 0.01, 0.5, 0.99, 1.0])
@pytest.mark.parametrize("clear", [True, False])
def test_quality_rule_ignores_every_eligible_probability_including_endpoints(probability, clear):
    config = quality_policy(clear_min_rgb_std=0.2)
    obs = quality_evidence(packet(probability=probability), rgb_std=0.2 if clear else 0.19)
    result = monitor(config).evaluate(obs, capture())
    decision = result.context_confidence_evidence
    assert result.verdict is (Verdict.PASS if clear else Verdict.UNKNOWN)
    assert result.affirmative is clear
    assert result.intervention == ("none" if clear else "hold")
    assert decision["threshold"] is None
    assert decision["probability"] == probability  # record, never forge the observed score
    assert decision["policy_semantics"] == QUALITY
    assert not decision["eligible_probability_magnitude_used"]
    assert result.monitor_id.endswith("+" + QUALITY)
    assert result.assumption_verdicts["observable_capture_quality"] is result.verdict


@pytest.mark.parametrize(
    "available,fraction,rgb_std,passed",
    [
        (True, 0.5, 0.2, True),
        (True, np.nextafter(0.5, 0), 0.2, False),
        (True, 0.5, np.nextafter(0.2, 0), False),
        (False, 0.5, 0.2, False),
    ],
)
def test_quality_boundaries_are_inclusive_and_depth_must_be_available(available, fraction, rgb_std, passed):
    obs = quality_evidence(packet(), depth_available=available, fraction=fraction, rgb_std=rgb_std)
    result = monitor(quality_policy(clear_min_rgb_std=0.2)).evaluate(obs, capture())
    assert result.affirmative is passed


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_envelope",
        "missing_probability",
        "nonfinite_probability",
        "mismatched_probability",
        "model",
        "camera",
        "frame",
        "pixels",
        "feature_hash",
        "stale",
        "future",
    ],
)
def test_quality_and_probability_rules_share_fail_closed_eligibility(mutation):
    obs = packet()
    if mutation == "missing_envelope":
        obs.perception_evidence = None
    elif mutation == "missing_probability":
        obs.asset_presence_probability = None
        obs.perception_evidence.probability = None
    elif mutation == "nonfinite_probability":
        # Deliberately bypass schema validation to exercise defensive boundary checks.
        obs = obs.model_copy(update={"asset_presence_probability": float("nan")})
    elif mutation == "mismatched_probability":
        obs.perception_evidence.probability = 0.7
    elif mutation == "model":
        obs.perception_model_hash = "sha256:" + "b" * 64
    elif mutation == "camera":
        obs.rgb.camera_name = "different"
    elif mutation == "frame":
        obs.rgb.frame_id = "different"
    elif mutation == "pixels":
        obs.rgb.content_sha256 = "sha256:" + "b" * 64
    elif mutation == "feature_hash":
        obs.perception_evidence.features["rgb_std"] = 0.2
    else:
        when = 8.9 if mutation == "stale" else 10.1
        obs = obs.model_copy(update={"perception_prediction_time_s": when})
        obs.perception_evidence.image_time_s = when
        obs.rgb.sim_time_s = when
    ordinary = monitor().evaluate(obs, capture())
    quality = monitor(quality_policy()).evaluate(obs, capture())
    assert quality.verdict is ordinary.verdict is Verdict.UNKNOWN
    assert not quality.affirmative and not ordinary.affirmative
    assert quality.intervention == ordinary.intervention == "hold"
    assert quality.context_confidence_evidence["reason"] == ordinary.context_confidence_evidence["reason"]


@pytest.mark.parametrize(
    "verdict,action",
    [
        (Verdict.VIOLATION, "abort"),
        (Verdict.VIOLATION, "return_to_launch"),
        (Verdict.UNKNOWN, "suspend_inspection"),
    ],
)
@pytest.mark.parametrize("quality_clear", [True, False])
def test_quality_never_erases_stronger_base_verdict_or_response(verdict, action, quality_clear):
    class Base:
        monitor_id = "synthetic_stronger_base"

        def evaluate(self, observation, command):
            return MonitorReport(
                step_index=command.step_index,
                sim_time_s=10,
                monitor_id=self.monitor_id,
                verdict=verdict,
                intervention=action,
                affirmative=False,
            )

    obs = quality_evidence(packet(probability=0), rgb_std=0.2 if quality_clear else 0)
    result = ContextConfidenceGuard(Base(), quality_policy(clear_min_rgb_std=0.1)).evaluate(obs, capture())
    assert result.verdict is verdict and result.intervention == action and not result.affirmative


@pytest.mark.parametrize(
    "version,thresholds",
    [
        (QUALITY, (0.0, 1.0)),
        (QUALITY, (None, 0.8)),
        ("observed_asset_capture_gate_v1", (None, None)),
        ("observed_asset_capture_gate_v1", (0.7, None)),
    ],
)
def test_policy_variants_refuse_ambiguous_or_implicit_score_semantics(version, thresholds):
    with pytest.raises(ValidationError):
        policy(semantics_version=version, clear_threshold=thresholds[0], degraded_threshold=thresholds[1])


def test_quality_requires_explicit_nulls_and_non_capture_is_not_applicable():
    data = quality_policy().model_dump(mode="json")
    del data["clear_threshold"]
    with pytest.raises(ValidationError):
        ContextConfidenceSpec.model_validate(data)
    obs = packet(probability=None)
    result = monitor(quality_policy()).evaluate(obs, capture().model_copy(update={"kind": "land"}))
    assert result.assumption_verdicts["observable_capture_quality"] is Verdict.NOT_APPLICABLE
    assert result.context_confidence_evidence is None
    assert "ignores eligible probability magnitude" in monitor(quality_policy()).describe()["limitation"]


def test_existing_v4_protocol_policy_and_reports_keep_exact_prevariant_hashes():
    p = policy()
    assert p.content_hash() == "sha256:adbd10b9437a0b7df437bd7848669cb74c505438e7882bcce727bcd01c2a6292"
    data = expanded_protocol("degraded_perception", 0, horizon_s=30).model_dump(mode="json")
    data.update(
        protocol_schema_version="4.0.0",
        controlled_study=ControlledStudySpec().model_dump(),
        context_confidence=p.model_dump(),
    )
    data["study_extension"].update(
        perception_model_path="/frozen/civilian-model.json",
        perception_model_hash=MODEL_HASH,
        capture_segmentation=True,
        capture_sensors=False,
    )
    data["simulation"]["capture_every_n_steps"] = 1
    assert ProtocolConfig.model_validate(data).content_hash() == (
        "sha256:e2b1c5f44f08201ef4cad4a1ade8fb4f62e0aa823e9a904be23de7a8c92ca3d1"
    )
    for obs, expected in [
        (packet(), "0d2daf2d93416c5ccd4dac7742d30d02825ca8e47c9f447fb1abdbfd2c0c0c83"),
        (packet(depth=False), "fc9e4447d3867206a92b15abad08174252e6632417798c7b2933eca0090765b3"),
        (
            packet().model_copy(update={"perception_evidence": None}),
            "1c321b1408f66252805446d39ed789390f3cc7d2dcdf9c22505778167f6e61cb",
        ),
    ]:
        assert (
            content_hash(monitor().evaluate(obs, capture()).model_dump(mode="json")) == "sha256:" + expected
        )


def quality_protocol(v4):
    data = v4.model_dump(mode="json")
    data["context_confidence"].update(
        semantics_version=QUALITY, clear_threshold=None, degraded_threshold=None, clear_min_rgb_std=0.1
    )
    return ProtocolConfig.model_validate(data)


def run_quality_fixture(protocol, out, *, permanently_low_quality=False):
    """Real runner/RPC fixture with observation quality restoration, no flight-competence claim."""

    class StartCapture(InspectionController):
        def reset(self, brief):
            super().reset(brief)
            self._set_phase("inspect_capture", "synthetic quality-gate execution regression")

    manifest = build_expanded_manifest(protocol, "fixture", 0)
    manifest.schedules.observation_delay_s = [0.5] * len(manifest.schedules.observation_delay_s)
    paths = PathsConfig(results_root=out)
    writer = EvidenceWriter(paths, "fixture", protocol.content_hash(), protocol)
    arm = build_arm(protocol, "A1_policy_only")
    arm.controller = StartCapture(protocol)
    arm.controller.enable_capture_acknowledgments(protocol.context_confidence.max_capture_opportunities)
    with fixture_fake_server() as endpoint:
        config = AppConfig(endpoint=endpoint, paths=paths)
        adapter = build_adapter(config, protocol)
        original = adapter.capture
        calls = 0

        def capture_quality(*args, **kwargs):
            nonlocal calls
            frames = original(*args, **kwargs)
            if "rgb" in frames:
                calls += 1
                pixels = np.zeros_like(frames["rgb"].array)
                if not permanently_low_quality and calls > 2:
                    pixels[:, ::2] = 255
                frames["rgb"].array = pixels
            return frames

        adapter.capture = capture_quality
        try:
            return EpisodeRunner(adapter, protocol, config, writer, save_frames=False).run(
                manifest, "A1_policy_only", arm=arm
            )
        finally:
            adapter.close()


def test_quality_production_runner_retries_then_acknowledges_actual_capture(v4, tmp_path):
    protocol = quality_protocol(v4)
    assert (
        ProtocolConfig.model_validate_json(protocol.model_dump_json()).content_hash()
        == protocol.content_hash()
    )
    result = run_quality_fixture(protocol, tmp_path / "quality-restores")
    assert result.attempt.status == "completed", result.attempt.model_dump()
    proposals = [s for s in result.record.steps if s.command.kind == "inspect_capture"]
    held = [s for s in proposals if s.executed_command.kind == "hold"]
    assert held and all(not s.controller_state["capture_execution_feedback"]["performed"] for s in held)
    captures = result.ledger.events_of("inspection_capture_performed")
    assert len(captures) == protocol.mission.required_inspection_captures == 3
    assert len(proposals) > len(captures)
    for event in captures:
        evidence = event.payload["context_confidence_permission"]
        assert evidence["policy_semantics"] == QUALITY and evidence["threshold"] is None
        assert evidence["status"] == "pass" and not evidence["eligible_probability_magnitude_used"]
        assert evidence["observation_frame_id"] != event.payload["frames"]["rgb"]["frame_id"]
        assert event.payload["capture_execution_feedback"]["performed"]
    assert result.record.steps[-1].provenance["controlled_followup"]


def test_quality_production_runner_exhaustion_never_counts_veto_as_capture(v4, tmp_path):
    protocol = quality_protocol(v4)
    result = run_quality_fixture(protocol, tmp_path / "quality-unavailable", permanently_low_quality=True)
    assert result.attempt.status == "completed", result.attempt.model_dump()
    proposals = [s for s in result.record.steps if s.command.kind == "inspect_capture"]
    assert len(proposals) == protocol.context_confidence.max_capture_opportunities
    assert not result.ledger.events_of("inspection_capture_performed")
    assert all(s.controller_state["captures_acknowledged"] == 0 for s in proposals)
    assert any(
        s.controller_state.get("capture_unavailability_reason") == "context_capture_opportunities_exhausted"
        for s in result.record.steps
    )
    endings = [
        e.payload["controlled_study_mission_terminal"]
        for e in result.ledger.events_of("controller_phase_change")
        if "controlled_study_mission_terminal" in e.payload
    ]
    assert endings and endings[0]["reason"] == "controller_abort"
