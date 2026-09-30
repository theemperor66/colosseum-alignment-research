"""Actual reset/prefix/branch execution through production RPC adapter; fixture data only."""

from __future__ import annotations

import json

import pytest

from colosseum_assurance.config import AppConfig, PathsConfig
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.runtime.episode import EpisodeRunner
from colosseum_assurance.runtime.evidence import EvidenceWriter
from colosseum_assurance.scenario.manifest import build_manifest
from colosseum_assurance.sim import build_adapter, fixture_fake_server
from colosseum_assurance.workflows.counterfactual import run_counterfactual


@pytest.fixture
def saved_episode(tmp_path):
    protocol = ProtocolConfig()
    protocol = protocol.model_copy(update={"mission": protocol.mission.model_copy(
        update={"episode_horizon_s": 40})})
    paths = PathsConfig(results_root=tmp_path / "source")
    writer = EvidenceWriter(paths, "fixture", protocol.content_hash(), protocol)
    manifest = build_manifest(protocol, "fixture", protocol.cells()[0]["cell_id"], 0)
    with fixture_fake_server() as endpoint:
        config = AppConfig(endpoint=endpoint, paths=paths)
        adapter = build_adapter(config, protocol)
        try:
            result = EpisodeRunner(adapter, protocol, config, writer, save_frames=False).run(
                manifest, "A0_unguarded")
        finally:
            adapter.close()
        assert result.record is not None and result.ledger is not None
        selected = next(s for s in result.record.steps
                        if s.executed_command.kind == "move_to" and s.step_index >= 2)
        yield writer.root, result.record, selected.step_index, config, tmp_path


def test_actual_original_and_hold_replay_share_start_and_diverge(saved_episode):
    root, record, step, config, tmp = saved_episode
    result = run_counterfactual(root, record.episode_id, step, config, tmp / "replay")
    assert result["status"] == "completed", result.get("refusal_reason")
    assert result["analysis_role"] == "exploration" and result["source_run_class"] == "fixture"
    assert result["between_branch_start_match"]["within_tolerance"]
    original, alternative = result["branches"]["original"], result["branches"]["hold"]
    assert original["source_endpoint_match"]["within_tolerance"]
    assert original["command"]["kind"] == "move_to" and alternative["command"]["kind"] == "hold"
    assert original["consequence"]["endpoint_position"] != alternative["consequence"]["endpoint_position"]
    assert len(original["prefix_trace"]) == step + 1
    assert len(original["samples"]) > 2
    assert original["consequence"]["observed_duration_s"] == pytest.approx(record.dt_s)
    assert json.loads((tmp / "replay" / "counterfactual.json").read_text())["status"] == "completed"
    assert (tmp / "replay" / "protocol.json").is_file()
    with pytest.raises(FileExistsError):
        run_counterfactual(root, record.episode_id, step, config, tmp / "replay")


def test_start_mismatch_is_retained_as_refusal_without_alternative(saved_episode):
    root, record, step, config, tmp = saved_episode
    path = root / "privileged_ledgers" / f"{record.episode_id}.json"
    payload = json.loads(path.read_text())
    at = record.steps[step].sim_time_s
    sample = min(payload["samples"], key=lambda s: abs(s["sim_time_s"] - at))
    sample["position"]["x"] += 1
    path.write_text(json.dumps(payload))
    result = run_counterfactual(root, record.episode_id, step, config, tmp / "mismatch")
    assert result["status"] == "refused" and "state mismatch" in result["refusal_reason"]
    assert "hold" not in result["branches"]
    assert not result["branches"]["original"]["prefix_trace"][-1]["state_check"]["within_tolerance"]
    assert (tmp / "mismatch" / "counterfactual.json").is_file()


def test_missing_executed_prefix_never_substitutes_proposal(saved_episode):
    root, record, step, config, tmp = saved_episode
    path = root / "episodes" / f"{record.episode_id}.json"
    payload = json.loads(path.read_text())
    payload["steps"][0]["executed_command"] = None
    path.write_text(json.dumps(payload))
    result = run_counterfactual(root, record.episode_id, step, config, tmp / "missing")
    assert result["status"] == "refused" and "executed-command prefix" in result["refusal_reason"]
    assert result["branches"] == {}


def test_cannot_write_derived_result_into_source_run(saved_episode):
    root, record, step, config, _ = saved_episode
    with pytest.raises(ValueError, match="outside the source"):
        run_counterfactual(root, record.episode_id, step, config, root / "counterfactual")


def test_source_endpoint_mismatch_invalidates_contrast(saved_episode):
    root, record, step, config, tmp = saved_episode
    path = root / "privileged_ledgers" / f"{record.episode_id}.json"
    payload = json.loads(path.read_text())
    at = record.steps[step].sim_time_s + record.dt_s
    sample = min(payload["samples"], key=lambda s: abs(s["sim_time_s"] - at))
    sample["velocity"]["x"] += 1
    path.write_text(json.dumps(payload))
    result = run_counterfactual(root, record.episode_id, step, config, tmp / "endpoint-mismatch")
    assert result["status"] == "refused" and "endpoint differs" in result["refusal_reason"]
    assert "hold" not in result["branches"]


def test_cannot_promote_fixture_source_to_experimental_replay(saved_episode):
    root, record, step, config, tmp = saved_episode
    # The malformed source must fail closed even if a user relabels both files consistently.
    for folder in ("episodes", "privileged_ledgers"):
        path = root / folder / f"{record.episode_id}.json"
        payload = json.loads(path.read_text())
        payload["run_class"] = "heldout"
        path.write_text(json.dumps(payload))
    result = run_counterfactual(root, record.episode_id, step, config, tmp / "invalid-provenance")
    assert result["status"] == "refused"
    assert "no anchored live" in result["refusal_reason"]
    assert result["analysis_role"] == "exploration"
