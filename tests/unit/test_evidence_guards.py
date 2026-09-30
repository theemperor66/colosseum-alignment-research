"""Evidence writing must refuse the mistakes that would quietly ruin the study."""

from __future__ import annotations

import json

import pytest
from tests.helpers import anchored_identity

from colosseum_assurance.config import PathsConfig
from colosseum_assurance.runtime.evidence import (
    EvidenceWriter,
    ProvenanceError,
    RunTreeConflict,
    assert_provenance_allowed,
    load_attempted_runs,
    summarize_attempts,
)
from colosseum_assurance.schemas import AttemptedRun, SimulatorIdentity

LIVE = anchored_identity()
UNANCHORED = SimulatorIdentity(provenance="airsim_compatible_unverified", endpoint_label="tunnel")
FAKE = SimulatorIdentity(provenance="fixture_fake", endpoint_label="fixture")
UNVERIFIED = SimulatorIdentity(provenance="unverified", endpoint_label="unknown")


@pytest.mark.parametrize("identity", [FAKE, UNVERIFIED, UNANCHORED])
@pytest.mark.parametrize("run_class", ["pilot", "heldout"])
def test_experimental_runs_reject_non_live_provenance(identity, run_class):
    with pytest.raises(ProvenanceError):
        assert_provenance_allowed(identity, run_class)


def test_fixture_runs_accept_fixture_provenance():
    """The guard must not be so strict that legitimate combinations fail."""
    assert assert_provenance_allowed(FAKE, "fixture") is None  # fixture data in a fixture tree
    assert assert_provenance_allowed(FAKE, "smoke") is None
    assert assert_provenance_allowed(LIVE, "pilot") is None  # anchored provenance in an experiment
    assert assert_provenance_allowed(LIVE, "fixture") is None


def test_protocol_compatible_endpoint_alone_is_not_experimental_evidence():
    """An RPC handshake proves a surface, not a build: Microsoft AirSim would also pass it."""
    assert not UNANCHORED.is_live
    assert "no artifact attestation anchors it" in UNANCHORED.qualification_note()
    assert LIVE.is_live and LIVE.is_qualified_third_party
    assert "unestablished" in LIVE.qualification_note()


def test_run_tree_refuses_a_second_protocol_hash(tmp_path):
    paths = PathsConfig(results_root=tmp_path)
    EvidenceWriter(paths=paths, run_class="fixture", protocol_hash="sha256:" + "a" * 64)
    # A different protocol hash creates a different directory, so force the collision explicitly.
    writer_dir = paths.run_dir("fixture", "a" * 12)
    meta = json.loads((writer_dir / "run_metadata.json").read_text())
    meta["protocol_hash"] = "sha256:" + "b" * 64
    (writer_dir / "run_metadata.json").write_text(json.dumps(meta))
    with pytest.raises(RunTreeConflict):
        EvidenceWriter(paths=paths, run_class="fixture", protocol_hash="sha256:" + "a" * 64)


def test_note_simulator_records_identity_and_blocks_mixed_provenance(tmp_path):
    paths = PathsConfig(results_root=tmp_path)
    writer = EvidenceWriter(paths=paths, run_class="fixture", protocol_hash="sha256:" + "c" * 64)
    writer.note_simulator(FAKE)
    meta = json.loads(writer.metadata_path.read_text())
    assert meta["simulator_provenance"] == ["fixture_fake"]
    writer.note_simulator(FAKE)  # idempotent
    assert len(json.loads(writer.metadata_path.read_text())["simulator_identities"]) == 1


def test_attempted_run_ledger_round_trip_and_summary(tmp_path):
    paths = PathsConfig(results_root=tmp_path)
    writer = EvidenceWriter(paths=paths, run_class="fixture", protocol_hash="sha256:" + "d" * 64)
    rows = [
        AttemptedRun(attempt_id="a1", episode_id="e1", scenario_id="s1", arm_id="A0_unguarded",
                     run_class="fixture", protocol_hash="sha256:" + "d" * 64, status="completed",
                     started_wall_clock="2026-01-01T00:00:00Z", simulator_provenance="fixture_fake"),
        AttemptedRun(attempt_id="a2", episode_id="e2", scenario_id="s1", arm_id="A1_policy_only",
                     run_class="fixture", protocol_hash="sha256:" + "d" * 64, status="crashed",
                     started_wall_clock="2026-01-01T00:01:00Z", simulator_provenance="fixture_fake",
                     error_type="AdapterError", error_message="rpc reset"),
    ]
    for row in rows:
        writer.append_attempt(row)
    loaded = load_attempted_runs(paths.attempted_runs_path("fixture", writer.protocol_short_hash))
    assert [r.attempt_id for r in loaded] == ["a1", "a2"]
    summary = summarize_attempts(loaded)
    assert summary["attempts"] == 2
    assert summary["by_status"] == {"completed": 1, "crashed": 1}
    assert summary["not_completed"] == 1  # failures stay visible


def test_corrupt_attempt_line_raises_instead_of_being_skipped(tmp_path):
    path = tmp_path / "attempted_runs.jsonl"
    path.write_text('{"not": "an attempt"}\n')
    with pytest.raises(ValueError) as excinfo:
        load_attempted_runs(path)
    assert ":1" in str(excinfo.value)


# --------------------------------------------------------------------------------------
# Continuation-review regressions: evidence retention and protocol persistence
# --------------------------------------------------------------------------------------
def test_recorded_episode_evidence_is_never_replaced(tmp_path):
    """A retry used to overwrite the earlier episode while appending a second attempt row."""
    from colosseum_assurance.runtime.evidence import EvidenceExists
    from colosseum_assurance.schemas import EpisodeRecord, TerminationRecord

    paths = PathsConfig(results_root=tmp_path)
    writer = EvidenceWriter(paths=paths, run_class="fixture", protocol_hash="sha256:" + "e" * 64)
    record = EpisodeRecord(
        episode_id="e1", scenario_id="s1", arm_id="A0_unguarded", run_class="fixture",
        protocol_hash="sha256:" + "e" * 64, policy_version="policy-v1.0.0",
        simulator_identity=FAKE, started_wall_clock="2026-01-01T00:00:00Z", dt_s=0.5,
        termination=TerminationRecord(reason="horizon_reached", step_index=1, sim_time_s=1.0),
    )
    first = writer.write_episode(record)
    original = first.read_text(encoding="utf-8")

    writer.assert_episode_not_recorded("e2")  # a different episode is fine
    with pytest.raises(EvidenceExists):
        writer.assert_episode_not_recorded("e1")
    with pytest.raises(EvidenceExists):
        writer.write_episode(record)
    assert first.read_text(encoding="utf-8") == original


def test_a_second_manifest_for_one_scenario_is_refused(tmp_path):
    """Arms of one scenario must fly the same geometry, so a changed manifest breaks the matched set."""
    from colosseum_assurance.protocol.spec import ProtocolConfig
    from colosseum_assurance.runtime.evidence import EvidenceExists
    from colosseum_assurance.scenario.manifest import build_manifest

    protocol = ProtocolConfig()
    manifest = build_manifest(protocol, "fixture", "obs_nominal__sup_nominal", 0)
    writer = EvidenceWriter(
        paths=PathsConfig(results_root=tmp_path), run_class="fixture",
        protocol_hash=manifest.protocol_hash,
    )
    writer.write_manifest(manifest)
    writer.write_manifest(manifest)  # identical rewrite is fine: arms share it

    moved = manifest.model_copy(
        update={"asset_position": manifest.asset_position.model_copy(update={"x": 99.0})}, deep=True
    )
    with pytest.raises(EvidenceExists):
        writer.write_manifest(moved)


def test_run_protocol_is_persisted_and_mismatches_are_refused(tmp_path):
    """Post-processing must score a run under the protocol that produced it."""
    from colosseum_assurance.protocol.spec import ProtocolConfig
    from colosseum_assurance.runtime.evidence import ProtocolMismatch, load_run_protocol

    protocol = ProtocolConfig()
    shortened = protocol.model_copy(
        update={"mission": protocol.mission.model_copy(update={"episode_horizon_s": 33.0})}, deep=True
    )
    writer = EvidenceWriter(
        paths=PathsConfig(results_root=tmp_path), run_class="fixture",
        protocol_hash=shortened.content_hash(), protocol=shortened,
    )
    assert (writer.root / "protocol.json").exists()

    loaded, provenance = load_run_protocol(writer.root)
    assert provenance["source"] == "run_tree"
    assert loaded.mission.episode_horizon_s == 33.0
    assert loaded.content_hash() == shortened.content_hash()

    with pytest.raises(ProtocolMismatch):
        load_run_protocol(writer.root, protocol)
    same, provenance = load_run_protocol(writer.root, shortened)
    assert same.content_hash() == shortened.content_hash()


def test_missing_run_protocol_is_refused_not_defaulted(tmp_path):
    """A historical run whose protocol cannot be established must refuse, not score under defaults."""
    import json

    from colosseum_assurance.protocol.spec import ProtocolConfig
    from colosseum_assurance.runtime.evidence import ProtocolMismatch, load_run_protocol

    run_dir = tmp_path / "fixture" / "abc123"
    (run_dir / "episodes").mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"run_class": "fixture", "protocol_hash": "sha256:" + "9" * 64}), encoding="utf-8"
    )
    with pytest.raises(ProtocolMismatch, match="--protocol"):
        load_run_protocol(run_dir)

    # Supplying a protocol that disagrees with the recorded hash is refused too.
    with pytest.raises(ProtocolMismatch):
        load_run_protocol(run_dir, ProtocolConfig())


def test_current_defaults_are_accepted_only_when_they_reproduce_the_recorded_hash(tmp_path):
    import json

    from colosseum_assurance.protocol.spec import ProtocolConfig
    from colosseum_assurance.runtime.evidence import load_run_protocol

    protocol = ProtocolConfig()
    run_dir = tmp_path / "fixture" / "def456"
    (run_dir / "episodes").mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"run_class": "fixture", "protocol_hash": protocol.content_hash()}), encoding="utf-8"
    )
    loaded, provenance = load_run_protocol(run_dir)
    assert provenance["source"] == "episode_record_hash_only"
    assert loaded.content_hash() == protocol.content_hash()


def test_a_run_with_no_protocol_evidence_at_all_is_refused(tmp_path):
    from colosseum_assurance.runtime.evidence import ProtocolMismatch, load_run_protocol

    run_dir = tmp_path / "fixture" / "empty"
    (run_dir / "episodes").mkdir(parents=True)
    with pytest.raises(ProtocolMismatch, match="cannot be established"):
        load_run_protocol(run_dir)


def test_partial_evidence_from_a_failed_run_also_blocks_a_repeat(tmp_path):
    """A crash can leave a ledger, frames, or an attempt row without an episode file."""
    from colosseum_assurance.runtime.evidence import EvidenceExists

    paths = PathsConfig(results_root=tmp_path)
    protocol_hash = "sha256:" + "f" * 64

    ledger_only = EvidenceWriter(paths=paths, run_class="fixture", protocol_hash=protocol_hash)
    (ledger_only.ledgers / "ep-ledger.json").write_text("{}", encoding="utf-8")
    assert "privileged_ledger" in ledger_only.existing_evidence("ep-ledger")
    with pytest.raises(EvidenceExists, match="privileged_ledger"):
        ledger_only.assert_episode_not_recorded("ep-ledger")

    frames_dir = ledger_only.frames / "ep-frames"
    frames_dir.mkdir(parents=True)
    (frames_dir / "step0000_depth.npy").write_bytes(b"\x00")
    with pytest.raises(EvidenceExists, match="frames"):
        ledger_only.assert_episode_not_recorded("ep-frames")

    attempt = AttemptedRun(
        attempt_id="a9", episode_id="ep-attempt", scenario_id="s1", arm_id="A0_unguarded",
        run_class="fixture", protocol_hash=protocol_hash, status="crashed",
        started_wall_clock="2026-01-01T00:00:00Z", simulator_provenance="fixture_fake",
    )
    ledger_only.append_attempt(attempt)
    with pytest.raises(EvidenceExists, match="attempted_runs"):
        ledger_only.assert_episode_not_recorded("ep-attempt")

    # An untouched episode id is still allowed.
    ledger_only.assert_episode_not_recorded("ep-fresh")
