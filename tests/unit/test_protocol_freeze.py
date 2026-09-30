"""Protocol freezing, tamper detection, sizing, and workload estimation."""

from __future__ import annotations

import json

import pytest

from colosseum_assurance.protocol.freeze import (
    ProtocolIntegrityError,
    append_deviation,
    estimate_workload,
    freeze_protocol,
    load_frozen,
    paired_sample_size,
)
from colosseum_assurance.protocol.spec import ProtocolConfig


def test_freeze_and_load_round_trip(tmp_path):
    protocol = ProtocolConfig()
    path = freeze_protocol(protocol, label="pilot", out_dir=tmp_path, rationale="pilot freeze")
    loaded, meta = load_frozen(path)
    assert meta["label"] == "pilot"
    assert loaded.protocol_label == "pilot"
    assert loaded.content_hash() == meta["protocol_hash"]
    assert meta["code_version"]["package_version"]


def test_edited_frozen_file_is_detected(tmp_path):
    path = freeze_protocol(ProtocolConfig(), label="pilot", out_dir=tmp_path)
    payload = json.loads(path.read_text())
    payload["protocol"]["obligations"]["geofence_tolerance_m"] = 99.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ProtocolIntegrityError):
        load_frozen(path)


def test_deviations_are_appended_without_changing_the_protocol(tmp_path):
    path = freeze_protocol(ProtocolConfig(), label="heldout", out_dir=tmp_path)
    before_hash = json.loads(path.read_text())["protocol_hash"]
    append_deviation(path, description="raised RPC timeout", rationale="server latency", applies_from="ep42")
    payload = json.loads(path.read_text())
    assert payload["protocol_hash"] == before_hash
    assert len(payload["deviations"]) == 1
    assert payload["deviations"][0]["description"] == "raised RPC timeout"
    loaded, _ = load_frozen(path)
    assert loaded.content_hash() == before_hash


def test_protocol_hash_changes_when_any_field_changes():
    a = ProtocolConfig()
    b = a.model_copy(update={"obligations": a.obligations.model_copy(update={"geofence_tolerance_m": 0.75})})
    assert a.content_hash() != b.content_hash()


def test_paired_sample_size_grows_when_discordance_grows():
    small = paired_sample_size(1, 2, 40, target_half_width=0.10)
    large = paired_sample_size(8, 12, 40, target_half_width=0.10)
    assert large["required_pairs_total"] > small["required_pairs_total"]
    tighter = paired_sample_size(8, 12, 40, target_half_width=0.05)
    assert tighter["required_pairs_total"] > large["required_pairs_total"]


def test_paired_sample_size_handles_zero_discordance_conservatively():
    result = paired_sample_size(0, 0, 18, target_half_width=0.10)
    assert result["required_pairs_total"] > 0
    assert "no discordant pairs" in result["caveat"]


def test_workload_estimate_uses_measured_episode_cost():
    protocol = ProtocolConfig()
    serial = estimate_workload(protocol, "heldout", measured_episode_wall_clock_s=60.0)
    parallel = estimate_workload(protocol, "heldout", measured_episode_wall_clock_s=60.0,
                                 parallel_capacity=4)
    assert serial["episodes_planned"] == 1080
    assert parallel["estimated_wall_clock_hours"] == pytest.approx(
        serial["estimated_wall_clock_hours"] / 4, rel=1e-6
    )
    with_failures = estimate_workload(protocol, "heldout", 60.0, failure_rate=0.5)
    assert with_failures["episodes_with_reruns"] == pytest.approx(2160.0)
