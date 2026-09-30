"""Configuration guard rails: private endpoints, run-class safety, environment overrides."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from colosseum_assurance.config import AppConfig, EndpointConfig


def test_loopback_endpoint_is_accepted():
    cfg = EndpointConfig(host="127.0.0.1", port=41451)
    assert cfg.is_loopback
    assert "41451" in cfg.description


@pytest.mark.parametrize("host", ["10.1.2.3", "192.168.1.9", "203.0.113.5", "sim.example.com"])
def test_non_loopback_endpoint_is_refused_without_explicit_opt_in(host):
    """The Colosseum RPC API is unauthenticated, so a remote host must be a deliberate choice."""
    with pytest.raises(ValidationError) as excinfo:
        EndpointConfig(host=host)
    assert "ssh" in str(excinfo.value).lower() or "private" in str(excinfo.value).lower()


def test_private_endpoint_allowed_with_explicit_flag():
    cfg = EndpointConfig(host="10.1.2.3", allow_direct_remote=True)
    assert not cfg.is_loopback


@pytest.mark.parametrize("run_class", ["pilot", "heldout"])
def test_experimental_run_classes_refuse_fixture_fake(run_class, tmp_path):
    frozen = tmp_path / "protocol-pilot-abc.json"
    frozen.write_text("{}")
    attestation = tmp_path / "attestation.json"
    attestation.write_text("{}")
    common = dict(protocol_path=frozen, simulator_attestation_path=attestation)
    with pytest.raises(ValidationError):
        AppConfig(run_class=run_class, allow_fixture_fake=True, require_live_simulator=True, **common)
    with pytest.raises(ValidationError):
        AppConfig(run_class=run_class, allow_fixture_fake=False, require_live_simulator=False, **common)
    cfg = AppConfig(run_class=run_class, allow_fixture_fake=False, require_live_simulator=True, **common)
    assert cfg.require_live_simulator


@pytest.mark.parametrize("run_class", ["pilot", "heldout"])
def test_experimental_run_classes_require_an_existing_frozen_protocol(run_class, tmp_path):
    """A held-out run must never fall back to draft protocol defaults."""
    with pytest.raises(ValidationError, match="requires protocol_path"):
        AppConfig(run_class=run_class, allow_fixture_fake=False, require_live_simulator=True)
    with pytest.raises(ValidationError, match="does not exist"):
        AppConfig(run_class=run_class, allow_fixture_fake=False, require_live_simulator=True,
                  protocol_path=tmp_path / "absent.json")


@pytest.mark.parametrize("run_class", ["pilot", "heldout"])
def test_experimental_run_classes_require_a_simulator_attestation(run_class, tmp_path):
    """An RPC handshake identifies a protocol surface, not which simulator answered."""
    frozen = tmp_path / "protocol.json"
    frozen.write_text("{}")
    with pytest.raises(ValidationError, match="requires simulator_attestation_path"):
        AppConfig(run_class=run_class, allow_fixture_fake=False, require_live_simulator=True,
                  protocol_path=frozen)
    with pytest.raises(ValidationError, match="does not exist"):
        AppConfig(run_class=run_class, allow_fixture_fake=False, require_live_simulator=True,
                  protocol_path=frozen, simulator_attestation_path=tmp_path / "absent.json")


def test_environment_overrides_are_applied(monkeypatch):
    monkeypatch.setenv("COLASSURE_SIM_PORT", "41999")
    monkeypatch.setenv("COLASSURE_SIM_LABEL", "tunnel-under-test")
    monkeypatch.setenv("COLASSURE_LOG_LEVEL", "DEBUG")
    cfg = AppConfig.load()
    assert cfg.endpoint.port == 41999
    assert cfg.endpoint.label == "tunnel-under-test"
    assert cfg.log_level == "DEBUG"


def test_qualified_map_asset_name_has_an_explicit_environment_binding(monkeypatch):
    monkeypatch.setenv("COLASSURE_SCENE_ASSET_ACTOR_NAME", "RealInspectionTower")
    cfg = AppConfig.load()
    assert cfg.scene_asset_actor_name == "RealInspectionTower"
