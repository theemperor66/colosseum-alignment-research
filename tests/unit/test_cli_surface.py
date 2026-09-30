"""The documented command surface must exist and expose help without a simulator."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from colosseum_assurance.cli import COMMANDS

REPO = Path(__file__).resolve().parents[2]
EXPECTED = {
    "doctor", "live-gate", "attest", "freeze", "deviate", "size", "smoke", "run",
    "evaluate", "analyze", "audit", "replay", "schemas", "version",
}


def test_all_documented_commands_are_registered():
    assert EXPECTED <= set(COMMANDS)


def test_every_command_has_a_docstring():
    missing = [name for name, fn in COMMANDS.items() if not (fn.__doc__ or "").strip()]
    assert not missing, f"commands without help text: {missing}"


@pytest.mark.parametrize("args", [["--help"], ["version"]])
def test_cli_runs_without_a_simulator(args):
    result = subprocess.run(
        [sys.executable, "-m", "colosseum_assurance.cli", *args],
        capture_output=True, text=True, cwd=REPO, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_refused_experimental_run_explains_itself_and_exits_non_zero(tmp_path):
    """An operator who runs an experiment too early must get a reason, not a stack trace."""
    protocol = tmp_path / "frozen.json"
    protocol.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "colosseum_assurance.cli", "run",
         "--run-class", "pilot", "--protocol", str(protocol)],
        capture_output=True, text=True, cwd=REPO, timeout=120,
        env={**os.environ, "COLASSURE_SIM_HOST": "127.0.0.1", "COLASSURE_SIM_ATTESTATION": ""},
    )
    assert result.returncode == 4, result.stderr
    assert "not allowed" in result.stderr
    assert "simulator_attestation_path" in result.stderr
    assert "Traceback" not in result.stderr


def test_gate_report_filename_is_covered_by_a_test():
    """The CLI writes the gate report under this name; a typo here used to fail only at run time."""
    from colosseum_assurance.sim.diagnostics import GateReport

    report = GateReport(
        created_at="2026-09-17T02:09:17+00:00", endpoint_label="tunnel", endpoint_host="127.0.0.1",
        endpoint_port=41451, protocol_hash="sha256:" + "0" * 64,
    )
    name = report.suggested_filename()
    assert name.startswith("live_gate_") and name.endswith(".json")
    assert ":" not in name and "+" not in name
    assert "20260917T020917" in name


def test_schema_export_writes_every_record_type(tmp_path):
    from colosseum_assurance.workflows.schema_export import MODELS, export_schemas

    result = export_schemas(tmp_path)
    assert result["written"] == len(MODELS)
    for name in MODELS:
        assert (tmp_path / f"{name}.schema.json").exists()
