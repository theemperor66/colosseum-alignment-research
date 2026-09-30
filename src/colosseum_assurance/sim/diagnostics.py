"""One command that explains what is wrong: configuration, connection, reset and camera readiness.

Two entry points:

* :func:`run_diagnostics` answers "can this machine talk to a simulator, and which part is broken?".
  Every check reports a status, a measured detail and a plain remedy. Unreachable parts are skipped
  with a reason; nothing is invented.
* :func:`run_live_readiness_gate` answers the stricter question "may this endpoint back experimental
  claims?". It passes only with (a) an anchored Colosseum artifact attestation, (b) a successful reset, (c) a
  commanded motion that measurably moved the vehicle across at least three samples, and (d) a nonempty
  RGB frame and a nonempty depth frame. The fixture fake fails it at (a) by construction, and
  ``tests/unit/test_colosseum_adapter.py`` pins that.

A specific distinction this module makes on purpose, because it cost real debugging time: an SSH local
forward with no simulator behind it *accepts* the TCP connection and then resets it. That looks nothing
like "connection refused", and the two need different fixes.
"""

from __future__ import annotations

import json
import math
import socket
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from colosseum_assurance.config import AppConfig, EndpointConfig
from colosseum_assurance.interfaces import AdapterError, AdapterTimeout
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.schemas import SimulatorIdentity, StrictModel, Vec3
from colosseum_assurance.sim.colosseum_adapter import ColosseumAdapter
from colosseum_assurance.sim.identity import load_attestation

CheckStatus = Literal["ok", "fail", "skip"]

GATE_MIN_SAMPLES = 3
GATE_MIN_DISPLACEMENT_M = 0.5
FIXTURE_GATE_REASON = "fixture fake, not a live Colosseum"


def _configured_adapter(config: AppConfig, protocol: ProtocolConfig) -> ColosseumAdapter:
    """Preserve operator configuration while letting diagnostics report provenance failures.

    Unlike ``build_adapter``, this does not connect or reject an endpoint before the diagnostic
    checks can explain it. It must nevertheless use the same attestation and scene configuration.
    """
    attestation = (load_attestation(config.simulator_attestation_path)
                   if config.simulator_attestation_path is not None else None)
    return ColosseumAdapter(
        config.endpoint, protocol, attestation=attestation, scene_mode=config.scene_mode,
        geometry_contract_path=config.scene_geometry_contract_path,
        scene_asset_actor_name=config.scene_asset_actor_name,
    )


class CheckResult(StrictModel):
    """One diagnostic check. ``detail`` carries measurements, ``remedy`` carries the next action."""

    id: str
    status: CheckStatus
    detail: str = ""
    remedy: str = ""
    duration_s: float = 0.0


class DiagnosticReport(StrictModel):
    """Full doctor output for one endpoint."""

    created_at: str
    endpoint_label: str
    endpoint_host: str
    endpoint_port: int
    protocol_hash: str
    run_class: str
    provenance: str = "unverified"
    identity: SimulatorIdentity | None = None
    checks: list[CheckResult] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    @property
    def counts(self) -> dict[str, int]:
        out = {"ok": 0, "fail": 0, "skip": 0}
        for check in self.checks:
            out[check.status] += 1
        return out

    def first_failure(self) -> CheckResult | None:
        for check in self.checks:
            if check.status == "fail":
                return check
        return None

    def render_text(self) -> str:
        lines = [
            f"simulator doctor: {self.endpoint_label} ({self.endpoint_host}:{self.endpoint_port})",
            f"  protocol {self.protocol_hash[:19]}...  run_class={self.run_class}  "
            f"provenance={self.provenance}",
            "",
        ]
        symbol = {"ok": "PASS", "fail": "FAIL", "skip": "SKIP"}
        for check in self.checks:
            lines.append(f"  [{symbol[check.status]}] {check.id}  ({check.duration_s:.2f} s)")
            if check.detail:
                lines.append(f"         {check.detail}")
            if check.status != "ok" and check.remedy:
                lines.append(f"         remedy: {check.remedy}")
        counts = self.counts
        lines.append("")
        lines.append(f"  {counts['ok']} ok, {counts['fail']} failed, {counts['skip']} skipped")
        return "\n".join(lines)

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True))
        return target


class GateReport(StrictModel):
    """Live-readiness gate result. ``evidence`` holds numbers, never adjectives."""

    created_at: str
    endpoint_label: str
    endpoint_host: str
    endpoint_port: int
    protocol_hash: str
    passed: bool = False
    reason: str = ""
    provenance: str = "unverified"
    identity: SimulatorIdentity | None = None
    checks: list[CheckResult] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)

    def suggested_filename(self) -> str:
        """Stable file name for a saved gate report, derived from its own timestamp.

        The CLI used to build this string from a field name that did not exist, which only failed at the
        moment a report was written. Keeping it on the model lets a unit test cover it.
        """
        stamp = "".join(ch for ch in self.created_at if ch.isalnum())
        return f"live_gate_{stamp}.json"

    def render_text(self) -> str:
        head = "LIVE READINESS GATE: " + ("PASSED" if self.passed else "NOT PASSED")
        lines = [head, f"  endpoint {self.endpoint_label} ({self.endpoint_host}:{self.endpoint_port})",
                 f"  provenance {self.provenance}", f"  reason: {self.reason}", ""]
        for check in self.checks:
            lines.append(f"  [{check.status.upper():4}] {check.id}: {check.detail}")
        if self.evidence:
            lines.append("")
            lines.append("  evidence:")
            for key, value in sorted(self.evidence.items()):
                lines.append(f"    {key} = {value}")
        return "\n".join(lines)

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True))
        return target


# ---------------------------------------------------------------------------- TCP probing
class TcpProbe(StrictModel):
    """Outcome of a bare TCP connect, before any RPC is attempted."""

    status: Literal["accepted", "refused", "timeout", "unresolved", "error"]
    detail: str
    remedy: str
    duration_s: float


def probe_tcp(endpoint: EndpointConfig) -> TcpProbe:
    """Try a bare TCP connect and classify the failure precisely."""
    started = time.monotonic()
    tunnel_hint = (
        "Open the tunnel first: "
        f"ssh -N -L {endpoint.port}:127.0.0.1:{endpoint.port} user@server, then retry."
    )
    try:
        with socket.create_connection((endpoint.host, endpoint.port),
                                      timeout=endpoint.connect_timeout_s):
            pass
    except ConnectionRefusedError:
        return TcpProbe(
            status="refused",
            detail=f"nothing is listening on {endpoint.host}:{endpoint.port} (connection refused)",
            remedy=("No listener at all: neither an SSH forward nor a local simulator. " + tunnel_hint),
            duration_s=time.monotonic() - started,
        )
    except socket.gaierror as exc:
        return TcpProbe(
            status="unresolved", detail=f"host {endpoint.host!r} does not resolve: {exc}",
            remedy="Fix the hostname in the configuration or use 127.0.0.1 with an SSH tunnel.",
            duration_s=time.monotonic() - started,
        )
    except TimeoutError:
        return TcpProbe(
            status="timeout",
            detail=(f"connect to {endpoint.host}:{endpoint.port} timed out after "
                    f"{endpoint.connect_timeout_s:g} s"),
            remedy="The address is filtered or unreachable. Check the network path and the firewall.",
            duration_s=time.monotonic() - started,
        )
    except OSError as exc:
        return TcpProbe(status="error", detail=f"connect failed: {exc}",
                        remedy="Check the endpoint configuration and the local network stack.",
                        duration_s=time.monotonic() - started)
    return TcpProbe(status="accepted", detail=f"{endpoint.host}:{endpoint.port} accepted a TCP connection",
                    remedy="", duration_s=time.monotonic() - started)


def classify_rpc_failure(endpoint: EndpointConfig, error: BaseException) -> tuple[str, str]:
    """Explain an RPC failure that happened *after* TCP accepted the connection.

    The interesting case is an SSH local forward with no simulator behind it: the forward accepts
    locally and then resets or closes the stream. That is a completely different fix from "refused".
    """
    text = str(error)
    reset_markers = ("reset by peer", "closed the connection", "Broken pipe", "ECONNRESET")
    if any(marker.lower() in text.lower() for marker in reset_markers):
        return (
            f"TCP accepted but the RPC stream was reset ({text}). The port is open locally, so a "
            "tunnel or proxy is listening, but no simulator answered behind it.",
            (
                "The SSH forward is up and the remote side has no simulator on "
                f"127.0.0.1:{endpoint.port}. Start the Colosseum server on the remote host, then "
                "re-run the doctor. Do not record episodes until this passes."
            ),
        )
    if isinstance(error, AdapterTimeout) or "within" in text:
        return (
            f"TCP accepted but no RPC answer arrived in time ({text}).",
            "The process behind the port is not a Colosseum RPC server, or it is blocked. Check the "
            "simulator log and whether the level is still loading.",
        )
    return (f"RPC call failed after a successful TCP connect: {text}",
            "Check that the listening process really is a Colosseum simulator.")


# ---------------------------------------------------------------------------- doctor
class _Doctor:
    """Runs the checks in order and stops calling the simulator once the connection is known bad."""

    def __init__(self, config: AppConfig, protocol: ProtocolConfig,
                 adapter_factory: Callable[[], ColosseumAdapter] | None = None) -> None:
        self.config = config
        self.protocol = protocol
        self.endpoint = config.endpoint
        self.adapter_factory = adapter_factory or (lambda: _configured_adapter(config, protocol))
        self.adapter: ColosseumAdapter | None = None
        self.report = DiagnosticReport(
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            endpoint_label=self.endpoint.label,
            endpoint_host=self.endpoint.host,
            endpoint_port=self.endpoint.port,
            protocol_hash=protocol.content_hash(),
            run_class=config.run_class,
        )

    def add(self, check_id: str, status: CheckStatus, detail: str = "", remedy: str = "",
            duration_s: float = 0.0) -> CheckResult:
        result = CheckResult(id=check_id, status=status, detail=detail, remedy=remedy,
                             duration_s=duration_s)
        self.report.checks.append(result)
        return result

    def run(self) -> DiagnosticReport:
        self.check_configuration()
        probe = self.check_tcp()
        if probe.status != "accepted":
            for check_id in ("rpc_ping", "version_handshake", "fixture_fake_probe", "reset",
                             "api_control", "state_sample", "rgb_capture", "depth_capture",
                             "scene_objects", "stepping_mode"):
                self.add(check_id, "skip", f"skipped: TCP probe said {probe.status}", probe.remedy)
            return self.report
        if not self.check_connect():
            for check_id in ("version_handshake", "fixture_fake_probe", "reset", "api_control",
                             "state_sample", "rgb_capture", "depth_capture", "scene_objects",
                             "stepping_mode"):
                self.add(check_id, "skip", "skipped: no usable RPC session")
            return self.report
        try:
            self.check_versions()
            self.check_fixture_probe()
            if self.check_reset():
                self.check_api_control()
                self.check_state_sample()
                self.check_capture()
            else:
                for check_id in ("api_control", "state_sample", "rgb_capture", "depth_capture"):
                    self.add(check_id, "skip", "skipped: reset did not complete")
            self.check_scene_objects()
            self.check_stepping()
        finally:
            if self.adapter is not None:
                self.adapter.close()
        return self.report

    # -- individual checks ---------------------------------------------
    def check_configuration(self) -> None:
        started = time.monotonic()
        notes = [f"endpoint {self.endpoint.description}",
                 f"loopback={self.endpoint.is_loopback}",
                 f"run_class={self.config.run_class}",
                 f"require_live={self.config.require_live_simulator}",
                 f"rpc_timeout={self.endpoint.rpc_timeout_s:g}s"]
        if not self.endpoint.is_loopback and not self.endpoint.allow_direct_remote:
            self.add("configuration", "fail", "; ".join(notes),
                     "A non-loopback endpoint needs allow_direct_remote and an authorized private path.",
                     time.monotonic() - started)
            return
        self.add("configuration", "ok", "; ".join(notes), "", time.monotonic() - started)

    def check_tcp(self) -> TcpProbe:
        probe = probe_tcp(self.endpoint)
        self.add("tcp_reachable", "ok" if probe.status == "accepted" else "fail",
                 probe.detail, probe.remedy, probe.duration_s)
        return probe

    def check_connect(self) -> bool:
        started = time.monotonic()
        adapter = self.adapter_factory()
        try:
            identity = adapter.connect()
        except AdapterError as exc:
            detail, remedy = classify_rpc_failure(self.endpoint, exc)
            self.add("rpc_ping", "fail", detail, remedy or exc.remedy, time.monotonic() - started)
            adapter.close()
            return False
        self.adapter = adapter
        self.report.identity = identity
        self.report.provenance = identity.provenance
        try:
            ping_ok = adapter.ping()
        except AdapterError as exc:
            detail, remedy = classify_rpc_failure(self.endpoint, exc)
            self.add("rpc_ping", "fail", detail, remedy, time.monotonic() - started)
            return False
        if not ping_ok:
            self.add("rpc_ping", "fail", "the server answered ping with something other than True",
                     "The listening process is not a Colosseum RPC server.", time.monotonic() - started)
            return False
        self.add("rpc_ping", "ok", f"ping answered True in {time.monotonic() - started:.3f} s", "",
                 time.monotonic() - started)
        return True

    def check_versions(self) -> None:
        started = time.monotonic()
        identity = self.report.identity
        assert identity is not None
        if identity.server_version is None:
            self.add("version_handshake", "fail", "getServerVersion did not return an integer",
                     "The server did not complete the version handshake.", time.monotonic() - started)
            return
        detail = (f"server_version={identity.server_version} "
                  f"min_required_client_version={identity.min_required_client_version} "
                  f"(upstream hardcodes 1; this identifies the protocol, not the build)")
        self.add("version_handshake", "ok", detail, "", time.monotonic() - started)

    def check_fixture_probe(self) -> None:
        started = time.monotonic()
        identity = self.report.identity
        assert identity is not None
        if identity.provenance == "fixture_fake":
            self.add("fixture_fake_probe", "ok",
                     "the server identified itself as the local fixture fake; any data from it is a "
                     "software test double, never experimental evidence",
                     "Use a genuine Colosseum endpoint for pilot or held-out runs.",
                     time.monotonic() - started)
            return
        if identity.provenance == "airsim_compatible_unverified":
            self.add("fixture_fake_probe", "ok",
                     "the server rejected the private fixture probe, as a genuine Colosseum does; note "
                     "that this proves an AirSim/Colosseum-compatible RPC surface only",
                     "Supply an artifact attestation (`colassure attest`) to anchor provenance.",
                     time.monotonic() - started)
            return
        if identity.is_live:
            self.add("fixture_fake_probe", "ok",
                     f"provenance anchored: {identity.qualification_note()}",
                     "", time.monotonic() - started)
            return
        self.add("fixture_fake_probe", "fail",
                 f"provenance could not be established: {identity.notes[:300]}",
                 "Provenance must be established before any run that could back a claim.",
                 time.monotonic() - started)

    def check_reset(self) -> bool:
        started = time.monotonic()
        adapter = self.adapter
        assert adapter is not None
        try:
            adapter.reset()
            adapter.wait_until_ready(timeout_s=min(self.endpoint.reset_timeout_s, 30.0))
        except AdapterError as exc:
            self.add("reset", "fail", f"reset or readiness failed: {exc}", exc.remedy,
                     time.monotonic() - started)
            return False
        self.add("reset", "ok", f"reset and readiness completed in {time.monotonic() - started:.2f} s",
                 "", time.monotonic() - started)
        return True

    def check_api_control(self) -> None:
        started = time.monotonic()
        adapter = self.adapter
        assert adapter is not None
        try:
            adapter.acquire_control()
            adapter.release_control()
        except AdapterError as exc:
            self.add("api_control", "fail", f"acquire/release failed: {exc}", exc.remedy,
                     time.monotonic() - started)
            return
        self.add("api_control", "ok", "enableApiControl + armDisarm acquired and released, verified "
                 "with isApiControlEnabled", "", time.monotonic() - started)

    def check_state_sample(self) -> None:
        started = time.monotonic()
        adapter = self.adapter
        assert adapter is not None
        try:
            state = adapter.sample_state()
        except AdapterError as exc:
            self.add("state_sample", "fail", f"getMultirotorState failed: {exc}", exc.remedy,
                     time.monotonic() - started)
            return
        self.add("state_sample", "ok",
                 f"position=({state.position.x:.2f}, {state.position.y:.2f}, {state.position.z:.2f}) m "
                 f"NED, landed={state.landed}, sim_time={state.sim_time_s:.2f} s",
                 "", time.monotonic() - started)

    def check_capture(self) -> None:
        adapter = self.adapter
        assert adapter is not None
        for kind, check_id in (("rgb", "rgb_capture"), ("depth", "depth_capture")):
            started = time.monotonic()
            try:
                frames = adapter.capture((kind,))
            except AdapterError as exc:
                self.add(check_id, "fail", f"{kind} capture failed: {exc}", exc.remedy,
                         time.monotonic() - started)
                continue
            frame = frames.get(kind)
            if frame is None or not frame.ref.is_nonempty:
                detail = "no frame returned" if frame is None else (
                    f"frame {frame.ref.width}x{frame.ref.height} is empty "
                    f"(nonzero_fraction={frame.ref.nonzero_fraction}, max={frame.ref.max_value})")
                self.add(check_id, "fail", detail,
                         "The camera produced no pixels. Check the render device: a headless server "
                         "without a GPU or without an offscreen renderer returns empty frames.",
                         time.monotonic() - started)
                continue
            ref = frame.ref
            self.add(check_id, "ok",
                     f"{ref.width}x{ref.height} {kind} frame, min={ref.min_value:.3f} "
                     f"max={ref.max_value:.3f} mean={ref.mean_value:.3f} "
                     f"nonzero_fraction={ref.nonzero_fraction:.3f}", "", time.monotonic() - started)

    def check_scene_objects(self) -> None:
        started = time.monotonic()
        adapter = self.adapter
        assert adapter is not None
        try:
            objects = adapter.list_scene_objects(".*")
        except AdapterError as exc:
            self.add("scene_objects", "fail", f"simListSceneObjects failed: {exc}", exc.remedy,
                     time.monotonic() - started)
            return
        if not objects:
            self.add("scene_objects", "fail", "the level reported zero scene objects",
                     "The level may not be loaded yet, or the vehicle is in an empty map.",
                     time.monotonic() - started)
            return
        sample = ", ".join(sorted(objects)[:5])
        self.add("scene_objects", "ok", f"{len(objects)} scene objects (first few: {sample})", "",
                 time.monotonic() - started)

    def check_stepping(self) -> None:
        started = time.monotonic()
        adapter = self.adapter
        assert adapter is not None
        configured = self.protocol.simulation.stepping_mode
        try:
            before = adapter.sim_time_s()
            adapter.step(self.protocol.mission.control_dt_s)
            after = adapter.sim_time_s()
        except AdapterError as exc:
            self.add("stepping_mode", "fail", f"stepping failed: {exc}", exc.remedy,
                     time.monotonic() - started)
            return
        report = adapter.stepping_report()
        detail = (f"configured={configured} used={report['mode_used']} "
                  f"clock advanced {after - before:.3f} s")
        if report["mode_used"] != configured:
            self.add("stepping_mode", "fail",
                     detail + f"; fallback reason: {report['fallback_reason']}",
                     "Paused stepping is required for reproducible episodes. Check that the "
                     "simulator accepts simPause and simContinueForTime.",
                     time.monotonic() - started)
            return
        self.add("stepping_mode", "ok", detail, "", time.monotonic() - started)


def run_diagnostics(config: AppConfig, protocol: ProtocolConfig, *,
                    adapter_factory: Callable[[], ColosseumAdapter] | None = None
                    ) -> DiagnosticReport:
    """Diagnose configuration, connection, reset, control, sampling, cameras and stepping."""
    return _Doctor(config, protocol, adapter_factory).run()


# ---------------------------------------------------------------------------- live gate
def run_live_readiness_gate(config: AppConfig, protocol: ProtocolConfig, *,
                            adapter_factory: Callable[[], ColosseumAdapter] | None = None,
                            min_displacement_m: float = GATE_MIN_DISPLACEMENT_M,
                            min_samples: int = GATE_MIN_SAMPLES) -> GateReport:
    """Decide whether this endpoint may back experimental claims. Returns structured evidence."""
    endpoint = config.endpoint
    report = GateReport(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        endpoint_label=endpoint.label,
        endpoint_host=endpoint.host,
        endpoint_port=endpoint.port,
        protocol_hash=protocol.content_hash(),
    )

    def add(check_id: str, status: CheckStatus, detail: str, remedy: str = "",
            duration_s: float = 0.0) -> None:
        report.checks.append(CheckResult(id=check_id, status=status, detail=detail, remedy=remedy,
                                         duration_s=duration_s))

    factory = adapter_factory or (lambda: _configured_adapter(config, protocol))
    adapter = factory()
    started = time.monotonic()
    try:
        identity = adapter.connect()
    except AdapterError as exc:
        detail, remedy = classify_rpc_failure(endpoint, exc)
        add("connect", "fail", detail, remedy, time.monotonic() - started)
        report.reason = f"no simulator connection: {detail}"
        adapter.close()
        return report

    report.identity = identity
    report.provenance = identity.provenance
    add("connect", "ok", f"connected to {endpoint.description}", "", time.monotonic() - started)

    try:
        # (a) provenance -------------------------------------------------
        if not identity.is_live:
            if identity.provenance == "fixture_fake":
                reason = f"provenance is {identity.provenance!r}: {FIXTURE_GATE_REASON}"
            elif identity.provenance == "airsim_compatible_unverified":
                reason = (
                    "the endpoint speaks the AirSim/Colosseum RPC surface but no artifact attestation "
                    "anchors it; a protocol handshake does not identify which simulator answered"
                )
            else:
                reason = f"provenance is {identity.provenance!r} and could not be established"
            add("provenance", "fail", reason,
                "Anchor the running package with `colassure attest` (package name, SHA-256, origin, "
                "verifier), then re-run the gate. Only an anchored simulator may produce evidence.")
            report.reason = reason
            report.evidence["provenance"] = identity.provenance
            return report
        add("provenance", "ok", identity.qualification_note())
        report.evidence["provenance_qualification"] = identity.qualification_note()
        if identity.artifact is not None:
            report.evidence["artifact_sha256"] = identity.artifact.artifact_sha256
            report.evidence["artifact_name"] = identity.artifact.artifact_name
            report.evidence["artifact_caveats"] = list(identity.artifact.caveats)
            report.evidence["scene_binding_verified"] = identity.scene_binding_verified

        # (b) reset ------------------------------------------------------
        step_started = time.monotonic()
        try:
            adapter.reset()
            adapter.wait_until_ready(timeout_s=min(endpoint.reset_timeout_s, 45.0))
            adapter.acquire_control()
        except AdapterError as exc:
            add("reset", "fail", f"reset/readiness/control failed: {exc}", exc.remedy,
                time.monotonic() - step_started)
            report.reason = f"reset did not succeed: {exc}"
            return report
        add("reset", "ok", "reset, readiness and API control completed",
            "", time.monotonic() - step_started)

        # (c) commanded motion -------------------------------------------
        step_started = time.monotonic()
        dt = protocol.mission.control_dt_s
        try:
            adapter.takeoff(min(protocol.mission.cruise_altitude_m, 3.0), timeout_s=30.0)
            origin = adapter.sample_state().position
            target = Vec3(x=origin.x + 4.0, y=origin.y, z=origin.z)
            positions = [origin]
            for index in range(max(min_samples, 6)):
                if index % 2 == 0:
                    adapter.move_to(target, protocol.mission.cruise_speed_mps,
                                    protocol.simulation.max_command_duration_s)
                adapter.step(dt)
                positions.append(adapter.sample_state().position)
        except AdapterError as exc:
            add("trajectory", "fail", f"commanded motion failed: {exc}", exc.remedy,
                time.monotonic() - step_started)
            report.reason = f"commanded motion failed: {exc}"
            return report
        displacement = max(math.dist(origin.as_tuple(), p.as_tuple()) for p in positions)
        report.evidence.update({
            "trajectory_samples": len(positions),
            "max_displacement_m": round(displacement, 4),
            "min_displacement_required_m": min_displacement_m,
            "sampled_positions_ned": [[round(v, 3) for v in p.as_tuple()] for p in positions],
        })
        if len(positions) < min_samples or displacement < min_displacement_m:
            reason = (f"commanded motion moved the vehicle {displacement:.3f} m across "
                      f"{len(positions)} samples; at least {min_displacement_m:g} m over "
                      f"{min_samples} samples is required")
            add("trajectory", "fail", reason,
                "The vehicle did not move. Check API control, arming and the flight controller.",
                time.monotonic() - step_started)
            report.reason = reason
            return report
        add("trajectory", "ok",
            f"{len(positions)} samples, max displacement {displacement:.3f} m", "",
            time.monotonic() - step_started)

        # (d) frames ------------------------------------------------------
        step_started = time.monotonic()
        try:
            frames = adapter.capture(("rgb", "depth"))
        except AdapterError as exc:
            add("frames", "fail", f"capture failed: {exc}", exc.remedy,
                time.monotonic() - step_started)
            report.reason = f"camera capture failed: {exc}"
            return report
        for kind in ("rgb", "depth"):
            frame = frames.get(kind)
            if frame is None:
                report.evidence[f"{kind}_frame"] = "missing"
                continue
            ref = frame.ref
            report.evidence[f"{kind}_frame"] = {
                "width": ref.width, "height": ref.height,
                "min": ref.min_value, "max": ref.max_value, "mean": ref.mean_value,
                "nonzero_fraction": ref.nonzero_fraction, "is_nonempty": ref.is_nonempty,
            }
        empty = [kind for kind in ("rgb", "depth")
                 if frames.get(kind) is None or not frames[kind].ref.is_nonempty]
        if empty:
            reason = f"these frames were empty: {empty}"
            add("frames", "fail", reason,
                "The renderer produced no pixels. A headless server needs an offscreen renderer.",
                time.monotonic() - step_started)
            report.reason = reason
            return report
        add("frames", "ok", "RGB and depth frames both carry pixel content", "",
            time.monotonic() - step_started)

        report.passed = True
        report.reason = (
            "live Colosseum provenance, successful reset, measurably changed trajectory "
            f"({displacement:.2f} m over {len(positions)} samples), and nonempty RGB and depth frames"
        )
        return report
    finally:
        try:
            adapter.release_control()
        except AdapterError:
            pass
        adapter.close()
