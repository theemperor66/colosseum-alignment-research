"""Simulator identity and the provenance rule that separates a live server from our test fixture.

The honesty problem this module solves: every artifact of this study records *which simulator produced
it*. A record that claims ``live_colosseum`` must be impossible to produce from the software test
fixture, even by accident, because a mislabelled fixture run would be fabricated evidence.

Two facts make the naive approach useless (see docs/upstream-colosseum-facts.md):

* ``getServerVersion`` is hardcoded ``return 1`` in ``AirLib/src/api/RpcLibServerBase.cpp:95`` at the
  pinned commit. Any fake can return 1. Versions are recorded as facts, never as provenance.
* A fake that mimics the RPC surface is indistinguishable from the real server on the happy path.

So provenance is decided by an *inverted* test. Our fixture fake answers the private method
``__colassure_fixture_fake__`` with ``True``. A genuine Colosseum has no such binding and answers with
an RPC error. Provenance is ``fixture_fake`` whenever the probe answers affirmatively, and
``airsim_compatible_unverified`` only when the probe *fails* with a server error **and** the basic
handshake worked. Anything else (transport failure, a server that answers an unknown private method
without erroring) is ``unverified``. The burden of proof sits on the live claim.

A second, independent review made the remaining gap explicit: a successful handshake plus a rejected
probe proves only that *something* speaks the AirSim/Colosseum RPC surface. Microsoft AirSim, or any
unrelated server, passes that test. Provenance that may back experimental claims therefore requires an
anchored :class:`~colosseum_assurance.schemas.SimulatorArtifactAttestation`: a named package, its
SHA-256, its origin, and who verified it. With no attestation the identity stays
``airsim_compatible_unverified`` and the live gate fails. With one, the class comes from the attestation
(``third_party_colosseum_build`` for a qualified third-party package such as the ESAR publication, whose
upstream Colosseum commit stays unknown, or ``colosseum_build_verified`` for a pinned upstream build).
When the attestation declares ``expected_scene_signature`` the live scene listing must match it, which is
a real cross-check between the declared package and the answering server.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from colosseum_assurance.rpc.msgpack_rpc import MsgpackRpcClient, RpcError, RpcTransportError
from colosseum_assurance.schemas import SimulatorArtifactAttestation, SimulatorIdentity

FIXTURE_FAKE_PROBE_METHOD = "__colassure_fixture_fake__"
"""Private RPC method implemented only by :mod:`colosseum_assurance.sim.fixture_fake`."""

SETTINGS_KEYS_RECORDED = (
    "SettingsVersion",
    "SimMode",
    "ClockType",
    "ClockSpeed",
    "ViewMode",
    "ApiServerPort",
    "LocalHostIp",
    "PhysicsEngineName",
)
"""Settings fields worth recording. Paths and any other free text are deliberately not copied."""

Provenance = Literal[
    "colosseum_build_verified",
    "third_party_colosseum_build",
    "airsim_compatible_unverified",
    "fixture_fake",
    "unverified",
]


@dataclass(slots=True)
class FixtureProbeResult:
    """Outcome of the ``__colassure_fixture_fake__`` probe."""

    answered_affirmative: bool = False
    rejected_with_rpc_error: bool = False
    answered_other: bool = False
    transport_failed: bool = False
    detail: str = ""

    @property
    def is_conclusive(self) -> bool:
        return self.answered_affirmative or self.rejected_with_rpc_error


def probe_fixture_fake(client: MsgpackRpcClient, *, timeout_s: float = 5.0) -> FixtureProbeResult:
    """Ask the server whether it is our fixture fake.

    An RPC error is the *expected* answer from a genuine Colosseum, so it is not a failure here.
    """
    try:
        answer = client.call(FIXTURE_FAKE_PROBE_METHOD, timeout_s=timeout_s)
    except RpcError as exc:
        return FixtureProbeResult(
            rejected_with_rpc_error=True,
            detail=f"server rejected {FIXTURE_FAKE_PROBE_METHOD!r}: {exc.text[:200]}",
        )
    except RpcTransportError as exc:
        return FixtureProbeResult(transport_failed=True, detail=f"probe transport failure: {exc}")
    if answer is True or answer == 1:
        return FixtureProbeResult(
            answered_affirmative=True,
            detail="server identifies itself as the local fixture fake simulator",
        )
    return FixtureProbeResult(
        answered_other=True,
        detail=(
            f"server answered the private probe with {answer!r} instead of erroring; a genuine "
            "Colosseum has no such binding, so provenance cannot be called live"
        ),
    )


def decide_provenance(
    *,
    ping_ok: bool,
    probe: FixtureProbeResult,
    server_version: int | None,
) -> tuple[Provenance, str]:
    """Pure decision function for provenance. Kept separate so it can be tested without a socket."""
    if probe.answered_affirmative:
        return "fixture_fake", probe.detail
    if probe.transport_failed:
        return "unverified", f"fixture-fake probe could not complete: {probe.detail}"
    if probe.answered_other:
        return "unverified", probe.detail
    if not ping_ok:
        return "unverified", "server did not answer ping with True"
    if server_version is None:
        return "unverified", "server did not report getServerVersion"
    return "airsim_compatible_unverified", (
        "handshake succeeded and the server rejected the private fixture-fake probe "
        f"({probe.detail}); this proves an AirSim/Colosseum-compatible RPC surface only, so an artifact "
        "attestation is required before any experimental claim"
    )


def scene_signature(object_names: Sequence[str]) -> str:
    """Stable hash of the scene object listing.

    Order from ``simListSceneObjects`` is not guaranteed, so names are sorted and de-duplicated first.
    Two episodes with the same signature saw the same set of actors; a changed signature means the
    scene changed under us, which must invalidate comparisons rather than pass unnoticed.
    """
    unique = sorted({str(name) for name in object_names})
    digest = hashlib.sha256("\n".join(unique).encode("utf-8")).hexdigest()
    return "sha256:" + digest


def _settings_facts(settings_text: str) -> tuple[str, dict[str, Any]]:
    """Return (digest, whitelisted fields) for a settings JSON document."""
    digest = "sha256:" + hashlib.sha256(settings_text.encode("utf-8", "surrogateescape")).hexdigest()
    facts: dict[str, Any] = {}
    try:
        parsed = json.loads(settings_text)
    except (ValueError, TypeError):
        return digest, {"parse_error": "settings string is not valid JSON"}
    if not isinstance(parsed, dict):
        return digest, {"parse_error": "settings document is not a JSON object"}
    for key in SETTINGS_KEYS_RECORDED:
        if key in parsed:
            facts[key] = parsed[key]
    vehicles = parsed.get("Vehicles")
    if isinstance(vehicles, dict):
        facts["VehicleNames"] = sorted(vehicles)
    return digest, facts


def load_attestation(path: str | Path) -> SimulatorArtifactAttestation:
    """Read and validate an artifact attestation file.

    A malformed or self-contradictory attestation raises. Silently ignoring it would let an experimental
    run proceed on an unanchored endpoint.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"simulator attestation not found: {p}. Create one with `colassure attest` after the package "
            "hash has been verified on the server."
        )
    return SimulatorArtifactAttestation.model_validate_json(p.read_text(encoding="utf-8"))


def build_identity(
    client: MsgpackRpcClient,
    *,
    endpoint_label: str,
    timeout_s: float = 10.0,
    scene_regex: str = ".*",
    client_version: int = 1,
    min_required_server_version: int = 1,
    attestation: SimulatorArtifactAttestation | None = None,
) -> SimulatorIdentity:
    # The two defaults are the upstream client-side constants, not RPC calls:
    # PythonClient/airsim/client.py:37-38 (getClientVersion -> 1) and client.py:43-44
    # (getMinRequiredServerVersion -> 1) at commit 84fc0c1c75bc73a0135ee80a325d470577c66c52.
    """Run the handshake and build a :class:`SimulatorIdentity` from what the server actually said.

    Fields we cannot obtain stay ``None``. In particular ``engine_version`` is left unset: the RPC API
    at the pinned commit exposes no Unreal Engine version, and inventing one would be fabrication.
    """
    notes: list[str] = []

    ping_ok = False
    try:
        ping_ok = bool(client.call("ping", timeout_s=timeout_s))
    except (RpcError, RpcTransportError) as exc:
        notes.append(f"ping failed: {exc}")

    server_version = _call_int(client, "getServerVersion", timeout_s, notes)
    min_client_version = _call_int(client, "getMinRequiredClientVersion", timeout_s, notes)

    probe = probe_fixture_fake(client, timeout_s=timeout_s)
    provenance, provenance_note = decide_provenance(
        ping_ok=ping_ok, probe=probe, server_version=server_version
    )
    notes.append(provenance_note)

    scene_objects: list[str] = []
    try:
        raw = client.call("simListSceneObjects", scene_regex, timeout_s=timeout_s)
        scene_objects = [str(name) for name in (raw or [])]
    except (RpcError, RpcTransportError) as exc:
        notes.append(f"simListSceneObjects unavailable: {exc}")

    settings_digest: str | None = None
    api_settings: dict[str, Any] = {}
    try:
        settings_text = client.call("getSettingsString", timeout_s=timeout_s)
        if isinstance(settings_text, str) and settings_text.strip():
            settings_digest, api_settings = _settings_facts(settings_text)
    except (RpcError, RpcTransportError) as exc:
        notes.append(f"getSettingsString unavailable: {exc}")

    if server_version is not None and server_version < min_required_server_version:
        notes.append(
            f"server protocol version {server_version} is below the minimum {min_required_server_version} "
            "this client supports; the simulator build is older than the pinned protocol"
        )
    if min_client_version is not None and client_version < min_client_version:
        notes.append(
            f"this client reports version {client_version} but the server requires at least "
            f"{min_client_version}; the recorded data may not mean what the schema says"
        )
    notes.append(
        "engine_version is not recorded: the Colosseum RPC API at commit "
        "84fc0c1c75bc73a0135ee80a325d470577c66c52 exposes no engine version"
    )

    live_scene_signature = scene_signature(scene_objects) if scene_objects else None

    # Anchoring step. Only an attestation can raise a protocol-compatible endpoint to a provenance class
    # that may back experimental claims, and an attestation that declares an expected scene signature must
    # match the scene the server actually reports.
    scene_binding_verified: bool | None = None
    if attestation is not None:
        if provenance == "fixture_fake":
            notes.append(
                "an artifact attestation was supplied but the server identified itself as the fixture "
                "fake; provenance stays fixture_fake"
            )
        elif provenance != "airsim_compatible_unverified":
            notes.append(
                f"an artifact attestation was supplied but the handshake left provenance {provenance!r}; "
                "the attestation was recorded and not applied"
            )
        else:
            expected = attestation.expected_scene_signature
            if expected:
                scene_binding_verified = bool(
                    live_scene_signature is not None and live_scene_signature == expected
                )
                if scene_binding_verified:
                    notes.append("live scene signature matches the attestation binding")
                else:
                    notes.append(
                        "live scene signature "
                        f"{live_scene_signature or 'unavailable'} does not match the attested "
                        f"{expected}; provenance stays airsim_compatible_unverified"
                    )
            if expected is None or scene_binding_verified:
                provenance = attestation.provenance_class
                notes.append(
                    f"provenance anchored to artifact {attestation.artifact_name} "
                    f"({attestation.artifact_sha256}) attested by {attestation.verified_by} on "
                    f"{attestation.verified_utc}"
                )
                for caveat in attestation.caveats:
                    notes.append(f"caveat: {caveat}")

    simulator_name = {
        "fixture_fake": "colassure_fixture_fake",
        "unverified": None,
    }.get(provenance, "colosseum_rpc_compatible")
    if provenance in {"third_party_colosseum_build", "colosseum_build_verified"}:
        simulator_name = "colosseum"

    return SimulatorIdentity(
        provenance=provenance,
        endpoint_label=endpoint_label,
        server_version=server_version,
        min_required_client_version=min_client_version,
        client_version=client_version,
        min_required_server_version=min_required_server_version,
        simulator_name=simulator_name,
        engine_version=(attestation.engine_version_declared + " (declared, not measured)")
        if (attestation is not None and attestation.engine_version_declared) else None,
        scene_name=None,
        scene_object_count=len(scene_objects) if scene_objects else None,
        scene_signature=live_scene_signature,
        settings_digest=settings_digest,
        api_settings=api_settings,
        artifact=attestation if provenance in {"third_party_colosseum_build", "colosseum_build_verified"}
        else None,
        scene_binding_verified=scene_binding_verified,
        observed_at_wall_clock=datetime.now(UTC).isoformat(timespec="seconds"),
        notes="; ".join(notes),
    )


def _call_int(client: MsgpackRpcClient, method: str, timeout_s: float, notes: list[str]) -> int | None:
    try:
        value = client.call(method, timeout_s=timeout_s)
    except (RpcError, RpcTransportError) as exc:
        notes.append(f"{method} failed: {exc}")
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        notes.append(f"{method} returned {value!r}, which is not an integer")
        return None
