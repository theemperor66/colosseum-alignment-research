"""Runtime configuration: simulator endpoint, paths, run class, and validation.

Configuration comes from a YAML file and/or ``COLASSURE_*`` environment variables. No credentials are
ever stored here: remote access is expected to arrive through an SSH tunnel or authorized private
networking, so the client normally talks to a loopback port (see docs/deployment.md).
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from colosseum_assurance.schemas import Box, StrictModel

ENV_PREFIX = "COLASSURE_"
DEFAULT_API_PORT = 41451  # Colosseum/AirSim default RPC port (see docs/upstream-colosseum-facts.md)
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "host.docker.internal"}


class EndpointConfig(StrictModel):
    """Where the simulator RPC server is reachable, and how patiently to talk to it."""

    host: str = "127.0.0.1"
    port: int = Field(default=DEFAULT_API_PORT, ge=1, le=65535)
    label: str = Field(default="local-tunnel", description="Human label recorded in simulator identity.")
    connect_timeout_s: float = Field(default=10.0, gt=0.0)
    rpc_timeout_s: float = Field(default=20.0, gt=0.0)
    reset_timeout_s: float = Field(default=60.0, gt=0.0)
    allow_direct_remote: bool = Field(
        default=False,
        description=(
            "Guard rail. A non-loopback host is refused unless this is explicitly enabled, because the "
            "Colosseum RPC API is unauthenticated and must not be reached over a public path."
        ),
    )
    vehicle_name: str = "Drone1"

    @field_validator("host")
    @classmethod
    def _host_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("endpoint host must not be empty")
        return v.strip()

    @property
    def is_loopback(self) -> bool:
        host = self.host.lower()
        if host in LOOPBACK_HOSTS:
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    @property
    def is_private_network(self) -> bool:
        try:
            return ipaddress.ip_address(self.host).is_private
        except ValueError:
            return False

    @model_validator(mode="after")
    def _refuse_public_endpoint(self) -> EndpointConfig:
        if self.is_loopback or self.allow_direct_remote:
            return self
        if self.is_private_network:
            raise ValueError(
                f"endpoint host {self.host!r} is a private-network address. The Colosseum RPC API is "
                "unauthenticated. Forward it over SSH "
                "(ssh -N -L 41451:127.0.0.1:41451 user@host) and connect to 127.0.0.1, or set "
                "allow_direct_remote=true only on an authorized private network."
            )
        raise ValueError(
            f"endpoint host {self.host!r} looks public. Refusing: the simulator RPC API is "
            "unauthenticated and must stay private. Use an SSH tunnel to 127.0.0.1."
        )

    @property
    def description(self) -> str:
        return f"{self.label} ({self.host}:{self.port})"


class PathsConfig(StrictModel):
    """Filesystem layout for evidence. Run classes are stored in separate trees on purpose."""

    results_root: Path = Path("results")
    frames_subdir: str = "frames"
    protocol_dir: Path = Path("configs/frozen")
    figures_dir: Path = Path("docs/figures")

    def run_dir(self, run_class: str, protocol_short_hash: str) -> Path:
        return self.results_root / run_class / protocol_short_hash

    def episodes_dir(self, run_class: str, protocol_short_hash: str) -> Path:
        return self.run_dir(run_class, protocol_short_hash) / "episodes"

    def ledgers_dir(self, run_class: str, protocol_short_hash: str) -> Path:
        return self.run_dir(run_class, protocol_short_hash) / "privileged_ledgers"

    def frames_dir(self, run_class: str, protocol_short_hash: str) -> Path:
        return self.run_dir(run_class, protocol_short_hash) / self.frames_subdir

    def attempted_runs_path(self, run_class: str, protocol_short_hash: str) -> Path:
        return self.run_dir(run_class, protocol_short_hash) / "attempted_runs.jsonl"

    def analysis_dir(self, run_class: str, protocol_short_hash: str) -> Path:
        return self.run_dir(run_class, protocol_short_hash) / "analysis"


class SceneLaunchPlatform(StrictModel):
    """Explicit authored support geometry, separate from the immutable historical protocol."""

    bounds: Box
    name: str = Field(default="launch_ground_plane", pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    spawn_clearance_m: float = Field(default=0.2, gt=0.0, le=2.0)

    @model_validator(mode="after")
    def _measurable_support(self) -> SceneLaunchPlatform:
        if "ground_plane" not in self.name.lower():
            raise ValueError("launch support name must contain ground_plane for its explicit contact role")
        if not all(math.isfinite(value) for value in self.bounds.model_dump().values()):
            raise ValueError("launch platform bounds must be finite")
        return self


class AppConfig(StrictModel):
    """Complete runtime configuration for a command invocation."""

    endpoint: EndpointConfig = Field(default_factory=EndpointConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    run_class: Literal["fixture", "pilot", "heldout", "smoke"] = "fixture"
    protocol_path: Path | None = Field(
        default=None, description="Frozen protocol JSON. Required for pilot and held-out runs."
    )
    simulator_attestation_path: Path | None = Field(
        default=None,
        description=(
            "Artifact attestation JSON that anchors simulator provenance (package name, SHA-256, origin, "
            "verifier). Required for pilot and held-out runs: an RPC handshake alone does not establish "
            "which simulator answered."
        ),
    )
    require_live_simulator: bool = Field(
        default=False,
        description=(
            "True for any run that may back experimental claims. The runner then refuses to start unless "
            "simulator identity proves a live Colosseum server."
        ),
    )
    allow_fixture_fake: bool = Field(
        default=True, description="Only fixture and smoke run classes may use the fake server."
    )
    scene_mode: Literal["verify_only", "instantiate", "qualified_map"] = Field(
        default="verify_only",
        description=(
            "How the study geometry reaches the live level. verify_only: the map already contains the "
            "study actors and their transforms are checked. instantiate: spawn them through verified "
            "RPC. qualified_map: measure a third-party map's own bodies and bind the manifest to them, "
            "which changes the scenario definition and is recorded with the episode. "
            "See docs/scene-integration.md."
        ),
    )
    scene_geometry_contract_path: Path | None = Field(
        default=None,
        description=(
            "Reviewed geometry contract binding scene actors to a known source mesh and base "
            "dimensions. A live scene whose extents are not covered by it cannot pass the scene gate, "
            "because an unmeasured extent makes the evaluator's clearance assumption unfounded. "
            "See docs/scene-integration.md."
        ),
    )
    scene_asset_actor_name: str | None = Field(
        default=None, min_length=1,
        description=(
            "Exact measured inspection-asset actor for qualified_map. The live binding must find "
            "this actor among the measured geometry and use its position and identity for the mission "
            "and evaluator-only segmentation; a synthetic target coordinate is not sufficient."
        ),
    )
    scene_launch_platform: SceneLaunchPlatform | None = None
    scene_inventory_max_probes: int = Field(default=256, ge=1, le=10000)
    scene_inventory_wall_budget_s: float = Field(default=120.0, gt=0.0, le=3600.0)
    camera_max_attitude_pairing_gap_s: float = Field(default=0.05, ge=0.0, le=1.0)
    scene_inventory_exclusions: dict[str, str] = Field(
        default_factory=dict,
        description="Exact nongeometry actor names mapped to their reviewed exclusion rationale.")
    max_episode_wall_clock_s: float = Field(default=900.0, gt=0.0)
    log_level: str = "INFO"

    @field_validator("scene_inventory_exclusions")
    @classmethod
    def _explicit_exclusion_reasons(cls, values: dict[str, str]) -> dict[str, str]:
        if any(not name.strip() or not reason.strip() for name, reason in values.items()):
            raise ValueError("every exact actor exclusion requires a nonempty name and review reason")
        return values

    @model_validator(mode="after")
    def _run_class_consistency(self) -> AppConfig:
        """Experimental run classes must name a live simulator and a readable frozen protocol.

        Without the protocol check, a held-out run could validate with ``protocol_path=None`` and silently
        fall back to draft defaults, which would destroy the separation between development and
        confirmatory evidence.
        """
        if self.scene_launch_platform is not None and self.scene_mode != "instantiate":
            raise ValueError("scene_launch_platform requires instantiate mode and verified authored geometry")
        if self.run_class in {"pilot", "heldout"}:
            if self.allow_fixture_fake:
                raise ValueError(
                    f"run_class={self.run_class!r} must set allow_fixture_fake=false: experimental runs "
                    "may never use the fixture fake simulator."
                )
            if not self.require_live_simulator:
                raise ValueError(
                    f"run_class={self.run_class!r} must set require_live_simulator=true so a missing live "
                    "simulator fails loudly instead of producing unusable evidence."
                )
            if self.protocol_path is None:
                raise ValueError(
                    f"run_class={self.run_class!r} requires protocol_path: an experimental run must load a "
                    "frozen protocol artifact (see `colassure freeze`), never draft defaults."
                )
            if not Path(self.protocol_path).is_file():
                raise ValueError(
                    f"protocol_path {self.protocol_path} does not exist or is not a file; refusing to run "
                    f"a {self.run_class!r} experiment without its frozen protocol."
                )
            if self.simulator_attestation_path is None:
                raise ValueError(
                    f"run_class={self.run_class!r} requires simulator_attestation_path: experimental "
                    "evidence must name the hash-verified simulator package that produced it. Create one "
                    "with `colassure attest`."
                )
            if not Path(self.simulator_attestation_path).is_file():
                raise ValueError(
                    f"simulator_attestation_path {self.simulator_attestation_path} does not exist or is "
                    "not a file; refusing to record experimental evidence from an unanchored simulator."
                )
        return self

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
        """Load YAML (if given), then environment variables, then explicit overrides."""
        data: dict[str, Any] = {}
        if path is not None:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"config file not found: {p}")
            loaded = yaml.safe_load(p.read_text()) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"config file {p} must contain a YAML mapping")
            data = loaded
        env = cls._env_overrides()
        data = _deep_merge(data, env)
        if overrides:
            data = _deep_merge(data, overrides)
        return cls.model_validate(data)

    @staticmethod
    def _env_overrides() -> dict[str, Any]:
        """Map COLASSURE_* variables onto the config tree. Unknown variables are ignored."""
        mapping = {
            "SIM_HOST": ("endpoint", "host", str),
            "SIM_PORT": ("endpoint", "port", int),
            "SIM_LABEL": ("endpoint", "label", str),
            "SIM_CONNECT_TIMEOUT_S": ("endpoint", "connect_timeout_s", float),
            "SIM_RPC_TIMEOUT_S": ("endpoint", "rpc_timeout_s", float),
            "SIM_RESET_TIMEOUT_S": ("endpoint", "reset_timeout_s", float),
            "SIM_ALLOW_DIRECT_REMOTE": ("endpoint", "allow_direct_remote", _as_bool),
            "SIM_VEHICLE": ("endpoint", "vehicle_name", str),
            "RESULTS_ROOT": ("paths", "results_root", Path),
            "RUN_CLASS": (None, "run_class", str),
            "PROTOCOL_PATH": (None, "protocol_path", Path),
            "SIM_ATTESTATION": (None, "simulator_attestation_path", Path),
            "REQUIRE_LIVE": (None, "require_live_simulator", _as_bool),
            "ALLOW_FIXTURE_FAKE": (None, "allow_fixture_fake", _as_bool),
            "SCENE_MODE": (None, "scene_mode", str),
            "SCENE_GEOMETRY_CONTRACT": (None, "scene_geometry_contract_path", Path),
            "SCENE_ASSET_ACTOR_NAME": (None, "scene_asset_actor_name", str),
            "SCENE_INVENTORY_MAX_PROBES": (None, "scene_inventory_max_probes", int),
            "SCENE_INVENTORY_WALL_BUDGET_S": (None, "scene_inventory_wall_budget_s", float),
            "LOG_LEVEL": (None, "log_level", str),
        }
        out: dict[str, Any] = {}
        for suffix, (section, key, caster) in mapping.items():
            raw = os.environ.get(ENV_PREFIX + suffix)
            if raw is None or raw == "":
                continue
            value = caster(raw)
            if section is None:
                out[key] = value
            else:
                out.setdefault(section, {})[key] = value
        return out

    def scene_configuration_evidence(self) -> dict[str, Any]:
        """Hash the deployment's geometry declarations without rewriting historical protocol hashes.

        Preserve this alongside the frozen protocol before held-out runs. The same data and digest
        travel with every scene binding; a digest records configuration, not measurement approval.
        """
        path = self.scene_geometry_contract_path
        declaration = {
            "schema_version": "1.0",
            "scene_mode": self.scene_mode,
            "scene_asset_actor_name": self.scene_asset_actor_name,
            "scene_launch_platform": (self.scene_launch_platform.model_dump(mode="json")
                                      if self.scene_launch_platform else None),
            "scene_inventory_max_probes": self.scene_inventory_max_probes,
            "scene_inventory_wall_budget_s": self.scene_inventory_wall_budget_s,
            "camera_max_attitude_pairing_gap_s": self.camera_max_attitude_pairing_gap_s,
            "scene_inventory_exclusions": self.scene_inventory_exclusions,
            "scene_geometry_contract_path": str(path) if path else None,
            "scene_geometry_contract_sha256": (
                "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
                if path is not None and path.is_file() else None),
        }
        digest = hashlib.sha256(json.dumps(declaration, sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest()
        return {"configuration": declaration, "sha256": "sha256:" + digest,
                "review_status": "configuration_declaration_not_measurement_approval"}

    def describe(self) -> str:
        return (
            f"endpoint={self.endpoint.description} run_class={self.run_class} "
            f"require_live={self.require_live_simulator} allow_fake={self.allow_fixture_fake} "
            f"results_root={self.paths.results_root}"
        )


def _as_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"cannot read {raw!r} as a boolean")


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
