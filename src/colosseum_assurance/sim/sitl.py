"""Offline preparation for a pinned, single-instance civilian PX4 SITL integration.

No function opens a socket, starts a simulator, or modifies PX4 failsafes. In particular TCP 4560 is
Colosseum's simulator listener, not a readiness endpoint: an ordinary probe can consume its HIL slot.
The source contract is pinned; compatibility remains unverified until the real integration gate runs.
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import shutil
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field, field_validator, model_validator

from colosseum_assurance.schemas import StrictModel

COLOSSEUM_COMMIT = "84fc0c1c75bc73a0135ee80a325d470577c66c52"
PX4_COMMIT = "a6274bc5ed01e5c86af79f89890689c239b1d944"
PX4_TAG = "v1.11.3"


class SITLConfig(StrictModel):
    """One PX4 process and one Colosseum vehicle sharing a private loopback network namespace."""

    model_config = ConfigDict(extra="forbid", strict=True)
    vehicle_name: str = "Drone1"
    camera_name: str = "front_center"
    host: str = "127.0.0.1"
    tcp_port: int = Field(default=4560, ge=1024, le=65535)
    control_port_local: int = Field(default=14540, ge=1024, le=65535)
    control_port_remote: int = Field(default=14580, ge=1024, le=65535)
    rpc_port: int = Field(default=41451, ge=1024, le=65535)
    camera_width: int = Field(default=256, ge=16, le=4096)
    camera_height: int = Field(default=144, ge=16, le=4096)
    camera_hfov_deg: float = Field(default=90.0, gt=10.0, lt=180.0)
    # Declared engineering mount; actual PX4 camera/vehicle transforms need live qualification.
    camera_x_m: float = Field(default=0.5, allow_inf_nan=False)
    camera_y_m: float = Field(default=0.0, allow_inf_nan=False)
    camera_z_m: float = Field(default=-0.1, allow_inf_nan=False)
    camera_pitch_deg: float = Field(default=0.0, allow_inf_nan=False)
    camera_roll_deg: float = Field(default=0.0, allow_inf_nan=False)
    camera_yaw_deg: float = Field(default=0.0, allow_inf_nan=False)

    @field_validator("vehicle_name", "camera_name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                            for c in value):
            raise ValueError("vehicle and camera names must contain only letters, digits, '_' or '-'")
        return value

    @field_validator("host")
    @classmethod
    def _loopback(cls, value: str) -> str:
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise ValueError("use an explicit loopback IPv4 address") from exc
        if address.version != 4 or not address.is_loopback:
            raise ValueError("this SITL profile only supports loopback IPv4 on one host")
        return value

    @model_validator(mode="after")
    def _ports(self) -> SITLConfig:
        if self.tcp_port == self.rpc_port:
            raise ValueError("PX4 HIL and simulator RPC TCP listeners require different ports")
        if self.control_port_local == self.control_port_remote:
            raise ValueError("the local and remote command UDP endpoints require different ports")
        return self


def generate_px4_settings(config: SITLConfig | None = None) -> dict[str, Any]:
    """Generate the bounded study profile; never insert failsafe-disabling PX4 parameters."""
    c = config or SITLConfig()
    camera = {"X": c.camera_x_m, "Y": c.camera_y_m, "Z": c.camera_z_m,
              "Pitch": c.camera_pitch_deg, "Roll": c.camera_roll_deg, "Yaw": c.camera_yaw_deg,
              "CaptureSettings": [
        {"ImageType": kind, "Width": c.camera_width, "Height": c.camera_height,
         "FOV_Degrees": c.camera_hfov_deg, "MotionBlurAmount": 0}
        for kind in (0, 2, 5)
    ]}
    return {
        "SettingsVersion": 1.2, "SimMode": "Multirotor", "ClockType": "SteppableClock",
        "ClockSpeed": 1.0, "LocalHostIp": c.host, "ApiServerPort": c.rpc_port,
        "ViewMode": "NoDisplay", "Vehicles": {c.vehicle_name: {
            "VehicleType": "PX4Multirotor", "UseSerial": False, "UseTcp": True,
            "LockStep": True, "TcpPort": c.tcp_port, "ControlIp": c.host,
            "ControlPortLocal": c.control_port_local, "ControlPortRemote": c.control_port_remote,
            "LocalHostIp": c.host, "EnableCollisions": True,
            "Cameras": {c.camera_name: camera},
        }},
    }


def validate_px4_settings(settings: Mapping[str, Any]) -> SITLConfig:
    """Validate the exact supported profile, refusing unknown fields and silent coerced values.

    This deliberately does not validate arbitrary PX4/AirSim settings. Extensions need a separately
    reviewed contract; accepting a configuration containing unexamined ``Parameters`` would silently
    permit upstream example failsafe overrides that this study never authorized.
    """
    try:
        vehicles = settings["Vehicles"]
        if not isinstance(vehicles, Mapping) or len(vehicles) != 1:
            raise ValueError("exactly one PX4 SITL vehicle is supported")
        name, vehicle = next(iter(vehicles.items()))
        cameras = vehicle["Cameras"]
        if not isinstance(cameras, Mapping) or len(cameras) != 1:
            raise ValueError("exactly one study camera is supported")
        camera_name, camera = next(iter(cameras.items()))
        captures = camera["CaptureSettings"]
        if not isinstance(captures, list) or len(captures) != 3:
            raise ValueError("RGB, DepthPerspective and segmentation capture settings are required")
        image = captures[0]
        config = SITLConfig(
            vehicle_name=name, camera_name=camera_name, host=settings["LocalHostIp"],
            rpc_port=settings["ApiServerPort"], tcp_port=vehicle["TcpPort"],
            control_port_local=vehicle["ControlPortLocal"],
            control_port_remote=vehicle["ControlPortRemote"],
            camera_width=image["Width"], camera_height=image["Height"],
            camera_hfov_deg=image["FOV_Degrees"],
            camera_x_m=camera["X"], camera_y_m=camera["Y"], camera_z_m=camera["Z"],
            camera_pitch_deg=camera["Pitch"], camera_roll_deg=camera["Roll"],
            camera_yaw_deg=camera["Yaw"],
        )
        expected = generate_px4_settings(config)
        # JSON-shaped equality alone treats True == 1. This recursive check preserves exact scalar
        # kinds while allowing JSON integer numeric FOV/clock fields where floats were generated.
        _assert_profile(settings, expected, "settings")
        return config
    except (KeyError, TypeError, IndexError, AttributeError, StopIteration) as exc:
        raise ValueError(f"malformed PX4 SITL settings: {exc}") from exc


def _assert_profile(actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"{path}: missing or unsupported fields")
        for key, value in expected.items():
            _assert_profile(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{path}: required capture settings differ")
        for index, value in enumerate(expected):
            _assert_profile(actual[index], value, f"{path}[{index}]")
    else:
        same_type = (type(actual) is type(expected) or
                     (isinstance(expected, float) and type(actual) in {int, float}))
        if not same_type or actual != expected:
            raise ValueError(f"{path}: expected {expected!r}, got {actual!r}")


def pinned_source_manifest() -> dict[str, Any]:
    """Versioned primary-source contract, not evidence that binaries were built or connected."""
    colosseum = f"https://github.com/CodexLabsLLC/Colosseum/blob/{COLOSSEUM_COMMIT}"
    px4 = f"https://github.com/PX4/PX4-Autopilot/blob/{PX4_COMMIT}"
    return {
        "schema_version": "px4-sitl-preparation-v1", "live_verified": False,
        "compatibility_status": "unverified_source_contract_only",
        "colosseum": {"repository": "https://github.com/CodexLabsLLC/Colosseum",
                      "commit": COLOSSEUM_COMMIT},
        "px4": {"repository": "https://github.com/PX4/PX4-Autopilot",
                "tag": PX4_TAG, "commit": PX4_COMMIT,
                "qualification": "Historical reference pin, not a current release recommendation."},
        "sources": [f"{colosseum}/AirLib/include/common/AirSimSettings.hpp",
                    f"{colosseum}/docs/px4_lockstep.md",
                    f"{colosseum}/AirLib/include/vehicles/multirotor/firmwares/mavlink/MavLinkMultirotorApi.hpp",
                    f"{px4}/src/modules/simulator/simulator_mavlink.cpp",
                    f"{px4}/ROMFS/px4fmu_common/init.d-posix/rcS"],
        "network": {
            "hil": "Colosseum listens on TCP4560; PX4 connects. Never probe that listener.",
            "commands": "Separate UDP command channel: Colosseum local14540, PX4 remote14580.",
            "scope": "One instance, same loopback network namespace; no public listeners.",
        },
        "required_live_checks": ["PX4 simulator handshake", "GPS fix and home establishment",
                                 "lockstep advancement", "commanded motion and reset",
                                 "RGB/depth/segmentation and study geometry qualification"],
    }


def _git_read(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                                timeout=3, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _binary_evidence(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"path": str(path) if path else None, "present": False, "executable": False}
    result: dict[str, Any] = {"path": str(path.resolve()), "present": True,
                              "executable": os.access(path, os.X_OK), "size_bytes": path.stat().st_size}
    if result["size_bytes"] > 1024**3:
        result["hash_unavailable_reason"] = "file exceeds bounded 1 GiB hashing budget"
        return result
    digest, deadline = hashlib.sha256(), time.monotonic() + 3
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
                if time.monotonic() > deadline:
                    result["hash_unavailable_reason"] = "hash exceeded bounded 3 s budget"
                    return result
    except OSError as exc:
        result["hash_unavailable_reason"] = type(exc).__name__
        return result
    result["sha256"] = digest.hexdigest()
    return result


def check_prerequisites(
    settings: Mapping[str, Any], *, px4_source: Path | None = None,
    colosseum_binary: Path | None = None,
) -> dict[str, Any]:
    """Bounded read-only local checks. No port probe, executable launch, build, or download."""
    config = validate_px4_settings(settings)
    source = Path(px4_source) if px4_source is not None else None
    commit = _git_read(source, "rev-parse", "HEAD") if source is not None else None
    dirty = _git_read(source, "status", "--porcelain", "--untracked-files=no") if source else None
    px4_binary = source / "build/px4_sitl_default/bin/px4" if source is not None else None
    binaries = {"px4": _binary_evidence(px4_binary),
                "colosseum": _binary_evidence(Path(colosseum_binary) if colosseum_binary else None)}
    missing = []
    if commit != PX4_COMMIT:
        missing.append("PX4 source checkout does not match the reviewed commit")
    if dirty is None or dirty:
        missing.append("PX4 source cleanliness could not be established" if dirty is None else
                       "PX4 source has tracked modifications")
    for name, evidence in binaries.items():
        if not evidence["present"] or not evidence["executable"] or not evidence.get("sha256"):
            missing.append(f"{name} executable and bounded SHA-256 evidence are required")
    return {
        **pinned_source_manifest(), "settings_valid": True,
        "settings": config.model_dump(mode="json"), "px4_observed_commit": commit,
        "px4_tracked_clean": dirty == "" if dirty is not None else None,
        "binaries": binaries, "local_prerequisites_present": not missing,
        "missing_prerequisites": missing,
        "build_tools": {name: shutil.which(name) for name in ("git", "make", "cmake", "ninja", "clang++")},
        "network_probes_performed": False, "processes_launched": False,
        "note": "Binary hashes establish file identity, not that these files were built from the pinned "
                "source. Retain build logs and simulator artifact attestation before any live run.",
    }
