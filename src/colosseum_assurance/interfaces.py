"""Structural interfaces between the simulator, the controller, the guards, and the runner.

These Protocols are the contract that keeps components substitutable and testable. They also encode the
information separation: a :class:`Controller` only ever receives an
:class:`~colosseum_assurance.schemas.ObservationPacket` plus a :class:`MissionBrief`; privileged truth is
reachable only through :meth:`SimAdapter.sample_truth`, which the runner calls on the ledger path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from colosseum_assurance.protocol.spec import MissionSpec, ObligationSpec
from colosseum_assurance.schemas import (
    ControlCommand,
    FrameRef,
    MonitorReport,
    ObservationPacket,
    SimulatorIdentity,
    TruthSample,
    Vec3,
    VehicleState,
)


class AdapterError(RuntimeError):
    """Any simulator-side failure. Carries a plain-language explanation for the diagnostics command."""

    def __init__(self, message: str, *, remedy: str = "", cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.remedy = remedy
        self.cause = cause


class AdapterTimeout(AdapterError):
    """A simulator call exceeded its timeout."""


@dataclass(slots=True)
class CapturedFrame:
    """A camera frame plus its metadata. ``array`` is None when only metadata was retained."""

    ref: FrameRef
    array: np.ndarray | None = None


@dataclass(slots=True)
class MissionBrief:
    """Non-privileged mission information available to the vehicle and its guards.

    Contains the *nominal* asset position from the protocol, never the per-scenario jittered truth, and
    never the obstacle list. Perception must resolve the rest.
    """

    mission: MissionSpec
    obligations: ObligationSpec
    home: Vec3
    nominal_asset_position: Vec3
    declared_observation_delay_s: float
    declared_supervision_delay_s: float
    policy_version: str
    camera_name: str = "front_center"
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SimAdapter(Protocol):
    """A genuine or fixture simulator, driven with bounded, timed calls.

    Time advancement has exactly one owner. ``issue_*`` methods send a command and return immediately;
    ``step`` is the only call that advances simulated time. The blocking convenience wrappers
    (``hold``, ``takeoff``, ``land``) advance the clock themselves and are for diagnostics and the
    readiness gate, never for a control step inside an episode.
    """

    def connect(self) -> SimulatorIdentity: ...
    def identity(self) -> SimulatorIdentity: ...
    def ping(self) -> bool: ...
    def reset(self) -> None: ...
    def wait_until_ready(self, timeout_s: float | None = None) -> None: ...
    def acquire_control(self) -> None: ...
    def release_control(self) -> None: ...
    def arm(self) -> None: ...
    def disarm(self) -> None: ...
    def takeoff(self, altitude_m: float, timeout_s: float | None = None) -> None: ...
    def issue_hold(self) -> None: ...
    def issue_rotate_to_yaw(
        self, yaw_rad: float, timeout_s: float | None = None, margin_deg: float = 5.0
    ) -> None: ...
    def issue_takeoff(self, timeout_s: float | None = None) -> None: ...
    def issue_land(self, timeout_s: float | None = None) -> None: ...
    def move_to(
        self, target: Vec3, speed_mps: float, duration_s: float, yaw_rad: float | None = None
    ) -> None: ...
    def hold(self, duration_s: float) -> None: ...
    def land(self, timeout_s: float | None = None) -> None: ...
    def step(self, dt_s: float) -> float: ...
    def sim_time_s(self) -> float: ...
    def sample_state(self) -> VehicleState: ...
    def sample_truth(self) -> TruthSample: ...
    def capture(self, kinds: tuple[str, ...] = ("rgb", "depth"), save_prefix: str | None = None
                ) -> dict[str, CapturedFrame]: ...
    def set_start_pose(self, position: Vec3, yaw_rad: float) -> None: ...
    def list_scene_objects(self, name_regex: str = ".*") -> list[str]: ...
    def close(self) -> None: ...


@runtime_checkable
class Controller(Protocol):
    """Fixed goal-directed controller. Identical across arms; sees only delayed observations."""

    controller_id: str

    def reset(self, brief: MissionBrief) -> None: ...
    def step(self, observation: ObservationPacket) -> ControlCommand: ...
    def internal_state(self) -> dict[str, Any]: ...


@runtime_checkable
class Monitor(Protocol):
    """A runtime guard. Produces a verdict and may request a bounded intervention."""

    monitor_id: str

    def reset(self, brief: MissionBrief) -> None: ...
    def evaluate(self, observation: ObservationPacket, command: ControlCommand) -> MonitorReport: ...
    def describe(self) -> dict[str, Any]: ...
