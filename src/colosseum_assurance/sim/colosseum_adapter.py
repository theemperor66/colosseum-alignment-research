"""``ColosseumAdapter``: the genuine Colosseum client, built on our own bounded msgpack-RPC transport.

Every RPC name and every structure layout below was read from the pinned upstream source at commit
``84fc0c1c75bc73a0135ee80a325d470577c66c52`` of <https://github.com/CodexLabsLLC/Colosseum>, confirmed
twice: once at the Python client call site and once at the C++ server ``bind(...)``. Nothing here is
guessed. A name that cannot be verified raises :class:`NotImplementedError` instead.

Wire name           | Python client (PythonClient/airsim/client.py) | C++ server bind
------------------- | --------------------------------------------- | ---------------------------------
ping                | client.py:35                                  | RpcLibServerBase.cpp:93
getServerVersion    | client.py:41                                  | RpcLibServerBase.cpp:95
getMinRequiredClientVersion | client.py:47                          | RpcLibServerBase.cpp:99
getSettingsString   | client.py:1128                                | RpcLibServerBase.cpp:515
reset               | client.py:26                                  | RpcLibServerBase.cpp:281
enableApiControl    | client.py:58                                  | RpcLibServerBase.cpp:131
isApiControlEnabled | client.py:72                                  | RpcLibServerBase.cpp:135
armDisarm           | client.py:85                                  | RpcLibServerBase.cpp:139
simPause            | client.py:94                                  | RpcLibServerBase.cpp:103
simIsPaused         | client.py:103 (Python method is simIsPause)   | RpcLibServerBase.cpp:107
simContinueForTime  | client.py:112                                 | RpcLibServerBase.cpp:111
simGetImages        | client.py:309                                 | RpcLibServerBase.cpp:147
simGetCollisionInfo | client.py:447                                 | RpcLibServerBase.cpp:358
simSetVehiclePose   | client.py:460                                 | RpcLibServerBase.cpp:248
simGetVehiclePose   | client.py:472                                 | RpcLibServerBase.cpp:252
simListSceneObjects | client.py:556                                 | RpcLibServerBase.cpp:363
simListSceneObjectsByTag | client.py:570                            | RpcLibServerBase.cpp:367
simEnableWeather    | client.py:251                                 | RpcLibServerBase.cpp:123
simSetWeatherParameter | client.py:261                              | RpcLibServerBase.cpp:127
simGetObjectPose    | client.py:498                                 | RpcLibServerBase.cpp:387
simGetGroundTruthKinematics | client.py:820                         | RpcLibServerBase.cpp:453
takeoff             | client.py:1159 (takeoffAsync)                 | MultirotorRpcLibServer.cpp:45
land                | client.py:1172 (landAsync)                    | MultirotorRpcLibServer.cpp:48
hover               | client.py:1291 (hoverAsync)                   | MultirotorRpcLibServer.cpp:111
moveToPosition      | client.py:1254 (moveToPositionAsync)          | MultirotorRpcLibServer.cpp:95
getMultirotorState  | client.py:1594                                | MultirotorRpcLibServer.cpp:140

Structure layout: this fork encodes structures as POSITIONAL ARRAYS. ``MsgpackMixin.to_msgpack``
(types.py:14-27) walks an explicit ``attribute_order`` list, and ``from_msgpack`` (types.py:29-41)
raises ``ValueError`` on a length mismatch; the C++ side uses ``MSGPACK_DEFINE_ARRAY`` throughout
(``AirLib/include/api/RpcLibAdaptorsBase.hpp``, 24 occurrences, zero ``MSGPACK_DEFINE_MAP``). The
orders below are copied from ``types.py`` with their line numbers. The decoder also accepts named maps,
because being robust to a server variant costs one ``isinstance`` check and a wrong guess would cost an
experiment.

Clock discipline (WHY it matters for reproducibility): **exactly one call advances simulated time**.
``move_to`` only *issues* a command and returns at once; ``step(dt)`` advances the clock through
``simContinueForTime`` (or a wall-clock sleep when the protocol says so). The setup and teardown
actions ``wait_until_ready``, ``takeoff``, ``hold`` and ``land`` advance the clock themselves, because
they have no meaning without it, and each one is bounded by an explicit timeout.

VERIFIED STEPPING SEMANTICS (read at the pin, cited, not assumed)
-----------------------------------------------------------------
``simContinueForTime`` is **synchronous for the multirotor path**. The chain is
``RpcLibServerBase.cpp:111-113`` -> ``WorldSimApi::continueForTime``
(``Unreal/Plugins/AirSim/Source/WorldSimApi.cpp:278-281``) -> ``ASimModeWorldBase::continueForTime``
(``Unreal/Plugins/AirSim/Source/SimMode/SimModeWorldBase.cpp:101-118``), which unpauses, calls
``physics_world_->continueForTime(seconds)`` and then BUSY-WAITS twice: first
``while (!physics_world_->isPaused())`` and then until ``UKismetSystemLibrary::GetFrameCount()`` has
moved on, before pausing the game again. So the reply arrives after the world has advanced. The
bounded verification poll in :meth:`step` is therefore **not** a workaround for an upstream defect; it
is a defensive check that the server we are actually talking to behaved that way. It protects against
a non-conforming build, a different SimMode, and our own fixture.

Unsupported stepping variants fail loudly upstream, and must fail loudly here:
``ASimModeBase::continueForTime`` throws ``std::domain_error("continueForTime is not implemented by
SimMode")`` (``SimModeBase.cpp:294-299``) and ``ASimModeBase::continueForFrames`` throws the matching
error (``SimModeBase.cpp:301-306``). Those become RPC errors, which :meth:`_call` converts into
:class:`AdapterError`. This adapter never calls ``simContinueForFrames``: the protocol's stepping grid
is defined in seconds, and a frame count would not be reproducible across builds.

``simPause`` (``RpcLibServerBase.cpp:103-105``) and ``simIsPaused`` (``RpcLibServerBase.cpp:107-109``)
reach ``ASimModeWorldBase::pause``/``isPaused`` (``SimModeWorldBase.cpp:90-99``), which drive the
physics world and the Unreal game pause. :meth:`_try_enter_paused_mode` therefore VERIFIES with
``simIsPaused`` instead of assuming the pause took effect.

SIMULATOR TIMESTAMPS: field, unit, epoch (all verified at the pin)
------------------------------------------------------------------
+--------------------------------+-------+------------------------------------------------------------+
| Field                          | Unit  | Source of the value                                        |
+================================+=======+============================================================+
| ``MultirotorState.timestamp``  | ns    | ``clock()->nowNanos()`` in ``MultirotorApiBase.hpp:134-140``|
+--------------------------------+-------+------------------------------------------------------------+
| ``CollisionInfo.time_stamp``   | ns    | ``ClockFactory::get()->nowNanos()`` at the contact instant, |
|                                |       | ``PawnSimApi.cpp:160``                                     |
+--------------------------------+-------+------------------------------------------------------------+
| ``ImageResponse.time_stamp``   | ns    | ``ClockFactory::get()->nowNanos()`` at read-back,          |
|                                |       | ``RenderRequest.cpp:172``, copied ``UnrealImageCapture.cpp:99``|
+--------------------------------+-------+------------------------------------------------------------+
| ``KinematicsState``            | none  | **NO timestamp field exists**: ``RpcLibAdaptorsBase.hpp:398-409``|
|                                |       | and ``types.py:517`` both carry six vectors and nothing else|
+--------------------------------+-------+------------------------------------------------------------+

Conversion: seconds = nanoseconds * 1e-9, then rebased against the first stamp seen after connect or
reset, so an episode starts at 0 (``docs/timing-semantics.md``). The epoch depends on the configured
clock and is therefore never assumed: ``SteppableClock`` starts at ``Utils::getTimeSinceEpochNanos()``
unless a start is given (``SteppableClock.hpp:24-28``) and only moves when the physics world steps it
(``World.hpp:43-45``); ``ScalableClock`` returns scaled Unix-epoch nanoseconds and its ``step()`` is a
no-op (``ScalableClock.hpp:31-45``, ``ClockBase.hpp:57-65``), which means a ScalableClock server keeps
reporting rising timestamps while the game is paused. The clock in use is selected by ``ClockType`` in
the settings (``SimModeBase.cpp:335-344``), so :meth:`connect` reads ``getSettingsString``
(``client.py:1121-1128``, ``RpcLibServerBase.cpp:515``) and records it in :meth:`timing_report`.

Because ``simGetGroundTruthKinematics`` carries no timestamp, :meth:`sample_truth` pairs it with a real
clock read (``getMultirotorState.timestamp``) taken immediately before it. That is a genuine simulator
timestamp with a measured pairing gap, NOT a locally accumulated value; the pairing is recorded under
``truth_timestamp_source`` so nobody can mistake it for a timestamp carried by the kinematics payload.

Depth convention: this adapter requests ``ImageType.DepthPerspective`` (2), i.e. the distance along the
projection ray for each pixel (``ImageCaptureBase.hpp:19-25``; upstream ``docs/image_apis.md:226``:
"For ImageType = DepthPerspective, you get depth from camera using a projection ray that hits that
pixel"). ``control/perception.py`` projects full pixel rays under that same convention. Requesting
``DepthPlanar`` (1) here while perception assumes ray distance would silently misplace every obstacle
off the optical axis.

Information separation: :meth:`sample_state` deliberately ignores the ``collision`` field of
``MultirotorState``. Collision data is privileged truth and is reachable only through
:meth:`sample_truth`, which the runner routes to the evaluator ledger.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from colosseum_assurance.config import AppConfig, EndpointConfig, SceneLaunchPlatform
from colosseum_assurance.interfaces import AdapterError, AdapterTimeout, CapturedFrame
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.rpc.msgpack_rpc import (
    MsgpackRpcClient,
    RpcError,
    RpcProtocolError,
    RpcTimeout,
    RpcTransportError,
    as_bytes,
)
from colosseum_assurance.scenario.manifest import ScenarioManifest
from colosseum_assurance.schemas import (
    FrameRef,
    SimulatorArtifactAttestation,
    SimulatorIdentity,
    TruthSample,
    Vec3,
    VehicleState,
)
from colosseum_assurance.sim.identity import build_identity, load_attestation

LOGGER = logging.getLogger("colosseum_assurance.sim.colosseum_adapter")

# --- structure layouts, from PythonClient/airsim/types.py at the pin --------------------------------
VECTOR3R_ORDER = ("x_val", "y_val", "z_val")                                    # types.py:134
QUATERNIONR_ORDER = ("w_val", "x_val", "y_val", "z_val")                        # types.py:207
POSE_ORDER = ("position", "orientation")                                        # types.py:309
KINEMATICS_ORDER = ("position", "orientation", "linear_velocity", "angular_velocity",
                    "linear_acceleration", "angular_acceleration")              # types.py:517
COLLISION_ORDER = ("has_collided", "penetration_depth", "time_stamp", "normal", "impact_point",
                   "position", "object_name", "object_id")                      # types.py:341
MULTIROTOR_STATE_ORDER = ("collision", "kinematics_estimated", "gps_location", "timestamp",
                          "landed_state", "rc_data", "ready", "ready_message",
                          "can_arm")                                            # types.py:578
IMAGE_RESPONSE_ORDER = ("image_data_uint8", "image_data_float", "camera_position", "camera_name",
                        "camera_orientation", "time_stamp", "message", "pixels_as_float", "compress",
                        "width", "height", "image_type")                        # types.py:453

IMAGE_TYPE_SCENE = 0        # types.py:71 and AirLib/include/common/ImageCaptureBase.hpp:19-25
IMAGE_TYPE_DEPTH_PLANAR = 1        # distance in the camera plane (docs/image_apis.md:226)
IMAGE_TYPE_DEPTH_PERSPECTIVE = 2   # distance along the projection ray; the convention we use
DRIVETRAIN_MAX_DEGREE_OF_FREEDOM = 0    # types.py: class DrivetrainType
WEATHER_PARAMETER_FOG = 7               # types.py: class WeatherParameter
LANDED_STATE_LANDED = 0                 # types.py: class LandedState

FIXTURE_LOAD_SCENE_METHOD = "__colassure_load_scene__"
FIXTURE_TRUTH_EXTRAS_METHOD = "__colassure_truth_extras__"

SUPPORTED_CAPTURE_KINDS = ("rgb", "depth", "segmentation")

# One nanosecond-to-second factor, written once so no call site can invent another.
NANOS_PER_SECOND = 1e9
# Default tolerance when checking that the world really advanced. One SteppableClock step at the
# default step size is 20 ms (SteppableClock.hpp:19), so an advance may legitimately land one quantum
# short of, or beyond, the requested dt.
DEFAULT_ADVANCE_TOLERANCE_S = 0.02


class ColosseumAdapter:
    """Bounded, timed client for one Colosseum vehicle. Implements ``interfaces.SimAdapter``."""

    def __init__(
        self,
        endpoint: EndpointConfig,
        protocol: ProtocolConfig,
        *,
        frames_dir: Path | None = None,
        client: MsgpackRpcClient | None = None,
        settle_speed_mps: float = 0.5,
        attestation: SimulatorArtifactAttestation | None = None,
        scene_mode: str = "verify_only",
        geometry_contract_path: Path | None = None,
        scene_asset_actor_name: str | None = None,
        scene_launch_platform: SceneLaunchPlatform | None = None,
        scene_inventory_max_probes: int = 256,
        scene_inventory_wall_budget_s: float = 120.0,
        scene_inventory_exclusions: dict[str, str] | None = None,
        camera_max_attitude_pairing_gap_s: float = 0.05,
    ) -> None:
        self.endpoint = endpoint
        self.protocol = protocol
        self.simulation = protocol.simulation
        self.vehicle_name = endpoint.vehicle_name or protocol.simulation.vehicle_name
        self.camera_name = protocol.simulation.camera_name
        self.frames_dir = Path(frames_dir) if frames_dir is not None else None
        self.settle_speed_mps = float(settle_speed_mps)
        # Anchors provenance. Without it the identity can rise no higher than
        # "airsim_compatible_unverified", because a protocol handshake identifies a surface, not a build.
        self.attestation = attestation
        # How the study geometry reaches the level: "verify_only" (the map already contains the study
        # actors), "instantiate" (spawn them through verified RPC), or "qualified_map" (measure a
        # third-party map's own bodies and bind the manifest to them, which changes the scenario
        # definition and is recorded as such). See docs/scene-integration.md.
        self.scene_mode: str = scene_mode
        # The operator's reviewed geometry contract. Without it a live scene whose extents cannot be
        # measured is blocked, because an unmeasured extent leaves the evaluator's clearance assumption
        # unfounded (docs/scene-integration.md).
        self.geometry_contract_path: Path | None = geometry_contract_path
        self.scene_asset_actor_name = scene_asset_actor_name
        deployment = AppConfig(
            scene_mode=scene_mode, scene_geometry_contract_path=geometry_contract_path,
            scene_asset_actor_name=scene_asset_actor_name, scene_launch_platform=scene_launch_platform,
            scene_inventory_max_probes=scene_inventory_max_probes,
            scene_inventory_wall_budget_s=scene_inventory_wall_budget_s,
            scene_inventory_exclusions=scene_inventory_exclusions or {},
            camera_max_attitude_pairing_gap_s=camera_max_attitude_pairing_gap_s,
        )
        self.scene_launch_platform = (scene_launch_platform.model_copy(deep=True)
                                      if scene_launch_platform else None)
        self.scene_inventory_max_probes = scene_inventory_max_probes
        self.scene_inventory_wall_budget_s = scene_inventory_wall_budget_s
        self.scene_inventory_exclusions = dict(scene_inventory_exclusions or {})
        self.scene_deployment_evidence = deployment.scene_configuration_evidence()
        self._requested_manifest_hashes: dict[str, str] = {}
        self.scene_frame_report: dict[str, Any] | None = None
        self._scene_frame_reference: np.ndarray | None = None
        self.camera_mount = None
        self._camera_mount_reference = None
        self.camera_hfov_rad = None
        self._camera_hfov_reference = None
        self.start_pose_report: dict[str, Any] | None = None
        self._bound_actor_names: dict[str, str] = {}
        self.bound_manifest = None
        self.max_speed_mps = 1.5 * max(protocol.mission.cruise_speed_mps,
                                       protocol.mission.approach_speed_mps)
        self._client = client or MsgpackRpcClient(
            endpoint.host, endpoint.port,
            connect_timeout_s=endpoint.connect_timeout_s,
            call_timeout_s=endpoint.rpc_timeout_s,
        )
        self._identity: SimulatorIdentity | None = None
        self._paused = False
        self._api_control_enabled = False
        self._armed = False
        self._disarm_acknowledgement: dict[str, Any] | None = None
        self._pending: dict[str, int] = {}
        self.landing_report: dict[str, Any] | None = None
        self._landing_active = False
        self._landing_task_counter = 0
        self._sim_time_s = 0.0
        self._time_origin_ns: int | None = None
        self._last_clock_ns: int | None = None
        self._collision_count = 0
        self._last_collision_stamp: int | None = None
        self._last_landed = False
        self.clock_regressions = 0
        self.stepping_mode_used: str | None = None
        self.stepping_fallback_reason: str | None = None
        self.frames_saved = 0
        # --- advance accounting. The runner records these, so a degraded run is visible in evidence.
        self.advance_tolerance_s = DEFAULT_ADVANCE_TOLERANCE_S
        self.advance_poll_iterations = 0
        self.short_advances = 0
        self.advance_overshoots = 0
        self.steps_taken = 0
        self.total_measured_advance_s = 0.0
        self.last_measured_advance_s: float | None = None
        self.clock_reads = 0
        self.clock_type: str | None = None
        self.clock_type_note = "getSettingsString has not been read yet"
        self.pause_verified: bool | None = None
        self.truth_pairing_gap_s = 0.0
        self.max_truth_pairing_gap_s = 0.0
        self.frames_without_timestamp = 0
        # Wall-clock budget for confirming one advance. None means "derive it from dt and the RPC
        # timeout"; the live gate and the tests set it explicitly to bound how long a stalled
        # simulator may hold this process.
        self.advance_budget_s: float | None = None

    # ================================================================== lifecycle
    def connect(self) -> SimulatorIdentity:
        """Open the connection, run the handshake, and record simulator identity."""
        self._invalidate_landing("connection_changed")
        self.landing_report = None
        self.scene_frame_report = None
        self._scene_frame_reference = None
        self._disarm_acknowledgement = None
        self._control_mode_sample = None
        self.camera_mount = None
        self._camera_mount_reference = None
        self.camera_hfov_rad = None
        self._camera_hfov_reference = None
        try:
            self._client.connect()
        except RpcTransportError as exc:
            raise AdapterError(
                f"cannot reach the simulator at {self.endpoint.description}: {exc}",
                remedy=(
                    "Start the Colosseum simulator and make its RPC port reachable on this host. "
                    "For a remote server open an SSH tunnel first: "
                    f"ssh -N -L {self.endpoint.port}:127.0.0.1:{self.endpoint.port} user@server."
                ),
                cause=exc,
            ) from exc
        self._identity = build_identity(
            self._client,
            endpoint_label=self.endpoint.label,
            timeout_s=self.endpoint.rpc_timeout_s,
            attestation=self.attestation,
        )
        LOGGER.info("simulator identity: provenance=%s endpoint=%s",
                    self._identity.provenance, self.endpoint.description)
        self._read_clock_type()
        if self.simulation.stepping_mode == "paused_continue_for_time":
            self._try_enter_paused_mode()
        return self._identity

    def _read_clock_type(self) -> None:
        """Record which clock the server runs, because it decides what a timestamp means.

        ``ClockType`` selects ``ScalableClock`` (scaled Unix-epoch wall clock, ``step()`` is a no-op)
        or ``SteppableClock`` (advanced only by the physics world) at ``SimModeBase.cpp:335-344``. With
        a ScalableClock the reported timestamps keep rising while the game is paused, so a measured
        advance is wall-clock time, not stepped time. That is a reproducibility caveat which must be
        recorded rather than discovered afterwards, so it is read once and reported, never guessed.
        """
        try:
            settings_text = self._client.call("getSettingsString",
                                              timeout_s=min(self.endpoint.rpc_timeout_s, 10.0))
        except (RpcError, RpcTransportError, RpcProtocolError, RpcTimeout) as exc:
            self.clock_type = None
            self.clock_type_note = f"getSettingsString failed ({exc}); clock type UNVERIFIED"
            LOGGER.warning("%s", self.clock_type_note)
            return
        if not isinstance(settings_text, str) or not settings_text.strip():
            self.clock_type = None
            self.clock_type_note = "getSettingsString returned no document; clock type UNVERIFIED"
            return
        try:
            document = json.loads(settings_text)
        except ValueError:
            self.clock_type = None
            self.clock_type_note = "settings string is not valid JSON; clock type UNVERIFIED"
            return
        if not isinstance(document, dict):
            self.clock_type = None
            self.clock_type_note = "settings document is not a JSON object; clock type UNVERIFIED"
            return
        value = document.get("ClockType")
        if isinstance(value, str) and value.strip():
            self.clock_type = value.strip()
            self.clock_type_note = "read from getSettingsString"
        else:
            # AirSimSettings defaults ClockType to an empty string, which SimModeBase then rejects
            # unless it is one of the two names, so an absent key tells us nothing on its own.
            self.clock_type = None
            self.clock_type_note = "settings document carries no ClockType; clock type UNVERIFIED"
        if self.clock_type == "ScalableClock" and self.simulation.stepping_mode == (
                "paused_continue_for_time"):
            LOGGER.warning(
                "the server runs a ScalableClock, whose nowNanos() is the scaled wall clock "
                "(ScalableClock.hpp:31-45) and whose step() does nothing (ClockBase.hpp:57-65). "
                "Paused stepping then measures wall-clock time, not stepped time. Set "
                "\"ClockType\": \"SteppableClock\" in settings.json for a reproducible run.")

    def identity(self) -> SimulatorIdentity:
        if self._identity is None:
            raise AdapterError("identity is unknown: connect() has not run",
                               remedy="Call connect() before reading identity().")
        return self._identity

    @property
    def is_fixture_fake(self) -> bool:
        return self._identity is not None and self._identity.provenance == "fixture_fake"

    def close(self) -> None:
        """Release the connection. Best effort: never raises, so teardown cannot mask a real error."""
        self._invalidate_landing("connection_closed")
        self._disarm_acknowledgement = None
        self._control_mode_sample = None
        for msgid in self._pending.values():
            self._client.abandon(msgid)
        self._pending.clear()
        if self._client.is_connected and self._paused:
            # Leave the simulator running, otherwise the next client inherits a frozen world.
            try:
                self._client.call("simPause", False, timeout_s=2.0)
            except (RpcError, RpcTransportError, RpcProtocolError):
                LOGGER.debug("could not unpause the simulator during close()", exc_info=True)
        self._paused = False
        self._client.close()

    def __enter__(self) -> ColosseumAdapter:
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ================================================================== rpc plumbing
    def _call(self, method: str, *params: Any, timeout_s: float | None = None,
              remedy: str = "") -> Any:
        """Single funnel for every RPC: applies the timeout and converts failures to adapter errors."""
        budget = self.endpoint.rpc_timeout_s if timeout_s is None else float(timeout_s)
        try:
            return self._client.call(method, *params, timeout_s=budget)
        except RpcTimeout as exc:
            self._invalidate_landing("rpc_timeout")
            self._disarm_acknowledgement = None
            self._control_mode_sample = None
            raise AdapterTimeout(
                f"{method} did not answer within {budget:g} s",
                remedy=remedy or (
                    "The simulator is unresponsive. Check that the Unreal process is running and not "
                    "blocked on a modal dialog, then raise endpoint.rpc_timeout_s if the scene is heavy. "
                    "An episode interrupted this way is incomplete, never a pass."
                ),
                cause=exc,
            ) from exc
        except RpcError as exc:
            raise AdapterError(
                f"{method} was rejected by the simulator: {exc.text}",
                remedy=remedy or (
                    "The server understood the call and refused it. Check API control, the vehicle "
                    "name in settings.json, and that the requested camera exists."
                ),
                cause=exc,
            ) from exc
        except (RpcTransportError, RpcProtocolError) as exc:
            self._invalidate_landing("rpc_transport_uncertainty")
            self._disarm_acknowledgement = None
            self._control_mode_sample = None
            raise AdapterError(
                f"{method} failed on the transport: {exc}",
                remedy=remedy or (
                    "The connection dropped. Check the simulator process and the SSH tunnel, then "
                    "re-run. Partial episodes must be recorded as incomplete."
                ),
                cause=exc,
            ) from exc

    def _send(self, label: str, method: str, *params: Any) -> None:
        """Issue a command without waiting. Its reply is collected on the next clock advance."""
        if label != "land":
            self._invalidate_landing(f"superseded_by_{method}")
        previous = self._pending.pop(label, None)
        if previous is not None:
            self._client.abandon(previous)
        try:
            self._pending[label] = self._client.send_request(method, *params)
        except RpcTransportError as exc:
            raise AdapterError(f"could not send {method}: {exc}",
                               remedy="The connection to the simulator dropped.", cause=exc) from exc

    def _drain_pending(self, timeout_s: float = 0.0) -> None:
        """Collect replies to issued commands so a server-side refusal cannot pass unnoticed."""
        for label, msgid in list(self._pending.items()):
            try:
                done, result = self._client.try_receive(msgid, timeout_s=timeout_s)
            except RpcError as exc:
                if label == "land":
                    self._landing_event("native_land_error", error=exc.text)
                    self._landing_active = False
                self._pending.pop(label, None)
                raise AdapterError(
                    f"the simulator refused the {label} command: {exc.text}",
                    remedy=(
                        "Commands are refused when API control was lost, for example after a reset "
                        "without a new enableApiControl/armDisarm pair."
                    ),
                    cause=exc,
                ) from exc
            except (RpcTransportError, RpcProtocolError) as exc:
                self._invalidate_landing("pending_rpc_transport_uncertainty")
                self._pending.pop(label, None)
                raise AdapterError(f"lost the reply to {label}: {exc}",
                                   remedy="The connection dropped mid-command.", cause=exc) from exc
            if done:
                self._pending.pop(label, None)
                if label == "land" and self._landing_active and self.landing_report is not None:
                    self.landing_report["native_result"] = result
                    self._landing_event("native_land_completed", result=result, message_id=msgid)
                    if result is not True:
                        self._landing_active = False
                        raise AdapterError("native land did not report successful completion")

    # ================================================================== decoding
    @staticmethod
    def _struct(value: Any, order: tuple[str, ...], what: str) -> dict[str, Any]:
        """Decode a msgpack structure that upstream encodes as a positional array."""
        if isinstance(value, dict):
            missing = [key for key in order if key not in value]
            if missing:
                raise AdapterError(
                    f"{what}: map form is missing fields {missing}",
                    remedy="The server speaks a different struct layout than the pinned Colosseum.",
                )
            return {key: value[key] for key in order}
        if isinstance(value, (list, tuple)):
            if len(value) != len(order):
                raise AdapterError(
                    f"{what}: expected {len(order)} positional fields {list(order)}, got {len(value)}",
                    remedy=(
                        "The server's struct layout does not match Colosseum at commit "
                        "84fc0c1c75bc73a0135ee80a325d470577c66c52. Check the simulator build before "
                        "trusting any recorded data."
                    ),
                )
            return dict(zip(order, value, strict=True))
        raise AdapterError(f"{what}: expected an array or map, got {type(value).__name__}",
                           remedy="The server is not speaking the Colosseum RPC structure layout.")

    @classmethod
    def _vec3(cls, value: Any, what: str) -> Vec3:
        fields = cls._struct(value, VECTOR3R_ORDER, what)
        return Vec3(x=float(fields["x_val"]), y=float(fields["y_val"]), z=float(fields["z_val"]))

    @classmethod
    def _yaw(cls, value: Any, what: str) -> float:
        q = cls._struct(value, QUATERNIONR_ORDER, what)
        w, x, y, z = (float(q[k]) for k in QUATERNIONR_ORDER)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _quaternion_from_yaw(yaw_rad: float) -> list[float]:
        return [math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)]

    @staticmethod
    def _as_stamp_ns(timestamp_ns: Any) -> int | None:
        """Return a usable nanosecond stamp, or None when the server did not supply one."""
        try:
            stamp = int(timestamp_ns)
        except (TypeError, ValueError):
            return None
        return stamp if stamp > 0 else None

    def _to_episode_time(self, stamp_ns: int) -> float:
        """Convert one genuine simulator timestamp to episode-relative seconds WITHOUT moving the clock.

        A camera frame can legitimately carry an older timestamp than the newest state read: the image
        was acquired earlier. Feeding that stamp through :meth:`_record_stamp` would rewind the
        authoritative clock and count a false regression, so capture-time conversion is kept separate
        from authoritative current-clock updates. The origin is shared, so both produce the same
        episode-relative scale.
        """
        if self._time_origin_ns is None:
            self._time_origin_ns = stamp_ns
        return max(0.0, (stamp_ns - self._time_origin_ns) / NANOS_PER_SECOND)

    def _record_stamp(self, stamp_ns: int) -> float:
        """Rebase one genuine simulator timestamp onto the episode-relative clock.

        The origin is the first timestamp seen after connect or reset, so every episode starts at 0 and
        records stay comparable across runs (docs/timing-semantics.md). Nanoseconds to seconds is the
        only unit conversion, and it happens here alone.
        """
        if self._time_origin_ns is None:
            self._time_origin_ns = stamp_ns
        observed = max(0.0, (stamp_ns - self._time_origin_ns) / NANOS_PER_SECOND)
        if observed + 1e-6 < self._sim_time_s:
            # The simulator clock is authoritative, so it is not clamped. A backwards jump means
            # something reset or reloaded the simulator under this session, which would silently
            # corrupt an episode record, so it is counted and logged instead of smoothed away.
            self.clock_regressions += 1
            LOGGER.warning(
                "simulator clock went backwards: %.3f s -> %.3f s. Something reset the simulator "
                "outside this session; the episode is not trustworthy.", self._sim_time_s, observed,
            )
        self._sim_time_s = observed
        self._last_clock_ns = stamp_ns
        return self._sim_time_s

    def _sim_time_from_ns(self, timestamp_ns: Any, *, what: str) -> float:
        """Convert a REQUIRED simulator timestamp to episode-relative seconds, or fail loudly.

        Silently substituting a locally accumulated value here is exactly the defect this method
        exists to prevent: it would let a record claim a simulator time the simulator never reported.
        """
        stamp = self._as_stamp_ns(timestamp_ns)
        if stamp is None:
            raise AdapterError(
                f"{what}: the server returned no usable simulator timestamp ({timestamp_ns!r})",
                remedy=(
                    "Upstream fills this field from the simulator clock "
                    "(MultirotorApiBase.hpp:134-140 for MultirotorState.timestamp). A server that "
                    "leaves it empty is not a build we can time an episode against. Check the "
                    "simulator build and settings.json ClockType before recording anything."
                ),
            )
        return self._record_stamp(stamp)

    def _multirotor_state(self) -> dict[str, Any]:
        """One decoded ``MultirotorState``. The single RPC that carries the simulator clock."""
        raw = self._call("getMultirotorState", self.vehicle_name)
        state = self._struct(raw, MULTIROTOR_STATE_ORDER, "MultirotorState")
        self._last_landed = int(state["landed_state"]) == LANDED_STATE_LANDED
        return state

    def _read_sim_clock_ns(self) -> int:
        """Read the simulator's OWN clock in nanoseconds (MultirotorApiBase.hpp:134-140).

        There is no cheaper verified clock getter at the pin: ``simGetGroundTruthKinematics`` carries no
        timestamp at all (RpcLibAdaptorsBase.hpp:398-409), so this is the honest way to ask what time
        the simulator thinks it is.
        """
        state = self._multirotor_state()
        stamp = self._as_stamp_ns(state["timestamp"])
        if stamp is None:
            raise AdapterError(
                "MultirotorState.timestamp is empty, so the simulator clock cannot be read",
                remedy=(
                    "Upstream sets it from clock()->nowNanos() (MultirotorApiBase.hpp:134-140). "
                    "Without it no episode can be timed; do not record data from this server."
                ),
            )
        self.clock_reads += 1
        return stamp

    # ================================================================== session control
    def ping(self) -> bool:
        return bool(self._call("ping", timeout_s=min(self.endpoint.rpc_timeout_s, 5.0)))

    def reset(self) -> None:
        """Reset the simulator and drop every cached session fact.

        After ``reset`` the upstream server requires ``enableApiControl`` and ``armDisarm`` again
        (client.py:20-26 docstring and docs/apis.md), so the cached flags are cleared here.
        """
        self._invalidate_landing("reset")
        self.landing_report = None
        self._disarm_acknowledgement = None
        self._control_mode_sample = None
        self.scene_frame_report = None
        self.camera_mount = None
        self.camera_hfov_rad = None
        self.start_pose_report = None
        for msgid in self._pending.values():
            self._client.abandon(msgid)
        self._pending.clear()
        self._call("reset", timeout_s=self.endpoint.reset_timeout_s,
                   remedy="The simulator did not complete a reset. Restart the simulator process.")
        self._api_control_enabled = False
        self._armed = False
        self._paused = False
        self._time_origin_ns = None
        self._last_clock_ns = None
        self._sim_time_s = 0.0
        self._collision_count = 0
        self._last_collision_stamp = None
        self._last_landed = False
        self.clock_regressions = 0
        self.advance_poll_iterations = 0
        self.short_advances = 0
        self.advance_overshoots = 0
        self.steps_taken = 0
        self.total_measured_advance_s = 0.0
        self.last_measured_advance_s = None
        self.truth_pairing_gap_s = 0.0
        self.max_truth_pairing_gap_s = 0.0
        self.frames_without_timestamp = 0
        if self.simulation.stepping_mode == "paused_continue_for_time":
            self._try_enter_paused_mode()

    def wait_until_ready(self, timeout_s: float | None = None) -> None:
        """Block until the vehicle answers, accepts API control, and has settled.

        Bounded by ``timeout_s``. A vehicle that never settles raises :class:`AdapterError` with a
        remedy rather than letting an episode start from an unknown state.
        """
        budget = float(timeout_s if timeout_s is not None else self.endpoint.reset_timeout_s)
        deadline = time.monotonic() + budget
        settle_dt = max(self.simulation.reset_settle_s, 0.1)
        last_reason = "no attempt completed"
        attempts = 0
        while time.monotonic() < deadline:
            attempts += 1
            try:
                if not self.ping():
                    last_reason = "ping did not return True"
                else:
                    self._call("enableApiControl", True, self.vehicle_name)
                    enabled = bool(self._call("isApiControlEnabled", self.vehicle_name))
                    if not enabled:
                        last_reason = "isApiControlEnabled stayed False after enableApiControl"
                    else:
                        self._api_control_enabled = True
                        state = self.sample_state()
                        speed = math.dist(state.velocity.as_tuple(), (0.0, 0.0, 0.0))
                        if speed <= self.settle_speed_mps:
                            LOGGER.debug("vehicle ready after %d attempt(s)", attempts)
                            return
                        last_reason = f"vehicle still moving at {speed:.2f} m/s"
            except AdapterError as exc:
                # Retry inside the budget: right after a reset the vehicle may not answer yet. A
                # persistent refusal still ends in the bounded failure below, naming the last reason.
                last_reason = f"a readiness call failed: {exc}"
            try:
                self._advance_time(settle_dt)
            except AdapterError as exc:
                # The clock is read from the simulator, so a vehicle that cannot answer yet also
                # cannot be stepped yet. That is a reason to keep waiting inside the budget, not to
                # abandon the readiness check with a different error.
                last_reason = f"advancing the clock failed: {exc}"
                time.sleep(min(settle_dt, 0.2))
        raise AdapterError(
            f"vehicle was not ready within {budget:g} s after {attempts} attempt(s): {last_reason}",
            remedy=(
                "Check that the simulator finished loading the level, that the vehicle name "
                f"{self.vehicle_name!r} exists in settings.json, and that no other client holds API "
                "control. Do not record an episode that starts from an unknown state."
            ),
        )

    def acquire_control(self, *, arm: bool = True) -> None:
        """Take verified API control; setup callers explicitly request a disarmed vehicle."""
        self._invalidate_landing("control_reacquired")
        self._disarm_acknowledgement = None
        self._control_mode_sample = None
        self._call("enableApiControl", True, self.vehicle_name)
        if not bool(self._call("isApiControlEnabled", self.vehicle_name)):
            raise AdapterError(
                "the simulator did not grant API control",
                remedy=(
                    "Another client may hold control, or the vehicle name does not match "
                    "settings.json. Check the Vehicles block in the simulator settings."
                ),
            )
        self._api_control_enabled = True
        if self._call("armDisarm", bool(arm), self.vehicle_name) is not True:
            raise AdapterError(f"the simulator did not acknowledge {'arming' if arm else 'disarming'}")
        self._armed = bool(arm)
        if not arm:
            self._disarm_acknowledgement = {
                "method": "armDisarm", "args": [False, self.vehicle_name], "result": True,
                "source": "acquire_control_disarmed_after_latest_reset_or_connection",
            }

    def release_control(self) -> None:
        """Disarm and hand API control back. Verified, so a stuck session is visible."""
        self._invalidate_landing("control_released")
        self._disarm_acknowledgement = None
        self._control_mode_sample = None
        try:
            self._call("armDisarm", False, self.vehicle_name)
        finally:
            self._armed = False
        self._call("enableApiControl", False, self.vehicle_name)
        if bool(self._call("isApiControlEnabled", self.vehicle_name)):
            raise AdapterError(
                "API control was still enabled after releasing it",
                remedy="The simulator kept the session. Restart the simulator before the next episode.",
            )
        self._api_control_enabled = False

    def arm(self) -> None:
        self._invalidate_landing("arming_requested")
        self._disarm_acknowledgement = None
        self._control_mode_sample = None
        self._call("armDisarm", True, self.vehicle_name)
        self._armed = True

    def disarm(self) -> None:
        self._disarm_acknowledgement = None
        self._control_mode_sample = None
        acknowledged = self._call("armDisarm", False, self.vehicle_name)
        self._last_disarm_result = acknowledged
        if acknowledged is True:
            self._disarm_acknowledgement = {
                "method": "armDisarm", "args": [False, self.vehicle_name], "result": True,
                "source": "explicit_disarm_after_latest_reset_or_connection",
            }
        self._armed = False

    # ================================================================== clock
    def _try_enter_paused_mode(self) -> None:
        """Pause the simulator so ``step`` can advance it deterministically, and VERIFY the pause.

        ``simPause`` returns nothing (RpcLibServerBase.cpp:103-105), so the only evidence that the
        world actually paused is ``simIsPaused`` (RpcLibServerBase.cpp:107-109 ->
        SimModeWorldBase.cpp:90-93). A server that accepts the pause and keeps running would otherwise
        produce episodes in which the world moved between our steps, unobserved.
        """
        try:
            self._client.call("simPause", True, timeout_s=self.endpoint.rpc_timeout_s)
        except (RpcError, RpcTransportError, RpcProtocolError, RpcTimeout) as exc:
            self._paused = False
            self.pause_verified = False
            self.stepping_mode_used = "wall_clock"
            self.stepping_fallback_reason = (
                f"simPause failed ({exc}); falling back to wall-clock stepping, which is less "
                "reproducible. This is recorded, not hidden."
            )
            LOGGER.warning("%s", self.stepping_fallback_reason)
            return
        try:
            confirmed = bool(self._client.call("simIsPaused", timeout_s=self.endpoint.rpc_timeout_s))
            self.pause_verified = confirmed
        except (RpcError, RpcTransportError, RpcProtocolError, RpcTimeout) as exc:
            confirmed = False
            self.pause_verified = None
            self.stepping_fallback_reason = (
                f"simIsPaused could not be read ({exc}); the pause is UNVERIFIED, so stepping falls "
                "back to wall clock rather than assuming a frozen world."
            )
            LOGGER.warning("%s", self.stepping_fallback_reason)
        else:
            if not confirmed:
                self.stepping_fallback_reason = (
                    "simPause was accepted but simIsPaused stayed False, so the server does not "
                    "support paused stepping. Falling back to wall-clock stepping, which is less "
                    "reproducible. This is recorded, not hidden."
                )
                LOGGER.warning("%s", self.stepping_fallback_reason)
        if not confirmed:
            self._paused = False
            self.stepping_mode_used = "wall_clock"
            return
        self._paused = True
        self.stepping_mode_used = "paused_continue_for_time"
        self.stepping_fallback_reason = None

    def _advance_budget_s(self, dt_s: float) -> float:
        """Wall-clock budget for one advance: generous, bounded, and never infinite."""
        if self.advance_budget_s is not None:
            return max(float(self.advance_budget_s), 0.05)
        return max(float(self.endpoint.rpc_timeout_s), float(dt_s) * 4.0 + 2.0)

    def _await_advance(self, before_ns: int, dt_s: float, budget_s: float, how: str) -> float:
        """Poll the simulator's OWN clock until it advanced by at least ``dt_s`` (minus tolerance).

        WHY this exists even though upstream blocks: ``ASimModeWorldBase::continueForTime``
        busy-waits for the physics world and a rendered frame before returning
        (SimModeWorldBase.cpp:101-118), so a conforming server needs one confirming read and no
        waiting at all. The poll is the check, not the fix. A build that does not behave that way, an
        unsupported SimMode, or a fake must produce a loud :class:`AdapterTimeout` instead of a step
        that silently recorded time which never passed.
        """
        tolerance = max(0.0, float(self.advance_tolerance_s))
        deadline = time.monotonic() + max(budget_s, 0.05)
        poll_interval = min(0.02, max(0.002, float(dt_s) / 20.0))
        iterations = 0
        advance = 0.0
        while True:
            now_ns = self._read_sim_clock_ns()
            iterations += 1
            self.advance_poll_iterations += 1
            advance = (now_ns - before_ns) / NANOS_PER_SECOND
            if advance + tolerance >= float(dt_s):
                break
            if time.monotonic() >= deadline:
                self._record_stamp(now_ns)
                raise AdapterTimeout(
                    f"the simulator clock advanced {advance:.4f} s of the requested {dt_s:.4f} s "
                    f"within {budget_s:g} s of wall clock ({how}, {iterations} clock reads)",
                    remedy=(
                        "The server accepted the advance and its own clock did not follow. Upstream "
                        "simContinueForTime returns only after the physics world paused again "
                        "(SimModeWorldBase.cpp:101-118), so this server is not behaving like the "
                        "pinned build: check SimMode (a non-world SimMode throws 'continueForTime is "
                        "not implemented by SimMode', SimModeBase.cpp:294-299), check ClockType in "
                        "settings.json, and check that no other client is pausing the simulator. "
                        "The episode is incomplete; it is never a pass."
                    ),
                )
            time.sleep(poll_interval)
        if iterations > 1:
            # The first read did not show the full advance: the server returned before its own clock
            # had moved. Recorded, because it changes what the timestamps of this step mean.
            self.short_advances += 1
        if advance > float(dt_s) + tolerance:
            self.advance_overshoots += 1
        self._record_stamp(now_ns)
        return advance

    def step(self, dt_s: float) -> float:
        """Advance simulated time by ``dt_s`` and return the advance the SIMULATOR actually reported.

        The returned value is measured from the simulator clock before and after the advance, never
        assumed from ``dt_s``. A step that cannot be confirmed raises :class:`AdapterTimeout`.
        """
        if dt_s <= 0.0:
            raise ValueError("step dt_s must be positive")
        if (self.simulation.stepping_mode == "paused_continue_for_time" and not self._paused):
            self._try_enter_paused_mode()
        before_ns = self._read_sim_clock_ns()
        if self.simulation.stepping_mode == "paused_continue_for_time" and self._paused:
            self._call("simContinueForTime", float(dt_s),
                       timeout_s=self._advance_budget_s(dt_s),
                       remedy="simContinueForTime stalled; the simulator is not stepping.")
            self.stepping_mode_used = "paused_continue_for_time"
            how = "simContinueForTime"
        else:
            time.sleep(float(dt_s))
            self.stepping_mode_used = "wall_clock"
            how = "wall-clock sleep"
        advance = self._await_advance(before_ns, float(dt_s), self._advance_budget_s(dt_s), how)
        self.steps_taken += 1
        self.last_measured_advance_s = advance
        self.total_measured_advance_s += advance
        self._drain_pending()
        return advance

    def _advance_time(self, dt_s: float) -> None:
        """Internal clock advance used by setup and teardown actions."""
        try:
            self.step(dt_s)
        except AdapterError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise AdapterError(f"advancing the simulator clock failed: {exc}",
                               remedy="The simulator stopped stepping.", cause=exc) from exc

    def sim_time_s(self) -> float:
        """Last known simulator clock in episode-relative seconds (no extra RPC)."""
        return self._sim_time_s

    def stepping_report(self) -> dict[str, Any]:
        """What the runner records about how time was advanced."""
        report = {
            "configured_mode": self.simulation.stepping_mode,
            "mode_used": self.stepping_mode_used or "not_stepped_yet",
            "paused": self._paused,
            "fallback_reason": self.stepping_fallback_reason,
            "sim_time_s": self._sim_time_s,
            "clock_regressions": self.clock_regressions,
        }
        report.update(self.timing_report())
        return report

    def timing_report(self) -> dict[str, Any]:
        """Measured facts about the clock, so a degraded run is visible in the evidence.

        Every value here is measured, not configured: how often the simulator clock had to be polled
        before it showed the requested advance, how often it fell short at first read, how often it
        overshot, and what the last measured advance actually was.
        """
        return {
            "clock_type": self.clock_type,
            "clock_type_note": self.clock_type_note,
            "clock_source": "MultirotorState.timestamp (MultirotorApiBase.hpp:134-140), nanoseconds",
            "truth_timestamp_source": (
                "paired MultirotorState.timestamp read; KinematicsState carries no timestamp "
                "(RpcLibAdaptorsBase.hpp:398-409)"
            ),
            "pause_verified": self.pause_verified,
            "advance_tolerance_s": self.advance_tolerance_s,
            "steps_taken": self.steps_taken,
            "advance_poll_iterations": self.advance_poll_iterations,
            "short_advances": self.short_advances,
            "advance_overshoots": self.advance_overshoots,
            "last_measured_advance_s": self.last_measured_advance_s,
            "total_measured_advance_s": self.total_measured_advance_s,
            "clock_reads": self.clock_reads,
            "max_truth_pairing_gap_s": self.max_truth_pairing_gap_s,
            "frames_without_timestamp": self.frames_without_timestamp,
        }

    def reapply_environment(self, manifest: ScenarioManifest | None = None) -> dict[str, Any]:
        """Re-apply everything a ``reset`` may have cleared, and report what was applied.

        WHAT RESET IS VERIFIED TO DO. ``reset`` (RpcLibServerBase.cpp:281-296) calls
        ``WorldSimApi::reset`` (WorldSimApi.cpp:265-271), which runs ``ASimModeWorldBase::reset``
        (SimModeWorldBase.cpp:172-180) on the game thread; that resets the physics world
        (PhysicsWorld.hpp:43-48) and deliberately does NOT call the base implementation, which is the
        one that resets each vehicle sim API (SimModeBase.cpp:413-422). The upstream Python client
        states the client-visible consequence directly: "you must call `enableApiControl` and
        `armDisarm` again after the call to reset" (client.py:20-26).

        WHAT IS UNVERIFIED. No reset path at the pin mentions the weather actor, which is initialised
        once in ``BeginPlay`` (SimModeBase.cpp:149-153), the pause state, or actors spawned through the
        API. Absence of code is weak evidence, and nothing here has ever been observed against a live
        build, so this method re-applies all of them defensively instead of assuming they survived.
        Returns the applied settings so the runner can record them with the episode.
        """
        applied: dict[str, Any] = {"upstream_reset_semantics": "api_control_and_arming_verified;"
                                   " weather/pause/spawned-actor survival UNVERIFIED"}
        self.acquire_control(arm=False)
        applied["api_control_enabled"] = self._api_control_enabled
        applied["armed"] = self._armed
        applied["arming_policy"] = "disarmed_until_placement_and_settlement_are_verified"
        if manifest is not None:
            applied["visibility"] = self.set_visibility(manifest.visibility)
            extension = self.protocol.study_extension
            if extension is not None:
                if not extension.solar_datetimes:
                    raise AdapterError("expanded environment has no frozen solar datetime levels")
                solar = extension.solar_datetimes[manifest.seed % len(extension.solar_datetimes)]
                # Pinned PythonClient/client.py:224–241. Freeze solar time rather
                # than wall-clock time; pause/time stepping remain runner-owned.
                self._call("simSetTimeOfDay", True, solar, False, 0.0, 1.0, True)
                applied["lighting"] = {"solar_datetime": solar, "clock_speed": 0.0,
                                       "rpc_completed": True, "rendered_efficacy_verified": False}
        if self.simulation.stepping_mode == "paused_continue_for_time":
            self._try_enter_paused_mode()
        applied["paused"] = self._paused
        applied["pause_verified"] = self.pause_verified
        applied["stepping_fallback_reason"] = self.stepping_fallback_reason
        return applied

    # ================================================================== motion
    def takeoff(self, altitude_m: float, timeout_s: float | None = None) -> None:
        """Take off and climb to ``altitude_m`` above home, bounded by ``timeout_s``."""
        if altitude_m <= 0.0:
            raise ValueError("altitude_m must be positive (it is a height above home)")
        budget = float(timeout_s if timeout_s is not None else 30.0)
        self.issue_takeoff(min(budget, 20.0))
        half = budget / 2.0
        self._advance_until(lambda st: not st.landed and st.position.z < -0.3, half, "leave the ground")
        state = self.sample_state()
        target = Vec3(x=state.position.x, y=state.position.y, z=-abs(float(altitude_m)))
        self.move_to(target, self.protocol.mission.cruise_speed_mps,
                     min(half, self.simulation.max_command_duration_s))
        tolerance = 0.6
        self._advance_until(lambda st: abs(st.position.z + abs(altitude_m)) <= tolerance, half,
                            f"reach {altitude_m:.1f} m altitude",
                            resend=lambda: self.move_to(
                                target, self.protocol.mission.cruise_speed_mps,
                                min(half, self.simulation.max_command_duration_s)))

    def move_to(self, target: Vec3, speed_mps: float, duration_s: float,
                yaw_rad: float | None = None) -> None:
        """Issue a bounded move. Never blocks: the clock is advanced by :meth:`step`.

        ``duration_s`` is passed to the server as ``timeout_sec``, so the simulator itself abandons the
        move when the budget runs out. Speed is clamped, so a controller bug cannot command a dash.
        """
        speed = min(max(float(speed_mps), 0.05), self.max_speed_mps)
        duration = min(max(float(duration_s), 0.05), float(self.simulation.max_command_duration_s))
        yaw_mode = [False, math.degrees(yaw_rad)] if yaw_rad is not None else [True, 0.0]
        self._send(
            "move_to", "moveToPosition",
            float(target.x), float(target.y), float(target.z), speed, duration,
            DRIVETRAIN_MAX_DEGREE_OF_FREEDOM, yaw_mode, -1.0, 1.0, self.vehicle_name,
        )

    # ---- issue-only commands -------------------------------------------------------------------
    # ONE component owns the clock. The episode runner advances time in truth-sample sub-steps, so every
    # executed action is covered by dense privileged samples. The methods below only *issue* a command.
    # The convenience wrappers underneath (hold/takeoff/land) additionally advance the clock and exist
    # for diagnostics and the readiness gate, which run outside an episode. Mixing the two inside an
    # episode would advance the clock twice for one control step and leave an unsampled interval
    # (independent review: "double stepping").
    def issue_rotate_to_yaw(
        self, yaw_rad: float, timeout_s: float | None = None, margin_deg: float = 5.0
    ) -> None:
        """Command a rotation in place to ``yaw_rad``. Does not advance the clock.

        Upstream `rotateToYaw(yaw, timeout_sec, margin, vehicle_name)` is bound in
        `AirLib/src/vehicles/multirotor/api/MultirotorRpcLibServer.cpp` and implemented in
        `MultirotorApiBase::rotateToYaw`, which holds the start position and turns until
        `isYawWithinMargin`. That helper converts the current yaw to DEGREES before comparing
        (`const float yaw_current = VectorMath::getYaw(getOrientation()) * 180 / M_PIf;`), so the wire
        units of both `yaw` and `margin` are degrees. This method takes radians and converts, because
        every other angle in this package is radians.

        A hover does NOT rotate, so a controller that wants to face a direction must send this. The
        command is bounded by `timeout_s` like any other motion command, and the runner refreshes it.
        """
        budget = float(timeout_s if timeout_s is not None else self.simulation.max_command_duration_s)
        self._send(
            "rotate", "rotateToYaw",
            float(math.degrees(float(yaw_rad))), max(budget, 0.05), float(abs(margin_deg)),
            self.vehicle_name,
        )

    def issue_hold(self) -> None:
        """Command a hover. Does not advance the clock."""
        if (self._landing_active and self.landing_report is not None
                and self.landing_report.get("disarm_acknowledged")
                and self._disarm_acknowledgement is not None and not self._armed):
            # A terminal guard hold is already satisfied by the acknowledged stationary disarmed
            # state. Preserve the requested guard command without sending hover to stopped motors.
            self._landing_event("hold_satisfied_by_acknowledged_disarmed_state", native_rpc_issued=False)
            return
        self._send("hold", "hover", self.vehicle_name)

    def issue_takeoff(self, timeout_s: float | None = None) -> None:
        """Command a takeoff. Does not advance the clock."""
        budget = float(timeout_s if timeout_s is not None else 20.0)
        self._send("takeoff", "takeoff", float(min(budget, 20.0)), self.vehicle_name)

    def issue_land(self, timeout_s: float | None = None) -> None:
        """Issue/continue a native landing and qualify actuator-side disarming without clock advances."""
        if self._landing_active and self.landing_report is not None:
            if self.landing_report.get("disarm_acknowledged") or "land" in self._pending:
                return
            if self.landing_report.get("native_result") is True and self._qualify_landing_completion():
                return
        budget = float(timeout_s if timeout_s is not None else 40.0)
        self._invalidate_landing("fresh_native_land_task")
        self._landing_task_counter += 1
        self.landing_report = {
            "task_id": self._landing_task_counter, "vehicle_name": self.vehicle_name,
            "native_result": None, "consecutive_qualified_samples": 0,
            "last_state_timestamp_ns": None, "disarm_acknowledged": False,
            "speed_limit_mps": .02, "angular_speed_limit_rad_s": .02,
            "height_tolerance_m": .01, "home_xy_tolerance_m": self.protocol.mission.return_tolerance_m,
            "required_fresh_samples": 3, "events": [],
            "evidence_scope": ("actuator-side native completion and onboard estimate only; "
                               "no truth/contact input"),
        }
        self._landing_active = True
        self._send("land", "land", float(min(budget, 60.0)), self.vehicle_name)
        self._landing_event("native_land_issued", method="land", timeout_s=min(budget, 60.0),
                            message_id=self._pending.get("land"))

    def _landing_event(self, kind: str, **payload: Any) -> None:
        if self.landing_report is None:
            return
        row = {"kind": kind, "task_id": self.landing_report["task_id"],
               "vehicle_name": self.vehicle_name, "adapter_sim_time_s": self._sim_time_s, **payload}
        self.landing_report["events"].append(row)
        if self.frames_dir is not None:
            path = self.frames_dir.parent / "landing_handshake.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            self.landing_report["evidence_path"] = str(path)

    def _invalidate_landing(self, reason: str) -> None:
        if self._landing_active:
            self._landing_event("landing_proof_invalidated", reason=reason)
        self._landing_active = False

    def _qualify_landing_completion(self) -> bool:
        """True retains a qualified task; False requests a fresh native descent task, never disarm."""
        report = self.landing_report
        assert report is not None and report["native_result"] is True
        # This deployment-specific handshake uses the declared launch geometry, never its truth
        # readback or collision fields. Other deployments keep their native landing behavior.
        setup = self.start_pose_report or {}
        if self.scene_launch_platform is None or not setup.get("verified"):
            self._landing_event("completion_not_qualified", reason="no_verified_declared_launch_reference")
            return False
        try:
            declared = setup["requested_pose_local_ned"][0]
            target_z = float(declared[2])
            state = self._multirotor_state()
            kin = self._struct(state["kinematics_estimated"], KINEMATICS_ORDER, "landing onboard estimate")
            position = self._vec3(kin["position"], "landing onboard position")
            velocity = self._vec3(kin["linear_velocity"], "landing onboard velocity")
            angular = self._vec3(kin["angular_velocity"], "landing onboard angular velocity")
            stamp = self._as_stamp_ns(state["timestamp"])
            if stamp is None or not all(math.isfinite(v) for v in (
                target_z, *position.as_tuple(), *velocity.as_tuple(), *angular.as_tuple(),
            )):
                raise AdapterError("landing completion requires finite timestamped onboard state")
            prior_stamp = report["last_state_timestamp_ns"]
            fresh = prior_stamp is None or stamp > prior_stamp
            speed = math.dist(velocity.as_tuple(), (0, 0, 0))
            angular_speed = math.dist(angular.as_tuple(), (0, 0, 0))
            eligible = (abs(position.z - target_z) <= .01 and speed <= .02 and angular_speed <= .02
                        and position.horizontal_distance_to(self.protocol.mission.home)
                        <= self.protocol.mission.return_tolerance_m)
            report["consecutive_qualified_samples"] = (
                report["consecutive_qualified_samples"] + 1 if fresh and eligible else 0)
            report["last_state_timestamp_ns"] = stamp
            self._landing_event(
                "onboard_completion_qualification", timestamp_ns=stamp,
                kinematics_estimated=kin, landed_state=state["landed_state"],
                declared_landing_z=target_z, home_xy=self.protocol.mission.home.model_dump(mode="json"),
                speed_mps=speed, angular_speed_rad_s=angular_speed, fresh=fresh, eligible=eligible,
                consecutive_qualified_samples=report["consecutive_qualified_samples"],
            )
            if not eligible:
                self._landing_event("completion_not_qualified",
                                    reason="height_position_or_motion_requires_descent")
                return False
            if report["consecutive_qualified_samples"] >= 3:
                self._landing_event("disarm_requested", method="armDisarm", args=[False, self.vehicle_name])
                self.disarm()
                acknowledged = self._disarm_acknowledgement is not None
                self._landing_event("disarm_result", result=self._last_disarm_result,
                                    acknowledged=acknowledged)
                if not acknowledged:
                    raise AdapterError("landing disarm was not acknowledged")
                report["disarm_acknowledged"] = True
            return True
        except Exception as exc:
            self._landing_event("landing_handshake_error", error_type=type(exc).__name__, error=str(exc))
            self._landing_active = False
            raise

    def hold(self, duration_s: float) -> None:
        """Hover in place for ``duration_s`` of simulated time (advances the clock)."""
        self.issue_hold()
        remaining = float(duration_s)
        dt = min(self.protocol.mission.control_dt_s, remaining)
        while remaining > 1e-9:
            advance = min(dt, remaining)
            self._advance_time(advance)
            remaining -= advance

    def land(self, timeout_s: float | None = None) -> None:
        """Land, bounded by ``timeout_s`` (advances the clock)."""
        budget = float(timeout_s if timeout_s is not None else 40.0)
        self.issue_land(budget)
        self._advance_until(lambda st: st.landed, budget, "land")

    def _advance_until(self, predicate: Callable[[VehicleState], bool], budget_s: float, what: str,
                       resend: Callable[[], None] | None = None) -> None:
        """Advance the clock in control steps until ``predicate(state)`` holds or the budget runs out."""
        dt = max(self.protocol.mission.control_dt_s, 0.05)
        elapsed = 0.0
        resend_interval = max(dt, self.simulation.max_command_duration_s * 0.8)
        since_resend = 0.0
        while elapsed < budget_s:
            self._advance_time(dt)
            elapsed += dt
            since_resend += dt
            state = self.sample_state()
            if predicate(state):
                return
            if resend is not None and since_resend >= resend_interval:
                resend()
                since_resend = 0.0
        raise AdapterTimeout(
            f"the vehicle did not {what} within {budget_s:g} s of simulated time",
            remedy=(
                "Check API control, the flight controller state, and whether an obstacle blocks the "
                "path. Record the episode as incomplete; it is not a pass."
            ),
        )

    def set_start_pose(self, position: Vec3, yaw_rad: float) -> None:
        """Place the vehicle before arming; flush queued live transforms and verify physical readback.

        Upstream queues the rendered transform while paused. A successful RPC alone does not prove
        the next physics sample starts at the declared pose. This bounded setup step is recorded
        separately from flight and cannot masquerade as a flown trajectory.
        """
        from colosseum_assurance.control.camera_geometry import unit_quaternion

        pose = [[float(position.x), float(position.y), float(position.z)],
                self._quaternion_from_yaw(float(yaw_rad))]
        report = {"verified": False, "requested_pose_local_ned": pose,
                  "phase": "pre_arm_setup_placement_not_flight", "raw_rpc": [],
                  "position_tolerance_m": .01, "rotation_tolerance_rad": .01,
                  "requested_flush_s": .01,
                  "settlement_speed_limit_mps": .02, "native_takeoff_speed_limit_mps": .05,
                  "settlement_required_consecutive_samples": 3,
                  "settlement_step_s": .01, "settlement_sim_budget_s": 2.0,
                  "settlement_wall_budget_s": 30.0, "settlement_samples": [],
                  "collision_read_semantics": "getCollisionInfoAndReset consumes has_collided; "
                                              "timestamp and object remain as event history"}
        self.start_pose_report = report
        settle_deadline: float | None = None
        support = (self._bound_actor_names.get(self.scene_launch_platform.name,
                   "colassure-"+self.scene_launch_platform.name)
                   if self.scene_launch_platform is not None else None)
        baseline_contact_stamp = 0
        support_event = None

        def read(method: str, *args: Any) -> Any:
            row = {"method": method, "args": list(args)}
            report["raw_rpc"].append(row)
            remaining = (None if settle_deadline is None else settle_deadline-time.monotonic())
            if remaining is not None and remaining <= 0:
                raise AdapterTimeout("preflight settlement exhausted its wall-clock budget")
            result = self._call(method, *args, timeout_s=(None if remaining is None else
                                min(self.endpoint.rpc_timeout_s, remaining)))
            row["result"] = result
            return result

        try:
            if not self.is_fixture_fake and read("simIsPaused") is not True:
                raise AdapterError("start-pose placement requires an actually paused simulator")
            if not self.is_fixture_fake:
                reused_acknowledgement = self._disarm_acknowledgement is not None
                if self._disarm_acknowledgement is None:
                    if read("armDisarm", False, self.vehicle_name) is not True:
                        raise AdapterError("start-pose placement requires acknowledged disarming")
                    self._disarm_acknowledgement = {
                        "method": "armDisarm", "args": [False, self.vehicle_name], "result": True,
                        "source": "start_pose_explicit_disarm",
                    }
                self._armed = False
                report["disarm_acknowledged_before_placement"] = True
                report["disarm_acknowledgement"] = dict(self._disarm_acknowledgement)
                report["disarm_acknowledgement_reused"] = reused_acknowledgement
                if support is not None:
                    baseline_contact = self._struct(read("simGetCollisionInfo", self.vehicle_name),
                                                    COLLISION_ORDER, "pre-placement contact baseline")
                    baseline_contact_stamp = self._as_stamp_ns(baseline_contact["time_stamp"]) or 0
                    report["pre_placement_collision"] = baseline_contact
            read("simSetVehiclePose", pose, True, self.vehicle_name)
            if self.is_fixture_fake:
                report.update(verified=True, method="fixture_immediate_placement_not_live_evidence",
                              requested_flush_s=0.0)
                return
            self._advance_time(.01)
            report["measured_flush_s"] = self.last_measured_advance_s
            if read("simIsPaused") is not True:
                raise AdapterError("start-pose flush did not return to a paused simulator")
            rendered = self._struct(read("simGetVehiclePose", self.vehicle_name), POSE_ORDER,
                                    "start-pose rendered pose")
            physical = self._struct(read("simGetGroundTruthKinematics", self.vehicle_name),
                                    KINEMATICS_ORDER, "start-pose physics")
            errors, angles = [], []
            desired_q = unit_quaternion(pose[1])
            for observed in (rendered, physical):
                xyz = self._vec3(observed["position"], "start-pose measured position")
                if not all(math.isfinite(v) for v in xyz.as_tuple()):
                    raise AdapterError("start-pose readback returned a nonfinite position")
                q_fields = self._struct(observed["orientation"], QUATERNIONR_ORDER, "start attitude")
                q = unit_quaternion([q_fields[key] for key in QUATERNIONR_ORDER])
                errors.append(xyz.distance_to(position))
                angles.append(2*math.acos(min(1.0, abs(float(np.dot(q, desired_q))))))
            report.update(position_errors_m=errors, orientation_errors_rad=angles)
            if max(errors) > .01 or max(angles) > .01:
                raise AdapterError("queued start pose was not physically applied within setup tolerance")
            # A correct rendered pose can still be falling a few millimetres onto its support.
            # Native takeoff rejects speed >0.05m/s; settle DISARMED below0.02 with repeated measured
            # state, retaining every advance and physical contact instead of hiding the motion.
            settle_deadline = time.monotonic()+30.0
            consecutive = 0
            last_stamp = None
            measured_settlement = 0.0
            report["required_support_actor"] = support
            for index in range(200):
                state = self._struct(read("getMultirotorState", self.vehicle_name),
                                     MULTIROTOR_STATE_ORDER, "preflight onboard state")
                estimated = self._struct(state["kinematics_estimated"], KINEMATICS_ORDER,
                                         "preflight onboard kinematics")
                physical = self._struct(read("simGetGroundTruthKinematics", self.vehicle_name),
                                        KINEMATICS_ORDER, "preflight physical kinematics")
                collision = self._struct(read("simGetCollisionInfo", self.vehicle_name),
                                         COLLISION_ORDER, "preflight support contact")
                positions, speeds = [], []
                for observed in (estimated, physical):
                    xyz = self._vec3(observed["position"], "settlement measured position")
                    velocity = self._vec3(observed["linear_velocity"], "settlement measured velocity")
                    if not all(math.isfinite(v) for v in (*xyz.as_tuple(), *velocity.as_tuple())):
                        raise AdapterError("preflight settlement returned nonfinite state")
                    q_fields = self._struct(observed["orientation"], QUATERNIONR_ORDER, "settle attitude")
                    q = unit_quaternion([q_fields[key] for key in QUATERNIONR_ORDER])
                    angle = 2*math.acos(min(1.0, abs(float(np.dot(q, desired_q)))))
                    if xyz.distance_to(position) > .01 or angle > .01:
                        raise AdapterError(
                            "preflight settlement drifted outside declared start-pose tolerance")
                    positions.append(xyz.model_dump(mode="json"))
                    speeds.append(math.dist(velocity.as_tuple(), (0, 0, 0)))
                stamp = self._as_stamp_ns(state["timestamp"])
                fresh = stamp is not None and (last_stamp is None or stamp > last_stamp)
                contact = bool(collision["has_collided"])
                contact_stamp = self._as_stamp_ns(collision["time_stamp"])
                new_contact = contact_stamp is not None and contact_stamp > baseline_contact_stamp
                wrong_contact = support is not None and (contact or new_contact) and (
                    str(collision["object_name"]) != support)
                if support is not None and support_event is None and contact and new_contact and (
                        str(collision["object_name"]) == support):
                    support_event = {"settlement_sample_index": index, "collision": dict(collision)}
                    report["observed_support_impact_after_placement"] = support_event
                # This RPC consumes the event flag; its later False says nothing about detachment.
                # Retain a fresh observed pad impact while requiring independent stable physics.
                support_ok = support is None or support_event is not None
                quiet = max(speeds) <= .02 and int(state["landed_state"]) == LANDED_STATE_LANDED
                consecutive = consecutive+1 if fresh and quiet and support_ok and not wrong_contact else 0
                report["settlement_samples"].append({
                    "index": index, "onboard_timestamp_ns": stamp, "positions_local_ned_m": positions,
                    "onboard_and_physics_speed_mps": speeds, "landed_state": state["landed_state"],
                    "collision": collision, "support_matches": support_ok,
                    "observed_support_impact_retained": support_event is not None,
                    "new_collision_since_placement_baseline": new_contact,
                    "wrong_support_contact": wrong_contact,
                    "fresh_after_previous_step": fresh, "consecutive_qualified_samples": consecutive,
                    "measured_settlement_elapsed_s": measured_settlement,
                })
                if wrong_contact:
                    raise AdapterError("preflight settlement observed contact with an unexpected body")
                if read("simIsPaused") is not True:
                    raise AdapterError("preflight settlement lost simulator pause")
                if consecutive >= 3:
                    report.update(verified=True, measured_settlement_s=measured_settlement,
                                  final_positions_local_ned_m=positions,
                                  method="disarmed_bounded_placement_then_measured_support_settlement")
                    break
                last_stamp = stamp
                remaining = settle_deadline-time.monotonic()
                if remaining <= 0 or measured_settlement >= 2.0 or index == 199:
                    raise AdapterTimeout(
                        "preflight settlement did not establish stable support within budget")
                original_budget = self.advance_budget_s
                self.advance_budget_s = min(self._advance_budget_s(.01), remaining)
                try:
                    self._advance_time(.01)
                finally:
                    self.advance_budget_s = original_budget
                advance = self.last_measured_advance_s
                report["settlement_samples"][-1]["following_measured_advance_s"] = advance
                if advance is None or not math.isfinite(advance) or advance <= 0:
                    raise AdapterError("preflight settlement clock did not measurably advance")
                measured_settlement += advance
        except Exception as exc:
            report.update(error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            if self.frames_dir is not None:
                directory = self.frames_dir.parent/"start_pose_qualification"
                directory.mkdir(parents=True, exist_ok=True)
                # repr fallback is only for rejected/nonstandard RPC diagnostics, never geometry.
                payload = json.dumps(report, sort_keys=True, default=repr)+"\n"
                digest = hashlib.sha256(payload.encode()).hexdigest()
                path = directory/f"{digest[:16]}.json"
                path.write_text(payload)
                report.update(evidence_path=str(path), evidence_sha256="sha256:"+digest)

    # ================================================================== observation
    def sample_sensors(self) -> dict[str, Any]:
        """Read physical sensor RPCs independently; unavailable sensors remain explicit."""
        from colosseum_assurance.sim.sensors import decode_gps, decode_imu

        result: dict[str, Any] = {}
        for name, method, decoder in (("imu", "getImuData", decode_imu), ("gps", "getGpsData", decode_gps)):
            try:
                sample = decoder(self._call(method, "", self.vehicle_name))
                sample["sim_time_s"] = self._to_episode_time(sample["capture_timestamp_ns"])
                result[name] = {"available": True, "sample": sample}
            except (AdapterError, ValueError, TypeError, KeyError) as exc:
                result[name] = {"available": False, "error": str(exc)[:300]}
        return result

    def configure_asset_segmentation(self, mesh_name: str, object_id: int = 42) -> dict[str, Any]:
        """Establish a unique mask ID through pinned public APIs, or mark labels unavailable.

        This changes only the evaluator's segmentation render. The controller
        never sees masks or mesh identity. A map must expose the exact mesh name.
        """
        requested_name = mesh_name
        mesh_name = self._bound_actor_names.get(mesh_name, mesh_name)
        try:
            cleared = self._call("simSetSegmentationObjectID", ".*", 0, True)
            assigned = self._call("simSetSegmentationObjectID", mesh_name, object_id, False)
            readback = self._call("simGetSegmentationObjectID", mesh_name)
            verified = cleared is True and assigned is True and readback == object_id
            return {"identity_verified": verified, "mesh_name": mesh_name, "object_id": object_id,
                    "manifest_asset_name": requested_name,
                    "method": "clear_all_ids_then_assign_exact_mesh_and_readback",
                    "palette_sha256": "c4377b4fe7e33863a68dee6e714729b6715aaa22f544ffb38dab7e9fd610d2d5"}
        except AdapterError as exc:
            return {"identity_verified": False, "mesh_name": mesh_name, "error": str(exc)[:300]}

    def sample_state(self) -> VehicleState:
        """Onboard state estimate. The ``collision`` field of MultirotorState is deliberately ignored.

        ``sim_time_s`` is ``MultirotorState.timestamp``, which upstream fills from the simulator clock
        in nanoseconds (MultirotorApiBase.hpp:134-140). It is never a locally accumulated value.
        """
        state = self._multirotor_state()
        kinematics = self._struct(state["kinematics_estimated"], KINEMATICS_ORDER, "KinematicsState")
        onboard_q = self._struct(kinematics["orientation"], QUATERNIONR_ORDER, "onboard orientation")
        sim_time = self._sim_time_from_ns(state["timestamp"], what="MultirotorState.timestamp")
        observed = VehicleState(
            sim_time_s=sim_time,
            position=self._vec3(kinematics["position"], "KinematicsState.position"),
            velocity=self._vec3(kinematics["linear_velocity"], "KinematicsState.linear_velocity"),
            yaw_rad=self._yaw(kinematics["orientation"], "KinematicsState.orientation"),
            orientation_wxyz=tuple(float(onboard_q[key]) for key in QUATERNIONR_ORDER),
            landed=int(state["landed_state"]) == LANDED_STATE_LANDED,
            api_control_enabled=self._api_control_enabled,
            source="colosseum_multirotor_state",
        )
        if self.protocol.controlled_study is not None:
            angular = self._vec3(kinematics["angular_velocity"], "onboard angular velocity")
            requested = (self.start_pose_report or {}).get("requested_pose_local_ned")
            self._control_mode_sample = {
                "sim_time_s": sim_time, "contract": "reset_valid_disarm_ack_v1",
                "angular_speed_rps": math.dist(angular.as_tuple(), (0, 0, 0)),
                "declared_landing_z": None if requested is None else requested[0][2],
                "armed": self._armed, "api_control_enabled": self._api_control_enabled,
                "acknowledged_disarmed": self._disarm_acknowledgement is not None and not self._armed,
                "disarm_acknowledgement": copy.deepcopy(self._disarm_acknowledgement),
            }
        if (self._landing_active and self.landing_report is not None
                and self.landing_report.get("disarm_acknowledged") and observed.landed
                and not self.landing_report.get("subsequent_onboard_landed_observed")):
            self.landing_report["subsequent_onboard_landed_observed"] = True
            self._landing_event("subsequent_onboard_landed", timestamp_ns=state["timestamp"],
                                kinematics_estimated=kinematics, landed_state=state["landed_state"])
        return observed

    def control_mode_sample(self) -> dict[str, Any]:
        """Latest SAME onboard-state sample plus reset-valid actuator acknowledgment; no truth RPC."""
        sample = getattr(self, "_control_mode_sample", None)
        if sample is None:
            return {"available": False, "error": "no controlled-study onboard mode sample"}
        return {"available": True, "sample": copy.deepcopy(sample)}

    def sample_truth(self) -> TruthSample:
        """PRIVILEGED ground truth. Only the runner's ledger path may call this.

        TIMESTAMP PROVENANCE. ``simGetGroundTruthKinematics`` returns a ``KinematicsState``, which has
        no timestamp field at all: six vectors and nothing else (RpcLibAdaptorsBase.hpp:398-409 and
        PythonClient/airsim/types.py:517). The sample is therefore paired with a real clock read taken
        immediately before it (``MultirotorState.timestamp``, MultirotorApiBase.hpp:134-140), and the
        pairing gap is measured and reported through :meth:`timing_report`. Under paused stepping the
        world cannot move between the two calls, so the gap is zero by construction; under wall-clock
        stepping it is small but real, and it is recorded rather than denied. What this method never
        does is take the time from a locally accumulated float.
        """
        clock_before_ns = self._read_sim_clock_ns()
        sim_time = self._record_stamp(clock_before_ns)
        kinematics = self._struct(
            self._call("simGetGroundTruthKinematics", self.vehicle_name),
            KINEMATICS_ORDER, "KinematicsState (ground truth)",
        )
        collision = self._struct(
            self._call("simGetCollisionInfo", self.vehicle_name), COLLISION_ORDER, "CollisionInfo",
        )
        has_collided = bool(collision["has_collided"])
        stamp = int(collision["time_stamp"] or 0)
        if has_collided and stamp != self._last_collision_stamp:
            # CollisionInfo carries no counter, so distinct contacts are counted by their timestamp.
            self._collision_count += 1
            self._last_collision_stamp = stamp
        object_name = collision["object_name"]
        clearance = self._truth_clearance()
        clock_after_ns = self._read_sim_clock_ns()
        gap_s = max(0.0, (clock_after_ns - clock_before_ns) / NANOS_PER_SECOND)
        self.truth_pairing_gap_s = gap_s
        self.max_truth_pairing_gap_s = max(self.max_truth_pairing_gap_s, gap_s)
        self._record_stamp(clock_after_ns)
        return TruthSample(
            sim_time_s=sim_time,
            position=self._vec3(kinematics["position"], "ground truth position"),
            velocity=self._vec3(kinematics["linear_velocity"], "ground truth velocity"),
            yaw_rad=self._yaw(kinematics["orientation"], "ground truth orientation"),
            collision_active=has_collided,
            collision_count=self._collision_count,
            collision_object=str(object_name) if object_name else None,
            collision_penetration_m=float(collision["penetration_depth"]) if has_collided else None,
            # The ground-truth kinematics carry no landed flag, so this repeats the simulator's own
            # landed_state from the most recent state sample instead of inferring one from altitude.
            landed=self._last_landed,
            min_obstacle_clearance_m=clearance,
            source="fixture_fake_ground_truth" if self.is_fixture_fake else "simulator_ground_truth",
        )

    def _truth_clearance(self) -> float | None:
        """Obstacle clearance, only where it can be obtained honestly.

        The Colosseum RPC API exposes no distance-to-nearest-body query, so on a live server this stays
        ``None`` and the evaluator computes clearance from the scenario manifest instead of a guess.
        """
        if not self.is_fixture_fake:
            return None
        try:
            extras = self._client.call(FIXTURE_TRUTH_EXTRAS_METHOD, self.vehicle_name,
                                       timeout_s=self.endpoint.rpc_timeout_s)
        except (RpcError, RpcTransportError, RpcProtocolError, RpcTimeout):
            return None
        if isinstance(extras, dict) and "min_obstacle_clearance_m" in extras:
            return float(extras["min_obstacle_clearance_m"])
        return None

    def capture(self, kinds: tuple[str, ...] = ("rgb", "depth"), save_prefix: str | None = None
                ) -> dict[str, CapturedFrame]:
        """Capture RGB and/or depth frames in one ``simGetImages`` round trip.

        Images are requested uncompressed (``compress=False``): a PNG round trip would add a decoder
        dependency and hide pixel statistics that the live-readiness gate needs.
        """
        unknown = [kind for kind in kinds if kind not in SUPPORTED_CAPTURE_KINDS]
        if unknown:
            raise ValueError(f"unsupported capture kinds {unknown}; expected {SUPPORTED_CAPTURE_KINDS}")
        if not kinds:
            return {}
        requests: list[list[Any]] = []
        for kind in kinds:
            if kind == "rgb":
                requests.append([str(self.camera_name), IMAGE_TYPE_SCENE, False, False])
            elif kind == "segmentation":
                requests.append([str(self.camera_name), 5, False, False])
            else:
                # DepthPerspective (2), i.e. distance along the projection ray, because that is what
                # control/perception.py projects. See the module docstring: requesting DepthPlanar
                # here would misplace every obstacle away from the optical axis.
                requests.append([str(self.camera_name), IMAGE_TYPE_DEPTH_PERSPECTIVE, True, False])
        raw = self._call(
            "simGetImages", requests, self.vehicle_name, False,
            remedy=(
                f"Camera {self.camera_name!r} did not return images. Check the CameraDefaults/Cameras "
                "block in settings.json and that the renderer is not running headless without a GPU."
            ),
        )
        if not isinstance(raw, (list, tuple)) or len(raw) != len(requests):
            raise AdapterError(
                f"simGetImages returned {len(raw) if isinstance(raw, (list, tuple)) else raw!r} "
                f"responses for {len(requests)} requests",
                remedy="The simulator did not answer every image request; treat the step as degraded.",
            )
        out: dict[str, CapturedFrame] = {}
        for kind, response in zip(kinds, raw, strict=True):
            out[kind] = self._decode_image(kind, response, save_prefix)
        return out

    def _decode_image(self, kind: str, response: Any, save_prefix: str | None) -> CapturedFrame:
        fields = self._struct(response, IMAGE_RESPONSE_ORDER, "ImageResponse")
        # Absolute rendered camera poses are privileged QA only. Never copy them into FrameRef,
        # CapturedFrame or the controller's ObservationPacket: doing so bypasses localization faults.
        if self.frames_dir is not None:
            qa_path = self.frames_dir.parent / "privileged_camera_poses.jsonl"
            qa_path.parent.mkdir(parents=True, exist_ok=True)
            with qa_path.open("a") as stream:
                stream.write(json.dumps({
                    "kind": kind, "camera_name": self.camera_name,
                    "capture_timestamp_ns": fields["time_stamp"], "save_prefix": save_prefix,
                    "camera_position_local_ned_m": fields["camera_position"],
                    "camera_orientation_local_ned_wxyz": fields["camera_orientation"],
                    "role": "privileged_offline_QA_not_controller_input",
                }, sort_keys=True, allow_nan=False)+"\n")
        width = int(fields["width"])
        height = int(fields["height"])
        stamp = self._as_stamp_ns(fields["time_stamp"])
        if stamp is None:
            # Upstream stamps every response from the simulator clock (RenderRequest.cpp:172,
            # UnrealImageCapture.cpp:99). A server that does not leaves the ACQUISITION TIME UNKNOWN.
            # Reading the clock again would date the frame to the moment it was decoded, which turns an
            # arbitrarily old image into apparently fresh evidence for the guard. The frame is therefore
            # returned with no timestamp and marked unusable, and the runner drops it for that step.
            self.frames_without_timestamp += 1
            LOGGER.warning(
                "%s frame carried no simulator timestamp; its acquisition time stays unknown and the "
                "frame cannot be used as evidence",
                kind,
            )
            sim_time = None
        else:
            # Capture time only. The world clock is not moved by reading an image.
            sim_time = self._to_episode_time(stamp)
        array: np.ndarray | None = None
        if width <= 0 or height <= 0:
            message = str(fields["message"] or "")
            LOGGER.warning("empty %s frame from the simulator (%s)", kind, message)
        elif kind == "depth":
            values = np.asarray(fields["image_data_float"], dtype=np.float32)
            if values.size != width * height:
                raise AdapterError(
                    f"depth frame has {values.size} floats but {width}x{height} pixels were announced",
                    remedy="The camera returned a malformed depth image; treat the step as degraded.",
                )
            array = values.reshape(height, width)
        else:
            payload = np.frombuffer(as_bytes(fields["image_data_uint8"]), dtype=np.uint8)
            if payload.size == 0 or payload.size % (width * height) != 0:
                raise AdapterError(
                    f"RGB frame has {payload.size} bytes, which is not a whole number of "
                    f"{width}x{height} pixels",
                    remedy="The camera returned a malformed colour image; treat the step as degraded.",
                )
            channels = payload.size // (width * height)
            if channels not in (3, 4):
                raise AdapterError("colour frame must contain 3 or 4 channels")
            # Pinned Unreal RenderRequest.cpp:112–114 writes B,G,R bytes. Convert
            # both scene and segmentation images to the package's RGB convention.
            array = payload.reshape(height, width, channels)[:, :, :3][:, :, ::-1].copy()
        ref = self._frame_ref(kind, width, height, sim_time, array, save_prefix)
        return CapturedFrame(ref=ref, array=array)

    def _frame_ref(self, kind: str, width: int, height: int, sim_time: float | None,
                   array: np.ndarray | None, save_prefix: str | None) -> FrameRef:
        min_value = max_value = mean_value = nonzero = None
        if array is not None and array.size:
            values = array.astype(np.float64, copy=False)
            finite = values[np.isfinite(values)]
            if finite.size:
                min_value = float(finite.min())
                max_value = float(finite.max())
                mean_value = float(finite.mean())
                nonzero = float(np.count_nonzero(finite) / finite.size)
        path, pixels_as = self._save_frame(kind, array, save_prefix)
        return FrameRef(
            kind=kind,
            camera_name=str(self.camera_name),
            sim_time_s=None if sim_time is None else float(sim_time),
            acquisition_time_known=sim_time is not None,
            width=width,
            height=height,
            path=path,
            pixels_as=pixels_as,
            min_value=min_value,
            max_value=max_value,
            mean_value=mean_value,
            nonzero_fraction=nonzero,
        )

    def _save_frame(self, kind: str, array: np.ndarray | None, save_prefix: str | None
                    ) -> tuple[str | None, str]:
        if array is None or save_prefix is None or self.frames_dir is None:
            return None, "none"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        if kind in {"rgb", "segmentation"}:
            target = self.frames_dir / f"{save_prefix}_{kind}.png"
            from matplotlib import image as mpimage  # local import: only needed when saving

            mpimage.imsave(target, array)
            self.frames_saved += 1
            return str(target), "png"
        target = self.frames_dir / f"{save_prefix}_depth.npy"
        np.save(target, array)
        self.frames_saved += 1
        return str(target), "npy"

    # ================================================================== scene
    def list_scene_objects(self, name_regex: str = ".*") -> list[str]:
        raw = self._call("simListSceneObjects", str(name_regex),
                         remedy="simListSceneObjects failed; the level may still be loading.")
        if raw is None:
            return []
        return [str(name) for name in raw]

    def list_scene_objects_by_tag(self, tag_regex: str = ".*") -> list[str]:
        """List actors by their Unreal *tag* (client.py:570, RpcLibServerBase.cpp:367).

        A second, independent route to the same question as :meth:`list_scene_objects`: an actor may be
        renamed in the level while keeping its tag, and a mismatch between the two routes is itself
        evidence that the scene drifted.
        """
        raw = self._call("simListSceneObjectsByTag", str(tag_regex),
                         remedy="simListSceneObjectsByTag failed; the level may still be loading.")
        return [str(name) for name in (raw or [])]

    def scene_name(self) -> str:
        """Not available. The Colosseum RPC API at the pin exposes no level-name getter.

        ``simLoadLevel`` sets a level but nothing reads the current one back, so this would have to be
        guessed. A guessed scene name in ``SimulatorIdentity`` would be fabricated provenance, so the
        field stays ``None`` and this call fails loudly instead.
        """
        raise NotImplementedError(
            "no verified RPC returns the current level name at Colosseum commit "
            "84fc0c1c75bc73a0135ee80a325d470577c66c52 (simLoadLevel sets one, nothing reads it). "
            "Record the scene through scene_signature() and the manifest instead."
        )

    def set_visibility(self, level: str) -> dict[str, Any]:
        """Apply a scenario visibility level with the verified weather API.

        ``simEnableWeather`` (client.py:251, RpcLibServerBase.cpp:123) and ``simSetWeatherParameter``
        (client.py:261, RpcLibServerBase.cpp:127); ``WeatherParameter.Fog == 7`` from types.py. Only the
        two levels the manifests use are implemented; anything else raises instead of being invented.
        """
        settings = {"clear": 0.0, "reduced": 0.5}
        if level not in settings:
            raise NotImplementedError(
                f"visibility level {level!r} has no verified simulator mapping; the manifests define "
                f"only {sorted(settings)}. Add a mapping deliberately rather than guessing one."
            )
        self._call("simEnableWeather", True)
        self._call("simSetWeatherParameter", WEATHER_PARAMETER_FOG, float(settings[level]))
        return {"visibility": level, "weather_parameter": "Fog", "value": settings[level]}

    def calibrate_scene_frame(self) -> dict[str, Any]:
        """Measure global-object minus vehicle-local NED using a paused, bracketed vehicle pose.

        Pinned NedTransform.cpp:25-45 uses the same axis/rotation convention but different translation
        origins. PawnSimApi vehicle/camera/kinematics/collision use local NED; WorldSimApi object poses
        and spawning use global NED. This does not establish an independently initialized PX4 frame.
        """
        from colosseum_assurance.control.camera_geometry import (
            CameraMount,
            quaternion_product,
            rotation_matrix,
        )

        position_tolerance_m, rotation_tolerance_rad = 0.001, 0.001
        report: dict[str, Any] = {
            "verified": False, "vehicle_name": self.vehicle_name,
            "measured_utc": datetime.now(UTC).isoformat(),
            "canonical_frame": "vehicle_local_ned_m",
            "object_rpc_frame": "global_ned_m",
            "method": "paused_local_global_local_global_local_with_truth_crosscheck",
            "source_pin": "84fc0c1c75bc73a0135ee80a325d470577c66c52",
            "position_tolerance_m": position_tolerance_m,
            "rotation_tolerance_rad": rotation_tolerance_rad, "raw_rpc": [],
        }
        self.scene_frame_report = report
        self.camera_mount = None
        self.camera_hfov_rad = None

        def read(method: str, *args: Any) -> Any:
            row: dict[str, Any] = {"method": method, "args": list(args)}
            report["raw_rpc"].append(row)
            try:
                result = self._call(method, *args)
                row["result"] = result
                return result
            except Exception as exc:
                row.update(error_type=type(exc).__name__, error=str(exc))
                raise

        def pose(raw: Any) -> tuple[np.ndarray, np.ndarray]:
            value = self._struct(raw, POSE_ORDER, "scene-frame pose")
            p = np.array(self._vec3(value["position"], "scene-frame position").as_tuple())
            q_fields = self._struct(value["orientation"], QUATERNIONR_ORDER, "scene-frame quaternion")
            q = np.array([float(q_fields[key]) for key in QUATERNIONR_ORDER])
            if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
                raise AdapterError("scene-frame calibration returned nonfinite vehicle pose")
            norm = float(np.linalg.norm(q))
            if abs(norm-1.0) > 0.001:
                raise AdapterError("scene-frame calibration returned a non-unit quaternion")
            return p, q/norm

        try:
            if read("simIsPaused") is not True:
                raise AdapterError("scene-frame calibration requires an actually paused simulator")
            local0 = pose(read("simGetVehiclePose", self.vehicle_name))
            global0 = pose(read("simGetObjectPose", self.vehicle_name))
            camera0_raw = read("simGetCameraInfo", str(self.camera_name), self.vehicle_name, False)
            camera0_info = self._struct(camera0_raw, ("pose", "fov", "proj_mat"), "CameraInfo")
            camera0 = pose(camera0_info["pose"])
            local1 = pose(read("simGetVehiclePose", self.vehicle_name))
            global1 = pose(read("simGetObjectPose", self.vehicle_name))
            camera1_raw = read("simGetCameraInfo", str(self.camera_name), self.vehicle_name, False)
            camera1_info = self._struct(camera1_raw, ("pose", "fov", "proj_mat"), "CameraInfo")
            camera1 = pose(camera1_info["pose"])
            local2 = pose(read("simGetVehiclePose", self.vehicle_name))
            truth_raw = read("simGetGroundTruthKinematics", self.vehicle_name)
            truth = self._struct(truth_raw, KINEMATICS_ORDER, "scene-frame ground truth")
            truth_pose = pose([truth["position"], truth["orientation"]])
            positions = [local0[0], local1[0], local2[0], truth_pose[0]]
            drift = max(float(np.linalg.norm(value-local0[0])) for value in positions)
            rotations = [local0[1], global0[1], local1[1], global1[1], local2[1], truth_pose[1]]
            angle = max(2*math.acos(min(1.0, abs(float(np.dot(q, local0[1]))))) for q in rotations)
            offsets = [global0[0]-local0[0], global1[0]-local1[0]]
            offset_drift = float(np.linalg.norm(offsets[1]-offsets[0]))
            translation = np.mean(offsets, axis=0)
            reference_drift = (float(np.linalg.norm(translation-self._scene_frame_reference))
                               if self._scene_frame_reference is not None else 0.0)
            report.update(local_pose_drift_m=drift, orientation_difference_rad=angle,
                          translation_repeatability_m=offset_drift,
                          translation_change_since_session_start_m=reference_drift,
                          global_minus_local_ned_m=translation.tolist())
            if read("simIsPaused") is not True:
                raise AdapterError("scene-frame calibration lost simulator pause during measurement")
            if max(drift, offset_drift, reference_drift) > position_tolerance_m:
                raise AdapterError("scene-frame calibration found moving/stale poses or changed frame origin")
            if angle > rotation_tolerance_rad:
                raise AdapterError("scene-frame calibration orientations disagree; translation is unproven")
            fovs = [float(camera0_info["fov"]), float(camera1_info["fov"])]
            if not all(math.isfinite(fov) and 0 < fov < 180 for fov in fovs):
                raise AdapterError("camera calibration returned invalid horizontal FOV")
            fov_reference = self._camera_hfov_reference
            fov_change = max(abs(value-fovs[0]) for value in fovs + (
                [fov_reference] if fov_reference is not None else []))
            fov_declared_error = abs(fovs[0]-self.simulation.camera_hfov_deg)
            if fov_change > .001 or fov_declared_error > .2:
                raise AdapterError("camera horizontal FOV changed or disagrees with declared intrinsics")
            mounts = []
            for vehicle, camera in ((local0, camera0), (local1, camera1)):
                inverse_q = vehicle[1] * np.array([1, -1, -1, -1])
                mounts.append(CameraMount(
                    position_body_m=tuple(rotation_matrix(vehicle[1]).T @ (camera[0]-vehicle[0])),
                    orientation_body_wxyz=tuple(quaternion_product(inverse_q, camera[1])),
                ))
            reference_mounts = mounts + ([self._camera_mount_reference]
                                          if self._camera_mount_reference is not None else [])
            mount_position_difference = max(float(np.linalg.norm(
                np.asarray(m.position_body_m)-mounts[0].position_body_m)) for m in reference_mounts)
            mount_rotation_difference = max(2*math.acos(min(1.0, abs(float(np.dot(
                m.orientation_body_wxyz, mounts[0].orientation_body_wxyz))))) for m in reference_mounts)
            mount_evidence = {
                "camera_name": self.camera_name, "mount": mounts[0].to_dict(),
                "method": "paused_two_camera_vehicle_pose_pairs_rigid_relative_transform",
                "role": "fixed_sensor_calibration_shared_across_arms_not_live_world_pose",
                "position_repeatability_and_reset_change_m": mount_position_difference,
                "rotation_repeatability_and_reset_change_rad": mount_rotation_difference,
                "source_pin": report["source_pin"],
                "camera_hfov_deg": fovs[0], "declared_hfov_deg": self.simulation.camera_hfov_deg,
                "hfov_declaration_tolerance_deg": .2, "hfov_repeat_reset_tolerance_deg": .001,
                "hfov_declaration_difference_deg": fov_declared_error,
                "hfov_repeat_reset_difference_deg": fov_change,
            }
            mount_payload = json.dumps(mount_evidence, sort_keys=True, allow_nan=False)
            mount_evidence["sha256"] = "sha256:"+hashlib.sha256(mount_payload.encode()).hexdigest()
            report["camera_mount_calibration"] = mount_evidence
            if (mount_position_difference > position_tolerance_m
                    or mount_rotation_difference > rotation_tolerance_rad):
                raise AdapterError("camera mount changed or camera/vehicle poses are not synchronized")
            self.camera_mount = mounts[0]
            self._camera_mount_reference = mounts[0]
            self.camera_hfov_rad = math.radians(fovs[0])
            self._camera_hfov_reference = fovs[0]
            self._scene_frame_reference = translation.copy()
            report["verified"] = True
            return report
        except Exception as exc:
            report.update(error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            if self.frames_dir is not None:
                # Preserve rejected measurements too, before any episode can exist. This is diagnostic
                # evidence, never an affirmative qualification when verified is false.
                directory = self.frames_dir.parent / "scene_frame_calibration"
                directory.mkdir(parents=True, exist_ok=True)

                def json_safe(value: Any) -> Any:
                    if isinstance(value, float) and not math.isfinite(value):
                        return {"nonfinite_float": repr(value)}
                    if isinstance(value, dict):
                        return {key: json_safe(item) for key, item in value.items()}
                    if isinstance(value, (list, tuple)):
                        return [json_safe(item) for item in value]
                    return value

                payload = json.dumps(json_safe(report), sort_keys=True, indent=2, allow_nan=False)+"\n"
                digest = hashlib.sha256(payload.encode()).hexdigest()
                path = directory / f"{digest[:16]}.json"
                path.write_text(payload)
                report["evidence_path"] = str(path)
                report["evidence_sha256"] = "sha256:"+digest

    def scene_rpc_call(self, method: str, *params: Any, timeout_s: float | None = None) -> Any:
        """Translate only world-object scene RPCs; all vehicle/camera/collision APIs stay local."""
        positional = {"simGetObjectPose", "simSpawnObject", "simSetObjectPose"}
        if method not in positional:
            return self._call(method, *params, timeout_s=timeout_s)
        if not self.scene_frame_report or not self.scene_frame_report.get("verified"):
            self.calibrate_scene_frame()
        offset = np.array(self.scene_frame_report["global_minus_local_ned_m"])

        def shift(raw: Any, translation: np.ndarray) -> list[Any]:
            fields = self._struct(raw, POSE_ORDER, "scene object pose")
            xyz = np.array(self._vec3(fields["position"], "scene object position").as_tuple())
            q = self._struct(fields["orientation"], QUATERNIONR_ORDER, "scene object orientation")
            return [(xyz+translation).tolist(), [q[key] for key in QUATERNIONR_ORDER]]

        values = list(params)
        if method in {"simSpawnObject", "simSetObjectPose"}:
            pose_index = 2 if method == "simSpawnObject" else 1
            values[pose_index] = shift(values[pose_index], offset)
        result = self._call(method, *values, timeout_s=timeout_s)
        return shift(result, -offset) if method == "simGetObjectPose" else result

    def configure_scene(self, manifest: ScenarioManifest) -> dict[str, Any]:
        """Prepare the scene for one scenario realization.

        The declared scene mode either verifies existing actors, explicitly instantiates study actors,
        or binds measured map geometry. Every live mode applies the same strict geometry gate. Only
        the fixture fake, which has no level, loads geometry over a private RPC.
        """
        from colosseum_assurance.sim.scene import (  # local: avoids an import cycle
            apply_launch_platform,
            assert_scene_ready,
            bind_scene,
        )

        self._bound_actor_names = {}
        requested_hash = self._requested_manifest_hashes.setdefault(
            manifest.scenario_id, manifest.content_hash())
        input_hash = manifest.content_hash()
        manifest = apply_launch_platform(manifest, self.scene_launch_platform, self.protocol)
        recorded_contract_hash = self.scene_deployment_evidence["configuration"][
            "scene_geometry_contract_sha256"]
        if recorded_contract_hash is not None:
            path = self.geometry_contract_path
            current_hash = ("sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
                            if path is not None and path.is_file() else None)
            if current_hash != recorded_contract_hash:
                raise AdapterError("geometry contract changed after deployment configuration was captured",
                                   remedy="Freeze the new deployment configuration and rebuild the adapter.")

        def declaration(payload: dict[str, Any], effective: ScenarioManifest) -> dict[str, Any]:
            payload["deployment_configuration"] = self.scene_deployment_evidence
            payload["requested_manifest_hash"] = requested_hash
            payload["input_manifest_hash"] = input_hash
            payload["scene_frame_calibration"] = self.scene_frame_report
            payload["scenario_definition_changed"] = effective.content_hash() != requested_hash
            if payload["scenario_definition_changed"]:
                payload["bound_manifest_hash"] = effective.content_hash()
                payload["effective_manifest"] = effective.model_dump(mode="json")
            self.bound_manifest = effective
            return payload

        if self.is_fixture_fake:
            payload = json.dumps({
                "obstacles": [obstacle.model_dump(mode="json") for obstacle in manifest.obstacles],
                "start_position": manifest.start_position.model_dump(mode="json"),
                "start_yaw_rad": float(manifest.start_yaw_rad),
                "asset_position": manifest.asset_position.model_dump(mode="json"),
            })
            result = self._call(FIXTURE_LOAD_SCENE_METHOD, payload,
                                remedy="The fixture fake refused the scene payload.")
            self._time_origin_ns = None
            self._sim_time_s = 0.0
            self._api_control_enabled = False
            self._armed = False
            self._paused = False
            if self.simulation.stepping_mode == "paused_continue_for_time":
                self._try_enter_paused_mode()
            loaded = dict(result) if isinstance(result, dict) else {"mode": "fixture_loaded"}
            loaded.setdefault("mode", "fixture_loaded")
            loaded["provenance"] = "fixture_fake"
            loaded["warning"] = "fixture geometry: software test double, not experimental evidence"
            return declaration(loaded, manifest)
        self.calibrate_scene_frame()
        # Live path. `bind_scene` applies the configured scene mode and measures real transforms and
        # extents; `assert_scene_ready` then FAILS CLOSED. A name list is not geometry proof: a level
        # whose actors carry the right names at the wrong positions must not produce evidence
        # (independent review, "live scene integration completeness").
        #
        # STRICT BY DEFAULT. The geometry gate runs with `require_complete_measurement=True`, so an
        # actor whose extent could not be measured blocks the episode instead of passing on its name
        # and position alone. An evaluator's clearance figure is only as true as the extent it assumed.
        strict = {"require_complete_measurement": True}
        bind_kwargs = dict(strict) if self._accepts_strict(bind_scene) else {}
        bind_kwargs.update(max_inventory_probes=self.scene_inventory_max_probes,
                           inventory_wall_budget_s=self.scene_inventory_wall_budget_s,
                           inventory_exclusions=self.scene_inventory_exclusions)
        if self.scene_mode == "qualified_map":
            bind_kwargs["asset_actor_name"] = self.scene_asset_actor_name
        result = bind_scene(self, manifest, self.protocol, self.scene_mode, **bind_kwargs)
        if not self._accepts_strict(assert_scene_ready):
            raise AdapterError(
                "sim.scene.assert_scene_ready no longer accepts require_complete_measurement, so the "
                "strict geometry gate cannot be applied",
                remedy=(
                    "The scene module changed its contract. Re-check configure_scene against it "
                    "before running anything: a silently relaxed geometry gate would let an episode "
                    "fly against unmeasured obstacles."
                ),
            )
        verification = assert_scene_ready(result, **strict)
        if not bool(getattr(result, "ok", False)):
            # `ok` is false for reasons that the per-actor checks do not always raise on, for example
            # a derived map with zero usable obstacles. Fail closed on it too.
            raise AdapterError(
                f"scene binding in mode {result.mode!r} reported ok=False, so the level that would be "
                "flown is not the level the manifest describes",
                remedy=(
                    "Inspect the returned scene report, rebuild or re-derive the scene, and re-run. "
                    "Never relax this check to obtain a pilot."
                ),
            )
        payload = result.model_dump(mode="json")
        payload["mode"] = result.mode
        payload["ok"] = bool(result.ok)
        payload["geometry_provenance"] = result.geometry_provenance
        payload["require_complete_measurement"] = True
        payload["measurement_complete"] = bool(getattr(verification, "measurement_complete", False))
        payload["inventory_complete"] = bool(getattr(verification, "inventory_complete",
                                                     getattr(verification, "measurement_complete",
                                                             False)))
        payload["blocking_reasons"] = self._blocking_reasons(verification)
        self._bound_actor_names = {
            check.spec.obstacle_name: check.resolved_actor_name
            for check in verification.actors if check.resolved_actor_name is not None
        }
        payload["resolved_actor_names"] = dict(self._bound_actor_names)
        return declaration(payload, result.manifest)

    @staticmethod
    def _accepts_strict(function: Callable[..., Any]) -> bool:
        """True when ``function`` takes ``require_complete_measurement``.

        The scene module is maintained separately. Probing the signature means a renamed or removed
        strictness flag is detected here and reported, instead of being dropped by a stray kwarg.
        """
        import inspect

        try:
            return "require_complete_measurement" in inspect.signature(function).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtins have no signature
            return False

    @staticmethod
    def _blocking_reasons(verification: Any) -> list[str]:
        """Collect every reason the scene would block an episode, for the episode record."""
        reasons: list[str] = []
        for check in getattr(verification, "actors", []) or []:
            if getattr(check, "status", "ok") != "ok":
                reasons.append(f"{check.status}:{check.spec.obstacle_name}")
        for finding in getattr(verification, "extras", []) or []:
            if getattr(finding, "status", "") != "ok":
                reasons.append(f"{finding.status}:{finding.actor_name}")
        reasons.extend(str(caveat) for caveat in (getattr(verification, "caveats", []) or []))
        return reasons


def build_adapter(config: AppConfig, protocol: ProtocolConfig, *, frames_dir: Path | None = None
                  ) -> ColosseumAdapter:
    """Build and connect a :class:`ColosseumAdapter` from an :class:`AppConfig`.

    Refuses to return an adapter whose provenance is not allowed by the configuration, so a run that
    claims to need a live simulator can never silently proceed against the fixture fake.
    """
    attestation = None
    if config.simulator_attestation_path is not None:
        attestation = load_attestation(config.simulator_attestation_path)
    if frames_dir is None:
        frames_dir = config.paths.frames_dir(config.run_class, protocol.short_hash)
    adapter = ColosseumAdapter(
        config.endpoint, protocol, frames_dir=frames_dir, attestation=attestation,
        scene_mode=config.scene_mode,
        geometry_contract_path=config.scene_geometry_contract_path,
        scene_asset_actor_name=config.scene_asset_actor_name,
        scene_launch_platform=config.scene_launch_platform,
        scene_inventory_max_probes=config.scene_inventory_max_probes,
        scene_inventory_wall_budget_s=config.scene_inventory_wall_budget_s,
        camera_max_attitude_pairing_gap_s=config.camera_max_attitude_pairing_gap_s,
        scene_inventory_exclusions=config.scene_inventory_exclusions,
    )
    identity = adapter.connect()
    if config.require_live_simulator and not identity.is_live:
        adapter.close()
        raise AdapterError(
            f"configuration requires a live Colosseum but the endpoint {config.endpoint.description} "
            f"has provenance {identity.provenance!r}",
            remedy=(
                "Point COLASSURE_SIM_HOST/COLASSURE_SIM_PORT at a genuine Colosseum server (through an "
                "SSH tunnel) AND supply COLASSURE_SIM_ATTESTATION with the hash-verified package "
                "attestation, or set run_class=fixture for software tests. "
                + identity.qualification_note()
            ),
        )
    if identity.provenance == "fixture_fake" and not config.allow_fixture_fake:
        adapter.close()
        raise AdapterError(
            "the endpoint is the fixture fake simulator but this run class forbids it",
            remedy="Fixture data may never be recorded as pilot or held-out evidence.",
        )
    return adapter
