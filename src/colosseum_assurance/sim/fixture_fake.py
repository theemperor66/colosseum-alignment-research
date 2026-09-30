"""LOCAL SOFTWARE TEST FIXTURE that speaks the Colosseum msgpack-RPC wire protocol.

**This module is not a simulator of record. Nothing it produces is experimental evidence.**
It exists so the adapter, the runner, the monitors and the evaluator can be exercised end to end on a
laptop with no Unreal Engine. Every ground-truth sample it emits is tagged
``fixture_fake_ground_truth``, and it answers the private probe ``__colassure_fixture_fake__`` with
``True`` so :mod:`colosseum_assurance.sim.identity` downgrades provenance to ``fixture_fake``. The
live-readiness gate can therefore never pass against this server.

What it models, deliberately crudely:

* first-order velocity tracking of a commanded position (no aerodynamics, no attitude dynamics),
* axis-aligned-box obstacles with a sphere-vs-box contact test,
* a pinhole depth camera ray-cast against the same boxes plus a ground plane,
* an RGB image shaded from the same ray-cast, so depth and colour agree with each other,
* ``simPause`` / ``simContinueForTime`` semantics, which make stepping deterministic,
* bounded command timeouts, because upstream bounds them on the simulator clock (see below).

Wire encoding follows upstream exactly: structures are POSITIONAL ARRAYS ordered by the
``attribute_order`` lists in ``PythonClient/airsim/types.py`` at commit
84fc0c1c75bc73a0135ee80a325d470577c66c52, matching ``MSGPACK_DEFINE_ARRAY`` on the C++ side
(``AirLib/include/api/RpcLibAdaptorsBase.hpp``). ``encoding="map"`` is available only to prove that the
adapter's defensive decoder also accepts named maps.

Depth convention
----------------
``ImageType`` values are ``Scene=0``, ``DepthPlanar=1``, ``DepthPerspective=2``, ``DepthVis=3``
(``AirLib/include/common/ImageCaptureBase.hpp:19-25``). Upstream ``docs/image_apis.md:226`` states that
``DepthPlanar`` is "depth in camera plane" and ``DepthPerspective`` is "depth from camera using a
projection ray that hits that pixel". This fixture honours the requested type: type 1 returns the
distance along the optical axis, type 2 returns the distance along the projection ray. The study path
requests type 2, which is what ``control/perception.py`` assumes.

Command timeouts
----------------
``moveToPosition`` delegates to ``moveOnPath`` (``MultirotorApiBase.cpp:470-476``), whose control loop
is bounded by ``Waiter waiter(getCommandPeriod(), timeout_sec, getCancelToken())``
(``MultirotorApiBase.cpp:340``) and stops when ``waiter.sleep()`` reports a timeout
(``MultirotorApiBase.cpp:361-362``); ``Waiter::isTimeout`` compares against the SIMULATOR clock
(``Waiter.hpp:57-62,73-75``). ``takeoff`` (``MultirotorApiBase.cpp:36-44``) and ``land``
(``MultirotorApiBase.cpp:52-77``) are bounded the same way. This fixture therefore expires a command
``timeout_sec`` of simulated seconds after it was issued, so a client that issues one bounded command
and then waits cannot silently rely on a command that upstream would already have dropped.

Deliberately NON-CONFORMING modes (opt-in, default off)
-------------------------------------------------------
``async_advance_delay_s``, ``ignore_continue_for_time``, ``pause_ignored`` and ``clock_quantum_s`` make
this fixture violate the verified upstream contract on purpose. Upstream ``simContinueForTime`` BLOCKS
until the physics world has paused itself again and a new frame was rendered
(``Unreal/Plugins/AirSim/Source/SimMode/SimModeWorldBase.cpp:101-118``), so these modes do not model
Colosseum. They exist only to prove that our client detects a non-conforming or unsupported server
instead of recording time that never passed.
"""

from __future__ import annotations

import json
import logging
import math
import re
import socketserver
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import msgpack
import numpy as np

from colosseum_assurance.config import EndpointConfig
from colosseum_assurance.scenario.manifest import ObstacleSpec
from colosseum_assurance.schemas import Vec3

LOGGER = logging.getLogger("colosseum_assurance.sim.fixture_fake")

FIXTURE_BANNER = "FIXTURE FAKE SIMULATOR - software test double, not experimental evidence"

REQUEST = 0
RESPONSE = 1
NOTIFY = 2

_LANDED = 0
_FLYING = 1


def default_obstacles(asset_position: Vec3 | None = None) -> list[ObstacleSpec]:
    """A minimal 3D scene: one inspection tower and one occluding wall in front of it."""
    asset = asset_position or Vec3(x=20.0, y=0.0, z=-7.0)
    return [
        ObstacleSpec(
            name="fixture_inspection_tower",
            kind="inspection_asset",
            center=Vec3(x=asset.x, y=asset.y, z=-7.0),
            extent=Vec3(x=1.2, y=1.2, z=7.0),
            unreal_actor_tag="InspectionTower",
        ),
        ObstacleSpec(
            name="fixture_occluding_wall",
            kind="wall",
            center=Vec3(x=asset.x - 9.0, y=asset.y + 1.0, z=-4.0),
            extent=Vec3(x=0.6, y=5.0, z=4.0),
            unreal_actor_tag="OccludingWall",
            occludes_asset=True,
        ),
    ]


@dataclass(slots=True)
class FixtureSceneGeometry:
    """Obstacle boxes plus a ground plane, in NED metres, prepared for vectorised ray casting."""

    obstacles: list[ObstacleSpec]
    ground_z: float = 0.0
    lower: np.ndarray = field(init=False)
    upper: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.obstacles:
            self.lower = np.array(
                [[o.center.x - o.extent.x, o.center.y - o.extent.y, o.center.z - o.extent.z]
                 for o in self.obstacles], dtype=np.float64)
            self.upper = np.array(
                [[o.center.x + o.extent.x, o.center.y + o.extent.y, o.center.z + o.extent.z]
                 for o in self.obstacles], dtype=np.float64)
        else:
            self.lower = np.zeros((0, 3))
            self.upper = np.zeros((0, 3))

    def names(self) -> list[str]:
        """Body names plus their Unreal actor tags, the way a level lists placed actors."""
        out: list[str] = []
        for obstacle in self.obstacles:
            out.append(obstacle.name)
            if obstacle.unreal_actor_tag and obstacle.unreal_actor_tag not in out:
                out.append(obstacle.unreal_actor_tag)
        return out

    def contact(self, position: Vec3, radius_m: float) -> tuple[str | None, float]:
        """Return (object name, penetration depth) for the deepest sphere-vs-box contact, if any.

        The ground plane is NOT included. Resting on the ground is landing, not a collision; the ground
        is handled separately in the physics step so that only a fast descent counts as an impact.
        """
        worst_name: str | None = None
        worst_depth = 0.0
        for obstacle in self.obstacles:
            distance = obstacle.surface_distance(position)
            penetration = radius_m - distance
            if penetration > worst_depth:
                worst_depth = penetration
                worst_name = obstacle.name
        return worst_name, worst_depth

    def clearance(self, position: Vec3, radius_m: float) -> float:
        """Smallest hull-to-obstacle surface distance (negative in contact); excludes the ground plane.

        This matches :meth:`ScenarioManifest.obstacle_clearance`, which the evaluator uses, so the two
        never disagree about what "clearance" means.
        """
        if not self.obstacles:
            return float("inf")
        return min(o.surface_distance(position) for o in self.obstacles) - radius_m


def _ray_cast(
    geometry: FixtureSceneGeometry,
    origin: np.ndarray,
    directions: np.ndarray,
    max_range_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ray/AABB + ground-plane cast.

    Returns ``(t, body_index)`` where ``t`` is the hit distance along each unit direction (``max_range``
    when nothing is hit) and ``body_index`` is the obstacle index, ``-1`` for the ground plane and
    ``-2`` for the sky.
    """
    n_rays = directions.shape[0]
    best_t = np.full(n_rays, float(max_range_m), dtype=np.float64)
    best_body = np.full(n_rays, -2, dtype=np.int32)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / directions
        for index in range(geometry.lower.shape[0]):
            t_low = (geometry.lower[index] - origin) * inv
            t_high = (geometry.upper[index] - origin) * inv
            t_near = np.nanmax(np.minimum(t_low, t_high), axis=1)
            t_far = np.nanmin(np.maximum(t_low, t_high), axis=1)
            t_entry = np.maximum(t_near, 0.0)
            hit = (t_far >= t_entry) & (t_far > 0.0) & (t_entry < best_t)
            best_t = np.where(hit, t_entry, best_t)
            best_body = np.where(hit, index, best_body)

        # Ground plane at z = ground_z, only reachable by rays pointing down (+z in NED).
        dz = directions[:, 2]
        t_ground = (geometry.ground_z - origin[2]) / dz
        ground_hit = (dz > 1e-9) & (t_ground > 0.0) & (t_ground < best_t)
        best_t = np.where(ground_hit, t_ground, best_t)
        best_body = np.where(ground_hit, -1, best_body)

    return np.clip(best_t, 0.0, float(max_range_m)), best_body


class FixtureFakeSimulator:
    """State and RPC behaviour of the fixture fake. Thread safe; one instance serves all connections."""

    def __init__(
        self,
        *,
        obstacles: Sequence[ObstacleSpec] | None = None,
        start_position: Vec3 | None = None,
        start_yaw_rad: float = 0.0,
        vehicle_radius_m: float = 0.35,
        velocity_tau_s: float = 0.35,
        yaw_tau_s: float = 0.4,
        arrival_tau_s: float = 0.8,
        physics_dt_s: float = 0.02,
        takeoff_altitude_m: float = 3.0,
        ground_impact_speed_mps: float = 2.0,
        image_width: int = 256,
        image_height: int = 144,
        max_range_m: float = 100.0,
        camera_hfov_deg: float = 90.0,
        advance_with_wall_clock: bool = True,
        encoding: Literal["array", "map"] = "array",
        hang_methods: Sequence[str] = (),
        fail_methods: dict[str, str] | None = None,
        async_advance_delay_s: float = 0.0,
        ignore_continue_for_time: bool = False,
        pause_ignored: bool = False,
        clock_quantum_s: float = 0.0,
        clock_epoch_ns: int = 0,
    ) -> None:
        self.geometry = FixtureSceneGeometry(list(obstacles) if obstacles is not None
                                             else default_obstacles())
        self.start_position = start_position or Vec3(x=0.0, y=0.0, z=0.0)
        self.start_yaw_rad = float(start_yaw_rad)
        self.vehicle_radius_m = float(vehicle_radius_m)
        self.velocity_tau_s = float(velocity_tau_s)
        self.yaw_tau_s = float(yaw_tau_s)
        self.arrival_tau_s = float(arrival_tau_s)
        self.physics_dt_s = float(physics_dt_s)
        self.takeoff_altitude_m = float(takeoff_altitude_m)
        self.ground_impact_speed_mps = float(ground_impact_speed_mps)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.max_range_m = float(max_range_m)
        self.camera_hfov_deg = float(camera_hfov_deg)
        self.advance_with_wall_clock = bool(advance_with_wall_clock)
        self.encoding = encoding
        self.hang_methods = frozenset(hang_methods)
        self.fail_methods = dict(fail_methods or {})
        # --- deliberately non-conforming knobs; all default to the upstream-conforming behaviour ----
        self.async_advance_delay_s = float(async_advance_delay_s)
        """When > 0, simContinueForTime returns AT ONCE and the world advances this many wall-clock
        seconds later on a background thread. Upstream blocks instead (SimModeWorldBase.cpp:101-118),
        so this is a fake defect used to prove the client waits for the clock it was promised."""
        self.ignore_continue_for_time = bool(ignore_continue_for_time)
        """When True, simContinueForTime is accepted and NOTHING happens: the server claims a step it
        never took. A client that trusts the return value would record time that never passed."""
        self.pause_ignored = bool(pause_ignored)
        """When True, simPause is accepted and the world stays unpaused, so simIsPaused disagrees."""
        self.clock_quantum_s = float(clock_quantum_s)
        """Quantises every advance upward to a whole number of quanta, the way a stepped clock does
        (SteppableClock advances by a fixed step per physics update: SteppableClock.hpp:19-45,
        World.hpp:43-45). With a quantum the world clock and a client-side sum of requested dt values
        measurably disagree, which is the point."""
        self.clock_epoch_ns = int(clock_epoch_ns)
        """Origin of the reported timestamps in nanoseconds. Upstream clocks do not start at zero
        (SteppableClock.hpp:24-28 starts at Unix-epoch nanos), so a non-zero value here proves the
        client rebases instead of assuming an origin."""

        self.lock = threading.RLock()
        self.call_log: list[str] = []
        self.last_image_requests: list[tuple[str, int, bool, bool]] = []
        self._async_threads: list[threading.Thread] = []
        self._stopping = threading.Event()
        self._reset_state()

    # ------------------------------------------------------------------ state
    def _reset_state(self) -> None:
        self.position = np.array(self.start_position.as_tuple(), dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.target = self.position.copy()
        self.command_speed = 0.0
        self.yaw_rad = self.start_yaw_rad
        self.target_yaw_rad = self.start_yaw_rad
        self.rotate_margin_rad = math.radians(5.0)
        self.sim_time_s = 0.0
        self.paused = False
        self.api_control = False
        self.armed = False
        self.landed = True
        self.has_collided = False
        self.collision_object = ""
        self.collision_penetration_m = 0.0
        self.collision_time_ns = 0
        self.collision_events = 0
        self.collision_position = self.position.copy()
        self._in_contact = False
        self.weather_enabled = False
        self.weather: dict[int, float] = {}
        self._wall_clock_ref = time.monotonic()
        # Upstream bounds every motion command on the simulator clock (Waiter.hpp:57-62), so the
        # fixture keeps a deadline per command instead of letting one command steer forever.
        self.command_deadline_s: float | None = None
        self.expired_commands = 0
        self.advance_calls = 0
        self.applied_advance_s = 0.0

    def _position_vec(self) -> Vec3:
        return Vec3(x=float(self.position[0]), y=float(self.position[1]), z=float(self.position[2]))

    def now_ns(self) -> int:
        """The fixture's OWN clock in nanoseconds, the only clock its samples ever carry."""
        return int(self.clock_epoch_ns + round(self.sim_time_s * 1e9))

    # ------------------------------------------------------------------ physics
    def _sync_wall_clock(self) -> None:
        """Advance an unpaused fixture with real time, the way a running simulator would."""
        now = time.monotonic()
        elapsed = now - self._wall_clock_ref
        self._wall_clock_ref = now
        if self.paused or not self.advance_with_wall_clock:
            return
        if elapsed > 0.0:
            self._advance(min(elapsed, 2.0))

    def _quantize(self, dt_s: float) -> float:
        """Round an advance up to a whole number of clock quanta, as a stepped clock does."""
        if self.clock_quantum_s <= 0.0:
            return float(dt_s)
        return math.ceil(dt_s / self.clock_quantum_s - 1e-9) * self.clock_quantum_s

    def _advance(self, dt_s: float) -> None:
        if dt_s <= 0.0:
            return
        dt_s = self._quantize(float(dt_s))
        substeps = max(1, int(math.ceil(dt_s / self.physics_dt_s)))
        step = dt_s / substeps
        for _ in range(substeps):
            self._substep(step)
        self.advance_calls += 1
        self.applied_advance_s += dt_s

    def _schedule_async_advance(self, seconds: float) -> None:
        """Apply an advance LATER, on a background thread. A deliberate contract violation."""

        def _run() -> None:
            if self._stopping.wait(self.async_advance_delay_s):
                return
            with self.lock:
                self._advance(seconds)
                self.paused = True
                self._wall_clock_ref = time.monotonic()

        thread = threading.Thread(target=_run, name="fixture-fake-async-advance", daemon=True)
        self._async_threads.append(thread)
        thread.start()

    def shutdown(self) -> None:
        """Stop pending background advances. Called by the server so tests leave no live threads."""
        self._stopping.set()
        for thread in list(self._async_threads):
            thread.join(timeout=2.0)
        self._async_threads.clear()

    def _expire_command(self) -> None:
        """Drop a command whose simulator-clock budget ran out, as the upstream loop does."""
        self.command_deadline_s = None
        self.expired_commands += 1
        self.target = self.position.copy()
        self.command_speed = 0.0

    def _substep(self, h: float) -> None:
        if self.command_deadline_s is not None and self.sim_time_s >= self.command_deadline_s:
            # Waiter::isTimeout uses >= on the SIMULATOR clock (Waiter.hpp:57-62), so the bound is
            # inclusive and is measured in simulated seconds, not wall-clock seconds.
            self._expire_command()
        to_target = self.target - self.position
        distance = float(np.linalg.norm(to_target))
        if distance > 1e-6 and self.command_speed > 0.0:
            speed = min(self.command_speed, distance / max(self.arrival_tau_s, h))
            desired = to_target / distance * speed
        else:
            desired = np.zeros(3)
        alpha = min(1.0, h / max(self.velocity_tau_s, 1e-6))
        self.velocity += (desired - self.velocity) * alpha

        yaw_error = math.atan2(math.sin(self.target_yaw_rad - self.yaw_rad),
                               math.cos(self.target_yaw_rad - self.yaw_rad))
        self.yaw_rad += yaw_error * min(1.0, h / max(self.yaw_tau_s, 1e-6))

        candidate = self.position + self.velocity * h
        self.sim_time_s += h

        # Ground: the floor stops the vehicle. Only a fast descent is recorded as an impact.
        descent_rate = float(self.velocity[2])
        if candidate[2] > self.geometry.ground_z:
            hard_landing = descent_rate > self.ground_impact_speed_mps
            candidate[2] = self.geometry.ground_z
            self.velocity[2] = 0.0
            if hard_landing:
                self._register_contact("fixture_ground_plane", descent_rate, candidate)

        candidate_vec = Vec3(x=float(candidate[0]), y=float(candidate[1]), z=float(candidate[2]))
        name, penetration = self.geometry.contact(candidate_vec, self.vehicle_radius_m)
        self.position = candidate
        if name is None:
            self._in_contact = False
        else:
            # Contact: stop at the obstacle and latch a collision record, as the upstream server does.
            self.velocity = np.zeros(3)
            self._register_contact(name, penetration, candidate)
        self.landed = bool(self.position[2] >= -0.15 and float(np.linalg.norm(self.velocity)) < 0.2)

    def _register_contact(self, name: str, penetration: float, position: np.ndarray) -> None:
        self.has_collided = True
        self.collision_object = name
        self.collision_penetration_m = float(penetration)
        self.collision_position = position.copy()
        if not self._in_contact:
            self.collision_events += 1
            # Upstream stamps the contact with the simulator clock (PawnSimApi.cpp:160) and the RPC
            # adaptor carries no collision counter (RpcLibAdaptorsBase.hpp:95), so the timestamp is
            # the only thing a client can use to tell two contacts apart.
            self.collision_time_ns = self.now_ns()
        self._in_contact = True

    # ------------------------------------------------------------------ rendering
    def _camera_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = np.array([math.cos(self.yaw_rad), math.sin(self.yaw_rad), 0.0])
        down = np.array([0.0, 0.0, 1.0])
        right = np.cross(down, forward)
        return forward, right, down

    def render(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(depth_planar, depth_perspective, rgb_uint8)`` rendered from the current pose.

        Both depth conventions come from the SAME ray cast, so they cannot drift apart:
        ``depth_planar`` is the distance along the optical axis (``ImageType.DepthPlanar``, 1) and
        ``depth_perspective`` is the distance along the projection ray (``ImageType.DepthPerspective``,
        2). See ``AirLib/include/common/ImageCaptureBase.hpp:19-25`` and upstream
        ``docs/image_apis.md:226`` for the two definitions.
        """
        width = max(int(width), 1)
        height = max(int(height), 1)
        forward, right, down = self._camera_basis()
        tan_half = math.tan(math.radians(self.camera_hfov_deg) / 2.0)
        us = (np.arange(width) + 0.5) / width * 2.0 - 1.0
        vs = (np.arange(height) + 0.5) / height * 2.0 - 1.0
        grid_u, grid_v = np.meshgrid(us * tan_half, vs * tan_half * (height / width))
        directions = (forward[None, None, :]
                      + grid_u[..., None] * right[None, None, :]
                      + grid_v[..., None] * down[None, None, :])
        directions = directions.reshape(-1, 3)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        t_hit, body = _ray_cast(self.geometry, self.position, directions, self.max_range_m)
        planar = t_hit * (directions @ forward)
        depth_planar = np.clip(planar, 0.0, self.max_range_m).reshape(height, width).astype(np.float32)
        depth_ray = np.clip(t_hit, 0.0, self.max_range_m).reshape(height, width).astype(np.float32)

        shade = np.clip(1.0 - t_hit / self.max_range_m, 0.05, 1.0)
        rgb = np.zeros((t_hit.shape[0], 3), dtype=np.float64)
        sky = body == -2
        ground = body == -1
        rgb[sky] = np.array([120.0, 160.0, 210.0])
        rgb[ground] = np.array([90.0, 80.0, 60.0]) * shade[ground, None] + 25.0
        for index, obstacle in enumerate(self.geometry.obstacles):
            mask = body == index
            if not mask.any():
                continue
            base = _colour_for(obstacle.name)
            rgb[mask] = base[None, :] * shade[mask, None] + 20.0
        image = np.clip(rgb, 0, 255).astype(np.uint8).reshape(height, width, 3)
        hour = getattr(self, "solar_hour", 12)
        light = 1.0 if hour == 12 else 0.55
        image = np.clip(image.astype(float) * light, 0, 255).astype(np.uint8)
        segmentation = np.zeros_like(image)
        ids = getattr(self, "segmentation_ids", {})
        for index, obstacle in enumerate(self.geometry.obstacles):
            if ids.get(obstacle.name) == 42:
                segmentation.reshape(-1, 3)[body == index] = [92, 31, 106]
        self.last_segmentation = segmentation
        return depth_planar, depth_ray, image

    # ------------------------------------------------------------------ rpc surface
    def dispatch(self, method: str, params: list[Any]) -> Any:
        handler = self._methods().get(method)
        if handler is None:
            raise _FixtureRpcError(f"rpc: method '{method}' not found")
        with self.lock:
            self.call_log.append(method)
            if method in self.fail_methods:
                raise _FixtureRpcError(self.fail_methods[method])
            self._sync_wall_clock()
            return handler(*params)

    def _methods(self) -> dict[str, Callable[..., Any]]:
        return {
            "__colassure_fixture_fake__": self.rpc_fixture_probe,
            "__colassure_load_scene__": self.rpc_load_scene,
            "__colassure_truth_extras__": self.rpc_truth_extras,
            "ping": self.rpc_ping,
            "getServerVersion": self.rpc_get_server_version,
            "getMinRequiredClientVersion": self.rpc_get_min_required_client_version,
            "getSettingsString": self.rpc_get_settings_string,
            "reset": self.rpc_reset,
            "enableApiControl": self.rpc_enable_api_control,
            "isApiControlEnabled": self.rpc_is_api_control_enabled,
            "armDisarm": self.rpc_arm_disarm,
            "takeoff": self.rpc_takeoff,
            "land": self.rpc_land,
            "hover": self.rpc_hover,
            "rotateToYaw": self.rpc_rotate_to_yaw,
            "moveToPosition": self.rpc_move_to_position,
            "getMultirotorState": self.rpc_get_multirotor_state,
            "getImuData": self.rpc_get_imu_data,
            "getGpsData": self.rpc_get_gps_data,
            "simSetSegmentationObjectID": self.rpc_set_segmentation_id,
            "simGetSegmentationObjectID": self.rpc_get_segmentation_id,
            "simGetGroundTruthKinematics": self.rpc_sim_get_ground_truth_kinematics,
            "simGetCollisionInfo": self.rpc_sim_get_collision_info,
            "simGetImages": self.rpc_sim_get_images,
            "simListSceneObjects": self.rpc_sim_list_scene_objects,
            "simListSceneObjectsByTag": self.rpc_sim_list_scene_objects_by_tag,
            "simEnableWeather": self.rpc_sim_enable_weather,
            "simSetWeatherParameter": self.rpc_sim_set_weather_parameter,
            "simSetTimeOfDay": self.rpc_sim_set_time_of_day,
            "simGetObjectPose": self.rpc_sim_get_object_pose,
            "simGetVehiclePose": self.rpc_sim_get_vehicle_pose,
            "simGetCameraInfo": self.rpc_sim_get_camera_info,
            "simSetVehiclePose": self.rpc_sim_set_vehicle_pose,
            "simPause": self.rpc_sim_pause,
            "simIsPaused": self.rpc_sim_is_paused,
            "simContinueForTime": self.rpc_sim_continue_for_time,
            "simContinueForFrames": self.rpc_sim_continue_for_frames,
        }

    # -- identity -------------------------------------------------------
    def rpc_fixture_probe(self) -> bool:
        """Answer the anti-self-deception probe. Always True: this is NOT a Colosseum server."""
        return True

    def rpc_load_scene(self, payload: str) -> dict[str, Any]:
        """Private RPC: replace the fixture world with one scenario realization.

        A genuine Colosseum answers this name with an RPC error, which is exactly what we want: scene
        geometry on a live server is built in the Unreal level and is verified, never injected.
        """
        document = json.loads(payload)
        obstacles = [ObstacleSpec.model_validate(item) for item in document.get("obstacles", [])]
        self.geometry = FixtureSceneGeometry(obstacles)
        start = document.get("start_position")
        if start is not None:
            self.start_position = Vec3.model_validate(start)
        self.start_yaw_rad = float(document.get("start_yaw_rad", self.start_yaw_rad))
        self._reset_state()
        LOGGER.info("%s loaded %d fixture obstacles", FIXTURE_BANNER, len(obstacles))
        return {"mode": "fixture_loaded", "obstacles": len(obstacles),
                "scene_objects": self.geometry.names()}

    def rpc_truth_extras(self, vehicle_name: str = "") -> dict[str, Any]:
        """Private RPC: privileged geometry facts the fixture can compute exactly.

        Only the fixture can answer this. On a live server the adapter must leave the corresponding
        ``TruthSample`` fields unset rather than estimate them.
        """
        return {
            "min_obstacle_clearance_m": float(
                self.geometry.clearance(self._position_vec(), self.vehicle_radius_m)
            ),
            "collision_events": int(self.collision_events),
        }

    def rpc_ping(self) -> bool:
        return True

    def rpc_get_server_version(self) -> int:
        return 1  # same constant the upstream server returns; it proves nothing, by design

    def rpc_get_min_required_client_version(self) -> int:
        return 1

    def rpc_get_settings_string(self) -> str:
        return json.dumps({
            "SettingsVersion": 1.2,
            "SimMode": "Multirotor",
            "ClockType": "SteppableClock",
            "LocalHostIp": "127.0.0.1",
            "ApiServerPort": 41451,
            "PhysicsEngineName": FIXTURE_BANNER,
            "Vehicles": {"Drone1": {"VehicleType": "SimpleFlight"}},
        })

    # -- session --------------------------------------------------------
    def rpc_reset(self) -> None:
        self._reset_state()
        return None

    def rpc_enable_api_control(self, is_enabled: bool, vehicle_name: str = "") -> None:
        self.api_control = bool(is_enabled)
        if not self.api_control:
            self.armed = False
        return None

    def rpc_is_api_control_enabled(self, vehicle_name: str = "") -> bool:
        return self.api_control

    def rpc_arm_disarm(self, arm: bool, vehicle_name: str = "") -> bool:
        if not self.api_control:
            raise _FixtureRpcError("armDisarm refused: API control is not enabled")
        self.armed = bool(arm)
        return True

    # -- motion ---------------------------------------------------------
    def _require_control(self, method: str) -> None:
        if not self.api_control:
            raise _FixtureRpcError(f"{method} refused: API control is not enabled")

    def rpc_takeoff(self, timeout_sec: float = 20.0, vehicle_name: str = "") -> bool:
        self._require_control("takeoff")
        self.target = np.array([self.position[0], self.position[1], -self.takeoff_altitude_m])
        self.command_speed = 2.0
        self.landed = False
        # Upstream takeoff is moveToPosition(..., timeout_sec, ...) (MultirotorApiBase.cpp:36-44),
        # so it expires like any other bounded command.
        self.command_deadline_s = self.sim_time_s + max(float(timeout_sec), 0.0)
        return True

    def rpc_land(self, timeout_sec: float = 60.0, vehicle_name: str = "") -> bool:
        self._require_control("land")
        self.target = np.array([self.position[0], self.position[1], 0.0])
        self.command_speed = 1.0
        # Upstream land runs a Waiter bounded by timeout_sec (MultirotorApiBase.cpp:52-77).
        self.command_deadline_s = self.sim_time_s + max(float(timeout_sec), 0.0)
        return True

    def rpc_rotate_to_yaw(
        self, yaw: float, timeout_sec: float = 3e38, margin: float = 5.0, vehicle_name: str = ""
    ) -> bool:
        """Rotate in place toward ``yaw`` DEGREES, holding the current position.

        Upstream `rotateToYaw(yaw, timeout_sec, margin, vehicle_name)` is bound in
        MultirotorRpcLibServer.cpp and implemented in MultirotorApiBase.cpp: it holds the start position
        and turns until `isYawWithinMargin`, which converts the current yaw to DEGREES before comparing,
        so both `yaw` and `margin` are degrees. The call is bounded by `timeout_sec`, like any other
        motion command here.
        """
        self._require_control("rotateToYaw")
        self.target = self.position.copy()
        self.command_speed = 0.0
        self.target_yaw_rad = math.radians(float(yaw))
        self.rotate_margin_rad = math.radians(abs(float(margin)))
        budget = float(timeout_sec)
        self.command_deadline_s = None if budget >= 1e30 else self.sim_time_s + max(budget, 0.0)
        return True

    def rpc_hover(self, vehicle_name: str = "") -> bool:
        self._require_control("hover")
        self.target = self.position.copy()
        self.command_speed = 0.0
        # A hover holds the heading it already has: upstream hover does not rotate.
        self.target_yaw_rad = self.yaw_rad
        # MultirotorApiBase::hover takes no timeout, so a hover never expires.
        self.command_deadline_s = None
        return True

    def rpc_move_to_position(
        self,
        x: float, y: float, z: float, velocity: float, timeout_sec: float = 5.0,
        drivetrain: int = 0, yaw_mode: Any = None, lookahead: float = -1.0,
        adaptive_lookahead: float = 1.0, vehicle_name: str = "",
    ) -> bool:
        """Set the commanded target. The clock is advanced by simContinueForTime, not by this call.

        The upstream server holds this request open until the move finishes or ``timeout_sec`` elapses.
        Our adapter always sends it without waiting, so returning at once is observationally equivalent
        and keeps the fixture deterministic.
        """
        self._require_control("moveToPosition")
        self.target = np.array([float(x), float(y), float(z)], dtype=np.float64)
        self.command_speed = max(float(velocity), 0.0)
        # The command steers for at most timeout_sec of SIMULATED time; after that the vehicle stops
        # being driven, exactly as the upstream moveOnPath loop does (MultirotorApiBase.cpp:340,
        # 361-362, 470-476; Waiter.hpp:57-62). Ignoring this would let one bounded command fly a whole
        # episode in the fixture while the same command expires on a real server.
        self.command_deadline_s = self.sim_time_s + max(float(timeout_sec), 0.0)
        if int(drivetrain) == 1:  # ForwardOnly
            delta = self.target - self.position
            if float(np.linalg.norm(delta[:2])) > 1e-3:
                self.target_yaw_rad = math.atan2(delta[1], delta[0])
        elif isinstance(yaw_mode, (list, tuple)) and len(yaw_mode) == 2 and not bool(yaw_mode[0]):
            self.target_yaw_rad = math.radians(float(yaw_mode[1]))
        return True

    # -- clock ----------------------------------------------------------
    def rpc_sim_pause(self, is_paused: bool) -> None:
        if self.pause_ignored:
            # Non-conforming server: the call is accepted and the world keeps running, so a client
            # that assumes simPause worked would step a world that is also moving by itself.
            return None
        self.paused = bool(is_paused)
        self._wall_clock_ref = time.monotonic()
        return None

    def rpc_sim_is_paused(self) -> bool:
        return self.paused

    def rpc_sim_continue_for_time(self, seconds: float) -> None:
        """Advance the world and stay paused, as the upstream server does when it returns.

        Upstream blocks until the physics world paused itself again and a new frame was rendered
        (SimModeWorldBase.cpp:101-118), so the default path here advances BEFORE returning. The two
        opt-in defects (``async_advance_delay_s``, ``ignore_continue_for_time``) deliberately break
        that contract so a test can prove the client checks the clock instead of trusting the reply.
        """
        if self.ignore_continue_for_time:
            self.paused = True
            return None
        if self.async_advance_delay_s > 0.0:
            self._schedule_async_advance(float(seconds))
            return None
        self._advance(float(seconds))
        self.paused = True
        self._wall_clock_ref = time.monotonic()
        return None

    def rpc_sim_continue_for_frames(self, frames: int) -> None:
        """Not implemented on purpose: this fixture has no frame counter to be honest about.

        Upstream implements it only for world-based SimModes (SimModeWorldBase.cpp:120-133); the base
        SimMode throws ``domain_error`` (SimModeBase.cpp:301-306). Refusing it here matches a server
        that does not support it, and is what the adapter must be able to survive.
        """
        raise _FixtureRpcError(
            "simContinueForFrames is not implemented by this fixture: it has no frame counter, and "
            "inventing one would make a frame-stepped episode unreproducible")

    # -- state ----------------------------------------------------------
    def _kinematics(self) -> Any:
        return self._struct(
            ["position", "orientation", "linear_velocity", "angular_velocity",
             "linear_acceleration", "angular_acceleration"],
            [self._vector3(self.position), self._quaternion(self.yaw_rad),
             self._vector3(self.velocity), self._vector3(np.zeros(3)),
             self._vector3(np.zeros(3)), self._vector3(np.zeros(3))],
        )

    def _collision_info(self) -> Any:
        return self._struct(
            ["has_collided", "penetration_depth", "time_stamp", "normal", "impact_point",
             "position", "object_name", "object_id"],
            [self.has_collided, self.collision_penetration_m, self.collision_time_ns,
             self._vector3(np.array([0.0, 0.0, -1.0])), self._vector3(self.collision_position),
             self._vector3(self.collision_position), self.collision_object,
             self.collision_events],
        )

    def rpc_get_multirotor_state(self, vehicle_name: str = "") -> Any:
        rc_data = self._struct(
            ["timestamp", "pitch", "roll", "throttle", "yaw", "left_z", "right_z", "switches",
             "vendor_id", "is_initialized", "is_valid"],
            [self.now_ns(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, "", False, False],
        )
        return self._struct(
            ["collision", "kinematics_estimated", "gps_location", "timestamp", "landed_state",
             "rc_data", "ready", "ready_message", "can_arm"],
            [self._collision_info(), self._kinematics(),
             self._struct(["latitude", "longitude", "altitude"], [0.0, 0.0, 0.0]),
             self.now_ns(), _LANDED if self.landed else _FLYING,
             rc_data, True, FIXTURE_BANNER, True],
        )

    def rpc_sim_get_ground_truth_kinematics(self, vehicle_name: str = "") -> Any:
        return self._kinematics()

    def rpc_sim_get_collision_info(self, vehicle_name: str = "") -> Any:
        return self._collision_info()

    def rpc_sim_get_vehicle_pose(self, vehicle_name: str = "") -> Any:
        return self._struct(["position", "orientation"],
                            [self._vector3(self.position), self._quaternion(self.yaw_rad)])

    def rpc_sim_set_vehicle_pose(self, pose: Any, ignore_collision: bool = True,
                                 vehicle_name: str = "") -> None:
        position, orientation = _unpack_pair(pose, ["position", "orientation"])
        values = _unpack_triple(position)
        self.position = np.array(values, dtype=np.float64)
        self.target = self.position.copy()
        self.command_deadline_s = None
        self.velocity = np.zeros(3)
        self.yaw_rad = _yaw_from_quaternion(orientation)
        self.target_yaw_rad = self.yaw_rad
        return None

    def rpc_sim_list_scene_objects(self, name_regex: str = ".*") -> list[str]:
        names = [*self.geometry.names(), "fixture_ground_plane", "Drone1", "PlayerStart"]
        try:
            pattern = re.compile(name_regex)
        except re.error as exc:
            raise _FixtureRpcError(f"simListSceneObjects: bad regex {name_regex!r}: {exc}") from exc
        return [name for name in names if pattern.search(name)]

    def rpc_sim_list_scene_objects_by_tag(self, tag_regex: str = ".*") -> list[str]:
        tags = [o.unreal_actor_tag for o in self.geometry.obstacles if o.unreal_actor_tag]
        try:
            pattern = re.compile(tag_regex)
        except re.error as exc:
            raise _FixtureRpcError(f"simListSceneObjectsByTag: bad regex {tag_regex!r}: {exc}") from exc
        return [tag for tag in tags if pattern.search(tag)]

    def rpc_sim_enable_weather(self, enable: bool) -> None:
        self.weather_enabled = bool(enable)
        return None

    def rpc_sim_set_weather_parameter(self, param: int, value: float) -> None:
        self.weather[int(param)] = float(value)
        return None

    def rpc_sim_set_time_of_day(self, enabled: bool, start: str, dst: bool,
                                speed: float, interval: float, move_sun: bool) -> None:
        self.solar_hour = int(start.split(" ")[1].split(":")[0]) if enabled else 12

    def rpc_sim_get_object_pose(self, object_name: str) -> Any:
        """Pose of any actor this fixture also LISTS, by body name or by its Unreal actor tag.

        A level answers ``simGetObjectPose`` for the names it returned from ``simListSceneObjects``,
        and this fixture lists both the body name and its tag (see :meth:`FixtureSceneGeometry.names`).
        Answering only one of the two would make a scene checker see an actor it cannot measure, which
        is a fixture artefact and not a property of any simulator. Unknown names still get the NaN pose
        that the pinned server returns for an actor missing from ``scene_object_map``
        (WorldSimApi.cpp:367-369).
        """
        if object_name == "Drone1":
            # This software fixture declares coincident vehicle-local and object-global origins.
            # Exposing the listed vehicle makes the production calibration exercise that declaration.
            return self.rpc_sim_get_vehicle_pose(object_name)
        for obstacle in self.geometry.obstacles:
            if object_name in (obstacle.name, obstacle.unreal_actor_tag):
                centre = np.array(obstacle.center.as_tuple())
                return self._struct(["position", "orientation"],
                                    [self._vector3(centre), self._quaternion(0.0)])
        nan = float("nan")
        return self._struct(["position", "orientation"],
                            [self._vector3(np.array([nan, nan, nan])), self._quaternion(0.0)])

    def rpc_get_imu_data(self, sensor_name: str = "", vehicle_name: str = "") -> Any:
        return self._struct(["time_stamp", "orientation", "angular_velocity", "linear_acceleration"],
                            [self.now_ns(), self._quaternion(self.yaw_rad),
                             self._vector3(np.zeros(3)), self._vector3(np.zeros(3))])

    def rpc_get_gps_data(self, sensor_name: str = "", vehicle_name: str = "") -> Any:
        point = self._struct(["latitude", "longitude", "altitude"],
                             [47 + self.position[0] / 111000, 8 + self.position[1] / 75000,
                              400 - self.position[2]])
        gnss = self._struct(["geo_point", "eph", "epv", "velocity", "fix_type", "time_utc"],
                            [point, 1.0, 1.0, self._vector3(self.velocity), 3, self.now_ns() // 1000])
        return self._struct(["time_stamp", "gnss", "is_valid"], [self.now_ns(), gnss, True])

    def rpc_set_segmentation_id(self, mesh_name: str, object_id: int, is_name_regex: bool = False) -> bool:
        import re

        ids = getattr(self, "segmentation_ids", {})
        matched = False
        for obstacle in self.geometry.obstacles:
            if (re.fullmatch(mesh_name, obstacle.name) if is_name_regex else mesh_name == obstacle.name):
                ids[obstacle.name] = int(object_id)
                matched = True
        self.segmentation_ids = ids
        return matched

    def rpc_get_segmentation_id(self, mesh_name: str) -> int:
        return getattr(self, "segmentation_ids", {}).get(mesh_name, -1)

    # -- images ---------------------------------------------------------
    def rpc_sim_get_camera_info(self, camera_name: str, vehicle_name: str = "",
                                external: bool = False) -> Any:
        # The software fixture renderer places its optical center at the vehicle origin.
        return self._struct(["pose", "fov", "proj_mat"],
                            [self.rpc_sim_get_vehicle_pose(vehicle_name), 90.0, []])

    def rpc_sim_get_images(self, requests: Any, vehicle_name: str = "", external: bool = False
                           ) -> list[Any]:
        if not isinstance(requests, (list, tuple)) or not requests:
            raise _FixtureRpcError("simGetImages: empty request list")
        out: list[Any] = []
        for raw in requests:
            camera_name, image_type, pixels_as_float, compress = _unpack_image_request(raw)
            if compress:
                raise _FixtureRpcError(
                    "simGetImages: the fixture fake renders uncompressed images only; "
                    "request compress=False"
                )
            width, height = self.image_width, self.image_height
            self.last_image_requests.append(
                (str(camera_name), int(image_type), bool(pixels_as_float), bool(compress)))
            depth_planar, depth_ray, rgb = self.render(width, height)
            timestamp = self.now_ns()
            if pixels_as_float:
                # 1 = DepthPlanar (optical-axis distance), 2 = DepthPerspective (ray distance);
                # ImageCaptureBase.hpp:19-25 and upstream docs/image_apis.md:226. Anything else would
                # have to be invented, so it is refused instead.
                if int(image_type) == 1:
                    depth = depth_planar
                elif int(image_type) == 2:
                    depth = depth_ray
                else:
                    raise _FixtureRpcError(
                        f"simGetImages: image_type {image_type} has no verified float depth meaning "
                        "in this fixture; request DepthPlanar (1) or DepthPerspective (2)"
                    )
                payload_uint8 = b""
                payload_float = [float(v) for v in depth.reshape(-1)]
            else:
                if int(image_type) == 5:
                    rgb = self.last_segmentation
                elif int(image_type) != 0:
                    raise _FixtureRpcError("fixture colour image type must be Scene or Segmentation")
                payload_uint8 = rgb[:, :, ::-1].copy().reshape(-1).tobytes()
                payload_float = []
            out.append(self._struct(
                ["image_data_uint8", "image_data_float", "camera_position", "camera_name",
                 "camera_orientation", "time_stamp", "message", "pixels_as_float", "compress",
                 "width", "height", "image_type"],
                [payload_uint8, payload_float, self._vector3(self.position), str(camera_name),
                 self._quaternion(self.yaw_rad), timestamp, FIXTURE_BANNER, bool(pixels_as_float),
                 False, width, height, int(image_type)],
            ))
        return out

    # -- encoding -------------------------------------------------------
    def _struct(self, order: list[str], values: list[Any]) -> Any:
        if self.encoding == "map":
            return dict(zip(order, values, strict=True))
        return list(values)

    def _vector3(self, values: Any) -> Any:
        return self._struct(["x_val", "y_val", "z_val"],
                            [float(values[0]), float(values[1]), float(values[2])])

    def _quaternion(self, yaw_rad: float) -> Any:
        return self._struct(["w_val", "x_val", "y_val", "z_val"],
                            [math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)])


class _FixtureRpcError(Exception):
    """Raised inside the fixture to produce a msgpack-RPC error response."""


def _colour_for(name: str) -> np.ndarray:
    """Deterministic per-body colour. Not hash() based: that is salted and would vary per process."""
    digest = sum(ord(character) * (index + 7) for index, character in enumerate(name))
    return np.array([80 + digest % 140, 60 + (digest // 7) % 150, 70 + (digest // 13) % 140],
                    dtype=np.float64)


def _unpack_pair(value: Any, order: list[str]) -> tuple[Any, Any]:
    if isinstance(value, dict):
        return value[order[0]], value[order[1]]
    return value[0], value[1]


def _unpack_triple(value: Any) -> tuple[float, float, float]:
    if isinstance(value, dict):
        return float(value["x_val"]), float(value["y_val"]), float(value["z_val"])
    return float(value[0]), float(value[1]), float(value[2])


def _yaw_from_quaternion(value: Any) -> float:
    if isinstance(value, dict):
        w, x, y, z = (float(value[k]) for k in ("w_val", "x_val", "y_val", "z_val"))
    else:
        w, x, y, z = (float(v) for v in value)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _unpack_image_request(raw: Any) -> tuple[str, int, bool, bool]:
    if isinstance(raw, dict):
        return (str(raw.get("camera_name", "0")), int(raw.get("image_type", 0)),
                bool(raw.get("pixels_as_float", False)), bool(raw.get("compress", True)))
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise _FixtureRpcError(
            "simGetImages: ImageRequest must be a 4-element array "
            "[camera_name, image_type, pixels_as_float, compress]"
        )
    return str(raw[0]), int(raw[1]), bool(raw[2]), bool(raw[3])


# ---------------------------------------------------------------------------- server
class _Handler(socketserver.BaseRequestHandler):
    """One TCP connection. Decodes msgpack-RPC requests and answers them from the simulator."""

    def handle(self) -> None:
        server: FixtureFakeServer = self.server  # type: ignore[assignment]
        simulator = server.simulator
        unpacker = msgpack.Unpacker(raw=False, strict_map_key=False, unicode_errors="surrogateescape")
        self.request.settimeout(1.0)
        while not server.stop_event.is_set():
            try:
                chunk = self.request.recv(262144)
            except TimeoutError:
                continue
            except OSError:
                return
            if not chunk:
                return
            unpacker.feed(chunk)
            for message in unpacker:
                if not isinstance(message, (list, tuple)) or not message:
                    return
                if message[0] == NOTIFY:
                    continue
                if message[0] != REQUEST or len(message) != 4:
                    return
                _, msgid, method, params = message
                method = str(method)
                if method in simulator.hang_methods:
                    # Deliberate black hole: accepted and never answered, so a test can prove that the
                    # client's per-call deadline really bounds the call. The connection keeps serving.
                    continue
                try:
                    result = simulator.dispatch(method, list(params or []))
                    payload = [RESPONSE, msgid, None, result]
                except _FixtureRpcError as exc:
                    payload = [RESPONSE, msgid, str(exc), None]
                except Exception as exc:  # noqa: BLE001 - mirror a server-side fault as an RPC error
                    payload = [RESPONSE, msgid, f"fixture fake internal error: {exc!r}", None]
                try:
                    self.request.sendall(msgpack.packb(payload, use_bin_type=True))
                except OSError:
                    return


class FixtureFakeServer(socketserver.ThreadingTCPServer):
    """Threaded TCP server for :class:`FixtureFakeSimulator`, bound to loopback on an ephemeral port."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, simulator: FixtureFakeSimulator, host: str = "127.0.0.1", port: int = 0) -> None:
        super().__init__((host, port), _Handler)
        self.simulator = simulator
        self.stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def endpoint(self) -> EndpointConfig:
        host, port = self.server_address[0], self.server_address[1]
        return EndpointConfig(
            host=str(host), port=int(port), label="fixture-fake",
            connect_timeout_s=5.0, rpc_timeout_s=10.0, reset_timeout_s=10.0,
        )

    def start(self) -> FixtureFakeServer:
        LOGGER.info("%s listening on %s", FIXTURE_BANNER, self.server_address)
        self._thread = threading.Thread(target=self.serve_forever, name="fixture-fake",
                                        kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.stop_event.set()
        self.simulator.shutdown()
        self.shutdown()
        self.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


@contextmanager
def fixture_fake_server(
    obstacles: Sequence[ObstacleSpec] | None = None,
    asset_position: Vec3 | None = None,
    start_position: Vec3 | None = None,
    image_size: tuple[int, int] = (256, 144),
    hfov_deg: float = 90.0,
    **kwargs: Any,
) -> Iterator[EndpointConfig]:
    """Run the fixture fake in-process and yield a loopback :class:`EndpointConfig` for it.

    Usage::

        with fixture_fake_server(obstacles=manifest.obstacles) as endpoint:
            adapter = ColosseumAdapter(endpoint, protocol)

    The endpoint is always ``127.0.0.1`` on an ephemeral port: the fixture is never exposed off-host.
    ``asset_position`` only moves the default inspection body; when ``obstacles`` is given, that list is
    the scene. The world can be reloaded per scenario over ``__colassure_load_scene__``.
    """
    if obstacles is None and asset_position is not None:
        obstacles = default_obstacles(asset_position)
    simulator = FixtureFakeSimulator(
        obstacles=obstacles,
        start_position=start_position,
        image_width=int(image_size[0]),
        image_height=int(image_size[1]),
        camera_hfov_deg=float(hfov_deg),
        **kwargs,
    )
    server = FixtureFakeServer(simulator).start()
    try:
        yield server.endpoint
    finally:
        server.stop()
