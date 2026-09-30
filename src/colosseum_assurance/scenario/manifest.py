"""Scenario manifests: the matched unit of the experiment.

A *scenario realization* fixes the scene geometry, the start state, and every exogenous disturbance
schedule. All three arms replay the same manifest, so matching survives trajectory divergence: the
runner only *indexes* precomputed schedules by step index or request ordinal and never draws fresh
random numbers during an episode (research-plan.md: "Precompute disturbances independently of
controller actions; simply reusing a random seed may not preserve matched conditions").
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.schemas import Box, StrictModel, Vec3

ObstacleKind = Literal["inspection_asset", "building", "mast", "tree", "wall", "ground_plane"]


class ObstacleSpec(StrictModel):
    """One scene body. Scene-construction and evaluator-side geometry only.

    The controller never receives this object; it must infer obstacles from depth or RGB frames
    (see :mod:`colosseum_assurance.control.perception`).
    """

    name: str
    kind: ObstacleKind
    center: Vec3
    extent: Vec3 = Field(description="Half-extents in metres (x, y, z) of an axis-aligned bounding box.")
    unreal_actor_tag: str | None = None
    occludes_asset: bool = False

    def as_box(self) -> Box:
        return Box(
            x_min=self.center.x - self.extent.x,
            x_max=self.center.x + self.extent.x,
            y_min=self.center.y - self.extent.y,
            y_max=self.center.y + self.extent.y,
            z_min=self.center.z - self.extent.z,
            z_max=self.center.z + self.extent.z,
        )

    def surface_distance(self, p: Vec3) -> float:
        """Distance from ``p`` to this body's surface; negative inside (approximate AABB distance)."""
        dx = max(abs(p.x - self.center.x) - self.extent.x, 0.0)
        dy = max(abs(p.y - self.center.y) - self.extent.y, 0.0)
        dz = max(abs(p.z - self.center.z) - self.extent.z, 0.0)
        outside = (dx * dx + dy * dy + dz * dz) ** 0.5
        if outside > 0.0:
            return outside
        inside = max(
            abs(p.x - self.center.x) - self.extent.x,
            abs(p.y - self.center.y) - self.extent.y,
            abs(p.z - self.center.z) - self.extent.z,
        )
        return inside


class ScheduleSet(StrictModel):
    """Precomputed exogenous disturbances for one scenario realization.

    Everything is indexed by control step or by request ordinal, never by a live RNG draw.
    """

    steps: int = Field(ge=1)
    dt_s: float = Field(gt=0.0)
    observation_delay_s: list[float]
    state_dropout: list[bool]
    depth_dropout: list[bool]
    heartbeat_times_s: list[float]
    supervision_outages_s: list[tuple[float, float]] = Field(default_factory=list)
    authorization_response_delay_s: list[float]
    authorization_decision: list[Literal["granted", "denied"]]
    authorization_validity_s: float
    visibility: str
    schedule_seed: int

    @model_validator(mode="after")
    def _lengths_match(self) -> ScheduleSet:
        for field in ("observation_delay_s", "state_dropout", "depth_dropout"):
            values = getattr(self, field)
            if len(values) != self.steps:
                raise ValueError(f"{field} must have exactly {self.steps} entries, got {len(values)}")
        if not self.authorization_response_delay_s:
            raise ValueError("at least one authorization response must be scheduled")
        if len(self.authorization_decision) != len(self.authorization_response_delay_s):
            raise ValueError("authorization decision and delay schedules must have equal length")
        for start, end in self.supervision_outages_s:
            if end <= start:
                raise ValueError("supervision outage windows must have positive duration")
        return self

    # ------------------------------------------------------------------ access
    def delay_at_step(self, step_index: int) -> float:
        return self.observation_delay_s[min(step_index, self.steps - 1)]

    def state_dropped(self, step_index: int) -> bool:
        return self.state_dropout[min(step_index, self.steps - 1)]

    def depth_dropped(self, step_index: int) -> bool:
        return self.depth_dropout[min(step_index, self.steps - 1)]

    def authorization_response(self, request_ordinal: int) -> tuple[float, str]:
        """Response delay and decision for the n-th authorization request (0-based).

        Keyed by ordinal, not by time, so two arms that request at different moments still receive the
        same scheduled treatment for their n-th request.
        """
        idx = min(request_ordinal, len(self.authorization_response_delay_s) - 1)
        return self.authorization_response_delay_s[idx], self.authorization_decision[idx]

    def supervision_available(self, sim_time_s: float) -> bool:
        """True when the supervisory link is up in the exogenous schedule at ``sim_time_s``."""
        return not any(start <= sim_time_s < end for start, end in self.supervision_outages_s)

    def last_heartbeat_before(self, sim_time_s: float) -> float | None:
        """The most recent scheduled heartbeat that was actually deliverable by ``sim_time_s``."""
        best: float | None = None
        for t in self.heartbeat_times_s:
            if t <= sim_time_s and self.supervision_available(t):
                best = t
        return best

    def content_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


class ScenarioManifest(StrictModel):
    """One reproducible scenario realization, shared by all arms."""

    schema_version: str = "1.0.0"
    scenario_id: str
    protocol_hash: str
    run_class: Literal["fixture", "pilot", "heldout", "smoke"]
    cell_id: str
    observation_delay_level: str
    supervision_delay_level: str
    layout_variant: str
    visibility: str
    realization_index: int
    seed: int
    start_position: Vec3
    start_yaw_rad: float
    asset_position: Vec3
    inspection_viewpoint: Vec3
    obstacles: list[ObstacleSpec]
    schedules: ScheduleSet
    scene_name: str = "CivilianInspectionYard"
    notes: str = ""

    def content_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

    def obstacle_clearance(self, p: Vec3) -> float:
        """Smallest surface distance from ``p`` to any scene body (privileged geometry)."""
        if not self.obstacles:
            return float("inf")
        return min(o.surface_distance(p) for o in self.obstacles)


def scenario_seed(protocol_hash: str, run_class: str, cell_id: str, realization_index: int) -> int:
    """Derive a stable 63-bit seed. Depends on protocol, run class, cell, realization -- never on arm."""
    key = f"{protocol_hash}|{run_class}|{cell_id}|{realization_index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") >> 1


def scenario_id(protocol_short_hash: str, run_class: str, cell_id: str, realization_index: int) -> str:
    return f"{run_class}-{protocol_short_hash}-{cell_id}-r{realization_index:03d}"


def _layout(variant: str, asset: Vec3, rng: np.random.Generator) -> list[ObstacleSpec]:
    """Build a small 3D obstacle layout around the inspection asset.

    Layouts differ in occlusion and clutter so that depth-based perception, not privileged state,
    determines what the controller can see.
    """
    obstacles: list[ObstacleSpec] = [
        ObstacleSpec(
            name="inspection_tower",
            kind="inspection_asset",
            center=Vec3(x=asset.x, y=asset.y, z=-7.0),
            extent=Vec3(x=1.2, y=1.2, z=7.0),
            unreal_actor_tag="InspectionTower",
        )
    ]
    jitter = lambda scale: float(rng.uniform(-scale, scale))  # noqa: E731 - local readability

    if variant in {"occluded", "cluttered"}:
        obstacles.append(
            ObstacleSpec(
                name="occluding_wall",
                kind="wall",
                center=Vec3(x=asset.x - 9.0 + jitter(1.0), y=1.0 + jitter(1.5), z=-4.0),
                extent=Vec3(x=0.6, y=5.0, z=4.0),
                unreal_actor_tag="OccludingWall",
                occludes_asset=True,
            )
        )
    if variant == "cluttered":
        obstacles.extend(
            [
                ObstacleSpec(
                    name="service_building",
                    kind="building",
                    center=Vec3(x=asset.x - 15.0 + jitter(1.5), y=-7.0 + jitter(1.5), z=-3.5),
                    extent=Vec3(x=3.0, y=3.0, z=3.5),
                    unreal_actor_tag="ServiceBuilding",
                ),
                ObstacleSpec(
                    name="guy_mast",
                    kind="mast",
                    center=Vec3(x=asset.x - 5.0 + jitter(0.8), y=-4.0 + jitter(0.8), z=-5.0),
                    extent=Vec3(x=0.35, y=0.35, z=5.0),
                    unreal_actor_tag="GuyMast",
                    occludes_asset=True,
                ),
                ObstacleSpec(
                    name="tree_row",
                    kind="tree",
                    center=Vec3(x=asset.x - 20.0 + jitter(2.0), y=6.0 + jitter(2.0), z=-3.0),
                    extent=Vec3(x=1.5, y=4.0, z=3.0),
                    unreal_actor_tag="TreeRow",
                ),
            ]
        )
    if variant == "open":
        obstacles.append(
            ObstacleSpec(
                name="far_shed",
                kind="building",
                center=Vec3(x=asset.x - 18.0 + jitter(2.0), y=-11.0 + jitter(2.0), z=-2.5),
                extent=Vec3(x=2.5, y=2.5, z=2.5),
                unreal_actor_tag="FarShed",
            )
        )
    return obstacles


def build_manifest(
    protocol: ProtocolConfig,
    run_class: str,
    cell_id: str,
    realization_index: int,
) -> ScenarioManifest:
    """Create one scenario realization deterministically from the protocol and its indices."""
    cells = {c["cell_id"]: c for c in protocol.cells()}
    if cell_id not in cells:
        raise KeyError(f"cell_id {cell_id!r} is not in the protocol; known cells: {sorted(cells)}")
    cell = cells[cell_id]
    obs_level = next(
        level for level in protocol.conditions.observation_delay_levels
        if level.level_id == cell["observation_delay_level"]
    )
    sup_level = next(
        level for level in protocol.conditions.supervision_delay_levels
        if level.level_id == cell["supervision_delay_level"]
    )

    phash = protocol.content_hash()
    seed = scenario_seed(phash, run_class, cell_id, realization_index)
    rng = np.random.default_rng(seed)

    layouts = protocol.conditions.layout_variants
    layout_variant = layouts[realization_index % len(layouts)]
    visibilities = protocol.conditions.visibility_levels
    visibility = visibilities[(realization_index // max(len(layouts), 1)) % len(visibilities)]

    mission = protocol.mission
    asset = mission.asset_nominal_position
    asset_jitter = Vec3(
        x=asset.x + float(rng.uniform(-1.0, 1.0)),
        y=asset.y + float(rng.uniform(-1.5, 1.5)),
        z=asset.z,
    )
    obstacles = _layout(layout_variant, asset_jitter, rng)
    approach_sign = 1.0 if rng.random() < 0.5 else -1.0
    viewpoint = Vec3(
        x=asset_jitter.x - mission.inspection_standoff_m,
        y=asset_jitter.y + approach_sign * float(rng.uniform(0.0, 1.0)),
        z=asset_jitter.z,
    )

    steps = int(round(mission.episode_horizon_s / mission.control_dt_s))
    schedules = build_schedules(
        protocol=protocol,
        steps=steps,
        obs_delay_value_s=obs_level.value_s,
        obs_jitter_s=obs_level.jitter_s,
        dropout_probability=obs_level.dropout_probability,
        supervision_delay_s=sup_level.value_s,
        supervision_jitter_s=sup_level.jitter_s,
        visibility=visibility,
        rng=rng,
        seed=seed,
    )

    return ScenarioManifest(
        scenario_id=scenario_id(protocol.short_hash, run_class, cell_id, realization_index),
        protocol_hash=phash,
        run_class=run_class,  # type: ignore[arg-type]
        cell_id=cell_id,
        observation_delay_level=obs_level.level_id,
        supervision_delay_level=sup_level.level_id,
        layout_variant=layout_variant,
        visibility=visibility,
        realization_index=realization_index,
        seed=seed,
        start_position=Vec3(x=mission.home.x, y=mission.home.y, z=mission.home.z),
        start_yaw_rad=0.0,
        asset_position=asset_jitter,
        inspection_viewpoint=viewpoint,
        obstacles=obstacles,
        schedules=schedules,
    )


def build_schedules(
    protocol: ProtocolConfig,
    steps: int,
    obs_delay_value_s: float,
    obs_jitter_s: float,
    dropout_probability: float,
    supervision_delay_s: float,
    supervision_jitter_s: float,
    visibility: str,
    rng: np.random.Generator,
    seed: int,
) -> ScheduleSet:
    """Draw every exogenous disturbance up front, before any control action exists."""
    dt = protocol.mission.control_dt_s
    obligations = protocol.obligations

    delays = np.clip(rng.normal(loc=obs_delay_value_s, scale=obs_jitter_s / 2.0 if obs_jitter_s else 0.0,
                                size=steps), 0.0, obs_delay_value_s + obs_jitter_s)
    state_dropout = rng.random(steps) < dropout_probability
    depth_dropout_p = dropout_probability + (0.10 if visibility == "reduced" else 0.0)
    depth_dropout = rng.random(steps) < depth_dropout_p

    horizon = steps * dt
    period = obligations.supervision_heartbeat_period_s
    heartbeats = [round(k * period, 6) for k in range(int(horizon / period) + 2)]

    outages: list[tuple[float, float]] = []
    if rng.random() < protocol.conditions.supervision_outage_probability:
        duration = float(rng.choice(protocol.conditions.supervision_outage_duration_s))
        earliest = 0.25 * horizon
        latest = max(earliest + dt, 0.8 * horizon - duration)
        start = float(rng.uniform(earliest, latest))
        outages.append((round(start, 3), round(start + duration, 3)))

    n_responses = 6
    auth_delays = np.clip(
        rng.normal(loc=supervision_delay_s, scale=supervision_jitter_s / 2.0 if supervision_jitter_s else 0.0,
                   size=n_responses),
        0.0, supervision_delay_s + supervision_jitter_s,
    )
    decisions: list[str] = ["granted"] * n_responses

    return ScheduleSet(
        steps=steps,
        dt_s=dt,
        observation_delay_s=[round(float(d), 4) for d in delays],
        state_dropout=[bool(v) for v in state_dropout],
        depth_dropout=[bool(v) for v in depth_dropout],
        heartbeat_times_s=heartbeats,
        supervision_outages_s=outages,
        authorization_response_delay_s=[round(float(d), 4) for d in auth_delays],
        authorization_decision=decisions,  # type: ignore[arg-type]
        authorization_validity_s=obligations.authorization_validity_s,
        visibility=visibility,
        schedule_seed=seed,
    )


def enumerate_manifests(protocol: ProtocolConfig, run_class: str, realizations: int | None = None
                        ) -> list[ScenarioManifest]:
    """Enumerate all manifests for a run class in a deterministic order."""
    if realizations is None:
        realizations = (protocol.sampling.pilot_realizations_per_cell if run_class == "pilot"
                        else protocol.sampling.heldout_realizations_per_cell)
    out: list[ScenarioManifest] = []
    for cell in protocol.cells():
        for i in range(realizations):
            out.append(build_manifest(protocol, run_class, cell["cell_id"], i))
    return out
