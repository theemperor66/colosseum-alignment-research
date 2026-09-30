"""Bind a scenario manifest to real 3D geometry in a Colosseum level, and prove the binding.

WHY THIS MODULE IS STRICT
-------------------------
An experiment that compares guarded and unguarded flight only means something if all three arms fly
through the *same* physical scene. A previous version of this module checked actor *names* only. A
name list is not geometry: the same names can sit at different positions, at a different scale, or
next to extra bodies that change occlusion and clearance. Every function here therefore measures real
numbers through the simulator and reports per-actor expected-versus-observed values. A live run must
pass through :func:`assert_scene_ready`, which fails closed on any mismatch.

VERIFIED RPC SURFACE (upstream pin 84fc0c1c75bc73a0135ee80a325d470577c66c52, tag v2.3.0)
----------------------------------------------------------------------------------------
Every call below was read in the pinned source before it was used here. The Python client name, the
C++ ``bind`` site, and the Unreal implementation are cited so a reviewer can check the semantics, not
just the spelling.

* ``simListSceneObjects`` -- client.py:544 / RpcLibServerBase.cpp:363 / WorldSimApi.cpp:331-339 /
  AirBlueprintLib.cpp:433-446. Iterates every actor in the world and matches ``actor->GetName()`` with
  ``std::regex_match`` (line 441), so the regex must match the WHOLE name. It returns NAMES ONLY: there
  is no listing that carries poses, so the relevant world costs one pose call per actor.
* ``simListSceneObjectsByTag`` -- client.py:558 / RpcLibServerBase.cpp:367 / AirBlueprintLib.cpp:448-466.
  Returns actor NAMES for matching tags.
* ``simGetObjectPose`` -- client.py:488 / RpcLibServerBase.cpp:387 / WorldSimApi.cpp:362-374. Looks the
  actor up in ``scene_object_map`` (line 367) and returns ``Pose::nanPose()`` (line 369) when it is
  absent. The pose passes through ``toGlobalNed`` (line 368), so it is already in the NED metre frame
  our manifests use. ``scene_object_map`` is filled once, at ``SimModeBase.cpp:154`` through
  ``AirBlueprintLib.cpp:265-274``, so an actor created after level start is listed but has NO pose.
* ``simGetObjectScale`` -- client.py:518 / RpcLibServerBase.cpp:392 / WorldSimApi.cpp:376-387. Returns
  ``AActor::GetActorScale()`` (line 382), a UNITLESS multiplier, and ``Vector3r::Zero()`` (line 383) for
  an absent actor. It is not metres and cannot become metres without the base size of the source mesh.
* ``simSpawnObject`` -- client.py:593 / RpcLibServerBase.cpp:375 / WorldSimApi.cpp:91-140. Renames the
  actor when a similar name already exists (lines 108-118) and RETURNS the final name.
* ``simDestroyObject`` -- client.py:609 / RpcLibServerBase.cpp:379 / WorldSimApi.cpp:62-78. Also removes
  the actor from ``scene_object_map``.
* ``simListAssets`` -- client.py:584 / RpcLibServerBase.cpp:383 / WorldSimApi.cpp:80-89. Lists the
  cooked static-mesh and blueprint registry.

WHAT CANNOT MEASURE AN EXTENT AT THIS PIN (checked, not assumed)
---------------------------------------------------------------
* ``simGetWorldExtents`` -- client.py:402 / RpcLibServerBase.cpp:231 / WorldSimApi.cpp:801-846. Returns
  the min/max corner of the WHOLE world as geodetic points, not a per-actor box.
* ``simGetMeshPositionVertexBuffers`` -- client.py:427 / RpcLibServerBase.cpp:239 /
  WorldSimApi.cpp:701-709 / AirBlueprintLib.cpp:468-...: iterates ``UStaticMeshComponent`` objects, keys
  them by the lower-cased MESH name (lines 475, 491) rather than the actor name, reports the component
  location and rotation but NOT its scale (lines 493-501), and returns LOD0 vertices in mesh-local
  space. A world extent cannot be reconstructed from that response.
* ``simGetDetections`` -- client.py:691 / RpcLibServerBase.cpp:277 / WorldSimApi.cpp:1040-1076 /
  DetectionComponent.cpp:43-73. This is the ONLY pinned call that returns a 3D box per actor:
  ``actor->GetComponentsBoundingBox(true)`` (DetectionComponent.cpp:61), keyed by the actor name
  (WorldSimApi.cpp:1055). It is not usable as a general extent query, and this module does not pretend
  otherwise: the box is expressed CAMERA-RELATIVE (DetectionComponent.cpp:62, converted per corner at
  WorldSimApi.cpp:1064-1065), the actor must match a wildcard filter on its name or mesh name
  (ObjectFilter.cpp:21-27, 157-159 -- ``MatchesWildcard``, not a regex), it must be closer than the
  filter radius (DetectionComponent.cpp:18, default 20000 cm) AND inside the camera view AND unoccluded
  by a line trace (DetectionComponent.cpp:151-195). A body behind the vehicle or behind another body
  returns nothing. UNVERIFIED: no simulator in this project has ever answered an RPC call, so a
  detection-based extent measurement could not be validated here. It is documented for a future lane
  instead of being implemented blind (docs/scene-integration.md).

Four consequences drive the design and are not negotiable:

1. **There is no general world-extent query at the pin.** A box half-extent in metres can only be
   established from a live scale multiplier TIMES the base size of a known source mesh, and that base
   size needs measurement evidence. It must come from a reviewed :class:`GeometryContract` entry that
   names who measured the mesh, how, and when. An actor with no such evidence is ``unverifiable`` and
   BLOCKS a scientific episode; it is never quietly reported as ``ok``.
2. **A missing actor is observable but ambiguous.** ``simGetObjectPose`` answers NaN both for an actor
   that does not exist and for an actor that exists in the level but is absent from
   ``scene_object_map``. The report distinguishes the two by cross-checking the scene listing, and never
   turns an unmeasured actor into a pass.
3. **``simSpawnObject`` can rename our actor.** The returned name, not the requested one, is what later
   calls must use, and it is what the binding report records.
4. **The relevant world is established by geometry, not by names.** Every listed actor that a declared
   exclusion does not remove is probed with ``simGetObjectPose``. When that cannot be completed -- no
   pose for an actor, an unusable listing, or more actors than the probe budget -- the inventory is
   reported INCOMPLETE and the episode is blocked, because an unknown body inside the mission volume
   changes clearance and occlusion.

COORDINATE CONVENTIONS
----------------------
* Manifests use vehicle-local NED metres: +X north, +Y east, +Z **down**. Object RPCs use GLOBAL NED
  with a different origin. The production adapter measures that translation while paused after each
  reset; its scene_rpc_call subtracts it on object reads and adds it on object spawn/set arguments.
  Generic vehicle, camera, kinematics and collision RPCs already use the local frame and stay unchanged.
* Unreal editor placement (see ``deploy/scripts/generate_study_scene.py``) uses centimetres with +Z up,
  hence ``ned_to_unreal_cm``. ``origin_offset_cm`` stays zero and UNMEASURED until somebody measures the
  built level.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import os
import re
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from colosseum_assurance.scenario.manifest import ObstacleSpec, ScenarioManifest
from colosseum_assurance.schemas import Box, StrictModel, Vec3

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from colosseum_assurance.config import SceneLaunchPlatform
    from colosseum_assurance.interfaces import SimAdapter
    from colosseum_assurance.protocol.spec import ProtocolConfig

UNREAL_CM_PER_M = 100.0

COLASSURE_ACTOR_PREFIX = "colassure-"
"""Name prefix of every actor this module creates.

Idempotent re-runs delete actors with this prefix and nothing else. An actor without the prefix was
placed by the level designer or by the map author, and we never destroy it.
"""

DEFAULT_IGNORED_ACTOR_REGEX = r"(?i)(playerstart|worldsettings|^drone|_?ground_?plane|skysphere|sunsky)"
"""Actors declared irrelevant for the inventory: the vehicle, the player start, the ground, the sky.

This is a DECLARED EXCLUSION, not a filter that hides evidence. Every actor removed by this pattern is
listed in :class:`SceneInventoryReport.excluded` with the pattern that removed it, so a reviewer sees
exactly what was left out. Everything else in the level is probed by geometry, whatever it is called.

The previous implementation did the opposite: it probed only actors whose name contained ``tower``,
``wall``, ``mast``, ``shed``, ``building``, ``tree`` or our own prefix. A collidable body called
``BP_Crane_7`` in the middle of the flight volume was therefore invisible to the check (independent
review, "unexpected relevant collidable actors cannot evade checks just because their name lacks
tower/wall/tree"). Name keywords no longer decide what counts as relevant.
"""

MAX_INVENTORY_PROBES = 256
"""Upper bound on ``simGetObjectPose`` calls spent on establishing the relevant world.

``simListSceneObjects`` returns names only (AirBlueprintLib.cpp:433-446), so the only way to place a
body is one pose call per actor. A packaged level can hold thousands of actors; beyond this bound the
inventory is reported INCOMPLETE with the count, which blocks a scientific episode instead of pretending
the rest of the level is empty.
"""

STUDY_BOX_ASSET_NAME = "Cube"
"""Default static mesh used by the instantiate path.

The Unreal engine primitive ``Cube`` is a 1 m cube at scale 1 (``/Engine/BasicShapes/Cube``). That
figure is an Unreal convention, NOT something this project has measured on a running server, so it is
not sufficient evidence for a scientific episode: the metre extents of a spawned box are accepted only
when a reviewed :class:`GeometryContract` entry states the mesh and its base size. The asset name is
also checked against ``simListAssets`` before any spawn, because ``WorldSimApi.cpp:94-98`` dereferences
the registry lookup without a null check: spawning an unknown asset name can take the simulator down
instead of returning an error.
"""

SceneMode = Literal["instantiate", "verify_only", "qualified_map"]
"""How a scenario manifest is bound to a level.

``instantiate``     We create the study geometry at runtime with ``simSpawnObject`` from a known mesh.
                    Expected geometry comes from the manifest; the scene is made to match it.
``verify_only``     The packaged map already contains the study actors (built by the Unreal step in
                    ``deploy/scripts/generate_study_scene.py``). We only measure and compare.
``qualified_map``   A third-party map whose *natural* actors are measured and the manifest geometry is
                    DERIVED from them (:func:`derive_manifest_geometry`). The study keeps real 3D
                    geometry and occlusion, and the record says plainly that the scenario definition
                    came from the map, not from us.
"""

ActorStatus = Literal["ok", "missing", "pose_mismatch", "scale_mismatch", "extra", "unverifiable"]

GeometryProvenance = Literal["manifest", "scene_derived"]


# ======================================================================================= errors
class SceneError(RuntimeError):
    """A scene problem that stops a run. Carries a plain-language remedy for the operator."""

    def __init__(self, message: str, *, remedy: str = "", report: Any | None = None) -> None:
        super().__init__(message)
        self.remedy = remedy
        self.report = report


class SceneMismatch(SceneError):
    """The level is not the level the manifest describes.

    Raised by :func:`assert_scene_ready`. Live episodes must die here rather than produce numbers from
    an unknown scene: a wrong obstacle position changes clearance, occlusion and the inspection task
    itself, so the episode would answer a different question than the protocol asks.
    """


class SceneUnavailable(SceneError):
    """The simulator cannot provide what this scene mode needs (missing RPC, missing asset).

    Separate from :class:`SceneMismatch` because the remedy is different: the mismatch means *fix the
    level*, this one means *use another scene mode or another build*.
    """


# ==================================================================================== geometry contract
GEOMETRY_CONTRACT_ENV_VAR = "COLASSURE_SCENE_GEOMETRY_CONTRACT"
"""Environment variable holding the path of the reviewed geometry contract.

Resolution order is: the explicit ``geometry_contract`` argument, then ``geometry_contract_path``, then
an ``adapter.geometry_contract`` / ``adapter.geometry_contract_path`` attribute, then this variable.
The chosen source path and its SHA-256 are copied into every report, so a reader can always see which
document authorised the extent numbers in an episode record.
"""

ExtentSource = Literal[
    "scale_times_contract_base",
    "contract_declared_world_extent",
    "fixture_fake_declared",
    "declared_asset_base_unreviewed",
    "unmeasured",
]
"""Where a half-extent in metres came from. Only the first three count as evidence.

``scale_times_contract_base``       live ``simGetObjectScale`` multiplier times the mesh base size of a
                                    reviewed contract entry. The strongest form: the live level takes
                                    part in the number.
``contract_declared_world_extent``  an independently measured world half-extent from a reviewed entry
                                    that explicitly accepts being used without a live scale. Weaker:
                                    the live level contributes the position only.
``fixture_fake_declared``           the adapter is the in-repo fixture fake, whose world IS the manifest
                                    it was handed. Software double, never experimental evidence.
``declared_asset_base_unreviewed``  a base size passed as a function argument with nothing reviewing it
                                    (for example the "a Cube is 1 m" Unreal convention). Reported for
                                    information, NOT accepted.
``unmeasured``                      nothing establishes the extent. Blocks a scientific episode.
"""


class GeometryContractEntry(StrictModel):
    """One reviewed statement: this actor is this mesh, and this mesh is this big.

    WHY THIS TYPE EXISTS. No RPC at the pin returns the world extents of an actor: ``simGetObjectScale``
    is a unitless multiplier (WorldSimApi.cpp:376-387) and the alternatives are unusable for this
    purpose (see the module docstring). Metre extents therefore need traceable independent evidence of
    the source mesh's base size. That evidence must be checkable, so an entry must
    name who measured the mesh, how, and when. Disclosure that a number is assumed does not make the
    evaluator's assumed clearance true; a named measurement can at least be audited or repeated.
    """

    pattern: str = Field(
        description=("Whole-name regular expression for the actor(s) this entry describes. Whole-name "
                     "matching mirrors upstream ``std::regex_match`` (AirBlueprintLib.cpp:441), so "
                     "``.*`` is needed on both sides for a substring."),
    )
    mesh_name: str = Field(description="The source mesh/asset the actor is built from.")
    mesh_base_size_m: Vec3 = Field(
        description="Full size of that mesh in metres at scale 1 (x, y, z), not the half-extent.",
    )
    world_half_extent_m: Vec3 | None = Field(
        default=None,
        description=("Independently measured world half-extents of the placed actor, in metres. Used "
                     "only when the server cannot answer simGetObjectScale and the entry opts in."),
    )
    accept_world_half_extent_without_live_scale: bool = Field(
        default=False,
        description=("Opt-in: accept ``world_half_extent_m`` as the extent evidence on a server that "
                     "does not answer simGetObjectScale. False by default so that an entry can never "
                     "satisfy the gate by accident."),
    )
    measured_by: str = Field(description="Actual measurer, including accurately named automated tools.")
    measurement_method: str = Field(description="How it was measured, concretely enough to repeat.")
    measured_at: str = Field(description="ISO-8601 date of the measurement.")
    evidence_uri: str = Field(default="", description="Where the measurement record lives, if anywhere.")
    note: str = ""

    @field_validator("pattern")
    @classmethod
    def _pattern_compiles(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("pattern must not be empty")
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"pattern {value!r} is not a valid regular expression: {exc}") from exc
        return value

    @field_validator("mesh_name", "measured_by", "measurement_method")
    @classmethod
    def _not_blank(cls, value: str, info: Any) -> str:
        if not value.strip():
            raise ValueError(
                f"{info.field_name} must not be empty: a geometry contract entry is a measurement "
                "claim, and a claim with no measurer, no method or no mesh cannot be audited"
            )
        return value

    @field_validator("measured_at")
    @classmethod
    def _iso_date(cls, value: str) -> str:
        try:
            date.fromisoformat(value.strip()[:10])
        except ValueError as exc:
            raise ValueError(f"measured_at must be an ISO-8601 date, got {value!r}") from exc
        return value

    @field_validator("mesh_base_size_m", "world_half_extent_m")
    @classmethod
    def _positive(cls, value: Vec3 | None, info: Any) -> Vec3 | None:
        if value is None:
            return value
        if min(value.as_tuple()) <= 0.0:
            raise ValueError(f"{info.field_name} must be positive in all three axes, got {value.as_tuple()}")
        return value

    @model_validator(mode="after")
    def _opt_in_needs_a_number(self) -> GeometryContractEntry:
        if self.accept_world_half_extent_without_live_scale and self.world_half_extent_m is None:
            raise ValueError(
                "accept_world_half_extent_without_live_scale=true requires world_half_extent_m: the "
                "entry claims the extent may be used without a live scale but states no extent"
            )
        return self

    def matches(self, actor_name: str) -> bool:
        """Whole-name match, like ``simListSceneObjects`` upstream (AirBlueprintLib.cpp:441)."""
        return re.fullmatch(self.pattern, actor_name) is not None

    def half_extent_from_scale(self, scale: Vec3) -> Vec3:
        """Convert a live scale multiplier into metre half-extents with this entry's mesh base size."""
        return Vec3(
            x=abs(scale.x) * self.mesh_base_size_m.x / 2.0,
            y=abs(scale.y) * self.mesh_base_size_m.y / 2.0,
            z=abs(scale.z) * self.mesh_base_size_m.z / 2.0,
        )

    def provenance_line(self) -> str:
        """One line naming the evidence, for the per-actor explanation in the report."""
        return (f"mesh {self.mesh_name!r} base {self.mesh_base_size_m.as_tuple()} m measured by "
                f"{self.measured_by} ({self.measurement_method}, {self.measured_at})")


class GeometryContract(StrictModel):
    """The reviewed configuration that binds actors to known meshes and known base dimensions.

    ``status`` exists so that the shipped example cannot authorise an experiment. An ``example``
    contract loads, computes and reports exactly like a reviewed one, but it also adds a blocking reason
    of its own, so nobody can obtain a scientific episode by pointing at the template file.
    """

    contract_version: str = "1.0.0"
    status: Literal["example", "reviewed"] = Field(
        description=("``reviewed``: an attributed review accepted traceable measurements for use. "
                     "``example``: a template. An example contract never satisfies the gate."),
    )
    reviewed_by: str = ""
    reviewed_at: str = ""
    note: str = ""
    entries: list[GeometryContractEntry] = Field(default_factory=list)
    source_path: str = ""
    source_sha256: str = ""

    @model_validator(mode="after")
    def _reviewed_needs_a_reviewer(self) -> GeometryContract:
        if self.status == "reviewed" and not self.reviewed_by.strip():
            raise ValueError(
                "status: reviewed requires reviewed_by: a contract that authorises experimental "
                "geometry must say who accepted it"
            )
        return self

    def entry_for(self, actor_name: str) -> GeometryContractEntry | None:
        """First entry whose pattern matches the whole actor name. File order decides."""
        for entry in self.entries:
            if entry.matches(actor_name):
                return entry
        return None

    def summary(self) -> dict[str, Any]:
        """Compact provenance block for the episode record."""
        return {
            "contract_version": self.contract_version,
            "status": self.status,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "entries": len(self.entries),
            "patterns": [entry.pattern for entry in self.entries],
        }


def load_geometry_contract(path: str | Path) -> GeometryContract:
    """Load and validate a geometry contract from YAML, recording the file hash.

    Fails loudly. A contract that cannot be read, parsed or validated must stop the run: the
    alternative is an episode whose clearance geometry rests on a file nobody could check.
    """
    file_path = Path(path)
    remedy = (
        "Write a contract like configs/scene-geometry.example.yaml: one entry per actor pattern with "
        "mesh_name, mesh_base_size_m, measured_by, measurement_method and measured_at, and "
        "status: reviewed plus reviewed_by after an attributed review of the measurements."
    )
    try:
        raw = file_path.read_bytes()
    except OSError as exc:
        raise SceneUnavailable(f"geometry contract {file_path} could not be read: {exc}",
                               remedy=remedy) from exc
    try:
        document = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise SceneUnavailable(f"geometry contract {file_path} is not readable YAML: {exc}",
                               remedy=remedy) from exc
    if not isinstance(document, dict):
        raise SceneUnavailable(
            f"geometry contract {file_path} must be a YAML mapping, got {type(document).__name__}",
            remedy=remedy,
        )
    try:
        contract = GeometryContract.model_validate(document)
    except ValidationError as exc:
        raise SceneUnavailable(f"geometry contract {file_path} is not valid: {exc}",
                               remedy=remedy) from exc
    contract.source_path = str(file_path)
    contract.source_sha256 = hashlib.sha256(raw).hexdigest()
    return contract


def resolve_geometry_contract(
    adapter: SimAdapter | Any,
    contract: GeometryContract | None = None,
    contract_path: str | Path | None = None,
) -> GeometryContract | None:
    """Find the contract that applies to this run, in a fixed and reportable order.

    The adapter attributes exist so that the production adapter can carry the operator's configured
    contract without every call site passing it. The environment variable exists so that an operator can
    bind a contract to a run without editing code. Both record their source in the report.
    """
    if contract is not None:
        return contract
    if contract_path:
        return load_geometry_contract(contract_path)
    value = getattr(adapter, "geometry_contract", None)
    if isinstance(value, GeometryContract):
        return value
    if isinstance(value, (str, Path)) and str(value).strip():
        return load_geometry_contract(value)
    attribute_path = getattr(adapter, "geometry_contract_path", None)
    if isinstance(attribute_path, (str, Path)) and str(attribute_path).strip():
        return load_geometry_contract(attribute_path)
    from_env = os.environ.get(GEOMETRY_CONTRACT_ENV_VAR, "").strip()
    if from_env:
        return load_geometry_contract(from_env)
    return None


# ======================================================================================= models
class SceneActorSpec(StrictModel):
    """What one manifest obstacle requires of the level, in numbers.

    Holds both the NED expectation used for verification and the Unreal placement values used by the
    editor-side generator, so a reviewer can see that both describe the same body.
    """

    obstacle_name: str
    kind: str
    actor_name: str = Field(description="Exact actor name we expect or will create in the level.")
    actor_tag: str = Field(description="Unreal actor tag; the second, independent resolution route.")
    occludes_asset: bool = False
    expected_position_ned_m: Vec3
    expected_yaw_rad: float = 0.0
    expected_half_extent_m: Vec3
    expected_scale: Vec3 | None = Field(
        default=None,
        description=(
            "Expected Unreal actor scale multiplier. Only known when we spawn a mesh of known base "
            "size; None elsewhere, because no RPC at the pin returns world extents."
        ),
    )
    asset_name: str | None = None
    asset_base_size_m: float | None = None
    unreal_location_cm: tuple[float, float, float]
    unreal_box_scale_cm: tuple[float, float, float]
    geometry_provenance: GeometryProvenance = "manifest"


class ActorMeasurement(StrictModel):
    """Raw numbers read back from the simulator for one actor, plus where the extent came from."""

    observed_position_ned_m: Vec3 | None = None
    observed_yaw_rad: float | None = None
    observed_scale: Vec3 | None = None
    observed_half_extent_m: Vec3 | None = None
    extent_source: ExtentSource = "unmeasured"
    extent_evidence: str = Field(
        default="",
        description="One line naming the evidence behind observed_half_extent_m, or why there is none.",
    )
    position_error_m: float | None = None
    yaw_error_rad: float | None = None
    scale_relative_error: float | None = None
    extent_error_m: float | None = None
    pose_source: str = ""
    scale_source: str = ""
    unmeasured_reason: str = ""


class SceneActorCheck(StrictModel):
    """Expected versus observed for one actor, with the tolerance that produced the verdict.

    ``position_verified`` and ``extent_verified`` are kept apart on purpose. "The bodies are in the
    right places but nobody knows how big they are" is a real and common state of the world, and the
    report has to be able to say exactly that without letting it count as a verified scene
    (independent review, "scene gate still accepts incomplete geometry").
    """

    spec: SceneActorSpec
    resolved_actor_name: str | None = None
    match_route: Literal["exact_name", "exact_tag", "tag_route", "normalized", "none"] = "none"
    candidate_names: list[str] = Field(default_factory=list)
    measurement: ActorMeasurement = Field(default_factory=ActorMeasurement)
    contract_pattern: str | None = Field(
        default=None,
        description="Pattern of the geometry-contract entry that supplied the extent evidence, if any.",
    )
    position_tolerance_m: float
    scale_tolerance: float
    yaw_tolerance_rad: float
    position_verified: bool = False
    extent_verified: bool = False
    status: ActorStatus = "unverifiable"
    explanation: str = ""

    @property
    def is_blocking(self) -> bool:
        """True when this actor must stop a scientific episode."""
        return self.status != "ok"


class ExtraActorFinding(StrictModel):
    """An actor the manifest does not describe, found by geometry inside or near the mission volume."""

    actor_name: str
    observed_position_ned_m: Vec3 | None = None
    observed_half_extent_m: Vec3 | None = None
    inside_mission_volume: bool | None = None
    distance_to_volume_m: float | None = None
    status: Literal["extra", "unverifiable"] = "unverifiable"
    note: str = ""


class InventoryExclusion(StrictModel):
    """One actor that a DECLARED pattern removed from the inventory, and why."""

    actor_name: str
    pattern: str
    reason: str


class SceneInventoryReport(StrictModel):
    """Whether the relevant world could be established at all.

    A verification that only looks at the actors the manifest names cannot see a body nobody declared.
    This record answers the other question: was every actor in the level either matched, excluded by a
    declared rule, or placed by measurement? When the answer is no, ``complete`` is false and the run is
    blocked, because an unplaced collidable body inside the mission volume changes clearance and
    occlusion just as much as a moved one.
    """

    name_regex: str = ".*"
    listed_actors: int = 0
    matched_actors: int = 0
    probed_actors: int = 0
    candidate_actors: int = 0
    unprobed_actors: int = 0
    max_probes: int = MAX_INVENTORY_PROBES
    wall_budget_s: float = 120.0
    elapsed_s: float = 0.0
    budget_exhausted: bool = False
    excluded: list[InventoryExclusion] = Field(default_factory=list)
    unplaced_actors: list[str] = Field(
        default_factory=list,
        description="Listed by simListSceneObjects but simGetObjectPose returned nothing usable.",
    )
    inside_volume: list[str] = Field(default_factory=list)
    complete: bool = False
    incomplete_reasons: list[str] = Field(default_factory=list)
    method: str = (
        "every listed actor that a declared exclusion did not remove was probed with simGetObjectPose "
        "and placed against the mission volume by geometry; names were not used to decide relevance"
    )


class SceneVerificationReport(StrictModel):
    """Measured comparison of the live level against one scenario manifest.

    Returned instead of raised: a mismatch is evidence about the run and belongs in the episode record.
    :func:`assert_scene_ready` turns it into the fail-closed exception.

    Three separate facts, three separate fields: ``positions_verified`` (the bodies are where the
    manifest says), ``measurement_complete`` (their sizes rest on evidence), ``inventory_complete``
    (nothing else relevant is in the volume). ``ok`` is the conjunction plus the scene rules, and only
    ``ok`` may back a scientific episode.
    """

    mode: SceneMode = "verify_only"
    scenario_id: str
    scene_name: str
    checked_at: str
    geometry_provenance: GeometryProvenance = "manifest"
    evidence_class: Literal["scientific_candidate", "software_double"] = "scientific_candidate"
    scene_object_count: int
    position_tolerance_m: float
    scale_tolerance: float
    yaw_tolerance_rad: float
    mission_volume: Box | None = None
    actors: list[SceneActorCheck] = Field(default_factory=list)
    extras: list[ExtraActorFinding] = Field(default_factory=list)
    inventory: SceneInventoryReport = Field(default_factory=SceneInventoryReport)
    geometry_contract: dict[str, Any] | None = Field(
        default=None,
        description="Summary of the reviewed contract that authorised the extent numbers, if any.",
    )
    # -- compact views, kept stable for the episode record and for older call sites ----------------
    expected_actors: list[str] = Field(default_factory=list)
    matched_actors: dict[str, list[str]] = Field(default_factory=dict)
    missing_actors: list[str] = Field(default_factory=list)
    extra_actors: list[str] = Field(default_factory=list)
    alias_actors: list[str] = Field(
        default_factory=list,
        description=("Level names that describe a body already matched, proven by measuring the same "
                     "position. Recorded so a reviewer can see what was forgiven and why."),
    )
    mismatched_actors: list[str] = Field(default_factory=list)
    unverifiable_actors: list[str] = Field(default_factory=list)
    unmeasured_extent_actors: list[str] = Field(
        default_factory=list,
        description="Actors whose position matched but whose extent rests on no evidence.",
    )
    tag_route_available: bool = False
    tag_route_actors: list[str] = Field(default_factory=list)
    tag_route_missing: list[str] = Field(default_factory=list)
    route_disagreement: list[str] = Field(default_factory=list)
    max_position_error_m: float | None = None
    positions_verified: bool = Field(
        default=False,
        description=("Every expected actor was found and sits within tolerance of the manifest "
                     "position. This ALONE is not a verified scene: extents may be unknown."),
    )
    measurement_complete: bool = False
    inventory_complete: bool = False
    blocking_reasons: list[str] = Field(
        default_factory=list,
        description=("Why this scene may not back a scientific episode. Each entry starts with a "
                     "category tag: [position], [measurement], [extra], [inventory], [scene], "
                     "[contract]."),
    )
    ok: bool = False
    caveats: list[str] = Field(default_factory=list)
    detail: str = ""

    @property
    def is_complete(self) -> bool:
        """Every expected actor was found, measured, and agreed with the manifest."""
        return self.ok and self.measurement_complete and not self.missing_actors


class SpawnedActorRecord(StrictModel):
    """One actor this process created, as the simulator confirmed it."""

    obstacle_name: str
    requested_name: str
    actual_name: str
    asset_name: str
    position_ned_m: Vec3
    yaw_rad: float
    scale: Vec3
    physics_enabled: bool
    renamed_by_simulator: bool = False


class SceneBindingReport(StrictModel):
    """What the runtime binding path did to the level, and whether the result verifies."""

    mode: SceneMode
    scenario_id: str
    scene_name: str
    created_at: str
    dry_run: bool
    asset_name: str
    asset_base_size_m: float
    planned_actors: list[SceneActorSpec] = Field(default_factory=list)
    planned_destroy: list[str] = Field(default_factory=list)
    spawned: list[SpawnedActorRecord] = Field(default_factory=list)
    destroyed: list[str] = Field(default_factory=list)
    destroy_failures: list[str] = Field(default_factory=list)
    rpc_methods_used: list[str] = Field(default_factory=list)
    available_assets_sampled: list[str] = Field(default_factory=list)
    verification: SceneVerificationReport | None = None
    ok: bool = False
    detail: str = ""


class DerivedActorRecord(StrictModel):
    """One natural map actor measured for the qualified-map path.

    ``extent_source`` is part of the record because a derived obstacle with an invented size would give
    the evaluator a clearance geometry that nothing supports. ``unmeasured`` here makes the whole
    binding not ok (independent review, "qualified-map derivation similarly invents default extents").
    """

    actor_name: str
    observed_position_ned_m: Vec3
    observed_yaw_rad: float
    observed_scale: Vec3 | None = None
    half_extent_m: Vec3
    extent_source: ExtentSource = "unmeasured"
    extent_evidence: str = ""
    contract_pattern: str | None = None
    inside_mission_volume: bool = True


class SceneDerivationProvenance(StrictModel):
    """Why a manifest carries scene-derived geometry, and what exactly was measured.

    This record must travel with the run metadata. A scene-derived manifest is a DIFFERENT scenario
    definition from a manifest we authored: the obstacles are whatever the third-party map contains.
    Comparisons across arms stay valid (all arms replay the same derived manifest), but comparisons
    against runs on our own study scene do not.
    """

    geometry_provenance: Literal["scene_derived"] = "scene_derived"
    derived_at: str
    name_regex: str
    scene_object_count: int
    candidates_considered: int
    actors: list[DerivedActorRecord] = Field(default_factory=list)
    asset_actor_name: str | None = None
    asset_position_ned_m: Vec3 | None = None
    mission_volume: Box
    placeholder_half_extent_m: Vec3 = Field(
        description=("Value written into an obstacle whose extent could NOT be established. It is a "
                     "placeholder, never a measurement: such an actor is listed in "
                     "``actors_without_extent_evidence`` and blocks the binding."),
    )
    geometry_contract: dict[str, Any] | None = None
    actors_without_extent_evidence: list[str] = Field(default_factory=list)
    unplaced_actors: list[str] = Field(
        default_factory=list,
        description="Candidates that simGetObjectPose could not place; the map inventory is then partial.",
    )
    extent_measurement_available: bool = False
    warnings: list[str] = Field(default_factory=list)
    note: str = (
        "Scene-derived geometry: obstacle boxes come from measured third-party map actors, not from "
        "the study scene definition. Record this with the run; it changes what the scenario is."
    )


class SceneDerivationResult(StrictModel):
    """Updated obstacle list plus the provenance record that explains where it came from."""

    obstacles: list[ObstacleSpec] = Field(default_factory=list)
    provenance: SceneDerivationProvenance


# ======================================================================================= rpc access
_MUTATING_METHODS = frozenset({"simSpawnObject", "simDestroyObject", "simSetObjectPose",
                               "simSetObjectScale", "simLoadLevel"})


class _SceneRpc:
    """Bounded access to the scene RPCs the adapter does not expose as public methods.

    The adapter owns timeouts, error translation and provenance, so this facade never opens its own
    socket. It prefers a public hook, then the adapter internal call funnel (same package), then a raw
    ``call`` object, which is what the unit-test stubs provide. Every method name used is recorded so a
    dry run can be *proved* to have touched nothing.
    """

    def __init__(self, adapter: SimAdapter | Any) -> None:
        self.adapter = adapter
        self.used: list[str] = []
        route: Callable[..., Any] | None = None
        route_name = "none"
        for attribute in ("scene_rpc_call", "rpc_call", "_call", "call"):
            candidate = getattr(adapter, attribute, None)
            if callable(candidate):
                route, route_name = candidate, attribute
                break
        self._route = route
        self._route_accepts_timeout = False
        if route is not None:
            parameters = inspect.signature(route).parameters.values()
            self._route_accepts_timeout = any(
                item.name == "timeout_s" or item.kind == inspect.Parameter.VAR_KEYWORD
                for item in parameters)
        self.route_name = route_name

    @property
    def available(self) -> bool:
        return self._route is not None

    def call(self, method: str, *params: Any, timeout_s: float | None = None) -> Any:
        """Issue one RPC. Raises :class:`SceneUnavailable` when the adapter offers no call route."""
        if self._route is None:
            raise SceneUnavailable(
                f"this adapter exposes no RPC route, so {method} cannot be issued",
                remedy=("Pass the ColosseumAdapter (or an object with a call(method, *params) "
                        "method). Scene binding needs raw RPC access."),
            )
        if method not in self.used:
            self.used.append(method)
        if timeout_s is not None and self._route_accepts_timeout:
            endpoint = getattr(self.adapter, "endpoint", None)
            if endpoint is not None:
                timeout_s = min(timeout_s, endpoint.rpc_timeout_s)
            return self._route(method, *params, timeout_s=timeout_s)
        return self._route(method, *params)

    def try_call(self, method: str, *params: Any,
                 timeout_s: float | None = None) -> tuple[Any, str]:
        """Issue one RPC and convert any failure into a short reason string.

        Verification must keep going after a refused optional call: a simulator without
        ``simGetObjectScale`` still allows position checks, and the report says which measurement is
        missing instead of pretending it passed.
        """
        try:
            return self.call(method, *params, timeout_s=timeout_s), ""
        except SceneUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - any adapter/transport error becomes evidence
            return None, f"{method} failed: {type(exc).__name__}: {exc}"

    def list_scene_objects(self, name_regex: str = ".*") -> list[str]:
        """Prefer the adapter public method; fall back to the raw RPC for bare stubs."""
        lister = getattr(self.adapter, "list_scene_objects", None)
        if callable(lister):
            return [str(name) for name in lister(name_regex)]
        raw = self.call("simListSceneObjects", str(name_regex))
        return [str(name) for name in (raw or [])]

    def list_scene_objects_by_tag(self, tag_regex: str = ".*") -> tuple[list[str], bool]:
        """Second resolution route. Returns (names, available)."""
        lister = getattr(self.adapter, "list_scene_objects_by_tag", None)
        if callable(lister):
            try:
                return [str(name) for name in lister(tag_regex)], True
            except Exception:  # noqa: BLE001 - an optional route must not break verification
                return [], False
        value, error = self.try_call("simListSceneObjectsByTag", str(tag_regex))
        if error or value is None:
            return [], False
        return [str(name) for name in value], True


# ======================================================================================= geometry
def _normalise(name: str) -> str:
    """Lower-case and strip separators so naming styles do not cause false misses."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _matches(tag: str, scene_object: str) -> bool:
    """Case- and separator-insensitive containment match.

    Unreal decorates placed actors ("BP_InspectionTower_C_1") and our manifests use snake_case body
    names, so exact comparison alone would report false misses. Containment is permissive on purpose;
    the resolved name is always recorded, and geometry is measured afterwards, so a wrong match cannot
    turn into a silent pass.
    """
    return _normalise(tag) in _normalise(scene_object)


def _finite_vec(values: Iterable[Any]) -> Vec3 | None:
    try:
        x, y, z = (float(v) for v in values)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, z)):
        return None
    return Vec3(x=x, y=y, z=z)


def _decode_vec3(value: Any) -> Vec3 | None:
    """Decode a Vector3r sent as a positional array or as a map. NaN or junk becomes None."""
    if value is None:
        return None
    if isinstance(value, dict):
        if not all(key in value for key in ("x_val", "y_val", "z_val")):
            return None
        return _finite_vec((value["x_val"], value["y_val"], value["z_val"]))
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return _finite_vec(value)
    return None


def _decode_quaternion_yaw(value: Any) -> float | None:
    """Yaw in radians from a Quaternionr (w, x, y, z) array or map."""
    if isinstance(value, dict):
        try:
            w, x, y, z = (float(value[key]) for key in ("w_val", "x_val", "y_val", "z_val"))
        except (KeyError, TypeError, ValueError):
            return None
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            w, x, y, z = (float(v) for v in value)
        except (TypeError, ValueError):
            return None
    else:
        return None
    if not all(math.isfinite(v) for v in (w, x, y, z)):
        return None
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _decode_pose(value: Any) -> tuple[Vec3, float] | None:
    """Decode a Pose (position, orientation).

    Returns None for the NaN pose that ``WorldSimApi.cpp:369`` sends when the actor is not in
    ``scene_object_map``; the caller turns that into ``missing`` or ``unverifiable``, never into a pass.
    """
    if isinstance(value, dict):
        position, orientation = value.get("position"), value.get("orientation")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        position, orientation = value
    else:
        return None
    point = _decode_vec3(position)
    if point is None:
        return None
    yaw = _decode_quaternion_yaw(orientation)
    return point, 0.0 if yaw is None else yaw


def _angle_error(observed: float, expected: float) -> float:
    """Smallest absolute angle between two yaw values, in radians."""
    return abs(math.atan2(math.sin(observed - expected), math.cos(observed - expected)))


def actor_tag_for(obstacle: ObstacleSpec) -> str:
    """The Unreal actor tag a body must carry. Falls back to the body name when no tag was set."""
    return obstacle.unreal_actor_tag or obstacle.name


def study_actor_name(obstacle_name: str) -> str:
    """Deterministic actor name for study geometry we own.

    The prefix is what makes cleanup safe: :func:`instantiate_scene` destroys actors with this prefix
    and refuses to touch anything else.
    """
    return f"{COLASSURE_ACTOR_PREFIX}{obstacle_name}"


def apply_launch_platform(manifest: ScenarioManifest, platform: SceneLaunchPlatform | None,
                          protocol: ProtocolConfig) -> ScenarioManifest:
    """Add frozen deployment support geometry idempotently, retaining the protocol's ground home.

    This only declares the requested world. Normal instantiate measurement and inventory gates must
    still prove that world exists. Rebinding an effective manifest must not accumulate duplicate bodies.
    """
    if platform is None:
        return manifest
    bounds, home = platform.bounds, protocol.mission.home
    clearance = platform.spawn_clearance_m
    if min(home.x-bounds.x_min, bounds.x_max-home.x,
           home.y-bounds.y_min, bounds.y_max-home.y) < clearance:
        raise SceneMismatch("launch platform does not cover home with the declared spawn clearance")
    if abs(bounds.z_min-home.z) > 1e-6:
        raise SceneMismatch("launch platform top must coincide with protocol mission.home ground height")
    body = ObstacleSpec(
        name=platform.name, kind="ground_plane",
        center=Vec3(x=(bounds.x_min+bounds.x_max)/2, y=(bounds.y_min+bounds.y_max)/2,
                    z=(bounds.z_min+bounds.z_max)/2),
        extent=Vec3(x=(bounds.x_max-bounds.x_min)/2, y=(bounds.y_max-bounds.y_min)/2,
                    z=(bounds.z_max-bounds.z_min)/2),
        unreal_actor_tag=platform.name,
    )
    existing = [item for item in manifest.obstacles if item.name == body.name]
    if any(item != body for item in existing) or len(existing) > 1:
        raise SceneMismatch("launch platform name conflicts with existing manifest geometry")
    data = manifest.model_dump(mode="json")
    if not existing:
        data["obstacles"].append(body.model_dump(mode="json"))
    data["start_position"] = Vec3(x=home.x, y=home.y,
                                  z=bounds.z_min-clearance).model_dump(mode="json")
    return ScenarioManifest.model_validate(data)


def ned_to_unreal_cm(point: Vec3, origin_offset_cm: tuple[float, float, float] = (0.0, 0.0, 0.0)
                     ) -> tuple[float, float, float]:
    """Convert an NED point in metres to Unreal world centimetres (+Z up)."""
    return (
        point.x * UNREAL_CM_PER_M + origin_offset_cm[0],
        point.y * UNREAL_CM_PER_M + origin_offset_cm[1],
        -point.z * UNREAL_CM_PER_M + origin_offset_cm[2],
    )


def mission_volume(manifest: ScenarioManifest, protocol: ProtocolConfig | None = None,
                   margin_m: float = 1.0) -> Box:
    """The volume an unexpected body must stay out of.

    With a protocol this is the geofence the vehicle is allowed to fly in, widened by ``margin_m``:
    anything solid inside it can change clearance, collision and occlusion. Without a protocol it falls
    back to the bounding box of the manifest itself (start, asset, viewpoint, obstacles), which is the
    most that can be said from the manifest alone.
    """
    if protocol is not None:
        fence = protocol.obligations.geofence
        return Box(
            x_min=fence.x_min - margin_m, x_max=fence.x_max + margin_m,
            y_min=fence.y_min - margin_m, y_max=fence.y_max + margin_m,
            z_min=fence.z_min - margin_m, z_max=fence.z_max + margin_m,
        )
    points: list[Vec3] = [manifest.start_position, manifest.asset_position, manifest.inspection_viewpoint]
    lows = [(p.x, p.y, p.z) for p in points]
    for obstacle in manifest.obstacles:
        lows.append((obstacle.center.x - obstacle.extent.x, obstacle.center.y - obstacle.extent.y,
                     obstacle.center.z - obstacle.extent.z))
        lows.append((obstacle.center.x + obstacle.extent.x, obstacle.center.y + obstacle.extent.y,
                     obstacle.center.z + obstacle.extent.z))
    xs = [p[0] for p in lows]
    ys = [p[1] for p in lows]
    zs = [p[2] for p in lows]
    return Box(
        x_min=min(xs) - margin_m, x_max=max(xs) + margin_m,
        y_min=min(ys) - margin_m, y_max=max(ys) + margin_m,
        z_min=min(zs) - margin_m, z_max=max(zs) + margin_m,
    )


def expected_actors(manifest: ScenarioManifest,
                    origin_offset_cm: tuple[float, float, float] = (0.0, 0.0, 0.0),
                    *,
                    mode: SceneMode = "verify_only",
                    asset_name: str | None = None,
                    asset_base_size_m: float | None = None,
                    geometry_provenance: GeometryProvenance = "manifest") -> list[SceneActorSpec]:
    """Map every manifest obstacle onto the actor the level must contain.

    ``asset_base_size_m`` is the edge length in metres of the source mesh at scale 1. It is the only
    way to turn a scale multiplier into metres, because no RPC at the pin reports world extents; when
    it is None the extent check is reported as unverifiable rather than guessed.
    """
    out: list[SceneActorSpec] = []
    for obstacle in sorted(manifest.obstacles, key=lambda o: o.name):
        scale: Vec3 | None = None
        if asset_base_size_m:
            scale = Vec3(
                x=2.0 * obstacle.extent.x / asset_base_size_m,
                y=2.0 * obstacle.extent.y / asset_base_size_m,
                z=2.0 * obstacle.extent.z / asset_base_size_m,
            )
        out.append(SceneActorSpec(
            obstacle_name=obstacle.name,
            kind=obstacle.kind,
            actor_name=study_actor_name(obstacle.name),
            actor_tag=actor_tag_for(obstacle),
            occludes_asset=obstacle.occludes_asset,
            expected_position_ned_m=obstacle.center,
            expected_yaw_rad=0.0,
            expected_half_extent_m=obstacle.extent,
            expected_scale=scale,
            asset_name=asset_name,
            asset_base_size_m=asset_base_size_m,
            unreal_location_cm=ned_to_unreal_cm(obstacle.center, origin_offset_cm),
            unreal_box_scale_cm=(obstacle.extent.x * 2 * UNREAL_CM_PER_M,
                                 obstacle.extent.y * 2 * UNREAL_CM_PER_M,
                                 obstacle.extent.z * 2 * UNREAL_CM_PER_M),
            geometry_provenance=geometry_provenance,
        ))
    del mode  # the mode does not change the expectation; it changes who creates the actor
    return out


class SceneBuildPlan(StrictModel):
    """Everything the Unreal build lane needs to place one scenario realization."""

    scenario_id: str
    scene_name: str
    origin_offset_cm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_offset_status: str = (
        "UNMEASURED: the PlayerStart offset of the built level has not been measured. Placing actors "
        "with a zero offset is only correct when PlayerStart sits at the level origin."
    )
    actor_prefix: str = COLASSURE_ACTOR_PREFIX
    actors: list[SceneActorSpec] = Field(default_factory=list)
    notes: str = ""


def build_plan(manifest: ScenarioManifest,
               origin_offset_cm: tuple[float, float, float] = (0.0, 0.0, 0.0),
               *, asset_name: str | None = None, asset_base_size_m: float | None = None
               ) -> SceneBuildPlan:
    """Declarative placement plan for one scenario realization."""
    return SceneBuildPlan(
        scenario_id=manifest.scenario_id,
        scene_name=manifest.scene_name,
        origin_offset_cm=origin_offset_cm,
        actors=expected_actors(manifest, origin_offset_cm, asset_name=asset_name,
                               asset_base_size_m=asset_base_size_m),
        notes=(
            "Boxes are axis aligned in NED. Unreal locations are centimetres with +Z up. Verify the "
            "PlayerStart offset in the built level before trusting these coordinates."
        ),
    )


# ======================================================================================= verification
def _resolve_actor(spec: SceneActorSpec, scene_objects: Sequence[str], tag_actors: Sequence[str]
                   ) -> tuple[str | None, str, list[str]]:
    """Find the level actor that should carry this obstacle.

    Order: exact actor name, exact obstacle name, exact tag, tag route, then a normalised containment
    match. An ambiguous containment match resolves to nothing on purpose; guessing between two bodies
    would be exactly the silent error this module exists to prevent.
    """
    objects = list(scene_objects)
    seen: list[str] = []
    for candidate, route in ((spec.actor_name, "exact_name"),
                             (spec.obstacle_name, "exact_name"),
                             (spec.actor_tag, "exact_tag")):
        if candidate in seen:
            continue
        seen.append(candidate)
        if candidate in objects:
            return candidate, route, [candidate]
    for name in tag_actors:
        if name in objects and _matches(spec.actor_tag, name):
            return name, "tag_route", [name]
    hits = sorted({name for candidate in seen for name in objects if _matches(candidate, name)})
    if len(hits) == 1:
        return hits[0], "normalized", hits
    return None, "none", hits


def _measure_actor(rpc: _SceneRpc, actor_name: str) -> tuple[ActorMeasurement, bool]:
    """Read pose and scale for one actor. Returns (measurement, pose_available)."""
    measurement = ActorMeasurement()
    pose_value, pose_error = rpc.try_call("simGetObjectPose", actor_name)
    if pose_error:
        measurement.unmeasured_reason = pose_error
        return measurement, False
    decoded = _decode_pose(pose_value)
    if decoded is None:
        measurement.unmeasured_reason = (
            "simGetObjectPose returned a NaN pose. The pinned server answers that way for an actor "
            "missing from scene_object_map (WorldSimApi.cpp:367-369). That map is filled once at level "
            "start (SimModeBase.cpp:154 -> AirBlueprintLib.cpp:265-274) and extended by simSpawnObject "
            "(WorldSimApi.cpp:135), so an actor that appeared by any other route is listed but has no "
            "pose."
        )
        return measurement, False
    position, yaw = decoded
    measurement.observed_position_ned_m = position
    measurement.observed_yaw_rad = yaw
    measurement.pose_source = "simGetObjectPose"

    scale_value, scale_error = rpc.try_call("simGetObjectScale", actor_name)
    if scale_error:
        measurement.scale_source = ""
        measurement.unmeasured_reason = scale_error
        return measurement, True
    scale = _decode_vec3(scale_value)
    if scale is None:
        measurement.unmeasured_reason = "simGetObjectScale returned a value this client cannot decode"
        return measurement, True
    if (scale.x, scale.y, scale.z) == (0.0, 0.0, 0.0):
        # WorldSimApi.cpp:383 returns a zero vector for an unknown actor; a real actor never has zero
        # scale, so this is "not found", not "scaled to nothing".
        measurement.unmeasured_reason = (
            "simGetObjectScale returned a zero vector (actor not in scene_object_map)")
        return measurement, True
    measurement.observed_scale = scale
    measurement.scale_source = "simGetObjectScale"
    return measurement, True

def _extent_evidence(measurement: ActorMeasurement, spec: SceneActorSpec,
                     entry: GeometryContractEntry | None, *, fixture_declared: bool) -> None:
    """Decide where this actor's metre half-extents may come from, and record the decision.

    Order of evidence, strongest first. Nothing here invents a number: when no rule applies the extent
    stays ``None`` with ``extent_source="unmeasured"``, which later makes the actor ``unverifiable``.
    """
    scale = measurement.observed_scale
    if entry is not None and scale is not None:
        measurement.observed_half_extent_m = entry.half_extent_from_scale(scale)
        measurement.extent_source = "scale_times_contract_base"
        measurement.extent_evidence = (
            f"live simGetObjectScale {scale.as_tuple()} times {entry.provenance_line()}"
        )
    elif (entry is not None and entry.world_half_extent_m is not None
            and entry.accept_world_half_extent_without_live_scale):
        measurement.observed_half_extent_m = entry.world_half_extent_m.model_copy()
        measurement.extent_source = "contract_declared_world_extent"
        measurement.extent_evidence = (
            f"reviewed contract entry {entry.pattern!r}: world half-extent "
            f"{entry.world_half_extent_m.as_tuple()} m measured by {entry.measured_by} "
            f"({entry.measurement_method}, {entry.measured_at}); this server did not supply a live "
            "scale, so the live level contributed the position only"
        )
    elif fixture_declared:
        # The in-repo fixture fake builds its world FROM the manifest it is handed, so its extents are
        # exactly the expected ones by construction. That is a software double, not a measurement, and
        # the report says so through evidence_class="software_double". A genuine server can never reach
        # this branch: the adapter sets is_fixture_fake only when the private probe
        # __colassure_fixture_fake__ answers True, which upstream Colosseum does not bind.
        measurement.observed_half_extent_m = spec.expected_half_extent_m.model_copy()
        measurement.extent_source = "fixture_fake_declared"
        measurement.extent_evidence = (
            "in-repo fixture fake: the level was created from this manifest, so the extent is declared, "
            "not measured. Software test double, never experimental evidence."
        )
    elif scale is not None and spec.asset_base_size_m:
        measurement.observed_half_extent_m = Vec3(
            x=abs(scale.x) * spec.asset_base_size_m / 2.0,
            y=abs(scale.y) * spec.asset_base_size_m / 2.0,
            z=abs(scale.z) * spec.asset_base_size_m / 2.0,
        )
        measurement.extent_source = "declared_asset_base_unreviewed"
        measurement.extent_evidence = (
            f"scale {scale.as_tuple()} times the base size {spec.asset_base_size_m} m passed to this "
            f"call for asset {spec.asset_name!r}. Nobody reviewed that base size, so it is reported "
            "for information and does NOT satisfy the gate."
        )
    else:
        measurement.extent_source = "unmeasured"
        measurement.extent_evidence = (
            "no extent evidence: no world-extent RPC exists at the pin, "
            + ("simGetObjectScale gave no usable multiplier, " if scale is None else "")
            + "and no reviewed geometry-contract entry covers this actor"
        )
        return
    expected = spec.expected_half_extent_m
    observed = measurement.observed_half_extent_m
    if observed is not None:
        measurement.extent_error_m = max(abs(observed.x - expected.x), abs(observed.y - expected.y),
                                         abs(observed.z - expected.z))


def _check_actor(spec: SceneActorSpec, scene_objects: Sequence[str], tag_actors: Sequence[str],
                 rpc: _SceneRpc, *, position_tolerance_m: float, scale_tolerance: float,
                 yaw_tolerance_rad: float, contract: GeometryContract | None = None,
                 fixture_declared: bool = False) -> SceneActorCheck:
    """Resolve, measure and classify one expected actor.

    ``status`` is ``ok`` only when the position AND the extent are both backed by evidence. An actor at
    the right place whose size nothing establishes is ``unverifiable``: the evaluator's clearance
    geometry would otherwise rest on an assumption that the level never confirmed.
    """
    resolved, route, candidates = _resolve_actor(spec, scene_objects, tag_actors)
    check = SceneActorCheck(
        spec=spec, resolved_actor_name=resolved, match_route=route,  # type: ignore[arg-type]
        candidate_names=candidates, position_tolerance_m=position_tolerance_m,
        scale_tolerance=scale_tolerance, yaw_tolerance_rad=yaw_tolerance_rad,
    )
    if resolved is None:
        if candidates:
            check.status = "unverifiable"
            check.explanation = (
                f"{len(candidates)} level actors could be {spec.obstacle_name!r} ({candidates}); the "
                "match is ambiguous, so no geometry claim can be made"
            )
        else:
            check.status = "missing"
            check.explanation = (
                f"no actor named {spec.actor_name!r} / {spec.obstacle_name!r} / tagged "
                f"{spec.actor_tag!r} exists in the level"
            )
        return check

    entry = contract.entry_for(resolved) if contract is not None else None
    check.contract_pattern = entry.pattern if entry is not None else None
    measurement, pose_ok = _measure_actor(rpc, resolved)
    _extent_evidence(measurement, spec, entry, fixture_declared=fixture_declared)
    check.measurement = measurement
    if not pose_ok:
        listed = resolved in set(scene_objects)
        check.status = "unverifiable" if listed else "missing"
        check.explanation = (
            f"actor {resolved!r} is {'listed but' if listed else 'not listed and'} not measurable: "
            f"{measurement.unmeasured_reason}"
        )
        return check

    observed = measurement.observed_position_ned_m
    if observed is None:  # defensive: pose_ok implies a position, but never trust that silently
        check.status = "unverifiable"
        check.explanation = f"actor {resolved!r} reported no usable position"
        return check
    measurement.position_error_m = observed.distance_to(spec.expected_position_ned_m)
    measurement.yaw_error_rad = _angle_error(float(measurement.observed_yaw_rad or 0.0),
                                             spec.expected_yaw_rad)
    if measurement.position_error_m > position_tolerance_m:
        check.status = "pose_mismatch"
        check.explanation = (
            f"{spec.obstacle_name} sits {measurement.position_error_m:.3f} m from the manifest "
            f"position (tolerance {position_tolerance_m:g} m): expected "
            f"{spec.expected_position_ned_m.as_tuple()}, measured {observed.as_tuple()}"
        )
        return check
    if measurement.yaw_error_rad > yaw_tolerance_rad:
        check.status = "pose_mismatch"
        check.explanation = (
            f"{spec.obstacle_name} is rotated {measurement.yaw_error_rad:.3f} rad from the expected "
            f"{spec.expected_yaw_rad:.3f} rad (tolerance {yaw_tolerance_rad:g} rad); a rotated body is "
            "not the axis-aligned box the evaluator uses for clearance"
        )
        return check
    check.position_verified = True

    scale_mismatch = _scale_verdict(spec, measurement, scale_tolerance)
    if scale_mismatch is not None:
        check.status = "scale_mismatch"
        check.explanation = scale_mismatch
        return check

    if measurement.extent_source in {"unmeasured", "declared_asset_base_unreviewed"}:
        check.status = "unverifiable"
        check.explanation = (
            f"{spec.obstacle_name} matched actor {resolved!r} via {route} and sits "
            f"{measurement.position_error_m:.3f} m from the manifest position, but its size is "
            f"UNVERIFIABLE: {measurement.extent_evidence}. The evaluator would assume half-extents "
            f"{spec.expected_half_extent_m.as_tuple()} m that nothing supports, so this actor blocks a "
            "scientific episode. Bind it in a reviewed geometry contract "
            f"({GEOMETRY_CONTRACT_ENV_VAR} or the geometry_contract argument)."
        )
        return check

    check.extent_verified = True
    check.status = "ok"
    check.explanation = (
        f"{spec.obstacle_name} matched actor {resolved!r} via {route}; position error "
        f"{measurement.position_error_m:.3f} m within {position_tolerance_m:g} m; extent from "
        f"{measurement.extent_source}: {measurement.extent_evidence}"
    )
    return check


def _scale_verdict(spec: SceneActorSpec, measurement: ActorMeasurement,
                   scale_tolerance: float) -> str | None:
    """Compare scale and derived extent against the manifest. Returns why it is wrong, else None.

    Returning None does NOT mean "verified": an actor with no measurement at all also lands here. The
    caller decides that separately from ``measurement.extent_source``, which is the whole point of the
    split (the previous version treated "no scale available" as "no objection" and passed the actor).
    """
    scale = measurement.observed_scale
    if scale is not None and spec.expected_scale is not None:
        errors = [abs(observed - expected) / max(abs(expected), 1e-9)
                  for observed, expected in ((abs(scale.x), spec.expected_scale.x),
                                             (abs(scale.y), spec.expected_scale.y),
                                             (abs(scale.z), spec.expected_scale.z))]
        measurement.scale_relative_error = max(errors)
        if measurement.scale_relative_error > scale_tolerance:
            return (
                f"{spec.obstacle_name} has scale {(scale.x, scale.y, scale.z)} but the manifest box "
                f"needs {spec.expected_scale.as_tuple()} from a {spec.asset_base_size_m} m asset: "
                f"relative error {measurement.scale_relative_error:.3f} exceeds {scale_tolerance:g}"
            )
    if measurement.extent_error_m is not None:
        observed_extent = measurement.observed_half_extent_m
        expected_extent = spec.expected_half_extent_m
        # A wide launch deck must not tolerate metre-scale height errors just because its x extent
        # is large. Every axis must match its own dimension, including thin support surfaces.
        axis_errors = [] if observed_extent is None else [
            abs(observed-expected) > max(scale_tolerance*abs(expected), 1e-6)
            for observed, expected in zip(observed_extent.as_tuple(), expected_extent.as_tuple(),
                                           strict=True)]
        if any(axis_errors):
            return (
                f"{spec.obstacle_name} measures half-extents "
                f"{observed_extent.as_tuple() if observed_extent else None} "
                f"({measurement.extent_source}: {measurement.extent_evidence}) "
                f"but the manifest says {spec.expected_half_extent_m.as_tuple()}: error "
                f"{measurement.extent_error_m:.3f} m exceeds the per-axis "
                f"{scale_tolerance:g} relative tolerance"
            )
    return None


def _alias_of(name: str, checks: Sequence[SceneActorCheck]) -> SceneActorCheck | None:
    """The expected actor whose name or tag could also describe ``name``."""
    for check in checks:
        spec = check.spec
        if any(_matches(candidate, name)
               for candidate in (spec.actor_name, spec.obstacle_name, spec.actor_tag)):
            return check
    return None

def _body_intrudes(volume: Box, position: Vec3, half_extent: Vec3 | None) -> bool:
    """Does this body reach into the mission volume?

    With a known half-extent the real axis-aligned box is tested, which is the honest question. Without
    one only the actor ORIGIN can be tested, and a large body whose origin sits outside the volume can
    still intrude: that limit is recorded as a caveat rather than hidden, because no RPC at the pin
    returns world bounds.
    """
    if half_extent is None:
        return volume.contains(position)
    return (
        position.x + half_extent.x >= volume.x_min and position.x - half_extent.x <= volume.x_max
        and position.y + half_extent.y >= volume.y_min and position.y - half_extent.y <= volume.y_max
        and position.z + half_extent.z >= volume.z_min and position.z - half_extent.z <= volume.z_max
    )


def _take_inventory(rpc: _SceneRpc, scene_objects: Sequence[str], checks: Sequence[SceneActorCheck],
                    volume: Box, *, ignored_actor_regex: str, max_probes: int,
                    position_tolerance_m: float, name_regex: str,
                    contract: GeometryContract | None = None,
                    wall_budget_s: float = 120.0,
                    exact_exclusions: dict[str, str] | None = None,
                    ) -> tuple[list[ExtraActorFinding], SceneInventoryReport, list[str], list[str]]:
    """Establish the relevant world by geometry, and say plainly when that is impossible.

    Every actor the level lists is either (a) one of the expected bodies, (b) removed by a DECLARED
    exclusion pattern and recorded with the pattern that removed it, or (c) probed with
    ``simGetObjectPose`` and placed against the mission volume. Nothing is skipped because its name
    lacks a study keyword: a collidable body called ``BP_Crane_7`` changes clearance exactly as much as
    one called ``inspection_tower``.

    Returns ``(findings, inventory, caveats, aliases)``. The inventory is INCOMPLETE -- which blocks a
    scientific episode -- when an actor cannot be placed, or when there are more actors than the probe
    budget, because in both cases the run cannot state what else is inside the flight volume.
    """
    started = time.monotonic()
    ignored = re.compile(ignored_actor_regex)
    claimed = {check.resolved_actor_name for check in checks if check.resolved_actor_name}
    findings: list[ExtraActorFinding] = []
    caveats: list[str] = []
    aliases: list[str] = []
    exclusions: list[InventoryExclusion] = []
    candidates: list[str] = []
    for name in sorted(scene_objects):
        if name in claimed:
            continue
        if name in (exact_exclusions or {}):
            exclusions.append(InventoryExclusion(
                actor_name=name, pattern=f"exact:{name}", reason=exact_exclusions[name]))
            continue
        if ignored.search(name):
            exclusions.append(InventoryExclusion(
                actor_name=name, pattern=ignored_actor_regex,
                reason=("declared irrelevant for clearance and occlusion (vehicle, player start, "
                        "ground, sky). Listed here so the exclusion is visible and reviewable."),
            ))
            continue
        candidates.append(name)

    inventory = SceneInventoryReport(
        name_regex=name_regex,
        listed_actors=len(scene_objects),
        matched_actors=len(claimed),
        excluded=exclusions,
        candidate_actors=len(candidates), max_probes=max_probes, wall_budget_s=wall_budget_s,
    )
    if len(candidates) > max_probes:
        inventory.incomplete_reasons.append(
            f"{len(candidates)} actors needed a pose probe but the budget is {max_probes}; "
            f"{len(candidates) - max_probes} actor(s) were never placed, so unexpected geometry may "
            "exist inside the mission volume. Narrow name_regex, declare exclusions, or raise the "
            "budget deliberately."
        )
    probed_outside = 0
    for name in candidates[:max_probes]:
        remaining = wall_budget_s - (time.monotonic() - started)
        if remaining <= 0:
            inventory.budget_exhausted = True
            break
        ours = name.startswith(COLASSURE_ACTOR_PREFIX)
        alias = _alias_of(name, checks)
        pose_value, error = rpc.try_call("simGetObjectPose", name, timeout_s=remaining)
        decoded = None if error else _decode_pose(pose_value)
        inventory.probed_actors += 1
        if decoded is None:
            # A name that merely LOOKS like an expected actor is not forgiven here any more. Without a
            # pose nothing can say whether this is a second identifier for a body we already measured
            # or a second body, and guessing from the name is the hole this rewrite closes.
            inventory.unplaced_actors.append(name)
            why = error or ("NaN pose: the actor is absent from scene_object_map, "
                            "WorldSimApi.cpp:367-369")
            inventory.incomplete_reasons.append(
                f"{name!r} is listed by simListSceneObjects but simGetObjectPose returned no usable "
                f"pose ({why}), so it cannot be placed inside or outside the mission volume"
            )
            findings.append(ExtraActorFinding(
                actor_name=name, status="extra" if ours else "unverifiable",
                note=("a leftover actor with our own prefix that cannot be measured" if ours else
                      "unexpected actor; position not measurable, so the relevant world is unknown"),
            ))
            continue
        position, _yaw = decoded
        if alias is not None and not ours:
            matched_position = alias.measurement.observed_position_ned_m
            if (matched_position is not None
                    and position.distance_to(matched_position) <= position_tolerance_m):
                aliases.append(name)
                continue
        entry = contract.entry_for(name) if contract is not None else None
        half_extent: Vec3 | None = None
        if entry is not None:
            remaining = wall_budget_s - (time.monotonic() - started)
            if remaining <= 0:
                inventory.budget_exhausted = True
                inventory.incomplete_reasons.append(f"{name!r}: budget expired before scale measurement")
                break
            scale_value, scale_error = rpc.try_call("simGetObjectScale", name, timeout_s=remaining)
            scale = None if scale_error else _decode_vec3(scale_value)
            if scale is not None and (scale.x, scale.y, scale.z) != (0.0, 0.0, 0.0):
                half_extent = entry.half_extent_from_scale(scale)
            elif entry.accept_world_half_extent_without_live_scale:
                half_extent = entry.world_half_extent_m
        inside = _body_intrudes(volume, position, half_extent)
        if not inside:
            probed_outside += 1
            if not ours:
                continue  # measured, outside the flight volume, not ours: not evidence about this run
        findings.append(ExtraActorFinding(
            actor_name=name, observed_position_ned_m=position, observed_half_extent_m=half_extent,
            inside_mission_volume=inside,
            distance_to_volume_m=max(0.0, volume.signed_exceedance(position)),
            status="extra",
            note=("inside the mission volume: it changes clearance and occlusion" if inside else
                  "outside the mission volume but carries our actor prefix, so an earlier run left it"),
        ))
        if inside:
            inventory.inside_volume.append(name)
    if probed_outside:
        caveats.append(
            f"{probed_outside} actor(s) were placed outside the mission volume by their ORIGIN. No RPC "
            "at the pin returns world bounds, so a large body whose origin lies outside the volume can "
            "still intrude into it unless a geometry-contract entry supplies its extent"
        )
    inventory.elapsed_s = time.monotonic() - started
    inventory.unprobed_actors = len(candidates)-inventory.probed_actors
    if inventory.elapsed_s >= wall_budget_s:
        inventory.budget_exhausted = True
    if inventory.budget_exhausted:
        inventory.incomplete_reasons.append(
            f"inventory wall budget {wall_budget_s:g} s exhausted after "
            f"{inventory.probed_actors}/{len(candidates)} actor pose probes; "
            f"{inventory.unprobed_actors} actors remain unprobed")
    inventory.complete = not inventory.incomplete_reasons
    return findings, inventory, caveats, aliases


FIXTURE_EVIDENCE_CAVEAT = (
    "SOFTWARE DOUBLE: the adapter reports the in-repo fixture fake, whose world is built from this very "
    "manifest. Extents are declared by that construction, not measured. This report proves the code "
    "path, never the physics, and must never be recorded as experimental evidence."
)


def verify_scene(
    adapter: SimAdapter | Any,
    manifest: ScenarioManifest,
    protocol: ProtocolConfig | None = None,
    position_tolerance_m: float = 0.25,
    scale_tolerance: float = 0.05,
    *,
    yaw_tolerance_rad: float = 0.05,
    mode: SceneMode = "verify_only",
    expected: Sequence[SceneActorSpec] | None = None,
    geometry_provenance: GeometryProvenance = "manifest",
    geometry_contract: GeometryContract | None = None,
    geometry_contract_path: str | Path | None = None,
    ignored_actor_regex: str = DEFAULT_IGNORED_ACTOR_REGEX,
    name_regex: str = ".*",
    mission_volume_margin_m: float = 1.0,
    max_inventory_probes: int = MAX_INVENTORY_PROBES,
    inventory_wall_budget_s: float = 120.0,
    inventory_exclusions: dict[str, str] | None = None,
    allow_empty_scene: bool = False,
) -> SceneVerificationReport:
    """Measure the live level and compare it with the manifest, body by body.

    A name match never produces ``ok``, and neither does a matching position on its own. For every
    expected obstacle this reads the actual transform through ``simGetObjectPose``, establishes the
    metre extent from the evidence rules in :func:`_extent_evidence`, and compares both against the
    manifest with the stated tolerances. Then it establishes the relevant world: every other listed
    actor is either excluded by a declared rule or placed by measurement.

    ``report.ok`` is true only when the positions match, every extent rests on evidence, the inventory
    is complete, no unexpected body is in the volume, and the scene rules hold. ``blocking_reasons``
    says which of those failed, one tagged line each.

    ``protocol`` is optional only so that existing call sites keep working; pass it, because without it
    the mission volume falls back to the manifest bounding box instead of the real geofence.
    """
    rpc = _SceneRpc(adapter)
    contract = resolve_geometry_contract(adapter, geometry_contract, geometry_contract_path)
    fixture_declared = bool(getattr(adapter, "is_fixture_fake", False))
    listing_error = ""
    try:
        scene_objects = rpc.list_scene_objects(name_regex)
    except SceneUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - an unusable listing is evidence, not a crash
        # Without a listing the relevant world cannot be established at all. That is reported as an
        # INCOMPLETE inventory (which blocks) instead of an exception, so the episode record keeps the
        # reason the scene could not be proven.
        scene_objects = []
        listing_error = f"simListSceneObjects failed: {type(exc).__name__}: {exc}"
    tag_actors, tag_available = rpc.list_scene_objects_by_tag(".*")
    specs = list(expected) if expected is not None else expected_actors(
        manifest, mode=mode, geometry_provenance=geometry_provenance)
    volume = mission_volume(manifest, protocol, mission_volume_margin_m)

    checks = [_check_actor(spec, scene_objects, tag_actors, rpc,
                           position_tolerance_m=position_tolerance_m,
                           scale_tolerance=scale_tolerance,
                           yaw_tolerance_rad=yaw_tolerance_rad,
                           contract=contract, fixture_declared=fixture_declared)
              for spec in specs]

    extras, inventory, caveats, aliases = _take_inventory(
        rpc, scene_objects, checks, volume,
        ignored_actor_regex=ignored_actor_regex, max_probes=max_inventory_probes,
        position_tolerance_m=position_tolerance_m, name_regex=name_regex, contract=contract,
        wall_budget_s=inventory_wall_budget_s, exact_exclusions=inventory_exclusions)
    if listing_error:
        inventory.incomplete_reasons.insert(0, listing_error + "; the level could not be enumerated, so "
                                            "nothing can be said about what else is in the mission volume")
        inventory.complete = False

    matched = {check.spec.actor_tag: [check.resolved_actor_name]
               for check in checks if check.resolved_actor_name}
    missing = [check.spec.actor_tag for check in checks if check.status == "missing"]
    mismatched = [check.spec.actor_tag for check in checks
                  if check.status in {"pose_mismatch", "scale_mismatch"}]
    unverifiable = [check.spec.actor_tag for check in checks if check.status == "unverifiable"]
    unmeasured_extents = [check.spec.actor_tag for check in checks if not check.extent_verified]
    tag_missing = [spec.actor_tag for spec in specs
                   if tag_available and not any(_matches(spec.actor_tag, name) for name in tag_actors)]
    errors = [check.measurement.position_error_m for check in checks
              if check.measurement.position_error_m is not None]
    blocking_extras = [finding for finding in extras if finding.status == "extra"]
    unverifiable_extras = [finding for finding in extras if finding.status == "unverifiable"]

    positions_verified = bool(specs) and all(check.position_verified for check in checks)
    measurement_complete = all(check.extent_verified for check in checks) and not unverifiable_extras
    if unmeasured_extents:
        caveats.append(
            "extent evidence is missing for " + ", ".join(unmeasured_extents) + ": the pinned RPC "
            "surface has no world-extent query, so metre extents need a live scale plus a reviewed "
            "geometry-contract entry naming the source mesh and its base size"
        )

    blocking_reasons: list[str] = []
    if not specs and not allow_empty_scene:
        # WHY zero obstacles is a BLOCK and not a pass: this study measures clearance to real bodies and
        # occlusion of the inspection asset. A scene with no obstacle cannot produce either, so an
        # episode flown in it answers a different question than the protocol asks. The rule is visible
        # here rather than hidden in a caller: allow_empty_scene=True exists for software smoke tests
        # only, and it is recorded in the report when it is used.
        blocking_reasons.append(
            "[scene] the binding expects ZERO obstacles: this study needs real bodies for clearance and "
            "occlusion, so an empty scene cannot answer the protocol question. Zero obstacles blocks a "
            "scientific episode; pass allow_empty_scene=True only for a software smoke test."
        )
    if not specs and allow_empty_scene:
        caveats.append("allow_empty_scene=True: an obstacle-free scene was accepted deliberately; this "
                       "is a software smoke test, not a scientific episode")
    if missing:
        blocking_reasons.append(f"[position] expected actor(s) absent from the level: {missing}")
    if mismatched:
        blocking_reasons.append(f"[position] actor(s) do not match the manifest geometry: {mismatched}")
    if unverifiable and not unmeasured_extents:
        blocking_reasons.append(f"[position] actor(s) could not be measured: {unverifiable}")
    if not measurement_complete:
        detail = ", ".join(unmeasured_extents) or "an unmeasurable extra actor"
        blocking_reasons.append(
            f"[measurement] extent evidence is missing for {detail}. The evaluator's clearance geometry "
            "would rest on an assumption the level never confirmed. Bind these actors in a reviewed "
            f"geometry contract ({GEOMETRY_CONTRACT_ENV_VAR} or the geometry_contract argument)."
        )
    if blocking_extras:
        blocking_reasons.append(
            "[extra] unexpected actor(s) relevant to this run: "
            f"{[finding.actor_name for finding in blocking_extras]}"
        )
    if not inventory.complete:
        blocking_reasons.append(
            "[inventory] the relevant world could not be established: "
            + "; ".join(inventory.incomplete_reasons)
        )
    if contract is not None and contract.status == "example":
        blocking_reasons.append(
            f"[contract] the geometry contract {contract.source_path or '(in memory)'} is marked "
            "status: example. An example is a template, not a review; set status: reviewed and name "
            "reviewed_by after an attributed review of the measurements."
        )

    report = SceneVerificationReport(
        mode=mode,
        scenario_id=manifest.scenario_id,
        scene_name=manifest.scene_name,
        checked_at=datetime.now(UTC).isoformat(timespec="seconds"),
        geometry_provenance=geometry_provenance,
        evidence_class="software_double" if fixture_declared else "scientific_candidate",
        scene_object_count=len(scene_objects),
        position_tolerance_m=position_tolerance_m,
        scale_tolerance=scale_tolerance,
        yaw_tolerance_rad=yaw_tolerance_rad,
        mission_volume=volume,
        actors=checks,
        extras=extras,
        inventory=inventory,
        geometry_contract=contract.summary() if contract is not None else None,
        expected_actors=[spec.actor_tag for spec in specs],
        matched_actors=matched,
        missing_actors=missing,
        extra_actors=[finding.actor_name for finding in blocking_extras],
        alias_actors=sorted(aliases),
        mismatched_actors=mismatched,
        unverifiable_actors=unverifiable,
        unmeasured_extent_actors=unmeasured_extents,
        tag_route_available=tag_available,
        tag_route_actors=sorted(tag_actors),
        tag_route_missing=tag_missing,
        route_disagreement=sorted(set(tag_missing) - set(missing)),
        max_position_error_m=max(errors) if errors else None,
        positions_verified=positions_verified,
        measurement_complete=measurement_complete,
        inventory_complete=inventory.complete,
        blocking_reasons=blocking_reasons,
        ok=not blocking_reasons,
        caveats=caveats,
        detail="",
    )
    if fixture_declared:
        report.caveats.insert(0, FIXTURE_EVIDENCE_CAVEAT)
    report.detail = _verification_detail(report)
    return report


def _verification_detail(report: SceneVerificationReport) -> str:
    """One sentence a human can act on. The phrasing names the failure, never softens it."""
    expected_count = len(report.actors)
    if report.missing_actors:
        return (
            f"{len(report.missing_actors)} expected actor(s) are absent from the level: "
            f"{report.missing_actors}. The scene is not the scene the manifest describes."
        )
    if report.mismatched_actors:
        first = next(check for check in report.actors
                     if check.status in {"pose_mismatch", "scale_mismatch"})
        return (
            f"{len(report.mismatched_actors)} actor(s) do not match the manifest geometry: "
            f"{first.explanation}"
        )
    if report.unverifiable_actors:
        first = next(check for check in report.actors if check.status == "unverifiable")
        return (
            f"{len(report.unverifiable_actors)} actor(s) could not be measured, so the scene is "
            f"unproven: {first.explanation}"
        )
    if report.extra_actors:
        return (
            f"{len(report.extra_actors)} unexpected actor(s) sit in the mission volume: "
            f"{report.extra_actors}. Extra geometry changes clearance and occlusion."
        )
    if not report.inventory_complete:
        return (
            "the relevant world could not be established, so the scene is unproven: "
            + "; ".join(report.inventory.incomplete_reasons)
        )
    if not report.ok:
        return "the scene does not satisfy the gate: " + "; ".join(report.blocking_reasons)
    sources = sorted({check.measurement.extent_source for check in report.actors})
    return (
        f"all {expected_count} expected actors are present at the manifest positions "
        f"(max position error {report.max_position_error_m:.3f} m, tolerance "
        f"{report.position_tolerance_m:g} m, extents from {sources}) in "
        f"{report.scene_object_count} scene objects, inventory complete over "
        f"{report.inventory.probed_actors} probed actor(s)"
        if report.max_position_error_m is not None else
        f"all {expected_count} expected actors resolved, but no position was measured"
    )


# ======================================================================================= instantiate
def _assert_contract_agrees_with_asset(contract: GeometryContract | None,
                                       specs: Sequence[SceneActorSpec], asset_name: str,
                                       asset_base_size_m: float) -> None:
    """Refuse to spawn geometry that contradicts the reviewed contract.

    If the contract says the actor is a 2 m mesh and this call spawns a 1 m mesh scaled as if it were
    2 m, every later extent number would be wrong in a way the report could not see, because the
    contract is what converts a scale multiplier into metres.
    """
    if contract is None:
        return
    problems: list[str] = []
    for spec in specs:
        entry = contract.entry_for(spec.actor_name)
        if entry is None:
            continue
        if entry.mesh_name != asset_name:
            problems.append(
                f"{spec.actor_name}: the contract entry {entry.pattern!r} names mesh "
                f"{entry.mesh_name!r} but this call spawns {asset_name!r}"
            )
        base = entry.mesh_base_size_m.as_tuple()
        if max(abs(value - asset_base_size_m) for value in base) > 1e-9:
            problems.append(
                f"{spec.actor_name}: the contract entry {entry.pattern!r} states base size {base} m "
                f"but this call assumes a cubic {asset_base_size_m} m mesh"
            )
    if problems:
        raise SceneError(
            "the spawn parameters contradict the reviewed geometry contract:\n  " + "\n  ".join(problems),
            remedy=("Spawn the mesh the contract describes, or update the contract after measuring the "
                    "mesh you really spawn. Do not let the two disagree: the contract is what turns a "
                    "scale multiplier into metres."),
        )


def instantiate_scene(
    adapter: SimAdapter | Any,
    manifest: ScenarioManifest,
    protocol: ProtocolConfig | None = None,
    dry_run: bool = False,
    *,
    asset_name: str = STUDY_BOX_ASSET_NAME,
    asset_base_size_m: float = 1.0,
    physics_enabled: bool = False,
    verify: bool = True,
    position_tolerance_m: float = 0.25,
    scale_tolerance: float = 0.05,
    geometry_contract: GeometryContract | None = None,
    geometry_contract_path: str | Path | None = None,
    allow_empty_scene: bool = False,
    max_inventory_probes: int = MAX_INVENTORY_PROBES,
    inventory_wall_budget_s: float = 120.0,
    inventory_exclusions: dict[str, str] | None = None,
) -> SceneBindingReport:
    """Create the manifest geometry in a running level, then prove it landed where it should.

    Rules this function enforces, in order:

    1. **Fail closed on capability.** ``simListAssets`` must answer and must contain ``asset_name``
       before anything is spawned. ``WorldSimApi.cpp:94-98`` dereferences the asset lookup without a
       null check, so an unknown asset name is a crash risk, not a polite error.
    2. **Idempotent.** Actors whose name starts with :data:`COLASSURE_ACTOR_PREFIX` are destroyed first,
       so a re-run produces the same level rather than a second copy of every box.
    3. **Never touch foreign actors.** Only prefixed names are destroyed. Everything else in the level
       belongs to whoever built it.
    4. **Deterministic.** Obstacles are spawned in sorted name order, so two runs issue identical RPC
       sequences and any renaming by the server is reproducible.
    5. **Verified.** The report carries the measured verification, and ``ok`` is false unless it passed.

    ``dry_run=True`` performs the capability checks and returns the plan without issuing a single
    mutating RPC.
    """
    rpc = _SceneRpc(adapter)
    if not rpc.available:
        raise SceneUnavailable(
            "the adapter exposes no RPC route, so scene instantiation is impossible",
            remedy="Pass a ColosseumAdapter connected to the simulator.",
        )
    specs = expected_actors(manifest, mode="instantiate", asset_name=asset_name,
                            asset_base_size_m=asset_base_size_m)
    contract = resolve_geometry_contract(adapter, geometry_contract, geometry_contract_path)
    _assert_contract_agrees_with_asset(contract, specs, asset_name, asset_base_size_m)
    report = SceneBindingReport(
        mode="instantiate",
        scenario_id=manifest.scenario_id,
        scene_name=manifest.scene_name,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        dry_run=dry_run,
        asset_name=asset_name,
        asset_base_size_m=asset_base_size_m,
        planned_actors=specs,
    )

    assets_value, assets_error = rpc.try_call("simListAssets")
    if assets_error:
        raise SceneUnavailable(
            f"this simulator does not answer simListAssets ({assets_error}), so study geometry cannot "
            "be spawned at runtime",
            remedy=("Use scene mode verify_only with a map built by deploy/scripts/"
                    "generate_study_scene.py, or scene mode qualified_map on a third-party map."),
            report=report,
        )
    assets = [str(name) for name in (assets_value or [])]
    report.available_assets_sampled = sorted(assets)[:25]
    if asset_name not in assets:
        raise SceneUnavailable(
            f"asset {asset_name!r} is not in the cooked asset registry ({len(assets)} assets); "
            "refusing to call simSpawnObject with a name the server cannot resolve",
            remedy=("Cook a package that contains the mesh, or pass asset_name= one of the assets the "
                    "server lists. WorldSimApi.cpp:94-98 dereferences an unknown asset without a null "
                    "check, so this check protects the simulator process."),
            report=report,
        )

    existing = [name for name in rpc.list_scene_objects(f".*{COLASSURE_ACTOR_PREFIX}.*")
                if name.startswith(COLASSURE_ACTOR_PREFIX)]
    report.planned_destroy = sorted(existing)

    if dry_run:
        report.ok = True
        report.rpc_methods_used = list(rpc.used)
        report.detail = (
            f"dry run: would destroy {len(existing)} previously spawned actor(s) and spawn "
            f"{len(specs)} box actor(s) from asset {asset_name!r}; no mutating RPC was issued"
        )
        return report

    for name in sorted(existing):
        value, error = rpc.try_call("simDestroyObject", name)
        if error or value is False:
            report.destroy_failures.append(f"{name}: {error or 'simDestroyObject returned False'}")
        else:
            report.destroyed.append(name)
    if report.destroy_failures:
        report.rpc_methods_used = list(rpc.used)
        report.detail = f"could not remove earlier study actors: {report.destroy_failures}"
        raise SceneError(
            "refusing to spawn study geometry on top of actors from an earlier run that could not be "
            f"removed: {report.destroy_failures}",
            remedy=("Restart the simulator to get a clean level, then re-run. Leftover duplicates make "
                    "simSpawnObject rename our actors (WorldSimApi.cpp:108-118) and change the scene."),
            report=report,
        )

    placed: list[SceneActorSpec] = []
    for spec in specs:
        scale = spec.expected_scale
        if scale is None:  # pragma: no cover - expected_actors always sets it in this mode
            raise SceneError(f"no spawn scale computed for {spec.obstacle_name}",
                             remedy="Pass asset_base_size_m so extents can be converted to a scale.")
        pose = [list(spec.expected_position_ned_m.as_tuple()), [1.0, 0.0, 0.0, 0.0]]
        try:
            actual = rpc.call("simSpawnObject", spec.actor_name, asset_name, pose,
                              list(scale.as_tuple()), bool(physics_enabled), False)
        except Exception as exc:  # noqa: BLE001 - reported as a scene failure with its cause
            report.rpc_methods_used = list(rpc.used)
            raise SceneUnavailable(
                f"simSpawnObject refused {spec.actor_name!r} from asset {asset_name!r}: {exc}",
                remedy=("The server accepts the call only for assets in its cooked registry and only "
                        "while the level is loaded. Check simListAssets output and the map."),
                report=report,
            ) from exc
        actual_name = str(actual) if actual else spec.actor_name
        report.spawned.append(SpawnedActorRecord(
            obstacle_name=spec.obstacle_name,
            requested_name=spec.actor_name,
            actual_name=actual_name,
            asset_name=asset_name,
            position_ned_m=spec.expected_position_ned_m,
            yaw_rad=0.0,
            scale=scale,
            physics_enabled=bool(physics_enabled),
            renamed_by_simulator=actual_name != spec.actor_name,
        ))
        placed.append(spec.model_copy(update={"actor_name": actual_name}))

    report.rpc_methods_used = list(rpc.used)
    if not verify:
        report.ok = False
        report.detail = ("geometry was spawned but verification was disabled, so the scene is "
                         "unproven; a live episode must not use this binding")
        return report

    report.verification = verify_scene(
        adapter, manifest, protocol, position_tolerance_m, scale_tolerance,
        mode="instantiate", expected=placed, geometry_contract=contract,
        allow_empty_scene=allow_empty_scene,
        max_inventory_probes=max_inventory_probes,
        inventory_wall_budget_s=inventory_wall_budget_s,
        inventory_exclusions=inventory_exclusions,
    )
    report.ok = report.verification.ok
    report.detail = (
        f"spawned {len(report.spawned)} actor(s) from {asset_name!r} after removing "
        f"{len(report.destroyed)}; verification: {report.verification.detail}"
    )
    return report


# ======================================================================================= qualified map
DEFAULT_QUALIFIED_MAP_REGEX = r".*"
"""Default actor filter for the qualified-map path.

Upstream matches the WHOLE actor name (``std::regex_match``, AirBlueprintLib.cpp:441), so a substring
filter must be written as ``.*Rock.*``. The default lists everything and lets the mission volume do the
filtering, which costs one pose RPC per candidate actor.
"""


def derive_manifest_geometry(
    adapter: SimAdapter | Any,
    manifest: ScenarioManifest,
    protocol: ProtocolConfig | None = None,
    name_regex: str = DEFAULT_QUALIFIED_MAP_REGEX,
    *,
    geometry_contract: GeometryContract | None = None,
    geometry_contract_path: str | Path | None = None,
    placeholder_half_extent_m: Vec3 | None = None,
    default_kind: str = "building",
    ignored_actor_regex: str = DEFAULT_IGNORED_ACTOR_REGEX,
    mission_volume_margin_m: float = 0.0,
    max_actors: int = 40,
    yaw_tolerance_rad: float = 0.05,
) -> SceneDerivationResult:
    """Measure the natural actors of a third-party map and bind the manifest to what is really there.

    This is the honest version of "run the study on someone else's map". We do not pretend that the map
    contains our study scene. We measure the bodies that sit inside the mission volume and rewrite the
    manifest geometry from those measurements, so the evaluator's clearance geometry, the perception
    task and the occlusion all refer to real 3D bodies.

    What this costs, stated plainly:

    * The scenario definition changes. Obstacle positions are the map author's, not the protocol's.
      :class:`SceneDerivationProvenance` records that, and it must travel with the run metadata.
    * **Extents are not invented any more.** A half-extent needs a reviewed geometry-contract entry for
      that actor: its measured scale times the entry's mesh base size, or the entry's independently
      measured world half-extent. An actor with no entry keeps ``extent_source="unmeasured"``, is listed
      in ``actors_without_extent_evidence``, and makes the binding not ok. The earlier version filled in
      a default box, which handed the evaluator a clearance geometry nobody had measured (independent
      review, "qualified-map derivation similarly invents default extents").
    * Rotated bodies are approximated by axis-aligned boxes, which is recorded as a warning.
    """
    rpc = _SceneRpc(adapter)
    contract = resolve_geometry_contract(adapter, geometry_contract, geometry_contract_path)
    placeholder = placeholder_half_extent_m or Vec3(x=0.5, y=0.5, z=0.5)
    volume = mission_volume(manifest, protocol, mission_volume_margin_m)
    ignored = re.compile(ignored_actor_regex)
    scene_objects = rpc.list_scene_objects(name_regex)
    candidates = [name for name in sorted(scene_objects) if not ignored.search(name)]

    records: list[DerivedActorRecord] = []
    warnings: list[str] = []
    without_evidence: list[str] = []
    unplaced: list[str] = []
    extent_measurable = False
    for name in candidates[:max_actors]:
        pose_value, pose_error = rpc.try_call("simGetObjectPose", name)
        decoded = None if pose_error else _decode_pose(pose_value)
        if decoded is None:
            # Unmeasurable actors cannot become manifest geometry, and they are not silently dropped:
            # a body nobody could place may still sit in the flight volume.
            unplaced.append(name)
            continue
        position, yaw = decoded
        if not volume.contains(position):
            continue
        scale_value, scale_error = rpc.try_call("simGetObjectScale", name)
        scale = None if scale_error else _decode_vec3(scale_value)
        if scale is not None and (scale.x, scale.y, scale.z) == (0.0, 0.0, 0.0):
            scale = None
        entry = contract.entry_for(name) if contract is not None else None
        source: ExtentSource = "unmeasured"
        evidence = ""
        if entry is not None and scale is not None:
            extent = entry.half_extent_from_scale(scale)
            source = "scale_times_contract_base"
            evidence = f"live scale {scale.as_tuple()} times {entry.provenance_line()}"
            extent_measurable = True
        elif (entry is not None and entry.world_half_extent_m is not None
                and entry.accept_world_half_extent_without_live_scale):
            extent = entry.world_half_extent_m.model_copy()
            source = "contract_declared_world_extent"
            evidence = (f"reviewed contract entry {entry.pattern!r}: world half-extent measured by "
                        f"{entry.measured_by} ({entry.measurement_method}, {entry.measured_at})")
            extent_measurable = True
        else:
            extent = placeholder.model_copy()
            evidence = ("no reviewed geometry-contract entry covers this actor and no world-extent RPC "
                        "exists at the pin, so its size is a placeholder, not a measurement")
            without_evidence.append(name)
            warnings.append(
                f"{name}: extent UNMEASURED. The obstacle carries the placeholder "
                f"{placeholder.as_tuple()} m, which is not evidence; add a reviewed geometry-contract "
                "entry for this actor before flying a scientific episode."
            )
        if abs(yaw) > yaw_tolerance_rad:
            warnings.append(
                f"{name} is rotated {yaw:.3f} rad; it is recorded as an axis-aligned box, which "
                "over- or under-states clearance near its corners"
            )
        records.append(DerivedActorRecord(
            actor_name=name, observed_position_ned_m=position, observed_yaw_rad=yaw,
            observed_scale=scale, half_extent_m=extent, extent_source=source,
            extent_evidence=evidence, contract_pattern=entry.pattern if entry else None,
        ))
    if len(candidates) > max_actors:
        warnings.append(
            f"only the first {max_actors} of {len(candidates)} candidate actors were measured; raise "
            "max_actors or narrow name_regex before using this map for a real run"
        )
    if unplaced:
        warnings.append(
            f"{len(unplaced)} candidate actor(s) could not be placed by simGetObjectPose ({unplaced}); "
            "the map inventory is partial, so the relevant world is not established"
        )
    if not records:
        warnings.append(
            "no measurable actor was found inside the mission volume: this map gives the study no "
            "3D obstacles, so it cannot support the occlusion the protocol requires"
        )
    if without_evidence:
        warnings.append(
            f"{len(without_evidence)} derived obstacle(s) have no extent evidence: {without_evidence}. "
            "The binding is not ok until a reviewed geometry contract covers them."
        )
    if contract is None:
        warnings.append(
            "no geometry contract was supplied, so no derived obstacle can carry a measured extent; "
            f"set {GEOMETRY_CONTRACT_ENV_VAR} or pass geometry_contract"
        )

    asset_position: Vec3 | None = None
    provenance = SceneDerivationProvenance(
        derived_at=datetime.now(UTC).isoformat(timespec="seconds"),
        name_regex=name_regex,
        scene_object_count=len(scene_objects),
        candidates_considered=len(candidates),
        actors=records,
        asset_position_ned_m=asset_position,
        mission_volume=volume,
        placeholder_half_extent_m=placeholder,
        geometry_contract=contract.summary() if contract is not None else None,
        actors_without_extent_evidence=without_evidence,
        unplaced_actors=unplaced,
        extent_measurement_available=extent_measurable,
        warnings=warnings,
    )
    obstacles = [
        ObstacleSpec(
            name=record.actor_name,
            kind=default_kind,  # type: ignore[arg-type]
            center=record.observed_position_ned_m,
            extent=record.half_extent_m,
            unreal_actor_tag=record.actor_name,
            occludes_asset=False,
        )
        for record in records
    ]
    return SceneDerivationResult(obstacles=obstacles, provenance=provenance)


def bind_asset_actor(rpc_source: SimAdapter | Any, actor_name: str) -> Vec3 | None:
    """Measure one named actor and return its NED position, or None when it is not measurable.

    Used by the qualified-map path when the inspection target is itself a map actor: the mission then
    refers to a body that really exists instead of a coordinate the map knows nothing about.
    """
    rpc = _SceneRpc(rpc_source)
    value, error = rpc.try_call("simGetObjectPose", actor_name)
    decoded = None if error else _decode_pose(value)
    return None if decoded is None else decoded[0]


def apply_derivation(manifest: ScenarioManifest, result: SceneDerivationResult,
                     *, asset_position: Vec3 | None = None) -> ScenarioManifest:
    """Return the manifest with scene-derived geometry and a note that says so.

    The manifest content hash changes, and that is the point: a scene-derived scenario is a different
    scenario from the authored one, and every record must be able to show which was flown.
    """
    update: dict[str, Any] = {
        "obstacles": list(result.obstacles),
        "notes": (manifest.notes if result.provenance.note in manifest.notes else
                  (f"{manifest.notes} " if manifest.notes else "") + result.provenance.note),
    }
    if asset_position is not None:
        update["asset_position"] = asset_position
        # Retain the declared viewing offset when the real asset replaces the authored coordinate.
        # Rebinding the same map is idempotent, as required across arms of one scenario.
        if asset_position != manifest.asset_position:
            update["inspection_viewpoint"] = Vec3(
                x=manifest.inspection_viewpoint.x + (asset_position.x - manifest.asset_position.x),
                y=manifest.inspection_viewpoint.y + (asset_position.y - manifest.asset_position.y),
                z=manifest.inspection_viewpoint.z + (asset_position.z - manifest.asset_position.z),
            )
    return manifest.model_copy(update=update)


def assert_scene_ready(report: SceneVerificationReport | SceneBindingReport | SceneBindResult,
                       *, require_complete_measurement: bool = True) -> SceneVerificationReport:
    """Fail closed unless the level was measured, matches the manifest, and holds nothing else.

    Every live episode must pass through this function. A scene that is missing an obstacle, holds one
    at the wrong place or size, contains extra bodies in the flight volume, could not be measured, or
    whose inventory could not be established, produces numbers about an unknown world. Those numbers
    would silently corrupt the arm comparison, so the run stops here instead.

    ``require_complete_measurement`` now defaults to **True**: the production path is strict, and a
    caller must deliberately write ``require_complete_measurement=False`` to accept a scene whose
    extents rest on nothing. That inversion is the point of this revision. The relaxation is limited to
    the ``[measurement]`` reasons; a missing body, a moved body, an extra body or an unestablished
    inventory still stops the run.
    """
    if isinstance(report, SceneBindResult):
        verification: SceneVerificationReport | None = report.verification
        extra_reasons = [reason for reason in report.blocking_reasons
                         if verification is None or reason not in verification.blocking_reasons]
    elif isinstance(report, SceneBindingReport):
        verification = report.verification
        extra_reasons = []
    else:
        verification = report
        extra_reasons = []
    if verification is None:
        raise SceneMismatch(
            "the scene binding carries no verification, so the level is unproven",
            remedy="Call instantiate_scene(..., verify=True) or verify_scene(...) before flying.",
            report=report,
        )
    reasons = [*verification.blocking_reasons, *extra_reasons]
    if not require_complete_measurement:
        reasons = [reason for reason in reasons if not reason.startswith("[measurement]")]
    problems = list(reasons)
    for check in verification.actors:
        if check.status != "ok":
            relaxed = (not require_complete_measurement and check.status == "unverifiable"
                       and check.position_verified)
            if relaxed:
                continue
            problems.append(f"[{check.status}] {check.spec.obstacle_name}: {check.explanation}")
    for finding in verification.extras:
        if finding.status == "extra":
            problems.append(f"[extra] {finding.actor_name}: {finding.note}")
        elif require_complete_measurement:
            problems.append(f"[unverifiable] {finding.actor_name}: {finding.note}")
    if not problems:
        return verification
    raise SceneMismatch(
        "the level does not match the scenario manifest, so this episode would measure an unknown "
        "scene:\n  " + "\n  ".join(problems),
        remedy=(
            "Rebuild or re-instantiate the scene for this manifest and re-verify. For unknown extents, "
            "bind every actor in a reviewed geometry contract (configs/scene-geometry.example.yaml is "
            f"the template; {GEOMETRY_CONTRACT_ENV_VAR} points at yours) -- disclosure alone does not "
            "make the evaluator's assumed clearance true. For scene mode instantiate, re-run "
            "instantiate_scene; for verify_only, rebuild the packaged map with "
            "deploy/scripts/generate_study_scene.py; for a third-party map, use scene mode "
            "qualified_map so the manifest is derived from the map instead of assumed. Never disable "
            "this check to obtain results."
        ),
        report=verification,
    )


class SceneBindResult(StrictModel):
    """One entry point for the runner: what was bound, how, and whether it verified.

    ``ok`` is the only field a caller may treat as permission to fly. The three components are kept
    separately so an episode record can show WHICH part failed, and ``blocking_reasons`` carries the
    tagged lines a human can act on.
    """

    mode: SceneMode
    manifest: ScenarioManifest = Field(
        description="The EFFECTIVE manifest: for qualified_map this is the derived one that was flown.",
    )
    geometry_provenance: GeometryProvenance
    verification: SceneVerificationReport
    binding: SceneBindingReport | None = None
    derivation: SceneDerivationResult | None = None
    geometry_contract: dict[str, Any] | None = None
    ok: bool = False
    measurement_complete: bool = False
    inventory_complete: bool = False
    blocking_reasons: list[str] = Field(default_factory=list)


def bind_scene(
    adapter: SimAdapter | Any,
    manifest: ScenarioManifest,
    protocol: ProtocolConfig | None = None,
    mode: SceneMode = "verify_only",
    *,
    position_tolerance_m: float = 0.25,
    scale_tolerance: float = 0.05,
    asset_name: str = STUDY_BOX_ASSET_NAME,
    asset_base_size_m: float = 1.0,
    name_regex: str = DEFAULT_QUALIFIED_MAP_REGEX,
    asset_actor_name: str | None = None,
    geometry_contract: GeometryContract | None = None,
    geometry_contract_path: str | Path | None = None,
    allow_empty_scene: bool = False,
    derivation_kwargs: dict[str, Any] | None = None,
    max_inventory_probes: int = MAX_INVENTORY_PROBES,
    inventory_wall_budget_s: float = 120.0,
    inventory_exclusions: dict[str, str] | None = None,
) -> SceneBindResult:
    """Apply one scene-mode policy and return the manifest that was actually flown.

    ``qualified_map`` returns a DIFFERENT manifest from the one passed in: its obstacles are measured
    map bodies. The caller must record ``SceneBindResult.manifest`` (and its content hash) with the
    episode, otherwise the record would describe geometry that was never in the level.
    """
    contract = resolve_geometry_contract(adapter, geometry_contract, geometry_contract_path)
    if mode == "instantiate":
        binding = instantiate_scene(adapter, manifest, protocol, asset_name=asset_name,
                                    asset_base_size_m=asset_base_size_m,
                                    position_tolerance_m=position_tolerance_m,
                                    scale_tolerance=scale_tolerance, geometry_contract=contract,
                                    allow_empty_scene=allow_empty_scene,
                                    max_inventory_probes=max_inventory_probes,
                                    inventory_wall_budget_s=inventory_wall_budget_s,
                                    inventory_exclusions=inventory_exclusions)
        verification = binding.verification
        if verification is None:  # pragma: no cover - instantiate_scene always verifies by default
            raise SceneMismatch("instantiate_scene returned no verification",
                                remedy="Do not disable verification on a live run.")
        return _bind_result(mode, manifest, "manifest", verification, contract, binding=binding)
    if mode == "verify_only":
        verification = verify_scene(adapter, manifest, protocol, position_tolerance_m, scale_tolerance,
                                    mode=mode, geometry_contract=contract,
                                    allow_empty_scene=allow_empty_scene,
                                    max_inventory_probes=max_inventory_probes,
                                    inventory_wall_budget_s=inventory_wall_budget_s,
                                    inventory_exclusions=inventory_exclusions)
        return _bind_result(mode, manifest, "manifest", verification, contract)
    if mode == "qualified_map":
        derivation = derive_manifest_geometry(adapter, manifest, protocol, name_regex,
                                              geometry_contract=contract,
                                              **(derivation_kwargs or {}))
        extra: list[str] = []
        asset_position = None
        asset = next((body for body in derivation.obstacles if body.name == asset_actor_name), None)
        if not asset_actor_name:
            extra.append("[scene] qualified_map requires an explicit inspection asset actor name; "
                         "the authored target coordinate does not establish a real map asset")
        elif asset is None:
            extra.append(f"[scene] inspection asset {asset_actor_name!r} is not among the measured "
                         "actors inside the mission volume; its identity and geometry are unverified")
        else:
            asset_position = asset.center.model_copy()
            derivation.obstacles = [
                body.model_copy(update={"kind": "inspection_asset"})
                if body.name == asset_actor_name else body
                for body in derivation.obstacles
            ]
        derivation.provenance.asset_actor_name = asset_actor_name
        derivation.provenance.asset_position_ned_m = asset_position
        bound = apply_derivation(manifest, derivation, asset_position=asset_position)
        verification = verify_scene(adapter, bound, protocol, position_tolerance_m, scale_tolerance,
                                    mode=mode, geometry_provenance="scene_derived",
                                    geometry_contract=contract, allow_empty_scene=allow_empty_scene,
                                    max_inventory_probes=max_inventory_probes,
                                    inventory_wall_budget_s=inventory_wall_budget_s,
                                    inventory_exclusions=inventory_exclusions)
        if derivation.provenance.actors_without_extent_evidence:
            extra.append(
                "[measurement] derived obstacle(s) with no extent evidence: "
                f"{derivation.provenance.actors_without_extent_evidence}. Their size is a placeholder, "
                "so the evaluator's clearance geometry would be invented."
            )
        if derivation.provenance.unplaced_actors:
            extra.append(
                "[inventory] map actor(s) that simGetObjectPose could not place: "
                f"{derivation.provenance.unplaced_actors}"
            )
        return _bind_result(mode, bound, "scene_derived", verification, contract,
                            derivation=derivation, extra_reasons=extra)
    raise SceneUnavailable(f"unknown scene mode {mode!r}",
                           remedy="Use one of: instantiate, verify_only, qualified_map.")


def _bind_result(mode: SceneMode, manifest: ScenarioManifest, provenance: GeometryProvenance,
                 verification: SceneVerificationReport, contract: GeometryContract | None,
                 *, binding: SceneBindingReport | None = None,
                 derivation: SceneDerivationResult | None = None,
                 extra_reasons: Sequence[str] = ()) -> SceneBindResult:
    """Assemble the bind result so every path reports ``ok`` by the same rule."""
    reasons = [*verification.blocking_reasons, *extra_reasons]
    return SceneBindResult(
        mode=mode,
        manifest=manifest,
        geometry_provenance=provenance,
        verification=verification,
        binding=binding,
        derivation=derivation,
        geometry_contract=contract.summary() if contract is not None else None,
        ok=not reasons,
        measurement_complete=verification.measurement_complete,
        inventory_complete=verification.inventory_complete,
        blocking_reasons=reasons,
    )


def scene_summary(report: SceneVerificationReport | SceneBindingReport | SceneBindResult
                  ) -> dict[str, Any]:
    """Compact dict for the episode record provenance block."""
    if isinstance(report, SceneBindResult):
        return {
            "mode": report.mode,
            "ok": report.ok,
            "geometry_provenance": report.geometry_provenance,
            "measurement_complete": report.measurement_complete,
            "inventory_complete": report.inventory_complete,
            "blocking_reasons": report.blocking_reasons,
            "flown_manifest_hash": report.manifest.content_hash(),
            "geometry_contract": report.geometry_contract,
            "verification": scene_summary(report.verification),
        }
    if isinstance(report, SceneBindingReport):
        verification = report.verification
        return {
            "mode": report.mode,
            "ok": report.ok,
            "dry_run": report.dry_run,
            "spawned": [record.actual_name for record in report.spawned],
            "destroyed": report.destroyed,
            "asset_name": report.asset_name,
            "verification": scene_summary(verification) if verification else None,
            "detail": report.detail,
        }
    return {
        "mode": report.mode,
        "ok": report.ok,
        "evidence_class": report.evidence_class,
        "geometry_provenance": report.geometry_provenance,
        "positions_verified": report.positions_verified,
        "measurement_complete": report.measurement_complete,
        "inventory_complete": report.inventory_complete,
        "blocking_reasons": report.blocking_reasons,
        "geometry_contract": report.geometry_contract,
        "scene_object_count": report.scene_object_count,
        "position_tolerance_m": report.position_tolerance_m,
        "max_position_error_m": report.max_position_error_m,
        "actor_status": {check.spec.obstacle_name: check.status for check in report.actors},
        "extent_source": {check.spec.obstacle_name: check.measurement.extent_source
                          for check in report.actors},
        "missing_actors": report.missing_actors,
        "mismatched_actors": report.mismatched_actors,
        "unverifiable_actors": report.unverifiable_actors,
        "unmeasured_extent_actors": report.unmeasured_extent_actors,
        "extra_actors": report.extra_actors,
        "inventory": {
            "listed_actors": report.inventory.listed_actors,
            "probed_actors": report.inventory.probed_actors,
            "excluded": len(report.inventory.excluded),
            "unplaced_actors": report.inventory.unplaced_actors,
            "complete": report.inventory.complete,
            "incomplete_reasons": report.inventory.incomplete_reasons,
        },
        "route_disagreement": report.route_disagreement,
        "caveats": report.caveats,
        "detail": report.detail,
    }
