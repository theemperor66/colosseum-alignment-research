"""Shared record schemas: the single contract between simulator, controller, monitors, evaluator, audit.

Three information channels are kept strictly apart (research-plan.md, "The question worth answering"):

1. ``TruthSample`` / ``TruthEvent`` / ``PrivilegedLedger`` -- what happened in the simulated world.
   Written by :mod:`colosseum_assurance.runtime`, read only by :mod:`colosseum_assurance.evaluation`.
2. ``ObservationPacket`` / ``ControlCommand`` / ``MonitorReport`` / ``EpisodeRecord`` -- what the
   controller and monitor could observe and did at the time. Never contains privileged world state.
3. ``AuditRecordSet`` (see :mod:`colosseum_assurance.audit`) -- what a later evaluator can establish
   from retained records after prespecified offline field ablation.

All models forbid extra fields so a leakage attempt fails loudly instead of silently succeeding.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator

SCHEMA_VERSION = "1.1.0"

# Timestamps are episode-relative simulator seconds (t = 0 at the first step after a successful reset).
# See docs/timing-semantics.md for the single executable definition shared by monitors and the evaluator.
TIME_EPSILON_S = 1e-6


def _require_finite(value: float | None, field: str) -> float | None:
    """Reject NaN and infinity in any timestamp or age. Missing values stay None."""
    if value is None:
        return None
    if not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number, got {value!r}")
    return value


class StrictModel(BaseModel):
    """Base model: unknown fields are an error, values are validated on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=False)


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------
class Vec3(StrictModel):
    """A point or vector in Colosseum NED world coordinates (metres, z negative up)."""

    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def distance_to(self, other: Vec3) -> float:
        return math.dist(self.as_tuple(), other.as_tuple())

    def horizontal_distance_to(self, other: Vec3) -> float:
        return math.dist((self.x, self.y), (other.x, other.y))

    @classmethod
    def from_iterable(cls, values: Any) -> Vec3:
        x, y, z = values
        return cls(x=float(x), y=float(y), z=float(z))


class Box(StrictModel):
    """Axis-aligned box in NED coordinates. ``z_min`` is the more negative (higher) bound."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @field_validator("x_max")
    @classmethod
    def _x_ordered(cls, v: float, info: Any) -> float:
        if "x_min" in info.data and v <= info.data["x_min"]:
            raise ValueError("x_max must exceed x_min")
        return v

    @field_validator("y_max")
    @classmethod
    def _y_ordered(cls, v: float, info: Any) -> float:
        if "y_min" in info.data and v <= info.data["y_min"]:
            raise ValueError("y_max must exceed y_min")
        return v

    @field_validator("z_max")
    @classmethod
    def _z_ordered(cls, v: float, info: Any) -> float:
        if "z_min" in info.data and v <= info.data["z_min"]:
            raise ValueError("z_max must exceed z_min (NED: z_min is the upper altitude bound)")
        return v

    def contains(self, p: Vec3, tolerance: float = 0.0) -> bool:
        return (
            self.x_min - tolerance <= p.x <= self.x_max + tolerance
            and self.y_min - tolerance <= p.y <= self.y_max + tolerance
            and self.z_min - tolerance <= p.z <= self.z_max + tolerance
        )

    def signed_exceedance(self, p: Vec3) -> float:
        """Return how far outside the box ``p`` lies (metres); <= 0 means inside."""
        dx = max(self.x_min - p.x, p.x - self.x_max)
        dy = max(self.y_min - p.y, p.y - self.y_max)
        dz = max(self.z_min - p.z, p.z - self.z_max)
        return max(dx, dy, dz)


# --------------------------------------------------------------------------------------
# Simulator identity and observations
# --------------------------------------------------------------------------------------
ANCHORED_LIVE_PROVENANCES = frozenset({"colosseum_build_verified", "third_party_colosseum_build"})
"""Provenance classes that may back experimental claims.

Both are *anchored*: a named artifact with a SHA-256 hash and a stated origin. An endpoint that merely
speaks the msgpack-RPC protocol is not anchored, because Microsoft AirSim or any unrelated server would
also pass that test (independent review, "false simulator provenance").
"""


class SimulatorArtifactAttestation(StrictModel):
    """Operator declaration binding a running simulator to a hash-verified package.

    This is the only thing that can raise an endpoint above ``airsim_compatible_unverified``. The client
    cannot interrogate a remote binary, so the attestation records what was verified, by whom, and how,
    and it is stored with the evidence. Two rules keep it honest:

    * ``colosseum_build_verified`` requires a 40-character upstream Colosseum commit. It is for a package
      built from pinned upstream source.
    * ``third_party_colosseum_build`` is for a qualified third-party Colosseum-based package. Its
      ``upstream_colosseum_commit`` must stay ``None`` when it is unknown, and at least one caveat must
      say so. Inventing an upstream identity for a third-party build is fabrication.
    """

    attestation_version: str = "1.0.0"
    provenance_class: Literal["colosseum_build_verified", "third_party_colosseum_build"]
    artifact_name: str
    artifact_sha256: str = Field(min_length=64, max_length=71)
    artifact_size_bytes: int | None = Field(default=None, ge=0)
    source_kind: Literal[
        "upstream_release", "upstream_source_build", "third_party_publication", "local_build"
    ]
    source_url: str | None = None
    source_repo: str | None = None
    source_revision: str | None = None
    upstream_colosseum_commit: str | None = None
    engine_version_declared: str | None = None
    scene_package_path: str | None = None
    expected_scene_signature: str | None = Field(
        default=None,
        description="Optional binding: the scene-object signature this package must produce. Checked live.",
    )
    verification_manifest_path: str | None = None
    verification_manifest_sha256: str | None = None
    verified_by: str
    verified_utc: str
    verification_note: str = ""
    caveats: list[str] = Field(default_factory=list)

    @field_validator("artifact_sha256")
    @classmethod
    def _hash_shape(cls, v: str) -> str:
        raw = v.split(":", 1)[-1].strip().lower()
        if len(raw) != 64 or any(c not in "0123456789abcdef" for c in raw):
            raise ValueError("artifact_sha256 must be a 64-character hexadecimal SHA-256 digest")
        return f"sha256:{raw}"

    @model_validator(mode="after")
    def _class_requirements(self) -> SimulatorArtifactAttestation:
        commit = (self.upstream_colosseum_commit or "").strip()
        if self.provenance_class == "colosseum_build_verified":
            if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit.lower()):
                raise ValueError(
                    "colosseum_build_verified requires upstream_colosseum_commit to be the 40-character "
                    "commit the package was built from"
                )
            if self.source_kind not in {"upstream_release", "upstream_source_build", "local_build"}:
                raise ValueError(
                    "colosseum_build_verified requires a source_kind that names an upstream build"
                )
        else:
            if commit:
                raise ValueError(
                    "third_party_colosseum_build must leave upstream_colosseum_commit unset: the exact "
                    "upstream Colosseum source commit of a third-party package is unknown, and must stay "
                    "recorded as unknown"
                )
            if not self.caveats:
                raise ValueError(
                    "third_party_colosseum_build requires at least one explicit caveat, for example that "
                    "the upstream Colosseum source commit is unestablished"
                )
        return self

    @property
    def short_hash(self) -> str:
        return self.artifact_sha256.split(":", 1)[-1][:12]


class SimulatorIdentity(StrictModel):
    """Recorded provenance of the simulator that produced an episode.

    ``provenance`` is deliberately conservative:

    | value | meaning |
    | --- | --- |
    | ``fixture_fake`` | our software test fixture answered the private probe |
    | ``unverified`` | the handshake did not complete, so nothing is established |
    | ``airsim_compatible_unverified`` | an endpoint speaks the msgpack-RPC surface, nothing anchors it |
    | ``third_party_colosseum_build`` | anchored to a hash-verified third-party Colosseum-based package |
    | ``colosseum_build_verified`` | anchored to a package built from a pinned upstream Colosseum commit |

    Only the last two may back experimental claims.
    """

    provenance: Literal[
        "colosseum_build_verified",
        "third_party_colosseum_build",
        "airsim_compatible_unverified",
        "fixture_fake",
        "unverified",
    ] = "unverified"
    endpoint_label: str = "unset"
    server_version: int | None = None
    min_required_client_version: int | None = None
    client_version: int | None = None
    min_required_server_version: int | None = None
    simulator_name: str | None = None
    engine_version: str | None = None
    scene_name: str | None = None
    scene_object_count: int | None = None
    scene_signature: str | None = None
    settings_digest: str | None = None
    api_settings: dict[str, Any] = Field(default_factory=dict)
    artifact: SimulatorArtifactAttestation | None = None
    scene_binding_verified: bool | None = Field(
        default=None,
        description="True when the live scene signature matched the attestation's expected signature.",
    )
    observed_at_wall_clock: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _anchored_requires_attestation(self) -> SimulatorIdentity:
        if self.provenance in ANCHORED_LIVE_PROVENANCES:
            if self.artifact is None:
                raise ValueError(
                    f"provenance={self.provenance!r} requires an artifact attestation; an RPC handshake "
                    "alone does not establish which simulator answered"
                )
            if self.artifact.provenance_class != self.provenance:
                raise ValueError(
                    f"identity provenance {self.provenance!r} disagrees with the attestation class "
                    f"{self.artifact.provenance_class!r}"
                )
        return self

    @property
    def is_live(self) -> bool:
        """True only for an anchored, non-fixture simulator."""
        return self.provenance in ANCHORED_LIVE_PROVENANCES

    @property
    def is_qualified_third_party(self) -> bool:
        return self.provenance == "third_party_colosseum_build"

    def qualification_note(self) -> str:
        """One sentence a report can print beside any result produced by this simulator."""
        if self.provenance == "colosseum_build_verified" and self.artifact is not None:
            return (
                f"Colosseum build verified against upstream commit "
                f"{self.artifact.upstream_colosseum_commit} (artifact {self.artifact.short_hash})."
            )
        if self.provenance == "third_party_colosseum_build" and self.artifact is not None:
            return (
                f"Qualified third-party Colosseum-based package {self.artifact.artifact_name} "
                f"(sha256 {self.artifact.short_hash}, source {self.artifact.source_repo or 'unstated'}); "
                "its exact upstream Colosseum source commit is unestablished."
            )
        if self.provenance == "airsim_compatible_unverified":
            return (
                "Endpoint speaks the AirSim/Colosseum RPC surface but no artifact attestation anchors it; "
                "this cannot back experimental claims."
            )
        if self.provenance == "fixture_fake":
            return "Local software test fixture. Never experimental evidence."
        return "Simulator identity was not established."


class FrameRef(StrictModel):
    """Reference to a captured camera frame stored on disk (never inlined in a record).

    ``sim_time_s`` is the ACQUISITION time reported by the simulator, never the time the client decoded
    or processed the frame. When the server supplies no usable timestamp the field is ``None`` and
    ``acquisition_time_known`` is False: the age of that evidence is unknown, and unknown age must never
    be presented as fresh. Substituting a current clock read would make stale evidence look new to the
    guard, which is the exact failure this study measures.
    """

    frame_id: str | None = None
    content_sha256: str | None = None
    kind: Literal["rgb", "depth", "segmentation"]
    camera_name: str
    sim_time_s: float | None
    acquisition_time_known: bool = True
    width: int
    height: int
    path: str | None = None
    pixels_as: Literal["png", "npy", "none"] = "none"
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None
    nonzero_fraction: float | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_v4_identity(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        for key in ("frame_id", "content_sha256"):
            if getattr(self, key) is None:
                data.pop(key, None)
        return data

    @model_validator(mode="after")
    def _acquisition_time_consistency(self) -> FrameRef:
        if self.sim_time_s is None:
            if self.acquisition_time_known:
                raise ValueError(
                    "a frame without an acquisition timestamp cannot claim a known acquisition time"
                )
            return self
        _require_finite(self.sim_time_s, "FrameRef.sim_time_s")
        if self.sim_time_s < 0.0:
            raise ValueError("FrameRef.sim_time_s must be episode-relative and non-negative")
        return self

    @property
    def is_nonempty(self) -> bool:
        """True when the frame has real pixel content (used by the live-readiness gate)."""
        if self.width <= 0 or self.height <= 0:
            return False
        if self.nonzero_fraction is None:
            return False
        return self.nonzero_fraction > 0.0 and (self.max_value or 0.0) > 0.0

    @property
    def is_usable_evidence(self) -> bool:
        """Nonempty pixels AND a known acquisition time.

        Perception may consume only frames that satisfy both. A nonempty frame of unknown age is not
        usable evidence, because nothing can be said about how stale it is.
        """
        return self.is_nonempty and self.acquisition_time_known and self.sim_time_s is not None


class VehicleState(StrictModel):
    """Onboard state estimate sample, stamped with its ACQUISITION time.

    ``sim_time_s`` is when the simulator produced the measurement, not when the vehicle received it.
    The receipt time lives in :class:`ObservationPacket`.
    """

    sim_time_s: float
    position: Vec3
    velocity: Vec3
    yaw_rad: float
    orientation_wxyz: tuple[float, float, float, float] | None = None
    """Full ONBOARD estimated attitude. None only in historical/fixture records, never ground truth."""
    landed: bool = False
    api_control_enabled: bool = True
    source: str = "colosseum_multirotor_state"

    @field_validator("orientation_wxyz")
    @classmethod
    def _onboard_attitude(cls, value: tuple[float, float, float, float] | None):
        if value is not None:
            if not all(math.isfinite(v) for v in value) or abs(math.sqrt(sum(v*v for v in value))-1) > .001:
                raise ValueError("onboard attitude requires a finite unit quaternion in w,x,y,z order")
        return value

    @field_validator("sim_time_s")
    @classmethod
    def _finite_time(cls, v: float) -> float:
        _require_finite(v, "VehicleState.sim_time_s")
        if v < 0.0:
            raise ValueError("VehicleState.sim_time_s must be episode-relative and non-negative")
        return v


class DepthSummary(StrictModel):
    """Features from camera pixels, fixed camera calibration and onboard attitude only.

    Produced by :mod:`colosseum_assurance.control.perception`. No scenario obstacle list is used.
    """

    sim_time_s: float
    camera_name: str
    valid: bool
    min_range_m: float | None = None
    free_path_m: float | None = None
    sector_min_range_m: list[float] = Field(default_factory=list)
    obstacle_bearing_rad: float | None = None
    target_visible: bool = False
    target_bearing_rad: float | None = None
    target_range_m: float | None = None
    target_horizontal_range_m: float | None = None
    """Vehicle-center horizontal target distance, for XY reconstruction after camera calibration."""
    geometry_source: str = "legacy_camera_ray_level_assumption"
    projection_attitude_sim_time_s: float | None = None
    camera_mount_evidence_sha256: str | None = None
    coverage_fraction: float = 0.0
    frame: FrameRef | None = None
    degraded_reason: str | None = None


class SupervisionView(StrictModel):
    """What the vehicle knows about the supervisory link at a given step.

    Two distinct times are recorded because they can differ:

    * ``last_heartbeat_sim_time_s`` -- when the supervisor PRODUCED the newest heartbeat the vehicle holds.
    * ``last_heartbeat_received_at_s`` -- when the vehicle RECEIVED it. This is the operative time for the
      loss-of-supervision obligation, because the vehicle cannot act on a message it has not received.

    ``heartbeat_age_s`` is defined as ``sim_time_s - last_heartbeat_received_at_s`` and is validated
    against those fields, so a monitor cannot be handed a freshness value computed from another clock.
    """

    sim_time_s: float
    last_heartbeat_sim_time_s: float | None
    last_heartbeat_received_at_s: float | None = None
    heartbeat_age_s: float | None
    link_state: Literal["nominal", "degraded", "lost", "unknown"] = "unknown"

    @model_validator(mode="after")
    def _consistent_times(self) -> SupervisionView:
        _require_finite(self.sim_time_s, "SupervisionView.sim_time_s")
        _require_finite(self.last_heartbeat_sim_time_s, "SupervisionView.last_heartbeat_sim_time_s")
        _require_finite(self.last_heartbeat_received_at_s, "SupervisionView.last_heartbeat_received_at_s")
        _require_finite(self.heartbeat_age_s, "SupervisionView.heartbeat_age_s")
        produced = self.last_heartbeat_sim_time_s
        received = self.last_heartbeat_received_at_s
        if received is None and produced is not None:
            # Link-level heartbeats are delivered at production time when the link is up.
            object.__setattr__(self, "last_heartbeat_received_at_s", produced)
            received = produced
        if produced is not None and received is not None and received + TIME_EPSILON_S < produced:
            raise ValueError("a heartbeat cannot be received before it was produced")
        if received is not None:
            if received > self.sim_time_s + TIME_EPSILON_S:
                raise ValueError("a heartbeat cannot be received in the future")
            expected_age = self.sim_time_s - received
            if self.heartbeat_age_s is None:
                object.__setattr__(self, "heartbeat_age_s", expected_age)
            elif abs(self.heartbeat_age_s - expected_age) > 1e-3:
                raise ValueError(
                    f"heartbeat_age_s={self.heartbeat_age_s} disagrees with "
                    f"sim_time_s - last_heartbeat_received_at_s={expected_age}"
                )
        if self.heartbeat_age_s is not None and self.heartbeat_age_s < 0.0:
            raise ValueError("heartbeat_age_s must not be negative")
        return self


class AuthorizationStatus(str, Enum):
    ABSENT = "absent"
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    EXPIRED = "expired"


class AuthorizationView(StrictModel):
    """Authorization record as visible to the vehicle (may be stale or expired)."""

    token_id: str | None = None
    status: AuthorizationStatus = AuthorizationStatus.ABSENT
    requested_at_s: float | None = None
    granted_at_s: float | None = None
    expires_at_s: float | None = None
    scope: str | None = None
    received_at_s: float | None = None
    issuer: str | None = None


class SensorHealth(StrictModel):
    """Per-step sensing health as observable onboard."""

    state_sample_available: bool = True
    depth_available: bool = True
    rgb_available: bool = True
    state_age_s: float | None = None
    depth_age_s: float | None = None
    dropouts_in_window: int = 0
    notes: str = ""


class PerceptionPredictionEvidence(StrictModel):
    """Observation-only binding, never a label or probability of safety."""

    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    evidence_version: Literal["observed_asset_prediction_v1"] = "observed_asset_prediction_v1"
    frame_id: str
    rgb_sha256: str
    camera_name: str
    image_time_s: float = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    model_hash: str
    feature_version: str
    label_spec_hash: str
    features_sha256: str
    features: dict[str, float]
    depth_available: bool
    depth_valid_fraction: float = Field(ge=0, le=1)
    rgb_std: float = Field(ge=0, le=1)
    probability: float | None = Field(default=None, ge=0, le=1)


class ObservationPacket(StrictModel):
    """The complete, delayed information available to controller and monitors at one step."""

    step_index: int
    receive_sim_time_s: float
    state: VehicleState | None
    state_age_s: float | None
    depth: DepthSummary | None
    rgb: FrameRef | None
    supervision: SupervisionView
    authorization: AuthorizationView
    sensor_health: SensorHealth
    sensor_samples: dict[str, Any] = Field(default_factory=dict)
    perception_evidence: PerceptionPredictionEvidence | None = None
    asset_presence_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    perception_model_hash: str | None = None
    perception_prediction_time_s: float | None = None
    mission_phase_hint: str | None = None
    declared_observation_delay_s: float | None = Field(
        default=None,
        description="Nominal delay bound declared in the scenario manifest (an assumption, not truth).",
    )

    @model_serializer(mode="wrap")
    def _omit_absent_v4_evidence(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        if self.perception_evidence is None:
            data.pop("perception_evidence", None)
        return data

    @model_validator(mode="after")
    def _timestamps_are_consistent(self) -> ObservationPacket:
        """Evidence age must follow from the timestamps, and nothing may arrive from the future.

        Without this, a monitor could be handed a freshness value that does not match the observation it
        actually holds, and would look better informed than its inputs permit.
        """
        _require_finite(self.receive_sim_time_s, "ObservationPacket.receive_sim_time_s")
        _require_finite(self.perception_prediction_time_s, "ObservationPacket.perception_prediction_time_s")
        if self.asset_presence_probability is not None:
            _require_finite(self.asset_presence_probability, "ObservationPacket.asset_presence_probability")
            if self.perception_prediction_time_s is None or not self.perception_model_hash:
                raise ValueError("a perception probability requires a model hash and capture timestamp")
        if self.perception_prediction_time_s is not None and (
            self.perception_prediction_time_s < 0
            or self.perception_prediction_time_s > self.receive_sim_time_s + TIME_EPSILON_S
        ):
            raise ValueError("perception prediction has an invalid acquisition time")
        if self.receive_sim_time_s < 0.0:
            raise ValueError("receive_sim_time_s must be episode-relative and non-negative")
        if self.state is not None:
            if self.state.sim_time_s > self.receive_sim_time_s + TIME_EPSILON_S:
                raise ValueError(
                    "observation carries a state acquired after it was received "
                    f"({self.state.sim_time_s} > {self.receive_sim_time_s})"
                )
            expected = self.receive_sim_time_s - self.state.sim_time_s
            if self.state_age_s is None:
                object.__setattr__(self, "state_age_s", expected)
            elif abs(self.state_age_s - expected) > 1e-3:
                raise ValueError(
                    f"state_age_s={self.state_age_s} disagrees with receive - acquisition = {expected}"
                )
        elif self.state_age_s is not None:
            raise ValueError("state_age_s must be None when no state sample was received")
        if self.depth is not None and self.depth.sim_time_s > self.receive_sim_time_s + TIME_EPSILON_S:
            raise ValueError("observation carries a depth summary from the future")
        health_age = self.sensor_health.state_age_s
        own_age = self.state_age_s
        if health_age is not None and own_age is not None and abs(health_age - own_age) > 1e-3:
            raise ValueError("sensor_health.state_age_s disagrees with the packet's state_age_s")
        if self.sensor_health.state_sample_available != (self.state is not None):
            raise ValueError("sensor_health.state_sample_available disagrees with the attached state")
        if self.sensor_health.depth_available != (self.depth is not None):
            raise ValueError("sensor_health.depth_available disagrees with the attached depth summary")
        return self


# --------------------------------------------------------------------------------------
# Actions, verdicts, monitor reports
# --------------------------------------------------------------------------------------
CommandKind = Literal[
    "noop",
    "arm",
    "takeoff",
    "move_to",
    "hold",
    "inspect_capture",
    "request_authorization",
    "land",
    "return_to_launch",
    "abort",
]


class ControlCommand(StrictModel):
    """A bounded command emitted by the controller or a guard intervention."""

    step_index: int
    issued_sim_time_s: float
    kind: CommandKind
    target: Vec3 | None = None
    speed_mps: float | None = None
    duration_s: float | None = None
    yaw_rad: float | None = None
    reason: str = ""
    issued_by: Literal["controller", "guard", "runner"] = "controller"
    controller_phase: str | None = None


class Verdict(str, Enum):
    """Verdict lattice. ``UNKNOWN`` is a first-class outcome, never silently a pass."""

    PASS = "pass"
    VIOLATION = "violation"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


InterventionKind = Literal["none", "hold", "suspend_inspection", "return_to_launch", "abort"]


class MonitorReport(StrictModel):
    """One monitor evaluation at one step, with per-obligation and per-assumption detail."""

    step_index: int
    sim_time_s: float
    monitor_id: str
    verdict: Verdict
    obligation_verdicts: dict[str, Verdict] = Field(default_factory=dict)
    assumption_verdicts: dict[str, Verdict] = Field(default_factory=dict)
    intervention: InterventionKind = "none"
    rationale: str = ""
    evidence_age_s: float | None = None
    context_confidence_evidence: dict[str, Any] | None = None
    affirmative: bool = False

    @model_serializer(mode="wrap")
    def _omit_absent_v4_decision(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        if self.context_confidence_evidence is None:
            data.pop("context_confidence_evidence", None)
        return data

    @field_validator("affirmative")
    @classmethod
    def _affirmative_consistent(cls, v: bool, info: Any) -> bool:
        verdict = info.data.get("verdict")
        if v and verdict is not Verdict.PASS:
            raise ValueError("affirmative=True requires verdict == PASS")
        return v


# --------------------------------------------------------------------------------------
# Episode records (exposed channel)
# --------------------------------------------------------------------------------------
class StepRecord(StrictModel):
    """One closed-loop control step as retained in the exposed evidence record."""

    step_index: int
    sim_time_s: float
    wall_clock_s: float
    observation: ObservationPacket
    command: ControlCommand
    executed_command: ControlCommand | None = None
    monitor_report: MonitorReport | None = None
    controller_state: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


TerminationReason = Literal[
    "mission_complete",
    "setup_failed",
    "horizon_reached",
    "controller_abort",
    "guard_abort",
    "guard_return_complete",
    "landed_early",
    "simulator_error",
    "rpc_timeout",
    "reset_failed",
    "runner_exception",
    "operator_interrupt",
]


NON_TERMINAL_REASONS = frozenset({
    "setup_failed",
    "simulator_error",
    "rpc_timeout",
    "reset_failed",
    "runner_exception",
    "operator_interrupt",
})


class TerminationRecord(StrictModel):
    """How an episode ended. Error endings are incomplete by construction, never by remembering to say so."""

    reason: TerminationReason
    detail: str = ""
    step_index: int
    sim_time_s: float
    completed_mission: bool = False
    reached_terminal_state: bool = Field(
        default=True,
        description=(
            "False for crashes, RPC failures, failed resets, and interrupts. The validator forces this, so "
            "error-handling code that forgets the flag cannot produce a superficially complete record."
        ),
    )

    @model_validator(mode="after")
    def _error_endings_are_not_terminal(self) -> TerminationRecord:
        if self.reason in NON_TERMINAL_REASONS and self.reached_terminal_state:
            object.__setattr__(self, "reached_terminal_state", False)
        if self.completed_mission and self.reason != "mission_complete":
            raise ValueError(
                f"completed_mission=True is only valid with reason='mission_complete', got {self.reason!r}"
            )
        if self.completed_mission and not self.reached_terminal_state:
            raise ValueError("an episode cannot both complete the mission and fail to reach a terminal state")
        _require_finite(self.sim_time_s, "TerminationRecord.sim_time_s")
        return self


class EpisodeRecord(StrictModel):
    """Exposed (non-privileged) episode evidence: the audit and monitor-facing record."""

    schema_version: str = SCHEMA_VERSION
    episode_id: str
    scenario_id: str
    arm_id: str
    run_class: Literal["fixture", "pilot", "heldout", "smoke"]
    protocol_hash: str
    policy_version: str
    code_version: dict[str, str] = Field(default_factory=dict)
    simulator_identity: SimulatorIdentity
    started_wall_clock: str
    dt_s: float
    steps: list[StepRecord] = Field(default_factory=list)
    termination: TerminationRecord
    monitor_id: str | None = None
    interventions: list[dict[str, Any]] = Field(default_factory=list)
    timing_report: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Measured clock facts from the adapter: which clock the simulator used, whether the pause "
            "was verified, how often an advance had to be polled, and the last measured advance. A "
            "degraded or fallback stepping mode must be visible in the evidence, not inferred."
        ),
    )
    notes: str = ""


# --------------------------------------------------------------------------------------
# Privileged truth channel (evaluator only)
# --------------------------------------------------------------------------------------
class TruthSample(StrictModel):
    """Undelayed simulator ground truth at one sample time. Never exposed to controller/monitor."""

    sim_time_s: float
    position: Vec3
    velocity: Vec3
    yaw_rad: float
    collision_active: bool = False
    collision_count: int = 0
    collision_object: str | None = None
    collision_penetration_m: float | None = None
    landed: bool = False
    min_obstacle_clearance_m: float | None = None
    source: Literal["simulator_ground_truth", "fixture_fake_ground_truth"] = "simulator_ground_truth"


TruthEventKind = Literal[
    "fault_started",
    "fault_ended",
    "segmentation_identity",
    "episode_start",
    "reset_ok",
    "reset_failed",
    "api_control_acquired",
    "api_control_released",
    "takeoff_complete",
    "authorization_requested",
    "authorization_granted",
    "authorization_denied",
    "authorization_expired",
    "supervision_heartbeat",
    "supervision_lost",
    "supervision_restored",
    "inspection_capture_performed",
    "command_executed",
    "guard_intervention",
    "controller_phase_change",
    "geofence_exceeded",
    "collision",
    "mission_objective_reached",
    "episode_end",
    "simulator_error",
]


class TruthEvent(StrictModel):
    """A privileged, timestamped world/procedural event used by the evaluator and audit reference."""

    sim_time_s: float
    kind: TruthEventKind
    detail: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


def sampling_quality(samples: list[TruthSample], expected_sample_count: int
                     ) -> tuple[float, float | None, float | None]:
    """Return (coverage_fraction, max_sample_gap_s, achieved_mean_interval_s) from real timestamps.

    Coverage is measured, never assumed. An empty ledger has zero coverage even if the few samples it
    does hold look safe.
    """
    if not samples:
        return 0.0, None, None
    times = sorted(s.sim_time_s for s in samples)
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    max_gap = max(gaps) if gaps else None
    mean_interval = (sum(gaps) / len(gaps)) if gaps else None
    coverage = 0.0 if expected_sample_count <= 0 else min(1.0, len(samples) / expected_sample_count)
    return coverage, max_gap, mean_interval


class PrivilegedLedger(StrictModel):
    """Evaluator-only ground-truth ledger for one episode.

    Coverage metadata is required and validated. A ledger with no samples cannot claim complete truth
    coverage, and ``max_sample_gap_s`` exposes long blind intervals that a single coverage number hides.
    """

    schema_version: str = SCHEMA_VERSION
    episode_id: str
    scenario_id: str
    arm_id: str
    run_class: Literal["fixture", "pilot", "heldout", "smoke"]
    protocol_hash: str
    simulator_identity: SimulatorIdentity
    sample_interval_s: float = Field(gt=0.0, description="Requested interval; the achieved one is measured.")
    samples: list[TruthSample] = Field(default_factory=list)
    events: list[TruthEvent] = Field(default_factory=list)
    termination: TerminationRecord
    truth_coverage_fraction: float = Field(
        ge=0.0, le=1.0,
        description="Measured fraction of expected truth samples actually captured (no default).",
    )
    expected_sample_count: int = Field(
        ge=0, description="Expected samples over the realised episode duration at sample_interval_s."
    )
    max_sample_gap_s: float | None = Field(
        default=None, description="Largest observed gap between consecutive truth samples."
    )
    achieved_sample_interval_s: float | None = Field(
        default=None, description="Mean observed interval, measured from timestamps, not requested."
    )
    timing_report: dict[str, Any] = Field(
        default_factory=dict,
        description="Measured clock facts from the adapter for the episode that produced this ledger.",
    )
    notes: str = ""

    @model_validator(mode="after")
    def _coverage_is_consistent_with_samples(self) -> PrivilegedLedger:
        if not self.samples:
            if self.truth_coverage_fraction != 0.0:
                raise ValueError("a ledger with no truth samples must report zero coverage")
            return self
        if self.truth_coverage_fraction <= 0.0:
            raise ValueError("a ledger with truth samples must report positive coverage")
        measured, max_gap, mean_interval = sampling_quality(self.samples, self.expected_sample_count)
        if self.expected_sample_count > 0 and abs(measured - self.truth_coverage_fraction) > 0.05:
            raise ValueError(
                f"truth_coverage_fraction={self.truth_coverage_fraction} disagrees with the measured "
                f"value {measured:.4f} for {len(self.samples)} samples and "
                f"expected_sample_count={self.expected_sample_count}"
            )
        if self.max_sample_gap_s is None and max_gap is not None:
            object.__setattr__(self, "max_sample_gap_s", round(max_gap, 6))
        if self.achieved_sample_interval_s is None and mean_interval is not None:
            object.__setattr__(self, "achieved_sample_interval_s", round(mean_interval, 6))
        return self

    def events_of(self, *kinds: str) -> list[TruthEvent]:
        wanted = set(kinds)
        return [e for e in self.events if e.kind in wanted]

    @property
    def duration_s(self) -> float:
        if not self.samples:
            return 0.0
        times = [s.sim_time_s for s in self.samples]
        return max(times) - min(times)


# --------------------------------------------------------------------------------------
# Attempted-run ledger
# --------------------------------------------------------------------------------------
RunStatus = Literal["completed", "crashed", "timeout", "reset_failed", "skipped", "aborted", "partial"]


class AttemptedRun(StrictModel):
    """One attempted episode, including failures. Nothing is silently discarded."""

    attempt_id: str
    episode_id: str | None
    scenario_id: str
    arm_id: str
    run_class: Literal["fixture", "pilot", "heldout", "smoke"]
    protocol_hash: str
    status: RunStatus
    started_wall_clock: str
    finished_wall_clock: str | None = None
    wall_clock_duration_s: float | None = None
    sim_duration_s: float | None = None
    termination_reason: str | None = None
    simulator_provenance: str = "unverified"
    error_type: str | None = None
    error_message: str | None = None
    episode_record_path: str | None = None
    ledger_path: str | None = None
    retry_of: str | None = None
    notes: str = ""
