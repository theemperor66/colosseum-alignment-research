"""Offline audit record variants: prespecified field ablation on a fixed trajectory.

Why this module exists: the audit experiment compares what an auditor can establish from an
action-only record against what a provenance-rich record supports, on the *same* completed episodes
(research-plan.md, "A bounded audit experiment"). That comparison is only meaningful if the ablation
touches the record and nothing else. Two safeguards enforce that here.

* **Trajectory invariance.** Command kinds, targets, speeds, issuers, step indices and the sim/wall
  timestamps are never ablatable, and :func:`trajectory_signature` proves that every produced variant
  carries the same trajectory as its source. Without this proof a logging change could be mistaken for
  a safety improvement, which is exactly the confusion the study is designed to avoid. The single
  exception is a command's free-text annotation (``reason``, ``controller_phase``), which the frozen
  protocol removes in the action-only variant: deleting an explanation does not change what was
  commanded, so it is excluded from the signature and permitted as a removal target.
* **No silent no-op ablation.** An ablation that removes nothing would look like a successful
  degradation and would quietly inflate the reconstruction rate of the degraded arm. A path that never
  existed in the episode is therefore an :class:`AblationError`, not a shrug.

Removal deletes the key. Setting it to ``None`` would leave the auditor a usable signal ("this field
was recorded but empty") that a truly action-only record does not contain.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import Field

from colosseum_assurance.protocol.spec import AuditAblation, AuditSpec, ProtocolConfig
from colosseum_assurance.schemas import EpisodeRecord, StepRecord, StrictModel

__all__ = [
    "ABLATABLE_COMMAND_ANNOTATIONS",
    "COMMAND_FIELDS",
    "PROTECTED_STEP_FIELDS",
    "SIGNED_STEP_FIELDS",
    "AblationError",
    "AuditRecordSet",
    "build_record_variant",
    "build_record_variants",
    "build_variants",
    "field_paths",
    "lookup_path",
    "paths_under",
    "removed_paths",
    "trajectory_signature",
]

#: Step fields that carry the trajectory itself. Ablating one of these would change what happened,
#: not what was retained about it, so requesting their removal is refused.
PROTECTED_STEP_FIELDS = frozenset(
    {"step_index", "sim_time_s", "wall_clock_s", "command", "executed_command"}
)

#: The two command objects of a step.
COMMAND_FIELDS = ("command", "executed_command")

#: Free-text annotations *inside* a command. They explain the command; they do not define it. The frozen
#: protocol removes them in the action-only variant so that no free-text reason survives anywhere, and
#: that removal leaves the trajectory identical. Every other command sub-field is refused.
ABLATABLE_COMMAND_ANNOTATIONS = frozenset({"reason", "controller_phase"})

#: Top-level step fields hashed into the trajectory signature, in this order.
SIGNED_STEP_FIELDS = ("step_index", "sim_time_s", "wall_clock_s", "command", "executed_command")

#: Episode-level fields promoted to typed attributes of :class:`AuditRecordSet`; everything else stays
#: in ``episode_fields`` so the auditor still sees simulator identity, code version and interventions.
_PROMOTED_EPISODE_FIELDS = frozenset(
    {"steps", "episode_id", "scenario_id", "arm_id", "run_class", "protocol_hash", "policy_version",
     "dt_s"}
)

class AblationError(ValueError):
    """Raised when an ablation is malformed, removes nothing, or would alter the trajectory."""


# --------------------------------------------------------------------------------------
# Path utilities over JSON-safe step dictionaries
# --------------------------------------------------------------------------------------
def _as_step_dict(step: Mapping[str, Any] | StepRecord) -> dict[str, Any]:
    """Accept either a live :class:`StepRecord` or an already serialized step."""
    if isinstance(step, StepRecord):
        return step.model_dump(mode="json")
    return dict(step)


def field_paths(node: Any, prefix: str = "") -> set[str]:
    """Return every dotted path inside ``node``, treating lists as leaf values.

    Used to *diff* a variant against its source: the difference of two path sets is the exact set of
    fields an ablation removed, including everything under a removed subtree. Lists are leaves because
    no ablatable path in the frozen protocol addresses a list element.
    """
    out: set[str] = set()
    if isinstance(node, Mapping):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.add(path)
            out |= field_paths(value, path)
    return out


def paths_under(node: Mapping[str, Any], target: str) -> set[str]:
    """Return ``target`` and every existing path below it, i.e. the subtree an ablation would delete."""
    present = field_paths(node)
    return {p for p in present if p == target or p.startswith(f"{target}.")}


def removed_paths(source: Mapping[str, Any], ablated: Mapping[str, Any]) -> set[str]:
    """Paths present in ``source`` but absent from ``ablated``."""
    return field_paths(source) - field_paths(ablated)


def lookup_path(node: Any, path: str, default: Any = None) -> Any:
    """Read a dotted path, returning ``default`` when any component is missing or not a mapping."""
    current: Any = node
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _path_exists(node: Any, parts: Sequence[str]) -> bool:
    """True when the dotted path is present, so a missing path can be told from a removed one."""
    if isinstance(node, list):
        return any(_path_exists(item, parts) for item in node)
    if not parts:
        return True
    if not isinstance(node, Mapping) or parts[0] not in node:
        return False
    return _path_exists(node[parts[0]], parts[1:])


def _remove_path(node: Any, parts: Sequence[str]) -> bool:
    """Delete ``parts`` from ``node`` in place; return True when a key was actually deleted.

    Lists are traversed element-wise so a future list-valued sub-record (several monitor reports per
    step, say) degrades consistently instead of silently surviving the ablation.
    """
    if isinstance(node, list):
        hit = False
        for item in node:
            if _remove_path(item, parts):
                hit = True
        return hit
    if not isinstance(node, dict) or not parts:
        return False
    head = parts[0]
    if len(parts) == 1:
        if head in node:
            del node[head]
            return True
        return False
    if head not in node:
        return False
    return _remove_path(node[head], parts[1:])


def _validate_removed_field(path: str) -> list[str]:
    """Reject empty paths and any path that would touch the trajectory.

    A command annotation (``command.reason``, ``executed_command.controller_phase``) is the one
    permitted exception: the frozen protocol ablates it, and deleting an explanatory string cannot
    change which command was issued, when, by whom, or with what target and speed. Everything else
    under a command, and the command objects themselves, stay.
    """
    if not path or not path.strip():
        raise AblationError("removed_field must be a non-empty dotted path")
    parts = path.split(".")
    if any(not part for part in parts):
        raise AblationError(f"removed_field {path!r} has an empty path component")
    if parts[0] in COMMAND_FIELDS:
        if len(parts) == 2 and parts[1] in ABLATABLE_COMMAND_ANNOTATIONS:
            return parts
        raise AblationError(
            f"removed_field {path!r} targets protected step field {parts[0]!r}: a command's kind, "
            "target, speed, timing and issuer fix the trajectory and are never ablatable (only "
            f"{sorted(ABLATABLE_COMMAND_ANNOTATIONS)} may be removed from a command)"
        )
    if parts[0] in PROTECTED_STEP_FIELDS:
        raise AblationError(
            f"removed_field {path!r} targets protected step field {parts[0]!r}: commands, executed "
            "commands, step indices and timestamps fix the trajectory and are never ablatable"
        )
    return parts


# --------------------------------------------------------------------------------------
# Trajectory signature
# --------------------------------------------------------------------------------------
def _command_core(value: Any) -> Any:
    """A command stripped of its free-text annotations: what was done, not why it was explained.

    The annotations are excluded from the signature because the frozen protocol removes them; keeping
    them in would make a legal ablation look like a trajectory change and hide real ones behind noise.
    """
    if not isinstance(value, Mapping):
        return value
    return {k: v for k, v in value.items() if k not in ABLATABLE_COMMAND_ANNOTATIONS}


def trajectory_signature(steps: Iterable[Mapping[str, Any] | StepRecord]) -> str:
    """Hash the unablatable part of a step sequence.

    Equality of this digest across variants is the evidence that offline ablation changed the record
    only. A missing signed field hashes as JSON ``null``, so an illegal removal changes the digest
    instead of passing unnoticed.
    """
    payload = []
    for step in steps:
        as_dict = _as_step_dict(step)
        row: dict[str, Any] = {}
        for name in SIGNED_STEP_FIELDS:
            value = as_dict.get(name)
            row[name] = _command_core(value) if name in COMMAND_FIELDS else value
        payload.append(row)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# Record variants
# --------------------------------------------------------------------------------------
class AuditRecordSet(StrictModel):
    """One episode as exposed to the reconstruction procedure under one prespecified ablation.

    This is the *only* object the reconstructor may read (besides the public protocol). It deliberately
    carries no truth samples, no manifest geometry and no reference answers.
    """

    variant_id: str
    episode_id: str
    scenario_id: str
    arm_id: str
    run_class: str
    protocol_hash: str
    policy_version: str | None = Field(
        default=None, description="None once a future ablation removes the episode-level policy version."
    )
    removed_fields: list[str] = Field(default_factory=list)
    missed_paths: list[str] = Field(default_factory=list)
    vacuous_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Removed paths whose whole parent was absent for structural reasons, for example a monitor "
            "field in an unguarded episode. Recorded so a vacuous removal stays visible."
        ),
    )
    trajectory_signature: str
    dt_s: float
    termination_reason: str
    steps: list[dict[str, Any]] = Field(default_factory=list)
    episode_fields: dict[str, Any] = Field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def step_at(self, step_index: int) -> dict[str, Any] | None:
        """Return the step with this ``step_index`` (list position is not assumed to match)."""
        for step in self.steps:
            if step.get("step_index") == step_index:
                return step
        return None


def _lookup_or_none(step: Mapping[str, Any], parts: Sequence[str]) -> Any:
    """Return the value at ``parts`` or None when any level is missing or itself None.

    A serialized optional field is present as JSON ``null``. For ablation purposes that is the same as
    absent: there is nothing there for an auditor to read.
    """
    node: Any = step
    for part in parts:
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
        if node is None:
            return None
    return node


def build_record_variant(
    record: EpisodeRecord,
    ablation: AuditAblation,
    *,
    strict: bool = True,
) -> AuditRecordSet:
    """Apply one prespecified ablation to ``record`` and return the auditor-visible record set.

    ``strict`` exists for exploratory use only: with ``strict=False`` a path that never existed in the
    episode is collected in ``missed_paths`` instead of raising, so a schema drift can be measured
    across a whole run before it is fixed. Reported audit results use ``strict=True``.
    """
    payload = record.model_dump(mode="json")
    source_steps: list[dict[str, Any]] = payload.get("steps", [])
    steps = copy.deepcopy(source_steps)
    missed: list[str] = []

    vacuous: list[str] = []
    for path in ablation.removed_fields:
        parts = _validate_removed_field(path)
        existed = any(_path_exists(step, parts) for step in source_steps)
        for step in steps:
            _remove_path(step, parts)
        if existed:
            continue
        # A nested path whose whole parent is absent for structural reasons is vacuous, not a bug. The
        # unguarded arm has no monitor, so it has no `monitor_report` and no field inside it. Removing
        # `monitor_report.evidence_age_s` there removes nothing and overstates nothing, because the
        # parent is already absent for every auditor. A leaf that is missing while its parent EXISTS is
        # still an error: that is the schema drift this check was written to catch.
        parent_absent = len(parts) > 1 and all(
            _lookup_or_none(step, parts[:-1]) is None for step in source_steps
        )
        if parent_absent:
            vacuous.append(path)
            continue
        message = (
            f"ablation {ablation.ablation_id!r} removes {path!r}, which exists in no step of "
            f"episode {record.episode_id!r}: a no-op ablation would silently overstate what the "
            "degraded record still supports"
        )
        if strict:
            raise AblationError(message)
        missed.append(path)

    source_signature = trajectory_signature(source_steps)
    variant_signature = trajectory_signature(steps)
    if variant_signature != source_signature:
        raise AblationError(
            f"ablation {ablation.ablation_id!r} changed the trajectory of episode "
            f"{record.episode_id!r} ({source_signature} -> {variant_signature}); record ablation must "
            "leave commands and timestamps untouched"
        )

    episode_fields = {k: v for k, v in payload.items() if k not in _PROMOTED_EPISODE_FIELDS}
    return AuditRecordSet(
        variant_id=ablation.ablation_id,
        episode_id=record.episode_id,
        scenario_id=record.scenario_id,
        arm_id=record.arm_id,
        run_class=record.run_class,
        protocol_hash=record.protocol_hash,
        policy_version=record.policy_version,
        removed_fields=list(ablation.removed_fields),
        missed_paths=missed,
        vacuous_paths=vacuous,
        trajectory_signature=source_signature,
        dt_s=record.dt_s,
        termination_reason=record.termination.reason,
        steps=steps,
        episode_fields=episode_fields,
    )


def build_record_variants(
    record: EpisodeRecord,
    audit: AuditSpec,
    *,
    strict: bool = True,
) -> dict[str, AuditRecordSet]:
    """Build every variant in the frozen audit schedule, keyed by ``ablation_id``.

    All variants share one source record, so any difference in reconstruction between them is caused by
    the retained fields alone.
    """
    variants: dict[str, AuditRecordSet] = {}
    for ablation in audit.record_variants:
        if ablation.ablation_id in variants:
            raise AblationError(f"duplicate ablation_id {ablation.ablation_id!r} in the audit schedule")
        variants[ablation.ablation_id] = build_record_variant(record, ablation, strict=strict)
    signatures = {v.trajectory_signature for v in variants.values()}
    if len(signatures) > 1:
        raise AblationError(f"variants of episode {record.episode_id!r} disagree on the trajectory")
    return variants


def build_variants(record: EpisodeRecord, protocol: ProtocolConfig) -> list[AuditRecordSet]:
    """Ordered-list form of :func:`build_record_variants` for callers that iterate the schedule.

    The order is exactly ``protocol.audit.record_variants``, so a caller can zip variants against the
    frozen schedule without trusting dictionary ordering.
    """
    variants = build_record_variants(record, protocol.audit, strict=True)
    return [variants[ablation.ablation_id] for ablation in protocol.audit.record_variants]
