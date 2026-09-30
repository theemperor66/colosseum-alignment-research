"""Pure source eligibility for future one-step replay; never simulator access or outcomes."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.runtime.evidence import utc_now
from colosseum_assurance.scenario.manifest import ScenarioManifest
from colosseum_assurance.schemas import EpisodeRecord, PrivilegedLedger

DEFAULT_TOLERANCES = {"position_m": 0.05, "velocity_mps": 0.05, "yaw_rad": 0.01, "time_s": 0.01}
FORMAT = "executed_prefix_replay_source_plan_v1"
SELECTORS = {
    "first_move_after_step1": "First executed move_to at retained step_index >=2; apparatus contrast only.",
    "first_direct_guard_nonhold": "First direct guard-issued executed command distinct from pure hold.",
}


class ReplayRefused(ValueError):
    """A prerequisite fails; never substitute a proposed action or inferred world state."""


def package_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def source_bytes(source: Path, relative: str) -> bytes:
    path = source / relative
    if not path.is_file() or path.is_symlink() or source not in path.resolve().parents:
        raise ReplayRefused(f"missing or unsafe source file: {relative}")
    return path.read_bytes()


def truth_at(ledger, at, tolerance):
    samples = sorted(ledger.samples, key=lambda s: abs(s.sim_time_s - at))
    if not samples or abs(samples[0].sim_time_s - at) > tolerance:
        raise ReplayRefused(f"source truth does not establish the state at {at:.6f} s")
    sample = samples[0]
    if not all(
        math.isfinite(x)
        for x in (sample.sim_time_s, sample.yaw_rad, *sample.position.as_tuple(), *sample.velocity.as_tuple())
    ):
        raise ReplayRefused("source physical anchor contains nonfinite values")
    return sample


def prepare_source(
    source: Path | str, episode_id: str, step_index: int, tolerances: dict[str, float] | None = None
):
    """Return parsed source and a byte-bound contract without any RPC or filesystem write."""
    source = Path(source).resolve()
    tolerances = dict(DEFAULT_TOLERANCES if tolerances is None else tolerances)
    if (
        type(step_index) is not int
        or step_index < 0
        or set(tolerances) != set(DEFAULT_TOLERANCES)
        or any(not math.isfinite(v) or v < 0 for v in tolerances.values())
    ):
        raise ReplayRefused("step and finite nonnegative tolerances are required")
    if Path(episode_id).name != episode_id or episode_id in {"", ".", ".."}:
        raise ReplayRefused("episode_id must be a single identifier, not a path")
    raw = {}

    def read(relative):
        raw[relative] = source_bytes(source, relative)
        return raw[relative]

    envelope = json.loads(read("protocol.json"))
    protocol = ProtocolConfig.model_validate(envelope["protocol"])
    if protocol.content_hash() != envelope.get("protocol_hash"):
        raise ReplayRefused("stored protocol content hash mismatch")
    record = EpisodeRecord.model_validate_json(read(f"episodes/{episode_id}.json"))
    ledger = PrivilegedLedger.model_validate_json(read(f"privileged_ledgers/{episode_id}.json"))
    if Path(record.scenario_id).name != record.scenario_id:
        raise ReplayRefused("unsafe scenario identity")
    manifest = ScenarioManifest.model_validate_json(read(f"manifests/{record.scenario_id}.json"))
    identity = record.simulator_identity
    if record.run_class in {"pilot", "heldout"} and not identity.is_live:
        raise ReplayRefused("experimental source has no anchored live simulator identity")
    if (
        record.episode_id != episode_id
        or record.episode_id != ledger.episode_id
        or record.scenario_id != ledger.scenario_id
        or record.scenario_id != manifest.scenario_id
        or record.arm_id != ledger.arm_id
        or record.run_class != ledger.run_class
        or record.run_class != manifest.run_class
        or any(m.protocol_hash != protocol.content_hash() for m in (record, ledger, manifest))
    ):
        raise ReplayRefused("source record, ledger, manifest and protocol identities disagree")
    if record.simulator_identity != ledger.simulator_identity:
        raise ReplayRefused("source record and truth ledger disagree on simulator identity")
    if not identity.is_live and identity.provenance != "fixture_fake":
        raise ReplayRefused("source simulator provenance is unverified")
    if identity.is_live and not re.fullmatch(r"[0-9a-f]{64}", record.code_version.get("source_sha256", "")):
        raise ReplayRefused("live source has no exact recorded package source digest")
    prefix = record.steps[: step_index + 1]
    if (
        len(prefix) != step_index + 1
        or [s.step_index for s in prefix] != list(range(step_index + 1))
        or any(s.executed_command is None for s in prefix)
    ):
        raise ReplayRefused("complete ordered executed-command prefix is unavailable")
    if any(s.executed_command.step_index != s.step_index for s in prefix):
        raise ReplayRefused("executed command and retained step indexes disagree")
    if any(
        not math.isfinite(s.executed_command.issued_sim_time_s)
        or abs(s.executed_command.issued_sim_time_s - s.sim_time_s) > tolerances["time_s"]
        for s in prefix
    ):
        raise ReplayRefused("executed command timestamp differs from retained step")
    selected = prefix[-1]
    if abs(record.dt_s - protocol.mission.control_dt_s) > 1e-9:
        raise ReplayRefused("recorded control interval differs from protocol")
    if selected.executed_command.kind == "hold" and selected.executed_command.yaw_rad is None:
        raise ReplayRefused("selected command is already a pure hold; no distinct alternative")
    times = [s.sim_time_s for s in prefix]
    if any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise ReplayRefused("executed prefix timestamps are not increasing")
    anchors = [truth_at(ledger, s.sim_time_s, tolerances["time_s"]) for s in prefix]
    end = truth_at(ledger, selected.sim_time_s + protocol.mission.control_dt_s, tolerances["time_s"])
    files = {
        name: {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
        for name, value in raw.items()
    }
    if any(source_bytes(source, name) != value for name, value in raw.items()):
        raise ReplayRefused("source changed during preflight")
    contract = {
        "status": "source_eligible_live_alignment_unverified",
        "episode_id": episode_id,
        "scenario_id": record.scenario_id,
        "arm_id": record.arm_id,
        "run_class": record.run_class,
        "step_index": step_index,
        "selected_executed_command": selected.executed_command.model_dump(mode="json"),
        "protocol_hash": protocol.content_hash(),
        "manifest_hash": manifest.content_hash(),
        "source_code_version": record.code_version,
        "source_simulator_identity": identity.model_dump(mode="json"),
        "source_input_files": files,
        "tolerances": tolerances,
        "prefix_length": len(prefix),
        "executed_prefix_sha256": canonical_digest(
            [s.executed_command.model_dump(mode="json") for s in prefix]
        ),
        "source_time_anchors": {
            "prefix_max_gap_s": max(
                abs(a.sim_time_s - s.sim_time_s) for a, s in zip(anchors, prefix, strict=True)
            ),
            "selected_start_s": selected.sim_time_s,
            "selected_end_s": selected.sim_time_s + record.dt_s,
            "start_sample_s": anchors[-1].sim_time_s,
            "end_sample_s": end.sim_time_s,
            "end_gap_s": abs(end.sim_time_s - selected.sim_time_s - protocol.mission.control_dt_s),
        },
        "alternative_consequences_measured": False,
    }
    provenance = {
        "source": "run_tree",
        "path": str(source / "protocol.json"),
        "protocol_hash": protocol.content_hash(),
        "written_utc": envelope.get("written_utc"),
    }
    return protocol, provenance, record, ledger, manifest, contract


def source_inventory(source: Path):
    paths = [source / "protocol.json"]
    paths += [
        source / name for name in ("attempted_runs.jsonl", "run_metadata.json") if (source / name).is_file()
    ]
    paths += [
        p
        for folder in ("episodes", "privileged_ledgers", "manifests")
        for p in sorted((source / folder).glob("*.json"))
    ]
    return {
        p.relative_to(source).as_posix(): {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        for p in paths
        for raw in [source_bytes(source, p.relative_to(source).as_posix())]
    }


def select_step(record, selector):
    if selector not in SELECTORS:
        raise ReplayRefused("unsupported prespecified replay selector")
    for step in record.steps:
        command = step.executed_command
        if command is None:
            continue
        if selector == "first_move_after_step1":
            if step.step_index >= 2 and command.kind == "move_to":
                return step
        elif command.issued_by == "guard" and not (command.kind == "hold" and command.yaw_rad is None):
            return step
    return None


def compile_replay_plan(
    run_dir: Path | str, out: Path | str, *, selector: str = "first_move_after_step1"
) -> dict[str, Any]:
    """Inventory every retained attempt/episode; select first executed move_to at index >=2.

    The rule uses no outcomes or alternative results. Historical outcomes may already be known.
    Refused and partial records remain visible; this is source feasibility, not a flight count.
    """
    source, target = Path(run_dir).resolve(), Path(out).resolve()
    if selector not in SELECTORS:
        raise ValueError("unsupported prespecified replay selector")
    if target == source or source in target.parents:
        raise ValueError("plan output must be outside the source evidence run")
    if target.exists():
        raise FileExistsError(target)
    implementation_hash = package_digest()
    before = source_inventory(source)
    attempts = []
    if "attempted_runs.jsonl" in before:
        attempts = [
            json.loads(line)
            for line in source_bytes(source, "attempted_runs.jsonl").splitlines()
            if line.strip()
        ]
    counts = Counter(row["episode_id"] for row in attempts)
    episode_files = {p.stem for p in (source / "episodes").glob("*.json")}
    ledger_files = {p.stem for p in (source / "privileged_ledgers").glob("*.json")}
    identities = sorted(set(counts) | episode_files | ledger_files)
    rows = []
    for eid in identities:
        base = {
            "episode_id": eid,
            "attempt_rows": counts[eid],
            "attempt_statuses": [r.get("status") for r in attempts if r["episode_id"] == eid],
        }
        try:
            if counts[eid] != 1:
                raise ReplayRefused("exactly one retained attempt row required; orphan/duplicate retained")
            episode = EpisodeRecord.model_validate_json(source_bytes(source, f"episodes/{eid}.json"))
            candidate = select_step(episode, selector)
            if candidate is None:
                raise ReplayRefused(f"no command meets {selector}; no substituted branch")
            *_, contract = prepare_source(source, eid, candidate.step_index)
            attempt = next(r for r in attempts if r["episode_id"] == eid)
            if any(attempt.get(k) != contract[k] for k in ("scenario_id", "arm_id", "protocol_hash")):
                raise ReplayRefused("attempt row differs from source episode identity")
            rows.append(dict(base, **contract))
        except (ValueError, KeyError, OSError, TypeError) as exc:
            rows.append(dict(base, status="refused", refusal_reason=f"{type(exc).__name__}: {exc}"))
    if source_inventory(source) != before or package_digest() != implementation_hash:
        raise ReplayRefused("source inventory or compiler implementation changed during compilation")
    report = {
        "format": FORMAT,
        "created_utc": utc_now(),
        "source_run": str(source),
        "status": "compiled_source_eligibility_only",
        "new_flights": 0,
        "alternative_consequences_measured": 0,
        "replay_implementation_source_sha256": implementation_hash,
        "selector": selector,
        "selection_rule": SELECTORS[selector] + " No outcome selection.",
        "prior_knowledge": "Historical outcomes may be known; rule fixed before new alternative execution.",
        "source_inventory": before,
        "retained_source_identities": len(identities),
        "eligible_sources": sum(r["status"] != "refused" for r in rows),
        "refused_sources": sum(r["status"] == "refused" for r in rows),
        "candidates": rows,
        "limits": [
            "Source eligibility does not establish reset, hidden-state or prefix equivalence.",
            "Early movement contrasts test apparatus, not substantive ethical alternatives.",
            "Guard selection measures recorded guard action versus hold, "
            "not unguarded policy or an indirect suspension.",
            "Fresh live gate, exact simulator artifact and measured branch alignment remain mandatory.",
            "No source outcome or alternative consequence is inferred from command replay eligibility.",
            "Source record inventory is not a new independent empirical sample or a full study denominator.",
        ],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def verify_compiled_plan(path: Path | str, source: Path, contract: dict[str, Any]) -> dict[str, str]:
    raw = Path(path).read_bytes()
    plan = json.loads(raw)
    if plan.get("format") != FORMAT or plan.get("replay_implementation_source_sha256") != package_digest():
        raise ReplayRefused("compiled plan format or replay implementation source changed")
    if source_inventory(source) != plan.get("source_inventory"):
        raise ReplayRefused("source inventory differs from compiled plan")
    record = EpisodeRecord.model_validate_json(
        source_bytes(source, f"episodes/{contract['episode_id']}.json")
    )
    selected = select_step(record, plan.get("selector"))
    if selected is None or selected.step_index != contract["step_index"]:
        raise ReplayRefused("selected branch differs from compiled selection rule")
    rows = [r for r in plan.get("candidates", []) if r.get("episode_id") == contract["episode_id"]]
    if len(rows) != 1 or any(rows[0].get(k) != v for k, v in contract.items()):
        raise ReplayRefused("compiled candidate differs from current source preflight")
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
