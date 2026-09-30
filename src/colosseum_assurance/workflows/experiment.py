"""Experimental run workflow: live Colosseum only.

Before any episode runs, the live readiness gate must pass. The gate checks genuine simulator provenance, a
successful reset, a changed trajectory, and nonempty RGB and depth frames. Its report is copied into the run
tree, so every experimental result carries the evidence that a real simulator produced it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from colosseum_assurance.config import AppConfig
from colosseum_assurance.interfaces import AdapterError
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.runtime.episode import EpisodeRunner
from colosseum_assurance.runtime.evidence import (
    EvidenceWriter,
    load_attempted_runs,
    summarize_attempts,
    utc_now,
)
from colosseum_assurance.scenario.manifest import ScenarioManifest, build_manifest
from colosseum_assurance.scenario.qualification import arm_order_plan, inspection_geometry_qualification
from colosseum_assurance.schemas import AttemptedRun
from colosseum_assurance.sim.scene import SceneError
from colosseum_assurance.version import code_version


class LiveGateFailure(RuntimeError):
    """Raised when the live readiness gate does not pass. Experimental runs must not start."""


def _record_setup_failure(
    writer: EvidenceWriter, manifest: ScenarioManifest, arm_id: str, error: Exception,
    started: str, wall_start: float, provenance: str,
) -> AttemptedRun:
    """Retain failures before the runner could produce an episode, without duplicating its ledger.

    The requested manifest places a failed binding in its planned severity cell. It is never described
    as flown geometry. If the runner already wrote an attempt, that attempt remains authoritative.
    """
    episode_id = f"{manifest.scenario_id}__{arm_id}"
    prior = load_attempted_runs(writer.root / "attempted_runs.jsonl")
    for attempt in prior:
        if attempt.episode_id == episode_id:
            return attempt
    writer.write_manifest(manifest)
    attempt = AttemptedRun(
        attempt_id=str(uuid4()), episode_id=episode_id, scenario_id=manifest.scenario_id,
        arm_id=arm_id, run_class=manifest.run_class, protocol_hash=manifest.protocol_hash,
        status="partial", started_wall_clock=started, finished_wall_clock=utc_now(),
        wall_clock_duration_s=round(time.monotonic() - wall_start, 3),
        termination_reason="setup_failed", simulator_provenance=provenance,
        error_type=type(error).__name__, error_message=str(error)[:4000],
        notes="Setup failed before an episode record existed; manifest is the planned scenario.",
    )
    writer.append_attempt(attempt)
    return attempt


def run_experiment(
    config: AppConfig,
    protocol: ProtocolConfig,
    protocol_meta: dict[str, Any],
    realizations: int | None = None,
    cell_filter: list[str] | None = None,
    arm_filter: list[str] | None = None,
    results_root: Path | None = None,
    limit: int | None = None,
    skip_gate: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Run pilot or held-out episodes against a live simulator, recording every attempt."""
    from colosseum_assurance.sim import build_adapter
    from colosseum_assurance.sim.diagnostics import run_live_readiness_gate

    if config.run_class not in {"pilot", "heldout"}:
        raise ValueError("run_experiment is for run_class 'pilot' or 'heldout' only")
    if not protocol_meta.get("frozen"):
        raise ValueError(
            "experimental runs require a frozen protocol file (use `colassure freeze`), so the protocol "
            "hash in every record refers to a fixed, inspectable specification"
        )
    extension = protocol.study_extension
    if config.run_class == "heldout" and extension is not None and extension.search_role == "exploration":
        raise ValueError("exploratory search protocols cannot produce held-out evidence")
    if results_root is not None:
        new_paths = config.paths.model_copy(update={"results_root": results_root})
        config = config.model_copy(update={"paths": new_paths})

    writer = EvidenceWriter(
        paths=config.paths, run_class=config.run_class, protocol_hash=protocol.content_hash(),
        protocol=protocol,
    )
    session_id = str(uuid4())
    if resume:
        prior_source = json.loads(writer.metadata_path.read_text())["code_version"].get("source_sha256")
        if (not prior_source or prior_source == "unknown"
                or prior_source != code_version().get("source_sha256")):
            raise ValueError("resume requires the exact recorded package source SHA; use a new study")
    metadata_path = writer.root / "frozen_protocol_meta.json"
    if not metadata_path.exists():
        metadata_path.write_text(
            json.dumps(protocol_meta, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    per_cell = realizations or (
        protocol.sampling.pilot_realizations_per_cell
        if config.run_class == "pilot"
        else protocol.sampling.heldout_realizations_per_cell
    )
    cells = [c["cell_id"] for c in protocol.cells()]
    if cell_filter:
        unknown = sorted(set(cell_filter) - set(cells))
        if unknown:
            raise ValueError(f"unknown cell ids: {unknown}")
        cells = [c for c in cells if c in cell_filter]
    arms = arm_filter or protocol.arms.arm_ids
    unknown_arms = sorted(set(arms) - set(protocol.arms.arm_ids))
    if unknown_arms:
        raise ValueError(f"unknown arm ids: {unknown_arms}")

    manifests = []
    for cell_id in cells:
        for realization in range(per_cell):
            if extension is not None:
                from colosseum_assurance.scenario.expanded import build_expanded_manifest

                manifests.append(build_expanded_manifest(protocol, config.run_class, realization))
            else:
                manifests.append(build_manifest(protocol, config.run_class, cell_id, realization))
    order = arm_order_plan(protocol, manifests)
    manifest_by_id = {m.scenario_id: m for m in manifests}
    if len(manifest_by_id) != len(manifests):
        raise ValueError("collection would repeat a scenario ID")
    if protocol.controlled_study is not None:
        order_payload = {"protocol_hash": protocol.content_hash(), "plan": order,
                         "note": "Cyclic/reverse rotations; incomplete blocks are not fully counterbalanced."}
        order_path = writer.root / "collection_order.json"
        if order_path.exists() and json.loads(order_path.read_text()) != order_payload:
            raise ValueError("resume collection order/manifest inventory differs from the original plan")
        if not order_path.exists():
            order_path.write_text(json.dumps(order_payload, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    planned_ids = {f"{m.scenario_id}__{a}" for m in manifests for a in protocol.arms.arm_ids}
    existing_attempts = load_attempted_runs(writer.root / "attempted_runs.jsonl")
    if (len({a.episode_id for a in existing_attempts}) != len(existing_attempts)
            or any(a.episode_id not in planned_ids for a in existing_attempts)):
        raise ValueError("existing attempts contain duplicate or unplanned episodes")
    intent_path = writer.root / "attempt_starts.jsonl"
    intents = ([json.loads(row) for row in intent_path.read_text().splitlines() if row]
               if intent_path.exists() else [])
    if any(row["episode_id"] not in planned_ids for row in intents):
        raise ValueError("existing start intents contain unplanned episodes")
    prior_started = {row["episode_id"] for row in intents}
    resume_skipped: dict[str, Any] = {}

    pilot_coverage = None
    if extension is not None and config.run_class == "pilot":
        from colosseum_assurance.scenario.expanded import coverage_plan

        full_sets = per_cell if limit is None else min(per_cell, limit // max(len(arms), 1))
        pilot_coverage = coverage_plan(protocol, full_sets)
        pilot_coverage["all_comparison_arms_planned"] = set(arms) == set(protocol.arms.arm_ids)
        pilot_coverage["coverage_floor_met"] = (
            pilot_coverage["covers_declared_domain"] and pilot_coverage["all_comparison_arms_planned"]
        )
        pilot_coverage["note"] = (
            "Planned coverage only; setup failures and missing completed arm sets must also be reviewed "
            "before freezing held-out evaluation. Limited pilots do not satisfy the coverage floor."
        )
        if realizations is None and limit is None and not pilot_coverage["covers_declared_domain"]:
            raise ValueError("the default pilot fails to cover declared layout/visibility strata")
        coverage_name = f"pilot_coverage-{session_id}.json" if resume else "pilot_coverage.json"
        coverage_path = writer.root / coverage_name
        coverage_path.write_text(
            json.dumps(pilot_coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Validate source, inventory and frozen order before any RPC/reset in a resumed collection.
    gate_payload: dict[str, Any] | None = None
    if not skip_gate:
        gate = run_live_readiness_gate(config, protocol)
        gate_payload = json.loads(gate.model_dump_json())
        gate_path = writer.root / "live_gate_report.json"
        if gate_path.exists():
            gate_path = writer.root / f"live_gate_report-{session_id}.json"
        gate_path.write_text(json.dumps(gate_payload, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        if not gate.passed:
            raise LiveGateFailure(
                "live readiness gate failed; refusing to record experimental episodes.\n"+gate.render_text())
    adapter = build_adapter(config, protocol)
    episode_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    try:
        runner = EpisodeRunner(adapter, protocol, config, writer=writer, save_frames=True)
        count = 0
        for group in order:
            # Keep the historical loop's failure/accounting behavior; only the declared order varies.
            for manifest in [manifest_by_id[group["scenario_id"]]]:
                if limit is not None and count >= limit:
                    break
                group_arms = ([arm for arm in group["arms"] if arm in arms]
                              if protocol.controlled_study is not None else arms)
                if resume:
                    remaining = []
                    for arm_id in group_arms:
                        eid = f"{manifest.scenario_id}__{arm_id}"
                        evidence = writer.existing_evidence(eid)
                        if evidence or eid in prior_started:
                            resume_skipped[eid] = {"evidence": evidence, "start_intent": eid in prior_started,
                                                   "attempt_recorded": any(a.episode_id == eid
                                                                           for a in existing_attempts)}
                        else:
                            remaining.append(arm_id)
                    group_arms = remaining
                planned_arms = group_arms if limit is None else group_arms[:max(0, limit - count)]
                if not planned_arms:
                    continue
                # Refuse repeats before resetting or mutating the world, including prior setup failures.
                for arm_id in planned_arms:
                    writer.assert_episode_not_recorded(f"{manifest.scenario_id}__{arm_id}")
                started, wall_start = utc_now(), time.monotonic()
                # One binding per scenario: the matched arm set must fly one verified world.
                try:
                    if (protocol.controlled_study is not None
                            and protocol.controlled_study.require_feasible_inspection_geometry):
                        qualification = inspection_geometry_qualification(protocol, manifest)
                        location = writer.root / "geometry_qualification"
                        location.mkdir(exist_ok=True)
                        qpath = location / f"{manifest.scenario_id}.json"
                        if qpath.exists() and json.loads(qpath.read_text()) != qualification:
                            raise ValueError("authored geometry qualification changed on resume")
                        if not qpath.exists():
                            qpath.write_text(json.dumps(qualification, sort_keys=True, indent=2)+"\n",
                                             encoding="utf-8")
                        if not qualification["passed"]:
                            raise SceneError("authored mission geometry qualification failed: "
                                             + "; ".join(qualification["failures"]))
                    binding = runner.bind_scenario(manifest)
                except (AdapterError, SceneError) as exc:
                    errors.append({
                        "scenario_id": manifest.scenario_id, "arm_id": "*",
                        "error_type": type(exc).__name__, "error": str(exc),
                        "stage": "scene_binding",
                    })
                    for arm_id in planned_arms:
                        _record_setup_failure(
                            writer, manifest, arm_id, exc, started, wall_start,
                            gate_payload.get("provenance", "unverified") if gate_payload else "unverified",
                        )
                        count += 1
                    continue
                for arm_id in planned_arms:
                    started, wall_start = utc_now(), time.monotonic()
                    if protocol.controlled_study is not None:
                        with intent_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps({"episode_id": f"{manifest.scenario_id}__{arm_id}",
                                                     "session_id": session_id, "started_utc": started,
                                                     "protocol_hash": protocol.content_hash(),
                                                     "manifest_hash": binding.effective.content_hash()})+"\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                    try:
                        result = runner.run(manifest, arm_id, binding=binding)
                        episode_rows.append({
                            "episode_id": result.attempt.episode_id,
                            "scenario_id": manifest.scenario_id,
                            "arm_id": arm_id,
                            "status": result.attempt.status,
                            "termination": result.attempt.termination_reason,
                            "wall_clock_s": result.attempt.wall_clock_duration_s,
                        })
                    except (AdapterError, SceneError) as exc:
                        # The runner records adapter failures itself; reaching here means setup failed
                        # before an attempt existed, which still must stay visible.
                        errors.append({
                            "scenario_id": manifest.scenario_id, "arm_id": arm_id,
                            "error_type": type(exc).__name__, "error": str(exc),
                        })
                        _record_setup_failure(
                            writer, binding.effective, arm_id, exc, started, wall_start,
                            gate_payload.get("provenance", "unverified") if gate_payload else "unverified",
                        )
                    count += 1
    finally:
        adapter.close()

    attempts = load_attempted_runs(
        config.paths.attempted_runs_path(config.run_class, writer.protocol_short_hash)
    )
    if resume or protocol.controlled_study is not None:
        (writer.root / f"collection_session-{session_id}.json").write_text(
            json.dumps({"session_id": session_id, "resume": resume, "skipped_existing": resume_skipped,
                        "new_attempts": count, "protocol_hash": protocol.content_hash(),
                        "note": "Orphan evidence/start intents are retained unresolved, never rerun."},
                       sort_keys=True, indent=2)+"\n", encoding="utf-8")
    return {
        "run_dir": str(writer.root),
        "run_class": config.run_class,
        "protocol_hash": protocol.content_hash(),
        "cells": cells,
        "arms": arms,
        "realizations_per_cell": per_cell,
        "episodes_attempted": len(attempts),
        "setup_errors": errors,
        "episodes": episode_rows,
        "attempts": summarize_attempts(attempts),
        "live_gate_passed": None if skip_gate else True,
        "pilot_coverage": pilot_coverage,
        "resume_skipped": resume_skipped,
        "next_step": (
            f"colassure evaluate --run-dir {writer.root} && colassure audit --run-dir {writer.root} && "
            f"colassure analyze --run-dir {writer.root}"
        ),
    }
