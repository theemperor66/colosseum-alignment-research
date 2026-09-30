"""Representative episode export: trajectory, verdict timeline, events, and camera frames.

Used to inspect embodiment and to include one concrete counterexample in the write-up. An exported episode
demonstrates the closed loop; it is not a substitute for the statistical evidence.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from colosseum_assurance.scenario.manifest import ScenarioManifest
from colosseum_assurance.schemas import EpisodeRecord, PrivilegedLedger


def export_replay(
    run_dir: Path,
    episode_id: str | None = None,
    out_dir: Path = Path("results/replay"),
    copy_frames: bool = True,
) -> dict[str, Any]:
    """Export one episode to a self-contained directory and return the written paths."""
    run_dir = Path(run_dir)
    episodes = sorted((run_dir / "episodes").glob("*.json"))
    if not episodes:
        raise FileNotFoundError(f"no episodes under {run_dir / 'episodes'}")
    if episode_id is None:
        chosen = _choose_representative(episodes)
    else:
        candidate = run_dir / "episodes" / f"{episode_id}.json"
        if not candidate.exists():
            raise FileNotFoundError(f"episode {episode_id} not found in {run_dir}")
        chosen = candidate

    record = EpisodeRecord.model_validate_json(chosen.read_text(encoding="utf-8"))
    ledger_path = run_dir / "privileged_ledgers" / f"{record.episode_id}.json"
    manifest_path = run_dir / "manifests" / f"{record.scenario_id}.json"
    ledger = (
        PrivilegedLedger.model_validate_json(ledger_path.read_text(encoding="utf-8"))
        if ledger_path.exists() else None
    )
    manifest = (
        ScenarioManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists() else None
    )

    target = Path(out_dir) / record.episode_id
    target.mkdir(parents=True, exist_ok=True)

    decisions_path = target / "decisions.csv"
    with decisions_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "step", "sim_time_s", "observed_x", "observed_y", "observed_z", "state_age_s",
            "depth_min_range_m", "target_visible", "supervision_link", "heartbeat_age_s",
            "authorization_status", "command", "command_target", "issued_by", "monitor_verdict",
            "intervention", "controller_phase",
        ])
        for step in record.steps:
            obs = step.observation
            state = obs.state
            depth = obs.depth
            cmd = step.executed_command or step.command
            writer.writerow([
                step.step_index, round(step.sim_time_s, 3),
                None if state is None else round(state.position.x, 3),
                None if state is None else round(state.position.y, 3),
                None if state is None else round(state.position.z, 3),
                None if obs.state_age_s is None else round(obs.state_age_s, 3),
                None if depth is None or depth.min_range_m is None else round(depth.min_range_m, 3),
                None if depth is None else depth.target_visible,
                obs.supervision.link_state,
                None if obs.supervision.heartbeat_age_s is None
                else round(obs.supervision.heartbeat_age_s, 3),
                obs.authorization.status.value,
                cmd.kind,
                None if cmd.target is None else f"({cmd.target.x:.2f},{cmd.target.y:.2f},{cmd.target.z:.2f})",
                cmd.issued_by,
                None if step.monitor_report is None else step.monitor_report.verdict.value,
                None if step.monitor_report is None else step.monitor_report.intervention,
                step.controller_state.get("phase"),
            ])

    truth_path = target / "true_trajectory.csv"
    if ledger is not None:
        with truth_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["sim_time_s", "x", "y", "z", "speed_mps", "collision_active",
                             "collision_object", "min_obstacle_clearance_m"])
            for s in ledger.samples:
                speed = (s.velocity.x**2 + s.velocity.y**2 + s.velocity.z**2) ** 0.5
                writer.writerow([
                    round(s.sim_time_s, 3), round(s.position.x, 3), round(s.position.y, 3),
                    round(s.position.z, 3), round(speed, 3), s.collision_active, s.collision_object,
                    None if s.min_obstacle_clearance_m is None else round(s.min_obstacle_clearance_m, 3),
                ])
        (target / "events.json").write_text(
            json.dumps([e.model_dump(mode="json") for e in ledger.events], indent=2) + "\n",
            encoding="utf-8",
        )

    frames_src = run_dir / "frames" / record.episode_id
    frames_copied = 0
    if copy_frames and frames_src.exists():
        frames_dst = target / "frames"
        frames_dst.mkdir(exist_ok=True)
        for f in sorted(frames_src.iterdir()):
            if f.is_file():
                shutil.copy2(f, frames_dst / f.name)
                frames_copied += 1

    summary = {
        "episode_id": record.episode_id,
        "scenario_id": record.scenario_id,
        "arm_id": record.arm_id,
        "run_class": record.run_class,
        "simulator_provenance": record.simulator_identity.provenance,
        "termination": record.termination.model_dump(mode="json"),
        "steps": len(record.steps),
        "interventions": record.interventions,
        "layout_variant": None if manifest is None else manifest.layout_variant,
        "visibility": None if manifest is None else manifest.visibility,
        "frames_copied": frames_copied,
        "paths": {
            "decisions_csv": str(decisions_path),
            "true_trajectory_csv": str(truth_path) if ledger is not None else None,
            "events_json": str(target / "events.json") if ledger is not None else None,
            "directory": str(target),
        },
        "note": (
            "true_trajectory.csv and events.json come from the privileged ledger. They are evaluator-side "
            "evidence and were not available to the controller or the guard at run time."
        ),
    }
    (target / "replay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return summary


def _choose_representative(episode_paths: list[Path]) -> Path:
    """Prefer a completed guarded episode with at least one intervention, else the first completed one."""
    best: Path | None = None
    for path in episode_paths:
        record = EpisodeRecord.model_validate_json(path.read_text(encoding="utf-8"))
        if not record.termination.reached_terminal_state:
            continue
        if record.interventions:
            return path
        if best is None:
            best = path
    return best or episode_paths[0]
