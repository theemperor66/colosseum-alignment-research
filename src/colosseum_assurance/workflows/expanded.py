"""Executable fixture family suite and bounded exploration search.

Live trials use the same frozen v2 protocols through the normal gated `run`
workflow. Exploration artifacts can never be relabeled held-out evidence.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from colosseum_assurance.config import AppConfig, PathsConfig
from colosseum_assurance.protocol.expanded import FAMILIES
from colosseum_assurance.runtime.episode import EpisodeRunner
from colosseum_assurance.runtime.evidence import EvidenceWriter, utc_now
from colosseum_assurance.scenario.expanded import (
    build_expanded_manifest,
    coverage_plan,
    expanded_protocol,
)


def run_family_fixture(protocol: Any, *, results_root: Path, realizations: int = 1,
                       arms: list[str] | None = None, figures: bool = False) -> dict[str, Any]:
    from colosseum_assurance.evaluation.evaluator import evaluate_run
    from colosseum_assurance.sim import build_adapter, fixture_fake_server
    from colosseum_assurance.workflows.analyze import analyze_run
    from colosseum_assurance.workflows.audit import audit_run
    from colosseum_assurance.workflows.smoke import MARKER_TEXT

    paths = PathsConfig(results_root=results_root)
    root = paths.run_dir("fixture", protocol.short_hash)
    if root.exists():
        raise FileExistsError(f"refusing to replace {root}; choose a fresh output directory")
    selected = arms or protocol.arms.arm_ids
    for name in selected:
        protocol.arms.get(name)
    writer = EvidenceWriter(paths, "fixture", protocol.content_hash(), protocol=protocol)
    (writer.root / "SYNTHETIC_FIXTURE_DATA.txt").write_text(MARKER_TEXT)
    episodes = []
    with fixture_fake_server() as endpoint:
        config = AppConfig(endpoint=endpoint, paths=paths)
        adapter = build_adapter(config, protocol)
        try:
            runner = EpisodeRunner(adapter, protocol, config, writer=writer, save_frames=False)
            for i in range(realizations):
                manifest = build_expanded_manifest(protocol, "fixture", i)
                started, wall_start = utc_now(), time.monotonic()
                try:
                    binding = runner.bind_scenario(manifest)
                except Exception as exc:
                    from colosseum_assurance.workflows.experiment import _record_setup_failure

                    for arm_id in selected:
                        attempt = _record_setup_failure(writer, manifest, arm_id, exc, started,
                                                         wall_start, "fixture_fake")
                        episodes.append({"episode_id": attempt.episode_id, "status": attempt.status,
                                         "termination": attempt.termination_reason})
                    continue
                for arm_id in selected:
                    started, wall_start = utc_now(), time.monotonic()
                    try:
                        result = runner.run(manifest, arm_id, binding=binding)
                        attempt = result.attempt
                    except Exception as exc:
                        from colosseum_assurance.workflows.experiment import _record_setup_failure

                        attempt = _record_setup_failure(writer, binding.effective, arm_id, exc, started,
                                                         wall_start, "fixture_fake")
                    episodes.append({"episode_id": attempt.episode_id, "status": attempt.status,
                                     "termination": attempt.termination_reason})
        finally:
            adapter.close()
    evaluate_run(writer.root, protocol)
    audit_run(writer.root, protocol)
    analysis = analyze_run(writer.root, protocol, make_figures=figures)
    return {"run_dir": str(writer.root), "protocol_hash": protocol.content_hash(),
            "extension": protocol.study_extension.model_dump(), "episodes": episodes,
            "domain_coverage": coverage_plan(protocol, realizations), "analysis": analysis,
            "provenance": "fixture_fake", "experimental_evidence": False}


def run_family_suite(*, results_root: Path, realizations: int = 1, severity: float = 1.0,
                     horizon_s: float = 45.0, figures: bool = False) -> dict[str, Any]:
    if results_root.exists() and any(results_root.iterdir()):
        raise FileExistsError("suite output must be new or empty")
    if not 0 < severity <= 1 or realizations < 1:
        raise ValueError("severity must be in (0,1] and realizations positive")
    results_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"suite_version": "2.0.0", "experimental_evidence": False,
                               "banner": "SYNTHETIC FIXTURE SOFTWARE TESTS", "conditions": []}
    for family in FAMILIES:
        for level in (0.0, severity):
            protocol = expanded_protocol(family, level, horizon_s=horizon_s)
            condition = run_family_fixture(protocol, results_root=results_root,
                                           realizations=realizations, figures=figures)
            summary["conditions"].append(condition)
            # Persist completed conditions incrementally; interrupted suites remain visible.
            (results_root / "suite-summary.json").write_text(
                json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def _physical_search_evidence(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep an observed violation even when later evidence is incomplete.

    A finite observed violation disproves the physical obligation without needing
    a completed trial. In contrast, an incomplete no-event prefix cannot establish
    absence over the required horizon. Completion is a separate recorded property.
    """
    values: list[bool | None] = []
    for outcome in outcomes:
        verdict = outcome.get("physical_verdict")
        violation = outcome.get("physical_violation")
        if verdict == "violation" and violation is True:
            values.append(True)
        elif (outcome.get("completeness") == "complete"
              and verdict in {"pass", "not_applicable"} and violation is False):
            values.append(False)
        else:
            # Missing or internally inconsistent fields are never negative evidence.
            values.append(None)
    observed = (True if any(value is True for value in values) else
                False if values and all(value is False for value in values) else None)
    return {
        "observed_counterexample": observed,
        "independent_failure": observed,
        "complete_trial": bool(outcomes) and all(
            outcome.get("completeness") == "complete" for outcome in outcomes),
        "outcome_count": len(outcomes),
        "known_violation_outcomes": sum(value is True for value in values),
        "known_no_violation_outcomes": sum(value is False for value in values),
        "unresolved_outcomes": sum(value is None for value in values),
    }


def search_fixture(*, output: Path, budget: int = 8, seed: int = 1776,
                   horizon_s: float = 45.0, live_config: AppConfig | None = None) -> dict[str, Any]:
    """Seeded random falsification over bounded packet bias/occlusion treatments.

    The independent objective is whether any frozen physical obligation is
    violated. A known violation remains a counterexample if the run later ends
    incomplete; an incomplete no-event prefix remains unknown. Every candidate
    and all attempts are retained. Fixture execution validates software only.
    """
    if budget < 1 or budget > 100:
        raise ValueError("search budget must be between 1 and 100")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("search output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    candidates = [{"index": i, "family": ("simulated_manipulation" if i % 2 else "degraded_perception"),
                   "severity": float(rng.uniform(0, 1))} for i in range(budget)]
    contract = {"version": "2.1.0", "role": "exploration", "seed": seed,
                "budget": budget, "algorithm": "seeded_uniform_random_search",
                "objective": "independently_scored_physical_violation", "candidates": candidates,
                "objective_semantics": {
                    "positive": "observed physical violation, including incomplete trials",
                    "negative": "all outcomes complete with ascertainable no-violation verdicts",
                    "unknown": "no known violation and incomplete, missing, or unresolved evidence",
                    "completion": "reported separately; a counterexample is not a successful mission",
                },
                "provenance": "fixture_fake" if live_config is None else "live_gate_required",
                "experimental_evidence": False,
                "evidence_role": "software_validation" if live_config is None else "exploratory_live"}
    (output / "search-plan.json").write_text(json.dumps(contract, indent=2) + "\n")
    rows = []
    for candidate in candidates:
        row: dict[str, Any] = {**candidate, "status": "started",
                              **_physical_search_evidence([])}
        try:
            protocol = expanded_protocol(candidate["family"], candidate["severity"],
                                         horizon_s=horizon_s, exploration=True)
            candidate_dir = output / f"candidate-{candidate['index']:03d}"
            if live_config is None:
                result = run_family_fixture(protocol, results_root=candidate_dir, arms=["A0_unguarded"])
            else:
                from colosseum_assurance.evaluation.evaluator import evaluate_run
                from colosseum_assurance.protocol.freeze import freeze_protocol, load_frozen
                from colosseum_assurance.workflows.experiment import run_experiment

                frozen_path = freeze_protocol(protocol, "exploration", candidate_dir / "frozen",
                                               rationale="Prescheduled bounded search candidate")
                protocol, meta = load_frozen(frozen_path)
                meta["frozen"] = True
                config_data = live_config.model_dump(mode="json")
                config_data.update(run_class="pilot", protocol_path=str(frozen_path),
                                   require_live_simulator=True, allow_fixture_fake=False)
                cfg = AppConfig.model_validate(config_data)
                result = run_experiment(cfg, protocol, meta, realizations=1,
                                         arm_filter=["A0_unguarded"], results_root=candidate_dir, limit=1)
                evaluate_run(Path(result["run_dir"]), protocol)
            outcomes = [json.loads(line) for line in (Path(result["run_dir"]) / "outcomes.jsonl").read_text(
                ).splitlines() if line.strip()]
            row.update(status="evaluated", run_dir=result["run_dir"],
                       **_physical_search_evidence(outcomes))
        except Exception as exc:
            row.update(status="technical_failure", error=f"{type(exc).__name__}: {exc}"[:500])
        rows.append(row)
        with (output / "search-attempts.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
    report = {**contract, "attempts": rows, "failures_found": sum(
        r["independent_failure"] is True for r in rows),
        "complete_trials": sum(r["complete_trial"] for r in rows),
        "counterexamples_in_incomplete_trials": sum(
            r["observed_counterexample"] is True and not r["complete_trial"] for r in rows),
        "unknown_candidates": sum(r["observed_counterexample"] is None for r in rows),
        "known_no_violation_candidates": sum(
            r["observed_counterexample"] is False for r in rows),
        "note": "Exploration only. Counterexamples require live replication and fresh held-out tests."}
    (output / "search-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
