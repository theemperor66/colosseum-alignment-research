"""Fixture-only smoke workflow: engineering validation, never experimental evidence.

This workflow runs the full software path -- scenario generation, closed-loop episodes, independent
evaluation, analysis, figures, and audit reconstruction -- against the local fixture fake simulator. It
exists so a developer can prove the plumbing works without a Colosseum server.

Everything it writes is marked synthetic: the run class is ``fixture``, the run tree receives a
``SYNTHETIC_FIXTURE_DATA.txt`` marker, and the analysis report carries a banner. The evidence writer
refuses to place fixture output in a pilot or held-out tree.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from colosseum_assurance.config import AppConfig, PathsConfig
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.runtime.episode import EpisodeRunner
from colosseum_assurance.runtime.evidence import EvidenceWriter, load_attempted_runs, summarize_attempts
from colosseum_assurance.scenario.manifest import build_manifest

MARKER_TEXT = """SYNTHETIC FIXTURE DATA - NOT EXPERIMENTAL EVIDENCE

Every episode in this directory was produced by the local fixture fake simulator in
colosseum_assurance.sim.fixture_fake. It exercises the software path only. It is not Colosseum, it is not a
measurement of any simulated world of research interest, and it must never appear in results, figures, or
text that describe the study's findings.
"""


def run_smoke(
    realizations: int = 1,
    cells: int = 2,
    results_root: Path = Path("results"),
    make_figures: bool = True,
    keep_frames: bool = False,
    horizon_s: float | None = 40.0,
    fresh: bool = True,
) -> dict[str, Any]:
    """Run the fixture end-to-end workflow and return a summary dictionary.

    ``fresh`` removes an existing FIXTURE tree for this protocol before starting. Recorded evidence is
    never overwritten, so repeating the workflow into a used tree is refused; a fixture tree holds no
    experimental evidence, so starting from a clean one is the honest way to repeat it. The deletion is
    hard-limited to the ``fixture`` run class, and an accumulating attempt ledger with fewer episode
    files than attempts is exactly the misleading artifact this avoids.
    """
    from colosseum_assurance.sim import build_adapter, fixture_fake_server

    protocol = ProtocolConfig()
    if horizon_s is not None:
        # A short horizon keeps the smoke workflow fast; it changes the protocol hash, which is exactly
        # what we want: fixture data must never share a hash with an experimental protocol.
        protocol = protocol.model_copy(
            update={"mission": protocol.mission.model_copy(update={"episode_horizon_s": horizon_s})},
            deep=True,
        )
    paths = PathsConfig(results_root=results_root)
    base_config = AppConfig(
        run_class="fixture", paths=paths, allow_fixture_fake=True, require_live_simulator=False
    )

    cell_ids = [c["cell_id"] for c in protocol.cells()][:cells]
    manifests = [
        build_manifest(protocol, "fixture", cell_id, i)
        for cell_id in cell_ids
        for i in range(realizations)
    ]

    if fresh:
        stale = paths.run_dir("fixture", protocol.short_hash)
        if stale.exists():
            if "fixture" not in stale.parts:
                raise RuntimeError(f"refusing to delete {stale}: it is not a fixture tree")
            shutil.rmtree(stale)
    writer = EvidenceWriter(
        paths=paths, run_class="fixture", protocol_hash=protocol.content_hash(), protocol=protocol
    )
    (writer.root / "SYNTHETIC_FIXTURE_DATA.txt").write_text(MARKER_TEXT, encoding="utf-8")

    episode_summaries: list[dict[str, Any]] = []
    with fixture_fake_server() as endpoint:
        config = base_config.model_copy(update={"endpoint": endpoint})
        adapter = build_adapter(config, protocol)
        try:
            runner = EpisodeRunner(adapter, protocol, config, writer=writer, save_frames=keep_frames)
            for manifest in manifests:
                # Bind the scene ONCE per scenario, after a reset, so every arm of the matched set flies
                # the same geometry and is briefed from it.
                binding = runner.bind_scenario(manifest)
                for arm_id in protocol.arms.arm_ids:
                    result = runner.run(manifest, arm_id, binding=binding)
                    episode_summaries.append(
                        {
                            "episode_id": result.attempt.episode_id,
                            "arm_id": arm_id,
                            "scenario_id": manifest.scenario_id,
                            "status": result.attempt.status,
                            "termination": result.attempt.termination_reason,
                            "sim_duration_s": result.attempt.sim_duration_s,
                            "wall_clock_s": result.attempt.wall_clock_duration_s,
                            "steps": len(result.record.steps) if result.record else 0,
                        }
                    )
        finally:
            adapter.close()

    run_dir = writer.root
    attempts = load_attempted_runs(paths.attempted_runs_path("fixture", writer.protocol_short_hash))
    summary: dict[str, Any] = {
        "banner": "SYNTHETIC FIXTURE DATA - NOT EXPERIMENTAL EVIDENCE",
        "run_dir": str(run_dir),
        "protocol_hash": protocol.content_hash(),
        "episodes": episode_summaries,
        "attempts": summarize_attempts(attempts),
    }

    from colosseum_assurance.evaluation.evaluator import evaluate_run
    from colosseum_assurance.workflows.analyze import analyze_run
    from colosseum_assurance.workflows.audit import audit_run

    summary["evaluation"] = evaluate_run(run_dir, protocol=protocol)
    summary["audit"] = audit_run(run_dir, protocol=protocol)
    summary["analysis"] = analyze_run(run_dir, protocol=protocol, make_figures=make_figures)
    (run_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return summary
