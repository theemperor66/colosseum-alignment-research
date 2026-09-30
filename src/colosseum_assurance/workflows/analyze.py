"""Analysis workflow: outcomes in, report and figures out.

Reads only saved artifacts, so the same command reproduces a report from stored data without touching a
simulator. Every reported number is tied to the run class, the protocol hash, and the code version that
produced the episodes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.runtime.evidence import (
    load_attempted_runs,
    load_run_protocol,
    summarize_attempts,
)


def analyze_run(
    run_dir: Path,
    protocol: ProtocolConfig | None = None,
    make_figures: bool = True,
) -> dict[str, Any]:
    """Compute metrics, write the report, and render figures for one run tree.

    Two inputs are read from the run tree rather than reconstructed here, because reconstructing them
    is how a report ends up describing a run that never happened:

    * the **protocol** comes from ``load_run_protocol``, so the report names the specification that
      actually scored the run and carries a warning when a fallback was used. ``protocol`` is a
      *supplied* protocol that must match, not a replacement;
    * the **attempted-run ledger** is passed on as typed records. It is the only evidence of a run that
      crashed before producing an outcome, so it - not the outcome file - defines every attempted
      denominator, and the scenario manifests place those attempts in their strata.
    """
    from colosseum_assurance.analysis.figures import render_all
    from colosseum_assurance.analysis.metrics import (
        compute_run_analysis,
        load_outcomes,
        load_scenario_strata,
    )
    from colosseum_assurance.analysis.report import write_report

    run_dir = Path(run_dir)
    outcomes_path = run_dir / "outcomes.jsonl"
    if not outcomes_path.exists():
        raise FileNotFoundError(
            f"{outcomes_path} not found. Run `colassure evaluate --run-dir {run_dir}` first: analysis "
            "never recomputes verdicts, it only reads the independent evaluator's output."
        )
    outcomes = load_outcomes(outcomes_path)
    metadata = _load_metadata(run_dir)
    attempts = load_attempted_runs(run_dir / "attempted_runs.jsonl")
    attempts_summary = summarize_attempts(attempts)
    scenario_strata = load_scenario_strata(run_dir)

    protocol, protocol_provenance = load_run_protocol(run_dir, protocol)
    analysis = compute_run_analysis(
        outcomes,
        protocol,
        run_class=str(metadata.get("run_class", "unknown")),
        protocol_hash=str(metadata.get("protocol_hash", protocol.content_hash())),
        attempted=attempts,
        scenario_strata=scenario_strata,
        protocol_provenance=protocol_provenance,
        # Provenance of the code that produced the episodes, taken from the run tree rather than from
        # the machine running the analysis: a report must name the version that generated the data.
        code_version=_code_version(run_dir, metadata),
    )

    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_summary = _load_audit(run_dir)
    report_paths = write_report(
        analysis,
        out_dir,
        audit=audit_summary,
        extra={
            "run_metadata": metadata,
            "attempts": attempts_summary,
            "protocol_provenance": protocol_provenance,
            "scenario_manifests_found": len(scenario_strata),
            "audit_summary_path": str(run_dir / "audit" / "audit_summary.json"),
        },
    )
    figure_paths: list[str] = []
    if make_figures:
        figure_paths = [str(p) for p in render_all(
            analysis, out_dir / "figures", audit=audit_summary, protocol=protocol)]

    return {
        "run_dir": str(run_dir),
        "run_class": metadata.get("run_class"),
        "protocol_hash": metadata.get("protocol_hash"),
        "protocol_provenance": protocol_provenance,
        "episodes_analyzed": len(outcomes),
        "attempts": attempts_summary,
        "attempts_without_outcome": max(0, len(attempts) - len(outcomes)),
        "arms_without_outcomes": list(analysis.arms_without_outcomes),
        "report": report_paths,
        "figures": figure_paths,
    }


def _code_version(run_dir: Path, metadata: dict[str, Any]) -> dict[str, str]:
    """Code provenance recorded when the episodes ran, not when the analysis ran."""
    recorded = metadata.get("code_version")
    if isinstance(recorded, dict) and recorded:
        return {str(k): str(v) for k, v in recorded.items()}
    for path in sorted((run_dir / "episodes").glob("*.json"))[:1]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        episode_version = payload.get("code_version")
        if isinstance(episode_version, dict) and episode_version:
            return {str(k): str(v) for k, v in episode_version.items()}
    return {}


def _load_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_audit(run_dir: Path) -> Any:
    from colosseum_assurance.audit.scoring import AuditScoreSummary

    path = run_dir / "audit" / "audit_summary.json"
    if not path.exists():
        return None
    envelope = json.loads(path.read_text(encoding="utf-8"))
    aggregate = envelope.get("aggregate")
    return None if aggregate is None else AuditScoreSummary.model_validate(aggregate)
