"""Audit workflow: prespecified offline record ablation, independent reconstruction, and scoring.

Trajectories are fixed. Only the retained record changes. Any difference in what an auditor can establish
is therefore a property of the records, not of flight safety.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.scenario.manifest import ScenarioManifest
from colosseum_assurance.schemas import EpisodeRecord, PrivilegedLedger


def audit_run(run_dir: Path, protocol: ProtocolConfig | None = None) -> dict[str, Any]:
    """Score reconstruction for every completed episode in a run tree."""
    from colosseum_assurance.audit.reconstructor import reconstruct
    from colosseum_assurance.audit.records import build_variants
    from colosseum_assurance.audit.reference import reference_answers
    from colosseum_assurance.audit.scoring import aggregate, score_episode

    run_dir = Path(run_dir)
    protocol = protocol or ProtocolConfig()
    episodes_dir = run_dir / "episodes"
    ledgers_dir = run_dir / "privileged_ledgers"
    manifests_dir = run_dir / "manifests"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"{episodes_dir} not found: nothing to audit")

    scores: list[Any] = []
    skipped: list[dict[str, str]] = []
    audited_episodes = 0

    for episode_path in sorted(episodes_dir.glob("*.json")):
        record = EpisodeRecord.model_validate_json(episode_path.read_text(encoding="utf-8"))
        if not record.termination.reached_terminal_state:
            skipped.append({
                "episode_id": record.episode_id,
                "reason": f"incomplete episode ({record.termination.reason}); audit needs a finished record",
            })
            continue
        ledger_path = ledgers_dir / f"{record.episode_id}.json"
        manifest_path = manifests_dir / f"{record.scenario_id}.json"
        if not ledger_path.exists() or not manifest_path.exists():
            skipped.append({
                "episode_id": record.episode_id,
                "reason": f"missing {'ledger' if not ledger_path.exists() else 'manifest'} file",
            })
            continue
        ledger = PrivilegedLedger.model_validate_json(ledger_path.read_text(encoding="utf-8"))
        manifest = ScenarioManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

        reference = reference_answers(record, ledger, manifest, protocol)
        for variant in build_variants(record, protocol):
            reconstructed = reconstruct(variant, protocol)
            scores.append(
                score_episode(
                    reference,
                    reconstructed,
                    protocol,
                    episode_id=record.episode_id,
                    variant_id=variant.variant_id,
                )
            )
        audited_episodes += 1

    out_dir = run_dir / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "audit_scores.jsonl"
    with scores_path.open("w", encoding="utf-8") as fh:
        for score in scores:
            fh.write(json.dumps(_dump(score), sort_keys=True, default=str) + "\n")

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "episodes_audited": audited_episodes,
        "score_rows": len(scores),
        "skipped": skipped,
        "scores_path": str(scores_path),
        "caveat": (
            "Offline record ablation changes only what an auditor can establish. It cannot change flight "
            "safety, and reconstructability is claimed only within this record schema and procedure."
        ),
    }
    if scores:
        aggregated = _dump(aggregate(scores, protocol))
        summary["aggregate"] = aggregated
        # A silently empty audit is the defect this workflow was found to have, so the alert is lifted
        # into the printed summary instead of staying inside the aggregate object.
        if isinstance(aggregated, dict):
            summary["scored_units"] = aggregated.get("scored_units")
            summary["excluded_units"] = aggregated.get("excluded_units")
            summary["zero_scored_units"] = aggregated.get("zero_scored_units")
            if aggregated.get("scoring_alert"):
                summary["scoring_alert"] = aggregated["scoring_alert"]
    else:
        summary["scored_units"] = 0
        summary["zero_scored_units"] = True
        summary["scoring_alert"] = (
            "no audit units were scored: every episode was skipped or produced no comparable answer"
        )
    summary_path = out_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
                            encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _dump(obj: Any) -> Any:
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return obj
