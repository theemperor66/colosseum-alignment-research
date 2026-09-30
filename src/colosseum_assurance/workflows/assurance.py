"""Machine-readable, GSN-style claim/obligation/scenario/evidence graph.

This is a research assurance case, never certification. File existence alone is
not treated as a successful test or as independent expert validation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from colosseum_assurance.evaluation.outcomes import EpisodeOutcome
from colosseum_assurance.runtime.evidence import load_attempted_runs, load_run_protocol
from colosseum_assurance.schemas import EpisodeRecord, PrivilegedLedger
from colosseum_assurance.version import code_version


def export_assurance_case(run_dir: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(out)
    protocol, provenance = load_run_protocol(run_dir)
    attempts = load_attempted_runs(run_dir / "attempted_runs.jsonl")
    path = run_dir / "outcomes.jsonl"
    outcomes = [EpisodeOutcome.model_validate_json(row).model_dump(mode="json")
                for row in path.read_text().splitlines() if row.strip()] if path.exists() else []
    if any(o["protocol_hash"] != protocol.content_hash() for o in outcomes):
        raise ValueError("outcome protocol does not match assurance-case protocol")
    attempts_by_id = {a.episode_id: a for a in attempts}
    outcomes_by_id = {o["episode_id"]: o for o in outcomes}
    if len(attempts_by_id) != len(attempts) or len(outcomes_by_id) != len(outcomes):
        raise ValueError("duplicate attempt or outcome identities cannot support an assurance claim")
    if set(outcomes_by_id) - set(attempts_by_id):
        raise ValueError("outcome has no corresponding recorded attempt")
    for episode_id, outcome in outcomes_by_id.items():
        attempt = attempts_by_id[episode_id]
        if (attempt.scenario_id != outcome["scenario_id"] or attempt.arm_id != outcome["arm_id"]
                or attempt.protocol_hash != outcome["protocol_hash"]):
            raise ValueError("outcome identity conflicts with attempted-run ledger")
    complete_pairing = (set(attempts_by_id) == set(outcomes_by_id) and bool(attempts)
                        and all(a.status == "completed" for a in attempts)
                        and all(o.get("completeness") == "complete" for o in outcomes))
    actual_provenances = {a.simulator_provenance for a in attempts}
    for attempt in attempts:
        for folder, model in (("episodes", EpisodeRecord), ("privileged_ledgers", PrivilegedLedger)):
            file = run_dir / folder / f"{attempt.episode_id}.json"
            if not file.exists():
                complete_pairing = False
                continue
            record = model.model_validate_json(file.read_text())
            actual_provenances.add(record.simulator_identity.provenance)
            if (record.episode_id != attempt.episode_id or record.scenario_id != attempt.scenario_id
                    or record.arm_id != attempt.arm_id or record.run_class != attempt.run_class
                    or record.protocol_hash != attempt.protocol_hash
                    or record.simulator_identity.provenance != attempt.simulator_provenance):
                raise ValueError("source record/ledger provenance or identity conflicts with attempted run")
    if actual_provenances - {"fixture_fake", "colosseum_build_verified", "third_party_colosseum_build"}:
        complete_pairing = False
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    fixture = "fixture_fake" in actual_provenances

    def evidence(node_id: str, file: Path) -> str:
        exists = file.is_file()
        nodes.append({"id": node_id, "type": "evidence", "path": str(file.resolve()),
                      "status": "retained" if exists else "missing",
                      "sha256": hashlib.sha256(file.read_bytes()).hexdigest() if exists else None})
        return node_id

    nodes.append({"id": "C0", "type": "claim", "statement": (
        "Retained evidence supports finite sampled-state conformance of these recorded civilian episodes"),
        "status": "pending_evaluation"})
    nodes.append({"id": "S0", "type": "strategy", "statement": (
        "Decompose into frozen obligations, all attempted scenario-arm episodes and independent verdicts")})
    edges.append({"from": "C0", "to": "S0", "relation": "supported_by"})
    evidence("E-protocol", run_dir / "protocol.json")
    evidence("E-attempts", run_dir / "attempted_runs.jsonl")
    evidence("E-outcomes", path)
    for target in ("E-protocol", "E-attempts", "E-outcomes"):
        edges.append({"from": "S0", "to": target, "relation": "uses"})
    seen_scenarios: set[str] = set()
    for attempt in attempts:
        sid = attempt.scenario_id
        if sid not in seen_scenarios:
            seen_scenarios.add(sid)
            nodes.append({"id": f"scenario:{sid}", "type": "scenario", "scenario_id": sid})
            mid = evidence(f"manifest:{sid}", run_dir / "manifests" / f"{sid}.json")
            edges.append({"from": f"scenario:{sid}", "to": mid, "relation": "defined_by"})
        eid = attempt.episode_id
        nodes.append({"id": f"episode:{eid}", "type": "attempt", "status": attempt.status,
                      "arm": attempt.arm_id, "termination": attempt.termination_reason})
        edges.append({"from": f"scenario:{sid}", "to": f"episode:{eid}", "relation": "exercised_by"})
        for kind, folder in (("record", "episodes"), ("ledger", "privileged_ledgers")):
            target = evidence(f"{kind}:{eid}", run_dir / folder / f"{eid}.json")
            edges.append({"from": f"episode:{eid}", "to": target, "relation": "retains"})
    claim_states = []
    for obligation in protocol.obligations.obligation_ids:
        verdicts = [o.get("obligations", {}).get(obligation, {}).get("verdict", "unknown") for o in outcomes]
        if "violation" in verdicts:
            status = "refuted_by_recorded_violation"
        elif not verdicts or not set(verdicts) <= {"pass", "not_applicable"} or not complete_pairing:
            status = "insufficient_evidence"
        elif any(n.get("status") == "missing" for n in nodes):
            status = "insufficient_evidence"
        else:
            status = "supported_fixture_only" if fixture else "supported_for_recorded_samples"
        claim_states.append(status)
        cid, oid = f"claim:{obligation}", f"obligation:{obligation}"
        nodes.append({"id": cid, "type": "claim", "statement": f"Conformance to {obligation}",
                      "status": status, "counts": {v: verdicts.count(v) for v in sorted(set(verdicts))}})
        nodes.append({"id": oid, "type": "policy_obligation", "obligation_id": obligation,
                      "policy_version": protocol.obligations.policy_version,
                      "protocol_hash": protocol.content_hash(),
                      "evaluator": "colosseum_assurance.evaluation.spec",
                      "tests": ["tests/unit/test_evaluator_boundaries.py",
                                "tests/unit/test_evaluator_semantics.py"],
                      "test_status": "definition_link_only; consult recorded verification run"})
        edges.extend([{"from": "S0", "to": cid, "relation": "decomposed_into"},
                      {"from": cid, "to": oid, "relation": "interpreted_by"},
                      {"from": oid, "to": "E-outcomes", "relation": "evaluated_in"}])
        edges.extend({"from": oid, "to": f"scenario:{sid}", "relation": "tested_in"}
                     for sid in sorted(seen_scenarios))
    nodes[0]["status"] = (
        "refuted_by_recorded_violation" if "refuted_by_recorded_violation" in claim_states
        else "insufficient_evidence" if "insufficient_evidence" in claim_states
        else "supported_fixture_only" if fixture else "supported_for_recorded_samples")
    for name in ("independent_expert_review", "human_factors_validation", "PX4_live_handshake",
                 "hardware_in_the_loop", "field_transfer", "certification"):
        nodes.append({"id": f"external:{name}", "type": "unresolved_obligation", "statement": name,
                      "status": "unsupported_external_validation"})
        edges.append({"from": "C0", "to": f"external:{name}", "relation": "does_not_establish"})
    report = {"graph_version": "1.0.0", "notation": "GSN-style typed directed graph",
              "protocol": provenance, "code_version": code_version(),
              "scope": "finite sampled simulator obligations; no normative or certification claim",
              "fixture_only": fixture, "nodes": nodes, "edges": edges}
    ids = {n["id"] for n in nodes}
    if len(ids) != len(nodes) or any(e["from"] not in ids or e["to"] not in ids for e in edges):
        raise ValueError("invalid assurance graph identity or dangling edge")
    out.mkdir(parents=True, exist_ok=False)
    (out / "assurance-case.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Civilian research assurance case", "", report["scope"], "",
             "**SYNTHETIC FIXTURE ONLY**" if fixture else "Recorded simulator evidence only", "",
             "| Claim | Status |", "| --- | --- |"]
    lines.extend(f"| {n['statement']} | {n['status']} |" for n in nodes if n["type"] in {
        "claim", "unresolved_obligation"})
    lines += ["", "The JSON graph binds claims to hashed protocol, manifest, episode, "
              "ledger and outcome files.",
              "Test paths identify specifications; they do not assert that those tests were executed.", ""]
    (out / "assurance-case.md").write_text("\n".join(lines))
    return report
