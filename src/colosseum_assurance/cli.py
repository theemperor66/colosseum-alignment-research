"""Command line interface.

Commands are grouped by what they prove:

* ``doctor`` and ``live-gate`` inspect the *real* simulator. They cannot pass without genuine provenance.
* ``smoke`` is a fixture-only engineering workflow. Its output is conspicuously synthetic and can never
  be experimental evidence.
* ``freeze``, ``run``, ``evaluate``, ``analyze``, ``audit``, ``size`` operate on the study itself.

Run ``colassure --help`` for the list, or ``colassure <command> --help`` for one command.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tyro

from colosseum_assurance.config import AppConfig
from colosseum_assurance.protocol.freeze import (
    append_deviation,
    estimate_workload,
    freeze_protocol,
    load_frozen,
    paired_sample_size,
)
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.version import PACKAGE_VERSION, code_version


def _load_config(config: Path | None, host: str | None, port: int | None, **extra: Any) -> AppConfig:
    overrides: dict[str, Any] = {}
    endpoint: dict[str, Any] = {}
    if host is not None:
        endpoint["host"] = host
    if port is not None:
        endpoint["port"] = port
    if endpoint:
        overrides["endpoint"] = endpoint
    overrides.update({k: v for k, v in extra.items() if v is not None})
    return AppConfig.load(config, overrides=overrides)


def _load_protocol(protocol: Path | None) -> tuple[ProtocolConfig, dict[str, Any]]:
    if protocol is None:
        return ProtocolConfig(), {"label": "draft", "frozen": False}
    proto, meta = load_frozen(protocol)
    meta["frozen"] = True
    return proto, meta


def _run_protocol(run_dir: Path, protocol: Path | None) -> tuple[ProtocolConfig, dict[str, Any]]:
    """Load the protocol a run was produced under, refusing a mismatched supplied one.

    Post-processing that silently falls back to current defaults scores a run against a specification it
    never ran with, so the provenance of this choice is returned and printed with every result.
    """
    from colosseum_assurance.runtime.evidence import load_run_protocol

    supplied = None if protocol is None else _load_protocol(protocol)[0]
    return load_run_protocol(run_dir, supplied)


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


# --------------------------------------------------------------------------------------
# Simulator-facing commands
# --------------------------------------------------------------------------------------
def doctor(
    config: Path | None = None,
    host: str | None = None,
    port: int | None = None,
    protocol: Path | None = None,
    json_out: Path | None = None,
) -> None:
    """Diagnose configuration, connection, reset, control, and camera readiness, and explain failures."""
    from colosseum_assurance.sim.diagnostics import run_diagnostics

    cfg = _load_config(config, host, port)
    proto, _ = _load_protocol(protocol)
    report = run_diagnostics(cfg, proto)
    print(report.render_text())
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {json_out}")
    sys.exit(0 if report.ok else 2)


def live_gate(
    config: Path | None = None,
    host: str | None = None,
    port: int | None = None,
    protocol: Path | None = None,
    out: Path = Path("results/live_gate"),
) -> None:
    """Live readiness gate: genuine provenance, successful reset, changed trajectory, nonempty frames."""
    from colosseum_assurance.sim.diagnostics import run_live_readiness_gate

    cfg = _load_config(config, host, port)
    proto, _ = _load_protocol(protocol)
    report = run_live_readiness_gate(cfg, proto)
    print(report.render_text())
    out.mkdir(parents=True, exist_ok=True)
    path = out / report.suggested_filename()
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")
    sys.exit(0 if report.passed else 3)


# --------------------------------------------------------------------------------------
# Protocol commands
# --------------------------------------------------------------------------------------
def freeze(label: str = "pilot", out_dir: Path = Path("configs/frozen"), rationale: str = "") -> None:
    """Freeze the current protocol defaults to a hashed JSON file."""
    protocol = ProtocolConfig()
    path = freeze_protocol(protocol, label=label, out_dir=out_dir, rationale=rationale)
    proto, meta = load_frozen(path)
    _print_json({
        "frozen_path": str(path),
        "protocol_hash": meta["protocol_hash"],
        "cells": len(proto.cells()),
        "episode_budget_pilot": proto.episode_budget("pilot"),
        "episode_budget_heldout": proto.episode_budget("heldout"),
    })


def deviate(frozen: Path, description: str, rationale: str, applies_from: str = "") -> None:
    """Append a post-freeze deviation to a frozen protocol file."""
    append_deviation(frozen, description=description, rationale=rationale, applies_from=applies_from)
    print(f"recorded deviation in {frozen}")


def size(
    run_dir: Path,
    target_half_width: float = 0.10,
    parallel_capacity: int = 1,
    failure_rate: float = 0.05,
    protocol: Path | None = None,
) -> None:
    """Derive held-out sample size and workload from measured pilot outcomes."""
    from colosseum_assurance.analysis.metrics import load_outcomes
    from colosseum_assurance.runtime.evidence import load_attempted_runs

    proto, protocol_provenance = _run_protocol(run_dir, protocol)
    outcomes = load_outcomes(run_dir / "outcomes.jsonl")
    by_scenario: dict[str, dict[str, bool]] = {}
    for o in outcomes:
        if o.completeness != "complete":
            continue
        by_scenario.setdefault(o.scenario_id, {})[o.arm_id] = bool(o.physical_violation)
    b = c = pairs = 0
    for arms in by_scenario.values():
        if "A1_policy_only" in arms and "A2_assumption_aware" in arms:
            pairs += 1
            a1, a2 = arms["A1_policy_only"], arms["A2_assumption_aware"]
            b += int(a1 and not a2)
            c += int(a2 and not a1)
    attempts = load_attempted_runs(run_dir / "attempted_runs.jsonl")
    durations = [a.wall_clock_duration_s for a in attempts if a.wall_clock_duration_s]
    mean_wall = sum(durations) / len(durations) if durations else 0.0
    sizing = paired_sample_size(b, c, max(pairs, 1), target_half_width) if pairs else {
        "error": "no paired complete episodes found; cannot size the held-out study"
    }
    workload = estimate_workload(
        proto, "heldout", measured_episode_wall_clock_s=mean_wall,
        parallel_capacity=parallel_capacity, failure_rate=failure_rate,
    )
    _print_json({
        "pilot_dir": str(run_dir),
        "protocol_provenance": protocol_provenance,
        "paired_complete_scenarios": pairs,
        "measured_mean_episode_wall_clock_s": round(mean_wall, 3),
        "sizing": sizing,
        "workload": workload,
        "note": "sizing is a planning calculation from a small pilot, not a power guarantee",
    })


# --------------------------------------------------------------------------------------
# Execution commands
# --------------------------------------------------------------------------------------
def smoke(
    realizations: int = 1,
    cells: int = 2,
    results_root: Path = Path("results"),
    figures: bool = True,
    keep_frames: bool = False,
    fresh: bool = True,
) -> None:
    """Fixture-only end-to-end workflow. SYNTHETIC: never experimental evidence.

    ``--no-fresh`` keeps an existing fixture tree, which then refuses to re-run episodes it already
    recorded. Recorded evidence is never overwritten.
    """
    from colosseum_assurance.workflows.smoke import run_smoke

    summary = run_smoke(
        realizations=realizations,
        cells=cells,
        results_root=results_root,
        make_figures=figures,
        keep_frames=keep_frames,
        fresh=fresh,
    )
    _print_json(summary)


def run(
    run_class: str,
    protocol: Path,
    config: Path | None = None,
    host: str | None = None,
    port: int | None = None,
    realizations: int | None = None,
    cells: str | None = None,
    arms: str | None = None,
    results_root: Path = Path("results"),
    limit: int | None = None,
    resume: bool = False,
) -> None:
    """Execute experimental episodes against a live Colosseum simulator (pilot or heldout)."""
    from colosseum_assurance.workflows.experiment import run_experiment

    if run_class not in {"pilot", "heldout"}:
        raise SystemExit("run_class must be 'pilot' or 'heldout'; use 'smoke' for fixture runs")
    cfg = _load_config(
        config, host, port,
        run_class=run_class, require_live_simulator=True, allow_fixture_fake=False,
        protocol_path=protocol,
    )
    proto, meta = _load_protocol(protocol)
    summary = run_experiment(
        config=cfg,
        protocol=proto,
        protocol_meta=meta,
        realizations=realizations,
        cell_filter=[c for c in cells.split(",") if c] if cells else None,
        arm_filter=[a for a in arms.split(",") if a] if arms else None,
        results_root=results_root,
        limit=limit,
        resume=resume,
    )
    _print_json(summary)


def evaluate(run_dir: Path, protocol: Path | None = None) -> None:
    """Score a run with the independent evaluator (privileged ledger in, outcomes out).

    The protocol comes from the RUN, not from current defaults. Passing ``--protocol`` is allowed only
    when it matches the protocol the run was produced under.
    """
    from colosseum_assurance.evaluation.evaluator import evaluate_run

    proto, provenance = _run_protocol(run_dir, protocol)
    summary = evaluate_run(run_dir, protocol=proto)
    payload = summary if isinstance(summary, dict) else {"outcomes": len(summary)}
    payload["protocol_provenance"] = provenance
    _print_json(payload)


def analyze(run_dir: Path, protocol: Path | None = None, figures: bool = True) -> None:
    """Compute paired episode-level statistics, write the report, and render figures.

    The report names the protocol that actually scored the run, which is the run's own protocol.
    """
    from colosseum_assurance.workflows.analyze import analyze_run

    proto = None if protocol is None else _load_protocol(protocol)[0]
    _print_json(analyze_run(run_dir, protocol=proto, make_figures=figures))


def audit(run_dir: Path, protocol: Path | None = None) -> None:
    """Run the offline audit ablation and reconstruction scoring on completed episodes.

    The protocol comes from the run, so the frozen audit questions and ablations are the ones the run
    was produced under.
    """
    from colosseum_assurance.workflows.audit import audit_run

    proto, provenance = _run_protocol(run_dir, protocol)
    payload = audit_run(run_dir, protocol=proto)
    payload["protocol_provenance"] = provenance
    _print_json(payload)


def replay(run_dir: Path, episode_id: str | None = None, out: Path = Path("results/replay")) -> None:
    """Export a representative episode: trajectory table, camera frames, and a timeline summary."""
    from colosseum_assurance.workflows.replay import export_replay

    _print_json(export_replay(run_dir, episode_id=episode_id, out_dir=out))


def schemas(out: Path = Path("docs/schemas")) -> None:
    """Write JSON Schemas for every record type, so external tools can validate our evidence."""
    from colosseum_assurance.workflows.schema_export import export_schemas

    _print_json(export_schemas(out))


def attest(
    artifact: Path,
    out: Path = Path("configs/simulator-attestation.json"),
    provenance_class: str = "third_party_colosseum_build",
    source_kind: str = "third_party_publication",
    artifact_sha256: str | None = None,
    source_url: str | None = None,
    source_repo: str | None = None,
    source_revision: str | None = None,
    upstream_commit: str | None = None,
    engine_version: str | None = None,
    scene_package_path: str | None = None,
    expected_scene_signature: str | None = None,
    verification_manifest: Path | None = None,
    verified_by: str = "",
    note: str = "",
    caveat: list[str] | None = None,
) -> None:
    """Write the artifact attestation that anchors simulator provenance.

    Provenance cannot come from an RPC handshake: any AirSim-compatible server answers it. This command
    records the package that runs the simulator, its SHA-256, where it came from, and who verified it.
    Pass ``--artifact-sha256`` when the package lives on another machine and was hashed there.
    """
    from colosseum_assurance.schemas import SimulatorArtifactAttestation

    digest = artifact_sha256
    size = None
    if digest is None:
        if not artifact.is_file():
            raise SystemExit(
                f"{artifact} is not a local file. Hash the package where it lives and pass "
                "--artifact-sha256 <digest>."
            )
        sha = hashlib.sha256()
        with artifact.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                sha.update(block)
        digest = sha.hexdigest()
        size = artifact.stat().st_size

    manifest_sha = None
    if verification_manifest is not None and verification_manifest.is_file():
        manifest_sha = "sha256:" + hashlib.sha256(verification_manifest.read_bytes()).hexdigest()

    if not verified_by.strip():
        raise SystemExit("--verified-by is required: an attestation must name who verified the package")

    caveats = list(caveat or [])
    if provenance_class == "third_party_colosseum_build" and not caveats:
        caveats = ["the exact upstream Colosseum source commit of this package is unestablished"]

    attestation = SimulatorArtifactAttestation(
        provenance_class=provenance_class,  # type: ignore[arg-type]
        artifact_name=artifact.name,
        artifact_sha256=digest,
        artifact_size_bytes=size,
        source_kind=source_kind,  # type: ignore[arg-type]
        source_url=source_url,
        source_repo=source_repo,
        source_revision=source_revision,
        upstream_colosseum_commit=upstream_commit,
        engine_version_declared=engine_version,
        scene_package_path=scene_package_path,
        expected_scene_signature=expected_scene_signature,
        verification_manifest_path=str(verification_manifest) if verification_manifest else None,
        verification_manifest_sha256=manifest_sha,
        verified_by=verified_by,
        verified_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        verification_note=note,
        caveats=caveats,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(attestation.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _print_json({
        "wrote": str(out),
        "provenance_class": attestation.provenance_class,
        "artifact_sha256": attestation.artifact_sha256,
        "caveats": attestation.caveats,
        "reminder": (
            "point COLASSURE_SIM_ATTESTATION at this file; the live gate still requires a successful "
            "reset, a changed trajectory, and nonempty RGB and depth frames"
        ),
    })


def version() -> None:
    """Print package and code provenance."""
    _print_json({"package_version": PACKAGE_VERSION, "code_version": code_version(),
                 "default_protocol_hash": ProtocolConfig().content_hash()})


def family_suite(out: Path = Path("results/family-suite"), realizations: int = 1,
                 severity: float = 1.0, horizon_s: float = 45.0, figures: bool = False) -> None:
    """Run all five civilian families, control and treatment, as synthetic software verification."""
    from colosseum_assurance.workflows.expanded import run_family_suite

    _print_json(run_family_suite(results_root=out, realizations=realizations, severity=severity,
                                 horizon_s=horizon_s, figures=figures))


def freeze_family(family: str, severity: float, label: str = "pilot",
                  out: Path = Path("configs/frozen"), operator_queue: Path | None = None) -> None:
    """Freeze a v2 family condition for the ordinary gated live run workflow."""
    from colosseum_assurance.protocol.expanded import FAMILIES
    from colosseum_assurance.scenario.expanded import coverage_plan, expanded_protocol

    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    protocol = expanded_protocol(family, severity)
    if operator_queue is not None:
        data = protocol.model_dump(mode="json")
        data["study_extension"].update(supervision_mode="operator_queue",
                                       operator_queue_directory=str(operator_queue.resolve()))
        protocol = ProtocolConfig.model_validate(data)
    path = freeze_protocol(protocol, label=label, out_dir=out,
                           rationale="Civilian five-family extension; original v1 freezes preserved.")
    protocol, _ = load_frozen(path)
    _print_json({"frozen_path": str(path), "protocol_hash": protocol.content_hash(),
                 "pilot_coverage": coverage_plan(protocol, 6)})


def falsify(out: Path = Path("results/search"), budget: int = 8, seed: int = 1776,
            horizon_s: float = 45.0) -> None:
    """Bounded simulation-fault search on the fixture, labeled exploration and never held-out evidence."""
    from colosseum_assurance.workflows.expanded import search_fixture

    _print_json(search_fixture(output=out, budget=budget, seed=seed, horizon_s=horizon_s))


def falsify_live(config: Path, out: Path, budget: int = 8, seed: int = 1776,
                 horizon_s: float = 120.0) -> None:
    """Execute a bounded, independently scored civilian fault search through the genuine-live gate."""
    from colosseum_assurance.workflows.expanded import search_fixture

    cfg = AppConfig.load(config)
    _print_json(search_fixture(output=out, budget=budget, seed=seed, horizon_s=horizon_s, live_config=cfg))


def counterfactual(run_dir: Path, episode_id: str, step_index: int, out: Path,
                   config: Path | None = None, preflight_plan: Path | None = None) -> None:
    """Replay an executed prefix and measure one-step original-versus-hold physical consequences."""
    from colosseum_assurance.schemas import EpisodeRecord
    from colosseum_assurance.workflows.counterfactual import run_counterfactual

    if Path(episode_id).name != episode_id:
        raise ValueError("episode_id must be a single identifier")
    record = EpisodeRecord.model_validate_json((run_dir / "episodes" / f"{episode_id}.json").read_text())
    if record.simulator_identity.provenance == "fixture_fake":
        from colosseum_assurance.sim import fixture_fake_server

        with fixture_fake_server() as endpoint:
            result = run_counterfactual(run_dir, episode_id, step_index, AppConfig(endpoint=endpoint), out,
                                        preflight_plan=preflight_plan)
    else:
        cfg = AppConfig.load(config, overrides={"require_live_simulator": True, "allow_fixture_fake": False})
        result = run_counterfactual(run_dir, episode_id, step_index, cfg, out, preflight_plan=preflight_plan)
    _print_json(result)


def counterfactual_plan(run_dir: Path, out: Path, selector: str = "first_move_after_step1") -> None:
    """Compile all retained source candidates/refusals offline; no RPC or alternative consequence."""
    from colosseum_assurance.workflows.counterfactual_source import compile_replay_plan

    _print_json(compile_replay_plan(run_dir, out, selector=selector))


def perception_fit(dataset: Path, model: Path) -> None:
    """Fit an observation-only probabilistic asset-presence model on grouped training/calibration data."""
    from colosseum_assurance.perception_eval import fit_dataset

    artifact = fit_dataset(dataset, model)
    _print_json({"model_path": str(model), "model_hash": artifact.model_hash,
                 "artifact": artifact.model_dump(mode="json")})


def perception_evaluate(dataset: Path, model: Path, out: Path) -> None:
    """Evaluate frozen perception probabilities against independent labels on held-out scenario groups."""
    from colosseum_assurance.perception_eval import evaluate_dataset

    _print_json(evaluate_dataset(dataset, model, out))


def sitl_prepare(out: Path = Path("configs/px4-sitl"), px4_source: Path | None = None,
                 colosseum_binary: Path | None = None) -> None:
    """Write and validate a pinned local PX4 connection profile; report missing build prerequisites."""
    from colosseum_assurance.sim.sitl import (
        check_prerequisites,
        generate_px4_settings,
        pinned_source_manifest,
        validate_px4_settings,
    )

    if out.exists():
        raise FileExistsError(out)
    settings = generate_px4_settings()
    validate_px4_settings(settings)
    report = check_prerequisites(settings, px4_source=px4_source, colosseum_binary=colosseum_binary)
    out.mkdir(parents=True, exist_ok=False)
    for name, value in (("settings.json", settings), ("source-manifest.json", pinned_source_manifest()),
                        ("prerequisites.json", report)):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")
    _print_json({"output": str(out), "prerequisites": report, "live_handshake_verified": False})


def assurance_case(run_dir: Path, out: Path) -> None:
    """Export a hashed claim-to-evidence graph with unsupported and refuted claims visible."""
    from colosseum_assurance.workflows.assurance import export_assurance_case

    report = export_assurance_case(run_dir, out)
    _print_json({"output": str(out), "root_claim": report["nodes"][0],
                 "fixture_only": report["fixture_only"]})


def operator_respond(request: Path, decision: str, operator: str) -> None:
    """Grant or deny a pending local civilian inspection request, retaining the response for audit."""
    from colosseum_assurance.runtime.operator import submit_response

    _print_json({"response_file": str(submit_response(request, decision=decision, operator=operator))})


def perception_smoke(out: Path = Path("results/perception-smoke"), realizations: int = 18,
                      horizon_s: float = 45.0) -> None:
    """Verify camera capture, independent labels, grouped model fit and held-out evaluation on fixtures."""
    from colosseum_assurance.workflows.perception import perception_smoke as workflow

    _print_json(workflow(out=out, realizations=realizations, horizon_s=horizon_s))


def freeze_perception(out: Path = Path("configs/frozen"), realizations: int = 18) -> None:
    """Freeze scenario groups and split assignment before live pilot image capture."""
    from colosseum_assurance.workflows.perception import prepare_perception_protocol

    protocol = prepare_perception_protocol(run_class="pilot", realizations=realizations)
    path = freeze_protocol(protocol, label="perception-pilot", out_dir=out,
                           rationale="Scenario-group splits assigned before images or labels exist.")
    frozen, _ = load_frozen(path)
    _print_json({"frozen_path": str(path), "protocol_hash": frozen.content_hash(),
                 "scenario_groups": frozen.study_extension.perception_split_by_group})


def freeze_perception_guard(protocol: Path, model: Path, threshold: float,
                            out: Path = Path("configs/frozen")) -> None:
    """Bind a trained model, threshold and content hash into a new civilian capture-guard protocol."""
    from colosseum_assurance.perception_eval import load_model

    proto, _ = load_frozen(protocol)
    if proto.study_extension is None:
        raise ValueError("the confidence guard requires a version 2 family protocol")
    artifact = load_model(model)
    data = proto.model_dump(mode="json")
    data["study_extension"].update(capture_segmentation=True, perception_model_path=str(model.resolve()),
                                   perception_model_hash=artifact.model_hash,
                                   perception_guard_threshold=threshold)
    proto = ProtocolConfig.model_validate(data)
    path = freeze_protocol(proto, label="perception-guard", out_dir=out,
                           rationale="New optional civilian capture-confidence guard; not a v1 comparator.")
    _print_json({"frozen_path": str(path), "model_hash": artifact.model_hash,
                 "threshold": threshold, "model_evidence_class": artifact.evidence_class})


COMMANDS = {
    "doctor": doctor,
    "live-gate": live_gate,
    "attest": attest,
    "freeze": freeze,
    "deviate": deviate,
    "size": size,
    "smoke": smoke,
    "run": run,
    "evaluate": evaluate,
    "analyze": analyze,
    "audit": audit,
    "replay": replay,
    "schemas": schemas,
    "version": version,
    "family-suite": family_suite,
    "freeze-family": freeze_family,
    "falsify": falsify,
    "falsify-live": falsify_live,
    "counterfactual": counterfactual,
    "counterfactual-plan": counterfactual_plan,
    "perception-fit": perception_fit,
    "perception-evaluate": perception_evaluate,
    "sitl-prepare": sitl_prepare,
    "assurance-case": assurance_case,
    "operator-respond": operator_respond,
    "perception-smoke": perception_smoke,
    "freeze-perception": freeze_perception,
    "freeze-perception-guard": freeze_perception_guard,
}


def main() -> None:
    """Entry point for the ``colassure`` console script.

    Expected failures are printed as plain language with their remedy and a non-zero exit status. A
    stack trace is the right output for a bug, not for "this run is not allowed yet", and an operator
    reading a refusal should see why and what to do next.
    """
    from pydantic import ValidationError

    from colosseum_assurance.interfaces import AdapterError
    from colosseum_assurance.protocol.freeze import ProtocolIntegrityError
    from colosseum_assurance.runtime.evidence import (
        ProtocolMismatch,
        ProvenanceError,
        RunTreeConflict,
    )
    from colosseum_assurance.sim.scene import SceneError
    from colosseum_assurance.workflows.experiment import LiveGateFailure

    try:
        tyro.extras.subcommand_cli_from_dict(COMMANDS, prog="colassure")
    except ValidationError as exc:
        print("colassure: this configuration is not allowed.\n", file=sys.stderr)
        for error in exc.errors():
            message = str(error.get("msg", "")).removeprefix("Value error, ")
            print(f"  - {message}", file=sys.stderr)
        sys.exit(4)
    except LiveGateFailure as exc:
        print(f"colassure: the live readiness gate did not pass.\n\n{exc}", file=sys.stderr)
        print(
            "\n  No experimental episode was recorded. See docs/live-readiness-gate.md.",
            file=sys.stderr,
        )
        sys.exit(3)
    except SceneError as exc:
        print(f"colassure: the scene is not the scene this scenario needs.\n\n  {exc}", file=sys.stderr)
        if getattr(exc, "remedy", ""):
            print(f"\n  remedy: {exc.remedy}", file=sys.stderr)
        sys.exit(8)
    except (ProvenanceError, RunTreeConflict, ProtocolIntegrityError, ProtocolMismatch) as exc:
        print(f"colassure: refusing to continue.\n\n  {exc}", file=sys.stderr)
        sys.exit(5)
    except AdapterError as exc:
        print(f"colassure: the simulator could not be used.\n\n  {exc}", file=sys.stderr)
        if getattr(exc, "remedy", ""):
            print(f"\n  remedy: {exc.remedy}", file=sys.stderr)
        sys.exit(6)
    except FileNotFoundError as exc:
        print(f"colassure: a required file is missing.\n\n  {exc}", file=sys.stderr)
        sys.exit(7)
    except (FileExistsError, ValueError) as exc:
        print(f"colassure: refusing this operation.\n\n  {exc}", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":  # pragma: no cover
    main()
