"""Bounded original-versus-hold simulator replay, always a separate exploratory artifact.

Saved proposals are not counterfactual outcomes. This workflow actually resets and replays the entire
executed command prefix twice, checks physical start equivalence, then measures one control step.
It is neither a human causal explanation nor evidence about unobserved longer-term consequences.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from colosseum_assurance.config import AppConfig
from colosseum_assurance.evaluation.measurements import outside_envelope
from colosseum_assurance.evaluation.spec import collision_instances
from colosseum_assurance.runtime.episode import EpisodeRunner, _GuardState
from colosseum_assurance.runtime.evidence import utc_now
from colosseum_assurance.runtime.supervision import AuthorizationBroker
from colosseum_assurance.schemas import ControlCommand
from colosseum_assurance.version import code_version
from colosseum_assurance.workflows.counterfactual_source import (
    ReplayRefused,
    prepare_source,
    verify_compiled_plan,
)
from colosseum_assurance.workflows.counterfactual_source import (
    package_digest as _source_digest,
)
from colosseum_assurance.workflows.counterfactual_source import (
    truth_at as _truth_at,
)


def _difference(actual, expected, tolerances):
    yaw = abs((actual.yaw_rad - expected.yaw_rad + math.pi) % (2 * math.pi) - math.pi)
    values = {
        "position_m": actual.position.distance_to(expected.position),
        "velocity_mps": actual.velocity.distance_to(expected.velocity),
        "yaw_rad": yaw, "time_s": abs(actual.sim_time_s - expected.sim_time_s),
        "landed_matches": actual.landed == expected.landed,
        "collision_count_matches": actual.collision_count == expected.collision_count,
    }
    values["within_tolerance"] = (
        all(values[k] <= limit for k, limit in tolerances.items())
        and values["landed_matches"] and values["collision_count_matches"]
    )
    return values


def _consequence(samples, ledger, manifest, protocol):
    geofence = [outside_envelope(s, manifest, protocol) for s in samples]
    # Samples contain the initial cumulative collision count, which is excluded as pre-existing.
    branch_ledger = ledger.model_copy(update={"samples": samples, "events": []})
    contacts = [c for c in collision_instances(branch_ledger, manifest, protocol)
                if c.sim_time_s > samples[0].sim_time_s + 1e-9 and not c.exempt]
    width = sum((right.sim_time_s - left.sim_time_s) * flag
                for left, right, flag in zip(samples, samples[1:], geofence, strict=False))
    return {
        "geofence_violation_observed": any(geofence), "sample_held_outside_geofence_s": width,
        "new_nonexempt_collision_count": len(contacts),
        "sample_count": len(samples),
        "observed_duration_s": samples[-1].sim_time_s - samples[0].sim_time_s,
        "endpoint_position": samples[-1].position.model_dump(mode="json"),
        "interpretation": "Sampled one-step physical consequences only; no mission or authorization verdict.",
    }


def run_counterfactual(
    run_dir: Path | str, episode_id: str, step_index: int, config: AppConfig, out_dir: Path | str,
    *, position_tolerance_m: float = 0.05, velocity_tolerance_mps: float = 0.05,
    yaw_tolerance_rad: float = 0.01, time_tolerance_s: float = 0.01,
    preflight_plan: Path | str | None = None,
) -> dict[str, Any]:
    """Replay a recorded decision against a hold alternative, retaining failures and branch traces.

    The output directory must be new and outside the source run. Held-out inputs keep their original
    role; this derived, selected replay is always exploratory and never a new held-out observation.
    For live sources, the standard live gate and exact simulator-artifact match are mandatory.
    """
    from colosseum_assurance.sim import build_adapter
    from colosseum_assurance.sim.diagnostics import run_live_readiness_gate

    source, target = Path(run_dir).resolve(), Path(out_dir).resolve()
    if target == source or source in target.parents:
        raise ValueError("counterfactual output must be outside the source evidence run")
    target.mkdir(parents=True, exist_ok=False)
    tolerances = {"position_m": position_tolerance_m, "velocity_mps": velocity_tolerance_mps,
                  "yaw_rad": yaw_tolerance_rad, "time_s": time_tolerance_s}
    result: dict[str, Any] = {
        "schema_version": "2.0.0", "method": "executed_prefix_one_step_original_vs_hold",
        "status": "refused", "source_run": str(source), "source_episode_id": episode_id,
        "step_index": step_index, "analysis_role": "exploration", "created_utc": utc_now(),
        "code_version": code_version(), "source_sha256": _source_digest(), "tolerances": tolerances,
        "branches": {}, "eligibility_checks": [],
        "interpretation": (
            "Bounded simulator contrast conditional on matched recorded physical states and an executed "
            "prefix. Not a human causal explanation, a full-state equivalence proof, a mission-safety "
            "claim, or a confirmatory held-out observation. Hidden simulator state may differ."
        ),
    }
    adapter = None
    try:
        protocol, provenance, record, ledger, manifest, contract = prepare_source(
            source, episode_id, step_index, tolerances)
        result["source_preflight"] = contract
        if preflight_plan is not None:
            result["compiled_preflight_plan"] = verify_compiled_plan(preflight_plan, source, contract)
        result["source_input_sha256"] = {
            name: hashlib.sha256(model.model_dump_json().encode()).hexdigest()
            for name, model in (("record", record), ("ledger", ledger), ("manifest", manifest))
        }
        result.update(source_run_class=record.run_class, protocol_hash=protocol.content_hash(),
                      source_code_version=record.code_version, protocol_provenance=provenance,
                      source_simulator_identity=record.simulator_identity.model_dump(mode="json"),
                      manifest_hash=manifest.content_hash())
        for name, model in (("protocol", protocol), ("manifest", manifest)):
            (target / f"{name}.json").write_text(model.model_dump_json(indent=2) + "\n")
        prefix = record.steps[:step_index + 1]
        selected = prefix[-1]
        result["executed_prefix"] = [s.executed_command.model_dump(mode="json") for s in prefix]
        dt = protocol.mission.control_dt_s
        expected_start = _truth_at(ledger, selected.sim_time_s, time_tolerance_s)
        expected_end = _truth_at(ledger, selected.sim_time_s + dt, time_tolerance_s)
        # Do not infer old deployment settings from a source record or downgrade a live source.
        live = record.simulator_identity.is_live
        if record.run_class in {"pilot", "heldout"} and not live:
            raise ReplayRefused("experimental source has no anchored live simulator identity")
        if live:
            if not config.require_live_simulator or config.allow_fixture_fake:
                raise ReplayRefused("live source requires a live-only replay configuration")
            gate = run_live_readiness_gate(config, protocol)
            result["live_gate"] = gate.model_dump(mode="json")
            if not gate.passed:
                raise ReplayRefused("live readiness gate failed")
        elif record.simulator_identity.provenance != "fixture_fake":
            raise ReplayRefused("source simulator provenance is unverified")
        adapter = build_adapter(config, protocol)
        identity = adapter.identity()
        if identity.provenance != record.simulator_identity.provenance:
            raise ReplayRefused("source and replay simulator provenance differ")
        if live and (identity.artifact is None or record.simulator_identity.artifact is None
                     or identity.artifact.artifact_sha256
                     != record.simulator_identity.artifact.artifact_sha256):
            raise ReplayRefused("source and replay simulator artifact hashes differ")
        result["replay_simulator_identity"] = identity.model_dump(mode="json")
        result["eligibility_checks"].append("source identities, executed prefix and truth endpoints verified")
        branch_starts = []
        for branch in ("original", "hold"):
            events, prefix_trace = [], []
            runner = EpisodeRunner(adapter, protocol, config, save_frames=False)
            adapter.reset()
            adapter.wait_until_ready()
            runner._time_origin_s = float(adapter.sim_time_s())
            reapply = getattr(adapter, "reapply_environment", None)
            if callable(reapply):
                reapply(manifest)
            configure = getattr(adapter, "configure_scene", None)
            if callable(configure):
                verified = runner._effective_manifest(manifest, dict(configure(manifest)))
                if verified.content_hash() != manifest.content_hash():
                    raise ReplayRefused("replayed scene does not match the saved effective manifest")
            adapter.set_start_pose(manifest.start_position, manifest.start_yaw_rad)
            adapter.acquire_control()
            adapter.arm()
            _, reached = runner._run_takeoff(protocol.mission.cruise_altitude_m, events)
            if not reached:
                raise ReplayRefused("replay takeoff did not reach the production setup criterion")
            broker = AuthorizationBroker(manifest.schedules, protocol.obligations)
            for step in prefix:
                actual = runner._rebase_truth(adapter.sample_truth())
                expected = _truth_at(ledger, step.sim_time_s, time_tolerance_s)
                check = _difference(actual, expected, tolerances)
                prefix_trace.append({"step_index": step.step_index, "state_check": check,
                                     "actual_truth": actual.model_dump(mode="json"),
                                     "expected_truth": expected.model_dump(mode="json")})
                result["branches"][branch] = {
                    "prefix_trace": prefix_trace,
                    "events": [e.model_dump(mode="json") for e in events], "samples": [],
                }
                if not check["within_tolerance"]:
                    raise ReplayRefused(f"{branch} prefix state mismatch at step {step.step_index}")
                if step.step_index == step_index:
                    break
                events.extend(broker.advance_to(runner._now()))
                runner._execute(step.executed_command, _GuardState(), broker, manifest, runner._now(),
                                step.step_index, events, {})
                runner._advance(dt, events)
            start = runner._rebase_truth(adapter.sample_truth())
            branch_starts.append(start)
            command = selected.executed_command if branch == "original" else ControlCommand(
                step_index=step_index, issued_sim_time_s=start.sim_time_s, kind="hold", duration_s=dt,
                issued_by="guard", reason="bounded exploratory counterfactual alternative",
            )
            events.extend(broker.advance_to(runner._now()))
            runner._execute(command, _GuardState(), broker, manifest, runner._now(), step_index, events, {})
            samples = [start, *runner._advance(dt, events)]
            branch_result = result["branches"][branch]
            branch_result.update(command=command.model_dump(mode="json"),
                                 samples=[s.model_dump(mode="json") for s in samples],
                                 events=[e.model_dump(mode="json") for e in events],
                                 start_match=_difference(start, expected_start, tolerances))
            gaps = [b.sim_time_s - a.sim_time_s for a, b in zip(samples, samples[1:], strict=False)]
            if (not gaps or any(g <= 0 or g > protocol.simulation.max_permitted_truth_gap_s for g in gaps)
                    or abs(samples[-1].sim_time_s - start.sim_time_s - dt) > time_tolerance_s):
                raise ReplayRefused(f"{branch} truth samples do not cover exactly one control step")
            branch_result["consequence"] = _consequence(samples, ledger, manifest, protocol)
            runner._safe_teardown(events)
            if branch == "original":
                match = _difference(samples[-1], expected_end, tolerances)
                branch_result["source_endpoint_match"] = match
                if not match["within_tolerance"]:
                    raise ReplayRefused("original replay endpoint differs from recorded source consequence")
        match = _difference(branch_starts[0], branch_starts[1], tolerances)
        result["between_branch_start_match"] = match
        if not match["within_tolerance"]:
            raise ReplayRefused("original and alternative do not share a matched physical start")
        result["status"] = "completed"
        result["eligibility_checks"].append("both full prefixes, branch starts and original endpoint matched")
    except Exception as exc:  # noqa: BLE001 - a failed replay is retained, never promoted to an outcome
        result["refusal_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if adapter is not None:
            try:
                adapter.release_control()
                adapter.close()
            except Exception as exc:  # noqa: BLE001 - teardown cannot erase replay diagnostics
                result["teardown_error"] = f"{type(exc).__name__}: {exc}"
        (target / "counterfactual.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
