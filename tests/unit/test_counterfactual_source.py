"""No-RPC source compiler tests; source trajectories are synthetic fixtures only."""

import copy
import json

import pytest
from tests.unit.test_counterfactual import saved_episode as source_fixture

from colosseum_assurance.schemas import SimulatorArtifactAttestation, SimulatorIdentity
from colosseum_assurance.workflows.counterfactual import run_counterfactual
from colosseum_assurance.workflows.counterfactual_source import (
    DEFAULT_TOLERANCES,
    compile_replay_plan,
    prepare_source,
    verify_compiled_plan,
)

saved_episode = source_fixture


def no_simulator(monkeypatch):
    import colosseum_assurance.sim
    import colosseum_assurance.sim.diagnostics

    calls = []

    def refuse(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("source eligibility must not contact a simulator")

    monkeypatch.setattr(colosseum_assurance.sim, "build_adapter", refuse)
    monkeypatch.setattr(colosseum_assurance.sim.diagnostics, "run_live_readiness_gate", refuse)
    return calls


def rewrite(path, mutate):
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value))


def test_compiles_hash_bound_first_move_without_rpc_or_outcome_read(saved_episode, monkeypatch):
    root, record, step, _, tmp = saved_episode
    calls = no_simulator(monkeypatch)
    (root / "outcomes.jsonl").write_text("must never parse this as an outcome")
    path = tmp / "plan.json"
    report = compile_replay_plan(root, path)
    assert not calls
    assert report["new_flights"] == report["alternative_consequences_measured"] == 0
    assert report["retained_source_identities"] == report["eligible_sources"] == 1
    assert report["refused_sources"] == 0
    candidate = report["candidates"][0]
    assert candidate["step_index"] == step
    assert candidate["tolerances"] == DEFAULT_TOLERANCES
    assert candidate["source_code_version"] == record.code_version
    assert len(candidate["source_input_files"]) == 4
    *_, contract = prepare_source(root, record.episode_id, step)
    assert verify_compiled_plan(path, root, contract)["sha256"]
    with pytest.raises(FileExistsError):
        compile_replay_plan(root, path)


def test_compiled_plan_can_drive_real_fixture_prefix_replay(saved_episode):
    root, record, step, config, tmp = saved_episode
    path = tmp / "plan.json"
    compile_replay_plan(root, path)
    result = run_counterfactual(root, record.episode_id, step, config, tmp / "replay", preflight_plan=path)
    assert result["status"] == "completed", result.get("refusal_reason")
    assert result["source_preflight"]["alternative_consequences_measured"] is False
    assert result["compiled_preflight_plan"]["sha256"]
    assert result["between_branch_start_match"]["within_tolerance"]


@pytest.mark.parametrize(
    "mutation",
    ["same_bytes_identity_tamper", "missing_ledger", "ordering", "missing_command", "command_time"],
)
def test_invalid_source_refused_before_any_simulator_call(saved_episode, monkeypatch, mutation):
    root, record, step, config, tmp = saved_episode
    path = tmp / "plan.json"
    compile_replay_plan(root, path)
    ep = root / "episodes" / (record.episode_id + ".json")
    if mutation == "same_bytes_identity_tamper":
        # A harmless textual change is enough: the reviewed input bytes have changed.
        ep.write_bytes(ep.read_bytes() + b"\n")
    elif mutation == "missing_ledger":
        (root / "privileged_ledgers" / ep.name).unlink()
    elif mutation == "ordering":
        rewrite(ep, lambda d: d["steps"].__setitem__(slice(0, 2), list(reversed(d["steps"][:2]))))
    elif mutation == "command_time":
        rewrite(ep, lambda d: d["steps"][0]["executed_command"].update(issued_sim_time_s=1000))
    else:
        rewrite(ep, lambda d: d["steps"][0].update(executed_command=None))
    calls = no_simulator(monkeypatch)
    result = run_counterfactual(root, record.episode_id, step, config, tmp / "refused", preflight_plan=path)
    assert result["status"] == "refused" and result["branches"] == {} and not calls
    assert {p.name for p in (tmp / "refused").iterdir()} == {"counterfactual.json"}


def test_selected_pure_hold_is_not_a_distinct_alternative(saved_episode, monkeypatch):
    root, record, step, config, tmp = saved_episode
    path = root / "episodes" / (record.episode_id + ".json")
    rewrite(path, lambda d: d["steps"][step]["executed_command"].update(kind="hold", yaw_rad=None))
    calls = no_simulator(monkeypatch)
    result = run_counterfactual(root, record.episode_id, step, config, tmp / "hold")
    assert "already a pure hold" in result["refusal_reason"] and not calls


def test_missing_early_truth_anchor_fails_before_live_gate(saved_episode, monkeypatch):
    root, record, step, config, tmp = saved_episode
    path = root / "privileged_ledgers" / (record.episode_id + ".json")
    at = record.steps[0].sim_time_s
    rewrite(path, lambda d: d.update(samples=[s for s in d["samples"] if abs(s["sim_time_s"] - at) > 0.01]))
    calls = no_simulator(monkeypatch)
    result = run_counterfactual(root, record.episode_id, step, config, tmp / "truth-gap")
    assert "source truth does not establish" in result["refusal_reason"]
    assert result["branches"] == {} and not calls


@pytest.mark.parametrize(
    "field", ["executed_prefix_sha256", "selector", "replay_implementation_source_sha256"]
)
def test_changed_plan_contract_refuses_before_simulator(saved_episode, monkeypatch, field):
    root, record, step, config, tmp = saved_episode
    path = tmp / "plan.json"
    compile_replay_plan(root, path)

    def change(d):
        if field == "executed_prefix_sha256":
            d["candidates"][0][field] = "0" * 64
        else:
            d[field] = "first_direct_guard_nonhold" if field == "selector" else "0" * 64

    rewrite(path, change)
    calls = no_simulator(monkeypatch)
    result = run_counterfactual(root, record.episode_id, step, config, tmp / "tampered", preflight_plan=path)
    assert result["status"] == "refused" and not calls


def test_guard_selector_does_not_substitute_a_movement_when_no_guard_exists(saved_episode, monkeypatch):
    root, record, _, _, tmp = saved_episode
    calls = no_simulator(monkeypatch)
    report = compile_replay_plan(root, tmp / "guard.json", selector="first_direct_guard_nonhold")
    assert report["eligible_sources"] == 0 and report["refused_sources"] == 1 and not calls
    assert report["candidates"][0]["episode_id"] == record.episode_id


def test_guard_selector_uses_first_direct_action_and_skips_pure_hold(saved_episode):
    root, record, step, _, tmp = saved_episode
    path = root / "episodes" / (record.episode_id + ".json")

    def guarded(d):
        d["steps"][0]["executed_command"].update(kind="hold", issued_by="guard", yaw_rad=None)
        d["steps"][step]["executed_command"]["issued_by"] = "guard"

    rewrite(path, guarded)
    report = compile_replay_plan(root, tmp / "guard.json", selector="first_direct_guard_nonhold")
    assert report["eligible_sources"] == 1 and report["candidates"][0]["step_index"] == step


def test_all_orphan_missing_and_duplicate_attempts_remain_refused(saved_episode):
    root, record, _, _, tmp = saved_episode
    path = root / "attempted_runs.jsonl"
    row = json.loads(path.read_text().splitlines()[0])
    extra = copy.deepcopy(row)
    extra["episode_id"] = "missing-episode"
    path.write_text("\n".join(json.dumps(x) for x in (row, row, extra)) + "\n")
    report = compile_replay_plan(root, tmp / "all-refusals.json")
    assert report["retained_source_identities"] == report["refused_sources"] == 2
    assert report["eligible_sources"] == 0
    assert {x["episode_id"] for x in report["candidates"]} == {record.episode_id, "missing-episode"}


def test_no_plan_can_be_written_inside_original_run(saved_episode):
    root, _, _, _, _ = saved_episode
    with pytest.raises(ValueError, match="outside"):
        compile_replay_plan(root, root / "new-plan.json")


def test_live_shaped_source_tampering_cannot_reach_gate_or_adapter(saved_episode, monkeypatch):
    root, record, step, config, tmp = saved_episode
    # An explicitly synthetic declared identity exercises the live control-flow branch;
    # this fixture is never empirical evidence or connected to a live endpoint.
    identity = SimulatorIdentity(
        provenance="third_party_colosseum_build",
        artifact=SimulatorArtifactAttestation(
            provenance_class="third_party_colosseum_build",
            artifact_name="SYNTHETIC-TEST-ONLY",
            artifact_sha256="0" * 64,
            source_kind="third_party_publication",
            verified_by="unit-test-not-live",
            verified_utc="2026-09-27T00:00:00Z",
            caveats=["Synthetic declaration solely for a no-RPC refusal test; no live artifact verified."],
        ),
    ).model_dump(mode="json")
    for folder in ("episodes", "privileged_ledgers"):
        rewrite(
            root / folder / (record.episode_id + ".json"), lambda d: d.update(simulator_identity=identity)
        )
    path = tmp / "live-shaped-plan.json"
    report = compile_replay_plan(root, path)
    assert report["eligible_sources"] == 1
    source = root / "privileged_ledgers" / (record.episode_id + ".json")
    source.write_bytes(source.read_bytes() + b"\n")
    calls = no_simulator(monkeypatch)
    result = run_counterfactual(
        root, record.episode_id, step, config, tmp / "live-refused", preflight_plan=path
    )
    assert result["status"] == "refused" and "source inventory differs" in result["refusal_reason"]
    assert not calls and result["branches"] == {}
    assert {p.name for p in (tmp / "live-refused").iterdir()} == {"counterfactual.json"}
