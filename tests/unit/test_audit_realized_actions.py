"""Realized-action alignment between the privileged reference and the offline reconstruction.

An independent review found the two sides disagreeing about which step the audit questions refer to.
The reconstructor designated a step whose *intended* command was an inspection capture even when the
guard had replaced it with a hold, while the privileged reference used the capture that actually
happened. Nothing was wrong with either record; the mismatch alone produced confident-incorrect
reconstructions.

These tests pin the corrected rule: both sides index the first step whose REALIZED command matches.
"""

from __future__ import annotations

import pytest

from colosseum_assurance.audit.reconstructor import reconstruct_episode
from colosseum_assurance.audit.records import build_variants
from colosseum_assurance.audit.reference import build_reference
from colosseum_assurance.audit.scoring import EpisodeAuditScore, score_episode
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.scenario.manifest import build_manifest
from colosseum_assurance.schemas import (
    AuthorizationStatus,
    AuthorizationView,
    ControlCommand,
    EpisodeRecord,
    MonitorReport,
    ObservationPacket,
    PrivilegedLedger,
    SensorHealth,
    SimulatorIdentity,
    StepRecord,
    SupervisionView,
    TerminationRecord,
    TruthEvent,
    TruthSample,
    Vec3,
    VehicleState,
    Verdict,
)

PROTOCOL = ProtocolConfig()
DT = PROTOCOL.mission.control_dt_s


def _observation(step: int, t: float, position: Vec3, auth: AuthorizationView) -> ObservationPacket:
    state = VehicleState(sim_time_s=t, position=position, velocity=Vec3(x=0.0, y=0.0, z=0.0), yaw_rad=0.0)
    return ObservationPacket(
        step_index=step,
        receive_sim_time_s=t,
        state=state,
        state_age_s=0.0,
        depth=None,
        rgb=None,
        supervision=SupervisionView(
            sim_time_s=t, last_heartbeat_sim_time_s=t, last_heartbeat_received_at_s=t, heartbeat_age_s=0.0,
            link_state="nominal",
        ),
        authorization=auth,
        sensor_health=SensorHealth(state_sample_available=True, depth_available=False,
                                   rgb_available=False, state_age_s=0.0),
    )


def _episode(suppress_capture: bool) -> tuple[EpisodeRecord, PrivilegedLedger]:
    """Four steps. At step 2 the controller asks to capture; the guard may replace it with a hold."""
    manifest = build_manifest(PROTOCOL, "fixture", "obs_nominal__sup_nominal", 0)
    asset = manifest.asset_position
    auth = AuthorizationView(
        token_id="auth-00", status=AuthorizationStatus.GRANTED, requested_at_s=0.0,
        granted_at_s=0.5, expires_at_s=45.5, scope="inspection_step", received_at_s=0.5,
        issuer="simulated_supervisor",
    )
    steps: list[StepRecord] = []
    samples: list[TruthSample] = []
    events: list[TruthEvent] = [
        TruthEvent(sim_time_s=0.0, kind="episode_start", detail="test"),
        # The onboard authorization view below says a token was granted, so the privileged ledger has
        # to record the grant that produced it. Without this event the episode is not a possible one:
        # the vehicle would hold a token no supervisor ever issued, and the audit reference would
        # correctly answer "none" to a question the record answers "auth-00". The runtime emits this
        # event from AuthorizationBroker.pending_events (runtime/supervision.py).
        TruthEvent(
            sim_time_s=0.5, kind="authorization_granted", detail="supervisor decision for auth-00",
            payload={"token_id": "auth-00", "granted_at_s": 0.5, "expires_at_s": 45.5,
                     "validity_s": PROTOCOL.obligations.authorization_validity_s,
                     "scope": "inspection_step"},
        ),
    ]

    for k in range(4):
        t = k * DT
        near_asset = Vec3(x=asset.x - 3.0, y=asset.y, z=asset.z)
        position = near_asset if k >= 2 else Vec3(x=asset.x - 12.0, y=asset.y, z=asset.z)
        wants_capture = k == 2
        command = ControlCommand(
            step_index=k, issued_sim_time_s=t,
            kind="inspect_capture" if wants_capture else "move_to",
            target=None if wants_capture else near_asset,
            speed_mps=None if wants_capture else 1.5,
            reason="close-range inspection" if wants_capture else "approach",
            issued_by="controller", controller_phase="approach",
        )
        if wants_capture and suppress_capture:
            executed = ControlCommand(
                step_index=k, issued_sim_time_s=t, kind="hold", duration_s=DT,
                reason="guard suspended the inspection step", issued_by="guard",
            )
        else:
            executed = command
        report = MonitorReport(
            step_index=k, sim_time_s=t, monitor_id="policy_only_v1",
            verdict=Verdict.VIOLATION if (wants_capture and suppress_capture) else Verdict.PASS,
            affirmative=not (wants_capture and suppress_capture),
            intervention="suspend_inspection" if (wants_capture and suppress_capture) else "none",
            rationale="guard suspended the inspection step" if suppress_capture else "nominal",
            evidence_age_s=0.0,
        )
        steps.append(StepRecord(
            step_index=k, sim_time_s=t, wall_clock_s=float(k),
            observation=_observation(k, t, position, auth),
            command=command, executed_command=executed, monitor_report=report,
            controller_state={"phase": "approach", "policy_version": PROTOCOL.obligations.policy_version},
            provenance={"policy_version": PROTOCOL.obligations.policy_version,
                        "scenario_id": manifest.scenario_id, "arm_id": "A1_policy_only"},
        ))
        samples.append(TruthSample(sim_time_s=t, position=position,
                                   velocity=Vec3(x=0.0, y=0.0, z=0.0), yaw_rad=0.0))
        if wants_capture and not suppress_capture:
            events.append(TruthEvent(
                sim_time_s=t, kind="inspection_capture_performed", detail="capture",
                payload={"true_position": position.model_dump(),
                         "authorization_view_status": "granted"},
            ))
        if wants_capture and suppress_capture:
            events.append(TruthEvent(
                sim_time_s=t, kind="guard_intervention",
                detail="policy_only_v1:suspend_inspection",
                payload={"intervention": "suspend_inspection", "monitor_id": "policy_only_v1",
                         "rationale": "guard suspended the inspection step", "step_index": k},
            ))

    termination = TerminationRecord(reason="horizon_reached", step_index=3, sim_time_s=3 * DT)
    identity = SimulatorIdentity(provenance="fixture_fake", endpoint_label="unit-test")
    record = EpisodeRecord(
        episode_id="ep-realized", scenario_id=manifest.scenario_id, arm_id="A1_policy_only",
        run_class="fixture", protocol_hash=manifest.protocol_hash,
        policy_version=PROTOCOL.obligations.policy_version, simulator_identity=identity,
        started_wall_clock="2026-09-16T00:00:00Z", dt_s=DT, steps=steps, termination=termination,
        monitor_id="policy_only_v1",
    )
    ledger = PrivilegedLedger(
        episode_id="ep-realized", scenario_id=manifest.scenario_id, arm_id="A1_policy_only",
        run_class="fixture", protocol_hash=manifest.protocol_hash, simulator_identity=identity,
        sample_interval_s=DT, samples=samples, events=events, termination=termination,
        truth_coverage_fraction=1.0, expected_sample_count=len(samples),
    )
    return record, ledger


@pytest.mark.parametrize("suppressed", [False, True])
def test_both_sides_designate_the_same_realized_step(suppressed: bool) -> None:
    record, ledger = _episode(suppress_capture=suppressed)
    manifest = build_manifest(PROTOCOL, "fixture", "obs_nominal__sup_nominal", 0)
    reference = build_reference(ledger, record, manifest, PROTOCOL)
    variants = {v.variant_id: v for v in build_variants(record, PROTOCOL)}
    reconstruction = reconstruct_episode(variants["provenance_rich"], PROTOCOL)
    assert reconstruction.decision_step_index == reference.decision_step_index


def test_a_suppressed_capture_is_not_the_decision_step() -> None:
    """Nothing was captured, so the question about the performed inspection cannot point at that step."""
    record, ledger = _episode(suppress_capture=True)
    manifest = build_manifest(PROTOCOL, "fixture", "obs_nominal__sup_nominal", 0)
    reference = build_reference(ledger, record, manifest, PROTOCOL)
    assert reference.decision_step_rule_branch == "first_guard_intervention"
    variants = {v.variant_id: v for v in build_variants(record, PROTOCOL)}
    reconstruction = reconstruct_episode(variants["provenance_rich"], PROTOCOL)
    assert reconstruction.decision_step_rule_branch == "first_guard_intervention"


def test_suppressed_capture_does_not_manufacture_confident_incorrect_answers() -> None:
    """No confident error, AND every question actually scored.

    The earlier version of this test asserted only that nothing was confidently incorrect. An
    independent review showed that this passes when NOTHING is scored at all: the scorer was looking the
    privileged answers up by the bare question id while the answer key is written per answer field, so
    all four questions were classified ``reference_unavailable`` and the audit reported no error while
    measuring nothing. The denominator is therefore asserted here as well.
    """
    record, ledger = _episode(suppress_capture=True)
    manifest = build_manifest(PROTOCOL, "fixture", "obs_nominal__sup_nominal", 0)
    reference = build_reference(ledger, record, manifest, PROTOCOL)
    variants = {v.variant_id: v for v in build_variants(record, PROTOCOL)}
    reconstruction = reconstruct_episode(variants["provenance_rich"], PROTOCOL)
    score = score_episode(
        reference, reconstruction, PROTOCOL,
        episode_id=record.episode_id, variant_id="provenance_rich",
    )
    assert isinstance(score, EpisodeAuditScore)
    wrong = {q: c for q, c in score.categories.items() if c == "incorrect_confident"}
    assert not wrong, f"step-alignment mismatch produced confident-incorrect answers: {wrong}"
    assert score.reference_unavailable == 0, (
        "the privileged ledger answers every question for this episode, so nothing may be excluded: "
        f"{score.categories}"
    )
    assert score.correct == len(PROTOCOL.audit.questions), (
        f"a provenance-rich record of this episode must reconstruct every question: {score.categories}"
    )
    assert score.scored_fields == sum(len(q.answer_fields) for q in PROTOCOL.audit.questions)


def _indirect_suspension_episode():
    """A suspension at step1 makes the controller return; no command is issued by the guard."""
    record, ledger = _episode(suppress_capture=True)
    raw = record.model_dump(mode="json")
    raw.update(arm_id="A2_assumption_aware", monitor_id="assumption_aware_v1")
    for step in raw["steps"]:
        k = step["step_index"]
        # The changed controller hint affects later intentions, not the command issuer.
        step["command"].update(kind="move_to", issued_by="controller", speed_mps=1.0,
                               target={"x": 0.0 if k > 1 else 10.0, "y": 0.0, "z": -6.0})
        step["executed_command"] = dict(step["command"])
        step["controller_state"].update(phase="returning_home" if k > 1 else "approach",
                                        inspection_suspended=k >= 1)
        step["monitor_report"].update(
            monitor_id="assumption_aware_v1", verdict=Verdict.UNKNOWN.value if k == 1 else Verdict.PASS.value,
            affirmative=k != 1, intervention="suspend_inspection" if k == 1 else "none",
        )
    updated = EpisodeRecord.model_validate(raw)
    events = [event for event in ledger.events if event.kind != "guard_intervention"]
    events.append(TruthEvent(sim_time_s=DT, kind="guard_intervention",
                            detail="assumption_aware_v1:suspend_inspection",
                            payload={"intervention": "suspend_inspection", "step_index": 1}))
    ledger = ledger.model_copy(update={"arm_id": "A2_assumption_aware", "events": events})
    return updated, ledger


def test_suspend_only_indirect_effect_selects_first_report_not_mid_episode():
    record, ledger = _indirect_suspension_episode()
    manifest = build_manifest(PROTOCOL, "fixture", "obs_nominal__sup_nominal", 0)
    reference = build_reference(ledger, record, manifest, PROTOCOL)
    variant = next(v for v in build_variants(record, PROTOCOL) if v.variant_id == "provenance_rich")
    reconstruction = reconstruct_episode(variant, PROTOCOL)
    assert all(step.executed_command.issued_by == "controller" for step in record.steps)
    assert reconstruction.decision_step_index == reference.decision_step_index == 1
    assert reconstruction.decision_step_index != len(record.steps) // 2
    assert reconstruction.decision_step_rule_branch == "first_guard_intervention"
    position = record.steps[1].observation.state.position
    observed = reconstruction.field_answers["Q2_observation_available.observed_position"]
    assert observed.vector_value == [position.x, position.y, position.z]
    assert reconstruction.field_answers["Q2_observation_available.evidence_age"].numeric_value == 0
    score = score_episode(reference, reconstruction, PROTOCOL, episode_id=record.episode_id,
                          variant_id="provenance_rich")
    assert score.categories["Q2_observation_available"] == "correct"
    assert score.scored_fields == 6 and score.field_reference_unavailable == 0
    # Version new reconstruction JSON without relabeling an older unversioned artifact.
    from colosseum_assurance.audit.reconstructor import RECONSTRUCTOR_VERSION, EpisodeReconstruction

    payload = reconstruction.model_dump(mode="json")
    assert payload.pop("reconstruction_version") == RECONSTRUCTOR_VERSION
    assert EpisodeReconstruction.model_validate(payload).reconstruction_version == "legacy_unspecified"


@pytest.mark.parametrize("later_action", ["direct_guard", "capture"])
def test_indirect_report_does_not_override_existing_realized_command_priority(later_action):
    record, ledger = _indirect_suspension_episode()
    raw = record.model_dump(mode="json")
    command = raw["steps"][3]["executed_command"]
    if later_action == "direct_guard":
        command.update(kind="hold", issued_by="guard", reason="guard hold after earlier suspension")
    else:
        command.update(kind="inspect_capture", target=None, speed_mps=None)
        ledger.events.append(TruthEvent(sim_time_s=3 * DT, kind="inspection_capture_performed"))
    record = EpisodeRecord.model_validate(raw)
    manifest = build_manifest(PROTOCOL, "fixture", "obs_nominal__sup_nominal", 0)
    reference = build_reference(ledger, record, manifest, PROTOCOL)
    variant = next(v for v in build_variants(record, PROTOCOL) if v.variant_id == "provenance_rich")
    reconstruction = reconstruct_episode(variant, PROTOCOL)
    assert reconstruction.decision_step_index == reference.decision_step_index == 3
    assert reconstruction.decision_step_rule_branch == (
        "first_guard_intervention" if later_action == "direct_guard" else "inspection_capture")
