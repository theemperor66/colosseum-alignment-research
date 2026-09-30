"""Unit tests for offline audit record ablation.

The point of these tests is that the ablation is *real*: fields are deleted rather than blanked, the
deletion is exactly the prespecified set, and the trajectory that the record describes is provably
untouched. All data here is synthetic and marked as fixture provenance; none of it is evidence.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import pytest

from colosseum_assurance.audit.records import (
    ABLATABLE_COMMAND_ANNOTATIONS,
    AblationError,
    AuditRecordSet,
    build_record_variant,
    build_record_variants,
    build_variants,
    field_paths,
    paths_under,
    removed_paths,
    trajectory_signature,
)
from colosseum_assurance.protocol.spec import AuditAblation, ProtocolConfig
from colosseum_assurance.schemas import (
    AuthorizationStatus,
    AuthorizationView,
    ControlCommand,
    DepthSummary,
    EpisodeRecord,
    MonitorReport,
    ObservationPacket,
    SensorHealth,
    SimulatorIdentity,
    StepRecord,
    SupervisionView,
    TerminationRecord,
    Vec3,
    VehicleState,
    Verdict,
)

# --------------------------------------------------------------------------------------
# Synthetic fixtures
#
# These builders are software-test fixtures. They are never experimental evidence: the simulator
# identity is marked ``fixture_fake`` and the truth samples are marked ``fixture_fake_ground_truth``,
# so any attempt to write them into a pilot or held-out run tree is refused by the evidence writer.
# --------------------------------------------------------------------------------------
SYNTHETIC_DT_S = 0.5
SYNTHETIC_STEPS = 12
SYNTHETIC_DELAY_S = 1.0
SYNTHETIC_TRUTH_INTERVAL_S = 0.1
INSPECTION_STEP = 9
GUARD_STEP = 10
GUARD_REASON = "loss_of_supervision_response"
AUTH_TOKEN = "auth-00"
AUTH_REQUESTED_AT_S = 1.0
AUTH_GRANTED_AT_S = 2.0
AUTH_VALIDITY_S = 45.0


def synthetic_truth_position(sim_time_s: float) -> Vec3:
    """Straight approach along +x that stops 7 m short of the asset and then holds."""
    return Vec3(x=min(7.0 * sim_time_s, 21.0), y=0.0, z=-6.0)


def _step_time(step_index: int) -> float:
    return round(step_index * SYNTHETIC_DT_S, 6)


def _synthetic_identity() -> SimulatorIdentity:
    return SimulatorIdentity(
        provenance="fixture_fake",
        endpoint_label="unit-test-fixture",
        simulator_name="synthetic-fixture",
        notes="software test fixture; never experimental evidence",
    )


def synthetic_episode_record(
    protocol: ProtocolConfig,
    *,
    episode_id: str = "fixture-synthetic-ep000",
    with_inspection: bool = True,
    with_guard: bool = True,
    dropout_steps: Sequence[int] = (),
) -> EpisodeRecord:
    """A 12-step exposed record with observations, authorization view and monitor reports."""
    dropped = set(dropout_steps)
    steps: list[StepRecord] = []
    for k in range(SYNTHETIC_STEPS):
        t = _step_time(k)
        measured_at = round(t - SYNTHETIC_DELAY_S, 6)
        has_state = measured_at >= 0.0 and k not in dropped
        state = (
            VehicleState(
                sim_time_s=measured_at,
                position=synthetic_truth_position(measured_at),
                velocity=Vec3(x=7.0, y=0.0, z=0.0),
                yaw_rad=0.0,
            )
            if has_state
            else None
        )
        authorization = (
            AuthorizationView(
                token_id=AUTH_TOKEN,
                status=AuthorizationStatus.GRANTED,
                requested_at_s=AUTH_REQUESTED_AT_S,
                granted_at_s=AUTH_GRANTED_AT_S,
                expires_at_s=AUTH_GRANTED_AT_S + AUTH_VALIDITY_S,
                scope="inspection_step",
                received_at_s=AUTH_GRANTED_AT_S,
                issuer="simulated_supervisor",
            )
            if t >= AUTH_GRANTED_AT_S
            else AuthorizationView(
                token_id=AUTH_TOKEN,
                status=AuthorizationStatus.PENDING,
                requested_at_s=AUTH_REQUESTED_AT_S,
            )
        )
        observation = ObservationPacket(
            step_index=k,
            receive_sim_time_s=t,
            state=state,
            state_age_s=None if state is None else SYNTHETIC_DELAY_S,
            depth=(
                DepthSummary(
                    sim_time_s=measured_at,
                    camera_name="front_center",
                    valid=True,
                    min_range_m=6.0,
                    free_path_m=9.0,
                    coverage_fraction=0.8,
                )
                if has_state
                else None
            ),
            rgb=None,
            supervision=SupervisionView(
                sim_time_s=t,
                last_heartbeat_sim_time_s=max((h for h in (0.0, 2.0, 4.0) if h <= t), default=None),
                heartbeat_age_s=t - max((h for h in (0.0, 2.0, 4.0) if h <= t), default=0.0),
                link_state="nominal",
            ),
            authorization=authorization,
            sensor_health=SensorHealth(
                state_sample_available=has_state,
                depth_available=has_state,
                rgb_available=False,
                state_age_s=None if state is None else SYNTHETIC_DELAY_S,
                depth_age_s=None if state is None else SYNTHETIC_DELAY_S,
                dropouts_in_window=1 if k in dropped else 0,
            ),
            mission_phase_hint="approach",
            declared_observation_delay_s=SYNTHETIC_DELAY_S,
        )
        kind = "move_to"
        if with_inspection and k == INSPECTION_STEP:
            kind = "inspect_capture"
        command = ControlCommand(
            step_index=k,
            issued_sim_time_s=t,
            kind=kind,  # type: ignore[arg-type]
            target=Vec3(x=21.0, y=0.0, z=-6.0),
            speed_mps=3.0,
            reason="approach the inspection viewpoint",
            issued_by="controller",
            controller_phase="approach",
        )
        executed = command
        intervention = "none"
        rationale = "assumptions hold; policy predicates satisfied"
        if with_guard and k == GUARD_STEP:
            executed = ControlCommand(
                step_index=k,
                issued_sim_time_s=t,
                kind="hold",
                speed_mps=0.0,
                reason=GUARD_REASON,
                issued_by="guard",
                controller_phase="approach",
            )
            intervention = "hold"
            rationale = GUARD_REASON
        report = MonitorReport(
            step_index=k,
            sim_time_s=t,
            monitor_id="assumption_aware_v1",
            verdict=Verdict.PASS if intervention == "none" else Verdict.UNKNOWN,
            obligation_verdicts={"geofence": Verdict.PASS, "inspection_authorization": Verdict.PASS},
            assumption_verdicts={"observation_freshness": Verdict.PASS},
            intervention=intervention,  # type: ignore[arg-type]
            rationale=rationale,
            evidence_age_s=SYNTHETIC_DELAY_S,
            affirmative=intervention == "none",
        )
        steps.append(
            StepRecord(
                step_index=k,
                sim_time_s=t,
                wall_clock_s=round(100.0 + k * 0.55, 6),
                observation=observation,
                command=command,
                executed_command=executed,
                monitor_report=report,
                controller_state={"phase": "approach", "waypoint_index": k // 3},
                provenance={"observation_source": "synthetic_fixture"},
            )
        )
    interventions = (
        [{"step_index": GUARD_STEP, "sim_time_s": _step_time(GUARD_STEP), "reason": GUARD_REASON}]
        if with_guard
        else []
    )
    return EpisodeRecord(
        episode_id=episode_id,
        scenario_id="fixture-synthetic-r000",
        arm_id="A2_assumption_aware",
        run_class="fixture",
        protocol_hash=protocol.content_hash(),
        policy_version=protocol.obligations.policy_version,
        code_version={"git_commit": "0000000", "dirty": "true"},
        simulator_identity=_synthetic_identity(),
        started_wall_clock="2026-01-01T00:00:00.000+00:00",
        dt_s=SYNTHETIC_DT_S,
        steps=steps,
        termination=TerminationRecord(
            reason="mission_complete",
            detail="synthetic fixture",
            step_index=SYNTHETIC_STEPS - 1,
            sim_time_s=_step_time(SYNTHETIC_STEPS - 1),
            completed_mission=True,
        ),
        monitor_id="assumption_aware_v1",
        interventions=interventions,
        notes="synthetic software-test fixture, not experimental evidence",
    )


@pytest.fixture(scope="module")
def protocol() -> ProtocolConfig:
    return ProtocolConfig()


@pytest.fixture(scope="module")
def record(protocol: ProtocolConfig) -> EpisodeRecord:
    return synthetic_episode_record(protocol)


@pytest.fixture(scope="module")
def variants(record: EpisodeRecord, protocol: ProtocolConfig) -> dict[str, AuditRecordSet]:
    return build_record_variants(record, protocol.audit)


def _source_steps(record: EpisodeRecord) -> list[dict]:
    return record.model_dump(mode="json")["steps"]


def test_variant_ids_match_the_frozen_schedule(
    variants: dict[str, AuditRecordSet], protocol: ProtocolConfig
) -> None:
    assert set(variants) == {a.ablation_id for a in protocol.audit.record_variants}
    for ablation in protocol.audit.record_variants:
        assert variants[ablation.ablation_id].removed_fields == list(ablation.removed_fields)
        assert variants[ablation.ablation_id].missed_paths == []


def test_ablation_removes_exactly_the_specified_fields(
    record: EpisodeRecord, variants: dict[str, AuditRecordSet], protocol: ProtocolConfig
) -> None:
    """The diff between source and variant must equal the prespecified paths, subtree expanded."""
    source_steps = _source_steps(record)
    for ablation in protocol.audit.record_variants:
        variant = variants[ablation.ablation_id]
        assert len(variant.steps) == len(source_steps)
        for source_step, variant_step in zip(source_steps, variant.steps, strict=True):
            expected: set[str] = set()
            for target in ablation.removed_fields:
                expected |= paths_under(source_step, target)
            assert removed_paths(source_step, variant_step) == expected
            assert field_paths(variant_step) == field_paths(source_step) - expected


def test_action_only_deletes_keys_instead_of_blanking_them(variants: dict[str, AuditRecordSet]) -> None:
    """Action-only means actions only.

    The independent review showed that keeping ``command.reason`` (or the monitor verdict) left the Q4
    answer in the record, so a "degraded" variant still carried the tested answer. The frozen contract in
    ``AuditSpec`` now removes the whole monitor report and both free-text reasons.
    """
    for step in variants["action_only"].steps:
        assert "observation" not in step, "a blanked key still tells the auditor the field was logged"
        assert "controller_state" not in step
        assert "provenance" not in step
        assert "monitor_report" not in step, "the verdict and intervention answer Q4 by themselves"
        assert "command" in step and "executed_command" in step
        assert "reason" not in step["command"], "the command reason is an alternative Q4 answer route"
        assert "reason" not in step["executed_command"]
        # What remains is the trajectory itself.
        assert step["command"]["kind"]
        assert "issued_sim_time_s" in step["command"]


def test_no_evidence_age_removes_every_route_to_the_age(
    variants: dict[str, AuditRecordSet],
) -> None:
    """After the review this variant also drops the acquisition timestamp.

    Keeping ``observation.state.sim_time_s`` let an auditor recompute the age exactly, so the variant
    claimed to remove information it still carried.
    """
    for step in variants["no_evidence_age"].steps:
        observation = step["observation"]
        assert "state_age_s" not in observation
        assert "sensor_health" not in observation
        assert "receive_sim_time_s" in observation, "the receipt time is not an age by itself"
        assert "authorization" in observation
        state = observation.get("state")
        if state is not None:
            assert "sim_time_s" not in state, "the acquisition time would make the age recoverable"
            assert "position" in state, "the position observation itself is deliberately retained"


def test_explicit_age_fields_only_is_a_labelled_redundancy_probe(
    variants: dict[str, AuditRecordSet], protocol: ProtocolConfig
) -> None:
    """This variant keeps the timestamps on purpose and must be described that way."""
    spec = next(v for v in protocol.audit.record_variants if v.ablation_id == "explicit_age_fields_only")
    assert spec.retains_redundant_routes is True
    assert spec.expected_unanswerable == {}
    for step in variants["explicit_age_fields_only"].steps:
        observation = step["observation"]
        assert "state_age_s" not in observation
        assert "receive_sim_time_s" in observation
        state = observation.get("state")
        if state is not None:
            assert "sim_time_s" in state, "age stays recoverable here; that is the point of the probe"


def test_trajectory_signature_is_identical_across_all_protocol_variants(
    record: EpisodeRecord, variants: dict[str, AuditRecordSet]
) -> None:
    source_signature = trajectory_signature(_source_steps(record))
    signatures = {variant.trajectory_signature for variant in variants.values()}
    assert signatures == {source_signature}
    for variant in variants.values():
        assert trajectory_signature(variant.steps) == source_signature


def test_trajectory_fields_are_identical_and_only_annotations_differ(
    variants: dict[str, AuditRecordSet],
) -> None:
    """The flown trajectory is untouched; only explanatory annotations are removed.

    ``command.reason`` is an ablatable annotation, not part of the trajectory, so action-only records must
    differ from the rich records in exactly that key and nowhere else among the signed fields.
    """
    rich = variants["provenance_rich"].steps
    action_only = variants["action_only"].steps
    plain = ("step_index", "sim_time_s", "wall_clock_s")
    for rich_step, lean_step in zip(rich, action_only, strict=True):
        for name in plain:
            assert rich_step[name] == lean_step[name]
        for name in ("command", "executed_command"):
            difference = set(rich_step[name]) ^ set(lean_step[name])
            assert difference <= ABLATABLE_COMMAND_ANNOTATIONS, (
                f"{name} differs in {sorted(difference)}, which is not an ablatable annotation"
            )
            for key, value in lean_step[name].items():
                assert rich_step[name][key] == value, f"{name}.{key} changed between variants"


def test_trajectory_signature_detects_a_changed_command(record: EpisodeRecord) -> None:
    """Without this the invariance test above could pass for a signature that ignores its input."""
    steps = copy.deepcopy(_source_steps(record))
    baseline = trajectory_signature(steps)
    steps[3]["command"]["kind"] = "abort"
    assert trajectory_signature(steps) != baseline
    steps = copy.deepcopy(_source_steps(record))
    steps[2]["sim_time_s"] = steps[2]["sim_time_s"] + 0.001
    assert trajectory_signature(steps) != baseline
    steps = copy.deepcopy(_source_steps(record))
    del steps[5]["executed_command"]
    assert trajectory_signature(steps) != baseline


def test_missing_path_raises_in_strict_mode(record: EpisodeRecord) -> None:
    ablation = AuditAblation(
        ablation_id="typo_variant",
        removed_fields=["observation.state_age_seconds"],
        description="field name that never existed",
    )
    with pytest.raises(AblationError, match="exists in no step"):
        build_record_variant(record, ablation)


def test_missing_path_is_recorded_when_strict_is_disabled(record: EpisodeRecord) -> None:
    ablation = AuditAblation(
        ablation_id="typo_variant",
        removed_fields=["observation.state_age_seconds", "controller_state"],
        description="one missing path and one real path",
    )
    variant = build_record_variant(record, ablation, strict=False)
    assert variant.missed_paths == ["observation.state_age_seconds"]
    assert all("controller_state" not in step for step in variant.steps)
    assert all("state_age_s" in step["observation"] for step in variant.steps)


def test_removing_a_path_whose_parent_is_already_gone_is_not_a_miss(record: EpisodeRecord) -> None:
    ablation = AuditAblation(
        ablation_id="overlapping",
        removed_fields=["observation", "observation.state_age_s"],
        description="the child path is already gone when it is applied",
    )
    variant = build_record_variant(record, ablation)
    assert variant.missed_paths == []
    assert all("observation" not in step for step in variant.steps)


@pytest.mark.parametrize(
    "path",
    ["command", "command.kind", "executed_command", "executed_command.target.x", "step_index",
     "sim_time_s", "wall_clock_s"],
)
def test_protected_paths_are_refused(record: EpisodeRecord, path: str) -> None:
    ablation = AuditAblation(
        ablation_id="illegal", removed_fields=[path], description="would alter the trajectory"
    )
    with pytest.raises(AblationError, match="protected step field"):
        build_record_variant(record, ablation)


def test_empty_path_is_refused(record: EpisodeRecord) -> None:
    ablation = AuditAblation(ablation_id="empty", removed_fields=[""], description="malformed")
    with pytest.raises(AblationError):
        build_record_variant(record, ablation)


def test_nested_path_under_a_removable_subtree_is_allowed(record: EpisodeRecord) -> None:
    """``observation.step_index`` is a copy inside the packet, not the step index of the trajectory."""
    ablation = AuditAblation(
        ablation_id="nested", removed_fields=["observation.step_index"], description="inside the packet"
    )
    variant = build_record_variant(record, ablation)
    assert all("step_index" not in step["observation"] for step in variant.steps)
    assert all(step["step_index"] == k for k, step in enumerate(variant.steps))


def test_episode_level_fields_survive_every_variant(
    record: EpisodeRecord, variants: dict[str, AuditRecordSet]
) -> None:
    for variant in variants.values():
        assert variant.episode_id == record.episode_id
        assert variant.policy_version == record.policy_version
        assert variant.dt_s == record.dt_s
        assert variant.termination_reason == record.termination.reason
        assert variant.episode_fields["simulator_identity"]["provenance"] == "fixture_fake"
        assert variant.episode_fields["monitor_id"] == "assumption_aware_v1"
        assert variant.episode_fields["interventions"][0]["reason"] == GUARD_REASON
        assert variant.episode_fields["code_version"]["git_commit"] == "0000000"
        assert variant.episode_fields["termination"]["reason"] == "mission_complete"
        assert "steps" not in variant.episode_fields


def test_source_record_is_not_mutated(record: EpisodeRecord, protocol: ProtocolConfig) -> None:
    before = record.model_dump(mode="json")
    build_record_variants(record, protocol.audit)
    assert record.model_dump(mode="json") == before


def test_build_variants_alias_returns_a_list_in_protocol_order(
    record: EpisodeRecord, protocol: ProtocolConfig
) -> None:
    listed = build_variants(record, protocol)
    assert isinstance(listed, list)
    assert [v.variant_id for v in listed] == [a.ablation_id for a in protocol.audit.record_variants]
    keyed = build_record_variants(record, protocol.audit)
    assert [v.model_dump(mode="json") for v in listed] == [
        keyed[a.ablation_id].model_dump(mode="json") for a in protocol.audit.record_variants
    ]


def test_step_lookup_uses_step_index_not_list_position(variants: dict[str, AuditRecordSet]) -> None:
    variant = variants["provenance_rich"]
    assert variant.step_count == SYNTHETIC_STEPS
    step = variant.step_at(GUARD_STEP)
    assert step is not None
    assert step["executed_command"]["issued_by"] == "guard"
    assert variant.step_at(SYNTHETIC_STEPS + 5) is None
