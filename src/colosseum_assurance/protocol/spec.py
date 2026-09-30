"""Frozen protocol specification: mission, obligations, arms, conditions, sampling, audit, analysis.

The protocol is a data object with a content hash. Every episode record and ledger carries that hash,
so pilot data, held-out data, and fixture data cannot be mixed without detection
(research-acceptance.md section 2: "Freeze the protocol ... before using fresh held-out scenarios").
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import Field, model_serializer, model_validator

from colosseum_assurance.protocol.context_confidence import ContextConfidenceSpec
from colosseum_assurance.protocol.controlled import ControlledStudySpec
from colosseum_assurance.protocol.expanded import ExpandedStudySpec
from colosseum_assurance.schemas import Box, StrictModel, Vec3

PROTOCOL_SCHEMA_VERSION = "1.0.0"


class MissionSpec(StrictModel):
    """One civilian inspection mission: launch, transit, close inspection, return."""

    mission_id: str = "civilian_tower_inspection_v1"
    home: Vec3 = Vec3(x=0.0, y=0.0, z=0.0)
    cruise_altitude_m: float = Field(default=6.0, gt=0.0, description="Height above home, positive up.")
    asset_nominal_position: Vec3 = Vec3(x=28.0, y=0.0, z=-6.0)
    inspection_standoff_m: float = Field(default=4.0, gt=0.0)
    inspection_tolerance_m: float = Field(default=1.5, gt=0.0)
    inspection_dwell_s: float = Field(default=3.0, gt=0.0)
    required_inspection_captures: int = Field(default=3, ge=1)
    cruise_speed_mps: float = Field(default=3.0, gt=0.0)
    approach_speed_mps: float = Field(default=1.5, gt=0.0)
    episode_horizon_s: float = Field(default=120.0, gt=0.0)
    control_dt_s: float = Field(default=0.5, gt=0.0)
    return_tolerance_m: float = Field(default=3.0, gt=0.0)


class ObligationSpec(StrictModel):
    """The measurable obligations of the frozen policy. Physical and procedural are scored apart."""

    policy_version: str = "policy-v1.0.0"
    geofence: Box = Box(x_min=-12.0, x_max=42.0, y_min=-22.0, y_max=22.0, z_min=-22.0, z_max=0.6)
    geofence_tolerance_m: float = Field(default=0.5, ge=0.0)
    min_obstacle_clearance_m: float = Field(default=0.0, ge=0.0)
    collision_is_violation: bool = True
    authorized_inspection_radius_m: float = Field(
        default=8.0, gt=0.0, description="Closer than this to the asset counts as the inspection step."
    )
    authorization_validity_s: float = Field(default=45.0, gt=0.0)
    supervision_heartbeat_period_s: float = Field(default=2.0, gt=0.0)
    loss_of_supervision_threshold_s: float = Field(
        default=6.0, gt=0.0, description="Heartbeat gap after which supervision counts as lost."
    )
    loss_of_supervision_reaction_deadline_s: float = Field(
        default=5.0, gt=0.0, description="Time allowed to enter a safe response after loss is detectable."
    )
    safe_response_speed_mps: float = Field(
        default=0.6, gt=0.0, description="Speed at or below which a hold response counts as entered."
    )
    safe_response_requires_leaving_inspection_zone: bool = True
    safe_response_must_be_maintained: bool = Field(
        default=True,
        description=(
            "True: the safe response must be entered by the deadline AND maintained until supervision is "
            "restored or the episode ends. False: entering it once is sufficient. See "
            "docs/timing-semantics.md."
        ),
    )
    supervision_loss_trigger: Literal["received_gap_exceeds_threshold"] = Field(
        default="received_gap_exceeds_threshold",
        description=(
            "Loss holds when (now - last RECEIVED heartbeat time) > loss_of_supervision_threshold_s, "
            "strictly greater. Equality is not yet a loss."
        ),
    )
    deadline_starts_at: Literal["detectable_onboard"] = Field(
        default="detectable_onboard",
        description=(
            "The reaction deadline starts at last_received_heartbeat + loss_of_supervision_threshold_s, "
            "which is the first moment the vehicle could know, not the moment the outage began."
        ),
    )
    response_entry_grace_s: float = Field(
        default=0.0, ge=0.0,
        description="Extra slack added to the reaction deadline before a violation is declared.",
    )

    @property
    def physical_obligation_ids(self) -> tuple[str, ...]:
        return ("geofence", "collision")

    @property
    def procedural_obligation_ids(self) -> tuple[str, ...]:
        return ("inspection_authorization", "loss_of_supervision_response")

    @property
    def obligation_ids(self) -> tuple[str, ...]:
        return self.physical_obligation_ids + self.procedural_obligation_ids


class ConditionLevel(StrictModel):
    """One level of an exogenous stress factor, expressed relative to modelled task timing."""

    level_id: str
    value_s: float = Field(ge=0.0)
    jitter_s: float = Field(default=0.0, ge=0.0)
    dropout_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    label: str = ""
    relative_to: str = ""


class ConditionsSpec(StrictModel):
    """Crossed observation-delay and supervisory-response-delay levels, including a nominal cell."""

    observation_delay_levels: list[ConditionLevel] = Field(
        default_factory=lambda: [
            ConditionLevel(
                level_id="obs_nominal", value_s=0.0, jitter_s=0.0, dropout_probability=0.0,
                label="nominal", relative_to="0 control steps",
            ),
            ConditionLevel(
                level_id="obs_moderate", value_s=1.0, jitter_s=0.2, dropout_probability=0.05,
                label="moderate", relative_to="2 control steps at dt=0.5 s",
            ),
            ConditionLevel(
                level_id="obs_severe", value_s=2.5, jitter_s=0.5, dropout_probability=0.15,
                label="severe",
                relative_to="5 control steps; ~1.7x the 1.5 m inspection tolerance at 1.5 m/s",
            ),
        ]
    )
    supervision_delay_levels: list[ConditionLevel] = Field(
        default_factory=lambda: [
            ConditionLevel(
                level_id="sup_nominal", value_s=1.0, jitter_s=0.0, dropout_probability=0.0,
                label="nominal", relative_to="0.5x heartbeat period",
            ),
            ConditionLevel(
                level_id="sup_moderate", value_s=6.0, jitter_s=1.0, dropout_probability=0.0,
                label="moderate", relative_to="1.0x loss-of-supervision threshold",
            ),
            ConditionLevel(
                level_id="sup_severe", value_s=14.0, jitter_s=2.0, dropout_probability=0.0,
                label="severe", relative_to="2.3x loss-of-supervision threshold",
            ),
        ]
    )
    supervision_outage_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    supervision_outage_duration_s: list[float] = Field(default_factory=lambda: [8.0, 16.0])
    visibility_levels: list[str] = Field(default_factory=lambda: ["clear", "reduced"])
    layout_variants: list[str] = Field(default_factory=lambda: ["open", "occluded", "cluttered"])

    @property
    def cell_ids(self) -> list[str]:
        return [f"{o.level_id}__{s.level_id}" for o in self.observation_delay_levels
                for s in self.supervision_delay_levels]


class ArmSpec(StrictModel):
    """One comparison arm. The base controller and mission are identical across arms."""

    arm_id: str
    monitor_id: str | None
    guard_enabled: bool
    description: str


class ArmsSpec(StrictModel):
    arms: list[ArmSpec] = Field(
        default_factory=lambda: [
            ArmSpec(
                arm_id="A0_unguarded", monitor_id=None, guard_enabled=False,
                description="Fixed controller, no runtime guard. Context for scenario difficulty.",
            ),
            ArmSpec(
                arm_id="A1_policy_only", monitor_id="policy_only_v1", guard_enabled=True,
                description=(
                    "Explicitly limited diagnostic baseline: checks the received state against the frozen "
                    "policy predicates and treats received information as current and valid."
                ),
            ),
            ArmSpec(
                arm_id="A2_assumption_aware", monitor_id="assumption_aware_v1", guard_enabled=True,
                description=(
                    "Assumption-aware comparator inspired by Zudaire et al. (2021) assumption monitoring "
                    "and ModelPlex-style model-validity checking; emits unknown verdicts and intervenes "
                    "when the assumptions required to interpret evidence fail."
                ),
            ),
        ]
    )

    @property
    def arm_ids(self) -> list[str]:
        return [a.arm_id for a in self.arms]

    def get(self, arm_id: str) -> ArmSpec:
        for a in self.arms:
            if a.arm_id == arm_id:
                return a
        raise KeyError(f"unknown arm_id: {arm_id}")


class SamplingSpec(StrictModel):
    """Sample sizing. Pilot values are provisional; held-out values are frozen after the pilot."""

    pilot_realizations_per_cell: int = Field(default=2, ge=1)
    heldout_realizations_per_cell: int = Field(default=40, ge=1)
    planning_budget_episodes: int = 1080
    sizing_rationale: str = (
        "40 realizations per cell x 9 cells x 3 arms = 1080 episodes is the planning budget from the "
        "research plan, not a power calculation. The pilot measures episode runtime and event prevalence; "
        "freeze_sample_size() then recomputes the held-out count from measured prevalence and the "
        "target half-width for the paired difference."
    )
    target_ci_half_width: float = Field(default=0.10, gt=0.0, le=0.5)
    base_seed: int = 20260916


class AcceptanceRuleSpec(StrictModel):
    """The exact finite-horizon acceptance rule behind every conditional false-assurance rate."""

    rule_id: str = "finite_horizon_affirmative_v1"
    statement: str = (
        "An episode is ACCEPTED by an arm if and only if: (a) the episode reached a terminal state "
        "within the episode horizon, (b) the arm's monitor issued an affirmative PASS verdict at every "
        "evaluated step over the whole retained horizon, (c) no step produced VIOLATION or UNKNOWN, and "
        "(d) monitor step coverage is at least min_step_coverage of expected control steps. "
        "Unguarded arm A0 has no monitor and is therefore never accepted; its acceptance-conditional "
        "rates are undefined by construction."
    )
    min_step_coverage: float = Field(default=0.95, gt=0.0, le=1.0)
    unknown_counts_as_pass: Literal[False] = False
    incomplete_counts_as_pass: Literal[False] = False


class AnalysisSpec(StrictModel):
    primary_outcome: str = (
        "Paired per-episode difference between A1 and A2 in the fraction of scenario realizations with an "
        "independently assessed physical violation; procedural violations reported separately."
    )
    statistical_unit: Literal["scenario_realization"] = "scenario_realization"
    bootstrap_resamples: int = Field(default=10000, ge=1000)
    bootstrap_seed: int = 7717
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    paired_test: Literal["exact_mcnemar"] = "exact_mcnemar"
    proportion_ci: Literal["wilson"] = "wilson"
    acceptance_rule: AcceptanceRuleSpec = Field(default_factory=AcceptanceRuleSpec)
    strata: list[str] = Field(default_factory=lambda: ["observation_delay_level", "supervision_delay_level"])


class AnswerField(StrictModel):
    """One concrete fact an audit question asks for, with its own unit, tolerance, and record routes.

    ``routes`` is the explicit permitted-information contract asked for in review: every dotted step-record
    path from which this single fact can be recovered. An ablation that claims to defeat a fact must remove
    all of that fact's routes, which :meth:`AuditSpec.validate_ablations` checks before the freeze.
    """

    name: str
    kind: Literal["categorical", "numeric", "identifier", "timestamp", "boolean"]
    unit: str = ""
    tolerance: float | None = Field(
        default=None, description="Numeric match tolerance in `unit`. None means exact match."
    )
    routes: list[str] = Field(default_factory=list)


class AuditQuestion(StrictModel):
    """An audit question and the separate facts it asks for.

    Position (metres) and evidence age (seconds) are separate answer fields with separate tolerances, so a
    single numeric tolerance cannot be applied to two different units.
    """

    question_id: str
    text: str
    answer_fields: list[AnswerField]

    @property
    def answer_routes(self) -> list[str]:
        seen: list[str] = []
        for field in self.answer_fields:
            for route in field.routes:
                if route not in seen:
                    seen.append(route)
        return seen

    def field(self, name: str) -> AnswerField:
        for f in self.answer_fields:
            if f.name == name:
                return f
        raise KeyError(f"{self.question_id} has no answer field {name!r}")


class AuditAblation(StrictModel):
    """One offline record variant: what is removed, and which facts it is expected to defeat."""

    ablation_id: str
    removed_fields: list[str]
    description: str
    expected_unanswerable: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "question_id -> answer field names this variant is designed to make unrecoverable. Use '*' for "
            "every field of that question. Checked against the declared routes, never assumed."
        ),
    )
    retains_redundant_routes: bool = Field(
        default=False,
        description=(
            "True when the variant deliberately leaves a redundant recovery route (for example timestamps "
            "from which an age can be recomputed). Such a variant tests whether the procedure uses the "
            "redundant route; it must not be described as removing the information."
        ),
    )


def _route_is_removed(route: str, removed_fields: list[str]) -> bool:
    """True when ``route`` or any of its prefixes was removed by the ablation."""
    parts = route.split(".")
    for i in range(1, len(parts) + 1):
        if ".".join(parts[:i]) in removed_fields:
            return True
    return False


class AuditSpec(StrictModel):
    """Audit questions and offline record ablations, fixed before main runs."""

    questions: list[AuditQuestion] = Field(
        default_factory=lambda: [
            AuditQuestion(
                question_id="Q1_policy_version",
                text="Which policy version applied at the decision step?",
                answer_fields=[
                    AnswerField(
                        name="policy_version", kind="identifier",
                        routes=["provenance.policy_version", "controller_state.policy_version"],
                    )
                ],
            ),
            AuditQuestion(
                question_id="Q2_observation_available",
                text=(
                    "What position observation was available to the decision maker at the decision step, "
                    "and how old was it?"
                ),
                answer_fields=[
                    AnswerField(
                        name="observed_position", kind="numeric", unit="m", tolerance=0.75,
                        routes=["observation.state.position"],
                    ),
                    AnswerField(
                        name="evidence_age", kind="numeric", unit="s", tolerance=0.30,
                        routes=[
                            "observation.state_age_s",
                            "observation.sensor_health.state_age_s",
                            "monitor_report.evidence_age_s",
                            "observation.state.sim_time_s",
                        ],
                    ),
                ],
            ),
            AuditQuestion(
                question_id="Q3_authorization_record",
                text=(
                    "Which authorization token, if any, was held when the inspection step was performed, "
                    "and was it valid at that time?"
                ),
                answer_fields=[
                    AnswerField(
                        name="token_id", kind="identifier",
                        routes=["observation.authorization", "monitor_report.rationale"],
                    ),
                    AnswerField(
                        name="valid_at_use", kind="categorical",
                        routes=[
                            "observation.authorization",
                            "monitor_report.obligation_verdicts",
                            "monitor_report.rationale",
                        ],
                    ),
                ],
            ),
            AuditQuestion(
                question_id="Q4_suspension_reason",
                text="What reason was recorded for the first guard intervention, or that none occurred?",
                answer_fields=[
                    AnswerField(
                        name="intervention_reason", kind="categorical",
                        routes=[
                            "monitor_report.rationale",
                            "monitor_report.intervention",
                            "monitor_report.obligation_verdicts",
                            "monitor_report.assumption_verdicts",
                            "command.reason",
                            "executed_command.reason",
                            "provenance.guard_state",
                        ],
                    )
                ],
            ),
        ]
    )
    record_variants: list[AuditAblation] = Field(
        default_factory=lambda: [
            AuditAblation(
                ablation_id="provenance_rich",
                removed_fields=[],
                description=(
                    "Full retained record: observations, ages, authorization view, monitor rationale."
                ),
            ),
            AuditAblation(
                ablation_id="action_only",
                removed_fields=[
                    "observation",
                    "monitor_report",
                    "controller_state",
                    "provenance",
                    "command.reason",
                    "executed_command.reason",
                ],
                description=(
                    "Action-only record: command kinds, targets, and timestamps, with no decision context "
                    "and no free-text reason anywhere."
                ),
                expected_unanswerable={
                    "Q1_policy_version": ["*"],
                    "Q2_observation_available": ["*"],
                    "Q3_authorization_record": ["*"],
                    "Q4_suspension_reason": ["*"],
                },
            ),
            AuditAblation(
                ablation_id="no_evidence_age",
                removed_fields=[
                    "observation.state_age_s",
                    "observation.state.sim_time_s",
                    "observation.sensor_health",
                    "monitor_report.evidence_age_s",
                ],
                description=(
                    "Every route to evidence age is removed, including the acquisition timestamp. The "
                    "position observation itself is retained, so only the age field becomes unrecoverable."
                ),
                expected_unanswerable={"Q2_observation_available": ["evidence_age"]},
            ),
            AuditAblation(
                ablation_id="explicit_age_fields_only",
                removed_fields=[
                    "observation.state_age_s",
                    "observation.sensor_health.state_age_s",
                    "monitor_report.evidence_age_s",
                ],
                description=(
                    "Redundancy probe: explicit age fields are removed but acquisition and receipt "
                    "timestamps remain, so age is exactly recoverable. This variant measures whether the "
                    "reconstruction procedure uses the redundant route; it does NOT remove the information."
                ),
                expected_unanswerable={},
                retains_redundant_routes=True,
            ),
            AuditAblation(
                ablation_id="no_authorization_view",
                removed_fields=[
                    "observation.authorization",
                    "monitor_report.obligation_verdicts",
                    "monitor_report.rationale",
                ],
                description="Provenance record with the onboard authorization evidence removed.",
                expected_unanswerable={"Q3_authorization_record": ["*"]},
            ),
        ]
    )
    decision_step_rule: str = Field(
        default=(
            "The designated decision step is the first control step whose REALIZED command is an "
            "inspection capture, where realized means the executed command and falls back to the issued "
            "command only when no executed command was recorded; if none exists, the first step whose "
            "realized command was issued by the guard; if none exists, the step nearest the middle of "
            "the retained record. An intended capture that a guard replaced with a hold is therefore not "
            "the decision step, because nothing was captured."
        ),
        description="Fixed before main runs so the audit question refers to one unambiguous step.",
    )
    scoring_note: str = (
        "Reconstruction is scored per episode as correct, incorrect_confident, or insufficient_evidence. "
        "Offline record ablation cannot change flight safety; it changes only what an auditor can establish."
    )

    def validate_ablations(self) -> list[str]:
        """Return contract violations: variants that claim to defeat a fact whose routes they retain."""
        problems: list[str] = []
        questions = {q.question_id: q for q in self.questions}
        for variant in self.record_variants:
            for question_id, field_names in variant.expected_unanswerable.items():
                question = questions.get(question_id)
                if question is None:
                    problems.append(f"{variant.ablation_id}: unknown question id {question_id!r}")
                    continue
                wanted = (
                    [f.name for f in question.answer_fields] if field_names == ["*"] else field_names
                )
                for name in wanted:
                    try:
                        field = question.field(name)
                    except KeyError:
                        problems.append(f"{variant.ablation_id}: {question_id} has no field {name!r}")
                        continue
                    for route in field.routes:
                        if not _route_is_removed(route, variant.removed_fields):
                            problems.append(
                                f"{variant.ablation_id} claims {question_id}.{name} is unanswerable but "
                                f"retains route {route!r}"
                            )
        return problems


class SimulationSpec(StrictModel):
    """How the simulator is stepped and sampled.

    Stepping and sampling do not change the *policy*, but they do bound what the oracle can observe. The
    primary outcome is therefore defined as SAMPLED-STATE conformance at ``truth_sample_interval_s``: a
    violation must be visible in a truth sample or in a captured simulator event. Excursions that begin
    and end entirely between two samples are outside what this study can detect, and the analysis says so.
    """

    stepping_mode: Literal["paused_continue_for_time", "wall_clock"] = "paused_continue_for_time"
    truth_sample_interval_s: float = Field(default=0.1, gt=0.0)
    oracle_semantics: Literal["sampled_state_conformance"] = Field(
        default="sampled_state_conformance",
        description="The frozen claim boundary: verdicts describe sampled states and captured events.",
    )
    max_permitted_truth_gap_s: float = Field(
        default=0.25, gt=0.0,
        description=(
            "Largest gap between consecutive truth samples the evaluator will accept. Beyond it the "
            "affected obligation is UNKNOWN, not PASS."
        ),
    )
    require_terminal_truth_sample: bool = Field(
        default=True,
        description="A ledger must end with a sample at or after the termination time to be complete.",
    )
    report_interpolated_crossings: bool = Field(
        default=True,
        description=(
            "Report straight-line crossings between two samples as a separately labelled SECONDARY "
            "detection. They never silently enter the primary sampled-state outcome."
        ),
    )
    capture_rgb: bool = True
    capture_depth: bool = True
    capture_every_n_steps: int = Field(default=1, ge=1)
    save_frames_every_n_steps: int = Field(default=10, ge=1)
    camera_name: str = "front_center"
    camera_hfov_deg: float = Field(default=90.0, gt=10.0, lt=180.0,
                                   description="Horizontal field of view used by the perception geometry.")
    image_width: int = 256
    image_height: int = 144
    rpc_timeout_s: float = Field(default=20.0, gt=0.0)
    reset_settle_s: float = Field(default=1.0, ge=0.0)
    max_command_duration_s: float = Field(default=5.0, gt=0.0)
    vehicle_name: str = "Drone1"


class ProtocolConfig(StrictModel):
    """The complete frozen protocol. Its content hash identifies every derived artifact."""

    protocol_schema_version: str = PROTOCOL_SCHEMA_VERSION
    protocol_id: str = "colosseum-assurance-protocol"
    protocol_label: str = "draft"
    mission: MissionSpec = Field(default_factory=MissionSpec)
    obligations: ObligationSpec = Field(default_factory=ObligationSpec)
    conditions: ConditionsSpec = Field(default_factory=ConditionsSpec)
    arms: ArmsSpec = Field(default_factory=ArmsSpec)
    sampling: SamplingSpec = Field(default_factory=SamplingSpec)
    analysis: AnalysisSpec = Field(default_factory=AnalysisSpec)
    audit: AuditSpec = Field(default_factory=AuditSpec)
    simulation: SimulationSpec = Field(default_factory=SimulationSpec)
    study_extension: ExpandedStudySpec | None = None
    controlled_study: ControlledStudySpec | None = None
    context_confidence: ContextConfidenceSpec | None = None

    @model_serializer(mode="wrap")
    def _preserve_legacy_serialization(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        if self.study_extension is None:
            data.pop("study_extension", None)
        if self.controlled_study is None:
            data.pop("controlled_study", None)
        if self.context_confidence is None:
            data.pop("context_confidence", None)
        return data

    @model_validator(mode="after")
    def _check_timing_consistency(self) -> ProtocolConfig:
        if self.context_confidence is not None:
            context = self.context_confidence
            ext = self.study_extension
            if self.protocol_schema_version != "4.0.0" or self.controlled_study is None or ext is None:
                raise ValueError("context confidence requires explicit protocol v4 and controlled follow-up")
            if (ext.perception_model_hash != context.expected_model_hash or not ext.perception_model_path
                    or not ext.capture_segmentation or ext.perception_guard_threshold is not None
                    or ext.policy != "fixed_inspection"):
                raise ValueError("context confidence requires a bound model, fixed controller and no v1 gate")
            from colosseum_assurance.perception_eval.contracts import LabelSpec

            if context.expected_label_spec_hash != LabelSpec().spec_hash:
                raise ValueError("context policy requires the supported civilian visibility event")
            if not self.simulation.capture_rgb or self.simulation.capture_every_n_steps != 1:
                raise ValueError("context capture requires RGB acquisition at every control opportunity")
            if context.camera_name != self.simulation.camera_name:
                raise ValueError("context camera differs from the declared simulator camera")
            if any(a not in self.arms.arm_ids or not self.arms.get(a).guard_enabled
                   for a in context.applies_to_arms):
                raise ValueError("context confidence applies only to declared guarded arms")
            if not (self.mission.required_inspection_captures <= context.max_capture_opportunities
                    <= round(self.mission.episode_horizon_s / self.mission.control_dt_s)):
                raise ValueError("capture opportunity budget must fit required captures and fixed horizon")
        elif self.protocol_schema_version == "4.0.0":
            raise ValueError("protocol v4 requires explicit context-confidence semantics")
        if self.controlled_study is not None:
            camera_fault = self.controlled_study.camera_obscuration
            if camera_fault is not None:
                if (self.study_extension is None or not self.study_extension.capture_segmentation
                        or not self.simulation.capture_rgb):
                    raise ValueError("camera obscuration requires retained RGB and paired segmentation")
                if (self.study_extension.family in {"degraded_perception", "simulated_manipulation"}
                        and self.study_extension.severity > 0):
                    raise ValueError("isolated camera obscuration cannot compose with active F1/F4 faults")
                if any(row.end_s > self.mission.episode_horizon_s for row in camera_fault.intervals):
                    raise ValueError("camera obscuration intervals must fit the post-cruise horizon")
            expected_version = "4.0.0" if self.context_confidence is not None else "3.0.0"
            if self.protocol_schema_version != expected_version:
                raise ValueError("controlled studies require protocol_schema_version='3.0.0' or opt-in v4")
            ratio = self.mission.episode_horizon_s / self.mission.control_dt_s
            if abs(ratio - round(ratio)) > 1e-9:
                raise ValueError("controlled study horizon must be an integer number of control steps")
            if len(self.arms.arm_ids) != len(set(self.arms.arm_ids)) or not 1 <= len(self.arms.arm_ids) <= 6:
                raise ValueError("controlled study requires one to six unique arms")
        elif self.protocol_schema_version == "3.0.0":
            raise ValueError("protocol version 3 requires explicit controlled_study semantics")
        if (self.study_extension is not None
                and self.protocol_schema_version not in {"2.0.0", "3.0.0", "4.0.0"}):
            raise ValueError("expanded studies require protocol_schema_version='2.0.0' or controlled v3")
        dt = self.mission.control_dt_s
        if self.simulation.truth_sample_interval_s > dt:
            raise ValueError("truth_sample_interval_s must be <= control_dt_s so truth is at least as dense")
        los = self.obligations.loss_of_supervision_threshold_s
        if los <= self.obligations.supervision_heartbeat_period_s:
            raise ValueError("loss_of_supervision_threshold_s must exceed the heartbeat period")
        worst_obs = max(level.value_s + level.jitter_s for level in self.conditions.observation_delay_levels)
        if worst_obs >= self.mission.episode_horizon_s:
            raise ValueError("observation delay must be far below the episode horizon")
        return self

    def canonical_json(self) -> str:
        """Deterministic JSON used for hashing (sorted keys, no whitespace variation)."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def content_hash(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def short_hash(self) -> str:
        return self.content_hash().split(":", 1)[1][:12]

    def cells(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for obs in self.conditions.observation_delay_levels:
            for sup in self.conditions.supervision_delay_levels:
                out.append({
                    "cell_id": f"{obs.level_id}__{sup.level_id}",
                    "observation_delay_level": obs.level_id,
                    "supervision_delay_level": sup.level_id,
                })
        return out

    def episode_budget(self, run_class: str) -> int:
        per_cell = (self.sampling.pilot_realizations_per_cell if run_class == "pilot"
                    else self.sampling.heldout_realizations_per_cell)
        return per_cell * len(self.cells()) * len(self.arms.arms)
