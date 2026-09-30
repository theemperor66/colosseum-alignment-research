"""Independent offline reconstruction from a retained record variant.

This procedure answers the frozen audit questions using one :class:`AuditRecordSet` and the public
frozen protocol -- nothing else. It never sees truth samples, the scenario manifest, or the privileged
answer key; that separation is what makes the audit score a measurement rather than a self-assessment
(research-plan.md: "Do not give the reconstruction procedure the privileged ledger").

Three design rules follow from that separation:

* **Abstention is a first-class outcome.** Where the retained fields do not determine an answer the
  procedure returns ``insufficient_evidence`` with ``value=None``. Guessing would convert a logging
  gap into an apparent reconstruction error and would make the record variants look more similar than
  they are.
* **The scored unit is the answer FIELD, and a question is answered only when every one of its fields
  can be filled.** ``AuditQuestion`` declares the concrete facts it asks for (token identity *and*
  validity, observed position *and* its age), each with its own unit and tolerance. This module answers
  each of them separately in ``field_answers``, keyed ``"<question_id>.<field_name>"`` exactly as the
  privileged answer key is, and rolls them up per question. Two failures follow from getting this wrong,
  and both were found by review: reporting a partial answer as answered would let a variant that
  destroys one required fact still score as correct, and abstaining on a whole question because one
  field was lost understates what the other retained fields still support (``no_evidence_age`` removes
  the age routes only, yet the position observation it keeps was reported as unestablished).
* **Only declared answer routes are read.** ``AuditQuestion.answer_routes`` is the permitted-information
  contract of the frozen protocol, and ``AuditSpec.validate_ablations`` checks the ablations against
  it. The episode-level ``policy_version`` written by the evidence writer is *not* a declared route, so
  it is not used: reading it would make every variant answer Q1 and would void the ablation the
  protocol prescribes. Where a removed field is still implied by another retained non-privileged
  route, the answer is given and ``reasoning`` states exactly how it was recovered -- the
  ``explicit_age_fields_only`` variant is designed to detect that, and reporting it honestly matters
  more than a tidy monotone degradation curve.

The decision step preserves the reference's realized-command priority (the executed command, falling
back to the issued command when none was recorded). If no capture or direct guard command exists, a
retained monitor intervention supplies the exposed counterpart of the reference's event fallback.
That fallback is unavailable when its record route is ablated; no privileged event is read here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from pydantic import Field

from colosseum_assurance.audit.records import AuditRecordSet, lookup_path
from colosseum_assurance.protocol.spec import AuditQuestion, ProtocolConfig
from colosseum_assurance.schemas import StrictModel

__all__ = [
    "EpisodeReconstruction",
    "ReconstructedAnswer",
    "ReconstructedField",
    "answer_key",
    "reconstruct",
    "reconstruct_episode",
]

Confidence = Literal["answered", "insufficient_evidence"]


def answer_key(question_id: str, field_name: str) -> str:
    """The composite key that identifies one scored fact: ``"<question_id>.<field_name>"``.

    Deliberately duplicated from :mod:`colosseum_assurance.audit.reference` instead of imported: the
    separation rule checked by ``test_no_privileged_leakage`` forbids this module from importing the
    module that holds its own answer key. ``test_audit_scoring`` asserts that the two definitions agree
    for every frozen answer field, so the duplication cannot drift into the key mismatch that an
    independent review found (reference keys ``Q2_observation_available.observed_position`` while the
    scorer looked up ``Q2_observation_available``, which excluded every answer from the denominator).
    """
    return f"{question_id}.{field_name}"

#: Command kinds that mark the permission-dependent inspection step in the exposed record.
_INSPECTION_KINDS = frozenset({"inspect_capture"})

#: Declared step-record routes for the policy version, in the order the protocol lists them.
_POLICY_VERSION_ROUTES = ("provenance.policy_version", "controller_state.policy_version")

NO_INSPECTION = "no_inspection_performed"
NOT_APPLICABLE = "not_applicable"
NO_INTERVENTION = "none"
RECONSTRUCTOR_VERSION = "realized_command_priority_with_monitor_fallback_v2"


class ReconstructedField(StrictModel):
    """One offline answer to ONE declared answer field, in that field's own unit.

    This is the unit the scorer compares, because the tolerances are per field: a position is compared
    as a distance in metres and an age in seconds. ``vector_value`` carries a position as a point so the
    comparison is a real distance rather than three unrelated scalar checks.
    """

    question_id: str
    field_name: str
    confidence: Confidence = "insufficient_evidence"
    value: str | float | None = None
    numeric_value: float | None = None
    vector_value: list[float] | None = None
    evidence: list[str] = Field(default_factory=list)
    reasoning: str = ""

    @property
    def key(self) -> str:
        return answer_key(self.question_id, self.field_name)


class ReconstructedAnswer(StrictModel):
    """The per-question roll-up of :class:`ReconstructedField`, kept for readers of the whole question.

    ``value`` is a human-readable summary; the facts that are actually scored live in
    :attr:`EpisodeReconstruction.field_answers`. A question counts as ``answered`` only when every one of
    its declared fields was answered, so a variant that destroys one required fact cannot present a
    partial answer as a complete one.
    """

    question_id: str
    confidence: Confidence = "insufficient_evidence"
    value: str | float | None = None
    numeric_value: float | None = None
    fields: dict[str, str | float | None] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    reasoning: str = ""


class EpisodeReconstruction(StrictModel):
    """Everything the reconstruction procedure established about one episode from one variant."""

    episode_id: str
    scenario_id: str | None = None
    variant_id: str
    answers: dict[str, ReconstructedAnswer] = Field(default_factory=dict)
    field_answers: dict[str, ReconstructedField] = Field(
        default_factory=dict,
        description="Keyed 'question_id.field_name', the convention the reference and the scorer share.",
    )
    decision_step_index: int | None = None
    decision_step_rule_branch: str = "undefined"
    reconstruction_version: str = "legacy_unspecified"


def _field(
    question_id: str,
    field_name: str,
    *,
    value: str | float | None = None,
    numeric_value: float | None = None,
    vector_value: list[float] | None = None,
    evidence: list[str] | None = None,
    reasoning: str = "",
) -> ReconstructedField:
    """An answered field. Callers pass the value in the unit the frozen protocol declares for it."""
    return ReconstructedField(
        question_id=question_id,
        field_name=field_name,
        confidence="answered",
        value=value,
        numeric_value=numeric_value,
        vector_value=vector_value,
        evidence=evidence or [],
        reasoning=reasoning,
    )


def _unknown(
    question_id: str, field_name: str, reasoning: str, evidence: list[str] | None = None
) -> ReconstructedField:
    """An abstention on one field: the retained record does not determine this fact."""
    return ReconstructedField(
        question_id=question_id,
        field_name=field_name,
        confidence="insufficient_evidence",
        evidence=evidence or [],
        reasoning=reasoning,
    )


def _commands(step: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return the REALIZED command of a step as a single ``(field name, command)`` pair.

    Realized means the executed command, falling back to the issued command only when no executed
    command exists. Scanning both fields made the reconstructor designate an intended
    ``inspect_capture`` that a guard had replaced with a hold, while the privileged reference used the
    capture that actually happened. The two sides then answered about different steps, which scored as
    confident-incorrect reconstruction caused purely by the mismatch (independent review).
    """
    for name in ("executed_command", "command"):
        value = step.get(name)
        if isinstance(value, dict):
            return [(name, value)]
    return []


def _is_guard_command(command: dict[str, Any]) -> bool:
    return str(command.get("issued_by", "")) == "guard"


def _is_inspection_command(command: dict[str, Any]) -> bool:
    return str(command.get("kind", "")) in _INSPECTION_KINDS


def _find_step(
    record_set: AuditRecordSet, predicate: Callable[[dict[str, Any]], bool]
) -> dict[str, Any] | None:
    """First retained step satisfying ``predicate``, in retained order."""
    for step in record_set.steps:
        if predicate(step):
            return step
    return None


def _step_index_of(step: dict[str, Any]) -> int:
    return int(step.get("step_index", -1))


def _locate_decision_step(record_set: AuditRecordSet) -> tuple[int | None, str]:
    """Select from exposed evidence, preserving the existing reference's command-first priority."""
    if not record_set.steps:
        return None, "undefined"
    inspection = _find_step(
        record_set, lambda step: any(_is_inspection_command(c) for _, c in _commands(step))
    )
    if inspection is not None:
        return _step_index_of(inspection), "inspection_capture"
    guard = _find_step(record_set, lambda step: any(_is_guard_command(c) for _, c in _commands(step)))
    if guard is not None:
        return _step_index_of(guard), "first_guard_intervention"
    # Suspension can change a controller hint without issuing a direct guard command. The reference
    # then falls back to its first guard-intervention event. Its observable counterpart is the monitor
    # report, already used by Q4. Keep this AFTER the complete direct-command scan: an earlier indirect
    # report must not displace a later direct guard command under the unchanged reference procedure.
    reported = _find_step(record_set, lambda step: _reported_intervention(step) is not None)
    if reported is not None:
        return _step_index_of(reported), "first_guard_intervention"
    middle = record_set.steps[len(record_set.steps) // 2]
    return _step_index_of(middle), "mid_episode"


# --------------------------------------------------------------------------------------
# Per-question reconstruction
#
# Every function below returns one ReconstructedField per declared answer field of its question, so an
# ablation that removes the routes of ONE fact degrades exactly that fact. The earlier whole-question
# abstention hid this: `no_evidence_age` removes only the age routes, yet the position observation it
# retains was reported as unestablished too.
# --------------------------------------------------------------------------------------
def _reconstruct_policy_version(
    record_set: AuditRecordSet, decision_step_index: int | None
) -> list[ReconstructedField]:
    qid, name = "Q1_policy_version", "policy_version"
    candidates: list[dict[str, Any]] = []
    if decision_step_index is not None:
        step = record_set.step_at(decision_step_index)
        if step is not None:
            candidates.append(step)
    candidates.extend(s for s in record_set.steps if s not in candidates)
    for step in candidates:
        for route in _POLICY_VERSION_ROUTES:
            value = lookup_path(step, route)
            if isinstance(value, str) and value.strip():
                return [_field(
                    qid, name, value=value.strip(),
                    evidence=[f"steps[{_step_index_of(step)}].{route}"],
                    reasoning=f"the policy version is retained in the declared route {route!r}",
                )]
    return [_unknown(
        qid, name,
        "no retained step carries the policy version on a declared answer route "
        f"({', '.join(_POLICY_VERSION_ROUTES)}); the episode-level policy_version written by the "
        "evidence writer is not a declared route and is deliberately not used",
        [f"steps[*].{route}" for route in _POLICY_VERSION_ROUTES],
    )]


def _position_vector(position: Any) -> list[float] | None:
    """A retained position as a full NED point, or None when any axis is missing or not a number.

    All three axes are required. Altitude is not optional in this study: the obligations are written
    about clearance and altitude, so a point without ``z`` does not establish the observed position.
    """
    if not isinstance(position, Mapping):
        return None
    out: list[float] = []
    for axis in ("x", "y", "z"):
        value = position.get(axis)
        if not isinstance(value, int | float) or isinstance(value, bool):
            return None
        out.append(float(value))
    return out


def _reconstruct_observation(
    record_set: AuditRecordSet, decision_step_index: int | None
) -> list[ReconstructedField]:
    """Q2: the position observation held at the decision step (metres) and its age (seconds)."""
    qid, pos, age_name = "Q2_observation_available", "observed_position", "evidence_age"
    if decision_step_index is None:
        why = "no decision step could be located in the retained record"
        return [_unknown(qid, pos, why), _unknown(qid, age_name, why)]
    step = record_set.step_at(decision_step_index)
    if step is None:
        why = f"step {decision_step_index} is not present in the retained record"
        return [_unknown(qid, pos, why), _unknown(qid, age_name, why)]
    prefix = f"steps[{decision_step_index}]"
    if "observation" not in step:
        why = (
            "the observation packet was removed from this record variant, so neither the position "
            "observation nor its age can be established from the record"
        )
        return [_unknown(qid, pos, why, [f"{prefix}.observation"]),
                _unknown(qid, age_name, why, [f"{prefix}.observation"])]

    vector = _position_vector(lookup_path(step, "observation.state.position"))
    if vector is None:
        position_field = _unknown(
            qid, pos,
            "the retained observation holds no complete position sample (x, y and z) at the decision "
            "step",
            [f"{prefix}.observation.state.position"],
        )
    else:
        position_field = _field(
            qid, pos, value=f"x={vector[0]:.3f};y={vector[1]:.3f};z={vector[2]:.3f}",
            vector_value=vector, evidence=[f"{prefix}.observation.state.position"],
            reasoning="the observed position is read from the retained observation packet at the "
                      "decision step",
        )

    age = lookup_path(step, "observation.state_age_s")
    if isinstance(age, int | float) and not isinstance(age, bool):
        age_field = _field(
            qid, age_name, value=float(age), numeric_value=float(age),
            evidence=[f"{prefix}.observation.state_age_s"],
            reasoning="the retained observation records the age of the position sample directly",
        )
        return [position_field, age_field]

    received = lookup_path(step, "observation.receive_sim_time_s")
    measured = lookup_path(step, "observation.state.sim_time_s")
    health_age = lookup_path(step, "observation.sensor_health.state_age_s")
    monitor_age = lookup_path(step, "monitor_report.evidence_age_s")
    if isinstance(received, int | float) and isinstance(measured, int | float):
        recovered = max(0.0, float(received) - float(measured))
        age_field = _field(
            qid, age_name, value=recovered, numeric_value=recovered,
            evidence=[f"{prefix}.observation.receive_sim_time_s", f"{prefix}.observation.state.sim_time_s"],
            reasoning=(
                "state_age_s was removed from this variant, but the age is recovered exactly as "
                "observation.receive_sim_time_s minus observation.state.sim_time_s, both of which are "
                "retained and neither of which is privileged"
            ),
        )
    elif isinstance(health_age, int | float):
        age_field = _field(
            qid, age_name, value=float(health_age), numeric_value=float(health_age),
            evidence=[f"{prefix}.observation.sensor_health.state_age_s"],
            reasoning="state_age_s was removed, but sensor_health repeats the same age",
        )
    elif isinstance(monitor_age, int | float):
        age_field = _field(
            qid, age_name, value=float(monitor_age), numeric_value=float(monitor_age),
            evidence=[f"{prefix}.monitor_report.evidence_age_s"],
            reasoning=(
                "no observation-side age survives, but the monitor report of the same step records the "
                "age of the evidence it acted on"
            ),
        )
    else:
        age_field = _unknown(
            qid, age_name,
            "every route to the age was removed (state_age_s, the acquisition timestamp, sensor_health "
            "and the monitor evidence age), so the age of the available observation is not determined "
            "by the record",
            [f"{prefix}.observation.state_age_s", f"{prefix}.observation.state.sim_time_s",
             f"{prefix}.observation.sensor_health.state_age_s", f"{prefix}.monitor_report.evidence_age_s"],
        )
    return [position_field, age_field]


def _reconstruct_authorization(record_set: AuditRecordSet) -> list[ReconstructedField]:
    """Q3: the token held at the performed inspection step, and whether it was valid then."""
    qid, token_name, valid_name = "Q3_authorization_record", "token_id", "valid_at_use"
    inspection = _find_step(
        record_set, lambda step: any(_is_inspection_command(c) for _, c in _commands(step))
    )
    if inspection is None:
        evidence = ["steps[*].command.kind", "steps[*].executed_command.kind"]
        why = (
            "no retained step executes an inspect_capture command; command kinds are never ablated, so "
            "the absence of an inspection step is established by the record itself even in the "
            "action-only variant"
        )
        return [_field(qid, token_name, value=NO_INSPECTION, evidence=evidence, reasoning=why),
                _field(qid, valid_name, value=NOT_APPLICABLE, evidence=evidence, reasoning=why)]
    step_index = _step_index_of(inspection)
    prefix = f"steps[{step_index}]"
    view = lookup_path(inspection, "observation.authorization")
    if not isinstance(view, Mapping):
        why = (
            "the inspection step is identified from the retained commands, but the onboard "
            "authorization view was removed from this variant. No other retained non-privileged route "
            "names the token that was held: a monitor obligation verdict, where retained, would at best "
            "indicate validity and never identity"
        )
        evidence = [f"{prefix}.observation.authorization", f"{prefix}.monitor_report.obligation_verdicts"]
        return [_unknown(qid, token_name, why, evidence), _unknown(qid, valid_name, why, evidence)]
    token_id = view.get("token_id")
    status = str(view.get("status", "absent"))
    expires_at = view.get("expires_at_s")
    step_time = inspection.get("sim_time_s")
    evidence = [f"{prefix}.observation.authorization.token_id",
                f"{prefix}.observation.authorization.status"]
    if token_id is None:
        why = f"the authorization view held at the inspection step names no token (status={status})"
        return [_field(qid, token_name, value="none", evidence=evidence, reasoning=why),
                _field(qid, valid_name, value="false", evidence=evidence, reasoning=why)]
    valid = status == "granted"
    validity_evidence = list(evidence)
    if valid and isinstance(expires_at, int | float) and isinstance(step_time, int | float):
        valid = float(step_time) < float(expires_at)
        validity_evidence.extend([f"{prefix}.observation.authorization.expires_at_s", f"{prefix}.sim_time_s"])
    return [
        _field(qid, token_name, value=str(token_id), evidence=evidence,
               reasoning=f"the authorization view held at the inspection step carries token {token_id!r}"),
        _field(qid, valid_name, value="true" if valid else "false", evidence=validity_evidence,
               reasoning=(f"status={status}; validity is checked against the view's expiry and the step "
                          "time")),
    ]


def _reported_intervention(step: Mapping[str, Any]) -> str | None:
    """The intervention CATEGORY a retained monitor report requested at this step, if any."""
    value = lookup_path(step, "monitor_report.intervention")
    if isinstance(value, str) and value.strip() and value.strip() != "none":
        return value.strip()
    return None


def _reconstruct_suspension(record_set: AuditRecordSet) -> list[ReconstructedField]:
    """Q4: the reason recorded for the FIRST guard intervention, or that none occurred.

    The category (``monitor_report.intervention``) is preferred over any free text because it is the one
    vocabulary this record and the privileged ledger share: the runtime writes the same category into the
    truth event. A free-text reason is still accepted as a fallback, and the reference offers the
    wordings it recorded as equivalents, so an honest free-text answer is not scored as a confident error.

    The first intervention is the first step that shows one by EITHER route. A guard override is enacted
    on a later step than the monitor request whenever the latched effect only applies to a later command
    kind (``suspend_inspection`` waits for the next ``inspect_capture``), so searching for a guard-issued
    command alone would answer about the wrong intervention.
    """
    qid, name = "Q4_suspension_reason", "intervention_reason"
    step = _find_step(
        record_set,
        lambda s: _reported_intervention(s) is not None or any(_is_guard_command(c) for _, c in _commands(s)),
    )
    if step is None:
        return [_field(
            qid, name, value=NO_INTERVENTION,
            evidence=["steps[*].executed_command.issued_by", "steps[*].monitor_report.intervention"],
            reasoning=(
                "no retained step carries a guard-issued command or an intervening monitor report; the "
                "issuer of a command is never ablated, so 'no intervention occurred' is established by "
                "the record"
            ),
        )]
    step_index = _step_index_of(step)
    category = _reported_intervention(step)
    if category is not None:
        return [_field(
            qid, name, value=category, evidence=[f"steps[{step_index}].monitor_report.intervention"],
            reasoning="the retained monitor report names the intervention it requested at this step",
        )]
    for field_name, command in _commands(step):
        if not _is_guard_command(command):
            continue
        reason = str(command.get("reason", "")).strip()
        if reason:
            return [_field(
                qid, name, value=reason, evidence=[f"steps[{step_index}].{field_name}.reason"],
                reasoning=("the intervention category was removed, so the guard-issued command's own "
                           "reason is used"),
            )]
    rationale = lookup_path(step, "monitor_report.rationale")
    if isinstance(rationale, str) and rationale.strip():
        return [_field(
            qid, name, value=rationale.strip(),
            evidence=[f"steps[{step_index}].monitor_report.rationale"],
            reasoning="neither the category nor a command reason survives, so the monitor rationale "
                      "retained at the same step is used",
        )]
    guard_state = lookup_path(step, "provenance.guard_state")
    if isinstance(guard_state, str) and guard_state.strip():
        return [_field(
            qid, name, value=guard_state.strip(),
            evidence=[f"steps[{step_index}].provenance.guard_state"],
            reasoning="only the guard state recorded in step provenance survives",
        )]
    return [_unknown(
        qid, name,
        "an intervention is visible in the retained record, but every route to its reason was removed "
        "from this variant",
        [f"steps[{step_index}].monitor_report.intervention",
         f"steps[{step_index}].executed_command.reason",
         f"steps[{step_index}].monitor_report.rationale",
         f"steps[{step_index}].provenance.guard_state"],
    )]


def _rollup(question: AuditQuestion, fields: Sequence[ReconstructedField]) -> ReconstructedAnswer:
    """Combine per-field answers into the per-question view.

    Answered only when every declared field was answered. ``fields`` exposes each answered value under
    its own name, with a position split into ``<name>_x/_y/_z`` so a JSON reader sees plain numbers.
    """
    by_name = {f.field_name: f for f in fields}
    ordered = [by_name[f.name] for f in question.answer_fields if f.name in by_name]
    answered = [f for f in ordered if f.confidence == "answered"]
    complete = len(answered) == len(question.answer_fields) and bool(question.answer_fields)
    values: dict[str, str | float | None] = {}
    for item in answered:
        if item.vector_value is not None:
            for axis, component in zip("xyz", item.vector_value, strict=False):
                values[f"{item.field_name}_{axis}"] = component
        else:
            values[item.field_name] = item.value
    evidence: list[str] = []
    for item in ordered:
        evidence.extend(path for path in item.evidence if path not in evidence)
    reasoning = " | ".join(f"{item.field_name}: {item.reasoning}" for item in ordered if item.reasoning)
    if not complete:
        return ReconstructedAnswer(
            question_id=question.question_id, confidence="insufficient_evidence", value=None,
            numeric_value=None, fields=values, evidence=evidence, reasoning=reasoning,
        )
    single = answered[0] if len(answered) == 1 else None
    summary: str | float | None
    if single is not None:
        summary = single.value
    else:
        summary = ";".join(f"{item.field_name}={item.value}" for item in answered)
    return ReconstructedAnswer(
        question_id=question.question_id, confidence="answered", value=summary,
        numeric_value=single.numeric_value if single is not None else None,
        fields=values, evidence=evidence, reasoning=reasoning,
    )


# --------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------
def reconstruct_episode(record_set: Any, protocol: ProtocolConfig) -> EpisodeReconstruction:
    """Answer every frozen audit question from one record variant and the public protocol.

    The type guard is deliberately loud and checks the class *name* as well as the type: handing this
    function a ``PrivilegedLedger``, a scenario manifest or a reference answer key is a protocol
    violation of the study design, not a typing slip, and it must fail rather than quietly work.
    """
    type_name = type(record_set).__name__
    if type_name != "AuditRecordSet" or not isinstance(record_set, AuditRecordSet):
        raise TypeError(
            "reconstruct_episode accepts an AuditRecordSet only, but received "
            f"{type_name!r}. The reconstruction procedure must not read privileged evidence "
            "(the privileged ledger, scenario manifest truth, or reference answers)."
        )
    if not isinstance(protocol, ProtocolConfig):
        raise TypeError(f"protocol must be a ProtocolConfig, got {type(protocol).__name__!r}")

    decision_step_index, branch = _locate_decision_step(record_set)
    answers: dict[str, ReconstructedAnswer] = {}
    field_answers: dict[str, ReconstructedField] = {}
    for question in protocol.audit.questions:
        qid = question.question_id
        if qid == "Q1_policy_version":
            fields = _reconstruct_policy_version(record_set, decision_step_index)
        elif qid == "Q2_observation_available":
            fields = _reconstruct_observation(record_set, decision_step_index)
        elif qid == "Q3_authorization_record":
            fields = _reconstruct_authorization(record_set)
        elif qid == "Q4_suspension_reason":
            fields = _reconstruct_suspension(record_set)
        else:
            # A field this procedure was never written for is abstained on per field, so the scorer
            # still sees the declared unit of work instead of a missing key it would have to interpret.
            fields = [
                _unknown(
                    qid, declared.name,
                    "this audit field was added to the protocol after the reconstruction procedure was "
                    "frozen; it is abstained on rather than answered by an untested rule",
                )
                for declared in question.answer_fields
            ]
        declared_names = {declared.name for declared in question.answer_fields}
        unexpected = sorted({f.field_name for f in fields} - declared_names)
        if unexpected:
            raise ValueError(
                f"reconstruction of {qid} produced undeclared answer field(s) {unexpected}; the scorer "
                "compares declared fields only, so an undeclared answer would be silently dropped"
            )
        for item in fields:
            field_answers[answer_key(qid, item.field_name)] = item
        answers[qid] = _rollup(question, fields)
    return EpisodeReconstruction(
        episode_id=record_set.episode_id,
        scenario_id=record_set.scenario_id,
        variant_id=record_set.variant_id,
        answers=answers,
        field_answers=field_answers,
        decision_step_index=decision_step_index,
        decision_step_rule_branch=branch,
        reconstruction_version=RECONSTRUCTOR_VERSION,
    )


def reconstruct(record_set: Any, protocol: ProtocolConfig) -> dict[str, Any]:
    """JSON-safe form of :func:`reconstruct_episode` for the CLI.

    The same type guard applies: a privileged ledger, a manifest, or a serialized reference answer key
    raises :class:`TypeError` here too, so the serialized call path cannot become a side door into
    privileged evidence.
    """
    return reconstruct_episode(record_set, protocol).model_dump(mode="json")
