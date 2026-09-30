"""Privileged reference answers for the audit questions.

This module is the *only* audit module allowed to read the :class:`PrivilegedLedger` and the scenario
manifest. It produces the answer key that the reconstruction procedure is scored against
(research-acceptance.md section 5: "Retain a privileged reference event ledger that the reconstruction
procedure cannot access").

One answer per answer field
---------------------------
``AuditQuestion`` asks for several concrete facts with different units and tolerances (a position in
metres, its age in seconds). Answers are therefore keyed by the composite key
``"<question_id>.<field_name>"`` and each carries its own availability: the ``no_evidence_age`` variant
is designed to defeat the age of an observation while leaving the observation itself, and a
question-level answer could not express that.

The same composite key is produced by :attr:`EpisodeReconstruction.field_answers` and consumed by
:mod:`colosseum_assurance.audit.scoring`; :func:`answer_key` is the single definition of the convention
on this side. An independent review found the scorer looking facts up by the bare question id: every
answer was then missing, every question was classified ``reference_unavailable``, and an audit that
scored nothing at all reported no error. The three modules must therefore agree at field level, and the
regression tests in ``tests/unit/test_audit_scoring.py`` assert a positive scored denominator so that a
silent return to zero cannot pass again.

Designated decision step
------------------------
The rule is the one frozen in ``AuditSpec.decision_step_rule``; it is copied into every
:class:`EpisodeReference` so an answer key can be audited against the rule that produced it:

1. ``inspection_capture`` -- the first control step whose executed command is an inspection capture,
   the permission-dependent step the obligations are written about.
2. ``first_guard_intervention`` -- otherwise the first step with a guard intervention.
3. ``mid_episode`` -- otherwise the step nearest the middle of the retained record, an arbitrary but
   prespecified fallback so that uneventful episodes still contribute scored rows.

The branch that fired is recorded, because a rate pooled over branches would hide that inspection
episodes and uneventful episodes are very different reconstruction problems. The rule is stated over
*command* evidence, which no ablation may remove, so the reference and the reconstruction index the
same step and the audit measures reconstruction rather than decision-step drift. Privileged proximity
to the asset is reported as detail only (``within_authorized_inspection_radius``), never as the rule.

What this module reads from the EpisodeRecord, and why
-----------------------------------------------------
The ledger timestamps events in simulator time, but the questions are indexed by *control step*. When
an :class:`EpisodeRecord` is supplied, the step grid and the decision step are taken from
``step_index``, ``sim_time_s``, ``command.kind`` and ``command.issued_by`` -- fields that
:mod:`colosseum_assurance.audit.records` refuses to ablate. No ablatable field is read. Without a
record the grid is rebuilt from ``control_dt_s`` and the termination step index, and the decision step
falls back to the ``inspection_capture_performed`` and ``guard_intervention`` truth events.

All timestamps are episode-relative seconds with t = 0 just after a successful reset
(docs/timing-semantics.md).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from colosseum_assurance.protocol.spec import AuditQuestion, ProtocolConfig
from colosseum_assurance.scenario.manifest import ScenarioManifest
from colosseum_assurance.schemas import (
    EpisodeRecord,
    PrivilegedLedger,
    StrictModel,
    TruthEvent,
    TruthSample,
)

__all__ = [
    "DESIGNATED_DECISION_STEP_RULE",
    "EpisodeReference",
    "ReferenceAnswer",
    "answer_key",
    "build_reference",
    "reference_answers",
]

DESIGNATED_DECISION_STEP_RULE = (
    "Implemented branches of AuditSpec.decision_step_rule: (1) inspection_capture -- the first control "
    "step whose executed command is an inspection capture; else (2) first_guard_intervention -- the "
    "first step with a guard intervention; else (3) mid_episode -- the step nearest the middle of the "
    "retained record. The frozen wording lives in protocol.audit.decision_step_rule and is copied into "
    "every EpisodeReference."
)

DecisionBranch = Literal["inspection_capture", "first_guard_intervention", "mid_episode", "undefined"]

#: Payload keys searched, in order, for the CATEGORY of the first guard intervention. The runtime writes
#: ``intervention`` (runtime/episode.py builds the guard_intervention payload with ``step_index``,
#: ``sim_time_s``, ``intervention``, ``monitor_id``, ``verdict``, ``rationale``, ``first_of_kind``), and
#: that category is the one vocabulary both sides of the audit share: the retained record carries it as
#: ``monitor_report.intervention``. The remaining keys are accepted so a producer-side rename degrades to
#: a readable answer instead of silently becoming "unspecified".
_REASON_KEYS = ("intervention", "reason", "intervention_reason", "suspension_reason", "kind")

#: Payload keys whose free text is an equally recorded wording of the same intervention. They are offered
#: to the scorer as equivalents, never as the canonical answer, because free text is not a vocabulary two
#: independent derivations can be expected to reproduce exactly.
_REASON_EQUIVALENT_KEYS = ("rationale", "reason", "intervention_reason", "suspension_reason")

_INSPECTION_KINDS = frozenset({"inspect_capture"})

NO_INSPECTION = "no_inspection_performed"
NOT_APPLICABLE = "not_applicable"
NO_INTERVENTION = "none"


def answer_key(question_id: str, field_name: str) -> str:
    """The composite key that identifies one scored fact."""
    return f"{question_id}.{field_name}"


class ReferenceAnswer(StrictModel):
    """One privileged fact, with the derivation that produced it.

    ``available`` is first class: when the ledger genuinely cannot answer (no state sample existed at
    the decision step, a guard intervention carries no reason), the reference says so and the scorer
    drops the row from the denominator. Guessing here would turn an unanswerable question into an
    apparent reconstruction failure.
    """

    question_id: str
    field_name: str
    answer_kind: str
    unit: str = ""
    value: str | float | None = None
    numeric_value: float | None = None
    vector_value: list[float] | None = Field(
        default=None,
        description="Position answers, compared by Euclidean distance against the field tolerance.",
    )
    equivalent_values: list[str] = Field(
        default_factory=list,
        description=(
            "Other strings the privileged ledger LITERALLY recorded for this same fact, for example the "
            "monitor rationale recorded beside the intervention category. The scorer accepts any of them "
            "for a categorical or identifier field. Without this, a record that retains a different but "
            "equally recorded wording of one fact would be scored as a confident error, which would "
            "measure vocabulary rather than reconstructability."
        ),
    )
    detail: dict[str, Any] = Field(default_factory=dict)
    derivation: str = ""
    available: bool = True
    unavailable_reason: str | None = None


class EpisodeReference(StrictModel):
    """The privileged answer key for one episode, keyed by ``question_id.field_name``."""

    episode_id: str
    scenario_id: str
    arm_id: str
    run_class: str
    protocol_hash: str
    decision_step_index: int | None = None
    decision_step_rule_branch: DecisionBranch = "undefined"
    decision_step_rule: str = ""
    decision_sim_time_s: float | None = None
    answers: dict[str, ReferenceAnswer] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Step grid, truth lookup, decision step
# --------------------------------------------------------------------------------------
def _step_grid(
    ledger: PrivilegedLedger, record: EpisodeRecord | None, protocol: ProtocolConfig
) -> list[tuple[int, float]]:
    """Return ``(step_index, sim_time_s)`` pairs, preferring the record's unablatable timestamps."""
    if record is not None and record.steps:
        return [(s.step_index, s.sim_time_s) for s in record.steps]
    dt = protocol.mission.control_dt_s
    n_steps = max(int(ledger.termination.step_index) + 1, 1)
    return [(k, round(k * dt, 6)) for k in range(n_steps)]


def _sorted_samples(ledger: PrivilegedLedger) -> list[TruthSample]:
    return sorted(ledger.samples, key=lambda s: s.sim_time_s)


def _sample_at_or_before(samples: list[TruthSample], t_s: float) -> TruthSample | None:
    """Newest truth sample measured at or before ``t_s`` (the sample a delayed channel could deliver)."""
    best: TruthSample | None = None
    for sample in samples:
        if sample.sim_time_s <= t_s + 1e-9 and (best is None or sample.sim_time_s > best.sim_time_s):
            best = sample
    return best


def _first_event(ledger: PrivilegedLedger, kind: str) -> TruthEvent | None:
    events = [e for e in ledger.events if e.kind == kind]
    if not events:
        return None
    return min(events, key=lambda e: e.sim_time_s)


def _step_in_force(grid: list[tuple[int, float]], t_s: float) -> tuple[int, float] | None:
    """The last step that had started at ``t_s``; falls back to the first step for pre-start events."""
    chosen: tuple[int, float] | None = None
    for index, t_step in grid:
        if t_step <= t_s + 1e-9:
            chosen = (index, t_step)
    if chosen is None and grid:
        return grid[0]
    return chosen


def _record_step_matching(record: EpisodeRecord | None, wanted: str) -> tuple[int, float] | None:
    """First step whose REALIZED command matches ``wanted`` (``inspection`` or ``guard``).

    Realized means the executed command, falling back to the issued command only when no executed
    command was recorded. An independent review found the earlier both-fields scan incoherent: a guard
    can replace an intended ``inspect_capture`` with a hold, and an audit question about what was
    *performed* must then not designate that step. The reconstructor applies the identical rule from the
    exposed record, so both sides index the same step without sharing code or privileged data.
    """
    if record is None:
        return None
    for step in record.steps:
        command = step.executed_command or step.command
        if command is None:
            continue
        if wanted == "inspection" and command.kind in _INSPECTION_KINDS:
            return step.step_index, step.sim_time_s
        if wanted == "guard" and command.issued_by == "guard":
            return step.step_index, step.sim_time_s
    return None


def _designated_decision_step(
    ledger: PrivilegedLedger,
    record: EpisodeRecord | None,
    grid: list[tuple[int, float]],
) -> tuple[int | None, float | None, DecisionBranch]:
    if not grid:
        return None, None, "undefined"
    from_record = _record_step_matching(record, "inspection")
    if from_record is not None:
        return from_record[0], from_record[1], "inspection_capture"
    capture = _first_event(ledger, "inspection_capture_performed")
    if capture is not None:
        in_force = _step_in_force(grid, capture.sim_time_s)
        if in_force is not None:
            return in_force[0], in_force[1], "inspection_capture"
    from_record = _record_step_matching(record, "guard")
    if from_record is not None:
        return from_record[0], from_record[1], "first_guard_intervention"
    intervention = _first_event(ledger, "guard_intervention")
    if intervention is not None:
        in_force = _step_in_force(grid, intervention.sim_time_s)
        if in_force is not None:
            return in_force[0], in_force[1], "first_guard_intervention"
    index, t_step = grid[len(grid) // 2]
    return index, t_step, "mid_episode"


# --------------------------------------------------------------------------------------
# Per-field references
# --------------------------------------------------------------------------------------
def _answer(question: AuditQuestion, field_name: str, **kwargs: Any) -> ReferenceAnswer:
    """Build an answer pre-filled with the frozen kind and unit of its answer field."""
    field = question.field(field_name)
    return ReferenceAnswer(
        question_id=question.question_id,
        field_name=field_name,
        answer_kind=str(field.kind),
        unit=str(field.unit),
        **kwargs,
    )


def _policy_version_answers(
    question: AuditQuestion, record: EpisodeRecord | None, protocol: ProtocolConfig
) -> dict[str, ReferenceAnswer]:
    """Q1: the policy version in force, taken from the frozen protocol, never from the record."""
    authoritative = protocol.obligations.policy_version
    recorded = record.policy_version if record is not None else None
    return {
        answer_key(question.question_id, "policy_version"): _answer(
            question, "policy_version",
            value=authoritative,
            detail={
                "protocol_policy_version": authoritative,
                "recorded_policy_version": recorded,
                "record_agrees": None if recorded is None else recorded == authoritative,
            },
            derivation=(
                "ObligationSpec.policy_version of the frozen protocol whose hash the episode carries; "
                "the value recorded in the episode is reported only as a cross-check"
            ),
        )
    }


def _observation_answers(
    question: AuditQuestion,
    ledger: PrivilegedLedger,
    manifest: ScenarioManifest,
    protocol: ProtocolConfig,
    step_index: int | None,
    t_decision: float | None,
) -> dict[str, ReferenceAnswer]:
    """Q2: which position observation the decision maker could hold, and how old it was.

    Derived from the exogenous schedule and the privileged samples alone: the observation pipeline
    delivers the newest sample measured at or before ``t - scheduled_delay(step)`` and delivers nothing
    at a scheduled dropout.
    """
    qid = question.question_id
    position_key = answer_key(qid, "observed_position")
    age_key = answer_key(qid, "evidence_age")

    def unavailable(reason: str, derivation: str, detail: dict[str, Any]) -> dict[str, ReferenceAnswer]:
        return {
            position_key: _answer(question, "observed_position", available=False,
                                  unavailable_reason=reason, derivation=derivation, detail=detail),
            age_key: _answer(question, "evidence_age", available=False,
                             unavailable_reason=reason, derivation=derivation, detail=detail),
        }

    if step_index is None or t_decision is None:
        return unavailable("no_decision_step", "episode has no control steps", {})
    schedules = manifest.schedules
    if schedules.state_dropped(step_index):
        return unavailable(
            "state_dropout_at_decision_step",
            "ScheduleSet.state_dropout marks this step: no position observation reached the decision "
            "maker, so neither the observed position nor its age is defined",
            {"step_index": step_index, "scheduled_state_dropout": True},
        )
    delay_s = schedules.delay_at_step(step_index)
    cutoff = t_decision - delay_s
    sample = _sample_at_or_before(_sorted_samples(ledger), cutoff)
    if sample is None:
        return unavailable(
            "no_truth_sample_older_than_the_scheduled_delay",
            "no privileged sample was measured at or before the delayed-channel cutoff",
            {"step_index": step_index, "scheduled_delay_s": delay_s, "cutoff_sim_time_s": cutoff},
        )
    age = max(0.0, t_decision - sample.sim_time_s)
    distance = sample.position.distance_to(manifest.asset_position)
    radius = protocol.obligations.authorized_inspection_radius_m
    shared_detail = {
        "step_index": step_index,
        "decision_sim_time_s": t_decision,
        "scheduled_delay_s": delay_s,
        "observation_sim_time_s": sample.sim_time_s,
        "truth_distance_to_asset_m": distance,
        "within_authorized_inspection_radius": distance <= radius,
    }
    selection = (
        "the newest privileged sample measured at or before (decision time - "
        "ScheduleSet.delay_at_step) is the one the delayed channel could deliver; this reproduces the "
        "selection rule of ObservationPipeline"
    )
    return {
        position_key: _answer(
            question, "observed_position",
            value=f"x={sample.position.x:.3f};y={sample.position.y:.3f};z={sample.position.z:.3f}",
            vector_value=[sample.position.x, sample.position.y, sample.position.z],
            detail=dict(shared_detail),
            derivation=f"{selection}; its position is the observation that was available",
        ),
        age_key: _answer(
            question, "evidence_age", value=age, numeric_value=age,
            detail=dict(shared_detail),
            derivation=(
                f"{selection}; the age is the decision step time minus that sample's measurement time. "
                "When the runner samples vehicle state once per control step instead of at the "
                "truth-sample period, this age can differ from the onboard age by less than one "
                "control step"
            ),
        ),
    }


def _authorization_answers(
    question: AuditQuestion, ledger: PrivilegedLedger, protocol: ProtocolConfig
) -> dict[str, ReferenceAnswer]:
    """Q3: which authorization token was held at the inspection capture, and was it valid then."""
    qid = question.question_id
    token_key = answer_key(qid, "token_id")
    valid_key = answer_key(qid, "valid_at_use")
    capture = _first_event(ledger, "inspection_capture_performed")
    if capture is None:
        derivation = "no inspection_capture_performed event exists in the privileged ledger"
        detail = {"inspection_capture_performed": False}
        return {
            token_key: _answer(question, "token_id", value=NO_INSPECTION, detail=detail,
                               derivation=derivation),
            valid_key: _answer(question, "valid_at_use", value=NOT_APPLICABLE, detail=detail,
                               derivation=derivation),
        }
    t_capture = capture.sim_time_s
    validity_s = protocol.obligations.authorization_validity_s
    granted = [e for e in ledger.events
               if e.kind == "authorization_granted" and e.sim_time_s <= t_capture + 1e-9]
    expired_tokens = {
        str(e.payload.get("token_id"))
        for e in ledger.events
        if e.kind == "authorization_expired" and e.sim_time_s <= t_capture + 1e-9
    }
    if not granted:
        derivation = "no authorization_granted event precedes the inspection_capture_performed event"
        detail = {"capture_sim_time_s": t_capture, "granted_tokens_before_capture": 0}
        return {
            token_key: _answer(question, "token_id", value="none", detail=detail, derivation=derivation),
            valid_key: _answer(question, "valid_at_use", value="false", detail=detail,
                               derivation=derivation),
        }
    newest = max(granted, key=lambda e: e.sim_time_s)
    token_id = str(newest.payload.get("token_id", "unknown"))
    expires_at = newest.payload.get("expires_at_s")
    expires_at_s = float(expires_at) if expires_at is not None else newest.sim_time_s + validity_s
    valid = t_capture < expires_at_s and token_id not in expired_tokens
    detail = {
        "capture_sim_time_s": t_capture,
        "token_id": token_id,
        "granted_at_s": newest.sim_time_s,
        "expires_at_s": expires_at_s,
        "authorization_validity_s": validity_s,
        "expired_before_capture": token_id in expired_tokens,
    }
    return {
        token_key: _answer(
            question, "token_id", value=token_id, detail=dict(detail),
            derivation=(
                "token of the newest authorization_granted event at or before the "
                "inspection_capture_performed event"
            ),
        ),
        valid_key: _answer(
            question, "valid_at_use", value="true" if valid else "false", detail=dict(detail),
            derivation=(
                "validity from that grant's expires_at_s (or granted_at + "
                "ObligationSpec.authorization_validity_s) and from any authorization_expired event for "
                "the token before the capture"
            ),
        ),
    }


def _suspension_answers(
    question: AuditQuestion, ledger: PrivilegedLedger
) -> dict[str, ReferenceAnswer]:
    """Q4: the reason recorded with the first guard intervention, or the exact string "none".

    The canonical answer is the intervention CATEGORY (``hold``, ``suspend_inspection``,
    ``return_to_launch``, ``abort``), because that is the only vocabulary the privileged ledger and the
    retained record share: the runtime writes it into the truth event as ``intervention`` and into the
    step record as ``monitor_report.intervention``. The free-text rationale recorded beside it is offered
    as an equivalent, so a variant whose only surviving route is the wording still scores correct.

    Known limitation: a retained free-text reason that the privileged ledger never recorded (for example
    a purely runtime-generated command annotation) cannot be checked against this answer key and would be
    scored as a confident error. None of the frozen record variants produces that case, because each of
    them either retains ``monitor_report.intervention`` or removes the monitor report entirely.
    """
    key = answer_key(question.question_id, "intervention_reason")
    event = _first_event(ledger, "guard_intervention")
    if event is None:
        return {
            key: _answer(
                question, "intervention_reason", value=NO_INTERVENTION, detail={"interventions": 0},
                derivation="the privileged ledger holds no guard_intervention event",
            )
        }
    reason: str | None = None
    used_key: str | None = None
    for candidate_key in _REASON_KEYS:
        candidate = event.payload.get(candidate_key)
        if isinstance(candidate, str) and candidate.strip():
            reason, used_key = candidate.strip(), candidate_key
            break
    if reason is None and event.detail.strip():
        reason, used_key = event.detail.strip(), "detail"
    equivalents: list[str] = []
    for candidate_key in _REASON_EQUIVALENT_KEYS:
        candidate = event.payload.get(candidate_key)
        if isinstance(candidate, str) and candidate.strip() and candidate.strip() != reason:
            equivalents.append(candidate.strip())
    if event.detail.strip() and event.detail.strip() != reason:
        equivalents.append(event.detail.strip())
    if reason is None:
        return {
            key: _answer(
                question, "intervention_reason", available=False,
                unavailable_reason="guard_intervention_event_carries_no_reason",
                detail={"event_sim_time_s": event.sim_time_s, "payload_keys": sorted(event.payload)},
                derivation="the first guard_intervention event has neither a reason payload nor a detail",
            )
        }
    return {
        key: _answer(
            question, "intervention_reason", value=reason, equivalent_values=equivalents,
            detail={"event_sim_time_s": event.sim_time_s, "source_key": used_key,
                    "equivalent_recorded_wordings": list(equivalents)},
            derivation=(
                f"first guard_intervention truth event, category read from {used_key!r}; "
                f"{len(equivalents)} other wording(s) recorded with the same event are accepted as "
                "equivalent"
            ),
        )
    }


# --------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------
def build_reference(
    ledger: PrivilegedLedger,
    record: EpisodeRecord | None,
    manifest: ScenarioManifest,
    protocol: ProtocolConfig,
) -> EpisodeReference:
    """Derive the privileged answer key for one episode.

    ``record`` is optional and is read only for the step grid and the decision step
    (``step_index``, ``sim_time_s``, command kind and issuer), so the answer key never depends on a
    field that an ablation could remove.
    """
    if record is not None and record.episode_id != ledger.episode_id:
        raise ValueError(
            f"record {record.episode_id!r} and ledger {ledger.episode_id!r} describe different episodes"
        )
    if manifest.scenario_id != ledger.scenario_id:
        raise ValueError(
            f"manifest {manifest.scenario_id!r} does not match ledger scenario {ledger.scenario_id!r}"
        )
    grid = _step_grid(ledger, record, protocol)
    step_index, t_decision, branch = _designated_decision_step(ledger, record, grid)
    answers: dict[str, ReferenceAnswer] = {}
    for question in protocol.audit.questions:
        qid = question.question_id
        if qid == "Q1_policy_version":
            answers.update(_policy_version_answers(question, record, protocol))
        elif qid == "Q2_observation_available":
            answers.update(
                _observation_answers(question, ledger, manifest, protocol, step_index, t_decision)
            )
        elif qid == "Q3_authorization_record":
            answers.update(_authorization_answers(question, ledger, protocol))
        elif qid == "Q4_suspension_reason":
            answers.update(_suspension_answers(question, ledger))
        else:
            for field in question.answer_fields:
                answers[answer_key(qid, field.name)] = _answer(
                    question, field.name, available=False,
                    unavailable_reason="no_reference_derivation_implemented",
                    derivation=(
                        "this audit field was added to the protocol after the reference module was "
                        "frozen; it is reported as unavailable rather than guessed"
                    ),
                )
    return EpisodeReference(
        episode_id=ledger.episode_id,
        scenario_id=ledger.scenario_id,
        arm_id=ledger.arm_id,
        run_class=ledger.run_class,
        protocol_hash=ledger.protocol_hash,
        decision_step_index=step_index,
        decision_step_rule_branch=branch,
        decision_step_rule=protocol.audit.decision_step_rule,
        decision_sim_time_s=None if t_decision is None else float(t_decision),
        answers=answers,
    )


def reference_answers(
    record: EpisodeRecord,
    ledger: PrivilegedLedger,
    manifest: ScenarioManifest,
    protocol: ProtocolConfig,
) -> dict[str, Any]:
    """JSON-safe form of :func:`build_reference` for the CLI, which serializes both sides.

    The full :class:`EpisodeReference` is dumped, so the per-field answers stay under ``answers``
    together with the decision-step branch, the frozen rule, and every derivation string. Nothing is
    summarised away: an answer key that loses its derivation cannot be challenged later.
    """
    return build_reference(ledger, record, manifest, protocol).model_dump(mode="json")
