"""Field-level audit scoring: the answer key, the reconstruction and the scorer must agree.

An independent review found that the audit produced no result at all. ``audit/reference.py`` keys every
privileged fact as ``question_id.field_name`` (a position in metres and its age in seconds are separate
facts with separate tolerances), while ``audit/scoring.py`` looked the answers up by the bare question
id. On the provenance-rich record of ``test_audit_realized_actions._episode``, SIX available reference
facts became FOUR ``reference_unavailable`` classifications, ZERO scored units, and an undefined
reconstruction accuracy -- and the test that guarded this path asserted only that nothing was
confidently incorrect, which is exactly what a total failure to score looks like.

These tests therefore assert denominators, exact expected answers, deliberately wrong answers, the
declared tolerance of each field in its own unit, and that an empty audit is loud. They drive the
production path used by ``workflows/audit.py``: ``build_reference`` + ``build_variants`` +
``reconstruct_episode`` + ``score_episode`` + ``aggregate``, including its serialized CLI form.

All data here is synthetic (simulator identity ``fixture_fake``). It is software evidence about the
scoring code, never experimental evidence about a simulator or a vehicle.
"""

from __future__ import annotations

from typing import Any

import pytest
from tests.unit.test_audit_realized_actions import PROTOCOL, _episode

from colosseum_assurance.audit.reconstructor import (
    EpisodeReconstruction,
    ReconstructedField,
    reconstruct,
    reconstruct_episode,
)
from colosseum_assurance.audit.reconstructor import answer_key as reconstruction_answer_key
from colosseum_assurance.audit.records import AuditRecordSet, build_variants
from colosseum_assurance.audit.reference import (
    EpisodeReference,
    answer_key,
    build_reference,
    reference_answers,
)
from colosseum_assurance.audit.scoring import (
    EpisodeAuditScore,
    aggregate,
    score_episode,
    score_run,
)
from colosseum_assurance.protocol.spec import AnswerField, ProtocolConfig
from colosseum_assurance.scenario.manifest import ScenarioManifest, build_manifest
from colosseum_assurance.schemas import EpisodeRecord, PrivilegedLedger

#: The step the frozen decision-step rule designates in this fixture (the realized inspection capture,
#: or the guard intervention that replaced it). Every perturbation below targets exactly this step,
#: because that is the step both sides answer about.
DECISION_STEP = 2

Q2 = "Q2_observation_available"
Q3 = "Q3_authorization_record"
POSITION = answer_key(Q2, "observed_position")
AGE = answer_key(Q2, "evidence_age")
TOKEN = answer_key(Q3, "token_id")
VALID = answer_key(Q3, "valid_at_use")
POLICY = answer_key("Q1_policy_version", "policy_version")
REASON = answer_key("Q4_suspension_reason", "intervention_reason")

ALL_FIELD_KEYS = {POLICY, POSITION, AGE, TOKEN, VALID, REASON}


def _field_spec(question_id: str, field_name: str) -> AnswerField:
    """The frozen declaration of one answer field, so tests use the protocol's tolerance, not a copy."""
    question = next(q for q in PROTOCOL.audit.questions if q.question_id == question_id)
    return question.field(field_name)


POSITION_TOLERANCE_M = float(_field_spec(Q2, "observed_position").tolerance or 0.0)
AGE_TOLERANCE_S = float(_field_spec(Q2, "evidence_age").tolerance or 0.0)


def _manifest() -> ScenarioManifest:
    return build_manifest(PROTOCOL, "fixture", "obs_nominal__sup_nominal", 0)


def _mislead(
    record: EpisodeRecord,
    *,
    position_offset_m: float = 0.0,
    reported_age_s: float = 0.0,
    token_id: str | None = None,
) -> EpisodeRecord:
    """Return a record whose RETAINED observation at the decision step disagrees with privileged truth.

    This is a misleading record, not a different flight: the trajectory, the commands and every
    timestamp are untouched, and the result is re-validated through :class:`EpisodeRecord`, so it obeys
    every schema invariant the evidence writer enforces (in particular
    ``state_age_s == receive_sim_time_s - state.sim_time_s``). Without such a case the scorer could
    never be shown to compare anything: an audit that only ever sees agreeing sides cannot distinguish a
    working comparison from one that classifies everything as correct.
    """
    payload: dict[str, Any] = record.model_dump(mode="json")
    for step in payload["steps"]:
        if step["step_index"] != DECISION_STEP or step.get("observation") is None:
            continue
        observation = step["observation"]
        received = float(observation["receive_sim_time_s"])
        measured = received - float(reported_age_s)
        assert measured >= 0.0, "a fixture may not claim an observation acquired before the episode"
        observation["state"]["position"]["x"] += float(position_offset_m)
        observation["state"]["sim_time_s"] = measured
        observation["state_age_s"] = float(reported_age_s)
        observation["sensor_health"]["state_age_s"] = float(reported_age_s)
        if token_id is not None:
            observation["authorization"]["token_id"] = token_id
    return EpisodeRecord.model_validate(payload)


def _variant(record: EpisodeRecord, variant_id: str) -> AuditRecordSet:
    variants = {v.variant_id: v for v in build_variants(record, PROTOCOL)}
    return variants[variant_id]


def _score(
    record: EpisodeRecord,
    ledger: PrivilegedLedger,
    variant_id: str = "provenance_rich",
) -> EpisodeAuditScore:
    """The production path of ``workflows/audit.py`` for one (episode, variant) pair."""
    reference = build_reference(ledger, record, _manifest(), PROTOCOL)
    reconstruction = reconstruct_episode(_variant(record, variant_id), PROTOCOL)
    score = score_episode(
        reference, reconstruction, PROTOCOL, episode_id=record.episode_id, variant_id=variant_id
    )
    assert isinstance(score, EpisodeAuditScore)
    return score


# --------------------------------------------------------------------------------------
# The key convention itself
# --------------------------------------------------------------------------------------
def test_reference_reconstruction_and_scoring_share_one_field_key() -> None:
    """The three modules must produce the same composite key for every declared answer field.

    This is the defect itself: the reference wrote ``Q2_observation_available.observed_position`` and the
    scorer asked for ``Q2_observation_available``. The reconstructor defines its own ``answer_key``
    because the separation rule forbids it from importing the module holding its answer key, so the two
    definitions are compared here directly.
    """
    record, ledger = _episode(suppress_capture=False)
    reference = build_reference(ledger, record, _manifest(), PROTOCOL)
    reconstruction = reconstruct_episode(_variant(record, "provenance_rich"), PROTOCOL)

    declared = {
        answer_key(question.question_id, field.name)
        for question in PROTOCOL.audit.questions
        for field in question.answer_fields
    }
    assert declared == ALL_FIELD_KEYS
    for question in PROTOCOL.audit.questions:
        for field in question.answer_fields:
            assert answer_key(question.question_id, field.name) == reconstruction_answer_key(
                question.question_id, field.name
            )
    assert set(reference.answers) == declared, "the privileged answer key lost or renamed a fact"
    assert set(reconstruction.field_answers) == declared, (
        "the reconstruction must answer every declared field under the shared key, so the scorer can "
        "find it"
    )


def test_the_bare_question_id_is_not_a_valid_answer_key() -> None:
    """The defect in its smallest form, using only the plain mapping the scorer has always returned.

    Against the pre-fix module this fails with every question ``reference_unavailable``: the answer key
    holds ``Q2_observation_available.observed_position`` and the scorer asked for
    ``Q2_observation_available``. No new attribute is used here, so the failure is behavioural.
    """
    record, ledger = _episode(suppress_capture=False)
    reference = build_reference(ledger, record, _manifest(), PROTOCOL)
    reconstruction = reconstruct_episode(_variant(record, "provenance_rich"), PROTOCOL)
    categories = score_episode(reconstruction, reference, PROTOCOL)

    assert isinstance(categories, dict)
    assert set(categories) == {q.question_id for q in PROTOCOL.audit.questions}
    excluded = [qid for qid, category in categories.items() if category == "reference_unavailable"]
    assert not excluded, (
        "the privileged reference answers all six facts of this episode, so no question may be excluded "
        f"from the denominator; excluded {excluded}"
    )
    assert all(category == "correct" for category in categories.values()), categories


# --------------------------------------------------------------------------------------
# Exact expected correct answers, with a positive denominator
# --------------------------------------------------------------------------------------
def test_a_provenance_rich_record_scores_every_declared_field_correct() -> None:
    """Six available facts, six scored, six correct, and every question rolled up as correct."""
    record, ledger = _episode(suppress_capture=False)
    manifest = _manifest()
    reference = build_reference(ledger, record, manifest, PROTOCOL)
    reconstruction = reconstruct_episode(_variant(record, "provenance_rich"), PROTOCOL)

    # The exact expected answers, stated independently of the code that produced them.
    decision_time_s = DECISION_STEP * PROTOCOL.mission.control_dt_s
    truth_at_decision = next(
        s for s in ledger.samples if s.sim_time_s == pytest.approx(decision_time_s)
    )
    assert reference.decision_step_index == DECISION_STEP
    assert reference.answers[POLICY].value == PROTOCOL.obligations.policy_version
    assert reference.answers[POSITION].vector_value == pytest.approx(
        [truth_at_decision.position.x, truth_at_decision.position.y, truth_at_decision.position.z]
    )
    assert reference.answers[AGE].numeric_value == pytest.approx(0.0)
    assert reference.answers[TOKEN].value == "auth-00"
    assert reference.answers[VALID].value == "true"
    assert reference.answers[REASON].value == "none"
    assert all(a.available for a in reference.answers.values())

    assert reconstruction.field_answers[POLICY].value == PROTOCOL.obligations.policy_version
    assert reconstruction.field_answers[POSITION].vector_value == pytest.approx(
        [truth_at_decision.position.x, truth_at_decision.position.y, truth_at_decision.position.z]
    )
    assert reconstruction.field_answers[AGE].numeric_value == pytest.approx(0.0)
    assert reconstruction.field_answers[TOKEN].value == "auth-00"
    assert reconstruction.field_answers[VALID].value == "true"
    assert reconstruction.field_answers[REASON].value == "none"

    score = score_episode(
        reference, reconstruction, PROTOCOL,
        episode_id=record.episode_id, variant_id="provenance_rich",
    )
    assert isinstance(score, EpisodeAuditScore)
    assert len(reference.answers) == 6
    assert score.scored_fields == 6, f"positive denominator required, got {score.field_categories}"
    assert score.field_reference_unavailable == 0
    assert score.field_correct == 6, score.field_comparisons
    assert score.field_incorrect_confident == 0
    assert score.field_insufficient_evidence == 0
    assert score.categories == {q.question_id: "correct" for q in PROTOCOL.audit.questions}
    assert score.reference_unavailable == 0


def test_the_serialized_cli_path_scores_identically_to_the_object_path() -> None:
    """``workflows/audit.py`` passes JSON on both sides; it must not score differently."""
    record, ledger = _episode(suppress_capture=True)
    manifest = _manifest()
    objects = _score(record, ledger, "provenance_rich")

    reference_json = reference_answers(record, ledger, manifest, PROTOCOL)
    reconstructed_json = reconstruct(_variant(record, "provenance_rich"), PROTOCOL)
    serialized = score_episode(
        reference_json, reconstructed_json, PROTOCOL,
        episode_id=record.episode_id, variant_id="provenance_rich",
    )
    assert isinstance(serialized, EpisodeAuditScore)
    assert serialized.categories == objects.categories
    assert serialized.field_categories == objects.field_categories
    assert serialized.scored_fields == 6


# --------------------------------------------------------------------------------------
# The comparison really compares
# --------------------------------------------------------------------------------------
def test_a_misleading_record_is_scored_as_confidently_incorrect() -> None:
    """A record that misstates the position, the age and the token must not score as correct."""
    record, ledger = _episode(suppress_capture=False)
    misleading = _mislead(
        record,
        position_offset_m=POSITION_TOLERANCE_M + 2.0,
        reported_age_s=AGE_TOLERANCE_S + 0.6,
        token_id="auth-99",
    )
    score = _score(misleading, ledger, "provenance_rich")

    assert score.scored_fields == 6, "the denominator must stay positive when answers are wrong"
    assert score.field_categories[POSITION] == "incorrect_confident", score.field_comparisons[POSITION]
    assert score.field_categories[AGE] == "incorrect_confident", score.field_comparisons[AGE]
    assert score.field_categories[TOKEN] == "incorrect_confident", score.field_comparisons[TOKEN]
    assert score.field_categories[POLICY] == "correct"
    assert score.categories[Q2] == "incorrect_confident"
    assert score.categories[Q3] == "incorrect_confident"
    assert score.field_incorrect_confident == 3
    # The measured error and the declared tolerance are reported, so a reviewer can check the verdict.
    assert f"{POSITION_TOLERANCE_M:.3f} m" in score.field_comparisons[POSITION]
    assert f"{AGE_TOLERANCE_S:.3f} s" in score.field_comparisons[AGE]
    assert "auth-99" in score.field_comparisons[TOKEN] and "auth-00" in score.field_comparisons[TOKEN]


def test_each_field_is_compared_with_its_own_unit_and_tolerance() -> None:
    """The same number is inside the metre tolerance and outside the second tolerance.

    A 0.5 error is correct for the position (tolerance 0.75 m) and confidently incorrect for the
    evidence age (tolerance 0.30 s). A scorer that applied one tolerance to both facts, or compared a
    position as a scalar, cannot produce this pair.
    """
    shared_error = 0.5
    assert shared_error <= POSITION_TOLERANCE_M and shared_error > AGE_TOLERANCE_S
    record, ledger = _episode(suppress_capture=False)
    score = _score(
        _mislead(record, position_offset_m=shared_error, reported_age_s=shared_error),
        ledger,
        "provenance_rich",
    )
    assert score.field_categories[POSITION] == "correct", score.field_comparisons[POSITION]
    assert score.field_categories[AGE] == "incorrect_confident", score.field_comparisons[AGE]
    assert score.categories[Q2] == "incorrect_confident", "one wrong field makes the question wrong"


@pytest.mark.parametrize(
    ("offset_m", "expected"),
    [
        (POSITION_TOLERANCE_M - 0.01, "correct"),
        (POSITION_TOLERANCE_M + 0.01, "incorrect_confident"),
    ],
)
def test_position_tolerance_boundary_in_metres(offset_m: float, expected: str) -> None:
    """Just inside and just outside the declared metre tolerance, measured as a distance."""
    record, ledger = _episode(suppress_capture=False)
    score = _score(_mislead(record, position_offset_m=offset_m), ledger, "provenance_rich")
    assert score.field_categories[POSITION] == expected, score.field_comparisons[POSITION]
    assert score.field_categories[AGE] == "correct", "only the position was perturbed"


@pytest.mark.parametrize(
    ("age_s", "expected"),
    [
        (AGE_TOLERANCE_S - 0.01, "correct"),
        (AGE_TOLERANCE_S + 0.01, "incorrect_confident"),
    ],
)
def test_evidence_age_tolerance_boundary_in_seconds(age_s: float, expected: str) -> None:
    """Just inside and just outside the declared second tolerance."""
    record, ledger = _episode(suppress_capture=False)
    score = _score(_mislead(record, reported_age_s=age_s), ledger, "provenance_rich")
    assert score.field_categories[AGE] == expected, score.field_comparisons[AGE]
    assert score.field_categories[POSITION] == "correct", "only the age was perturbed"


def test_an_answered_field_with_no_value_is_not_scored_as_correct() -> None:
    """A content-free answer must not match the recorded answer "none".

    Q3 legitimately answers the string ``"none"`` when no token was held. If a reconstruction reported a
    field as answered while supplying nothing, a plain string comparison would read the missing value as
    that same ``"none"`` and award a correct answer for an empty one.
    """
    record, ledger = _episode(suppress_capture=False)
    reference = build_reference(ledger, record, _manifest(), PROTOCOL)
    reference.answers[TOKEN].value = "none"
    reconstruction = reconstruct_episode(_variant(record, "provenance_rich"), PROTOCOL)
    reconstruction.field_answers[TOKEN] = ReconstructedField(
        question_id=Q3, field_name="token_id", confidence="answered", value=None,
        reasoning="deliberately content-free answer",
    )
    score = score_episode(
        reference, reconstruction, PROTOCOL,
        episode_id=record.episode_id, variant_id="provenance_rich",
    )
    assert isinstance(score, EpisodeAuditScore)
    assert score.field_categories[TOKEN] == "insufficient_evidence", score.field_comparisons[TOKEN]
    assert "supplied no value" in score.field_comparisons[TOKEN]


# --------------------------------------------------------------------------------------
# Ablations degrade exactly the facts whose routes they remove
# --------------------------------------------------------------------------------------
def test_ablations_degrade_only_the_fields_whose_routes_they_remove() -> None:
    """Same episode, three record variants, three different outcomes for the two Q2 facts.

    ``no_evidence_age`` removes every route to the age and keeps the position observation, so exactly
    one of the two facts must become unknown. A whole-question abstention would hide that the retained
    position is still established, and would make a targeted ablation look like an action-only record.
    """
    record, ledger = _episode(suppress_capture=False)
    rich = _score(record, ledger, "provenance_rich")
    aged_out = _score(record, ledger, "no_evidence_age")
    action_only = _score(record, ledger, "action_only")
    redundant = _score(record, ledger, "explicit_age_fields_only")

    assert rich.field_categories[POSITION] == "correct"
    assert rich.field_categories[AGE] == "correct"

    assert aged_out.field_categories[POSITION] == "correct", (
        "the position observation survives this ablation and must still be scored: "
        f"{aged_out.field_comparisons[POSITION]}"
    )
    assert aged_out.field_categories[AGE] == "insufficient_evidence"
    assert aged_out.field_reference_unavailable == 0, (
        "the privileged ledger still knows the answer; the RECORD is what lost it"
    )
    assert aged_out.categories[Q2] == "insufficient_evidence"

    assert action_only.field_categories[POSITION] == "insufficient_evidence"
    assert action_only.field_categories[AGE] == "insufficient_evidence"
    assert action_only.field_categories[POLICY] == "insufficient_evidence"
    assert action_only.field_correct < rich.field_correct

    # The redundancy probe keeps the acquisition timestamps, so the age is recoverable and the variant
    # must NOT be reported as having removed the information.
    assert redundant.field_categories[AGE] == "correct", redundant.field_comparisons[AGE]


def test_removing_the_authorization_view_makes_the_token_unknown_not_wrong() -> None:
    """An auditor who cannot see the token must abstain, never guess: the two are different failures."""
    record, ledger = _episode(suppress_capture=False)
    blinded = _score(record, ledger, "no_authorization_view")
    assert blinded.field_categories[TOKEN] == "insufficient_evidence"
    assert blinded.field_categories[VALID] == "insufficient_evidence"
    assert blinded.field_incorrect_confident == 0
    assert blinded.categories[Q3] == "insufficient_evidence"


# --------------------------------------------------------------------------------------
# A run that scores nothing must say so
# --------------------------------------------------------------------------------------
def _rekey_by_question(reference: EpisodeReference) -> EpisodeReference:
    """The pre-fix shape: one answer per QUESTION, so no field key resolves.

    This is what the scorer used to ask for. Reproducing it proves that such a run is now reported as
    scoring nothing instead of quietly reporting an audit with no denominator.
    """
    collapsed: dict[str, Any] = {}
    for answer in reference.answers.values():
        collapsed.setdefault(answer.question_id, answer.model_copy())
    return reference.model_copy(update={"answers": collapsed})


def test_a_healthy_run_reports_positive_scored_units() -> None:
    record, ledger = _episode(suppress_capture=False)
    reference = build_reference(ledger, record, _manifest(), PROTOCOL)
    pairs: list[tuple[EpisodeReconstruction, EpisodeReference]] = [
        (reconstruct_episode(variant, PROTOCOL), reference)
        for variant in build_variants(record, PROTOCOL)
    ]
    summary = score_run(pairs, PROTOCOL, "fixture", resamples=64)

    assert summary.zero_scored_units is False
    assert summary.scoring_alert is None
    assert summary.scored_units == len(pairs) * len(PROTOCOL.audit.questions)
    assert summary.scored_field_units == len(pairs) * 6
    assert summary.excluded_units == 0
    rich = summary.variants["provenance_rich"]
    assert rich.per_question[Q2].n_episodes == 1
    assert rich.per_question[Q2].correct == 1
    assert rich.per_question[Q2].per_field["observed_position"].n_episodes == 1
    assert rich.per_question[Q2].per_field["observed_position"].unit == "m"
    assert rich.per_question[Q2].per_field["observed_position"].tolerance == POSITION_TOLERANCE_M
    assert rich.per_question[Q2].per_field["evidence_age"].unit == "s"
    assert summary.variants["explicit_age_fields_only"].retains_redundant_routes is True
    assert summary.variants["provenance_rich"].retains_redundant_routes is False


def test_a_run_that_scores_nothing_is_reported_in_a_machine_readable_field() -> None:
    """The exact failure of the review: every answer excluded, nothing scored, no complaint."""
    record, ledger = _episode(suppress_capture=False)
    broken = _rekey_by_question(build_reference(ledger, record, _manifest(), PROTOCOL))
    pairs: list[tuple[EpisodeReconstruction, EpisodeReference]] = [
        (reconstruct_episode(variant, PROTOCOL), broken)
        for variant in build_variants(record, PROTOCOL)
    ]
    summary = score_run(pairs, PROTOCOL, "fixture", resamples=64)

    assert summary.scored_units == 0
    assert summary.scored_field_units == 0
    assert summary.zero_scored_units is True
    assert summary.scoring_alert is not None
    assert "NO AUDIT UNIT WAS SCORED" in summary.scoring_alert
    assert "question_id.field_name" in summary.scoring_alert
    assert summary.excluded_units == len(pairs) * len(PROTOCOL.audit.questions)
    assert summary.variants["provenance_rich"].overall_correct_rate.point is None


def test_the_alert_survives_the_serialized_aggregate_path() -> None:
    """``aggregate`` is what the CLI calls; the alert must not exist only in the in-process path."""
    record, ledger = _episode(suppress_capture=False)
    broken = _rekey_by_question(build_reference(ledger, record, _manifest(), PROTOCOL))
    rows = []
    for variant in build_variants(record, PROTOCOL):
        reconstruction = reconstruct_episode(variant, PROTOCOL)
        score = score_episode(
            broken, reconstruction, PROTOCOL,
            episode_id=record.episode_id, variant_id=variant.variant_id,
        )
        assert isinstance(score, EpisodeAuditScore)
        rows.append(score.model_dump(mode="json"))
    summary = aggregate(rows, PROTOCOL, resamples=64)
    assert summary.zero_scored_units is True
    assert summary.scoring_alert is not None

    healthy = aggregate(
        [
            _score(record, ledger, variant.variant_id).model_dump(mode="json")
            for variant in build_variants(record, PROTOCOL)
        ],
        PROTOCOL,
        resamples=64,
    )
    assert healthy.zero_scored_units is False
    assert healthy.scored_field_units == 5 * 6
    assert healthy.run_class == "fixture"


def test_a_protocol_question_without_a_reference_derivation_is_excluded_not_guessed() -> None:
    """An added question must widen ``reference_unavailable``, never invent an answer.

    This keeps the exclusion category meaningful: it exists for facts the privileged ledger cannot
    supply, and a run whose new question is unanswerable must still score its other questions.
    """
    extended = ProtocolConfig.model_validate(PROTOCOL.model_dump())
    extended.audit.questions.append(
        type(PROTOCOL.audit.questions[0])(
            question_id="Q9_future_question",
            text="A question added after the reference module was frozen.",
            answer_fields=[AnswerField(name="unknown_fact", kind="identifier")],
        )
    )
    record, ledger = _episode(suppress_capture=False)
    reference = build_reference(ledger, record, _manifest(), extended)
    reconstruction = reconstruct_episode(
        {v.variant_id: v for v in build_variants(record, extended)}["provenance_rich"], extended
    )
    score = score_episode(
        reference, reconstruction, extended,
        episode_id=record.episode_id, variant_id="provenance_rich",
    )
    assert isinstance(score, EpisodeAuditScore)
    assert score.categories["Q9_future_question"] == "reference_unavailable"
    assert score.field_reference_unavailable == 1
    assert score.scored_fields == 6, "the other six facts must still be scored"
    assert score.correct == len(PROTOCOL.audit.questions)
