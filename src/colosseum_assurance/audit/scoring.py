"""Scoring of offline reconstruction against the privileged answer key.

Offline record ablation cannot change flight safety: it changes only what an auditor can establish
afterwards. Every number produced here is a statement about retained evidence, never about the
behaviour of the vehicle, and no figure derived from this module may be described as a safety effect.

Three outcomes, not two
-----------------------
``correct``, ``incorrect_confident`` and ``insufficient_evidence`` are counted separately, because an
abstention and a confident error are different failures: an auditor who knows the record is silent is
in a very different position from one who is confidently misled (research-acceptance.md section 5).
A fourth bookkeeping category, ``reference_unavailable``, covers rows where the privileged ledger
itself cannot answer; those rows are excluded from the denominator and reported, so an unanswerable
question never inflates or deflates a rate.

Uncertainty is scenario level
-----------------------------
Matched arms and audit questions share a scenario realization. Every interval resamples entire
scenarios, retaining all their arms and answer rows. One scenario cannot supply an uncertainty
interval, and historical scores without scenario identity retain counts but no invented clustering.

Two call paths, one aggregation
-------------------------------
In-process callers pass pydantic objects to :func:`score_run`. The CLI serializes both sides to JSON
and calls :func:`score_episode` per (episode, variant), then :func:`aggregate`. Both paths build the
same row list and share :func:`_summarise`, so the two routes cannot drift apart.

The scored unit is the answer FIELD
-----------------------------------
``AuditQuestion`` declares several facts with different units and tolerances, so a question-level
comparison cannot be right: a position must be compared as a distance in metres and its age in seconds.
Every declared field is therefore looked up by the composite key ``"<question_id>.<field_name>"`` -- the
key the privileged reference writes and the reconstruction now also writes -- compared with that field's
own tolerance, and then rolled up per question so the existing per-question reporting keeps working.

This module previously looked the reference up by the bare question id. Nothing matched, every question
became ``reference_unavailable``, every denominator was zero, and the audit reported no error at all. A
run that scores nothing is therefore now stated in machine-readable form on the summary
(:attr:`AuditScoreSummary.zero_scored_units` and :attr:`AuditScoreSummary.scoring_alert`), because the
failure this module has to protect against is the silent one.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, NamedTuple

from pydantic import Field

from colosseum_assurance.analysis.stats import (
    BootstrapEstimate,
    ProportionEstimate,
    cluster_bootstrap_statistic,
)
from colosseum_assurance.audit.reconstructor import EpisodeReconstruction, ReconstructedField
from colosseum_assurance.audit.reference import EpisodeReference, ReferenceAnswer, answer_key
from colosseum_assurance.protocol.spec import AnswerField, AuditQuestion, ProtocolConfig
from colosseum_assurance.schemas import StrictModel

__all__ = [
    "AUDIT_BOOTSTRAP_RESAMPLES",
    "SCORE_CATEGORIES",
    "AuditAnalysis",
    "AuditScoreSummary",
    "EpisodeAuditScore",
    "FieldScore",
    "QuestionScore",
    "VariantScore",
    "aggregate",
    "score_episode",
    "score_run",
]

#: The three scored outcomes plus the excluded bookkeeping category.
SCORE_CATEGORIES = ("correct", "incorrect_confident", "insufficient_evidence", "reference_unavailable")

#: Reduced resample count for the audit bootstrap. ``cluster_bootstrap_statistic`` rebuilds an index
#: array per replicate in Python, and the audit set is at most a few thousand rows; 2000 percentile
#: replicates put the Monte-Carlo error on the interval endpoints far below the sampling width that
#: the interval itself reports, so the extra 8000 replicates of the primary analysis buy nothing here.
AUDIT_BOOTSTRAP_RESAMPLES = 2000

_UNDEFINED_REASON = "no episode has a privileged reference answer for this question"


class EpisodeAuditScore(StrictModel):
    """One episode scored under one record variant, as written by the per-episode CLI step.

    The counts without a prefix are per QUESTION (the reporting unit the figures use); the ``field_*``
    counts are per declared answer field (the unit that is actually compared). Both are written out so a
    reader can see the denominator that produced a rate, and so an empty audit is visible in the row
    itself rather than only in the aggregate.
    """

    episode_id: str
    scenario_id: str | None = None
    variant_id: str
    run_class: str = "unknown"
    categories: dict[str, str] = Field(default_factory=dict)
    field_categories: dict[str, str] = Field(
        default_factory=dict, description="'question_id.field_name' -> one of SCORE_CATEGORIES."
    )
    field_comparisons: dict[str, str] = Field(
        default_factory=dict,
        description="Why each field got its category, including the measured error and the tolerance.",
    )
    correct: int = Field(default=0, ge=0)
    incorrect_confident: int = Field(default=0, ge=0)
    insufficient_evidence: int = Field(default=0, ge=0)
    reference_unavailable: int = Field(default=0, ge=0)
    field_correct: int = Field(default=0, ge=0)
    field_incorrect_confident: int = Field(default=0, ge=0)
    field_insufficient_evidence: int = Field(default=0, ge=0)
    field_reference_unavailable: int = Field(default=0, ge=0)
    reference_decision_step_index: int | None = None
    reference_decision_branch: str = "undefined"
    reconstructed_decision_step_index: int | None = None
    reconstructed_decision_branch: str = "undefined"

    @property
    def decision_step_agrees(self) -> bool | None:
        """None when either side has no decision step; the audit reads this per branch, not pooled."""
        if self.reference_decision_step_index is None or self.reconstructed_decision_step_index is None:
            return None
        return self.reference_decision_step_index == self.reconstructed_decision_step_index

    @property
    def scored_fields(self) -> int:
        """Fields that entered a denominator: everything the privileged reference could answer."""
        return (
            self.field_correct + self.field_incorrect_confident + self.field_insufficient_evidence
        )


class AuditProportionEstimate(ProportionEstimate):
    """A row proportion with uncertainty clustered over independent scenarios."""

    method: Literal["cluster_percentile", "none"] = "none"
    n_units: int = Field(default=0, ge=0)
    unit: str = "scenario_realization"
    interval_unavailable_reason: str | None = None


class FieldScore(StrictModel):
    """Per (variant, question, answer field) counts, with the tolerance the comparison used.

    The tolerance and unit are carried here so a reported rate can be read without opening the protocol,
    and so a later tolerance change is visible in the artifact it produced.
    """

    variant_id: str
    question_id: str
    field_name: str
    kind: str = ""
    unit: str = ""
    tolerance: float | None = None
    n_episodes: int = Field(ge=0, description="Denominator actually scored (reference-unavailable excluded).")
    correct: int = Field(ge=0)
    incorrect_confident: int = Field(ge=0)
    insufficient_evidence: int = Field(ge=0)
    reference_unavailable: int = Field(default=0, ge=0)
    correct_rate: AuditProportionEstimate


class QuestionScore(StrictModel):
    """Per (variant, question) counts and episode-level rates, plus the fields they roll up from."""

    variant_id: str
    question_id: str
    n_episodes: int = Field(ge=0, description="Denominator actually scored (reference-unavailable excluded).")
    correct: int = Field(ge=0)
    incorrect_confident: int = Field(ge=0)
    insufficient_evidence: int = Field(ge=0)
    reference_unavailable: int = Field(default=0, ge=0)
    correct_rate: AuditProportionEstimate
    incorrect_confident_rate: AuditProportionEstimate
    insufficient_evidence_rate: AuditProportionEstimate
    per_field: dict[str, FieldScore] = Field(default_factory=dict)


class VariantScore(StrictModel):
    """One record variant: its per-question scores and its pooled correct rate."""

    variant_id: str
    removed_fields: list[str] = Field(default_factory=list)
    retains_redundant_routes: bool = Field(
        default=False,
        description=(
            "Copied from the frozen ablation. A variant that keeps a redundant recovery route did not "
            "remove the information, and the report and figure label it as a probe rather than a removal."
        ),
    )
    n_episodes: int = Field(ge=0)
    n_scenarios: int = Field(default=0, ge=0)
    scored_units: int = Field(default=0, ge=0, description="Scored (question, episode) rows.")
    scored_field_units: int = Field(default=0, ge=0, description="Scored (field, episode) rows.")
    per_question: dict[str, QuestionScore] = Field(default_factory=dict)
    overall_correct_rate: BootstrapEstimate


class AuditScoreSummary(StrictModel):
    """The complete audit result for one run class under one frozen protocol."""

    protocol_hash: str
    protocol_short_hash: str
    run_class: str
    n_episodes: int = Field(ge=0)
    n_scenarios: int = Field(default=0, ge=0)
    question_ids: list[str] = Field(default_factory=list)
    variant_ids: list[str] = Field(default_factory=list)
    variants: dict[str, VariantScore] = Field(default_factory=dict)
    scored_units: int = Field(
        default=0, ge=0,
        description="(variant, question, episode) rows that entered a denominator.",
    )
    excluded_units: int = Field(
        default=0, ge=0, description="Rows excluded because the privileged reference could not answer."
    )
    scored_field_units: int = Field(default=0, ge=0, description="(variant, field, episode) rows scored.")
    excluded_field_units: int = Field(default=0, ge=0)
    zero_scored_units: bool = Field(
        default=False,
        description=(
            "True when rows were submitted but none could be scored. A key mismatch between the "
            "reference, the reconstruction and this module once produced exactly that, and it looked "
            "like a clean audit. It must never again be readable only by noticing an absence."
        ),
    )
    scoring_alert: str | None = Field(
        default=None, description="Human-readable statement of the same condition, or None when healthy."
    )
    notes: str = ""


#: Alias for the CLI, which imports the aggregated object under a different name.
AuditAnalysis = AuditScoreSummary


# --------------------------------------------------------------------------------------
# Comparison of one answer FIELD
# --------------------------------------------------------------------------------------
#: Strings accepted as the two truth values of a ``boolean`` answer field. Applied to both sides, so a
#: reference that says "true" and a reconstruction that says "True" are the same answer, while an
#: identifier that happens to read "1" is never mapped (identifiers are compared as written).
_TRUE_TOKENS = frozenset({"true", "yes", "1", "1.0"})
_FALSE_TOKENS = frozenset({"false", "no", "0", "0.0"})


def _canonical(value: str | float | None, *, boolean: bool = False) -> str:
    """Case- and whitespace-insensitive comparison form."""
    text = " ".join(str(value).strip().split()).casefold()
    if boolean and text in _TRUE_TOKENS:
        return "true"
    if boolean and text in _FALSE_TOKENS:
        return "false"
    return text


def _numeric(value: Any) -> float | None:
    """A finite float from a number or a numeric string, else None. ``bool`` is never a measurement."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _tolerance_of(field: AnswerField) -> float:
    """``None`` means an exact match is required, which is a tolerance of zero."""
    return 0.0 if field.tolerance is None else float(field.tolerance)


def _compare_numeric(
    field: AnswerField, reference: ReferenceAnswer, answer: ReconstructedField
) -> tuple[str, str]:
    """Compare a numeric field in its own declared unit.

    A position is compared as a Euclidean distance in metres against the metre tolerance; an age is
    compared as a difference in seconds against the second tolerance. An answer that cannot be read in
    the declared unit at all (a scalar offered for a position, text offered for an age) is counted as
    ``incorrect_confident``: the procedure asserted an answer, and an answer that does not answer the
    declared field is not an abstention.
    """
    tolerance = _tolerance_of(field)
    unit = field.unit or "units"
    if reference.vector_value is not None:
        expected = [float(v) for v in reference.vector_value]
        candidate = answer.vector_value
        if candidate is None or len(candidate) != len(expected):
            got = "no vector" if candidate is None else f"{len(candidate)} component(s)"
            return "incorrect_confident", (
                f"{field.name}: the reference is a {len(expected)}-component point but the "
                f"reconstruction offered {got}, so the answer cannot be read in {unit}"
            )
        if any(not math.isfinite(float(v)) for v in candidate):
            return "incorrect_confident", f"{field.name}: the reconstructed point is not finite"
        error = math.dist([float(v) for v in candidate], expected)
        category = "correct" if error <= tolerance else "incorrect_confident"
        return category, (
            f"{field.name}: distance {error:.3f} {unit} "
            f"{'<=' if category == 'correct' else '>'} tolerance {tolerance:.3f} {unit}"
        )
    expected_scalar = reference.numeric_value
    if expected_scalar is None:
        expected_scalar = _numeric(reference.value)
    if expected_scalar is None:
        return "reference_unavailable", (
            f"{field.name}: the privileged reference carries no value readable in {unit}, so this field "
            "cannot be scored either way"
        )
    candidate_scalar = answer.numeric_value
    if candidate_scalar is None:
        candidate_scalar = _numeric(answer.value)
    if candidate_scalar is None:
        return "incorrect_confident", (
            f"{field.name}: the reconstruction answered {answer.value!r}, which cannot be read in {unit}"
        )
    error = abs(candidate_scalar - expected_scalar)
    category = "correct" if error <= tolerance else "incorrect_confident"
    return category, (
        f"{field.name}: |{candidate_scalar:.3f} - {expected_scalar:.3f}| = {error:.3f} {unit} "
        f"{'<=' if category == 'correct' else '>'} tolerance {tolerance:.3f} {unit}"
    )


def _compare_text(
    field: AnswerField, reference: ReferenceAnswer, answer: ReconstructedField
) -> tuple[str, str]:
    """Compare an identifier, categorical or boolean field exactly, after normalisation.

    ``ReferenceAnswer.equivalent_values`` holds other wordings the privileged ledger literally recorded
    for the same fact (the monitor rationale beside the intervention category, say). Accepting them is
    not a loosened tolerance: each is a string the ledger itself recorded for this fact, and rejecting
    them would score vocabulary rather than reconstructability.
    """
    boolean = field.kind == "boolean"
    got = _canonical(answer.value, boolean=boolean)
    expected = _canonical(reference.value, boolean=boolean)
    if got == expected:
        return "correct", f"{field.name}: {answer.value!r} matches the reference answer"
    for alternative in reference.equivalent_values:
        if got == _canonical(alternative, boolean=boolean):
            return "correct", (
                f"{field.name}: {answer.value!r} matches {alternative!r}, an equally recorded wording of "
                f"the reference answer {reference.value!r}"
            )
    return "incorrect_confident", (
        f"{field.name}: {answer.value!r} != reference {reference.value!r}"
        + (f" (equivalents {reference.equivalent_values})" if reference.equivalent_values else "")
    )


def _compare_field(
    field: AnswerField, reference: ReferenceAnswer | None, answer: ReconstructedField | None
) -> tuple[str, str]:
    """Categorize one declared answer field and say why, in one line."""
    if reference is None:
        return "reference_unavailable", (
            f"{field.name}: the privileged reference produced no answer under the key "
            f"{field.name!r}; reference, reconstruction and scoring must use the same field key"
        )
    if not reference.available:
        return "reference_unavailable", (
            f"{field.name}: {reference.unavailable_reason or 'the privileged ledger cannot answer this'}"
        )
    if answer is None:
        return "insufficient_evidence", (
            f"{field.name}: the reconstruction produced no answer for this field"
        )
    if answer.confidence != "answered":
        return "insufficient_evidence", f"{field.name}: {answer.reasoning or 'the reconstruction abstained'}"
    if answer.value is None and answer.numeric_value is None and answer.vector_value is None:
        # Claimed as answered but carrying nothing. Nothing was established, so this is an abstention in
        # substance. It must never fall through to a string comparison, where a missing value and the
        # recorded answer "none" would look like the same answer.
        return "insufficient_evidence", (
            f"{field.name}: the reconstruction reported this field as answered but supplied no value"
        )
    if reference.value is None and reference.numeric_value is None and reference.vector_value is None:
        return "reference_unavailable", (
            f"{field.name}: the privileged reference is marked available but carries no value, so it "
            "cannot be an answer key for this field"
        )
    if field.kind == "numeric":
        return _compare_numeric(field, reference, answer)
    if field.kind == "timestamp":
        # A timestamp may be frozen as seconds or as text. Compare it numerically when the reference is
        # a number, and exactly otherwise, rather than dropping the row for being unreadable.
        numeric_reference = reference.numeric_value
        if numeric_reference is None:
            numeric_reference = _numeric(reference.value)
        if reference.vector_value is not None or numeric_reference is not None:
            return _compare_numeric(field, reference, answer)
    return _compare_text(field, reference, answer)


def _rollup_category(field_categories: Sequence[str]) -> str:
    """Roll answer fields up to their question.

    Correct only when every scored field is correct; one confident error makes the question a confident
    error (an auditor was misled about part of it); otherwise the question is an abstention. A question
    is ``reference_unavailable`` only when the privileged ledger could answer none of its fields, so a
    question is never dropped from the denominator merely because one of several fields is unanswerable.
    """
    scored = [c for c in field_categories if c != "reference_unavailable"]
    if not scored:
        return "reference_unavailable"
    if "incorrect_confident" in scored:
        return "incorrect_confident"
    if "insufficient_evidence" in scored:
        return "insufficient_evidence"
    return "correct"


class _Classification(NamedTuple):
    """One episode under one variant: question categories, field categories, and the reasons."""

    questions: dict[str, str]
    fields: dict[str, str]
    comparisons: dict[str, str]


def _classify(
    reconstruction: EpisodeReconstruction,
    reference: EpisodeReference,
    protocol: ProtocolConfig,
) -> _Classification:
    """Categorize every declared answer field of every audit question for one episode.

    A field the reconstruction never addressed counts as ``insufficient_evidence``: silence is an
    abstention, and scoring it as an error would punish an honest procedure. The per-question category
    is derived from the fields, never compared separately, so the two can never disagree.
    """
    if reconstruction.episode_id != reference.episode_id:
        raise ValueError(
            f"reconstruction {reconstruction.episode_id!r} and reference {reference.episode_id!r} "
            "describe different episodes"
        )
    questions: dict[str, str] = {}
    fields: dict[str, str] = {}
    comparisons: dict[str, str] = {}
    for question in protocol.audit.questions:
        qid = question.question_id
        per_question: list[str] = []
        for declared in question.answer_fields:
            key = answer_key(qid, declared.name)
            category, why = _compare_field(
                declared, reference.answers.get(key), reconstruction.field_answers.get(key)
            )
            fields[key] = category
            comparisons[key] = why
            per_question.append(category)
        questions[qid] = _rollup_category(per_question)
    return _Classification(questions=questions, fields=fields, comparisons=comparisons)


# --------------------------------------------------------------------------------------
# Normalisation of the serialized CLI call path
# --------------------------------------------------------------------------------------
def _looks_like_reconstruction(payload: Mapping[str, Any]) -> bool:
    """A reconstruction dict carries ``variant_id``; its answers carry ``confidence``."""
    if "variant_id" in payload:
        return True
    answers = payload.get("answers")
    if isinstance(answers, Mapping):
        return any(isinstance(a, Mapping) and "confidence" in a for a in answers.values())
    return False


def _as_reference(obj: Any) -> EpisodeReference:
    if isinstance(obj, EpisodeReference):
        return obj
    if isinstance(obj, Mapping):
        return EpisodeReference.model_validate(dict(obj))
    raise TypeError(f"expected an EpisodeReference or its JSON form, got {type(obj).__name__!r}")


def _as_reconstruction(obj: Any) -> EpisodeReconstruction:
    if isinstance(obj, EpisodeReconstruction):
        return obj
    if isinstance(obj, Mapping):
        return EpisodeReconstruction.model_validate(dict(obj))
    raise TypeError(f"expected an EpisodeReconstruction or its JSON form, got {type(obj).__name__!r}")


def _order_sides(first: Any, second: Any) -> tuple[EpisodeReconstruction, EpisodeReference]:
    """Accept (reconstruction, reference) or (reference, reconstruction), in object or dict form.

    The two call paths of this module disagree on argument order, and a silently swapped pair would
    score every question wrong while looking healthy, so the sides are identified by shape instead of
    by position.
    """
    first_is_reconstruction = isinstance(first, EpisodeReconstruction) or (
        isinstance(first, Mapping) and _looks_like_reconstruction(first)
    )
    if first_is_reconstruction:
        return _as_reconstruction(first), _as_reference(second)
    return _as_reconstruction(second), _as_reference(first)


def score_episode(
    first: Any,
    second: Any,
    protocol: ProtocolConfig,
    *,
    episode_id: str | None = None,
    variant_id: str | None = None,
) -> dict[str, str] | EpisodeAuditScore:
    """Score one episode under one record variant.

    Two shapes are supported, distinguished by the keyword arguments:

    * ``score_episode(reconstruction, reference, protocol)`` returns the plain
      ``question_id -> category`` mapping used inside this module and by in-process callers.
    * ``score_episode(reference, reconstructed, protocol, episode_id=..., variant_id=...)`` returns an
      :class:`EpisodeAuditScore` for the CLI, which passes both sides as JSON dictionaries.

    The positional arguments may be given in either order and in either form; the sides are identified
    by their content.
    """
    reconstruction, reference = _order_sides(first, second)
    if reconstruction.scenario_id is not None and reconstruction.scenario_id != reference.scenario_id:
        raise ValueError("reconstruction and reference belong to different scenarios")
    classification = _classify(reconstruction, reference, protocol)
    categories = classification.questions
    if episode_id is None and variant_id is None:
        return categories
    resolved_episode = episode_id or reconstruction.episode_id
    resolved_variant = variant_id or reconstruction.variant_id
    if episode_id is not None and episode_id != reconstruction.episode_id:
        raise ValueError(
            f"episode_id {episode_id!r} does not match the reconstruction "
            f"({reconstruction.episode_id!r}); scoring the wrong episode would be undetectable later"
        )
    if variant_id is not None and variant_id != reconstruction.variant_id:
        raise ValueError(
            f"variant_id {variant_id!r} does not match the reconstruction "
            f"({reconstruction.variant_id!r})"
        )
    values = list(categories.values())
    field_values = list(classification.fields.values())
    return EpisodeAuditScore(
        episode_id=resolved_episode,
        scenario_id=reference.scenario_id,
        variant_id=resolved_variant,
        run_class=reference.run_class,
        categories=categories,
        field_categories=classification.fields,
        field_comparisons=classification.comparisons,
        correct=values.count("correct"),
        incorrect_confident=values.count("incorrect_confident"),
        insufficient_evidence=values.count("insufficient_evidence"),
        reference_unavailable=values.count("reference_unavailable"),
        field_correct=field_values.count("correct"),
        field_incorrect_confident=field_values.count("incorrect_confident"),
        field_insufficient_evidence=field_values.count("insufficient_evidence"),
        field_reference_unavailable=field_values.count("reference_unavailable"),
        reference_decision_step_index=reference.decision_step_index,
        reference_decision_branch=reference.decision_step_rule_branch,
        reconstructed_decision_step_index=reconstruction.decision_step_index,
        reconstructed_decision_branch=reconstruction.decision_step_rule_branch,
    )


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------
def _counts(outcomes: Sequence[str]) -> tuple[int, int, int, int]:
    """``(n_scored, correct, incorrect_confident, insufficient_evidence)`` for one denominator."""
    scored = [c for c in outcomes if c != "reference_unavailable"]
    return (
        len(scored),
        sum(1 for c in scored if c == "correct"),
        sum(1 for c in scored if c == "incorrect_confident"),
        sum(1 for c in scored if c == "insufficient_evidence"),
    )


def _clustered_estimate(
    values: Sequence[float], scenarios: Sequence[str | None], confidence_level: float,
    seed: int, resamples: int,
) -> BootstrapEstimate:
    """Never manufacture independent units or a zero-width interval from a single scenario."""
    known = {s for s in scenarios if s is not None}
    if not values or None in scenarios or len(known) < 2:
        reason = ("no scored audit rows" if not values else
                  "scenario identity is missing from historical audit rows" if None in scenarios else
                  "at least two independent scenario realizations are needed for an interval")
        return BootstrapEstimate(
            point=sum(values) / len(values) if values else None,
            n_units=len(known), unit="scenario_realization", method="none",
            seed=seed, resamples=0, confidence_level=confidence_level, undefined_reason=reason,
        )
    return cluster_bootstrap_statistic(
        values, [str(s) for s in scenarios], seed=seed, resamples=resamples,
        confidence_level=confidence_level, unit="scenario_realization",
    )


def _cluster_rate(
    outcomes: Sequence[str], scenarios: Sequence[str | None], category: str,
    confidence_level: float, seed: int, resamples: int,
) -> AuditProportionEstimate:
    scored = [(c, s) for c, s in zip(outcomes, scenarios, strict=True)
              if c != "reference_unavailable"]
    values = [float(c == category) for c, _ in scored]
    estimate = _clustered_estimate(values, [s for _, s in scored], confidence_level, seed, resamples)
    return AuditProportionEstimate(
        numerator=sum(c == category for c, _ in scored), denominator=len(scored),
        point=estimate.point, ci_low=estimate.ci_low, ci_high=estimate.ci_high,
        confidence_level=confidence_level, method=estimate.method,
        n_units=estimate.n_units, unit=estimate.unit,
        undefined_reason=_UNDEFINED_REASON if not scored else None,
        interval_unavailable_reason=estimate.undefined_reason,
    )


def _field_score(
    variant_id: str,
    question: AuditQuestion,
    field: AnswerField,
    outcomes: Sequence[str],
    confidence_level: float,
    scenarios: Sequence[str | None], seed: int, resamples: int,
) -> FieldScore:
    n, correct, incorrect, abstained = _counts(outcomes)
    return FieldScore(
        variant_id=variant_id,
        question_id=question.question_id,
        field_name=field.name,
        kind=str(field.kind),
        unit=str(field.unit),
        tolerance=None if field.tolerance is None else float(field.tolerance),
        n_episodes=n,
        correct=correct,
        incorrect_confident=incorrect,
        insufficient_evidence=abstained,
        reference_unavailable=len(outcomes) - n,
        correct_rate=_cluster_rate(outcomes, scenarios, "correct", confidence_level, seed, resamples),
    )


def _question_score(
    variant_id: str,
    question_id: str,
    outcomes: Sequence[str],
    confidence_level: float,
    per_field: dict[str, FieldScore],
    scenarios: Sequence[str | None], seed: int, resamples: int,
) -> QuestionScore:
    n, correct, incorrect, abstained = _counts(outcomes)
    return QuestionScore(
        variant_id=variant_id,
        question_id=question_id,
        n_episodes=n,
        correct=correct,
        incorrect_confident=incorrect,
        insufficient_evidence=abstained,
        reference_unavailable=len(outcomes) - n,
        correct_rate=_cluster_rate(outcomes, scenarios, "correct", confidence_level, seed, resamples),
        incorrect_confident_rate=_cluster_rate(
            outcomes, scenarios, "incorrect_confident", confidence_level, seed, resamples),
        insufficient_evidence_rate=_cluster_rate(
            outcomes, scenarios, "insufficient_evidence", confidence_level, seed, resamples),
        per_field=per_field,
    )


def _summarise(
    rows: Sequence[tuple[str, str, str | None, Mapping[str, str], Mapping[str, str]]],
    protocol: ProtocolConfig,
    run_class: str,
    seed: int | None,
    resamples: int | None,
) -> AuditScoreSummary:
    """Aggregate ``(variant_id, episode_id, question categories, field categories)`` rows.

    The single aggregation path for both call routes. Field rows are aggregated beside the question rows
    rather than instead of them: the figures and the report are per question, while the per-field
    denominators are what make a tolerance and a unit auditable.
    """
    questions = list(protocol.audit.questions)
    question_ids = [q.question_id for q in questions]
    confidence_level = protocol.analysis.confidence_level
    bootstrap_seed = protocol.analysis.bootstrap_seed if seed is None else seed
    bootstrap_resamples = AUDIT_BOOTSTRAP_RESAMPLES if resamples is None else resamples
    ablation_by_id = {ablation.ablation_id: ablation for ablation in protocol.audit.record_variants}

    outcomes: dict[str, dict[str, list[str]]] = {}
    field_outcomes: dict[str, dict[str, list[str]]] = {}
    pooled: dict[str, list[tuple[str | None, float]]] = {}
    scenarios_by_variant: dict[str, list[str | None]] = {}
    episodes_by_variant: dict[str, list[str]] = {}
    variant_order: list[str] = []
    all_episodes: list[str] = []
    rows_without_field_categories = 0

    for variant_id, episode_id, scenario_id, categories, field_categories in rows:
        if variant_id not in outcomes:
            outcomes[variant_id] = {qid: [] for qid in question_ids}
            field_outcomes[variant_id] = {
                answer_key(q.question_id, f.name): [] for q in questions for f in q.answer_fields
            }
            pooled[variant_id] = []
            scenarios_by_variant[variant_id] = []
            episodes_by_variant[variant_id] = []
            variant_order.append(variant_id)
        if episode_id not in episodes_by_variant[variant_id]:
            episodes_by_variant[variant_id].append(episode_id)
        if episode_id not in all_episodes:
            all_episodes.append(episode_id)
        scenarios_by_variant[variant_id].append(scenario_id)
        if not field_categories:
            rows_without_field_categories += 1
        for qid in question_ids:
            category = categories.get(qid, "reference_unavailable")
            outcomes[variant_id][qid].append(category)
            if category != "reference_unavailable":
                pooled[variant_id].append((scenario_id, 1.0 if category == "correct" else 0.0))
        for key in field_outcomes[variant_id]:
            field_outcomes[variant_id][key].append(
                field_categories.get(key, "reference_unavailable")
            )

    variants: dict[str, VariantScore] = {}
    for variant_id in variant_order:
        per_question: dict[str, QuestionScore] = {}
        for question in questions:
            per_field = {
                field.name: _field_score(
                    variant_id, question, field,
                    field_outcomes[variant_id][answer_key(question.question_id, field.name)],
                    confidence_level,
                    scenarios_by_variant[variant_id], bootstrap_seed, bootstrap_resamples,
                )
                for field in question.answer_fields
            }
            per_question[question.question_id] = _question_score(
                variant_id, question.question_id, outcomes[variant_id][question.question_id],
                confidence_level, per_field,
                scenarios_by_variant[variant_id], bootstrap_seed, bootstrap_resamples,
            )
        variant_rows = pooled[variant_id]
        ablation = ablation_by_id.get(variant_id)
        variants[variant_id] = VariantScore(
            variant_id=variant_id,
            removed_fields=list(ablation.removed_fields) if ablation is not None else [],
            retains_redundant_routes=bool(ablation.retains_redundant_routes) if ablation else False,
            n_episodes=len(episodes_by_variant[variant_id]),
            n_scenarios=len({s for s in scenarios_by_variant[variant_id] if s is not None}),
            scored_units=sum(score.n_episodes for score in per_question.values()),
            scored_field_units=sum(
                field.n_episodes for score in per_question.values() for field in score.per_field.values()
            ),
            per_question=per_question,
            overall_correct_rate=_clustered_estimate(
                [value for _, value in variant_rows],
                [scenario for scenario, _ in variant_rows],
                confidence_level, bootstrap_seed, bootstrap_resamples,
            ),
        )

    scored_units = sum(variant.scored_units for variant in variants.values())
    excluded_units = sum(
        score.reference_unavailable
        for variant in variants.values()
        for score in variant.per_question.values()
    )
    scored_field_units = sum(variant.scored_field_units for variant in variants.values())
    excluded_field_units = sum(
        field.reference_unavailable
        for variant in variants.values()
        for score in variant.per_question.values()
        for field in score.per_field.values()
    )
    zero_scored = bool(rows) and scored_units == 0
    alert = None
    if zero_scored:
        alert = (
            f"NO AUDIT UNIT WAS SCORED: {len(rows)} (variant, episode) rows were submitted and all "
            f"{excluded_units} (variant, question, episode) rows were excluded as reference_unavailable. "
            "Every reconstruction accuracy in this summary is undefined. Check that the reference, the "
            "reconstruction and the scorer agree on the 'question_id.field_name' answer key before "
            "reading any number here as a result."
        )
    legacy = (
        f" {rows_without_field_categories} row(s) carried no per-field categories (scored before "
        "field-level scoring existed) and contribute to the per-question denominators only."
        if rows_without_field_categories else ""
    )
    notes = (
        "Offline record ablation cannot change flight safety; it changes only what an auditor can "
        f"establish afterwards. {scored_units} (variant, question, episode) rows were scored and "
        f"{excluded_units} were excluded because the privileged reference could not answer; at field "
        f"level {scored_field_units} were scored and {excluded_field_units} excluded. Interval method: "
        f"cluster bootstrap ({bootstrap_resamples} resamples, seed {bootstrap_seed}) over scenario "
        "realizations for field, question and pooled rates; all matched arms and question rows travel "
        "together. Intervals are unavailable with fewer than two scenarios or missing scenario IDs."
        + legacy
        + (f" {alert}" if alert else "")
    )
    return AuditScoreSummary(
        protocol_hash=protocol.content_hash(),
        protocol_short_hash=protocol.short_hash,
        run_class=run_class,
        n_episodes=len(all_episodes),
        n_scenarios=len({s for values in scenarios_by_variant.values() for s in values if s is not None}),
        question_ids=question_ids,
        variant_ids=variant_order,
        variants=variants,
        scored_units=scored_units,
        excluded_units=excluded_units,
        scored_field_units=scored_field_units,
        excluded_field_units=excluded_field_units,
        zero_scored_units=zero_scored,
        scoring_alert=alert,
        notes=notes,
    )


def score_run(
    pairs: Iterable[tuple[EpisodeReconstruction, EpisodeReference]],
    protocol: ProtocolConfig,
    run_class: str,
    *,
    seed: int | None = None,
    resamples: int | None = None,
) -> AuditScoreSummary:
    """Aggregate episode scores into the per-variant audit summary consumed by the figures module.

    ``pairs`` may mix variants: rows are grouped by ``reconstruction.variant_id``, so one call scores a
    whole run. The denominators are per (variant, question) and per (variant, question, field), because
    a fact can be unanswerable in the privileged ledger of some episodes only.
    """
    rows: list[tuple[str, str, str | None, Mapping[str, str], Mapping[str, str]]] = []
    for reconstruction, reference in pairs:
        classification = _classify(reconstruction, reference, protocol)
        rows.append((reconstruction.variant_id, reconstruction.episode_id,
                     reference.scenario_id,
                     classification.questions, classification.fields))
    return _summarise(rows, protocol, run_class, seed, resamples)


def aggregate(
    scores: list[Any],
    protocol: ProtocolConfig,
    *,
    seed: int | None = None,
    resamples: int | None = None,
) -> AuditScoreSummary:
    """Aggregate per-episode scores produced by :func:`score_episode` (objects or their JSON form).

    The run class is taken from the scored episodes rather than from an argument, so a summary can
    never be labelled with a run class that its episodes do not carry. Mixed inputs are labelled
    ``mixed`` instead of silently adopting the first value.
    """
    rows: list[tuple[str, str, str | None, Mapping[str, str], Mapping[str, str]]] = []
    run_classes: list[str] = []
    for entry in scores:
        score = entry if isinstance(entry, EpisodeAuditScore) else EpisodeAuditScore.model_validate(
            dict(entry) if isinstance(entry, Mapping) else entry
        )
        rows.append((score.variant_id, score.episode_id, score.scenario_id,
                     score.categories, score.field_categories))
        if score.run_class not in run_classes:
            run_classes.append(score.run_class)
    if not run_classes:
        run_class = "unknown"
    elif len(run_classes) == 1:
        run_class = run_classes[0]
    else:
        run_class = "mixed"
    return _summarise(rows, protocol, run_class, seed, resamples)
