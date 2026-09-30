"""Independent evaluation outcome schema.

Shape only. The semantics live in :mod:`colosseum_assurance.evaluation.spec` and
:mod:`colosseum_assurance.evaluation.evaluator`, which are implemented separately from the monitors.
``assess_obligations`` must never receive an :class:`~colosseum_assurance.schemas.EpisodeRecord`:
obligation verdicts come from privileged world evidence alone. Monitor-relative quantities
(acceptance, missed detection, detection delay) necessarily read the *recorded* monitor verdicts, which
is different from calling the monitor's verdict functions.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from colosseum_assurance.evaluation.measurement_models import EpisodeMeasurements, ProxyDiagnostic
from colosseum_assurance.schemas import StrictModel, Verdict

EVALUATOR_VERSION = "evaluator-v2.0.0"
FALSE_ASSURANCE_SEMANTICS = "ascertainable_v2"


class ObligationOutcome(StrictModel):
    """Independent verdict for one frozen obligation in one episode."""

    obligation_id: str
    category: Literal["physical", "procedural"]
    verdict: Verdict
    first_violation_sim_time_s: float | None = None
    violation_count: int = 0
    evidence: str = ""
    measurements: dict[str, float] = Field(default_factory=dict)
    unknown_reason: str | None = None


class EpisodeOutcome(StrictModel):
    """Everything the independent evaluator establishes about one episode."""

    schema_version: str = "2.0.0"
    evaluator_version: str = EVALUATOR_VERSION
    false_assurance_semantics: str = FALSE_ASSURANCE_SEMANTICS
    episode_id: str
    scenario_id: str
    arm_id: str
    run_class: Literal["fixture", "pilot", "heldout", "smoke"]
    protocol_hash: str
    cell_id: str | None = None
    observation_delay_level: str | None = None
    supervision_delay_level: str | None = None

    completeness: Literal["complete", "incomplete"]
    incomplete_reason: str | None = None
    termination_reason: str

    episode_verdict: Verdict
    physical_verdict: Verdict
    procedural_verdict: Verdict
    obligations: dict[str, ObligationOutcome] = Field(default_factory=dict)

    physical_violation: bool = False
    procedural_violation: bool = False
    any_violation: bool = False
    first_violation_sim_time_s: float | None = None

    mission_completed: bool = False
    mission_completed_safely: bool = False
    completion_time_s: float | None = None

    accepted_by_monitor: bool | None = Field(
        default=None, description="None when the arm has no monitor: acceptance is undefined, not False."
    )
    acceptance_reason: str = ""
    monitor_step_coverage: float | None = None
    false_assurance: bool | None = Field(
        default=None,
        description=("True: accepted and independently VIOLATION. False: accepted and independently "
                     "PASS/NOT_APPLICABLE. None: not accepted or independent evidence UNKNOWN."),
    )

    missed_detection: bool | None = None
    detection_delay_s: float | None = None
    monitor_verdict_at_violation: str | None = None

    interventions: int = 0
    first_intervention_sim_time_s: float | None = None
    suspended: bool = False
    abandoned: bool = False
    unnecessary_intervention: bool | None = Field(
        default=None,
        description=(
            "True only when an independent criterion shows no obligation was at risk: the episode has no "
            "independently assessed violation and the counterfactual unguarded arm also had none."
        ),
    )

    truth_coverage_fraction: float = 1.0
    measurements: EpisodeMeasurements | None = Field(
        default=None, description="None in historical outcomes that did not retain these measurements."
    )
    proxy_diagnostic: ProxyDiagnostic | None = None
    notes: str = ""
    diagnostics: dict[str, Any] = Field(default_factory=dict)
