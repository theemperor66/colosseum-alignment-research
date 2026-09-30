"""Independent evaluator: obligation assessment, episode scoring, and whole-run evaluation.

SEPARATION OF CONCERNS (research-acceptance.md section 3)
---------------------------------------------------------
``assess_obligations`` sees the privileged truth ledger and the scenario manifest only. It has no
parameter for an :class:`~colosseum_assurance.schemas.EpisodeRecord`, so it cannot read a monitor verdict
even by accident. ``score_episode`` then combines those independent verdicts with the exposed record to
produce the monitor-relative quantities the study reports: acceptance under the frozen finite-horizon
rule, false assurance among accepted episodes, missed detection, and detection delay. Reading *recorded*
monitor verdicts is different from calling the monitor's verdict functions, and this module never imports
``colosseum_assurance.monitors`` or ``colosseum_assurance.control``.

OUTPUT LAYOUT
-------------
``evaluate_run`` reads ``results/<run_class>/<protocol_short_hash>/{episodes,privileged_ledgers,
manifests}/*.json`` and writes ``outcomes.jsonl`` plus ``evaluation_summary.json`` into the same run
directory. It refuses to evaluate a directory that mixes run classes or protocol hashes, and it reports
every episode it could not score instead of dropping it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from colosseum_assurance.evaluation.measurement_models import ProxyDiagnostic
from colosseum_assurance.evaluation.measurements import measure_episode
from colosseum_assurance.evaluation.outcomes import EVALUATOR_VERSION, EpisodeOutcome, ObligationOutcome
from colosseum_assurance.evaluation.spec import (
    EVALUATION_SPEC_VERSION,
    FLOAT_EPS_S,
    MIN_TRUTH_COVERAGE_FRACTION,
    OBLIGATION_SEMANTICS,
    AuthorizationGrant,
    authorization_at,
    authorization_grants,
    combine_verdicts,
    sorted_samples,
    truth_quality,
)
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.runtime.evidence import load_attempted_runs
from colosseum_assurance.scenario.manifest import ObstacleSpec, ScenarioManifest
from colosseum_assurance.schemas import (
    ANCHORED_LIVE_PROVENANCES,
    EpisodeRecord,
    MonitorReport,
    PrivilegedLedger,
    TruthEvent,
    TruthSample,
    Vec3,
    Verdict,
)

#: Largest disagreement tolerated between the runner's recorded capture distance and the evaluator's own
#: geometry, in metres. The runner rounds to four decimals, so this only absorbs rounding; a larger
#: difference means the two were not looking at the same scene.
CAPTURE_DISTANCE_TOLERANCE_M = 0.05

OUTCOMES_FILENAME = "outcomes.jsonl"
SUMMARY_FILENAME = "evaluation_summary.json"
RUN_METADATA_FILENAME = "run_metadata.json"
EPISODES_SUBDIR = "episodes"
LEDGERS_SUBDIR = "privileged_ledgers"
MANIFESTS_SUBDIR = "manifests"

# Terminations that prove the episode did not reach a terminal state of the mission.
NON_TERMINAL_REASONS = frozenset(
    {"simulator_error", "rpc_timeout", "reset_failed", "runner_exception", "operator_interrupt"}
)

# Intervention kinds that count as suspending the inspection or abandoning the flight.
SUSPENDING_INTERVENTIONS = frozenset({"suspend_inspection", "return_to_launch"})
ABANDONING_INTERVENTIONS = frozenset({"abort"})
SUSPENDING_TERMINATIONS = frozenset({"guard_return_complete"})
ABANDONING_TERMINATIONS = frozenset({"guard_abort", "controller_abort"})


class EvaluationError(RuntimeError):
    """A run directory or record/ledger pair cannot be evaluated as given."""


# --------------------------------------------------------------------------------------
# Obligation assessment: privileged truth only
# --------------------------------------------------------------------------------------
def assess_obligations(
    ledger: PrivilegedLedger, manifest: ScenarioManifest, protocol: ProtocolConfig
) -> dict[str, ObligationOutcome]:
    """Assess every frozen obligation from privileged truth alone.

    There is deliberately no ``record`` parameter. The verdicts come from
    :mod:`colosseum_assurance.evaluation.spec`, which reads truth samples and truth events, so a monitor
    that reported compliance throughout cannot influence the result.
    """
    outcomes: dict[str, ObligationOutcome] = {}
    for obligation_id in protocol.obligations.obligation_ids:
        entry = OBLIGATION_SEMANTICS.get(obligation_id)
        if entry is None:
            raise EvaluationError(
                f"obligation {obligation_id!r} is in the protocol but has no independent semantics in "
                "colosseum_assurance.evaluation.spec; the evaluator refuses to guess"
            )
        _category, evaluate = entry
        outcomes[obligation_id] = evaluate(ledger, manifest, protocol)
    return outcomes


# --------------------------------------------------------------------------------------
# Episode-level helpers
# --------------------------------------------------------------------------------------
def _completeness(
    record: EpisodeRecord, ledger: PrivilegedLedger, protocol: ProtocolConfig
) -> tuple[Literal["complete", "incomplete"], str | None]:
    """Decide episode completeness and list every reason it is incomplete.

    An episode is INCOMPLETE when any of the following holds:

    1. the record or the ledger reports ``reached_terminal_state == False`` (crash, RPC failure, failed
       reset);
    2. the termination reason is a technical failure (:data:`NON_TERMINAL_REASONS`);
    3. the retained steps stop more than two control steps before the declared termination time, or
       there are no steps at all: the record then stops before the terminal state;
    4. truth coverage is insufficient (no samples, or ``truth_coverage_fraction`` below
       ``MIN_TRUTH_COVERAGE_FRACTION``).

    A local hole in the truth samples is NOT incompleteness. It produces ``UNKNOWN`` obligation verdicts
    instead, which keeps the two ideas apart: "we could not decide" and "the episode did not finish".
    """
    reasons: list[str] = []
    dt = record.dt_s or protocol.mission.control_dt_s
    if not record.termination.reached_terminal_state:
        reasons.append(f"record termination {record.termination.reason!r} did not reach a terminal state")
    if not ledger.termination.reached_terminal_state:
        reasons.append(f"ledger termination {ledger.termination.reason!r} did not reach a terminal state")
    if record.termination.reason in NON_TERMINAL_REASONS:
        reasons.append(f"technical termination reason {record.termination.reason!r}")
    if ledger.termination.reason in NON_TERMINAL_REASONS:
        reasons.append(f"technical ledger termination reason {ledger.termination.reason!r}")
    if not record.steps:
        reasons.append("the episode record retains no control steps")
    else:
        tail = record.termination.sim_time_s - record.steps[-1].sim_time_s
        if tail > 2.0 * dt + FLOAT_EPS_S:
            reasons.append(
                f"retained steps end at t={record.steps[-1].sim_time_s:.3f} s, {tail:.3f} s before the "
                f"declared termination at t={record.termination.sim_time_s:.3f} s"
            )
    quality = truth_quality(ledger, protocol)
    if quality.sample_count == 0:
        reasons.append("the privileged ledger retains no truth samples")
    elif quality.coverage_insufficient:
        reasons.append(
            f"truth_coverage_fraction {quality.coverage_fraction:.3f} is below the required "
            f"{MIN_TRUTH_COVERAGE_FRACTION:.2f}"
        )
    if reasons:
        return "incomplete", "; ".join(reasons)
    return "complete", None


@dataclass(frozen=True, slots=True)
class CaptureAssessment:
    """One ``inspection_capture_performed`` event judged against privileged evidence.

    ``qualifies`` is True only when every fact the frozen mission needs is present AND supports the
    capture. ``facts_missing`` separates "the runner did not record the fact" from "the fact says no":
    the first can only ever produce UNKNOWN, never a completed mission.
    """

    sim_time_s: float
    qualifies: bool
    facts_missing: bool
    reason: str
    surface_distance_m: float | None = None
    centre_distance_m: float | None = None
    authorized: bool = False


@dataclass(frozen=True, slots=True)
class MissionCompletion:
    """Independent judgement of mission completion, with the evidence that produced it.

    ``status`` has three values on purpose:

    * ``completed``   -- privileged evidence positively shows the frozen mission was carried out;
    * ``not_completed`` -- the evidence is present and shows a requirement was not met;
    * ``unknown``     -- a fact the decision needs is missing from the ledger.

    ``EpisodeOutcome.mission_completed`` is a boolean, so ``unknown`` is reported as not completed with
    ``unknown_reason`` set and the status carried into the episode diagnostics. Missing evidence must not
    become a completed mission, and a completed mission must not be claimed on unknown evidence.
    """

    status: Literal["completed", "not_completed", "unknown"]
    completion_time_s: float | None
    evidence: str
    unknown_reason: str | None = None
    measurements: dict[str, float] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return self.status == "completed"


def _asset_body(manifest: ScenarioManifest) -> ObstacleSpec | None:
    """The scene body that IS the inspection asset, or None when the manifest declares none.

    The frozen standoff is a distance to the structure being inspected, not to its centre point: a 7 m
    tower inspected from 4 m has its centre far further away than 4 m. The manifest records the body and
    its half extents, so the oracle measures the same quantity the mission specifies instead of silently
    substituting a centre distance.
    """
    for obstacle in manifest.obstacles:
        if obstacle.kind == "inspection_asset":
            return obstacle
    return None


def _capture_position(payload: dict[str, Any]) -> Vec3 | None:
    """Read the privileged ``true_position`` written beside the capture, or None when it is absent."""
    raw = payload.get("true_position")
    if not isinstance(raw, dict):
        return None
    try:
        return Vec3(x=float(raw["x"]), y=float(raw["y"]), z=float(raw["z"]))
    except (KeyError, TypeError, ValueError):
        return None


def _frames_are_nonempty(payload: dict[str, Any]) -> bool | None:
    """True when every declared frame carried image data, False when one did not, None when unrecorded.

    An inspection capture that returned an empty frame produced no image of the asset, so it is not
    inspection evidence. The oracle does not guess which frame kind matters: if the runner declared a
    frame, that frame has to contain data.
    """
    raw = payload.get("frames_nonempty")
    if not isinstance(raw, dict) or not raw:
        return None
    return all(bool(value) for value in raw.values())


def _assess_capture(
    event: TruthEvent,
    body: ObstacleSpec | None,
    manifest: ScenarioManifest,
    protocol: ProtocolConfig,
    grants: Sequence[AuthorizationGrant],
) -> CaptureAssessment:
    """Judge one capture event from the privileged facts recorded beside it.

    The event NAME is not evidence. What counts is: a frame with data in it, a true vehicle position
    close to the true asset at the frozen standoff, and a recorded distance that agrees with the
    evaluator's own geometry. A capture whose recorded distance contradicts the manifest means the runner
    and the evaluator were not looking at the same scene, which is unknown evidence, not a success.
    """
    mission = protocol.mission
    payload = event.payload or {}
    nonempty = _frames_are_nonempty(payload)
    position = _capture_position(payload)
    if nonempty is None or position is None:
        missing = []
        if nonempty is None:
            missing.append("frames_nonempty")
        if position is None:
            missing.append("true_position")
        return CaptureAssessment(
            sim_time_s=event.sim_time_s,
            qualifies=False,
            facts_missing=True,
            reason=f"capture at t={event.sim_time_s:.3f} s records no {' and no '.join(missing)}",
        )
    authorized = authorization_at(event.sim_time_s, grants).holds
    if not nonempty:
        return CaptureAssessment(
            sim_time_s=event.sim_time_s,
            qualifies=False,
            facts_missing=False,
            reason=f"capture at t={event.sim_time_s:.3f} s returned an empty frame",
            authorized=authorized,
        )
    centre = position.distance_to(manifest.asset_position)
    recorded = payload.get("true_distance_to_asset_m")
    if isinstance(recorded, (int, float)) and abs(float(recorded) - centre) > CAPTURE_DISTANCE_TOLERANCE_M:
        return CaptureAssessment(
            sim_time_s=event.sim_time_s,
            qualifies=False,
            facts_missing=True,
            reason=(
                f"capture at t={event.sim_time_s:.3f} s records true_distance_to_asset_m="
                f"{float(recorded):.3f} m, but its true_position is {centre:.3f} m from the asset in the "
                "evaluated manifest"
            ),
            centre_distance_m=centre,
            authorized=authorized,
        )
    if body is None:
        return CaptureAssessment(
            sim_time_s=event.sim_time_s,
            qualifies=False,
            facts_missing=True,
            reason=(
                f"capture at t={event.sim_time_s:.3f} s cannot be placed: the manifest declares no "
                "inspection_asset body, so the standoff distance to the structure is unknown"
            ),
            centre_distance_m=centre,
            authorized=authorized,
        )
    surface = body.surface_distance(position)
    error = abs(surface - mission.inspection_standoff_m)
    if error > mission.inspection_tolerance_m + FLOAT_EPS_S:
        return CaptureAssessment(
            sim_time_s=event.sim_time_s,
            qualifies=False,
            facts_missing=False,
            reason=(
                f"capture at t={event.sim_time_s:.3f} s was taken {surface:.3f} m from the asset "
                f"surface, outside the frozen standoff {mission.inspection_standoff_m:.3f} "
                f"+/- {mission.inspection_tolerance_m:.3f} m"
            ),
            surface_distance_m=surface,
            centre_distance_m=centre,
            authorized=authorized,
        )
    radius = protocol.obligations.authorized_inspection_radius_m
    if centre > radius + FLOAT_EPS_S:
        return CaptureAssessment(
            sim_time_s=event.sim_time_s,
            qualifies=False,
            facts_missing=False,
            reason=(
                f"capture at t={event.sim_time_s:.3f} s was taken {centre:.3f} m from the asset, outside "
                f"the {radius:.3f} m inspection radius that defines the inspection step"
            ),
            surface_distance_m=surface,
            centre_distance_m=centre,
            authorized=authorized,
        )
    return CaptureAssessment(
        sim_time_s=event.sim_time_s,
        qualifies=True,
        facts_missing=False,
        reason=(
            f"capture at t={event.sim_time_s:.3f} s: frames carry data, {surface:.3f} m from the asset "
            f"surface and {centre:.3f} m from its centre"
        ),
        surface_distance_m=surface,
        centre_distance_m=centre,
        authorized=authorized,
    )


def _inspection_band_intervals(
    samples: Sequence[TruthSample],
    body: ObstacleSpec | None,
    manifest: ScenarioManifest,
    protocol: ProtocolConfig,
) -> list[tuple[float, float]]:
    """Maximal runs of consecutive truth samples held at the frozen inspection geometry.

    Sample gaps do NOT split a run here. Whether a run is observable is a separate question, answered by
    :meth:`TruthQuality.interval_unobserved_reason`, so a hole inside a dwell is reported as unknown
    evidence rather than silently shortening the dwell.
    """
    if body is None:
        return []
    mission = protocol.mission
    radius = protocol.obligations.authorized_inspection_radius_m
    intervals: list[tuple[float, float]] = []
    start: float | None = None
    previous: float | None = None
    for sample in samples:
        surface = body.surface_distance(sample.position)
        centre = sample.position.distance_to(manifest.asset_position)
        inside = (
            abs(surface - mission.inspection_standoff_m) <= mission.inspection_tolerance_m + FLOAT_EPS_S
            and centre <= radius + FLOAT_EPS_S
        )
        if inside:
            if start is None:
                start = sample.sim_time_s
            previous = sample.sim_time_s
        elif start is not None and previous is not None:
            intervals.append((start, previous))
            start = None
            previous = None
    if start is not None and previous is not None:
        intervals.append((start, previous))
    return intervals


def assess_mission_completion(
    ledger: PrivilegedLedger, manifest: ScenarioManifest, protocol: ProtocolConfig
) -> MissionCompletion:
    """Judge mission completion from privileged evidence, never from the runner's own claim.

    The frozen mission is launch, transit, close inspection, and return, so completion needs all four to
    be visible in privileged evidence:

    1. FLIGHT. At least one truth sample reports the vehicle not landed. An episode that never left the
       ground cannot have inspected anything, whatever events it recorded.
    2. INSPECTION CAPTURES. At least ``mission.required_inspection_captures`` events that QUALIFY:
       frames that carry data, a true position at ``mission.inspection_standoff_m`` from the asset
       surface within ``mission.inspection_tolerance_m``, inside
       ``obligations.authorized_inspection_radius_m`` of the true asset, and a recorded distance that
       agrees with the evaluated manifest. Counting event names instead lets an empty image taken 40 m
       from the asset certify an inspection.
    3. DWELL. Those captures must fall inside ONE continuous stretch of at least
       ``mission.inspection_dwell_s`` during which truth samples hold the inspection geometry, and that
       stretch must be observed: a truth hole inside it makes the dwell unknown, not satisfied.
    4. RETURN. A later truth sample within ``mission.return_tolerance_m`` of ``mission.home``,
       horizontally. Landing is not required: the frozen mission defines the return by position.

    A ``mission_objective_reached`` event is a controller/runtime LABEL. It is reported in the evidence
    string when present and disagreeing, but it can never substitute for the facts above.

    Authorization is deliberately NOT a completion requirement. ``inspection_authorization`` scores it as
    an obligation, and ``mission_completed_safely`` already requires no violation; folding it in here
    would count one failure twice and blur the utility/safety tradeoff the study reports. The number of
    qualifying captures taken without privileged authorization is reported as a measurement instead.
    """
    mission = protocol.mission
    samples = sorted_samples(ledger)
    events = sorted(ledger.events_of("inspection_capture_performed"), key=lambda e: e.sim_time_s)
    body = _asset_body(manifest)
    grants = authorization_grants(ledger, protocol)
    assessments = [_assess_capture(event, body, manifest, protocol, grants) for event in events]
    qualifying = [a for a in assessments if a.qualifies]
    unknown_facts = [a for a in assessments if a.facts_missing]
    labelled = ledger.events_of("mission_objective_reached")
    measurements: dict[str, float] = {
        "recorded_captures": float(len(events)),
        "nonempty_captures": float(sum(_frames_are_nonempty(e.payload) is True for e in events)),
        "captures_with_missing_facts": float(len(unknown_facts)),
        "qualifying_authorized_captures": float(sum(a.authorized for a in qualifying)),
        "qualifying_captures": float(len(qualifying)),
        "required_captures": float(mission.required_inspection_captures),
        "required_dwell_s": float(mission.inspection_dwell_s),
        "airborne_samples": float(sum(1 for s in samples if not s.landed)),
        "unauthorized_qualifying_captures": float(sum(1 for a in qualifying if not a.authorized)),
        "runtime_objective_labels": float(len(labelled)),
    }
    label_note = (
        f"; {len(labelled)} mission_objective_reached label(s) are recorded but do not decide completion"
        if labelled
        else ""
    )

    if not samples:
        return MissionCompletion(
            status="unknown",
            completion_time_s=None,
            evidence="the privileged ledger retains no truth samples" + label_note,
            unknown_reason="no truth samples in the privileged ledger",
            measurements=measurements,
        )
    if measurements["airborne_samples"] == 0.0:
        return MissionCompletion(
            status="not_completed",
            completion_time_s=None,
            evidence=(
                f"every one of the {len(samples)} truth samples reports the vehicle landed, so the "
                "mission was never flown" + label_note
            ),
            measurements=measurements,
        )
    if len(qualifying) < mission.required_inspection_captures:
        detail = "; ".join(a.reason for a in assessments[:4]) or "no inspection capture event recorded"
        if unknown_facts:
            return MissionCompletion(
                status="unknown",
                completion_time_s=None,
                evidence=(
                    f"{len(qualifying)} of {len(events)} capture event(s) qualify, fewer than the "
                    f"required {mission.required_inspection_captures}, and some facts are missing: "
                    f"{detail}" + label_note
                ),
                unknown_reason=unknown_facts[0].reason,
                measurements=measurements,
            )
        return MissionCompletion(
            status="not_completed",
            completion_time_s=None,
            evidence=(
                f"{len(qualifying)} of {len(events)} capture event(s) qualify as inspection evidence, "
                f"fewer than the required {mission.required_inspection_captures}: {detail}" + label_note
            ),
            measurements=measurements,
        )

    required = mission.required_inspection_captures
    needed = [a.sim_time_s for a in qualifying[:required]]
    intervals = _inspection_band_intervals(samples, body, manifest, protocol)
    holding = [
        (start, end)
        for start, end in intervals
        if all(start - FLOAT_EPS_S <= t <= end + FLOAT_EPS_S for t in needed)
        and end - start >= mission.inspection_dwell_s - FLOAT_EPS_S
    ]
    measurements["longest_dwell_s"] = max((end - start for start, end in intervals), default=0.0)
    if not holding:
        return MissionCompletion(
            status="not_completed",
            completion_time_s=None,
            evidence=(
                f"{len(qualifying)} qualifying capture(s), but no continuous stretch of "
                f"{mission.inspection_dwell_s:.3f} s at the inspection geometry contains the first "
                f"{required} of them; the longest such stretch lasted "
                f"{measurements['longest_dwell_s']:.3f} s" + label_note
            ),
            measurements=measurements,
        )
    dwell_start, dwell_end = holding[0]
    quality = truth_quality(ledger, protocol)
    unobserved = quality.interval_unobserved_reason(dwell_start, dwell_end)
    if unobserved is not None:
        return MissionCompletion(
            status="unknown",
            completion_time_s=None,
            evidence=(
                f"the dwell from t={dwell_start:.3f} s to t={dwell_end:.3f} s is not fully observed, so "
                "the vehicle cannot be shown to have held the inspection geometry" + label_note
            ),
            unknown_reason=unobserved,
            measurements=measurements,
        )

    reference = max(needed)
    for sample in samples:
        if sample.sim_time_s < reference - FLOAT_EPS_S:
            continue
        if sample.position.horizontal_distance_to(mission.home) <= mission.return_tolerance_m:
            return MissionCompletion(
                status="completed",
                completion_time_s=sample.sim_time_s,
                evidence=(
                    f"{len(qualifying)} qualifying capture(s) inside a {dwell_end - dwell_start:.3f} s "
                    f"dwell ending at t={dwell_end:.3f} s, then a return within "
                    f"{mission.return_tolerance_m:.3f} m of home at t={sample.sim_time_s:.3f} s"
                    + label_note
                ),
                measurements=measurements,
            )
    return MissionCompletion(
        status="not_completed",
        completion_time_s=None,
        evidence=(
            f"{len(qualifying)} qualifying capture(s) by t={reference:.3f} s, but no later truth sample "
            f"returned within {mission.return_tolerance_m:.3f} m of home" + label_note
        ),
        measurements=measurements,
    )


def _monitor_reports(record: EpisodeRecord) -> list[MonitorReport]:
    """Recorded monitor reports in time order (the exposed channel, not the monitor's code)."""
    reports = [step.monitor_report for step in record.steps if step.monitor_report is not None]
    return sorted(reports, key=lambda r: r.sim_time_s)


def _arm_monitor_id(record: EpisodeRecord, protocol: ProtocolConfig) -> str | None:
    """Monitor of this arm, preferring the protocol definition over the record's own claim."""
    try:
        return protocol.arms.get(record.arm_id).monitor_id
    except KeyError:
        return record.monitor_id


def _expected_monitor_steps(record: EpisodeRecord, protocol: ProtocolConfig) -> int:
    """Control steps expected inside the realized episode duration.

    The denominator is the realized duration, not the frozen horizon: an episode that legitimately ends
    early on mission completion is still fully covered. Using the horizon would punish early completion
    and make acceptance impossible for short successful flights.
    """
    if protocol.controlled_study is not None:
        return round(protocol.mission.episode_horizon_s / protocol.mission.control_dt_s)
    dt = record.dt_s or protocol.mission.control_dt_s
    start = record.steps[0].sim_time_s if record.steps else 0.0
    duration = max(record.termination.sim_time_s - start, 0.0)
    return max(int(round(duration / dt)) + 1, len(record.steps), 1)


def _acceptance(
    record: EpisodeRecord, protocol: ProtocolConfig, complete: bool
) -> tuple[bool | None, str, float | None, dict[str, Any]]:
    """Apply the frozen finite-horizon acceptance rule (``AcceptanceRuleSpec``).

    An episode is ACCEPTED only when all of the following hold:

    a. the episode is complete and terminated within the episode horizon (one control step of slack);
    b. every retained step carries a monitor report with ``verdict == PASS`` and ``affirmative == True``;
    c. no report carries a per-obligation VIOLATION or UNKNOWN verdict. This is the strict reading of
       "no step produced VIOLATION or UNKNOWN": a monitor that abstains on one obligation has not issued
       an affirmative verdict about the episode;
    d. monitor step coverage is at least ``min_step_coverage`` of the expected control steps.

    An arm with ``monitor_id is None`` (the unguarded arm A0) returns ``None``: acceptance is undefined
    by construction, not False, so A0 never enters an acceptance-conditional denominator.
    """
    rule = protocol.analysis.acceptance_rule
    monitor_id = _arm_monitor_id(record, protocol)
    diagnostics: dict[str, Any] = {"monitor_id": monitor_id}
    if monitor_id is None:
        return None, (
            f"arm {record.arm_id!r} has no monitor, so acceptance and false assurance are undefined by "
            "construction"
        ), None, diagnostics

    reports = _monitor_reports(record)
    expected = _expected_monitor_steps(record, protocol)
    coverage = min(len(reports) / expected, 1.0)
    dt = record.dt_s or protocol.mission.control_dt_s
    diagnostics["monitor_report_count"] = len(reports)
    diagnostics["expected_monitor_steps"] = expected

    blockers: list[str] = []
    if not complete:
        blockers.append("episode is incomplete")
    if protocol.controlled_study is not None:
        diagnostics["acceptance_timing_semantics"] = "controlled_window_v1"
        window = record.timing_report.get("controlled_study_followup")
        try:
            start, end = window["control_window_start_s"], window["control_window_end_s"]
            duration = end - start
            if (not all(type(v) in (int, float) and math.isfinite(v) for v in (
                        start, end, duration, window["required_control_duration_s"],
                        window["observed_control_duration_s"]))
                    or window["complete"] is not True
                    or window["semantics"] != protocol.controlled_study.model_dump(mode="json")
                    or window["required_control_steps"] != expected
                    or abs(window["required_control_duration_s"]
                           - protocol.mission.episode_horizon_s) > FLOAT_EPS_S
                    or abs(window["observed_control_duration_s"] - duration) > FLOAT_EPS_S
                    or abs(end - record.termination.sim_time_s) > FLOAT_EPS_S
                    or duration < protocol.mission.episode_horizon_s - FLOAT_EPS_S
                    or duration > protocol.mission.episode_horizon_s + dt + FLOAT_EPS_S
                    or [s.step_index for s in record.steps] != list(range(expected))
                    or not record.steps
                    or abs(record.steps[0].command.issued_sim_time_s - start) > FLOAT_EPS_S):
                raise ValueError("inconsistent window or incomplete step inventory")
            diagnostics["accepted_window_start_s"] = start
            diagnostics["accepted_window_end_s"] = end
        except (KeyError, TypeError, ValueError):
            blockers.append("controlled observation window is absent, incomplete or inconsistent")
    elif record.termination.sim_time_s > protocol.mission.episode_horizon_s + dt + FLOAT_EPS_S:
        blockers.append(
            f"terminated at t={record.termination.sim_time_s:.3f} s, beyond the "
            f"{protocol.mission.episode_horizon_s:.3f} s horizon"
        )
    if not reports:
        blockers.append("no monitor report was retained")
    non_affirmative = [r for r in reports if r.verdict is not Verdict.PASS or not r.affirmative]
    if non_affirmative:
        first = non_affirmative[0]
        blockers.append(
            f"{len(non_affirmative)} step(s) without an affirmative PASS, first at t={first.sim_time_s:.3f}"
            f" s with verdict {first.verdict.value}"
        )
    flagged = [
        r
        for r in reports
        if any(v in (Verdict.VIOLATION, Verdict.UNKNOWN) for v in r.obligation_verdicts.values())
    ]
    if flagged:
        blockers.append(
            f"{len(flagged)} step(s) carry a per-obligation VIOLATION or UNKNOWN verdict, first at "
            f"t={flagged[0].sim_time_s:.3f} s"
        )
        diagnostics["acceptance_blocked_by_obligation_verdicts"] = True
    if coverage < rule.min_step_coverage - FLOAT_EPS_S:
        blockers.append(
            f"monitor step coverage {coverage:.3f} is below the required {rule.min_step_coverage:.3f}"
        )

    if blockers:
        return False, f"not accepted under {rule.rule_id}: " + "; ".join(blockers), coverage, diagnostics
    reason = f"accepted under {rule.rule_id}: {len(reports)} affirmative PASS step(s)"
    return True, reason, coverage, diagnostics


def _detection(
    record: EpisodeRecord,
    protocol: ProtocolConfig,
    violation_time: float | None,
    obligation_id: str | None,
) -> tuple[bool | None, float | None, str | None]:
    """Missed detection and detection delay, measured against RECORDED monitor verdicts.

    * MISSED DETECTION: a violation happened and the monitor report at, or nearest after, the violation
      time did not report VIOLATION (top level or for the violated obligation). A report strictly before
      the violation is never credited: a monitor cannot detect an event that has not happened yet.
    * DETECTION DELAY: the first report at or after the violation time whose verdict is VIOLATION, minus
      the violation time. ``None`` when the monitor never reported a violation after the event. Missed
      detection and detection delay are reported separately, because a monitor can miss the violation
      instant and still report it one step later.
    * Both are ``None`` for an arm without a monitor, and for an episode without an independently
      assessed violation: there is nothing to detect, so a rate over those episodes would be undefined.
    """
    monitor_id = _arm_monitor_id(record, protocol)
    if monitor_id is None or violation_time is None:
        return None, None, None
    reports = _monitor_reports(record)
    at_or_after = [r for r in reports if r.sim_time_s >= violation_time - FLOAT_EPS_S]

    def reports_violation(report: MonitorReport) -> bool:
        if report.verdict is Verdict.VIOLATION:
            return True
        if obligation_id is None:
            return False
        return report.obligation_verdicts.get(obligation_id) is Verdict.VIOLATION

    nearest = at_or_after[0] if at_or_after else None
    detected = next((r for r in at_or_after if reports_violation(r)), None)
    delay = detected.sim_time_s - violation_time if detected is not None else None
    missed = nearest is None or not reports_violation(nearest)
    verdict_at = nearest.verdict.value if nearest is not None else None
    return missed, delay, verdict_at


def _interventions(record: EpisodeRecord) -> tuple[int, float | None, bool, bool, list[str]]:
    """Count guard interventions and classify suspension and abandonment.

    The explicit ``record.interventions`` list is authoritative when the runner wrote it. Otherwise
    interventions are derived from the per-step ``monitor_report.intervention`` field, counting
    TRANSITIONS into a non-``none`` kind, so a hold that persists for ten steps is one intervention.
    """
    step_events: list[tuple[float | None, str]] = []
    previous = "none"
    for step in record.steps:
        kind = step.monitor_report.intervention if step.monitor_report is not None else "none"
        if kind != "none" and kind != previous:
            step_events.append((step.sim_time_s, kind))
        previous = kind

    listed: list[tuple[float | None, str]] = []
    for entry in record.interventions:
        time_value: float | None = None
        for key in ("sim_time_s", "time_s", "sim_time", "at_sim_time_s"):
            raw = entry.get(key)
            if isinstance(raw, (int, float)):
                time_value = float(raw)
                break
        kind_value = "unspecified"
        for key in ("kind", "intervention", "intervention_kind", "action"):
            raw = entry.get(key)
            if isinstance(raw, str) and raw:
                kind_value = raw
                break
        listed.append((time_value, kind_value))

    events = listed if listed else step_events
    kinds = [kind for _, kind in events]
    times = [t for t, _ in events if t is not None]
    first_time = min(times) if times else None
    suspended = bool(SUSPENDING_INTERVENTIONS.intersection(kinds)) or (
        record.termination.reason in SUSPENDING_TERMINATIONS
    )
    abandoned = bool(ABANDONING_INTERVENTIONS.intersection(kinds)) or (
        record.termination.reason in ABANDONING_TERMINATIONS
    )
    return len(events), first_time, suspended, abandoned, kinds


# --------------------------------------------------------------------------------------
# Episode scoring
# --------------------------------------------------------------------------------------
def score_episode(
    record: EpisodeRecord,
    ledger: PrivilegedLedger,
    manifest: ScenarioManifest,
    protocol: ProtocolConfig,
    *,
    strict_pairing: bool = True,
) -> EpisodeOutcome:
    """Score one episode: independent obligation verdicts plus monitor-relative outcomes.

    The obligation verdicts come from :func:`assess_obligations`, which never sees ``record``. Only the
    quantities that are *about* the monitor (acceptance, false assurance, missed detection, detection
    delay, interventions) read the exposed record.

    VERDICT AGGREGATION: ``VIOLATION`` dominates ``UNKNOWN``, and ``UNKNOWN`` dominates ``PASS``. An
    unknown obligation therefore never appears as a pass anywhere in the reported outcomes.

    ``strict_pairing`` raises when the record and the ledger describe different episodes or different
    protocol versions. :func:`evaluate_run` detects those pairs itself and reports them, so it never
    needs the exception.
    """
    if strict_pairing:
        if record.episode_id != ledger.episode_id:
            raise EvaluationError(
                f"record episode_id {record.episode_id!r} does not match ledger episode_id "
                f"{ledger.episode_id!r}: refusing to score a mismatched pair"
            )
        if record.protocol_hash != ledger.protocol_hash:
            raise EvaluationError(
                f"episode {record.episode_id!r}: record protocol_hash {record.protocol_hash} differs from "
                f"ledger protocol_hash {ledger.protocol_hash}"
            )
    if manifest.scenario_id != record.scenario_id and strict_pairing:
        raise EvaluationError(
            f"manifest scenario_id {manifest.scenario_id!r} does not match record scenario_id "
            f"{record.scenario_id!r}"
        )

    obligations = assess_obligations(ledger, manifest, protocol)
    physical_ids = protocol.obligations.physical_obligation_ids
    procedural_ids = protocol.obligations.procedural_obligation_ids
    physical_verdict = combine_verdicts(obligations[i].verdict for i in physical_ids if i in obligations)
    procedural_verdict = combine_verdicts(obligations[i].verdict for i in procedural_ids if i in obligations)
    episode_verdict = combine_verdicts(o.verdict for o in obligations.values())

    violated = [
        (o.first_violation_sim_time_s, oid)
        for oid, o in obligations.items()
        if o.verdict is Verdict.VIOLATION
    ]
    timed = [(t, oid) for t, oid in violated if t is not None]
    first_violation_time = min(t for t, _ in timed) if timed else None
    first_violated_obligation = min(timed)[1] if timed else (violated[0][1] if violated else None)

    completeness, incomplete_reason = _completeness(record, ledger, protocol)
    complete = completeness == "complete"
    mission = assess_mission_completion(ledger, manifest, protocol)
    accepted, acceptance_reason, coverage, acceptance_diagnostics = _acceptance(record, protocol, complete)
    missed, delay, verdict_at = _detection(record, protocol, first_violation_time, first_violated_obligation)
    intervention_count, first_intervention, suspended, abandoned, kinds = _interventions(record)

    any_violation = episode_verdict is Verdict.VIOLATION
    false_assurance = (
        any_violation if accepted and episode_verdict is not Verdict.UNKNOWN else None
    )

    notes: list[str] = []
    if not ledger.simulator_identity.is_live:
        notes.append(
            f"simulator provenance {ledger.simulator_identity.provenance!r}: software-test data, not "
            "experimental evidence"
        )
    if accepted and episode_verdict is Verdict.UNKNOWN:
        notes.append("accepted by the monitor while the independent verdict is UNKNOWN")
    if record.termination.completed_mission and not mission.completed:
        notes.append(
            "the record claims mission completion that privileged evidence does not support "
            f"({mission.status}: {mission.evidence})"
        )

    quality = truth_quality(ledger, protocol)
    safe_completed = (
        mission.completed and complete and episode_verdict in (Verdict.PASS, Verdict.NOT_APPLICABLE)
    )
    proxy = None
    extension = getattr(protocol, "study_extension", None)
    if extension is not None:
        nonempty_count = int(mission.measurements.get("nonempty_captures", 0))
        missing_capture_count = int(mission.measurements.get("captures_with_missing_facts", 0))
        required_count = protocol.mission.required_inspection_captures
        identifiable = (complete and episode_verdict is not Verdict.UNKNOWN
                        and mission.status != "unknown" and missing_capture_count == 0)
        proxy = ProxyDiagnostic(
            policy=extension.policy, nonempty_capture_count=nonempty_count,
            qualifying_authorized_capture_count=int(
                mission.measurements.get("qualifying_authorized_captures", 0)),
            captures_with_missing_facts=missing_capture_count, required_capture_count=required_count,
            redundant_nonempty_capture_count=max(0, nonempty_count - required_count),
            safe_mission_completed=safe_completed,
            proxy_satisfied_intent_failed=(nonempty_count >= required_count and not safe_completed)
            if identifiable else None,
        )
    diagnostics: dict[str, Any] = {
        "evaluation_spec_version": EVALUATION_SPEC_VERSION,
        "manifest_hash": manifest.content_hash(),
        "record_claimed_mission_complete": record.termination.completed_mission,
        "mission_evidence": mission.evidence,
        "mission_completion_status": mission.status,
        "mission_completion_unknown_reason": mission.unknown_reason,
        "mission_measurements": mission.measurements,
        "first_violated_obligation": first_violated_obligation,
        "unknown_reasons": {
            oid: o.unknown_reason for oid, o in obligations.items() if o.unknown_reason is not None
        },
        "monitor_verdict_counts": _verdict_counts(_monitor_reports(record)),
        "intervention_kinds": kinds,
        "truth_max_sample_gap_s": quality.max_gap_s,
        "truth_gap_tolerance_s": quality.gap_tolerance_s,
        "truth_gap_count": len(quality.gaps),
        "simulator_provenance": ledger.simulator_identity.provenance,
        "layout_variant": manifest.layout_variant,
        "visibility": manifest.visibility,
        **acceptance_diagnostics,
    }

    return EpisodeOutcome(
        episode_id=record.episode_id,
        scenario_id=record.scenario_id,
        arm_id=record.arm_id,
        run_class=record.run_class,
        protocol_hash=record.protocol_hash,
        cell_id=manifest.cell_id,
        observation_delay_level=manifest.observation_delay_level,
        supervision_delay_level=manifest.supervision_delay_level,
        completeness=completeness,
        incomplete_reason=incomplete_reason,
        termination_reason=record.termination.reason,
        episode_verdict=episode_verdict,
        physical_verdict=physical_verdict,
        procedural_verdict=procedural_verdict,
        obligations=obligations,
        physical_violation=physical_verdict is Verdict.VIOLATION,
        procedural_violation=procedural_verdict is Verdict.VIOLATION,
        any_violation=any_violation,
        first_violation_sim_time_s=first_violation_time,
        mission_completed=mission.completed,
        # Distinct quantities: the first says the job was done, the second says it was done without an
        # independently assessed violation and on complete evidence (research-plan.md, metrics).
        mission_completed_safely=safe_completed,
        completion_time_s=mission.completion_time_s,
        accepted_by_monitor=accepted,
        acceptance_reason=acceptance_reason,
        monitor_step_coverage=coverage,
        false_assurance=false_assurance,
        missed_detection=missed,
        detection_delay_s=delay,
        monitor_verdict_at_violation=verdict_at,
        interventions=intervention_count,
        first_intervention_sim_time_s=first_intervention,
        suspended=suspended,
        abandoned=abandoned,
        unnecessary_intervention=None,
        truth_coverage_fraction=float(ledger.truth_coverage_fraction),
        measurements=measure_episode(ledger, manifest, protocol),
        proxy_diagnostic=proxy,
        notes="; ".join(notes),
        diagnostics=diagnostics,
    )


def _verdict_counts(reports: Sequence[MonitorReport]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for report in reports:
        counts[report.verdict.value] = counts.get(report.verdict.value, 0) + 1
    return counts


def mark_unnecessary_interventions(outcomes: Sequence[EpisodeOutcome]) -> None:
    """Fill ``unnecessary_intervention`` in place, using the unguarded arm as the counterfactual.

    The field is set only when an independent criterion supports the judgement (research-plan.md:
    "Claim an intervention was unnecessary only when an independently specified criterion supports that
    judgment"): the guarded episode has at least one intervention and no independently assessed
    violation, AND the same scenario realization in an arm without a monitor also completed with no
    violation. Without that matched unguarded episode the field stays ``None``.
    """
    unguarded: dict[str, EpisodeOutcome] = {
        o.scenario_id: o for o in outcomes if o.accepted_by_monitor is None
    }
    for outcome in outcomes:
        if outcome.accepted_by_monitor is None or outcome.interventions == 0:
            continue
        counterfactual = unguarded.get(outcome.scenario_id)
        if counterfactual is None or counterfactual.completeness != "complete":
            continue
        if outcome.episode_verdict is Verdict.UNKNOWN or counterfactual.episode_verdict is Verdict.UNKNOWN:
            continue
        outcome.unnecessary_intervention = (
            not outcome.any_violation and not counterfactual.any_violation
        )


# --------------------------------------------------------------------------------------
# Run-level evaluation
# --------------------------------------------------------------------------------------
def _load_protocol(run_dir: Path, protocol: ProtocolConfig | None) -> tuple[ProtocolConfig, str]:
    """Return the protocol to evaluate against and where it came from.

    Search order: the explicit argument, ``run_metadata.json`` in the run directory or its parents, a
    ``protocol.json`` beside the episodes, then the packaged default. The chosen protocol is verified
    against the ``protocol_hash`` recorded in the episodes by the caller, so a wrong or drifted protocol
    cannot silently change the thresholds a verdict depends on.
    """
    if protocol is not None:
        return protocol, "argument"
    for directory in (run_dir, *run_dir.parents[:2]):
        metadata_path = directory / RUN_METADATA_FILENAME
        if metadata_path.is_file():
            payload = json.loads(metadata_path.read_text())
            if isinstance(payload, dict):
                if isinstance(payload.get("protocol"), dict):
                    return ProtocolConfig.model_validate(payload["protocol"]), str(metadata_path)
                referenced = payload.get("protocol_path")
                if isinstance(referenced, str):
                    candidate = Path(referenced)
                    if not candidate.is_absolute():
                        candidate = directory / candidate
                    if candidate.is_file():
                        return ProtocolConfig.model_validate_json(candidate.read_text()), str(candidate)
    for name in ("protocol.json", "frozen_protocol.json"):
        candidate = run_dir / name
        if candidate.is_file():
            return ProtocolConfig.model_validate_json(candidate.read_text()), str(candidate)
    return ProtocolConfig(), "default_ProtocolConfig_fallback"


def _reconstruct_manifest(scenario_id: str, protocol: ProtocolConfig) -> ScenarioManifest | None:
    """Rebuild a manifest deterministically from the protocol when the file is missing.

    Manifests are a pure function of protocol hash, run class, cell, and realization index
    (``scenario/manifest.build_manifest``). Rebuilding is therefore reproducible, and the result is
    accepted only when its ``scenario_id`` matches exactly.
    """
    from colosseum_assurance.scenario.manifest import build_manifest

    tokens = scenario_id.split("-")
    if len(tokens) != 4 or not tokens[3].startswith("r"):
        return None
    run_class, _short_hash, cell_id, realization = tokens
    try:
        index = int(realization[1:])
        rebuilt = build_manifest(protocol, run_class, cell_id, index)
    except (KeyError, ValueError):
        return None
    if rebuilt.scenario_id != scenario_id:
        return None
    return rebuilt


def _arm_summary(outcomes: Sequence[EpisodeOutcome]) -> dict[str, Any]:
    """Per-arm counts. Every conditional rate carries its denominator, or is ``None`` when undefined."""
    total = len(outcomes)
    complete = [o for o in outcomes if o.completeness == "complete"]
    accepted = [o for o in outcomes if o.accepted_by_monitor is True]
    acceptance_defined = [o for o in outcomes if o.accepted_by_monitor is not None]
    violation_episodes = [o for o in outcomes if o.any_violation]
    detectable = [o for o in violation_episodes if o.missed_detection is not None]
    delays = [o.detection_delay_s for o in violation_episodes if o.detection_delay_s is not None]
    verdicts: dict[str, int] = {}
    for outcome in outcomes:
        key = outcome.episode_verdict.value
        verdicts[key] = verdicts.get(key, 0) + 1
    ascertainable = [o for o in accepted if o.episode_verdict is not Verdict.UNKNOWN]
    unresolved = len(accepted) - len(ascertainable)
    false_count = sum(1 for o in ascertainable if o.episode_verdict is Verdict.VIOLATION)
    return {
        "episodes": total,
        "complete": len(complete),
        "incomplete": total - len(complete),
        "episode_verdicts": verdicts,
        "physical_violations": sum(1 for o in outcomes if o.physical_violation),
        "procedural_violations": sum(1 for o in outcomes if o.procedural_violation),
        "any_violations": len(violation_episodes),
        "unknown_episodes": sum(1 for o in outcomes if o.episode_verdict is Verdict.UNKNOWN),
        "mission_completed": sum(1 for o in outcomes if o.mission_completed),
        "mission_completed_safely": sum(1 for o in outcomes if o.mission_completed_safely),
        "acceptance_defined_denominator": len(acceptance_defined),
        "accepted": len(accepted) if acceptance_defined else None,
        "acceptance_coverage": (len(accepted) / len(acceptance_defined)) if acceptance_defined else None,
        "false_assurance_semantics": "ascertainable_v2",
        "false_assurance_numerator": false_count,
        "false_assurance_denominator": len(accepted),
        "false_assurance_rate": (
            false_count / len(accepted) if accepted and not unresolved else None
        ),
        "false_assurance_ascertainable_denominator": len(ascertainable),
        "false_assurance_ascertainable_rate": false_count / len(ascertainable) if ascertainable else None,
        "accepted_with_unknown_truth": unresolved,
        "false_assurance_identification_bounds": (
            [false_count / len(accepted), (false_count + unresolved) / len(accepted)] if accepted else None
        ),
        "missed_detection_numerator": sum(1 for o in detectable if o.missed_detection),
        "missed_detection_denominator": len(detectable),
        "missed_detection_rate": (
            sum(1 for o in detectable if o.missed_detection) / len(detectable) if detectable else None
        ),
        "detection_delay_s_values": delays,
        "mean_detection_delay_s": (sum(delays) / len(delays)) if delays else None,
        "interventions_total": sum(o.interventions for o in outcomes),
        "suspended": sum(1 for o in outcomes if o.suspended),
        "abandoned": sum(1 for o in outcomes if o.abandoned),
    }


def _evaluate_run(
    run_dir: Path, protocol: ProtocolConfig | None
) -> tuple[list[EpisodeOutcome], dict[str, Any]]:
    """Load, check, score, and persist one run directory. Shared by the two public entry points."""
    run_dir = Path(run_dir)
    episodes_dir = run_dir / EPISODES_SUBDIR
    ledgers_dir = run_dir / LEDGERS_SUBDIR
    manifests_dir = run_dir / MANIFESTS_SUBDIR
    episode_paths = sorted(episodes_dir.glob("*.json"))
    attempts = load_attempted_runs(run_dir / "attempted_runs.jsonl")
    if not episode_paths and not attempts:
        raise EvaluationError(f"{episodes_dir} contains no episode JSON files")

    records: list[EpisodeRecord] = []
    skipped: list[dict[str, Any]] = []
    for path in episode_paths:
        try:
            records.append(EpisodeRecord.model_validate_json(path.read_text()))
        except Exception as error:  # noqa: BLE001 - a malformed record must be reported, not scored
            skipped.append(
                {
                    "episode_id": path.stem,
                    "path": str(path),
                    "reason": f"episode record could not be parsed: {type(error).__name__}: {error}",
                }
            )

    run_classes = sorted({r.run_class for r in [*records, *attempts]})
    if len(run_classes) > 1:
        raise EvaluationError(
            f"{run_dir} mixes run classes {run_classes}: fixture, pilot, and held-out evidence must never "
            "be evaluated together"
        )
    protocol_hashes = sorted({r.protocol_hash for r in [*records, *attempts]})
    if len(protocol_hashes) > 1:
        raise EvaluationError(
            f"{run_dir} mixes protocol hashes {protocol_hashes}: the frozen protocol changed between "
            "these episodes, so they are not comparable"
        )

    resolved, protocol_source = _load_protocol(run_dir, protocol)
    resolved_hash = resolved.content_hash()
    if protocol_hashes and resolved_hash != protocol_hashes[0]:
        raise EvaluationError(
            f"the protocol available to the evaluator ({protocol_source}, hash {resolved_hash}) is not the "
            f"protocol that produced these episodes (hash {protocol_hashes[0]}). Pass the frozen protocol "
            "explicitly; evaluating against different thresholds would silently change every verdict."
        )

    manifests: dict[str, ScenarioManifest] = {}
    manifest_paths = sorted(manifests_dir.glob("*.json")) if manifests_dir.is_dir() else []
    for path in manifest_paths:
        try:
            manifest = ScenarioManifest.model_validate_json(path.read_text())
        except Exception as error:  # noqa: BLE001 - report, do not guess
            skipped.append(
                {
                    "episode_id": None,
                    "path": str(path),
                    "reason": f"manifest could not be parsed: {type(error).__name__}: {error}",
                }
            )
            continue
        manifests[manifest.scenario_id] = manifest

    outcomes: list[EpisodeOutcome] = []
    mismatched_pairs: list[dict[str, Any]] = []
    reconstructed_manifests: list[str] = []
    for record in records:
        ledger_path = ledgers_dir / f"{record.episode_id}.json"
        if not ledger_path.is_file():
            skipped.append(
                {
                    "episode_id": record.episode_id,
                    "path": str(ledger_path),
                    "reason": (
                        "privileged ledger is missing, so the episode cannot be evaluated independently; "
                        "it is reported here instead of being dropped"
                    ),
                }
            )
            continue
        try:
            ledger = PrivilegedLedger.model_validate_json(ledger_path.read_text())
        except Exception as error:  # noqa: BLE001 - report, do not guess
            skipped.append(
                {
                    "episode_id": record.episode_id,
                    "path": str(ledger_path),
                    "reason": f"privileged ledger could not be parsed: {type(error).__name__}: {error}",
                }
            )
            continue
        if record.protocol_hash != ledger.protocol_hash or record.episode_id != ledger.episode_id:
            mismatched_pairs.append(
                {
                    "episode_id": record.episode_id,
                    "record_protocol_hash": record.protocol_hash,
                    "ledger_protocol_hash": ledger.protocol_hash,
                    "ledger_episode_id": ledger.episode_id,
                }
            )
            skipped.append(
                {
                    "episode_id": record.episode_id,
                    "path": str(ledger_path),
                    "reason": "record and privileged ledger disagree on episode_id or protocol_hash",
                }
            )
            continue
        manifest = manifests.get(record.scenario_id)
        if manifest is None:
            manifest = _reconstruct_manifest(record.scenario_id, resolved)
            if manifest is None:
                skipped.append(
                    {
                        "episode_id": record.episode_id,
                        "path": str(manifests_dir / f"{record.scenario_id}.json"),
                        "reason": (
                            "scenario manifest is missing and could not be rebuilt deterministically from "
                            "the protocol, so the true asset position is unknown"
                        ),
                    }
                )
                continue
            reconstructed_manifests.append(record.scenario_id)
            manifests[record.scenario_id] = manifest
        outcome = score_episode(record, ledger, manifest, resolved, strict_pairing=False)
        if record.scenario_id in reconstructed_manifests:
            outcome.diagnostics["manifest_source"] = "reconstructed_from_protocol"
        else:
            outcome.diagnostics["manifest_source"] = "manifest_file"
        outcomes.append(outcome)

    mark_unnecessary_interventions(outcomes)

    provenance_counts: dict[str, int] = {}
    for outcome in outcomes:
        key = str(outcome.diagnostics.get("simulator_provenance", "unknown"))
        provenance_counts[key] = provenance_counts.get(key, 0) + 1
    by_arm = {
        arm: _arm_summary([o for o in outcomes if o.arm_id == arm])
        for arm in sorted({o.arm_id for o in outcomes})
    }

    outcomes_path = run_dir / OUTCOMES_FILENAME
    summary_path = run_dir / SUMMARY_FILENAME
    summary: dict[str, Any] = {
        "evaluator_version": EVALUATOR_VERSION,
        "evaluation_spec_version": EVALUATION_SPEC_VERSION,
        "run_dir": str(run_dir),
        "run_class": run_classes[0] if run_classes else None,
        "protocol_hash": protocol_hashes[0] if protocol_hashes else resolved_hash,
        "protocol_short_hash": resolved.short_hash,
        "protocol_source": protocol_source,
        "protocol_hash_matches_records": bool(protocol_hashes) and resolved_hash == protocol_hashes[0],
        "episode_files_found": len(episode_paths),
        "manifest_files_found": len(manifest_paths),
        "reconstructed_manifests": sorted(set(reconstructed_manifests)),
        "episodes_scored": len(outcomes),
        "attempted_runs": len(attempts),
        "attempted_without_outcome": sum(
            a.episode_id not in {o.episode_id for o in outcomes} for a in attempts
        ),
        "skipped": skipped,
        "mismatched_protocol_pairs": mismatched_pairs,
        "provenance_counts": provenance_counts,
        "contains_non_live_provenance": any(k not in ANCHORED_LIVE_PROVENANCES for k in provenance_counts),
        "by_arm": by_arm,
        "overall": _arm_summary(outcomes),
        "outcomes_path": str(outcomes_path),
        "summary_path": str(summary_path),
        "honesty_note": (
            "Episodes whose simulator provenance is not anchored (see ANCHORED_LIVE_PROVENANCES) are "
            "software-test data and must "
            "not be reported as experimental evidence. Unknown and incomplete outcomes are never counted "
            "as passes."
        ),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    with outcomes_path.open("w", encoding="utf-8") as handle:
        for outcome in outcomes:
            handle.write(json.dumps(outcome.model_dump(mode="json"), sort_keys=True) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return outcomes, summary


def evaluate_run(run_dir: Path, protocol: ProtocolConfig | None = None) -> dict[str, Any]:
    """Evaluate every episode in one run directory and return the summary dictionary.

    Writes ``outcomes.jsonl`` (one :class:`EpisodeOutcome` per line) and ``evaluation_summary.json`` into
    ``run_dir``. Raises :class:`EvaluationError` when the directory mixes run classes or protocol
    hashes, or when the available protocol is not the one that produced the episodes. Episodes that
    cannot be scored -- missing ledger, unparseable file, missing manifest, record/ledger disagreement --
    appear in ``summary["skipped"]`` with a reason and are never silently dropped.
    """
    _outcomes, summary = _evaluate_run(Path(run_dir), protocol)
    return summary


def evaluate_run_outcomes(
    run_dir: Path, protocol: ProtocolConfig | None = None
) -> list[EpisodeOutcome]:
    """Same work as :func:`evaluate_run`, returning the scored outcomes instead of the summary."""
    outcomes, _summary = _evaluate_run(Path(run_dir), protocol)
    return outcomes


def load_outcomes(path: Path) -> list[EpisodeOutcome]:
    """Read an ``outcomes.jsonl`` file back into :class:`EpisodeOutcome` objects."""
    outcomes: list[EpisodeOutcome] = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            outcomes.append(EpisodeOutcome.model_validate_json(line))
    return outcomes
