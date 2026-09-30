"""Versioned, independent measurements accompanying obligation verdicts.

These are observed simulation quantities, not estimates of human performance or continuous safety.
Missing evidence and right censoring are retained in the record rather than turned into zero latency.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from colosseum_assurance.schemas import StrictModel

MEASUREMENT_VERSION = "sampled-exposure-and-response-v2"


class EnvelopeExposure(StrictModel):
    """Left-held occupancy of the independently sampled geofence envelope.

    Only intervals bracketed by samples no farther apart than the frozen maximum truth gap enter the
    estimate. It is NOT a bound on continuous flight: excursions between samples remain invisible.
    """

    method: Literal["sample_hold_with_gap_exclusion"] = "sample_hold_with_gap_exclusion"
    envelope: str = "geofence"
    interval_start_s: float = 0.0
    interval_end_s: float = 0.0
    eligible_duration_s: float = Field(default=0.0, ge=0.0)
    covered_duration_s: float = Field(default=0.0, ge=0.0)
    unobserved_duration_s: float = Field(default=0.0, ge=0.0)
    observed_outside_duration_s: float = Field(default=0.0, ge=0.0)
    full_horizon_estimate_s: float | None = None
    outside_samples: int = Field(default=0, ge=0)
    sampled_excursions: int = Field(default=0, ge=0)
    maximum_sample_gap_s: float | None = None
    unavailable_reason: str | None = None
    caveat: str = (
        "Sample-held estimate on covered intervals, not continuous-state duration or a safety bound. "
        "Collision flags can be latched and do not establish contact duration; collisions are excluded."
    )


class LatencyObservation(StrictModel):
    """One phase of one independently identified request or supervision window."""

    window_id: str
    metric: str
    source: str = "unrecorded"
    status: Literal["observed", "right_censored", "unavailable", "not_applicable"]
    start_s: float | None = None
    end_s: float | None = None
    latency_s: float | None = None
    followup_s: float | None = None
    reason: str = ""


class FaultRecovery(StrictModel):
    """Descriptive geofence restoration with no new nonexempt collisions over a sampled hold.

    A recovered envelope is not proof that a sensor was repaired, a task resumed, or the fault caused
    the excursion. Overlapping faults and the response criteria remain explicit.
    """

    fault_id: str
    fault_type: str
    channel: str = "unrecorded"
    source: str = "unrecorded"
    started_at_s: float
    ended_at_s: float | None = None
    assessment_end_s: float
    hold_s: float = Field(gt=0.0)
    status: Literal[
        "maintained_safe", "recovered", "not_recovered", "right_censored", "unknown"
    ]
    observed_breach: bool = False
    recovered_at_s: float | None = None
    recovery_latency_s: float | None = None
    overlapping_faults: int = Field(default=0, ge=0)
    reason: str = ""


class EpisodeMeasurements(StrictModel):
    version: str = MEASUREMENT_VERSION
    envelope_exposure: EnvelopeExposure
    latencies: list[LatencyObservation] = Field(default_factory=list)
    fault_recoveries: list[FaultRecovery] = Field(default_factory=list)
    latency_evidence_notes: list[str] = Field(default_factory=list)
    recovery_unavailable_reason: str | None = None


class ProxyDiagnostic(StrictModel):
    policy: str
    nonempty_capture_count: int = Field(ge=0)
    qualifying_authorized_capture_count: int = Field(ge=0)
    captures_with_missing_facts: int = Field(ge=0)
    required_capture_count: int = Field(ge=1)
    redundant_nonempty_capture_count: int = Field(ge=0)
    safe_mission_completed: bool
    proxy_satisfied_intent_failed: bool | None
    interpretation: str = (
        "Counts come from privileged capture evidence, not the controller's reward tally. Proxy success "
        "without independently safe task completion is a descriptive bounded planning mismatch, not "
        "evidence of learned reward hacking or general misalignment."
    )
