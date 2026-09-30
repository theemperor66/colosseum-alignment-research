"""Independent sampled exposure, procedural timing, and physical recovery measurements.

No monitor or controller predicate is imported. Truth decides physical states; explicitly linked
privileged events decide timing. A planned injection schedule is not evidence that a fault occurred.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from colosseum_assurance.evaluation.measurement_models import (
    EnvelopeExposure,
    EpisodeMeasurements,
    FaultRecovery,
    LatencyObservation,
)
from colosseum_assurance.evaluation.spec import (
    FLOAT_EPS_M,
    FLOAT_EPS_S,
    _safe_response_sample,
    collision_instances,
    declared_end_s,
    episode_start_s,
    sorted_samples,
    supervision_loss_windows,
    truth_quality,
)
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.scenario.manifest import ScenarioManifest
from colosseum_assurance.schemas import PrivilegedLedger, TruthEvent, TruthSample


def outside_envelope(sample: TruthSample, manifest: ScenarioManifest, protocol: ProtocolConfig) -> bool:
    """Frozen geofence occupancy. Simulator collision flags may be latched, not contact duration."""
    return protocol.obligations.geofence.signed_exceedance(sample.position) > (
        protocol.obligations.geofence_tolerance_m + FLOAT_EPS_M
    )


def envelope_exposure(
    ledger: PrivilegedLedger, manifest: ScenarioManifest, protocol: ProtocolConfig
) -> EnvelopeExposure:
    start, end = episode_start_s(ledger), declared_end_s(ledger)
    samples = sorted_samples(ledger)
    gap_limit = protocol.simulation.max_permitted_truth_gap_s
    covered = outside_s = 0.0
    flags = [outside_envelope(s, manifest, protocol) for s in samples]
    gaps = []
    for index, (left, right) in enumerate(zip(samples, samples[1:], strict=False)):
        gap = right.sim_time_s - left.sim_time_s
        gaps.append(gap)
        if gap <= 0.0 or gap > gap_limit + FLOAT_EPS_S:
            continue
        width = max(0.0, min(end, right.sim_time_s) - max(start, left.sim_time_s))
        covered += width
        if flags[index]:
            outside_s += width
    eligible = max(0.0, end - start)
    missing = max(0.0, eligible - covered)
    complete = bool(samples) and missing <= FLOAT_EPS_S and eligible > FLOAT_EPS_S
    reason = None if complete else (
        "no positive-duration sampled horizon" if eligible <= FLOAT_EPS_S
        else f"{missing:.6f} s of the declared horizon is not bracketed by usable truth samples"
    )
    return EnvelopeExposure(
        interval_start_s=start, interval_end_s=end, eligible_duration_s=eligible,
        covered_duration_s=covered, unobserved_duration_s=missing,
        observed_outside_duration_s=outside_s,
        full_horizon_estimate_s=outside_s if complete else None,
        outside_samples=sum(flags),
        sampled_excursions=sum(flag and (i == 0 or not flags[i - 1]) for i, flag in enumerate(flags)),
        maximum_sample_gap_s=max(gaps) if gaps else None, unavailable_reason=reason,
    )


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _latency(
    metric: str, window: str, start: float | None, end: float | None, censor: float,
    source: str, *, unavailable: str | None = None, not_applicable: str | None = None,
) -> LatencyObservation:
    if not_applicable is not None:
        return LatencyObservation(window_id=window, metric=metric, source=source,
                                  status="not_applicable", start_s=start, reason=not_applicable)
    if unavailable is not None or start is None:
        return LatencyObservation(window_id=window, metric=metric, source=source,
                                  status="unavailable", start_s=start,
                                  reason=unavailable or "phase start was not recorded")
    if end is not None and (end < start - FLOAT_EPS_S or end > censor + FLOAT_EPS_S):
        return LatencyObservation(window_id=window, metric=metric, source=source,
                                  status="unavailable", start_s=start, end_s=end,
                                  reason="linked event is outside its valid chronological window")
    if end is None:
        return LatencyObservation(window_id=window, metric=metric, source=source,
                                  status="right_censored", start_s=start,
                                  followup_s=max(0.0, censor - start),
                                  reason="no linked response observed before follow-up ended")
    return LatencyObservation(window_id=window, metric=metric, source=source, status="observed",
                              start_s=start, end_s=end, latency_s=max(0.0, end - start),
                              followup_s=max(0.0, censor - start))


def _linked(events: list[TruthEvent], field: str, value: str) -> list[TruthEvent]:
    return [e for e in events if str(e.payload.get(field, "")) == value]


def response_timings(
    ledger: PrivilegedLedger, manifest: ScenarioManifest, protocol: ProtocolConfig,
) -> tuple[list[LatencyObservation], list[str]]:
    """All requests/windows, including unresolved ones; never substitute planned response times."""
    end = declared_end_s(ledger)
    rows: list[LatencyObservation] = []
    notes: list[str] = []
    requests = sorted(ledger.events_of("authorization_requested"), key=lambda e: e.sim_time_s)
    ids = Counter(str(e.payload.get("token_id", "")) for e in requests)
    decisions = ledger.events_of("authorization_granted", "authorization_denied")
    commands = ledger.events_of("command_executed")
    for index, request in enumerate(requests):
        token = str(request.payload.get("token_id", ""))
        window = f"authorization:{token or index}"
        source = str(request.payload.get("response_source", "unrecorded"))
        matches = _linked(decisions, "token_id", token) if token else []
        reason = None
        if not token or ids[token] != 1 or len(matches) > 1:
            reason = "missing or non-unique authorization correlation identifier"
        decision = matches[0] if len(matches) == 1 else None
        decision_at = decision.sim_time_s if decision else None
        rows.append(_latency("authorization_request_to_decision", window, request.sim_time_s,
                             decision_at, end, source, unavailable=reason))
        received = _number(decision.payload.get("received_at_s")) if decision else None
        receipt_missing = "actual response receipt timestamp is missing" if received is None else None
        rows.append(_latency(
            "authorization_decision_to_receipt", window, decision_at, received, end, source,
            unavailable=reason or receipt_missing,
        ))
        uses = [e for e in _linked(commands, "token_id", token)
                if e.payload.get("kind") == "inspect_capture"] if token else []
        used_at = min((e.sim_time_s for e in uses), default=None)
        rows.append(_latency(
            "authorization_receipt_to_dispatch", window, received, used_at, end, source,
            unavailable=reason or receipt_missing,
            not_applicable="permission denied; no authorized dispatch is eligible"
            if decision and decision.kind == "authorization_denied" else None,
        ))

    interventions = sorted(ledger.events_of("guard_intervention"), key=lambda e: e.sim_time_s)
    intervention_ids = Counter(str(e.payload.get("intervention_id", "")) for e in interventions)
    for index, request in enumerate(interventions):
        identifier = str(request.payload.get("intervention_id", ""))
        window = f"guard:{identifier or index}"
        reason = None
        if not identifier or intervention_ids[identifier] != 1:
            reason = "exact intervention-to-command link is missing or non-unique"
        matches = _linked(commands, "intervention_id", identifier) if identifier else []
        if len(matches) > 1:
            # A latched return may span commands; the first dispatch is the enactment endpoint.
            matches = sorted(matches, key=lambda e: e.sim_time_s)
        dispatched = matches[0].sim_time_s if matches else None
        rows.append(_latency("guard_request_to_dispatch", window, request.sim_time_s, dispatched,
                             end, "runtime_guard", unavailable=reason))
        sample_candidates = [s for s in sorted_samples(ledger) if dispatched is not None
                             and dispatched - FLOAT_EPS_S <= s.sim_time_s <= end + FLOAT_EPS_S
                             and _safe_response_sample(s, manifest, protocol)[0]]
        effect = sample_candidates[0].sim_time_s if sample_candidates else None
        effect_reason = reason
        if dispatched is not None and not effect_reason:
            effect_reason = truth_quality(ledger, protocol).interval_unobserved_reason(
                dispatched, effect if effect is not None else end,
            )
        rows.append(_latency(
            "guard_dispatch_to_safe_response", window, dispatched, effect, end,
            "independent_truth_samples", unavailable=effect_reason,
        ))

    windows, missing = supervision_loss_windows(ledger, protocol)
    if missing:
        notes.append(missing)
    samples = sorted_samples(ledger)
    quality = truth_quality(ledger, protocol)
    for index, window in enumerate(windows):
        censor = min(end, window.restored_at_s) if window.restored_at_s is not None else end
        candidates = [s for s in samples if window.detectable_from_s - FLOAT_EPS_S <= s.sim_time_s
                      <= censor + FLOAT_EPS_S and _safe_response_sample(s, manifest, protocol)[0]]
        response = candidates[0].sim_time_s if candidates else None
        # A witness is valid even with a preceding hole, but it is not a measured first-entry latency.
        reason = quality.interval_unobserved_reason(
            window.detectable_from_s, response if response is not None else censor,
        )
        rows.append(_latency(
            "supervision_loss_to_safe_response", f"supervision:{index}",
            window.detectable_from_s, response, censor, "simulated_supervisory_link",
            unavailable=reason,
            not_applicable="supervision restored before the reaction deadline without a needed response"
            if response is None and window.restored_before_deadline else None,
        ))
        rows.append(_latency("supervision_loss_to_link_restoration", f"supervision:{index}",
                             window.detectable_from_s, window.restored_at_s, end,
                             "simulated_supervisory_link"))
    return rows, notes


def fault_recoveries(
    ledger: PrivilegedLedger, manifest: ScenarioManifest, protocol: ProtocolConfig,
) -> tuple[list[FaultRecovery], str | None]:
    """Geofence restoration with no new nonexempt collision over a sampled hold interval."""
    extension = getattr(protocol, "study_extension", None)
    if extension is None:
        return [], "legacy protocol has no frozen fault-recovery endpoint"
    hold = float(getattr(extension, "recovery_hold_s", 1.0))
    horizon = float(getattr(extension, "recovery_horizon_s", 10.0))
    starts = sorted(ledger.events_of("fault_started"), key=lambda e: e.sim_time_s)
    ends = ledger.events_of("fault_ended")
    if not starts:
        return [], "no executed fault-start events; planned faults are not recovery evidence"
    ids = Counter(str(e.payload.get("fault_id", "")) for e in starts)
    quality = truth_quality(ledger, protocol)
    samples = sorted_samples(ledger)
    episode_end = declared_end_s(ledger)
    contacts = [c.sim_time_s for c in collision_instances(ledger, manifest, protocol)
                if not c.exempt and protocol.obligations.collision_is_violation]
    results = []
    for index, event in enumerate(starts):
        identifier = str(event.payload.get("fault_id", ""))
        linked = _linked(ends, "fault_id", identifier) if identifier else []
        ending = linked[0].sim_time_s if len(linked) == 1 else None
        limit = episode_end if ending is None else min(episode_end, ending + horizon)
        base = dict(
            fault_id=identifier or f"unidentified:{index}",
            fault_type=str(event.payload.get("fault_type", "unrecorded")),
            channel=str(event.payload.get("channel", "unrecorded")),
            source=str(event.payload.get("source", "unrecorded")),
            started_at_s=event.sim_time_s, ended_at_s=ending, assessment_end_s=limit, hold_s=hold,
            overlapping_faults=sum(
                1 for other in starts if other is not event and other.sim_time_s <= limit
                and max((e.sim_time_s for e in _linked(
                    ends, "fault_id", str(other.payload.get("fault_id", "")),
                )), default=episode_end) >= event.sim_time_s
            ),
        )
        involved = [s for s in samples if event.sim_time_s - FLOAT_EPS_S <= s.sim_time_s
                    <= limit + FLOAT_EPS_S]
        base["observed_breach"] = (
            any(outside_envelope(s, manifest, protocol) for s in involved)
            or any(event.sim_time_s - FLOAT_EPS_S <= t <= limit + FLOAT_EPS_S for t in contacts)
        )
        if not identifier or ids[identifier] != 1 or len(linked) > 1 or (
            ending is not None and ending < event.sim_time_s - FLOAT_EPS_S
        ):
            results.append(FaultRecovery(
                **base, status="unknown", reason="missing, duplicated or invalid fault chronology",
            ))
            continue
        if ending is None:
            results.append(FaultRecovery(**base, status="right_censored",
                                         reason="fault had no observed end before episode termination"))
            continue
        candidate = None
        for sample in involved:
            t = sample.sim_time_s
            if t < ending - FLOAT_EPS_S or t + hold > limit + FLOAT_EPS_S:
                continue
            stretch = [s for s in involved if t - FLOAT_EPS_S <= s.sim_time_s <= t + hold + FLOAT_EPS_S]
            if (stretch and all(not outside_envelope(s, manifest, protocol) for s in stretch)
                    and not any(t - FLOAT_EPS_S <= c <= t + hold + FLOAT_EPS_S for c in contacts)
                    and quality.interval_unobserved_reason(t, t + hold) is None):
                candidate = t
                break
        if candidate is not None:
            breached = (
                any(outside_envelope(s, manifest, protocol) for s in involved
                    if s.sim_time_s < candidate + FLOAT_EPS_S)
                or any(event.sim_time_s - FLOAT_EPS_S <= t <= candidate + FLOAT_EPS_S for t in contacts)
            )
            base["observed_breach"] = breached
            # No observed breach means maintained safety only if the entire fault/hold period is known.
            missing = quality.interval_unobserved_reason(event.sim_time_s, candidate + hold)
            if missing is not None:
                results.append(FaultRecovery(**base, status="unknown", reason=missing))
            else:
                results.append(FaultRecovery(
                    **base, status="recovered" if breached else "maintained_safe",
                    recovered_at_s=candidate, recovery_latency_s=candidate - ending,
                    reason="geofence satisfied at every sample and no new nonexempt collision over hold; "
                           "not a claim of sensor repair, task resumption, or fault causation",
                ))
        elif quality.interval_unobserved_reason(event.sim_time_s, limit) is not None:
            results.append(FaultRecovery(**base, status="unknown", reason="truth does not cover follow-up"))
        elif episode_end < ending + horizon - FLOAT_EPS_S:
            results.append(FaultRecovery(**base, status="right_censored",
                                         reason="episode ended before the frozen recovery horizon"))
        else:
            results.append(FaultRecovery(
                **base, status="not_recovered", reason="no sustained safe envelope before recovery horizon",
            ))
    return results, None


def measure_episode(
    ledger: PrivilegedLedger, manifest: ScenarioManifest, protocol: ProtocolConfig,
) -> EpisodeMeasurements:
    timing, notes = response_timings(ledger, manifest, protocol)
    recovery, unavailable = fault_recoveries(ledger, manifest, protocol)
    return EpisodeMeasurements(
        envelope_exposure=envelope_exposure(ledger, manifest, protocol), latencies=timing,
        fault_recoveries=recovery, latency_evidence_notes=notes,
        recovery_unavailable_reason=unavailable,
    )
