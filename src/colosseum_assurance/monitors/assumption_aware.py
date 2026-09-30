"""``assumption_aware_v1``: the literature-grounded comparator guard.

INSPIRED IMPLEMENTATION, NOT A REPRODUCTION
-------------------------------------------
This guard is *inspired by* three published lines of work. It is not a faithful reimplementation of any
of them, and no result here should be read as a reproduction of their results.

* **Assumption monitoring for UAV temporal task plans** (Zudaire, Gorostiaga, Sanchez, Schneider and
  Uchitel, "Assumption Monitoring Using Runtime Verification for UAV Temporal Task Plan Executions",
  ICRA 2021, DOI 10.1109/ICRA48506.2021.9561671; title, authors and mechanism recorded in
  docs/literature-check.md): the idea that an executing UAV must monitor the assumptions its plan
  relies on, not only the plan's own predicates, because an unmonitored assumption violation can
  permit a failure the user cannot distinguish from success. Departures: their Sections II-III keep
  two things apart that this project does not use at all - discrete controller synthesis from FLTL
  goals with MTSA, and HLOLA stream monitors compiled from *engineer-supplied* assumptions. Ours are
  five hand-written Python checks over one fixed inspection mission: no controller synthesis, no
  temporal logic, no stream specification language, and no execution-time or memory guarantee. The
  environment / sensing / capability grouping used below is a useful organising interpretation of
  their motivation, not a validated three-part taxonomy taken from the paper.
* **ModelPlex-style model-validity monitoring** (Mitsch and Platzer, "Verified Runtime Model Validation
  for Partially Observable Hybrid Systems", arXiv:1811.06502v2, 24 February 2019): the idea that a
  safety argument transfers to the running system only while the model it was proved against remains
  valid, so model validity must be checked at runtime. Departures: we have no theorem-proved hybrid
  model and no synthesised monitor formula. Our "model validity" is a bounded reachability envelope
  from the last confirmed position plus a declared delay bound; it is a plausibility heuristic, not a
  proof-carrying monitor. Their guarantees require assumptions and proofs this implementation does not
  establish: their Section 5 assumes non-faulty sensors inside known uncertainty bounds, and their
  Theorems 4-5 require contraction conditions. General sensor failure and an arbitrarily stale record -
  both of which this guard must answer for - lie outside those cited guarantees, so nothing here
  inherits them.
* **Symbolic runtime verification under uncertainties and assumptions** (Kallwies, Leucker and Sanchez,
  ATVA 2022, LNCS 13505, pp. 117-134, DOI 10.1007/978-3-031-19992-9_8): the rule that a monitor should
  return UNKNOWN when the available evidence and the explicit assumptions do not determine the
  obligation, while uncertain inputs may still support a definite verdict. Departures: we do not run a
  symbolic three-valued evaluation over a specification language; we compute explicit bounds per
  obligation and return UNKNOWN when those bounds straddle a threshold. Treating a failed *assumption*
  as a reason to withhold an affirmative verdict is this implementation's own policy, not an account of
  their symbolic inference.

Checks reported in ``assumption_verdicts``
------------------------------------------
``a_evidence_age``            state (and depth) evidence within the declared delay bound plus margin.
``b_sensing_validity``        a usable depth frame, sufficient coverage, bounded dropout burst, and a
                              still-conclusive extrapolation when the current frame is missing.
``c_capability_consistency``  consecutive received states consistent with the bounded speed, and a
                              reachability envelope small enough to be informative.
``d_authorization_validity``  whether the held record decides the authorization *at the time of use*: a
                              granted record is treated as valid when it carries no expiry at all, or
                              when its remaining validity is AT LEAST the margin, which is the larger of
                              the clock margin and the record's own age; it is undecidable when the
                              remaining validity is strictly inside that margin, or the request is still
                              pending; it is invalid when the expiry is AT OR BEFORE now, or the status
                              is denied, expired or absent. Record age alone is
                              therefore not a verdict, and revocation is not observable onboard at all.
``e_supervision_liveness``    the link state derived from the newest *received* heartbeat, re-timed
                              against the vehicle clock. It is undecidable when the supervision view
                              itself lags by more than one heartbeat period, or when the age the view
                              states disagrees with its own timestamps by more than one control step,
                              or when no heartbeat receipt time can be derived at all; otherwise the
                              received gap decides live or lost, with no ambiguity band around the loss
                              threshold. The declared supervisory response delay is *not* part of this
                              verdict; when it exceeds the reaction deadline it only adds
                              ``unknown_grace_steps`` to the unknown streak, so escalation happens
                              earlier (see ``_supervisory_rescue_feasible``).

An assumption verdict of VIOLATION means the assumption is demonstrably false; UNKNOWN means it cannot
be decided. Neither is an obligation violation, and both prevent an affirmative verdict.

Why it is not an always-abstain monitor
---------------------------------------
Abstention would make assurance coverage collapse, and the analysis reports coverage next to every
conditional rate, so a guard that abstains on everything loses by construction. Two rules keep it
informative: a conclusion is affirmed when the *bounds* still support it (a clearance of 25 m stays a
clearance even if the vehicle could have moved 2 m since the frame), and an obligation that cannot be
engaged at all is NOT_APPLICABLE rather than UNKNOWN. In a clean nominal step this guard returns an
affirmative PASS, which ``tests/unit/test_monitor_assumption_aware.py`` pins.

What a comparison against ``policy_only_v1`` can mean
----------------------------------------------------
Both guards read the same packet, share the frozen policy thresholds and share one intervention
vocabulary. That makes the two arms *comparable*; it does not isolate the causal effect of any single
check. A1 and A2 are two frozen monitor **configurations** whose differences include uncertainty
propagation, authorization-boundary handling, stale-view handling and unknown-triggered escalation. A
paired A1-A2 difference is therefore an effect of the whole configuration. A mechanism-specific claim
would need component ablations, which this study does not run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from colosseum_assurance.interfaces import MissionBrief
from colosseum_assurance.monitors.base import (
    ASSET_POSITION_UNCERTAINTY_M,
    COLLISION_PROXIMITY_M,
    MonitorBase,
    geofence_exceedance_m,
    horizontal_distance,
    reachable_radius_m,
    received_heartbeat_time_s,
    segment_min_distance,
    speed_of,
)
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.schemas import (
    AuthorizationStatus,
    ControlCommand,
    DepthSummary,
    InterventionKind,
    MonitorReport,
    ObservationPacket,
    VehicleState,
    Verdict,
)

__all__ = ["AssumptionAwareMonitor", "AssumptionThresholds", "CHECK_IDS"]

CHECK_IDS: tuple[str, ...] = (
    "a_evidence_age",
    "b_sensing_validity",
    "c_capability_consistency",
    "d_authorization_validity",
    "e_supervision_liveness",
)

_INTERVENTION_RANK: dict[str, int] = {
    "none": 0,
    "hold": 1,
    "suspend_inspection": 2,
    "return_to_launch": 3,
    "abort": 4,
}


@dataclass(frozen=True, slots=True)
class AssumptionThresholds:
    """Frozen thresholds of the assumption checks. Changing one changes the comparator."""

    evidence_age_margin_s: float | None = None
    """Slack on the declared observation delay. ``None`` means half a control period, which is enough
    for delivery quantisation and not enough to absorb a dropped packet: a dropout is exactly the event
    that breaks the declared bound, and this guard exists to notice that."""

    max_evidence_extrapolation_s: float = 2.0
    min_depth_coverage_fraction: float = 0.25
    max_dropout_burst: int = 3
    motion_speed_tolerance: float = 1.5
    motion_slack_m: float = 0.5
    clock_margin_s: float = 0.25
    max_useful_envelope_m: float = 10.0
    """Above this the reachability envelope is wider than the features being tested, so position
    evidence stops supporting any conclusion."""

    unknown_grace_steps: int = 2
    unknown_hold_steps: int = 6
    unknown_rtl_steps: int = 20


class AssumptionAwareMonitor(MonitorBase):
    """Policy predicates plus the assumptions required to interpret the evidence behind them."""

    monitor_id = "assumption_aware_v1"
    monitor_class = "assumption_aware_inspired_implementation"

    def __init__(self, protocol: ProtocolConfig, thresholds: AssumptionThresholds | None = None) -> None:
        super().__init__(protocol)
        self.thresholds = thresholds or AssumptionThresholds()
        self._last_state: VehicleState | None = None
        self._previous_state: VehicleState | None = None
        self._last_depth: DepthSummary | None = None
        self._unknown_streak = 0
        self._safe_response_entered = False

    def reset(self, brief: MissionBrief) -> None:
        super().reset(brief)
        self._last_state = None
        self._previous_state = None
        self._last_depth = None
        self._unknown_streak = 0
        self._safe_response_entered = False

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "emits_unknown": True,
                "grounding": "inspired_implementation",
                "grounding_sources": [
                    "Zudaire et al. 2021, assumption monitoring for UAV temporal task plan execution",
                    "Mitsch and Platzer, verified runtime validation for partially observable hybrid systems",
                    "symbolic runtime verification under uncertainties and assumptions",
                ],
                "assumption_checks": list(CHECK_IDS),
                "thresholds": {
                    "evidence_age_margin_s": self._age_margin_s(),
                    "max_evidence_extrapolation_s": self.thresholds.max_evidence_extrapolation_s,
                    "min_depth_coverage_fraction": self.thresholds.min_depth_coverage_fraction,
                    "max_dropout_burst": self.thresholds.max_dropout_burst,
                    "clock_margin_s": self.thresholds.clock_margin_s,
                    "max_useful_envelope_m": self.thresholds.max_useful_envelope_m,
                    "asset_position_uncertainty_m": ASSET_POSITION_UNCERTAINTY_M,
                },
                "escalation": (
                    "violation -> obligation specific (geofence and supervision go home, authorization "
                    "suspends the inspection, proximity holds); unknown -> none for "
                    f"{self.thresholds.unknown_grace_steps} steps, then hold, then suspend_inspection "
                    f"after {self.thresholds.unknown_hold_steps}, then return_to_launch after "
                    f"{self.thresholds.unknown_rtl_steps}; an undecidable authorization while the "
                    "inspection radius may already be engaged suspends immediately; one level earlier "
                    "when the declared supervisory response delay exceeds the reaction deadline"
                ),
                "limitations": [
                    "inspired implementation, not a reproduction of any published monitor",
                    "uses the nominal asset position plus a declared survey uncertainty, not truth",
                    "has no contact truth; the collision obligation is a depth proximity proxy",
                    "assumption checks are hand written, not synthesised from a specification",
                ],
            }
        )
        return info

    # ------------------------------------------------------------------ helpers
    def _age_margin_s(self) -> float:
        margin = self.thresholds.evidence_age_margin_s
        return 0.5 * self.mission.control_dt_s if margin is None else margin

    def _effective_state_age_s(
        self, observation: ObservationPacket
    ) -> tuple[VehicleState | None, float | None]:
        """Best available position sample and how old it is on the vehicle clock."""
        now = observation.receive_sim_time_s
        if observation.state is not None:
            age = observation.state_age_s
            if age is None:
                age = now - observation.state.sim_time_s
            return observation.state, max(age, 0.0)
        if self._last_state is not None:
            return self._last_state, max(now - self._last_state.sim_time_s, 0.0)
        return None, None

    def _envelope_m(self, age_s: float | None) -> float | None:
        """Radius the vehicle could have left the sampled position by, before this command takes effect."""
        if age_s is None:
            return None
        horizon = age_s + self.mission.control_dt_s + self.thresholds.clock_margin_s
        return reachable_radius_m(self.mission.cruise_speed_mps, horizon)

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, observation: ObservationPacket, command: ControlCommand) -> MonitorReport:
        brief = self._require_brief()
        self.steps_evaluated += 1
        now = observation.receive_sim_time_s
        notes: list[str] = []

        state, age = self._effective_state_age_s(observation)
        envelope = self._envelope_m(age)
        grounded, mode_reason = self.ground_mode.evaluate(observation, command)
        self._stationary_disarmed_confirmed = grounded
        if self.protocol.controlled_study is not None:
            notes.append(mode_reason)
        if grounded:
            ground = self.protocol.controlled_study
            assert ground is not None and age is not None
            envelope = (ground.ground_position_slack_m + ground.ground_max_speed_mps
                        * (age+self.mission.control_dt_s+self.thresholds.clock_margin_s))
        assumptions: dict[str, Verdict] = {}
        assumptions["a_evidence_age"] = (Verdict.PASS if grounded
                                          else self._check_evidence_age(observation, age, notes))
        assumptions["b_sensing_validity"] = (Verdict.NOT_APPLICABLE if grounded
                                              else self._check_sensing(observation, envelope, notes))
        assumptions["c_capability_consistency"] = self._check_capability(observation, envelope, notes)

        obligations: dict[str, Verdict] = {}
        obligations["geofence"] = self._geofence_verdict(state, envelope, notes)
        obligations["collision"] = (Verdict.NOT_APPLICABLE if grounded
                                      else self._collision_verdict(observation, envelope, notes))
        if grounded:
            # No motion is commanded in a positively evidenced disarmed rest mode. This removes
            # the airborne depth-proximity prediction only; independent collision truth is unchanged.
            assumptions["b_sensing_validity"] = Verdict.NOT_APPLICABLE
            obligations["collision"] = Verdict.NOT_APPLICABLE
        engaged, definitely_engaged = self._radius_engagement(state, envelope, command, brief)
        auth_state, auth_note = self._classify_authorization(observation, now)
        assumptions["d_authorization_validity"] = (
            Verdict.NOT_APPLICABLE
            if not engaged
            else (Verdict.UNKNOWN if auth_state == "ambiguous" else Verdict.PASS)
        )
        obligations["inspection_authorization"] = self._authorization_verdict(
            engaged, definitely_engaged, auth_state, auth_note, notes
        )
        supervision_state, effective_age = self._supervision_state(observation, now)
        assumptions["e_supervision_liveness"] = {
            "live": Verdict.PASS,
            "lost": Verdict.PASS,
            "ambiguous": Verdict.UNKNOWN,
            "unreadable": Verdict.UNKNOWN,
        }[supervision_state]
        obligations["loss_of_supervision_response"] = self._supervision_verdict(
            supervision_state, observation, effective_age, state, envelope, now, notes
        )

        if observation.state is not None:
            self._previous_state = self._last_state
            self._last_state = observation.state
        if observation.depth is not None and observation.depth.valid:
            self._last_depth = observation.depth

        verdict, affirmative = self._aggregate(obligations, assumptions)
        if verdict is Verdict.PASS:
            self._unknown_streak = 0
        else:
            self._unknown_streak += 1
        intervention = self._intervention(
            obligations, assumptions, supervision_state, engaged, verdict
        )
        if not notes:
            notes.append("all obligations decidable and satisfied on evidence within the declared bounds")
        if envelope is not None:
            notes.append(f"reachability envelope {envelope:.2f} m")

        return MonitorReport(
            step_index=observation.step_index,
            sim_time_s=now,
            monitor_id=self.monitor_id,
            verdict=verdict,
            obligation_verdicts=obligations,
            assumption_verdicts=assumptions,
            intervention=intervention,
            rationale="; ".join(notes)[:900],
            evidence_age_s=age,
            affirmative=affirmative,
        )

    # ------------------------------------------------------------------ assumption checks
    def _check_evidence_age(
        self, observation: ObservationPacket, age: float | None, notes: list[str]
    ) -> Verdict:
        """(a) Is the evidence within the delay bound that was declared for this episode?"""
        if age is None:
            notes.append("no position sample has ever been received; evidence age undecidable")
            return Verdict.UNKNOWN
        bound = self._declared_observation_delay_s(observation) + self._age_margin_s()
        if age > bound:
            notes.append(
                f"evidence age {age:.2f} s exceeds the declared bound {bound:.2f} s "
                "(declared delay plus margin)"
            )
            return Verdict.VIOLATION
        depth_age = observation.sensor_health.depth_age_s
        if depth_age is not None and depth_age > bound:
            notes.append(f"depth evidence age {depth_age:.2f} s exceeds the declared bound {bound:.2f} s")
            return Verdict.VIOLATION
        return Verdict.PASS

    def _check_sensing(
        self, observation: ObservationPacket, envelope: float | None, notes: list[str]
    ) -> Verdict:
        """(b) Is there sensing evidence good enough to bound obstacle proximity over this step?"""
        t = self.thresholds
        health = observation.sensor_health
        if health.dropouts_in_window > t.max_dropout_burst:
            notes.append(f"sensor dropout burst {health.dropouts_in_window} exceeds {t.max_dropout_burst}")
            return Verdict.VIOLATION
        depth = observation.depth
        if depth is not None and depth.valid:
            if depth.coverage_fraction < t.min_depth_coverage_fraction:
                notes.append(
                    f"depth coverage {depth.coverage_fraction:.2f} below the usable floor "
                    f"{t.min_depth_coverage_fraction:.2f}"
                )
                return Verdict.VIOLATION
            return Verdict.PASS
        reason = depth.degraded_reason if depth is not None else "no depth summary in packet"
        last = self._last_depth
        if last is None:
            notes.append(f"no usable depth evidence has been received ({reason})")
            return Verdict.VIOLATION
        gap = observation.receive_sim_time_s - last.sim_time_s
        if gap > t.max_evidence_extrapolation_s:
            notes.append(f"last usable depth frame is {gap:.2f} s old ({reason})")
            return Verdict.VIOLATION
        bound = self._extrapolated_clearance_m(last, envelope)
        if bound is None or bound <= COLLISION_PROXIMITY_M:
            notes.append(
                f"depth missing this step ({reason}) and the extrapolated clearance is not conclusive"
            )
            return Verdict.UNKNOWN
        return Verdict.PASS

    def _check_capability(
        self, observation: ObservationPacket, envelope: float | None, notes: list[str]
    ) -> Verdict:
        """(c) Is the received motion consistent with the vehicle's bounds, and is the envelope useful?"""
        t = self.thresholds
        if envelope is None:
            notes.append("no position evidence; capability envelope undecidable")
            return Verdict.UNKNOWN
        current = observation.state
        previous = self._last_state
        if current is not None and previous is not None:
            span = current.sim_time_s - previous.sim_time_s
            if span > 1e-6:
                moved = current.position.distance_to(previous.position)
                allowed = self.mission.cruise_speed_mps * t.motion_speed_tolerance * span + t.motion_slack_m
                if moved > allowed:
                    notes.append(
                        f"received motion {moved:.2f} m in {span:.2f} s exceeds the capability bound "
                        f"{allowed:.2f} m"
                    )
                    return Verdict.VIOLATION
        if envelope > t.max_useful_envelope_m:
            notes.append(
                f"reachability envelope {envelope:.2f} m exceeds the useful limit "
                f"{t.max_useful_envelope_m:.2f} m; position evidence supports no conclusion"
            )
            return Verdict.UNKNOWN
        return Verdict.PASS

    # ------------------------------------------------------------------ obligation verdicts
    def _geofence_verdict(
        self, state: VehicleState | None, envelope: float | None, notes: list[str]
    ) -> Verdict:
        if state is None or envelope is None:
            notes.append("geofence undecidable: no position evidence")
            return Verdict.UNKNOWN
        exceedance = geofence_exceedance_m(
            self.obligations.geofence, state.position, self.obligations.geofence_tolerance_m
        )
        if exceedance > 0.0:
            notes.append(f"sampled position is {exceedance:.2f} m outside the geofence")
            return Verdict.VIOLATION
        if exceedance + envelope > 0.0:
            notes.append(
                f"geofence undecidable: the sample sits {-exceedance:.2f} m inside, within the "
                f"{envelope:.2f} m the vehicle could have travelled since"
            )
            return Verdict.UNKNOWN
        return Verdict.PASS

    def _extrapolated_clearance_m(self, depth: DepthSummary, envelope: float | None) -> float | None:
        """Lower bound on current clearance from an earlier frame, given how far the vehicle could move."""
        if depth.min_range_m is None:
            return None if not depth.valid else float("inf")
        if envelope is None:
            return None
        return depth.min_range_m - envelope

    def _collision_verdict(
        self, observation: ObservationPacket, envelope: float | None, notes: list[str]
    ) -> Verdict:
        depth = observation.depth
        if depth is not None and depth.valid:
            if depth.min_range_m is None:
                return Verdict.PASS
            if depth.min_range_m < COLLISION_PROXIMITY_M:
                notes.append(f"depth proximity proxy {depth.min_range_m:.2f} m below the contact margin")
                return Verdict.VIOLATION
            return Verdict.PASS
        last = self._last_depth
        if last is None:
            notes.append("contact proximity undecidable: no usable depth evidence")
            return Verdict.UNKNOWN
        gap = observation.receive_sim_time_s - last.sim_time_s
        bound = self._extrapolated_clearance_m(last, envelope)
        if gap > self.thresholds.max_evidence_extrapolation_s or bound is None:
            notes.append(f"contact proximity undecidable: last usable frame {gap:.2f} s old")
            return Verdict.UNKNOWN
        if bound <= COLLISION_PROXIMITY_M:
            notes.append(
                f"contact proximity undecidable: extrapolated clearance {bound:.2f} m from a "
                f"{gap:.2f} s old frame"
            )
            return Verdict.UNKNOWN
        return Verdict.PASS

    def _radius_engagement(
        self,
        state: VehicleState | None,
        envelope: float | None,
        command: ControlCommand,
        brief: MissionBrief,
    ) -> tuple[bool, bool]:
        """Could the inspection radius be engaged, and is it certainly engaged?

        The uncertainty of the nominal asset position is added to the reachability envelope, because a
        guard that only knows the surveyed position cannot place the boundary more precisely than that.
        """
        if state is None or envelope is None:
            return True, False
        radius = self.obligations.authorized_inspection_radius_m
        slack = envelope + ASSET_POSITION_UNCERTAINTY_M
        asset = brief.nominal_asset_position
        distance = horizontal_distance(state.position, asset)
        commanded = distance
        if command.target is not None:
            commanded = segment_min_distance(asset, state.position, command.target)
        nearest = min(distance, commanded)
        could = (nearest - slack) <= radius
        certain = (max(distance, commanded) + slack) <= radius
        return could, certain

    def _classify_authorization(
        self, observation: ObservationPacket, now: float
    ) -> tuple[str, str]:
        """Decide the authorization at the *time of use*: valid, invalid, or undecidable."""
        auth = observation.authorization
        margin = self.thresholds.clock_margin_s
        if auth.received_at_s is not None:
            margin = max(margin, now - auth.received_at_s)
        if auth.status is AuthorizationStatus.GRANTED:
            if auth.expires_at_s is None:
                return "valid", "authorization granted without an expiry"
            remaining = auth.expires_at_s - now
            if remaining <= 0.0:
                return "invalid", f"authorization expired {-remaining:.2f} s ago"
            if remaining < margin:
                return (
                    "ambiguous",
                    f"authorization expires in {remaining:.2f} s, inside the {margin:.2f} s clock and "
                    "record-age margin",
                )
            return "valid", f"authorization valid for another {remaining:.2f} s"
        if auth.status is AuthorizationStatus.PENDING:
            return "ambiguous", "authorization request is still pending"
        return "invalid", f"authorization status is {auth.status.value}"

    def _authorization_verdict(
        self,
        engaged: bool,
        definitely_engaged: bool,
        auth_state: str,
        auth_note: str,
        notes: list[str],
    ) -> Verdict:
        if not engaged:
            return Verdict.NOT_APPLICABLE
        if auth_state == "valid":
            return Verdict.PASS
        if auth_state == "invalid" and definitely_engaged:
            notes.append(f"inspection radius certainly engaged and {auth_note}")
            return Verdict.VIOLATION
        notes.append(
            f"inspection authorization undecidable: {auth_note}"
            + ("" if definitely_engaged else "; radius engagement itself is within the position envelope")
        )
        return Verdict.UNKNOWN

    def _supervision_state(self, observation: ObservationPacket, now: float) -> tuple[str, float | None]:
        """Re-time the newest received heartbeat against the vehicle clock and classify the link.

        The obligation is defined on the received gap, which is exactly decidable once the receipt time
        is known, so this guard does not invent an ambiguity band around the threshold. What it does
        check is whether its own supervision view is current: if the view lags by more than one
        heartbeat period, a heartbeat could have arrived without appearing in it, and the link state is
        then undecidable rather than lost.
        """
        supervision = observation.supervision
        received = received_heartbeat_time_s(supervision)
        if received is None:
            return "unreadable", None
        age = max(0.0, now - received)
        view_lag = now - supervision.sim_time_s
        if view_lag > self.obligations.supervision_heartbeat_period_s:
            return "ambiguous", age
        stated = supervision.heartbeat_age_s
        if stated is not None:
            disagreement = abs((supervision.sim_time_s - received) - stated)
            if disagreement > self.mission.control_dt_s:
                return "ambiguous", age
        if age > self.obligations.loss_of_supervision_threshold_s:
            return "lost", age
        return "live", age

    def _supervision_verdict(
        self,
        supervision_state: str,
        observation: ObservationPacket,
        age: float | None,
        state: VehicleState | None,
        envelope: float | None,
        now: float,
        notes: list[str],
    ) -> Verdict:
        """Loss-of-supervision response, with every ingredient checked before it is believed."""
        obligations = self.obligations
        if supervision_state == "live":
            self._safe_response_entered = False
            return Verdict.NOT_APPLICABLE
        if supervision_state in {"ambiguous", "unreadable"} or age is None:
            notes.append(
                "loss-of-supervision response undecidable: "
                + ("no heartbeat receipt time" if age is None else "the supervision view is not current")
            )
            return Verdict.UNKNOWN
        if state is None or envelope is None:
            notes.append("supervision lost but the response cannot be confirmed: no position evidence")
            return Verdict.UNKNOWN

        brief = self._require_brief()
        slack = envelope + ASSET_POSITION_UNCERTAINTY_M
        radius = obligations.authorized_inspection_radius_m
        distance = horizontal_distance(state.position, brief.nominal_asset_position)
        certainly_outside = (distance - slack) > radius
        certainly_inside = (distance + slack) <= radius
        slow = speed_of(state) <= obligations.safe_response_speed_mps
        if obligations.safe_response_requires_leaving_inspection_zone:
            in_response = slow and certainly_outside
            confirmable = certainly_outside or certainly_inside
        else:
            in_response = slow
            confirmable = True

        received = received_heartbeat_time_s(observation.supervision)
        assert received is not None
        detectable_at = received + obligations.loss_of_supervision_threshold_s
        deadline = (
            detectable_at
            + obligations.loss_of_supervision_reaction_deadline_s
            + obligations.response_entry_grace_s
        )
        if in_response:
            self._safe_response_entered = True
            notes.append(f"supervision lost {age:.1f} s ago; safe response confirmed")
            return Verdict.PASS
        if self._safe_response_entered and obligations.safe_response_must_be_maintained:
            if not confirmable:
                notes.append("safe response maintenance undecidable from the available position evidence")
                return Verdict.UNKNOWN
            notes.append("safe response was left while loss of supervision persists")
            return Verdict.VIOLATION
        # Inclusive boundary: at exactly the deadline the evaluator already scores a violation
        # (docs/timing-semantics.md, section 3), so the guard must judge the same instant the
        # same way. A strict ">" here would make every monitor one control step late by
        # construction and would inflate the measured detection delay.
        if now >= deadline:
            if not confirmable:
                notes.append(
                    f"supervision reaction deadline {deadline:.1f} s passed and the response cannot be "
                    "confirmed from the available position evidence"
                )
                return Verdict.UNKNOWN
            notes.append(
                f"supervision lost since {detectable_at:.1f} s; reaction deadline {deadline:.1f} s passed "
                "without a confirmed safe response"
            )
            return Verdict.VIOLATION
        notes.append(f"supervision lost {age:.1f} s ago; inside the reaction window until {deadline:.1f} s")
        return Verdict.PASS

    # ------------------------------------------------------------------ aggregation and escalation
    def _aggregate(
        self, obligations: dict[str, Verdict], assumptions: dict[str, Verdict]
    ) -> tuple[Verdict, bool]:
        if any(v is Verdict.VIOLATION for v in obligations.values()):
            return Verdict.VIOLATION, False
        if any(v is Verdict.UNKNOWN for v in obligations.values()):
            return Verdict.UNKNOWN, False
        if any(v in {Verdict.UNKNOWN, Verdict.VIOLATION} for v in assumptions.values()):
            return Verdict.UNKNOWN, False
        return Verdict.PASS, True

    def _supervisory_rescue_feasible(self) -> bool:
        """Can a supervisory response still arrive inside the reaction deadline?"""
        brief = self._require_brief()
        return brief.declared_supervision_delay_s <= self.obligations.loss_of_supervision_reaction_deadline_s

    def _intervention(
        self,
        obligations: dict[str, Verdict],
        assumptions: dict[str, Verdict],
        supervision_state: str,
        engaged: bool,
        verdict: Verdict,
    ) -> InterventionKind:
        choice: InterventionKind = "none"

        def escalate(candidate: InterventionKind) -> None:
            nonlocal choice
            if _INTERVENTION_RANK[candidate] > _INTERVENTION_RANK[choice]:
                choice = candidate

        for obligation, value in obligations.items():
            if value is not Verdict.VIOLATION:
                continue
            if obligation in {"geofence", "loss_of_supervision_response"}:
                escalate("return_to_launch")
            elif obligation == "inspection_authorization":
                escalate("suspend_inspection")
            elif obligation == "collision":
                escalate("hold")
        if supervision_state == "lost":
            escalate("return_to_launch")
        if verdict is Verdict.UNKNOWN:
            if engaged and obligations.get("inspection_authorization") is Verdict.UNKNOWN:
                escalate("suspend_inspection")
            streak = self._unknown_streak
            if not self._supervisory_rescue_feasible():
                # No supervisor can resolve this inside the deadline, so escalate one level earlier.
                streak += self.thresholds.unknown_grace_steps
            if streak > self.thresholds.unknown_rtl_steps:
                escalate("return_to_launch")
            elif streak > self.thresholds.unknown_hold_steps:
                escalate("suspend_inspection")
            elif streak > self.thresholds.unknown_grace_steps:
                escalate("hold")
        return choice
