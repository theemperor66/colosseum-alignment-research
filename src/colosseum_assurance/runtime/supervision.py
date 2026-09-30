"""Simulated supervisory link and authorization broker.

Both are driven entirely by the precomputed :class:`ScheduleSet` of the scenario manifest, so two arms
whose trajectories diverge still receive the same exogenous treatment: heartbeats arrive at the same
scheduled simulator times, and the *n*-th authorization request receives the *n*-th scheduled response
delay and decision.

Synthetic supervisory delays model timing assumptions. They are not measurements of human performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from colosseum_assurance.protocol.spec import ObligationSpec
from colosseum_assurance.scenario.manifest import ScheduleSet
from colosseum_assurance.schemas import (
    AuthorizationStatus,
    AuthorizationView,
    SupervisionView,
    TruthEvent,
)


@dataclass
class SupervisionLink:
    """Heartbeat delivery and the observable link state."""

    schedules: ScheduleSet
    obligations: ObligationSpec
    _events: list[TruthEvent] = field(default_factory=list)
    _delivered: list[float] = field(default_factory=list)
    _loss_open: bool = False
    _last_seen_time_s: float = 0.0

    def advance_to(self, now_s: float) -> list[TruthEvent]:
        """Deliver every scheduled heartbeat up to ``now_s`` and emit loss/restore events."""
        new_events: list[TruthEvent] = []
        for t in self.schedules.heartbeat_times_s:
            if self._last_seen_time_s < t <= now_s and self.schedules.supervision_available(t):
                self._delivered.append(t)
                new_events.append(
                    TruthEvent(sim_time_s=t, kind="supervision_heartbeat", detail="delivered")
                )
                if self._loss_open:
                    self._loss_open = False
                    new_events.append(
                        TruthEvent(
                            sim_time_s=t,
                            kind="supervision_restored",
                            detail="heartbeat delivered after a loss",
                        )
                    )
        self._last_seen_time_s = now_s

        gap = self.heartbeat_age_s(now_s)
        if gap is not None and gap > self.obligations.loss_of_supervision_threshold_s and not self._loss_open:
            self._loss_open = True
            detectable_at = (
                self.last_delivered_before(now_s) or 0.0
            ) + self.obligations.loss_of_supervision_threshold_s
            new_events.append(
                TruthEvent(
                    sim_time_s=now_s,
                    kind="supervision_lost",
                    detail="heartbeat gap exceeded the loss-of-supervision threshold",
                    payload={
                        "threshold_s": self.obligations.loss_of_supervision_threshold_s,
                        "gap_s": gap,
                        "detectable_onboard_at_s": round(detectable_at, 6),
                        "reaction_deadline_s": self.obligations.loss_of_supervision_reaction_deadline_s,
                        "deadline_at_s": round(
                            detectable_at + self.obligations.loss_of_supervision_reaction_deadline_s, 6
                        ),
                    },
                )
            )
        self._events.extend(new_events)
        return new_events

    def last_delivered_before(self, now_s: float) -> float | None:
        best: float | None = None
        for t in self._delivered:
            if t <= now_s and (best is None or t > best):
                best = t
        return best

    def heartbeat_age_s(self, now_s: float) -> float | None:
        last = self.last_delivered_before(now_s)
        if last is None:
            return now_s  # nothing ever received: the age is the whole elapsed mission time
        return max(0.0, now_s - last)

    def view(self, now_s: float) -> SupervisionView:
        """The observable link state: only heartbeat arrival times, never the outage schedule."""
        last = self.last_delivered_before(now_s)
        age = self.heartbeat_age_s(now_s)
        period = self.obligations.supervision_heartbeat_period_s
        threshold = self.obligations.loss_of_supervision_threshold_s
        if age is None:
            link = "unknown"
        elif age <= 1.5 * period:
            link = "nominal"
        elif age <= threshold:
            link = "degraded"
        else:
            link = "lost"
        return SupervisionView(
            sim_time_s=now_s,
            last_heartbeat_sim_time_s=last,
            # Heartbeats are link-level liveness signals: when the link is up they are delivered at the
            # moment they are produced, so production time and receipt time coincide here. The supervisory
            # response delay applies to authorization decisions, not to heartbeats
            # (see docs/timing-semantics.md).
            last_heartbeat_received_at_s=last,
            heartbeat_age_s=age,
            link_state=link,  # type: ignore[arg-type]
        )

    @property
    def events(self) -> list[TruthEvent]:
        return list(self._events)


@dataclass
class AuthorizationBroker:
    """Issues authorizations for the permission-dependent inspection step.

    A request made at simulator time ``t`` is decided at ``t + scheduled_delay(ordinal)``. The vehicle
    sees nothing until that moment, which is why a slow supervisor can leave an autonomous approach
    unauthorized.
    """

    schedules: ScheduleSet
    obligations: ObligationSpec
    _requests: list[dict[str, float | str]] = field(default_factory=list)
    _events: list[TruthEvent] = field(default_factory=list)
    _expired_announced: set[str] = field(default_factory=set)

    def request(self, now_s: float) -> TruthEvent:
        ordinal = len(self._requests)
        delay_s, decision = self.schedules.authorization_response(ordinal)
        decided_at = now_s + delay_s
        token_id = f"auth-{ordinal:02d}"
        self._requests.append(
            {
                "token_id": token_id,
                "requested_at_s": now_s,
                "decided_at_s": decided_at,
                "decision": decision,
                "expires_at_s": decided_at + self.obligations.authorization_validity_s,
                "ordinal": float(ordinal),
            }
        )
        event = TruthEvent(
            sim_time_s=now_s,
            kind="authorization_requested",
            detail=f"request {ordinal} for the inspection step",
            payload={
                "token_id": token_id,
                "ordinal": ordinal,
                "response_source": "synthetic_broker",
                "scheduled_response_delay_s": delay_s,
                "decision": decision,
                "decided_at_s": decided_at,
            },
        )
        self._events.append(event)
        return event

    def advance_to(self, now_s: float) -> list[TruthEvent]:
        """Emit grant, denial, and expiry events whose simulator time has arrived."""
        new_events: list[TruthEvent] = []
        for req in self._requests:
            token = str(req["token_id"])
            decided_at = float(req["decided_at_s"])
            expires_at = float(req["expires_at_s"])
            decision = str(req["decision"])
            granted_key = f"granted:{token}"
            if decided_at <= now_s and granted_key not in self._expired_announced:
                self._expired_announced.add(granted_key)
                new_events.append(
                    TruthEvent(
                        sim_time_s=decided_at,
                        kind="authorization_granted" if decision == "granted" else "authorization_denied",
                        detail=f"supervisor decision for {token}",
                        payload={
                            "token_id": token,
                            "decided_at_s": decided_at,
                            "received_at_s": now_s,
                            "response_source": "synthetic_broker",
                            "granted_at_s": decided_at,
                            "expires_at_s": expires_at,
                            "validity_s": self.obligations.authorization_validity_s,
                            "scope": "inspection_step",
                        },
                    )
                )
            expiry_key = f"expired:{token}"
            if decision == "granted" and expires_at <= now_s and expiry_key not in self._expired_announced:
                self._expired_announced.add(expiry_key)
                new_events.append(
                    TruthEvent(
                        sim_time_s=expires_at,
                        kind="authorization_expired",
                        detail=f"{token} reached the end of its validity window",
                        payload={"token_id": token, "expires_at_s": expires_at},
                    )
                )
        self._events.extend(new_events)
        return new_events

    def truth_status(self, now_s: float) -> tuple[AuthorizationStatus, dict[str, float | str] | None]:
        """The privileged authorization state: the newest granted, unexpired token if one exists."""
        best: dict[str, float | str] | None = None
        for req in self._requests:
            if str(req["decision"]) != "granted":
                continue
            if float(req["decided_at_s"]) <= now_s < float(req["expires_at_s"]):
                if best is None or float(req["decided_at_s"]) > float(best["decided_at_s"]):
                    best = req
        if best is not None:
            return AuthorizationStatus.GRANTED, best
        expired = [r for r in self._requests
                   if str(r["decision"]) == "granted" and float(r["expires_at_s"]) <= now_s]
        if expired:
            newest = max(expired, key=lambda r: float(r["expires_at_s"]))
            return AuthorizationStatus.EXPIRED, newest
        denied = [r for r in self._requests
                  if str(r["decision"]) == "denied" and float(r["decided_at_s"]) <= now_s]
        if denied:
            return AuthorizationStatus.DENIED, denied[-1]
        if self._requests:
            return AuthorizationStatus.PENDING, self._requests[-1]
        return AuthorizationStatus.ABSENT, None

    def view(self, now_s: float) -> AuthorizationView:
        """What the vehicle holds. Identical content to truth, but only after the decision arrives."""
        status, req = self.truth_status(now_s)
        if req is None:
            return AuthorizationView(status=status)
        decided_at = float(req["decided_at_s"])
        received = decided_at if decided_at <= now_s else None
        return AuthorizationView(
            token_id=str(req["token_id"]),
            status=status,
            requested_at_s=float(req["requested_at_s"]),
            granted_at_s=decided_at if (received is not None and str(req["decision"]) == "granted") else None,
            expires_at_s=float(req["expires_at_s"]) if received is not None else None,
            scope="inspection_step" if received is not None else None,
            received_at_s=received,
            issuer="simulated_supervisor" if received is not None else None,
        )

    @property
    def events(self) -> list[TruthEvent]:
        return list(self._events)

    @property
    def request_count(self) -> int:
        return len(self._requests)
