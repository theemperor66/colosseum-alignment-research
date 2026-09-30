"""Supervisory link and authorization broker: timing semantics that the obligations depend on."""

from __future__ import annotations

from colosseum_assurance.protocol.spec import ObligationSpec
from colosseum_assurance.runtime.supervision import AuthorizationBroker, SupervisionLink
from colosseum_assurance.scenario.manifest import ScheduleSet
from colosseum_assurance.schemas import AuthorizationStatus


def schedules(outages=(), response_delays=(2.0,), decisions=("granted",), validity=45.0) -> ScheduleSet:
    return ScheduleSet(
        steps=100,
        dt_s=0.5,
        observation_delay_s=[0.0] * 100,
        state_dropout=[False] * 100,
        depth_dropout=[False] * 100,
        heartbeat_times_s=[round(k * 2.0, 3) for k in range(40)],
        supervision_outages_s=list(outages),
        authorization_response_delay_s=list(response_delays),
        authorization_decision=list(decisions),
        authorization_validity_s=validity,
        visibility="clear",
        schedule_seed=7,
    )


OBLIGATIONS = ObligationSpec()


def test_heartbeats_are_delivered_while_the_link_is_up():
    link = SupervisionLink(schedules=schedules(), obligations=OBLIGATIONS)
    link.advance_to(5.0)
    view = link.view(5.0)
    assert view.last_heartbeat_sim_time_s == 4.0
    assert view.heartbeat_age_s == 1.0
    assert view.link_state == "nominal"


def test_outage_suppresses_heartbeats_and_raises_loss_after_the_threshold():
    """Outage windows are half-open [start, end): the heartbeat scheduled at t=10.0 is already lost."""
    link = SupervisionLink(schedules=schedules(outages=[(10.0, 30.0)]), obligations=OBLIGATIONS)
    link.advance_to(9.0)
    assert link.view(9.0).last_heartbeat_sim_time_s == 8.0
    assert link.view(9.0).link_state == "nominal"
    link.advance_to(14.0)
    view = link.view(14.0)
    assert view.last_heartbeat_sim_time_s == 8.0
    assert view.heartbeat_age_s == 6.0
    # age == threshold is still "degraded": loss requires the gap to exceed the threshold.
    assert view.link_state == "degraded"
    events = link.advance_to(18.0)
    kinds = [e.kind for e in events]
    assert "supervision_lost" in kinds
    loss = next(e for e in events if e.kind == "supervision_lost")
    assert loss.payload["detectable_onboard_at_s"] == 14.0  # last heartbeat 8.0 + threshold 6.0
    assert loss.payload["deadline_at_s"] == 19.0            # + reaction deadline 5.0
    assert link.view(18.0).link_state == "lost"


def test_supervision_restored_event_is_emitted_once_after_a_loss():
    link = SupervisionLink(schedules=schedules(outages=[(10.0, 20.0)]), obligations=OBLIGATIONS)
    link.advance_to(19.0)
    events = link.advance_to(23.0)
    kinds = [e.kind for e in events]
    assert "supervision_restored" in kinds
    assert link.view(23.0).link_state == "nominal"
    later = link.advance_to(27.0)
    assert "supervision_restored" not in [e.kind for e in later]


def test_authorization_is_pending_until_the_scheduled_response_arrives():
    broker = AuthorizationBroker(schedules=schedules(response_delays=(6.0,)), obligations=OBLIGATIONS)
    broker.request(10.0)
    broker.advance_to(12.0)
    view = broker.view(12.0)
    assert view.status is AuthorizationStatus.PENDING
    assert view.granted_at_s is None and view.expires_at_s is None
    broker.advance_to(16.0)
    view = broker.view(16.0)
    assert view.status is AuthorizationStatus.GRANTED
    assert view.granted_at_s == 16.0
    assert view.expires_at_s == 16.0 + OBLIGATIONS.authorization_validity_s


def test_authorization_expires_exactly_at_the_end_of_its_validity_window():
    broker = AuthorizationBroker(
        schedules=schedules(response_delays=(1.0,), validity=10.0), obligations=ObligationSpec(
            authorization_validity_s=10.0
        )
    )
    broker.request(0.0)
    broker.advance_to(1.0)
    assert broker.view(5.0).status is AuthorizationStatus.GRANTED
    assert broker.view(10.999).status is AuthorizationStatus.GRANTED
    assert broker.view(11.0).status is AuthorizationStatus.EXPIRED
    events = broker.advance_to(11.0)
    assert [e.kind for e in events] == ["authorization_expired"]


def test_second_request_uses_the_second_scheduled_response():
    broker = AuthorizationBroker(
        schedules=schedules(response_delays=(1.0, 9.0), decisions=("granted", "denied")),
        obligations=OBLIGATIONS,
    )
    broker.request(0.0)
    broker.request(20.0)
    broker.advance_to(30.0)
    assert broker.request_count == 2
    kinds = [e.kind for e in broker.events]
    assert kinds.count("authorization_requested") == 2
    assert "authorization_denied" in kinds


def test_denied_authorization_is_never_reported_as_granted():
    broker = AuthorizationBroker(
        schedules=schedules(response_delays=(1.0,), decisions=("denied",)), obligations=OBLIGATIONS
    )
    broker.request(0.0)
    broker.advance_to(2.0)
    assert broker.view(2.0).status is AuthorizationStatus.DENIED
    assert broker.truth_status(2.0)[0] is AuthorizationStatus.DENIED
