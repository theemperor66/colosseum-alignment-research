"""The delayed observation channel is the study's independent variable; its semantics must be exact."""

from __future__ import annotations

from colosseum_assurance.runtime.observation import ObservationPipeline
from colosseum_assurance.scenario.manifest import ScheduleSet
from colosseum_assurance.schemas import (
    AuthorizationView,
    DepthSummary,
    SupervisionView,
    Vec3,
    VehicleState,
)


def make_schedules(steps=10, delay=1.0, state_dropout=None, depth_dropout=None) -> ScheduleSet:
    return ScheduleSet(
        steps=steps,
        dt_s=0.5,
        observation_delay_s=[delay] * steps,
        state_dropout=state_dropout or [False] * steps,
        depth_dropout=depth_dropout or [False] * steps,
        heartbeat_times_s=[0.0, 2.0, 4.0],
        authorization_response_delay_s=[1.0],
        authorization_decision=["granted"],
        authorization_validity_s=45.0,
        visibility="clear",
        schedule_seed=1,
    )


def state_at(t: float, x: float) -> VehicleState:
    return VehicleState(
        sim_time_s=t, position=Vec3(x=x, y=0.0, z=-5.0), velocity=Vec3(x=1.0, y=0.0, z=0.0), yaw_rad=0.0
    )


def supervision(t: float) -> SupervisionView:
    return SupervisionView(sim_time_s=t, last_heartbeat_sim_time_s=t, heartbeat_age_s=0.0,
                           link_state="nominal")


def test_delayed_observation_returns_an_older_sample():
    pipeline = ObservationPipeline(schedules=make_schedules(delay=1.0), declared_delay_bound_s=1.0)
    for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
        pipeline.push_state(state_at(t, x=t * 2.0))
    obs = pipeline.build(4, now_s=2.0, supervision=supervision(2.0), authorization=AuthorizationView())
    assert obs.state is not None
    assert obs.state.sim_time_s == 1.0  # newest sample at or before now - delay
    assert obs.state_age_s == 1.0
    assert obs.state.position.x == 2.0  # the true position at t=2.0 would be 4.0


def test_zero_delay_returns_the_current_sample():
    pipeline = ObservationPipeline(schedules=make_schedules(delay=0.0), declared_delay_bound_s=0.0)
    pipeline.push_state(state_at(0.0, 0.0))
    pipeline.push_state(state_at(1.0, 5.0))
    obs = pipeline.build(1, now_s=1.0, supervision=supervision(1.0), authorization=AuthorizationView())
    assert obs.state is not None and obs.state.sim_time_s == 1.0
    assert obs.state_age_s == 0.0


def test_dropout_removes_the_sample_and_is_visible_in_sensor_health():
    schedules = make_schedules(delay=0.0, state_dropout=[False, True] + [False] * 8)
    pipeline = ObservationPipeline(schedules=schedules, declared_delay_bound_s=0.0)
    pipeline.push_state(state_at(0.0, 0.0))
    pipeline.push_state(state_at(0.5, 1.0))
    obs = pipeline.build(1, now_s=0.5, supervision=supervision(0.5), authorization=AuthorizationView())
    assert obs.state is None
    assert obs.state_age_s is None
    assert not obs.sensor_health.state_sample_available
    assert obs.sensor_health.dropouts_in_window == 1


def test_no_sample_old_enough_yields_no_state():
    """Early in an episode with a long delay, nothing is old enough to be received yet."""
    pipeline = ObservationPipeline(schedules=make_schedules(delay=2.5), declared_delay_bound_s=2.5)
    pipeline.push_state(state_at(0.0, 0.0))
    obs = pipeline.build(0, now_s=1.0, supervision=supervision(1.0), authorization=AuthorizationView())
    assert obs.state is None


def test_depth_and_rgb_follow_the_depth_dropout_schedule():
    schedules = make_schedules(delay=0.0, depth_dropout=[True] + [False] * 9)
    pipeline = ObservationPipeline(schedules=schedules, declared_delay_bound_s=0.0)
    pipeline.push_depth(DepthSummary(sim_time_s=0.0, camera_name="front_center", valid=True,
                                     min_range_m=8.0, coverage_fraction=0.99))
    first = pipeline.build(0, now_s=0.0, supervision=supervision(0.0), authorization=AuthorizationView())
    assert first.depth is None and not first.sensor_health.depth_available
    second = pipeline.build(1, now_s=0.5, supervision=supervision(0.5), authorization=AuthorizationView())
    assert second.depth is not None and second.depth.min_range_m == 8.0


def test_declared_delay_bound_is_carried_as_an_assumption():
    pipeline = ObservationPipeline(schedules=make_schedules(delay=1.0), declared_delay_bound_s=1.2)
    pipeline.push_state(state_at(0.0, 0.0))
    obs = pipeline.build(2, now_s=1.0, supervision=supervision(1.0), authorization=AuthorizationView())
    assert obs.declared_observation_delay_s == 1.2
    assert obs.state_age_s == 1.0
