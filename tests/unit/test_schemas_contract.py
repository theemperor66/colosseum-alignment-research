"""Record-schema semantics that the rest of the study depends on."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from colosseum_assurance.schemas import (
    AuthorizationView,
    Box,
    FrameRef,
    MonitorReport,
    ObservationPacket,
    SensorHealth,
    SupervisionView,
    Vec3,
    Verdict,
)


def test_box_rejects_inverted_bounds():
    with pytest.raises(ValidationError):
        Box(x_min=1.0, x_max=0.0, y_min=-1.0, y_max=1.0, z_min=-5.0, z_max=0.0)


def test_box_contains_and_exceedance_use_ned_convention():
    box = Box(x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, z_min=-10.0, z_max=0.0)
    assert box.contains(Vec3(x=0.0, y=0.0, z=-5.0))
    assert not box.contains(Vec3(x=0.0, y=0.0, z=-10.5))
    assert box.contains(Vec3(x=0.0, y=0.0, z=-10.5), tolerance=0.6)
    assert box.signed_exceedance(Vec3(x=2.5, y=0.0, z=-5.0)) == pytest.approx(1.5)
    assert box.signed_exceedance(Vec3(x=0.0, y=0.0, z=-5.0)) < 0.0


def test_affirmative_requires_pass_verdict():
    MonitorReport(step_index=0, sim_time_s=0.0, monitor_id="m", verdict=Verdict.PASS, affirmative=True)
    with pytest.raises(ValidationError):
        MonitorReport(
            step_index=0, sim_time_s=0.0, monitor_id="m", verdict=Verdict.UNKNOWN, affirmative=True
        )
    with pytest.raises(ValidationError):
        MonitorReport(
            step_index=0, sim_time_s=0.0, monitor_id="m", verdict=Verdict.VIOLATION, affirmative=True
        )


def test_frame_ref_nonempty_requires_real_pixel_content():
    empty = FrameRef(kind="depth", camera_name="front_center", sim_time_s=0.0, width=4, height=4)
    assert not empty.is_nonempty
    blank = FrameRef(
        kind="depth", camera_name="front_center", sim_time_s=0.0, width=4, height=4,
        nonzero_fraction=0.0, max_value=0.0,
    )
    assert not blank.is_nonempty
    real = FrameRef(
        kind="depth", camera_name="front_center", sim_time_s=0.0, width=4, height=4,
        nonzero_fraction=0.9, max_value=12.0,
    )
    assert real.is_nonempty


def test_observation_packet_forbids_unknown_fields():
    """A leak attempt must fail loudly rather than ride along in an extra field."""
    with pytest.raises(ValidationError):
        ObservationPacket(
            step_index=0,
            receive_sim_time_s=0.0,
            state=None,
            state_age_s=None,
            depth=None,
            rgb=None,
            supervision=SupervisionView(sim_time_s=0.0, last_heartbeat_sim_time_s=None, heartbeat_age_s=None),
            authorization=AuthorizationView(),
            sensor_health=SensorHealth(),
            true_collision=True,
        )
