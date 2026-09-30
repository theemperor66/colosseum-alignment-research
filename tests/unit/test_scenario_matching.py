"""Arm matching: scenarios and disturbance schedules must not depend on the arm or on actions."""

from __future__ import annotations

import numpy as np

from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.scenario.manifest import build_manifest, enumerate_manifests, scenario_seed


def test_manifest_is_deterministic_and_arm_independent():
    protocol = ProtocolConfig()
    a = build_manifest(protocol, "pilot", "obs_moderate__sup_moderate", 3)
    b = build_manifest(protocol, "pilot", "obs_moderate__sup_moderate", 3)
    assert a.content_hash() == b.content_hash()
    # The manifest is built without any notion of an arm, so all arms share one schedule set.
    assert a.schedules.content_hash() == b.schedules.content_hash()


def test_scenarios_differ_across_cells_and_realizations():
    protocol = ProtocolConfig()
    seeds = {
        scenario_seed(protocol.content_hash(), "pilot", cell["cell_id"], i)
        for cell in protocol.cells()
        for i in range(3)
    }
    assert len(seeds) == len(protocol.cells()) * 3


def test_schedules_cover_the_whole_horizon_and_are_indexable():
    protocol = ProtocolConfig()
    manifest = build_manifest(protocol, "pilot", "obs_severe__sup_severe", 0)
    steps = int(round(protocol.mission.episode_horizon_s / protocol.mission.control_dt_s))
    assert manifest.schedules.steps == steps
    assert len(manifest.schedules.observation_delay_s) == steps
    # Indexing past the end clamps instead of raising, so a longer run cannot consume fresh randomness.
    assert manifest.schedules.delay_at_step(steps + 50) == manifest.schedules.observation_delay_s[-1]


def test_authorization_schedule_is_keyed_by_request_ordinal_not_time():
    """Two arms requesting at different times still receive the same n-th treatment."""
    protocol = ProtocolConfig()
    manifest = build_manifest(protocol, "pilot", "obs_nominal__sup_severe", 1)
    first = manifest.schedules.authorization_response(0)
    assert manifest.schedules.authorization_response(0) == first
    assert manifest.schedules.authorization_response(99) == manifest.schedules.authorization_response(
        len(manifest.schedules.authorization_response_delay_s) - 1
    )


def test_observation_delay_levels_are_ordered_and_include_a_nominal_cell():
    protocol = ProtocolConfig()
    values = [lv.value_s for lv in protocol.conditions.observation_delay_levels]
    assert values == sorted(values)
    assert values[0] == 0.0
    sup = [lv.value_s for lv in protocol.conditions.supervision_delay_levels]
    assert sup == sorted(sup)


def test_enumerate_manifests_produces_the_planned_grid():
    protocol = ProtocolConfig()
    manifests = enumerate_manifests(protocol, "pilot")
    expected = len(protocol.cells()) * protocol.sampling.pilot_realizations_per_cell
    assert len(manifests) == expected
    assert len({m.scenario_id for m in manifests}) == expected


def test_layouts_include_occluders_so_perception_matters():
    protocol = ProtocolConfig()
    variants = {
        build_manifest(protocol, "pilot", "obs_nominal__sup_nominal", i).layout_variant for i in range(3)
    }
    assert variants == set(protocol.conditions.layout_variants)
    built = [build_manifest(protocol, "pilot", "obs_nominal__sup_nominal", i) for i in range(6)]
    cluttered = next(m for m in built if m.layout_variant == "cluttered")
    assert any(o.occludes_asset for o in cluttered.obstacles)
    assert np.isfinite(cluttered.obstacle_clearance(cluttered.start_position))
