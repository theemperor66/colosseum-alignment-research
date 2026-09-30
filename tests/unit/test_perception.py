"""Perception is the only place where scene geometry enters the vehicle, so its output must be pinned.

WHY these tests exist
---------------------
The study claims the controller resolves the scene from simulated sensor data rather than from the
scenario manifest. That claim is only worth as much as the depth summary behind it. Every frame here is
built as a float32 array with KNOWN geometry, so each expected value is derived from the pinhole model
documented in ``control/perception.py`` and not from a previous run of the code.

The second reason is the degradation contract the guards depend on:

* ``valid=False``  -> no range information at all; ``degraded_reason`` says why.
* ``valid=True`` with ``min_range_m is None`` -> positively clear within the sensing horizon.

An assumption-aware guard treats those two cases differently, so a change that blurs them must fail here.

Sign convention, read from the source and asserted below: ``bearing = atan((u + 0.5 - width/2) / f)``,
so image columns left of centre give NEGATIVE bearings and columns right of centre give POSITIVE ones.
Elevation is positive upward. All tests use that convention consistently.

Depth convention under test
---------------------------
The study captures ``ImageType = DepthPerspective`` (value 2 in upstream
``AirLib/include/common/ImageCaptureBase.hpp:19-25``; ``docs/image_apis.md:226`` says "For ImageType =
DepthPerspective, you get depth from camera using a projection ray that hits that pixel"), so one
pixel is the RAY distance to the surface point, not the distance to the camera plane. Every frame
below is therefore built from explicit 3D coordinates for a camera at the origin looking along +x
with +y to the right and +z up, and converted with ``r = sqrt(x**2 + y**2 + z**2)``. The expected
numbers are read off those coordinates, never off the projection that ``perception.py`` performs.

The reported contract these tests pin (see the module docstring of ``control/perception.py``):
``min_range_m``, ``sector_min_range_m``, ``free_path_m`` and ``target_range_m`` are RAY distances,
while the corridor gate, the sector height gate and ``width_m`` use the FORWARD (camera-plane)
distance ``r / sqrt(1 + tan(bearing)**2 + tan(elev)**2)``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from colosseum_assurance.control.perception import (
    PerceptionParams,
    column_bearings_rad,
    detect_vertical_structure,
    focal_length_px,
    ray_offsets,
    row_elevations_rad,
    sector_bearings_rad,
    summarize_depth,
)
from colosseum_assurance.schemas import DepthSummary

WIDTH = 256
HEIGHT = 144
HFOV_RAD = math.pi / 2.0  # the Colosseum default camera, 90 degrees
MAX_RANGE_M = 60.0
BEYOND_HORIZON_M = 100.0  # finite, but further than the sensing horizon


def blank_frame(fill_m: float = BEYOND_HORIZON_M) -> np.ndarray:
    """A frame with nothing inside the sensing horizon, used as the background of every scene."""
    return np.full((HEIGHT, WIDTH), fill_m, dtype=np.float32)


def planar_wall(distance_m: float) -> np.ndarray:
    """Ray distances for a fronto-parallel plane ``distance_m`` in front of an ideal pinhole camera.

    Colosseum's ``DepthPerspective`` reports distance ALONG the projection ray, so a flat wall is not a
    constant-depth image: for a pixel with tan(bearing)=u and tan(elevation)=v the ray length is
    ``d * sqrt(1 + u^2 + v^2)``. Building the frame from that exact expression means the expected
    minimum range is the wall distance by construction.
    """
    f = focal_length_px(WIDTH, HFOV_RAD)
    u = (np.arange(WIDTH) + 0.5 - WIDTH / 2.0) / f
    v = (HEIGHT / 2.0 - np.arange(HEIGHT) - 0.5) / f
    return (distance_m * np.sqrt(1.0 + u[None, :] ** 2 + v[:, None] ** 2)).astype(np.float32)


def pixel_tangents() -> tuple[np.ndarray, np.ndarray]:
    """``tan(bearing)`` per column and ``tan(elevation)`` per row for the camera of this module.

    Written out here rather than imported, so that a scene built by these tests never depends on the
    projection code under test; only the pinhole indexing convention is shared.
    """
    f = focal_length_px(WIDTH, HFOV_RAD)
    return (np.arange(WIDTH) + 0.5 - WIDTH / 2.0) / f, (HEIGHT / 2.0 - np.arange(HEIGHT) - 0.5) / f


def fronto_parallel_patch(
    *, forward_m: float, y_range: tuple[float, float], z_range: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Ray distances for a flat rectangle at ``forward_m``, built from its 3D corners.

    The pixel with tangents ``(u, v)`` sees the point ``(d, d*u, d*v)`` of the plane ``x = d``, so the
    value stored is the length of that vector. The boolean mask of the pixels that hit the rectangle
    comes back too, so a test can state its expectation in terms of the 3D points it just built.
    """
    u_t, v_t = pixel_tangents()
    y = forward_m * u_t[None, :]
    z = forward_m * v_t[:, None]
    hit = (y >= y_range[0]) & (y <= y_range[1]) & (z >= z_range[0]) & (z <= z_range[1])
    hit = np.broadcast_to(hit, (HEIGHT, WIDTH))
    ray = np.sqrt(forward_m**2 + y**2 + z**2)
    return np.where(hit, ray, BEYOND_HORIZON_M).astype(np.float32), hit


def summarize(
    frame: np.ndarray | None,
    *,
    sim_time_s: float = 1.0,
    max_range_m: float = MAX_RANGE_M,
    n_sectors: int = 7,
    rgb: np.ndarray | None = None,
    degraded_reason: str | None = None,
) -> DepthSummary:
    """Call ``summarize_depth`` with the fixed camera of this test module."""
    return summarize_depth(
        frame,
        sim_time_s=sim_time_s,
        camera_name="front_center",
        hfov_rad=HFOV_RAD,
        rgb=rgb,
        n_sectors=n_sectors,
        max_range_m=max_range_m,
        degraded_reason=degraded_reason,
    )


# ----------------------------------------------------------------------------------------------
# Geometry: the conventions every other assertion in this file relies on
# ----------------------------------------------------------------------------------------------
def test_pinhole_geometry_puts_negative_bearings_on_the_left_and_positive_on_the_right() -> None:
    """The bearing sign convention is a contract: the controller adds vehicle yaw to these values."""
    bearings = column_bearings_rad(WIDTH, HFOV_RAD)
    assert bearings.shape == (WIDTH,)
    assert bearings[0] < 0.0 < bearings[-1]
    # Half the field of view at the outermost pixel centre, from f = (W/2)/tan(hfov/2).
    f = focal_length_px(WIDTH, HFOV_RAD)
    assert f == pytest.approx(128.0, abs=1e-6)
    assert bearings[0] == pytest.approx(math.atan((0.5 - 128.0) / f), abs=1e-12)
    assert bearings[-1] == pytest.approx(math.atan((255.5 - 128.0) / f), abs=1e-12)
    assert bearings[-1] == pytest.approx(-bearings[0], abs=1e-12)
    # Monotone left to right, so "more negative" always means "further left".
    assert np.all(np.diff(bearings) > 0.0)

    elevations = row_elevations_rad(HEIGHT, WIDTH, HFOV_RAD)
    assert elevations[0] > 0.0 > elevations[-1]  # positive elevation is up, row 0 is the top row
    assert np.all(np.diff(elevations) < 0.0)


def test_sector_bearings_match_the_split_used_by_the_summary() -> None:
    """The controller converts sector indices back to bearings, so both sides must use one split."""
    centres = sector_bearings_rad(7, HFOV_RAD, WIDTH)
    assert len(centres) == 7
    assert centres[0] < 0.0 < centres[-1]
    assert np.all(np.diff(centres) > 0.0)
    assert abs(float(centres[3])) < 0.05  # the middle sector looks along the camera axis


def test_focal_length_rejects_impossible_cameras() -> None:
    """A zero-width image or a 180-degree field of view has no pinhole focal length."""
    with pytest.raises(ValueError):
        focal_length_px(0, HFOV_RAD)
    with pytest.raises(ValueError):
        focal_length_px(WIDTH, math.pi)


def test_ray_offsets_project_a_known_3d_point_onto_the_camera_axes() -> None:
    """Off the optical axis AND off the horizon at once: the case a mixed convention gets wrong.

    A fronto-parallel plane 10 m ahead puts the 3D point ``(10, 10u, 10v)`` on the ray of the pixel
    with tangents ``(u, v)``, so the three expected offsets are the coordinates of that point. The
    pixel used is the one nearest to bearing 40 deg and elevation 25 deg, where the ray distance is
    13.86 m: 39 percent longer than the forward distance, so any formula that feeds the raw ray
    distance into a trigonometric factor is visibly wrong there.
    """
    u_t, v_t = pixel_tangents()
    col = int(np.argmin(np.abs(u_t - math.tan(math.radians(40.0)))))
    row = int(np.argmin(np.abs(v_t - math.tan(math.radians(25.0)))))
    # Half a pixel is 0.25 deg here, so the nearest pixel centre cannot sit exactly on 40/25 deg.
    assert math.degrees(math.atan(u_t[col])) == pytest.approx(40.0, abs=0.25)
    assert math.degrees(math.atan(v_t[row])) == pytest.approx(25.0, abs=0.25)

    plane = planar_wall(10.0)
    assert float(plane[row, col]) == pytest.approx(13.8615, abs=1e-3)  # sqrt(10^2 + y^2 + z^2)
    offsets = ray_offsets(plane, hfov_rad=HFOV_RAD)

    # The 3D point this pixel looks at, in metres: forward 10, right 10*tan(40), up 10*tan(25).
    # The tolerance is the float32 resolution of the frame itself (a real capture is float32).
    assert offsets.forward[row, col] == pytest.approx(10.0, rel=1e-6)
    assert offsets.lateral[row, col] == pytest.approx(10.0 * u_t[col], rel=1e-6)
    assert offsets.vertical[row, col] == pytest.approx(10.0 * v_t[row], rel=1e-6)
    # Same statement in degrees. The slack is the sub-pixel offset of the chosen pixel, at most
    # 0.5 px * 10 m / 128 px = 0.04 m. The pre-fix geometry returned 8.91 m and 6.44 m for this
    # pixel (r*sin(bearing) and r*tan(elev)), i.e. 0.52 m and 1.78 m of pure convention error.
    assert offsets.lateral[row, col] == pytest.approx(10.0 * math.tan(math.radians(40.0)), abs=0.05)
    assert offsets.vertical[row, col] == pytest.approx(10.0 * math.tan(math.radians(25.0)), abs=0.05)

    # The projection is a rotation of the same vector, so its length is preserved everywhere.
    recovered = np.sqrt(offsets.forward**2 + offsets.lateral**2 + offsets.vertical**2)
    assert np.allclose(recovered, plane.astype(np.float64), rtol=1e-12, atol=1e-9)
    assert np.allclose(offsets.forward, 10.0, atol=1e-4)  # a flat wall has one forward distance


def test_ray_offsets_rejects_a_frame_that_is_not_two_dimensional() -> None:
    """Projecting a three-channel image as depth would fabricate positions, so it must raise."""
    with pytest.raises(ValueError):
        ray_offsets(np.zeros((4, 4, 3), dtype=np.float32), hfov_rad=HFOV_RAD)


# ----------------------------------------------------------------------------------------------
# Known scenes
# ----------------------------------------------------------------------------------------------
def test_frontal_wall_at_eight_metres_is_reported_at_eight_metres() -> None:
    """The headline range measurement. Tolerance covers the 2nd-percentile robust minimum only."""
    summary = summarize(planar_wall(8.0))
    assert summary.valid is True
    assert summary.min_range_m is not None
    assert summary.min_range_m == pytest.approx(8.0, abs=0.10)
    assert summary.free_path_m == pytest.approx(8.0, abs=0.10)
    # A wall spanning the whole frame has no lateral offset: the nearest surface is straight ahead.
    assert summary.obstacle_bearing_rad == pytest.approx(0.0, abs=0.02)
    assert summary.coverage_fraction == pytest.approx(1.0, abs=1e-9)
    assert len(summary.sector_min_range_m) == 7
    assert all(r < MAX_RANGE_M for r in summary.sector_min_range_m)


def test_constant_range_surface_is_reported_exactly() -> None:
    """With every pixel at 8.0 m the robust quantile has no spread to absorb, so 8.0 must come back."""
    summary = summarize(blank_frame(8.0))
    assert summary.min_range_m == pytest.approx(8.0, abs=1e-6)
    assert summary.free_path_m == pytest.approx(8.0, abs=1e-6)


def test_box_on_the_left_gives_a_negative_obstacle_bearing_and_leaves_the_corridor_free() -> None:
    """A compact obstacle off the flight axis must be located on the correct side, not braked for.

    The box occupies columns 40-80, whose bearings are all negative (left of centre). It is short
    (31 of 144 rows), so it must NOT be mistaken for the tall inspection asset.
    """
    frame = blank_frame()
    frame[60:91, 40:81] = 6.0
    summary = summarize(frame)

    assert summary.valid is True
    assert summary.min_range_m == pytest.approx(6.0, abs=1e-6)
    assert summary.obstacle_bearing_rad is not None
    assert summary.obstacle_bearing_rad < 0.0
    # The box centre is column 60, i.e. atan((60.5 - 128)/128) = -0.487 rad.
    assert summary.obstacle_bearing_rad == pytest.approx(math.atan((60.5 - 128.0) / 128.0), abs=0.03)
    # It is 6 m away but more than 1.2 m off the axis, so the straight-ahead corridor stays clear.
    assert summary.free_path_m == pytest.approx(MAX_RANGE_M, abs=1e-6)
    assert summary.target_visible is False


def test_box_on_the_right_mirrors_the_sign() -> None:
    """The sign convention must be symmetric; a mirrored scene may not produce a mirrored mistake."""
    frame = blank_frame()
    frame[60:91, 175:216] = 6.0
    summary = summarize(frame)
    assert summary.obstacle_bearing_rad is not None
    assert summary.obstacle_bearing_rad > 0.0
    assert summary.min_range_m == pytest.approx(6.0, abs=1e-6)


def test_tall_narrow_structure_is_detected_with_a_plausible_range_and_width() -> None:
    """The inspection asset is exactly this: tall, narrow, vertically continuous, at a known range.

    The patch is cut out of a fronto-parallel wall at 12 m, so every pixel of it belongs to one flat
    surface 12 m from the camera plane. It used to be a block of constant ray distance, which is not
    a flat surface at all but a piece of a sphere, and that is the only reason the old
    ``w = ray * (tan(right) - tan(left))`` could reproduce the expected 2.16 m here. With a real
    plane the expected width is unchanged but now correct for the right reason, and the pre-fix code
    returns 2.22 m for it.
    """
    frame = blank_frame()
    frame[20:131, 120:143] = planar_wall(12.0)[20:131, 120:143]  # 23 columns, 111 of 144 rows
    summary = summarize(frame)

    assert summary.target_visible is True
    assert summary.target_range_m == pytest.approx(12.4, abs=0.2)  # ray distance, so above 12.0 m
    assert summary.target_bearing_rad is not None
    # Column centre 131.5 -> atan((131.5 - 128)/128) = +0.027 rad, just right of the axis.
    assert summary.target_bearing_rad == pytest.approx(math.atan(3.5 / 128.0), abs=0.02)

    detection = detect_vertical_structure(
        frame.astype(np.float64), hfov_rad=HFOV_RAD, max_range_m=MAX_RANGE_M
    )
    assert detection.visible is True
    assert detection.column_span == (120, 142)
    # The plane is 12 m from the camera plane, so the surface between the outer edges of columns 120
    # and 142 is exactly w = 12 * ((143-128)/128 - (120-128)/128) = 2.15625 m across.
    assert detection.width_m == pytest.approx(12.0 * (15.0 / 128.0 + 8.0 / 128.0), abs=0.03)
    assert detection.support_fraction == pytest.approx(111.0 / 144.0, abs=0.05)
    # The reported range is the RAY distance to the surface, which exceeds the 12 m forward distance.
    assert detection.range_m is not None and detection.range_m > 12.1
    assert detection.range_m == pytest.approx(float(np.mean(frame[20:131, 120:143])), abs=0.1)


def test_axis_aligned_box_is_measured_at_its_true_width_bearing_and_ray_range() -> None:
    """A box of known size and centre, rendered here by ray/box intersection, must measure true.

    Box: x in [7.68, 9.18], y in [-1.2, 1.2], z in [-0.1, 4.2] metres, i.e. 2.40 m wide and 4.30 m
    tall, centred on the camera axis in bearing. The camera at the origin lies inside both the y-span
    and the z-span of the box, so only the FRONT face can be seen: every visible point is at x =
    7.68 m and the forward distance is constant, while the ray distance runs from 7.68 m to 8.82 m.
    The face half width is 1.2 / 7.68 = 20/128 of the focal length, so its edges fall on pixel
    boundaries and the expected width carries no quantisation error at all: exactly 2.40 m.
    The pre-fix code measured 2.52 m, because it multiplied the tangent difference by the ray
    distance (mean 8.07 m) instead of the forward distance.
    """
    u_t, v_t = pixel_tangents()
    lo = np.array([7.68, -1.2, -0.1])
    hi = np.array([9.18, 1.2, 4.2])
    direction = np.stack(  # unit ray per pixel, camera at the origin looking along +x
        np.broadcast_arrays(np.ones((HEIGHT, WIDTH)), u_t[None, :], v_t[:, None]), axis=-1
    )
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    # Slab test: enter at the largest per-axis entry time, leave at the smallest exit time.
    t_lo = np.minimum(lo / direction, hi / direction).max(axis=-1)
    t_hi = np.maximum(lo / direction, hi / direction).min(axis=-1)
    hit = (t_lo <= t_hi) & (t_lo > 0.0)
    frame = np.where(hit, t_lo, BEYOND_HORIZON_M).astype(np.float32)

    # Self-check of the scene before it is used as evidence about the code under test.
    assert hit.sum() == 2880
    forward = frame / np.sqrt(1.0 + u_t[None, :] ** 2 + v_t[:, None] ** 2)
    assert np.allclose(forward[hit], 7.68, atol=1e-5)
    assert float(frame[hit].min()) == pytest.approx(7.68, abs=1e-3)

    detection = detect_vertical_structure(
        frame.astype(np.float64), hfov_rad=HFOV_RAD, max_range_m=MAX_RANGE_M
    )
    assert detection.visible is True
    assert detection.column_span == (108, 147)  # 1.2 m / 7.68 m = 20 px each side of the centre
    assert detection.width_m == pytest.approx(2.40, abs=0.05)
    assert detection.bearing_rad == pytest.approx(0.0, abs=1e-9)
    # Reported range is the ray distance averaged over the visible face, not the 7.68 m forward one.
    assert detection.range_m == pytest.approx(float(frame[hit].mean()), abs=0.05)
    assert detection.range_m is not None and detection.range_m > 7.9

    summary = summarize(frame)
    assert summary.target_visible is True
    assert summary.min_range_m == pytest.approx(7.68, abs=0.05)  # nearest ray, straight ahead


def test_corridor_gate_classifies_a_high_obstacle_by_its_projected_position() -> None:
    """The corridor is a 1.2 m box around the flight axis, so entry must be decided in metres.

    Two identical patches sit 2.5 m ahead at about 25 deg elevation (1.10 m to 1.18 m above the
    axis). One is 0.91 m to 1.14 m right of the axis, inside the 1.2 m half width; the other is
    1.24 m to 1.49 m right of it, outside. Only the second may leave the corridor clear.
    The pre-fix code scaled both offsets by the ray-norm factor (1.15 at this angle) and so pushed
    the inside patch out: it reported free_path_m = 60.0 m, the sensing horizon, for an obstacle
    2.9 m ahead and inside the corridor.
    """
    params = PerceptionParams()
    assert params.corridor_half_width_m == 1.2 and params.corridor_half_height_m == 1.2

    inside, inside_hit = fronto_parallel_patch(forward_m=2.5, y_range=(0.90, 1.16), z_range=(1.10, 1.18))
    outside, outside_hit = fronto_parallel_patch(forward_m=2.5, y_range=(1.24, 1.50), z_range=(1.10, 1.18))
    assert inside_hit.sum() >= params.min_pixels_per_sector
    assert outside_hit.sum() >= params.min_pixels_per_sector

    inside_summary = summarize(inside)
    nearest_inside = float(inside[inside_hit].min())  # = sqrt(2.5^2 + 0.908^2 + 1.104^2) = 2.88 m
    assert nearest_inside == pytest.approx(2.880, abs=0.01)
    assert inside_summary.min_range_m == pytest.approx(nearest_inside, abs=0.05)
    assert inside_summary.free_path_m == pytest.approx(nearest_inside, abs=0.05)

    outside_summary = summarize(outside)
    nearest_outside = float(outside[outside_hit].min())
    assert outside_summary.min_range_m == pytest.approx(nearest_outside, abs=0.05)
    # Seen, located, and correctly outside the corridor: the path straight ahead stays clear.
    assert outside_summary.free_path_m == pytest.approx(MAX_RANGE_M, abs=1e-9)


def test_reported_ranges_are_ray_distances_not_forward_distances() -> None:
    """The documented contract: every reported range is the distance to the surface point.

    The patch is 10 m ahead of the camera plane but 8 m to the right, so its forward distance is
    10 m and its ray distance is about 12.7 m. ``min_range_m`` must carry the ray distance.
    """
    frame, hit = fronto_parallel_patch(forward_m=10.0, y_range=(7.8, 8.2), z_range=(-0.3, 0.3))
    assert hit.sum() >= PerceptionParams().min_pixels_per_sector
    nearest = float(frame[hit].min())
    assert nearest == pytest.approx(math.sqrt(10.0**2 + 7.8**2), abs=0.05)

    summary = summarize(frame)
    assert summary.min_range_m == pytest.approx(nearest, abs=0.1)
    assert summary.min_range_m is not None and summary.min_range_m > 12.0  # not the 10 m forward
    assert summary.obstacle_bearing_rad is not None and summary.obstacle_bearing_rad > 0.6
    assert summary.free_path_m == pytest.approx(MAX_RANGE_M, abs=1e-9)  # far outside the corridor


def test_a_wide_wall_is_not_reported_as_the_inspection_asset() -> None:
    """Width is the discriminator: a wall is too wide for the declared asset window, so it is rejected."""
    summary = summarize(planar_wall(12.0))
    assert summary.valid is True
    assert summary.min_range_m is not None
    assert summary.target_visible is False


def test_occluder_in_front_of_the_structure_hides_it() -> None:
    """Occlusion must remove the detection, not degrade it silently into a wrong range.

    The wall at 5 m covers the tower at 12 m. The frame still yields a range (the wall), but the asset
    is no longer visible, which is what forces the controller to search instead of servo.
    """
    frame = blank_frame()
    frame[20:131, 120:143] = planar_wall(12.0)[20:131, 120:143]
    assert summarize(frame).target_visible is True  # the same scene without the occluder

    occluded = np.minimum(frame, planar_wall(5.0))
    summary = summarize(occluded)
    assert summary.valid is True
    assert summary.target_visible is False
    assert summary.target_bearing_rad is None
    assert summary.target_range_m is None
    assert summary.min_range_m == pytest.approx(5.0, abs=0.10)


def test_all_nan_frame_is_invalid_and_says_why() -> None:
    """No finite pixel means "we cannot tell", which must never be reported as "nothing is there"."""
    summary = summarize(np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32))
    assert summary.valid is False
    assert summary.degraded_reason == "no_finite_depth_pixels"
    assert summary.coverage_fraction == 0.0
    assert summary.min_range_m is None
    assert summary.free_path_m is None
    assert summary.target_visible is False


def test_frame_entirely_beyond_the_horizon_is_positively_clear() -> None:
    """valid=True with min_range_m=None is the positive statement the guards rely on."""
    summary = summarize(blank_frame(BEYOND_HORIZON_M))
    assert summary.valid is True
    assert summary.min_range_m is None  # nothing within the horizon, not "unknown"
    assert summary.obstacle_bearing_rad is None
    assert summary.free_path_m == pytest.approx(MAX_RANGE_M, abs=1e-9)
    assert summary.coverage_fraction == pytest.approx(1.0, abs=1e-9)
    assert summary.target_visible is False
    # An empty sector reports the sensing horizon, never a fabricated near value.
    assert summary.sector_min_range_m == [MAX_RANGE_M] * 7


def test_frame_exactly_at_the_horizon_is_still_clear() -> None:
    """The horizon is the edge of knowledge, so a return AT it must not become a near obstacle."""
    summary = summarize(blank_frame(MAX_RANGE_M))
    assert summary.valid is True
    assert summary.min_range_m is None
    assert summary.free_path_m == pytest.approx(MAX_RANGE_M, abs=1e-9)


def test_coverage_fraction_counts_invalid_pixels_and_gates_the_frame() -> None:
    """Coverage is the guards' sensing-quality input, so it must track the invalid pixels exactly."""
    half_bad = np.full((HEIGHT, WIDTH), 20.0, dtype=np.float32)
    half_bad[: HEIGHT // 2, :] = np.nan
    summary = summarize(half_bad)
    assert summary.coverage_fraction == pytest.approx(0.5, abs=1e-9)
    assert summary.valid is True  # 0.50 is above the 0.15 usable floor
    assert summary.min_range_m == pytest.approx(20.0, abs=1e-6)

    mostly_bad = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
    mostly_bad[:14, :] = 20.0  # 14/144 = 0.097 of the frame
    summary = summarize(mostly_bad)
    assert summary.coverage_fraction == pytest.approx(14.0 / HEIGHT, abs=1e-6)
    assert summary.valid is False
    assert summary.degraded_reason is not None
    assert summary.degraded_reason.startswith("low_depth_coverage")
    assert summary.min_range_m is None

    default_floor = PerceptionParams().min_coverage_fraction
    assert 14.0 / HEIGHT < default_floor <= 0.5  # the two cases really do straddle the floor


def test_scattered_invalid_pixels_do_not_break_the_summary() -> None:
    """Real reduced-visibility depth is speckled with NaN; the summary must survive it."""
    frame = np.full((HEIGHT, WIDTH), 20.0, dtype=np.float32)
    frame[::2, ::2] = np.nan
    summary = summarize(frame)
    assert summary.valid is True
    assert summary.coverage_fraction == pytest.approx(0.75, abs=1e-9)
    assert summary.min_range_m == pytest.approx(20.0, abs=1e-6)


def test_missing_frame_is_reported_not_raised() -> None:
    """``depth=None`` is an ordinary step outcome (a dropout), so it must never raise."""
    summary = summarize(None)
    assert summary.valid is False
    assert summary.degraded_reason == "depth_frame_missing"
    assert summary.coverage_fraction == 0.0
    assert summary.min_range_m is None
    assert summary.sector_min_range_m == []
    assert summary.target_visible is False


def test_a_caller_supplied_degraded_reason_survives() -> None:
    """A scheduled dropout and an unreadable frame are different events for the guards."""
    summary = summarize(None, degraded_reason="scheduled_sensor_dropout")
    assert summary.valid is False
    assert summary.degraded_reason == "scheduled_sensor_dropout"


def test_rgb_fallback_supplies_a_bearing_but_never_a_range() -> None:
    """Without depth the vehicle may align on the structure; it may not servo distance to it."""
    rgb = np.full((HEIGHT, WIDTH, 3), 200, dtype=np.uint8)
    rgb[20:130, 150:175, :] = 40  # a dark vertical structure right of centre
    summary = summarize(None, rgb=rgb)
    assert summary.valid is False  # there is still no depth evidence
    assert summary.degraded_reason == "depth_frame_missing+rgb_bearing_only"
    assert summary.target_visible is True
    assert summary.target_bearing_rad is not None
    assert summary.target_bearing_rad > 0.0  # right of centre, matching the sign convention
    assert summary.target_range_m is None


def test_sector_count_is_honoured_and_empty_sectors_report_the_horizon() -> None:
    """The controller indexes sectors by position, so the length and the empty-sector value are a contract."""
    frame = blank_frame()
    frame[60:90, 0:20] = 7.0  # obstacle in the leftmost sector only
    for n_sectors in (3, 5, 7, 9):
        summary = summarize(frame, n_sectors=n_sectors)
        assert len(summary.sector_min_range_m) == n_sectors
        assert summary.sector_min_range_m[0] == pytest.approx(7.0, abs=1e-6)
        assert all(r == pytest.approx(MAX_RANGE_M, abs=1e-9) for r in summary.sector_min_range_m[1:])


def test_a_frame_that_is_not_two_dimensional_is_a_programming_error() -> None:
    """Silently summarising an RGB array as depth would fabricate ranges, so it must raise."""
    with pytest.raises(ValueError):
        summarize(np.zeros((4, 4, 3), dtype=np.float32))


def test_empty_frame_is_invalid() -> None:
    """A zero-sized capture is a failed capture, not a clear scene."""
    summary = summarize(np.zeros((0, 0), dtype=np.float32))
    assert summary.valid is False
    assert summary.degraded_reason == "depth_frame_empty"
