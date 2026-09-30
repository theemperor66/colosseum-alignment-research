"""Depth-first perception: one captured frame in, one small non-privileged feature set out.

WHY this module exists at all
-----------------------------
The study claims something about a *perception-based* controller, so the controller must resolve the
scene from simulated sensor data. The scenario manifest knows where every obstacle is; the controller
must not. This module is therefore the only place where scene geometry enters the vehicle, and it may
read pixels, fixed camera calibration, and onboard attitude only. It imports no scenario, evaluation,
or runtime module. Absolute simulator camera poses are deliberately excluded.

Live camera calibration
-----------------------
When ``camera_mount`` is supplied, every pixel's camera-frame point is transformed through the fixed
body mounting and full ONBOARD estimated attitude. The resulting frame is centered on the vehicle,
gravity-level, with the onboard yaw removed; the controller adds its delivered onboard yaw for world
heading. Sector bins and corridor gates use these transformed points, including pitch and roll.
Ranges then mean vehicle-center 3D distance. The explicitly named ``target_horizontal_range_m``
serves XY reconstruction; obstacle/target matching keeps the 3D distance. Missing angular coverage
is unknown, not free space. Raw camera-ray validity and the sensor horizon are checked before the
transform. The vertical-structure detector itself remains an image-column heuristic and may fail
on banked imagery; calibration does not turn that heuristic into a perfect detector.

The historical fixture path without a mount retains the level camera-at-origin calculations below;
its summary explicitly labels that assumption. Production runtime requires measured mounting and
full onboard attitude. The following camera-ray convention describes raw data and that legacy path.

Depth convention: DepthPerspective, distance along the projection ray
--------------------------------------------------------------------
The study captures ``ImageType = DepthPerspective`` (enum value 2; ``Scene=0``, ``DepthPlanar=1``,
``DepthPerspective=2``, ``DepthVis=3`` in ``AirLib/include/common/ImageCaptureBase.hpp:19-25`` at the
pinned upstream commit ``84fc0c1c75bc73a0135ee80a325d470577c66c52`` of
github.com/CodexLabsLLC/Colosseum). Upstream ``docs/image_apis.md:226`` states the difference: "For
ImageType = DepthPlanar, you get depth in camera plane, i.e., all points that are plane-parallel to
the camera have same depth. For ImageType = DepthPerspective, you get depth from camera using a
projection ray that hits that pixel." A pixel value ``r`` is therefore the distance from the camera
centre to the surface point ALONG that pixel's ray, not the distance to the camera plane.

A ray distance is not yet a position, so it must be projected before it can be compared with a
rectangular corridor or turned into the width of an object. The camera is modelled as an ideal
pinhole with square pixels and no distortion, and the horizontal field of view ``hfov_rad`` comes
from the caller (the simulator settings), never from a hardcoded constant::

    f_px     = (width / 2) / tan(hfov / 2)
    u_t      = (u + 0.5 - width / 2) / f_px         # tan(bearing), positive right of the axis
    v_t      = (height / 2 - v - 0.5) / f_px        # tan(elevation), positive up
    bearing  = atan(u_t)
    elev     = atan(v_t)
    norm     = sqrt(1 + u_t**2 + v_t**2)            # ray length per unit of axis length
    forward  = r / norm                             # distance to the camera plane (planar depth)
    lateral  = forward * u_t                        # metres right of the camera axis
    vertical = forward * v_t                        # metres above the camera axis

Those three offsets are the camera-frame coordinates of the surface point, so
``r**2 = forward**2 + lateral**2 + vertical**2`` holds exactly. :func:`ray_offsets` is the only
place that computes them, because a second copy of this projection is how the two axes drift apart:
``r * sin(bearing)`` for the lateral offset is exact only at zero elevation, and ``r * tan(elev)``
for the vertical offset is exact nowhere for a ray distance. Mixing the two describes two different
scenes on the two axes.

The vertical field of view follows from the same focal length, so a non-square image is handled
correctly. Bearings are relative to the camera axis; the caller adds the vehicle yaw. In NED a
positive yaw turns from +x (north) toward +y (east), i.e. to the right, so ``yaw + bearing`` is the
world heading of a feature. The frame is assumed to be captured with a level camera; a strongly
banked vehicle would break the vertical gating below, which is acceptable for a bounded inspection
mission flown at low speed.

What each reported number carries
---------------------------------
Every reported RANGE is a ray distance; forward distance is used only where the geometry demands it
(the rectangular corridor gate, the sector height gate, and the width of a fronto-parallel surface):

===========================================  ====================================================
Field                                        Quantity
===========================================  ====================================================
``DepthSummary.min_range_m``                 ray distance (m) to the nearest gated surface
``DepthSummary.sector_min_range_m[i]``       ray distance (m) inside sector ``i``
``DepthSummary.free_path_m``                 ray distance (m) of the nearest corridor pixel
``DepthSummary.target_range_m``              ray distance (m) to the detected surface
``TargetDetection.width_m``                  metres across, computed from the FORWARD distance
corridor gate, sector gate, column banding   forward/lateral/vertical (m) from :func:`ray_offsets`
===========================================  ====================================================

WHY ray distance for every reported range: it is the physical camera-to-surface separation that the
simulator measured, it does not depend on where in the image the surface happens to fall, and the
consumers compare all of these numbers against the same physical thresholds (contact margin,
stand-off, emergency stop). One convention for the reported set is the point; the defect this module
had was exactly two conventions at once. The known cost is stated rather than hidden: inside the
corridor the ray distance exceeds the along-axis clearance by at most ``sqrt(1 + 2 * (1.2 / d)**2)``,
i.e. about 2 percent at 8 m but about 15 percent at 3 m, so the emergency-stop margin must absorb a
near-field optimism of that size.

Degradation is reported, never hidden
-------------------------------------
Every failure mode sets ``valid=False`` plus a machine-readable ``degraded_reason``:
missing frame, no finite pixels, coverage below the usable threshold. ``valid=True`` with
``min_range_m is None`` is a positive statement ("nothing within the sensing horizon"), which is very
different from ``valid=False`` ("we cannot tell"). The assumption-aware guard depends on that
distinction, so it must not be blurred.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from colosseum_assurance.control.camera_geometry import CameraMount, camera_to_vehicle_level
from colosseum_assurance.schemas import DepthSummary, FrameRef

__all__ = [
    "PerceptionParams",
    "RayOffsets",
    "TargetDetection",
    "column_bearings_rad",
    "detect_vertical_structure",
    "detect_vertical_structure_rgb",
    "focal_length_px",
    "ray_offsets",
    "row_elevations_rad",
    "sector_bearings_rad",
    "summarize_depth",
]


@dataclass(frozen=True, slots=True)
class PerceptionParams:
    """Frozen perception thresholds.

    They are part of the frozen configuration of the experiment: changing one changes what the
    controller can see, so they live in one inspectable object instead of being scattered as literals.
    """

    min_coverage_fraction: float = 0.15
    """Below this fraction of finite pixels the frame is declared unusable rather than summarised."""

    robust_quantile: float = 0.02
    """Ranges are the 2nd percentile of a pixel set, not its minimum: one speckle must not brake the
    vehicle, and reduced-visibility depth is noisy."""

    min_pixels_per_sector: int = 8
    """A sector needs at least this many in-range pixels before its range is believed."""

    corridor_half_width_m: float = 1.2
    """Half width of the straight-ahead corridor used for ``free_path_m`` (vehicle plus margin)."""

    corridor_half_height_m: float = 1.2
    """Half height of that corridor. Vertical gating removes the ground plane and the sky, which are
    always present in a forward-looking depth frame and are not obstacles at cruise altitude."""

    sector_half_height_m: float = 2.5
    """Vertical gate for the per-sector ranges; taller than the corridor so that walls and towers are
    seen early while the ground plane is still excluded."""

    target_band_m: float = 1.5
    """Depth quantisation used to find a vertically continuous surface inside a column."""

    target_min_height_fraction: float = 0.28
    """A column supports a target only if this fraction of its rows sits in one depth band."""

    target_min_columns: int = 3
    target_min_width_m: float = 1.0
    target_max_width_m: float = 6.0
    """Width window of the inspection asset. It rejects wide walls and thin masts, and it is the
    detector's main confusion mode: a thin mast measured at long range can pass the lower bound."""

    target_range_tolerance_m: float = 1.5
    target_range_rel_tolerance: float = 0.08
    """Tolerance for calling two adjacent columns part of the same surface."""

    rgb_min_height_fraction: float = 0.30
    rgb_min_contrast: float = 10.0
    rgb_max_width_fraction: float = 0.60


@dataclass(frozen=True, slots=True)
class TargetDetection:
    """One vertical-structure hypothesis, in camera-relative terms only."""

    visible: bool
    bearing_rad: float | None = None
    range_m: float | None = None
    width_m: float | None = None
    support_fraction: float = 0.0
    column_span: tuple[int, int] | None = None
    source: str = "depth"
    camera_point_frd_m: tuple[float, float, float] | None = None
    """Representative selected surface point, forward/right/down; never world coordinates."""


# ----------------------------------------------------------------------------------------------
# Pinhole geometry
# ----------------------------------------------------------------------------------------------
def focal_length_px(width: int, hfov_rad: float) -> float:
    """Focal length in pixels for an ideal pinhole with horizontal field of view ``hfov_rad``."""
    if width <= 0:
        raise ValueError("width must be positive")
    if not (0.0 < hfov_rad < math.pi):
        raise ValueError(f"hfov_rad must be in (0, pi), got {hfov_rad}")
    return (width / 2.0) / math.tan(hfov_rad / 2.0)


def column_bearings_rad(width: int, hfov_rad: float) -> np.ndarray:
    """Bearing of every image column, positive to the right of the camera axis."""
    f = focal_length_px(width, hfov_rad)
    u = np.arange(width, dtype=np.float64) + 0.5
    return np.arctan((u - width / 2.0) / f)


def sector_bearings_rad(n_sectors: int, hfov_rad: float, width: int = 256) -> np.ndarray:
    """Centre bearing of each sector produced by :func:`summarize_depth`.

    The controller receives sector ranges without their bearings, so both sides must derive them from
    the same column split. Keeping that split in one function stops the two from drifting apart.
    """
    if n_sectors < 1:
        raise ValueError("n_sectors must be >= 1")
    f = focal_length_px(width, hfov_rad)
    centres = [
        math.atan((columns[len(columns) // 2] + 0.5 - width / 2.0) / f)
        for columns in np.array_split(np.arange(width), n_sectors)
    ]
    return np.asarray(centres, dtype=np.float64)


def row_elevations_rad(height: int, width: int, hfov_rad: float) -> np.ndarray:
    """Elevation of every image row, positive upward. Uses the horizontal focal length (square pixels)."""
    if height <= 0:
        raise ValueError("height must be positive")
    f = focal_length_px(width, hfov_rad)
    v = np.arange(height, dtype=np.float64) + 0.5
    return np.arctan((height / 2.0 - v) / f)


@dataclass(frozen=True, slots=True)
class RayOffsets:
    """Camera-frame coordinates of every pixel's surface point, in metres.

    ``forward`` is the planar depth (distance to the camera plane), ``lateral`` is positive to the
    right of the camera axis and ``vertical`` is positive above it. All three have the shape of the
    depth frame and carry NaN wherever the frame did.
    """

    forward: np.ndarray
    lateral: np.ndarray
    vertical: np.ndarray


def _tangent_grids(height: int, width: int, hfov_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """``tan(bearing)`` per column and ``tan(elevation)`` per row, as 1-D arrays."""
    f = focal_length_px(width, hfov_rad)
    if height <= 0:
        raise ValueError("height must be positive")
    u_t = (np.arange(width, dtype=np.float64) + 0.5 - width / 2.0) / f
    v_t = (height / 2.0 - np.arange(height, dtype=np.float64) - 0.5) / f
    return u_t, v_t


def ray_offsets(ray_distance_m: np.ndarray, *, hfov_rad: float) -> RayOffsets:
    """Project a DepthPerspective frame (ray distance in metres) onto the camera axes.

    WHY this exists as one function: every gate in this module needs the same three numbers, and a
    second hand-written copy of the projection is how the horizontal and the vertical axis end up on
    two different conventions. The full-ray form is used, so it stays exact off the optical axis and
    off the horizon at the same time::

        norm = sqrt(1 + tan(bearing)**2 + tan(elev)**2)
        forward, lateral, vertical = r / norm, forward * tan(bearing), forward * tan(elev)

    The returned grids satisfy ``forward**2 + lateral**2 + vertical**2 == r**2`` for every finite
    pixel, which is the identity the tests pin.
    """
    arr = np.asarray(ray_distance_m, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"ray_distance_m must be a 2-D array of metres, got shape {arr.shape}")
    height, width = arr.shape
    u_t, v_t = _tangent_grids(height, width, hfov_rad)
    norm = np.sqrt(1.0 + u_t[None, :] ** 2 + v_t[:, None] ** 2)
    forward = arr / norm
    return RayOffsets(forward=forward, lateral=forward * u_t[None, :], vertical=forward * v_t[:, None])


def _robust_min(values: np.ndarray, quantile: float, fallback: float) -> float:
    """Quantile-based minimum. Falls back when the sample is empty."""
    if values.size == 0:
        return fallback
    return float(np.quantile(values, quantile))


# ----------------------------------------------------------------------------------------------
# Asset detector
# ----------------------------------------------------------------------------------------------
def detect_vertical_structure(
    depth: np.ndarray,
    *,
    hfov_rad: float,
    max_range_m: float,
    params: PerceptionParams | None = None,
) -> TargetDetection:
    """Find the tall, narrow, vertically continuous surface that the inspection asset presents.

    The discriminator is deliberately simple and explainable, because the study must be able to say
    what perception did and where it fails:

    1. Quantise each column's in-range FORWARD distances into bands and keep the band with the most
       rows. A tower occupies one band over most of a column; the ground plane sweeps continuously
       through many bands, so it never wins by height.
    2. Keep columns whose best band covers at least ``target_min_height_fraction`` of the rows.
    3. Group adjacent surviving columns whose forward distances agree, then measure the group's
       physical width with the exact pinhole expression for a fronto-parallel surface,
       ``w = forward * (tan(theta_right) - tan(theta_left))``.
    4. Accept a group whose width lies inside the asset window, and prefer the tallest (then the
       nearest) accepted group.

    ``depth`` is a DepthPerspective frame, i.e. ray distance per pixel (see the module docstring), so
    every step above first converts it with :func:`ray_offsets`. ``range_m`` is reported back as the
    ray distance; only the banding, the grouping and the width use the forward distance.

    Known confusions: a thin mast at long range can measure wide enough to pass, and a wall seen
    edge-on can measure narrow enough to pass. Both are genuine perception errors and are left in.
    """
    p = params or PerceptionParams()
    if depth.ndim != 2:
        raise ValueError(f"depth must be a 2-D array, got shape {depth.shape}")
    height, width = depth.shape
    finite = np.isfinite(depth) & (depth > 0.0)
    in_range = finite & (depth <= max_range_m)
    if not in_range.any():
        return TargetDetection(visible=False)

    # Band in FORWARD distance, not in ray distance. A vertical planar face has one forward distance
    # for every row and column it covers, while its ray distance grows with the off-axis angle (by
    # 11 percent at the corner of a 90-degree frame), which would smear one surface across bands.
    offsets = ray_offsets(depth, hfov_rad=hfov_rad)
    forward = offsets.forward
    n_bands = max(int(math.ceil(max_range_m / p.target_band_m)), 1)
    # NaN and infinity are legitimate depth values from a real camera, and casting them to int is
    # undefined. Replace them with zero before the cast and rely on `in_range` to mask them out.
    safe_forward = np.where(finite, forward, 0.0)
    band_index = np.clip((safe_forward / p.target_band_m).astype(np.int64), 0, n_bands - 1)
    flat = np.where(in_range, band_index + np.arange(width)[None, :] * n_bands, -1).ravel()
    flat = flat[flat >= 0]
    counts = np.bincount(flat, minlength=width * n_bands).reshape(width, n_bands).astype(np.float64)
    # A surface may straddle two bands, so score a 3-band window.
    window = counts.copy()
    window[:, 1:] += counts[:, :-1]
    window[:, :-1] += counts[:, 1:]

    best_band = np.argmax(window, axis=1)
    best_support = window[np.arange(width), best_band] / float(height)
    candidate = best_support >= p.target_min_height_fraction
    if not candidate.any():
        return TargetDetection(visible=False)

    # Two ranges per column, both averaged over exactly the pixels of the winning band: the reported
    # one is the RAY distance (what the sensor measured), the working one is the FORWARD distance
    # (what the width formula and the same-surface test need).
    col_range = np.full(width, np.nan)
    col_forward = np.full(width, np.nan)
    col_points = np.full((width, 3), np.nan)
    for u in np.flatnonzero(candidate):
        lo = (best_band[u] - 1) * p.target_band_m
        hi = (best_band[u] + 2) * p.target_band_m
        sel = in_range[:, u] & (forward[:, u] >= lo) & (forward[:, u] < hi)
        if sel.any():
            col_range[u] = float(np.mean(depth[sel, u]))
            col_forward[u] = float(np.mean(forward[sel, u]))
            col_points[u] = (col_forward[u], float(np.mean(offsets.lateral[sel, u])),
                             -float(np.mean(offsets.vertical[sel, u])))
        else:
            candidate[u] = False

    f = focal_length_px(width, hfov_rad)
    groups: list[tuple[int, int]] = []
    start: int | None = None
    for u in range(width):
        if candidate[u] and start is None:
            start = u
        elif start is not None:
            # Same-surface test in FORWARD distance: a flat wall keeps one forward distance across
            # columns, while its ray distance grows with the bearing and would split the wall in two.
            tol = p.target_range_tolerance_m + p.target_range_rel_tolerance * col_forward[u - 1]
            if not candidate[u] or abs(col_forward[u] - col_forward[u - 1]) > tol:
                groups.append((start, u - 1))
                start = u if candidate[u] else None
    if start is not None:
        groups.append((start, width - 1))

    best: TargetDetection = TargetDetection(visible=False)
    best_key: tuple[float, float] = (-1.0, 0.0)
    for lo, hi in groups:
        n_cols = hi - lo + 1
        if n_cols < p.target_min_columns:
            continue
        rng = float(np.nanmedian(col_range[lo : hi + 1]))  # ray distance, reported as-is
        rng_forward = float(np.nanmedian(col_forward[lo : hi + 1]))
        if not (math.isfinite(rng) and math.isfinite(rng_forward)):
            continue
        left = (lo - width / 2.0) / f
        right = (hi + 1.0 - width / 2.0) / f
        # w = d * (tan(theta_right) - tan(theta_left)) is the width of a fronto-parallel surface at
        # distance d from the camera PLANE, so it takes the forward distance. Feeding it the ray
        # distance overstates the width by the ray-norm factor (11 percent at a 90-degree frame
        # corner), which moves a real asset out of the accepted width window.
        phys_width = rng_forward * (right - left)
        if not (p.target_min_width_m <= phys_width <= p.target_max_width_m):
            continue
        support = float(np.mean(best_support[lo : hi + 1]))
        centre = 0.5 * (lo + hi) + 0.5
        bearing = float(np.arctan((centre - width / 2.0) / f))
        key = (round(support, 3), -rng_forward)
        if key > best_key:
            best_key = key
            best = TargetDetection(
                visible=True,
                bearing_rad=bearing,
                range_m=rng,
                width_m=float(phys_width),
                support_fraction=support,
                column_span=(int(lo), int(hi)),
                source="depth",
                camera_point_frd_m=tuple(float(v) for v in np.nanmedian(col_points[lo:hi+1], axis=0)),
            )
    return best


def detect_vertical_structure_rgb(
    rgb: np.ndarray,
    *,
    hfov_rad: float,
    params: PerceptionParams | None = None,
) -> TargetDetection:
    """Bearing-only fallback when depth is unusable but colour is not.

    A vertical structure shows up as a run of columns that differ from the per-row background in a
    consistent direction over many rows. This yields a bearing and no range at all, which is exactly
    what should be reported: the vehicle may align, it may not servo distance.
    """
    p = params or PerceptionParams()
    if rgb.ndim == 3:
        lum = rgb[..., :3].astype(np.float64).mean(axis=2)
    elif rgb.ndim == 2:
        lum = rgb.astype(np.float64)
    else:
        raise ValueError(f"rgb must be 2-D or 3-D, got shape {rgb.shape}")
    height, width = lum.shape
    if height == 0 or width == 0:
        return TargetDetection(visible=False, source="rgb")

    background = np.median(lum, axis=1, keepdims=True)
    deviation = lum - background
    dark = (deviation < -p.rgb_min_contrast).mean(axis=0)
    bright = (deviation > p.rgb_min_contrast).mean(axis=0)
    support = np.maximum(dark, bright)
    candidate = support >= p.rgb_min_height_fraction
    if not candidate.any():
        return TargetDetection(visible=False, source="rgb")

    f = focal_length_px(width, hfov_rad)
    best: TargetDetection = TargetDetection(visible=False, source="rgb")
    best_support = -1.0
    lo: int | None = None
    for u in range(width + 1):
        inside = bool(candidate[u]) if u < width else False
        if inside and lo is None:
            lo = u
        elif not inside and lo is not None:
            hi = u - 1
            span = hi - lo + 1
            if span >= 2 and span <= p.rgb_max_width_fraction * width:
                mean_support = float(np.mean(support[lo : hi + 1]))
                if mean_support > best_support:
                    best_support = mean_support
                    centre = 0.5 * (lo + hi) + 0.5
                    best = TargetDetection(
                        visible=True,
                        bearing_rad=float(np.arctan((centre - width / 2.0) / f)),
                        range_m=None,
                        support_fraction=mean_support,
                        column_span=(int(lo), int(hi)),
                        source="rgb",
                    )
            lo = None
    return best


# ----------------------------------------------------------------------------------------------
# Frame summary
# ----------------------------------------------------------------------------------------------
def summarize_depth(
    depth: np.ndarray | None,
    *,
    sim_time_s: float,
    camera_name: str,
    hfov_rad: float,
    frame_ref: FrameRef | None = None,
    rgb: np.ndarray | None = None,
    n_sectors: int = 7,
    max_range_m: float = 60.0,
    degraded_reason: str | None = None,
    params: PerceptionParams | None = None,
    camera_mount: CameraMount | None = None,
    onboard_orientation_wxyz: tuple[float, float, float, float] | None = None,
    onboard_yaw_rad: float = 0.0,
    attitude_sim_time_s: float | None = None,
    mount_evidence_sha256: str | None = None,
) -> DepthSummary:
    """Summarise one depth frame (metres, HxW, NaN/inf allowed) into a :class:`DepthSummary`.

    ``depth=None`` means no frame arrived this step. ``degraded_reason`` lets the caller pass in the
    reason it already knows (for example a scheduled sensor dropout); it is preserved, because the
    guards distinguish "the frame was dropped" from "the frame was unreadable".

    Semantics that the guards rely on:

    * ``valid=False``  -> no range information; the reason says why.
    * ``valid=True`` and ``min_range_m is None`` -> nothing within ``max_range_m``, positively clear.
    * ``sector_min_range_m`` always has ``n_sectors`` entries; an empty sector reports ``max_range_m``
      (the sensing horizon), never a fabricated near value.

    Units: ``depth`` is a DepthPerspective frame (ray distance per pixel). ``min_range_m``,
    ``free_path_m``, ``sector_min_range_m`` and ``target_range_m`` are ray distances; the corridor
    and sector gates use the forward/lateral/vertical offsets from :func:`ray_offsets`. The module
    docstring holds the full table and the reason for that split.
    """
    p = params or PerceptionParams()
    if camera_mount is not None and onboard_orientation_wxyz is None:
        raise ValueError("calibrated camera projection requires ONBOARD full attitude, never camera truth")
    if depth is None:
        summary = DepthSummary(
            sim_time_s=sim_time_s,
            camera_name=camera_name,
            valid=False,
            coverage_fraction=0.0,
            frame=frame_ref,
            degraded_reason=degraded_reason or "depth_frame_missing",
        )
        if rgb is not None:
            hit = detect_vertical_structure_rgb(rgb, hfov_rad=hfov_rad, params=p)
            if hit.visible:
                summary.target_visible = True
                summary.target_bearing_rad = hit.bearing_rad
                if camera_mount is not None:
                    direction = camera_to_vehicle_level(
                        np.array([math.cos(hit.bearing_rad), math.sin(hit.bearing_rad), 0.0]),
                        mount=camera_mount, onboard_orientation_wxyz=onboard_orientation_wxyz,
                        onboard_yaw_rad=onboard_yaw_rad, directions_only=True)
                    summary.target_bearing_rad = math.atan2(direction[1], direction[0])
                summary.degraded_reason = (summary.degraded_reason or "") + "+rgb_bearing_only"
        return summary

    arr = np.asarray(depth, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"depth must be a 2-D array of metres, got shape {arr.shape}")
    height, width = arr.shape
    if height == 0 or width == 0:
        return DepthSummary(
            sim_time_s=sim_time_s,
            camera_name=camera_name,
            valid=False,
            coverage_fraction=0.0,
            frame=frame_ref,
            degraded_reason=degraded_reason or "depth_frame_empty",
        )
    if n_sectors < 1:
        raise ValueError("n_sectors must be >= 1")

    finite = np.isfinite(arr) & (arr > 0.0)
    coverage = float(finite.mean())
    if coverage <= 0.0:
        return DepthSummary(
            sim_time_s=sim_time_s,
            camera_name=camera_name,
            valid=False,
            coverage_fraction=0.0,
            frame=frame_ref,
            degraded_reason=degraded_reason or "no_finite_depth_pixels",
        )
    if coverage < p.min_coverage_fraction:
        return DepthSummary(
            sim_time_s=sim_time_s,
            camera_name=camera_name,
            valid=False,
            coverage_fraction=coverage,
            frame=frame_ref,
            degraded_reason=degraded_reason or f"low_depth_coverage:{coverage:.3f}",
        )

    if camera_mount is not None:
        return _summarize_calibrated(
            arr, finite=finite, coverage=coverage, sim_time_s=sim_time_s, camera_name=camera_name,
            hfov_rad=hfov_rad, frame_ref=frame_ref, n_sectors=n_sectors, max_range_m=max_range_m,
            degraded_reason=degraded_reason, params=p, mount=camera_mount,
            orientation=onboard_orientation_wxyz, yaw=onboard_yaw_rad,
            attitude_sim_time_s=attitude_sim_time_s, mount_evidence_sha256=mount_evidence_sha256,
        )

    # The sensing horizon is a ray distance, because that is what the sensor measured. The corridor
    # and the sector height gate are rectangular volumes around the camera axis, so they need the
    # projected offsets instead: one full-ray projection, used by both gates.
    in_range = finite & (arr <= max_range_m)
    bearings = column_bearings_rad(width, hfov_rad)
    offsets = ray_offsets(arr, hfov_rad=hfov_rad)
    lateral = offsets.lateral
    vertical = offsets.vertical

    # Per-sector ranges, with the ground plane and the sky gated out vertically.
    sector_mask = in_range & (np.abs(vertical) <= p.sector_half_height_m)
    sector_ranges: list[float] = []
    sector_centre_bearings: list[float] = []
    for columns in np.array_split(np.arange(width), n_sectors):
        sel = sector_mask[:, columns]
        values = arr[:, columns][sel]
        if values.size < p.min_pixels_per_sector:
            sector_ranges.append(float(max_range_m))
        else:
            sector_ranges.append(min(_robust_min(values, p.robust_quantile, max_range_m), max_range_m))
        sector_centre_bearings.append(float(bearings[columns[len(columns) // 2]]))

    detected = [r for r in sector_ranges if r < max_range_m - 1e-9]
    min_range = float(min(detected)) if detected else None
    obstacle_bearing: float | None = None
    if min_range is not None:
        # Bearing of the nearest surface, taken from the pixels that actually produced the minimum
        # rather than from the centre of the winning sector: a wide obstacle spanning several sectors
        # would otherwise be reported at the edge of the first one.
        tol = max(0.5, 0.05 * min_range)
        nearest = sector_mask & (arr <= min_range + tol)
        if nearest.any():
            obstacle_bearing = float(np.median(np.broadcast_to(bearings[None, :], arr.shape)[nearest]))
        else:
            obstacle_bearing = sector_centre_bearings[int(np.argmin(sector_ranges))]

    corridor = (
        in_range
        & (np.abs(lateral) <= p.corridor_half_width_m)
        & (np.abs(vertical) <= p.corridor_half_height_m)
    )
    corridor_values = arr[corridor]
    free_path = (
        min(_robust_min(corridor_values, p.robust_quantile, max_range_m), max_range_m)
        if corridor_values.size >= p.min_pixels_per_sector
        else float(max_range_m)
    )

    target = detect_vertical_structure(arr, hfov_rad=hfov_rad, max_range_m=max_range_m, params=p)

    return DepthSummary(
        sim_time_s=sim_time_s,
        camera_name=camera_name,
        valid=True,
        min_range_m=min_range,
        free_path_m=float(free_path),
        sector_min_range_m=[float(r) for r in sector_ranges],
        obstacle_bearing_rad=obstacle_bearing,
        target_visible=target.visible,
        target_bearing_rad=target.bearing_rad,
        target_range_m=target.range_m,
        coverage_fraction=coverage,
        frame=frame_ref,
        degraded_reason=degraded_reason,
    )


def _summarize_calibrated(
    arr: np.ndarray, *, finite: np.ndarray, coverage: float, sim_time_s: float,
    camera_name: str, hfov_rad: float, frame_ref: FrameRef | None, n_sectors: int,
    max_range_m: float, degraded_reason: str | None, params: PerceptionParams,
    mount: CameraMount, orientation: object, yaw: float, attitude_sim_time_s: float | None,
    mount_evidence_sha256: str | None,
) -> DepthSummary:
    """Sensor-only projection for calibrated live runs; legacy fixture calculations stay readable.

    All reported ranges here are vehicle-center 3D distances, except the explicitly named target
    horizontal distance used for XY reconstruction. The sensor horizon still applies to the raw
    camera ray. A sector without observed rays is UNKNOWN, never synthesized as clear space.
    """
    p = params
    offsets = ray_offsets(arr, hfov_rad=hfov_rad)
    points = camera_to_vehicle_level(
        np.stack((offsets.forward, offsets.lateral, -offsets.vertical), axis=-1),
        mount=mount, onboard_orientation_wxyz=orientation, onboard_yaw_rad=yaw)
    forward, lateral, down = (points[..., axis] for axis in range(3))
    ranges = np.linalg.norm(points, axis=-1)
    bearings = np.arctan2(lateral, forward)
    # Check the visible vertical band at the sensing horizon, independently of whether the rays
    # hit a nearby object or distant background. Counting azimuths alone falsely declares a clear
    # vehicle-level corridor when a pitched camera sees only sky or only ground.
    horizon = ray_offsets(np.full(arr.shape, max_range_m), hfov_rad=hfov_rad)
    horizon_points = camera_to_vehicle_level(
        np.stack((horizon.forward, horizon.lateral, -horizon.vertical), axis=-1),
        mount=mount, onboard_orientation_wxyz=orientation, onboard_yaw_rad=yaw)
    horizon_bearings = np.arctan2(horizon_points[..., 1], horizon_points[..., 0])
    in_range = finite & (arr <= max_range_m)
    sector_mask = in_range & (forward > 0) & (np.abs(down) <= p.sector_half_height_m)
    width = arr.shape[1]
    f = focal_length_px(width, hfov_rad)
    groups = np.array_split(np.arange(width), n_sectors)
    if any(len(columns) == 0 for columns in groups):
        raise ValueError("n_sectors cannot exceed image width")
    edges = [math.atan((columns[0]-width/2)/f) for columns in groups]
    edges.append(hfov_rad/2)
    sector_ranges, unobserved = [], []
    for i in range(n_sectors):
        angular = (bearings >= edges[i]) & (bearings < edges[i+1]) & (forward > 0)
        view = (finite & (horizon_bearings >= edges[i]) & (horizon_bearings < edges[i+1])
                & (horizon_points[..., 0] > 0))
        down_view = horizon_points[..., 2][view]
        vertical_covered = (down_view.size >= p.min_pixels_per_sector
                            and float(down_view.min()) <= -p.sector_half_height_m
                            and float(down_view.max()) >= p.sector_half_height_m)
        if np.count_nonzero(finite & angular) < p.min_pixels_per_sector or not vertical_covered:
            # A changed mount or attitude can put this vehicle-centered sector outside the image.
            # Zero is conservative for the controller; valid=False tells guards it is unknown.
            sector_ranges.append(0.0)
            unobserved.append(i)
            continue
        values = ranges[sector_mask & angular]
        sector_ranges.append(min(_robust_min(values, p.robust_quantile, max_range_m), max_range_m)
                             if values.size >= p.min_pixels_per_sector else float(max_range_m))
    detected = [value for value in sector_ranges if value < max_range_m-1e-9]
    min_range = min(detected) if detected else None
    nearest = (sector_mask & (ranges <= min_range+max(.5, .05*min_range))
               if min_range is not None else np.zeros(arr.shape, dtype=bool))
    obstacle_bearing = float(np.median(bearings[nearest])) if nearest.any() else None
    corridor = (in_range & (forward > 0) & (np.abs(lateral) <= p.corridor_half_width_m)
                & (np.abs(down) <= p.corridor_half_height_m))
    values = ranges[corridor]
    free_path = min(_robust_min(values, p.robust_quantile, max_range_m), max_range_m)
    target = detect_vertical_structure(arr, hfov_rad=hfov_rad, max_range_m=max_range_m, params=p)
    target_bearing = target_range = target_horizontal = None
    if target.visible and target.camera_point_frd_m is not None:
        point = camera_to_vehicle_level(
            np.asarray(target.camera_point_frd_m), mount=mount,
            onboard_orientation_wxyz=orientation, onboard_yaw_rad=yaw)
        target_bearing = math.atan2(point[1], point[0])
        target_range = float(np.linalg.norm(point))
        target_horizontal = float(np.linalg.norm(point[:2]))
    if unobserved:
        degraded_reason = (degraded_reason+";" if degraded_reason else "") + "camera_sector_unobserved"
    return DepthSummary(
        sim_time_s=sim_time_s, camera_name=camera_name, valid=not unobserved,
        min_range_m=min_range, free_path_m=0.0 if unobserved else free_path,
        sector_min_range_m=sector_ranges, obstacle_bearing_rad=obstacle_bearing,
        target_visible=target.visible and target_bearing is not None,
        target_bearing_rad=target_bearing, target_range_m=target_range,
        target_horizontal_range_m=target_horizontal, coverage_fraction=coverage,
        frame=frame_ref, degraded_reason=degraded_reason,
        geometry_source="vehicle_center_from_calibrated_mount_and_onboard_attitude",
        projection_attitude_sim_time_s=attitude_sim_time_s,
        camera_mount_evidence_sha256=mount_evidence_sha256,
    )
