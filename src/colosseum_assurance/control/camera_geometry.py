"""Rigid camera mounting and onboard-attitude geometry, with no world-truth inputs.

A mount is a fixed calibration, shared by every experimental arm. Runtime functions accept only
that calibration and the vehicle's onboard attitude. Absolute simulator camera poses belong to
qualification/QA evidence and must never be supplied to these functions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def unit_quaternion(value: object) -> np.ndarray:
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("attitude must contain four finite quaternion components (w,x,y,z)")
    norm = float(np.linalg.norm(q))
    if abs(norm - 1.0) > 0.001:
        raise ValueError("attitude quaternion must have unit norm within 0.001")
    return q / norm


def rotation_matrix(value: object) -> np.ndarray:
    """Body/camera forward-right-down to parent NED, quaternion order w,x,y,z."""
    w, x, y, z = unit_quaternion(value)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def quaternion_product(left: object, right: object) -> np.ndarray:
    w, x, y, z = unit_quaternion(left)
    a, b, c, d = unit_quaternion(right)
    return unit_quaternion([w*a-x*b-y*c-z*d, w*b+x*a+y*d-z*c,
                            w*c-x*d+y*a+z*b, w*d+x*c-y*b+z*a])


@dataclass(frozen=True, slots=True)
class CameraMount:
    """Camera origin and orientation in the vehicle's rigid body forward-right-down frame."""

    position_body_m: tuple[float, float, float]
    orientation_body_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if len(self.position_body_m) != 3 or not np.all(np.isfinite(self.position_body_m)):
            raise ValueError("camera mounting position must contain three finite metres")
        unit_quaternion(self.orientation_body_wxyz)

    def to_dict(self) -> dict[str, list[float]]:
        return {"position_body_m": list(self.position_body_m),
                "orientation_body_wxyz": list(self.orientation_body_wxyz)}


def camera_to_vehicle_level(
    points_camera_frd_m: np.ndarray, *, mount: CameraMount,
    onboard_orientation_wxyz: object, onboard_yaw_rad: float,
    directions_only: bool = False,
) -> np.ndarray:
    """Transform camera points to a gravity-level frame centered on the vehicle, yaw removed.

    Adding the *onboard* yaw later recovers world heading, without exposing absolute camera truth.
    Directions have no range, so a camera translation cannot honestly parallax-correct them.
    """
    points = np.asarray(points_camera_frd_m, dtype=float)
    if points.shape[-1] != 3 or not math.isfinite(onboard_yaw_rad):
        raise ValueError("camera points require xyz and a finite onboard yaw")
    body = points @ rotation_matrix(mount.orientation_body_wxyz).T
    if not directions_only:
        body = body + np.asarray(mount.position_body_m)
    world = body @ rotation_matrix(onboard_orientation_wxyz).T
    c, s = math.cos(onboard_yaw_rad), math.sin(onboard_yaw_rad)
    world_to_level = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    return world @ world_to_level.T
