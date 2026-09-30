"""Strict codecs for the pinned Colosseum IMU/GPS RPC contracts.

No state estimate is substituted if a sensor is absent. GPS EPH/EPV are dilution
values in the pinned source, not error bounds in metres.
"""
from __future__ import annotations

import math
from typing import Any


def structure(value: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    if isinstance(value, dict) and all(k in value for k in keys):
        return {k: value[k] for k in keys}
    if isinstance(value, (list, tuple)) and len(value) == len(keys):
        return dict(zip(keys, value, strict=True))
    raise ValueError(f"malformed sensor structure; expected {keys}")


def finite(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean sensor measurement")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite sensor measurement")
    return result


def vector(value: Any) -> dict[str, float]:
    return {k: finite(v) for k, v in structure(value, ("x_val", "y_val", "z_val")).items()}


def timestamp(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("sensor acquisition timestamp must be positive integer nanoseconds")
    return value


def decode_imu(value: Any) -> dict[str, Any]:
    row = structure(value, ("time_stamp", "orientation", "angular_velocity", "linear_acceleration"))
    orientation = {k: finite(v) for k, v in structure(
        row["orientation"], ("w_val", "x_val", "y_val", "z_val")
    ).items()}
    if not 0.95 <= sum(v * v for v in orientation.values()) <= 1.05:
        raise ValueError("IMU orientation is not a unit quaternion")
    return {"capture_timestamp_ns": timestamp(row["time_stamp"]),
            "orientation": orientation, "angular_velocity_rad_s": vector(row["angular_velocity"]),
            "linear_acceleration_body_m_s2": vector(row["linear_acceleration"]),
            "gravity_subtracted": True, "orientation_noise_modeled": False}


def decode_gps(value: Any) -> dict[str, Any]:
    row = structure(value, ("time_stamp", "gnss", "is_valid"))
    if not isinstance(row["is_valid"], bool):
        raise ValueError("GPS is_valid must be boolean")
    gnss = structure(row["gnss"], ("geo_point", "eph", "epv", "velocity", "fix_type", "time_utc"))
    point = {k: finite(v) for k, v in structure(
        gnss["geo_point"], ("latitude", "longitude", "altitude")
    ).items()}
    if abs(point["latitude"]) > 90 or abs(point["longitude"]) > 180:
        raise ValueError("GPS latitude/longitude out of range")
    fix = gnss["fix_type"]
    if isinstance(fix, bool) or fix not in (0, 1, 2, 3):
        raise ValueError("unknown GPS fix type")
    eph, epv = finite(gnss["eph"]), finite(gnss["epv"])
    if min(eph, epv) < 0:
        raise ValueError("negative GPS dilution")
    return {"capture_timestamp_ns": timestamp(row["time_stamp"]), "geo_point": point,
            "velocity_ned_m_s": vector(gnss["velocity"]), "eph_dilution": eph, "epv_dilution": epv,
            "fix_type": fix, "is_valid": row["is_valid"],
            "has_position_fix": row["is_valid"] and fix >= 2,
            "time_utc_raw_sim_microseconds": timestamp(gnss["time_utc"])}
