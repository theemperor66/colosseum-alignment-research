"""Protocol freezing, tamper detection, pilot-derived sample sizing, and workload estimation.

Freezing writes the protocol to JSON together with its content hash and the code version that produced
it. Loading recomputes the hash, so an edited frozen file is detected instead of silently used. Any change
after freezing must be appended as an explicit deviation, which keeps the record honest rather than tidy.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.version import code_version


class ProtocolIntegrityError(RuntimeError):
    """Raised when a frozen protocol file does not match its recorded hash."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def freeze_protocol(
    protocol: ProtocolConfig,
    label: str,
    out_dir: Path,
    rationale: str = "",
) -> Path:
    """Write a frozen protocol file and return its path.

    The file name carries the label and the short content hash, so two frozen protocols cannot
    overwrite each other.
    """
    frozen = protocol.model_copy(update={"protocol_label": label})
    content_hash = frozen.content_hash()
    payload: dict[str, Any] = {
        "frozen_at_utc": _now(),
        "label": label,
        "protocol_hash": content_hash,
        "protocol_short_hash": content_hash.split(":", 1)[1][:12],
        "code_version": code_version(),
        "rationale": rationale,
        "deviations": [],
        "protocol": frozen.model_dump(mode="json"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"protocol-{label}-{payload['protocol_short_hash']}.json"
    if path.exists():
        existing, _ = load_frozen(path)
        if existing.content_hash() != content_hash:
            raise ProtocolIntegrityError("existing freeze does not match requested protocol")
        return path  # Preserve original freeze time, evidence and deviation ledger.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_frozen(path: str | Path) -> tuple[ProtocolConfig, dict[str, Any]]:
    """Load a frozen protocol and verify its hash. Returns the protocol and the envelope metadata."""
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if "protocol" not in payload or "protocol_hash" not in payload:
        raise ProtocolIntegrityError(f"{p} is not a frozen protocol envelope")
    protocol = ProtocolConfig.model_validate(payload["protocol"])
    recomputed = protocol.content_hash()
    if recomputed != payload["protocol_hash"]:
        raise ProtocolIntegrityError(
            f"{p} was modified after freezing: recorded {payload['protocol_hash']}, recomputed {recomputed}"
        )
    return protocol, {k: v for k, v in payload.items() if k != "protocol"}


def append_deviation(path: str | Path, description: str, rationale: str, applies_from: str = "") -> None:
    """Record a post-freeze deviation in the frozen file without changing the protocol itself."""
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    payload.setdefault("deviations", []).append(
        {
            "recorded_at_utc": _now(),
            "description": description,
            "rationale": rationale,
            "applies_from": applies_from,
            "code_version": code_version(),
        }
    )
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------
# Pilot-derived sizing and workload
# --------------------------------------------------------------------------------------
def paired_sample_size(
    discordant_b: int,
    discordant_c: int,
    n_pairs: int,
    target_half_width: float,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Sample size for a paired difference of proportions, from observed pilot discordance.

    Uses the large-sample variance of the paired difference
    ``var = (p_b + p_c - (p_b - p_c)^2) / n``, so the required number of pairs for a two-sided interval
    half-width ``w`` is ``n = z^2 (p_b + p_c - (p_b - p_c)^2) / w^2``.

    This is a planning calculation from a small pilot, not a power guarantee. With few discordant pairs the
    estimate is very uncertain, which the returned ``caveat`` states explicitly.
    """
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    z = _z_for(confidence_level)
    p_b = discordant_b / n_pairs
    p_c = discordant_c / n_pairs
    diff = p_b - p_c
    variance_term = max(p_b + p_c - diff * diff, 0.0)
    if variance_term == 0.0:
        # No discordant pairs observed: fall back to a conservative single-pair prior so the
        # recommendation is finite and visibly conservative.
        variance_term = 1.0 / max(n_pairs, 1)
        caveat = (
            "no discordant pairs were observed in the pilot; the variance term was replaced by 1/n_pilot as "
            "a conservative placeholder and the result is a lower bound on what is needed"
        )
    else:
        caveat = (
            f"based on {discordant_b + discordant_c} discordant pairs out of {n_pairs}; the estimate is "
            "unstable with few discordant pairs"
        )
    required = math.ceil(z * z * variance_term / (target_half_width**2))
    return {
        "method": "large_sample_paired_difference_half_width",
        "confidence_level": confidence_level,
        "target_half_width": target_half_width,
        "pilot_pairs": n_pairs,
        "discordant_b": discordant_b,
        "discordant_c": discordant_c,
        "p_discordant_b": round(p_b, 6),
        "p_discordant_c": round(p_c, 6),
        "variance_term": round(variance_term, 6),
        "required_pairs_total": required,
        "caveat": caveat,
    }


def _z_for(confidence_level: float) -> float:
    """Two-sided normal quantile without scipy (Acklam-style rational approximation)."""
    p = 1.0 - (1.0 - confidence_level) / 2.0
    if not 0.0 < p < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    # Inverse standard normal CDF, absolute error < 4.5e-4 (Peter Acklam, public domain algorithm).
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def estimate_workload(
    protocol: ProtocolConfig,
    run_class: str,
    measured_episode_wall_clock_s: float,
    parallel_capacity: int = 1,
    reset_overhead_s: float = 0.0,
    failure_rate: float = 0.0,
) -> dict[str, Any]:
    """Replace a guessed run window with measured episode time times the frozen run count.

    ``failure_rate`` inflates the episode count for expected reruns. The result is a schedule estimate,
    not a promise: it assumes the measured episode cost is representative of the held-out conditions.
    """
    if parallel_capacity < 1:
        raise ValueError("parallel_capacity must be at least 1")
    if not 0.0 <= failure_rate < 1.0:
        raise ValueError("failure_rate must be in [0, 1)")
    episodes = protocol.episode_budget(run_class)
    effective = episodes / (1.0 - failure_rate)
    per_episode = measured_episode_wall_clock_s + reset_overhead_s
    serial_s = effective * per_episode
    wall_s = serial_s / parallel_capacity
    return {
        "run_class": run_class,
        "episodes_planned": episodes,
        "episodes_with_reruns": round(effective, 1),
        "measured_episode_wall_clock_s": measured_episode_wall_clock_s,
        "reset_overhead_s": reset_overhead_s,
        "per_episode_cost_s": round(per_episode, 3),
        "parallel_capacity": parallel_capacity,
        "serial_compute_hours": round(serial_s / 3600.0, 3),
        "estimated_wall_clock_hours": round(wall_s / 3600.0, 3),
        "assumption": (
            "measured episode cost is representative; parallel capacity is demonstrated, not assumed; "
            "analysis and inspection time are not included"
        ),
    }
