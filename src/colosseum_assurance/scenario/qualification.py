"""Offline, authored-geometry necessary-condition checks, not live competence evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.scenario.manifest import ScenarioManifest
from colosseum_assurance.schemas import Vec3


def inspection_geometry_qualification(protocol: ProtocolConfig, manifest: ScenarioManifest) -> dict:
    """Exhibit one permitted standoff point or refuse; do not claim a collision-free route exists.

    This finite witness search is deliberately one-sided: failure may be an unresolved search, but
    it may never authorize a known-disjoint inspection shell. A witness establishes static geometric
    compatibility only. Pixel quality, controller competence and native world bounds require live QA.
    """
    mission, obligations = protocol.mission, protocol.obligations
    assets = [o for o in manifest.obstacles if o.kind == "inspection_asset"]
    failures = []
    if len(assets) != 1:
        return {"passed": False, "failures": ["exactly_one_inspection_asset_required"], "witness": None}
    asset = assets[0]
    fence = obligations.geofence
    tolerance = obligations.geofence_tolerance_m
    nearest = Vec3(x=min(max(asset.center.x, fence.x_min-tolerance), fence.x_max+tolerance),
                   y=min(max(asset.center.y, fence.y_min-tolerance), fence.y_max+tolerance),
                   z=min(max(asset.center.z, fence.z_min-tolerance), fence.z_max+tolerance))
    minimum_surface_distance = max(0., asset.surface_distance(nearest))
    lower = max(0., mission.inspection_standoff_m-mission.inspection_tolerance_m)
    upper = mission.inspection_standoff_m+mission.inspection_tolerance_m
    if minimum_surface_distance > upper:
        failures.append("permitted_region_and_inspection_shell_disjoint")
    for name, point in (("home", mission.home), ("start", manifest.start_position)):
        if fence.signed_exceedance(point) > tolerance:
            failures.append(f"{name}_outside_permitted_region")
    candidates = [manifest.inspection_viewpoint]
    # Sample rays around the measured asset at the declared inspection altitude. Use the AABB
    # support function to put each candidate exactly standoff beyond its nearest radial surface.
    for index in range(360):
        angle = 2*math.pi*index/360
        dx, dy = math.cos(angle), math.sin(angle)
        extent = min(asset.extent.x/max(abs(dx), 1e-12), asset.extent.y/max(abs(dy), 1e-12))
        candidates.append(Vec3(x=asset.center.x+(extent+mission.inspection_standoff_m)*dx,
                               y=asset.center.y+(extent+mission.inspection_standoff_m)*dy,
                               z=manifest.asset_position.z))
    witness = None
    for point in candidates:
        distance = asset.surface_distance(point)
        if (fence.signed_exceedance(point) <= tolerance and lower <= distance <= upper
                and point.distance_to(manifest.asset_position) <= obligations.authorized_inspection_radius_m
                and all(o.surface_distance(point) > obligations.min_obstacle_clearance_m
                        for o in manifest.obstacles if o.name != asset.name)):
            witness = point.model_dump(mode="json")
            break
    if witness is None:
        failures.append("no_static_unobstructed_authorized_inspection_witness_found")
    return {"qualification_version": "authored_inspection_geometry_v1", "passed": not failures,
            "failures": failures, "witness": witness,
            "minimum_asset_surface_distance_from_permitted_region_m": minimum_surface_distance,
            "inspection_surface_distance_range_m": [lower, upper],
            "manifest_hash": manifest.content_hash(), "protocol_hash": protocol.content_hash(),
            "limitations": ["necessary static geometry check, not controller competence",
                            "does not prove route connectivity, native obstacle coverage or image quality",
                            "failure of finite witness search can require independent geometry review"]}


def arm_order_plan(protocol: ProtocolConfig, manifests: list[ScenarioManifest]) -> list[dict]:
    """Cyclic rotations then reversed rotations balance arm positions within each stratum.

    Incomplete blocks are explicit and not described as balanced. No outcome affects ordering.
    """
    counts: Counter = Counter()
    arms = protocol.arms.arm_ids
    permutations = list(dict.fromkeys(
        tuple(sequence[offset:]+sequence[:offset])
        for sequence in (arms, list(reversed(arms))) for offset in range(len(arms))))
    plan = []
    spec = protocol.controlled_study
    for manifest in manifests:
        stratum = (manifest.cell_id, manifest.layout_variant, manifest.visibility)
        index = counts[stratum]
        counts[stratum] += 1
        if spec is None:
            order = protocol.arms.arm_ids
        else:
            seed = json.dumps([spec.order_seed, *stratum], separators=(",", ":")).encode()
            # Rotate arm labels, not the row sequence: every first n rows remains position-balanced.
            offset = int.from_bytes(hashlib.sha256(seed).digest()[:8], "big") % len(arms)
            labels = dict(zip(arms, arms[offset:]+arms[:offset], strict=True))
            order = [labels[arm] for arm in permutations[index % len(permutations)]]
        plan.append({"scenario_id": manifest.scenario_id, "manifest_hash": manifest.content_hash(),
                     "stratum": list(stratum), "within_stratum_index": index, "arms": order,
                     "permutation_block_size": len(permutations),
                     "method": "legacy_fixed_order" if spec is None else spec.arm_order})
    return plan
