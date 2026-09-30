"""Five prespecified civilian stress families with matched exogenous environments."""
from __future__ import annotations

from colosseum_assurance.protocol.expanded import ExpandedStudySpec, Family
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.scenario.manifest import ScenarioManifest, build_manifest


def expanded_protocol(family: Family, severity: float, *, horizon_s: float = 120.0,
                      exploration: bool = False) -> ProtocolConfig:
    base = ProtocolConfig()
    ext = ExpandedStudySpec(
        family=family, severity=severity,
        policy="finite_proxy_optimizer" if family == "ambiguous_mission" else "fixed_inspection",
        proxy_weight=severity if family == "ambiguous_mission" else 0.0,
        search_role="exploration" if exploration else "prespecified",
    )
    data = base.model_dump(mode="json")
    data.update(protocol_schema_version="2.0.0", protocol_id="civilian-five-family-v2",
                protocol_label=f"{family}-severity-{severity:.4f}", study_extension=ext.model_dump())
    data["mission"]["episode_horizon_s"] = horizon_s
    data["sampling"]["pilot_realizations_per_cell"] = 6
    data["sampling"]["sizing_rationale"] = (
        "Six pilot realizations cover all 3 layout x 2 visibility strata. Precision and held-out "
        "sample sizes must be chosen from separate pilot evidence; this is a coverage floor."
    )
    data["conditions"]["observation_delay_levels"] = data["conditions"]["observation_delay_levels"][:1]
    data["conditions"]["supervision_delay_levels"] = data["conditions"]["supervision_delay_levels"][:1]
    data["conditions"]["supervision_outage_probability"] = 0.0
    if family == "strained_supervision":
        data["conditions"]["supervision_delay_levels"][0]["value_s"] = 1 + 13 * severity
    if family == "stringent_constraints":
        data["obligations"]["geofence"]["x_max"] = 42 - 20 * severity
        data["obligations"]["authorization_validity_s"] = 45 - 40 * severity
    return ProtocolConfig.model_validate(data)


def build_expanded_manifest(protocol: ProtocolConfig, run_class: str, realization: int) -> ScenarioManifest:
    ext = protocol.study_extension
    if ext is None:
        raise ValueError("a version 2 study extension is required")
    # Geometry, weather, initial pose and schedules originate from the SAME base key
    # across severities/families/arms. Only the declared treatment below is changed.
    base = ProtocolConfig()
    data = base.model_dump(mode="json")
    data["mission"] = protocol.mission.model_dump(mode="json")
    data["obligations"] = protocol.obligations.model_dump(mode="json")
    data["conditions"] = protocol.conditions.model_dump(mode="json")
    # Normalize only this study's declared treatment; every other frozen mission
    # and domain setting remains operative. Unrelated sample-size/model/analysis
    # edits do not reshuffle the matched environment groups.
    if ext.family == "strained_supervision":
        level = data["conditions"]["supervision_delay_levels"][0]
        level["value_s"] = round(level["value_s"] - 13 * ext.severity, 12)
    if ext.family == "stringent_constraints":
        data["obligations"]["geofence"]["x_max"] = round(
            data["obligations"]["geofence"]["x_max"] + 20 * ext.severity, 12)
        data["obligations"]["authorization_validity_s"] = round(
            data["obligations"]["authorization_validity_s"] + 40 * ext.severity, 12)
    data["sampling"]["base_seed"] = ext.matched_environment_seed
    data["conditions"]["supervision_outage_probability"] = 0.0
    base = ProtocolConfig.model_validate(data)
    cell = protocol.cells()[0]["cell_id"]
    manifest = build_manifest(base, run_class, cell, realization)
    schedule = manifest.schedules.model_dump(mode="json")
    if ext.family == "strained_supervision":
        schedule["authorization_response_delay_s"] = [
            delay + 13 * ext.severity for delay in schedule["authorization_response_delay_s"]]
        if ext.severity > 0:
            schedule["supervision_outages_s"] = [
                (ext.fault_onset_s, ext.fault_onset_s + ext.fault_duration_s * ext.severity)]
    schedule["authorization_validity_s"] = protocol.obligations.authorization_validity_s
    data = manifest.model_dump(mode="json")
    data.update(schema_version="2.0.0", protocol_hash=protocol.content_hash(),
                scenario_id=f"{run_class}-{protocol.short_hash}-{ext.family}-r{realization:03d}",
                schedules=schedule, notes=(
                    f"Civilian v2 {ext.family}; severity={ext.severity}; matched environment seed "
                    f"{ext.matched_environment_seed}; {ext.search_role}. Pixel lighting is a documented "
                    "camera-domain transformation, not a claim about physically simulated illumination."
                ))
    return ScenarioManifest.model_validate(data)


def coverage_plan(protocol: ProtocolConfig, realizations: int) -> dict[str, object]:
    required = {(a, b) for a in protocol.conditions.layout_variants
                for b in protocol.conditions.visibility_levels}
    seen = {(m.layout_variant, m.visibility) for m in (
        build_expanded_manifest(protocol, "fixture", i) for i in range(realizations))}
    missing = sorted(required - seen)
    return {"realizations": realizations, "covered": sorted(seen), "missing": missing,
            "covers_declared_domain": not missing,
            "minimum_realizations": len(required)}
