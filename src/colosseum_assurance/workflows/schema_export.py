"""JSON Schema export for every persisted record type.

External reviewers can validate our saved evidence without importing this package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from colosseum_assurance.evaluation.outcomes import EpisodeOutcome, ObligationOutcome
from colosseum_assurance.perception_eval import LabelEvidence, LabelSpec, ModelArtifact, PerceptionRow
from colosseum_assurance.protocol.expanded import ExpandedStudySpec
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.scenario.manifest import ScenarioManifest, ScheduleSet
from colosseum_assurance.schemas import (
    SCHEMA_VERSION,
    AttemptedRun,
    EpisodeRecord,
    MonitorReport,
    ObservationPacket,
    PrivilegedLedger,
    SimulatorIdentity,
    StepRecord,
    TruthEvent,
    TruthSample,
)
from colosseum_assurance.sim.sitl import SITLConfig

MODELS = {
    "episode_record": EpisodeRecord,
    "step_record": StepRecord,
    "observation_packet": ObservationPacket,
    "monitor_report": MonitorReport,
    "privileged_ledger": PrivilegedLedger,
    "truth_sample": TruthSample,
    "truth_event": TruthEvent,
    "attempted_run": AttemptedRun,
    "simulator_identity": SimulatorIdentity,
    "scenario_manifest": ScenarioManifest,
    "schedule_set": ScheduleSet,
    "episode_outcome": EpisodeOutcome,
    "obligation_outcome": ObligationOutcome,
    "protocol_config": ProtocolConfig,
    "expanded_study_spec": ExpandedStudySpec,
    "perception_label_spec": LabelSpec,
    "perception_label_evidence": LabelEvidence,
    "perception_row": PerceptionRow,
    "perception_model": ModelArtifact,
    "sitl_config": SITLConfig,
}


def export_schemas(out_dir: Path) -> dict[str, Any]:
    """Write one JSON Schema per record type and an index file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, model in MODELS.items():
        path = out_dir / f"{name}.schema.json"
        path.write_text(json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        written[name] = str(path)
    index = {
        "schema_version": SCHEMA_VERSION,
        "models": written,
        "channels": {
            "exposed": ["episode_record", "step_record", "observation_packet", "monitor_report"],
            "privileged": ["privileged_ledger", "truth_sample", "truth_event"],
            "evaluation": ["episode_outcome", "obligation_outcome", "perception_label_evidence",
                           "perception_row", "perception_model"],
            "configuration": ["protocol_config", "scenario_manifest", "schedule_set",
                              "expanded_study_spec", "perception_label_spec", "sitl_config"],
            "operations": ["attempted_run", "simulator_identity"],
        },
    }
    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"out_dir": str(out_dir), "written": len(written), "index": str(index_path)}
