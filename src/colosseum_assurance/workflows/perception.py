"""Freeze scenario groups before image capture, then fit and evaluate the visual model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.scenario.expanded import build_expanded_manifest, expanded_protocol


def prepare_perception_protocol(*, run_class: str, realizations: int,
                                horizon_s: float = 120.0) -> ProtocolConfig:
    from colosseum_assurance.perception_eval import assign_scenario_splits

    if realizations < 6:
        raise ValueError("at least six scenario realizations are required for domain and split coverage")
    protocol = expanded_protocol("degraded_perception", 0.5, horizon_s=horizon_s)
    groups = [f"{run_class}:environment-{build_expanded_manifest(protocol, run_class, i).seed}"
              for i in range(realizations)]
    splits = assign_scenario_splits(groups)  # before a connection, image, label or fit exists
    data = protocol.model_dump(mode="json")
    data["study_extension"].update(capture_segmentation=True, perception_split_by_group=splits)
    data["sampling"]["pilot_realizations_per_cell"] = realizations
    data["simulation"]["save_frames_every_n_steps"] = 1
    return ProtocolConfig.model_validate(data)


def perception_smoke(*, out: Path, realizations: int = 18, horizon_s: float = 45.0) -> dict[str, Any]:
    from colosseum_assurance.perception_eval import evaluate_dataset, fit_dataset
    from colosseum_assurance.workflows.expanded import run_family_fixture

    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    protocol = prepare_perception_protocol(run_class="fixture", realizations=realizations,
                                           horizon_s=horizon_s)
    result = run_family_fixture(protocol, results_root=out, realizations=realizations,
                                arms=["A0_unguarded"])
    run_dir = Path(result["run_dir"])
    rows = run_dir / "perception" / "rows.jsonl"
    model_path = out / "fixture-model.json"
    model = fit_dataset(rows, model_path)
    report = evaluate_dataset(rows, model_path, out / "vision-evaluation")
    summary = {"experimental_evidence": False, "run_dir": str(run_dir),
               "model_path": str(model_path), "model_hash": model.model_hash,
               "split_groups": protocol.study_extension.perception_split_by_group,
               "evaluation_path": str(out / "vision-evaluation"),
               "overall": report["overall"], "note": "Fixture fit/evaluation verifies software only."}
    (out / "perception-smoke-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
