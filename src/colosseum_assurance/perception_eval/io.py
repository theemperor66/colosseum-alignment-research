"""File entrypoints and reviewable artifacts; never overwrite an existing research artifact."""

from __future__ import annotations

import json
from pathlib import Path

from colosseum_assurance.perception_eval.contracts import PerceptionRow, content_hash, validate_rows
from colosseum_assurance.perception_eval.metrics import evaluate_model
from colosseum_assurance.perception_eval.model import ModelArtifact, fit_model


def load_rows(path: str | Path) -> list[PerceptionRow]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(PerceptionRow.model_validate_json(line))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid perception JSONL row {number}: {exc}") from exc
    validate_rows(rows)
    return rows


def write_rows(path: str | Path, rows: list[PerceptionRow]) -> None:
    validate_rows(rows)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def save_model(path: str | Path, model: ModelArtifact) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump({"model_hash": model.model_hash, "artifact": model.model_dump(mode="json")}, handle,
                  indent=2, allow_nan=False, sort_keys=True)
        handle.write("\n")


def load_model(path: str | Path) -> ModelArtifact:
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(envelope, dict) or set(envelope) != {"model_hash", "artifact"}:
        raise ValueError("model file requires exactly the hash and artifact envelope")
    model = ModelArtifact.model_validate_json(json.dumps(envelope["artifact"], allow_nan=False))
    if model.model_hash != envelope["model_hash"]:
        raise ValueError("model hash mismatch: frozen artifact was modified")
    return model


def fit_dataset(dataset_path: str | Path, model_path: str | Path, **fit_options) -> ModelArtifact:
    if Path(model_path).exists():
        raise FileExistsError(model_path)
    rows = load_rows(dataset_path)
    inventory = _group_inventory(dataset_path, rows)
    model = fit_model(rows, **fit_options)
    if inventory is not None:
        data = model.model_dump(mode="json")
        data.update(dataset_scope="planned_group_inventory", inventory_hash=content_hash(inventory))
        model = ModelArtifact.model_validate_json(json.dumps(data, allow_nan=False))
    save_model(model_path, model)
    return model


def _group_inventory(dataset_path: str | Path, rows: list[PerceptionRow]) -> dict | None:
    """Check planned groups before selection can disappear into the fitted row table.

    The manifest is frozen before collection by the capture workflow. Its absence explicitly limits
    scope to the declared rows; it is never inferred from those same rows and called independent.
    """
    source = Path(dataset_path).with_name("group-inventory.json")
    if not source.exists():
        return None
    inventory = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(inventory, dict) or not {"groups", "protocol_hash", "run_class"} <= set(inventory):
        raise ValueError("group inventory requires groups, protocol_hash and run_class")
    groups = inventory["groups"]
    if (not isinstance(groups, dict) or not groups
            or any(not isinstance(sid, str) or not sid or split not in ("train", "calibration", "test")
                   for sid, split in groups.items())):
        raise ValueError("group inventory must map nonempty scenario IDs to train/calibration/test")
    if any(not isinstance(inventory[key], str) or not inventory[key]
           for key in ("protocol_hash", "run_class")):
        raise ValueError("group inventory protocol_hash and run_class must be nonempty strings")
    recorded = {row.scenario_id: row.split for row in rows}
    missing = sorted(set(groups) - set(recorded))
    if missing:
        raise ValueError(f"missing planned scenario groups: {missing}; retain missing-capture rows")
    extra = sorted(set(recorded) - set(groups))
    if extra:
        raise ValueError(f"unplanned scenario groups in dataset: {extra}")
    changed = sorted(sid for sid in groups if groups[sid] != recorded[sid])
    if changed:
        raise ValueError(f"scenario split differs from frozen group inventory: {changed}")
    fixture = all(row.provenance == "fixture_fake" for row in rows)
    if fixture != (inventory["run_class"] == "fixture"):
        raise ValueError("group inventory run class conflicts with fixture/live row provenance")
    return inventory


def _markdown(report: dict) -> str:
    counts, rates, scores = (report["overall"][key] for key in ("counts", "rates", "scores"))
    banner = "FIXTURE ONLY — no live validation" if report["evidence_class"] == "fixture_only" else \
        "Held-out live-data evaluation — limited to the recorded build and label policy"
    lines = ["# Civilian visual asset-presence evaluation", "", f"**{banner}**", "", report["claim"] + ".",
             "", f"Model: `{report['model_hash']}`", f"Label policy: `{report['label_spec_hash']}`", "",
             f"Scored {counts['scored_frames']} of {counts['eligible_frames']} eligible frames across "
             f"{report['overall']['scenario_count']} test scenarios. Missing predictions: "
             f"{counts['eligible_frames'] - counts['predicted_frames']}; unknown labels: "
             f"{counts['eligible_frames'] - counts['labelled_frames']}.", "",
             "| Metric | Estimate | Numerator / denominator |", "| --- | ---: | ---: |"]
    for name, rate in rates.items():
        value = "undefined" if rate["value"] is None else f"{rate['value']:.6g}"
        lines.append(f"| {name} | {value} | {rate['numerator']} / {rate['denominator']} |")
    for name, value in scores.items():
        display = "undefined" if value is None else f"{value:.6g}"
        lines.append(f"| {name} | {display} | {counts['scored_frames']} scored frames |")
    lines += ["", "Confusion counts: " + ", ".join(f"{key}={counts[key]}" for key in
              ("true_positive", "false_positive", "false_negative", "true_negative")) + ".", "",
              "Intervals, reliability-bin counts, per-stratum results and a training-prior baseline are in "
              "`evaluation.json`. Intervals resample whole scenarios and preserve all arms and frames. "
              "An interval is unavailable if any stratum has fewer than two test scenarios.", "",
              "![Reliability diagram](reliability.png)", "", "## Limits", ""]
    lines.extend("- " + limitation for limitation in report["limitations"])
    return "\n".join(lines) + "\n"


def _plot_reliability(report: dict, destination: Path) -> None:
    # The object-oriented Agg backend works on a headless Linux host without setting global pyplot state.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7, 6), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot()
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.6", label="ideal calibration")
    nonempty = [b for b in report["overall"]["reliability_bins"] if b["count"]]
    if nonempty:
        axis.plot([b["mean_probability"] for b in nonempty], [b["positive_frequency"] for b in nonempty],
                  "o-", label="held-out bins")
        for item in nonempty:
            right = item["mean_probability"] > 0.85
            high = item["positive_frequency"] > 0.85
            axis.annotate(f"n={item['count']}", (item["mean_probability"], item["positive_frequency"]),
                          xytext=(-5 if right else 4, -12 if high else 5), textcoords="offset points",
                          ha="right" if right else "left", fontsize=8)
    else:
        axis.text(0.5, 0.5, "No jointly observed labels and predictions", ha="center", wrap=True)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted asset-visibility probability",
             ylabel="Observed positive-label fraction", title="Civilian asset-presence reliability")
    if report["evidence_class"] == "fixture_only":
        figure.text(0.5, 0.51, "FIXTURE ONLY", ha="center", va="center", fontsize=32,
                    color="firebrick", alpha=0.25, rotation=25)
    axis.legend(loc="lower right")
    figure.savefig(destination, dpi=180, metadata={"Model hash": report["model_hash"]})


def evaluate_dataset(
    dataset_path: str | Path, model_path: str | Path, output_dir: str | Path, **evaluation_options,
) -> dict:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    model, rows = load_model(model_path), load_rows(dataset_path)
    inventory = _group_inventory(dataset_path, rows)
    observed_hash = content_hash(inventory) if inventory is not None else None
    if model.inventory_hash != observed_hash:
        raise ValueError("evaluation group inventory does not match the model's frozen dataset scope/hash")
    report = evaluate_model(model, rows, **evaluation_options)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "evaluation.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    (output / "evaluation.md").write_text(_markdown(report), encoding="utf-8")
    _plot_reliability(report, output / "reliability.png")
    return report
