"""Information separation between the three channels, proved by scanning the source, not by trust.

research-acceptance.md section 3 requires simulator ground truth available to the evaluator to be kept
"separate from controller and monitor inputs", and requires the evaluator to be implemented separately
from the monitor's verdict-producing functions. A comment saying so is not evidence. These tests read
the shipped source with :mod:`ast` and fail with the module, the rule, and the line that broke it.

The four rules:

R1  ``control/*`` and ``monitors/*`` must not import ``evaluation``, ``audit``, ``scenario.manifest`` or
    ``runtime.evidence``; those are the privileged and evaluation channels.
R2  ``control/*`` and ``monitors/*`` must not mention the privileged type names anywhere in their source.
    A privileged type name inside the control lane is the first symptom of privileged data reaching it.
R3  ``evaluation/*`` must not import ``monitors`` or ``control``. An evaluator that can call the
    monitor's predicates is no longer an independent assessment of the same obligations.
R4  ``audit/reconstructor.py`` must not import ``audit.reference``, ``evaluation``, or
    ``PrivilegedLedger``. The reconstruction procedure must work from retained records alone, or the
    audit lane measures the answer key instead of the records.
R5  ``evaluation.evaluator.assess_obligations`` must have no ``EpisodeRecord`` parameter, so a monitor
    verdict cannot reach an obligation verdict even by accident.
"""

from __future__ import annotations

import ast
import inspect
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

import colosseum_assurance
from colosseum_assurance.evaluation.evaluator import assess_obligations

PACKAGE = "colosseum_assurance"
PACKAGE_ROOT = Path(colosseum_assurance.__file__).resolve().parent

#: Modules the control and monitor lanes may never import (prefix match on the dotted name).
FORBIDDEN_FOR_ONLINE_LANES = (
    f"{PACKAGE}.evaluation",
    f"{PACKAGE}.audit",
    f"{PACKAGE}.scenario.manifest",
    f"{PACKAGE}.runtime.evidence",
)

#: Type names that only exist in the privileged channel or in privileged scene geometry.
PRIVILEGED_TYPE_NAMES = ("TruthSample", "TruthEvent", "PrivilegedLedger", "ObstacleSpec", "ScenarioManifest")

ONLINE_LANES = ("control", "monitors")


@dataclass(frozen=True, slots=True)
class ImportSite:
    """One import statement: the module it names, the names it binds, and where it is."""

    module: str
    names: tuple[str, ...]
    lineno: int


def _module_files(relative_dir: str) -> list[Path]:
    """Every shipped ``.py`` file in one package subdirectory, ``__init__.py`` included."""
    directory = PACKAGE_ROOT / relative_dir
    assert directory.is_dir(), f"{directory} does not exist; the separation scan would check nothing"
    return sorted(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)


def _dotted_name(path: Path) -> str:
    """``.../colosseum_assurance/monitors/base.py`` -> ``colosseum_assurance.monitors.base``."""
    relative = path.resolve().relative_to(PACKAGE_ROOT.parent)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports(path: Path) -> list[ImportSite]:
    """Every imported module in one file, with relative imports resolved to absolute dotted names."""
    tree = ast.parse(path.read_text(), filename=str(path))
    own = _dotted_name(path)
    package = own.rsplit(".", 1)[0] if (path.name != "__init__.py" and "." in own) else own
    sites: list[ImportSite] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                sites.append(ImportSite(module=alias.name, names=(alias.name,), lineno=node.lineno))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                base = package.split(".")
                trimmed = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                module = ".".join([*trimmed, module]) if module else ".".join(trimmed)
            sites.append(
                ImportSite(
                    module=module,
                    names=tuple(alias.name for alias in node.names),
                    lineno=node.lineno,
                )
            )
    return sites


def _is_forbidden(module: str, forbidden: str) -> bool:
    """True when ``module`` is the forbidden module or a submodule of it."""
    return module == forbidden or module.startswith(forbidden + ".")


def _name_sites(path: Path, name: str) -> list[int]:
    """Line numbers where ``name`` appears as a whole word in the raw source, comments included."""
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    return [index for index, line in enumerate(path.read_text().splitlines(), 1) if pattern.search(line)]


# --------------------------------------------------------------------------------------
# R1 and R2: the online lanes cannot reach the privileged or evaluation channels
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("lane", ONLINE_LANES)
def test_online_lane_does_not_import_privileged_or_evaluation_modules(lane: str):
    """R1: a controller or monitor that imports the evaluator can borrow its ground truth."""
    breaches: list[str] = []
    files = _module_files(lane)
    assert files, f"no source files found under {lane}/"
    for path in files:
        for site in _imports(path):
            for forbidden in FORBIDDEN_FOR_ONLINE_LANES:
                if _is_forbidden(site.module, forbidden):
                    breaches.append(
                        f"[R1] {_dotted_name(path)}:{site.lineno} imports {site.module!r}, which is in "
                        f"the forbidden channel {forbidden!r}"
                    )
    assert breaches == [], (
        f"{lane}/ broke the information separation required by research-acceptance.md section 3:\n"
        + "\n".join(breaches)
    )


@pytest.mark.parametrize("lane", ONLINE_LANES)
def test_online_lane_never_mentions_a_privileged_type_name(lane: str):
    """R2: the privileged type names must not appear in the source of an online lane at all."""
    breaches: list[str] = []
    for path in _module_files(lane):
        for name in PRIVILEGED_TYPE_NAMES:
            for lineno in _name_sites(path, name):
                breaches.append(
                    f"[R2] {_dotted_name(path)}:{lineno} mentions the privileged type name {name!r}; the "
                    f"{lane} lane may only see ObservationPacket-level evidence"
                )
    assert breaches == [], "privileged type names leaked into an online lane:\n" + "\n".join(breaches)


# --------------------------------------------------------------------------------------
# R3: the evaluator is implemented separately from the monitors
# --------------------------------------------------------------------------------------
def test_evaluation_does_not_import_monitors_or_control():
    """R3: reading RECORDED monitor verdicts is allowed; calling the monitor's own code is not."""
    breaches: list[str] = []
    files = _module_files("evaluation")
    assert files, "no source files found under evaluation/"
    for path in files:
        for site in _imports(path):
            for forbidden in (f"{PACKAGE}.monitors", f"{PACKAGE}.control"):
                if _is_forbidden(site.module, forbidden):
                    breaches.append(
                        f"[R3] {_dotted_name(path)}:{site.lineno} imports {site.module!r}; the evaluator "
                        "must assess the frozen obligations from independent evidence, not by reusing "
                        "the monitor's predicates"
                    )
    assert breaches == [], "the evaluator is not independent of the monitors:\n" + "\n".join(breaches)


# --------------------------------------------------------------------------------------
# R4: the reconstruction procedure cannot see its own answer key
# --------------------------------------------------------------------------------------
def test_audit_reconstructor_cannot_reach_the_reference_answers_or_truth():
    """R4: reconstruction must run on retained records only (research-acceptance.md section 5)."""
    path = PACKAGE_ROOT / "audit" / "reconstructor.py"
    assert path.is_file(), f"{path} does not exist; the audit separation scan would check nothing"
    breaches: list[str] = []
    for site in _imports(path):
        for forbidden in (f"{PACKAGE}.audit.reference", f"{PACKAGE}.evaluation"):
            if _is_forbidden(site.module, forbidden):
                breaches.append(
                    f"[R4] audit.reconstructor:{site.lineno} imports {site.module!r}, which holds the "
                    "reference answers the reconstruction is scored against"
                )
        if "PrivilegedLedger" in site.names:
            breaches.append(
                f"[R4] audit.reconstructor:{site.lineno} imports PrivilegedLedger from {site.module!r}; "
                "the reconstruction procedure must not be able to read privileged truth"
            )
    assert breaches == [], "the audit reconstruction lane can see its answer key:\n" + "\n".join(breaches)


# --------------------------------------------------------------------------------------
# R5: obligation assessment has no way to see a monitor verdict
# --------------------------------------------------------------------------------------
def test_assess_obligations_takes_no_episode_record():
    """R5: the separation is enforced by the signature, so no code path can pass a record in."""
    signature = inspect.signature(assess_obligations)
    parameters = list(signature.parameters.values())
    offending = [
        f"[R5] parameter {p.name!r} (annotation {p.annotation!r})"
        for p in parameters
        if "EpisodeRecord" in str(p.annotation) or "record" in p.name.lower()
    ]
    assert offending == [], (
        "evaluation.evaluator.assess_obligations must not accept an EpisodeRecord: obligation verdicts "
        "come from privileged truth alone.\n" + "\n".join(offending)
    )
    assert [p.name for p in parameters] == ["ledger", "manifest", "protocol"], (
        f"assess_obligations signature changed to {signature}; the boundary cases and the separation "
        "argument both depend on its three privileged-only inputs"
    )


# --------------------------------------------------------------------------------------
# Runtime check: the AST scan only sees DIRECT imports, so confirm the real import graph too
# --------------------------------------------------------------------------------------
def test_importing_the_online_lanes_does_not_pull_in_the_evaluation_or_audit_channels():
    """A transitive import through a shared helper would defeat the source scan; check ``sys.modules``."""
    program = (
        "import json, sys\n"
        "import colosseum_assurance.monitors\n"
        "import colosseum_assurance.control\n"
        "leaked = sorted(\n"
        "    name for name in sys.modules\n"
        "    if name.startswith(('colosseum_assurance.evaluation', 'colosseum_assurance.audit'))\n"
        ")\n"
        "print(json.dumps(leaked))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=120, check=False
    )
    assert completed.returncode == 0, f"probe failed: {completed.stderr}"
    leaked = completed.stdout.strip().splitlines()[-1]
    assert leaked == "[]", (
        "importing the control and monitor lanes transitively imported the privileged evaluation or "
        f"audit channel: {leaked}"
    )
