"""Package and code-provenance version helpers."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path

PACKAGE_VERSION = "0.1.0"


def repo_root() -> Path:
    """Return the repository root (three levels above this file)."""
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def code_version() -> dict[str, str]:
    """Return git provenance for the working tree, or an explicit unknown marker.

    Every saved record carries this so results can be traced to exact code.
    """
    out: dict[str, str] = {
        "package_version": PACKAGE_VERSION,
        "git_commit": "unknown",
        "git_dirty": "unknown",
    }
    embedded = Path(__file__).with_name("_build_provenance.json")
    if embedded.is_file():
        try:
            metadata = json.loads(embedded.read_text())
            commit = metadata["git_commit"]
            expected = metadata["source_sha256"]
            if not re.fullmatch(r"[0-9a-f]{40}", commit):
                raise ValueError("invalid embedded revision")
            digest = hashlib.sha256()
            for source in sorted(embedded.parent.rglob("*.py")):
                digest.update(str(source.relative_to(embedded.parent)).encode() + b"\0")
                digest.update(source.read_bytes())
            if digest.hexdigest() != expected:
                out["build_integrity"] = "source_digest_mismatch"
                return out
            out.update(git_commit=commit, git_dirty=str(metadata["git_dirty"]),
                       source_sha256=expected, build_integrity="verified_package_source")
            return out
        except (KeyError, ValueError, OSError, TypeError):
            out["build_integrity"] = "invalid_embedded_provenance"
            return out
    try:
        root = repo_root()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10, check=False
        )
        if commit.returncode == 0:
            out["git_commit"] = commit.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True,
            text=True, timeout=15, check=False,
        )
        if status.returncode == 0:
            out["git_dirty"] = "true" if status.stdout.strip() else "false"
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - provenance must never crash a run
        pass
    digest = hashlib.sha256()
    try:
        package = Path(__file__).parent
        for source in sorted(package.rglob("*.py")):
            digest.update(str(source.relative_to(package)).encode() + b"\0")
            digest.update(source.read_bytes())
        out["source_sha256"] = digest.hexdigest()
    except OSError:
        out["source_sha256"] = "unknown"
    return out
