#!/usr/bin/env python3
"""Check the release's unchanged scientific files against their recorded hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def verify_source(root: Path) -> int:
    manifest = json.loads((root / "study/release-file-provenance.json").read_text())
    failures = []
    for name, binding in manifest["files"].items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"Unsafe manifest path: {name}")
        target = root / name
        if target.is_symlink() or not target.is_file():
            failures.append(f"missing or symlink: {name}")
        elif hashlib.sha256(target.read_bytes()).hexdigest() != binding["sha256"]:
            failures.append(f"changed: {name}")
    if failures:
        raise ValueError("Frozen-source integrity failed:\n" + "\n".join(failures))
    return len(manifest["files"])


if __name__ == "__main__":
    count = verify_source(Path(__file__).resolve().parents[1])
    print(f"Verified {count} unchanged files from the paper's evidence archive.")
