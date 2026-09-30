#!/usr/bin/env python3
"""Validate the exact compact paper archive, then check its saved-label arithmetic.

Only the trusted, unchanged verifier shipped in this repository is executed.
No code from a supplied archive or extracted directory is imported or executed.
This check does not rescore raw trajectories or run simulation experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ARCHIVE_SHA256 = "8fb48df5102ba4b06c1adcae83c7434152bdadff31009bc9f0d4d066d44ad894"
MANIFEST_SHA256 = "0441d6de82539a1bfec01bfa0d600293e9e5320ee044ef2d211b487aa99a6459"
MAX_TOTAL_BYTES = 64 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_name(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name or path.is_absolute() or ".." in path.parts
        or "\\" in name or "\x00" in name or str(path) != name
    ):
        raise ValueError(f"Unsafe archive path: {name!r}")


def validate_members(members: dict[str, bytes]) -> dict[str, bytes]:
    manifest_bytes = members.get("SHA256SUMS.json", b"")
    if digest(manifest_bytes) != MANIFEST_SHA256:
        raise ValueError("Evidence manifest does not match the paper's published archive.")
    manifest = json.loads(manifest_bytes)
    if set(members) != set(manifest) | {"SHA256SUMS.json"}:
        raise ValueError("Evidence contains missing or unexpected files.")
    for name, expected in manifest.items():
        safe_name(name)
        if digest(members[name]) != expected:
            raise ValueError(f"Evidence hash mismatch: {name}")
    return members


def load_evidence(source: Path) -> dict[str, bytes]:
    """Read bounded regular files; reject symlinks, traversal and modified evidence."""
    if source.is_symlink():
        raise ValueError("Provide a regular archive or directory, not a symlink.")
    members: dict[str, bytes] = {}
    if source.is_file():
        if source.stat().st_size > MAX_TOTAL_BYTES:
            raise ValueError("Archive exceeds the compact-evidence size limit.")
        archive_bytes = source.read_bytes()
        if digest(archive_bytes) != ARCHIVE_SHA256:
            raise ValueError("ZIP does not match the paper's published evidence.zip SHA-256.")
        with zipfile.ZipFile(source) as archive:
            if sum(info.file_size for info in archive.infolist()) > MAX_TOTAL_BYTES:
                raise ValueError("Uncompressed evidence exceeds the size limit.")
            for info in archive.infolist():
                safe_name(info.filename)
                mode = info.external_attr >> 16
                if info.is_dir() or stat.S_ISLNK(mode):
                    raise ValueError("The exact evidence archive contains regular files only.")
                if info.filename in members:
                    raise ValueError(f"Duplicate archive member: {info.filename}")
                members[info.filename] = archive.read(info)
    elif source.is_dir():
        total = 0
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Symlink in extracted evidence: {path.relative_to(source)}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError("Extracted evidence contains a non-regular file.")
            name = path.relative_to(source).as_posix()
            safe_name(name)
            total += path.stat().st_size
            if total > MAX_TOTAL_BYTES:
                raise ValueError("Extracted evidence exceeds the size limit.")
            members[name] = path.read_bytes()
    else:
        raise ValueError(f"Evidence path does not exist: {source}")
    return validate_members(members)


def verify_paper(source: Path) -> dict:
    members = load_evidence(source)
    verifier = (ROOT / "study/verify_reported_results.py").read_bytes()
    manifest = json.loads(members["SHA256SUMS.json"])
    if digest(verifier) != manifest["review-data/verify_reported_results.py"]:
        raise ValueError("The repository's arithmetic verifier has changed.")
    with tempfile.TemporaryDirectory(prefix="colosseum-paper-check-") as temporary:
        work = Path(temporary)
        for name, data in members.items():
            path = PurePosixPath(name)
            if path.parent == PurePosixPath("review-data") and path.suffix == ".json":
                (work / path.name).write_bytes(data)
        script = work / "verify_reported_results.py"
        script.write_bytes(verifier)
        # -I isolates imports and ignores PYTHON* environment settings. We do not
        # propagate -O: the frozen arithmetic verifier intentionally uses asserts.
        result = subprocess.run(
            [sys.executable, "-I", str(script)], check=True, capture_output=True,
            text=True, cwd=work, timeout=60,
        )
    report = json.loads(result.stdout)
    report["evidence_archive_sha256"] = ARCHIVE_SHA256
    report["archive_member_hashes_verified"] = len(members) - 1
    report["executed_verifier"] = "repository copy, byte-identical to paper archive"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, help="Original evidence.zip or its extracted root")
    args = parser.parse_args()
    try:
        report = verify_paper(args.evidence)
    except (OSError, ValueError, zipfile.BadZipFile, subprocess.SubprocessError) as error:
        parser.exit(1, f"Verification failed: {error}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
