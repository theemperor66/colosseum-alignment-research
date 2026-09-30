"""Release-wrapper checks use synthetic files, not new experimental evidence."""

import hashlib
import json
import stat
import zipfile
from pathlib import Path

import pytest
from scripts import verify_paper, verify_source


def miniature_evidence(monkeypatch):
    data = b'{"synthetic": true}\n'
    manifest = json.dumps({"review-data/example.json": hashlib.sha256(data).hexdigest()}).encode()
    monkeypatch.setattr(verify_paper, "MANIFEST_SHA256", hashlib.sha256(manifest).hexdigest())
    return {"SHA256SUMS.json": manifest, "review-data/example.json": data}


def test_frozen_release_sources_are_unchanged():
    root = Path(__file__).resolve().parents[1]
    assert verify_source.verify_source(root) == 109


def test_extracted_evidence_requires_exact_contents(tmp_path, monkeypatch):
    members = miniature_evidence(monkeypatch)
    for name, data in members.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    assert verify_paper.load_evidence(tmp_path) == members
    (tmp_path / "unexpected.py").write_text("raise RuntimeError('never execute me')")
    with pytest.raises(ValueError, match="unexpected"):
        verify_paper.load_evidence(tmp_path)


def test_changed_data_is_rejected(monkeypatch):
    members = miniature_evidence(monkeypatch)
    members["review-data/example.json"] = b'{"synthetic": false}'
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_paper.validate_members(members)


def test_rewritten_manifest_is_rejected():
    with pytest.raises(ValueError, match="manifest"):
        verify_paper.validate_members({"SHA256SUMS.json": b"{}"})


@pytest.mark.parametrize("name", ["../outside.json", "/tmp/outside.json", "a/../b", "a\\b", "a//b", "./a"])
def test_unsafe_member_paths_are_rejected(name):
    with pytest.raises(ValueError, match="Unsafe"):
        verify_paper.safe_name(name)


def test_unrecognised_zip_is_rejected_before_execution(tmp_path):
    archive = tmp_path / "untrusted.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("review-data/verify_reported_results.py", "raise RuntimeError('untrusted')")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_paper.load_evidence(archive)


def test_symlink_in_extracted_evidence_is_rejected(tmp_path):
    (tmp_path / "link").symlink_to("nonexistent")
    with pytest.raises(ValueError, match="Symlink"):
        verify_paper.load_evidence(tmp_path)


def test_zip_symlink_is_rejected(tmp_path, monkeypatch):
    archive = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(info, "outside")
    monkeypatch.setattr(verify_paper, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="regular files"):
        verify_paper.load_evidence(archive)


def test_bounded_directory_read(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_paper, "MAX_TOTAL_BYTES", 3)
    (tmp_path / "oversized").write_bytes(b"1234")
    with pytest.raises(ValueError, match="size limit"):
        verify_paper.load_evidence(tmp_path)
