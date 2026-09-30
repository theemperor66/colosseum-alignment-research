"""Shared builders for tests. Keeping these in one place stops each test inventing its own provenance."""

from __future__ import annotations

from colosseum_assurance.schemas import SimulatorArtifactAttestation, SimulatorIdentity


def third_party_attestation(**overrides) -> SimulatorArtifactAttestation:
    """A qualified third-party package attestation, shaped like the staged ESAR publication."""
    payload = dict(
        provenance_class="third_party_colosseum_build",
        artifact_name="ESARMaps_linux.zip",
        artifact_sha256="20a25418d63c60ccda9284e657d9a61f755d9b56adcea4195eb103fdbf853f91",
        artifact_size_bytes=7579779117,
        source_kind="third_party_publication",
        source_url="https://huggingface.co/datasets/4amGodvzx/ESAR",
        source_repo="4amGodvzx/ESAR",
        source_revision="0f4ae9cde09ecb7f0d60074f1b83c2372fb5122c",
        engine_version_declared="5.6.1",
        scene_package_path="/opt/esar/UEPackage/dapeng",
        verified_by="test-fixture",
        verified_utc="2026-09-16T21:35:25Z",
        verification_note="published SHA-256 matched the locally staged archive",
        caveats=["the exact upstream Colosseum source commit of this package is unestablished"],
    )
    payload.update(overrides)
    return SimulatorArtifactAttestation(**payload)


def anchored_identity(**overrides) -> SimulatorIdentity:
    """An identity that may back experimental claims: anchored to a hash-verified artifact."""
    payload = dict(
        provenance="third_party_colosseum_build",
        endpoint_label="tunnel",
        artifact=third_party_attestation(),
    )
    payload.update(overrides)
    return SimulatorIdentity(**payload)
