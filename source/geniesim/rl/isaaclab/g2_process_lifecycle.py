"""Durable pre-close contract for Isaac Sim process shutdown.

Isaac Sim's official ``fast_shutdown`` mode can terminate the interpreter from
inside ``SimulationApp.close``.  It is therefore suitable only after every
artifact owned by Python has been closed, flushed, fsynced, and independently
attested.  This module does not call ``sys.exit`` or ``os._exit`` and does not
hide native shutdown failures.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


G2_SHUTDOWN_GRACEFUL_AUDIT = "graceful-audit"
G2_SHUTDOWN_OFFICIAL_FAST = "official-fast"
G2_SHUTDOWN_MODES = (
    G2_SHUTDOWN_GRACEFUL_AUDIT,
    G2_SHUTDOWN_OFFICIAL_FAST,
)


def use_official_fast_shutdown(mode: str) -> bool:
    if mode not in G2_SHUTDOWN_MODES:
        raise ValueError(f"unsupported G2 shutdown mode: {mode!r}")
    return mode == G2_SHUTDOWN_OFFICIAL_FAST


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    # Open read-only: the producer must already have closed the writable
    # handle.  fsync here is a durability barrier, not a substitute for close.
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o644,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def commit_pre_close_artifacts(
    *,
    output_dir: Path,
    shutdown_mode: str,
    artifact_paths: Iterable[Path],
) -> Path:
    """Seal required artifacts before ``SimulationApp.close``.

    Missing or empty artifacts fail closed.  The returned attestation is itself
    written atomically and directory-fsynced.
    """

    use_official_fast_shutdown(shutdown_mode)
    records: list[dict[str, object]] = []
    unique_paths = sorted({Path(item).resolve() for item in artifact_paths})
    if not unique_paths:
        raise ValueError("at least one pre-close artifact is required")
    for path in unique_paths:
        if not path.is_file():
            raise FileNotFoundError(f"required pre-close artifact missing: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"required pre-close artifact is empty: {path}")
        _fsync_file(path)
        records.append(
            {
                "path": str(path),
                "size_bytes": size,
                "sha256": file_sha256(path),
            }
        )
    attestation = {
        "schema": "geniesim_g2_pre_close_durability_v1",
        "shutdown_mode": shutdown_mode,
        "python_owned_cleanup_completed_before_commit": True,
        "python_owned_artifacts_closed_before_commit": True,
        "artifacts_fsynced": True,
        "artifact_hashes_verified": True,
        "simulation_app_close_not_yet_called": True,
        "shutdown_claim": (
            "OFFICIAL_FAST_AFTER_DURABLE_COMMIT"
            if use_official_fast_shutdown(shutdown_mode)
            else "GRACEFUL_FINALIZATION_AUDIT"
        ),
        "artifacts": records,
    }
    path = output_dir / "pre_close_durability.json"
    _durable_json(path, attestation)
    return path


def verify_pre_close_attestation(path: Path) -> bool:
    """Verify a durability record from a supervisor process."""

    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text())
        records = value["artifacts"]
        if not value["artifact_hashes_verified"] or not records:
            return False
        for record in records:
            artifact = Path(record["path"])
            if not artifact.is_file():
                return False
            if artifact.stat().st_size != int(record["size_bytes"]):
                return False
            if file_sha256(artifact) != record["sha256"]:
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
