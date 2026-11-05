"""Atomic file writes and content hashing.

Every task writes to a temporary file in the destination directory and then
``os.replace``s it into place. A crash mid-write therefore never leaves a
half-written artefact, and re-running a task simply overwrites its partition
with identical content (idempotency).
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


def _atomic_write(path: Path, writer) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            writer(fh)
            fh.flush()
            # mkstemp creates 0600 files. Lambda executes as a different OS user, so
            # artefacts that end up in a deployment package must be world-readable.
            os.fchmod(fh.fileno(), 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def write_csv_atomic(df: pd.DataFrame, path) -> Path:
    return _atomic_write(Path(path), lambda fh: df.to_csv(fh, index=False))


def write_json_atomic(obj, path) -> Path:
    data = json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
    return _atomic_write(Path(path), lambda fh: fh.write(data))


def write_bytes_atomic(data: bytes, path) -> Path:
    return _atomic_write(Path(path), lambda fh: fh.write(data))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
