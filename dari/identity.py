"""Stable artifact fingerprints used for frozen model/policy compatibility."""
import hashlib
import json
from pathlib import Path


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def artifact_digest(path: str | Path | None) -> str:
    if path is None:
        return "bootstrap"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    digest = hashlib.sha256()
    for file in files:
        digest.update((file.name if path.is_file() else file.relative_to(path).as_posix()).encode())
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
