from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_system_identity(
    config_path: str | Path,
    profile_name: str,
    profile: Mapping[str, Any],
    repo_root: str | Path | None = None,
) -> Dict[str, object]:
    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    config_file = Path(config_path).resolve()
    git_commit = _git(root, "rev-parse", "HEAD")
    tracked_changes = _git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    try:
        normalized_config_path = str(config_file.relative_to(root)).replace(
            "\\",
            "/",
        )
    except ValueError:
        normalized_config_path = str(config_file)
    identity: Dict[str, object] = {
        "schema_version": 1,
        "git_commit": git_commit,
        "git_dirty": bool(tracked_changes),
        "config_path": normalized_config_path,
        "config_sha256": sha256_file(config_file),
        "profile": profile_name,
        "profile_sha256": stable_hash(profile),
    }
    identity["system_fingerprint"] = stable_hash(identity)
    return identity


def capture_run_identity(
    system_identity: Mapping[str, object],
    parameters: Mapping[str, object],
) -> Dict[str, object]:
    normalized_parameters = dict(parameters)
    return {
        **dict(system_identity),
        "parameters": normalized_parameters,
        "run_fingerprint": stable_hash(
            {
                "system_fingerprint": system_identity["system_fingerprint"],
                "parameters": normalized_parameters,
            }
        ),
    }
