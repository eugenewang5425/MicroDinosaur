"""Rebuild the public repository and simulation file inventories.

Run from any directory after adding intended files to Git. Deleted index entries
and ignored local outputs are excluded; both manifests deliberately omit
themselves to avoid recursive hashes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_MANIFEST = ROOT / "repository_files.json"
SOURCE_MANIFEST = ROOT / "simulation" / "SOURCE_SNAPSHOT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_paths() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    paths = []
    for raw in output.decode("utf-8").split("\0"):
        if not raw:
            continue
        path = ROOT / raw
        if path.is_file():
            paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(ROOT).as_posix())


def row(path: Path) -> dict:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    original_source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    paths = git_paths()
    simulation_paths = [
        path for path in paths
        if path.is_relative_to(ROOT / "simulation") and path != SOURCE_MANIFEST
    ]
    source = {
        "snapshot_date": "2026-09-21",
        "upstream": original_source.get("upstream"),
        "upstream_base_commit": original_source.get("upstream_base_commit"),
        "source_worktree_was_dirty": True,
        "note": (
            "Curated public simulation package after relative-path cleanup. "
            "Historical reports are frozen evidence; no physical hardware deployment is claimed."
        ),
        "files": [row(path) for path in simulation_paths],
        "asset_members": original_source["asset_members"],
    }
    write_json(SOURCE_MANIFEST, source)

    # SOURCE_SNAPSHOT changed above, so obtain repository rows afterwards.
    paths = git_paths()
    repository = {
        "schema_version": 3,
        "repository": "eugenewang5425/MicroDinosaur",
        "visibility": "public",
        "scope": (
            "Current MicroDinosaur CAD, curated training source, selected policies, "
            "and action requalification evidence"
        ),
        "model_path": "current/MicroDinosaur_v1.blender",
        "model_sha256": "0a4f86aefea24e0d5260c837849a16164ec83cc2d364f58e6c03947de95ce286",
        "model_storage": "Git LFS",
        "excluded_local_only": [
            "retired CAD assemblies",
            "virtual environments and caches",
            "complete raw training logs and checkpoint pools",
            "personal resumes and contact details",
            "unselected experiment trajectories and videos",
        ],
        "note": (
            "Inventory excludes itself. Paths are repository-relative. Research candidates remain "
            "separate from qualified policies, and historical reports retain their original scope."
        ),
        "files": [row(path) for path in paths if path != REPOSITORY_MANIFEST],
    }
    write_json(REPOSITORY_MANIFEST, repository)
    print(json.dumps({"repository_files": len(repository["files"]), "simulation_files": len(source["files"])}))


if __name__ == "__main__":
    main()
