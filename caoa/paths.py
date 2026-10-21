"""Data locations, read from configs/paths.yaml (or a local / CAOA_PATHS override)."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RESOURCES = Path(__file__).resolve().parent / "resources"


@dataclass(frozen=True)
class Paths:
    scannet_scans: Path
    shapenet: Path
    s2c_annotations: Path
    scan2cad_full_annotations: Path
    scan2cad_cad_appearances: Path
    legacy_repo: Path
    cache: Path


def load_paths(config: str | os.PathLike | None = None) -> Paths:
    if config is None:
        config = os.environ.get("CAOA_PATHS")
    if config is None:
        local = REPO_ROOT / "configs" / "paths.local.yaml"
        config = local if local.exists() else REPO_ROOT / "configs" / "paths.yaml"
    with open(config) as f:
        raw = yaml.safe_load(f)
    # Relative paths are relative to the repo root, not the working directory.
    return Paths(**{k: REPO_ROOT / Path(v) for k, v in raw.items()})
