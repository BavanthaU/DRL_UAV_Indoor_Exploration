"""Repository-local paths used by Isaac Lab task configs and scripts."""

from __future__ import annotations

from pathlib import Path

ISAAC45_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ISAAC45_ROOT.parent
ENVIRONMENT_DIR = ISAAC45_ROOT / "environments"
DRONE_MODEL_DIR = ISAAC45_ROOT / "drone_models"
EVALUATION_DIR = REPO_ROOT / "evaluation_files"
IMAGE_DIR = REPO_ROOT / "images"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
