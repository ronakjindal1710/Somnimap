"""Portable configuration loader for SomniMap.

Resolves repository root dynamically and avoids hardcoded machine paths.
Environment overrides:
  SOMNIMAP_CONFIG          Path to YAML config (defaults to <repo>/experiments/configs/base_config.yaml)
  SOMNIMAP_RAW_DATASET_DIR Optional override for paths.raw_dataset_dir
  SOMNIMAP_PROCESSED_DIR   Optional override for paths.processed_dir
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def get_repo_root() -> Path:
    """Return the SomniMap repository root (parent of `src/`)."""
    return Path(__file__).resolve().parents[1]


def get_default_config_path() -> Path:
    env_cfg = os.environ.get("SOMNIMAP_CONFIG")
    if env_cfg:
        return Path(env_cfg).expanduser().resolve()
    return get_repo_root() / "experiments" / "configs" / "base_config.yaml"


def _resolve_repo_relative(raw_val: str, repo_root: Path) -> str:
    p = Path(raw_val)
    if p.is_absolute():
        return str(p)
    return str((repo_root / p).resolve())


def load_config(cfg_path: Path | str | None = None) -> dict[str, Any]:
    """Load base_config.yaml and resolve repo-relative paths portably."""
    path = Path(cfg_path) if cfg_path is not None else get_default_config_path()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    repo_root = get_repo_root()
    paths = cfg.setdefault("paths", {})

    # Always bind project_root to the active repository checkout unless explicitly overridden
    if not paths.get("project_root") or not Path(paths["project_root"]).exists():
        paths["project_root"] = str(repo_root)
    else:
        paths["project_root"] = _resolve_repo_relative(paths["project_root"], repo_root)

    proj_root = Path(paths["project_root"])
    defaults = {
        "manifest_csv": str(proj_root / "data" / "manifest.csv"),
        "annotation_coverage_csv": str(proj_root / "data" / "annotation_coverage.csv"),
        "split_manifest_json": str(proj_root / "data" / "splits" / "split_manifest.json"),
        "processed_dir": str(proj_root / "data" / "processed"),
        "mlflow_tracking_uri": f"sqlite:///{(proj_root / 'mlflow.db').as_posix()}",
    }
    for key, fallback in defaults.items():
        val = paths.get(key)
        if not val:
            paths[key] = fallback
        elif key == "mlflow_tracking_uri":
            if val.startswith("sqlite:///"):
                db_part = val[len("sqlite:///"):]
                if not Path(db_part).is_absolute():
                    paths[key] = f"sqlite:///{(proj_root / db_part).as_posix()}"
                elif not Path(db_part).parent.exists():
                    paths[key] = fallback
        else:
            resolved = _resolve_repo_relative(val, proj_root)
            paths[key] = resolved if Path(resolved).parent.exists() else fallback

    if os.environ.get("SOMNIMAP_PROCESSED_DIR"):
        paths["processed_dir"] = str(Path(os.environ["SOMNIMAP_PROCESSED_DIR"]).expanduser().resolve())

    if os.environ.get("SOMNIMAP_RAW_DATASET_DIR"):
        paths["raw_dataset_dir"] = str(Path(os.environ["SOMNIMAP_RAW_DATASET_DIR"]).expanduser().resolve())
    elif paths.get("raw_dataset_dir"):
        raw_p = Path(paths["raw_dataset_dir"])
        if not raw_p.is_absolute():
            paths["raw_dataset_dir"] = str((proj_root / raw_p).resolve())

    return cfg
