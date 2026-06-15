"""
core/paths.py

Single source of truth for project path helpers.
"""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def safe_slug(value: str, max_len: int = 90) -> str:
    """
    Convert arbitrary text into a safe file/folder slug.
    """
    value = value or "unknown"
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return (value[:max_len] or "unknown")


def resolve_project_path(raw_path: str | Path | None) -> Path | None:
    """
    Resolve artifact paths saved in JSON.

    Supports:
    - absolute paths
    - paths relative to current working directory
    - paths relative to project root
    """
    if not raw_path:
        return None

    path = Path(str(raw_path))

    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                path,
                Path.cwd() / path,
                PROJECT_ROOT / path,
                Path.cwd().parent / path,
            ]
        )

    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue

    return None