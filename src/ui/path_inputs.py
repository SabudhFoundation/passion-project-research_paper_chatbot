from __future__ import annotations

from pathlib import Path


def resolve_path_input(raw_value: str, default_path: Path) -> Path:
    cleaned_value = (raw_value or "").strip()
    return Path(cleaned_value) if cleaned_value else default_path