"""Shared helper functions for saving outputs and standardizing result payloads."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def ensure_parent_dir(path: Path) -> Path:
    """Create the parent directory for a target file and return the resolved path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _coerce_output_path(output_path: Path, default_name: str) -> Path:
    if output_path.exists() and output_path.is_dir():
        return output_path / default_name

    if output_path.suffix:
        return output_path

    return output_path / default_name


def _rows_from_data(data: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(data, Mapping):
        return [data]

    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        rows: list[Mapping[str, Any]] = []
        for item in data:
            if isinstance(item, Mapping):
                rows.append(item)
            else:
                rows.append({"value": item})
        return rows

    return [{"value": data}]


def _stringify_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False)
    return value


def save_json(data: Any, output_path: Path, append: bool = False) -> Path:
    """Write JSON to `output_path`.

    If `append=True` then the file will be treated as a JSON array and the new
    `data` item will be appended. If the file does not exist it will be created
    with an array containing `data`.
    """
    output_path = ensure_parent_dir(output_path)

    if not append:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        return output_path

    # Append mode: maintain a JSON array in the file
    if not output_path.exists():
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump([data], handle, ensure_ascii=False, indent=2)
        return output_path

    # Read existing content
    try:
        with open(output_path, "r", encoding="utf-8") as handle:
            content = handle.read().strip()
            if not content:
                existing = []
            else:
                existing = json.loads(content)
    except (json.JSONDecodeError, OSError):
        # If file is malformed or unreadable, preserve raw content and append
        existing = [content if isinstance(content, str) else None]

    # Compute next order number from existing items (if any)
    next_order = 1
    if isinstance(existing, list):
        max_order = 0
        for item in existing:
            if isinstance(item, Mapping) and "order" in item and isinstance(item["order"], int):
                if item["order"] > max_order:
                    max_order = item["order"]
        next_order = max_order + 1

        # Prepare the item to append, ensuring it's a mapping with an `order`
        if isinstance(data, Mapping):
            new_item = dict(data)
            new_item.setdefault("order", next_order)
        else:
            new_item = {"order": next_order, "value": data}

        existing.append(new_item)
    else:
        # existing is not a list (malformed or raw content). Convert to list
        # and assign orders to both the preserved content and the new item.
        preserved = existing
        existing = []

        # Attempt to assign order 1 to preserved if it is a mapping
        if isinstance(preserved, Mapping):
            preserved_with_order = dict(preserved)
            preserved_with_order.setdefault("order", 1)
            existing.append(preserved_with_order)
            next_order = preserved_with_order.get("order", 1) + 1
        else:
            existing.append({"order": 1, "value": preserved})
            next_order = 2

        if isinstance(data, Mapping):
            new_item = dict(data)
            new_item.setdefault("order", next_order)
        else:
            new_item = {"order": next_order, "value": data}

        existing.append(new_item)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(existing, handle, ensure_ascii=False, indent=2)

    return output_path


def save_csv(data: Any, output_path: Path) -> Path:
    output_path = ensure_parent_dir(output_path)
    rows = list(_rows_from_data(data))

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _stringify_cell(row.get(key, "")) for key in fieldnames})

    return output_path


def save_excel(data: Any, output_path: Path, sheet_name: str = "Sheet1") -> Path:
    rows = list(_rows_from_data(data))
    output_path = ensure_parent_dir(output_path)

    try:
        from openpyxl import Workbook
    except ImportError as exc:
        raise ImportError(
            "Excel export requires the 'openpyxl' package to be installed."
        ) from exc

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = sheet_name[:31]

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    worksheet.append(fieldnames)
    for row in rows:
        worksheet.append([_stringify_cell(row.get(key, "")) for key in fieldnames])

    workbook.save(output_path)
    return output_path


def save_data(data: Any, output_path: Path, append: bool = False) -> Path:
    target_path = _coerce_output_path(output_path, "result.json")
    suffix = target_path.suffix.lower()

    if suffix == ".csv":
        return save_csv(data, target_path)
    if suffix in {".xlsx", ".xlsm"}:
        return save_excel(data, target_path)
    return save_json(data, target_path, append=append)


def save_result(output_path: Path, result: dict[str, Any], append: bool = False) -> Path:
    return save_data(result, output_path, append=append)


def reset_output_file(output_path: Path) -> Path:
    """Clear the output file and replace it with an empty JSON array."""

    output_path = ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump([], handle, ensure_ascii=False, indent=2)
    return output_path


def error_result(
    question: str,
    provider: str | None,
    model_name: str,
    temperature: float,
    vector_store_path: Path,
    top_k: int,
    elapsed: float,
    error: str,
) -> dict[str, Any]:
    return {
        "question": question,
        "route": "error",
        "answer": "",
        "error": error,
        "provider": provider,
        "model_name": model_name,
        "temperature": temperature,
        "vector_store_path": str(vector_store_path),
        "top_k": top_k,
        "time_taken_sec": elapsed,
        "retrieved_chunks": [],
    }
