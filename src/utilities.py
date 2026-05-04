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


def save_json(data: Any, output_path: Path) -> Path:
    output_path = ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
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


def save_data(data: Any, output_path: Path) -> Path:
    target_path = _coerce_output_path(output_path, "result.json")
    suffix = target_path.suffix.lower()

    if suffix == ".csv":
        return save_csv(data, target_path)
    if suffix in {".xlsx", ".xlsm"}:
        return save_excel(data, target_path)
    return save_json(data, target_path)


def save_result(output_path: Path, result: dict[str, Any]) -> Path:
    return save_data(result, output_path)


def error_result(
    question: str,
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
        "model_name": model_name,
        "temperature": temperature,
        "vector_store_path": str(vector_store_path),
        "top_k": top_k,
        "time_taken_sec": elapsed,
        "retrieved_chunks": [],
    }
