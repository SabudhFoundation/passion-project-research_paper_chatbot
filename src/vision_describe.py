"""
vision_describe.py

Vision preprocessing for multimodal RAG.

Reads table / figure / diagram PNG artifacts from combined_papers.json,
sends them to a vision-capable LLM, and saves vision_description back into JSON.

This script is intentionally separate from main.py:
- main.py answers user questions
- vision_describe.py prepares image descriptions before chunking/embedding

Example:
python src/vision_describe.py --combined_json data/parsed_combined/combined_papers.json --output_json data/parsed_combined/combined_papers.json --provider groq --model_name meta-llama/llama-4-scout-17b-16e-instruct --artifact_types tables --limit 1
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from config import (
    ENABLE_MERMAID_GENERATION,
    MODEL_CAPABILITIES,
    TEMPERATURE,
)
from core.constants import (
    ContentType,
    VISION_ELIGIBLE_TABLE_STATUSES,
    VisionStatus,
)
from core.paths import resolve_project_path
from llm import build_llm
from prompt import build_table_vision_prompt, build_visual_vision_prompt
from utilities import save_json


# =========================================================
# Config
# =========================================================

class VisionConfig(BaseModel):
    combined_json: Path
    output_json: Path | None = None
    provider: str
    model_name: str
    temperature: float = TEMPERATURE
    artifact_types: str = "all"
    table_statuses: list[str] = Field(
        default_factory=lambda: sorted(VISION_ELIGIBLE_TABLE_STATUSES)
    )
    limit: int = 0
    retries: int = 1
    retry_delay: float = 2.0
    dry_run: bool = False
    force: bool = False

    @field_validator("combined_json")
    @classmethod
    def file_must_exist(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"combined_json not found: {value}")
        return value

    @field_validator("artifact_types")
    @classmethod
    def valid_artifact_types(cls, value: str) -> str:
        value = value.strip().lower()
        allowed = {"all", "tables", "visuals"}
        if value not in allowed:
            raise ValueError(f"artifact_types must be one of {allowed}")
        return value


def parse_args() -> VisionConfig:
    parser = argparse.ArgumentParser(description="Generate vision descriptions for artifacts")

    parser.add_argument("--combined_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--provider", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument(
        "--artifact_types",
        type=str,
        default="all",
        choices=["all", "tables", "visuals"],
    )
    parser.add_argument(
        "--table_statuses",
        type=str,
        default="weak,text_fallback,caption_only",
        help="Comma-separated table statuses eligible for vision.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of artifacts to attempt. Use 0 for no limit.",
    )
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry_delay", type=float, default=2.0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    table_statuses = [
        status.strip()
        for status in args.table_statuses.split(",")
        if status.strip()
    ]

    return VisionConfig(
        combined_json=Path(args.combined_json),
        output_json=Path(args.output_json) if args.output_json else None,
        provider=args.provider,
        model_name=args.model_name,
        temperature=args.temperature,
        artifact_types=args.artifact_types,
        table_statuses=table_statuses,
        limit=args.limit,
        retries=args.retries,
        retry_delay=args.retry_delay,
        dry_run=args.dry_run,
        force=args.force,
    )


# =========================================================
# JSON helpers
# =========================================================

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def backup_once(path: Path) -> Path:
    backup_path = path.with_name(f"{path.stem}.before_vision{path.suffix}")

    if path.exists() and not backup_path.exists():
        backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    return backup_path


def extract_json_object(text: str) -> dict[str, Any] | None:
    """
    Extract first JSON object from model output.
    Handles outputs wrapped in ```json fences.
    """
    if not text:
        return None

    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    start = cleaned.find("{")
    if start == -1:
        return None

    try:
        parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


# =========================================================
# Eligibility helpers
# =========================================================

def already_generated(artifact: dict[str, Any]) -> bool:
    return artifact.get("vision_status") == VisionStatus.GENERATED.value


def has_existing_description(artifact: dict[str, Any]) -> bool:
    description = str(artifact.get("vision_description", "") or "").strip()
    return bool(description)


def table_is_eligible(
    table: dict[str, Any],
    config: VisionConfig,
) -> bool:
    status = table.get("structured_status", "")

    if status not in set(config.table_statuses):
        return False

    if not table.get("image_path"):
        return False

    if config.force:
        return True

    return not already_generated(table)


def visual_is_eligible(
    visual: dict[str, Any],
    config: VisionConfig,
) -> bool:
    content_type = visual.get("content_type", ContentType.FIGURE.value)

    if content_type not in {
        ContentType.FIGURE.value,
        ContentType.IMAGE.value,
        ContentType.DIAGRAM.value,
        ContentType.FLOWCHART.value,
    }:
        return False

    if not visual.get("image_path"):
        return False

    if config.force:
        return True

    return not already_generated(visual)


def image_exists(raw_path: str) -> tuple[bool, Path | None]:
    path = resolve_project_path(raw_path)
    return path is not None and path.exists(), path


# =========================================================
# Error helpers
# =========================================================

def is_quota_or_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()

    terms = [
        "429",
        "quota",
        "rate limit",
        "ratelimit",
        "resource exhausted",
        "resourceexhausted",
        "too many requests",
    ]

    return any(term in message for term in terms)


def mark_failed(
    artifact: dict[str, Any],
    error: str,
    model_name: str,
) -> None:
    artifact["vision_status"] = VisionStatus.FAILED.value
    artifact["vision_model"] = model_name
    artifact["vision_error"] = error[:3000]


# =========================================================
# LLM call helpers
# =========================================================

def ensure_model_supports_vision(config: VisionConfig) -> None:
    capabilities = MODEL_CAPABILITIES.get(config.model_name)

    if capabilities is None:
        print(
            f"[warning] Model '{config.model_name}' not found in MODEL_CAPABILITIES. "
            "Continuing anyway; provider API will decide if images are supported."
        )
        return

    if not capabilities.get("supports_vision"):
        raise ValueError(
            f"Model '{config.model_name}' is marked as supports_vision=False. "
            "Choose a vision-capable model."
        )


def call_vision_model(
    *,
    llm,
    prompt: str,
    image_path: Path,
    config: VisionConfig,
) -> str:
    last_error: Exception | None = None

    for attempt in range(1, max(config.retries, 1) + 1):
        try:
            return llm.generate_content(
                prompt,
                images=[image_path],
            ).text.strip()

        except Exception as exc:
            last_error = exc
            print(
                f"    -> vision call failed "
                f"(attempt {attempt}/{config.retries}): {exc}",
                flush=True,
            )

            if attempt < config.retries:
                time.sleep(config.retry_delay)

    assert last_error is not None
    raise last_error


# =========================================================
# Artifact processors
# =========================================================

def apply_table_vision_response(
    table: dict[str, Any],
    raw_text: str,
    model_name: str,
) -> None:
    parsed = extract_json_object(raw_text)

    if parsed:
        description = str(parsed.get("vision_description", "") or "").strip()
        markdown = str(parsed.get("vision_markdown", "") or "").strip()
        confidence = str(parsed.get("confidence", "") or "").strip()
        limitations = str(parsed.get("limitations", "") or "").strip()
    else:
        description = raw_text.strip()
        markdown = ""
        confidence = "low"
        limitations = "Model response was not valid JSON."

    table["vision_status"] = VisionStatus.GENERATED.value
    table["vision_model"] = model_name
    table["vision_description"] = description
    table["vision_markdown"] = markdown
    table["vision_confidence"] = confidence
    table["vision_limitations"] = limitations
    table["vision_raw_response"] = raw_text[:5000]


def apply_visual_vision_response(
    visual: dict[str, Any],
    raw_text: str,
    model_name: str,
) -> None:
    parsed = extract_json_object(raw_text)

    if parsed:
        description = str(parsed.get("vision_description", "") or "").strip()
        mermaid = str(parsed.get("mermaid", "") or "").strip()
        confidence = str(parsed.get("confidence", "") or "").strip()
        limitations = str(parsed.get("limitations", "") or "").strip()
    else:
        description = raw_text.strip()
        mermaid = ""
        confidence = "low"
        limitations = "Model response was not valid JSON."

    visual["vision_status"] = VisionStatus.GENERATED.value
    visual["vision_model"] = model_name
    visual["vision_description"] = description
    visual["vision_confidence"] = confidence
    visual["vision_limitations"] = limitations
    visual["vision_raw_response"] = raw_text[:5000]

    if mermaid:
        visual["mermaid"] = mermaid
        visual["mermaid_status"] = "generated"


def process_table(
    *,
    llm,
    table: dict[str, Any],
    image_path: Path,
    config: VisionConfig,
) -> None:
    prompt = build_table_vision_prompt(table)
    raw_text = call_vision_model(
        llm=llm,
        prompt=prompt,
        image_path=image_path,
        config=config,
    )
    apply_table_vision_response(table, raw_text, config.model_name)


def process_visual(
    *,
    llm,
    visual: dict[str, Any],
    image_path: Path,
    config: VisionConfig,
) -> None:
    prompt = build_visual_vision_prompt(
        visual,
        generate_mermaid=ENABLE_MERMAID_GENERATION,
    )
    raw_text = call_vision_model(
        llm=llm,
        prompt=prompt,
        image_path=image_path,
        config=config,
    )
    apply_visual_vision_response(visual, raw_text, config.model_name)


# =========================================================
# Main processing
# =========================================================

def can_attempt_more(config: VisionConfig, counters: dict[str, int]) -> bool:
    return config.limit <= 0 or counters["attempted"] < config.limit


def run_vision_describe(config: VisionConfig) -> dict[str, int]:
    output_path = config.output_json or config.combined_json

    papers = load_json(config.combined_json)

    if not isinstance(papers, list):
        raise ValueError("combined_json must contain a list of paper objects.")

    counters = {
        "eligible_tables": 0,
        "eligible_visuals": 0,
        "attempted": 0,
        "processed": 0,
        "skipped_existing": 0,
        "missing_image": 0,
        "failed": 0,
    }

    print("\nCONFIG")
    print(f"combined_json  : {config.combined_json}")
    print(f"output_json    : {output_path}")
    print(f"provider       : {config.provider}")
    print(f"model_name     : {config.model_name}")
    print(f"artifact_types : {config.artifact_types}")
    print(f"table_statuses : {config.table_statuses}")
    print(f"limit          : {config.limit}")
    print(f"dry_run        : {config.dry_run}")
    print(f"force          : {config.force}")
    print()

    reached_stop_error = False

    llm = None
    if not config.dry_run:
        ensure_model_supports_vision(config)
        llm = build_llm(
            config.model_name,
            config.temperature,
            provider=config.provider,
        )
        backup_path = backup_once(output_path)
        print(f"Backup path: {backup_path}\n")

    for paper_index, paper in enumerate(papers, start=1):
        paper_name = paper.get("paper_name", paper.get("source_file", "Unknown paper"))
        print(f"[PAPER {paper_index}] {paper_name}")

        # -----------------------------
        # Tables
        # -----------------------------
        if config.artifact_types in {"all", "tables"}:
            for table in paper.get("tables", []) or []:
                if not table_is_eligible(table, config):
                    if already_generated(table):
                        counters["skipped_existing"] += 1
                    continue

                counters["eligible_tables"] += 1

                caption = str(table.get("caption", ""))[:120]
                status = table.get("structured_status", "")

                exists, resolved_image_path = image_exists(table.get("image_path", ""))

                print(f"  [table eligible] status={status} | caption={caption}")

                if not exists or resolved_image_path is None:
                    counters["missing_image"] += 1
                    print("    -> missing image")
                    continue

                if config.dry_run:
                    continue

                if not can_attempt_more(config, counters):
                    continue

                try:
                    counters["attempted"] += 1

                    process_table(
                        llm=llm,
                        table=table,
                        image_path=resolved_image_path,
                        config=config,
                    )

                    counters["processed"] += 1
                    save_json(papers, output_path)
                    print("    -> generated")

                except Exception as exc:
                    counters["failed"] += 1
                    error_text = f"{exc}\n{traceback.format_exc()}"
                    mark_failed(table, error_text, config.model_name)
                    save_json(papers, output_path)
                    print(f"    -> failed: {exc}")

                    if is_quota_or_rate_limit_error(exc):
                        print("    -> quota/rate-limit detected. Stopping further calls.")
                        reached_stop_error = True
                        break

            if reached_stop_error:
                break

        # -----------------------------
        # Visuals
        # -----------------------------
        if config.artifact_types in {"all", "visuals"}:
            for visual in paper.get("figures", []) or []:
                if not visual_is_eligible(visual, config):
                    if already_generated(visual):
                        counters["skipped_existing"] += 1
                    continue

                counters["eligible_visuals"] += 1

                caption = str(visual.get("caption", ""))[:120]
                content_type = visual.get("content_type", "")

                exists, resolved_image_path = image_exists(visual.get("image_path", ""))

                print(f"  [visual eligible] type={content_type} | caption={caption}")

                if not exists or resolved_image_path is None:
                    counters["missing_image"] += 1
                    print("    -> missing image")
                    continue

                if config.dry_run:
                    continue

                if not can_attempt_more(config, counters):
                    continue

                try:
                    counters["attempted"] += 1

                    process_visual(
                        llm=llm,
                        visual=visual,
                        image_path=resolved_image_path,
                        config=config,
                    )

                    counters["processed"] += 1
                    save_json(papers, output_path)
                    print("    -> generated")

                except Exception as exc:
                    counters["failed"] += 1
                    error_text = f"{exc}\n{traceback.format_exc()}"
                    mark_failed(visual, error_text, config.model_name)
                    save_json(papers, output_path)
                    print(f"    -> failed: {exc}")

                    if is_quota_or_rate_limit_error(exc):
                        print("    -> quota/rate-limit detected. Stopping further calls.")
                        reached_stop_error = True
                        break

            if reached_stop_error:
                break

    print("\nSUMMARY")
    for key, value in counters.items():
        print(f"{key}: {value}")

    if config.dry_run:
        print("\nDry run completed. No model calls were made and no JSON was written.")
    else:
        save_json(papers, output_path)
        print(f"\nSaved updated JSON: {output_path}")

    return counters


if __name__ == "__main__":
    cfg = parse_args()
    run_vision_describe(cfg)