"""
rag_common.py

Shared RAG pipeline helpers used by:
- main.py
- uploaded_pdf_pipeline.py

This file contains common RAG logic only:
- typed retrieval decision
- retrieved chunk formatting
- retrieved chunk serialization
- route classification
- answer generation
- generation metrics
- overall metrics

Do not put generic JSON/file utilities here.
Those belong in utilities.py.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from config import (
    MAX_ROUTE_RETRIES,
    ROUTE_RETRY_DELAY,
)
from core.constants import ContentType
from metrics import (
    Timer,
    build_overall_metrics,
    build_stage_metrics,
)
from prompt import (
    route_query,
    summarize_non_rag,
    summarize_rag,
)


# =========================================================
# Typed retrieval routing
# =========================================================

TABLE_QUERY_TERMS = re.compile(
    r"\b("
    r"table|tab\.?|row|column|accuracy|score|metric|metrics|comparison|"
    r"performance|result\s+table|hyperparameter|hyperparameters"
    r")\b",
    re.IGNORECASE,
)

FIGURE_QUERY_TERMS = re.compile(
    r"\b("
    r"figure|fig\.?|image|plot|chart|graph|visualization|visualisation"
    r")\b",
    re.IGNORECASE,
)

DIAGRAM_QUERY_TERMS = re.compile(
    r"\b("
    r"diagram|flowchart|architecture|framework|pipeline|workflow|"
    r"model\s+structure|network\s+structure|encoder|decoder|block\s+diagram"
    r")\b",
    re.IGNORECASE,
)

METHOD_QUERY_TERMS = re.compile(
    r"\b("
    r"method|methodology|approach|system|model|architecture|framework|pipeline"
    r")\b",
    re.IGNORECASE,
)


def decide_content_types(question: str) -> list[str]:
    """
    Decide which chunk types to retrieve for a user question.

    The LLM router decides rag vs non_rag.
    This function only decides which content types are most relevant.
    """
    question = question or ""

    if TABLE_QUERY_TERMS.search(question):
        return [
            ContentType.TABLE.value,
            ContentType.TEXT.value,
        ]

    if DIAGRAM_QUERY_TERMS.search(question):
        return [
            ContentType.DIAGRAM.value,
            ContentType.FLOWCHART.value,
            ContentType.FIGURE.value,
            ContentType.IMAGE.value,
            ContentType.TEXT.value,
        ]

    if FIGURE_QUERY_TERMS.search(question):
        return [
            ContentType.FIGURE.value,
            ContentType.IMAGE.value,
            ContentType.DIAGRAM.value,
            ContentType.FLOWCHART.value,
            ContentType.TEXT.value,
        ]

    if METHOD_QUERY_TERMS.search(question):
        return [
            ContentType.TEXT.value,
            ContentType.DIAGRAM.value,
            ContentType.FIGURE.value,
            ContentType.IMAGE.value,
            ContentType.FLOWCHART.value,
        ]

    return [ContentType.TEXT.value]


# =========================================================
# Retrieved context formatting
# =========================================================

def format_chunk_for_context(chunk: dict[str, Any]) -> str:
    """
    Format one retrieved chunk into text context for the final LLM.

    This includes artifact paths as metadata, but does not send image bytes.
    """
    metadata = chunk.get("metadata", {}) or {}

    content_type = chunk.get("content_type") or metadata.get(
        "content_type",
        ContentType.TEXT.value,
    )

    paper = metadata.get("paper_name") or chunk.get("paper_name", "?")
    section = metadata.get("section") or chunk.get("section", "?")
    page = (
        metadata.get("page_number")
        or metadata.get("page_start")
        or chunk.get("page_start", "?")
    )
    caption = metadata.get("caption", "")

    if content_type == ContentType.TABLE.value:
        return f"""[table]
Paper: {paper}
Section: {section}
Page: {page}
Caption: {caption}

Table context:
{chunk.get("content", "")}

CSV path: {metadata.get("csv_path", "")}
HTML path: {metadata.get("html_path", "")}
JSON path: {metadata.get("json_path", "")}
Markdown path: {metadata.get("markdown_path", "")}
Image path: {metadata.get("image_path", "")}
Structured status: {metadata.get("structured_status", "")}
Display preference: {metadata.get("display_preference", "")}
Vision status: {metadata.get("vision_status", "")}
Vision description: {metadata.get("vision_description", "")}
""".strip()

    if content_type in {
        ContentType.FIGURE.value,
        ContentType.IMAGE.value,
        ContentType.DIAGRAM.value,
        ContentType.FLOWCHART.value,
    }:
        return f"""[{content_type}]
Paper: {paper}
Section: {section}
Page: {page}
Caption: {caption}

Visual context:
{chunk.get("content", "")}

Image path: {metadata.get("image_path", "")}
Vision status: {metadata.get("vision_status", "")}
Vision description: {metadata.get("vision_description", "")}
Mermaid status: {metadata.get("mermaid_status", "")}
Mermaid:
{metadata.get("mermaid", "")}
""".strip()

    return f"""[text]
Paper: {paper}
Section: {section}
Page: {page}

Content:
{chunk.get("content", "")}
""".strip()


def serialize_retrieved_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """
    Store retrieved chunk in output JSON with artifact paths preserved.
    Streamlit uses this to render sources.
    """
    metadata = chunk.get("metadata", {}) or {}

    content_type = chunk.get("content_type") or metadata.get(
        "content_type",
        ContentType.TEXT.value,
    )

    return {
        "similarity": chunk.get("similarity", ""),
        "content_type": content_type,
        "paper_name": metadata.get("paper_name", ""),
        "section": metadata.get("section", ""),
        "page_number": metadata.get(
            "page_number",
            metadata.get("page_start", ""),
        ),
        "page_start": metadata.get("page_start", ""),
        "page_end": metadata.get("page_end", ""),
        "caption": metadata.get("caption", ""),
        "image_path": metadata.get("image_path", ""),
        "csv_path": metadata.get("csv_path", ""),
        "html_path": metadata.get("html_path", ""),
        "json_path": metadata.get("json_path", ""),
        "markdown_path": metadata.get("markdown_path", ""),
        "table_id": metadata.get("table_id", ""),
        "figure_id": metadata.get("figure_id", ""),
        "structured_status": metadata.get("structured_status", ""),
        "display_preference": metadata.get("display_preference", ""),
        "vision_status": metadata.get("vision_status", ""),
        "vision_model": metadata.get("vision_model", ""),
        "vision_description": metadata.get("vision_description", ""),
        "mermaid": metadata.get("mermaid", ""),
        "mermaid_status": metadata.get("mermaid_status", ""),
        "metadata": metadata,
        "content": chunk.get("content", ""),
    }


# =========================================================
# Route classification
# =========================================================

def try_parse_route(raw_response: str) -> Optional[str]:
    cleaned = re.sub(r"```(?:json)?", "", raw_response)
    cleaned = cleaned.replace("```", "").strip()

    start = cleaned.find("{")
    if start == -1:
        return None

    try:
        parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError:
        return None

    route = parsed.get("route", "")
    return route if route in {"rag", "non_rag"} else None


def classify_route(
    question: str,
    llm,
    provider: str,
    model_name: str,
    memory: Optional[list[dict[str, Any]]] = None,
):
    routing_prompt = route_query(
        question,
        memory=memory,
    )

    route_timer = Timer()
    last_api_error: Optional[Exception] = None

    for attempt in range(1, MAX_ROUTE_RETRIES + 1):
        try:
            raw_response = llm.generate_content(routing_prompt).text.strip()
            last_api_error = None

        except Exception as exc:
            last_api_error = exc

            print(
                f"[router] Router LLM call failed "
                f"(attempt {attempt}/{MAX_ROUTE_RETRIES}): {exc!r}",
                flush=True,
            )

            if attempt < MAX_ROUTE_RETRIES:
                time.sleep(ROUTE_RETRY_DELAY)

            continue

        route = try_parse_route(raw_response)

        if route is not None:
            latency = route_timer.stop()

            routing_metrics = build_stage_metrics(
                prompt=routing_prompt,
                response=raw_response,
                provider=provider,
                model_name=model_name,
                latency=latency,
            )

            print(
                f"[router] Route decided: '{route}' (attempt {attempt})",
                flush=True,
            )

            return route, routing_metrics

        print(
            f"[router] Could not parse valid route "
            f"(attempt {attempt}/{MAX_ROUTE_RETRIES}): {raw_response!r}",
            flush=True,
        )

        if attempt < MAX_ROUTE_RETRIES:
            time.sleep(ROUTE_RETRY_DELAY)

    if last_api_error is not None:
        raise RuntimeError(
            f"[router] Router LLM call failed after {MAX_ROUTE_RETRIES} attempts. "
            f"Last error: {last_api_error!r}"
        ) from last_api_error

    print(
        f"[router] WARNING: Router returned invalid JSON on all "
        f"{MAX_ROUTE_RETRIES} attempts. Defaulting to non_rag.",
        flush=True,
    )

    latency = route_timer.stop()

    fallback_metrics = build_stage_metrics(
        prompt=routing_prompt,
        response="non_rag",
        provider=provider,
        model_name=model_name,
        latency=latency,
    )

    return "non_rag", fallback_metrics


# =========================================================
# Answer generation + metrics
# =========================================================

def build_empty_generation_metrics() -> dict[str, Any]:
    """
    Metrics used when no final LLM generation happens.
    Example: retrieval returns no chunks.
    """
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "time_taken_sec": 0.0,
    }


def generate_prompt_with_metrics(
    *,
    llm,
    prompt: str,
    provider: str,
    model_name: str,
) -> tuple[str, dict[str, Any]]:
    """
    Call LLM and build generation metrics.
    """
    generation_timer = Timer()

    answer = llm.generate_content(prompt).text

    generation_latency = generation_timer.stop()

    generation_metrics = build_stage_metrics(
        prompt=prompt,
        response=answer,
        provider=provider,
        model_name=model_name,
        latency=generation_latency,
    )

    return answer, generation_metrics


def generate_rag_answer_with_metrics(
    *,
    question: str,
    context: str,
    memory: list[dict[str, Any]] | None,
    llm,
    provider: str,
    model_name: str,
) -> tuple[str, dict[str, Any]]:
    """
    Build RAG prompt, call LLM, return answer + generation metrics.
    """
    final_prompt = summarize_rag(
        question,
        context,
        memory,
    )

    return generate_prompt_with_metrics(
        llm=llm,
        prompt=final_prompt,
        provider=provider,
        model_name=model_name,
    )


def generate_non_rag_answer_with_metrics(
    *,
    question: str,
    memory: list[dict[str, Any]] | None,
    llm,
    provider: str,
    model_name: str,
) -> tuple[str, dict[str, Any]]:
    """
    Build non-RAG prompt, call LLM, return answer + generation metrics.
    """
    final_prompt = summarize_non_rag(
        question,
        memory,
    )

    return generate_prompt_with_metrics(
        llm=llm,
        prompt=final_prompt,
        provider=provider,
        model_name=model_name,
    )


def build_overall_from_stage_metrics(
    *,
    routing_metrics: dict[str, Any],
    generation_metrics: dict[str, Any],
    total_pipeline_time: float,
) -> dict[str, Any]:
    """
    Build total pipeline metrics from routing + generation metrics.
    """
    return build_overall_metrics(
        routing_metrics,
        generation_metrics,
        total_pipeline_time,
    )