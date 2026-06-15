"""
main.py

Knowledge Base RAG pipeline:
- route query
- retrieve typed chunks
- generate answer
- save result + metrics

Supports multimodal RAG context:
- text chunks
- table chunks
- figure / image / diagram / flowchart chunks

Important:
This file does NOT send images directly to the LLM during normal QA.
Images are displayed in UI and can be described earlier by vision_describe.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from config import (
    EMBED_MODEL,
    MEMORY_WINDOW,
    METRICS_JSON_PATH,
    MODEL_NAME,
    OUTPUT_JSON_PATH,
    TEMPERATURE,
    TOP_K,
    VECTOR_STORE_PATH,
)
from llm import build_llm, normalize_provider_name
from metrics import save_metrics
from rag_common import (
    build_empty_generation_metrics,
    build_overall_from_stage_metrics,
    classify_route,
    decide_content_types,
    format_chunk_for_context,
    generate_non_rag_answer_with_metrics,
    generate_rag_answer_with_metrics,
    serialize_retrieved_chunk,
)
from utilities import error_result, save_result

load_dotenv()


# =========================================================
# Conversation memory
# =========================================================

CONVERSATION_MEMORY: list[dict[str, Any]] = []


# =========================================================
# CLI config
# =========================================================

class MainConfig(BaseModel):
    question: str
    model_name: str = MODEL_NAME
    provider: Optional[str] = None
    temperature: float = Field(default=TEMPERATURE, ge=0.0, le=1.0)
    vector_store_path: Path = VECTOR_STORE_PATH
    output_json_path: Path = OUTPUT_JSON_PATH
    metrics_json_path: Path = METRICS_JSON_PATH
    top_k: int = Field(default=TOP_K, gt=0, le=30)
    embed_model: str = EMBED_MODEL

    @field_validator("vector_store_path")
    @classmethod
    def folder_must_exist(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"Vector store folder not found: {value}")
        return value


def parse_args() -> MainConfig:
    parser = argparse.ArgumentParser(description="RAG pipeline main entry point")

    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--vector_store_path", type=str, default=str(VECTOR_STORE_PATH))
    parser.add_argument("--output_json_path", type=str, default=str(OUTPUT_JSON_PATH))
    parser.add_argument("--metrics_json_path", type=str, default=str(METRICS_JSON_PATH))
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--embed_model", type=str, default=EMBED_MODEL)

    args = parser.parse_args()

    return MainConfig(
        question=args.question,
        model_name=args.model_name,
        provider=args.provider,
        temperature=args.temperature,
        vector_store_path=Path(args.vector_store_path),
        output_json_path=Path(args.output_json_path),
        metrics_json_path=Path(args.metrics_json_path),
        top_k=args.top_k,
        embed_model=args.embed_model,
    )


# =========================================================
# Pipeline
# =========================================================

def run_pipeline(config: MainConfig) -> Dict[str, Any]:
    start = time.time()

    retrieved_chunks: list[dict[str, Any]] = []
    route = "error"
    resolved_provider = config.provider or "unknown"

    routing_metrics: dict[str, Any] = {}
    generation_metrics: dict[str, Any] = {}
    overall_metrics: dict[str, Any] = {}

    try:
        resolved_provider = normalize_provider_name(
            config.provider,
            config.model_name,
        )

        print("[INIT] Initializing LLM...", flush=True)

        llm = build_llm(
            config.model_name,
            config.temperature,
            provider=config.provider,
        )

        print(
            f"[INIT] LLM in use: provider={config.provider or 'auto'}, "
            f"model={config.model_name}",
            flush=True,
        )

        recent_memory = CONVERSATION_MEMORY[-MEMORY_WINDOW:]

        print("\n[ROUTER] Sending request to LLM...", flush=True)

        route, routing_metrics = classify_route(
            config.question,
            llm,
            provider=resolved_provider,
            model_name=config.model_name,
            memory=recent_memory,
        )

        print(f"[ROUTER DONE] {routing_metrics['time_taken_sec']}s", flush=True)
        print(f"[ROUTE] → {route}", flush=True)

        if route == "rag":
            from retriever import retrieve

            print("[RETRIEVER] Fetching relevant chunks...", flush=True)

            retrieve_start = time.time()
            content_types = decide_content_types(config.question)

            print(f"[RETRIEVER] Content type filter: {content_types}", flush=True)

            retrieved_chunks = retrieve(
                question=config.question,
                top_k=config.top_k,
                vector_store_folder_path=str(config.vector_store_path),
                embed_model=config.embed_model,
                content_types=content_types,
            )

            print(
                f"[RETRIEVER DONE] {round(time.time() - retrieve_start, 2)}s",
                flush=True,
            )

            if not retrieved_chunks:
                answer = (
                    "No relevant documents were found in the knowledge base for your "
                    "question. Please rephrase it or ask about a different paper or method."
                )

                generation_metrics = build_empty_generation_metrics()

            else:
                context = "\n\n".join(
                    format_chunk_for_context(chunk)
                    for chunk in retrieved_chunks
                )

                print("[LLM] Generating final answer...", flush=True)

                answer, generation_metrics = generate_rag_answer_with_metrics(
                    question=config.question,
                    context=context,
                    memory=recent_memory,
                    llm=llm,
                    provider=resolved_provider,
                    model_name=config.model_name,
                )

                print(
                    f"[LLM DONE] {generation_metrics['time_taken_sec']}s",
                    flush=True,
                )

        else:
            print("[LLM] Generating final answer...", flush=True)

            answer, generation_metrics = generate_non_rag_answer_with_metrics(
                question=config.question,
                memory=recent_memory,
                llm=llm,
                provider=resolved_provider,
                model_name=config.model_name,
            )

            print(
                f"[LLM DONE] {generation_metrics['time_taken_sec']}s",
                flush=True,
            )

        elapsed = round(time.time() - start, 2)

        overall_metrics = build_overall_from_stage_metrics(
            routing_metrics=routing_metrics,
            generation_metrics=generation_metrics,
            total_pipeline_time=elapsed,
        )

        CONVERSATION_MEMORY.append(
            {
                "question": config.question,
                "answer": answer,
                "route": route,
            }
        )

        if len(CONVERSATION_MEMORY) > MEMORY_WINDOW:
            CONVERSATION_MEMORY.pop(0)

        result: Dict[str, Any] = {
            "question": config.question,
            "route": route,
            "answer": answer,
            "provider": resolved_provider,
            "model_name": config.model_name,
            "temperature": config.temperature,
            "vector_store_path": str(config.vector_store_path),
            "top_k": config.top_k,
            "time_taken_sec": elapsed,
            "retrieved_chunks": [
                serialize_retrieved_chunk(chunk)
                for chunk in retrieved_chunks
            ],
        }

    except Exception as exc:
        elapsed = round(time.time() - start, 2)

        overall_metrics = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "grand_total_cost": 0.0,
            "total_pipeline_time_sec": elapsed,
        }

        result = error_result(
            question=config.question,
            provider=config.provider,
            model_name=config.model_name,
            temperature=config.temperature,
            vector_store_path=config.vector_store_path,
            top_k=config.top_k,
            elapsed=elapsed,
            error=str(exc),
        )

    out_path = save_result(
        config.output_json_path,
        result,
        append=True,
    )

    metrics_payload = {
        "question": config.question,
        "route": route,
        "provider": resolved_provider,
        "model_name": config.model_name,
        "routing_metrics": routing_metrics,
        "generation_metrics": generation_metrics,
        "overall_metrics": overall_metrics,
    }

    save_metrics(
        metrics_payload,
        config.metrics_json_path,
    )

    print("\n========== RESULT ==========", flush=True)

    if result.get("error"):
        print(f"ERROR: {result['error']}", flush=True)
    else:
        print(result.get("answer", ""), flush=True)

    print("============================", flush=True)
    print(f"Route        : {result.get('route')}", flush=True)
    print(f"Result saved : {out_path}", flush=True)
    print(f"Time taken   : {result.get('time_taken_sec')}s", flush=True)

    print("\n========== METRICS ==========", flush=True)
    print(json.dumps(metrics_payload, indent=2), flush=True)
    print("=============================", flush=True)

    return result


if __name__ == "__main__":
    print("[BOOT] Starting main.py...", flush=True)

    config = parse_args()

    print(f"[CONFIG] {config}", flush=True)

    run_pipeline(config)