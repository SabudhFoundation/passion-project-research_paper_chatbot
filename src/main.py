"""
main.py
Goal: Orchestrate routing → retrieval → answer generation.

Example:
    python src/main.py \\
        --question "What is the Transformer architecture?" \\
        --model_name  gemini-1.5-flash \\
        --temperature 0.1 \\
        --vector_store_path ../data/vector_store \\
        --output_json_path ../data/results/output.json \\
        --top_k 3 \\
        --embed_model Sentence-transformer model (default: all-MiniLM-L6-v2)

    python src/main.py --question "write a short note on transformer?" --vector_store_path data/vector_store --output_json_path data/results/output.json
"""
import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional
from config import (
    MODEL_NAME,
    TEMPERATURE,
    TOP_K,
    VECTOR_STORE_PATH,
    EMBED_MODEL,
    OUTPUT_JSON_PATH,
    MAX_ROUTE_RETRIES,
    ROUTE_RETRY_DELAY,
    MEMORY_WINDOW,
)
from dotenv import load_dotenv

from pydantic import BaseModel, Field, field_validator

from llm import build_llm
from prompt import route_query
from utilities import error_result, save_result

load_dotenv()

# Simple in-memory conversation buffer
CONVERSATION_MEMORY = []


class MainConfig(BaseModel):
    question: str
    model_name: str = MODEL_NAME
    provider: Optional[str] = None
    temperature: float = Field(default=TEMPERATURE, ge=0.0, le=1.0)
    vector_store_path: Path = VECTOR_STORE_PATH
    output_json_path: Path = OUTPUT_JSON_PATH
    top_k: int = Field(default=TOP_K, gt=0, le=20)
    embed_model: str = EMBED_MODEL

    @field_validator("vector_store_path")
    @classmethod
    def folder_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Vector store folder not found: {v}")
        return v


def parse_args() -> MainConfig:
    parser = argparse.ArgumentParser(description="RAG Pipeline — main entry point")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--vector_store_path", type=str, default=str(VECTOR_STORE_PATH))
    parser.add_argument("--output_json_path", type=str, default=str(OUTPUT_JSON_PATH))
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
        top_k=args.top_k,
        embed_model=args.embed_model,
    )


def _try_parse_route(raw_response: str) -> Optional[str]:
    cleaned = re.sub(r"```(?:json)?", "", raw_response).replace("```", "").strip()
    start = cleaned.find("{")
    if start == -1:
        return None

    try:
        parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError:
        return None

    route = parsed.get("route", "")
    return route if route in ("rag", "non_rag") else None


def classify_route(question: str, llm, memory=None) -> str:
    routing_prompt = route_query(question, memory=memory)
    last_api_error: Optional[Exception] = None

    for attempt in range(1, MAX_ROUTE_RETRIES + 1):
        try:
            raw_response = llm.generate_content(routing_prompt).text.strip()
            last_api_error = None
        except Exception as exc:
            last_api_error = exc
            print(
                f"[main] Router LLM call failed (attempt {attempt}/{MAX_ROUTE_RETRIES}): {exc!r}",
                flush=True,
            )
            if attempt < MAX_ROUTE_RETRIES:
                time.sleep(ROUTE_RETRY_DELAY)
            continue

        route = _try_parse_route(raw_response)
        if route is not None:
            print(f"[main] Route decided: '{route}' (attempt {attempt})", flush=True)
            return route

        print(
            f"[main] Could not parse a valid route from response "
            f"(attempt {attempt}/{MAX_ROUTE_RETRIES}): {raw_response!r}",
            flush=True,
        )
        if attempt < MAX_ROUTE_RETRIES:
            time.sleep(ROUTE_RETRY_DELAY)

    if last_api_error is not None:
        raise RuntimeError(
            f"[main] Router LLM call failed after {MAX_ROUTE_RETRIES} attempts. "
            f"Last error: {last_api_error!r}"
        ) from last_api_error

    print(
        f"[main] WARNING: Router returned unparseable JSON on all "
        f"{MAX_ROUTE_RETRIES} attempt(s). Defaulting to 'non_rag'.",
        flush=True,
    )
    return "non_rag"


def run_pipeline(config: MainConfig) -> Dict[str, Any]:
    from prompt import summarize_non_rag, summarize_rag
    start = time.time()
    retrieved_chunks: list[dict[str, Any]] = []

    try:
        print("[INIT] Initializing LLM...", flush=True)
        llm = build_llm(config.model_name, config.temperature, provider=config.provider)
        print(f"[INIT] LLM in use: provider={config.provider or 'auto'}, model={config.model_name}", flush=True)

        # Get last N interactions
        recent_memory = CONVERSATION_MEMORY[-MEMORY_WINDOW:]

        print("\n[ROUTER] Sending request to LLM...", flush=True)
        t1 = time.time()
        route = classify_route(config.question, llm, recent_memory)
        print(f"[ROUTER DONE] {round(time.time() - t1, 2)}s", flush=True)
        print(f"[ROUTE] → {route}", flush=True)

        if route == "rag":
            print("[RETRIEVER] Fetching relevant chunks...", flush=True)

            # Lazy import so non-RAG queries start fast
            from retriever import retrieve

            t2 = time.time()
            retrieved_chunks = retrieve(
                question=config.question,
                top_k=config.top_k,
                vector_store_folder_path=str(config.vector_store_path),
                embed_model=config.embed_model,
            )
            print(f"[RETRIEVER DONE] {round(time.time() - t2, 2)}s", flush=True)

            if not retrieved_chunks:
                answer = (
                    "No relevant documents were found in the knowledge base for your question. "
                    "Please rephrase it or ask about a different paper or method."
                )
            else:
                context = "\n\n".join(
                    f"[Source: {chunk['metadata'].get('paper_name', '?')} | "
                    f"Section: {chunk['metadata'].get('section', '?')}]\n{chunk['content']}"
                    for chunk in retrieved_chunks
                )

                print("[LLM] Generating final answer...", flush=True)
                t3 = time.time()
                answer = llm.generate_content(
                    summarize_rag(config.question, context, recent_memory)
                ).text
                print(f"[LLM DONE] {round(time.time() - t3, 2)}s", flush=True)
        else:
            print("[LLM] Generating final answer...", flush=True)
            t3 = time.time()
            answer = llm.generate_content(
                summarize_non_rag(config.question, recent_memory)
            ).text
            print(f"[LLM DONE] {round(time.time() - t3, 2)}s", flush=True)

        elapsed = round(time.time() - start, 2)

        # Store current interaction

        CONVERSATION_MEMORY.append({
            "question": config.question,
            "answer": answer,
            "route": route,
        })

        # Keep memory bounded
        if len(CONVERSATION_MEMORY) > MEMORY_WINDOW:
            CONVERSATION_MEMORY.pop(0)

        result: Dict[str, Any] = {
            "question": config.question,
            "route": route,
            "answer": answer,
            "model_name": config.model_name,
            "temperature": config.temperature,
            "vector_store_path": str(config.vector_store_path),
            "top_k": config.top_k,
            "time_taken_sec": elapsed,
            "retrieved_chunks": [
                {
                    "similarity": chunk["similarity"],
                    "paper_name": chunk["metadata"].get("paper_name", ""),
                    "section": chunk["metadata"].get("section", ""),
                    "content": chunk["content"],
                }
                for chunk in retrieved_chunks
            ],
        }

    except Exception as exc:
        elapsed = round(time.time() - start, 2)
        result = error_result(
            question=config.question,
            model_name=config.model_name,
            temperature=config.temperature,
            vector_store_path=config.vector_store_path,
            top_k=config.top_k,
            elapsed=elapsed,
            error=str(exc),
        )

    out_path = save_result(config.output_json_path, result)

    print("\n========== ANSWER ==========", flush=True)
    print(result.get("answer", ""), flush=True)
    print("============================", flush=True)
    print(f"Route        : {result.get('route')}", flush=True)
    print(f"Result saved : {out_path}", flush=True)
    print(f"Time taken   : {result.get('time_taken_sec')}s", flush=True)

    return result


if __name__ == "__main__":
    print("[BOOT] Starting main.py...", flush=True)
    config = parse_args()
    print(f"[CONFIG] {config}", flush=True)
    run_pipeline(config)