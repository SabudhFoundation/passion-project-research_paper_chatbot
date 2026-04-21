"""
main.py
Goal: Orchestrate routing → retrieval → answer generation.

Example:
    python main.py \\
        --question "What is the Transformer architecture?" \\
        --model_name llama-3.3-70b-versatile \\
        --temperature 0.1 \\
        --vector_store_path ../data/vector_store \\
        --output_json_path ../data/results/output.json \\
        --top_k 3 \\
        --embed_model Sentence-transformer model (default: all-MiniLM-L6-v2)

    python src/main.py --question "write a short note on transformer?" --vector_store_path data/vector_store --output_json_path data/results/output.json
"""
import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, field_validator

from prompt import route_query, summarize_non_rag, summarize_rag

load_dotenv()

MAX_ROUTE_RETRIES = 2
ROUTE_RETRY_DELAY = 0.75


class MainConfig(BaseModel):
    question: str
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    vector_store_path: Path
    output_json_path: Path
    top_k: int = Field(default=3, gt=0, le=20)
    embed_model: str = "all-MiniLM-L6-v2"

    @field_validator("vector_store_path")
    @classmethod
    def folder_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Vector store folder not found: {v}")
        return v


def parse_args() -> MainConfig:
    parser = argparse.ArgumentParser(description="RAG Pipeline — main entry point")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="llama-3.3-70b-versatile")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--vector_store_path", type=str, required=True)
    parser.add_argument("--output_json_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    return MainConfig(
        question=args.question,
        model_name=args.model_name,
        temperature=args.temperature,
        vector_store_path=Path(args.vector_store_path),
        output_json_path=Path(args.output_json_path),
        top_k=args.top_k,
        embed_model=args.embed_model,
    )


def build_llm(model_name: str, temperature: float) -> ChatGroq:
    from langchain_groq import ChatGroq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in environment / .env")
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=1024,
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


def classify_route(question: str, llm: ChatGroq) -> str:
    routing_prompt = route_query(question)
    last_api_error: Optional[Exception] = None

    for attempt in range(1, MAX_ROUTE_RETRIES + 1):
        try:
            raw_response = llm.invoke(routing_prompt).content.strip()
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


def _save_result(config: MainConfig, result: Dict[str, Any]) -> Path:
    out_path = config.output_json_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_dir():
        out_path = out_path / "result.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return out_path


def _error_result(config: MainConfig, error: str, elapsed: float) -> Dict[str, Any]:
    return {
        "question": config.question,
        "route": "error",
        "answer": "",
        "error": error,
        "model_name": config.model_name,
        "temperature": config.temperature,
        "vector_store_path": str(config.vector_store_path),
        "top_k": config.top_k,
        "time_taken_sec": elapsed,
        "retrieved_chunks": [],
    }


def run_pipeline(config: MainConfig) -> Dict[str, Any]:
    from prompt import route_query, summarize_non_rag, summarize_rag
    start = time.time()
    retrieved_chunks: list[dict[str, Any]] = []

    try:
        print("[INIT] Initializing LLM...", flush=True)
        llm = build_llm(config.model_name, config.temperature)

        print("\n[ROUTER] Sending request to LLM...", flush=True)
        t1 = time.time()
        route = classify_route(config.question, llm)
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
                answer = llm.invoke(summarize_rag(config.question, context)).content
                print(f"[LLM DONE] {round(time.time() - t3, 2)}s", flush=True)
        else:
            print("[LLM] Generating final answer...", flush=True)
            t3 = time.time()
            answer = llm.invoke(summarize_non_rag(config.question)).content
            print(f"[LLM DONE] {round(time.time() - t3, 2)}s", flush=True)

        elapsed = round(time.time() - start, 2)

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
        result = _error_result(config, str(exc), elapsed)

    out_path = _save_result(config, result)

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