import os
import sys
import json
import re
import time
from pathlib import Path
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(__file__))

from retriever import Retriever
from prompt import route_query, summarize_rag, summarize_non_rag
from query import build_llm, llm_generate

# -------------------- CONFIG --------------------

class MainConfig(BaseModel):
    question:          str
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    temperature:       float = Field(default=0.1, ge=0.0, le=1.0)
    vector_store_path: Path
    output_json_path:  Path
    top_k:             int   = Field(default=3, gt=0, le=20)
    embed_model:       str   = "all-MiniLM-L6-v2"

# -------------------- PIPELINE --------------------

def run_pipeline(config: MainConfig) -> dict:
    print("Starting RAG Pipeline...")

    # Step 1: Retrieve chunks
    print("\n[Step 1] Retrieving relevant chunks...")
    retriever = Retriever(
        vector_folder=str(config.vector_store_path),
        embed_model=config.embed_model
    )
    chunks = retriever.retrieve(config.question, top_k=config.top_k)

    # Step 2: Load LLM
    print("\n[Step 2] Loading Hugging Face LLM...")
    llm = build_llm(config.model_name, config.temperature)

    # Step 3: Route query
    print("\n[Step 3] Routing query...")
    route_prompt = route_query(config.question)
    route_response = llm_generate(llm, route_prompt, max_new_tokens=100)

    try:
        content = route_response.strip()
        match = re.search(r'\{.*?\}', content, re.DOTALL)
        if match:
            route_data = json.loads(match.group())
            route = route_data.get("route", "rag")
        else:
            route = "rag"
    except Exception:
        route = "rag"

    print(f"Route decided: {route}")

    # Step 4: Build prompt based on route
    print("\n[Step 4] Building prompt...")
    if route == "rag" and chunks:
        context = "\n\n".join([
            f"[Paper: {c['metadata'].get('paper_name', '?')} | "
            f"Section: {c['metadata'].get('section', '?')}]\n{c['content']}"
            for c in chunks
        ])
        prompt = summarize_rag(config.question, context)
    else:
        route = "non_rag"
        prompt = summarize_non_rag(config.question)

    # Step 5: Generate answer
    print("\n[Step 5] Generating answer using Hugging Face...")
    start = time.time()
    answer = llm_generate(llm, prompt, max_new_tokens=1024)
    elapsed = round(time.time() - start, 2)

    print(f"\n========== ANSWER ==========")
    print(answer)
    print(f"============================")
    print(f"Time taken: {elapsed}s")

    # Step 6: Save result
    result = {
        "question":         config.question,
        "answer":           answer,
        "route":            route,
        "model_name":       config.model_name,
        "temperature":      config.temperature,
        "time_taken_sec":   elapsed,
        "retrieved_chunks": [
            {
                "similarity": c["similarity"],
                "paper_name": c["metadata"].get("paper_name", ""),
                "section":    c["metadata"].get("section", ""),
                "content":    c["content"],
            }
            for c in chunks
        ],
    }

    out_path = config.output_json_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Result saved → {out_path}")
    return result


if __name__ == "__main__":
    config = MainConfig(
        question="What is attention mechanism?",
        vector_store_path=Path("data/vector_store"),
        output_json_path=Path("data/results/output.json"),
    )
    run_pipeline(config)