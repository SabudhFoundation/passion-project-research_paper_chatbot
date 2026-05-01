import json
import os
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator

load_dotenv()

# -------------------- SCHEMA --------------------

class QueryConfig(BaseModel):
    model_name:    str   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    temperature:   float = Field(default=0.1, ge=0.0, le=1.0)
    vector_folder: Path
    output_json:   Path
    top_k:         int   = Field(default=3, gt=0, le=20)
    embed_model:   str   = "all-MiniLM-L6-v2"

    @validator("vector_folder")
    def folder_must_exist(cls, v):
        if not v.exists():
            raise ValueError(f"Vector store folder not found: {v}")
        return v

# -------------------- ARG PARSER --------------------

def parse_args() -> QueryConfig:
    parser = argparse.ArgumentParser(description="RAG Interactive Chat")
    parser.add_argument("--question",      type=str,   default=None)
    parser.add_argument("--model_name",    type=str,   default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--temperature",   type=float, default=0.1)
    parser.add_argument("--vector_folder", type=str,   required=True)
    parser.add_argument("--output_json",   type=str,   required=True)
    parser.add_argument("--top_k",         type=int,   default=3)
    parser.add_argument("--embed_model",   type=str,   default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    return QueryConfig(
        model_name=args.model_name,
        temperature=args.temperature,
        vector_folder=Path(args.vector_folder),
        output_json=Path(args.output_json),
        top_k=args.top_k,
        embed_model=args.embed_model,
    )

# -------------------- RETRIEVER --------------------

class Retriever:
    def __init__(self, vector_folder: str, embed_model: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {embed_model}...")
        self.embedder = SentenceTransformer(embed_model)
        self.client = chromadb.PersistentClient(path=vector_folder)
        self.collection = self.client.get_or_create_collection("pdf_chunks")
        print(f"Vector store loaded. Total chunks in DB: {self.collection.count()}")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_emb = self.embedder.encode([query], convert_to_numpy=True)[0]
        results = self.collection.query(query_embeddings=[query_emb.tolist()], n_results=top_k)
        docs = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                docs.append({"content": doc, "metadata": meta, "similarity": round(1 - dist, 4)})
        return docs

# -------------------- LLM --------------------

def build_llm(model_name: str, temperature: float):
    print(f"Loading model from Hugging Face: {model_name}")
    hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
    pipe = pipeline(
        "text-generation",
        model=model_name,
        token=hf_token,
        max_new_tokens=256,   
        temperature=max(temperature, 0.01),
        do_sample=True,
    )
    return pipe

def llm_generate(llm, prompt: str, max_new_tokens: int = 512) -> str:
    messages = [{"role": "user", "content": prompt}]
    result = llm(messages, max_new_tokens=max_new_tokens)
    return result[0]["generated_text"][-1]["content"]

# -------------------- CHAT --------------------

def start_chat(config: QueryConfig):
    retriever = Retriever(str(config.vector_folder), config.embed_model)
    llm = build_llm(config.model_name, config.temperature)

    print("\n" + "="*50)
    print(f"Chatbot Ready! (Model: {config.model_name})")
    print("Type 'exit' or 'quit' to end.")
    print("="*50)

    while True:
        question = input("\n[User]: ")
        if question.lower() in ['exit', 'quit', 'bye']:
            break
        if not question.strip():
            continue

        start_time = time.time()
        chunks = retriever.retrieve(question, top_k=config.top_k)
        context = "\n\n".join([f"Context: {c['content']}" for c in chunks])
        prompt = f"Answer concisely based on context.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        response = llm_generate(llm, prompt)
        print(f"\n[Bot]: {response}")
        print(f"--- ({round(time.time() - start_time, 2)}s) ---")

if __name__ == "__main__":
    config = parse_args()
    start_chat(config)