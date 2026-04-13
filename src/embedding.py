"""
embedding.py
Goal: Load chunks JSON, generate embeddings, store in ChromaDB vector store.

Arguments:
    --chunk_file     : Path to chunks.json produced by chunking.py
    --vector_folder  : Path to persist the ChromaDB vector store

Returns:
    - Vector store persisted at vector_folder
    - Prints: vector store path, number of chunks processed, time taken
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import chromadb
import uuid
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field, validator


# -------------------- SCHEMA --------------------

class EmbedConfig(BaseModel):
    chunk_file:    Path
    vector_folder: Path
    model_name:    str  = "all-MiniLM-L6-v2"
    batch_size:    int  = Field(default=64, gt=0)

    @validator("chunk_file")
    def file_must_exist(cls, v):
        if not v.exists():
            raise ValueError(f"Chunk file not found: {v}")
        return v


# -------------------- ARG PARSER --------------------

def parse_args() -> EmbedConfig:
    parser = argparse.ArgumentParser(description="Generate embeddings and store in ChromaDB")
    parser.add_argument("--chunk_file",    type=str, required=True)
    parser.add_argument("--vector_folder", type=str, required=True)
    parser.add_argument("--model_name",    type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--batch_size",    type=int, default=64)
    args = parser.parse_args()

    return EmbedConfig(
        chunk_file=Path(args.chunk_file),
        vector_folder=Path(args.vector_folder),
        model_name=args.model_name,
        batch_size=args.batch_size,
    )


# -------------------- EMBEDDING MANAGER --------------------

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"Embedding dimension : {self.model.get_sentence_embedding_dimension()}")

    def generate(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        print(f"Embedding shape : {embeddings.shape}")
        return embeddings


# -------------------- VECTOR STORE --------------------

class VectorStore:
    def __init__(self, persist_directory: str, collection_name: str = "pdf_chunks"):
        self.persist_directory = persist_directory
        self.collection_name   = collection_name
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Research paper chunk embeddings"},
        )
        print(f"Collection       : {collection_name}")
        print(f"Existing docs    : {self.collection.count()}")

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        if len(chunks) != len(embeddings):
            raise ValueError("Mismatch: chunks vs embeddings count")

        ids, metadatas, documents, emb_list = [], [], [], []

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            doc_id = f"chunk_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            meta = {
                "chunk_id":    str(chunk.get("chunk_id", i)),
                "paper_name":  str(chunk.get("paper_name",  "")),
                "author":      str(chunk.get("author",      "")),
                "year":        str(chunk.get("year",        "")),
                "source_file": str(chunk.get("source_file", "")),
                "section":     str(chunk.get("section",     "")),
                "char_count":  str(chunk.get("char_count",  "")),
            }
            metadatas.append(meta)
            documents.append(chunk["text"])
            emb_list.append(emb.tolist())

        self.collection.add(
            ids=ids,
            embeddings=emb_list,
            metadatas=metadatas,
            documents=documents,
        )
        print(f"✓ Added {len(chunks)} chunks. Total in DB: {self.collection.count()}")


# -------------------- MAIN LOGIC --------------------

def embed_and_store(config: EmbedConfig):
    start = time.time()

    # Load chunks
    with open(config.chunk_file, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {config.chunk_file}\n")

    texts = [c["text"] for c in chunks]

    # Generate embeddings
    manager = EmbeddingManager(config.model_name)
    embeddings = manager.generate(texts, batch_size=config.batch_size)

    # Store
    store = VectorStore(str(config.vector_folder))
    store.add_chunks(chunks, embeddings)

    elapsed = round(time.time() - start, 2)

    print("\n========== SUMMARY ==========")
    print(f"Vector store path : {config.vector_folder}")
    print(f"Chunks processed  : {len(chunks)}")
    print(f"Time taken        : {elapsed}s")
    print("==============================")

    return {
        "vector_store_path": str(config.vector_folder),
        "chunks_processed":  len(chunks),
        "time_taken_sec":    elapsed,
    }


# -------------------- ENTRY --------------------

if __name__ == "__main__":
    config = parse_args()
    print("\nCONFIG:", config)
    embed_and_store(config)
