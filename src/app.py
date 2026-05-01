import builtins
import os
import sys
import threading
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
load_dotenv()

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TEMPERATURE = 0.1
TOP_K = 3
VECTOR_STORE_PATH = Path("data/vector_store")
EMBED_MODEL = "all-MiniLM-L6-v2"
OUTPUT_JSON_PATH = Path("data/results/output.json")

@st.cache_resource
def warmup_model():
    from retriever import _get_embedder
    _get_embedder(EMBED_MODEL)

warmup_model()

st.set_page_config(page_title="Research Paper Chatbot", layout="centered")
st.title("Research Paper Chatbot")
st.write("Ask questions about your research papers.")

question = st.text_area(
    "Your question",
    placeholder="e.g. What is transformer architecture?",
    height=110,
)
ask = st.button("Ask", type="primary")

live_logs = []

def run_pipeline_with_logs(config):
    from main import run_pipeline
    original_print = builtins.print
    
    # State dictionary to track if we should stop logging
    state = {"ignore_rest": False}

    def capture_print(*args, **kwargs):
        msg = " ".join(map(str, args))
        
        # 2. Filter out the final answer block and everything after it
        if "========== ANSWER ==========" in msg:
            state["ignore_rest"] = True
            
        if not state["ignore_rest"]:
            live_logs.append(msg)

    builtins.print = capture_print
    try:
        return run_pipeline(config)
    finally:
        builtins.print = original_print

if ask:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    if not VECTOR_STORE_PATH.exists():
        st.error(f"Vector store not found: {VECTOR_STORE_PATH}")
        st.stop()

    from main import MainConfig

    config = MainConfig(
        question=question.strip(),
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        vector_store_path=VECTOR_STORE_PATH,
        output_json_path=OUTPUT_JSON_PATH,
        top_k=TOP_K,
        embed_model=EMBED_MODEL,
    )

    result_container = {}
    live_logs.clear()
    log_placeholder = None
    thinking_visible = False

    def worker():
        result_container["result"] = run_pipeline_with_logs(config)

    thread = threading.Thread(target=worker)
    thread.start()

    while thread.is_alive():
        time.sleep(0.3)
        if live_logs:
            if not thinking_visible:
                with st.expander("Thinking", expanded=True):
                    log_placeholder = st.empty()
                thinking_visible = True
            log_placeholder.code("\n".join(live_logs[-20:]), language="text")

    thread.join()

    result = result_container.get("result", {})
    
    if thinking_visible and live_logs:
        log_placeholder.code("\n".join(live_logs), language="text")

    st.subheader("Answer")
    st.write(result.get("answer", ""))
    st.caption(f"Route: {result.get('route', 'unknown')}")

    chunks = result.get("retrieved_chunks", [])
    if chunks:
        st.subheader("Sources (Exact Chunks Used)")
        for i, chunk in enumerate(chunks, start=1):
            title = f"{i}. {chunk.get('paper_name', 'Unknown')} | {chunk.get('section', 'Unknown')}"
            with st.expander(title):
                st.write(f"Similarity: {chunk.get('similarity', '')}")
                st.markdown("**Full Chunk:**")
                st.write(chunk.get("content", ""))