import builtins
import os
import sys
import threading
import time
from pathlib import Path
from config import (
    EMBED_MODEL,
    MODEL_OPTIONS,
    METRICS_JSON_PATH,
    OUTPUT_JSON_PATH,
    TEMPERATURE,
    TOP_K,
    VECTOR_STORE_PATH,
)
import streamlit as st
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
load_dotenv()

from utilities import reset_output_file

if "output_file_initialized" not in st.session_state:
    reset_output_file(OUTPUT_JSON_PATH)
    st.session_state.output_file_initialized = True


@st.cache_resource
def warmup_model():
    from retriever import _get_embedder
    _get_embedder(EMBED_MODEL)

warmup_model()

st.set_page_config(page_title="Research Paper Chatbot", layout="centered")
st.markdown(
    """
    <style>
    div[data-baseweb="select"] input {
        caret-color: transparent !important;
        cursor: pointer !important;
    }
    div[data-baseweb="select"] [role="combobox"] {
        cursor: pointer !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Research Paper Chatbot")
st.write("Ask questions about your research papers.")

with st.sidebar:
    st.header("Settings")
    temperature = st.number_input(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(TEMPERATURE),
        step=0.05,
    )
    top_k = st.number_input(
        "Top K",
        min_value=1,
        max_value=20,
        value=int(TOP_K),
        step=1,
    )
    output_json_path = st.text_input(
        "Output JSON path",
        value=str(OUTPUT_JSON_PATH),
    )
    metrics_json_path = st.text_input(
        "Metrics JSON path",
        value=str(METRICS_JSON_PATH),
    )

with st.form("question_form"):
    selected_model_label = st.selectbox(
        "Select model",
        list(MODEL_OPTIONS.keys()),
        index=0,
    )
    provider_value, model_name = MODEL_OPTIONS[selected_model_label]

    question = st.text_area(
        "Your question",
        placeholder="e.g. What is transformer architecture?",
        height=110,
    )
    ask = st.form_submit_button("Ask", type="primary")

live_logs = []


def resolve_path_input(raw_value: str, default_path: Path) -> Path:
    cleaned_value = raw_value.strip()
    return Path(cleaned_value) if cleaned_value else default_path

def run_pipeline_with_logs(config):
    from main import run_pipeline
    original_print = builtins.print
    
    # State dictionary to track if we should stop logging
    state = {"ignore_rest": False}

    def capture_print(*args, **kwargs):
        msg = " ".join(map(str, args))
        
        # Stop logging after the result block begins.
        if "========== RESULT ==========" in msg:
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
        model_name=model_name,
        provider=provider_value,
        temperature=float(temperature),
        vector_store_path=VECTOR_STORE_PATH,
        output_json_path=resolve_path_input(output_json_path, OUTPUT_JSON_PATH),
        metrics_json_path=resolve_path_input(metrics_json_path, METRICS_JSON_PATH),
        top_k=int(top_k),
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

    if result.get("error"):
        st.subheader("Error")
        st.error(result["error"])
    else:
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