"""
app.py

Streamlit UI for the research paper chatbot.

Modes:
1. Knowledge Base Chat
2. Uploaded PDF Chat

This file contains mostly UI layout and user interaction code.

Heavy logic is delegated to:
- main.py
- uploaded_pdf_pipeline.py
- ui/components.py
- ui/runtime.py
- ui/state.py
- ui/startup.py
- ui/path_inputs.py
- ui/theme.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import streamlit as st
from dotenv import load_dotenv

# =========================================================
# Project path setup
# =========================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

for path in [CURRENT_DIR, PROJECT_ROOT]:
    if str(path) not in sys.path:
        sys.path.append(str(path))


# =========================================================
# Imports
# =========================================================

from config import (
    EMBED_MODEL,
    MEMORY_WINDOW,
    METRICS_JSON_PATH,
    MODEL_CAPABILITIES,
    OUTPUT_JSON_PATH,
    TEMPERATURE,
    TOP_K,
    UPLOAD_CHUNK_OVERLAP,
    UPLOAD_CHUNK_SIZE,
    UPLOAD_METRICS_JSON_PATH,
    UPLOAD_OUTPUT_JSON_PATH,
    VECTOR_STORE_PATH,
)
from ui.components import (
    model_selectbox,
    render_answer_result,
    text_answer_model_options,
    vision_model_options,
)
from ui.path_inputs import resolve_path_input
from ui.runtime import (
    run_blocking_with_live_logs,
    run_pipeline_with_logs,
)
from ui.startup import warmup_model_with_ui
from ui.state import (
    append_uploaded_pdf_memory,
    init_session_state,
)
from ui.theme import (
    apply_app_theme,
    render_app_header,
    render_section_card,
)


load_dotenv()


# =========================================================
# Page setup
# =========================================================

st.set_page_config(
    page_title="Research Paper Chatbot",
    layout="wide",
)


# =========================================================
# Startup UI
# =========================================================

init_session_state()
apply_app_theme()
render_app_header()

# Render visible UI first, then warm up the model.
# This prevents the app from looking blank/black during startup.
warmup_model_with_ui()


# =========================================================
# Small local UI helpers
# =========================================================

def keep_upload_tab_active() -> None:
    """
    Keep Uploaded PDF Chat selected after Streamlit reruns caused by upload widgets.
    """
    st.session_state.active_chat_tab = "Uploaded PDF Chat"


def normalize_uploaded_files_ui(uploaded_files: Any) -> list[Any]:
    """
    Lightweight UploadedFile normalization.

    Avoids importing uploaded_pdf_pipeline.py at startup.
    """
    if uploaded_files is None:
        return []

    if isinstance(uploaded_files, list):
        return uploaded_files

    return [uploaded_files]

def run_inline_with_live_logs(
    worker_fn: Callable[[], dict[str, Any]],
    live_logs: list[str],
) -> dict[str, Any]:
    """
    Run a blocking worker in a background thread and show live logs inside
    an expander. The expander disappears after processing finishes because
    the page reruns after successful processing.
    """
    result_container: dict[str, Any] = {}

    with st.expander("Uploaded PDF processing logs", expanded=True):
        log_placeholder = st.empty()

        def worker() -> None:
            try:
                result_container["result"] = worker_fn()
            except Exception as exc:
                result_container["result"] = {
                    "error": str(exc),
                }

        thread = threading.Thread(target=worker)
        thread.start()

        while thread.is_alive():
            time.sleep(0.3)

            visible_logs = live_logs[-25:]

            if visible_logs:
                log_placeholder.markdown(
                    "\n".join(f"- {message}" for message in visible_logs)
                )
            else:
                log_placeholder.markdown("Preparing uploaded PDF processing...")

        thread.join()

        visible_logs = live_logs[-25:]

        if visible_logs:
            log_placeholder.markdown(
                "\n".join(f"- {message}" for message in visible_logs)
            )

    return result_container.get("result", {})
    
# =========================================================
# Sidebar settings
# =========================================================

with st.sidebar:
    st.header("Settings")
    st.caption("Control model behavior, retrieval depth, and output locations.")

    temperature = st.number_input(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(TEMPERATURE),
        step=0.05,
        help="Lower values give more deterministic answers.",
    )

    top_k = st.number_input(
        "Top K",
        min_value=1,
        max_value=30,
        value=int(TOP_K),
        step=1,
        help="Number of retrieved chunks to use for answering.",
    )

    st.divider()

    output_json_path = st.text_input(
        "Knowledge Base output JSON path",
        value=str(OUTPUT_JSON_PATH),
        help="Used only by Knowledge Base Chat.",
    )

    metrics_json_path = st.text_input(
        "Knowledge Base metrics JSON path",
        value=str(METRICS_JSON_PATH),
        help="Used only by Knowledge Base Chat.",
    )

    st.divider()

    st.caption("Uploaded PDF outputs are saved separately:")

    uploaded_output_json_path = st.text_input(
        "Uploaded PDF output JSON path",
        value=str(UPLOAD_OUTPUT_JSON_PATH),
        help="Used only by Uploaded PDF Chat.",
    )

    uploaded_metrics_json_path = st.text_input(
        "Uploaded PDF metrics JSON path",
        value=str(UPLOAD_METRICS_JSON_PATH),
        help="Used only by Uploaded PDF Chat.",
    )


# =========================================================
# Stateful mode selector
# =========================================================

if "active_chat_tab" not in st.session_state:
    st.session_state.active_chat_tab = "Knowledge Base Chat"

selected_tab = st.radio(
    "Select chat mode",
    [
        "Knowledge Base Chat",
        "Uploaded PDF Chat",
    ],
    horizontal=True,
    key="active_chat_tab",
    label_visibility="collapsed",
)


# =========================================================
# Tab 1: Knowledge Base Chat
# =========================================================

if selected_tab == "Knowledge Base Chat":
    render_section_card(
        "Knowledge Base Chat",
        "Ask questions from your already-built research-paper vector store.",
    )

    with st.form("kb_question_form"):
        provider_value, model_name = model_selectbox(
            label="Select answer model",
            key="kb_model_select",
            default_index=0,
            options=text_answer_model_options(),
        )

        question = st.text_area(
            "Your question",
            placeholder="e.g. Explain Figure 4 using the visual information.",
            height=120,
            key="kb_question",
        )

        ask = st.form_submit_button(
            "Ask Knowledge Base",
            type="primary",
            use_container_width=True,
        )

    if ask:
        if not question.strip():
            st.warning("Please enter a question.")

        elif not VECTOR_STORE_PATH.exists():
            st.error(f"Vector store not found: {VECTOR_STORE_PATH}")

        else:
            from main import MainConfig

            config = MainConfig(
                question=question.strip(),
                model_name=model_name,
                provider=provider_value,
                temperature=float(temperature),
                vector_store_path=VECTOR_STORE_PATH,
                output_json_path=resolve_path_input(
                    output_json_path,
                    OUTPUT_JSON_PATH,
                ),
                metrics_json_path=resolve_path_input(
                    metrics_json_path,
                    METRICS_JSON_PATH,
                ),
                top_k=int(top_k),
                embed_model=EMBED_MODEL,
            )

            live_logs: list[str] = []

            def worker_fn() -> dict[str, Any]:
                return run_pipeline_with_logs(
                    config,
                    live_logs,
                )

            result = run_blocking_with_live_logs(
                worker_fn,
                live_logs,
                label="Answering from Knowledge Base...",
                success_label="Knowledge Base answer ready",
                error_label="Knowledge Base answer failed",
            )

            render_answer_result(result)


# =========================================================
# Tab 2: Uploaded PDF Chat
# =========================================================

if selected_tab == "Uploaded PDF Chat":
    render_section_card(
        "Uploaded PDF Chat",
        "Upload PDFs, build an isolated vector store, and ask questions only from those files.",
    )

    uploaded_files_raw = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="uploaded_pdf_files",
        help="Uploaded PDFs are indexed separately from the Knowledge Base.",
        on_change=keep_upload_tab_active,
    )

    uploaded_files = normalize_uploaded_files_ui(uploaded_files_raw)

    upload_provider_value, upload_model_name = model_selectbox(
        label="Select answer model",
        key="upload_answer_model_select",
        default_index=0,
        options=text_answer_model_options(),
    )

    # -----------------------------------------------------
    # Upload processing options
    # -----------------------------------------------------

    with st.expander("Upload processing options", expanded=False):
        force_rebuild = st.checkbox(
            "Force rebuild uploaded PDF index",
            value=False,
            key="force_rebuild_upload_index",
            help="Reprocess the uploaded PDF(s), even if an index already exists.",
            on_change=keep_upload_tab_active,
        )

        generate_upload_vision = st.checkbox(
            "Generate vision descriptions for uploaded tables/figures",
            value=False,
            help=(
                "Slower and uses vision-model quota. Keep this off for normal "
                "text/table QA. Turn it on only when you need image, figure, "
                "or diagram understanding."
            ),
            key="generate_upload_vision",
            on_change=keep_upload_tab_active,
        )

        col_a, col_b = st.columns(2)

        with col_a:
            upload_chunk_size = st.number_input(
                "Upload chunk size",
                min_value=300,
                max_value=2000,
                value=UPLOAD_CHUNK_SIZE,
                step=100,
                key="upload_chunk_size",
                on_change=keep_upload_tab_active,
            )

        with col_b:
            upload_chunk_overlap = st.number_input(
                "Upload chunk overlap",
                min_value=0,
                max_value=500,
                value=UPLOAD_CHUNK_OVERLAP,
                step=25,
                key="upload_chunk_overlap",
                on_change=keep_upload_tab_active,
            )

        vision_provider_value: str | None = None
        vision_model_name: str | None = None

        if generate_upload_vision:
            selected_answer_capabilities = MODEL_CAPABILITIES.get(
                upload_model_name,
                {},
            )

            if selected_answer_capabilities.get("supports_vision") is True:
                vision_provider_value = upload_provider_value
                vision_model_name = upload_model_name

                st.caption(
                    "Vision preprocessing will use the selected answer model: "
                    f"{vision_provider_value} / {vision_model_name}"
                )

            else:
                vision_options = vision_model_options()

                if not vision_options:
                    st.warning(
                        "No vision-capable models are configured. "
                        "Check MODEL_CAPABILITIES in config.py."
                    )

                else:
                    selected_vision_label = st.selectbox(
                        "Select vision preprocessing model",
                        list(vision_options.keys()),
                        index=0,
                        key="upload_vision_model_select",
                        on_change=keep_upload_tab_active,
                    )

                    vision_provider_value, vision_model_name = vision_options[
                        selected_vision_label
                    ]

                    st.caption(
                        "Selected answer model is text-only, so vision preprocessing "
                        f"will use: {vision_provider_value} / {vision_model_name}"
                    )

        else:
            st.caption("Vision preprocessing is disabled.")

    # -----------------------------------------------------
    # Upload state
    # -----------------------------------------------------

    if not uploaded_files:
        st.info("Upload one or more PDFs to start.")

    else:
        # Lazy import to avoid slowing down initial app render.
        from uploaded_pdf_pipeline import uploaded_files_signature

        current_signature = uploaded_files_signature(uploaded_files)
        previous_signature = st.session_state.get("uploaded_pdf_signature")

        current_index_matches_upload = (
            previous_signature == current_signature
            and "uploaded_pdf_index" in st.session_state
        )


        needs_new_index = not current_index_matches_upload
        show_process_button = needs_new_index or force_rebuild

        # -------------------------------------------------
        # Process uploaded PDFs
        # -------------------------------------------------

        if show_process_button:
            process_label = (
                "Rebuild uploaded PDF index"
                if force_rebuild and current_index_matches_upload
                else "Process uploaded PDF(s)"
            )

            if st.button(
                process_label,
                type="primary",
                use_container_width=True,
            ):
                
                live_logs: list[str] = []

                def worker_fn() -> dict[str, Any]:
                    from uploaded_pdf_pipeline import build_uploaded_pdf_index

                    index = build_uploaded_pdf_index(
                        uploaded_files,
                        chunk_size=int(upload_chunk_size),
                        chunk_overlap=int(upload_chunk_overlap),
                        embed_model=EMBED_MODEL,
                        force_rebuild=force_rebuild,
                        generate_vision=generate_upload_vision,
                        vision_provider=vision_provider_value,
                        vision_model_name=vision_model_name,
                        progress_callback=live_logs.append,
                    )

                    return {
                        "index": index,
                    }

                process_result = run_inline_with_live_logs(
                    worker_fn,
                    live_logs,
                )

                if process_result.get("error"):
                    st.error(process_result["error"])

                else:
                    index = process_result.get("index")

                    if index is None or not hasattr(index, "vector_store_path"):
                        st.error("Upload processing failed: no index was returned.")

                    else:
                        st.session_state.uploaded_pdf_signature = current_signature
                        st.session_state.uploaded_pdf_index = index
                        st.session_state.uploaded_pdf_memory = []

                        current_index_matches_upload = True
                        needs_new_index = False

                        st.caption(
                            f"Processed {index.pdf_count} PDF(s), created {index.chunk_count} chunks."
                        )

                        st.rerun()

            elif needs_new_index:
                st.info("Click **Process uploaded PDF(s)** before asking questions.")

        # -------------------------------------------------
        # Uploaded PDF chat
        # -------------------------------------------------

        if current_index_matches_upload and "uploaded_pdf_index" in st.session_state:
            index = st.session_state.uploaded_pdf_index

            with st.form("uploaded_pdf_question_form"):
                uploaded_question = st.text_area(
                    "Ask a question about the uploaded PDF(s)",
                    placeholder=(
                        "e.g. Summarize the uploaded PDF. "
                        "What does Table 2 say?"
                    ),
                    height=120,
                    key="uploaded_pdf_question",
                )

                ask_uploaded = st.form_submit_button(
                    "Ask uploaded PDF(s)",
                    type="primary",
                    use_container_width=True,
                )

            if ask_uploaded:

                if not uploaded_question.strip():
                    st.warning("Please enter a question.")

                else:
                    recent_uploaded_memory = st.session_state.uploaded_pdf_memory[
                        -MEMORY_WINDOW:
                    ]

                    with st.status(
                        "Preparing answer...",
                        expanded=True,
                    ) as status:
                        try:
                            from uploaded_pdf_pipeline import run_uploaded_pdf_question

                            def upload_progress_log(message: str) -> None:
                                st.write(message)

                            result = run_uploaded_pdf_question(
                                question=uploaded_question.strip(),
                                index=index,
                                provider=upload_provider_value,
                                model_name=upload_model_name,
                                temperature=float(temperature),
                                top_k=int(top_k),
                                embed_model=EMBED_MODEL,
                                memory=recent_uploaded_memory,
                                output_json_path=resolve_path_input(
                                    uploaded_output_json_path,
                                    UPLOAD_OUTPUT_JSON_PATH,
                                ),
                                metrics_json_path=resolve_path_input(
                                    uploaded_metrics_json_path,
                                    UPLOAD_METRICS_JSON_PATH,
                                ),
                                progress_callback=upload_progress_log,
                            )

                            success_label = (
                                "Direct answer generated"
                                if result.get("route") == "non_rag"
                                else "Grounded answer generated from uploaded PDF(s)"
                            )

                            status.update(
                                label=success_label,
                                state="complete",
                                expanded=False,
                            )

                        except Exception as exc:
                            result = {
                                "error": str(exc),
                                "route": "uploaded_pdf",
                            }

                            status.update(
                                label="Uploaded PDF answer failed",
                                state="error",
                                expanded=True,
                            )

                    render_answer_result(result)

                    append_uploaded_pdf_memory(
                        question=uploaded_question.strip(),
                        result=result,
                    )