from __future__ import annotations

from typing import Any

import streamlit as st

from config import MEMORY_WINDOW


def init_session_state() -> None:
    if "output_file_initialized" not in st.session_state:
        st.session_state.output_file_initialized = True

    if "uploaded_pdf_memory" not in st.session_state:
        st.session_state.uploaded_pdf_memory = []


def append_uploaded_pdf_memory(
    *,
    question: str,
    result: dict[str, Any],
) -> None:
    """
    Store one uploaded-PDF conversation turn.

    Kept separate from Knowledge Base Chat memory.
    """
    if result.get("error"):
        return

    st.session_state.uploaded_pdf_memory.append(
        {
            "question": question,
            "answer": result.get("answer", ""),
            "route": result.get("route", "uploaded_pdf"),
        }
    )

    st.session_state.uploaded_pdf_memory = st.session_state.uploaded_pdf_memory[
        -MEMORY_WINDOW:
    ]