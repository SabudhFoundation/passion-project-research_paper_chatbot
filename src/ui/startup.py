from __future__ import annotations

import streamlit as st

from config import EMBED_MODEL


@st.cache_resource(show_spinner=False)
def _load_embedding_model() -> tuple[bool, str]:
    """
    Load embedding model once and cache it.

    Returns:
        (success, error_message)
    """
    try:
        from retriever import get_embedder

        get_embedder(EMBED_MODEL)
        return True, ""

    except Exception as exc:
        return False, str(exc)


def warmup_model_with_ui() -> None:
    """
    Show visible feedback while the embedding model loads.

    This prevents a blank/black screen feeling during startup.
    """
    if st.session_state.get("model_warmed_up"):
        return

    with st.status("Loading embedding model...", expanded=True) as status:
        st.write("Preparing the retriever for faster first answers.")

        ok, error_message = _load_embedding_model()

        if ok:
            st.session_state.model_warmed_up = True
            status.update(
                label="Model ready",
                state="complete",
                expanded=False,
            )
        else:
            if error_message:
                st.caption(f"Warmup skipped: {error_message}")
                
            status.update(
                label="Model warmup skipped",
                state="complete",
                expanded=False,
            )