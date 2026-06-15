from __future__ import annotations

from typing import Any

import streamlit as st

from config import (
    MODEL_CAPABILITIES,
    MODEL_OPTIONS,
)
from ui.source_renderer import render_sources


def render_answer_result(result: dict[str, Any]) -> None:
    if result.get("error"):
        st.subheader("Error")
        st.error(result["error"])
    else:
        st.subheader("Answer")
        st.write(result.get("answer", ""))

    st.caption(f"Route: {result.get('route', 'unknown')}")
    render_sources(result.get("retrieved_chunks", []))


def model_selectbox(
    *,
    label: str,
    key: str,
    default_index: int = 0,
    options: dict[str, tuple[str, str]] | None = None,
) -> tuple[str, str]:
    model_options = options or MODEL_OPTIONS
    labels = list(model_options.keys())

    if not labels:
        raise ValueError("No model options available.")

    safe_index = min(max(default_index, 0), len(labels) - 1)

    selected_label = st.selectbox(
        label,
        labels,
        index=safe_index,
        key=key,
    )

    return model_options[selected_label]

def text_answer_model_options() -> dict[str, tuple[str, str]]:
    """
    Return models suitable for normal answer generation.

    Hides explicitly vision-labeled models from answer dropdowns.
    Vision models should appear only in vision preprocessing options.
    """
    options: dict[str, tuple[str, str]] = {}

    for label, value in MODEL_OPTIONS.items():
        provider, model_name = value
        capabilities = MODEL_CAPABILITIES.get(model_name, {})

        if capabilities.get("supports_text") is True and "vision" not in label.lower():
            options[label] = value

    return options

def vision_model_options() -> dict[str, tuple[str, str]]:
    """
    Return only models that support image input.

    Single source of truth:
    - MODEL_OPTIONS controls model labels shown in UI.
    - MODEL_CAPABILITIES controls supports_vision.
    """
    options: dict[str, tuple[str, str]] = {}

    for label, (provider, model_name) in MODEL_OPTIONS.items():
        capabilities = MODEL_CAPABILITIES.get(model_name, {})

        if capabilities.get("supports_vision") is True:
            options[label] = (provider, model_name)

    return options