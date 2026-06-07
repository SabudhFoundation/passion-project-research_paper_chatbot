"""
ui/source_renderer.py

Streamlit source renderer for retrieved chunks.

Supports:
- text chunks
- table chunks with hidden expanders for actual PNG / extracted HTML / downloads
- figure / image / diagram / flowchart chunks with PNG
- meaningful source titles for text chunks

No RAG logic.
No LLM calls.
No retrieval.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from core.constants import ContentType, SOURCE_TYPE_LABELS
from core.paths import resolve_project_path

import streamlit.components.v1 as components


def render_mermaid_chart(mermaid_code: str, *, height: int = 420) -> None:
    """
    Render Mermaid.js syntax as an actual diagram in Streamlit.

    Falls back safely if the Mermaid code is empty.
    """
    mermaid_code = (mermaid_code or "").strip()

    if not mermaid_code:
        return

    html = f"""
    <div class="mermaid">
    {mermaid_code}
    </div>

    <script type="module">
        import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
        mermaid.initialize({{
            startOnLoad: true,
            theme: "default",
            securityLevel: "loose"
        }});
    </script>
    """

    components.html(
        html,
        height=height,
        scrolling=True,
    )

# =========================================================
# Generic helpers
# =========================================================

def chunk_value(chunk: dict[str, Any], key: str, default: str = "") -> Any:
    """
    Read from top-level retrieved chunk first, then metadata.
    """
    value = chunk.get(key, None)
    if value not in (None, ""):
        return value

    metadata = chunk.get("metadata") or {}
    if isinstance(metadata, dict):
        value = metadata.get(key, None)
        if value not in (None, ""):
            return value

    return default


def source_type_label(content_type: str) -> str:
    return SOURCE_TYPE_LABELS.get(content_type, content_type.title())


def clean_source_preview(text: str, max_len: int = 115) -> str:
    """
    Clean a source preview so it is useful as an expander heading.
    """
    text = " ".join(str(text or "").split())

    if len(text) > max_len:
        return text[: max_len - 3] + "..."

    return text


# =========================================================
# Text preview helpers
# =========================================================

def extract_text_content_body(content: str) -> str:
    """
    Extract only the real passage text from formatted text chunks.

    main.py formats text chunks like:
    [text]
    Paper: ...
    Section: ...
    Page: ...

    Content:
    actual passage...
    """
    content = content or ""

    marker = "Content:"
    if marker in content:
        return content.split(marker, 1)[1].strip()

    return content.strip()


def is_bad_preview_line(line: str) -> bool:
    """
    Avoid ugly/noisy source headings.
    """
    line = " ".join(str(line or "").split())

    if not line:
        return True

    lowered = line.lower()

    bad_prefixes = (
        "[text]",
        "[table]",
        "[figure]",
        "[diagram]",
        "content type:",
        "paper:",
        "section:",
        "page:",
        "caption:",
        "columns:",
        "markdown table:",
        "table context:",
        "visual context:",
        "csv path:",
        "html path:",
        "json path:",
        "markdown path:",
        "image path:",
        "extraction method:",
        "extraction confidence:",
        "structured status:",
        "display preference:",
        "quality reason:",
        "description:",
        "mermaid status:",
        "mermaid:",
        "vision status:",
        "vision model:",
        "vision description:",
        "content:",
    )

    if lowered.startswith(bad_prefixes):
        return True

    bad_fragments = (
        "downloaded from",
        "all rights reserved",
        "copyright",
        "http://",
        "https://",
        "www.",
        "doi:",
        "@",
    )

    if any(fragment in lowered for fragment in bad_fragments):
        return True

    # Avoid references/citation-only headings.
    if lowered.startswith("[") and "]" in lowered[:8]:
        return True

    # Avoid very short/noisy lines.
    alpha_count = sum(ch.isalpha() for ch in line)
    if alpha_count < 20:
        return True

    return False


def first_meaningful_text_preview(content: str) -> str:
    """
    Pick a useful first sentence/line from a text chunk.
    """
    body = extract_text_content_body(content)

    lines = [
        " ".join(line.strip().split())
        for line in body.splitlines()
        if line.strip()
    ]

    candidates = [
        line
        for line in lines
        if not is_bad_preview_line(line)
    ]

    if not candidates:
        return "Retrieved text passage"

    first = candidates[0]

    # Prefer sentence-like preview when possible.
    sentence_parts = re.split(r"(?<=[.!?])\s+", first)
    if sentence_parts and len(sentence_parts[0]) >= 45:
        return clean_source_preview(sentence_parts[0])

    return clean_source_preview(first)


def make_text_source_label(chunk: dict[str, Any]) -> str:
    """
    Create a meaningful text-source title.

    Priority:
    1. Section + useful passage preview
    2. Useful passage preview
    3. Section name
    4. Generic fallback
    """
    content = chunk.get("content", "") or ""
    section = chunk_value(chunk, "section", "")

    preview = first_meaningful_text_preview(content)

    if section and section not in {"Body", "Unknown"}:
        if preview and preview != "Retrieved text passage":
            return clean_source_preview(f"{section}: {preview}")
        return clean_source_preview(section)

    return clean_source_preview(preview)


# =========================================================
# Source title / metadata
# =========================================================

def source_title(index: int, chunk: dict[str, Any]) -> str:
    content_type = chunk_value(chunk, "content_type", ContentType.TEXT.value)
    caption = chunk_value(chunk, "caption", "")
    page = (
        chunk_value(chunk, "page_number", "")
        or chunk_value(chunk, "page_start", "")
    )

    if content_type == ContentType.TABLE.value:
        main_label = caption or "Retrieved table source"

    elif content_type in {
        ContentType.FIGURE.value,
        ContentType.IMAGE.value,
        ContentType.DIAGRAM.value,
        ContentType.FLOWCHART.value,
    }:
        main_label = caption or "Retrieved visual source"

    else:
        main_label = make_text_source_label(chunk)

    main_label = " ".join(str(main_label).split())

    if len(main_label) > 110:
        main_label = main_label[:107] + "..."

    page_part = f" | page {page}" if page else ""
    return f"{index}. {main_label}{page_part}"


def render_metadata(chunk: dict[str, Any]) -> None:
    paper_name = chunk_value(chunk, "paper_name", "Unknown")
    section = chunk_value(chunk, "section", "")
    page = chunk_value(chunk, "page_number", "") or chunk_value(chunk, "page_start", "")
    similarity = chunk.get("similarity", "")

    st.caption(f"Paper: {paper_name}")

    if section and section not in {"Body", "Unknown"}:
        st.caption(f"Section: {section}")

    if page:
        st.caption(f"Page: {page}")

    if similarity != "":
        st.caption(f"Similarity: {similarity}")


# =========================================================
# Table rendering
# =========================================================

def render_html_table(html_path: Path, fallback_text: str = "") -> None:
    try:
        html_content = html_path.read_text(encoding="utf-8")

        wrapped_html = f"""
        <div style="background-color: white; color: black; padding: 12px; border-radius: 6px;">
        <style>
        table {{
            border-collapse: collapse;
            width: 100%;
            color: black;
            background-color: white;
            font-size: 14px;
        }}
        caption {{
            color: black;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        th {{
            color: black;
            background-color: #f2f2f2;
            border: 1px solid #777;
            padding: 6px;
        }}
        td {{
            color: black;
            background-color: white;
            border: 1px solid #777;
            padding: 6px;
        }}
        </style>
        {html_content}
        </div>
        """

        components.html(wrapped_html, height=420, scrolling=True)

    except Exception:
        if fallback_text:
            st.write(fallback_text)
        else:
            st.info("HTML table artifact exists, but could not be loaded.")


def render_table_source(chunk: dict[str, Any], index: int) -> None:
    """
    Render table source without showing actual/extracted table immediately.

    User sees:
    - caption
    - metadata
    - expander: View actual table
    - expander: View extracted table
    - expander: Download table files
    - expander: View retrieved context
    """
    caption = chunk_value(chunk, "caption", "")
    content = chunk.get("content", "") or ""

    structured_status = chunk_value(chunk, "structured_status", "")
    display_preference = chunk_value(chunk, "display_preference", "")
    extraction_method = chunk_value(chunk, "extraction_method", "")

    image_path = resolve_project_path(chunk_value(chunk, "image_path", ""))
    html_path = resolve_project_path(chunk_value(chunk, "html_path", ""))
    csv_path = resolve_project_path(chunk_value(chunk, "csv_path", ""))
    json_path = resolve_project_path(chunk_value(chunk, "json_path", ""))
    markdown_path = resolve_project_path(chunk_value(chunk, "markdown_path", ""))

    if caption:
        st.markdown("**Caption**")
        st.write(caption)

    meta_parts = []
    if structured_status:
        meta_parts.append(f"status: `{structured_status}`")
    if display_preference:
        meta_parts.append(f"display: `{display_preference}`")
    if extraction_method:
        meta_parts.append(f"method: `{extraction_method}`")

    if meta_parts:
        st.caption(" | ".join(meta_parts))

    with st.expander("View actual table", expanded=False):
        if image_path:
            st.image(
                str(image_path),
                caption=caption or "Actual table from PDF",
                use_container_width=True,
            )
        else:
            st.info("Actual table image was not available for this source.")

    with st.expander("View extracted table", expanded=False):
        if html_path:
            render_html_table(html_path, fallback_text=content)
        elif content:
            st.text_area(
                label="Extracted table text",
                value=content,
                height=240,
                key=f"extracted_table_text_{index}",
                label_visibility="collapsed",
                disabled=True,
            )
        else:
            st.info("Extracted table artifact was not available for this source.")

    has_download = any([csv_path, json_path, markdown_path])

    if has_download:
        with st.expander("Download table files", expanded=False):
            if csv_path:
                try:
                    st.download_button(
                        label="Download CSV",
                        data=csv_path.read_bytes(),
                        file_name=csv_path.name,
                        mime="text/csv",
                        key=f"download_table_csv_{index}_{csv_path.name}",
                    )
                except Exception:
                    st.info("CSV artifact exists, but could not be loaded.")

            if json_path:
                try:
                    st.download_button(
                        label="Download JSON",
                        data=json_path.read_bytes(),
                        file_name=json_path.name,
                        mime="application/json",
                        key=f"download_table_json_{index}_{json_path.name}",
                    )
                except Exception:
                    st.info("JSON artifact exists, but could not be loaded.")

            if markdown_path:
                try:
                    st.download_button(
                        label="Download Markdown",
                        data=markdown_path.read_bytes(),
                        file_name=markdown_path.name,
                        mime="text/markdown",
                        key=f"download_table_md_{index}_{markdown_path.name}",
                    )
                except Exception:
                    st.info("Markdown artifact exists, but could not be loaded.")

    vision_description = chunk_value(chunk, "vision_description", "")
    if vision_description:
        with st.expander("View vision description", expanded=False):
            st.write(vision_description)

    if content:
        with st.expander("View retrieved context", expanded=False):
            st.text_area(
                label="Table chunk text",
                value=content,
                height=240,
                key=f"table_context_{index}",
                label_visibility="collapsed",
                disabled=True,
            )


# =========================================================
# Visual rendering
# =========================================================

def render_visual_source(chunk: dict[str, Any], index: int) -> None:
    content_type = chunk_value(chunk, "content_type", ContentType.FIGURE.value)
    caption = chunk_value(chunk, "caption", "")
    content = chunk.get("content", "") or ""
    image_path = resolve_project_path(chunk_value(chunk, "image_path", ""))
    mermaid = chunk_value(chunk, "mermaid", "")

    if caption:
        st.markdown("**Caption**")
        st.write(caption)

    label = (
        "Diagram / Flowchart"
        if content_type in {ContentType.DIAGRAM.value, ContentType.FLOWCHART.value}
        else "Figure / Image"
    )

    with st.expander(f"View actual {label.lower()}", expanded=False):
        if image_path:
            st.image(
                str(image_path),
                caption=caption or label,
                use_container_width=True,
            )
        else:
            st.info("No image artifact path was available for this source.")

    vision_description = chunk_value(chunk, "vision_description", "")
    if vision_description:
        with st.expander("View vision description", expanded=False):
            st.write(vision_description)

    if mermaid:
        st.markdown("**Mermaid diagram**")
        render_mermaid_chart(mermaid)

        with st.expander("View Mermaid source", expanded=False):
            st.code(mermaid, language="mermaid")

    if content:
        with st.expander("View retrieved context", expanded=False):
            st.text_area(
                label="Visual chunk text",
                value=content,
                height=220,
                key=f"visual_context_{index}",
                label_visibility="collapsed",
                disabled=True,
            )


# =========================================================
# Text rendering
# =========================================================

def render_text_source(chunk: dict[str, Any], index: int) -> None:
    content = chunk.get("content", "") or ""
    preview = first_meaningful_text_preview(content)

    st.caption(f"Preview: {preview}")

    with st.expander("View full text passage", expanded=False):
        st.text_area(
            label="Text chunk",
            value=extract_text_content_body(content) or content,
            height=260,
            key=f"text_context_{index}",
            label_visibility="collapsed",
            disabled=True,
        )


# =========================================================
# Main source renderer
# =========================================================

def render_sources(chunks: list[dict[str, Any]]) -> None:
    """
    Unified source renderer for text, tables, figures, images, diagrams, and flowcharts.
    """
    if not chunks:
        return

    st.subheader("Sources")

    for index, chunk in enumerate(chunks, start=1):
        content_type = chunk_value(chunk, "content_type", ContentType.TEXT.value)
        title = source_title(index, chunk)

        with st.expander(title, expanded=index == 1):
            st.caption(f"Source type: {source_type_label(content_type)}")
            render_metadata(chunk)

            if content_type == ContentType.TABLE.value:
                render_table_source(chunk, index)

            elif content_type in {
                ContentType.FIGURE.value,
                ContentType.IMAGE.value,
                ContentType.DIAGRAM.value,
                ContentType.FLOWCHART.value,
            }:
                render_visual_source(chunk, index)

            else:
                render_text_source(chunk, index)