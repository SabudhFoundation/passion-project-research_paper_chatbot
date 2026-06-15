"""
ui/theme.py

Centralized Streamlit UI styling.

This file only contains visual styling helpers:
- app-wide CSS
- header rendering
- section card rendering
"""

from __future__ import annotations

import html

import streamlit as st


def apply_app_theme() -> None:
    """
    Apply custom CSS for a clean professional UI.

    Main accent color: #285C87.
    """
    st.markdown(
        """
        <style>
        /* =========================================================
           Main page spacing
        ========================================================= */

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        /* =========================================================
           Sidebar polish
        ========================================================= */

        div[data-testid="stSidebar"] {
            border-right: 1px solid rgba(120, 120, 120, 0.18);
        }

        /* =========================================================
           Header card
        ========================================================= */

        .app-hero {
            padding: 1.25rem 1.35rem;
            border-radius: 18px;
            background: linear-gradient(
                135deg,
                rgba(40, 92, 135, 0.18),
                rgba(40, 92, 135, 0.08)
            );
            border: 1px solid rgba(40, 92, 135, 0.35);
            margin-bottom: 1.25rem;
        }

        .app-title {
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
            letter-spacing: -0.02em;
        }

        .app-subtitle {
            font-size: 1rem;
            opacity: 0.78;
            margin-bottom: 0;
            line-height: 1.5;
        }

        /* =========================================================
           Section cards
        ========================================================= */

        .section-card {
            padding: 1.1rem 1.2rem;
            border-radius: 16px;
            border: 1px solid rgba(40, 92, 135, 0.30);
            background: rgba(40, 92, 135, 0.06);
            margin-bottom: 1rem;
        }

        .section-card h3 {
            margin-top: 0;
            margin-bottom: 0.35rem;
            font-weight: 750;
        }

        .section-card p {
            margin-bottom: 0;
            opacity: 0.75;
            line-height: 1.5;
        }

        /* =========================================================
           Selectbox cursor fix
        ========================================================= */

        div[data-baseweb="select"] input {
            caret-color: transparent !important;
            cursor: pointer !important;
        }

        div[data-baseweb="select"] [role="combobox"] {
            cursor: pointer !important;
        }

        /* =========================================================
           Inputs
        ========================================================= */

        textarea {
            border-radius: 14px !important;
        }

        input {
            border-radius: 10px !important;
        }

        /* =========================================================
           Buttons
        ========================================================= */

        .stButton > button,
        .stFormSubmitButton > button,
        div[data-testid="stFormSubmitButton"] button {
            border-radius: 999px !important;
            font-weight: 700 !important;
            padding: 0.45rem 1.2rem !important;
            transition: all 0.15s ease-in-out !important;
        }

        button[data-testid="stBaseButton-primary"],
        button[data-testid="stFormSubmitButton-primary"],
        .stButton button[kind="primary"],
        .stFormSubmitButton button[kind="primary"],
        div[data-testid="stFormSubmitButton"] button[kind="primary"] {
            background-color: #285C87 !important;
            border-color: #285C87 !important;
            color: #FFFFFF !important;
        }

        button[data-testid="stBaseButton-primary"]:hover,
        button[data-testid="stFormSubmitButton-primary"]:hover,
        .stButton button[kind="primary"]:hover,
        .stFormSubmitButton button[kind="primary"]:hover,
        div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
            background-color: #1F496B !important;
            border-color: #1F496B !important;
            color: #FFFFFF !important;
            transform: translateY(-1px);
        }

        button[data-testid="stBaseButton-primary"]:focus,
        button[data-testid="stFormSubmitButton-primary"]:focus,
        .stButton button[kind="primary"]:focus,
        .stFormSubmitButton button[kind="primary"]:focus,
        div[data-testid="stFormSubmitButton"] button[kind="primary"]:focus {
            box-shadow: 0 0 0 0.2rem rgba(40, 92, 135, 0.30) !important;
        }

        button[data-testid="stBaseButton-primary"]:active,
        button[data-testid="stFormSubmitButton-primary"]:active,
        .stButton button[kind="primary"]:active,
        .stFormSubmitButton button[kind="primary"]:active,
        div[data-testid="stFormSubmitButton"] button[kind="primary"]:active {
            background-color: #183852 !important;
            border-color: #183852 !important;
            color: #FFFFFF !important;
        }

        button[data-testid="stBaseButton-secondary"]:hover,
        .stButton button[kind="secondary"]:hover {
            border-color: #285C87 !important;
            color: #285C87 !important;
        }

        /* =========================================================
           Alerts / messages
        ========================================================= */

        div[data-testid="stAlert"] {
            border-radius: 14px !important;
        }

        div[data-testid="stAlert"] div[role="alert"] {
            border-color: rgba(40, 92, 135, 0.45) !important;
            background-color: rgba(40, 92, 135, 0.12) !important;
            color: #FFFFFF !important;
        }

        div[data-testid="stAlert"] svg {
            fill: #285C87 !important;
            color: #285C87 !important;
        }

        /* Extra fallback for success alerts */
        div[data-testid="stAlert"][kind="success"],
        div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
            color: #FFFFFF !important;
        }

        /* =========================================================
           Radio tabs
        ========================================================= */

        div[role="radiogroup"] {
            gap: 0.5rem;
        }

        div[role="radiogroup"] label {
            border-radius: 999px;
            padding: 0.25rem 0.6rem;
        }

        /* =========================================================
           Status blocks
        ========================================================= */

        div[data-testid="stStatusWidget"] {
            border-radius: 14px;
        }

        /* =========================================================
           Expander polish
        ========================================================= */

        details {
            border-radius: 14px !important;
        }

        /* =========================================================
           Code blocks
        ========================================================= */

        code {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header() -> None:
    """
    Render the main app header.
    """
    st.markdown(
        """
        <div class="app-hero">
            <div class="app-title">Research Paper Chatbot</div>
            <p class="app-subtitle">
                Ask questions across your knowledge base or upload PDFs for isolated document chat.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_card(title: str, description: str) -> None:
    """
    Render a clean section intro card.

    Escapes strings before inserting into HTML.
    """
    safe_title = html.escape(title)
    safe_description = html.escape(description)

    st.markdown(
        f"""
        <div class="section-card">
            <h3>{safe_title}</h3>
            <p>{safe_description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )