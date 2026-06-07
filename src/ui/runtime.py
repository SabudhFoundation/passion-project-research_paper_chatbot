from __future__ import annotations

import builtins
import threading
import time
from typing import Any, Callable

import streamlit as st


def run_pipeline_with_logs(config: Any, live_logs: list[str]) -> dict[str, Any]:
    """
    Capture print logs from main.run_pipeline() and show them in Streamlit.
    """
    from main import run_pipeline

    original_print = builtins.print
    state = {"ignore_rest": False}

    def capture_print(*args, **kwargs):
        message = " ".join(map(str, args))

        if "========== RESULT ==========" in message:
            state["ignore_rest"] = True

        if not state["ignore_rest"]:
            live_logs.append(message)

    builtins.print = capture_print

    try:
        return run_pipeline(config)
    finally:
        builtins.print = original_print


def run_blocking_with_live_logs(
    worker_fn: Callable[[], dict[str, Any]],
    live_logs: list[str],
    *,
    label: str = "Working...",
    success_label: str = "Done",
    error_label: str = "Failed",
) -> dict[str, Any]:
    """
    Run a blocking function in a thread while showing immediate UI feedback.
    """
    result_container: dict[str, Any] = {}

    def worker() -> None:
        try:
            result_container["result"] = worker_fn()
        except Exception as exc:
            result_container["result"] = {
                "error": str(exc),
            }

    thread = threading.Thread(target=worker)
    thread.start()

    with st.status(label, expanded=True) as status:
        log_placeholder = st.empty()
        last_rendered = ""

        while thread.is_alive():
            time.sleep(0.3)

            if live_logs:
                current_logs = "\n".join(live_logs[-25:])
            else:
                current_logs = "Starting..."

            if current_logs != last_rendered:
                log_placeholder.code(
                    current_logs,
                    language="text",
                )
                last_rendered = current_logs

        thread.join()

        result = result_container.get("result", {})

        if live_logs:
            log_placeholder.code(
                "\n".join(live_logs[-50:]),
                language="text",
            )

        if result.get("error"):
            status.update(
                label=error_label,
                state="error",
                expanded=True,
            )
        else:
            status.update(
                label=success_label,
                state="complete",
                expanded=False,
            )

    return result_container.get("result", {})