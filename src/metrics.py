from __future__ import annotations

import json
import time

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any

import tiktoken

from config import MODEL_PRICING


# =========================================================
# Tokenizer fallback mapping
# =========================================================

TOKENIZER_FALLBACKS = {
    "gemini": "cl100k_base",
    "groq": "cl100k_base",
    "huggingface": "cl100k_base",
}


# =========================================================
# Tokenizer
# =========================================================

def get_tokenizer(
    model_name: str,
    provider: str,
):

    try:
        return tiktoken.encoding_for_model(model_name)

    except Exception:

        fallback = TOKENIZER_FALLBACKS.get(
            provider,
            "cl100k_base",
        )

        return tiktoken.get_encoding(fallback)


def count_tokens(
    text: str,
    model_name: str,
    provider: str,
):

    tokenizer = get_tokenizer(
        model_name,
        provider,
    )

    return len(tokenizer.encode(text))


# =========================================================
# Cost Calculation
# =========================================================

def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model_name: str,
):
    pricing = MODEL_PRICING.get(model_name, {})

    unit_tokens = pricing.get("unit_tokens")
    input_price = pricing.get("input_price", 0.0)
    output_price = pricing.get("output_price", 0.0)

    # If unit_tokens is missing or invalid (<=0), treat costs as zero.
    if not unit_tokens or unit_tokens <= 0:
        return {
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
        }

    input_cost = (input_tokens / unit_tokens) * input_price
    output_cost = (output_tokens / unit_tokens) * output_price

    return {
        "input_cost": round(input_cost, 8),
        "output_cost": round(output_cost, 8),
        "total_cost": round(input_cost + output_cost, 8),
    }


# =========================================================
# Timer
# =========================================================

class Timer:

    def __init__(self):

        self.start = time.time()

    def stop(self):

        return round(
            time.time() - self.start,
            3,
        )


# =========================================================
# Stage Metrics
# =========================================================

@dataclass
class StageMetrics:

    input_tokens: int
    output_tokens: int
    total_tokens: int

    input_cost: float
    output_cost: float
    total_cost: float

    time_taken_sec: float

    def to_dict(self):

        return asdict(self)


def build_stage_metrics(
    prompt: str,
    response: str,
    provider: str,
    model_name: str,
    latency: float,
):

    input_tokens = count_tokens(
        prompt,
        model_name,
        provider,
    )

    output_tokens = count_tokens(
        response,
        model_name,
        provider,
    )

    total_tokens = input_tokens + output_tokens

    costs = calculate_cost(
        input_tokens,
        output_tokens,
        model_name,
    )

    metrics = StageMetrics(

        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,

        input_cost=costs["input_cost"],
        output_cost=costs["output_cost"],
        total_cost=costs["total_cost"],

        time_taken_sec=latency,
    )

    return metrics.to_dict()


# =========================================================
# Overall Metrics
# =========================================================

def build_overall_metrics(
    routing_metrics: Dict[str, Any],
    generation_metrics: Dict[str, Any],
    total_pipeline_time: float,
):

    return {

        "total_input_tokens": (
            routing_metrics["input_tokens"]
            + generation_metrics["input_tokens"]
        ),

        "total_output_tokens": (
            routing_metrics["output_tokens"]
            + generation_metrics["output_tokens"]
        ),

        "total_tokens": (
            (routing_metrics.get("total_tokens")
             or routing_metrics.get("input_tokens", 0) + routing_metrics.get("output_tokens", 0))
            + (generation_metrics.get("total_tokens")
               or generation_metrics.get("input_tokens", 0) + generation_metrics.get("output_tokens", 0))
        ),

        "grand_total_cost": round(
            routing_metrics["total_cost"]
            + generation_metrics["total_cost"],
            8,
        ),

        "total_pipeline_time_sec": round(
            total_pipeline_time,
            3,
        ),
    }


# =========================================================
# Save Metrics
# =========================================================

def save_metrics(
    metrics: Dict[str, Any],
    path: Path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = []

    if path.exists():

        try:

            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)

        except Exception:

            existing = []

    existing.append(metrics)

    with open(path, "w", encoding="utf-8") as f:

        json.dump(
            existing,
            f,
            indent=2,
            ensure_ascii=False,
        )