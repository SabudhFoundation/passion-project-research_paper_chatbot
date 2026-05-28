from pathlib import Path

# Model & LLM settings
MODEL_NAME = "gemini-2.5-flash" # Backend default model used by main.py / CLI fallback.

MODEL_OPTIONS: dict[str, tuple[str, str]] = {
    "Gemini: gemini-2.5-flash": (
        "gemini",
        "gemini-2.5-flash",
    ),

    "Groq: llama-3.3-70b-versatile": (
        "groq",
        "llama-3.3-70b-versatile",
    ),
	
    "Hugging Face: Qwen2.5-72B": (
        "huggingface",
        "Qwen/Qwen2.5-72B-Instruct",
    ),

    "Hugging Face: Qwen2.5-7B": (
        "huggingface",
        "Qwen/Qwen2.5-7B-Instruct",
    ),

}

TEMPERATURE = 0.1
TOP_K = 10
EMBED_MODEL = "all-MiniLM-L6-v2"

# Paths
VECTOR_STORE_PATH = Path("data/vector_store")
OUTPUT_JSON_PATH = Path("data/results/output.json")
METRICS_JSON_PATH = Path("data/results/metrics.json") # Path to save metrics results

# Retry settings
MAX_ROUTE_RETRIES = 2
ROUTE_RETRY_DELAY = 0.75

# Memory settings
MEMORY_WINDOW = 3  # number of previous interactions to consider

# Model Pricing
# Cost per 1M tokens for input and output, based on current pricing from providers.

MODEL_PRICING = {
    "gemini-2.5-flash": {
        "input_price": 0.30,
        "output_price": 2.50,
        "unit_tokens": 1000000,
    },

    "llama-3.3-70b-versatile": {
        "input_price": 0.59,
        "output_price": 0.79,
        "unit_tokens": 1000000,
    },

    "Qwen/Qwen2.5-72B-Instruct": {
        "input_price": 0.380,
        "output_price": 0.400,
        "unit_tokens": 1000000,
    },

    "Qwen/Qwen2.5-7B-Instruct": {
        "input_price": 0.040,
        "output_price": 0.100,
        "unit_tokens": 1000000,
    },

}
