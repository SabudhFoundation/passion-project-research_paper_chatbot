from pathlib import Path

# =========================================================
# Model & LLM settings
# =========================================================

MODEL_NAME = "gemini-2.5-flash"

# =========================================================
# Chunking settings
# =========================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_SECTION_CHARS = 60

# Uploaded PDF chunking uses the same defaults by default.
UPLOAD_CHUNK_SIZE = CHUNK_SIZE
UPLOAD_CHUNK_OVERLAP = CHUNK_OVERLAP

MODEL_OPTIONS: dict[str, tuple[str, str]] = {
    "Gemini: gemini-2.5-flash": (
        "gemini",
        "gemini-2.5-flash",
    ),
    "Groq: llama-3.3-70b-versatile": (
        "groq",
        "llama-3.3-70b-versatile",
    ),
    "Groq Vision: llama-4-scout-17b-16e-instruct": (
        "groq",
        "meta-llama/llama-4-scout-17b-16e-instruct",
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

# Single source of truth for model capabilities.
# Text models continue working normally.
# Vision support is used by vision_describe.py and optionally by Uploaded PDF Chat preprocessing.
MODEL_CAPABILITIES: dict[str, dict[str, object]] = {
    "gemini-2.5-flash": {
        "provider": "gemini",
        "supports_text": True,
        "supports_vision": True,
    },
    "llama-3.3-70b-versatile": {
        "provider": "groq",
        "supports_text": True,
        "supports_vision": False,
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "provider": "groq",
        "supports_text": True,
        "supports_vision": True,
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "provider": "huggingface",
        "supports_text": True,
        "supports_vision": False,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "provider": "huggingface",
        "supports_text": True,
        "supports_vision": False,
    },
}

TEMPERATURE = 0.1
TOP_K = 10
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "pdf_chunks"

# =========================================================
# Paths
# =========================================================

VECTOR_STORE_PATH = Path("data/vector_store")
OUTPUT_JSON_PATH = Path("data/results/output.json")
METRICS_JSON_PATH = Path("data/results/metrics.json")

ARTIFACTS_PATH = Path("data/artifacts")
TABLES_ARTIFACT_PATH = ARTIFACTS_PATH / "tables"
FIGURES_ARTIFACT_PATH = ARTIFACTS_PATH / "figures"
VISION_CACHE_PATH = ARTIFACTS_PATH / "vision_cache"

# =========================================================
# Uploaded PDF Paths
# =========================================================

UPLOAD_BASE_PATH = Path("data/uploaded_pdf")

UPLOAD_PARSED_INDIVIDUAL_PATH = UPLOAD_BASE_PATH / "parsed_individual"
UPLOAD_PARSED_COMBINED_PATH = UPLOAD_BASE_PATH / "parsed_combined"
UPLOAD_CHUNKS_PATH = UPLOAD_BASE_PATH / "chunks"

UPLOAD_VECTOR_STORE_PATH = UPLOAD_BASE_PATH / "vector_store"
UPLOAD_OUTPUT_JSON_PATH = UPLOAD_BASE_PATH / "results/output.json"
UPLOAD_METRICS_JSON_PATH = UPLOAD_BASE_PATH / "results/metrics.json"

UPLOAD_ARTIFACTS_PATH = UPLOAD_BASE_PATH / "artifacts"
UPLOAD_TABLES_ARTIFACT_PATH = UPLOAD_ARTIFACTS_PATH / "tables"
UPLOAD_FIGURES_ARTIFACT_PATH = UPLOAD_ARTIFACTS_PATH / "figures"
UPLOAD_VISION_CACHE_PATH = UPLOAD_ARTIFACTS_PATH / "vision_cache"

UPLOADS_PATH = UPLOAD_BASE_PATH / "uploads"

UPLOAD_COMBINED_JSON_PATH = UPLOAD_PARSED_COMBINED_PATH / "combined_papers.json"
UPLOAD_CHUNKS_JSON_PATH = UPLOAD_CHUNKS_PATH / "chunks.json"
UPLOAD_VECTOR_MARKER_PATH = UPLOAD_BASE_PATH / "vector_store_ready.json"
UPLOAD_INDEX_MANIFEST_PATH = UPLOAD_BASE_PATH / "index_manifest.json"

# =========================================================
# Multimodal feature flags
# =========================================================

ENABLE_TABLE_EXTRACTION = True
ENABLE_BASIC_IMAGE_EXTRACTION = True
ENABLE_FIGURE_REGION_CROPPING = True

# Vision is a separate preprocessing step.
# Keep disabled by default so normal parsing stays cheap and stable.
ENABLE_VISION_DESCRIPTIONS = False

# Mermaid generation is optional and experimental.
# If enabled, vision_describe.py may ask the vision model to generate Mermaid syntax
# for diagrams, workflows, architectures, and flowcharts.
# Streamlit can render saved Mermaid syntax through ui/source_renderer.py.
# Keep disabled by default to avoid extra hallucination risk.
ENABLE_MERMAID_GENERATION = False

# =========================================================
# Retry settings
# =========================================================

MAX_ROUTE_RETRIES = 2
ROUTE_RETRY_DELAY = 0.75

# =========================================================
# Memory settings
# =========================================================

MEMORY_WINDOW = 3

# =========================================================
# Model pricing
# Cost per 1M tokens
# =========================================================

MODEL_PRICING = {
    "gemini-2.5-flash": {
        "input_price": 0.30,
        "output_price": 2.50,
        "unit_tokens": 1_000_000,
    },
    "llama-3.3-70b-versatile": {
        "input_price": 0.59,
        "output_price": 0.79,
        "unit_tokens": 1_000_000,
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "input_price": 0.11,
        "output_price": 0.34,
        "unit_tokens": 1_000_000,
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "input_price": 0.380,
        "output_price": 0.400,
        "unit_tokens": 1_000_000,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "input_price": 0.040,
        "output_price": 0.100,
        "unit_tokens": 1_000_000,
    },
}