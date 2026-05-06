from pathlib import Path

# Model & LLM settings
MODEL_NAME = "gemini-2.5-flash" # Backend default model used by main.py / CLI fallback.

MODEL_OPTIONS: dict[str, tuple[str, str]] = {
	"Gemini: gemini-2.5-flash": ("gemini", "gemini-2.5-flash"),
	"Groq: llama-3.3-70b-versatile": ("groq", "llama-3.3-70b-versatile"),
}
TEMPERATURE = 0.1
TOP_K = 10
EMBED_MODEL = "all-MiniLM-L6-v2"

# Paths
VECTOR_STORE_PATH = Path("data/vector_store")
OUTPUT_JSON_PATH = Path("data/results/output.json")

# Retry settings
MAX_ROUTE_RETRIES = 2
ROUTE_RETRY_DELAY = 0.75

# Memory settings
MEMORY_WINDOW = 3  # number of previous interactions to consider
