"""Project-wide configuration: paths, env vars, LLM defaults."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # python-dotenv is optional at runtime; missing env file is fine.
    pass


# Anchored at the project root (parent of src/).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SAMPLE_DIR: Path = DATA_DIR / "sample"
DB_DIR: Path = PROJECT_ROOT / "db"
DB_PATH: Path = DB_DIR / "renteriq.duckdb"

# Subfolders the user may drop raw files into. Each maps to a dataset_type hint.
RAW_SUBFOLDERS: dict[str, str] = {
    "apartments": "apartments",
    "demographics": "demographics",
    "crime": "crime",
    "transit": "transit",
    "amenities": "amenities",
    "parks": "parks",
    "worship": "worship",
    "housing": "housing",
    "facilities": "facilities",
    "population": "population",
}


def ensure_directories() -> None:
    """Create the data/db directory tree if it does not yet exist."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for sub in RAW_SUBFOLDERS:
        (RAW_DIR / sub).mkdir(parents=True, exist_ok=True)


# LLM defaults - all overridable via .env.
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock").strip().lower()
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

WATSONX_API_KEY: str = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID: str = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL: str = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_MODEL_ID: str = os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-3-8b-instruct")


# Always make sure directories exist at import time so seed_data and the
# Streamlit app can write without crashing on a fresh checkout.
ensure_directories()
