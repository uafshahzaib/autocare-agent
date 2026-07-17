"""Central configuration for the AutoCare agent."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM provider selection. Set LLM_PROVIDER to "anthropic" or "openai" explicitly,
# or leave unset and it's inferred from whichever API key is present
# (Anthropic takes priority if both are set).
LLM_PROVIDER = os.getenv("LLM_PROVIDER")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
