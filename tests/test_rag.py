"""Integration test for the RAG retrieval pipeline (builds/loads a real FAISS index
using local PyTorch sentence-transformer embeddings — no network call to an LLM)."""
from app.tools import _search_knowledge_base


def test_rag_retrieves_relevant_chunk_for_oil_question():
    result = _search_knowledge_base("How often should I change my engine oil?")
    assert "oil" in result.lower()


def test_rag_retrieves_relevant_chunk_for_brake_question():
    result = _search_knowledge_base("My brakes are squealing, what does that mean?")
    assert "squeal" in result.lower() or "brake" in result.lower()
