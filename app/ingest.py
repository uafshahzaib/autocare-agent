"""
Builds a FAISS vector store from the markdown documents in knowledge_base/.

Embeddings are produced locally by a PyTorch-backed sentence-transformer model
(no external API calls needed for retrieval), keeping the RAG pipeline fast,
offline-capable, and cheap to run in CI.
"""
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    KNOWLEDGE_BASE_DIR,
    VECTOR_STORE_DIR,
)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Local, PyTorch-backed embedding model — no API key required."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vector_store() -> FAISS:
    loader = DirectoryLoader(
        str(KNOWLEDGE_BASE_DIR), glob="**/*.md", loader_cls=TextLoader
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(VECTOR_STORE_DIR))
    return vector_store


def load_vector_store() -> FAISS:
    """Load the persisted index, building it first if it doesn't exist yet."""
    embeddings = get_embeddings()
    if not VECTOR_STORE_DIR.exists():
        return build_vector_store()
    return FAISS.load_local(
        str(VECTOR_STORE_DIR), embeddings, allow_dangerous_deserialization=True
    )


if __name__ == "__main__":
    build_vector_store()
    print(f"Vector store built at {VECTOR_STORE_DIR}")
