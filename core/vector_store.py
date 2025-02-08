"""
Module for creating and persisting a Chroma vector store from document chunks.
"""

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from core.config import OPENAI_API_KEY, logger


def create_vector_store(documents, persist_directory=None, collection_name="liferay_docs"):
    """
    Creates a Chroma vector store from the given document chunks using the E5-base-v2 model.

    Args:
        documents: List of document chunks (output of the text splitter).
        persist_directory: Directory to persist the vector store. If None, the store is in-memory.
        collection_name: Name of the collection within Chroma.

    Returns:
        A Chroma vector store object.
    """
    # Instantiate the embedding model with E5-base-v2.
    # You can specify "cuda" for GPU acceleration if available; otherwise, use "cpu".
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={"device": "cuda"}  # Change to "cpu" if no GPU is available
    )

    vector_store = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    logger.info("Vector store created with %d documents", len(documents))
    return vector_store

if __name__ == "__main__":
    # For testing: load documents, chunk them, create the vector store, and persist it.
    from core.document_loader import load_documents
    from core.document_processor import chunk_documents

    docs = load_documents()
    chunks = chunk_documents(docs)
    store = create_vector_store(chunks, persist_directory="chroma_db")
    store.persist()  # Saves your embeddings and data to disk
    logger.info("Vector store persisted to disk.")