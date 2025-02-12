"""
Module for creating and persisting a Chroma vector store from document chunks.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from core.config import logger


def create_vector_store(documents, persist_directory=None, collection_name="liferay_docs", device_type="cpu"):
    """
    Creates a Chroma vector store from the given document chunks using the 'text-embedding-3-large'.

    Args:
        documents (List[Document]): A list of document chunks for which embeddings will be computed (output of the text splitter).
        persist_directory (str, optional): The directory where the vector store will be persisted.
            If None, the vector store will be created in-memory.
        collection_name (str, optional): The name of the collection within the vector store.
            Defaults to "liferay_docs".
        device_type (str, optional): The device on which the embedding model will run.
            Options are "cpu" for running on the CPU or "cuda" for GPU acceleration.
            Defaults to "cpu". Use "cuda" if a compatible GPU is available for faster computation.

    Returns:
        Chroma: A Chroma vector store object containing embeddings for the provided documents.
    """
    # Embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
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

    docs = load_documents(path_to_data="chroma_db")
    chunks = chunk_documents(docs)
    store = create_vector_store(chunks, persist_directory="chroma_db")
    logger.info("Vector store persisted to disk.")