"""
Module for creating and persisting a Chroma vector store from document chunks.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from core.config import logger
from core.utils import batch_documents
from typing import List
from langchain.schema import Document  # adjust import as necessary

def create_vector_store(
    documents: List[Document],
    persist_directory: str = "chroma_db",
    collection_name: str = "liferay_docs",
):
    """
    Creates and persists a Chroma vector store from the provided document chunks.

    This function processes a list of LangChain Document objects by:
      1. Batching the documents so that each embedding API call remains within a specified token limit.
      2. Initializing an embedding model using OpenAIEmbeddings (configured with the 'text-embedding-3-large' model).
      3. Creating a new Chroma vector store (or updating an existing one) with the document batches.
      4. Persisting the vector store to disk in the specified directory under the given collection name.

    The batching of documents is performed using the `batch_documents` function with a default token limit
    of 600,000 tokens and the "o200k_base" encoding.

    Args:
        documents (List[Document]): A list of LangChain Document objects representing the document chunks.
        persist_directory (str, optional): The directory where the vector store will be persisted. Defaults to "chroma_db".
        collection_name (str, optional): The name of the collection within the vector store. Defaults to "liferay_docs".

    Returns:
        Chroma: An instance of the Chroma vector store containing the embeddings of the provided documents.

    Raises:
        Exception: Propagates any exceptions raised during the creation or updating of the vector store.
    """
    # Initialize the embedding model.
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    # Batch the documents
    batches = batch_documents(documents, token_limit=600000, encoding_name="o200k_base")
    logger.info(f"Splitting {len(documents)} documents into {len(batches)} batches based on token limit.")

    vector_store = None
    for idx, batch in enumerate(batches):
        logger.info(f"Processing batch {idx+1}/{len(batches)} with {len(batch)} documents.")
        if vector_store is None:
            # Initialize the vector store with the first batch.
            vector_store = Chroma.from_documents(
                batch,
                embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name,
            )
        else:
            # Append the batch to the existing vector store.
            # (Method name may vary; here we assume Chroma has an `add_documents` method.)
            vector_store.add_documents(batch)
    logger.info("Vector store created and persisted to disk with %d documents", len(documents))
    return vector_store

if __name__ == "__main__":
    # For testing: load documents, chunk them, create the vector store, and persist it.
    from core.document_loader import load_documents
    from core.document_processor import chunk_documents

    docs = load_documents(path_to_data="chroma_db")
    chunks = chunk_documents(docs)
    store = create_vector_store(chunks, persist_directory="chroma_db")
    logger.info("Vector store persisted to disk.")