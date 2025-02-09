"""
Entry Point for the Document Processing Pipeline

This script orchestrates the following steps:
  1. Loads processed LangChain Documents from a JSONL data source.
  2. Splits the documents into smaller chunks that fit within the model's context window.
  3. Embeds these chunks and creates a Chroma vector store.
  4. Persists the vector store to disk for efficient retrieval in downstream tasks.

"""

from core.config import DATA_PATH, CHROMA_DB_DIR, logger
from core.document_loader import load_documents
from core.document_processor import chunk_documents
from core.vector_store import create_vector_store

def main():
    # Load the Documents from the JSONL file
    documents = load_documents(path_to_data=DATA_PATH)

    # Split Documents into manageable chunks
    document_chunks = chunk_documents(
        documents=documents,
        chunk_size=2000,    # Should always fit in the context window of the model used
        chunk_overlap=200
    )

    # Create the vector store and embed the document chunks
    vector_storage = create_vector_store(
        documents=document_chunks,
        persist_directory=CHROMA_DB_DIR,
        collection_name="liferay_docs",
        device_type="cpu"  # Only if a cuda-capable GPU is available, otherwise 'cpu'
    )

    logger.info("Vector store correctly persisted to disk.")

if __name__ == "__main__":
    main()  # Entry point of the program
