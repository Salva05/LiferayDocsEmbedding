from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.config import  logger

def chunk_documents(documents, chunk_size=2000, chunk_overlap=200):
    """
    Splits documents into smaller chunks using a recursive splitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(
        f"Document split into {len(chunks)} chunks with a chunk size of ~{chunk_size} tokens (overlap: {chunk_overlap}).")
    return chunks

if __name__ == "__main__":
    # For testing
    from core.document_loader import load_documents
    docs = load_documents()
    chunks = chunk_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks")
