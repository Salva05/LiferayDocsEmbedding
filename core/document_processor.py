from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.config import  logger

def chunk_documents(documents, chunk_size=1000, chunk_overlap=150):
    """
    Splits documents into smaller chunks using a recursive splitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    # This returns a list of new document chunks
    _chunks = text_splitter.split_documents(documents)
    return _chunks

if __name__ == "__main__":
    # For testing
    from core.document_loader import load_documents
    docs = load_documents()
    chunks = chunk_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks")
