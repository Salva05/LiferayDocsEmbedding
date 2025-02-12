from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.config import  logger

def chunk_documents(documents, chunk_size=4000, chunk_overlap=400):
    """
    Splits a list of documents into smaller text chunks using a recursive character text splitter.

    Args:
        documents (List[Document]): A list of Document objects to be split.
        chunk_size (int, optional): Maximum token size for each chunk. Defaults to 2000.
        chunk_overlap (int, optional): Number of tokens to overlap between consecutive chunks. Defaults to 200.

    Returns:
        List[Document]: A list of Document chunks resulting from splitting the original documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["[DOC_END]", "[DOC_START]", "\n\n", "\n"]
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
