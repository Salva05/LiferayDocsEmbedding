# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from core.config import logger, DATA_PATH


def chunk_documents(documents, chunk_size=3000, chunk_overlap=100, encoding_name="o200k_base"):
    """
    Splits a list of documents into smaller text chunks based on token counts.

    Args:
        documents (List[Document]): A list of Document objects to be split.
        chunk_size (int, optional): Maximum number of tokens per chunk. Defaults to 3000.
        chunk_overlap (int, optional): Number of tokens to overlap between chunks. Defaults to 200.
        encoding_name (str, optional): The tokenizer encoding to use. Defaults to "o200k_base" (base encoder for gpt-4o's token management).

    Returns:
        List[Document]: A list of Document chunks.
    """
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding_name
        #separators=["[DOC_END]", "\n \n \n", "\n\n", "\n"]
    )
    doc_chunks = text_splitter.split_documents(documents)
    logger.info(
        f"Document split into {len(doc_chunks)} chunks with a chunk size of ~{chunk_size} tokens (overlap: {chunk_overlap}).")
    return doc_chunks

if __name__ == "__main__":
    # For testing
    from core.document_loader import load_documents
    docs = load_documents(DATA_PATH)
    chunks = chunk_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks")
