from core.config import logger

def metadata_extractor(record: dict, metadata: dict) -> dict:
    """
    Extracts each key that is not 'content' from a DocumentationItem JSON record
    and returns it as the metadata of a LangChain Document.
    (for more information see https://github.com/Salva05/LiferayDocsScrape.git)

    Args:
        record (dict): The input dictionary containing the scraped document data.
        metadata (dict): A dictionary where extracted metadata will be stored.

    Returns:
        dict: A dictionary containing the structured metadata for the document.
    """

    # Basic metadata fields
    metadata["url"] = record.get("url")
    metadata["title"] = record.get("title")
    metadata["path"] = record.get("path")
    metadata["scraped_at"] = record.get("scraped_at")

    # Flatten the 'metadata' field (excluding 'Deployment Approach', not too relevant)
    for key, value in record.get("metadata", {}).items():
        if key == "Deployment Approach":
            continue
        # If value is a list, join them into a comma-separated string
        if isinstance(value, list):
            metadata[f"metadata_{key}"] = ", ".join(value)
        else:
            metadata[f"metadata_{key}"] = value

    return metadata

def post_process(documents):
    """
    Post-processes a list of LangChain Document objects by prepending
    metadata information (specifically 'title' and 'path') to each document's
    page content.

    Each document is expected to have:
      - a `metadata` attribute (a dictionary) with optional keys 'title' and 'path'
      - a `page_content` attribute containing the main text of the document

    The function constructs a header from the available metadata and
    prepends it to the existing page content, separated by blank lines.

    Args:
        documents (list): A list of LangChain Document objects.

    Returns:
        list: The list of Document objects with updated page content.
    """
    for doc in documents:
        # Retrieve metadata values, if they exist
        doc_path = doc.metadata.get("path", "").strip()
        source = doc.metadata.get("url", "").strip()

        # Build header lines based on available metadata
        header_lines = []
        if doc_path:
            header_lines.append(f"Documentation Path: {doc_path}")
        if source:
            header_lines.append(f"Source: {source}")

        # If header has been constructed
        if header_lines:
            header = "\n".join(header_lines)
            # Inject DOC_START and DOC_END markers to delineate the document boundaries
            doc.page_content = f"{header}\n\n{doc.page_content}\n\n[DOC_END]"

    return documents

def deduplicate_documents(documents):
    """
    Filters out duplicate document objects from a list based on their content.
    The need arises from the overlap chunk that may feed into the model same documents
    """
    unique_docs = []
    seen = set()
    for doc in documents:
        # Normalize the content for comparison.
        normalized = doc.page_content.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            unique_docs.append(doc)
        else:
            # Log a warning showing only the first 10 characters of the duplicate document's content.
            truncated = doc.page_content.strip()[:10]
            logger.warning(f"Duplicate document found: (Firs 10 chars) {truncated}...")
    return unique_docs

import tiktoken
from langchain.schema import Document  # adjust as needed
from typing import List

def count_tokens(text: str, encoding_name: str = "o200k_base") -> int:
    """
    Counts the number of tokens in the provided text using the specified tiktoken encoding.

    This function leverages the tiktoken library to encode the input text and compute the token count.
    This is useful for evaluating text length in contexts where token limits are enforced (e.g., language model inputs).

    Args:
        text (str): The text string to be tokenized.
        encoding_name (str, optional): The name of the tiktoken encoding to use. Defaults to "o200k_base".

    Returns:
        int: The number of tokens in the encoded text.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def batch_documents(
    documents: List[Document],
    token_limit: int = 600000,
    encoding_name: str = "o200k_base"
) -> List[List[Document]]:
    """
    Groups a list of LangChain Document objects into batches such that the cumulative
    token count of each batch does not exceed the specified token_limit.

    This function iterates through the provided documents, counts the tokens in each document's
    page content using the specified tiktoken encoding, and groups them into batches. Each batch's
    total token count is maintained below the provided limit, ensuring compatibility with models
    or systems that impose token constraints.

    Args:
        documents (List[Document]): A list of LangChain Document objects, each containing a 'page_content' attribute.
        token_limit (int, optional): The maximum allowed number of tokens per batch. Defaults to 600000.
        encoding_name (str, optional): The name of the tiktoken encoding to use for token counting. Defaults to "o200k_base".

    Raises:
        ValueError: If an individual document exceeds the token limit on its own.

    Returns:
        List[List[Document]]: A list of batches, where each batch is a list of Document objects whose
                                combined token count is below the token_limit.
    """
    batches = []
    current_batch = []
    current_tokens = 0
    for doc in documents:
        # Each Document has a 'page_content' property
        tokens = count_tokens(doc.page_content, encoding_name=encoding_name)
        if tokens > token_limit:
            raise ValueError(f"A single document exceeds the token limit: {tokens} tokens")
        # If adding this document would exceed the limit (and there is at least one doc in the batch)
        if current_tokens + tokens > token_limit and current_batch:
            batches.append(current_batch)
            current_batch = [doc]
            current_tokens = tokens
        else:
            current_batch.append(doc)
            current_tokens += tokens
    if current_batch:
        batches.append(current_batch)
    return batches
