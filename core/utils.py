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
        title = doc.metadata.get("title", "").strip()
        doc_path = doc.metadata.get("path", "").strip()
        source = doc.metadata.get("url", "").strip()

        # Build header lines based on available metadata
        header_lines = []
        if title:
            header_lines.append(f"Title: {title}")
        if doc_path:
            header_lines.append(f"Documentation Path: {doc_path}")
        if source:
            header_lines.append(f"Source: {source}")

        # If header has been constructed
        if header_lines:
            header = "\n".join(header_lines)
            # Prepend it and add a blank line for readability
            doc.page_content = f"{header}\n\n{doc.page_content}"

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