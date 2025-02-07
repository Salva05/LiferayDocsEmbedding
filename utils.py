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

    # Extract common metadata fields from the scraped record
    metadata["url"] = record.get("url")
    metadata["title"] = record.get("title")
    metadata["path"] = record.get("path")
    metadata["scraped_at"] = record.get("scraped_at")

    # Extract metadata field, but excludes the field Deployment Approach (not really relevant)
    metadata["metadata"] = {
        key: value
        for key, value in filter(lambda item: item[0] != "Deployment Approach", record.get("metadata", {}).items())
    }

    return metadata

def include_title(docs):
    """
    Prepend the title to 'page_content' field of each document
    """
    for doc in docs:
        title = doc.metadata.get("title", "Untitled")
        doc.page_content = f"Title: {title}\n\n{doc.page_content}"
