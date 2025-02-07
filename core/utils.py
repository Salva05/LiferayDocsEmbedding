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