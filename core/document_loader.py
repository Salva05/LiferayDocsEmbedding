from langchain_community.document_loaders import JSONLoader
from core.utils import metadata_extractor, post_process
from core.config import DATA_PATH, logger

def load_documents():
    """
    Loads documents from a JSON file, applies metadata extraction,
    and includes some metadata in the page content of the document.

    Returns:
        List[Document]: A list of Document objects loaded from the JSON file.
            Each Document contains:
              - page_content: The main content extracted from the JSON field specified by 'content_key'.
              - metadata: A dictionary with additional information
    """
    loader = JSONLoader(
        file_path=DATA_PATH,
        jq_schema=".",
        content_key="content",  # Field for page_content extraction
        text_content=True,
        json_lines=True,
        metadata_func=metadata_extractor,
    )
    documents = loader.load()
    documents = post_process(documents)  # Include some metadata in page_content for making them available to the model

    logger.info(f"Loaded {len(documents)} documents")
    return documents

if __name__ == "__main__":
    # For testing
    docs = load_documents()
    docs = post_process(docs)

    print(docs[0])  # Pick the first loaded document to analyze its shape