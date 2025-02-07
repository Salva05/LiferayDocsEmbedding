from langchain_community.document_loaders import JSONLoader
from core.utils import metadata_extractor
from core.config import DATA_PATH, logger

def load_documents():
    """
    Loads documents from a JSON file, applies metadata extraction,
    and includes the title in the page content.
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

    logger.info(f"Loaded {len(documents)} documents")
    return documents

if __name__ == "__main__":
    # For testing
    load_documents()