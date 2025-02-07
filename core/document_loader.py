import os
import logging
from langchain_community.document_loaders import JSONLoader
from utils import metadata_extractor, include_title
from core.config import DATA_PATH

# Logging configuration
log_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'run.log')

# Create log file if it doesn't exist.
if not os.path.exists(log_filename):
    open(log_filename, 'w').close()  # Create an empty log file if it doesn't exist

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger()

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
    include_title(documents)  # Prepend title to each Document's page_content
    return documents


if __name__ == "__main__":
    docs = load_documents()
    logger.info(f"Loaded {len(docs)} documents.")
