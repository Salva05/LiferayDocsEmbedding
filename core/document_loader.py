from langchain_community.document_loaders import JSONLoader
from utils import metadata_extractor, include_title
from config import DATA_PATH


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
    print(f"Loaded {len(docs)} documents.")
