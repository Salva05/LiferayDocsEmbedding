import os
import unittest
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from core.config import CHROMA_DB_DIR, logger

class TestChromaVectorStore(unittest.TestCase):
    """
    Test case for verifying that the persisted Chroma vector store
    is correctly loaded and that similarity search returns expected results.
    """

    def setUp(self):
        """
        Set up the embedding model and load the persisted vector store.
        """
        # Configure the embedding model
        self.device_type = "cpu"   # If CUDA-capable GPU is available, change to 'cuda'
        self.collection_name = "liferay_docs"
        self.path_to_db = os.path.join(os.path.dirname(os.path.abspath(__file__)), CHROMA_DB_DIR)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-base-v2",
            model_kwargs={"device": self.device_type}
        )

        # Load the persisted vector store.
        self.store = Chroma(
            persist_directory=self.path_to_db,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )

    def test_collection_data(self):
        """
        Test that the collection data contains stored document IDs.
        """
        collection_data = self.store._collection.get()
        # Check that collection_data has an "ids" key.
        self.assertIn("ids", collection_data)
        num_ids = len(collection_data.get("ids", []))
        # Log the number of documents stored
        logger.info("Number of documents stored: %d", num_ids)
        # Assert that at least one document is stored.
        self.assertGreater(num_ids, 0, "No documents found in the vector store.")

    def test_similarity_search(self):
        """
        Test that a similarity search returns at least one result.
        """
        test_query = "Blade CLI"
        similar_docs = self.store.similarity_search(test_query, k=1)
        # Ensure that we receive at least one document.
        self.assertTrue(len(similar_docs) > 0, "Similarity search returned no results.")

        # Optionally, further check that the returned document has the expected attributes.
        doc = similar_docs[0]
        self.assertIsInstance(doc.page_content, str, "Document page_content is not a string.")
        self.assertIsInstance(doc.metadata, dict, "Document metadata is not a dictionary.")
        # Optionally print a snippet for manual inspection (not required for automated tests)
        snippet = doc.page_content[:500]
        print("Snippet:", snippet)
        print("Metadata:", doc.metadata)

if __name__ == "__main__":
    unittest.main()
