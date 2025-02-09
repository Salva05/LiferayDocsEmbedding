# LiferayDocsEmbedding

## Overview  
This repository constitutes the **second phase** of a three-phase project aimed at building an advanced retrieval pipeline for Liferay documentation.  

- **Phase One** ([Salva05/LiferayDocsScrape.git](https://github.com/Salva05/LiferayDocsScrape.git), currently private) focuses on **web scraping** Liferay docs and exporting them in `.jl` format.  
- **Phase Two** (this repository) ingests the `.jl` data source, **chunks** the contents, **embeds** them using [HuggingFace Embeddings](https://github.com/hwchase17/langchain/blob/master/docs/modules/indexes/text_vectorstores.rst#huggingfaceembeddings), and **stores** them in a [Chroma](https://docs.trychroma.com/) vector database—relying on [LangChain](https://github.com/hwchase17/langchain) to manage the creation of `Document` objects and text splitting.  
- **Phase Three** will build upon the vector store created here to complete the end-to-end retrieval pipeline (details to come in its own repository).

**Key Highlights**:
- Load `.jsonl` files containing Liferay docs data (titles, metadata, paths, etc.).
- Transform them into LangChain `Document` objects.
- Split large documents into smaller chunks to optimize embedding and retrieval.
- Generate embeddings with `intfloat/e5-base-v2` model.
- Persist these embeddings in a Chroma vector store for similarity search.
- Provide a test suite to verify the correctness of the pipeline.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Code Overview](#code-overview)
3. [Overall Workflow](#overall-workflow)
4. [Setup & Installation](#setup--installation)
5. [Usage](#usage)
6. [Tech Stack](#tech-stack)

---

## Project Structure

~~~plaintext
.
├── core/
│   ├── config.py               # Centralized configuration (env variables, logging)
│   ├── document_loader.py      # Loads and preprocesses .jsonl data into LangChain Documents
│   ├── document_processor.py   # Splits documents using text splitting
│   ├── utils.py                # Utility functions (metadata extraction, post-processing, deduplication)
│   └── vector_store.py         # Creates and persists the Chroma vector store
├── main.py                     # Main entry-point pipeline that orchestrates the entire process
├── test.py                     # Unit tests that ensure the pipeline and vector store are working
├── shape.json                  # Sample JSON snippet showing the .jsonl data structure expected
├── .env.example                # Example environment variables file
├── requirements.txt            # Dependencies required to run this project
└── README.md                   # This documentation file
~~~

### Notable Files

- **`main.py`**  
  The single, primary entry-point script. It:
  1. Loads `.jsonl` documents.
  2. Splits them into smaller chunks.
  3. Creates a Chroma vector store with embeddings.
  4. Persists the resulting store on disk.

- **`test.py`**  
  Contains a suite of standardized Python tests (via `unittest`). It checks:
  1. Whether your Chroma vector store has valid document IDs.
  2. Whether similarity search retrieves results as expected.

- **`shape.json`**  
  Example of the JSON structure expected for each line in the `.jsonl` dataset:
  ```json
  {
    "url": "https://learn.liferay.com/w/dxp/index",
    "title": "This is a sample json line",
    "content": "Here goes the main content of the page scraped ...",
    "path": "/ DXP / Indexing",
    "metadata": {
      "Capability": [
        "Platform"
      ],
      "Resource Type": [
        "Official Documentation"
      ],
      "Deployment Approach": [
        "Liferay PaaS",
        "Liferay SaaS",
        "Liferay Self-Hosted"
      ]
    },
    "scraped_at": "2025-02-05T20:11:26.358643+00:00"
  }
  
## Code Overview

1. **`core/config.py`**  
   - Loads environment variables from `.env`.
   - Sets up global logging (to `run.log` and console).
   - Exports constants (`DATA_PATH`, `CHROMA_DB_DIR`) used throughout the project.

2. **`core/document_loader.py`**  
   - Uses `JSONLoader` from `langchain_community.document_loaders` to parse `.jsonl` data.
   - Applies `metadata_extractor` to build structured metadata for each document.
   - Applies `post_process` to inject relevant metadata (title, path, etc.) into each document’s `page_content` to make them accessible by the model, in the context.

3. **`core/document_processor.py`**  
   - Provides a `chunk_documents` function using `RecursiveCharacterTextSplitter` to split large documents into smaller, more manageable chunks.

4. **`core/utils.py`**  
   - `metadata_extractor`: Builds structured metadata from the original JSON fields, expected from LangChain's Document object (for db indexing and retrieval efficiency).
   - `post_process`: Prepends metadata to each document’s `page_content`.
   - `deduplicate_documents`: Removes duplicate documents based on their content (the overlap chunk sometimes makes the same document being retrieved from a RAG process).

5. **`core/vector_store.py`**  
   - Leverages `HuggingFaceEmbeddings` with the `intfloat/e5-base-v2` model.
   - Creates a Chroma vector store with `Chroma.from_documents`.
   - Persists the vector store to disk for later loading and usage.

6. **`main.py`**  
   - Orchestrates the entire workflow:
     1. Loads documents (JSON).
     2. Splits them into chunks.
     3. Embeds and creates a Chroma store.
     4. Persists the store.

7. **`test.py`**  
   - Provides a `unittest` suite that:
     1. Loads the persisted Chroma store.
     2. Checks that stored document IDs are present.
     3. Performs a similarity search to ensure the pipeline works end-to-end.

## Overall Workflow

1. **Load and Process Documents**  
   - The `document_loader.py` module reads the `.jsonl` file, applies metadata extraction, and converts each line into a LangChain `Document`.
   - The `document_processor.py` module splits these documents into smaller, manageable chunks using the `RecursiveCharacterTextSplitter`.

2. **Embedding & Vector Store Creation**  
   - The `vector_store.py` module leverages `HuggingFaceEmbeddings` (using the `intfloat/e5-base-v2` model) to generate embeddings for the document chunks.
   - A Chroma vector store is created from these embeddings and persisted to disk.

3. **Persist & Query**  
   - Once persisted, the vector store can be loaded at any time (for example, via `test.py`).
   - Similarity searches can be performed or, other vector-based retrieval operations on the stored embeddings.

## Setup & Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/YourUsername/LiferayDocsEmbedding.git
   cd LiferayDocsEmbedding
   
2. **Create and Activate a Python Virtual Environment**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate

3. **Install Dependencies**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt

4. **Set Up Environment Variables**  
   ```bash
   cp .env.example .env
   
  > [!NOTE]
  > DATA: The path to the jsonl data source.
  > CHROMA_DB_DIR: The directory where the Chroma vector store will be persisted.
   
## Usage

1. **Prepare JSONL Data**  
   - Place `.jsonl` file representing the dataset, in the directory specified by the `DATA` environment variable in `.env` file.

2. **Run the Pipeline**  
   - Execute the main entry point to process the data, split documents into chunks, generate embeddings, and persist the Chroma vector store:
     ```bash
     python main.py
     ```
   - This script will:
     - Load and preprocess the documents from your JSON Lines data source.
     - Split each document into smaller chunks using the text splitter.
     - Generate embeddings with the `intfloat/e5-base-v2` model.
     - Create and persist a Chroma vector store in the directory specified by `CHROMA_DB_DIR`.

3. **Query and Verify**  
   - To verify that everything is working correctly, run the test script:
     ```bash
     python test.py
     ```
   - The test script performs a similarity search using a test query, and prints a snippet along with metadata from the most similar document.

## Tech Stack

- **Python**  
  - Version 3.12 (or compatible)

- **LangChain Ecosystem**  
  - `langchain` (0.3.17)  
  - `langchain-chroma` (0.2.1)  
  - `langchain-community` (0.3.16)  
  - `langchain-core` (0.3.34)  
  - `langchain-huggingface` (0.1.2)  
  - `langchain-text-splitters` (0.3.6)

- **Vector Store**  
  - `chromadb` (0.6.3)  
  - `chroma-hnswlib` (0.7.6)

- **Embeddings & Transformers**  
  - `sentence-transformers` (3.4.1)  
  - `transformers` (4.48.3)  
  - `huggingface-hub` (0.28.1)

- **Deep Learning Framework**  
  - `torch` (2.6.0)

- **Environment Management**  
  - `python-dotenv` (1.0.1)
