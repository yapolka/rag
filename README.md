# RAG Question Answering System

A modular Retrieval-Augmented Generation (RAG) system built with Python, Weaviate, Sentence Transformers, and Groq LLM.

The system answers user questions based on a local document knowledge base and provides references to retrieved sources.

## 1. Project Overview

The project implements a complete RAG pipeline:

1. Load and preprocess local text documents.
2. Split documents into overlapping chunks.
3. Generate embeddings using a local Sentence Transformer model.
4. Store chunks, embeddings, and metadata in Weaviate.
5. Retrieve relevant chunks using semantic vector search.
6. Filter retrieved results based on cosine distance.
7. Generate answers using Groq LLM.
8. Return answers with source references.

If retrieval does not find sufficiently relevant context, the system returns a fallback response instead of calling the LLM.

## 2. Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Weaviate | Vector database |
| Docker Compose | Local database deployment |
| Sentence Transformers | Local embedding generation |
| multilingual-e5-small | Multilingual embedding model |
| Groq API | LLM inference |
| OpenAI Python SDK | OpenAI-compatible Groq API client |
| python-dotenv | Environment configuration |
| unittest | Automated testing |
| logging | Pipeline monitoring |

## 3. Architecture

**Document Indexing**

```text
Local Documents
      |
      v
Document Loader
      |
      v
Text Cleaning
      |
      v
Chunking
      |
      v
Sentence Transformer
      |
      v
Vector Embeddings
      |
      v
Weaviate
```

**Question Answering**

```text
User Question
      |
      v
Query Embedding
      |
      v
Weaviate Semantic Search
      |
      v
Top-K Retrieved Chunks
      |
      v
Relevance Filtering
      |
      +---- Weak Context ----> Fallback Answer
      |
      v
Context Construction
      |
      v
Groq LLM
      |
      v
Answer + Sources
```

## 4. Project Structure

```text
rag/
├── knowledge_base/
├── loader.py
├── chunking.py
├── embedder.py
├── weaviate_store.py
├── index_documents.py
├── retriever.py
├── prompts.py
├── evaluation.py
├── rag_pipeline.py
├── evaluate_rag.py
├── test_chunking.py
├── test_evaluation.py
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## 5. Installation

### Prerequisites

- Python 3.11 or 3.12
- Docker and Docker Compose v2
- Internet access for initial model download
- Groq API key

### Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Configure Environment Variables

Copy the example configuration:

```bash
cp .env.example .env
```

Edit `.env` and provide a valid Groq API key.

Example:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Groq API keys are available at:

https://console.groq.com/keys

## 6. Start Weaviate

Start the database using Docker Compose:

```bash
docker compose up -d
```

The local deployment exposes:

- HTTP: `localhost:8081`
- gRPC: `localhost:50051`

Check container status:

```bash
docker compose ps
```

Check database readiness:

```bash
curl -i http://localhost:8081/v1/.well-known/ready
```

Expected HTTP status: `200 OK`.

## 7. Prepare the Knowledge Base

Place at least five UTF-8 encoded `.txt` documents into:

```text
knowledge_base/
```

The document loader removes extra whitespace and empty lines.

Documents are automatically split into chunks.

Default chunking configuration:

```python
chunk_size = 500
chunk_overlap = 100
```

Chunk size and overlap are measured in characters.

Each chunk contains:

- `document_id`
- `source_name`
- `chunk_id`
- `text`

## 8. Generate Embeddings

The project uses the local embedding model:

```text
intfloat/multilingual-e5-small
```

Embedding dimensionality:

```text
384
```

Document embeddings use the `passage:` prefix.

Query embeddings use the `query:` prefix.

Both are L2-normalized.

Test embedding generation:

```bash
python embedder.py
```

The model is downloaded automatically on first use.

## 9. Index Documents

Run:

```bash
python index_documents.py
```

The indexing pipeline:

1. Loads documents.
2. Creates overlapping chunks.
3. Generates embeddings.
4. Creates or opens the Weaviate collection.
5. Inserts chunks and metadata.
6. Verifies the number of stored objects.

Collection name:

```text
KnowledgeChunk
```

Repeated indexing skips objects with existing deterministic UUIDs.

**Limitation:** The current implementation prevents duplicate insertion but does not automatically update modified chunks or delete obsolete chunks.

## 10. Semantic Retrieval

Run:

```bash
python retriever.py
```

Enter a question related to the knowledge base.

The retriever generates a query embedding and searches Weaviate for the nearest vectors.

Default configuration:

```python
top_k = 3
```

Each result includes:

- Document identifier
- Source filename
- Chunk identifier
- Retrieved text
- Cosine distance

Lower cosine distance indicates greater embedding similarity.

## 11. Run the RAG Application

Start the interactive question-answering application:

```bash
python rag_pipeline.py
```

Enter a question:

```text
What is supervised learning?
```

The system retrieves relevant chunks, filters them, builds a prompt, and sends the context to the Groq LLM.

The result contains:

- Generated answer
- Source references
- Retrieval distances
- Pipeline status

If no chunks satisfy the configured relevance threshold, the system skips LLM generation and returns an insufficient-context message.

## 12. Retrieval Quality and Evaluation

The system uses cosine distance filtering.

Initial configuration:

```python
distance_threshold = 0.5
```

This is a heuristic starting value, not a universal relevance threshold.

To evaluate alternative thresholds:

```bash
python evaluate_rag.py
```

The evaluation script compares:

```python
THRESHOLDS = [0.3, 0.4, 0.5, 0.6]
```

Evaluation should include:

- Questions answerable using the knowledge base
- Questions not answerable using the knowledge base
- Manual inspection of retrieved chunks
- Retrieval distances
- False acceptance and false rejection cases

The evaluation script reports answerability classification accuracy. It does not directly measure factual correctness of generated LLM answers.

For document-level retrieval evaluation, relevant source documents can additionally be annotated to calculate Hit Rate@K.

## 13. Testing

Run chunking tests:

```bash
python test_chunking.py
```

Run retrieval-evaluation and pipeline tests:

```bash
python -m unittest test_evaluation.py -v
```

The tests cover:

- Basic text chunking
- Overlapping chunks
- Empty input
- Invalid chunking parameters
- Metadata preservation
- Distance-based filtering
- Empty retrieval
- Weak-context handling
- Skipping LLM calls for rejected context
- Generation with accepted context

## 14. Demo Questions

Replace these example questions with questions supported by the actual knowledge base.

Example answerable questions:

1. What is machine learning?
2. What is supervised learning?
3. What is a neural network?

Example questions that should not be answerable when the corresponding information is absent:

4. What is the weather on Mars?
5. Who won the football World Cup in 1930?

The expected retrieval results and answers must be validated against the documents provided with the project.

## 15. Limitations

The current implementation has several limitations:

- Character-based chunking can split words and sentences.
- Embeddings are generated sequentially.
- The relevance threshold requires calibration.
- Semantic similarity does not guarantee factual relevance.
- LLM output can still contain unsupported claims.
- Source references identify retrieved context but are not independently verified citations.
- Modified and deleted documents are not automatically synchronized.
- The application currently provides a command-line interface rather than an HTTP API.

## 16. Future Improvements

Possible improvements include:

- Token-aware or sentence-aware chunking
- Batch embedding generation
- Automatic document synchronization
- Hybrid BM25 and vector retrieval
- Cross-encoder reranking
- More comprehensive retrieval evaluation
- Groundedness and answer-faithfulness evaluation
- Pydantic response schemas
- FastAPI endpoint
- Automated integration tests

## 17. Project Status

The core modules for document loading, chunking, vector indexing, retrieval, generation, and relevance filtering have been implemented.

Final project acceptance requires successfully running the integration pipeline and validating the results against the included knowledge base.

---

**Project:** RAG Question Answering System  
**Stack:** Python · Weaviate · Sentence Transformers · Groq
