# AI Research Assistant (RAG Pipeline)

A robust, localized Retrieval-Augmented Generation (RAG) system designed to extract precise insights from PDF documents. Built with a focus on observability and hallucination prevention, this API utilizes an "LLM-as-a-Judge" architecture to ensure all generated answers are strictly grounded in the provided document context.

## Features

* **Intelligent Document Ingestion:** Extracts and cleans text from PDF files using `PyMuPDF`, preserving pagination metadata for accurate citations.
* **Semantic Search Engine:** Utilizes `HuggingFace` embeddings (`all-MiniLM-L6-v2`) and `ChromaDB` for high-speed, localized vector retrieval.
* **Asynchronous REST API:** Built on `FastAPI` to handle concurrent LLM requests without blocking the event loop.
* **LLM-as-a-Judge Evaluation:** Implements an automated faithfulness check using a secondary LLM to detect and flag hallucinations before returning the response.
* **Production-Ready Observability:** Features structured file logging to track request latency, retrieval steps, and evaluation metrics.
* **Fully Containerized:** Dockerized deployment with host-network routing for seamless integration with local LLM providers.

## Tech Stack

* **Frameworks:** FastAPI, LangChain (LCEL)
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace (`sentence-transformers`)
* **LLM Engine:** Ollama (Local inference)
* **Models:** `qwen3:1.7b` (Generation), `qwen2.5:1.5b-instruct` (Evaluation)
* **Infrastructure:** Docker, Docker Compose

## Architecture Flow

1. **Ingestion:** PDF -> `PyMuPDFLoader` -> `RecursiveCharacterTextSplitter` -> `ChromaDB`.
2. **Retrieval:** User Query -> Embedding -> Top-K Semantic Search -> Context Assembly.
3. **Generation:** Query + Context -> Generation LLM -> Draft Answer.
4. **Evaluation:** Draft Answer + Context -> Judge LLM -> Faithfulness Score (True/False).
5. **Response:** Returns Draft Answer + Faithfulness Score via API.

## Getting Started

### Prerequisites

* Docker installed on your machine.
* [Ollama](https://ollama.com/) installed and running locally.
* Required Ollama models downloaded:
```bash
ollama run qwen3:1.7b
ollama run qwen2.5:1.5b-instruct

```



### Installation & Run

1. **Clone the repository:**
```bash
git clone https://github.com/Danylo20006/AI_Research_Assistant.git
cd AI-Research-Assistant

```


2. **Add a document:**
Place your target PDF document into the `data/papers/` directory (e.g., `data/papers/novel.pdf`).
3. **Build and run via Docker:**
```bash
docker build -t rag-assistant .
docker run -p 8000:8000 -v ${PWD}/data:/app/data rag-assistant

```


*The API will be available at `http://localhost:8000/docs`.*

## API Usage

**Endpoint:** `POST /ask`

**Request:**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/ask' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "How many people were left by AM?"
}'

```

**Response:**

```json
{
  "answer": "The context explicitly states that 'there were only five of us, down here inside, alone with AM.' Therefore, the number of people left by AM is five.",
  "is_faithful": true
}

```