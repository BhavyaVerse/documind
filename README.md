# DocuMind

**Production-grade RAG system for querying SEC 10-K financial filings.**

DocuMind answers questions about annual reports from Amazon, Alphabet, Visa, Apple, and Morgan Stanley using a hybrid retrieval pipeline, cross-encoder reranking, and cited answer generation — with full observability, automated evaluation, and a live REST API.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.16-1C3C3C?logo=langchain&logoColor=white)](https://langchain.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Railway](https://img.shields.io/badge/Deployed-Railway-0B0D0E?logo=railway&logoColor=white)](https://railway.app)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)](/.github/workflows/eval.yml)

<!--**[Live Demo →](https://your-app.railway.app/docs)** &nbsp;|&nbsp; **[API Docs →](https://your-app.railway.app/docs)** &nbsp;|&nbsp; **[Health Check →](https://your-app.railway.app/health)** -->

---

## What it does

Send a natural language question about any of the five 10-K filings and get back a grounded, cited answer with links to the exact source passages.

```bash
curl -X POST https://your-app.railway.app/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are Amazon'\''s primary business segments and how does AWS generate revenue?"}'
```

```json
{
  "answer": "Amazon operates three reportable segments: North America, International,
             and AWS [1]. AWS generates revenue primarily through usage-based fees
             for cloud services including compute, storage, and machine learning [2].",
  "citations_used": [1, 2],
  "sources": [
    { "citation_number": 1, "file": "amazon_10k_2025.pdf", "page": 6,  "relevance_score": 0.9821 },
    { "citation_number": 2, "file": "amazon_10k_2025.pdf", "page": 18, "relevance_score": 0.9134 }
  ],
  "validation": { "passed": true, "faithfulness_score": 0.96 },
  "latency_ms": 2340,
  "trace_id": "abc123"
}
```

---

## Architecture

```
User query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                   FastAPI  /query                   │
└───────────────────────┬─────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
  Vector Search               BM25 Keyword Search
  (ChromaDB +                 (rank-bm25, exact
  all-MiniLM-L6-v2)           term matching)
          │                           │
          └─────────────┬─────────────┘
                        │
                        ▼
              Reciprocal Rank Fusion
              (top-20 deduplicated
               candidates)
                        │
                        ▼
            Cross-Encoder Reranker
            (ms-marco-MiniLM-L-6-v2)
            (top-20 → top-5)
                        │
                        ▼
           LLM Answer Generation
           (any llm model, cited
            inline with [n] markers)
                        │
                        ▼
           Citation + Faithfulness
           Validation
                        │
                        ▼
              JSON response +
              Langfuse trace
```

---

## Document corpus

| Company | Filing | Period |
|---|---|---|
| Amazon | 10-K | FY ended December 31, 2025 |
| Alphabet | 10-K | FY ended December 31, 2025 |
| Visa | 10-K | FY ended September 30, 2025 |
| Apple | 10-K | FY ended September 27, 2025 |
| Morgan Stanley | 10-K | FY ended December 31, 2025 |

---

## Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| **API** | FastAPI, Uvicorn | REST layer, request/response models |
| **Retrieval** | ChromaDB, rank-bm25 | Vector store + keyword index |
| **Embeddings** | sentence-transformers `all-MiniLM-L6-v2` | Local, free, 384-dim vectors |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Pairwise relevance scoring |
| **Generation** | Groq `llama-3.3-70b-versatile` via LangChain | Cited answer generation |
| **Fusion** | Reciprocal Rank Fusion (RRF) | Combines BM25 + vector rankings |
| **Observability** | Langfuse | Per-step tracing, cost tracking, quality scores |
| **Evaluation** | RAGAS | Faithfulness, answer relevancy, context precision |
| **CI** | GitHub Actions | Runs RAGAS gate on every pull request |
| **Deployment** | Docker, Railway | Containerised, live public URL |

---

## Project structure

```
documind/
│
├── src/
│   ├── ingestion/
│   │   ├── loader.py          # PDF loading with LangChain DirectoryLoader
│   │   └── chunker.py         # RecursiveCharacterTextSplitter, 400-token chunks
│   │
│   ├── retrieval/
│   │   ├── vector_store.py    # ChromaDB build + load + search
│   │   ├── bm25_index.py      # BM25Okapi build, pickle persistence, search
│   │   ├── hybrid.py          # Reciprocal Rank Fusion over both result lists
│   │   └── reranker.py        # CrossEncoder reranking, top-20 → top-5
│   │
│   ├── generation/
│   │   ├── generator.py       # Prompt loading, context formatting, LLM call
│   │   └── validator.py       # Citation coverage + LLM faithfulness check
│   │
│   ├── monitoring/
│   │   └── tracer.py          # Langfuse trace/span/generation/score helpers
│   │
│   └── api/
│       └── main.py            # FastAPI app, lifespan model loading, /query endpoint
│
├── config/
│   └── prompts.yaml           # Version-controlled LLM prompt templates
│
├── data/
│   └── documents/             # Source PDF files (SEC 10-K filings)
│
├── evaluation/
│   ├── golden_dataset.json    # 50 hand-curated Q&A pairs across all 5 companies
│   ├── evaluator.py           # RAGAS pipeline runner + per-company breakdown
│   ├── ci_gate.py             # Threshold gate, exits 0 (pass) or 1 (fail)
│   └── results/               # Timestamped JSON output from every eval run
│
├── .github/
│   └── workflows/
│       └── eval.yml           # GitHub Actions: runs CI gate on every PR
│
├── Dockerfile                 # Production image, models baked in at build time
├── docker-compose.yml         # Local development orchestration
├── railway.json               # Railway deployment configuration
├── start.py                   # Container entry point: ingest if needed, then serve
├── ingest.py                  # Builds ChromaDB vector store + BM25 index from PDFs
├── .env                       # API keys (gitignored)
└── requirements.txt           # All Python dependencies, pinned versions
```

---

## Setup

### Prerequisites

- Python 3.11+ (avoid 3.14 version)
- Docker Desktop (for local container testing)
- OpenAI API key or any another LLM API key(I have used GROQ free API key for answer generation)
- Langfuse account (free tier at [cloud.langfuse.com](https://cloud.langfuse.com))

### 1. Clone and install

```bash
git clone https://github.com/BhavyaVerse/documind.git
cd documind

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your-key-here
OPENAI_API_KEY=sk-your-openai-key-here
LANGFUSE_PUBLIC_KEY=pk-lf-your-key-here
LANGFUSE_SECRET_KEY=sk-lf-your-key-here
```

### 3. Add documents

Place your SEC 10-K PDF files in `data/documents/`. The system expects at least one PDF to function.

### 4. Build the document indexes

```bash
python ingest.py
```

This downloads the embedding model (~90 MB, one time only), chunks all PDFs, embeds every chunk into ChromaDB, and builds the BM25 keyword index. Takes 2–5 minutes depending on document count.

---

## Running locally

### Option A — Python directly

```bash
uvicorn src.api.main:app --reload --port 8000
```

### Option B — Docker

```bash
# First run: builds image and runs ingest inside the container
docker compose up --build

# Subsequent runs: skips ingest, starts in ~5 seconds
docker compose up
```

Open **[http://localhost:8000/docs](http://localhost:8000/docs)** for the interactive Swagger UI.

---

## API reference

### `POST /query`

Run a question through the full RAG pipeline.

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `question` | string | required | Natural language question (5–500 chars) |
| `top_k` | int | 5 | Number of source passages to retrieve (1–10) |
| `run_validation` | bool | true | Run faithfulness check (adds ~1–2 s latency) |
| `session_id` | string | null | Groups related queries in Langfuse |

**Response:**

| Field | Type | Description |
|---|---|---|
| `answer` | string | Generated answer with inline `[n]` citations |
| `citations_used` | list[int] | Which source numbers were cited |
| `sources` | list | File, page, relevance score, text preview per source |
| `validation` | dict | Citation coverage + faithfulness score |
| `latency_ms` | float | Total pipeline latency |
| `trace_id` | string | Langfuse trace ID for this request |

### `GET /health`

```json
{
  "status": "ok",
  "vector_store_loaded": true,
  "bm25_loaded": true,
  "reranker_loaded": true
}
```

---

## How the retrieval pipeline works

### Why hybrid search?

Vector search (embeddings) and BM25 keyword search are complementary. Vector search finds semantically related content even when exact terms don't appear. BM25 finds exact matches for specific numbers, names, and financial terminology like "EBITDA" or "Tier 1 capital ratio". Neither alone performs as well as both combined.

### Reciprocal Rank Fusion

Rather than normalising scores across both systems (which is fragile — BM25 and cosine similarity live on incompatible scales), RRF uses only rank positions:

```
RRF_score = Σ  1 / (60 + rank_i)
```

A document ranked 1st in both lists scores highest. A document only appearing in one list still scores well. The `k=60` constant smooths outlier influence.

### Cross-encoder reranking

The bi-encoder (embedding model) encodes query and document separately — fast but less accurate. The cross-encoder reads both together in a single forward pass, attending to query-document interaction. This is 10× more accurate but too slow to run over all chunks. Running it only over the top-20 hybrid candidates gives accuracy at manageable cost (~0.3–0.8 s on CPU).

### Chunk size decision

The cross-encoder model (`ms-marco-MiniLM-L-6-v2`) has a **512-token limit covering query + chunk combined**. With a typical query of ~15 tokens, the safe chunk size is 400 tokens with 50-token overlap — preventing silent truncation that would corrupt relevance scores.

---

## Evaluation

DocuMind ships with a 50-sample golden dataset drawn from all five 10-K filings, covering executives, business segments, revenue drivers, risk factors, competitive landscape, and specific financial metrics.

### Run evaluation manually

```bash
# Quick run — 10 samples, ~$0.25
python -m evaluation.evaluator --n 10

# Full run — all ready samples
python -m evaluation.evaluator
```

Output is saved to `evaluation/results/eval_YYYYMMDD_HHMMSS.json` with aggregate scores, per-company breakdown, and per-sample scores.

### Run the CI gate

```bash
python -m evaluation.ci_gate
```

Exits `0` if all metrics pass, `1` if any fail. GitHub Actions runs this automatically on every pull request.

### RAGAS metrics and thresholds

| Metric | Threshold | What it measures |
|---|---|---|
| **Faithfulness** | ≥ 0.80 | Are all claims in the answer supported by retrieved context? |
| **Answer Relevancy** | ≥ 0.75 | Does the answer actually address the question asked? |
| **Context Precision** | ≥ 0.70 | Are the retrieved chunks ranked correctly by relevance? |

---

## Observability

Every API request creates a Langfuse trace with four child spans:

```
documind-query  (trace)
├── hybrid-search   → candidates returned, vector vs BM25 counts
├── rerank          → CE score range, input/output count
├── llm-generation  → model, token counts, cost, citations
└── validation      → citation coverage, faithfulness score
```

Scores (`citation_coverage`, `faithfulness`, `validation_passed`) are attached as named time-series values and appear as charts in the Langfuse dashboard, making quality regressions visible over time.

View traces at [cloud.langfuse.com](https://cloud.langfuse.com) after running any query.

---

## Deployment

### Local Docker

```bash
docker compose up --build   # first run
docker compose up           # subsequent runs
```

### Railway (live)

The repository includes `railway.json` which configures Railway to use the `Dockerfile` and `python start.py` as the entry point.

**To deploy:**

1. Push the repository to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub repo
3. Select the `documind` repository
4. Add environment variables in Railway dashboard: `OPENAI_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`
5. Railway builds the image and provides a public URL

The `Dockerfile` bakes both ML models (~175 MB) into the image at build time so cold starts take seconds, not minutes. The `start.py` entry point checks for built indexes on every startup and runs `ingest.py` automatically if they're missing.
<!--
### Test a live deployment

```bash
BASE_URL=https://your-app.railway.app ./test_docker.sh
```
-->

---

## Key technical decisions

**Local embedding model over OpenAI embeddings.** `all-MiniLM-L6-v2` runs entirely on CPU, costs nothing per query, and produces embeddings fast enough for real-time use (~14,000 sentences/second). For this corpus size the accuracy difference from `text-embedding-3-small` is negligible.

**Prompts in YAML, not Python.** Storing prompt templates in `config/prompts.yaml` means prompt changes are tracked in Git independently of code changes. This makes it possible to review prompt regressions in pull requests and roll back prompts without a code deployment.

**Validation as a runtime check, not just an eval metric.** The faithfulness check in `validator.py` runs on every live query, not just during evaluation. This means the API can flag potentially hallucinated answers in production and surface that signal to callers via the `validation.passed` field in the response.

**Fixed seed in CI.** The CI gate always evaluates the same 20 questions (seed=42) regardless of which PR triggered it. This makes scores directly comparable across commits — a score change is a real signal, not sampling noise.

---

## Repository

```
github.com/BhavyaVerse/documind
```
