# Enterprise Knowledge Transfer Agent

AI-powered RAG system for internal architecture, data pipelines, and deployment workflows. Built with LangGraph, LangChain, FAISS, and FastAPI.

## Features

- **RAG workflow**: Retrieve → Generate → Reflection → Confidence
- **Hybrid retrieval**: Semantic (FAISS) + keyword/MMR search
- **Citation enforcement**: Traceable, citation-backed responses
- **Reflection & confidence**: Hallucination validation, confidence scoring
- **Ingestion**: TXT, PDF; Confluence, GitHub, local files
- **Production-ready**: Logging middleware, env config, Docker, RBAC

---

## Run Instructions

### Prerequisites

- Python 3.11+
- OpenAI API key

### 1. Local Setup

```bash
# Clone and enter project
cd LanGraph

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install
pip install -e .

# Configure
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

### 2. Ingest Documents

```bash
# Ingest TXT/PDF from paths
python scripts/ingest.py ./docs ./runbooks

# Add to existing index
python scripts/ingest.py ./new_docs --add

# Full pipeline (Confluence, GitHub, files)
python scripts/ingest.py ./docs --full-pipeline
```

**Git / GitHub:** In `.env`, set `GITHUB_REPOS` to comma-separated **local repo paths** and/or **remote Git URLs** (for example `https://github.com/org/repo.git`). Remote URLs are **cloned or pulled** into `GITHUB_CLONE_CACHE_DIR` (default `./data/git_clones`), then text files are indexed. Requires **`git` installed**. For private **GitHub** HTTPS repos, set `GITHUB_TOKEN`.

**Confluence:** Uses the **Confluence REST API** (install `atlassian-python-api`). Set `CONFLUENCE_URL`, `CONFLUENCE_TOKEN`, and `CONFLUENCE_SPACE_KEYS`. All pages in each space are fetched with **pagination** (not only the first 100). If token-only auth fails on Cloud, set `CONFLUENCE_USERNAME` to your Atlassian email. HTML `body.storage` is converted to plain text with BeautifulSoup.

### 3. Run API

```bash
uvicorn knowledge_transfer_agent.api.main:app --reload
```

API: http://localhost:8000  
Docs: http://localhost:8000/docs  
Web UI: http://localhost:8000/app/

### 4. Docker

```bash
# Build
docker build -t knowledge-agent .

# Run (mount data volume for FAISS index)
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-your-key \
  -v $(pwd)/data:/app/data \
  knowledge-agent
```

Or with docker-compose:

```bash
# Set OPENAI_API_KEY in .env first
docker-compose up --build
```

---

## API Examples

### POST /api/v1/ask

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do we deploy the data pipeline?"}'
```

Response:

```json
{
  "answer": "...",
  "citations": [
    {"source": "/path/to/doc", "source_type": "file", "doc_id": "..."}
  ],
  "reflection_status": "Validation passed",
  "confidence_score": 0.9
}
```

### POST /api/v1/query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is our database architecture?"}'
```

### GET /api/v1/health

```bash
curl http://localhost:8000/api/v1/health
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key |
| `VECTOR_STORE_PATH` | No | `./data/faiss_index` | FAISS index directory |
| `API_HOST` | No | `0.0.0.0` | Bind address |
| `API_PORT` | No | `8000` | Server port |
| `ENABLE_RBAC` | No | `false` | Require API key for routes |
| `SECRET_KEY` | No | - | API key when RBAC enabled |

See `.env.example` for the full list.

---

## Project Structure

```
LanGraph/
├── src/knowledge_transfer_agent/
│   ├── config.py
│   ├── ingestion/          # Loaders, chunking, embedding pipeline
│   ├── retrieval/          # FAISS, hybrid retriever
│   ├── agent/              # LangGraph nodes, prompts, graph
│   └── api/                # FastAPI, routes, middleware
├── scripts/ingest.py       # CLI ingestion
├── ui/web/                 # Web UI (HTML/CSS/JS, served at /app/)
├── docs/                   # Sample docs
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/ask` | Ask question (answer, citations, reflection, confidence) |
| POST | `/api/v1/query` | Query with thread support |
| POST | `/api/v1/feedback` | Submit feedback |
| POST | `/api/v1/ingest` | Trigger ingestion |

---

## License

MIT
