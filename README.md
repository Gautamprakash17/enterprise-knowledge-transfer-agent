# Enterprise Knowledge Transfer Agent

AI-powered RAG system for internal onboarding and knowledge transfer. Ingests documents and code, builds a per-project FAISS index, and answers questions with **citation-backed** responses via a **LangGraph multi-agent** pipeline.

**Stack:** LangGraph · LangChain · FAISS · FastAPI · vanilla web UI

---

## Features

| Area | What you get |
|------|----------------|
| **RAG** | Retrieve → generate → reflection → confidence scoring |
| **Multi-agent** | Supervisor, retriever, writer, critic, guardrails, shared memory |
| **Retrieval** | Hybrid semantic + MMR keyword search; optional reranking |
| **Citations** | Answers cite retrieved chunks `[N]`; sources panel in the UI |
| **Ingestion** | PDF, TXT, Markdown, code files; folder upload; Git clone; Confluence |
| **Projects** | Workspace-scoped indexes — Project A never mixes with Project B |
| **UI** | ChatGPT-style web app: projects, conversations, upload, document library |
| **Production** | Rate limiting, retries, caching, audit log, Prometheus metrics, RBAC |

---

## Quick start

### Prerequisites

- Python 3.11+
- OpenAI API key
- `git` (optional, for Git clone ingest)

### 1. Install

```bash
git clone https://github.com/Gautamprakash17/enterprise-knowledge-transfer-agent.git
cd enterprise-knowledge-transfer-agent

python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

pip install -e .
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

### 2. Start the API + web UI

```bash
uvicorn knowledge_transfer_agent.api.main:app --reload --host 127.0.0.1 --port 8000
```

Or use the helper script:

```bash
./scripts/run_ui.sh
```

| URL | Purpose |
|-----|---------|
| http://localhost:8000/app/ | **Web UI** (chat, upload, citations) |
| http://localhost:8000/docs | OpenAPI / Swagger |
| http://localhost:8000/api/v1/health | Health + index status |

> If port `8000` is busy, use another port: `uvicorn ... --port 8002` and open `http://localhost:8002/app/`.

### 3. Add documents

**Option A — Web UI (recommended)**

1. Open http://localhost:8000/app/
2. Select or create a **project**
3. Click **Add documents** or **Upload documents to chat**
4. Wait for indexing to finish (progress bar in the upload dialog)
5. Ask a question — answers include `[1]`, `[2]` citation chips

**Option B — CLI**

```bash
# Index files or folders into the default project
python scripts/ingest.py ./docs

# Add to existing index (incremental)
python scripts/ingest.py ./new_docs --add

# Full pipeline: Confluence + GitHub (from .env) + local paths
python scripts/ingest.py ./docs --full-pipeline
```

### 4. Ask a question (API)

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -H "X-Workspace-Id: default" \
  -d '{"question": "How does the RAG pipeline work in this project?"}'
```

Example response:

```json
{
  "answer": "The pipeline loads documents, chunks them, embeds into FAISS, then retrieves context at query time [1].",
  "citations": [
    {
      "source": "/path/to/doc.md",
      "source_type": "file",
      "snippet": "...",
      "doc_id": "file:abc123"
    }
  ],
  "reflection_status": "Validation passed",
  "confidence_score": 0.9
}
```

---

## Web UI overview

The UI at `/app/` is a single-page app (HTML/CSS/JS) served by FastAPI.

- **Projects** — isolated knowledge bases with separate FAISS indexes
- **Conversations** — ChatGPT-style threads; lazy-created on first message or upload
- **Upload** — attach files to the current chat or add docs to the whole project
- **Citations panel** — click `[N]` in an answer to jump to the source chunk
- **Knowledge library** — indexed documents, audit trail, shared memory
- **Themes** — dark / light mode

---

## Vector store (where embeddings live)

Embeddings are **not** stored in SQLite. They persist as FAISS artifacts on disk:

| File | Contents |
|------|----------|
| `index.faiss` | Embedding vectors + index structure |
| `index.pkl` | Chunk text and metadata mapping |
| `ingestion_manifest.json` | Change detection for incremental re-index |

**Paths:**

- Default project (legacy): `./data/faiss_index/`
- Named projects: `./data/workspaces/{project-id}/faiss_index/`

After ingest, verify in the UI (**Knowledge library → Documents**) or:

```bash
curl -H "X-Workspace-Id: default" http://localhost:8000/api/v1/documents
```

---

## Ingestion sources

| Source | How |
|--------|-----|
| **Files / folders** | Web UI upload or `scripts/ingest.py` |
| **Code repos** | Web UI → Git clone tab, or set `GITHUB_REPOS` in `.env` |
| **Confluence** | Set `CONFLUENCE_URL`, `CONFLUENCE_TOKEN`, `CONFLUENCE_SPACE_KEYS` in `.env` |

**Git / GitHub:** `GITHUB_REPOS` accepts local paths and remote URLs (`https://github.com/org/repo.git`). Remote repos are cloned into `GITHUB_CLONE_CACHE_DIR` (default `./data/git_clones`). Set `GITHUB_TOKEN` for private repos.

**Confluence:** Uses the REST API (`atlassian-python-api`). Pages are fetched with pagination. HTML `body.storage` is converted to plain text.

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check; `vector_store_loaded` per workspace |
| GET | `/api/v1/meta` | Feature flags and API version |
| POST | `/api/v1/ask` | Ask question (answer, citations, reflection, confidence) |
| POST | `/api/v1/ask/stream` | Streaming answer (SSE) |
| POST | `/api/v1/query` | Query with thread support |
| POST | `/api/v1/ingest` | Trigger ingestion on server paths |
| POST | `/api/v1/ingest/upload` | Upload files (async job) |
| GET | `/api/v1/ingest/jobs/{id}` | Poll ingest job status |
| GET | `/api/v1/documents` | List indexed sources + chunk counts |
| GET | `/api/v1/workspaces` | List projects |
| POST | `/api/v1/workspaces` | Create project |
| GET | `/api/v1/chats` | List conversation threads |
| POST | `/api/v1/chats` | Create thread |
| GET | `/metrics` | Prometheus metrics |

Pass the active project via header: `X-Workspace-Id: default`

---

## Docker

```bash
docker build -t knowledge-agent .

docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-your-key \
  -v $(pwd)/data:/app/data \
  knowledge-agent
```

Or:

```bash
docker-compose up --build
```

---

## Project structure

```
enterprise-knowledge-transfer-agent/
├── src/knowledge_transfer_agent/
│   ├── agent/              # LangGraph graph + multi-agent nodes
│   ├── api/                # FastAPI routes, middleware, schemas
│   ├── core/               # Workspaces, database, guardrails, cache
│   ├── ingestion/          # Loaders, chunking, embedding pipeline
│   ├── retrieval/          # FAISS, hybrid retriever, reranker
│   └── services/           # Agent service, ingest jobs, follow-ups
├── ui/web/                 # Web UI (HTML/CSS/JS → /app/)
├── scripts/                # CLI ingest, RAGAS eval, run helpers
├── tests/
├── docs/                   # Interview guide, architecture diagrams
├── data/                   # FAISS indexes, uploads (gitignored)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## Configuration

Key environment variables (see `.env.example` for the full list):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required** — OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | Chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `VECTOR_STORE_PATH` | `./data/faiss_index` | Default FAISS directory |
| `CHUNK_SIZE` | `1000` | Chunk size in **characters** |
| `CHUNK_OVERLAP` | `200` | Overlap in characters |
| `TOP_K_SEMANTIC` | `6` | Semantic retrieval count |
| `TOP_K_KEYWORD` | `2` | MMR keyword retrieval count |
| `ENABLE_RBAC` | `false` | Require API key on routes |
| `SECRET_KEY` | — | API key when RBAC is enabled |

---

## Architecture

Full system design with diagrams: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

```
User → Web UI (/app/) → FastAPI → AgentService → LangGraph
                                              ↓
                         Guardrails → Shared Memory → Supervisor
                                              ↓
                              Retriever → Writer → Critic → Confidence
                                              ↓
                                         FAISS (per project)
```

Interview talking points: [docs/interview_guide.md](docs/interview_guide.md)

---

## Development

```bash
# Run tests
pytest

# Lint
ruff check src tests
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| **“No index” / empty answers** | Upload docs to the **active project**; check sidebar shows **Indexed** |
| **Upload done but no new chunks** | Same file re-upload may skip (incremental manifest) — tick **Replace index** |
| **Wrong project answers** | Confirm `X-Workspace-Id` / sidebar project matches where you uploaded |
| **Port in use** | Start on another port: `--port 8002` |
| **API key errors** | Set `OPENAI_API_KEY` in `.env` and restart the server |

---

## License

MIT
