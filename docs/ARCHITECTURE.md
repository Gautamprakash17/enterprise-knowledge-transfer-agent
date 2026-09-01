# Architecture — Enterprise Knowledge Transfer Agent

This document describes how the system is structured end-to-end: clients, API, agent graph, ingestion, storage, and deployment.

For interview talking points and Q&A, see [interview_guide.md](interview_guide.md).

---

## 1. System overview

The product is a **workspace-scoped RAG assistant**:

1. **Ingest** internal documents and code into a per-project **FAISS** index.
2. **Retrieve** relevant chunks at query time (hybrid semantic + keyword).
3. **Generate** grounded answers with `[N]` citations via a **LangGraph multi-agent** workflow.
4. **Validate** answers with a critic/reflection step and attach a **confidence** score.

```mermaid
flowchart LR
  subgraph Clients
    UI[Web UI /app/]
    API_CLIENT[HTTP clients]
  end

  subgraph Platform
    API[FastAPI]
    AGENT[LangGraph agents]
    VS[(FAISS per project)]
    DB[(SQLite audit / chats)]
  end

  UI --> API
  API_CLIENT --> API
  API --> AGENT
  AGENT --> VS
  API --> DB
  INGEST[Ingestion jobs] --> VS
```

---

## 2. Layered architecture

| Layer | Responsibility | Key modules |
|-------|----------------|-------------|
| **Presentation** | Chat UI, upload, citations, projects | `ui/web/` served at `/app/` |
| **API** | REST, streaming, auth, rate limits | `api/main.py`, `api/routes*.py` |
| **Agent** | Multi-hop RAG orchestration | `agent/graph.py`, `agent/multi_agent/` |
| **Retrieval** | Embeddings, FAISS, hybrid search | `retrieval/vector_store.py`, `hybrid_retriever.py` |
| **Ingestion** | Load, chunk, embed, persist | `ingestion/pipeline.py` |
| **Core** | Workspaces, guardrails, cache, metrics | `core/workspaces.py`, `core/database.py` |

```mermaid
flowchart TB
  subgraph Clients["Clients"]
    C1["Web UI\nui/web → /app/"]
    C2["HTTP clients\ncurl / Postman / integrations"]
  end

  subgraph API["API layer — FastAPI"]
    M["main.py + middleware"]
    R1["/health · /meta"]
    R2["/ingest · /ingest/upload"]
    R3["/ask · /ask/stream · /query"]
    R4["/workspaces · /chats · /documents"]
    SEC["RBAC: API key / Bearer"]
    RL["Rate limit: SlowAPI"]
    M --> SEC
    M --> RL
    M --> R1
    M --> R2
    M --> R3
    M --> R4
  end

  subgraph Core["Agent runtime"]
    SVC["AgentService"]
    G["LangGraph graph"]
    RET["HybridRetriever + FAISS"]
    EMB["OpenAI embeddings"]
    CACHE["Retrieve + LLM cache"]
    SVC --> G
    G --> RET
    RET --> EMB
    G --> CACHE
    RET --> CACHE
  end

  subgraph Ingest["Ingestion"]
    P["IngestionPipeline"]
    LOAD["File / Git / Confluence loaders"]
    CHK["Chunking"]
    VS["save_faiss_index"]
    P --> LOAD --> CHK --> VS
  end

  subgraph Data["Persistence"]
    FAISS[(index.faiss + index.pkl)]
    SQL[(kta.db)]
  end

  CFG[".env / config.py"]

  Clients --> API
  API --> SVC
  R2 --> P
  VS --> FAISS
  RET --> FAISS
  API --> SQL
  CFG -.-> API
  CFG -.-> Core
  CFG -.-> Ingest
```

> Source file for editing: [diagrams/project_system_architecture.mmd](diagrams/project_system_architecture.mmd)

---

## 3. Request flow (ask)

```mermaid
sequenceDiagram
  autonumber
  participant User
  participant UI as Web UI
  participant API as FastAPI /ask
  participant SVC as AgentService
  participant LG as LangGraph
  participant VS as FAISS
  participant LLM as OpenAI

  User->>UI: Question + workspace header
  UI->>API: POST /api/v1/ask
  API->>SVC: ask(question, workspace_id)
  SVC->>LG: invoke graph
  LG->>LG: Guardrails + shared memory load
  loop Supervisor routing
    LG->>VS: Hybrid retrieve (multi-hop)
    VS-->>LG: Top-k chunks
    LG->>LLM: Writer (grounded prompt)
    LG->>LLM: Critic (reflection)
  end
  LG->>LG: Memory + confidence + output guardrails
  LG-->>SVC: answer, citations, confidence
  SVC-->>API: Structured response
  API-->>UI: JSON or SSE stream
  UI-->>User: Answer + [N] citation chips
```

**Workspace scoping:** every request carries `X-Workspace-Id`. Retrieval loads only that project's FAISS index (`core/workspaces.py` → `workspace_index_path`).

---

## 4. LangGraph multi-agent workflow

The compiled graph (`agent/graph.py`) runs this pipeline:

```
guardrails → shared_memory → supervisor ⇄ (retriever | writer | critic) → memory → confidence → output_guardrails → shared_memory_save → END
```

```mermaid
flowchart TD
  START([User question]) --> GRD[guardrails\ninjection block · PII redact]
  GRD --> SM[shared_memory\nload workspace facts]
  SM --> SUP[supervisor\nroute next agent]

  SUP -->|need docs| RET[retriever\nplan · hybrid search · compress]
  SUP -->|draft answer| WRI[writer\ngrounded generation + citations]
  SUP -->|validate| CRI[critic\nreflection / groundedness]
  SUP -->|done| MEM[memory\nconversation update]

  RET --> SUP
  WRI --> SUP
  CRI -->|fail| SUP
  CRI -->|pass| SUP

  MEM --> CONF[confidence\nscore 0–1]
  CONF --> OGRD[output_guardrails\nredact secrets on output]
  OGRD --> SAVE[shared_memory_save\npersist episodic facts]
  SAVE --> END([Response])

  subgraph Retrieval
    RET --> FAISS[(FAISS index)]
  end
```

| Agent | Role |
|-------|------|
| **guardrails** | Block prompt injection; redact PII/secrets on input |
| **shared_memory** | Load long-term workspace memory before retrieval |
| **supervisor** | Routes to retriever, writer, critic, or finish |
| **retriever** | Sub-query planning, hybrid FAISS retrieval, optional compression |
| **writer** | Strict grounded prompt; requires citation markers `[N]` |
| **critic** | Validates groundedness; failed critique loops back via supervisor |
| **memory** | Updates per-thread conversation state |
| **confidence** | Maps critic outcome to a 0–1 trust score |
| **output_guardrails** | Final safety pass on the answer text |
| **shared_memory_save** | Writes durable facts back to workspace memory |

Each node is wrapped with **Prometheus timing** (`kta_agent_duration_seconds`).

---

## 5. Ingestion pipeline

Documents enter the system through the **web UI upload**, **API ingest endpoints**, or **CLI** (`scripts/ingest.py`).

```mermaid
flowchart LR
  SRC[Sources\nPDF · TXT · MD · code · Git · Confluence]
  LOAD[document_loader]
  CHK[RecursiveCharacterTextSplitter\nchunk_size · overlap]
  EMB[OpenAI embeddings]
  IDX[FAISS index]
  MAN[ingestion_manifest.json]

  SRC --> LOAD --> CHK --> EMB --> IDX
  IDX --> MAN
```

**Incremental indexing** (`ingestion/pipeline.py`):

| Manifest change | Behaviour |
|-----------------|-----------|
| No change | Skip re-embed (fast path) |
| Add only | `add_documents` to existing FAISS |
| Modified / removed | Full rebuild (`replace_index`) |

**Async uploads:** `POST /api/v1/ingest/upload` saves files under `data/web_uploads/{workspace}/` and queues a background job (`services/ingest_jobs.py`). Poll `GET /api/v1/ingest/jobs/{id}` for progress.

---

## 6. Data & storage layout

```
data/
├── faiss_index/                    # Default project (legacy path if index exists)
│   ├── index.faiss                 # Embedding vectors
│   ├── index.pkl                   # Chunk text + metadata
│   └── ingestion_manifest.json
├── workspaces/
│   └── {project-id}/
│       └── faiss_index/            # Isolated index per project
├── web_uploads/{project}/          # Raw uploaded files (server-side)
├── git_clones/                     # Cloned repos (Git ingest)
└── kta.db                          # SQLite: chats, audit, ingest jobs, memory
```

| Store | What it holds |
|-------|----------------|
| **FAISS** | Embeddings + chunk metadata (`source`, `doc_id`, `workspace_id`) |
| **SQLite** | Chat threads, messages, audit log, ingest job status, shared memory |
| **web_uploads/** | Original files before indexing (not queried at runtime) |

**Citation rule:** `[N]` in an answer refers to the **N-th retrieved context chunk** for that response, not a file number or PDF page by default.

---

## 7. Workspace isolation

Each **project** (workspace) has:

- Its own FAISS directory under `data/workspaces/{id}/faiss_index/`
- Its own upload folder under `data/web_uploads/{id}/`
- Scoped chat threads and shared memory in SQLite

The API resolves the active project from `X-Workspace-Id` (default: `default`). Retrieval **never** mixes chunks across projects.

---

## 8. API surface (summary)

| Group | Endpoints |
|-------|-----------|
| **Q&A** | `POST /api/v1/ask`, `POST /api/v1/ask/stream`, `POST /api/v1/query` |
| **Ingest** | `POST /api/v1/ingest`, `POST /api/v1/ingest/upload`, `GET /api/v1/ingest/jobs/{id}` |
| **Projects** | `GET/POST /api/v1/workspaces`, `DELETE /api/v1/workspaces/{id}` |
| **Chats** | `GET/POST /api/v1/chats`, messages CRUD |
| **Library** | `GET /api/v1/documents`, audit log, shared memory |
| **Ops** | `GET /api/v1/health`, `GET /metrics` |

---

## 9. Cross-cutting concerns

| Concern | Implementation |
|---------|----------------|
| **Auth** | Optional RBAC via `ENABLE_RBAC` + `SECRET_KEY` (API key / Bearer) |
| **Rate limiting** | SlowAPI on API routes |
| **Caching** | Retrieval and LLM invoke cache (`core/cache.py`) |
| **Retries** | Exponential backoff on LLM calls |
| **Observability** | Structured logging middleware, Prometheus `/metrics`, query audit |
| **Safety** | Input/output guardrails, abstain when evidence is insufficient |

---

## 10. Deployment view

```mermaid
flowchart TB
  subgraph Single host
    NGINX[Nginx optional\nreverse proxy]
    UVICORN[Uvicorn / Gunicorn\nFastAPI workers]
    DATA[(data/ volume\nFAISS + SQLite)]
    UVICORN --> DATA
    NGINX --> UVICORN
  end

  USER[Users] --> NGINX
  USER --> UVICORN

  OPENAI[OpenAI API\nchat + embeddings]
  UVICORN --> OPENAI
```

- **Docker:** `Dockerfile` + `docker-compose.yml` — mount `./data` for index persistence.
- **Scale-out:** run multiple API replicas behind a load balancer; share the `data/` volume or move FAISS to a shared/object store pattern; SQLite is single-writer — use Postgres for multi-replica chat/audit at scale.

Example scaled layout: `docker-compose.scale.yml`, `deploy/nginx/nginx.conf`, `scripts/run_scaled_api.sh`.

---

## 11. Architecture diagrams (images)

Pre-rendered diagrams in the repo:

| File | Description |
|------|-------------|
| [diagrams/kta_rag_langgraph_architecture.png](diagrams/kta_rag_langgraph_architecture.png) | RAG + LangGraph overview |
| [diagrams/kta_project_architecture_vertical.png](diagrams/kta_project_architecture_vertical.png) | Vertical system layout |
| [Generated_image.png](Generated_image.png) | Agent flowchart |

Edit Mermaid sources in `docs/diagrams/*.mmd` and render at [mermaid.live](https://mermaid.live).

---

## 12. Key design decisions

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **FAISS on disk** | Fast local retrieval, simple ops | Horizontal scale needs sharding or managed vector DB |
| **Per-workspace indexes** | Strong isolation between projects | Duplicate embeddings if same doc in two projects |
| **LangGraph multi-agent** | Explicit routing, retries, testable agents | More moving parts than a single chain |
| **Hybrid retrieval** | Better recall on keyword-heavy docs | More chunks to rank and filter |
| **Strict grounding + critic** | Enterprise trust; fewer hallucinations | May abstain more often |
| **Vanilla web UI** | No frontend build step; served by FastAPI | Less component ecosystem than React SPA |
