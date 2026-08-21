# Multi-Agent Support System

A full-stack AI customer-support platform combining a **multi-agent orchestration backend** built with LangGraph and FastAPI, and a **Next.js support-desk frontend** ("Relay Desk").

Users ask questions in natural language and the system routes the request to specialized agents — order lookup, policy RAG, support tickets, and returns/refunds — evaluates each response for grounding and safety, and streams live progress back to the UI. Conversations persist across turns via LangGraph checkpoints and a SQLite-backed conversation store.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Frontend  (Next.js / Relay Desk)                    │
│                Chat workspace · conversation sidebar · SSE streaming        │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │  /api/query  ·  /api/query/stream  ·  /api/conversations
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Backend  (FastAPI + LangGraph)                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Support Supervisor                           │  │
│  │                     Intent detection + routing                       │  │
│  └──────────┬────────────────────┬────────────────────┬─────────────────┘  │
│             ▼                    ▼                    ▼                     │
│      ┌────────────┐      ┌────────────┐      ┌──────────────┐              │
│      │   Order    │      │  Policy    │      │  Support     │              │
│      │   Agent    │      │   RAG      │      │  Ticket      │              │
│      └─────┬──────┘      └─────┬──────┘      └──────┬───────┘              │
│            │        ┌──────────┴─────────┐          │                       │
│            ▼        ▼                    ▼          ▼                       │
│       Order Tools  Hybrid RAG    Return/Refund   Ticket Tools              │
│      (real DB)     retrieval      Agent        (real DB records)           │
│                                   │                                        │
│                                   ▼                                        │
│                            Action Tools (real refunds)                     │
│                                   │                                        │
│                                   ▼                                        │
│                              Evaluator                                     │
│                            (grounding + safety)                            │
│                          ┌──────┴──────┐                                   │
│                          ▼             ▼                                   │
│                       PASS           FAIL ──► retry / repair              │
│                          │                                                 │
│                          ▼                                                 │
│                       Formatter ──► END                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agents

| Agent | Purpose | Handoff To | Tools |
|-------|---------|------------|-------|
| **Supervisor** | Intent detection, planning, and orchestration | order, policy_rag, support_ticket | — |
| **Order** | Retrieves customer order information from the database | evaluator | `search_orders`, `get_order_by_id`, `get_order_by_customer_name`, `get_order_items` |
| **Policy RAG** | Retrieves policies, FAQs, and docs via vector/BM25 search | return_refund, evaluator | — |
| **Support Ticket** | Creates and tracks support tickets | evaluator | `create_support_ticket`, `get_ticket_status`, `list_tickets_by_email`, `update_ticket_status` |
| **Return/Refund** | Processes return/refund requests and refund lookups | evaluator | `calculate_eligible_refund`, `create_refund_request`, `get_refund_status`, `list_refunds_by_order` |
| **Evaluator** | Scores responses for grounding, correctness, and safety; retries failed agents | formatter, all agents, end | — |
| **Formatter** | Produces the final user-facing response | end | — |

### Key Features

- **7 specialized agents** with controlled peer-to-peer handoffs (max 5 per request to prevent loops)
- **Real database records** — support tickets (`TKT-...`) and refunds (`REF-...`) are persisted and trackable
- **Structured routing & evaluation** via LLM structured output (`RouteDecision`, `EvaluationResult`)
- **Database-backed orders** with SQLAlchemy + SQLite
- **Agentic RAG pipeline** — contextualization → hybrid dense (Chroma/PGVector) + sparse (BM25) retrieval fused with Reciprocal Rank Fusion → LLM reranking (0–10) → context grading → generation → groundedness check, with query-rewrite and complementary-retrieval loops
- **Pluggable vector store** — local ChromaDB by default; opt-in Postgres PGVector (`USE_PGVECTOR=true`) for embeddings storage
- **LangGraph checkpoints** (SQLite or in-memory) for stateful multi-turn conversations
- **Streaming** agent progress over SSE (`/query/stream`)
- **Security pipeline** — input sanitization, prompt-injection detection, PII masking, output validation
- **Production features** — rate limiting (slowapi), response caching (Redis in production, in-memory in dev), structured JSON logging, metrics, LangSmith tracing
- **Conversation store** — persistent conversations/messages API with per-conversation LangGraph thread mapping

### Policy RAG Pipeline

The Policy RAG agent runs a full agentic retrieval loop as an internal sub-graph:

```
User Message
    │
    ▼
Contextualize (rewrite follow-up into standalone question)
    │
    ▼
Hybrid Retrieve (dense + BM25 fused via Reciprocal Rank Fusion)
    │
    ▼
Rerank (LLM scores each chunk 0–10 → top-k)
    │
    ▼
Grade Context (SUFFICIENT / PARTIAL / NONE / UNANSWERABLE)
    │
    ├─── SUFFICIENT ──► Generate Answer ──► Groundedness Check ──► Return Response
    │                         │ not grounded
    │                         ▼
    │                    Rewrite Query (loop, bounded by MAX_RETRIEVAL_RETRIES)
    │
    ├─── PARTIAL ──► Retrieve Complementary (new sources, missing aspects) ──► Rerank → Grade (loop)
    │
    ├─── NONE ──► Rewrite Query ──► Hybrid Retrieve → Rerank → Grade (loop)
    │
    └─── UNANSWERABLE ──► "I don't have sufficient information..."
```

## Repository Structure

```
multi-agents/
├── backend/                     # FastAPI + LangGraph multi-agent API
│   ├── app/
│   │   ├── agents/              # Supervisor, RAG, order, ticket, refund, evaluator, formatter, graph
│   │   ├── tools/               # LangChain @tool bindings (orders, tickets, refunds)
│   │   ├── api.py               # /query, /query/stream, /health, /metrics, /cache/stats
│   │   ├── api_conversations.py # /conversations CRUD + message endpoints
│   │   ├── config.py            # pydantic-settings configuration
│   │   ├── models.py            # API request/response models
│   │   ├── models_db.py         # SQLAlchemy ORM models
│   │   ├── db.py / db_init.py   # DB engine + seeding
│   │   ├── checkpoints.py       # LangGraph checkpoint factory
│   │   ├── cache.py             # Response caching (memory dev / Redis production)
│   │   ├── ingestion.py         # KB → Chroma/PGVector + BM25 indexing
│   │   ├── retrieval.py         # Hybrid dense+BM25 retrieval with RRF fusion
│   │   ├── reranker.py          # LLM-based chunk reranking (0–10 scoring)
│   │   ├── security.py          # Input/output security pipeline
│   │   ├── monitoring.py        # Logging and metrics
│   │   └── data/                # Seed orders + RAG knowledge base
│   ├── main.py                  # Uvicorn entry point
│   └── README.md                # Detailed backend documentation
└── frontend/                    # Next.js "Relay Desk" support workspace
    ├── app/
    │   ├── page.tsx             # Chat interface (SSE streaming)
    │   └── api/                 # Proxy routes → backend (query, stream, conversations)
    ├── components/              # Sidebar, Avatar, shadcn/ui primitives
    ├── contexts/                # Conversation + Theme providers
    ├── hooks/                   # TanStack Query hooks
    └── public/                  # Icons, fonts, placeholders
```

## Getting Started

### Prerequisites

- Python >= 3.12
- Node.js >= 20 and pnpm
- OpenRouter API key (for LLM calls)
- Optional: LangSmith API key (for tracing)

### 1. Backend

```bash
cd backend
pip install -e .

# Configure environment
cp .env.example .env
# edit .env with your OPEN_ROUTER_API_KEY (and LANGSMITH_API_KEY if using tracing)

# Initialize and seed the database
python app/db_init.py

# Run the API (http://127.0.0.1:8000)
python main.py
```

### 2. Frontend

```bash
cd frontend
pnpm install

# Optional: point to a non-local backend
# export NEXT_PUBLIC_API_URL=http://127.0.0.1:8000

# Run the dev server (http://localhost:3000)
pnpm dev
```

Open `http://localhost:3000`, create a conversation, and start asking about orders, policies, tickets, or refunds. Agent progress streams live into the chat.

## API Overview

| Method | Path | Description |
|--------|------|-------------|
| POST | `/query` | Send a query to the multi-agent system |
| POST | `/query/stream` | SSE streaming query with live agent status |
| GET | `/conversations` | List conversations |
| POST | `/conversations` | Create a conversation |
| GET | `/conversations/{id}` | Get a conversation with messages |
| POST | `/conversations/{id}/messages` | Add a message to a conversation |
| GET | `/health` | Health check |
| GET | `/metrics` | Metrics for monitoring dashboards |
| GET | `/cache/stats` | Cache performance statistics |

### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the status of ORD-1001?"}'
```

Pass `thread_id` or `conversation_id` for multi-turn context:

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Can I return it?", "thread_id": "my-conversation-123"}'
```

## Configuration

Key settings live in `backend/app/config.py` and are overridable via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `primary_model` | `openrouter/google/gemini-2.0-flash-exp:free` | LLM for all agents |
| `OPEN_ROUTER_API_KEY` | `""` | OpenRouter API key |
| `APP_ENV` | `development` | `production` enables 0.0.0.0 + multiple workers |
| `RATE_LIMIT` | `20/minute` | API rate limit |
| `CACHE_TTL_SECONDS` | `300` | Response cache TTL |
| `CHECKPOINT_STORAGE` | `sqlite` | Checkpoint backend: `memory` or `sqlite` |
| `CHECKPOINT_PATH` | `./checkpoints.db` | SQLite checkpoint file path |
| `CACHE_BACKEND` | `auto` | `auto` (Redis in production, memory in dev), `memory`, or `redis` |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL for production cache |
| `DATABASE_URL` | `sqlite:///./orders.db` | App database; point to PostgreSQL in production |
| `USE_PGVECTOR` | `false` | Store embeddings in Postgres via PGVector instead of ChromaDB |
| `RETRIEVAL_TOP_K` / `RERANK_TOP_K` | `10` / `5` | Hybrid retrieval candidates / context kept after reranking |
| `MAX_RETRIEVAL_RETRIES` | `1` | Max query rewrites when retrieval fails |

## Running with Docker Compose

Both services have Dockerfiles and a root-level `docker-compose.yml` orchestrates them:

- **backend** — FastAPI + LangGraph API on port `8000` (SQLite data persisted in a `backend_data` volume)
- **frontend** — Next.js Relay Desk on port `3000` (standalone build, waits for the backend healthcheck)

```bash
# Provide your API key (create a .env in the repo root or export it)
# OPEN_ROUTER_API_KEY=sk-...   LANGSMITH_API_KEY=lsv2-...

docker compose up --build
```

Open `http://localhost:3000`. To stop: `docker compose down` (add `-v` to also delete the database volume).

### Building images individually

```bash
docker build -t multi-agents-backend ./backend
docker build -t multi-agents-frontend ./frontend
```

## Development

```bash
# Backend tests & lint
cd backend
uv run python -m pytest tests/
ruff check .

# Frontend lint & build
cd frontend
pnpm lint
pnpm build
```

## Documentation

- **[Backend README](backend/README.md)** — detailed architecture, query flows, database schema, security, and monitoring.
- **`backend/workflow.txt`** — text-based architecture diagram.