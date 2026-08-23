# Multi-Agent Support System

A production-ready multi-agent support system built with LangGraph, FastAPI, SQLAlchemy, and OpenRouter. The system routes user queries to specialized support agents, enables peer-to-peer handoffs, evaluates responses for grounding and safety, and persists conversation state via LangGraph checkpoints.

## Architecture

```
                          START
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Support Supervisor  │
                 │ Intent + Planning   │
                 └──────────┬──────────┘
                            │
           ┌────────────────┼─────────────────────┐
           │                │                     │
           ▼                ▼                     ▼
      ┌─────────┐      ┌─────────┐         ┌─────────────┐
      │  Order  │      │ Policy  │         │   Support   │
      │  Agent  │      │   RAG   │         │ Ticket Agent│
      └────┬────┘      └────┬────┘         └──────┬──────┘
           │                │                     │
           │                │                     │
           ▼                ▼                     ▼
       Order Tools      Hybrid RAG            Ticket Tools
                            │
                            │
                            ▼
                     ┌──────────────┐
                     │ Return/Refund│
                     │    Agent     │
                     └──────┬───────┘
                            │
                            ▼
                       Action Tools
                            │
                            ▼
                  ┌──────────────────┐
                  │     Evaluator    │
                  └────────┬─────────┘
                           │
                   ┌───────┴────────┐
                   │                │
                 PASS              FAIL
                   │                │
                   ▼                ▼
               Formatter        Retry/Repair
                   │
                   ▼
                  END
```

### Agents

| Agent | Purpose | Handoff To | Tools |
|-------|---------|------------|-------|
| **Supervisor** | Intent detection, planning, and orchestration | order, policy_rag, support_ticket | — |
| **Order** | Retrieves customer order information from the database | evaluator | `search_orders`, `get_order_by_id`, `get_order_by_customer_name`, `get_order_items` |
| **Policy RAG** | Agentic RAG: contextualize → hybrid retrieval (dense + BM25/RRF) → LLM rerank → grade context → generate → groundedness check | return_refund, evaluator | — |
| **Support Ticket** | Creates and tracks support tickets | evaluator | `create_support_ticket`, `get_ticket_status`, `list_tickets_by_email`, `update_ticket_status` |
| **Return/Refund** | Processes return/refund requests and refund status lookups | evaluator | `calculate_eligible_refund`, `create_refund_request`, `get_refund_status`, `list_refunds_by_order` |
| **Evaluator** | Scores responses for grounding, correctness, and safety; retries failed agents | formatter, order, policy_rag, support_ticket, return_refund, end | — |
| **Formatter** | Produces the final user-facing response | end | — |

## Key Features

- **6 Specialized Agents**: Order, Policy RAG, Support Ticket, Return/Refund, Evaluator, and Formatter
- **LangChain @tool Bindings**: Agents use structured tool calls to create and retrieve real DB records (tickets, refunds, orders)
- **Actual DB Records**: Support tickets (`TKT-...`) and refund requests (`REF-...`) are persisted and trackable by ID
- **Controlled P2P Handoffs**: Agents hand off within declared capability boundaries (e.g., Policy RAG → Return/Refund)
- **Handoff Loop Protection**: Maximum 5 handoffs per request to prevent infinite loops
- **Structured Routing**: Router uses LLM structured output (`RouteDecision`) for reliable intent classification
- **Structured Evaluation**: Evaluator uses LLM structured output (`EvaluationResult`) for consistent scoring
- **Database-Backed Orders**: SQLAlchemy ORM with SQLite for order storage and retrieval
- **Agentic RAG Pipeline**: Contextualization → hybrid retrieval → LLM reranking → context grading → generation → groundedness verification, with query-rewrite and complementary-retrieval loops
- **Hybrid Retrieval**: Dense (ChromaDB or PGVector) + sparse (BM25) search fused via Reciprocal Rank Fusion (k=60)
- **LLM Reranking**: Retrieved chunks scored 0–10 by the primary LLM; only the top `RERANK_TOP_K` are kept for generation
- **Pluggable Vector Store**: Local ChromaDB by default; opt-in Postgres PGVector via `USE_PGVECTOR=true` (independent of the main app database choice)
- **LangGraph Checkpoints**: Persistent conversation state via `langgraph-checkpoint-sqlite` with configurable storage backends
- **Response Caching**: In-memory in development, Redis in production (`CACHE_BACKEND=auto`) — shared across instances with native TTL
- **Security Pipeline**: Input sanitization, PII masking, and output validation
- **Observability**: Structured JSON logging, metrics collection, and LangSmith tracing

## Project Structure

```
multi-agents/
├── app/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── state.py              # Shared AgentState TypedDict
│   │   ├── router_agent.py       # Supervisor node with structured output
│   │   ├── rag_agent.py          # Policy RAG retrieval
│   │   ├── order_agent.py        # Order lookup via bound DB tools
│   │   ├── support_ticket_agent.py  # Support ticket handling via bound tools
│   │   ├── return_refund_agent.py   # Return/refund processing via bound tools
│   │   ├── evaluator_agent.py    # Evaluation and retry routing
│   │   ├── formatter_agent.py    # Final response formatting
│   │   └── graph.py              # LangGraph workflow definition
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── order_tools.py        # @tool decorated order retrieval tools
│   │   ├── support_tools.py      # @tool decorated support ticket tools
│   │   └── refund_tools.py       # @tool decorated refund request tools
│   ├── api.py                    # FastAPI endpoints
│   ├── config.py                 # Centralized settings (pydantic-settings)
│   ├── models.py                 # API request/response models
│   ├── models_db.py              # SQLAlchemy ORM models (Order, SupportTicket, RefundRequest)
│   ├── db.py                     # Database engine and session management
│   ├── db_init.py                # Database initialization and seeding
│   ├── checkpoints.py            # LangGraph checkpoint factory
│   ├── cache.py                  # Response caching (memory dev / Redis production)
│   ├── ingestion.py              # KB ingestion → Chroma/PGVector + BM25 index
│   ├── retrieval.py              # Hybrid dense+BM25 retriever with RRF fusion
│   ├── reranker.py               # LLM relevance reranker (0–10 scoring)
│   ├── utils.py                  # OpenRouter embeddings wrapper, BM25 tokenizer, ID helpers
│   ├── security.py               # Input/output security pipeline
│   ├── monitoring.py             # Logging and metrics
│   └── data/
│       ├── orders.json           # Seed data for orders
│       └── rag_knowledge_base.json  # Sample RAG documents
├── tests/                        # Unit tests (retrieval, reranker, cache, ingestion)
├── main.py                       # Uvicorn entry point
├── pyproject.toml
└── workflow.txt                  # Architecture diagram
```

## Getting Started

### Prerequisites

- Python >= 3.12
- OpenRouter API key (for LLM calls)

### Installation

```bash
cd "D:\Data\PROGRAMMING PLANET\LangRag\multi-agents"
pip install -e .
```

### Database Setup

Initialize the SQLite database and seed it with sample orders:

```bash
python app/db_init.py
```

### Configuration

Create a `.env` file in the project root:

```env
OPEN_ROUTER_API_KEY=your_key_here
OPEN_ROUTER_BASE_URL=https://openrouter.ai/api/v1
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=multi-agent-system

# Checkpoint configuration
CHECKPOINT_STORAGE=sqlite
CHECKPOINT_PATH=./checkpoints.db
CHECKPOINT_TABLE=checkpoints

# Cache (Redis is selected automatically when APP_ENV=production)
CACHE_BACKEND=auto
REDIS_URL=redis://localhost:6379/0

# Database (PostgreSQL in production; also used by PGVector when enabled)
DATABASE_URL=sqlite:///./orders.db
DB_ECHO_LOGS=false

# Vector store for RAG embeddings
USE_PGVECTOR=false
PGVECTOR_COLLECTION=policy_docs

# RAG pipeline tuning
RETRIEVAL_TOP_K=10
RERANK_TOP_K=5
MAX_RETRIEVAL_RETRIES=1
INGEST_ON_STARTUP=true
```

Key settings in `app/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `primary_model` | `openrouter/google/gemini-2.0-flash-exp:free` | LLM for all agents |
| `openrouter_api_key` | `""` | OpenRouter API key |
| `app_env` | `development` | Environment mode |
| `cache_ttl_seconds` | `300` | Response cache TTL |
| `cache_backend` | `auto` | Cache backend: `auto`, `memory`, or `redis` |
| `redis_url` | `redis://localhost:6379/0` | Redis connection URL |
| `database_url` | `sqlite:///./orders.db` | App database connection string |
| `use_pgvector` | `false` | Store embeddings via PGVector instead of ChromaDB |
| `retrieval_top_k` / `rerank_top_k` | `10` / `5` | Retrieval candidates / context kept after reranking |
| `max_retrieval_retries` | `1` | Max query rewrites on retrieval failure |
| `checkpoint_storage` | `sqlite` | Checkpoint backend: `memory` or `sqlite` |
| `checkpoint_path` | `./checkpoints.db` | SQLite checkpoint file path |

### Running the Server

```bash
cd "D:\Data\PROGRAMMING PLANET\LangRag\multi-agents"
python main.py
```

The server starts at `http://127.0.0.1:8000`.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/query` | Send a query to the multi-agent system |
| GET | `/tickets` | List agent-created tickets (filters: `status`, `priority`, `email`, `order_id`, `q`; pagination) |
| GET | `/tickets/stats` | Ticket counts by status and priority |
| GET | `/tickets/{ticket_id}` | Get a single ticket |
| PATCH | `/tickets/{ticket_id}` | Update ticket status/priority |
| GET | `/health` | Health check |
| GET | `/metrics` | Metrics for monitoring dashboards |
| GET | `/cache/stats` | Cache performance statistics |

### Example Requests

**Order lookup:**
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the status of ORD-1001?"}'
```

**Create a support ticket:**
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "I have a complaint about my order ORD-1001, the item arrived damaged"}'
```
Response includes a real ticket ID: `TKT-...`

**Check ticket status:**
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the status of ticket TKT-B1AD8289?"}'
```

**Request a refund:**
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "I want to return the headphones from ORD-1001"}'
```
Response includes a real refund ID: `REF-...`

**Check refund status:**
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the status of refund REF-DE689742?"}'
```

### Multi-Turn Conversations

Pass a `thread_id` to maintain conversation context across requests:

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Can I return it?", "thread_id": "my-conversation-123"}'
```

## Workflow

See `workflow.txt` for the architecture diagram.

### Policy RAG Pipeline

The Policy RAG agent runs an agentic retrieval loop as a compiled sub-graph:

```
Contextualize (rewrite follow-up into standalone question)
    │
    ▼
Hybrid Retrieve (dense + BM25 fused via Reciprocal Rank Fusion, k=60)
    │
    ▼
Rerank (LLM scores each chunk 0–10 → top RERANK_TOP_K)
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

- **Ingestion** (`app/ingestion.py`): loads `app/data/rag_knowledge_base.json`, enriches chunks with contextual prefixes, embeds them via the custom OpenRouter embeddings wrapper (`app/utils.py`), and persists to ChromaDB or PGVector plus a pickled BM25 index. Runs at startup when `INGEST_ON_STARTUP=true`.
- **Retrieval** (`app/retrieval.py`): both retrievers fetch `2 × top_k` candidates before fusion; complementary retrieval excludes already-seen sources for diversity.
- **Reranking** (`app/reranker.py`): identical queries are cached to avoid re-scoring; LLM failures score 0 and are deprioritized.

### Query Flow

1. **Supervisor** receives the query, classifies intent, and routes to the appropriate specialist
2. **Specialized Agent** processes the request using bound LangChain tools:
   - **Order**: Uses `search_orders`, `get_order_by_id`, etc. to retrieve real order data from the database
   - **Policy RAG**: Retrieves relevant documents and generates a grounded answer; may hand off to Return/Refund
   - **Support Ticket**: Uses `create_support_ticket` to create real tickets with trackable `TKT-...` IDs; uses `get_ticket_status` for status lookups
   - **Return/Refund**: Uses `calculate_eligible_refund` and `create_refund_request` to create real refund requests with trackable `REF-...` IDs; uses `get_refund_status` for lookups
3. **Evaluator** checks the response for grounding, relevance, and factual support
4. If the evaluator **passes**, the response goes to the Formatter
5. If the evaluator **fails**, it routes back to the responsible agent for retry (up to 5 handoffs)

## Database

The system uses SQLAlchemy with SQLite for persistent storage:

- `app/models_db.py` — ORM models: `Order`, `OrderItem`, `SupportTicket`, `RefundRequest`
- `app/db.py` — Engine, session factory, and context managers
- `app/tools/order_tools.py` — `@tool` decorated order retrieval tools
- `app/tools/support_tools.py` — `@tool` decorated support ticket tools
- `app/tools/refund_tools.py` — `@tool` decorated refund request tools
- `app/db_init.py` — Seeding script

### Trackable IDs

| Entity | ID Format | Example |
|--------|-----------|---------|
| Order | `ORD-XXXX` | `ORD-1001` |
| Support Ticket | `TKT-XXXXXXXX` | `TKT-B1AD8289` |
| Refund Request | `REF-XXXXXXXX` | `REF-DE689742` |

The database is driven by `DATABASE_URL` in `.env` — SQLite by default, PostgreSQL in production (also required when `USE_PGVECTOR=true`, with the `vector` extension enabled: `CREATE EXTENSION vector;`).

## Checkpoints

LangGraph checkpoints enable stateful multi-turn conversations:

- **Memory**: In-memory (non-persistent, development)
- **SQLite**: Persistent local storage via `langgraph-checkpoint-sqlite`

Configure via `CHECKPOINT_STORAGE`, `CHECKPOINT_PATH`, and `CHECKPOINT_TABLE` in `.env`.

## Caching

Response caching deduplicates identical queries:

- **Memory** (`CACHE_BACKEND=memory`): per-instance dict with TTL (development default)
- **Redis** (`CACHE_BACKEND=redis`, or `auto` + `APP_ENV=production`): shared across instances, native TTL expiry, survives restarts; falls back to memory with a warning if Redis is unreachable

Inspect hit/miss rates via `GET /cache/stats`.

## Security

- **Input Sanitization**: Detects prompt injection patterns before LLM calls
- **PII Masking**: Redacts emails, phone numbers, SSNs, and credit cards from both input and output
- **Output Validation**: Blocks potentially harmful content in LLM responses

## Monitoring

- **Structured JSON Logging**: All logs output as JSON for ELK/Datadog ingestion
- **Metrics Collector**: Tracks request count, latency, error rate, cache hit rate, and token usage
- **LangSmith Tracing**: Full trace visibility via LangSmith integration

## Development

```bash
# Run tests (retrieval RRF fusion, reranker, cache, ingestion)
uv run python -m pytest tests/ -v

# Lint / typecheck
ruff check .
```
