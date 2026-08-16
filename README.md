# Multi-Agent System

A production-ready multi-agent system built with LangGraph, FastAPI, and OpenRouter. The system routes user queries to specialized agents, enables peer-to-peer handoffs, and evaluates responses before returning them.

## Architecture

```
User Request
     │
     ▼
 Router Agent (intent classification)
     │
     ├───────────────────────┐
     ▼                       ▼                       ▼
 RAG Agent            Order Agent            Coding Agent
 (documents)          (order lookup)         (code generation)
     │                       │                       │
     └──────────────┬────────┴───────────┬──────────┘
                    ▼                    ▼
                 Evaluator (grounding / correctness)
                    │
            ┌───────┴───────┐
            ▼               ▼
          PASS             FAIL
            │               │
            ▼               ▼
          END       Retry / Handoff to agent
```

### Agents

| Agent | Purpose | Handoff To |
|-------|---------|------------|
| **Router** | Classifies intent and routes to the right specialist | rag, order, coding |
| **RAG** | Retrieves and answers from the knowledge base | evaluator |
| **Order** | Looks up customer orders from `orders.json` | rag, evaluator |
| **Coding** | Generates code from requirements | review, evaluator |
| **Review** | Reviews generated code for correctness, security, performance, maintainability | coding, evaluator |
| **Evaluator** | Scores the final response for grounding and correctness; retries failed agents | rag, order, coding, review, end |

## Key Features

- **Controlled P2P Handoffs**: Agents can hand off to each other within declared capability boundaries. Invalid handoffs are blocked.
- **Handoff Loop Protection**: Maximum 5 handoffs per request to prevent infinite loops.
- **Evaluator Feedback Loop**: If the evaluator finds issues, it can route back to the responsible agent for retry.
- **Structured Routing**: Router uses LLM structured output (`RouteDecision`) for reliable intent classification.
- **Structured Evaluation**: Evaluator uses LLM structured output (`EvaluationResult`) for consistent scoring.

## Project Structure

```
multi-agents/
├── app/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── state.py           # Shared AgentState TypedDict
│   │   ├── router_agent.py    # Router node with structured output
│   │   ├── rag_agent.py       # RAG retrieval from knowledge base
│   │   ├── order_agent.py     # Order lookup from orders.json
│   │   ├── coding_agent.py    # Code generation
│   │   ├── review_agent.py    # Code review
│   │   ├── evaluator_agent.py # Evaluation and retry routing
│   │   └── graph.py           # LangGraph workflow definition
│   ├── api.py                 # FastAPI endpoints
│   ├── config.py              # Centralized settings (pydantic-settings)
│   ├── security.py            # Input sanitization, PII detection/masking
│   ├── monitoring.py          # Structured JSON logging + metrics
│   ├── utils.py               # Embeddings, BM25 tokenizer, stable IDs
│   ├── models.py              # (extend for DB models)
│   └── data/
│       ├── orders.json        # Sample order data
│       └── rag_knowledge_base.json  # Sample RAG documents
├── main.py                    # Uvicorn entry point
├── pyproject.toml
├── workflow.txt               # Original architecture diagram
└── prototypes.py              # Original prototypes
```

## Getting Started

### Prerequisites

- Python >= 3.12
- OpenRouter API key (for LLM calls)

### Installation

```bash
cd "D:\Data\PROGRAMMING PLANET\LangRag\multi-agents"
uv sync
```

### Configuration

Create a `.env` file in the project root:

```env
OPEN_ROUTER_API_KEY=your_key_here
OPEN_ROUTER_BASE_URL=https://openrouter.ai/api/v1
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=multi-agent-system
```

Key settings in `app/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `primary_model` | `openrouter/google/gemini-2.0-flash-exp:free` | LLM for all agents |
| `fallback_model` | same as primary | Fallback if primary fails |
| `embedding_model` | `text-embedding-3-small` | Embedding model for RAG |
| `dataset_dir` | `./dataset-docs` | Directory for ingestion |
| `chroma_persist_dir` | `./chroma_db` | ChromaDB persistence path |
| `retrieval_top_k` | `10` | Number of documents to retrieve |
| `rerank_top_k` | `5` | Documents after reranking |

### Running the Server

```bash
cd "D:\Data\PROGRAMMING PLANET\LangRag\multi-agents"
uv run python main.py
```

The server starts at `http://127.0.0.1:8000`.

### Environment Modes

Set `APP_ENV` in `.env` to control runtime behavior:

| `APP_ENV` | Host | Reload | Workers | Use Case |
|-----------|------|--------|---------|----------|
| `development` (default) | `127.0.0.1` | enabled | `1` | Local development |
| `production` | `0.0.0.0` | disabled | `WORKERS` env var (default `4`) | Docker / production |

**Production example:**
```bash
APP_ENV=production WORKERS=4 uv run python main.py
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/query` | Send a query to the multi-agent system |
| GET | `/health` | Health check |

### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?"}'
```

```json
{
  "query": "What is the refund policy?",
  "response": "...",
  "agent_used": "rag",
  "visited_agents": ["router", "rag", "evaluator"],
  "citations": ["DOC-RET-01"]
}
```

## Workflow

See `workflow.txt` for the original architecture diagram.

### Query Flow

1. **Router** receives the query and classifies it into one of three intents: `rag`, `order`, or `coding`.
2. **Specialized Agent** processes the request:
   - **RAG**: Retrieves relevant documents and generates a grounded answer.
   - **Order**: Looks up order information from `orders.json`.
   - **Coding**: Generates code, then hands off to **Review** for analysis.
3. **Evaluator** checks the response for grounding, relevance, and factual support.
4. If the evaluator **passes**, the response is returned to the user.
5. If the evaluator **fails**, it routes back to the responsible agent for retry (up to 5 handoffs).

## Security

- **Input Sanitization**: Detects prompt injection patterns before LLM calls.
- **PII Masking**: Redacts emails, phone numbers, SSNs, and credit cards from both input and output.
- **Output Validation**: Blocks potentially harmful content in LLM responses.

## Monitoring

- **Structured JSON Logging**: All logs output as JSON for ELK/Datadog ingestion.
- **Metrics Collector**: Tracks request count, latency, error rate, cache hit rate, and token usage.
- **LangSmith Tracing**: Full trace visibility via LangSmith integration.

## Development

```bash
# Run tests
pytest

# Lint / typecheck
ruff check .
```
