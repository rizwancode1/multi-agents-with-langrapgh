from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


AgentName = Literal[
    "router",
    "rag",
    "order",
    "coding",
    "review",
    "evaluator",
]


class AgentState(TypedDict, total=False):
    query: str

    current_agent: AgentName

    response: str

    context: list[str]

    order_data: dict

    code_result: str

    citations: list[str]

    next_agent: AgentName | None

    handoff_reason: str | None

    evaluation: dict

    handoff_count: int

    visited_agents: list[str]

    error: str | None


AGENT_CAPABILITIES = {
    "rag": {
        "description": "Retrieves information from company documents",
        "can_handle": [
            "documentation",
            "policies",
            "FAQs",
            "knowledge base",
        ],
        "can_handoff_to": [
            "evaluator",
        ],
    },

    "order": {
        "description": "Retrieves customer order information",
        "can_handle": [
            "order status",
            "order history",
            "customer order",
        ],
        "can_handoff_to": [
            "rag",
            "evaluator",
        ],
    },

    "coding": {
        "description": "Writes and analyzes code",
        "can_handle": [
            "coding",
            "bugs",
            "implementation",
        ],
        "can_handoff_to": [
            "review",
            "evaluator",
        ],
    },

    "review": {
        "description": "Reviews generated code",
        "can_handle": [
            "code review",
            "security review",
            "quality review",
        ],
        "can_handoff_to": [
            "coding",
            "evaluator",
        ],
    },
}




llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)


class RouteDecision(BaseModel):
    agent: Literal[
        "rag",
        "order",
        "coding",
    ]

    reason: str = Field(
        description="Why this agent should handle the request"
    )


router_llm = llm.with_structured_output(RouteDecision)



def router_node(state: AgentState):
    query = state["query"]

    decision = router_llm.invoke(
        f"""
You are a routing agent.

Choose the ONE specialized agent that should handle the user's request.

Available agents:

1. rag
   - documentation
   - policies
   - FAQs
   - knowledge base

2. order
   - customer orders
   - order status
   - order history

3. coding
   - programming
   - debugging
   - implementation

User request:
{query}
"""
    )

    return {
        "current_agent": "router",
        "next_agent": decision.agent,
        "handoff_reason": decision.reason,
    }



""""
Suppose the user asks:

What is our refund policy?

The router returns:

{
    "agent": "rag",
    "reason": "The request requires company policy information."
}

The Order Agent never runs.

"""



def rag_node(state: AgentState):
    query = state["query"]

    # Your actual retriever would be here
    # documents = retrieve_documents(query)

    documents = []

    context = [
        doc.page_content
        for doc in documents
    ]

    answer = llm.invoke(
        f"""
Answer the user using ONLY the provided context.

Question:
{query}

Context:
{context}
"""
    )

    return {
        "current_agent": "rag",
        "response": answer.content,
        "context": context,
        "visited_agents": state.get("visited_agents", []) + ["rag"],
        "handoff_count": state.get("handoff_count", 0),
    }


"""Hybrid Retriever
      ↓
BM25
      +
Dense Retrieval
      ↓
RRF
      ↓
Reranker
      ↓
Contextual Compression
      ↓
RAG Agent"""





# Order Agent

def order_node(state: AgentState):
    query = state["query"]

    # order_id = extract_order_id(query)

    # order = get_order_from_api(order_id)

    order = {
        "id": "ord1134",
        "price": "400"
    }

    return {
        "current_agent": "order",
        "order_data": order,
        "visited_agents": state.get("visited_agents", []) + ["order"],
        "handoff_count": state.get("handoff_count", 0),
    }



# Controlled P2P handoff 

def request_handoff(
    state: AgentState,
    target: AgentName,
    reason: str,
):
    current_agent = state["current_agent"]

    allowed = AGENT_CAPABILITIES.get(
        current_agent,
        {}
    ).get("can_handoff_to", [])

    if target not in allowed:
        raise ValueError(
            f"{current_agent} cannot handoff to {target}"
        )

    count = state.get("handoff_count", 0)

    if count >= 5:
        raise RuntimeError(
            "Maximum handoff limit reached"
        )

    visited = state.get("visited_agents", [])

    return {
        "next_agent": target,
        "handoff_reason": reason,
        "handoff_count": count + 1,
        "visited_agents": visited,
    }


"""
Now an agent can request:

return request_handoff(
    state,
    target="rag",
    reason="Refund policy is required.",
)

But the framework checks:

Is Order → RAG allowed?

Yes.

Therefore:

Order Agent
     ↓
RAG Agent

is allowed.
"""


# Coding → Review


def coding_node(state: AgentState):
    query = state["query"]

    generated_code = llm.invoke(
        f"""
Implement the following requirement:

{query}
"""
    )

    return {
        "current_agent": "coding",
        "code_result": generated_code.content,
        "visited_agents": state.get("visited_agents", []) + ["coding"],
    }


def review_node(state: AgentState):
    code = state["code_result"]

    review = llm.invoke(
        f"""
Review the following code for:

- correctness
- security
- performance
- maintainability

Code:

{code}
"""
    )

    return {
        "current_agent": "review",
        "response": review.content,
        "visited_agents": state.get("visited_agents", []) + ["review"],
    }



# The evaluator is extremely important

class EvaluationResult(BaseModel):
    passed: bool
    score: float
    feedback: str
    retry_agent: str | None


evaluator_llm = llm.with_structured_output(
    EvaluationResult
)


def evaluator_node(state: AgentState):
    response = state.get("response", "")
    context = state.get("context", [])

    result = evaluator_llm.invoke(
        f"""
Evaluate the answer.

Answer:
{response}

Context:
{context}

Check:

1. Is the answer grounded?
2. Is it relevant?
3. Is it factually supported?
4. Are there unsupported claims?
5. Should another agent retry?
"""
    )

    return {
        "current_agent": "evaluator",
        "evaluation": result.model_dump(),
    }



"""
Conditional routing after evaluation

Now we decide:

                Evaluator
                    │
             ┌──────┴───────┐
             │              │
           PASS           FAIL
             │              │
             ▼              ▼
            END       Retry appropriate agent

"""


def evaluator_router(state: AgentState):
    evaluation = state["evaluation"]

    if evaluation["passed"]:
        return "end"

    retry_agent = evaluation.get("retry_agent")

    if retry_agent:
        return retry_agent

    return "end"




# Build the LangGraph


from langgraph.graph import StateGraph, START, END


builder = StateGraph(AgentState)

builder.add_node("router", router_node)
builder.add_node("rag", rag_node)
builder.add_node("order", order_node)
builder.add_node("coding", coding_node)
builder.add_node("review", review_node)
builder.add_node("evaluator", evaluator_node)


builder.add_edge(START, "router")


def initial_router(state: AgentState):
    return state["next_agent"]


builder.add_conditional_edges(
    "router",
    initial_router,
    {
        "rag": "rag",
        "order": "order",
        "coding": "coding",
    },
)

builder.add_conditional_edges(
    "rag",
    lambda state: "evaluator",
    {
        "evaluator": "evaluator"
    },
)

builder.add_conditional_edges(
    "order",
    lambda state: "evaluator",
    {
        "evaluator": "evaluator"
    },
)

builder.add_conditional_edges(
    "coding",
    lambda state: "review",
    {
        "review": "review"
    },
)

builder.add_edge("review", "evaluator")

builder.add_conditional_edges(
    "evaluator",
    evaluator_router,
    {
        "end": END,
        "rag": "rag",
        "order": "order",
        "coding": "coding",
        "review": "review",
    },
)

graph = builder.compile()


"""12. But this isn't quite P2P yet

Notice:

Router
 ↓
Agent
 ↓
Evaluator

This is still mostly centralized.

To introduce genuine peer handoffs, we can use LangGraph's Command.

For example:from langgraph.types import Command


def order_node(state: AgentState):

    order = get_order_from_api(
        extract_order_id(state["query"])
    )

    # Suppose we discover we need policy information
    return Command(
        update={
            "current_agent": "order",
            "order_data": order,
            "handoff_reason": (
                "Need refund policy information"
            ),
        },
        goto="rag",
    )"""



