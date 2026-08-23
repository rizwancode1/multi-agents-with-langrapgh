"""
Shared helpers for agent nodes.

Previously every agent duplicated an ``_add_route`` function that used the
``__import__("time")`` hack. Centralized here with a proper import.
"""

import time

from app.agents.state import AgentState


def add_route(state: AgentState, agent: str, action: str) -> list[dict]:
    """Append a routing trace entry and return the new route list."""
    route = list(state.get("route", []))
    route.append({
        "agent": agent,
        "action": action,
        "timestamp": time.time(),
    })
    return route
