# src/agent.py — full file after all extractions

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from src.state import AgentState, DEFAULT_INITIAL_STATE
from src.agents.requirements_agent import requirement_gathering_node
from src.agents.query_agent import query_database_node
from src.agents.critique_agent import self_critique_node
from src.agents.response_agent import respond_to_user_node, stream_final_response

load_dotenv('.env', override=True)


# ── Supervisor ────────────────────────────────────────────────────────────────
def supervisor(state: AgentState) -> str:
    """
    Central router. Reads next_step from state and returns the node name.
    This is the only place routing logic lives — no node sets its own destination
    beyond writing next_step into state.

    Routing table:
        gather          → requirements agent
        query_database  → query agent
        self_critique   → critique agent
        respond_to_user → response agent
        awaiting_user   → END (waiting for next user message)
        end             → END (terminal)
        <anything else> → END (safe fallback)
    """
    next_step = state.get("next_step", "end")

    routing_map = {
        "gather":           "gather_requirements",
        "query_database":   "query_database",
        "self_critique":    "self_critique",
        "respond_to_user":  "respond_to_user",
        "awaiting_user":    END,
        "end":              END,
    }

    destination = routing_map.get(next_step, END)
    print(f"\n[Supervisor] '{next_step}' → '{destination}'")
    return destination


# ── Graph ─────────────────────────────────────────────────────────────────────
workflow = StateGraph(AgentState)

workflow.add_node("gather_requirements", requirement_gathering_node)
workflow.add_node("query_database",      query_database_node)
workflow.add_node("self_critique",       self_critique_node)
workflow.add_node("respond_to_user",     respond_to_user_node)

workflow.set_entry_point("gather_requirements")

# Every node routes through the supervisor — one conditional function, one place
for node in ["gather_requirements", "query_database", "self_critique", "respond_to_user"]:
    workflow.add_conditional_edges(node, supervisor, {
        "gather_requirements": "gather_requirements",
        "query_database":      "query_database",
        "self_critique":       "self_critique",
        "respond_to_user":     "respond_to_user",
        END:                   END,
    })

pc_config_agent = workflow.compile()
