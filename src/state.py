# src/state.py
import operator
from typing import TypedDict, List, Annotated, Optional


class AgentState(TypedDict):
    chat_history: List[dict]
    user_requirements: dict
    current_build: Optional[dict]
    next_step: str
    critique_feedback: Optional[list]
    no_build_reason: Optional[str]
    final_response: Optional[str]
    logs: Annotated[List[str], operator.add]
    critique_iterations: int
    supervisor_steps: int          # ← ADD THIS



DEFAULT_INITIAL_STATE: AgentState = {
    "chat_history": [],
    "user_requirements": {},
    "current_build": None,
    "next_step": "gather",
    "critique_feedback": None,
    "no_build_reason": None,
    "final_response": None,
    "logs": [],
    "critique_iterations": 0,
    "supervisor_steps": 0
}
