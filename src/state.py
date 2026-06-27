# src/state.py
import operator
from typing import TypedDict, List, Annotated, Optional
import uuid
from datetime import datetime


class AgentState(TypedDict):
    chat_history: List[dict]
    user_requirements: dict
    current_build: Optional[dict]
    next_step: str
    critique_feedback: Optional[list]
    no_build_reason: Optional[str]
    final_response: Optional[str]
    # logs will hold structured log entries (dicts). Keep the Annotated for append semantics.
    logs: Annotated[List[dict], operator.add]
    critique_iterations: int
    supervisor_steps: int
    # Observability fields
    run_id: Optional[str]
    started_at: Optional[str]
    llm_call_count: int
    last_llm_error: Optional[str]



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
    "supervisor_steps": 0,
    "run_id": str(uuid.uuid4()),
    "started_at": datetime.utcnow().isoformat() + "Z",
    "llm_call_count": 0,
    "last_llm_error": None,
}
