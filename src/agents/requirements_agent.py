# src/agents/requirements_agent.py
import json
import re
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END

from src.state import AgentState
from src.prompts import REQUIREMENT_GATHERING_PROMPT

load_dotenv('.env', override=True)

# ── LLM (lazy — instantiated once on first import) ────────────────────────────
_llm = None

def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv('MODEL_DEPLOYMENT', 'gpt-4o-mini'),
            temperature=float(os.getenv('TEMPERATURE', 0.0)),
            api_key=os.getenv('OPENAI_API_KEY')
        )
    return _llm


# ── Utilities ─────────────────────────────────────────────────────────────────
EXIT_PHRASES = {"exit", "quit", "bye", "goodbye", "stop", "end", "i want to exit"}

def format_chat_history(chat_history: list) -> str:
    """Converts chat history list into a readable string for prompt injection."""
    if not chat_history:
        return "No prior conversation."
    lines = []
    for msg in chat_history:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _get_latest_user_input(chat_history: list) -> str:
    for msg in reversed(chat_history):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


# ── Agent node ────────────────────────────────────────────────────────────────
def requirement_gathering_node(state: AgentState) -> dict:
    """
    Requirements Agent.
    Responsibility: gather, merge, and validate user requirements.
    Knows nothing about SQL, components, or compatibility.
    """
    print("\n--- [Requirements Agent] ---")

    chat_history = state.get("chat_history", [])
    current_requirements = state.get("user_requirements", {})
    user_input = _get_latest_user_input(chat_history)

    # ── Exit intent check ─────────────────────────────────────────────────
    if any(phrase in user_input.lower() for phrase in EXIT_PHRASES):
        return {
            "next_step": "end",
            "final_response": "Thanks for using the PC Builder! Come back anytime. 👋",
            "logs": ["User requested exit."]
        }

    # ── LLM call ──────────────────────────────────────────────────────────
    gather_prompt = ChatPromptTemplate.from_messages([
        ("system", REQUIREMENT_GATHERING_PROMPT),
        ("user", "Extract and update requirements now.")
    ])

    try:
        response = (gather_prompt | get_llm()).invoke({
            "chat_history": format_chat_history(chat_history),
            "current_requirements": str(current_requirements),
            "user_input": user_input
        })
    except Exception as e:
        return {
            "next_step": "respond_to_user",
            "no_build_reason": None,
            "logs": [f"[RequirementsAgent] LLM call failed: {str(e)}"]
        }

    # ── Parse ─────────────────────────────────────────────────────────────
    try:
        raw = re.sub(r"^```json|^```|```$", "", response.content.strip(), flags=re.MULTILINE).strip()
        new_requirements: dict = json.loads(raw)
    except json.JSONDecodeError as e:
        return {
            "next_step": "respond_to_user",
            "no_build_reason": None,
            "logs": [f"[RequirementsAgent] JSON parse failed: {str(e)}"]
        }

    # ── Merge — never discard already-collected fields ────────────────────
    merged = {
        **current_requirements,
        **{k: v for k, v in new_requirements.items() if v is not None and v != [] and v != ""}
    }

    print(f"-> Merged requirements: {merged}")
    is_complete = merged.get("is_complete", False)

    return {
        "user_requirements": merged,
        "next_step": "query_database" if is_complete else "respond_to_user",
        "logs": [f"[RequirementsAgent] Complete: {is_complete} | Requirements: {merged}"]
    }
