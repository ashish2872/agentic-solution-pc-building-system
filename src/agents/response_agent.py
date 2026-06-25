# src/agents/response_agent.py
import os
import json
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.state import AgentState
from src.agents.requirements_agent import format_chat_history

load_dotenv('.env', override=True)

# ── Lazy LLM ──────────────────────────────────────────────────────────────────
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


# ── Prompts ───────────────────────────────────────────────────────────────────
CLARIFICATION_PROMPT = """
You are a friendly PC building assistant having a conversation with a user.

Here is the conversation so far:
{chat_history}

The user's requirements collected so far:
{requirements}

The system needs clarification or has detected ambiguity/conflict.
Clarification message from the system: {clarification_message}

Rephrase this into a warm, conversational message to the user.
Do NOT ask for information they have already provided.
Keep it to 2-3 sentences.
"""

NO_BUILD_PROMPT = """
You are a helpful PC building assistant.

Here is the conversation so far:
{chat_history}

User requirements:
{requirements}

A compatible build could not be assembled. Reason:
{reason}

Explain this conversationally in 3-4 sentences:
1. Acknowledge what the user asked for
2. Explain specifically why it didn't work
3. Suggest ONE concrete change they can make (e.g. increase budget by $X,
   relax a specific constraint, choose a different component type)

Do not repeat information they already gave. Do not use bullet points.
Sound like a knowledgeable friend, not an error message.
"""

STREAMING_RESPONSE_PROMPT = """
You are a friendly expert PC builder assistant.
Given the final PC build configuration and the user's requirements,
present the build in a clear, conversational way.
Explain WHY each component was chosen for their specific use case.
If there are compatibility notes, explain them clearly.
Format the response with a short section per component.
End with the total price and a one-line summary of why this build fits their needs.
"""


# ── Sub-step A: Generate clarification message ────────────────────────────────
def _generate_clarification(
    requirements: dict,
    chat_history: list,
    clarification_message: str
) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", CLARIFICATION_PROMPT),
        ("user", "Generate the clarification message now.")
    ])
    try:
        response = (prompt | get_llm()).invoke({
            "chat_history": format_chat_history(chat_history),
            "requirements": str(requirements),
            "clarification_message": clarification_message
        })
        return response.content.strip()
    except Exception:
        # Fall back to raw clarification message if LLM fails
        return clarification_message


# ── Sub-step B: Generate no-build explanation ────────────────────────────────
def _generate_no_build_explanation(
    requirements: dict,
    chat_history: list,
    reason: str
) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", NO_BUILD_PROMPT),
        ("user", "Generate the explanation now.")
    ])
    try:
        response = (prompt | get_llm()).invoke({
            "chat_history": format_chat_history(chat_history),
            "requirements": str(requirements),
            "reason": reason
        })
        return response.content.strip()
    except Exception:
        # Graceful fallback — never show a raw exception to the user
        return (
            f"I wasn't able to put together a compatible build with your current requirements. "
            f"{reason} — would you like to adjust your budget or relax any constraints?"
        )


# ── Streaming generator (called by app.py directly) ──────────────────────────
def stream_final_response(build: dict, requirements: dict):
    """
    Streams the final build explanation token by token.
    Use as a generator — pass directly to st.write_stream().

    Usage:
        streamed = st.write_stream(stream_final_response(build, requirements))
    """
    streaming_llm = ChatOpenAI(
        model=os.getenv('MODEL_DEPLOYMENT', 'gpt-4o-mini'),
        temperature=0.3,
        api_key=os.getenv('OPENAI_API_KEY'),
        streaming=True
    )

    messages = [
        SystemMessage(content=STREAMING_RESPONSE_PROMPT),
        HumanMessage(content=(
            f"User Requirements:\n{json.dumps(requirements, indent=2)}\n\n"
            f"Final PC Build:\n{json.dumps(build, indent=2)}\n\n"
            f"Present this build to the user now."
        ))
    ]

    for chunk in streaming_llm.stream(messages):
        if chunk.content:
            yield chunk.content


# ── Main node ─────────────────────────────────────────────────────────────────
def respond_to_user_node(state: AgentState) -> dict:
    """
    Response Agent.
    Responsibility: generate all user-facing text.
    Knows nothing about SQL, compatibility logic, or routing decisions.
    Three cases: clarification needed, build ready, no compatible build.
    """
    print("\n--- [Response Agent] ---")

    requirements = state.get("user_requirements", {})
    build = state.get("current_build")
    no_build_reason = state.get("no_build_reason")
    chat_history = state.get("chat_history", [])

    # ── Case 1: Ambiguous or conflicting requirements ─────────────────────
    if requirements.get("is_ambiguous") or requirements.get("is_conflicting"):
        raw_clarification = requirements.get(
            "clarification_message",
            "Could you please clarify your requirements?"
        )
        conversational = _generate_clarification(
            requirements, chat_history, raw_clarification
        )
        print(f"-> Clarification needed: {conversational[:80]}...")
        return {
            "next_step": "awaiting_user",
            "final_response": conversational,
            "logs": ["[ResponseAgent] Clarification required. Returning to user."]
        }

    # ── Case 2: Compatible build assembled ───────────────────────────────
    if build and build.get("is_compatible"):
        print("-> Compatible build ready. Signalling UI to stream.")
        return {
            "next_step": "awaiting_user",
            "final_response": "build_ready",   # UI sentinel — triggers stream_final_response
            "logs": ["[ResponseAgent] Build ready. Streaming to user."]
        }

    # ── Case 3: No build or incompatible — explain conversationally ───────
    reason = no_build_reason or "No compatible configuration could be assembled."
    explanation = _generate_no_build_explanation(requirements, chat_history, reason)
    print(f"-> No build explanation: {explanation[:80]}...")

    return {
        "next_step": "awaiting_user",
        "current_build": None,       # never surface an incompatible build to the UI
        "final_response": explanation,
        "logs": [f"[ResponseAgent] No compatible build. Reason: {reason}"]
    }
