# src/agents/critique_agent.py
import os
import json
import re
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.state import AgentState
from src.prompts import SELF_CRITIQUE_PROMPT

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


# ── Sub-step A: Call LLM for compatibility verdict ────────────────────────────
def _run_critique(build: dict, budget: str) -> tuple[dict | None, str | None]:
    """
    Calls the critique LLM and returns (parsed_critique, error_message).
    parsed_critique is None if the call or parse failed.
    """
    critique_prompt = ChatPromptTemplate.from_messages([
        ("system", SELF_CRITIQUE_PROMPT),
        ("user", "Review this build now and return your JSON verdict.")
    ])

    try:
        response = (critique_prompt | get_llm()).invoke({
            "build": json.dumps(build, indent=2),
            "budget": budget
        })
    except Exception as e:
        return None, f"Critique LLM call failed: {str(e)}"

    try:
        raw = re.sub(r"^```json|^```|```$", "", response.content.strip(), flags=re.MULTILINE).strip()
        critique: dict = json.loads(raw)
        return critique, None
    except json.JSONDecodeError as e:
        return None, f"Critique JSON parse failed: {str(e)} | Raw: {response.content[:300]}"


# ── Sub-step B: Decide routing from critique verdict ─────────────────────────
def _route_from_verdict(
    build: dict,
    critique: dict,
    iterations: int,
    logs: list
) -> dict:
    """
    Interprets the critique verdict and returns the state update dict.
    All routing decisions live here — not scattered across conditionals.
    """
    is_compatible = critique.get("is_compatible", False)
    notes = critique.get("compatibility_notes", "")
    issues = critique.get("issues_found", [])
    needs_requery = critique.get("needs_requery", False)

    logs.append(f"[CritiqueAgent] Compatible: {is_compatible} | Needs requery: {needs_requery}")
    logs.append(f"[CritiqueAgent] Issues: {issues}")
    logs.append(f"[CritiqueAgent] Notes: {notes}")

    print(f"-> Compatible: {is_compatible} | Issues: {issues}")

    # Stamp verdict onto build
    build["is_compatible"] = is_compatible
    build["compatibility_notes"] = notes

    # ── Case 1: Hard incompatibility — send back to Query Agent ──────────
    if needs_requery:
        print("-> Hard incompatibility. Routing back to Query Agent.")
        return {
            "current_build": build,
            "critique_iterations": iterations + 1,
            "critique_feedback": issues,       # Query Agent reads this to fix specifically
            "no_build_reason": None,
            "next_step": "query_database",
            "logs": logs + [f"[CritiqueAgent] Re-querying due to: {issues}"]
        }

    # ── Case 2: Compatible — proceed to response ──────────────────────────
    if is_compatible:
        return {
            "current_build": build,
            "critique_iterations": iterations + 1,
            "critique_feedback": None,
            "no_build_reason": None,
            "next_step": "respond_to_user",
            "logs": logs
        }

    # ── Case 3: Issues found but LLM didn't flag needs_requery ───────────
    # Soft incompatibility — surface to user conversationally, don't serve broken build
    return {
        "current_build": None,
        "critique_iterations": iterations + 1,
        "critique_feedback": None,
        "no_build_reason": f"{notes} — Issues: {', '.join(issues)}",
        "next_step": "respond_to_user",
        "logs": logs + ["[CritiqueAgent] Soft incompatibility. Surfacing to user."]
    }


# ── Main node ─────────────────────────────────────────────────────────────────
def self_critique_node(state: AgentState) -> dict:
    """
    Critique Agent.
    Responsibility: verify compatibility of the assembled build.
    Knows nothing about SQL, conversation, or response formatting.
    Outputs either a re-query instruction (with specific issues) or a pass/fail verdict.
    """
    print("\n--- [Critique Agent] ---")

    build = state.get("current_build")
    iterations = state.get("critique_iterations", 0)
    requirements = state.get("user_requirements", {})
    budget = str(requirements.get("budget", "Not specified"))
    logs = []

    # ── Guard: no build to critique ───────────────────────────────────────
    if not build:
        return {
            "next_step": "respond_to_user",
            "no_build_reason": "No build was assembled to critique.",
            "logs": ["[CritiqueAgent] No build in state. Skipping critique."]
        }

    # ── Guard: max iterations hit ─────────────────────────────────────────
    if iterations >= 3:
        print("-> Max critique iterations reached. Proceeding with best available build.")
        build["is_compatible"] = True
        build["compatibility_notes"] = (
            "Max critique iterations reached. Build may have minor compatibility issues."
        )
        return {
            "current_build": build,
            "critique_iterations": iterations,
            "critique_feedback": None,
            "no_build_reason": None,
            "next_step": "respond_to_user",
            "logs": ["[CritiqueAgent] Max iterations (3) reached. Passing build through."]
        }

    # ── Run critique ──────────────────────────────────────────────────────
    critique, error = _run_critique(build, budget)

    if critique is None:
        # LLM or parse failure — don't block the user, pass build through with a note
        build["is_compatible"] = True
        build["compatibility_notes"] = "Compatibility check could not be completed."
        return {
            "current_build": build,
            "critique_iterations": iterations + 1,
            "critique_feedback": None,
            "no_build_reason": None,
            "next_step": "respond_to_user",
            "logs": [f"[CritiqueAgent] Critique failed: {error}. Passing build through."]
        }

    # ── Route based on verdict ────────────────────────────────────────────
    return _route_from_verdict(build, critique, iterations, logs)
