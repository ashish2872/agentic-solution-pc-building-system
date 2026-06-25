# src/agents/query_agent.py
import os
import json
import re
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.state import AgentState
from src.prompts import SQL_GENERATION_PROMPT, COMPONENT_SELECTION_PROMPT
from src.schema import PCConfiguration
from src.tools import run_sql_query, validate_sql_against_metadata
from src.db_metadata import get_metadata_as_text

load_dotenv('.env', override=True)

# ── Lazy LLM ─────────────────────────────────────────────────────────────────
_llm = None

# src/agents/query_agent.py — after imports
from pydantic import BaseModel
from typing import Optional

class SelectedComponent(BaseModel):
    name: str
    price: float
    socket: Optional[str] = None
    memory_type: Optional[str] = None
    wattage: Optional[int] = None

class SelectedBuild(BaseModel):
    cpu: Optional[SelectedComponent] = None
    gpu: Optional[SelectedComponent] = None
    motherboard: Optional[SelectedComponent] = None
    ram: Optional[SelectedComponent] = None
    storage: Optional[SelectedComponent] = None
    psu: Optional[SelectedComponent] = None
    case: Optional[SelectedComponent] = None


















def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv('MODEL_DEPLOYMENT', 'gpt-4o-mini'),
            temperature=float(os.getenv('TEMPERATURE', 0.0)),
            api_key=os.getenv('OPENAI_API_KEY')
        )
    return _llm


# src/agents/query_agent.py

def _extract_compatibility_constraints(current_build: dict | None) -> dict:
    """
    Reads already-selected components from current_build and returns
    hard compatibility constraints for remaining queries.
    """
    constraints = {}
    if not current_build:
        return constraints

    cpu = current_build.get("cpu")
    if cpu and isinstance(cpu, dict):
        socket = cpu.get("socket")
        if socket:
            constraints["motherboard_socket"] = socket

    motherboard = current_build.get("motherboard")
    if motherboard and isinstance(motherboard, dict):
        mem_type = motherboard.get("memory_type") or motherboard.get("supported_memory")
        if mem_type:
            constraints["ram_memory_type"] = mem_type
        socket = motherboard.get("socket")
        if socket and "motherboard_socket" not in constraints:
            constraints["motherboard_socket"] = socket

    gpu = current_build.get("gpu")
    if gpu and isinstance(gpu, dict):
        tdp = gpu.get("tdp") or gpu.get("power", 0)
        if tdp:
            constraints["min_psu_wattage"] = int(tdp) + 150  # GPU TDP + overhead

    return constraints


def _inject_constraints_into_prompt(
    requirements: dict,
    current_build: dict | None,
    critique_feedback: list
) -> str:
    """
    Builds a constraint string to inject into the SQL generation prompt
    so the LLM knows exactly what WHERE clauses to add.
    """
    constraints = _extract_compatibility_constraints(current_build)
    lines = []

    if constraints.get("motherboard_socket"):
        lines.append(
            f"HARD CONSTRAINT: Motherboard query MUST include "
            f"WHERE socket = '{constraints['motherboard_socket']}'"
        )
    if constraints.get("ram_memory_type"):
        lines.append(
            f"HARD CONSTRAINT: RAM query MUST include "
            f"WHERE memory_type = '{constraints['ram_memory_type']}'"
        )
    if constraints.get("min_psu_wattage"):
        lines.append(
            f"HARD CONSTRAINT: PSU query MUST include "
            f"WHERE wattage >= {constraints['min_psu_wattage']}"
        )
    if critique_feedback:
        lines.append(
            f"CRITIQUE FEEDBACK (fix these specific issues): {', '.join(critique_feedback)}"
        )

    return "\n".join(lines) if lines else "No prior compatibility constraints."


# ── Utilities ─────────────────────────────────────────────────────────────────
def trim_query_results(query_results: dict, max_rows: int = 3) -> dict:
    """Keep only header + first max_rows data rows per component."""
    trimmed = {}
    for component, result in query_results.items():
        lines = result.strip().split("\n")
        trimmed[component] = "\n".join(lines[:max_rows + 2])
    return trimmed


# ── Sub-step A: Generate + validate + execute SQL ─────────────────────────────
def _generate_and_execute_sql(
    requirements: dict,
    current_build: dict | None,
    critique_feedback: list,
    metadata: str,
    logs: list
) -> tuple[dict, list]:
    """
    Returns (query_results dict, updated logs).
    query_results is empty if generation or execution fully failed.
    """

    # ── Build explicit compatibility constraints from current build ────────
    constraint_string = _inject_constraints_into_prompt(
        requirements, current_build, critique_feedback
    )
    logs.append(f"[QueryAgent] Compatibility constraints applied: {constraint_string}")

    # ── SQL generation prompt ─────────────────────────────────────────────
    sql_gen_prompt = ChatPromptTemplate.from_messages([
        ("system", SQL_GENERATION_PROMPT),
        ("user", "Generate the SQL queries now.")
    ])

    try:
        sql_response = (sql_gen_prompt | get_llm()).invoke({
            "metadata": metadata,
            "requirements": str(requirements),
            "current_build": str(current_build),
            "critique_feedback": str(critique_feedback),
            "compatibility_constraints": constraint_string   # ← NEW
        })
    except Exception as e:
        logs.append(f"[QueryAgent] SQL generation LLM call failed: {str(e)}")
        return {}, logs

    # ── Parse JSON array of {component, sql} ─────────────────────────────
    try:
        raw = re.sub(
            r"^```json|^```|```$", "",
            sql_response.content.strip(),
            flags=re.MULTILINE
        ).strip()
        sql_queries: list = json.loads(raw)
    except json.JSONDecodeError as e:
        logs.append(
            f"[QueryAgent] SQL JSON parse failed: {str(e)} "
            f"| Raw: {sql_response.content[:300]}"
        )
        return {}, logs

    # ── SQL rewrite prompt — used per component if validation fails ───────
    sql_rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a SQL correction assistant for a SQLite PC components database.\n"
            "Fix ONLY the identified issue and return the corrected SQL query as a plain "
            "string — no explanation, no markdown, no JSON wrapping. Just the SQL."
        )),
        ("user", (
            "Original query: {sql}\n\n"
            "Validation error: {error}\n\n"
            "Database metadata: {metadata}\n\n"
            "Return only the corrected SQL query."
        ))
    ])
    sql_rewrite_chain = sql_rewrite_prompt | get_llm()

    query_results = {}

    # ── Execute each query with validate → rewrite → execute loop ────────
    for item in sql_queries:
        component = item.get("component")
        sql = item.get("sql")

        if not component or not sql:
            logs.append(f"[QueryAgent] Skipping malformed item: {item}")
            continue

        # Validate before hitting the DB
        is_valid, error_msg = validate_sql_against_metadata(sql)

        if not is_valid:
            logs.append(
                f"[QueryAgent] [{component}] Validation failed: {error_msg}. Rewriting."
            )
            try:
                rewrite_response = sql_rewrite_chain.invoke({
                    "sql": sql,
                    "error": error_msg,
                    "metadata": metadata
                })
                sql = rewrite_response.content.strip()
                logs.append(f"[QueryAgent] [{component}] Rewritten SQL: {sql}")

                # Second validation on rewritten query
                is_valid, error_msg = validate_sql_against_metadata(sql)
                if not is_valid:
                    logs.append(
                        f"[QueryAgent] [{component}] Rewrite still invalid: "
                        f"{error_msg}. Skipping."
                    )
                    continue

            except Exception as e:
                logs.append(
                    f"[QueryAgent] [{component}] Rewrite LLM failed: {str(e)}. Skipping."
                )
                continue

        # Execute validated SQL
        logs.append(f"[QueryAgent] Executing SQL for [{component}]: {sql}")
        result = run_sql_query(sql)
        query_results[component] = result
        logs.append(f"[QueryAgent] Result for [{component}]: {result[:200]}")

    return query_results, logs


# ── Sub-step B: Select best components from query results ─────────────────────
def _select_components(
    query_results: dict,
    requirements: dict,
    current_build: dict | None,
    logs: list
) -> tuple[dict | None, str | None, list]:

    trimmed = trim_query_results(query_results)

    selection_prompt = ChatPromptTemplate.from_messages([
        ("system", COMPONENT_SELECTION_PROMPT),
        ("user", "Select the best components now.")
    ])

    # Use renamed class — no conflict with schema.py's ComponentSelection
    structured_llm = get_llm().with_structured_output(SelectedBuild)

    try:
        selection_response = (selection_prompt | structured_llm).invoke({
            "primary_use": requirements.get("primary_use", "General Use"),
            "preferences": requirements.get("preferences", []),
            "query_results": str(trimmed),
            "current_build": str(current_build) if current_build else "No components selected yet."
        })
    except Exception as e:
        logs.append(f"[QueryAgent] Component selection LLM failed: {str(e)}")
        return None, f"Component selection failed: {str(e)}", logs

    if selection_response is None:
        logs.append("[QueryAgent] Selection returned None.")
        return None, "Component selection returned no result.", logs

    selected = selection_response.model_dump()
    logs.append(
        f"[QueryAgent] Components selected: "
        f"{[k for k, v in selected.items() if v is not None]}"
    )

    return selected, None, logs


# ── Sub-step C: Assemble PCConfiguration ─────────────────────────────────────
def _assemble_build(
    selected: dict,
    current_build: dict | None,
    logs: list
) -> tuple[dict | None, str | None, list]:

    existing = current_build or {}

    # Convert SelectedComponent dicts → ComponentSelection-compatible dicts
    def to_component_selection(component: dict | None) -> dict | None:
        if not component:
            return None

        # Build specifications string from compatibility fields
        spec_parts = []
        if component.get("socket"):
            spec_parts.append(f"Socket: {component['socket']}")
        if component.get("memory_type"):
            spec_parts.append(f"Memory: {component['memory_type']}")
        if component.get("wattage"):
            spec_parts.append(f"Wattage: {component['wattage']}W")

        return {
            "name": component.get("name", "Unknown"),
            "price": component.get("price", 0.0),
            "specifications": ", ".join(spec_parts) if spec_parts else "See manufacturer specs"
        }

    # Apply conversion to all components from selected
    converted = {}
    for key, val in selected.items():
        if isinstance(val, dict):
            converted[key] = to_component_selection(val)
        else:
            converted[key] = val

    # Merge converted new components over existing build
    merged = {
        **existing,
        **{k: v for k, v in converted.items() if v is not None}
    }

    try:
        updated_build = PCConfiguration(**merged)
    except Exception as e:
        logs.append(f"[QueryAgent] PCConfiguration construction failed: {str(e)}")
        return None, f"Could not assemble a valid PC configuration: {str(e)}", logs

    components = [
        updated_build.cpu, updated_build.gpu, updated_build.ram,
        updated_build.motherboard, updated_build.storage,
        updated_build.psu, updated_build.case
    ]
    total = sum(c.price for c in components if c is not None)
    updated_build.total_price = total
    logs.append(f"[QueryAgent] Build assembled. Total: ${total:.2f}")

    return updated_build.model_dump(), None, logs


# ── Main node ─────────────────────────────────────────────────────────────────
def query_database_node(state: AgentState) -> dict:
    """
    Query Agent.
    Responsibility: SQL generation, validation, execution, component selection,
                    and PCConfiguration assembly.
    Knows nothing about conversation or compatibility checking.
    """
    print("\n--- [Query Agent] ---")

    requirements = state.get("user_requirements", {})
    current_build = state.get("current_build")
    critique_feedback = state.get("critique_feedback") or []
    logs = []

    metadata = get_metadata_as_text()
    logs.append("[QueryAgent] Metadata loaded.")

    # Step A — SQL
    query_results, logs = _generate_and_execute_sql(
        requirements, current_build, critique_feedback, metadata, logs
    )

    if not query_results:
        return {
            "next_step": "respond_to_user",
            "no_build_reason": "No valid query results returned from the database.",
            "logs": logs
        }

    # Step B — Selection
    selected, reason, logs = _select_components(
        query_results, requirements, current_build, logs
    )

    if selected is None:
        return {
            "next_step": "respond_to_user",
            "no_build_reason": reason,
            "logs": logs
        }

    # Step C — Assembly
    build, reason, logs = _assemble_build(selected, current_build, logs)

    if build is None:
        return {
            "next_step": "respond_to_user",
            "no_build_reason": reason,
            "logs": logs
        }

    return {
        "current_build": build,
        "next_step": "self_critique",
        "no_build_reason": None,
        "logs": logs
    }
