# src/agents/query_agent.py
import os
import json
import re
from dotenv import load_dotenv
from typing import Optional, Tuple, List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.state import AgentState
from src.prompts import SQL_GENERATION_PROMPT, COMPONENT_SELECTION_PROMPT
from src.schema import PCConfiguration
from src.tools import run_sql_query, validate_sql_against_metadata
from src.db_metadata import get_metadata_as_text
from pydantic import BaseModel

from src.logging_utils import attach_log
from src.validation import estimate_prompt_size_chars, need_truncate
from src.errors import LLMError
from src.utils.sanitize import sanitize_json_for_prompt

load_dotenv('.env', override=True)


# ── Pydantic models ───────────────────────────────────────────────────────────
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


# ── Lazy LLM ──────────────────────────────────────────────────────────────────
_llm = None
def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("MODEL_DEPLOYMENT", "gpt-4o-mini"),
            temperature=float(os.getenv("TEMPERATURE", 0.0)),
            api_key=os.getenv("OPENAI_API_KEY")
        )
    return _llm


def _call_llm_direct(messages: List[Dict[str, str]], state: AgentState, max_retries: int = 3) -> str:
    """
    Call the OpenAI chat API directly with a plain list of message dicts.
    Retries on transient errors. Returns the response text string.
    This avoids ALL LangChain chain composition issues.
    """
    import time, random
    model_name = os.getenv("MODEL_DEPLOYMENT", "gpt-4o-mini")
    llm = get_llm()
    last_exc = None
    backoff = 1.0

    for attempt in range(1, max_retries + 1):
        try:
            attach_log(state, "llm_direct", f"Attempt {attempt}", meta={"model": model_name, "msg_count": len(messages)})
            resp = llm.client.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=2000
            )
            text = resp.choices[0].message.content
            attach_log(state, "llm_direct", "Call succeeded", meta={"preview": text[:300]})
            return text
        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            attach_log(state, "llm_direct", f"Attempt {attempt} failed: {str(e)}", level="error")
            transient = any(t in err_str for t in ["rate limit", "429", "timeout", "502", "503", "504"])
            if transient and attempt < max_retries:
                time.sleep(backoff + random.random() * 0.5)
                backoff *= 2
                continue
            break

    raise LLMError("LLM call failed after retries", cause=last_exc)


def _escape_curly_braces(text: str) -> str:
    """Escape braces so a pre-formatted string is safe inside ChatPromptTemplate."""
    return text.replace("{", "{{").replace("}", "}}")


# ── Utilities ─────────────────────────────────────────────────────────────────
def trim_query_results(query_results: dict, max_rows: int = 3) -> dict:
    trimmed = {}
    for component, result in (query_results or {}).items():
        lines = (result or "").splitlines()
        trimmed[component] = "\n".join(lines[:max_rows + 2])
    return trimmed


# ── Compatibility helpers ─────────────────────────────────────────────────────
def _extract_compatibility_constraints(current_build: dict | None) -> dict:
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
        tdp = gpu.get("tdp") or gpu.get("power") or 0
        if tdp:
            try:
                constraints["min_psu_wattage"] = int(tdp) + 150
            except Exception:
                constraints["min_psu_wattage"] = 450
    return constraints


def _inject_constraints_into_prompt(
    requirements: dict,
    current_build: dict | None,
    critique_feedback: list
) -> str:
    constraints = _extract_compatibility_constraints(current_build)
    lines: List[str] = []
    if constraints.get("motherboard_socket"):
        lines.append(f"HARD CONSTRAINT: Motherboard query MUST include WHERE socket = '{constraints['motherboard_socket']}'")
    if constraints.get("ram_memory_type"):
        lines.append(f"HARD CONSTRAINT: RAM query MUST include WHERE memory_type = '{constraints['ram_memory_type']}'")
    if constraints.get("min_psu_wattage"):
        lines.append(f"HARD CONSTRAINT: PSU query MUST include WHERE wattage >= {constraints['min_psu_wattage']}")
    if critique_feedback:
        lines.append(f"CRITIQUE FEEDBACK (fix these specific issues): {', '.join(str(i) for i in critique_feedback)}")
    return "\n".join(lines) if lines else "No prior compatibility constraints."


# ── Sub-step A: Generate + validate + execute SQL ─────────────────────────────
def _generate_and_execute_sql(
    requirements: dict,
    current_build: dict | None,
    critique_feedback: list,
    metadata: str,
    state: AgentState
) -> Tuple[dict, AgentState]:

    attach_log(state, "query_agent", "Starting SQL generation", meta={"requirements": requirements})

    constraint_string = _inject_constraints_into_prompt(requirements, current_build, critique_feedback)
    attach_log(state, "query_agent", "Constraints applied", meta={"constraints": constraint_string})

    prompt_est = estimate_prompt_size_chars(metadata, str(requirements), str(current_build), constraint_string)
    if need_truncate(prompt_est, soft_limit_chars=30000):
        attach_log(state, "query_agent", "Prompt estimated large", level="warning", meta={"chars": prompt_est})

    # Pre-format the SQL prompt — bake all values in so no {variables} remain
    try:
        sql_prompt_text = SQL_GENERATION_PROMPT.format(
            metadata=metadata,
            requirements=str(requirements),
            current_build=str(current_build),
            critique_feedback=str(critique_feedback),
            compatibility_constraints=constraint_string
        )
    except KeyError as e:
        attach_log(state, "query_agent", f"SQL prompt format failed: {e}", level="error")
        return {}, state

    # Call the LLM directly with pre-built messages — no chain composition
    messages = [
        {"role": "system", "content": sql_prompt_text},
        {"role": "user",   "content": "Generate the SQL queries now."}
    ]

    try:
        sql_text = _call_llm_direct(messages, state)
    except LLMError as e:
        attach_log(state, "query_agent", f"SQL generation failed: {str(e)}", level="error")
        return {}, state

    # Parse JSON array of {component, sql}
    try:
        raw = re.sub(r"^```json|^```|```$", "", sql_text.strip(), flags=re.MULTILINE).strip()
        sql_queries: list = json.loads(raw)
    except Exception as e:
        attach_log(state, "query_agent", f"SQL JSON parse failed: {str(e)}", level="error",
                   meta={"raw_preview": sql_text[:500]})
        return {}, state

    query_results: Dict[str, str] = {}

    for item in sql_queries:
        component = item.get("component")
        sql = item.get("sql")
        if not component or not sql:
            attach_log(state, "query_agent", "Skipping malformed SQL item", level="warning", meta={"item": item})
            continue

        # Validate before hitting the DB
        is_valid, err = validate_sql_against_metadata(sql)
        if not is_valid:
            attach_log(state, "query_agent", f"SQL validation failed for [{component}]",
                       level="warning", meta={"error": err, "sql": sql[:400]})
            # Attempt rewrite via LLM
            rewrite_messages = [
                {"role": "system", "content": (
                    "You are a SQL correction assistant for a SQLite PC components database. "
                    "Fix ONLY the identified issue and return the corrected SQL query as a plain string — "
                    "no explanation, no markdown, no JSON wrapping. Just the SQL."
                )},
                {"role": "user", "content": (
                    f"Original query: {sql}\n\n"
                    f"Validation error: {err}\n\n"
                    f"Database metadata:\n{metadata}\n\n"
                    f"Return only the corrected SQL query."
                )}
            ]
            try:
                sql = _call_llm_direct(rewrite_messages, state, max_retries=2)
                sql = sql.strip()
                attach_log(state, "query_agent", f"Rewritten SQL for [{component}]", meta={"sql": sql[:300]})
                is_valid, err = validate_sql_against_metadata(sql)
                if not is_valid:
                    attach_log(state, "query_agent", f"Rewritten SQL still invalid for [{component}]",
                               level="warning", meta={"error": err})
                    continue
            except LLMError as e:
                attach_log(state, "query_agent", f"SQL rewrite failed for [{component}]: {str(e)}", level="error")
                continue

        # Execute validated SQL
        try:
            res = run_sql_query(sql, state=state)
        except Exception as e:
            attach_log(state, "query_agent", f"DB execution error for [{component}]: {str(e)}",
                       level="error", meta={"sql_preview": sql[:200]})
            continue

        if isinstance(res, str) and (
            res.startswith("Error:") or
            res.startswith("Validation error:") or
            res.startswith("Execution Failed:")
        ):
            attach_log(state, "query_agent", f"DB returned error for [{component}]",
                       level="warning", meta={"result": res[:400]})
            continue

        query_results[component] = res
        attach_log(state, "query_agent", f"Collected results for [{component}]",
                   meta={"preview": res[:200]})

    return query_results, state


# ── Sub-step B: Select best components ───────────────────────────────────────
def _select_components(
    query_results: dict,
    requirements: dict,
    current_build: dict | None,
    state: AgentState
) -> Tuple[Optional[dict], Optional[str], AgentState]:

    attach_log(state, "query_agent", "Starting component selection",
               meta={"components": list(query_results.keys())})

    # Sanitize query_results — keeps structure, truncates large values
    sanitized_qr, summary = sanitize_json_for_prompt(
        query_results,
        preserve_top_level_keys=["cpu", "motherboard"]
    )
    attach_log(state, "query_agent", "Sanitized query_results", meta=summary)
    qr_str = json.dumps(sanitized_qr, indent=2)

    # Pre-format the selection prompt — bake all values in
    try:
        selection_prompt_text = COMPONENT_SELECTION_PROMPT.format(
            primary_use=requirements.get("primary_use", "General Use"),
            preferences=", ".join(requirements.get("preferences", [])) or "None",
            query_results=qr_str,
            current_build=json.dumps(current_build, indent=2) if current_build else "No components selected yet."
        )
    except KeyError as e:
        attach_log(state, "query_agent", f"Selection prompt format failed: {e}", level="error")
        return None, f"Selection prompt formatting failed: {e}", state

    # Call LLM directly with pre-built messages — no chain composition
    messages = [
        {"role": "system", "content": selection_prompt_text},
        {"role": "user",   "content": "Return ONLY a JSON object. Do NOT include any prose or explanation."}
    ]

    try:
        text = _call_llm_direct(messages, state)
        attach_log(state, "query_agent", "Selection LLM response received",
                   meta={"preview": text[:400]})
    except LLMError as e:
        attach_log(state, "query_agent", f"Selection LLM call failed: {str(e)}", level="error")
        return None, f"Selection LLM call failed: {str(e)}", state

    # Parse JSON
    text_clean = re.sub(r"^```json|^```|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(text_clean)
    except Exception as e:
        attach_log(state, "query_agent", f"Selection JSON parse failed: {str(e)}", level="error",
                   meta={"raw_preview": text[:1000]})
        return None, "Could not parse component selection JSON.", state

    # Validate with Pydantic
    try:
        selected_model = SelectedBuild.model_validate(parsed)
        selected = selected_model.model_dump()
    except Exception as e:
        attach_log(state, "query_agent", f"SelectedBuild validation failed: {str(e)}", level="warning",
                   meta={"parsed_keys": list(parsed.keys()) if isinstance(parsed, dict) else None})
        selected = parsed  # best-effort fallback

    if not isinstance(selected, dict):
        attach_log(state, "query_agent", "Selection output not a dict", level="error",
                   meta={"type": type(selected).__name__})
        return None, "Selection output malformed.", state

    attach_log(state, "query_agent", "Components selected",
               meta={"selected_keys": [k for k, v in (selected or {}).items() if v]})
    return selected, None, state


# ── Sub-step C: Assemble PCConfiguration ─────────────────────────────────────
def _assemble_build(
    selected: dict,
    current_build: dict | None,
    state: AgentState
) -> Tuple[Optional[dict], Optional[str], AgentState]:

    attach_log(state, "query_agent", "Assembling PCConfiguration")
    existing = current_build or {}

    def to_component_selection(component: dict | None) -> dict | None:
        if not component:
            return None
        spec_parts = []
        if component.get("socket"):
            spec_parts.append(f"Socket: {component['socket']}")
        if component.get("memory_type"):
            spec_parts.append(f"Memory: {component['memory_type']}")
        if component.get("wattage"):
            spec_parts.append(f"Wattage: {component['wattage']}W")
        return {
            "name":           component.get("name", "Unknown"),
            "price":          float(component.get("price", 0.0)),
            "specifications": ", ".join(spec_parts) if spec_parts else "See manufacturer specs"
        }

    converted = {}
    for key, val in (selected or {}).items():
        if isinstance(val, dict):
            converted[key] = to_component_selection(val)
        else:
            converted[key] = val

    merged = {**existing, **{k: v for k, v in converted.items() if v is not None}}

    try:
        updated_build = PCConfiguration(**merged)
    except Exception as e:
        attach_log(state, "query_agent", f"PCConfiguration construction failed: {str(e)}",
                   level="error", meta={"merged_keys": list(merged.keys())})
        return None, f"Could not assemble a valid PC configuration: {str(e)}", state

    components = [
        updated_build.cpu, updated_build.gpu, updated_build.ram,
        updated_build.motherboard, updated_build.storage,
        updated_build.psu, updated_build.case
    ]
    total = sum(c.price for c in components if c is not None)
    updated_build.total_price = total
    attach_log(state, "query_agent", f"Build assembled. Total: ${total:.2f}", meta={"total": total})
    return updated_build.model_dump(), None, state


# ── Main node ─────────────────────────────────────────────────────────────────
def query_database_node(state: AgentState) -> dict:
    attach_log(state, "query_agent", "Node invoked")

    requirements      = state.get("user_requirements", {})
    current_build     = state.get("current_build")
    critique_feedback = state.get("critique_feedback") or []

    metadata = get_metadata_as_text()
    attach_log(state, "query_agent", "Metadata loaded", meta={"preview": metadata[:300]})

    # Step A
    query_results, state = _generate_and_execute_sql(
        requirements, current_build, critique_feedback, metadata, state
    )
    if not query_results:
        attach_log(state, "query_agent", "No valid query results", level="warning")
        return {
            "next_step":       "respond_to_user",
            "no_build_reason": "No valid query results returned from the database.",
            "logs":            state.get("logs", [])
        }

    # Step B
    selected, reason, state = _select_components(
        query_results, requirements, current_build, state
    )
    if selected is None:
        attach_log(state, "query_agent", f"Selection failed: {reason}", level="warning")
        return {
            "next_step":       "respond_to_user",
            "no_build_reason": reason,
            "logs":            state.get("logs", [])
        }

    # Step C
    build, reason, state = _assemble_build(selected, current_build, state)
    if build is None:
        attach_log(state, "query_agent", f"Assembly failed: {reason}", level="warning")
        return {
            "next_step":       "respond_to_user",
            "no_build_reason": reason,
            "logs":            state.get("logs", [])
        }

    attach_log(state, "query_agent", "Completed — routing to critique",
               meta={"total_price": build.get("total_price", 0)})
    return {
        "current_build":   build,
        "next_step":       "self_critique",
        "no_build_reason": None,
        "logs":            state.get("logs", [])
    }
