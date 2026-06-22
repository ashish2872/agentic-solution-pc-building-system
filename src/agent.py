import os
import operator
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated, Optional
from langchain_core.prompts import ChatPromptTemplate
from src.prompts import (
    REQUIREMENT_GATHERING_PROMPT,
    SQL_GENERATION_PROMPT,
    COMPONENT_SELECTION_PROMPT,
    SELF_CRITIQUE_PROMPT
)

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from src.schema import PCConfiguration, UserRequirements
from src.tools import get_database_schema, run_sql_query
from src.db_metadata import get_metadata_as_text


load_dotenv('.env')
# Assuming you use an LLM wrapper like langchain_openai or langchain_anthropic


# In src/agent.py, after imports
from src.db_metadata import get_metadata_as_text
DB_METADATA_TEXT = get_metadata_as_text()  # built once at startup


LLM_MODEL = os.getenv('MODEL_DEPLOYMENT', 'gpt-4o-mini')
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.0))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


class AgentState(TypedDict):
    # Requirements gathered from the user (budget, usage, preferences)
    user_requirements: dict
    
    # Conversational history for taking continuous feedback
    chat_history: List[dict]
    
    # The current running build draft — stored as dict for LangGraph serialization
    # Convert to PCConfiguration inside nodes using PCConfiguration(**state["current_build"])
    current_build: Optional[dict]
    
    # Advanced reasoning track: logs thoughts or errors for self-critique/reflection
    logs: Annotated[List[str], operator.add]
    
    # Internal flag to decide graph navigation (e.g., 'gather', 'query', 'critique', 'end')
    next_step: str

    # Tracks how many times self_critique has looped back — prevents infinite loops
    critique_iterations: int

    final_response: Optional[str]  # The final message to the user, either clarification or build summary




# Initialize externalized configuration (Step 3/Expectations)
llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE, api_key = OPENAI_API_KEY)

# --- NODES ---

# src/agent.py — add after pc_config_agent = workflow.compile()

from langchain_core.messages import SystemMessage, HumanMessage

STREAMING_RESPONSE_PROMPT = """
You are a friendly expert PC builder assistant.
Given the final PC build configuration and the user's requirements, 
present the build in a clear, conversational way.
Explain WHY each component was chosen for their specific use case.
If there are compatibility notes or issues, explain them clearly.
Format the response in a readable way with sections for each component.
"""

def stream_final_response(build: dict, requirements: dict):
    """
    Streams the final build explanation token by token.
    Use this as a generator — iterate over it to get chunks.
    
    Usage:
        for chunk in stream_final_response(build, requirements):
            print(chunk, end="", flush=True)
    """
    import json

    streaming_llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.3,
        streaming=True
    )

    messages = [
        SystemMessage(content=STREAMING_RESPONSE_PROMPT),
        HumanMessage(content=f"""
User Requirements: {json.dumps(requirements, indent=2)}

Final PC Build:
{json.dumps(build, indent=2)}

Present this build to the user now.
        """)
    ]

    for chunk in streaming_llm.stream(messages):
        if chunk.content:
            yield chunk.content






structured_parser_llm = llm.with_structured_output(UserRequirements)




def requirement_gathering_node(state: AgentState) -> dict:
    print("\n--- [Node: Gathering & Validating Requirements] ---")
    
    # Extract the most recent message from the user
    if not state.get("chat_history"):
        return {
            "next_step": "respond_to_user",
            "logs": ["Error: Chat history is empty. No user input found."]
        }
    
    latest_user_message = state["chat_history"][-1]["content"]
    
    # Format the conversational context for the LLM
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", REQUIREMENT_GATHERING_PROMPT),
        ("user", "User Message: {user_input}")
    ])
    
    # Invoke the model to extract parameters
    try:
        parsed_requirements: UserRequirements = structured_parser_llm.invoke(
            prompt_template.format(user_input=latest_user_message)
        )
    except Exception as e:
        # Graceful API failure handling (Requirement 4)
        error_msg = f"LLM parsing failed: {str(e)}"
        return {
            "next_step": "respond_to_user",
            "logs": [error_msg],
            "user_requirements": {"error": True}
        }

    # Evaluate dynamic branching based on validation results
    if parsed_requirements.is_ambiguous or parsed_requirements.is_conflicting:
        print(f"-> Validation Alert: Ambiguous={parsed_requirements.is_ambiguous}, Conflicting={parsed_requirements.is_conflicting}")
        log_entry = f"Requirements invalid. Reason: Input is either ambiguous or conflicting. Asking for clarity."
        
        # Inject the clarification message directly into the log sequence
        return {
            "user_requirements": parsed_requirements.model_dump(),
            "next_step": "respond_to_user", # Bypass DB querying, go straight to response node
            "logs": [log_entry]
        }
        
    print("-> Requirements successfully validated.")
    log_entry = f"Extracted Requirements -> Budget: {parsed_requirements.budget}, Use: {parsed_requirements.primary_use}, Prefs: {parsed_requirements.preferences}"
    
    return {
        "user_requirements": parsed_requirements.model_dump(),
        "next_step": "query_database", # Criteria passed, clear to search components
        "logs": [log_entry]
    }



def query_database_node(state: AgentState) -> dict:
    """Generates SQL queries via LLM, executes them, then selects best components."""
    print("\n--- [Node: Querying Components Database] ---")

    requirements = state.get("user_requirements", {})
    current_build = state.get("current_build")
    logs = []

    # ── Step 1: Fetch schema ──────────────────────────────────────────────
    metadata = get_metadata_as_text()
    logs.append("Metadata retrieved successfully.")

    sql_gen_prompt = ChatPromptTemplate.from_messages([
        ("system", SQL_GENERATION_PROMPT),
        ("user", "Generate the SQL queries now.")
    ])

    sql_gen_chain = sql_gen_prompt | llm

    try:
        sql_response = sql_gen_chain.invoke({
            "metadata": metadata,
            "requirements": str(requirements),
            "current_build": str(current_build)
        })

    except Exception as e:
        return {
            "next_step": "respond_to_user",
            "logs": [f"LLM SQL generation failed: {str(e)}"]
        }

    # ── Step 3: Parse the JSON array of {component, sql} objects ─────────
    import json, re

    try:
        # Strip markdown code fences if LLM wrapped output in ```json ... ```
        raw = sql_response.content.strip()
        raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
        sql_queries: list = json.loads(raw)
    except json.JSONDecodeError as e:
        return {
            "next_step": "respond_to_user",
            "logs": [f"Failed to parse LLM SQL output as JSON: {str(e)}", f"Raw output: {sql_response.content}"]
        }

    # ── Step 4: Execute each query and collect results ────────────────────
    query_results = {}

    for item in sql_queries:
        component = item.get("component")
        sql = item.get("sql")

        if not component or not sql:
            logs.append(f"Skipping malformed query item: {item}")
            continue

        logs.append(f"Executing SQL for [{component}]: {sql}")
        result = run_sql_query(sql)
        query_results[component] = result
        logs.append(f"Result for [{component}]: {result[:200]}")  # truncate for log readability

    if not query_results:
        return {
            "next_step": "respond_to_user",
            "logs": logs + ["No valid query results returned. Cannot assemble build."]
        }

    # ── Step 5: LLM selects best component from each result set ──────────
    selection_prompt = ChatPromptTemplate.from_messages([
        ("system", COMPONENT_SELECTION_PROMPT),
        ("user", "Select the best components now.")
    ])

    selection_chain = selection_prompt | llm

    try:
        selection_response = selection_chain.invoke({
            "primary_use": requirements.get("primary_use", "General Use"),
            "preferences": requirements.get("preferences", []),
            "query_results": str(query_results)
        })
    except Exception as e:
        return {
            "next_step": "respond_to_user",
            "logs": logs + [f"LLM component selection failed: {str(e)}"]
        }

    # ── Step 6: Parse selection into PCConfiguration ──────────────────────
    try:
        raw = selection_response.content.strip()
        raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
        selected: dict = json.loads(raw)
    except json.JSONDecodeError as e:
        return {
            "next_step": "respond_to_user",
            "logs": logs + [f"Failed to parse component selection JSON: {str(e)}"]
        }

    # Merge with existing build (preserves already-selected components on feedback loops)
    existing = current_build or {}
    merged = {**existing, **{k: v for k, v in selected.items() if v is not None}}

    try:
        updated_build = PCConfiguration(**merged)
    except Exception as e:
        return {
            "next_step": "respond_to_user",
            "logs": logs + [f"Failed to construct PCConfiguration: {str(e)}"]
        }

    # ── Step 7: Calculate total price ─────────────────────────────────────
    components = [updated_build.cpu, updated_build.gpu, updated_build.ram,
                  updated_build.motherboard, updated_build.storage,
                  updated_build.psu, updated_build.case]

    total = sum(c.price for c in components if c is not None)
    updated_build.total_price = total
    logs.append(f"Build assembled. Total price: ${total:.2f}")

    return {
        "current_build": updated_build.model_dump(),
        "logs": logs,
        "next_step": "self_critique"
    }



def respond_to_user_node(state: AgentState) -> dict:
    """Formats structured output and prepares final response."""
    print("\n--- [Node: Formatting Final Response] ---")

    requirements = state.get("user_requirements", {})
    build = state.get("current_build")

    # Ambiguous or conflicting — return clarification message
    if requirements.get("is_ambiguous") or requirements.get("is_conflicting"):
        clarification = requirements.get(
            "clarification_message",
            "Could you please clarify your requirements?"
        )
        return {
            "next_step": "awaiting_user",
            "logs": ["Clarification required. Returning message to user."],
            "final_response": clarification
        }

    # Normal case — store build summary in state for UI to stream
    if build:
        return {
            "next_step": "end",
            "logs": ["Build complete. Ready to stream response."],
            "final_response": "build_ready"  # signal to UI to trigger streaming
        }

    return {
        "next_step": "end",
        "logs": ["No build found."],
        "final_response": "Sorry, I could not assemble a build with the given requirements."
    }


# --- ROUTING CONDITION ---

def route_next_node(state: AgentState) -> str:
    """Evaluates state dynamic flag to determine the graph execution edge."""
    if state["next_step"] == "query_database":
        return "query_database"
    elif state["next_step"] == "self_critique":
        return "self_critique"
    elif state["next_step"] == "respond_to_user":
        return "respond_to_user"
    else:
        return END

# --- COMPILING THE GRAPH ---

def self_critique_node(state: AgentState) -> dict:
    """Runs LLM compatibility check on the assembled build."""
    print("\n--- [Node: Self-Critique & Reflection] ---")

    import json, re

    build = state.get("current_build")
    iterations = state.get("critique_iterations", 0)
    requirements = state.get("user_requirements", {})
    logs = []

    # Hard exit: prevent infinite loop after 3 failed critique rounds
    if iterations >= 3:
        # Mark best available build as final even if not perfect
        if build:
            build["is_compatible"] = True
            build["compatibility_notes"] = "Max critique iterations reached. Build may have minor issues."
        return {
            "current_build": build,
            "next_step": "respond_to_user",
            "logs": ["Max critique iterations (3) reached. Proceeding with best available build."]
        }

    if not build:
        return {
            "next_step": "respond_to_user",
            "logs": ["No build found to critique."]
        }

    # ── LLM Compatibility Check ───────────────────────────────────────────
    critique_prompt = ChatPromptTemplate.from_messages([
        ("system", SELF_CRITIQUE_PROMPT),
        ("user", "Review this build now and return your JSON verdict.")
    ])

    critique_chain = critique_prompt | llm

    try:
        critique_response = critique_chain.invoke({
            "build": json.dumps(build, indent=2),
            "budget": requirements.get("budget", "Not specified")
        })
    except Exception as e:
        return {
            "next_step": "respond_to_user",
            "logs": [f"LLM critique call failed: {str(e)}"]
        }

    # ── Parse critique response ───────────────────────────────────────────
    try:
        raw = critique_response.content.strip()
        raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
        critique: dict = json.loads(raw)
    except json.JSONDecodeError as e:
        return {
            "next_step": "respond_to_user",
            "logs": [f"Failed to parse critique JSON: {str(e)}", f"Raw: {critique_response.content}"]
        }

    is_compatible = critique.get("is_compatible", False)
    notes = critique.get("compatibility_notes", "")
    issues = critique.get("issues_found", [])
    needs_requery = critique.get("needs_requery", False)

    logs.append(f"Critique result — Compatible: {is_compatible}, Needs requery: {needs_requery}")
    logs.append(f"Issues found: {issues}")
    logs.append(f"Notes: {notes}")

    print(f"-> Compatible: {is_compatible} | Issues: {issues}")

    # ── Update build with compatibility verdict ───────────────────────────
    build["is_compatible"] = is_compatible
    build["compatibility_notes"] = notes

    # ── Route based on critique verdict ──────────────────────────────────
    if needs_requery:
        print("-> Hard incompatibility found. Routing back to re-query.")
        # Store issues in logs so query_database_node can use them
        return {
            "current_build": build,
            "critique_iterations": iterations + 1,
            "next_step": "query_database",
            "logs": logs + [f"Routing back due to: {issues}"]
        }

    return {
        "current_build": build,
        "critique_iterations": iterations + 1,
        "next_step": "respond_to_user",
        "logs": logs
    }

workflow = StateGraph(AgentState)

# Add our modular steps
workflow.add_node("gather_requirements", requirement_gathering_node)
workflow.add_node("query_database", query_database_node)
workflow.add_node("self_critique", self_critique_node)
workflow.add_node("respond_to_user", respond_to_user_node)

# Set up orchestration flow
workflow.set_entry_point("gather_requirements")

# Define conditional transitions (Implementing the loop)
workflow.add_conditional_edges(
    "gather_requirements", route_next_node, {
        "query_database": "query_database",
        "self_critique": "self_critique",
        "respond_to_user": "respond_to_user",
        END: END
    }
)


workflow.add_conditional_edges(
    "query_database", route_next_node, {
        "query_database": "query_database",
        "self_critique": "self_critique",
        "respond_to_user": "respond_to_user",
        END: END
    }
)

workflow.add_conditional_edges(
    "self_critique", route_next_node, {
        "query_database": "query_database", 
        "respond_to_user": "respond_to_user",
        END: END
    }
)

workflow.add_conditional_edges(
    "respond_to_user", route_next_node, {
        "awaiting_user": END,
        END: END
    }
)


# Compile state machine
pc_config_agent = workflow.compile()

# Default initial state — use this as the base when invoking the agent in main.py
DEFAULT_INITIAL_STATE = {
    "user_requirements": {},
    "chat_history": [],
    "current_build": None,
    "logs": [],
    "next_step": "gather",
    "critique_iterations": 0,
    "final_response": None
}
