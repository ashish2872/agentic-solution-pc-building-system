# src/utils/prompt_utils.py
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List, Any
from src.logging_utils import attach_log

def render_prompt_to_message_dicts(prompt_template: ChatPromptTemplate, inputs: Dict[str, Any], state: Any) -> List[Dict[str, str]]:
    """
    Render a ChatPromptTemplate into a list of plain message dicts suitable for LLM APIs:
    [{'role': 'system'|'user', 'content': '...'}, ...]
    Ensures inputs are primitive (strings/lists/dicts) before formatting.
    """
    # Sanitize input values: stringify any non-primitive
    safe_inputs = {}
    for k, v in (inputs or {}).items():
        if v is None or isinstance(v, (str, int, float, bool, list, dict)):
            safe_inputs[k] = v
        else:
            safe_inputs[k] = str(v)
            attach_log(state, "prompt_utils", f"Sanitized non-primitive input for '{k}'", meta={"type": type(v).__name__})
    # Render prompt value and convert to messages
    pv = prompt_template.format_prompt(**safe_inputs)
    messages = pv.to_messages()  # list of Message objects
    msg_list = []
    for m in messages:
        role = "system" if getattr(m, "type", "system") == "system" else "user"
        msg_list.append({"role": role, "content": m.content})
    return msg_list
