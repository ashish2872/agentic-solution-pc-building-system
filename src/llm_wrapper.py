# src/llm_wrapper.py
import os
import time
import random
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.errors import LLMError
from src.logging_utils import attach_log

# Configure defaults
DEFAULT_TIMEOUT = 20
DEFAULT_MAX_RETRIES = 3
RETRYABLE_SUBSTRINGS = ["rate limit", "429", "timeout", "502", "503", "504", "connection aborted", "RateLimitError"]

_llm_singleton: Optional[ChatOpenAI] = None



# near top of call_llm()
def _sanitize_call_args(prompt_chain, input_vars, state):
    # ensure prompt_chain is not an AIMessage
    from langchain_core.messages import AIMessage, HumanMessage
    if isinstance(prompt_chain, (AIMessage,)):
        # Convert to string prompt if possible
        attach_log(state, "llm_wrapper", "Sanitized prompt_chain AIMessage -> using .content", level="warning")
        prompt_chain = str(prompt_chain.content)
    # sanitize input vars
    sanitized = {}
    for k, v in (input_vars or {}).items():
        if v is None or isinstance(v, (str, int, float, bool, list, dict)):
            sanitized[k] = v
        elif hasattr(v, "content"):
            sanitized[k] = v.content
            attach_log(state, "llm_wrapper", f"Sanitized Message input for '{k}' to content", level="info")
        elif isinstance(v, list) and all(hasattr(x, "content") for x in v):
            sanitized[k] = [x.content for x in v]
            attach_log(state, "llm_wrapper", f"Sanitized list[Message] for '{k}'", level="info")
        else:
            sanitized[k] = str(v)
            attach_log(state, "llm_wrapper", f"Stringified unexpected input for '{k}'", level="warning", meta={"type": type(v).__name__})
    return prompt_chain, sanitized




def get_llm_instance() -> ChatOpenAI:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = ChatOpenAI(
            model=os.getenv("MODEL_DEPLOYMENT", "gpt-4o-mini"),
            temperature=float(os.getenv("TEMPERATURE", 0.0)),
            api_key=os.getenv("OPENAI_API_KEY")
        )
    return _llm_singleton

def _is_transient_error(err_str: str) -> bool:
    if not err_str:
        return False
    s = err_str.lower()
    return any(tok in s for tok in RETRYABLE_SUBSTRINGS)

def call_llm(prompt_chain: ChatPromptTemplate, input_vars: Dict[str, Any], state: Optional[Dict[str, Any]] = None, timeout_seconds: int = DEFAULT_TIMEOUT, max_retries: int = DEFAULT_MAX_RETRIES) -> Any:
    """
    Call an LLM via a prompt chain with retries and exponential backoff.
    Returns the raw chain response on success or raises LLMError on failure.
    """
    llm = get_llm_instance()
    attempt = 0
    backoff = 1.0
    last_exc: Optional[Exception] = None

    while attempt < max_retries:
        attempt += 1
        try:
            if state is not None:
                attach_log(state, "llm_wrapper", f"LLM call attempt {attempt}", meta={"vars": list(input_vars.keys())})
            response = (prompt_chain | llm).invoke(input_vars)  # some wrappers accept timeout; adjust if your client supports it
            if state is not None:
                attach_log(state, "llm_wrapper", "LLM call succeeded", meta={"attempt": attempt})
            return response
        except Exception as e:
            last_exc = e
            err_str = str(e)
            if state is not None:
                attach_log(state, "llm_wrapper", f"LLM call failed on attempt {attempt}: {err_str}", level="error", meta={"attempt": attempt})
            # If transient, backoff and retry
            if _is_transient_error(err_str) and attempt < max_retries:
                sleep_for = backoff + random.random() * 0.5
                time.sleep(sleep_for)
                backoff *= 2
                continue
            break

    # Exhausted retries or non-transient error
    raise LLMError("LLM call failed after retries", cause=last_exc)
