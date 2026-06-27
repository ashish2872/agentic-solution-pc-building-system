# src/utils/sanitize.py
import json
import math
from datetime import datetime
from typing import Any, Tuple, Dict, List, Union, Optional

TRUNCATE_STR_CHARS = 1000    # max chars for any string value (tune as needed)
TRUNCATE_LIST_ITEMS = 6      # keep this many elements from large lists
TRUNCATE_DICT_KEYS = 40      # max keys to include from very large dicts

def _is_message_like(obj: Any) -> bool:
    """Heuristic to detect LangChain Message or similar objects."""
    return hasattr(obj, "content") and hasattr(obj, "type")

def _to_primitive(obj: Any) -> Any:
    """Try to convert unknown object to a primitive/dict safely."""
    # Pydantic BaseModel support
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # LangChain-like message
    if _is_message_like(obj):
        try:
            return getattr(obj, "content")
        except Exception:
            return str(obj)
    # objects with __dict__
    if hasattr(obj, "__dict__"):
        try:
            return {k: _to_primitive(v) for k, v in vars(obj).items()}
        except Exception:
            return str(obj)
    # fallback to str
    return str(obj)


def sanitize_json_for_prompt(
    data: Any,
    *,
    max_string_len: int = TRUNCATE_STR_CHARS,
    max_list_items: int = TRUNCATE_LIST_ITEMS,
    max_dict_keys: int = TRUNCATE_DICT_KEYS,
    preserve_top_level_keys: Optional[List[str]] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Deep-sanitize an arbitrary JSON-like object for safe inclusion in a prompt.

    Returns:
      - sanitized: a JSON-serializable structure (dict/list/primitives) with large items truncated
      - summary: meta-summary with counts and truncation notes for logging/tracing

    Behavior:
      - Converts message/model/class instances to primitives (content/dict/str)
      - Truncates long strings to max_string_len and appends "…[TRUNCATED]"
      - Truncates long lists to the first max_list_items and appends a note item
      - Truncates dicts with many keys to a subset and adds a "__truncated_keys_count" key
    """
    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "original_type": type(data).__name__,
        "truncated_strings": 0,
        "truncated_lists": 0,
        "truncated_dicts": 0,
        "coerced_objects": 0,
    }

    def _sanitize(obj: Any, depth: int = 0) -> Any:
        # primitives pass through
        if obj is None or isinstance(obj, (bool, int, float, str)):
            if isinstance(obj, str):
                if len(obj) > max_string_len:
                    summary["truncated_strings"] += 1
                    return obj[:max_string_len] + "…[TRUNCATED]"
            return obj

        # try to handle dict-like
        if isinstance(obj, dict):
            keys = list(obj.keys())
            sanitized = {}
            # allow preserving some top-level keys if requested
            if depth == 0 and preserve_top_level_keys:
                keys_to_iterate = [k for k in keys if k in preserve_top_level_keys] + [k for k in keys if k not in preserve_top_level_keys]
            else:
                keys_to_iterate = keys

            # truncate extremely wide dicts
            if len(keys_to_iterate) > max_dict_keys:
                keys_to_iterate = keys_to_iterate[:max_dict_keys]
                summary["truncated_dicts"] += 1
            for k in keys_to_iterate:
                sanitized_k = k if isinstance(k, str) else str(k)
                sanitized[sanitized_k] = _sanitize(obj[k], depth + 1)
            if len(obj) > len(keys_to_iterate):
                sanitized["__truncated_keys_count"] = len(obj) - len(keys_to_iterate)
            return sanitized

        # lists/tuples
        if isinstance(obj, (list, tuple)):
            length = len(obj)
            if length > max_list_items:
                summary["truncated_lists"] += 1
                sanitized_part = [_sanitize(x, depth + 1) for x in obj[:max_list_items]]
                sanitized_part.append(f"...[{length - max_list_items} more items truncated]")
                return sanitized_part
            else:
                return [_sanitize(x, depth + 1) for x in obj]

        # sets and other iterables - convert to list
        if isinstance(obj, set):
            lst = list(obj)
            return _sanitize(lst, depth)

        # objects: try to coerce to dict, message-like or pydantic
        primitive = _to_primitive(obj)
        if isinstance(primitive, (dict, list, tuple)):
            summary["coerced_objects"] += 1
            return _sanitize(primitive, depth)
        # primitive fallback (string)
        summary["coerced_objects"] += 1
        s = str(primitive)
        if len(s) > max_string_len:
            summary["truncated_strings"] += 1
            return s[:max_string_len] + "…[TRUNCATED]"
        return s

    sanitized = _sanitize(data, depth=0)
    return sanitized, summary
