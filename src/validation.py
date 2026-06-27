# src/validation.py
import re
from typing import Tuple, Optional, List, Dict

PROFANITY_BLACKLIST = {"fuck", "shit", "bastard", "idiot"}  # edit to taste / replace with a curated list

def sanitize_user_input(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Basic content filtering. Returns (clean_text, error_message).
    If text is unacceptable, returns (None, message).
    """
    if not text or not text.strip():
        return None, "Empty input."

    low = text.lower()
    for bad in PROFANITY_BLACKLIST:
        if bad in low:
            return None, "Please avoid offensive language."

    # Strip excessive whitespace
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned, None

def validate_requirements_schema(req: Dict) -> Tuple[bool, List[str]]:
    """
    Ensure requirements dict has sensible types and values.
    Returns (is_valid, list_of_errors).
    """
    errors = []
    if "budget" in req and req["budget"] is not None:
        try:
            b = float(req["budget"])
            if b <= 0:
                errors.append("budget must be positive")
        except Exception:
            errors.append("budget must be numeric")
    if "primary_use" in req and req["primary_use"] is not None:
        if not isinstance(req["primary_use"], str) or not req["primary_use"].strip():
            errors.append("primary_use must be a non-empty string")
    # preferences should be list
    if "preferences" in req and req["preferences"] is not None:
        if not isinstance(req["preferences"], (list, tuple)):
            errors.append("preferences must be an array/list")
    return (len(errors) == 0), errors

def estimate_prompt_size_chars(*parts: str) -> int:
    """Rudimentary estimator of prompt size in characters."""
    return sum(len(p or "") for p in parts)

def need_truncate(prompt_chars: int, soft_limit_chars: int = 30000) -> bool:
    """Return True if prompt likely exceeds a safe budget."""
    return prompt_chars > soft_limit_chars
