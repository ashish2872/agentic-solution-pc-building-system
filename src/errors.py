# src/errors.py
from typing import Optional

class LLMError(Exception):
    """Raised when LLM calls fail after retries."""
    def __init__(self, message: str, code: Optional[str] = None, cause: Optional[Exception] = None):
        super().__init__(message)
        self.code = code
        self.cause = cause

class ValidationError(Exception):
    """Raised for validation failures in inputs or intermediate data."""
    pass

