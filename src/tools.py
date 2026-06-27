# src/tools.py
from typing import List, Dict, Any, Optional, Tuple
import time
import re

from src.database import execute_read_query, get_db_connection
from src.db_metadata import get_table_column_names
from src.logging_utils import attach_log
from src.errors import ValidationError

# Config
MAX_ROWS = 50
DB_RETRY_ATTEMPTS = 3
DB_RETRY_BACKOFF = 0.25  # seconds


def get_database_schema(state: Optional[dict] = None) -> str:
    """
    Retrieves the schema information for all tables in the computer components database.
    Use this tool at the beginning of a task to understand what tables and columns
    are available for querying.
    """
    schema_query = """
    SELECT tbl_name, sql
    FROM sqlite_master
    WHERE type='table' AND tbl_name NOT LIKE 'sqlite_%';
    """
    try:
        tables = execute_read_query(schema_query)
    except Exception as e:
        if state is not None:
            attach_log(state, "tools", f"get_database_schema failed: {str(e)}", level="error")
        return "Could not retrieve database schema or database is empty."

    if not tables or "error" in (tables[0] or {}):
        if state is not None:
            attach_log(state, "tools", "get_database_schema returned no tables or an error", level="warning")
        return "Could not retrieve database schema or database is empty."

    schema_text = "Database Schema Layout:\n"
    for table in tables:
        schema_text += f"\nTable: {table.get('tbl_name')}\n"
        schema_text += f"Creation SQL:\n{table.get('sql')}\n"
        schema_text += "-" * 40 + "\n"

    if state is not None:
        attach_log(state, "tools", "Retrieved database schema", meta={"tables": [t.get("tbl_name") for t in tables]})

    return schema_text


def _safe_execute_query(sql_query: str, state: Optional[dict] = None) -> List[Dict[str, Any]]:
    """
    Executes the SQL SELECT query with a small retry for transient DB errors and
    logs errors to the agent state. Returns a list of row dicts or raises.
    """
    attempt = 0
    while attempt < DB_RETRY_ATTEMPTS:
        try:
            result = execute_read_query(sql_query)
            if state is not None:
                attach_log(state, "tools", "SQL executed", meta={"sql_preview": sql_query[:200], "rows": len(result) if isinstance(result, list) else 0})
            return result
        except Exception as e:
            attempt += 1
            if state is not None:
                attach_log(state, "tools", f"SQL execution attempt {attempt} failed: {str(e)}", level="warning", meta={"sql_preview": sql_query[:200]})
            if attempt < DB_RETRY_ATTEMPTS:
                time.sleep(DB_RETRY_BACKOFF * attempt)
            else:
                # Final failure: re-raise so caller can handle
                if state is not None:
                    attach_log(state, "tools", f"SQL execution ultimately failed after {attempt} attempts", level="error", meta={"sql_preview": sql_query[:200]})
                raise


def run_sql_query(sql_query: str, state: Optional[dict] = None) -> str:
    """
    Executes a read-only SQL SELECT query against the computer components database
    and returns the string representation of the results.

    Args:
        sql_query (str): A valid SQL SELECT statement.
        state (dict|None): optional agent state to log into.

    Returns:
        str: A formatted string of the rows returned, or an error message.
    """
    # Guardrail: Prevent destructive commands and multi-statement
    clean_sql = re.sub(r'--.*$', '', sql_query, flags=re.MULTILINE).strip().rstrip(';')
    if ';' in clean_sql:
        msg = "Error: Security Policy Violation. Multiple statements are not permitted."
        if state is not None:
            attach_log(state, "tools", f"Rejected multi-statement query", level="warning", meta={"sql_preview": sql_query[:200]})
        return msg

    forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "GRANT"]
    if any(keyword in sql_query.upper() for keyword in forbidden_keywords):
        msg = "Error: Security Policy Violation. Only read-only SELECT queries are permitted."
        if state is not None:
            attach_log(state, "tools", f"Rejected destructive query", level="warning", meta={"sql_preview": sql_query[:200]})
        return msg

    # Validate against metadata before executing
    is_valid, validation_msg = validate_sql_against_metadata(sql_query)
    if not is_valid:
        if state is not None:
            attach_log(state, "tools", f"SQL validation failed: {validation_msg}", level="warning", meta={"sql_preview": sql_query[:200]})
        return f"Validation error: {validation_msg}"

    try:
        results = _safe_execute_query(sql_query, state=state)
    except Exception as e:
        return f"Execution Failed: {str(e)}"

    if not results:
        if state is not None:
            attach_log(state, "tools", "Query executed but returned 0 results", level="info", meta={"sql_preview": sql_query[:200]})
        return "Query executed successfully, but returned 0 results."

    # Cap rows to avoid bloating LLM context window
    truncated = len(results) > MAX_ROWS
    rows_to_display = results[:MAX_ROWS]

    # Format headers
    headers = list(rows_to_display[0].keys())
    output_lines = []
    output_lines.append(" | ".join(headers))
    output_lines.append("-" * len(output_lines[0]))

    for row in rows_to_display:
        output_lines.append(" | ".join(str(row.get(h, "")) for h in headers))

    if truncated:
        output_lines.append(f"\n... (truncated — showing {MAX_ROWS} of {len(results)} rows)")

    formatted = "\n".join(output_lines)

    if state is not None:
        attach_log(state, "tools", "Formatted SQL result for LLM", meta={"rows_shown": min(len(results), MAX_ROWS), "total_rows": len(results)})

    return formatted


def validate_sql_against_metadata(sql_query: str) -> Tuple[bool, str]:
    """
    Validates that the SQL query only references known tables and columns.
    Returns (is_valid, error_message).
    """
    known = get_table_column_names()

    # Extract table name after FROM or JOIN (support multiple forms)
    table_match = re.search(r'FROM\s+`?(\w[\w\-]*)`?', sql_query, re.IGNORECASE)
    if not table_match:
        # Try JOIN
        join_match = re.search(r'JOIN\s+`?(\w[\w\-]*)`?', sql_query, re.IGNORECASE)
        if not join_match:
            return False, "Could not identify a table name in the query."
        table_name = join_match.group(1).lower()
    else:
        table_name = table_match.group(1).lower()

    if table_name not in known:
        return False, f"Unknown table '{table_name}'. Valid tables: {list(known.keys())}"

    # Extract column names from SELECT clause
    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE | re.DOTALL)
    if select_match:
        select_clause = select_match.group(1).strip()
        if select_clause != "*":
            columns = [c.strip().strip('`') for c in select_clause.split(",")]
            valid_cols = known.get(table_name, [])
            invalid = [c for c in columns if c not in valid_cols]
            if invalid:
                return False, f"Unknown columns {invalid} for table '{table_name}'. Valid columns: {valid_cols}"

    return True, "OK"
