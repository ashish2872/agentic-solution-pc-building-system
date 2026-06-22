from typing import List, Dict, Any
from src.database import execute_read_query, get_db_connection
from src.db_metadata import get_table_column_names

def get_database_schema() -> str:
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
    tables = execute_read_query(schema_query)
    
    if not tables or "error" in tables[0]:
        return "Could not retrieve database schema or database is empty."
    
    schema_text = "Database Schema Layout:\n"
    for table in tables:
        schema_text += f"\nTable: {table['tbl_name']}\n"
        schema_text += f"Creation SQL:\n{table['sql']}\n"
        schema_text += "-" * 40 + "\n"
        
    return schema_text

def run_sql_query(sql_query: str) -> str:
    """
    Executes a read-only SQL SELECT query against the computer components database 
    and returns the string representation of the results.
    
    Args:
        sql_query (str): A valid SQL SELECT statement.
        
    Returns:
        str: A string formatting of the rows returned, or an error message.
    """
    # Guardrail: Prevent destructive commands
    import re

    # Guardrail 1: Strip SQL comments, then check for multiple statements
    clean_sql = re.sub(r'--.*$', '', sql_query, flags=re.MULTILINE).strip().rstrip(';')
    if ';' in clean_sql:
        return "Error: Security Policy Violation. Multiple statements are not permitted."

    # Guardrail 2: Prevent destructive keywords
    forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "GRANT"]
    if any(keyword in sql_query.upper() for keyword in forbidden_keywords):
        return "Error: Security Policy Violation. Only read-only SELECT queries are permitted."


    results = execute_read_query(sql_query)
    
    if not results:
        return "Query executed successfully, but returned 0 results."
        
    if "error" in results[0]:
        return f"Execution Failed: {results[0]['error']}"

    # Cap rows to avoid bloating LLM context window
    MAX_ROWS = 50
    truncated = len(results) > MAX_ROWS
    rows_to_display = results[:MAX_ROWS]

    # Format the dictionary rows cleanly for the LLM to digest
    output = []
    headers = list(rows_to_display[0].keys())
    output.append(" | ".join(headers))
    output.append("-" * len(output[0]))
    
    for row in rows_to_display:
        output.append(" | ".join(str(row[h]) for h in headers))

    if truncated:
        output.append(f"\n... (truncated — showing 50 of {len(results)} rows)")
        
    return "\n".join(output)

# Add this to src/tools.py


def validate_sql_against_metadata(sql_query: str) -> tuple[bool, str]:
    """
    Validates that the SQL query only references known tables and columns.
    Returns (is_valid, error_message).
    """
    import re
    known = get_table_column_names()

    # Extract table name after FROM or JOIN
    table_match = re.search(r'FROM\s+`?(\w[\w\-]*)`?', sql_query, re.IGNORECASE)
    if not table_match:
        return False, "Could not identify a table name in the query."

    table_name = table_match.group(1).lower()

    if table_name not in known:
        return False, f"Unknown table '{table_name}'. Valid tables: {list(known.keys())}"

    # Extract column names from SELECT clause
    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE | re.DOTALL)
    if select_match:
        select_clause = select_match.group(1).strip()
        if select_clause != "*":
            columns = [c.strip().strip('`') for c in select_clause.split(",")]
            valid_cols = known[table_name]
            invalid = [c for c in columns if c not in valid_cols]
            if invalid:
                return False, f"Unknown columns {invalid} for table '{table_name}'. Valid columns: {valid_cols}"

    return True, "OK"
