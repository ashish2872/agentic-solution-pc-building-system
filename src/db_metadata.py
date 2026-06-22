# src/db_metadata.py
# Dynamically introspects the SQLite DB to build metadata at runtime.

import sqlite3
import os

DB_PATH = os.getenv("DATABASE_PATH", "pc_components.db")


def _get_connection():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_all_table_names() -> list[str]:
    """Returns all user-defined table names from the DB."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT tbl_name FROM sqlite_master 
            WHERE type='table' AND tbl_name NOT LIKE 'sqlite_%'
            ORDER BY tbl_name;
        """)
        return [row["tbl_name"] for row in cursor.fetchall()]


def get_table_columns(table_name: str) -> list[dict]:
    """
    Returns column metadata for a given table using PRAGMA.
    Each entry has: name, type, notnull, dflt_value, pk
    """
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        return [dict(row) for row in cursor.fetchall()]


def get_sample_rows(table_name: str, limit: int = 2) -> list[dict]:
    """
    Fetches a few sample rows so the LLM understands real data values.
    Helps ground things like price ranges, naming conventions etc.
    """
    with _get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(f"SELECT * FROM `{table_name}` LIMIT {limit};")
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error:
            return []


def build_dynamic_metadata() -> dict:
    """
    Introspects the entire DB and returns a structured metadata dict.
    Format:
    {
        "table_name": {
            "columns": [{"name": ..., "type": ..., "notnull": ...}, ...],
            "sample_rows": [{...}, {...}]
        },
        ...
    }
    """
    metadata = {}
    tables = get_all_table_names()

    for table in tables:
        columns = get_table_columns(table)
        samples = get_sample_rows(table)
        metadata[table] = {
            "columns": columns,
            "sample_rows": samples
        }

    return metadata


def get_table_column_names() -> dict[str, list[str]]:
    """
    Returns a simple dict of table -> list of valid column names.
    Used for SQL validation before execution.
    """
    tables = get_all_table_names()
    result = {}
    for table in tables:
        cols = get_table_columns(table)
        result[table] = [col["name"] for col in cols]
    return result


def get_metadata_as_text() -> str:
    """
    Returns a formatted string of the full DB metadata.
    Pass this into SQL generation prompts to ground the LLM.
    """
    metadata = build_dynamic_metadata()

    lines = ["DATABASE METADATA — Use this to write accurate SQL queries.\n"]
    lines.append("IMPORTANT RULES:")
    lines.append("- Always wrap table names in backticks e.g. `cases`, `power-supplies`")
    lines.append("- Only use column names listed below — do not invent columns")
    lines.append("- Always use LIMIT 3 and ORDER BY price DESC")
    lines.append("- Only write SELECT statements\n")

    for table, info in metadata.items():
        lines.append(f"Table: `{table}`")

        # Column info
        lines.append("  Columns:")
        for col in info["columns"]:
            nullable = "NOT NULL" if col["notnull"] else "nullable"
            pk = " (PRIMARY KEY)" if col["pk"] else ""
            lines.append(f"    - {col['name']}: {col['type']} {nullable}{pk}")

        # Sample rows — helps LLM understand real values
        if info["sample_rows"]:
            lines.append("  Sample rows:")
            for row in info["sample_rows"]:
                lines.append(f"    {row}")

        lines.append("")

    return "\n".join(lines)
