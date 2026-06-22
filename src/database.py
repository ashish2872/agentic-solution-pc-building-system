import os
import sqlite3
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


# In a real setup, pull this from config or env variables
DB_PATH = os.getenv("DATABASE_PATH", "pc_components.db")

def get_db_connection():
    """Establishes and returns a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        # Allows accessing rows by column name like a dictionary
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        raise e

def execute_read_query(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Executes a SELECT query and returns results as a list of dictionaries."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"SQL Execution Error: {str(e)} | Query: {query}")
            return [{"error": f"SQL Execution Error: {str(e)}"}]
