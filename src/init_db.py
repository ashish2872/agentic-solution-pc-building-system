import sqlite3
from pathlib import Path
import pandas as pd

def initialize_database():
    db_path = "pc_components.db"
    csv_dir = Path("data/csv")
    
    # Connect to SQLite database (it will create the file if it doesn't exist)
    conn = sqlite3.connect(db_path)
    print(f"Connected to database at: {db_path}")
    
    # Loop through your discovered CSV files
    for path in csv_dir.glob("*.csv"):
        # Convert file name (e.g., 'power-supply.csv') into a clean SQL table name ('power_supply')
        table_name = path.stem.replace("-", "_")
        print(f"Importing {path.name} into table '{table_name}'...")
        
        try:
            # Read CSV with pandas
            df = pd.read_csv(path)
            
            # Clean column names (SQL tables don't like spaces or dashes in headers)
            df.columns = [col.strip().lower().replace(" ", "_").replace("-", "_") for col in df.columns]
            
            # Dump dataframe into SQLite table
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            
        except Exception as e:
            print(f"Error processing {path.name}: {e}")
            
    conn.close()
    print("Database initialisation complete!")

if __name__ == "__main__":
    initialize_database()