import sqlite3
import os

db_path = 'drishti_memory.db'

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    print("Clearing 'memory' table...")
    c.execute("DELETE FROM memory")
    conn.commit()
    conn.close()
    print("Database cleared successfully. Inconsistent timestamp formats removed.")
else:
    print(f"Database {db_path} not found. Nothing to clear.")
