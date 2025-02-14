import sqlite3
import hashlib
import os
from datetime import datetime

# Initialize SQLite Database
def initialize_db(db_path="file_tracking.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS file_tracking (
            file_name TEXT PRIMARY KEY,
            file_hash TEXT,
            last_processed TIMESTAMP,
            pinecone_id TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Run Processing
if __name__ == "__main__":
    initialize_db()