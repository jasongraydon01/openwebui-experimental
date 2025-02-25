import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DATABASE_PATH")

def initialize_database():
    """Initialize the SQLite database with necessary tables."""
    # Check if the file_log.db exists
    if not os.path.exists(DB_PATH):
        print(f"⚠️ Database not found. Creating {DB_PATH}...")
    
    # Connect to the SQLite database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create file_log table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS file_log (
        file_name TEXT PRIMARY KEY,
        file_hash TEXT,
        last_modified REAL,
        last_processed REAL,
        slide_count INTEGER
    )
    ''')
    
    # Create slides table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS slides (
        vector_id TEXT PRIMARY KEY,
        file_name TEXT,
        slide_number INTEGER,
        content TEXT,
        keywords TEXT,
        context_slides TEXT,
        FOREIGN KEY (file_name) REFERENCES file_log (file_name) ON DELETE CASCADE
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {DB_PATH} with file_log and slides tables")

if __name__ == "__main__":
    initialize_database()