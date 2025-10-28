"""
Database initialization for Dexter-Gliksbot.

This module ensures all required SQLite databases exist before the system starts.
"""
import sqlite3
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def init_databases(data_dir: str = "./data") -> None:
    """Initialize all required SQLite databases."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    databases = {
        "brain.db": _init_brain_db,
        "brain.episodes.db": _init_episodes_db,
        "timeline.db": _init_timeline_db,
        "snapshots.db": _init_snapshots_db,
        "shared_brain.sqlite": _init_shared_brain_db,
    }
    
    for db_name, init_func in databases.items():
        db_path = data_path / db_name
        if not db_path.exists():
            logger.info(f"Creating database: {db_path}")
            init_func(str(db_path))
        else:
            logger.debug(f"Database already exists: {db_path}")


def _init_brain_db(db_path: str) -> None:
    """Initialize the main brain database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Observations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            bus TEXT NOT NULL,
            topic TEXT NOT NULL,
            content TEXT NOT NULL,
            summary TEXT,
            embedding BLOB
        )
    """)
    
    # Entities table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            value TEXT NOT NULL,
            first_seen REAL NOT NULL,
            last_seen REAL NOT NULL,
            count INTEGER DEFAULT 1,
            UNIQUE(type, value)
        )
    """)
    
    # Relations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            relation TEXT NOT NULL,
            strength REAL DEFAULT 1.0,
            first_seen REAL NOT NULL,
            last_seen REAL NOT NULL,
            UNIQUE(source, target, relation)
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Initialized brain database: {db_path}")


def _init_episodes_db(db_path: str) -> None:
    """Initialize the episodes database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            agent TEXT NOT NULL,
            action TEXT NOT NULL,
            result TEXT,
            context TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Initialized episodes database: {db_path}")


def _init_timeline_db(db_path: str) -> None:
    """Initialize the timeline database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS timeline (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            event_type TEXT NOT NULL,
            data TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Initialized timeline database: {db_path}")


def _init_snapshots_db(db_path: str) -> None:
    """Initialize the snapshots database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id TEXT PRIMARY KEY,
            timestamp REAL NOT NULL,
            data BLOB NOT NULL,
            compressed INTEGER DEFAULT 1
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Initialized snapshots database: {db_path}")


def _init_shared_brain_db(db_path: str) -> None:
    """Initialize the shared brain database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS shared_knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL UNIQUE,
            value TEXT NOT NULL,
            updated REAL NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Initialized shared brain database: {db_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_databases()
