from __future__ import annotations
import sqlite_utils
import logging
import time
from pathlib import Path
from typing import List, Callable

logger = logging.getLogger(__name__)

class MigrationManager:
    """
    Simple migration system for SQLite database schema evolution.
    Runs migrations in order and tracks which ones have been applied.
    """
    
    def __init__(self, db_path: Path):
        self.db = sqlite_utils.Database(db_path)
        self._init_migrations_table()
        self.migrations: List[tuple[str, Callable]] = []
        self._register_default_migrations()
    
    def _init_migrations_table(self):
        """Create migrations tracking table."""
        self.db["migrations"].create({
            "version": str,
            "applied_at": float,
            "description": str
        }, pk="version", if_not_exists=True)
    
    def _register_default_migrations(self):
        """Register default migrations for the brain database."""
        
        # Migration 001: Initial brain schema
        self.add_migration("001_initial", "Initial brain database schema", self._migration_001)
        
        # Migration 002: Add embedding support
        self.add_migration("002_embeddings", "Add embedding support for vectors", self._migration_002)
        
        # Migration 003: Add task-based isolation
        self.add_migration("003_task_isolation", "Add task-based memory isolation", self._migration_003)
        
        # Migration 004: Add outbox table
        self.add_migration("004_outbox", "Add transactional outbox table", self._migration_004)
        
        # Migration 005: Add STM/LTM tables
        self.add_migration("005_memory_tiers", "Add STM and LTM memory tables", self._migration_005)
        
        # Migration 006: Optimize outbox performance
        self.add_migration("006_outbox_optimize", "Optimize outbox table performance", self._migration_006)
    
    def add_migration(self, version: str, description: str, migration_func: Callable):
        """Add a new migration."""
        self.migrations.append((version, description, migration_func))
    
    def _migration_001(self):
        """Initial brain schema with basic memory tables."""
        # Main memories table
        self.db["memories"].create({
            "id": int,
            "kind": str,
            "content": str,
            "meta": str,
            "ts": float
        }, pk="id", if_not_exists=True)
        
        # Full-text search table
        self.db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content, 
                content='memories', 
                content_rowid='id'
            )
        """)
        
        # Edges table for relationships
        self.db["edges"].create({
            "id": int,
            "src": int,
            "rel": str,
            "dst": int,
            "ts": float
        }, pk="id", if_not_exists=True)
        
        # Create indexes
        self.db["memories"].create_index(["kind"], if_not_exists=True)
        self.db["memories"].create_index(["ts"], if_not_exists=True)
        self.db["edges"].create_index(["src"], if_not_exists=True)
        self.db["edges"].create_index(["dst"], if_not_exists=True)
        
        logger.info("Applied migration 001: Initial brain schema")
    
    def _migration_002(self):
        """Add embedding support."""
        # Add embedding column to memories
        try:
            self.db.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
            logger.info("Added embedding column to memories table")
        except Exception as e:
            if "duplicate column" not in str(e).lower():
                raise
            logger.info("Embedding column already exists")
        
        # Create vector search index (sqlite-vss would be better but this works)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories(embedding)
        """)
        
        logger.info("Applied migration 002: Added embedding support")
    
    def _migration_003(self):
        """Add task-based isolation."""
        # Add task_root column for namespace isolation
        try:
            self.db.execute("ALTER TABLE memories ADD COLUMN task_root TEXT DEFAULT 'default'")
            logger.info("Added task_root column to memories table")
        except Exception as e:
            if "duplicate column" not in str(e).lower():
                raise
            logger.info("task_root column already exists")
        
        # Add agent_id column
        try:
            self.db.execute("ALTER TABLE memories ADD COLUMN agent_id TEXT DEFAULT 'system'")
            logger.info("Added agent_id column to memories table")
        except Exception as e:
            if "duplicate column" not in str(e).lower():
                raise
            logger.info("agent_id column already exists")
        
        # Create indexes for task and agent filtering
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_memories_task_root ON memories(task_root)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_memories_agent_id ON memories(agent_id)")
        
        logger.info("Applied migration 003: Added task-based isolation")
    
    def _migration_004(self):
        """Add transactional outbox table."""
        self.db["outbox"].create({
            "id": int,
            "task": str,
            "payload": str,
            "status": str,
            "created_at": float,
            "retry_count": int
        }, pk="id", defaults={"retry_count": 0}, if_not_exists=True)
        
        # Create indexes
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_outbox_status ON outbox(status)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_outbox_created_at ON outbox(created_at)")
        
        logger.info("Applied migration 004: Added outbox table")
    
    def _migration_005(self):
        """Add STM and LTM memory tables."""
        # LTM tokens table
        self.db["ltm_tokens"].create({
            "id": str,
            "text": str,
            "embedding": bytes,
            "task_root": str,
            "agent_id": str,
            "meta": str,
            "size": int,
            "atime": float,
            "refs": str
        }, pk="id", if_not_exists=True)
        
        # Create indexes
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_ltm_tokens_task_root ON ltm_tokens(task_root)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_ltm_tokens_agent_id ON ltm_tokens(agent_id)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_ltm_tokens_atime ON ltm_tokens(atime)")
        
        logger.info("Applied migration 005: Added STM/LTM memory tables")
    
    def _migration_006(self):
        """Optimize outbox table performance."""
        # Ensure retry_count has a value for existing records
        self.db.execute("UPDATE outbox SET retry_count = COALESCE(retry_count, 0)")
        
        # Add composite index for status and created_at for faster queries
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_outbox_status_created 
            ON outbox(status, created_at)
        """)
        
        # Add index for task type for faster routing
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_outbox_task ON outbox(task)")
        
        logger.info("Applied migration 006: Optimized outbox table")
    
    def run_migrations(self):
        """Run all pending migrations."""
        logger.info("Starting database migrations...")
        
        # Get already applied migrations
        applied = set(row["version"] for row in self.db["migrations"].rows)
        
        # Run pending migrations
        for version, description, migration_func in self.migrations:
            if version in applied:
                logger.debug(f"Migration {version} already applied")
                continue
            
            logger.info(f"Applying migration {version}: {description}")
            
            try:
                migration_func()
                
                # Record migration as applied
                self.db["migrations"].insert({
                    "version": version,
                    "applied_at": time.time(),
                    "description": description
                })
                
                logger.info(f"Successfully applied migration {version}")
                
            except Exception as e:
                logger.error(f"Failed to apply migration {version}: {e}")
                raise
        
        logger.info("All migrations completed successfully")
    
    def get_status(self) -> dict:
        """Get migration status."""
        applied = set(row["version"] for row in self.db["migrations"].rows)
        
        return {
            "total_migrations": len(self.migrations),
            "applied_migrations": len(applied),
            "pending_migrations": len(self.migrations) - len(applied),
            "applied_versions": list(applied),
            "latest_version": max(applied) if applied else None
        }


def run_migrations(db_path: str | Path) -> None:
    """
    Convenience function to run all migrations on a database.
    
    Args:
        db_path: Path to the SQLite database file
    """
    db_path = Path(db_path)
    manager = MigrationManager(db_path)
    manager.run_migrations()
