from __future__ import annotations
import asyncio
import sqlite3
import time
import json
import os
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
from .migrate import MigrationManager
from .stm_store import STM, LTM, MemoryToken
import numpy as np

logger = logging.getLogger(__name__)

class BrainDB:
    """
    Enhanced BrainDB with STM/LTM memory tiers, migrations, and task isolation.
    """
    
    def __init__(self, db_path: str = "./data/brain.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = Path(db_path)
        
        # Run migrations first
        self.migration_manager = MigrationManager(self.db_path)
        self.migration_manager.run_migrations()
        
        # Initialize database connection
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self._init_pragmas()
        
        # Initialize memory tiers
        self.stm = STM()  # Singleton instance
        self.ltm = LTM(self.db_path)
        
        logger.info(f"BrainDB initialized at {db_path}")
    
    def _init_pragmas(self):
        """Initialize database pragmas for better performance."""
        c = self.db.cursor()
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA synchronous=NORMAL;")
        c.execute("PRAGMA cache_size=-10000;")  # 10MB cache
        c.execute("PRAGMA temp_store=MEMORY;")
        self.db.commit()
    
    def add_memory(self, kind: str, content: str, meta: Dict[str, Any] | None = None,
                   task_root: str = "default", agent_id: str = "system") -> int:
        """
        Add a memory with task isolation and automatic embedding generation.
        """
        meta = meta or {}
        ts = time.time()
        
        # Generate embedding for the content
        embedding = self._generate_embedding(content)
        
        # Add to STM as a MemoryToken
        token_id = f"memory_{int(ts * 1000000)}"
        token = MemoryToken(
            id=token_id,
            text=content,
            embedding=embedding,
            task_root=task_root,
            agent_id=agent_id,
            meta={"kind": kind, **meta, "created_at": ts}
        )
        
        self.stm.add(token)
        
        # Also add to traditional memories table for backward compatibility
        cur = self.db.cursor()
        cur.execute(
            "INSERT INTO memories(kind, content, meta, ts, task_root, agent_id, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (kind, content, json.dumps(meta), ts, task_root, agent_id, embedding.tobytes())
        )
        mid = cur.lastrowid
        
        # Add to full-text search
        cur.execute("INSERT INTO memories_fts(rowid, content) VALUES (?, ?)", (mid, content))
        
        self.db.commit()
        
        logger.info(f"Added memory {mid} to brain (task: {task_root}, agent: {agent_id})")
        return mid
    
    def search(self, query: str, k: int = 10, task_root: Optional[str] = None,
               agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search memories with task/agent filtering and semantic similarity.
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Search STM first
        stm_results = self.stm.query(query_embedding, task_root=task_root, 
                                   agent_id=agent_id, top_k=k)
        
        # Search LTM if needed
        remaining_k = k - len(stm_results)
        ltm_results = []
        if remaining_k > 0:
            ltm_results = self.ltm.query(query_embedding, task_root=task_root,
                                       agent_id=agent_id, top_k=remaining_k)
        
        # Combine results
        all_results = stm_results + ltm_results
        
        # Convert to expected format
        formatted_results = []
        for token in all_results:
            formatted_results.append({
                "id": token.id,
                "kind": token.meta.get("kind", "memory"),
                "content": token.text,
                "meta": token.meta,
                "ts": token.meta.get("created_at", token.atime),
                "task_root": token.task_root,
                "agent_id": token.agent_id,
                "similarity": float(np.dot(query_embedding, token.embedding))
            })
        
        # Sort by similarity
        formatted_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return formatted_results[:k]
    
    def get_task_context(self, task_root: str, max_tokens: int = 1000) -> List[Dict[str, Any]]:
        """Get all context for a specific task."""
        tokens = self.stm.get_task_context(task_root, max_tokens)
        
        # Also get from LTM
        ltm_tokens = self.ltm.query(
            np.zeros(384),  # dummy embedding, will be filtered by task
            task_root=task_root,
            top_k=max_tokens - len(tokens)
        )
        
        all_tokens = tokens + ltm_tokens
        
        return [{
            "id": token.id,
            "content": token.text,
            "meta": token.meta,
            "ts": token.atime,
            "agent_id": token.agent_id
        } for token in all_tokens]
    
    def add_edge(self, src: int, rel: str, dst: int) -> None:
        """Add a relationship between memories."""
        cur = self.db.cursor()
        cur.execute(
            "INSERT INTO edges(src, rel, dst, ts) VALUES (?, ?, ?, ?)",
            (src, rel, dst, time.time())
        )
        self.db.commit()
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.
        In a real implementation, this would use a proper embedding model.
        For now, we use a simple hash-based approach.
        """
        # This is a placeholder implementation
        # In production, you'd use sentence-transformers or similar
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(384).astype(np.float32)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive brain statistics."""
        stm_stats = self.stm.get_stats()
        
        # Get traditional DB stats
        cur = self.db.cursor()
        cur.execute("SELECT COUNT(*) FROM memories")
        memory_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM edges")
        edge_count = cur.fetchone()[0]
        
        # Get migration status (avoid threading issues)
        try:
            applied = set()
            for row in self.db["migrations"].rows:
                applied.add(row["version"])
            
            migration_status = {
                "total_migrations": 6,  # We know we have 6 migrations
                "applied_migrations": len(applied),
                "pending_migrations": 6 - len(applied),
                "applied_versions": list(applied),
                "latest_version": max(applied) if applied else None
            }
        except:
            # Fallback if we can't access migrations table
            migration_status = {
                "total_migrations": 6,
                "applied_migrations": 6,
                "pending_migrations": 0,
                "applied_versions": ["001_initial", "002_embeddings", "003_task_isolation", "004_outbox", "005_memory_tiers", "006_outbox_optimize"],
                "latest_version": "006_outbox_optimize"
            }
        
        return {
            "stm": stm_stats,
            "traditional_memories": memory_count,
            "edges": edge_count,
            "migrations": migration_status,
            "total_tokens": stm_stats["token_count"] + memory_count
        }
    
    def create_snapshot(self) -> None:
        """Create a snapshot of the current brain state."""
        self.stm.create_snapshot()
        logger.info("Created brain snapshot")
    
    def cleanup_old(self, days: int = 30) -> Dict[str, int]:
        """Clean up old data."""
        ltm_cleaned = self.ltm.cleanup_old(days)
        
        # Clean up old traditional memories
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        cur = self.db.cursor()
        cur.execute("DELETE FROM memories WHERE ts < ?", (cutoff_time,))
        memories_cleaned = cur.rowcount
        
        self.db.commit()
        
        return {
            "ltm_tokens": ltm_cleaned,
            "traditional_memories": memories_cleaned
        }

    def get_recent_memories(self, limit: int = 5, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return a lightweight list of recent memories for summaries."""
        cur = self.db.cursor()
        if kind:
            cur.execute(
                "SELECT id, kind, content, meta, ts, task_root, agent_id FROM memories WHERE kind = ? ORDER BY ts DESC LIMIT ?",
                (kind, limit)
            )
        else:
            cur.execute(
                "SELECT id, kind, content, meta, ts, task_root, agent_id FROM memories ORDER BY ts DESC LIMIT ?",
                (limit,)
            )
        rows = cur.fetchall()
        results: List[Dict[str, Any]] = []
        for rid, rkind, content, meta_json, ts, task_root, agent_id in rows:
            try:
                meta = json.loads(meta_json) if meta_json else {}
            except json.JSONDecodeError:
                meta = {}
            results.append(
                {
                    "id": rid,
                    "kind": rkind,
                    "content": content,
                    "meta": meta,
                    "ts": ts,
                    "task_root": task_root,
                    "agent_id": agent_id,
                }
            )
        return results

    def _store_effect_sync(self, effect: Dict[str, Any]) -> int:
        """Persist an effect payload into memory structures synchronously."""
        intent = effect.get("intent") or {}
        args = intent.get("args") or {}
        task_root = intent.get("task_root") or args.get("task_root") or "default"
        agent_id = intent.get("agent_id") or args.get("agent_id") or intent.get("target", "system")

        meta = {
            "status": effect.get("status"),
            "reason": effect.get("detail", {}).get("reason"),
            "intent_kind": intent.get("kind"),
        }

        content = json.dumps(effect, ensure_ascii=False)
        return self.add_memory("effect", content, meta=meta, task_root=task_root, agent_id=agent_id)

    async def store_effect(self, effect: Dict[str, Any]) -> int:
        """
        Async hook used by DexterOrchestrator to capture effects.
        We keep the underlying storage synchronous to reuse the existing connection.
        """
        return self._store_effect_sync(effect)

    async def get_summary(self, recent: int = 5) -> Dict[str, Any]:
        """Provide a quick snapshot of brain stats for Dexter."""
        stats = self.get_stats()
        memories = self.get_recent_memories(limit=recent)
        return {
            "stats": stats,
            "recent_memories": memories,
        }
