
"""Persistence utilities for episodic and working memory."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:  # Optional heavy dependencies
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

import logging

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """SQLite-backed long-term memory with optional vector search."""

    def __init__(
        self,
        db_path: str,
        *,
        embedding_model: str = "all-MiniLM-L6-v2",
        retention_days: Optional[int] = None,
    ) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self._conn = sqlite3.connect(self.path.as_posix(), check_same_thread=False)
        self._lock = threading.RLock()
        self._bootstrap()

        self._use_vectors = bool(SentenceTransformer and faiss and _np is not None)
        self._encoder: Optional[SentenceTransformer] = None
        self._index = None
        self._id_to_row: Dict[int, int] = {}
        self._next_index = 0

        if self._use_vectors:
            try:
                self._encoder = SentenceTransformer(embedding_model)
                self._index = faiss.IndexFlatL2(self._encoder.get_sentence_embedding_dimension())
                logger.info("Episodic memory using vector search with %s", embedding_model)
            except Exception as exc:  # pragma: no cover - optional path
                logger.warning("Vector search disabled: %s", exc)
                self._use_vectors = False

    # ------------------------------------------------------------------
    # schema helpers
    # ------------------------------------------------------------------
    def _bootstrap(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent TEXT,
                    role TEXT,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    importance REAL DEFAULT 0.5,
                    metadata TEXT
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(timestamp DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_episodes_importance ON episodes(importance DESC)")
            self._conn.commit()

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------
    def add(
        self,
        *,
        agent: str,
        role: str,
        content: str,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        content = content.strip()
        if not content:
            return 0

        payload = json.dumps(metadata or {})
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO episodes (agent, role, content, timestamp, importance, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (agent, role, content, time.time(), importance, payload),
            )
            episode_id = cur.lastrowid
            self._conn.commit()

        if self._use_vectors and self._encoder is not None and _np is not None:
            try:  # pragma: no cover - network heavy
                vector = self._encoder.encode([content])[0]
                vector = _np.asarray(vector, dtype="float32")
                self._index.add(vector.reshape(1, -1))  # type: ignore[union-attr]
                self._id_to_row[self._next_index] = episode_id
                self._next_index += 1
            except Exception as exc:  # pragma: no cover
                logger.debug("Vector index append failed: %s", exc)
        return episode_id

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------
    def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT agent, role, content, timestamp, importance, metadata FROM episodes ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            return [self._row_to_dict(row) for row in cur.fetchall()]

    def search(self, query: str, limit: int = 20, *, min_importance: float = 0.0) -> List[Dict[str, Any]]:
        like = f"%{query}%"
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT agent, role, content, timestamp, importance, metadata
                FROM episodes
                WHERE content LIKE ? AND importance >= ?
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
                """,
                (like, min_importance, limit),
            )
            return [self._row_to_dict(row) for row in cur.fetchall()]

    def search_semantic(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        if not self._use_vectors or self._encoder is None or self._index is None or _np is None:
            return self.search(query, limit)
        if self._index.ntotal == 0:  # type: ignore[union-attr]
            return []
        try:  # pragma: no cover - heavy dependency
            vector = self._encoder.encode([query])[0]
            vector = _np.asarray(vector, dtype="float32")
            distances, indices = self._index.search(vector.reshape(1, -1), min(limit, self._index.ntotal))  # type: ignore[operator]
            rows = []
            for idx, dist in zip(indices[0], distances[0]):
                if int(idx) in self._id_to_row:
                    rows.append((self._id_to_row[int(idx)], float(dist)))
            if not rows:
                return []
            episode_ids = [row_id for row_id, _ in rows]
            placeholders = ",".join(["?"] * len(episode_ids))
            with self._lock:
                cur = self._conn.cursor()
                cur.execute(
                    f"SELECT id, agent, role, content, timestamp, importance, metadata FROM episodes WHERE id IN ({placeholders})",
                    episode_ids,
                )
                data = {row[0]: row[1:] for row in cur.fetchall()}
            results = []
            for eid, dist in rows:
                if eid in data:
                    agent, role, content, ts, imp, meta = data[eid]
                    results.append(
                        {
                            "agent": agent,
                            "role": role,
                            "content": content,
                            "timestamp": ts,
                            "importance": imp,
                            "metadata": json.loads(meta) if meta else {},
                            "distance": dist,
                        }
                    )
            return results
        except Exception as exc:
            logger.debug("Semantic search failed, falling back: %s", exc)
            return self.search(query, limit)

    # ------------------------------------------------------------------
    def prune(self) -> None:
        if self.retention_days is None:
            return
        horizon = time.time() - (self.retention_days * 86400)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM episodes WHERE timestamp < ?", (horizon,))
            self._conn.commit()

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM episodes")
            self._conn.commit()
            self._id_to_row.clear()
            if self._index is not None:  # pragma: no cover - heavy dep
                self._index.reset()
            self._next_index = 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    def _row_to_dict(self, row: Iterable[Any]) -> Dict[str, Any]:
        agent, role, content, ts, importance, metadata = row
        return {
            "agent": agent,
            "role": role,
            "content": content,
            "timestamp": ts,
            "importance": importance,
            "metadata": json.loads(metadata) if metadata else {},
        }


class WorkingMemory:
    """Rolling buffer of the most recent conversational turns."""

    def __init__(self, *, context_window: int = 8) -> None:
        self.context_window = max(1, context_window)
        self._messages: List[Dict[str, str]] = []
        self._lock = threading.RLock()

    def add(self, role: str, content: str) -> None:
        payload = {"role": role, "content": content}
        with self._lock:
            self._messages.append(payload)
            if len(self._messages) > self.context_window:
                self._messages = self._messages[-self.context_window :]

    def get_messages(self) -> List[Dict[str, str]]:
        with self._lock:
            return list(self._messages)

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()

    def to_formatted_string(self) -> str:
        with self._lock:
            return "\n".join(f"{msg.get('role', 'agent')}: {msg.get('content', '')}" for msg in self._messages)

class BrainDB:
    """Thin facade to keep backwards compatibility with existing imports."""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.episodic = EpisodicMemory(self.db_path.with_suffix(".episodes.db").as_posix())
        self.working = WorkingMemory()

    async def start(self) -> None:  # pragma: no cover - async noop
        return

    async def stop(self) -> None:  # pragma: no cover - async noop
        self.episodic.close()

    def query(self, sql: str) -> Any:  # pragma: no cover - legacy stub
        raise NotImplementedError("Direct SQL queries are not supported on BrainDB")
