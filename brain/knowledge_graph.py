"""Advanced knowledge graph storage for Dexter."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses representing the typed objects stored in the graph
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EntityType:
    """Definition of an entity category."""

    name: str
    description: str = ""
    schema: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationType:
    """Definition of a relation predicate."""

    name: str
    description: str = ""
    domain: Optional[str] = None
    range: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    """Concrete entity stored in the knowledge graph."""

    name: str
    entity_type: str
    metadata: Dict[str, Any]
    first_seen: float
    last_updated: float
    confidence: float
    support: int
    contradiction: int
    embedding: Tuple[float, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "metadata": self.metadata,
            "first_seen": self.first_seen,
            "last_updated": self.last_updated,
            "confidence": self.confidence,
            "support": self.support,
            "contradiction": self.contradiction,
            "embedding": list(self.embedding),
        }


@dataclass
class Relation:
    """Directional relation between two entities."""

    relation_id: int
    subject: str
    predicate: str
    obj: str
    confidence: float
    support: int
    contradiction: int
    source: Optional[str]
    timestamp: float
    metadata: Dict[str, Any]
    embedding: Tuple[float, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.relation_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.obj,
            "confidence": self.confidence,
            "support": self.support,
            "contradiction": self.contradiction,
            "source": self.source,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "embedding": list(self.embedding),
        }


@dataclass
class KnowledgeGraphSnapshot:
    """Portable serialisation of the graph."""

    created_at: float
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at,
            "entities": [entity.as_dict() for entity in self.entities],
            "relations": [relation.as_dict() for relation in self.relations],
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Knowledge graph implementation
# ---------------------------------------------------------------------------
class KnowledgeGraph:
    """Persistent, thread-safe knowledge graph with lightweight embeddings."""

    EMBEDDING_DIM = 48

    def __init__(self, db_path: str) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path.as_posix(), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._bootstrap()

    # ------------------------------------------------------------------
    def _bootstrap(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS entity_types (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    schema TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS relation_types (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    domain TEXT,
                    range TEXT,
                    metadata TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    name TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    metadata TEXT,
                    first_seen REAL NOT NULL,
                    last_updated REAL NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    support INTEGER DEFAULT 0,
                    contradiction INTEGER DEFAULT 0,
                    embedding TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS triples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    source TEXT,
                    timestamp REAL NOT NULL,
                    support INTEGER DEFAULT 0,
                    contradiction INTEGER DEFAULT 0,
                    metadata TEXT,
                    embedding TEXT,
                    UNIQUE(subject, predicate, object)
                )
                """
            )
            # Ensure new columns exist when upgrading from older schema
            self._ensure_column(cur, "entities", "confidence", "REAL DEFAULT 0.5")
            self._ensure_column(cur, "entities", "support", "INTEGER DEFAULT 0")
            self._ensure_column(cur, "entities", "contradiction", "INTEGER DEFAULT 0")
            self._ensure_column(cur, "entities", "embedding", "TEXT")
            self._ensure_column(cur, "triples", "support", "INTEGER DEFAULT 0")
            self._ensure_column(cur, "triples", "contradiction", "INTEGER DEFAULT 0")
            self._ensure_column(cur, "triples", "metadata", "TEXT")
            self._ensure_column(cur, "triples", "embedding", "TEXT")
            self._conn.commit()

    @staticmethod
    def _ensure_column(cursor: sqlite3.Cursor, table: str, column: str, definition: str) -> None:
        cursor.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cursor.fetchall()}
        if column not in existing:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    # ------------------------------------------------------------------
    # Entity type & relation type management
    # ------------------------------------------------------------------
    def register_entity_type(self, entity_type: EntityType | str, description: Optional[str] = None, schema: Optional[Dict[str, Any]] = None) -> EntityType:
        if isinstance(entity_type, EntityType):
            payload = entity_type
        else:
            payload = EntityType(name=entity_type, description=description or "", schema=schema or {})
        with self._lock:
            self._conn.execute(
                "INSERT INTO entity_types (name, description, schema) VALUES (?, ?, ?) ON CONFLICT(name) DO UPDATE SET description=excluded.description, schema=excluded.schema",
                (payload.name, payload.description, json.dumps(payload.schema or {})),
            )
            self._conn.commit()
        return payload

    def register_relation_type(
        self,
        relation_type: RelationType | str,
        *,
        description: Optional[str] = None,
        domain: Optional[str] = None,
        range: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RelationType:
        if isinstance(relation_type, RelationType):
            payload = relation_type
        else:
            payload = RelationType(
                name=relation_type,
                description=description or "",
                domain=domain,
                range=range,
                metadata=metadata or {},
            )
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO relation_types (name, description, domain, range, metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description = excluded.description,
                    domain = excluded.domain,
                    range = excluded.range,
                    metadata = excluded.metadata
                """,
                (payload.name, payload.description, payload.domain, payload.range, json.dumps(payload.metadata or {})),
            )
            self._conn.commit()
        return payload

    def _auto_register_entity_type(self, name: str) -> None:
        with self._lock:
            cur = self._conn.execute("SELECT 1 FROM entity_types WHERE name = ?", (name,))
            if cur.fetchone() is None:
                self._conn.execute("INSERT INTO entity_types (name, description, schema) VALUES (?, '', '{}')", (name,))
                self._conn.commit()

    # ------------------------------------------------------------------
    # Entity operations
    # ------------------------------------------------------------------
    def add_entity(
        self,
        *,
        name: str,
        entity_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
        embedding: Optional[Sequence[float]] = None,
    ) -> Entity:
        if not name:
            raise ValueError("Entity name is required")
        entity_type = entity_type or "generic"
        metadata = metadata or {}
        now = time.time()

        embedding_vec = self._normalise_embedding(embedding) if embedding is not None else self._embed_entity(name, metadata)
        serialized_meta = json.dumps(metadata, ensure_ascii=True, sort_keys=True)

        with self._lock:
            self._auto_register_entity_type(entity_type)
            cur = self._conn.execute("SELECT * FROM entities WHERE name = ?", (name,))
            row = cur.fetchone()
            if row is None:
                support = 1
                contradiction = 0
                derived_conf = confidence if confidence is not None else self._confidence_from_counts(support, contradiction)
                self._conn.execute(
                    """
                    INSERT INTO entities (name, entity_type, metadata, first_seen, last_updated, confidence, support, contradiction, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        name,
                        entity_type,
                        serialized_meta,
                        now,
                        now,
                        derived_conf,
                        support,
                        contradiction,
                        self._serialize_embedding(embedding_vec),
                    ),
                )
            else:
                support = int(row["support"] or 0) + 1
                contradiction = int(row["contradiction"] or 0)
                derived_conf = confidence if confidence is not None else self._confidence_from_counts(support, contradiction)
                self._conn.execute(
                    """
                    UPDATE entities
                    SET entity_type = ?,
                        metadata = ?,
                        last_updated = ?,
                        confidence = ?,
                        support = ?,
                        contradiction = ?,
                        embedding = ?
                    WHERE name = ?
                    """,
                    (
                        entity_type,
                        serialized_meta,
                        now,
                        derived_conf,
                        support,
                        contradiction,
                        self._serialize_embedding(embedding_vec),
                        name,
                    ),
                )
            self._conn.commit()
        return self.get_entity(name)

    def get_entity(self, name: str) -> Optional[Entity]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM entities WHERE name = ?", (name,))
            row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_entity(row)

    def record_entity_evidence(self, name: str, observed: bool) -> Optional[Entity]:
        with self._lock:
            cur = self._conn.execute("SELECT support, contradiction FROM entities WHERE name = ?", (name,))
            row = cur.fetchone()
            if row is None:
                return None
            support = int(row["support"] or 0) + (1 if observed else 0)
            contradiction = int(row["contradiction"] or 0) + (0 if observed else 1)
            confidence = self._confidence_from_counts(support, contradiction)
            self._conn.execute(
                "UPDATE entities SET support = ?, contradiction = ?, confidence = ?, last_updated = ? WHERE name = ?",
                (support, contradiction, confidence, time.time(), name),
            )
            self._conn.commit()
        return self.get_entity(name)

    def all_entities(self) -> List[Entity]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM entities")
            rows = cur.fetchall()
        return [self._row_to_entity(row) for row in rows]

    # ------------------------------------------------------------------
    # Relation operations
    # ------------------------------------------------------------------
    def add_relation(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        confidence: Optional[float] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        observed: bool = True,
    ) -> Relation:
        if not subject or not predicate or not obj:
            raise ValueError("subject, predicate and object are required")
        metadata = metadata or {}
        now = time.time()
        embedding_vec = self._embed_relation(subject, predicate, obj, metadata)
        serialized_meta = json.dumps(metadata, ensure_ascii=True, sort_keys=True)

        with self._lock:
            self.add_entity(name=subject, entity_type="entity")
            self.add_entity(name=obj, entity_type="entity")
            cur = self._conn.execute(
                "SELECT id, support, contradiction FROM triples WHERE subject = ? AND predicate = ? AND object = ?",
                (subject, predicate, obj),
            )
            row = cur.fetchone()
            if row is None:
                support = 1 if observed else 0
                contradiction = 0 if observed else 1
                derived_conf = confidence if confidence is not None else self._confidence_from_counts(support, contradiction)
                self._conn.execute(
                    """
                    INSERT INTO triples (subject, predicate, object, confidence, source, timestamp, support, contradiction, metadata, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        subject,
                        predicate,
                        obj,
                        derived_conf,
                        source,
                        now,
                        support,
                        contradiction,
                        serialized_meta,
                        self._serialize_embedding(embedding_vec),
                    ),
                )
            else:
                support = int(row["support"] or 0) + (1 if observed else 0)
                contradiction = int(row["contradiction"] or 0) + (0 if observed else 1)
                derived_conf = confidence if confidence is not None else self._confidence_from_counts(support, contradiction)
                self._conn.execute(
                    """
                    UPDATE triples
                    SET confidence = ?,
                        source = COALESCE(?, source),
                        timestamp = ?,
                        support = ?,
                        contradiction = ?,
                        metadata = ?,
                        embedding = ?
                    WHERE subject = ? AND predicate = ? AND object = ?
                    """,
                    (
                        derived_conf,
                        source,
                        now,
                        support,
                        contradiction,
                        serialized_meta,
                        self._serialize_embedding(embedding_vec),
                        subject,
                        predicate,
                        obj,
                    ),
                )
            self._conn.commit()

        return self.get_relation(subject=subject, predicate=predicate, obj=obj)

    def add_triple(
        self,
        *,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: Optional[str] = None,
    ) -> None:
        self.add_relation(subject, predicate, obj, confidence=confidence, source=source, observed=True)

    def get_relation(
        self,
        relation_id: Optional[int] = None,
        *,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> Optional[Relation]:
        query = "SELECT * FROM triples WHERE "
        params: List[Any] = []
        if relation_id is not None:
            query += "id = ?"
            params.append(relation_id)
        else:
            clauses: List[str] = []
            if subject is not None:
                clauses.append("subject = ?")
                params.append(subject)
            if predicate is not None:
                clauses.append("predicate = ?")
                params.append(predicate)
            if obj is not None:
                clauses.append("object = ?")
                params.append(obj)
            if not clauses:
                raise ValueError("relation lookup requires id or at least one filter")
            query += " AND ".join(clauses)
        with self._lock:
            cur = self._conn.execute(query, params)
            row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_relation(row)

    def update_relation_confidence(self, relation_id: int, observed: bool) -> Optional[Relation]:
        with self._lock:
            cur = self._conn.execute("SELECT support, contradiction FROM triples WHERE id = ?", (relation_id,))
            row = cur.fetchone()
            if row is None:
                return None
            support = int(row["support"] or 0) + (1 if observed else 0)
            contradiction = int(row["contradiction"] or 0) + (0 if observed else 1)
            confidence = self._confidence_from_counts(support, contradiction)
            self._conn.execute(
                "UPDATE triples SET support = ?, contradiction = ?, confidence = ?, timestamp = ? WHERE id = ?",
                (support, contradiction, confidence, time.time(), relation_id),
            )
            self._conn.commit()
        return self.get_relation(relation_id)

    def record_relation_evidence(self, subject: str, predicate: str, obj: str, observed: bool) -> Optional[Relation]:
        rel = self.get_relation(subject=subject, predicate=predicate, obj=obj)
        if rel is None:
            return None
        return self.update_relation_confidence(rel.relation_id, observed)

    def query(
        self,
        *,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        limit: int = 100,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = ["confidence >= ?"]
        params: List[Any] = [min_confidence]
        if subject:
            clauses.append("subject = ?")
            params.append(subject)
        if predicate:
            clauses.append("predicate = ?")
            params.append(predicate)
        if obj:
            clauses.append("object = ?")
            params.append(obj)
        where = " AND ".join(clauses)
        sql = f"SELECT * FROM triples WHERE {where} ORDER BY confidence DESC, timestamp DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
        return [self._row_to_relation(row).as_dict() for row in rows]

    def get_related(self, entity: str, *, max_hops: int = 1, min_confidence: float = 0.0) -> List[Tuple[str, str, str]]:
        visited: set[str] = set()
        frontier: set[str] = {entity}
        results: set[Tuple[str, str, str]] = set()
        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                for relation in self.query(subject=node, min_confidence=min_confidence, limit=200):
                    triple = (relation["subject"], relation["predicate"], relation["object"])
                    results.add(triple)
                    next_frontier.add(relation["object"])
                for relation in self.query(obj=node, min_confidence=min_confidence, limit=200):
                    triple = (relation["subject"], relation["predicate"], relation["object"])
                    results.add(triple)
                    next_frontier.add(relation["subject"])
            frontier = next_frontier
            if not frontier:
                break
        return list(results)

    def to_adjacency(self, *, weight: str = "confidence", min_confidence: float = 0.0) -> Dict[str, Dict[str, float]]:
        graph: Dict[str, Dict[str, float]] = {}
        with self._lock:
            cur = self._conn.execute("SELECT subject, object, confidence FROM triples WHERE confidence >= ?", (min_confidence,))
            rows = cur.fetchall()
        for subject, obj, conf in rows:
            graph.setdefault(subject, {})[obj] = float(conf if weight == "confidence" else 1.0)
        return graph

    def to_networkx(self, *, directed: bool = True, min_confidence: float = 0.0):  # pragma: no cover - optional dependency wrapper
        try:
            import networkx as nx  # type: ignore
        except ImportError as exc:  # pragma: no cover - executed only if dependency missing
            raise RuntimeError("networkx is required for graph export") from exc

        graph = nx.DiGraph() if directed else nx.Graph()
        for entity in self.all_entities():
            graph.add_node(entity.name, **entity.as_dict())
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM triples WHERE confidence >= ?",
                (min_confidence,),
            )
            rows = cur.fetchall()
        for row in rows:
            relation = self._row_to_relation(row)
            graph.add_edge(
                relation.subject,
                relation.obj,
                predicate=relation.predicate,
                confidence=relation.confidence,
                support=relation.support,
                contradiction=relation.contradiction,
                metadata=relation.metadata,
            )
        return graph

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def find_similar_entities(self, name: str, *, top_k: int = 5, include_self: bool = False) -> List[Tuple[Entity, float]]:
        target = self.get_entity(name)
        if target is None:
            return []
        candidates = []
        for entity in self.all_entities():
            if not include_self and entity.name == name:
                continue
            similarity = self._cosine_similarity(target.embedding, entity.embedding)
            candidates.append((entity, similarity))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:top_k]

    def find_similar_relations(
        self,
        relation_id: int,
        *,
        top_k: int = 5,
    ) -> List[Tuple[Relation, float]]:
        target = self.get_relation(relation_id)
        if target is None:
            return []
        with self._lock:
            cur = self._conn.execute("SELECT * FROM triples WHERE id != ?", (relation_id,))
            rows = cur.fetchall()
        candidates: List[Tuple[Relation, float]] = []
        for row in rows:
            relation = self._row_to_relation(row)
            similarity = self._cosine_similarity(target.embedding, relation.embedding)
            candidates.append((relation, similarity))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Snapshotting
    # ------------------------------------------------------------------
    def snapshot(self, *, metadata: Optional[Dict[str, Any]] = None) -> KnowledgeGraphSnapshot:
        entities = self.all_entities()
        with self._lock:
            cur = self._conn.execute("SELECT * FROM triples")
            rows = cur.fetchall()
        relations = [self._row_to_relation(row) for row in rows]
        return KnowledgeGraphSnapshot(
            created_at=time.time(),
            entities=entities,
            relations=relations,
            metadata=metadata or {},
        )

    def merge_snapshot(self, snapshot: KnowledgeGraphSnapshot) -> None:
        for entity in snapshot.entities:
            self.add_entity(
                name=entity.name,
                entity_type=entity.entity_type,
                metadata=entity.metadata,
                confidence=entity.confidence,
                embedding=entity.embedding,
            )
        for relation in snapshot.relations:
            self.add_relation(
                relation.subject,
                relation.predicate,
                relation.obj,
                confidence=relation.confidence,
                source=relation.source,
                metadata=relation.metadata,
                observed=True,
            )

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return self.to_adjacency()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        metadata = json.loads(row["metadata"] or "{}")
        embedding = self._deserialize_embedding(row["embedding"])
        return Entity(
            name=row["name"],
            entity_type=row["entity_type"],
            metadata=metadata,
            first_seen=float(row["first_seen"] or 0.0),
            last_updated=float(row["last_updated"] or 0.0),
            confidence=float(row["confidence"] or 0.5),
            support=int(row["support"] or 0),
            contradiction=int(row["contradiction"] or 0),
            embedding=embedding,
        )

    def _row_to_relation(self, row: sqlite3.Row) -> Relation:
        metadata = json.loads(row["metadata"] or "{}")
        embedding = self._deserialize_embedding(row["embedding"])
        return Relation(
            relation_id=int(row["id"]),
            subject=row["subject"],
            predicate=row["predicate"],
            obj=row["object"],
            confidence=float(row["confidence"] or 0.5),
            support=int(row["support"] or 0),
            contradiction=int(row["contradiction"] or 0),
            source=row["source"],
            timestamp=float(row["timestamp"] or 0.0),
            metadata=metadata,
            embedding=embedding,
        )

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _confidence_from_counts(support: int, contradiction: int) -> float:
        return (support + 1.0) / (support + contradiction + 2.0)

    def _embed_entity(self, name: str, metadata: Dict[str, Any]) -> Tuple[float, ...]:
        payload = name + "::" + json.dumps(metadata, sort_keys=True)
        return self._hash_to_unit_vector(payload)

    def _embed_relation(self, subject: str, predicate: str, obj: str, metadata: Dict[str, Any]) -> Tuple[float, ...]:
        payload = "|".join([subject, predicate, obj, json.dumps(metadata, sort_keys=True)])
        return self._hash_to_unit_vector(payload)

    def _hash_to_unit_vector(self, text: str) -> Tuple[float, ...]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: List[float] = []
        for index in range(self.EMBEDDING_DIM):
            byte = digest[index % len(digest)]
            values.append((byte / 255.0) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return tuple(value / norm for value in values)

    def _normalise_embedding(self, embedding: Sequence[float]) -> Tuple[float, ...]:
        norm = math.sqrt(sum(float(value) * float(value) for value in embedding)) or 1.0
        return tuple(float(value) / norm for value in embedding)

    @staticmethod
    def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        numerator = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
        denom_a = math.sqrt(sum(float(a) * float(a) for a in vec_a)) or 1.0
        denom_b = math.sqrt(sum(float(b) * float(b) for b in vec_b)) or 1.0
        return max(min(numerator / (denom_a * denom_b), 1.0), -1.0)

    def _serialize_embedding(self, embedding: Sequence[float]) -> str:
        return json.dumps([float(value) for value in embedding], ensure_ascii=True)

    def _deserialize_embedding(self, payload: Optional[str]) -> Tuple[float, ...]:
        if not payload:
            return tuple()
        try:
            data = json.loads(payload)
            if isinstance(data, list):
                return tuple(float(value) for value in data)
        except Exception:  # pragma: no cover - fallback on corruption
            logger.warning("Failed to decode embedding payload")
        return tuple()

