
"""Context curation service combining episodic memory, knowledge graph and patterns."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .knowledge_graph import KnowledgeGraph
from .memory import EpisodicMemory
from .patterns import PatternStore

logger = logging.getLogger(__name__)


@dataclass
class ContextEntry:
    entry_id: str
    content: str
    relevance: float
    source: str
    timestamp: datetime
    metadata: Dict[str, Any]
    tokens: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.entry_id,
            "content": self.content,
            "relevance": self.relevance,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "tokens": self.tokens,
        }


class ContextualMemory:
    """Produces tailored context bundles for an agent or LLM call."""

    def __init__(
        self,
        episodic: EpisodicMemory,
        knowledge: KnowledgeGraph,
        patterns: PatternStore,
        *,
        max_tokens: int = 4000,
        max_entries: int = 50,
    ) -> None:
        self.episodic = episodic
        self.knowledge = knowledge
        self.patterns = patterns
        self.max_tokens = max_tokens
        self.max_entries = max_entries
        self._last_context: List[ContextEntry] = []

    def build_context(self, query: str, *, agent: Optional[str] = None) -> List[ContextEntry]:
        entries: List[ContextEntry] = []
        for episode in self.episodic.search_semantic(query, limit=20):
            content = episode.get("content", "")
            timestamp = datetime.fromtimestamp(episode.get("timestamp", 0.0))
            entry = ContextEntry(
                entry_id=f"episode:{episode.get('timestamp', 0)}",
                content=f"[{episode.get('role', 'unknown')}]: {content}",
                relevance=1.0 - float(episode.get("distance", 0.0)) if "distance" in episode else episode.get("importance", 0.5),
                source="episodic",
                timestamp=timestamp,
                metadata={"agent": episode.get("agent"), **episode.get("metadata", {})},
                tokens=max(1, len(content) // 4),
            )
            entries.append(entry)

        entities = self._extract_entities(query)
        for entity in entities[:5]:
            for fact in self.knowledge.query(subject=entity, limit=5):
                content = f"{fact['subject']} {fact['predicate']} {fact['object']}"
                entry = ContextEntry(
                    entry_id=f"fact:{hash((fact['subject'], fact['predicate'], fact['object']))}",
                    content=f"FACT: {content}",
                    relevance=float(fact.get("confidence", 0.5)),
                    source="knowledge",
                    timestamp=datetime.fromtimestamp(fact.get("timestamp", datetime.utcnow().timestamp())),
                    metadata={"source": fact.get("source")},
                    tokens=max(1, len(content) // 5),
                )
                entries.append(entry)

        for pattern in self.patterns.top_patterns(min_confidence=0.6, limit=10):
            description = pattern.description or pattern.pattern_name
            entry = ContextEntry(
                entry_id=f"pattern:{pattern.pattern_id}",
                content=f"PATTERN ({pattern.pattern_type}): {description}",
                relevance=pattern.confidence,
                source="pattern",
                timestamp=datetime.fromtimestamp(pattern.last_seen),
                metadata={"occurrences": pattern.occurrence_count},
                tokens=max(1, len(description) // 5),
            )
            entries.append(entry)

        entries.sort(key=lambda entry: entry.relevance, reverse=True)
        curated = self._prune(entries)
        self._last_context = curated
        logger.debug("Contextual memory curated %d entries", len(curated))
        return curated

    def format_context(self) -> str:
        if not self._last_context:
            return ""
        sections: Dict[str, List[str]] = {"knowledge": [], "pattern": [], "episodic": []}
        for entry in self._last_context:
            sections.setdefault(entry.source, []).append(entry.content)
        parts: List[str] = []
        if sections["knowledge"]:
            parts.append("=== Facts ===")
            parts.extend(sections["knowledge"])
        if sections["pattern"]:
            parts.append("\n=== Patterns ===")
            parts.extend(sections["pattern"])
        if sections["episodic"]:
            parts.append("\n=== Relevant history ===")
            parts.extend(sections["episodic"])
        return "\n".join(parts).strip()

    def summary(self) -> Dict[str, Any]:
        if not self._last_context:
            return {"entries": 0, "tokens": 0}
        return {
            "entries": len(self._last_context),
            "tokens": sum(entry.tokens for entry in self._last_context),
            "sources": {
                entry.source: sum(1 for candidate in self._last_context if candidate.source == entry.source)
                for entry in self._last_context
            },
        }

    def _prune(self, entries: List[ContextEntry]) -> List[ContextEntry]:
        pruned: List[ContextEntry] = []
        token_budget = 0
        for entry in entries:
            if len(pruned) >= self.max_entries:
                break
            if token_budget + entry.tokens > self.max_tokens:
                break
            pruned.append(entry)
            token_budget += entry.tokens
        return pruned

    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        tokens = text.split()
        entities: List[str] = []
        for index, token in enumerate(tokens):
            word = token.strip('.,!?"')
            if len(word) < 2:
                continue
            if not word[0].isupper():
                continue
            if index + 1 < len(tokens) and tokens[index + 1][0].isupper():
                next_word = tokens[index + 1].strip('.,!?"')
                entities.append(f"{word} {next_word}")
            else:
                entities.append(word)
        seen: set[str] = set()
        deduped: List[str] = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                deduped.append(entity)
        return deduped
