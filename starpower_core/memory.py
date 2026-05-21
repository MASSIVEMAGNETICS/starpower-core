from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx


@dataclass(slots=True)
class MemoryRecord:
    id: str
    text: str
    kind: str = "observation"
    tags: list[str] = field(default_factory=list)
    weight: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    checksum: str = ""

    def seal(self) -> "MemoryRecord":
        payload = {
            "id": self.id,
            "text": self.text,
            "kind": self.kind,
            "tags": self.tags,
            "weight": self.weight,
            "created_at": self.created_at,
        }
        self.checksum = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return self

    def verify(self) -> bool:
        expected = MemoryRecord(
            id=self.id,
            text=self.text,
            kind=self.kind,
            tags=list(self.tags),
            weight=self.weight,
            created_at=self.created_at,
        ).seal().checksum
        return self.checksum == expected


@dataclass(slots=True)
class RemMemoryGraph:
    """Checksum-protected semantic-episodic memory graph."""

    records: dict[str, MemoryRecord] = field(default_factory=dict)
    graph: nx.Graph = field(default_factory=nx.Graph)

    def remember(self, text: str, kind: str = "observation", tags: list[str] | None = None, weight: float = 1.0) -> MemoryRecord:
        if not text.strip():
            raise ValueError("memory text must be non-empty")
        tags = tags or []
        record_id = hashlib.sha256(f"{kind}|{text}|{tags}".encode()).hexdigest()[:16]
        record = MemoryRecord(id=record_id, text=text, kind=kind, tags=tags, weight=weight).seal()
        self.records[record.id] = record
        self.graph.add_node(record.id, kind=record.kind, tags=record.tags, weight=record.weight)
        for tag in tags:
            tag_node = f"tag:{tag}"
            self.graph.add_node(tag_node, kind="tag")
            self.graph.add_edge(record.id, tag_node, relation="tagged")
        return record

    def relate(self, source_id: str, target_id: str, relation: str = "related") -> None:
        if source_id not in self.records:
            raise KeyError(f"unknown source memory: {source_id}")
        if target_id not in self.records:
            raise KeyError(f"unknown target memory: {target_id}")
        self.graph.add_edge(source_id, target_id, relation=relation)

    def recall(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        terms = {token.lower() for token in query.split() if token.strip()}
        scored: list[tuple[float, MemoryRecord]] = []
        for record in self.records.values():
            haystack = f"{record.text} {' '.join(record.tags)} {record.kind}".lower()
            score = sum(1 for term in terms if term in haystack) + record.weight * 0.05
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:limit]]

    def audit(self) -> dict[str, Any]:
        corrupted = [record.id for record in self.records.values() if not record.verify()]
        dangling = [node for node in self.graph.nodes if not str(node).startswith("tag:") and node not in self.records]
        return {
            "records": len(self.records),
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "corrupted": corrupted,
            "dangling_nodes": dangling,
            "healthy": not corrupted and not dangling,
        }

    def self_heal(self) -> dict[str, Any]:
        audit = self.audit()
        for record_id in audit["corrupted"]:
            self.records[record_id].seal()
        for node in audit["dangling_nodes"]:
            self.graph.remove_node(node)
        return self.audit()

    def save_json(self, path: str | Path) -> None:
        payload = [asdict(record) for record in self.records.values()]
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "RemMemoryGraph":
        instance = cls()
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for item in raw:
            record = MemoryRecord(**item)
            instance.records[record.id] = record
            instance.graph.add_node(record.id, kind=record.kind, tags=record.tags, weight=record.weight)
            for tag in record.tags:
                tag_node = f"tag:{tag}"
                instance.graph.add_node(tag_node, kind="tag")
                instance.graph.add_edge(record.id, tag_node, relation="tagged")
        instance.self_heal()
        return instance
