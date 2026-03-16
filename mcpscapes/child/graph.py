import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from mcpscapes.shared.embedder import get_embedder
from mcpscapes.shared.models import MemoryNode

_CREATE_NODES = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    domain_weights TEXT NOT NULL,
    embedding BLOB,
    tags TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

_CREATE_EDGES = """
CREATE TABLE IF NOT EXISTS edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    PRIMARY KEY (source_id, target_id, relation),
    FOREIGN KEY (source_id) REFERENCES nodes(id),
    FOREIGN KEY (target_id) REFERENCES nodes(id)
);
"""


class KnowledgeGraph:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_CREATE_NODES)
        self._conn.execute(_CREATE_EDGES)
        self._conn.commit()

    def add_node(
        self,
        content: str,
        tags: list[str],
        domain_weights: dict[str, float],
    ) -> MemoryNode:
        embedder = get_embedder()
        embedding = embedder.embed(content)
        now = datetime.utcnow().isoformat()
        node = MemoryNode(
            id=str(uuid.uuid4()),
            content=content,
            domain_weights=domain_weights,
            embedding=embedding,
            tags=tags,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        emb_blob = np.array(embedding, dtype=np.float32).tobytes()
        self._conn.execute(
            "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                node.id,
                node.content,
                json.dumps(node.domain_weights),
                emb_blob,
                json.dumps(node.tags),
                now,
                now,
            ),
        )
        self._conn.commit()
        return node

    def get_node(self, id: str) -> MemoryNode | None:
        row = self._conn.execute(
            "SELECT id, content, domain_weights, embedding, tags, created_at, updated_at FROM nodes WHERE id = ?",
            (id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO edges (source_id, target_id, relation, weight) VALUES (?, ?, ?, ?)",
            (source_id, target_id, relation, weight),
        )
        self._conn.commit()

    def search(self, query: str, k: int = 5) -> list[tuple["MemoryNode", float]]:
        embedder = get_embedder()
        q_vec = np.array(embedder.embed(query), dtype=np.float32)
        rows = self._conn.execute(
            "SELECT id, content, domain_weights, embedding, tags, created_at, updated_at FROM nodes WHERE embedding IS NOT NULL"
        ).fetchall()
        scored: list[tuple[MemoryNode, float]] = []
        for row in rows:
            node = self._row_to_node(row)
            if node.embedding is None:
                continue
            n_vec = np.array(node.embedding, dtype=np.float32)
            denom = np.linalg.norm(q_vec) * np.linalg.norm(n_vec)
            score = float(np.dot(q_vec, n_vec) / denom) if denom else 0.0
            scored.append((node, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
