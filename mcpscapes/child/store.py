import sqlite3
from pathlib import Path

import numpy as np
import sqlite_vec


class VectorStore:
    """Performance layer wrapping sqlite-vec for ANN search.

    KnowledgeGraph defaults to numpy cosine search in graph.py.
    # TODO: swap graph.py search() to delegate here by setting
    # KnowledgeGraph(..., use_vec_store=True) once benchmarked.
    """

    def __init__(self, db_path: str, dim: int = 384) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._dim = dim
        # Integer rowid → string node_id mapping
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vec_id_map "
            "(rowid INTEGER PRIMARY KEY AUTOINCREMENT, node_id TEXT NOT NULL UNIQUE)"
        )
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_nodes "
            f"USING vec0(embedding float[{dim}])"
        )
        self._conn.commit()

    def add(self, id: str, embedding: list[float]) -> None:
        # Upsert id mapping
        self._conn.execute(
            "INSERT OR IGNORE INTO vec_id_map (node_id) VALUES (?)", (id,)
        )
        row = self._conn.execute(
            "SELECT rowid FROM vec_id_map WHERE node_id = ?", (id,)
        ).fetchone()
        rowid = row[0]
        vec = np.array(embedding, dtype=np.float32).tobytes()
        self._conn.execute(
            "INSERT OR REPLACE INTO vec_nodes(rowid, embedding) VALUES (?, ?)",
            (rowid, vec),
        )
        self._conn.commit()

    def search(self, query_embedding: list[float], k: int) -> list[tuple[str, float]]:
        vec = np.array(query_embedding, dtype=np.float32).tobytes()
        rows = self._conn.execute(
            "SELECT rowid, distance FROM vec_nodes WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (vec, k),
        ).fetchall()
        results = []
        for rowid, distance in rows:
            id_row = self._conn.execute(
                "SELECT node_id FROM vec_id_map WHERE rowid = ?", (rowid,)
            ).fetchone()
            if id_row:
                score = 1.0 - float(distance)  # convert cosine distance → similarity
                results.append((id_row[0], score))
        return results
