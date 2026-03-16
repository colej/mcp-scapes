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
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_nodes USING vec0(embedding float[{dim}])"
        )
        self._conn.commit()
