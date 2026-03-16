import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from mcpscapes.shared.models import ServerRegistration

_CREATE_SERVERS = """
CREATE TABLE IF NOT EXISTS servers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    url TEXT NOT NULL,
    centroid BLOB,
    registered_at TEXT NOT NULL
);
"""


class Registry:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_CREATE_SERVERS)
        self._conn.commit()

    def _row_to_reg(self, row: tuple) -> ServerRegistration:
        id_, name, desc, url, centroid_blob, registered_at = row
        centroid = (
            np.frombuffer(centroid_blob, dtype=np.float32).tolist()
            if centroid_blob is not None
            else None
        )
        return ServerRegistration(
            id=id_,
            name=name,
            description=desc,
            url=url,
            centroid=centroid,
            registered_at=datetime.fromisoformat(registered_at),
        )

    def get(self, id: str) -> ServerRegistration | None:
        row = self._conn.execute(
            "SELECT id, name, description, url, centroid, registered_at FROM servers WHERE id = ?",
            (id,),
        ).fetchone()
        return self._row_to_reg(row) if row else None

    def list_all(self) -> list[ServerRegistration]:
        rows = self._conn.execute(
            "SELECT id, name, description, url, centroid, registered_at FROM servers"
        ).fetchall()
        return [self._row_to_reg(r) for r in rows]

    def register(self, reg: ServerRegistration) -> None:
        centroid_blob = (
            np.array(reg.centroid, dtype=np.float32).tobytes()
            if reg.centroid is not None
            else None
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO servers (id, name, description, url, centroid, registered_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                reg.id,
                reg.name,
                reg.description,
                reg.url,
                centroid_blob,
                reg.registered_at.isoformat(),
            ),
        )
        self._conn.commit()
