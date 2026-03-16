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
