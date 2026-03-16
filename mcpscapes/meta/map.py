import numpy as np

from mcpscapes.shared.models import MapEdge, RouteResult, ServerRegistration


class TopographicMap:
    def __init__(self) -> None:
        self._distances: dict[tuple[str, str], float] = {}

    def rebuild(self, servers: list[ServerRegistration]) -> None:
        """Compute all pairwise cosine distances between server centroids."""
        self._distances = {}
        with_centroids = [s for s in servers if s.centroid is not None]
        for i, a in enumerate(with_centroids):
            for b in with_centroids[i + 1 :]:
                va = np.array(a.centroid, dtype=np.float32)
                vb = np.array(b.centroid, dtype=np.float32)
                denom = np.linalg.norm(va) * np.linalg.norm(vb)
                similarity = float(np.dot(va, vb) / denom) if denom else 0.0
                distance = 1.0 - similarity
                self._distances[(a.id, b.id)] = distance
                self._distances[(b.id, a.id)] = distance

    def nearest_servers(
        self,
        query_embedding: list[float],
        servers: list[ServerRegistration],
        top_k: int = 3,
    ) -> list[RouteResult]:
        """Return top_k servers sorted by descending cosine similarity to query."""
        q = np.array(query_embedding, dtype=np.float32)
        scored: list[RouteResult] = []
        for srv in servers:
            if srv.centroid is None:
                score = 0.0
            else:
                c = np.array(srv.centroid, dtype=np.float32)
                denom = np.linalg.norm(q) * np.linalg.norm(c)
                score = float(np.dot(q, c) / denom) if denom else 0.0
            scored.append(
                RouteResult(
                    server_id=srv.id,
                    score=score,
                    connection_info={"url": srv.url, "id": srv.id, "name": srv.name},
                )
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]
