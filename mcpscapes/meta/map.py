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

    def soft_weights(
        self,
        query_embedding: list[float],
        servers: list[ServerRegistration],
        temperature: float = 0.5,
    ) -> dict[str, float]:
        """Softmax over cosine similarities with temperature scaling.

        Low temperature → near-hard routing (winner-takes-most).
        High temperature → diffuse activation across all servers.
        """
        if not servers:
            return {}
        q = np.array(query_embedding, dtype=np.float32)
        similarities = []
        ids = []
        for srv in servers:
            if srv.centroid is None:
                sim = 0.0
            else:
                c = np.array(srv.centroid, dtype=np.float32)
                denom = np.linalg.norm(q) * np.linalg.norm(c)
                sim = float(np.dot(q, c) / denom) if denom else 0.0
            similarities.append(sim / temperature)
            ids.append(srv.id)
        sims = np.array(similarities, dtype=np.float64)
        sims -= sims.max()  # numerical stability
        exp_sims = np.exp(sims)
        weights = exp_sims / exp_sims.sum()
        return {id_: float(w) for id_, w in zip(ids, weights)}

    def server_distances(self) -> list[MapEdge]:
        """All pairwise edges sorted by ascending distance."""
        seen: set[frozenset[str]] = set()
        edges: list[MapEdge] = []
        for (src, tgt), dist in self._distances.items():
            key = frozenset([src, tgt])
            if key not in seen:
                seen.add(key)
                edges.append(MapEdge(source=src, target=tgt, distance=dist))
        edges.sort(key=lambda e: e.distance)
        return edges

    def interpolate(
        self,
        server_id_a: str,
        server_id_b: str,
        t: float,
        servers: list[ServerRegistration],
    ) -> list[float]:
        """Linear interpolation between two server centroids at parameter t in [0,1]."""
        srv_map = {s.id: s for s in servers}
        a = srv_map.get(server_id_a)
        b = srv_map.get(server_id_b)
        if a is None or a.centroid is None or b is None or b.centroid is None:
            raise ValueError(f"Cannot interpolate: missing centroid for {server_id_a!r} or {server_id_b!r}")
        va = np.array(a.centroid, dtype=np.float32)
        vb = np.array(b.centroid, dtype=np.float32)
        return ((1.0 - t) * va + t * vb).tolist()
