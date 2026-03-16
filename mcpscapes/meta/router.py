from mcpscapes.meta.map import TopographicMap
from mcpscapes.meta.registry import Registry
from mcpscapes.shared.embedder import get_embedder
from mcpscapes.shared.models import RouteResult


class Router:
    def __init__(self, registry: Registry, topo_map: TopographicMap) -> None:
        self._registry = registry
        self._map = topo_map

    def route(self, query: str, top_k: int = 3, temperature: float = 0.5) -> list[RouteResult]:
        """Embed query and return nearest servers by cosine similarity."""
        embedding = get_embedder().embed(query)
        servers = self._registry.list_all()
        return self._map.nearest_servers(embedding, servers, top_k)

    def soft_route(self, query: str, temperature: float = 0.5) -> dict[str, float]:
        """Embed query and return softmax weight distribution over all servers."""
        embedding = get_embedder().embed(query)
        servers = self._registry.list_all()
        return self._map.soft_weights(embedding, servers, temperature)
