import os

import uvicorn
from fastmcp import FastMCP

from mcpscapes.meta.map import TopographicMap
from mcpscapes.meta.registry import Registry
from mcpscapes.meta.router import Router

META_PORT = int(os.environ.get("META_PORT", "8000"))
META_DB_PATH = os.environ.get("META_DB_PATH", "./data/meta.db")

_registry = Registry(META_DB_PATH)
_topo_map = TopographicMap()
_router = Router(_registry, _topo_map)

mcp = FastMCP("mcp-scapes-meta")


@mcp.tool()
async def list_domains() -> list[dict]:
    """Return all registered servers with their nearest 3 neighbours."""
    servers = _registry.list_all()
    srv_map = {s.id: s for s in servers}
    result = []
    for srv in servers:
        if srv.centroid:
            neighbours = _topo_map.nearest_servers(srv.centroid, servers, top_k=4)
            neighbours = [n for n in neighbours if n.server_id != srv.id][:3]
            nb_info = [{"server_id": n.server_id, "distance": round(1.0 - n.score, 4)} for n in neighbours]
        else:
            nb_info = []
        result.append({
            "id": srv.id,
            "name": srv.name,
            "description": srv.description,
            "nearest_neighbours": nb_info,
        })
    return result
