import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from pydantic import BaseModel as PydanticBase

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


@mcp.tool()
async def route_query(query: str, top_k: int = 3, temperature: float = 0.5) -> list[dict]:
    """Route a query to the nearest servers and return connection info."""
    results = _router.route(query, top_k, temperature)
    return [r.model_dump() for r in results]


@mcp.tool()
async def soft_route_query(query: str, temperature: float = 0.5) -> dict:
    """Return softmax weight distribution over all domains for a query."""
    return _router.soft_route(query, temperature)


@mcp.tool()
async def domain_map() -> list[dict]:
    """Return all pairwise inter-server distances as an edge list."""
    return [e.model_dump() for e in _topo_map.server_distances()]


@mcp.tool()
async def interpolate_domains(domain_a: str, domain_b: str, t: float = 0.5) -> list[dict]:
    """Interpolate between two domain centroids and route the midpoint vector."""
    servers = _registry.list_all()
    point = _topo_map.interpolate(domain_a, domain_b, t, servers)
    results = _topo_map.nearest_servers(point, servers, top_k=3)
    return [r.model_dump() for r in results]


@mcp.tool()
async def search_across(
    query: str,
    top_k_servers: int = 2,
    top_k_results: int = 5,
) -> list[dict]:
    """Fan out query to top N servers, merge and re-rank results."""
    import httpx

    top_servers = _router.route(query, top_k=top_k_servers)
    all_results: list[dict] = []
    async with httpx.AsyncClient() as client:
        for srv_result in top_servers:
            url = srv_result.connection_info["url"]
            try:
                resp = await client.post(
                    f"{url}/mcp",
                    json={
                        "method": "tools/call",
                        "params": {
                            "name": "search_memory",
                            "arguments": {"query": query, "k": top_k_results},
                        },
                    },
                    timeout=10.0,
                )
                resp.raise_for_status()
                items = resp.json().get("result", []) or []
                for item in items:
                    item["source_server"] = srv_result.server_id
                    all_results.append(item)
            except Exception:
                pass
    all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return all_results[:top_k_results]


@mcp.tool()
async def add_to_domain(domain_id: str, content: str, tags: list[str]) -> dict:
    """Proxy add_memory to a specific child server."""
    import httpx

    srv = _registry.get(domain_id)
    if srv is None:
        raise ValueError(f"Unknown domain: {domain_id!r}")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{srv.url}/mcp",
            json={
                "method": "tools/call",
                "params": {
                    "name": "add_memory",
                    "arguments": {"content": content, "tags": tags},
                },
            },
            timeout=15.0,
        )
        resp.raise_for_status()
        return resp.json().get("result", {})


@mcp.tool()
async def add_with_soft_routing(
    content: str,
    tags: list[str],
    temperature: float = 0.3,
    threshold: float = 0.15,
) -> dict:
    """Embed content, compute soft weights, write to all servers above threshold."""
    import httpx

    weights = _router.soft_route(content, temperature)
    targets = {sid: w for sid, w in weights.items() if w >= threshold}
    written: list[str] = []
    async with httpx.AsyncClient() as client:
        for server_id, weight in targets.items():
            srv = _registry.get(server_id)
            if srv is None:
                continue
            try:
                resp = await client.post(
                    f"{srv.url}/mcp",
                    json={
                        "method": "tools/call",
                        "params": {
                            "name": "add_memory",
                            "arguments": {
                                "content": content,
                                "tags": tags,
                                "domain_weights": {k: v for k, v in weights.items()},
                            },
                        },
                    },
                    timeout=15.0,
                )
                resp.raise_for_status()
                written.append(server_id)
            except Exception:
                pass
    return {"written_to": written, "domain_weights": weights}


# --- Internal HTTP endpoints ---

_internal = FastAPI()


@_internal.post("/internal/register")
async def internal_register(reg: "ServerRegistrationBody"):
    from mcpscapes.shared.models import ServerRegistration
    server_reg = ServerRegistration(**reg.model_dump())
    _registry.register(server_reg)
    # Fire-and-forget centroid fetch
    import asyncio
    asyncio.create_task(_router.refresh_centroid(server_reg.id))
    return {"status": "registered", "id": server_reg.id}


@_internal.post("/internal/refresh_centroid/{server_id}")
async def internal_refresh_centroid(server_id: str):
    import asyncio
    asyncio.create_task(_router.refresh_centroid(server_id))
    return {"status": "refreshing", "server_id": server_id}


@_internal.get("/internal/health")
async def internal_health():
    return {"status": "ok", "server_count": len(_registry.list_all())}


class ServerRegistrationBody(PydanticBase):
    id: str
    name: str
    description: str
    url: str
    centroid: list[float] | None = None
    registered_at: str | None = None


def run(preload: bool = False) -> None:
    if preload:
        from mcpscapes.shared.embedder import get_embedder
        get_embedder().embed("warmup")

    from starlette.routing import Mount
    mcp_app = mcp.get_asgi_app()
    _internal.mount("/mcp", mcp_app)
    uvicorn.run(_internal, host="0.0.0.0", port=META_PORT)
