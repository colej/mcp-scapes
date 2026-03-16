import asyncio
import os
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastmcp import FastMCP

from mcpscapes.child.graph import KnowledgeGraph

CHILD_ID = os.environ.get("CHILD_ID", "child")
CHILD_NAME = os.environ.get("CHILD_NAME", "Child Server")
CHILD_DESCRIPTION = os.environ.get("CHILD_DESCRIPTION", "A child MCP server")
CHILD_PORT = int(os.environ.get("CHILD_PORT", "8001"))
CHILD_DB_PATH = os.environ.get("CHILD_DB_PATH", f"./data/{CHILD_ID}.db")
META_URL = os.environ.get("META_URL", "")

mcp = FastMCP(CHILD_NAME)
_graph: KnowledgeGraph | None = None


def get_graph() -> KnowledgeGraph:
    global _graph
    if _graph is None:
        _graph = KnowledgeGraph(CHILD_DB_PATH)
    return _graph


@mcp.tool()
async def add_memory(
    content: str,
    tags: list[str],
    domain_weights: dict[str, float] | None = None,
) -> dict:
    """Add a memory node to the local knowledge graph."""
    import asyncio
    import httpx

    weights = domain_weights if domain_weights is not None else {CHILD_ID: 1.0}
    node = get_graph().add_node(content, tags, weights)

    if META_URL:
        async def _notify():
            async with httpx.AsyncClient() as client:
                try:
                    await client.post(
                        f"{META_URL}/internal/refresh_centroid/{CHILD_ID}",
                        timeout=5.0,
                    )
                except Exception:
                    pass
        asyncio.create_task(_notify())

    return node.model_dump(mode="json")


@mcp.tool()
async def search_memory(query: str, k: int = 5) -> list[dict]:
    """Search the local knowledge graph by semantic similarity."""
    results = get_graph().search(query, k)
    return [
        {**node.model_dump(mode="json"), "score": score}
        for node, score in results
    ]


@mcp.tool()
async def get_memory(id: str) -> dict | None:
    """Return a single memory node by id."""
    node = get_graph().get_node(id)
    return node.model_dump(mode="json") if node else None


@mcp.tool()
async def add_relation(
    source_id: str,
    target_id: str,
    relation: str,
    weight: float = 1.0,
) -> dict:
    """Add a directed edge between two memory nodes."""
    get_graph().add_edge(source_id, target_id, relation, weight)
    return {"source_id": source_id, "target_id": target_id, "relation": relation, "weight": weight}


@mcp.tool()
async def describe() -> dict:
    """Return server identity, centroid, node count, and top tags."""
    graph = get_graph()
    centroid = graph.compute_centroid()
    row = graph._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
    node_count = row[0] if row else 0
    tag_rows = graph._conn.execute("SELECT tags FROM nodes").fetchall()
    import json
    tag_freq: dict[str, int] = {}
    for (tags_json,) in tag_rows:
        for tag in json.loads(tags_json):
            tag_freq[tag] = tag_freq.get(tag, 0) + 1
    top_tags = sorted(tag_freq, key=lambda t: tag_freq[t], reverse=True)[:10]
    return {
        "id": CHILD_ID,
        "name": CHILD_NAME,
        "description": CHILD_DESCRIPTION,
        "centroid": centroid,
        "node_count": node_count,
        "top_tags": top_tags,
    }


@mcp.tool()
async def list_memories(limit: int = 20, offset: int = 0) -> list[dict]:
    """Paginated listing of all memory nodes."""
    import json
    graph = get_graph()
    rows = graph._conn.execute(
        "SELECT id, content, domain_weights, embedding, tags, created_at, updated_at "
        "FROM nodes LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return [graph._row_to_node(row).model_dump(mode="json") for row in rows]


async def _register_with_meta() -> None:
    if not META_URL:
        return
    from mcpscapes.shared.models import ServerRegistration
    reg = ServerRegistration(
        id=CHILD_ID,
        name=CHILD_NAME,
        description=CHILD_DESCRIPTION,
        url=f"http://localhost:{CHILD_PORT}",
    )
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                f"{META_URL}/internal/register",
                json=reg.model_dump(mode="json"),
                timeout=10.0,
            )
        except Exception as exc:
            from loguru import logger
            logger.warning(f"Failed to register with meta-server: {exc}")


def run(preload: bool = False) -> None:
    if preload:
        from mcpscapes.shared.embedder import get_embedder
        get_embedder().embed("warmup")

    async def _startup():
        await _register_with_meta()

    asyncio.run(_startup())
    uvicorn.run(mcp.get_asgi_app(), host="0.0.0.0", port=CHILD_PORT)
