import os

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
