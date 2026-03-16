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
