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
