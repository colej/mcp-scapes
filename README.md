# mcp-scapes

A semantically-routed, topographic MCP federation with soft/overlapping context assignment.

## Installation

```bash
pip install -e ".[dev]"
```

## Quickstart

```bash
# Start the meta-server
topomcp-meta

# Start a child server
CHILD_ID=energy CHILD_NAME="Energy Markets" META_URL=http://localhost:8000 topomcp-child
```

See the full README for architecture details and all MCP tools.
