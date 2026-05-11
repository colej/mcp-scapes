# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## mcp-scapes

A distributed MCP (Model Context Protocol) server federation that routes queries across multiple domain-specific child servers using neural-inspired topographic mapping.

### Architecture

```
Meta-Server (:8000)
  ├── Registry  — SQLite table of registered child servers + their embedding centroids
  ├── Map       — Pairwise cosine distance matrix between server centroids
  └── Router    — Embeds incoming queries (all-MiniLM-L6-v2, 384D) → softmax over
                  server similarities → distributes calls to top-k children

Child Servers (:8001+)
  └── Each maintains a local knowledge graph (SQLite + sqlite-vec) and
      auto-registers its centroid with the meta-server on startup
```

**Routing:** Temperature-controlled softmax (τ=0.1 = hard/winner-take-most, τ=2.0 = diffuse across all servers). Queries are not hard-routed to one server — soft overlapping assignment is intentional.

**Key source locations:**
- [mcpscapes/meta/](mcpscapes/meta/) — router, registry, map logic
- [mcpscapes/child/](mcpscapes/child/) — knowledge graph storage
- [mcpscapes/shared/](mcpscapes/shared/) — embedder, shared models

### Commands

```bash
# Install (dev)
pip install -e ".[dev]"

# Run tests
pytest tests/

# Start meta-server
topomcp-meta

# Start a child server (env vars configure identity)
CHILD_ID=energy \
CHILD_NAME="Energy Markets" \
CHILD_DESCRIPTION="Electricity prices, grid infrastructure, renewables" \
META_URL=http://localhost:8000 \
topomcp-child

# Start everything via Docker
docker compose up --build
```

### Definition of done

- `pytest tests/` passes
- New code has tests using real embeddings, not mocked vectors
- If centroids or embedding model changed, registry migration documented

### Don't

- Don't swap the embedding model without re-running centroid registration for all child servers
- Don't hard-route queries to a single server — soft overlapping assignment is the point
- Don't add a child server without updating the distance matrix recomputation
