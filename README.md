# mcp-scapes

A semantically-routed, topographic MCP federation with soft/overlapping context assignment.
Inspired by Rebecca Schwarzlose, *Brainscapes* (2021).

## Concept

Topographic routing applies the neuroscientific insight that the brain organises representations continuously in space: concepts with similar features sit near each other, and boundaries between regions are gradients rather than hard lines. mcp-scapes applies this principle to MCP servers — each child server is a domain region, proximity between servers is encoded in a live distance matrix, and every piece of knowledge is assigned soft membership weights across all regions rather than belonging exclusively to one.

Rather than hard-routing a query to exactly one server, mcp-scapes embeds the query and computes a softmax over all server centroids. Low temperature produces near-hard routing (the closest domain wins most of the weight); high temperature diffuses activation across many servers simultaneously. This mirrors how cortical maps blend at their boundaries, where neurons respond to multiple stimuli with graded tuning curves rather than binary membership.

## Architecture

```
  ┌────────────────────────────────────────────┐
  │              meta-server :8000             │
  │  ┌──────────┐  ┌────────┐  ┌───────────┐  │
  │  │ Registry │  │  Map   │  │  Router   │  │
  │  │ (SQLite) │  │(cosine │  │(embed +   │  │
  │  │          │◄─│ dist.) │◄─│ softmax)  │  │
  │  └──────────┘  └────────┘  └───────────┘  │
  │        MCP tools + /internal/* HTTP        │
  └──────┬──────────────┬──────────────┬───────┘
         │              │              │
  :8001  ▼       :8002  ▼      :8003  ▼
  child-energy   child-research  child-code
  KnowledgeGraph KnowledgeGraph  KnowledgeGraph
  (SQLite + vec) (SQLite + vec)  (SQLite + vec)
```

## Quickstart — Docker Compose

```bash
docker compose up --build
```

This starts:
- `meta` on port 8000
- `child-energy` on port 8001
- `child-research` on port 8002
- `child-code` on port 8003

All child servers self-register with meta on startup.

## Quickstart — Manual

```bash
pip install -e ".[dev]"

# Terminal 1 — meta
topomcp-meta

# Terminal 2 — one child
CHILD_ID=energy \
CHILD_NAME="Energy Markets" \
CHILD_DESCRIPTION="Knowledge about energy markets and commodities" \
META_URL=http://localhost:8000 \
topomcp-child
```

Add `--preload` to either command to warm up the sentence-transformer model before accepting connections.

## MCP Tools — Meta Server

| Tool | Description |
|------|-------------|
| `list_domains` | All registered servers with nearest 3 neighbours |
| `route_query(query, top_k, temperature)` | Nearest servers with scores and connection info |
| `soft_route_query(query, temperature)` | Full softmax weight distribution over all domains |
| `domain_map` | Pairwise inter-server distances as edge list |
| `interpolate_domains(domain_a, domain_b, t)` | Route the interpolated midpoint between two domains |
| `search_across(query, top_k_servers, top_k_results)` | Fan-out search, merge and re-rank results |
| `add_to_domain(domain_id, content, tags)` | Proxy add_memory to a specific child |
| `add_with_soft_routing(content, tags, temperature, threshold)` | Write to all servers above weight threshold |

## MCP Tools — Child Server

| Tool | Description |
|------|-------------|
| `add_memory(content, tags, domain_weights)` | Add a node to the local knowledge graph |
| `search_memory(query, k)` | Semantic search, returns top-k with scores |
| `get_memory(id)` | Fetch a single node by id |
| `add_relation(source_id, target_id, relation, weight)` | Add a directed edge between nodes |
| `describe` | Server identity, centroid, node count, top tags |
| `list_memories(limit, offset)` | Paginated listing of all nodes |

## How Soft Routing Works

When a query or content arrives at the meta-server, mcp-scapes:

1. **Embeds** the text using `all-MiniLM-L6-v2` (384-dimensional vector).
2. **Scores** every registered server by cosine similarity between the query embedding and each server's centroid (the mean of all its stored node embeddings).
3. **Applies softmax with temperature** `τ`:

   ```
   weight_i = exp(sim_i / τ) / Σ exp(sim_j / τ)
   ```

   - `τ = 0.1` → near-hard routing: the closest domain gets ~all the weight.
   - `τ = 2.0` → diffuse activation: multiple domains receive substantial weight.

4. For `add_with_soft_routing`, any server whose weight exceeds `threshold` (default `0.15`) receives the write. The node is stored with `domain_weights` set to the full softmax distribution, encoding its fractional membership across all regions.

Server centroids update automatically: each `add_memory` call fires a background POST to `meta/internal/refresh_centroid/{id}`, which fetches the child's updated centroid and rebuilds the pairwise distance matrix.

## Contributing

1. Fork the repo and create a feature branch.
2. Install dev dependencies: `pip install -e ".[dev]"`.
3. Run tests: `pytest tests/`.
4. Open a PR against `main`.
