"""Tests for TopographicMap routing logic."""
import math

import pytest

from mcpscapes.meta.map import TopographicMap
from mcpscapes.shared.models import ServerRegistration


def _make_server(id: str, centroid: list[float]) -> ServerRegistration:
    return ServerRegistration(id=id, name=id, description=id, url=f"http://{id}:8000", centroid=centroid)


def test_soft_weights_sum_to_one():
    topo = TopographicMap()
    servers = [
        _make_server("a", [1.0, 0.0, 0.0]),
        _make_server("b", [0.0, 1.0, 0.0]),
        _make_server("c", [0.0, 0.0, 1.0]),
    ]
    topo.rebuild(servers)
    weights = topo.soft_weights([1.0, 0.5, 0.0], servers, temperature=0.5)
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_temperature_sharpness():
    topo = TopographicMap()
    servers = [
        _make_server("a", [1.0, 0.0]),
        _make_server("b", [0.0, 1.0]),
        _make_server("c", [-1.0, 0.0]),
    ]
    topo.rebuild(servers)
    query = [1.0, 0.0]
    low_temp = topo.soft_weights(query, servers, temperature=0.1)
    high_temp = topo.soft_weights(query, servers, temperature=2.0)
    # Lower temperature → more peaked: winning server should have higher weight
    assert low_temp["a"] > high_temp["a"]


def test_interpolate_midpoint():
    topo = TopographicMap()
    servers = [
        _make_server("x", [1.0, 0.0]),
        _make_server("y", [0.0, 1.0]),
    ]
    topo.rebuild(servers)
    mid = topo.interpolate("x", "y", 0.5, servers)
    # Midpoint should be equidistant from both endpoints
    import numpy as np
    vx = [1.0, 0.0]
    vy = [0.0, 1.0]
    dx = float(np.linalg.norm(np.array(mid) - np.array(vx)))
    dy = float(np.linalg.norm(np.array(mid) - np.array(vy)))
    assert abs(dx - dy) < 1e-6


def test_nearest_server_ordering():
    topo = TopographicMap()
    servers = [
        _make_server("close", [1.0, 0.0]),
        _make_server("far", [-1.0, 0.0]),
    ]
    topo.rebuild(servers)
    results = topo.nearest_servers([1.0, 0.0], servers, top_k=2)
    assert results[0].server_id == "close"
    assert results[0].score > results[1].score
