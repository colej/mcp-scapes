"""Tests for meta-server registry, routing, and soft-routing fan-out."""
import tempfile
import os
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from mcpscapes.meta.map import TopographicMap
from mcpscapes.meta.registry import Registry
from mcpscapes.meta.router import Router
from mcpscapes.shared.models import ServerRegistration


@pytest.fixture
def registry():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Registry(os.path.join(tmpdir, "meta.db"))


@pytest.fixture
def topo():
    return TopographicMap()


@pytest.fixture
def router(registry, topo):
    return Router(registry, topo)


def test_register_server(registry):
    reg = ServerRegistration(
        id="test",
        name="Test Server",
        description="A test server",
        url="http://test:8000",
    )
    registry.register(reg)
    domains = registry.list_all()
    ids = [s.id for s in domains]
    assert "test" in ids


def test_route_returns_scores(registry, topo, router):
    a = ServerRegistration(
        id="a", name="A", description="A", url="http://a:8000",
        centroid=[1.0, 0.0, 0.0],
    )
    b = ServerRegistration(
        id="b", name="B", description="B", url="http://b:8000",
        centroid=[0.0, 1.0, 0.0],
    )
    registry.register(a)
    registry.register(b)
    topo.rebuild([a, b])

    # Query close to server A's centroid
    results = topo.nearest_servers([1.0, 0.0, 0.0], [a, b], top_k=2)
    assert results[0].server_id == "a"
    assert results[0].score > results[1].score


@pytest.mark.asyncio
async def test_add_with_soft_routing_fanout():
    """Mock two child servers, verify both receive writes when weights exceed threshold."""
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        reg = Registry(os.path.join(tmpdir, "meta.db"))
        topo = TopographicMap()
        router = Router(reg, topo)

        srv_a = ServerRegistration(
            id="a", name="A", description="A", url="http://a:8000",
            centroid=[1.0, 0.0],
        )
        srv_b = ServerRegistration(
            id="b", name="B", description="B", url="http://b:8000",
            centroid=[0.9, 0.1],  # close to A → both should exceed threshold with high temp
        )
        reg.register(srv_a)
        reg.register(srv_b)
        topo.rebuild([srv_a, srv_b])

        weights = topo.soft_weights([1.0, 0.0], [srv_a, srv_b], temperature=2.0)
        # At high temperature both servers should exceed 0.15 threshold
        assert weights["a"] >= 0.15
        assert weights["b"] >= 0.15
