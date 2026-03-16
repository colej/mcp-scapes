"""Tests for KnowledgeGraph (child server knowledge layer)."""
import tempfile
import os

import pytest

from mcpscapes.child.graph import KnowledgeGraph


@pytest.fixture
def graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield KnowledgeGraph(os.path.join(tmpdir, "test.db"))


def test_add_and_search(graph):
    node = graph.add_node(
        content="solar panel efficiency improvements",
        tags=["solar", "energy"],
        domain_weights={"energy": 1.0},
    )
    results = graph.search("solar energy improvements", k=5)
    ids = [n.id for n, _ in results]
    assert node.id in ids


def test_domain_weights_preserved(graph):
    weights = {"energy": 0.7, "research": 0.3}
    node = graph.add_node(
        content="cross-domain research on energy policy",
        tags=["policy"],
        domain_weights=weights,
    )
    retrieved = graph.get_node(node.id)
    assert retrieved is not None
    assert retrieved.domain_weights == weights


def test_centroid_updates(graph):
    centroid_before = graph.compute_centroid()
    graph.add_node("first node content about topic A", tags=[], domain_weights={"x": 1.0})
    centroid_after_one = graph.compute_centroid()
    graph.add_node("second node content about topic B", tags=[], domain_weights={"x": 1.0})
    centroid_after_two = graph.compute_centroid()
    # Centroid should change as nodes are added
    assert centroid_before != centroid_after_one
    assert centroid_after_one != centroid_after_two
