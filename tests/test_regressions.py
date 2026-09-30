"""Regression tests for two bugs found during BQT-MDQW integration.

These are kept in their own file because each one guards a specific defect that
was present in an earlier revision of the codebase. If either of these tests
ever fails, the property it protects has been broken again.

1. ``test_total_cost_equals_edge_cost_plus_node_cost``
   The A* / uniform-cost search charged the *destination* node's penalty while
   :attr:`RouteResult.node_cost` summed only the intermediate nodes. The two
   reported figures therefore disagreed by exactly
   ``router.node_cost(target)`` whenever a non-zero hop or repeater penalty was
   configured, contradicting the documented invariant
   ``total_cost == edge_cost + node_cost``.

2. ``test_walk_prior_is_independent_of_python_hash_seed``
   :func:`src.quantum_walk.shunt_decomposition` originally keyed its bipartite
   double cover on ``("out", i)`` / ``("in", i)`` string tuples. CPython
   randomises ``str.__hash__`` per process, so the cycle cover chosen by
   Hopcroft-Karp -- and therefore the walk unitary and the routing prior --
   varied with ``PYTHONHASHSEED`` even at a fixed seed. That silently breaks
   the reproducibility guarantee the project advertises. The keys are now
   integers, whose hashing is not randomised.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.mdqw_routing import MDQWConfig, MDQWRouter, RoutingWeights
from src.network import build_example_network, random_network
from src.quantum_walk import quantum_walk_prior, shunt_decomposition
from src.mdqw_routing import walkable_proxy

REPO_ROOT = Path(__file__).resolve().parents[1]


def _prior_fingerprint(graph) -> str:
    """Return a stable hash of a graph's walk-derived prior.

    Args:
        graph: Any undirected graph.

    Returns:
        A short hexadecimal digest of the shunts and the resulting prior.
    """
    shunts = shunt_decomposition(graph)
    shunts_part = str([matrix.tolist() for matrix in shunts])
    prior = quantum_walk_prior(graph, steps=8)
    prior_part = json.dumps(
        {str(k): round(v, 9) for k, v in sorted(prior.items(), key=lambda kv: str(kv[0]))}
    )
    payload = shunts_part + prior_part
    return hashlib.md5(payload.encode()).hexdigest()[:12]


# ----------------------------------------------------------------------
# Bug 1: total_cost / node_cost inconsistency
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("hop_penalty", "repeater_penalty"),
    [(0.0, None), (0.5, 0.9), (2.0, None), (0.25, 0.25), (1.5, 3.0)],
)
@pytest.mark.parametrize("mode", ["none", "astar"])
def test_total_cost_equals_edge_cost_plus_node_cost(
    hop_penalty: float, repeater_penalty: float | None, mode: str
) -> None:
    """The reported cost identity must hold under every penalty setting."""
    net = build_example_network()
    router = MDQWRouter(
        net,
        MDQWConfig(
            weights=RoutingWeights(
                hop_penalty=hop_penalty, repeater_penalty=repeater_penalty
            ),
            mode=mode,
        ),
    )
    result = router.route("A", "F")

    assert result.total_cost == pytest.approx(
        result.edge_cost + result.node_cost, abs=1e-9
    ), (
        "RouteResult.total_cost disagrees with edge_cost + node_cost; the "
        "search is charging a node term that the result does not report"
    )


def test_destination_node_penalty_is_not_charged() -> None:
    """The destination is exempt from the node term, as documented.

    Node ``F`` is a repeater with a distinctive penalty. If the destination were
    charged, the identity above would break by exactly that amount.
    """
    net = build_example_network()
    router = MDQWRouter(
        net,
        MDQWConfig(
            weights=RoutingWeights(hop_penalty=0.5, repeater_penalty=0.9),
            mode="none",
        ),
    )
    result = router.route("A", "F")

    # F is the destination, so its 0.9 repeater penalty must not appear.
    assert router.node_cost("F") == pytest.approx(0.9)
    # Only the intermediates B and C are charged, at hop_penalty each.
    expected_intermediates = sum(
        router.node_cost(node) for node in result.path[1:-1]
    )
    assert result.node_cost == pytest.approx(expected_intermediates, abs=1e-9)
    assert result.path[-1] == "F"
    assert result.node_cost == pytest.approx(1.0, abs=1e-9)


# ----------------------------------------------------------------------
# Bug 2: PYTHONHASHSEED-dependent walk prior
# ----------------------------------------------------------------------
@pytest.mark.parametrize("seed", ["0", "1", "42"])
def test_walk_prior_is_independent_of_python_hash_seed(seed: str) -> None:
    """The walk-derived prior must not depend on CPython's hash randomisation.

    Runs a subprocess with an explicit ``PYTHONHASHSEED`` and compares a
    fingerprint against the in-process value. If the shunts were keyed on
    strings, these digests would differ.
    """
    expected = _prior_fingerprint(walkable_proxy(build_example_network().to_networkx())[0])

    script = (
        "import sys, hashlib, json;"
        f"sys.path.insert(0, {str(REPO_ROOT)!r});"
        "from src.quantum_walk import quantum_walk_prior, shunt_decomposition;"
        "from src.mdqw_routing import walkable_proxy;"
        "from src.network import build_example_network;"
        "g = walkable_proxy(build_example_network().to_networkx())[0];"
        "sh = shunt_decomposition(g);"
        "pr = quantum_walk_prior(g, steps=8);"
        "payload = str([m.tolist() for m in sh]) + json.dumps("
        "{str(k): round(v, 9) for k, v in sorted(pr.items(), key=lambda kv: str(kv[0]))});"
        "print(hashlib.md5(payload.encode()).hexdigest()[:12])"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env={"PYTHONHASHSEED": seed, "PATH": ""},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected, (
        f"prior fingerprint changed under PYTHONHASHSEED={seed}: "
        f"expected {expected}, got {completed.stdout.strip()}"
    )


def test_shunt_decomposition_is_stable_across_repeated_calls() -> None:
    """Repeated decomposition of the same graph returns identical shunts."""
    net = build_example_network()
    proxy = walkable_proxy(net.to_networkx())[0]
    first = [matrix.tolist() for matrix in shunt_decomposition(proxy)]
    second = [matrix.tolist() for matrix in shunt_decomposition(proxy)]
    assert first == second


def test_random_network_prior_uses_only_graph_nodes() -> None:
    """The prior covers exactly the proxy's nodes, whatever the graph."""
    net = random_network(10, 3, seed=4242)
    proxy, _strategy = walkable_proxy(net.to_networkx())
    prior = quantum_walk_prior(proxy, steps=6)
    assert set(prior) == set(proxy.nodes)
    assert max(prior.values()) == pytest.approx(1.0, abs=1e-9)
