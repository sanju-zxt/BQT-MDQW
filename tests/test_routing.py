"""Tests for :mod:`src.mdqw_routing`.

What this file is really for
---------------------------
The routing module makes one claim that the rest of the project leans on: with
the walk-informed prior switched off, the search returns a *minimum-cost* path
for the cost function it optimises. Everything else here exists to defend that
claim, and to defend the honesty of the module's self-description when the
claim does not hold:

* :func:`_manual_edge_cost` re-implements ``C(e)`` from the module's own
  documented formula using only public data (``metric_ranges`` and
  ``link_metrics``). Optimality claims are then checked against an
  independently computed cost rather than against ``MDQWRouter.edge_cost``
  twice, which would only prove the router is self-consistent.
* ``test_astar_matches_brute_force_over_every_simple_path`` enumerates *all*
  simple paths on a seeded random network and compares the router's answer with
  the minimum. This is the strongest correctness test in the file.
* ``test_walk_prior_can_only_make_the_pure_edge_cost_worse`` pins the key
  honesty invariant: adding a non-negative node term can never make the pure
  edge cost better than Dijkstra's minimum, only worse or equal.

Nothing here is statistical and nothing depends on wall clock time. Every
random network is built with an explicit ``seed=``.
"""

from __future__ import annotations

import math
import pathlib
import sys

import networkx as nx
import pytest

# Make ``src`` importable no matter how pytest is invoked (``pytest`` from the
# repository root, or ``python -m pytest`` from anywhere in the tree).
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mdqw_routing import (  # noqa: E402  (import after the sys.path fix)
    DEFAULT_WEIGHTS,
    DijkstraBaseline,
    MDQWConfig,
    MDQWRouter,
    RouteResult,
    RoutingError,
    RoutingWeights,
    compare_routes,
    enumerate_routes,
    summarise_comparison,
    walkable_proxy,
)
from src.network import EdgeMetrics, QuantumNetwork, build_example_network, random_network
from src.quantum_walk import quantum_walk_prior

#: Strategy labels :func:`walkable_proxy` is documented to be able to return.
PROXY_STRATEGIES = ("already-regular", "spanning-2-factor", "complete-graph-fallback")

#: Absolute tolerance for cost arithmetic re-derived from the public API.
COST_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _manual_edge_cost(router: MDQWRouter, u: str, v: str) -> float:
    """Recompute ``C(e)`` for link ``u-v`` straight from the module's formula.

    Independent of :meth:`MDQWRouter.edge_cost`: it re-implements

    .. code-block:: text

        C(e) = alpha*d_hat + beta*n_hat + gamma*l_hat
               + delta*(1 - f_hat) + rho*(1 - r_hat)

    using :meth:`QuantumNetwork.link_metrics` for the raw values and
    :meth:`QuantumNetwork.metric_ranges` for the normalisation bounds, with the
    documented "constant metric maps to 0.0" degenerate case. Returns
    ``float('nan')`` if the link does not exist.
    """
    if not router.graph.has_edge(u, v):
        return float("nan")
    metrics = router.network.link_metrics(u, v)
    bounds = router.network.metric_ranges()

    def normalised(key: str, value: float) -> float:
        low, high = bounds[key]
        if high - low < 1e-15:
            return 0.0
        return (value - low) / (high - low)

    weights = router.weights
    return (
        weights.alpha * normalised("distance_km", metrics.distance_km)
        + weights.beta * normalised("noise", metrics.noise)
        + weights.gamma * normalised("latency_ms", metrics.latency_ms)
        + weights.delta * (1.0 - normalised("fidelity", metrics.fidelity))
        + weights.rho * (1.0 - normalised("reliability", metrics.reliability))
    )


def _route_edge_cost(router: MDQWRouter, path) -> float:
    """Sum the manually recomputed link costs along a node sequence."""
    return sum(
        _manual_edge_cost(router, u, v) for u, v in zip(path, path[1:])
    )


def _intermediate_node_cost(router: MDQWRouter, result: RouteResult) -> float:
    """Recompute the node term from the public API over the intermediate nodes.

    Source and destination are exempt from the node term, so the sum runs over
    ``path[1:-1]`` only.
    """
    return sum(router.node_cost(node) for node in result.path[1:-1])


def _assert_valid_path(net: QuantumNetwork, result: RouteResult) -> None:
    """Every consecutive pair on the path must be a real edge of ``net``."""
    assert result.source == result.path[0]
    assert result.target == result.path[-1]
    assert result.hop_count == len(result.path) - 1
    for u, v in zip(result.path, result.path[1:]):
        assert net.graph.has_edge(u, v), (u, v)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def example() -> QuantumNetwork:
    """The hand-designed six-node A-F network."""
    return build_example_network()


@pytest.fixture
def chain() -> QuantumNetwork:
    """A three-node line A-B-C with no direct A-C link.

    Deliberately uniform links, so the only thing that can decide a route is the
    number of hops.
    """
    net = QuantumNetwork(name="chain-A-B-C")
    for node in ("A", "B", "C"):
        net.add_node(node)
    metrics = EdgeMetrics(100.0, 0.10, 2.0, 0.95, 0.99)
    net.add_link("A", "B", metrics)
    net.add_link("B", "C", metrics)
    return net


@pytest.fixture
def chain_with_direct_link() -> QuantumNetwork:
    """The same line plus a deliberately expensive direct A-C link.

    A hop-count-driven router would take A-C; a cost-driven router must prefer
    the two-hop path. This is the cheapest possible proof that the cost function
    is actually used.
    """
    net = QuantumNetwork(name="chain-with-direct-A-C")
    for node in ("A", "B", "C"):
        net.add_node(node)
    net.add_link("A", "B", EdgeMetrics(100.0, 0.10, 2.0, 0.95, 0.99))
    net.add_link("B", "C", EdgeMetrics(100.0, 0.10, 2.0, 0.95, 0.99))
    net.add_link("A", "C", EdgeMetrics(400.0, 0.35, 20.0, 0.80, 0.85))
    return net


@pytest.fixture
def repeater_pair() -> QuantumNetwork:
    """Two equal-hop branches from A to Z: one through P, one through repeater R.

    Both branches carry *identical* link metrics, so the only difference between
    them is the node term -- which is exactly what is under test.
    """
    net = QuantumNetwork(name="repeater-branches")
    net.add_node("A")
    net.add_node("P", is_repeater=False)
    net.add_node("R", is_repeater=True)
    net.add_node("Z")
    metrics = EdgeMetrics(100.0, 0.10, 2.0, 0.95, 0.99)
    net.add_link("A", "P", metrics)
    net.add_link("P", "Z", metrics)
    net.add_link("A", "R", metrics)
    net.add_link("R", "Z", metrics)
    return net


@pytest.fixture
def seeded() -> QuantumNetwork:
    """A reproducible ten-node network for the brute-force optimality check."""
    return random_network(10, 3, seed=42)


@pytest.fixture
def dominant_hop_penalty(example: QuantumNetwork) -> float:
    """A hop penalty larger than any single link cost.

    With ``penalty > max_e C(e)``, two routes differing by one hop can never tie,
    so minimising ``sum(C(e)) + (k - 1) * penalty`` is exactly minimising the
    hop count. That makes the resulting route predictable without hard-coding
    any float.
    """
    router = MDQWRouter(example, MDQWConfig(mode="none"))
    worst_link = max(_manual_edge_cost(router, u, v) for u, v in example.links)
    return 10.0 * worst_link


# ---------------------------------------------------------------------------
# 1. RoutingWeights
# ---------------------------------------------------------------------------


def test_routing_weights_defaults() -> None:
    """The documented balanced defaults are the module's contract."""
    weights = RoutingWeights()
    assert weights.alpha == pytest.approx(1.0)
    assert weights.beta == pytest.approx(1.0)
    assert weights.gamma == pytest.approx(0.5)
    assert weights.delta == pytest.approx(1.5)
    assert weights.rho == pytest.approx(0.25)
    assert weights.hop_penalty == pytest.approx(0.0)

    # ``DEFAULT_WEIGHTS`` documents the link-cost half of the defaults.
    for key, value in DEFAULT_WEIGHTS.items():
        assert getattr(weights, key) == pytest.approx(value)


def test_routing_weights_repeater_penalty_defaults_to_hop_penalty() -> None:
    """Omitting ``repeater_penalty`` means "charge the same as any other hop"."""
    assert RoutingWeights(hop_penalty=0.3).repeater_penalty == pytest.approx(0.3)
    assert RoutingWeights().repeater_penalty == pytest.approx(0.0)
    # An explicit value is kept, not overwritten.
    assert RoutingWeights(hop_penalty=0.3, repeater_penalty=0.9).repeater_penalty == pytest.approx(0.9)


def test_routing_weights_as_dict_has_all_seven_keys() -> None:
    """``as_dict`` is what the sweeps serialise, so its key set is a contract."""
    payload = RoutingWeights(hop_penalty=0.4, repeater_penalty=0.7).as_dict()
    assert set(payload) == {
        "alpha",
        "beta",
        "gamma",
        "delta",
        "rho",
        "hop_penalty",
        "repeater_penalty",
    }
    assert len(payload) == 7
    assert payload["hop_penalty"] == pytest.approx(0.4)
    assert payload["repeater_penalty"] == pytest.approx(0.7)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": -1.0},
        {"beta": -0.5},
        {"gamma": -1e-9},
        {"delta": -1.0},
        {"rho": -0.25},
        {"hop_penalty": -1.0},
        {"repeater_penalty": -0.1},
    ],
    ids=["alpha", "beta", "gamma", "delta", "rho", "hop", "repeater"],
)
def test_routing_weights_reject_negative_values(kwargs: dict) -> None:
    """A negative weight would let the router *reward* something.

    With a negative coefficient the "shortest path" search would minimise a cost
    that can go the wrong way, so every weight is required to be non-negative.
    """
    with pytest.raises(RoutingError, match="non-negative"):
        RoutingWeights(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": float("nan")},
        {"beta": float("inf")},
        {"delta": float("-inf")},
    ],
    ids=["nan", "inf", "-inf"],
)
def test_routing_weights_reject_non_finite_values(kwargs: dict) -> None:
    with pytest.raises(RoutingError, match="finite"):
        RoutingWeights(**kwargs)


def test_routing_weights_reject_the_all_zero_degenerate_case() -> None:
    """Every link would cost zero, so *every* path would tie.

    The module refuses this configuration outright instead of silently returning
    an arbitrary shortest-by-hops route, and the error says why.
    """
    with pytest.raises(RoutingError, match="degenerate"):
        RoutingWeights(alpha=0.0, beta=0.0, gamma=0.0, delta=0.0, rho=0.0)


def test_routing_error_is_a_value_error() -> None:
    """``RoutingError`` subclasses ``ValueError`` so generic handlers still work."""
    assert issubclass(RoutingError, ValueError)


# ---------------------------------------------------------------------------
# 2. MDQWConfig
# ---------------------------------------------------------------------------


def test_mdqw_config_defaults() -> None:
    """A* by default, with the walk-informed prior switched *off*.

    The prior being off by default is deliberate: it is a heuristic and voids the
    optimality guarantee.
    """
    config = MDQWConfig()
    assert config.mode == "astar"
    assert config.use_qw_prior is False
    assert config.qw_prior_weight == pytest.approx(0.0)
    assert config.prior_steps == 8
    assert config.max_hops is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "greedy"}, "mode"),
        ({"mode": "A*"}, "mode"),
        ({"mode": None}, "mode"),
        ({"qw_prior_weight": -0.1}, "qw_prior_weight"),
        ({"qw_prior_weight": float("nan")}, "qw_prior_weight"),
        ({"prior_steps": 0}, "prior_steps"),
        ({"prior_steps": -2}, "prior_steps"),
        ({"max_hops": -1}, "max_hops"),
    ],
    ids=[
        "unknown-mode",
        "wrong-case-mode",
        "none-mode",
        "negative-prior-weight",
        "nan-prior-weight",
        "zero-prior-steps",
        "negative-prior-steps",
        "negative-max-hops",
    ],
)
def test_mdqw_config_rejects_invalid_fields(kwargs: dict, message: str) -> None:
    """Every guard must fire before a search starts, naming the bad field."""
    with pytest.raises(RoutingError, match=message):
        MDQWConfig(**kwargs)


# ---------------------------------------------------------------------------
# 3. Hand-computed optimality
# ---------------------------------------------------------------------------


def test_dijkstra_finds_the_only_path_on_a_line(chain: QuantumNetwork) -> None:
    """On A-B-C the A->C answer is forced, so the *cost* is what is under test."""
    router = MDQWRouter(chain, MDQWConfig(mode="none"))
    result = DijkstraBaseline(chain).route("A", "C")

    assert result.path == ("A", "B", "C")
    assert result.hop_count == 2
    assert result.node_cost == pytest.approx(0.0)
    assert result.edge_cost == pytest.approx(
        router.edge_cost("A", "B") + router.edge_cost("B", "C")
    )
    assert result.total_cost == pytest.approx(result.edge_cost)


def test_reported_edge_cost_matches_an_independent_recomputation(chain: QuantumNetwork) -> None:
    """``edge_cost`` must equal a from-scratch evaluation of the documented formula.

    Recomputing through ``router.edge_cost`` twice would only prove the router is
    self-consistent; ``_manual_edge_cost`` re-derives the number from the raw
    metrics and the network's normalisation bounds instead.
    """
    router = MDQWRouter(chain, MDQWConfig(mode="none"))
    result = router.route("A", "C")

    expected = _manual_edge_cost(router, "A", "B") + _manual_edge_cost(router, "B", "C")
    assert expected > 0.0, "the fixture must not produce a degenerate zero cost"
    assert result.edge_cost == pytest.approx(expected, abs=COST_TOLERANCE)
    assert result.total_cost == pytest.approx(expected, abs=COST_TOLERANCE)


def test_cost_beats_hop_count_on_an_expensive_direct_link(chain_with_direct_link: QuantumNetwork) -> None:
    """A one-hop but expensive A-C link must lose to the cheap two-hop path.

    If the router were minimising hop count it would return A-C. This is the
    assertion that proves the cost function is really what is being optimised.
    """
    net = chain_with_direct_link
    router = MDQWRouter(net, MDQWConfig(mode="none"))

    two_hop = _manual_edge_cost(router, "A", "B") + _manual_edge_cost(router, "B", "C")
    direct = _manual_edge_cost(router, "A", "C")
    assert two_hop < direct, "fixture precondition: the detour must be cheaper"

    baseline = DijkstraBaseline(net).route("A", "C")
    assert baseline.path == ("A", "B", "C")
    assert baseline.hop_count == 2
    assert baseline.edge_cost == pytest.approx(two_hop, abs=COST_TOLERANCE)

    # Both MDQW search modes must agree with the classical baseline.
    for mode in ("none", "astar"):
        result = MDQWRouter(net, MDQWConfig(mode=mode)).route("A", "C")
        assert result.path == ("A", "B", "C"), mode
        assert result.edge_cost == pytest.approx(two_hop, abs=COST_TOLERANCE), mode


def test_direct_link_wins_when_it_is_the_cheap_one(chain: QuantumNetwork) -> None:
    """The converse check: with a cheap A-C link the one-hop route must win.

    Without this, "Dijkstra prefers two hops" could be satisfied by a router that
    simply never prefers a direct link.
    """
    net = chain.copy()
    net.add_link("A", "C", EdgeMetrics(50.0, 0.02, 0.5, 0.995, 0.999))

    router = MDQWRouter(net, MDQWConfig(mode="none"))
    assert _manual_edge_cost(router, "A", "C") < (
        _manual_edge_cost(router, "A", "B") + _manual_edge_cost(router, "B", "C")
    )

    result = DijkstraBaseline(net).route("A", "C")
    assert result.path == ("A", "C")
    assert result.hop_count == 1


# ---------------------------------------------------------------------------
# 4. A* versus uniform-cost search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "target"),
    [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("A", "F"), ("B", "F"), ("C", "E")],
)
def test_astar_and_uniform_cost_agree(example: QuantumNetwork, source: str, target: str) -> None:
    """A*'s heuristic is admissible, so it must not change the answer.

    ``h(v)`` is the minimum incident link cost at ``v``; any route out of ``v``
    pays at least that much, and node terms are non-negative, so ``h`` never
    over-estimates. The observable consequence is that the *same* path comes
    back with the same ``edge_cost`` and the same ``total_cost``.

    The number of nodes each search expands is not part of the public API (the
    frontier is a local list inside ``MDQWRouter.route``), so "A* expands fewer
    nodes" is not directly assertable without instrumenting the heap; what is
    assertable is that the heuristic changed nothing about the result.
    """
    uniform = MDQWRouter(example, MDQWConfig(mode="none")).route(source, target)
    astar = MDQWRouter(example, MDQWConfig(mode="astar")).route(source, target)

    assert astar.path == uniform.path, (source, target)
    assert astar.edge_cost == pytest.approx(uniform.edge_cost, abs=COST_TOLERANCE)
    assert astar.total_cost == pytest.approx(uniform.total_cost, abs=COST_TOLERANCE)
    assert astar.hop_count == uniform.hop_count
    assert "A*" in astar.algorithm
    assert "uniform-cost" in uniform.algorithm


# ---------------------------------------------------------------------------
# 5. A* is actually optimal (brute force)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seed", "source", "target"),
    [("42", "N0", "N5"), ("42", "N1", "N9"), ("42", "N3", "N7")],
    ids=["seed42-N0-N5", "seed42-N1-N9", "seed42-N3-N7"],
)
def test_astar_matches_brute_force_over_every_simple_path(seed: str, source: str, target: str) -> None:
    """The router's answer must equal the minimum over *all* simple paths.

    ``enumerate_routes`` is given a path budget far above the number of simple
    paths and a hop cutoff of ``n - 1``, so its cheapest row is the true optimum
    rather than a truncated search. This is the strongest correctness test in the
    file: an inadmissible heuristic, an off-by-one in the node term, or a broken
    edge cost would all show up here.
    """
    net = random_network(10, 3, seed=int(seed))
    router = MDQWRouter(net, MDQWConfig(mode="astar"))

    frame = enumerate_routes(net, source, target, max_paths=100_000, max_hops=len(net) - 1)
    assert not frame.empty
    assert len(frame) > 1, "the enumeration must be a genuine search space"

    result = router.route(source, target)
    _assert_valid_path(net, result)

    assert result.edge_cost == pytest.approx(
        float(frame["edge_cost"].min()), abs=COST_TOLERANCE
    )
    # The independently recomputed cost agrees with both.
    assert result.edge_cost == pytest.approx(_route_edge_cost(router, result.path), abs=COST_TOLERANCE)
    # And the Dijkstra baseline, which optimises the same objective, agrees too.
    baseline = DijkstraBaseline(net).route(source, target)
    assert result.edge_cost == pytest.approx(baseline.edge_cost, abs=COST_TOLERANCE)
    assert result.total_cost == pytest.approx(baseline.total_cost, abs=COST_TOLERANCE)


def test_brute_force_search_space_is_large_enough_to_be_meaningful(seeded: QuantumNetwork) -> None:
    """Guard for the test above: a two-path enumeration proves very little."""
    router = MDQWRouter(seeded, MDQWConfig(mode="astar"))
    counts = [
        len(enumerate_routes(seeded, u, v, max_paths=100_000, max_hops=len(seeded) - 1))
        for u, v in (("N0", "N5"), ("N1", "N9"), ("N3", "N7"))
    ]
    assert min(counts) >= 10, counts
    for u, v in (("N0", "N5"), ("N1", "N9"), ("N3", "N7")):
        result = router.route(u, v)
        assert result.is_cost_optimal is True
        assert result.notes == "Optimal for the stated objective."


# ---------------------------------------------------------------------------
# 6. Hop and repeater penalties change routing
# ---------------------------------------------------------------------------


def test_zero_hop_penalty_gives_the_minimum_edge_cost_route(
    example: QuantumNetwork,
) -> None:
    """With no node term the router is a pure ``C(e)`` optimiser.

    It must therefore agree with the Dijkstra baseline on ``edge_cost``, which is
    the fair cross-algorithm comparison the module's docstring insists on.
    """
    router = MDQWRouter(example, MDQWConfig(weights=RoutingWeights(hop_penalty=0.0)))
    result = router.route("A", "F")
    baseline = DijkstraBaseline(example).route("A", "F")

    assert result.node_cost == pytest.approx(0.0)
    assert result.edge_cost == pytest.approx(baseline.edge_cost, abs=COST_TOLERANCE)
    assert result.total_cost == pytest.approx(result.edge_cost, abs=COST_TOLERANCE)
    assert result.total_cost == pytest.approx(
        result.edge_cost + _intermediate_node_cost(router, result), abs=COST_TOLERANCE
    )


def test_large_hop_penalty_forces_the_minimum_hop_route(
    example: QuantumNetwork, dominant_hop_penalty: float
) -> None:
    """A dominant hop penalty must collapse the choice to pure hop count.

    The penalty is larger than the most expensive single link, so no amount of
    link-cost advantage can buy an extra hop. A->F has a 2-hop route (A-E-F) and
    a cheaper 3-hop route (A-B-C-F); the penalty has to flip the decision.
    """
    cheap = MDQWRouter(example, MDQWConfig(weights=RoutingWeights(hop_penalty=0.0)))
    expensive = MDQWRouter(
        example, MDQWConfig(weights=RoutingWeights(hop_penalty=dominant_hop_penalty))
    )

    min_hops = nx.shortest_path_length(example.graph, "A", "F")
    frame = enumerate_routes(example, "A", "F")
    longest = int(frame["hop_count"].max())

    greedy = cheap.route("A", "F")
    relay = expensive.route("A", "F")

    # Precondition: the cheapest route really is the longer one.
    assert greedy.hop_count > min_hops
    assert min_hops < greedy.hop_count <= longest

    assert relay.hop_count == min_hops
    assert relay.hop_count < longest
    assert relay.edge_cost > greedy.edge_cost, "fewer hops was bought at a link-cost premium"
    assert relay.node_cost > 0.0


def test_node_term_is_recomputable_from_the_public_api(
    example: QuantumNetwork, dominant_hop_penalty: float
) -> None:
    """``result.node_cost`` must equal the sum of ``node_cost`` over intermediates.

    Source and destination are exempt, so the sum runs over ``path[1:-1]`` --
    recomputed here rather than trusting the value under test.
    """
    router = MDQWRouter(
        example, MDQWConfig(weights=RoutingWeights(hop_penalty=dominant_hop_penalty))
    )
    result = router.route("A", "F")

    expected = _intermediate_node_cost(router, result)
    assert expected > 0.0
    assert result.node_cost == pytest.approx(expected, abs=COST_TOLERANCE)
    assert result.total_cost >= result.edge_cost + result.node_cost - COST_TOLERANCE


def test_total_cost_excludes_the_destination_node_term(
    example: QuantumNetwork, dominant_hop_penalty: float
) -> None:
    """The destination must be exempt from the node term, as documented.

    ``RouteResult.node_cost`` sums only the intermediate nodes, and the module
    docstring states that "source and destination nodes are exempt from the node
    term". An earlier revision of ``MDQWRouter.route`` added
    ``node_cost(neighbour)`` on *every* step, including the final step into the
    destination, so the two reported figures disagreed by exactly
    ``node_cost(destination)``. The search now skips that term on the step into
    the target, restoring the documented identity.

    This does not affect which path is selected: the destination term is the
    same constant for every candidate path, so it could never change the
    winner. It only affected the reported ``total_cost``.
    """
    router = MDQWRouter(
        example, MDQWConfig(weights=RoutingWeights(hop_penalty=dominant_hop_penalty))
    )
    result = router.route("A", "F")

    gap = result.total_cost - (result.edge_cost + result.node_cost)
    assert gap == pytest.approx(0.0, abs=COST_TOLERANCE)
    assert router.node_cost(result.target) > 0.0, (
        "the fixture must give the destination a non-zero penalty for this test "
        "to be meaningful"
    )
    assert result.path[1:-1], "the fixture must route through at least one relay"


def test_repeater_penalty_makes_the_repeater_branch_strictly_more_expensive(
    repeater_pair: QuantumNetwork,
) -> None:
    """Equal-hop branches through a repeater and through a plain node.

    The two branches carry identical link metrics, so the whole difference is the
    node term. With ``repeater_penalty > hop_penalty`` the route through the
    repeater R must be strictly more expensive, and the router must avoid it.
    """
    net = repeater_pair
    weights = RoutingWeights(hop_penalty=0.2, repeater_penalty=0.8)
    router = MDQWRouter(net, MDQWConfig(weights=weights))

    via_plain = (
        _manual_edge_cost(router, "A", "P")
        + router.node_cost("P")
        + _manual_edge_cost(router, "P", "Z")
    )
    via_repeater = (
        _manual_edge_cost(router, "A", "R")
        + router.node_cost("R")
        + _manual_edge_cost(router, "R", "Z")
    )
    # Both branches cost exactly the same in links ...
    assert _manual_edge_cost(router, "A", "P") == _manual_edge_cost(router, "A", "R")
    assert _manual_edge_cost(router, "P", "Z") == _manual_edge_cost(router, "R", "Z")
    # ... so the node term alone decides, and it must favour the plain node.
    assert router.node_cost("R") > router.node_cost("P")
    assert via_repeater > via_plain

    result = router.route("A", "Z")
    _assert_valid_path(net, result)
    assert result.path == ("A", "P", "Z")
    assert result.hop_count == 2
    assert result.node_cost == pytest.approx(router.node_cost("P"), abs=COST_TOLERANCE)
    assert result.edge_cost == pytest.approx(
        _manual_edge_cost(router, "A", "P") + _manual_edge_cost(router, "P", "Z"),
        abs=COST_TOLERANCE,
    )


def test_equal_penalties_make_the_two_branches_tie(repeater_pair: QuantumNetwork) -> None:
    """The repeater penalty is the *only* thing separating the two branches.

    With ``repeater_penalty == hop_penalty`` both branches have identical cost, so
    either is acceptable; what must not happen is one of them being cheaper.
    """
    weights = RoutingWeights(hop_penalty=0.4, repeater_penalty=0.4)
    router = MDQWRouter(repeater_pair, MDQWConfig(weights=weights))

    assert router.node_cost("P") == pytest.approx(router.node_cost("R"))
    via_plain = (
        _manual_edge_cost(router, "A", "P")
        + router.node_cost("P")
        + _manual_edge_cost(router, "P", "Z")
    )
    via_repeater = (
        _manual_edge_cost(router, "A", "R")
        + router.node_cost("R")
        + _manual_edge_cost(router, "R", "Z")
    )
    assert via_plain == pytest.approx(via_repeater, abs=COST_TOLERANCE)

    result = router.route("A", "Z")
    assert result.hop_count == 2
    _assert_valid_path(repeater_pair, result)


# ---------------------------------------------------------------------------
# 7. The quantum-walk prior breaks optimality -- and we assert that it does
# ---------------------------------------------------------------------------


def _extreme_prior(net: QuantumNetwork, focus: str) -> dict:
    """A prior of 1.0 on ``focus`` and its neighbours, 0.0 everywhere else."""
    focused = {focus} | set(net.graph.neighbors(focus))
    return {node: (1.0 if node in focused else 0.0) for node in net.nodes}


def test_walk_prior_breaks_optimality_and_says_so(example: QuantumNetwork) -> None:
    """With an extreme prior the route is worse, and the result admits it.

    ``use_qw_prior=True`` adds a non-negative node term, so the search minimises
    a *different* objective from Dijkstra's. The module's honesty contract is
    that it says so: ``is_cost_optimal`` is ``False`` and the notes name the
    prior as a heuristic. The prior is aimed at D and its neighbours (C and F),
    which are exactly the intermediate nodes of the cheapest A->F route, so the
    answer must actually move.
    """
    router = MDQWRouter(
        example,
        MDQWConfig(use_qw_prior=True, qw_prior_weight=100.0),
        prior=_extreme_prior(example, "D"),
    )
    result = router.route("A", "F")
    baseline = DijkstraBaseline(example).route("A", "F")

    # 1. It claims no optimality, and says why.
    assert result.is_cost_optimal is False
    assert "heuristic" in result.notes
    assert "MDQW" in result.algorithm
    assert "QW-prior" in result.algorithm

    # 2. The path is still a real, connected path.
    _assert_valid_path(example, result)

    # 3. The prior actually moved the decision off the optimal-for-C(e) path.
    assert result.path != baseline.path

    # 4. THE honesty invariant: a non-negative node term can only make the pure
    #    edge cost worse or equal, never better, because Dijkstra's answer is the
    #    global minimum of C(e) over all simple paths and every C(e) >= 0.
    assert result.edge_cost >= baseline.edge_cost - COST_TOLERANCE
    assert result.edge_cost > baseline.edge_cost


def test_walk_prior_never_beats_dijkstra_on_edge_cost(example: QuantumNetwork) -> None:
    """Sweep several priors and weights; the inequality must hold every time.

    Strengthens the single-case test above: the invariant is structural, not an
    accident of one configuration.
    """
    baseline = DijkstraBaseline(example)
    for focus in ("A", "D", "E", "F"):
        for weight in (0.5, 5.0, 250.0):
            router = MDQWRouter(
                example,
                MDQWConfig(use_qw_prior=True, qw_prior_weight=weight),
                prior=_extreme_prior(example, focus),
            )
            for source, target in (("A", "F"), ("A", "D"), ("B", "F")):
                result = router.route(source, target)
                _assert_valid_path(example, result)
                assert result.edge_cost >= (
                    baseline.route(source, target).edge_cost - COST_TOLERANCE
                ), (focus, weight, source, target)


def test_prior_is_ignored_when_use_qw_prior_is_false(example: QuantumNetwork) -> None:
    """The flag, not the presence of a prior, is what activates the heuristic.

    Guards the honesty contract in the other direction: passing a prior must not
    silently downgrade an optimal router.
    """
    baseline = DijkstraBaseline(example).route("A", "F")
    router = MDQWRouter(
        example,
        MDQWConfig(use_qw_prior=False, qw_prior_weight=100.0),
        prior=_extreme_prior(example, "D"),
    )
    result = router.route("A", "F")

    assert result.is_cost_optimal is True
    assert result.node_cost == pytest.approx(0.0)
    assert result.edge_cost == pytest.approx(baseline.edge_cost, abs=COST_TOLERANCE)


# ---------------------------------------------------------------------------
# 8. quantum_walk_prior integration
# ---------------------------------------------------------------------------


def test_quantum_walk_prior_feeds_the_router(example: QuantumNetwork) -> None:
    """End-to-end: walk proxy -> walk prior -> router node term.

    ``walkable_proxy`` produces the regular surrogate the coined walk needs, and
    ``quantum_walk_prior`` turns it into a ``node -> salience`` mapping. With a
    non-zero ``qw_prior_weight`` that mapping must show up as a strictly positive
    node cost on at least one intermediate node, which is the whole mechanism.
    """
    proxy, strategy = walkable_proxy(example.to_networkx())
    assert strategy in PROXY_STRATEGIES

    prior = quantum_walk_prior(proxy, steps=6)
    assert set(prior) == set(example.nodes)
    assert all(0.0 <= value <= 1.0 for value in prior.values())

    weight = 0.5
    router = MDQWRouter(
        example, MDQWConfig(use_qw_prior=True, qw_prior_weight=weight), prior=prior
    )
    result = router.route("A", "F")
    _assert_valid_path(example, result)

    intermediates = list(result.path[1:-1])
    assert intermediates, "the fixture must produce a multi-hop route"
    assert any(router.node_cost(node) > 0.0 for node in intermediates)
    assert result.node_cost == pytest.approx(
        _intermediate_node_cost(router, result), abs=COST_TOLERANCE
    )
    assert result.node_cost == pytest.approx(
        weight * sum(prior[node] for node in intermediates), abs=COST_TOLERANCE
    )
    # The walk prior is a heuristic, so the guarantee is withdrawn.
    assert result.is_cost_optimal is False
    assert "heuristic" in result.notes
    # And the honesty invariant still holds.
    assert result.edge_cost >= (
        DijkstraBaseline(example).route("A", "F").edge_cost - COST_TOLERANCE
    )


def test_zero_prior_weight_leaves_the_node_term_at_the_hop_penalty(example: QuantumNetwork) -> None:
    """The prior must be scaled by its weight, not added unconditionally."""
    proxy, _ = walkable_proxy(example.to_networkx())
    prior = quantum_walk_prior(proxy, steps=4)
    weights = RoutingWeights(hop_penalty=0.3)

    plain = MDQWRouter(example, MDQWConfig(weights=weights))
    scaled = MDQWRouter(
        example, MDQWConfig(weights=weights, use_qw_prior=True, qw_prior_weight=0.0),
        prior=prior,
    )
    for node in example.nodes:
        assert plain.node_cost(node) == pytest.approx(scaled.node_cost(node))


# ---------------------------------------------------------------------------
# 9. Prior validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prior",
    [
        {},
        {"ZZ": 0.5},
        {"A": 0.1, "NOPE": 0.2},
        {"B": -1.0},
        {"B": -1e-12},
        {"B": float("nan")},
        {"B": float("inf")},
        {"B": float("-inf")},
    ],
    ids=[
        "empty",
        "unknown-node",
        "one-unknown-node",
        "negative",
        "tiny-negative",
        "nan",
        "inf",
        "-inf",
    ],
)
def test_router_rejects_an_invalid_prior(example: QuantumNetwork, prior: dict) -> None:
    """A prior that is empty, unknown or negative must be refused at construction.

    These are silent-corruption risks otherwise: a negative prior would reward
    visiting a node, and a ``nan`` would poison every cost comparison.
    """
    with pytest.raises(RoutingError):
        MDQWRouter(
            example, MDQWConfig(use_qw_prior=True, qw_prior_weight=1.0), prior=prior
        )


def test_router_accepts_none_as_the_disabled_prior(example: QuantumNetwork) -> None:
    """``None`` is the documented way to say "no prior"; ``{}`` is an error."""
    router = MDQWRouter(example, MDQWConfig(), prior=None)
    assert router.prior is None
    assert router.route("A", "F").node_cost == pytest.approx(0.0)


def test_router_rejects_an_empty_network() -> None:
    """An empty network has no normalisation table and no route."""
    with pytest.raises(RoutingError, match="empty network"):
        MDQWRouter(QuantumNetwork(name="empty"))


# ---------------------------------------------------------------------------
# 10. walkable_proxy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder",
    [
        pytest.param(lambda: build_example_network().to_networkx(), id="example-network"),
        pytest.param(lambda: random_network(12, 3, seed=99).to_networkx(), id="random-12"),
        pytest.param(lambda: nx.cycle_graph(6), id="cycle-6"),
        pytest.param(lambda: nx.path_graph(5), id="path-5"),
        pytest.param(lambda: nx.complete_graph(4), id="complete-4"),
    ],
)
def test_walkable_proxy_returns_a_regular_graph_over_the_same_nodes(builder) -> None:
    """The surrogate must be ``m``-regular, because the coined walk requires it.

    ``shunt_decomposition`` refuses a non-regular graph outright, so a proxy that
    were not regular would break ``quantum_walk_prior`` -- and with it the router
    prior. The node *set* must be preserved because the prior is keyed by the
    real network's node names.
    """
    graph = builder()
    proxy, strategy = walkable_proxy(graph)

    assert strategy in PROXY_STRATEGIES
    assert set(proxy.nodes) == set(graph.nodes)
    assert len(proxy.nodes) == len(graph)

    degrees = set(dict(proxy.degree()).values())
    assert len(degrees) == 1, degrees
    # A coined walk needs m >= 2: degree 0 or 1 has no coin to speak of.
    assert min(degrees) >= 2


@pytest.mark.parametrize(
    ("builder", "size"),
    [
        pytest.param(nx.cycle_graph, 6, id="cycle-6"),
        pytest.param(nx.complete_graph, 5, id="complete-5"),
        pytest.param(nx.cycle_graph, 9, id="cycle-9"),
    ],
)
def test_walkable_proxy_returns_regular_graphs_unchanged(builder, size: int) -> None:
    """An already-regular input must be passed straight through, not rebuilt."""
    graph = builder(size)
    proxy, strategy = walkable_proxy(graph)

    assert strategy == "already-regular"
    assert proxy is graph


def test_walkable_proxy_falls_back_to_a_complete_graph() -> None:
    """A tree has no cycle cover, so the documented fallback is the complete graph.

    ``nx.path_graph(5)`` is the smallest useful case: maximum-cardinality
    matching cannot 2-factor it, and the result is ``(n - 1)``-regular.
    """
    proxy, strategy = walkable_proxy(nx.path_graph(5))
    assert strategy == "complete-graph-fallback"
    assert len(set(dict(proxy.degree()).values())) == 1
    assert set(dict(proxy.degree()).values()) == {4}


def test_walkable_proxy_rejects_a_single_node_graph() -> None:
    """A one-node graph cannot carry a coin, so it is refused up front."""
    with pytest.raises(ValueError, match="at least two nodes"):
        walkable_proxy(nx.empty_graph(1))


def test_walkable_proxy_output_feeds_the_walk(example: QuantumNetwork) -> None:
    """The proxy must actually be usable: a walk and a prior over it must work."""
    proxy, _ = walkable_proxy(example.to_networkx())
    states = quantum_walk_prior(proxy, steps=4)
    assert set(states) == set(example.nodes)
    assert max(states.values()) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 11. RouteResult
# ---------------------------------------------------------------------------


def test_route_result_describe_and_to_record(example: QuantumNetwork) -> None:
    """Both reporting helpers must expose the documented fields.

    ``to_record`` is the schema every comparison table is built from, so an
    accidental rename would silently drop a column rather than fail.
    """
    result = DijkstraBaseline(example).route("A", "F")

    text = result.describe()
    assert "Selected path" in text
    assert "A -> B -> C -> F" in text
    assert "Dijkstra" in text

    record = result.to_record()
    assert set(record) == {
        "algorithm",
        "source",
        "target",
        "path",
        "hop_count",
        "total_cost",
        "edge_cost",
        "node_cost",
        "total_distance_km",
        "total_noise",
        "mean_noise",
        "estimated_fidelity",
        "total_latency_ms",
        "is_cost_optimal",
    }
    assert record["path"] == "A -> B -> C -> F"
    assert record["hop_count"] == 3
    assert record["source"] == "A" and record["target"] == "F"


def test_route_result_len_is_the_node_count(example: QuantumNetwork) -> None:
    """``len(result)`` counts *nodes*, not hops -- that is what it is documented as."""
    result = DijkstraBaseline(example).route("A", "F")
    assert len(result) == len(result.path) == 4
    assert len(result) == result.hop_count + 1


def test_route_result_metric_breakdown_is_self_consistent(example: QuantumNetwork) -> None:
    """The reported aggregates must be sums over the reported edges.

    ``total_cost`` is asserted against ``edge_cost`` only because the node terms
    are zero here; the destination-term gap is covered separately.
    """
    router = MDQWRouter(example, MDQWConfig(mode="none"))
    result = router.route("A", "F")

    assert len(result.edges) == result.hop_count
    assert result.edge_cost == pytest.approx(sum(edge.cost for edge in result.edges))
    assert result.total_distance_km == pytest.approx(
        sum(edge.distance_km for edge in result.edges)
    )
    assert result.total_latency_ms == pytest.approx(
        sum(edge.latency_ms for edge in result.edges)
    )
    assert result.total_noise == pytest.approx(sum(edge.noise for edge in result.edges))
    assert result.mean_noise == pytest.approx(result.total_noise / result.hop_count)
    assert result.estimated_fidelity == pytest.approx(
        math.prod(edge.fidelity for edge in result.edges)
    )
    # Consecutive edges must chain into the reported path.
    assert (result.edges[0].source,) + tuple(
        edge.target for edge in result.edges
    ) == result.path


@pytest.mark.parametrize("attenuation_length_km", [50.0, 10.0, 1000.0])
def test_attenuation_fidelity_never_exceeds_the_product_model(
    example: QuantumNetwork, attenuation_length_km: float
) -> None:
    """Attenuation can only *reduce* fidelity.

    Each hop contributes ``f_e * exp(-d_e / L)`` and ``exp(-x) <= 1``, so the
    attenuated estimate is bounded above by the plain product of link fidelities.
    """
    result = DijkstraBaseline(example).route("A", "F")
    attenuated = result.attenuation_fidelity(attenuation_length_km)

    assert 0.0 < attenuated <= result.estimated_fidelity
    assert attenuated == pytest.approx(
        math.prod(
            edge.fidelity * math.exp(-edge.distance_km / attenuation_length_km)
            for edge in result.edges
        ),
        abs=COST_TOLERANCE,
    )


def test_attenuation_fidelity_gets_worse_as_the_attenuation_length_shrinks(
    example: QuantumNetwork,
) -> None:
    """Monotone in ``L``: a shorter characteristic length must mean more loss."""
    result = DijkstraBaseline(example).route("A", "F")
    values = [result.attenuation_fidelity(length) for length in (5000.0, 500.0, 50.0, 5.0)]
    assert all(earlier > later for earlier, later in zip(values, values[1:]))


@pytest.mark.parametrize("bad_length", [0.0, -1.0, -1e-12])
def test_attenuation_fidelity_rejects_a_non_positive_length(
    example: QuantumNetwork, bad_length: float
) -> None:
    """``exp(-d / L)`` is undefined at ``L = 0`` and nonsense for ``L < 0``."""
    result = DijkstraBaseline(example).route("A", "F")
    with pytest.raises(ValueError, match="must be positive"):
        result.attenuation_fidelity(bad_length)


# ---------------------------------------------------------------------------
# 12. compare_routes / summarise_comparison
# ---------------------------------------------------------------------------


def test_compare_routes_accepts_a_mapping(example: QuantumNetwork, dominant_hop_penalty: float) -> None:
    """A labelled mapping must produce one row per route plus the delta columns."""
    plain = MDQWRouter(example, MDQWConfig(weights=RoutingWeights(hop_penalty=0.0)))
    heavy = MDQWRouter(
        example, MDQWConfig(weights=RoutingWeights(hop_penalty=dominant_hop_penalty))
    )
    frame = compare_routes(
        {
            "Dijkstra": DijkstraBaseline(example).route("A", "F"),
            "MDQW-cheap": plain.route("A", "F"),
            "MDQW-hop-heavy": heavy.route("A", "F"),
        }
    )

    assert list(frame["label"]) == ["Dijkstra", "MDQW-cheap", "MDQW-hop-heavy"]
    assert "edge_cost_delta_vs_dijkstra" in frame.columns
    assert frame["edge_cost_delta_vs_dijkstra"].iloc[0] == 0.0
    assert frame["edge_cost_delta_vs_dijkstra"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert "hop_delta_vs_dijkstra" in frame.columns
    assert "fidelity_delta_vs_dijkstra" in frame.columns


def test_compare_routes_accepts_a_sequence(example: QuantumNetwork, dominant_hop_penalty: float) -> None:
    """A bare list is labelled by each result's ``algorithm`` field.

    The baseline is still detected, because the Dijkstra algorithm name contains
    "Dijkstra"; that is the only thing the delta columns key off.
    """
    heavy = MDQWRouter(
        example, MDQWConfig(weights=RoutingWeights(hop_penalty=dominant_hop_penalty))
    )
    frame = compare_routes(
        [DijkstraBaseline(example).route("A", "F"), heavy.route("A", "F")]
    )

    assert len(frame) == 2
    assert "label" in frame.columns
    assert frame["label"].iloc[0] == "Dijkstra (classical baseline)"
    assert frame["edge_cost_delta_vs_dijkstra"].iloc[0] == 0.0
    assert frame["edge_cost_delta_vs_dijkstra"].iloc[1] > 0.0


def test_summarise_comparison_counts_sum_to_the_non_baseline_rows(
    example: QuantumNetwork, dominant_hop_penalty: float
) -> None:
    """lower + higher + tied must equal the number of non-baseline rows exactly.

    Any row left unclassified would mean a configuration silently dropped from
    the report.
    """
    plain = MDQWRouter(example, MDQWConfig(weights=RoutingWeights(hop_penalty=0.0)))
    heavy = MDQWRouter(
        example, MDQWConfig(weights=RoutingWeights(hop_penalty=dominant_hop_penalty))
    )
    heavier = MDQWRouter(
        example, MDQWConfig(weights=RoutingWeights(hop_penalty=100.0 * dominant_hop_penalty))
    )
    frame = compare_routes(
        {
            "Dijkstra": DijkstraBaseline(example).route("A", "F"),
            "MDQW-cheap": plain.route("A", "F"),
            "MDQW-hop-heavy": heavy.route("A", "F"),
            "MDQW-very-hop-heavy": heavier.route("A", "F"),
        }
    )

    summary = summarise_comparison(frame)
    assert len(summary) == 1
    record = summary.iloc[0]
    assert record["configurations"] == 3
    assert (
        record["mdqw_lower_edge_cost"]
        + record["dijkstra_lower_edge_cost"]
        + record["tied"]
    ) == 3
    # The cheap MDQW configuration ties with Dijkstra (same objective).
    assert record["tied"] == 1
    assert record["dijkstra_lower_edge_cost"] == 2
    assert record["mdqw_lower_edge_cost"] == 0


def test_verdict_reports_dijkstra_winning_without_claiming_superiority(
    example: QuantumNetwork, dominant_hop_penalty: float
) -> None:
    """When Dijkstra wins the verdict must say so -- and stay instance-scoped.

    The module is explicit that no algorithm is presented as universally better,
    so the wording is part of the contract: "Dijkstra ... on this instance",
    never a bare claim of superiority.
    """
    heavy = MDQWRouter(
        example, MDQWConfig(weights=RoutingWeights(hop_penalty=dominant_hop_penalty))
    )
    frame = compare_routes(
        {
            "Dijkstra": DijkstraBaseline(example).route("A", "F"),
            "MDQW-hop-heavy": heavy.route("A", "F"),
        }
    )
    verdict = str(summarise_comparison(frame).iloc[0]["verdict"])

    assert "Dijkstra" in verdict
    assert "MDQW achieved a lower link cost" not in verdict
    assert "this instance" in verdict
    assert "better" not in verdict.lower()


def test_verdict_reports_a_tie_when_the_objectives_coincide(example: QuantumNetwork) -> None:
    """With no node term MDQW and Dijkstra optimise the same thing, so they tie."""
    router = MDQWRouter(example, MDQWConfig(weights=RoutingWeights(hop_penalty=0.0)))
    frame = compare_routes(
        {
            "Dijkstra": DijkstraBaseline(example).route("A", "F"),
            "MDQW": router.route("A", "F"),
        }
    )
    record = summarise_comparison(frame).iloc[0]

    assert record["mdqw_lower_edge_cost"] == 0
    assert record["dijkstra_lower_edge_cost"] == 0
    assert record["tied"] == 1
    assert "tied" in str(record["verdict"]).lower()


def test_summarise_comparison_rejects_a_frame_without_a_baseline(example: QuantumNetwork) -> None:
    """A comparison with no Dijkstra row cannot be scored against one."""
    router = MDQWRouter(example)
    frame = compare_routes({"MDQW": router.route("A", "F")})
    with pytest.raises(RoutingError, match="no Dijkstra baseline"):
        summarise_comparison(frame)


def test_summarise_comparison_rejects_an_empty_frame(example: QuantumNetwork) -> None:
    frame = compare_routes({"Dijkstra": DijkstraBaseline(example).route("A", "F")})
    with pytest.raises(RoutingError, match="empty"):
        summarise_comparison(frame.iloc[0:0])


# ---------------------------------------------------------------------------
# 13. Error handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["none", "astar"])
def test_route_rejects_unknown_endpoints(example: QuantumNetwork, mode: str) -> None:
    """An unknown node is a caller bug and must raise, not return an empty route."""
    router = MDQWRouter(example, MDQWConfig(mode=mode))
    with pytest.raises(KeyError, match="unknown source"):
        router.route("Q", "F")
    with pytest.raises(KeyError, match="unknown target"):
        router.route("A", "Q")


def test_dijkstra_rejects_unknown_endpoints(example: QuantumNetwork) -> None:
    baseline = DijkstraBaseline(example)
    with pytest.raises(KeyError, match="unknown source"):
        baseline.route("Q", "F")
    with pytest.raises(KeyError, match="unknown target"):
        baseline.route("A", "Q")


def test_route_reports_a_disconnected_network_as_a_routing_error() -> None:
    """Two components with no path between them: a ``RoutingError``, not ``False``.

    The error must name the endpoint pair so a caller can tell which pair of the
    sweep is infeasible.
    """
    net = QuantumNetwork(name="split")
    for node in ("X", "Y", "P", "Q"):
        net.add_node(node)
    net.add_link("X", "Y", EdgeMetrics(10.0, 0.1, 1.0, 0.99, 0.99))
    net.add_link("P", "Q", EdgeMetrics(10.0, 0.1, 1.0, 0.99, 0.99))

    router = MDQWRouter(net)
    with pytest.raises(RoutingError, match="X.*P"):
        router.route("X", "P")
    with pytest.raises(RoutingError, match="X.*P"):
        DijkstraBaseline(net).route("X", "P")

    # Within a component everything still works.
    assert router.route("X", "Y").path == ("X", "Y")


def test_max_hops_below_the_minimum_raises_naming_max_hops(example: QuantumNetwork) -> None:
    """A ``max_hops`` budget of 1 cannot admit A->F, whose minimum is 2 hops.

    The constrained search legitimately returns *no* route even though one
    exists, so the message has to say it was the hop budget that failed rather
    than claiming the network is disconnected.
    """
    assert nx.shortest_path_length(example.graph, "A", "F") == 2

    router = MDQWRouter(example, MDQWConfig(max_hops=1))
    with pytest.raises(RoutingError, match="max_hops"):
        router.route("A", "F")

    # One more hop is enough, and the message stops appearing.
    generous = MDQWRouter(example, MDQWConfig(max_hops=2))
    assert generous.route("A", "F").hop_count == 2


def test_max_hops_is_respected_when_it_is_generous(example: QuantumNetwork) -> None:
    router = MDQWRouter(example, MDQWConfig(max_hops=3))
    result = router.route("A", "F")
    assert result.hop_count <= 3
    assert "maxhops=3" in result.algorithm


def test_dijkstra_rejects_an_empty_network() -> None:
    with pytest.raises(RoutingError, match="empty network"):
        DijkstraBaseline(QuantumNetwork(name="empty"))


# ---------------------------------------------------------------------------
# 14. enumerate_routes
# ---------------------------------------------------------------------------


def test_enumerate_routes_on_a_to_f(example: QuantumNetwork) -> None:
    """The A->F search space is exactly three simple paths, cheapest first.

    Asserting the *set* of paths pins the topology; asserting the ordering and
    the single ``selected`` row pins the bookkeeping that makes the table
    readable.
    """
    frame = enumerate_routes(example, "A", "F")

    assert len(frame) == 3
    assert set(frame["path"]) == {"A -> B -> C -> F", "A -> B -> C -> D -> F", "A -> E -> F"}

    # Ascending by edge_cost.
    costs = list(frame["edge_cost"])
    assert costs == sorted(costs)

    # Exactly one row is marked selected, and it is the cheapest one.
    assert int(frame["selected"].sum()) == 1
    assert frame.loc[frame["selected"], "edge_cost"].iloc[0] == pytest.approx(
        float(frame["edge_cost"].min())
    )

    # The selected row agrees with the classical baseline.
    baseline = DijkstraBaseline(example).route("A", "F")
    selected = frame.loc[frame["selected"]].iloc[0]
    assert selected["path"] == "A -> B -> C -> F"
    assert selected["edge_cost"] == pytest.approx(baseline.edge_cost, abs=COST_TOLERANCE)
    assert selected["hop_count"] == baseline.hop_count == 3

    # The enumerated rows are themselves honest about optimality.
    assert not frame["is_cost_optimal"].any()
    assert not frame["is_dijkstra_baseline"].any()


def test_enumerate_routes_rows_are_consistent_with_the_router(example: QuantumNetwork) -> None:
    """Every enumerated path must be a real path with a recomputable cost."""
    router = MDQWRouter(example)
    frame = enumerate_routes(example, "A", "F")

    for _, row in frame.iterrows():
        path = tuple(row["path"].split(" -> "))
        assert all(
            example.graph.has_edge(u, v) for u, v in zip(path, path[1:])
        ), row["path"]
        assert row["edge_cost"] == pytest.approx(_route_edge_cost(router, path), abs=COST_TOLERANCE)
        assert row["hop_count"] == len(path) - 1


def test_enumerate_routes_reports_unreachable_endpoints() -> None:
    """A disconnected pair is reported as a ``RoutingError`` with context."""
    net = QuantumNetwork(name="split")
    for node in ("X", "Y", "P", "Q"):
        net.add_node(node)
    net.add_link("X", "Y", EdgeMetrics(10.0, 0.1, 1.0, 0.99, 0.99))
    net.add_link("P", "Q", EdgeMetrics(10.0, 0.1, 1.0, 0.99, 0.99))
    with pytest.raises(RoutingError, match="cannot enumerate routes"):
        enumerate_routes(net, "X", "P")


def test_enumerate_routes_honours_its_path_budget(example: QuantumNetwork) -> None:
    """``max_paths`` bounds the work; the frame must respect the budget."""
    frame = enumerate_routes(example, "A", "F", max_paths=2)
    assert len(frame) == 2
    assert list(frame["edge_cost"]) == sorted(frame["edge_cost"])


def test_enumerate_routes_max_hops_drops_the_long_paths(example: QuantumNetwork) -> None:
    """``max_hops=3`` removes the 4-hop route from the A->F enumeration."""
    frame = enumerate_routes(example, "A", "F", max_hops=3)
    assert set(frame["path"]) == {"A -> B -> C -> F", "A -> E -> F"}