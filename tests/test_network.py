"""Tests for :mod:`src.network`.

The module is pure data modelling with no quantum computation and no I/O, so
every assertion here is exact: range validation, structural counts, table
round-trips, and error types. Nothing is statistical and nothing depends on wall
clock time.

The one test that carries real weight for the project as a whole is
``test_random_network_is_reproducible_for_a_fixed_seed``. Every figure, table
and sweep elsewhere in the repository is generated from ``random_network`` with
an explicit ``seed=``; if that seed stopped reproducing the same weighted graph,
every published number would silently become unreproducible. The test therefore
pins both halves of the contract -- same seed gives an identical edge table and
node set, a different seed does not.
"""

from __future__ import annotations

import math
import pathlib
import sys

import networkx as nx
import pandas as pd
import pytest

# Make ``src`` importable no matter how pytest is invoked (``pytest`` from the
# repository root, or ``python -m pytest`` from anywhere in the tree).
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.network import (  # noqa: E402  (import after the sys.path fix)
    METRIC_KEYS,
    EdgeMetrics,
    MetricValidationError,
    QuantumNetwork,
    build_example_network,
    minmax_normalize,
    random_network,
)

#: The five metric names in the canonical column order used by every table.
ALL_METRIC_COLUMNS = ("distance_km", "noise", "latency_ms", "fidelity", "reliability")


@pytest.fixture
def example() -> QuantumNetwork:
    """The hand-designed A-F network used throughout the project."""
    return build_example_network()


@pytest.fixture
def twin_link() -> QuantumNetwork:
    """A minimal network holding exactly one link, A-B, for mutation tests."""
    net = QuantumNetwork(name="twin-link")
    net.add_node("A")
    net.add_node("B")
    net.add_link("A", "B", EdgeMetrics(200.0, 0.4, 6.0, 0.81, 0.9))
    return net


@pytest.fixture
def seeded() -> QuantumNetwork:
    """A reproducible 12-node network; every use of it pins ``seed=99``."""
    return random_network(12, 3, seed=99)


# ---------------------------------------------------------------------------
# 1. EdgeMetrics construction and validation
# ---------------------------------------------------------------------------


def test_edge_metrics_accepts_in_range_values() -> None:
    """Interior values and both endpoints of every range must be accepted."""
    interior = EdgeMetrics(120.0, 0.05, 2.0, 0.97, 0.99)
    assert interior.distance_km == pytest.approx(120.0)
    assert interior.noise == pytest.approx(0.05)

    # The ranges are closed: 0 and 1 are legal for noise/fidelity/reliability,
    # and 0 is legal for both non-negative quantities.
    boundary = EdgeMetrics(0.0, 0.0, 0.0, 1.0, 1.0)
    assert boundary.as_dict() == {
        "distance_km": 0.0,
        "noise": 0.0,
        "latency_ms": 0.0,
        "fidelity": 1.0,
        "reliability": 1.0,
    }

    # Integers are accepted as real numbers and are coerced by ``as_dict``.
    integral = EdgeMetrics(100, 0, 1, 1, 1)
    assert all(isinstance(value, float) for value in integral.as_dict().values())


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"noise": 1.5}, "noise"),
        ({"fidelity": -0.1}, "fidelity"),
        ({"reliability": 2.0}, "reliability"),
        ({"distance_km": -1}, "distance_km"),
        ({"latency_ms": -0.5}, "latency_ms"),
    ],
    ids=["noise-above-one", "fidelity-below-zero", "reliability-above-one", "negative-distance", "negative-latency"],
)
def test_edge_metrics_rejects_out_of_range_values(kwargs: dict, expected: str) -> None:
    """Every physically implausible metric must be rejected at construction.

    Validation happens in ``__post_init__`` so an invalid link can never reach a
    router; the error names the offending field so a bad call site is obvious.
    """
    base = {"distance_km": 100.0, "noise": 0.1, "latency_ms": 1.0, "fidelity": 0.95, "reliability": 0.99}
    with pytest.raises(MetricValidationError, match=expected):
        EdgeMetrics(**{**base, **kwargs})


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "inf", "-inf"],
)
def test_edge_metrics_rejects_non_finite_values(value: float) -> None:
    """``nan``/``inf`` must not slip through the range checks.

    ``nan`` is the dangerous one: every ``nan < 0`` and ``0 <= nan <= 1``
    comparison is ``False``, so a naive range check would accept it and poison
    every normalisation downstream.
    """
    with pytest.raises(MetricValidationError, match="finite"):
        EdgeMetrics(100.0, value, 1.0, 0.95, 0.99)


def test_edge_metrics_rejects_non_numeric_type() -> None:
    """A numeric-looking string must not be silently coerced.

    ``"0.5"`` is deliberately rejected: metrics arrive from configuration and
    from parsed files, and silently accepting a string would push the type error
    into the normalisation tables instead of raising it at the boundary.
    """
    with pytest.raises(MetricValidationError, match="real number"):
        EdgeMetrics(100.0, "0.5", 1.0, 0.95, 0.99)


def test_metric_validation_error_is_a_value_error() -> None:
    """Callers that only catch ``ValueError`` must still catch this."""
    assert issubclass(MetricValidationError, ValueError)


def test_edge_metrics_is_frozen() -> None:
    """A router must not be able to mutate metrics behind the network's back."""
    from dataclasses import FrozenInstanceError

    metrics = EdgeMetrics(120.0, 0.05, 2.0, 0.97, 0.99)
    with pytest.raises(FrozenInstanceError):
        metrics.noise = 0.5


# ---------------------------------------------------------------------------
# 2. EdgeMetrics serialisation
# ---------------------------------------------------------------------------


def test_as_dict_returns_floats_and_from_dict_round_trips() -> None:
    """``as_dict``/``from_dict`` must be an exact inverse pair.

    The router's normalisation tables are keyed by ``float`` metric value, so a
    round-trip that changed ``0.97`` into ``Decimal('0.97')`` or kept it as an
    ``int`` would produce a ``KeyError`` deep inside ``edge_cost``.
    """
    metrics = EdgeMetrics(120, 0.05, 2, 0.97, 0.99)
    payload = metrics.as_dict()

    assert set(payload) == set(ALL_METRIC_COLUMNS)
    assert all(type(value) is float for value in payload.values())

    restored = EdgeMetrics.from_dict(payload)
    assert restored == metrics
    assert restored.as_dict() == payload


def test_from_dict_rejects_a_missing_key() -> None:
    """A partially populated mapping must fail loudly, not default silently."""
    payload = EdgeMetrics(120.0, 0.05, 2.0, 0.97, 0.99).as_dict()
    for key in ALL_METRIC_COLUMNS:
        partial = {k: v for k, v in payload.items() if k != key}
        with pytest.raises(MetricValidationError, match=key):
            EdgeMetrics.from_dict(partial)


def test_from_dict_ignores_extra_keys() -> None:
    """Edge attributes carried alongside the metrics must not break loading.

    ``add_link`` stores ``EdgeMetrics`` fields directly on the networkx edge, so
    a row read back out of a network can legitimately contain extra keys.
    """
    payload = EdgeMetrics(120.0, 0.05, 2.0, 0.97, 0.99).as_dict()
    noisy = {**payload, "label": "spine", "is_dijkstra_baseline": False}
    assert EdgeMetrics.from_dict(noisy) == EdgeMetrics.from_dict(payload)


def test_from_dict_still_validates() -> None:
    """Round-tripping must not become a validation bypass."""
    payload = EdgeMetrics(120.0, 0.05, 2.0, 0.97, 0.99).as_dict()
    with pytest.raises(MetricValidationError):
        EdgeMetrics.from_dict({**payload, "noise": 3.0})


# ---------------------------------------------------------------------------
# 3. QuantumNetwork basics
# ---------------------------------------------------------------------------


def test_build_example_network_has_the_documented_shape(example: QuantumNetwork) -> None:
    """Six nodes, seven links, connected, named A through F.

    Every other module's example depends on this exact topology, so the counts
    are asserted rather than derived.
    """
    assert len(example) == 6
    assert len(example.links) == 7
    assert example.is_connected()
    assert sorted(example.nodes) == ["A", "B", "C", "D", "E", "F"]


def test_build_example_network_supports_len_and_contains(example: QuantumNetwork) -> None:
    """``len(net)`` counts nodes and ``in`` tests membership of the node set."""
    assert len(example) == example.graph.number_of_nodes()
    for node in example.nodes:
        assert node in example
    assert "Z" not in example
    assert 42 not in example


def test_build_example_network_link_set_is_the_documented_one(example: QuantumNetwork) -> None:
    """The link set, compared order-independently.

    The topology is the two-branch figure from the module docstring: a clean
    upper branch A-B-C-D, a lossy lower branch A-E-F-D, and the cross-over link
    C-F that lets the two branches compete.
    """
    links = {frozenset(pair) for pair in example.links}
    expected = {
        frozenset(("A", "B")),
        frozenset(("B", "C")),
        frozenset(("C", "D")),
        frozenset(("A", "E")),
        frozenset(("E", "F")),
        frozenset(("D", "F")),
        frozenset(("C", "F")),
    }
    assert links == expected


def test_build_example_network_node_attributes(example: QuantumNetwork) -> None:
    """Endpoints carry 800 ms memories; E and F are flagged as repeaters.

    The router charges ``repeater_penalty`` for an intermediate node flagged
    ``is_repeater``, so a mislabelled node would silently change routing.
    """
    for node in ("A", "B", "C", "D"):
        attributes = example.node(node)
        assert attributes["memory_lifetime_ms"] == pytest.approx(800.0)
        assert attributes["is_repeater"] is False
    for node in ("E", "F"):
        attributes = example.node(node)
        assert attributes["memory_lifetime_ms"] == pytest.approx(2000.0)
        assert attributes["is_repeater"] is True


def test_add_node_defaults_are_filled_in() -> None:
    """A node added without attributes still gets the documented defaults."""
    net = QuantumNetwork(name="defaults")
    net.add_node("A")
    attributes = net.node("A")
    assert attributes == {"memory_lifetime_ms": 1000.0, "is_repeater": False}


# ---------------------------------------------------------------------------
# 4. Link mutation
# ---------------------------------------------------------------------------


def test_add_link_rejects_a_non_edge_metrics_payload() -> None:
    """``add_link`` must refuse a dict so links always carry a full record.

    A partially-populated dict would leave the normalisation tables missing a
    key and fail much later, with a much less useful message.
    """
    net = QuantumNetwork(name="strict")
    with pytest.raises(MetricValidationError, match="EdgeMetrics"):
        net.add_link("A", "B", {"noise": 0.1})
    with pytest.raises(ValueError):
        net.add_link("A", "B", None)
    assert net.links == []


def test_add_link_rejects_a_duplicate_link(example: QuantumNetwork) -> None:
    """Re-adding a link must not silently overwrite the metrics."""
    with pytest.raises(ValueError, match="already exists"):
        example.add_link("A", "B", EdgeMetrics(1.0, 0.1, 1.0, 0.99, 0.99))
    assert example.link_metrics("A", "B") == EdgeMetrics(120.0, 0.05, 2.0, 0.97, 0.99)


def test_add_link_creates_missing_endpoints() -> None:
    """Endpoints are auto-created, so a link cannot be half-added."""
    net = QuantumNetwork(name="auto")
    net.add_link("A", "B", EdgeMetrics(10.0, 0.1, 1.0, 0.99, 0.99))
    assert sorted(net.nodes) == ["A", "B"]
    assert net.link_metrics("A", "B").distance_km == pytest.approx(10.0)


def test_update_link_replaces_the_metrics(example: QuantumNetwork) -> None:
    """``update_link`` is the only sanctioned way to change a link in place."""
    replacement = EdgeMetrics(999.0, 0.01, 0.5, 0.999, 0.999)
    example.update_link("A", "B", replacement)
    assert example.link_metrics("A", "B") == replacement
    # The rest of the network is untouched.
    assert len(example.links) == 7
    assert example.link_metrics("B", "C") == EdgeMetrics(140.0, 0.06, 2.5, 0.96, 0.99)


def test_update_link_rejects_a_missing_link(example: QuantumNetwork) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        example.update_link("A", "Z", EdgeMetrics(1.0, 0.1, 1.0, 0.99, 0.99))


def test_update_link_rejects_a_non_edge_metrics_payload(example: QuantumNetwork) -> None:
    with pytest.raises(MetricValidationError, match="EdgeMetrics"):
        example.update_link("A", "B", {"noise": 0.1})


def test_add_midpoint_node_subdivides_the_link(twin_link: QuantumNetwork) -> None:
    """``u-v`` becomes ``u-M-v`` with the metrics split and ``M`` a repeater.

    Distance, noise and latency are halved and fidelity becomes
    ``sqrt(f)``, which is the multiplicative-composition rule for two equal
    hops -- so ``2 * sqrt(f) != f`` and the model is explicit that subdividing a
    link is *not* cost-neutral in the fidelity term.
    """
    twin_link.add_midpoint_node("A", "B", "M")

    assert not twin_link.graph.has_edge("A", "B")
    assert twin_link.graph.has_edge("A", "M")
    assert twin_link.graph.has_edge("M", "B")
    assert len(twin_link.links) == 2

    assert twin_link.node("M")["is_repeater"] is True

    for u, v in (("A", "M"), ("M", "B")):
        half = twin_link.link_metrics(u, v)
        assert half.distance_km == pytest.approx(100.0)
        assert half.noise == pytest.approx(0.2)
        assert half.latency_ms == pytest.approx(3.0)
        assert half.fidelity == pytest.approx(0.9)  # sqrt(0.81)
        assert half.reliability == pytest.approx(0.9)


def test_add_midpoint_node_rejects_a_missing_link(twin_link: QuantumNetwork) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        twin_link.add_midpoint_node("A", "Z", "M")
    assert "M" not in twin_link


# ---------------------------------------------------------------------------
# 5. edge_table / metric_ranges
# ---------------------------------------------------------------------------


def test_edge_table_has_the_exact_documented_columns(example: QuantumNetwork) -> None:
    """Column names and order are load-bearing: plots and CSVs key off them."""
    table = example.edge_table()
    assert list(table.columns) == ["source", "target", *ALL_METRIC_COLUMNS]
    assert METRIC_KEYS == ALL_METRIC_COLUMNS[:4]


def test_edge_table_has_one_row_per_link(seeded: QuantumNetwork, example: QuantumNetwork) -> None:
    """Every link appears exactly once; nothing is dropped or duplicated."""
    for net in (example, seeded):
        table = net.edge_table()
        assert len(table) == len(net.links)
        pairs = [frozenset((row["source"], row["target"])) for _, row in table.iterrows()]
        assert len(set(pairs)) == len(pairs)
        assert set(pairs) == {frozenset(link) for link in net.links}


def test_edge_table_values_round_trip_through_link_metrics(example: QuantumNetwork) -> None:
    """The table is a faithful view of the stored metrics.

    If the table and the edge records could disagree, every figure would be
    computed from numbers that no link actually carries.
    """
    table = example.edge_table()
    for _, row in table.iterrows():
        stored = example.link_metrics(row["source"], row["target"])
        for key in ALL_METRIC_COLUMNS:
            assert float(row[key]) == pytest.approx(getattr(stored, key))


def test_metric_ranges_match_the_edge_table(example: QuantumNetwork) -> None:
    """``metric_ranges`` is exactly the table's per-column min and max.

    These are the denominators of every normalisation, so a wrong range would
    silently rescale the whole cost function.
    """
    table = example.edge_table()
    ranges = example.metric_ranges()
    assert set(ranges) == set(ALL_METRIC_COLUMNS)
    for key, (low, high) in ranges.items():
        assert isinstance(low, float) and isinstance(high, float)
        assert low == pytest.approx(float(table[key].min()))
        assert high == pytest.approx(float(table[key].max()))
        assert low <= high


def test_metric_ranges_on_a_linkless_network_is_degenerate_not_empty() -> None:
    """A network with no links has *undefined* ranges, not zeroed ones.

    The columns still exist (the frame is built with an explicit column list) but
    every column is empty, so pandas reports ``NaN`` bounds rather than
    ``(0.0, 0.0)``. Pinning that here documents why a linkless network is not a
    usable routing input even though nothing raises.
    """
    net = QuantumNetwork(name="empty")
    net.add_node("A")
    assert net.edge_table().empty

    ranges = net.metric_ranges()
    assert set(ranges) == set(ALL_METRIC_COLUMNS)
    for key, (low, high) in ranges.items():
        assert math.isnan(low) and math.isnan(high), key


# ---------------------------------------------------------------------------
# 6. Lookup errors
# ---------------------------------------------------------------------------


def test_node_raises_for_an_unknown_node(example: QuantumNetwork) -> None:
    with pytest.raises(KeyError, match="unknown node"):
        example.node("Q")


def test_link_metrics_raises_for_an_unknown_link(example: QuantumNetwork) -> None:
    with pytest.raises(KeyError, match="unknown link"):
        example.link_metrics("A", "Q")


def test_has_path_raises_for_unknown_endpoints(example: QuantumNetwork) -> None:
    """Reachability of an unknown node is a caller bug, not a ``False``."""
    with pytest.raises(KeyError, match="unknown endpoint"):
        example.has_path("Q", "F")
    with pytest.raises(KeyError, match="unknown endpoint"):
        example.has_path("A", "Q")


def test_has_path_answers_correctly_for_known_nodes(example: QuantumNetwork) -> None:
    """Sanity of the reachability query the router relies on for its guard."""
    assert example.has_path("A", "F") is True
    assert example.has_path("E", "C") is True
    assert example.has_path("A", "A") is True  # zero-hop path to itself


def test_has_path_is_false_across_a_disconnect() -> None:
    """A disconnected network reports ``False`` rather than raising."""
    net = QuantumNetwork(name="split")
    for node in ("X", "Y", "P", "Q"):
        net.add_node(node)
    net.add_link("X", "Y", EdgeMetrics(10.0, 0.1, 1.0, 0.99, 0.99))
    net.add_link("P", "Q", EdgeMetrics(10.0, 0.1, 1.0, 0.99, 0.99))
    assert net.has_path("X", "Y") is True
    assert net.has_path("X", "P") is False
    assert net.is_connected() is False


# ---------------------------------------------------------------------------
# 7. minmax_normalize
# ---------------------------------------------------------------------------


def test_minmax_normalize_handles_empty_input() -> None:
    """Empty input maps to empty output instead of raising on ``.min()``."""
    assert minmax_normalize([]) == {}


def test_minmax_normalize_maps_a_constant_metric_to_zero() -> None:
    """A metric with no variation must contribute nothing, not an arbitrary constant.

    This is what keeps a degenerate sweep from being decided by a rounding
    artefact: if every link had identical fidelity, fidelity would drop out of
    the cost entirely rather than dominate it.
    """
    assert minmax_normalize([1, 1, 1]) == {1.0: 0.0}
    assert minmax_normalize([4.2]) == {4.2: 0.0}


def test_minmax_normalize_maps_onto_the_unit_interval() -> None:
    """The exact documented mapping for a three-value spread."""
    assert minmax_normalize([0, 5, 10]) == {0.0: 0.0, 5.0: 0.5, 10.0: 1.0}


def test_minmax_normalize_deduplicates_repeated_values() -> None:
    """Repeated values collapse to one key with the same normalised value."""
    normalised = minmax_normalize([2.0, 2.0, 6.0])
    assert normalised == {2.0: 0.0, 6.0: 1.0}


@pytest.mark.parametrize(
    "values",
    [[1.0, -0.5], [-1.0], [0.0, 3.0, -1e-9]],
    ids=["one-negative", "all-negative", "tiny-negative"],
)
def test_minmax_normalize_rejects_negative_values(values: list) -> None:
    """A negative metric has no meaning after normalisation to ``[0, 1]``."""
    with pytest.raises(ValueError, match="non-negative"):
        minmax_normalize(values)


@pytest.mark.parametrize(
    "values",
    [[1.0, float("nan")], [float("inf"), 1.0], [float("-inf")]],
    ids=["nan", "inf", "-inf"],
)
def test_minmax_normalize_rejects_non_finite_values(values: list) -> None:
    """``nan`` would poison every lookup in the router's normalisation table."""
    with pytest.raises(ValueError, match="non-finite"):
        minmax_normalize(values)


# ---------------------------------------------------------------------------
# 8. Determinism -- the reproducibility guarantee
# ---------------------------------------------------------------------------


def test_random_network_is_reproducible_for_a_fixed_seed() -> None:
    """The same seed must rebuild the same weighted graph, bit for bit.

    This is the guarantee the whole project rests on: every published table and
    figure is generated from a seeded ``random_network``. Both the topology (via
    networkx's seeded ``G(n, p)``) and the five metrics (via a seeded
    ``numpy.random.Generator``) have to agree across calls.
    """
    first = random_network(12, 3, seed=99)
    second = random_network(12, 3, seed=99)

    pd.testing.assert_frame_equal(first.edge_table(), second.edge_table())
    assert first.nodes == second.nodes
    assert first.links == second.links
    assert first.metric_ranges() == second.metric_ranges()
    # Node attributes carry a seeded random memory lifetime and repeater flag.
    assert [first.node(node) for node in first.nodes] == [
        second.node(node) for node in second.nodes
    ]


def test_a_different_seed_produces_a_different_network() -> None:
    """Guard against the seed being accepted and then ignored.

    Without this half, a ``seed=`` that silently did nothing would still pass the
    reproducibility test above.
    """
    first = random_network(12, 3, seed=99)
    other = random_network(12, 3, seed=100)

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(first.edge_table(), other.edge_table())


def test_seeded_network_respects_its_declared_ranges() -> None:
    """Every drawn metric must lie inside the range that was requested."""
    net = random_network(12, 3, seed=99)
    ranges = net.metric_ranges()
    assert ranges["distance_km"][0] >= 50.0
    assert ranges["distance_km"][1] <= 400.0
    assert ranges["noise"][0] >= 0.02
    assert ranges["noise"][1] <= 0.35
    assert ranges["fidelity"][0] >= 0.80
    assert ranges["fidelity"][1] <= 0.995
    assert ranges["reliability"][0] >= 0.85
    assert ranges["reliability"][1] <= 0.999


def test_seeded_network_node_identifiers_and_sizes(seeded: QuantumNetwork) -> None:
    """Node ids are ``"N0" .. "N{n-1}"``; that naming is what the walk prior keys on."""
    assert len(seeded) == 12
    assert sorted(seeded.nodes) == sorted(f"N{index}" for index in range(12))
    assert len(seeded.links) > 0


# ---------------------------------------------------------------------------
# 9. random_network validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_nodes", [1, 0, -5])
def test_random_network_rejects_too_few_nodes(n_nodes: int) -> None:
    with pytest.raises(ValueError, match="n_nodes"):
        random_network(n_nodes, 2, seed=1)


@pytest.mark.parametrize("avg_degree", [0, -1])
def test_random_network_rejects_a_degenerate_degree(avg_degree: int) -> None:
    """``avg_degree < 1`` would sample ``G(n, p = 0)``, i.e. no links at all."""
    with pytest.raises(ValueError, match="avg_degree"):
        random_network(6, avg_degree, seed=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"noise_range": (0.9, 0.1)},
        {"distance_range": (500.0, 10.0)},
        {"latency_range": (30.0, 2.0)},
        {"fidelity_range": (0.99, 0.5)},
        {"reliability_range": (0.9, 1.5)},
        {"noise_range": (-0.1, 0.3)},
        {"distance_range": (-5.0, 10.0)},
    ],
    ids=[
        "noise-inverted",
        "distance-inverted",
        "latency-inverted",
        "fidelity-inverted",
        "reliability-above-one",
        "noise-negative",
        "distance-negative",
    ],
)
def test_random_network_rejects_malformed_ranges(kwargs: dict) -> None:
    """A malformed or impossible range must be refused before any sampling.

    Otherwise ``rng.uniform(low, high)`` would silently return values outside the
    range the caller asked for, or produce an ``EdgeMetrics`` that is invalid.
    """
    with pytest.raises(ValueError, match="malformed"):
        random_network(8, 3, seed=1, **kwargs)


def test_random_network_minimum_size_is_usable() -> None:
    """The smallest accepted network still has at least one link to report."""
    net = random_network(2, 1, seed=1)
    assert len(net) == 2
    assert len(net.links) == 1


# ---------------------------------------------------------------------------
# 10. copy()
# ---------------------------------------------------------------------------


def test_copy_is_independent_of_the_original(example: QuantumNetwork) -> None:
    """Mutating a copy must not leak back into the source network.

    Sweeps clone a base network and perturb the clone. If ``copy()`` shared
    attribute dictionaries, every configuration in a sweep would silently
    inherit the previous one's edits.
    """
    original_table = example.edge_table()
    clone = example.copy()

    # Structural edits.
    clone.add_node("Z")
    clone.add_midpoint_node("A", "B", "M")

    # Metric edits.
    clone.update_link("C", "D", EdgeMetrics(1.0, 0.0, 0.0, 1.0, 1.0))

    # Node-attribute edits, made through the documented ``graph`` property.
    clone.graph.nodes["A"]["is_repeater"] = True

    assert "Z" not in example and "M" not in example
    assert example.graph.has_edge("A", "B")
    assert example.link_metrics("C", "D") == EdgeMetrics(110.0, 0.05, 2.0, 0.97, 0.99)
    assert example.node("A")["is_repeater"] is False

    # The original's table is untouched, and the clone agrees with it *before*
    # the edits -- i.e. copy() started from an exact snapshot.
    pd.testing.assert_frame_equal(original_table, example.edge_table())
    assert clone is not example
    assert clone.name == example.name


def test_copy_preserves_node_attributes(example: QuantumNetwork) -> None:
    """Attributes are copied, not just re-derived with defaults."""
    clone = example.copy()
    for node in example.nodes:
        assert clone.node(node) == example.node(node)


def test_node_returns_a_copy_of_the_attributes(example: QuantumNetwork) -> None:
    """``node()`` hands out a detached dict so callers cannot corrupt the graph."""
    attributes = example.node("A")
    attributes["is_repeater"] = True
    attributes["injected"] = 1
    assert example.node("A")["is_repeater"] is False
    assert "injected" not in example.node("A")


# ---------------------------------------------------------------------------
# 11. hop_lengths()
# ---------------------------------------------------------------------------


def test_hop_lengths_is_sorted_and_ends_at_the_diameter(example: QuantumNetwork) -> None:
    """The largest hop count equals the networkx diameter.

    ``hop_lengths`` is built from ``all_pairs_shortest_path_length``, so the
    entries are ``n`` self-pairs at distance ``0`` plus ``n * (n - 1)`` genuine
    node pairs. Asserting both the ordering and the maximum pins the meaning of
    the list, which the summary statistics in the docs rely on.
    """
    lengths = example.hop_lengths()

    assert all(isinstance(value, int) for value in lengths)
    assert lengths == sorted(lengths)
    assert min(lengths) == 0  # each node's distance to itself
    assert max(lengths) == nx.diameter(example.graph)
    assert len(lengths) == len(example) ** 2

    distinct_pairs = [value for value in lengths if value >= 1]
    assert len(distinct_pairs) == len(example) * (len(example) - 1)
    assert all(value >= 1 for value in distinct_pairs)


def test_hop_lengths_counts_every_ordered_pair_exactly_once(seeded: QuantumNetwork) -> None:
    """The multiset of hop counts must match an independent BFS enumeration."""
    expected: list[int] = []
    for source, targets in nx.all_pairs_shortest_path_length(seeded.graph):
        expected.extend(targets.values())
    assert seeded.hop_lengths() == sorted(expected)
    assert max(seeded.hop_lengths()) == nx.diameter(seeded.graph)


def test_hop_lengths_of_a_chain_matches_its_distances() -> None:
    """A 3-node chain has a diameter of 2: P-Q and Q-R at one hop, P-R at two,
    and three self-pairs at zero."""
    chain = QuantumNetwork(name="chain")
    for node in ("P", "Q", "R"):
        chain.add_node(node)
    metrics = EdgeMetrics(10.0, 0.1, 1.0, 0.99, 0.99)
    chain.add_link("P", "Q", metrics)
    chain.add_link("Q", "R", metrics)
    # P-Q and Q-R at one hop; P-R at two; three self-pairs at zero.
    assert chain.hop_lengths() == [0, 0, 0, 1, 1, 1, 1, 2, 2]