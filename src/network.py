"""Quantum communication network model for the BQT-MDQW prototype.

Scope and provenance
--------------------
Everything in this module is ``[PROPOSED]`` *data modelling* plus ``[SIMULATED]``
bookkeeping. No part of this file performs quantum computation, and no link
metric here is measured from a physical quantum channel. The edge attributes are
synthetic parameters chosen to make the routing problem non-trivial; see
:func:`build_example_network` and :func:`random_network`.

Design
------
A quantum communication network is represented as a weighted undirected graph
``G = (V, E)``:

* ``V`` -- quantum nodes (endpoints or repeaters).
* ``E`` -- quantum communication links.
* each link carries the five routing-relevant quantities declared in
  :class:`EdgeMetrics`.

The link metrics are deliberately *not* independent physical quantities. In a
real deployment ``distance`` drives attenuation, ``noise`` and ``reliability``
are consequences of it, and ``latency`` includes any classical feed-forward
round trip. This prototype keeps them as independent inputs so that the
weighting coefficients ``alpha, beta, gamma, delta`` remain independently
controllable; that is a modelling convenience, and it is stated as such
wherever the metrics are used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Hashable, Iterable, Mapping

import networkx as nx
import numpy as np
import pandas as pd

__all__ = [
    "EdgeMetrics",
    "QuantumNetwork",
    "minmax_normalize",
    "build_example_network",
    "random_network",
    "METRIC_KEYS",
]

#: Keys of the four per-link routing metrics, in the canonical order used by
#: every table, plot and DataFrame produced by this project.
METRIC_KEYS: tuple[str, ...] = ("distance_km", "noise", "latency_ms", "fidelity")


class MetricValidationError(ValueError):
    """Raised when a link metric is outside its physically plausible range."""


@dataclass(frozen=True, slots=True)
class EdgeMetrics:
    """Per-link routing metrics for one quantum communication link.

    Attributes:
        distance_km: Physical link length in kilometres. Non-negative.
        noise: Dimensionless noise severity in ``[0, 1]`` where ``0`` is ideal
            and ``1`` is maximal. This is a *routing* score, not a calibrated
            error rate; see :mod:`src.quantum_teleportation` for the separate
            error probability actually fed to the noise model.
        latency_ms: One-way classical/control latency in milliseconds.
            Non-negative.
        fidelity: Per-hop state-transfer fidelity in ``[0, 1]`` as declared by
            this prototype's link model. See the note in the module docstring:
            this is a model input, not a measurement.
        reliability: Link availability probability in ``[0, 1]``.

    All fields are validated on construction. The class is frozen so that a
    :class:`QuantumNetwork` cannot be mutated behind a router's back.
    """

    distance_km: float
    noise: float
    latency_ms: float
    fidelity: float
    reliability: float

    def __post_init__(self) -> None:
        for name in METRIC_KEYS:
            value = getattr(self, name)
            if not isinstance(value, (int, float, np.floating, np.integer)):
                raise MetricValidationError(
                    f"{name} must be a real number, got {type(value).__name__}"
                )
            if not np.isfinite(value):
                raise MetricValidationError(f"{name} must be finite, got {value}")
        if self.distance_km < 0:
            raise MetricValidationError(
                f"distance_km must be non-negative, got {self.distance_km}"
            )
        if self.latency_ms < 0:
            raise MetricValidationError(
                f"latency_ms must be non-negative, got {self.latency_ms}"
            )
        for name in ("noise", "fidelity", "reliability"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise MetricValidationError(
                    f"{name} must lie in [0, 1], got {value}"
                )

    def as_dict(self) -> dict[str, float]:
        """Return the metrics as a plain dictionary."""
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EdgeMetrics":
        """Build :class:`EdgeMetrics` from a mapping, ignoring extra keys.

        Args:
            data: Mapping that must contain every :class:`EdgeMetrics` field.

        Returns:
            A validated :class:`EdgeMetrics` instance.

        Raises:
            MetricValidationError: If a required key is missing.
        """
        missing = [k for k in cls.__dataclass_fields__ if k not in data]
        if missing:
            raise MetricValidationError(
                f"missing required edge metric(s): {', '.join(sorted(missing))}"
            )
        return cls(**{k: data[k] for k in cls.__dataclass_fields__})


class QuantumNetwork:
    """A weighted graph of quantum nodes and quantum communication links.

    The class is a thin, validating wrapper around :class:`networkx.Graph`. Its
    job is to guarantee that every link carries a complete
    :class:`EdgeMetrics` record and to provide the derived quantities the
    routing layer needs.

    Args:
        name: Human-readable network label, used in plot titles.

    Raises:
        MetricValidationError: If a link is added with a non-:class:`EdgeMetrics`
            payload.
    """

    def __init__(self, name: str = "quantum-network") -> None:
        self.name = name
        self._graph = nx.Graph(name=name)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def add_node(
        self,
        node: Hashable,
        *,
        memory_lifetime_ms: float = 1000.0,
        is_repeater: bool = False,
        **attributes: Any,
    ) -> None:
        """Add (or update) a quantum node.

        Args:
            node: Node identifier.
            memory_lifetime_ms: Coherence time of the node's quantum memory.
                This is a *declared* model parameter; the multi-hop simulation
                in :mod:`src.simulation` uses it to report a memory-decay
                estimate, and does not simulate actual decoherence.
            is_repeater: Whether the node is a repeater. When true the MDQW
                router adds the per-relay ``hop_penalty`` for this node.
            **attributes: Additional node attributes stored verbatim.
        """
        payload = {
            "memory_lifetime_ms": float(memory_lifetime_ms),
            "is_repeater": bool(is_repeater),
        }
        payload.update(attributes)
        self._graph.add_node(node, **payload)

    def add_link(
        self,
        u: Hashable,
        v: Hashable,
        metrics: EdgeMetrics,
        **attributes: Any,
    ) -> None:
        """Add a bidirectional quantum link between ``u`` and ``v``.

        Both endpoints are created automatically if they do not exist yet.

        Args:
            u: Source node.
            v: Target node.
            metrics: Validated :class:`EdgeMetrics` for the link.
            **attributes: Extra edge attributes stored verbatim.

        Raises:
            MetricValidationError: If ``metrics`` is not an :class:`EdgeMetrics`.
            ValueError: If a link between ``u`` and ``v`` already exists.
        """
        if not isinstance(metrics, EdgeMetrics):
            raise MetricValidationError(
                "add_link expects an EdgeMetrics instance, got "
                f"{type(metrics).__name__}"
            )
        if self._graph.has_edge(u, v):
            raise ValueError(
                f"link {u!r}-{v!r} already exists; use update_link to modify it"
            )
        self._graph.add_edge(u, v, **metrics.as_dict(), **attributes)

    def update_link(
        self, u: Hashable, v: Hashable, metrics: EdgeMetrics
    ) -> None:
        """Replace the metrics of an existing link.

        Args:
            u: Source node.
            v: Target node.
            metrics: New validated metrics.

        Raises:
            ValueError: If the link does not exist.
            MetricValidationError: If ``metrics`` is not an :class:`EdgeMetrics`.
        """
        if not self._graph.has_edge(u, v):
            raise ValueError(f"link {u!r}-{v!r} does not exist")
        if not isinstance(metrics, EdgeMetrics):
            raise MetricValidationError(
                "update_link expects an EdgeMetrics instance, got "
                f"{type(metrics).__name__}"
            )
        self._graph.edges[u, v].clear()
        self._graph.edges[u, v].update(metrics.as_dict())

    def add_midpoint_node(
        self,
        u: Hashable,
        v: Hashable,
        new_node: Hashable,
        memory_lifetime_ms: float = 1000.0,
    ) -> None:
        """Insert a repeater node between two adjacent nodes.

        The direct link ``u-v`` is replaced by ``u-new_node-v`` with the direct
        link's metrics split evenly, which is a deliberately simple stand-in
        for "insert a repeater somewhere along this span".

        Args:
            u: One endpoint of the link to subdivide.
            v: The other endpoint.
            new_node: Identifier for the inserted repeater.
            memory_lifetime_ms: Memory lifetime assigned to the repeater.

        Raises:
            ValueError: If the link ``u-v`` does not exist.
        """
        if not self._graph.has_edge(u, v):
            raise ValueError(f"link {u!r}-{v!r} does not exist")
        original = EdgeMetrics.from_dict(self._graph.edges[u, v])
        self._graph.remove_edge(u, v)
        self.add_node(new_node, memory_lifetime_ms=memory_lifetime_ms, is_repeater=True)
        half = EdgeMetrics(
            distance_km=original.distance_km / 2.0,
            noise=original.noise / 2.0,
            latency_ms=original.latency_ms / 2.0,
            fidelity=float(np.sqrt(original.fidelity)),
            reliability=original.reliability,
        )
        self.add_link(u, new_node, half)
        self.add_link(new_node, v, half)

    # ------------------------------------------------------------------
    # Views and derived data
    # ------------------------------------------------------------------
    @property
    def graph(self) -> nx.Graph:
        """The underlying :class:`networkx.Graph`."""
        return self._graph

    @property
    def nodes(self) -> list[Hashable]:
        """Node identifiers in insertion order."""
        return list(self._graph.nodes)

    @property
    def links(self) -> list[tuple[Hashable, Hashable]]:
        """Link endpoints as ``(u, v)`` pairs."""
        return [(u, v) for u, v in self._graph.edges]

    def node(self, node: Hashable) -> dict[str, Any]:
        """Return the attribute dictionary of a single node.

        Args:
            node: Node identifier.

        Returns:
            A copy of the node's attributes.

        Raises:
            KeyError: If the node is not part of the network.
        """
        if node not in self._graph:
            raise KeyError(f"unknown node {node!r}; known nodes: {self.nodes}")
        return dict(self._graph.nodes[node])

    def link_metrics(self, u: Hashable, v: Hashable) -> EdgeMetrics:
        """Return the :class:`EdgeMetrics` of link ``u-v``.

        Args:
            u: Source node.
            v: Target node.

        Returns:
            The link's metrics.

        Raises:
            KeyError: If the link does not exist.
        """
        if not self._graph.has_edge(u, v):
            raise KeyError(f"unknown link {u!r}-{v!r}")
        return EdgeMetrics.from_dict(self._graph.edges[u, v])

    def is_connected(self) -> bool:
        """Return whether the underlying graph is connected."""
        return nx.is_connected(self._graph)

    def has_path(self, source: Hashable, target: Hashable) -> bool:
        """Return whether any path exists from ``source`` to ``target``.

        Args:
            source: Start node.
            target: End node.

        Raises:
            KeyError: If either node is unknown.
        """
        if source not in self._graph or target not in self._graph:
            raise KeyError(
                f"unknown endpoint(s): {source!r} -> {target!r}; "
                f"known nodes: {self.nodes}"
            )
        return nx.has_path(self._graph, source, target)

    def edge_table(self) -> pd.DataFrame:
        """Return every link's metrics as a tidy :class:`pandas.DataFrame`.

        Returns:
            A frame with one row per link and the columns
            ``source``, ``target``, ``distance_km``, ``noise``,
            ``latency_ms``, ``fidelity``, ``reliability``.
        """
        rows: list[dict[str, Any]] = []
        for u, v, data in self._graph.edges(data=True):
            row: dict[str, Any] = {"source": u, "target": v}
            row.update({k: data[k] for k in METRIC_KEYS})
            row["reliability"] = data["reliability"]
            rows.append(row)
        return pd.DataFrame(
            rows,
            columns=["source", "target", *METRIC_KEYS, "reliability"],
        )

    def metric_ranges(self) -> dict[str, tuple[float, float]]:
        """Return ``(min, max)`` of every link metric across the whole network.

        Normalising by these ranges is what makes ``alpha, beta, gamma, delta``
        unit-comparable; see :func:`minmax_normalize`.

        Returns:
            Mapping from metric name to ``(minimum, maximum)`` over all links.
        """
        table = self.edge_table()
        return {
            key: (float(table[key].min()), float(table[key].max()))
            for key in (*METRIC_KEYS, "reliability")
        }

    def hop_lengths(self) -> list[int]:
        """Return the length in hops of every simple source/target pair.

        Returns:
            Sorted list of shortest-path hop counts over all distinct node pairs.
        """
        lengths = nx.all_pairs_shortest_path_length(self._graph)
        out: list[int] = []
        for _, targets in lengths:
            out.extend(targets.values())
        return sorted(out)

    def copy(self) -> "QuantumNetwork":
        """Return an independent deep copy of this network."""
        clone = QuantumNetwork(name=self.name)
        for node, attrs in self._graph.nodes(data=True):
            clone.add_node(node, **attrs)
        for u, v, data in self._graph.edges(data=True):
            clone.add_link(u, v, EdgeMetrics.from_dict(data))
        return clone

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Number of nodes."""
        return self._graph.number_of_nodes()

    def __contains__(self, node: object) -> bool:
        """Membership test against the node set."""
        return node in self._graph

    def __repr__(self) -> str:
        """Compact developer-facing representation."""
        return (
            f"QuantumNetwork(name={self.name!r}, nodes={self._graph.number_of_nodes()}, "
            f"links={self._graph.number_of_edges()}, "
            f"connected={nx.is_connected(self._graph) if self._graph.number_of_nodes() else False})"
        )

    def to_networkx(self) -> nx.Graph:
        """Return the underlying :class:`networkx.Graph`.

        This is what the routing layer in :mod:`src.mdqw_routing` consumes.
        """
        return self._graph


def minmax_normalize(values: Iterable[float]) -> dict[float, float]:
    """Min-max normalise a sequence of non-negative metrics to ``[0, 1]``.

    Normalisation is what makes the routing coefficients ``alpha, beta, gamma,
    delta`` comparable: a 900 km fibre span and a 0.3 ms control latency are
    otherwise on incomparable scales, and no single set of weights would mean
    anything portable across networks.

    Behaviour on degenerate input:

    * empty input -> empty output;
    * constant input (max == min) -> every value maps to ``0.0`` so that a metric
      with no variation contributes nothing rather than an arbitrary constant.

    Args:
        values: Non-negative metric values.

    Returns:
        Mapping from each distinct input value to its normalised value.

    Raises:
        ValueError: If any value is negative or non-finite.
    """
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {}
    if not np.all(np.isfinite(array)):
        raise ValueError("minmax_normalize received a non-finite value")
    if np.any(array < 0):
        raise ValueError("minmax_normalize requires non-negative values")
    low, high = float(array.min()), float(array.max())
    if high - low < 1e-15:
        return {float(v): 0.0 for v in array}
    return {float(v): float((v - low) / (high - low)) for v in array}


def build_example_network() -> QuantumNetwork:
    """Build the hand-designed A-F network used throughout the project.

    Topology::

        A --- B --- C --- D
         \\             /
          \\   E --- F

    Concretely the links are ``A-B``, ``A-E``, ``E-F``, ``F-D``, ``D-C`` and
    ``B-C``.

    The metric assignment is chosen so the routing problem is *not* degenerate:
    there is a geometrically short path (``A-E-F-D``) whose links are lossy and
    slow, and a longer path (``A-B-C-D``) whose links are clean. Which one wins
    therefore depends on the coefficients ``alpha, beta, gamma, delta``, which
    is exactly the trade-off the MDQW cost function is meant to expose.

    Node memory lifetimes are ``800 ms`` for the endpoints and ``2000 ms`` for
    the repeaters, so that a route through more intermediates accumulates more
    memory decay.

    Returns:
        A :class:`QuantumNetwork` named ``BQT-MDQW/example-6-node``.
    """
    net = QuantumNetwork(name="BQT-MDQW/example-6-node")

    endpoints = {"A": 800.0, "B": 800.0, "C": 800.0, "D": 800.0}
    repeaters = {"E": 2000.0, "F": 2000.0}
    for node, lifetime in endpoints.items():
        net.add_node(node, memory_lifetime_ms=lifetime, is_repeater=False)
    for node, lifetime in repeaters.items():
        net.add_node(node, memory_lifetime_ms=lifetime, is_repeater=True)

    # Upper branch A-B-C-D: longer, but clean links.
    net.add_link("A", "B", EdgeMetrics(120.0, 0.05, 2.0, 0.97, 0.99))
    net.add_link("B", "C", EdgeMetrics(140.0, 0.06, 2.5, 0.96, 0.99))
    net.add_link("C", "D", EdgeMetrics(110.0, 0.05, 2.0, 0.97, 0.99))

    # Lower branch A-E-F-D: shorter in hops, but lossy and slow.
    net.add_link("A", "E", EdgeMetrics(160.0, 0.22, 8.0, 0.88, 0.92))
    net.add_link("E", "F", EdgeMetrics(150.0, 0.30, 11.0, 0.82, 0.88))
    net.add_link("F", "D", EdgeMetrics(100.0, 0.20, 7.0, 0.90, 0.93))

    # One extra clean-ish link so that the two branches can cross over.
    net.add_link("C", "F", EdgeMetrics(90.0, 0.10, 4.0, 0.93, 0.96))

    return net


def random_network(
    n_nodes: int = 12,
    avg_degree: int = 3,
    *,
    seed: int = 20260930,
    distance_range: tuple[float, float] = (50.0, 400.0),
    noise_range: tuple[float, float] = (0.02, 0.35),
    latency_range: tuple[float, float] = (1.0, 20.0),
    fidelity_range: tuple[float, float] = (0.80, 0.995),
    reliability_range: tuple[float, float] = (0.85, 0.999),
) -> QuantumNetwork:
    """Generate a reproducible random weighted quantum network.

    The topology is an Erdős-Rényi style ``G(n, p)`` graph sampled by
    :func:`networkx.gnp_random_graph` with an explicit integer seed, so the
    generated graph is identical for a given ``seed`` on any platform. The five
    link metrics are then drawn from independent uniform distributions in the
    supplied ranges, using a :class:`numpy.random.Generator` seeded from the
    same integer.

    The node identifiers are the strings ``"N0" ... "N{n-1}"``; every node
    carries a random memory lifetime and roughly a third are flagged as
    repeaters.

    Args:
        n_nodes: Number of nodes. Must be at least 2.
        avg_degree: Target average degree; converted to the ``G(n, p)`` parameter
            ``p = min(1.0, avg_degree / (n_nodes - 1))``.
        seed: Master seed for both topology and metrics.
        distance_range: Inclusive ``(low, high)`` bounds for ``distance_km``.
        noise_range: Inclusive bounds for ``noise``.
        latency_range: Inclusive bounds for ``latency_ms``.
        fidelity_range: Inclusive bounds for ``fidelity``.
        reliability_range: Inclusive bounds for ``reliability``.

    Returns:
        A connected-if-possible :class:`QuantumNetwork`. If the sampled graph is
        disconnected the function still returns it; use
        :meth:`QuantumNetwork.is_connected` or
        :meth:`QuantumNetwork.has_path` to check reachability.

    Raises:
        ValueError: If ``n_nodes`` is below 2 or a range is malformed.
    """
    if n_nodes < 2:
        raise ValueError(f"n_nodes must be at least 2, got {n_nodes}")
    if avg_degree < 1:
        raise ValueError(f"avg_degree must be at least 1, got {avg_degree}")

    ranges = {
        "distance": distance_range,
        "noise": noise_range,
        "latency": latency_range,
        "fidelity": fidelity_range,
        "reliability": reliability_range,
    }
    for key, (low, high) in ranges.items():
        if not (0.0 <= low <= high <= (1.0 if key in {"noise", "fidelity", "reliability"} else float("inf"))):
            raise ValueError(f"malformed {key}_range: {ranges[key]}")

    probability = min(1.0, avg_degree / (n_nodes - 1))
    topology = nx.gnp_random_graph(n_nodes, probability, seed=seed)

    rng = np.random.default_rng(seed)
    net = QuantumNetwork(name=f"BQT-MDQW/random-{n_nodes}n-seed{seed}")
    for index in range(n_nodes):
        net.add_node(
            f"N{index}",
            memory_lifetime_ms=float(rng.uniform(500.0, 3000.0)),
            is_repeater=bool(rng.random() < 0.34),
        )

    for u, v in topology.edges():
        metrics = EdgeMetrics(
            distance_km=float(rng.uniform(*distance_range)),
            noise=float(rng.uniform(*noise_range)),
            latency_ms=float(rng.uniform(*latency_range)),
            fidelity=float(rng.uniform(*fidelity_range)),
            reliability=float(rng.uniform(*reliability_range)),
        )
        net.add_link(f"N{u}", f"N{v}", metrics)

    # Guarantee at least one link so that metric ranges are well defined.
    if net.graph.number_of_edges() == 0:
        net.add_link("N0", "N1", EdgeMetrics(100.0, 0.1, 5.0, 0.9, 0.95))

    return net
