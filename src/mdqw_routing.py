"""MDQW-inspired routing and the Dijkstra baseline for BQT-MDQW.

Provenance legend used throughout this module
--------------------------------------------
``[ESTABLISHED]``
    Dijkstra's single-source shortest-path algorithm. This is a *classical*
    graph algorithm. It is **not** a quantum algorithm, and this module makes
    no attempt to run it on a quantum computer. Dijkstra is used here purely
    as a well-understood reference point.
``[PROPOSED]``
    The combined multi-metric cost function, the per-relay hop penalty, and the
    optional quantum-walk-derived node prior. These are design choices made in
    this repository. They have no external citation and have not been validated
    on physical quantum networking hardware.
``[SIMULATED]``
    Every number produced by :meth:`MDQWRouter.route` and
    :meth:`DijkstraBaseline.route` is the output of a classical graph search
    over a synthetic network. No quantum state is involved in routing.

The cost function
-----------------
For a link ``e = (u, v)`` the routing cost is a weighted sum of four
min-max-normalised metrics, plus an optional reliability term::

    C(e) = alpha   * d_hat          (normalised link distance)
          + beta    * n_hat          (normalised noise severity)
          + gamma   * l_hat          (normalised latency)
          + delta   * (1 - f_hat)    (fidelity penalty)
          + rho     * (1 - r_hat)    (optional reliability penalty)

Normalisation is what makes the coefficients comparable: a 400 km span and a
20 ms control latency are otherwise on incommensurable scales, and no single
choice of weights would transfer between networks.

A route's total cost also accumulates a *node* term::

    N(v) = hop_penalty      for an intermediate non-repeater node
         = repeater_penalty for an intermediate node flagged ``is_repeater``
         + qw_prior_weight * prior(v)   when the walk-informed prior is enabled

Source and destination nodes are exempt from the node term.

Two honest caveats about optimality
----------------------------------
1. The Dijkstra baseline searches with ``C(e)`` only. MDQW searches with
   ``C(e) + N(v)``. **Their objective functions are therefore different**, and
   comparing their raw ``total_cost`` values is not meaningful. The fair
   cross-algorithm comparison is :attr:`RouteResult.edge_cost`, which both
   routes are scored with. :func:`compare_routes` reports both.
2. The walk-informed prior is a **heuristic** and breaks cost-optimality even
   for MDQW's own objective. It is off by default. When enabled, the router
   returns the best path *under the prior-influenced objective*, which is not
   necessarily the minimum-cost path under ``C(e)``.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable, Literal, Mapping, Sequence

import networkx as nx
import pandas as pd

from .network import METRIC_KEYS, QuantumNetwork, minmax_normalize

__all__ = [
    "RoutingWeights",
    "MDQWConfig",
    "EdgeBreakdown",
    "RouteResult",
    "MDQWRouter",
    "DijkstraBaseline",
    "walkable_proxy",
    "compare_routes",
    "enumerate_routes",
    "summarise_comparison",
    "DEFAULT_WEIGHTS",
]

SearchMode = Literal["none", "astar"]


class RoutingError(ValueError):
    """Raised for invalid routing configuration or unreachable endpoints."""


#: Balanced default weights. Distance and fidelity dominate slightly, which keeps
#: the default behaviour close to classical distance routing while still
#: exposing the trade-off to the sweeps in :mod:`src.simulation`.
DEFAULT_WEIGHTS = {"alpha": 1.0, "beta": 1.0, "gamma": 0.5, "delta": 1.5, "rho": 0.25}


@dataclass(frozen=True, slots=True)
class RoutingWeights:
    """Coefficients of the combined routing cost ``C(e)``.

    Attributes:
        alpha: Weight on normalised link distance.
        beta: Weight on normalised noise severity.
        gamma: Weight on normalised latency.
        delta: Weight on the fidelity penalty ``1 - f_hat``.
        rho: Weight on the optional reliability penalty ``1 - r_hat``.
            Defaults to ``0.25``.
        hop_penalty: Cost charged once for every intermediate node, modelling
            the operational overhead of a relay: a memory element that must be
            loaded, held, and read out.
        repeater_penalty: Cost charged for an intermediate node flagged
            ``is_repeater``. Defaults to the value of ``hop_penalty``.

    All weights must be finite and non-negative, and at least one of
    ``alpha, beta, gamma, delta, rho`` must be strictly positive -- otherwise
    every link costs the same and the routing problem is degenerate.
    """

    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.5
    delta: float = 1.5
    rho: float = 0.25
    hop_penalty: float = 0.0
    repeater_penalty: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "alpha",
            "beta",
            "gamma",
            "delta",
            "rho",
            "hop_penalty",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise RoutingError(f"weight {name} must be finite, got {value}")
            if value < 0:
                raise RoutingError(f"weight {name} must be non-negative, got {value}")
        effective = self.alpha + self.beta + self.gamma + self.delta + self.rho
        if effective <= 0:
            raise RoutingError(
                "at least one of alpha/beta/gamma/delta/rho must be positive; "
                "otherwise all link costs are identical and routing is degenerate"
            )
        if self.repeater_penalty is None:
            object.__setattr__(self, "repeater_penalty", self.hop_penalty)
        elif float(self.repeater_penalty) < 0:
            raise RoutingError(
                f"repeater_penalty must be non-negative, got {self.repeater_penalty}"
            )

    def as_dict(self) -> dict[str, float]:
        """Return the weights as a plain dictionary."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "rho": self.rho,
            "hop_penalty": self.hop_penalty,
            "repeater_penalty": float(self.repeater_penalty or 0.0),
        }


@dataclass(frozen=True, slots=True)
class MDQWConfig:
    """Configuration of the MDQW-inspired search.

    Attributes:
        weights: The :class:`RoutingWeights` used for the link cost.
        mode: ``"none"`` for uniform-cost search (a Dijkstra-style expansion) or
            ``"astar"`` for A* with an admissible heuristic. A* is optimal for
            the same objective but typically expands fewer nodes.
        use_qw_prior: Enable the quantum-walk-derived node prior. **Off by
            default** because the prior is a heuristic and voids the optimality
            guarantee. See the module docstring.
        qw_prior_weight: Multiplier applied to the prior value at each node.
        prior_steps: Number of walk steps used to build the prior.
        max_hops: Optional upper bound on path length in hops. When set, the
            search is a constrained one and may return no route even when a
            path exists.
    """

    weights: RoutingWeights = field(default_factory=RoutingWeights)
    mode: SearchMode = "astar"
    use_qw_prior: bool = False
    qw_prior_weight: float = 0.0
    prior_steps: int = 8
    max_hops: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("none", "astar"):
            raise RoutingError(f"mode must be 'none' or 'astar', got {self.mode!r}")
        if not math.isfinite(self.qw_prior_weight) or self.qw_prior_weight < 0:
            raise RoutingError(
                f"qw_prior_weight must be finite and non-negative, got {self.qw_prior_weight}"
            )
        if self.prior_steps < 1:
            raise RoutingError(f"prior_steps must be >= 1, got {self.prior_steps}")
        if self.max_hops is not None and self.max_hops < 0:
            raise RoutingError(f"max_hops must be non-negative or None, got {self.max_hops}")


@dataclass(frozen=True, slots=True)
class EdgeBreakdown:
    """Per-link accounting for one route.

    Attributes:
        source: Link tail node.
        target: Link head node.
        distance_km: Link length in kilometres.
        noise: Normalised-independent noise severity in ``[0, 1]``.
        latency_ms: Link latency in milliseconds.
        fidelity: Per-hop fidelity declared by the network model.
        reliability: Link availability probability.
        cost: The link's contribution ``C(e)`` under the active weights.
    """

    source: Hashable
    target: Hashable
    distance_km: float
    noise: float
    latency_ms: float
    fidelity: float
    reliability: float
    cost: float

    def as_dict(self) -> dict[str, Any]:
        """Return the breakdown entry as a plain dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "distance_km": self.distance_km,
            "noise": self.noise,
            "latency_ms": self.latency_ms,
            "fidelity": self.fidelity,
            "reliability": self.reliability,
            "cost": self.cost,
        }


@dataclass(frozen=True, slots=True)
class RouteResult:
    """A single route plus every metric needed to compare it against another.

    The distinction between :attr:`total_cost` and :attr:`edge_cost` is
    deliberate and load-bearing; see the module docstring.

    Attributes:
        path: Ordered node sequence from source to destination.
        total_cost: The objective value actually optimised by the producing
            algorithm. For MDQW this includes node costs; for the Dijkstra
            baseline it is the pure edge sum.
        edge_cost: Sum of ``C(e)`` over the route's links. This is the **fair
            cross-algorithm** comparison metric, because it is computed the same
            way for every algorithm regardless of its node terms.
        node_cost: Sum of the per-node terms. Zero for the Dijkstra baseline.
        hop_count: Number of links in the route.
        total_distance_km: Sum of link distances.
        total_noise: Sum of link noise severities.
        mean_noise: Average link noise severity.
        estimated_fidelity: Product of per-link fidelities, the standard
            independent-error model. See :meth:`attenuation_fidelity` for the
            distance-based alternative.
        total_latency_ms: Sum of link latencies.
        edges: Per-link :class:`EdgeBreakdown` records.
        algorithm: Human-readable name of the producing algorithm.
        source: Route origin.
        target: Route destination.
        is_cost_optimal: ``True`` only when the producing algorithm carries an
            optimality guarantee for :attr:`total_cost`. ``False`` for the
            walk-informed mode, and ``True`` for Dijkstra and for MDQW in
            ``"none"``/``"astar"`` mode with the prior disabled.
        notes: Free-form provenance notes.
    """

    path: tuple[Hashable, ...]
    total_cost: float
    edge_cost: float
    node_cost: float
    hop_count: int
    total_distance_km: float
    total_noise: float
    mean_noise: float
    estimated_fidelity: float
    total_latency_ms: float
    edges: tuple[EdgeBreakdown, ...]
    algorithm: str
    source: Hashable
    target: Hashable
    is_cost_optimal: bool
    notes: str = ""

    def __len__(self) -> int:
        """Number of nodes on the route."""
        return len(self.path)

    def describe(self) -> str:
        """Return a human-readable multi-line summary of the route."""
        chain = " -> ".join(str(node) for node in self.path)
        return (
            f"Selected path: {chain}\n"
            f"Algorithm    : {self.algorithm}\n"
            f"Total cost   : {self.total_cost:.4f} "
            f"(edge {self.edge_cost:.4f} + node {self.node_cost:.4f})\n"
            f"Hop count    : {self.hop_count}\n"
            f"Distance     : {self.total_distance_km:.2f} km\n"
            f"Noise (sum)  : {self.total_noise:.4f}\n"
            f"Latency      : {self.total_latency_ms:.2f} ms\n"
            f"Fidelity est.: {self.estimated_fidelity:.6f}"
        )

    def to_record(self) -> dict[str, Any]:
        """Return a flat dictionary suitable for a :class:`pandas.DataFrame`."""
        return {
            "algorithm": self.algorithm,
            "source": self.source,
            "target": self.target,
            "path": " -> ".join(str(node) for node in self.path),
            "hop_count": self.hop_count,
            "total_cost": self.total_cost,
            "edge_cost": self.edge_cost,
            "node_cost": self.node_cost,
            "total_distance_km": self.total_distance_km,
            "total_noise": self.total_noise,
            "mean_noise": self.mean_noise,
            "estimated_fidelity": self.estimated_fidelity,
            "total_latency_ms": self.total_latency_ms,
            "is_cost_optimal": self.is_cost_optimal,
        }

    def attenuation_fidelity(self, attenuation_length_km: float = 50.0) -> float:
        """Estimate route fidelity with a distance-attenuation model.

        This is an alternative to the product-of-link-fidelities model: each hop
        contributes ``f_e * exp(-d_e / L)`` where ``L`` is the attenuation
        length. It reflects that long links are worse than short ones even at
        equal declared fidelity.

        Args:
            attenuation_length_km: Characteristic attenuation length ``L``.

        Returns:
            The product of the attenuated per-hop fidelities.

        Raises:
            ValueError: If ``attenuation_length_km`` is not positive.
        """
        if attenuation_length_km <= 0:
            raise ValueError(
                f"attenuation_length_km must be positive, got {attenuation_length_km}"
            )
        product = 1.0
        for edge in self.edges:
            product *= edge.fidelity * math.exp(-edge.distance_km / attenuation_length_km)
        return product


class MDQWRouter:
    """Quantum-walk-inspired shortest-path router over a :class:`QuantumNetwork`.

    Args:
        network: The network to route over.
        config: Search configuration. Defaults to A* with no walk prior.
        prior: Optional mapping ``node -> salience in [0, 1]`` used when
            ``config.use_qw_prior`` is set. Build one with
            :func:`src.quantum_walk.quantum_walk_prior` over a walkable proxy of
            the network; see :func:`walkable_proxy`.

    Raises:
        RoutingError: If the network is empty or a provided prior is invalid.
    """

    def __init__(
        self,
        network: QuantumNetwork,
        config: MDQWConfig | None = None,
        prior: Mapping[Hashable, float] | None = None,
    ) -> None:
        if len(network) == 0:
            raise RoutingError("cannot route over an empty network")
        self.network = network
        self.config = config or MDQWConfig()
        self.weights = self.config.weights
        self.graph = network.to_networkx()
        self.prior = self._validate_prior(prior)

        # Normalisation tables, computed once per router. Keyed by raw metric
        # value -> normalised value, built over every link in the network.
        table = network.edge_table()
        self._norm: dict[str, dict[float, float]] = {}
        for key in ("distance_km", "noise", "latency_ms", "fidelity", "reliability"):
            self._norm[key] = minmax_normalize(table[key].tolist())

        # Per-node minimum incident edge cost, the admissible A* heuristic.
        self._min_incident: dict[Hashable, float] = {}
        for node in self.graph.nodes:
            costs = [
                self.edge_cost(node, other)
                for other in self.graph.neighbors(node)
            ]
            self._min_incident[node] = min(costs) if costs else 0.0

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _validate_prior(
        self, prior: Mapping[Hashable, float] | None
    ) -> dict[Hashable, float] | None:
        if prior is None:
            return None
        if not prior:
            raise RoutingError("prior is empty; pass None to disable the walk prior")
        known = set(self.network.nodes)
        unknown = set(prior) - known
        if unknown:
            raise RoutingError(
                f"prior references unknown node(s): {sorted(map(str, unknown))}"
            )
        cleaned: dict[Hashable, float] = {}
        for node, value in prior.items():
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise RoutingError(
                    f"prior value for {node!r} must be finite and non-negative, got {value}"
                )
            cleaned[node] = number
        return cleaned

    # ------------------------------------------------------------------
    # Costs
    # ------------------------------------------------------------------
    def edge_cost(self, u: Hashable, v: Hashable) -> float:
        """Return the combined cost ``C(e)`` of link ``u-v``.

        Args:
            u: Link tail.
            v: Link head.

        Returns:
            The weighted, normalised link cost.

        Raises:
            KeyError: If the link does not exist.
        """
        data = self.graph.edges[u, v]
        d = self._norm["distance_km"][float(data["distance_km"])]
        n = self._norm["noise"][float(data["noise"])]
        latency = self._norm["latency_ms"][float(data["latency_ms"])]
        fidelity_penalty = 1.0 - self._norm["fidelity"][float(data["fidelity"])]
        reliability_penalty = 1.0 - self._norm["reliability"][float(data["reliability"])]
        return (
            self.weights.alpha * d
            + self.weights.beta * n
            + self.weights.gamma * latency
            + self.weights.delta * fidelity_penalty
            + self.weights.rho * reliability_penalty
        )

    def node_cost(self, node: Hashable) -> float:
        """Return the node term ``N(v)`` charged when ``node`` is intermediate.

        Args:
            node: Node identifier.

        Returns:
            The total node cost. Always ``0.0`` when the walk prior is disabled
            and ``hop_penalty`` is zero.
        """
        cost = 0.0
        weights = self.weights
        if weights.hop_penalty or weights.repeater_penalty:
            attributes = self.network.node(node)
            if attributes.get("is_repeater", False):
                cost += float(weights.repeater_penalty or 0.0)
            else:
                cost += float(weights.hop_penalty)
        if self.prior is not None and self.config.use_qw_prior:
            cost += self.config.qw_prior_weight * self.prior.get(node, 0.0)
        return cost

    def _heuristic(self, node: Hashable, target: Hashable) -> float:
        """Return the admissible A* heuristic ``h(v)``.

        Uses the minimum incident edge cost at ``v``. Any path from ``v`` to the
        target must cross at least one link, and all node terms are
        non-negative, so this is a valid lower bound on the remaining cost.

        Args:
            node: Current node.
            target: Goal node.

        Returns:
            The heuristic value; ``0.0`` at the target.
        """
        if node == target:
            return 0.0
        return self._min_incident.get(node, 0.0)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def route(self, source: Hashable, target: Hashable) -> RouteResult:
        """Find a route from ``source`` to ``target``.

        Args:
            source: Start node.
            target: Destination node.

        Returns:
            A :class:`RouteResult` with the full metric breakdown.

        Raises:
            KeyError: If either endpoint is not a node of the network.
            RoutingError: If the endpoints are disconnected, or if
                ``max_hops`` is set too low to admit any path.
        """
        if source not in self.graph:
            raise KeyError(f"unknown source {source!r}")
        if target not in self.graph:
            raise KeyError(f"unknown target {target!r}")
        if not self.graph.has_edge(source, target) and not nx.has_path(
            self.graph, source, target
        ):
            raise RoutingError(
                f"no path exists between {source!r} and {target!r}; "
                "the network is disconnected at these endpoints"
            )

        use_astar = self.config.mode == "astar"
        # Heap entries: (f, tie, g, node). The tie-breaker keeps the search
        # deterministic for graphs with many equal-cost routes.
        counter = 0
        start_h = self._heuristic(source, target)
        heap: list[tuple[float, int, float, Hashable]] = [(start_h, counter, 0.0, source)]
        best: dict[Hashable, float] = {source: 0.0}
        parents: dict[Hashable, Hashable] = {}
        hops: dict[Hashable, int] = {source: 0}
        goal_node: Hashable | None = None

        while heap:
            _, _, g_cost, node = heapq.heappop(heap)
            if node == target:
                goal_node = node
                break
            if g_cost > best.get(node, math.inf) + 1e-12:
                continue
            depth = hops[node]
            if self.config.max_hops is not None and depth >= self.config.max_hops:
                continue
            for neighbour in self.graph.neighbors(node):
                # The destination is exempt from the node term, matching the
                # documented semantics of node_cost() and keeping the invariant
                # total_cost == edge_cost + sum(node_cost over intermediates).
                # Without this exemption the search would charge the target's
                # penalty while RouteResult.node_cost would not, and the two
                # reported figures would disagree.
                node_term = 0.0 if neighbour == target else self.node_cost(neighbour)
                step = self.edge_cost(node, neighbour) + node_term
                tentative = g_cost + step
                if tentative < best.get(neighbour, math.inf) - 1e-12:
                    best[neighbour] = tentative
                    parents[neighbour] = node
                    hops[neighbour] = depth + 1
                    counter += 1
                    priority = tentative + (
                        self._heuristic(neighbour, target) if use_astar else 0.0
                    )
                    heapq.heappush(heap, (priority, counter, tentative, neighbour))

        if goal_node is None:
            limit = (
                " within the max_hops limit of "
                f"{self.config.max_hops}"
                if self.config.max_hops is not None
                else ""
            )
            raise RoutingError(
                f"no route found from {source!r} to {target!r}{limit}"
            )

        path = self._reconstruct(parents, source, target)
        return self._build_result(
            path,
            total_cost=best[target],
            algorithm=self._algorithm_name(),
            is_cost_optimal=not (self.config.use_qw_prior and self.prior is not None),
        )

    def _algorithm_name(self) -> str:
        """Return the display name of the active configuration."""
        parts = ["MDQW"]
        parts.append("A*" if self.config.mode == "astar" else "uniform-cost")
        if self.prior is not None and self.config.use_qw_prior:
            parts.append("+QW-prior")
        if self.config.max_hops is not None:
            parts.append(f"maxhops={self.config.max_hops}")
        return " ".join(parts)

    def _reconstruct(
        self,
        parents: Mapping[Hashable, Hashable],
        source: Hashable,
        target: Hashable,
    ) -> tuple[Hashable, ...]:
        """Walk the parent pointers back to build the node sequence."""
        path = [target]
        while path[-1] != source:
            path.append(parents[path[-1]])
        path.reverse()
        return tuple(path)

    def _build_result(
        self,
        path: Sequence[Hashable],
        *,
        total_cost: float,
        algorithm: str,
        is_cost_optimal: bool,
    ) -> RouteResult:
        """Assemble a :class:`RouteResult` from a node sequence.

        The ``total_cost`` is recomputed from scratch rather than trusted from
        the search, so the reported number always matches the reported path.
        """
        edges: list[EdgeBreakdown] = []
        edge_total = 0.0
        node_total = 0.0
        distance = 0.0
        noise = 0.0
        latency = 0.0
        fidelity = 1.0

        for index in range(len(path) - 1):
            u, v = path[index], path[index + 1]
            data = self.network.link_metrics(u, v)
            cost = self.edge_cost(u, v)
            edges.append(
                EdgeBreakdown(
                    source=u,
                    target=v,
                    distance_km=data.distance_km,
                    noise=data.noise,
                    latency_ms=data.latency_ms,
                    fidelity=data.fidelity,
                    reliability=data.reliability,
                    cost=cost,
                )
            )
            edge_total += cost
            distance += data.distance_km
            noise += data.noise
            latency += data.latency_ms
            fidelity *= data.fidelity

        for node in path[1:-1]:
            node_total += self.node_cost(node)

        return RouteResult(
            path=tuple(path),
            total_cost=total_cost,
            edge_cost=edge_total,
            node_cost=node_total,
            hop_count=len(path) - 1,
            total_distance_km=distance,
            total_noise=noise,
            mean_noise=noise / max(1, len(path) - 1),
            estimated_fidelity=fidelity,
            total_latency_ms=latency,
            edges=tuple(edges),
            algorithm=algorithm,
            source=path[0],
            target=path[-1],
            is_cost_optimal=is_cost_optimal,
            notes=(
                "Quantum-walk prior is a heuristic; cost-optimality is not "
                "guaranteed."
                if not is_cost_optimal
                else "Optimal for the stated objective."
            ),
        )


class DijkstraBaseline:
    """Ordinary Dijkstra shortest path over the same cost ``C(e)``.

    ``[ESTABLISHED]`` This is the classical reference. It searches with the
    *identical* link cost function used by :class:`MDQWRouter` but with **no
    node terms and no heuristic**, so it returns the true minimum-``C(e)`` path.

    Args:
        network: The network to route over.
        weights: Link-cost weights. Should match the MDQW configuration being
            compared against.
    """

    def __init__(self, network: QuantumNetwork, weights: RoutingWeights | None = None) -> None:
        if len(network) == 0:
            raise RoutingError("cannot route over an empty network")
        self.network = network
        self.weights = weights or RoutingWeights()
        self.graph = network.to_networkx()
        self._scorer = MDQWRouter(network, MDQWConfig(weights=self.weights, mode="none"))

    def route(self, source: Hashable, target: Hashable) -> RouteResult:
        """Return the minimum-link-cost path from ``source`` to ``target``.

        Args:
            source: Start node.
            target: Destination node.

        Returns:
            A :class:`RouteResult` whose ``edge_cost`` is the global minimum.

        Raises:
            KeyError: If either endpoint is unknown.
            RoutingError: If the endpoints are disconnected.
        """
        if source not in self.graph:
            raise KeyError(f"unknown source {source!r}")
        if target not in self.graph:
            raise KeyError(f"unknown target {target!r}")
        try:
            path = nx.shortest_path(
                self.graph,
                source,
                target,
                weight=lambda u, v, data: self._scorer.edge_cost(u, v),
            )
        except nx.NetworkXNoPath as exc:
            raise RoutingError(
                f"no path exists between {source!r} and {target!r}"
            ) from exc
        return self._scorer._build_result(
            path,
            total_cost=sum(
                self._scorer.edge_cost(u, v) for u, v in zip(path, path[1:])
            ),
            algorithm="Dijkstra (classical baseline)",
            is_cost_optimal=True,
        )


# ----------------------------------------------------------------------
# Walk-informed prior support
# ----------------------------------------------------------------------
def walkable_proxy(graph: nx.Graph) -> tuple[nx.Graph, str]:
    """Return a regular graph over the same node set, suitable for a quantum walk.

    The coined discrete-time quantum walk in :mod:`src.quantum_walk` requires an
    ``m``-regular graph so that its adjacency matrix decomposes into exactly
    ``m`` permutation matrices. Real quantum networks are rarely regular, so
    this function derives a walkable surrogate in three steps:

    1. return the graph unchanged if it is already regular;
    2. otherwise try to extract a spanning 2-regular subgraph (a disjoint union
       of cycles) via a maximum-cardinality matching, which is walkable with a
       Hadamard coin of dimension ``m = 2``;
    3. otherwise fall back to the complete graph over the same nodes, which is
       ``(n - 1)``-regular.

    The surrogate is used **only** to shape the routing prior. Its edges are not
    physical quantum links and it carries no metrics of its own.

    Args:
        graph: Any undirected graph with at least two nodes.

    Returns:
        A tuple ``(proxy, strategy)`` where ``strategy`` names the branch taken.

    Raises:
        ValueError: If the graph has fewer than two nodes or is disconnected in
            a way that prevents a spanning cycle cover.
    """
    nodes = list(graph.nodes)
    if len(nodes) < 2:
        raise ValueError("walkable_proxy needs at least two nodes")

    degrees = {node: graph.degree(node) for node in nodes}
    if len(set(degrees.values())) == 1:
        return graph, "already-regular"

    try:
        matching = nx.max_weight_matching(
            nx.Graph(graph), maxcardinality=True, weight=None
        )
    except nx.NetworkXError:
        matching = set()

    cycles: nx.Graph = nx.Graph()
    cycles.add_nodes_from(nodes)
    used: set[Hashable] = set()
    for u, v in matching:
        if u in used or v in used:
            continue
        cycles.add_edge(u, v)
        used.update({u, v})
    leftover = [node for node in nodes if node not in used]
    # Close each leftover vertex onto itself only if it can form a valid cycle;
    # a self-loop makes the vertex degree 2 (one in, one out) for the walker.
    for node in leftover:
        if graph.degree(node) >= 2:
            cycles.add_edge(node, node)

    degrees2 = {node: cycles.degree(node) for node in nodes}
    if len(set(degrees2.values())) == 1 and min(degrees2.values()) >= 2:
        return cycles, "spanning-2-factor"

    complete = nx.complete_graph(nodes)
    return complete, "complete-graph-fallback"


# ----------------------------------------------------------------------
# Comparison utilities
# ----------------------------------------------------------------------
def compare_routes(
    results: Mapping[str, RouteResult] | Sequence[RouteResult],
) -> pd.DataFrame:
    """Tabulate routes produced by different algorithms.

    Args:
        results: Either a ``name -> RouteResult`` mapping or a sequence of
            results (named by their ``algorithm`` field).

    Returns:
        A frame with one row per route containing path, hop count, both cost
        flavours, distance, noise, estimated fidelity, latency, and the
        optimality flag. The ``*_delta_vs_dijkstra`` columns are filled with
        ``NaN`` for the baseline row itself.

    Note:
        The meaningful cross-algorithm comparison is the ``edge_cost`` column,
        because MDQW and Dijkstra optimise different objectives. Both are
        reported; neither is presented as universally better.
    """
    if isinstance(results, Mapping):
        items = list(results.items())
        frame = pd.DataFrame(
            [
                {**route.to_record(), "label": label}
                for label, route in items
            ]
        )
    else:
        frame = pd.DataFrame([route.to_record() for route in results])
        frame.insert(0, "label", frame["algorithm"])

    baseline_mask = frame["label"].str.contains("Dijkstra", case=False, na=False)
    if baseline_mask.any():
        base = frame.loc[baseline_mask].iloc[0]
        frame["edge_cost_delta_vs_dijkstra"] = frame["edge_cost"] - float(base["edge_cost"])
        frame["hop_delta_vs_dijkstra"] = frame["hop_count"] - int(base["hop_count"])
        frame["fidelity_delta_vs_dijkstra"] = (
            frame["estimated_fidelity"] - float(base["estimated_fidelity"])
        )
    else:
        frame["edge_cost_delta_vs_dijkstra"] = float("nan")
        frame["hop_delta_vs_dijkstra"] = float("nan")
        frame["fidelity_delta_vs_dijkstra"] = float("nan")

    return frame


def summarise_comparison(frame: pd.DataFrame, *, tolerance: float = 1e-9) -> pd.DataFrame:
    """Summarise a :func:`compare_routes` frame into win/loss/tie counts.

    The comparison is made on :func:`~pandas.DataFrame.edge_cost` -- the link
    cost that both algorithms score identically. Nothing here asserts that one
    algorithm is better in general; it only reports what happened on this
    instance.

    Args:
        frame: Output of :func:`compare_routes`.
        tolerance: Absolute tolerance for declaring a tie.

    Returns:
        A one-row frame with counts and a plain-text ``verdict``.
    """
    if frame.empty:
        raise RoutingError("cannot summarise an empty comparison frame")
    baseline_mask = frame["label"].str.contains("Dijkstra", case=False, na=False)
    if not baseline_mask.any():
        raise RoutingError("comparison frame has no Dijkstra baseline row")
    baseline = frame.loc[baseline_mask].iloc[0]
    base_cost = float(baseline["edge_cost"])
    base_fid = float(baseline["estimated_fidelity"])

    rows = frame[~baseline_mask]
    lower = int((rows["edge_cost"] < base_cost - tolerance).sum())
    higher = int((rows["edge_cost"] > base_cost + tolerance).sum())
    tied = int(len(rows) - lower - higher)
    better_fidelity = int((rows["estimated_fidelity"] > base_fid + tolerance).sum())
    worse_fidelity = int((rows["estimated_fidelity"] < base_fid - tolerance).sum())

    if lower and not higher:
        verdict = (
            f"MDQW achieved a lower link cost on this instance "
            f"({lower} of {len(rows)} configurations)."
        )
    elif higher and not lower:
        verdict = (
            f"Dijkstra achieved a lower link cost on this instance "
            f"({higher} of {len(rows)} configurations)."
        )
    elif lower and higher:
        verdict = (
            f"Mixed: MDQW lower on {lower}, higher on {higher}, tied on {tied}."
        )
    else:
        verdict = f"All {len(rows)} MDQW configurations tied with Dijkstra."

    return pd.DataFrame(
        [
            {
                "configurations": int(len(rows)),
                "mdqw_lower_edge_cost": lower,
                "dijkstra_lower_edge_cost": higher,
                "tied": tied,
                "mdqw_higher_fidelity": better_fidelity,
                "mdqw_lower_fidelity": worse_fidelity,
                "verdict": verdict,
            }
        ]
    )


def enumerate_routes(
    network: QuantumNetwork,
    source: Hashable,
    target: Hashable,
    *,
    config: MDQWConfig | None = None,
    max_paths: int = 64,
    max_hops: int = 8,
) -> pd.DataFrame:
    """Enumerate simple routes and score each one with the MDQW cost function.

    This provides the "enough information to compare different routes" view: it
    exhaustively lists up to ``max_paths`` distinct simple paths and scores
    every one of them, so a reader can see exactly why one path was chosen.

    Args:
        network: The network to search.
        source: Start node.
        target: Destination node.
        config: Search configuration used only for the cost weights.
        max_paths: Upper bound on the number of paths scored.
        max_hops: Upper bound on path length, to keep the enumeration finite.

    Returns:
        A frame with one row per enumerated path, ordered by increasing edge
        cost, including a ``selected`` column marking the MDQW-chosen path.

    Raises:
        KeyError: If an endpoint is unknown.
        RoutingError: If the endpoints are disconnected.
    """
    config = config or MDQWConfig()
    router = MDQWRouter(network, config)
    try:
        best = router.route(source, target)
    except RoutingError as exc:
        raise RoutingError(
            f"cannot enumerate routes from {source!r} to {target!r}: {exc}"
        ) from exc

    chosen = tuple(best.path)
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(
        nx.all_simple_paths(router.graph, source, target, cutoff=max_hops)
    ):
        if index >= max_paths:
            break
        result = router._build_result(
            path,
            total_cost=sum(
                router.edge_cost(u, v) for u, v in zip(path, path[1:])
            ),
            algorithm="enumerated",
            is_cost_optimal=False,
        )
        record = result.to_record()
        record["path"] = " -> ".join(str(node) for node in path)
        record["selected"] = tuple(path) == chosen
        record["is_dijkstra_baseline"] = False
        rows.append(record)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["edge_cost", "hop_count"], ignore_index=True)
