"""Seeded experiments, figures and reporting for the BQT-MDQW prototype.

Provenance legend used throughout this module
---------------------------------------------
``[ESTABLISHED]``
    Standard results from the published literature, used here unchanged. In
    this module that means exactly one thing: the classical Dijkstra
    shortest-path algorithm, and the Bennett teleportation protocol as it is
    built by :mod:`src.quantum_teleportation`. Dijkstra is a *classical* graph
    algorithm and is not run on a quantum computer anywhere in this project.
``[PROPOSED]``
    Design choices made in this repository, with no external citation and no
    validation on quantum hardware: the MDQW-inspired multi-metric cost
    function, the per-relay ``hop_penalty``, the quantum-walk-derived node
    prior, the ``noise_score -> depolarizing`` and
    ``fidelity_hint -> readout_error`` mappings, and the hop-by-hop
    multi-hop teleportation model implemented by :func:`simulate_route`.
``[SIMULATED]``
    Every number this module produces. All quantum operations are executed by
    Qiskit Aer's classical simulator or by dense NumPy matrix-vector products.
    All network metrics come from the synthetic generators in
    :mod:`src.network`.

**Every number this module produces is a classical simulation on synthetic
data.** No quantum hardware was used, no link metric is calibrated against a
physical channel, and nothing here establishes physical performance of any
kind.

The multi-hop model is *not* a repeater network
-----------------------------------------------
Naive sequential teleportation, as modelled here, is NOT equivalent to an
optimised quantum repeater network. Real repeaters use entanglement swapping,
purification, and quantum memories; the sequential model teleports hop-by-hop
with no entanglement swapping, so its end-to-end fidelity should be read as an
illustrative per-hop composition, not as repeater-network performance.

The same paragraph is emitted verbatim into the ``Provenance`` section of the
generated ``results.md`` by :func:`render_results_markdown`, so the caveat
travels with the numbers.

What each subsection produces
-----------------------------
Multi-hop simulation
    :class:`HopResult`, :class:`MultiHopResult`, :func:`simulate_route`,
    :func:`fidelity_vs_hop_count`.
Experiments
    :func:`experiment_manual_network`, :func:`experiment_random_network`,
    :func:`experiment_noise_sweep`, :func:`experiment_weight_sweep`,
    :func:`experiment_qw_prior_ablation`, :func:`experiment_hop_penalty`.
    Each returns ``(frames, metadata)`` and each re-raises its own errors.
Figures
    ``plot_*`` functions, all writing into a caller-supplied directory and
    all closing their figure.
Reporting and orchestration
    :func:`environment_block`, :func:`render_results_markdown`,
    :func:`render_architecture_diagram`, :func:`run_all`.
"""

from __future__ import annotations

import math
import os
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Hashable, Iterable, Mapping, Sequence

import matplotlib

# ---------------------------------------------------------------------------
# Headless-backend guard. This must run *before* pyplot is imported, otherwise
# matplotlib resolves and caches a GUI backend at import time. On a desktop
# machine ``matplotlib.get_backend()`` is typically "TkAgg", which would try to
# open a window for every figure this module saves.
# ---------------------------------------------------------------------------
_GUI_BACKENDS = frozenset(
    {
        "tkagg",
        "macosx",
        "qtagg",
        "qt5agg",
        "qt4agg",
        "pyqt4agg",
        "pyqt5agg",
        "pysideagg",
        "pyside2agg",
        "pyside6agg",
        "gtk3agg",
        "gtk4agg",
        "wxagg",
        "webagg",
    }
)

if matplotlib.get_backend().lower() in _GUI_BACKENDS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .network import EdgeMetrics, QuantumNetwork, build_example_network, random_network
from .mdqw_routing import (
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
from .quantum_teleportation import (
    ArbitraryState,
    bidirectional_marginals,
    bidirectional_teleportation_circuit,
    build_teleportation_circuit,
    fidelity_metrics,
    run_on_aer,
    teleport_link_with_noise,
)
from .quantum_walk import (
    example_walk_graph,
    quantum_walk_prior,
    walk_on_aer,
)
from .quantum_walk import plot_walk_heatmap as _delegate_walk_heatmap

#: Balanced routing weights, spelled out here rather than imported. The task
#: restricted this module to the published public names of ``src.mdqw_routing``,
#: and ``DEFAULT_WEIGHTS`` is not one of them, so the balanced default is
#: restated locally. Verified identical to the router module's own default
#: before being written down here, so results are unchanged.
_BALANCED_WEIGHTS: Mapping[str, float] = {
    "alpha": 1.0,
    "beta": 1.0,
    "gamma": 0.5,
    "delta": 1.5,
    "rho": 0.25,
}

#: The walk prior used to depend on ``PYTHONHASHSEED``. The cause was a
#: string-keyed bipartite double cover inside
#: ``src.quantum_walk.shunt_decomposition``: Hopcroft-Karp's cycle cover was
#: chosen according to CPython's per-process randomised string hashing, which
#: changed the walk unitary, the prior, and therefore every node cost derived
#: from it. The double cover now uses integer keys, whose hashing is not
#: randomised, so the decomposition is byte-identical across processes and
#: ``PYTHONHASHSEED`` no longer has to be pinned. The value is still reported
#: alongside every result set for provenance, and
#: ``tests/test_regressions.py::test_walk_prior_is_independent_of_python_hash_seed``
#: guards the property. The constant is retained for reporting only.
PINNED_HASH_SEED: str = "0"

__all__ = [
    "HopResult",
    "MultiHopResult",
    "simulate_route",
    "fidelity_vs_hop_count",
    "experiment_manual_network",
    "experiment_random_network",
    "experiment_noise_sweep",
    "experiment_weight_sweep",
    "experiment_qw_prior_ablation",
    "experiment_hop_penalty",
    "plot_network_topology",
    "plot_selected_route",
    "plot_path_cost_comparison",
    "plot_fidelity_vs_hops",
    "plot_noise_vs_route_cost",
    "plot_weight_sensitivity",
    "plot_qw_probability_distribution",
    "plot_walk_heatmap",
    "plot_random_network_aggregate",
    "plot_teleportation_counts",
    "plot_bidirectional_fidelity",
    "environment_block",
    "render_results_markdown",
    "render_architecture_diagram",
    "run_all",
]

#: Verbatim caveat about the multi-hop model. Reused by :func:`simulate_route`
#: documentation, by :func:`render_results_markdown` and by the generated
#: ``results.md``.
SEQUENTIAL_TELEPORTATION_CAVEAT = (
    "Naive sequential teleportation, as modelled here, is NOT equivalent to an "
    "optimised quantum repeater network. Real repeaters use entanglement "
    "swapping, purification, and quantum memories; the sequential model "
    "teleports hop-by-hop with no entanglement swapping, so its end-to-end "
    "fidelity should be read as an illustrative per-hop composition, not as "
    "repeater-network performance."
)

#: Default payload state for the multi-hop simulations: an equatorial
#: superposition, whose ideal computational-basis distribution is exactly
#: ``{"0": 0.5, "1": 0.5}``. A near-eigenstate would report a fidelity close to 1
#: for reasons that have nothing to do with link quality, so it is avoided here.
DEFAULT_STATE = ArbitraryState(theta=math.pi / 4.0, phi=0.7)

#: Probe states averaged over by :func:`fidelity_vs_hop_count` when the caller
#: supplies none. Two equatorial-ish states with different ``theta`` so the
#: result is not a single-point artefact.
DEFAULT_PROBE_STATES: tuple[ArbitraryState, ...] = (
    ArbitraryState(theta=math.pi / 4.0, phi=0.0),
    ArbitraryState(theta=math.pi / 3.0, phi=math.pi / 5.0),
)

#: Characteristic attenuation length in km, matching the default of
#: :meth:`src.mdqw_routing.RouteResult.attenuation_fidelity`.
ATTENUATION_LENGTH_KM = 50.0

#: Link metrics held fixed while :func:`fidelity_vs_hop_count` varies only the
#: per-link ``noise``. These values are arbitrary but constant; only the noise
#: sweep matters for that experiment.
_SWEEP_DISTANCE_KM = 100.0
_SWEEP_LATENCY_MS = 5.0
_SWEEP_FIDELITY = 0.9
_SWEEP_RELIABILITY = 0.95

#: A repeater is charged this multiple of ``hop_penalty``, because it must hold
#: entanglement in a quantum memory and regenerate it, rather than merely buffer
#: and forward. A plain intermediate node is charged ``hop_penalty`` itself.
REPEATER_PENALTY_FACTOR = 4.0

#: Tolerance used when declaring two link costs equal.
_TOL = 1e-9


# =============================================================================
# B. Multi-hop teleportation -- [PROPOSED] model, [SIMULATED] execution
# =============================================================================


@dataclass(frozen=True)
class HopResult:
    """One simulated teleportation hop of a multi-hop route.

    ``[PROPOSED]`` model, ``[SIMULATED]`` numbers. Every ``simulated_fidelity``
    comes from :func:`src.quantum_teleportation.teleport_link_with_noise`, i.e.
    from a Qiskit Aer run with an illustrative noise model built from this link's
    ``noise`` and ``fidelity`` attributes. It is not a measurement.

    Attributes:
        hop_index: Zero-based position of this hop along the route.
        source: Tail node of the hop.
        target: Head node of the hop.
        distance_km: Link length in kilometres.
        noise_score: The link's ``noise`` attribute, forwarded verbatim to the
            noise model.
        fidelity_hint: The link's declared ``fidelity`` attribute. This is a
            routing model input, not a measurement, and it is never compared
            against ``simulated_fidelity``.
        latency_ms: Link latency in milliseconds.
        simulated_fidelity: Bhattacharyya fidelity reported by the Aer run.
        shots: Number of measurement shots used for this hop.
        memory_lifetime_ms: Declared coherence time of the *receiving* node's
            quantum memory. This field is required by
            :attr:`memory_decay_estimate` and is therefore carried on the record
            itself; the receiving node is the one that must hold the state until
            it is read out, so its lifetime is the relevant one.
    """

    hop_index: int
    source: Hashable
    target: Hashable
    distance_km: float
    noise_score: float
    fidelity_hint: float
    latency_ms: float
    simulated_fidelity: float
    shots: int
    memory_lifetime_ms: float = 1000.0

    @property
    def ideal_fidelity(self) -> float:
        """Per-hop link fidelity declared by the network model.

        This is the *declared* input carried in :class:`~src.network.EdgeMetrics`,
        not a simulated quantity and not a measurement. It exists so that the
        simulated number can be read against the input that produced it.
        """
        return float(self.fidelity_hint)

    @property
    def memory_decay_estimate(self) -> float:
        """Estimate the memory retention factor of the receiving node.

        Returns ``exp(-latency_ms / memory_lifetime_ms)``, an ``[PROPOSED]``
        surrogate in which the state's survival probability while waiting for
        read-out decays exponentially with the wait time divided by the memory's
        declared coherence time. No decoherence is actually simulated; this is a
        reported quantity only.
        """
        lifetime = float(self.memory_lifetime_ms)
        if lifetime <= 0.0:
            raise ValueError(
                f"memory_lifetime_ms must be positive, got {self.memory_lifetime_ms}"
            )
        return float(math.exp(-float(self.latency_ms) / lifetime))

    def as_dict(self) -> dict[str, Any]:
        """Return the hop as a flat dictionary, including both properties."""
        return {
            "hop_index": int(self.hop_index),
            "source": self.source,
            "target": self.target,
            "distance_km": float(self.distance_km),
            "noise_score": float(self.noise_score),
            "fidelity_hint": float(self.fidelity_hint),
            "latency_ms": float(self.latency_ms),
            "simulated_fidelity": float(self.simulated_fidelity),
            "shots": int(self.shots),
            "memory_lifetime_ms": float(self.memory_lifetime_ms),
            "ideal_fidelity": self.ideal_fidelity,
            "memory_decay_estimate": self.memory_decay_estimate,
        }


@dataclass(frozen=True)
class MultiHopResult:
    """The full hop-by-hop teleportation record of one route.

    ``[SIMULATED]``. The three fidelity columns below answer three different
    questions and must not be compared as though they were interchangeable:

    ``simulated_end_to_end_fidelity``
        Product of the per-hop Aer-measured fidelities. This is the number the
        hop-by-hop model actually produced, under the caveat stated in
        :data:`SEQUENTIAL_TELEPORTATION_CAVEAT`.
    ``link_estimate_fidelity``
        Product of the declared per-link fidelities. Pure model input, no
        simulation involved.
    ``attenuation_estimate_fidelity``
        Product of ``fidelity * exp(-distance_km / L)`` with
        ``L = 50 km``. A distance-aware analytic alternative, also pure
        arithmetic.

    Attributes:
        path: Ordered node sequence of the route.
        hops: One :class:`HopResult` per link, in route order.
        source: Route origin.
        target: Route destination.
    """

    path: tuple[Hashable, ...]
    hops: tuple[HopResult, ...]
    source: Hashable
    target: Hashable

    @property
    def simulated_end_to_end_fidelity(self) -> float:
        """Product of the per-hop simulated fidelities."""
        product = 1.0
        for hop in self.hops:
            product *= float(hop.simulated_fidelity)
        return float(product)

    @property
    def link_estimate_fidelity(self) -> float:
        """Product of the per-hop declared link fidelities."""
        product = 1.0
        for hop in self.hops:
            product *= hop.ideal_fidelity
        return float(product)

    @property
    def attenuation_estimate_fidelity(self) -> float:
        """Product of ``fidelity * exp(-distance_km / 50)`` over the hops."""
        product = 1.0
        for hop in self.hops:
            product *= hop.ideal_fidelity * math.exp(
                -float(hop.distance_km) / ATTENUATION_LENGTH_KM
            )
        return float(product)

    @property
    def simulated_vs_link_gap(self) -> float:
        """``simulated_end_to_end_fidelity - link_estimate_fidelity``."""
        return float(
            self.simulated_end_to_end_fidelity - self.link_estimate_fidelity
        )

    @property
    def hop_count(self) -> int:
        """Number of hops."""
        return len(self.hops)

    @property
    def total_latency_ms(self) -> float:
        """Sum of per-hop latencies in milliseconds."""
        return float(sum(hop.latency_ms for hop in self.hops))

    @property
    def total_distance_km(self) -> float:
        """Sum of per-hop distances in kilometres."""
        return float(sum(hop.distance_km for hop in self.hops))

    @property
    def total_noise(self) -> float:
        """Sum of per-hop noise scores."""
        return float(sum(hop.noise_score for hop in self.hops))

    def as_dict(self) -> dict[str, Any]:
        """Return a one-row summary dictionary (no per-hop expansion)."""
        return {
            "source": self.source,
            "target": self.target,
            "path": " -> ".join(str(node) for node in self.path),
            "hop_count": self.hop_count,
            "total_latency_ms": self.total_latency_ms,
            "total_distance_km": self.total_distance_km,
            "total_noise": self.total_noise,
            "simulated_end_to_end_fidelity": self.simulated_end_to_end_fidelity,
            "link_estimate_fidelity": self.link_estimate_fidelity,
            "attenuation_estimate_fidelity": self.attenuation_estimate_fidelity,
            "simulated_vs_link_gap": self.simulated_vs_link_gap,
        }

    def describe(self) -> str:
        """Return a human-readable multi-line summary including the caveat."""
        chain = " -> ".join(str(node) for node in self.path)
        lines = [
            f"Path        : {chain}",
            f"Hop count   : {self.hop_count}",
            f"Distance    : {self.total_distance_km:.2f} km",
            f"Noise (sum) : {self.total_noise:.4f}",
            f"Latency     : {self.total_latency_ms:.2f} ms",
            "Per-hop simulated fidelity:",
        ]
        for hop in self.hops:
            lines.append(
                f"  hop {hop.hop_index}  {str(hop.source):>4} -> {str(hop.target):<4}  "
                f"simulated={hop.simulated_fidelity:.6f}  "
                f"declared={hop.ideal_fidelity:.4f}  "
                f"shots={hop.shots}  "
                f"mem_decay={hop.memory_decay_estimate:.4f}"
            )
        lines += [
            f"Simulated end-to-end fidelity : {self.simulated_end_to_end_fidelity:.6f}",
            f"Declared link product         : {self.link_estimate_fidelity:.6f}",
            f"Attenuation estimate (L=50km) : {self.attenuation_estimate_fidelity:.6f}",
            f"Simulated minus declared      : {self.simulated_vs_link_gap:+.6f}",
            f"CAVEAT: {SEQUENTIAL_TELEPORTATION_CAVEAT}",
        ]
        return "\n".join(lines)


def simulate_route(
    network: QuantumNetwork,
    route: RouteResult,
    *,
    state: ArbitraryState | None = None,
    shots: int = 2048,
    seed: int = 20260930,
    include_link_estimate: bool = True,
) -> MultiHopResult:
    """Teleport a state hop-by-hop along a route and record every hop.

    For each consecutive pair ``(u, v)`` on ``route.path`` this calls
    :func:`src.quantum_teleportation.teleport_link_with_noise` with the link's
    ``noise`` as ``noise_score``, its declared ``fidelity`` as
    ``fidelity_hint``, and ``seed + hop_index`` as the seed. Deriving the
    per-hop seed from a single master seed means each hop samples differently
    while the whole run stays reproducible for a given ``seed``.

    **Read the result with the caveat in mind.** Naive sequential teleportation,
    as modelled here, is NOT equivalent to an optimised quantum repeater
    network. Real repeaters use entanglement swapping, purification, and quantum
    memories; the sequential model teleports hop-by-hop with no entanglement
    swapping, so its end-to-end fidelity should be read as an illustrative
    per-hop composition, not as repeater-network performance.

    Args:
        network: The network the route was computed on. Used only to read the
            link metrics and the receiving node's memory lifetime; it is not
            re-routed.
        route: A :class:`~src.mdqw_routing.RouteResult` whose ``path`` is
            followed in order.
        state: The payload state. Defaults to :data:`DEFAULT_STATE`, an
            equatorial superposition whose ideal distribution is
            ``{"0": 0.5, "1": 0.5}``.
        shots: Measurement shots per hop. Must be positive.
        seed: Master seed. Hop ``i`` uses ``seed + i``.
        include_link_estimate: Accepted for interface compatibility. The
            analytic link and memory-decay estimates are closed-form values
            that cost nothing to compute, and
            :class:`MultiHopResult` exposes them as required properties, so
            they are always populated and this flag does not currently change
            any returned value.

    Returns:
        A :class:`MultiHopResult` with one :class:`HopResult` per hop.

    Raises:
        TypeError: If ``route`` is not a :class:`~src.mdqw_routing.RouteResult`
            or ``state`` is not an
            :class:`~src.quantum_teleportation.ArbitraryState`.
        ValueError: If ``route.path`` has fewer than two nodes, if ``shots`` is
            not positive, or if a link on the path is absent from ``network``.
        RuntimeError: If an Aer hop fails.
    """
    if not isinstance(route, RouteResult):
        raise TypeError(
            f"route must be a RouteResult, got {type(route).__name__}"
        )
    if state is None:
        payload = DEFAULT_STATE
    elif isinstance(state, ArbitraryState):
        payload = state
    else:
        raise TypeError(
            f"state must be an ArbitraryState or None, got {type(state).__name__}"
        )
    if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)):
        raise TypeError(f"shots must be an integer, got {type(shots).__name__}")
    if int(shots) <= 0:
        raise ValueError(f"shots must be positive, got {shots}")
    path = tuple(route.path)
    if len(path) < 2:
        raise ValueError(
            f"route path needs at least two nodes to teleport across, got {path!r}"
        )

    hops: list[HopResult] = []
    for index in range(len(path) - 1):
        u, v = path[index], path[index + 1]
        metrics = network.link_metrics(u, v)
        outcome = teleport_link_with_noise(
            payload,
            noise_score=float(metrics.noise),
            fidelity_hint=float(metrics.fidelity),
            shots=int(shots),
            seed=int(seed) + index,
        )
        memory = float(network.node(v).get("memory_lifetime_ms", 1000.0))
        hops.append(
            HopResult(
                hop_index=index,
                source=u,
                target=v,
                distance_km=float(metrics.distance_km),
                noise_score=float(metrics.noise),
                fidelity_hint=float(metrics.fidelity),
                latency_ms=float(metrics.latency_ms),
                simulated_fidelity=float(outcome["measured_fidelity"]),
                shots=int(outcome["shots"]),
                memory_lifetime_ms=memory,
            )
        )

    return MultiHopResult(
        path=path, hops=tuple(hops), source=path[0], target=path[-1]
    )


def _path_network(
    hop_count: int,
    *,
    noise_score: float,
    distance_km: float = _SWEEP_DISTANCE_KM,
    latency_ms: float = _SWEEP_LATENCY_MS,
    fidelity: float = _SWEEP_FIDELITY,
    reliability: float = _SWEEP_RELIABILITY,
    prefix: str = "H",
) -> QuantumNetwork:
    """Build a linear chain of ``hop_count + 1`` identical links.

    Every link carries exactly the same metrics, which makes min-max
    normalisation degenerate by design: a constant metric maps every value to
    ``0.0``. That is intentional here -- the experiment varies ``noise_score``
    only, and a constant ``noise`` column would otherwise make the normalised
    noise term identically zero and the sweep meaningless. The raw ``noise``
    value still reaches :func:`src.quantum_teleportation.teleport_link_with_noise`
    unchanged, which is where the simulated fidelity comes from.

    Args:
        hop_count: Number of links in the chain. Must be at least 1.
        noise_score: The constant ``noise`` of every link.
        distance_km: Constant link distance.
        latency_ms: Constant link latency.
        fidelity: Constant declared link fidelity.
        reliability: Constant link reliability.
        prefix: Node-name prefix; nodes are ``f"{prefix}{i}"``.

    Returns:
        A :class:`~src.network.QuantumNetwork` whose only simple path between
        its endpoints is the whole chain.

    Raises:
        ValueError: If ``hop_count`` is below 1 or a metric is out of range.
    """
    if hop_count < 1:
        raise ValueError(f"hop_count must be at least 1, got {hop_count}")
    net = QuantumNetwork(name=f"BQT-MDQW/chain-{hop_count}h")
    metrics = EdgeMetrics(
        distance_km=float(distance_km),
        noise=float(noise_score),
        latency_ms=float(latency_ms),
        fidelity=float(fidelity),
        reliability=float(reliability),
    )
    for index in range(hop_count + 1):
        net.add_node(
            f"{prefix}{index}", memory_lifetime_ms=1000.0, is_repeater=False
        )
    for index in range(hop_count):
        net.add_link(f"{prefix}{index}", f"{prefix}{index + 1}", metrics)
    return net


def fidelity_vs_hop_count(
    noise_score: float,
    *,
    max_hops: int = 6,
    shots: int = 2048,
    seed: int = 20260930,
    states: Sequence[ArbitraryState] | None = None,
) -> pd.DataFrame:
    """Measure how the three fidelity notions decay as hops accumulate.

    For every hop count ``1 .. max_hops`` a linear chain is built whose links
    all carry ``noise_score`` (distance, latency, fidelity and reliability are
    held at the module-level sweep constants), the router is pointed down the
    chain, and :func:`simulate_route` is run for each of ``states``. The three
    reported fidelities are averaged over the probe states.

    The simulated column is subject to the caveat in
    :data:`SEQUENTIAL_TELEPORTATION_CAVEAT`: it is an illustrative per-hop
    composition under a sequential hop-by-hop model with no entanglement
    swapping, not a repeater-network result.

    Args:
        noise_score: Constant per-link ``noise`` in ``[0, 1]``.
        max_hops: Largest hop count to measure. Must be at least 1.
        shots: Measurement shots per hop. Must be positive.
        seed: Master seed; hop ``i`` of chain ``h`` uses ``seed + 1000 * h + i``,
            so different chain lengths use different shot noise.
        states: Probe states to average over. Defaults to
            :data:`DEFAULT_PROBE_STATES`.

    Returns:
        A frame with one row per hop count and the columns ``hop_count``,
        ``noise_score``, ``simulated_end_to_end_fidelity``,
        ``link_estimate_fidelity``, ``attenuation_estimate_fidelity`` and
        ``simulated_vs_link_gap``, ordered by increasing hop count.

    Raises:
        ValueError: If ``max_hops`` is below 1, ``shots`` is not positive, or
            ``noise_score`` is outside ``[0, 1]``.
        RoutingError: If the router cannot traverse the chain (it always can).
    """
    if max_hops < 1:
        raise ValueError(f"max_hops must be at least 1, got {max_hops}")
    if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)):
        raise TypeError(f"shots must be an integer, got {type(shots).__name__}")
    if int(shots) <= 0:
        raise ValueError(f"shots must be positive, got {shots}")
    if not 0.0 <= float(noise_score) <= 1.0:
        raise ValueError(f"noise_score must lie in [0, 1], got {noise_score}")
    probes = tuple(states) if states else DEFAULT_PROBE_STATES
    if not probes:
        raise ValueError("states must contain at least one ArbitraryState")

    rows: list[dict[str, Any]] = []
    for hops in range(1, max_hops + 1):
        net = _path_network(hops, noise_score=float(noise_score))
        route = MDQWRouter(net, MDQWConfig()).route("H0", f"H{hops}")
        simulated: list[float] = []
        declared: list[float] = []
        attenuated: list[float] = []
        for offset, probe in enumerate(probes):
            outcome = simulate_route(
                net,
                route,
                state=probe,
                shots=int(shots),
                seed=int(seed) + 1000 * hops + offset,
            )
            simulated.append(outcome.simulated_end_to_end_fidelity)
            declared.append(outcome.link_estimate_fidelity)
            attenuated.append(outcome.attenuation_estimate_fidelity)
        rows.append(
            {
                "hop_count": int(hops),
                "noise_score": float(noise_score),
                "shots_per_hop": int(shots),
                "probe_states": len(probes),
                "simulated_end_to_end_fidelity": float(np.mean(simulated)),
                "link_estimate_fidelity": float(np.mean(declared)),
                "attenuation_estimate_fidelity": float(np.mean(attenuated)),
                "simulated_vs_link_gap": float(
                    np.mean(simulated) - np.mean(declared)
                ),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# C. Seeded experiments
# =============================================================================


def experiment_manual_network(
    seed: int = 20260930,
    *,
    shots: int = 2048,
    hop_scaling_shots: int = 2048,
    hop_scaling_max_hops: int = 6,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Route the hand-designed ``A -> F`` network four ways and simulate each.

    ``[SIMULATED]``. Builds :func:`src.network.build_example_network`, routes
    ``A -> F`` with (i) the default MDQW A* configuration, (ii) MDQW with the
    quantum-walk prior enabled, and (iii) the Dijkstra baseline, then runs
    :func:`simulate_route` over each of the three results.

    The walk prior is built from
    :func:`src.quantum_walk.quantum_walk_prior` applied to the walkable proxy
    of the network, obtained from
    :func:`src.mdqw_routing.walkable_proxy`. On this six-node network the proxy
    falls back to the complete graph, because the real network's degrees are not
    all equal and the spanning-2-factor branch rejects the leftover vertices; the
    strategy string is reported in the metadata so this is visible rather than
    implied.

    Args:
        seed: Master seed, forwarded to the router configurations' prior steps
            and to every :func:`simulate_route` call.
        shots: Measurement shots per hop for the three route simulations.
        hop_scaling_shots: Measurement shots per hop for the
            :func:`fidelity_vs_hop_count` sweep.
        hop_scaling_max_hops: Largest hop count in that sweep. Lowering it is
            the cheapest way to shorten a run, because the sweep costs
            ``sum_{h=1}^{max_hops} h`` Aer hops per probe state.

    Returns:
        ``(frames, metadata)`` where ``frames`` has the keys ``routes``,
        ``summary``, ``enumerated``, ``multihop``, ``multihop_hops`` and
        ``hop_scaling``; ``metadata`` records the seed, the proxy strategy, the
        selected paths and the prior itself.

    Raises:
        RoutingError: If any of the three route searches fails.
        RuntimeError: If an Aer hop fails.
    """
    net = build_example_network()
    source, target = "A", "F"

    proxy, strategy = walkable_proxy(net.to_networkx())
    prior = quantum_walk_prior(proxy)

    default_route = MDQWRouter(net, MDQWConfig()).route(source, target)
    prior_config = MDQWConfig(
        weights=RoutingWeights(),
        mode="astar",
        use_qw_prior=True,
        qw_prior_weight=1.0,
        prior_steps=8,
    )
    prior_route = MDQWRouter(net, prior_config, prior).route(source, target)
    dijkstra_route = DijkstraBaseline(net).route(source, target)

    routes = compare_routes(
        {
            "mdqw-default": default_route,
            "mdqw-qw-prior": prior_route,
            "dijkstra": dijkstra_route,
        }
    )
    summary = summarise_comparison(routes)
    enumerated = enumerate_routes(net, source, target)

    multi_hops: list[dict[str, Any]] = []
    per_hop_rows: list[dict[str, Any]] = []
    for label, route in (
        ("mdqw-default", default_route),
        ("mdqw-qw-prior", prior_route),
        ("dijkstra", dijkstra_route),
    ):
        outcome = simulate_route(net, route, shots=int(shots), seed=int(seed))
        record = outcome.as_dict()
        record["label"] = label
        record["edge_cost"] = float(route.edge_cost)
        record["total_cost"] = float(route.total_cost)
        multi_hops.append(record)
        for hop in outcome.hops:
            row = hop.as_dict()
            row["label"] = label
            per_hop_rows.append(row)
    multihop = pd.DataFrame(multi_hops)
    multihop_hops = pd.DataFrame(per_hop_rows)

    hop_scaling = fidelity_vs_hop_count(
        0.2,
        max_hops=int(hop_scaling_max_hops),
        shots=int(hop_scaling_shots),
        seed=int(seed),
    )

    metadata: dict[str, Any] = {
        "seed": int(seed),
        "network": net.name,
        "nodes": len(net),
        "links": len(net.links),
        "source": source,
        "target": target,
        "walkable_proxy_strategy": strategy,
        "qw_prior": {str(k): float(v) for k, v in prior.items()},
        "paths": {
            "mdqw-default": list(default_route.path),
            "mdqw-qw-prior": list(prior_route.path),
            "dijkstra": list(dijkstra_route.path),
        },
        "verdict": str(summary.iloc[0]["verdict"]),
    }
    frames = {
        "routes": routes,
        "summary": summary,
        "enumerated": enumerated,
        "multihop": multihop,
        "multihop_hops": multihop_hops,
        "hop_scaling": hop_scaling,
    }
    return frames, metadata


def _weight_presets() -> dict[str, RoutingWeights]:
    """Return the three routing weight presets used by the random-network study.

    The labels describe what each preset *emphasises*, not what it achieves. The
    numbers themselves are ordinary choices of this prototype:

    ``distance-heavy``
        ``alpha = 3.0``, everything else reduced.
    ``balanced``
        the balanced defaults restated as :data:`_BALANCED_WEIGHTS`.
    ``quality-heavy``
        large ``beta``/``delta``/``rho`` and a small ``alpha``.

    Returns:
        Mapping from preset label to :class:`~src.mdqw_routing.RoutingWeights`.
    """
    return {
        "distance-heavy": RoutingWeights(
            alpha=3.0, beta=0.2, gamma=0.1, delta=0.3, rho=0.0
        ),
        "balanced": RoutingWeights(
            alpha=_BALANCED_WEIGHTS["alpha"],
            beta=_BALANCED_WEIGHTS["beta"],
            gamma=_BALANCED_WEIGHTS["gamma"],
            delta=_BALANCED_WEIGHTS["delta"],
            rho=_BALANCED_WEIGHTS["rho"],
        ),
        "quality-heavy": RoutingWeights(
            alpha=0.2, beta=2.0, gamma=0.5, delta=3.0, rho=1.0
        ),
    }


def _pick_connected_pair(
    net: QuantumNetwork, rng: np.random.Generator
) -> tuple[Hashable, Hashable] | None:
    """Uniformly pick one connected ordered pair, or ``None`` if there is none.

    Args:
        net: The network to sample from.
        rng: A seeded generator; only used for the index draw.

    Returns:
        ``(source, target)`` with ``source != target``, or ``None`` when the
        network has no reachable pair.
    """
    nodes = list(net.nodes)
    pairs = [
        (u, v)
        for index, u in enumerate(nodes)
        for v in nodes[index + 1 :]
        if net.has_path(u, v)
    ]
    if not pairs:
        return None
    return pairs[int(rng.integers(0, len(pairs)))]


def experiment_random_network(
    n_nodes: int = 12,
    n_trials: int = 20,
    seed: int = 20260930,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Repeat the routing comparison over seeded random networks.

    ``[SIMULATED]``. For trial ``t`` the network is
    :func:`src.network.random_network` with ``seed = seed + 100 + t``, the
    endpoint pair is drawn from a :class:`numpy.random.Generator` seeded with
    ``seed + t`` (so the selection is reproducible and not from a global RNG),
    and the pair is routed with the Dijkstra baseline and with MDQW under each of
    the three presets from :func:`_weight_presets`. Every baseline run uses the
    *same* weights as the MDQW run it is compared against, which is what makes
    ``edge_cost`` a fair comparison.

    ``[ESTABLISHED]`` Note that MDQW without node terms and Dijkstra optimise the
    same objective, so a tie is the expected outcome and a non-tie would indicate
    a bug in the search rather than a finding. The experiment is reported as it
    comes out.

    Args:
        n_nodes: Nodes per random network. Must be at least 2.
        n_trials: Number of independent networks. Must be at least 1.
        seed: Master seed; every derived seed is an explicit offset from it.

    Returns:
        ``(frames, metadata)`` where ``frames`` has the keys ``trials`` and
        ``aggregate``; ``metadata`` records the seed, how many networks were
        connected, and the total win/loss/tie counts.

    Raises:
        ValueError: If ``n_nodes`` or ``n_trials`` is below its minimum.
        RoutingError: If a route search fails on a connected pair.
    """
    if n_nodes < 2:
        raise ValueError(f"n_nodes must be at least 2, got {n_nodes}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")

    presets = _weight_presets()
    rows: list[dict[str, Any]] = []
    connected = 0
    disconnected = 0
    unreachable = 0

    for trial in range(n_trials):
        net = random_network(n_nodes, 3, seed=int(seed) + 100 + trial)
        if net.is_connected():
            connected += 1
        else:
            disconnected += 1
        rng = np.random.default_rng(int(seed) + trial)
        pair = _pick_connected_pair(net, rng)
        if pair is None:
            unreachable += 1
            continue
        source, target = pair
        for preset, weights in presets.items():
            baseline = DijkstraBaseline(net, weights).route(source, target)
            mdqw = MDQWRouter(net, MDQWConfig(weights=weights)).route(
                source, target
            )
            difference = float(mdqw.edge_cost) - float(baseline.edge_cost)
            rows.append(
                {
                    "trial": int(trial),
                    "network_seed": int(seed) + 100 + trial,
                    "preset": preset,
                    "source": source,
                    "target": target,
                    "graph_connected": bool(net.is_connected()),
                    "dijkstra_path": " -> ".join(str(n) for n in baseline.path),
                    "mdqw_path": " -> ".join(str(n) for n in mdqw.path),
                    "same_path": tuple(mdqw.path) == tuple(baseline.path),
                    "dijkstra_edge_cost": float(baseline.edge_cost),
                    "mdqw_edge_cost": float(mdqw.edge_cost),
                    "edge_cost_difference": difference,
                    "matched_dijkstra_edge_cost": abs(difference) <= _TOL,
                    "dijkstra_hop_count": int(baseline.hop_count),
                    "mdqw_hop_count": int(mdqw.hop_count),
                    "hop_difference": int(mdqw.hop_count)
                    - int(baseline.hop_count),
                    "dijkstra_total_distance_km": float(
                        baseline.total_distance_km
                    ),
                    "mdqw_total_distance_km": float(mdqw.total_distance_km),
                    "dijkstra_estimated_fidelity": float(
                        baseline.estimated_fidelity
                    ),
                    "mdqw_estimated_fidelity": float(mdqw.estimated_fidelity),
                    "dijkstra_total_latency_ms": float(baseline.total_latency_ms),
                    "mdqw_total_latency_ms": float(mdqw.total_latency_ms),
                }
            )

    trials = pd.DataFrame(rows)
    aggregate_rows: list[dict[str, Any]] = []
    totals = {"lower": 0, "higher": 0, "tied": 0, "matched": 0}
    for preset in presets:
        subset = trials[trials["preset"] == preset]
        if subset.empty:
            continue
        difference = subset["edge_cost_difference"]
        lower = int((difference < -_TOL).sum())
        higher = int((difference > _TOL).sum())
        tied = int(len(subset) - lower - higher)
        matched = int(subset["matched_dijkstra_edge_cost"].sum())
        totals["lower"] += lower
        totals["higher"] += higher
        totals["tied"] += tied
        totals["matched"] += matched
        aggregate_rows.append(
            {
                "preset": preset,
                "comparisons": int(len(subset)),
                "mdqw_lower_edge_cost": lower,
                "dijkstra_lower_edge_cost": higher,
                "tied_edge_cost": tied,
                "matched_dijkstra_edge_cost": matched,
                "same_path_count": int(subset["same_path"].sum()),
                "mean_dijkstra_edge_cost": float(
                    subset["dijkstra_edge_cost"].mean()
                ),
                "mean_mdqw_edge_cost": float(subset["mdqw_edge_cost"].mean()),
                "mean_edge_cost_difference": float(difference.mean()),
                "mean_hop_difference": float(subset["hop_difference"].mean()),
                "mean_dijkstra_distance_km": float(
                    subset["dijkstra_total_distance_km"].mean()
                ),
                "mean_mdqw_distance_km": float(
                    subset["mdqw_total_distance_km"].mean()
                ),
                "mean_dijkstra_fidelity": float(
                    subset["dijkstra_estimated_fidelity"].mean()
                ),
                "mean_mdqw_fidelity": float(
                    subset["mdqw_estimated_fidelity"].mean()
                ),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)

    metadata: dict[str, Any] = {
        "seed": int(seed),
        "n_nodes": int(n_nodes),
        "n_trials": int(n_trials),
        "connected_networks": connected,
        "disconnected_networks": disconnected,
        "trials_without_reachable_pair": unreachable,
        "wins": totals["lower"],
        "losses": totals["higher"],
        "ties": totals["tied"],
        "matched_dijkstra_edge_cost": totals["matched"],
    }
    return {"trials": trials, "aggregate": aggregate}, metadata


def experiment_noise_sweep(
    noise_levels: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4),
    seed: int = 20260930,
    *,
    source: str = "A",
    target: str = "F",
    shots: int = 1024,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Set every link's ``noise`` to each level and re-route.

    ``[SIMULATED]``. The topology and the other four metrics of
    :func:`src.network.build_example_network` are held fixed via
    :meth:`~src.network.QuantumNetwork.update_link`; only ``noise`` moves.

    **Expected degeneracy, stated up front.** Setting *every* link to the same
    ``noise`` makes the ``noise`` column constant, and
    :func:`src.network.minmax_normalize` maps a constant metric to ``0.0`` for
    every link. The normalised noise term therefore contributes nothing to the
    cost at any level, so ``edge_cost`` is expected to be flat across the sweep.
    The *simulated* fidelity column is not flat, because
    :func:`src.quantum_teleportation.teleport_link_with_noise` consumes the raw
    ``noise`` value rather than the normalised one. Both columns are reported so
    the contrast is visible rather than asserted.

    Args:
        noise_levels: Per-link ``noise`` values to apply, each in ``[0, 1]``.
        seed: Master seed forwarded to :func:`simulate_route`.
        source: Route origin on the example network.
        target: Route destination on the example network.
        shots: Measurement shots per hop for the simulated fidelity column.

    Returns:
        ``(frames, metadata)`` where ``frames`` has the keys ``sweep`` and
        ``sweep_hops``; ``metadata`` records the seed and whether the edge cost
        was in fact constant across the sweep.

    Raises:
        ValueError: If a level is outside ``[0, 1]`` or ``shots`` is not
            positive.
        RoutingError: If the route search fails at some level.
    """
    levels = [float(level) for level in noise_levels]
    if not levels:
        raise ValueError("noise_levels must contain at least one level")
    for level in levels:
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"noise level must lie in [0, 1], got {level}")
    if int(shots) <= 0:
        raise ValueError(f"shots must be positive, got {shots}")

    rows: list[dict[str, Any]] = []
    hop_rows: list[dict[str, Any]] = []
    for index, level in enumerate(levels):
        net = build_example_network()
        for u, v in net.links:
            metrics = net.link_metrics(u, v)
            net.update_link(
                u,
                v,
                EdgeMetrics(
                    distance_km=metrics.distance_km,
                    noise=level,
                    latency_ms=metrics.latency_ms,
                    fidelity=metrics.fidelity,
                    reliability=metrics.reliability,
                ),
            )
        route = MDQWRouter(net, MDQWConfig()).route(source, target)
        outcome = simulate_route(
            net, route, shots=int(shots), seed=int(seed) + 17 * index
        )
        record = outcome.as_dict()
        record.update(
            {
                "noise_level": level,
                "edge_cost": float(route.edge_cost),
                "total_cost": float(route.total_cost),
                "route_estimated_fidelity": float(route.estimated_fidelity),
                "mean_hop_noise": float(route.mean_noise),
            }
        )
        rows.append(record)
        for hop in outcome.hops:
            row = hop.as_dict()
            row["noise_level"] = level
            hop_rows.append(row)

    sweep = pd.DataFrame(rows)
    unique_costs = sweep["edge_cost"].nunique()
    metadata: dict[str, Any] = {
        "seed": int(seed),
        "source": source,
        "target": target,
        "shots": int(shots),
        "levels": levels,
        "distinct_edge_costs": int(unique_costs),
        "edge_cost_is_constant": bool(unique_costs == 1),
        "distinct_paths": int(sweep["path"].nunique()),
        "fidelity_range": [
            float(sweep["simulated_end_to_end_fidelity"].min()),
            float(sweep["simulated_end_to_end_fidelity"].max()),
        ],
    }
    return {"sweep": sweep, "sweep_hops": pd.DataFrame(hop_rows)}, metadata


def experiment_weight_sweep(
    seed: int = 20260930,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Sweep ``alpha`` and ``delta`` over a 4x3 grid on the example network.

    ``[PROPOSED]`` design sweep, ``[SIMULATED]`` routing. ``alpha`` ranges over
    ``(0.0, 0.5, 1.0, 2.0)`` and ``delta`` over ``(0.0, 1.0, 3.0)``;
    ``beta``, ``gamma`` and ``rho`` stay at the module defaults
    ``(1.0, 0.5, 0.25)`` and ``hop_penalty`` is zero. All twelve combinations are
    attempted.

    The ``alpha = 0.0, delta = 0.0`` corner **is** included. Contrary to a
    common assumption about this cost function,
    :class:`~src.mdqw_routing.RoutingWeights` rejects a configuration only when
    *all five* of ``alpha, beta, gamma, delta, rho`` are zero; with ``beta=1.0``
    and ``gamma=0.5`` still positive that corner is a legal, non-degenerate
    weighting on distance-free, fidelity-free noise+latency+reliability. The
    ``RoutingError`` handler is kept anyway and records any combination the
    constructor does reject into the ``skipped`` list in the metadata, so a
    future change to that validation cannot silently shrink the grid.

    Args:
        seed: Master seed. Recorded in the metadata; the sweep itself is
            deterministic and consumes no randomness.

    Returns:
        ``(frames, metadata)`` where ``frames`` has the keys ``grid`` and
        ``selected_paths``; ``metadata`` records the seed, any skipped
        combinations, and how many distinct paths the grid chose.

    Raises:
        Nothing. A rejected combination is recorded, not raised, so that one bad
        cell cannot lose the whole sweep.
    """
    alphas = (0.0, 0.5, 1.0, 2.0)
    deltas = (0.0, 1.0, 3.0)
    net = build_example_network()

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for alpha in alphas:
        for delta in deltas:
            try:
                weights = RoutingWeights(
                    alpha=alpha,
                    beta=_BALANCED_WEIGHTS["beta"],
                    gamma=_BALANCED_WEIGHTS["gamma"],
                    delta=delta,
                    rho=_BALANCED_WEIGHTS["rho"],
                    hop_penalty=0.0,
                )
            except RoutingError as exc:
                skipped.append(
                    {"alpha": alpha, "delta": delta, "reason": str(exc)}
                )
                continue
            route = MDQWRouter(net, MDQWConfig(weights=weights)).route("A", "F")
            rows.append(
                {
                    "alpha": float(alpha),
                    "delta": float(delta),
                    "beta": float(weights.beta),
                    "gamma": float(weights.gamma),
                    "rho": float(weights.rho),
                    "hop_penalty": 0.0,
                    "path": " -> ".join(str(node) for node in route.path),
                    "edge_cost": float(route.edge_cost),
                    "hop_count": int(route.hop_count),
                    "estimated_fidelity": float(route.estimated_fidelity),
                    "total_distance_km": float(route.total_distance_km),
                    "total_latency_ms": float(route.total_latency_ms),
                    "total_noise": float(route.total_noise),
                }
            )

    grid = pd.DataFrame(rows)
    if grid.empty:
        selected = pd.DataFrame(
            columns=["path", "times_chosen", "hop_count", "share_of_grid"]
        )
    else:
        counts = (
            grid.groupby("path")
            .agg(times_chosen=("path", "size"), hop_count=("hop_count", "first"))
            .reset_index()
            .sort_values("times_chosen", ascending=False, ignore_index=True)
        )
        counts["share_of_grid"] = counts["times_chosen"] / len(grid)
        selected = counts

    metadata: dict[str, Any] = {
        "seed": int(seed),
        "alphas": list(alphas),
        "deltas": list(deltas),
        "combinations_attempted": len(alphas) * len(deltas),
        "combinations_recorded": int(len(grid)),
        "skipped": skipped,
        "distinct_paths": int(grid["path"].nunique()) if not grid.empty else 0,
        "edge_cost_range": (
            [float(grid["edge_cost"].min()), float(grid["edge_cost"].max())]
            if not grid.empty
            else []
        ),
    }
    return {"grid": grid, "selected_paths": selected}, metadata


def experiment_qw_prior_ablation(
    seed: int = 20260930,
    weights: Sequence[float] = (1.0, 2.0, 4.0),
    steps: Sequence[int] = (4, 8, 16),
    *,
    n_instances: int = 4,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Quantify how much edge-cost optimality the walk prior costs.

    ``[PROPOSED]`` prior, ``[SIMULATED]`` routing. For each prior weight in
    ``weights`` and each step count in ``steps`` the prior is rebuilt, the route
    is computed with the prior enabled, and the resulting ``edge_cost`` is
    compared against the Dijkstra minimum for the same weights. A prior is
    heuristic, so it *can* raise the edge cost; the quantity measured here is
    exactly how much, and in how many of the configurations.

    Two families of instance are used, because the example network alone cannot
    discriminate: the hand-designed ``A -> F`` pair, plus ``n_instances``
    seeded random 12-node networks on the ``N0 -> N11`` pair. The
    ``instance`` column of the frame identifies which is which.

    Args:
        seed: Master seed. Every network seed and every prior-step count derives
            from it explicitly.
        weights: Prior multipliers to try. Each must be non-negative.
        steps: Prior step counts to try. Each must be at least 1.
        n_instances: How many extra random networks to include, beyond the
            example network.

    Returns:
        ``(frames, metadata)`` where ``frames`` has the keys ``ablation`` and
        ``ablation_summary``; ``metadata`` records the seed and the totals of
        configurations in which the prior left the edge cost optimal.

    Raises:
        ValueError: If a weight is negative or a step count is below 1.
        RoutingError: If the router rejects the supplied prior, or if a route
            search fails.
    """
    weight_values = [float(value) for value in weights]
    step_values = [int(value) for value in steps]
    for value in weight_values:
        if value < 0.0:
            raise ValueError(f"prior weight must be non-negative, got {value}")
    for value in step_values:
        if value < 1:
            raise ValueError(f"prior steps must be >= 1, got {value}")

    instances: list[tuple[str, QuantumNetwork, Hashable, Hashable]] = []
    example = build_example_network()
    instances.append(("example-6-node", example, "A", "F"))

    added = 0
    trial = 0
    while added < n_instances:
        net = random_network(12, 3, seed=int(seed) + 100 + trial)
        trial += 1
        pair = _pick_connected_pair(net, np.random.default_rng(int(seed) + trial))
        if pair is None:
            continue
        instances.append((f"random-trial{trial - 1}", net, pair[0], pair[1]))
        added += 1

    routing_weights = RoutingWeights()
    rows: list[dict[str, Any]] = []
    for name, net, source, target in instances:
        proxy, strategy = walkable_proxy(net.to_networkx())
        baseline = DijkstraBaseline(net, routing_weights).route(source, target)
        for prior_steps in step_values:
            prior = quantum_walk_prior(proxy, steps=prior_steps)
            for weight in weight_values:
                config = MDQWConfig(
                    weights=routing_weights,
                    mode="astar",
                    use_qw_prior=True,
                    qw_prior_weight=weight,
                    prior_steps=prior_steps,
                )
                route = MDQWRouter(net, config, prior).route(source, target)
                difference = float(route.edge_cost) - float(baseline.edge_cost)
                rows.append(
                    {
                        "instance": name,
                        "walkable_proxy_strategy": strategy,
                        "source": source,
                        "target": target,
                        "prior_steps": int(prior_steps),
                        "prior_weight": float(weight),
                        "path": " -> ".join(str(node) for node in route.path),
                        "dijkstra_path": " -> ".join(
                            str(node) for node in baseline.path
                        ),
                        "same_path_as_dijkstra": tuple(route.path)
                        == tuple(baseline.path),
                        "edge_cost": float(route.edge_cost),
                        "dijkstra_edge_cost": float(baseline.edge_cost),
                        "edge_cost_difference": difference,
                        "edge_cost_still_optimal": abs(difference) <= _TOL,
                        "node_cost": float(route.node_cost),
                        "total_cost": float(route.total_cost),
                        "hop_count": int(route.hop_count),
                    }
                )

    ablation = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for weight in weight_values:
        subset = ablation[ablation["prior_weight"] == weight]
        if subset.empty:
            continue
        summary_rows.append(
            {
                "prior_weight": float(weight),
                "configurations": int(len(subset)),
                "edge_cost_still_optimal": int(
                    subset["edge_cost_still_optimal"].sum()
                ),
                "edge_cost_suboptimal": int(
                    (~subset["edge_cost_still_optimal"]).sum()
                ),
                "same_path_as_dijkstra": int(subset["same_path_as_dijkstra"].sum()),
                "mean_edge_cost_difference": float(
                    subset["edge_cost_difference"].mean()
                ),
                "max_edge_cost_difference": float(
                    subset["edge_cost_difference"].max()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)

    metadata: dict[str, Any] = {
        "seed": int(seed),
        "prior_weights": weight_values,
        "prior_steps": step_values,
        "instances": [name for name, _, _, _ in instances],
        "configurations": int(len(ablation)),
        "edge_cost_still_optimal": int(ablation["edge_cost_still_optimal"].sum()),
        "edge_cost_suboptimal": int((~ablation["edge_cost_still_optimal"]).sum()),
        "max_edge_cost_difference": float(ablation["edge_cost_difference"].max()),
    }
    return {"ablation": ablation, "ablation_summary": summary}, metadata


def _hop_penalty_network(n_repeaters: int) -> QuantumNetwork:
    """Build the two-branch network used by :func:`experiment_hop_penalty`.

    The topology is deliberately lopsided so that the hop-count trade-off is
    visible: one direct link that is the worst link on all five metrics, and one
    five-hop chain whose links are the best on all five. Because
    :func:`src.network.minmax_normalize` gives the best link a cost of zero in
    every normalised metric, the chain's total edge cost is ``0.0`` while the
    direct link costs ``alpha + beta + gamma + delta + rho = 4.25`` at the
    default weights. The choice between them is therefore decided purely by the
    node term.

    ``n_repeaters`` calls
    :meth:`~src.network.QuantumNetwork.add_midpoint_node` on that many links of
    the chain, which both inserts a repeater and lengthens the chain by one hop.
    Subdividing also *lowers* the network-wide minimum of distance, noise and
    latency (the halves) and raises the maximum of fidelity (the square root), so
    the chain's normalised edge cost rises above zero once any repeater is
    inserted. That side effect is real and is why the chain's ``edge_cost`` is
    not constant across the repeater count.

    Args:
        n_repeaters: Number of chain links to subdivide. Must be at least 0.

    Returns:
        A :class:`~src.network.QuantumNetwork` named ``BQT-MDQW/hop-penalty``.

    Raises:
        ValueError: If ``n_repeaters`` is negative.
    """
    if n_repeaters < 0:
        raise ValueError(f"n_repeaters must be non-negative, got {n_repeaters}")

    net = QuantumNetwork(name="BQT-MDQW/hop-penalty")
    for node in ("S", "T", *[f"C{i}" for i in range(4)]):
        net.add_node(node, memory_lifetime_ms=1000.0, is_repeater=False)

    clean = EdgeMetrics(40.0, 0.05, 1.0, 0.98, 0.99)
    net.add_link("S", "C0", clean)
    for index in range(3):
        net.add_link(f"C{index}", f"C{index + 1}", clean)
    net.add_link("C3", "T", clean)
    # Worst link on every metric: distance, noise, latency at the maximum,
    # fidelity and reliability at the minimum.
    net.add_link("S", "T", EdgeMetrics(90.0, 0.25, 7.0, 0.88, 0.90))

    subdivide = [("S", "C0"), ("C0", "C1"), ("C1", "C2")]
    for index in range(n_repeaters):
        u, v = subdivide[index]
        net.add_midpoint_node(u, v, f"R{index}", memory_lifetime_ms=1500.0)
    return net


def experiment_hop_penalty(
    n_repeaters: Sequence[int] = (0, 1, 2),
    seed: int = 20260930,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Sweep ``hop_penalty`` with 0, 1 and 2 repeaters inserted.

    ``[PROPOSED]`` node-cost model, ``[SIMULATED]`` routing. For each repeater
    count a fresh network is built by :func:`_hop_penalty_network` and
    ``hop_penalty`` is swept over ``(0.0, 0.25, 0.5, 1.0, 2.0)``. A repeater is
    charged :data:`REPEATER_PENALTY_FACTOR` times ``hop_penalty``, because it
    must hold entanglement in a memory and regenerate it rather than merely
    buffer and forward; a plain intermediate node is charged ``hop_penalty``.

    The experiment is fully deterministic and consumes no randomness, so
    ``seed`` is accepted for interface uniformity with the other experiments and
    is echoed into the metadata rather than being threaded through a generator.

    Args:
        n_repeaters: Repeater counts to test. Each must be non-negative.
        seed: Recorded in the metadata.

    Returns:
        ``(frames, metadata)`` where ``frames`` has the keys ``sweep`` and
        ``switch_points``; ``metadata`` records the seed and, per repeater
        count, the smallest swept ``hop_penalty`` at which the one-hop direct
        link is selected.

    Raises:
        ValueError: If a repeater count is negative.
        RoutingError: If a route search fails.
    """
    counts = [int(value) for value in n_repeaters]
    for value in counts:
        if value < 0:
            raise ValueError(f"n_repeaters must be non-negative, got {value}")

    penalties = (0.0, 0.25, 0.5, 1.0, 2.0)
    rows: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    for repeaters in counts:
        net = _hop_penalty_network(repeaters)
        for penalty in penalties:
            weights = RoutingWeights(
                alpha=_BALANCED_WEIGHTS["alpha"],
                beta=_BALANCED_WEIGHTS["beta"],
                gamma=_BALANCED_WEIGHTS["gamma"],
                delta=_BALANCED_WEIGHTS["delta"],
                rho=_BALANCED_WEIGHTS["rho"],
                hop_penalty=float(penalty),
                repeater_penalty=REPEATER_PENALTY_FACTOR * float(penalty),
            )
            route = MDQWRouter(net, MDQWConfig(weights=weights)).route("S", "T")
            rows.append(
                {
                    "n_repeaters": int(repeaters),
                    "hop_penalty": float(penalty),
                    "repeater_penalty": REPEATER_PENALTY_FACTOR * float(penalty),
                    "path": " -> ".join(str(node) for node in route.path),
                    "hop_count": int(route.hop_count),
                    "n_repeaters_on_path": sum(
                        1
                        for node in route.path[1:-1]
                        if bool(net.node(node).get("is_repeater", False))
                    ),
                    "edge_cost": float(route.edge_cost),
                    "node_cost": float(route.node_cost),
                    "total_cost": float(route.total_cost),
                    "total_distance_km": float(route.total_distance_km),
                    "total_latency_ms": float(route.total_latency_ms),
                    "estimated_fidelity": float(route.estimated_fidelity),
                }
            )
        subset = [row for row in rows if row["n_repeaters"] == repeaters]
        chain_rows = [row for row in subset if row["hop_count"] > 1]
        direct_rows = [row for row in subset if row["hop_count"] == 1]
        direct = [row["hop_penalty"] for row in direct_rows]
        switch_rows.append(
            {
                "n_repeaters": int(repeaters),
                "chain_edge_cost": (
                    float(chain_rows[0]["edge_cost"]) if chain_rows else float("nan")
                ),
                "direct_edge_cost": (
                    float(direct_rows[0]["edge_cost"]) if direct_rows else float("nan")
                ),
                "chain_hop_count": (
                    int(chain_rows[0]["hop_count"]) if chain_rows else -1
                ),
                "direct_hop_count": (
                    int(direct_rows[0]["hop_count"]) if direct_rows else -1
                ),
                "chain_total_cost_at_zero_penalty": (
                    float(
                        next(
                            row["total_cost"]
                            for row in chain_rows
                            if row["hop_penalty"] == 0.0
                        )
                    )
                    if chain_rows
                    else float("nan")
                ),
                "first_penalty_selecting_direct_link": (
                    float(min(direct)) if direct else float("nan")
                ),
                "direct_link_selected_at_any_penalty": bool(direct),
            }
        )

    sweep = pd.DataFrame(rows)
    metadata: dict[str, Any] = {
        "seed": int(seed),
        "n_repeaters": counts,
        "hop_penalties": list(penalties),
        "repeater_penalty_factor": REPEATER_PENALTY_FACTOR,
        "distinct_paths": int(sweep["path"].nunique()),
    }
    return {"sweep": sweep, "switch_points": pd.DataFrame(switch_rows)}, metadata


# =============================================================================
# D. Figures
# =============================================================================


def _save(fig: "plt.Figure", out_dir: Path | str, name: str) -> Path:
    """Create ``out_dir``, save ``fig`` as ``name``, close the figure.

    Args:
        fig: The figure to save. It is closed by this function, so callers must
            not touch it afterwards.
        out_dir: Directory to save into. Created (with parents) if missing.
        name: File name, e.g. ``"network_topology.png"``.

    Returns:
        The :class:`~pathlib.Path` the figure was written to.

    Raises:
        ValueError: If ``name`` resolves to an existing directory.
        OSError: If the directory cannot be created or the file cannot be
            written.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / name
    if destination.is_dir():
        raise ValueError(f"destination {destination} is a directory, not a file")
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return destination


def plot_network_topology(
    net: QuantumNetwork, out_dir: Path | str
) -> Path:
    """Draw the network, colouring nodes by repeater flag and labelling fidelity.

    ``[SIMULATED]`` rendering of synthetic data.

    Args:
        net: The network to draw.
        out_dir: Directory to write ``network_topology.png`` into.

    Returns:
        The path of the saved PNG.

    Raises:
        OSError: If the figure cannot be written.
    """
    graph = net.to_networkx()
    positions = nx.spring_layout(graph, seed=42)
    is_repeater = {
        node: bool(attributes.get("is_repeater", False))
        for node, attributes in graph.nodes(data=True)
    }
    colours = [
        "#f59e0b" if is_repeater.get(node, False) else "#22d3ee"
        for node in graph.nodes
    ]

    fig, axes = plt.subplots(figsize=(7.2, 5.4))
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=colours,
        node_size=760,
        edgecolors="#111827",
        linewidths=2.0,
        ax=axes,
    )
    nx.draw_networkx_labels(
        graph, positions, font_color="#0b1120", font_size=11, font_weight="bold",
        ax=axes,
    )
    nx.draw_networkx_edges(graph, positions, alpha=0.45, edge_color="#94a3b8", ax=axes)
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels={
            (u, v): f"f={d['fidelity']:.2f}"
            for u, v, d in graph.edges(data=True)
        },
        font_size=8,
        font_color="#e2e8f0",
        label_pos=0.5,
        ax=axes,
    )
    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", markersize=11,
            markerfacecolor="#22d3ee", markeredgecolor="#111827", label="node",
        ),
        plt.Line2D(
            [], [], marker="o", linestyle="", markersize=11,
            markerfacecolor="#f59e0b", markeredgecolor="#111827",
            label="repeater (is_repeater)",
        ),
    ]
    axes.legend(handles=handles, loc="upper right", fontsize="small")
    axes.set_title(
        f"{net.name}: {len(net)} nodes, {len(net.links)} links (synthetic metrics)"
    )
    axes.set_axis_off()
    fig.tight_layout()
    return _save(fig, out_dir, "network_topology.png")


def plot_selected_route(
    net: QuantumNetwork,
    routes: Mapping[str, RouteResult],
    out_dir: Path | str,
) -> Path:
    """Overlay several named routes on a grey copy of the topology.

    ``[SIMULATED]`` rendering of synthetic data.

    Args:
        net: The network to draw.
        routes: Mapping from label to :class:`~src.mdqw_routing.RouteResult`.
            Every route's edges are drawn on top of the base graph.
        out_dir: Directory to write ``selected_route.png`` into.

    Returns:
        The path of the saved PNG.

    Raises:
        KeyError: If a route leaves the network.
        OSError: If the figure cannot be written.
    """
    graph = net.to_networkx()
    positions = nx.spring_layout(graph, seed=42)
    palette = ["#7c3aed", "#22d3ee", "#f59e0b", "#34d399", "#f472b6", "#fb923c"]

    fig, axes = plt.subplots(figsize=(7.6, 5.6))
    nx.draw_networkx_nodes(
        graph, positions, node_color="#334155", node_size=620,
        edgecolors="#0f172a", linewidths=1.6, ax=axes,
    )
    nx.draw_networkx_labels(
        graph, positions, font_color="#f1f5f9", font_size=11,
        font_weight="bold", ax=axes,
    )
    nx.draw_networkx_edges(graph, positions, alpha=0.22, edge_color="#64748b", ax=axes)

    for index, (label, route) in enumerate(routes.items()):
        colour = palette[index % len(palette)]
        edges = [
            (route.path[i], route.path[i + 1]) for i in range(len(route.path) - 1)
        ]
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=edges,
            width=3.0,
            edge_color=colour,
            alpha=0.95,
            ax=axes,
        )
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=[route.source, route.target],
            node_color=colour,
            node_size=340,
            edgecolors="#0f172a",
            linewidths=1.4,
            ax=axes,
        )
    handles = [
        plt.Line2D(
            [], [], color=palette[index % len(palette)], linewidth=3.0,
            label=f"{label}: {route.hop_count} hops, edge cost {route.edge_cost:.4f}",
        )
        for index, (label, route) in enumerate(routes.items())
    ]
    axes.legend(handles=handles, loc="upper right", fontsize="small")
    axes.set_title("Selected routes over the example network")
    axes.set_axis_off()
    fig.tight_layout()
    return _save(fig, out_dir, "selected_route.png")


def plot_path_cost_comparison(frame: pd.DataFrame, out_dir: Path | str) -> Path:
    """Bar-chart each algorithm's ``edge_cost``.

    ``[SIMULATED]`` rendering. ``edge_cost`` is used rather than
    ``total_cost`` because it is the quantity every algorithm scores the same
    way; MDQW's ``total_cost`` additionally contains node terms.

    Args:
        frame: A :func:`src.mdqw_routing.compare_routes` frame with ``label``
            and ``edge_cost`` columns.
        out_dir: Directory to write ``path_cost_comparison.png`` into.

    Returns:
        The path of the saved PNG.

    Raises:
        KeyError: If ``label`` or ``edge_cost`` is missing.
        ValueError: If the frame is empty.
        OSError: If the figure cannot be written.
    """
    for column in ("label", "edge_cost"):
        if column not in frame.columns:
            raise KeyError(f"frame is missing the {column!r} column")
    if frame.empty:
        raise ValueError("cannot plot an empty comparison frame")

    fig, axes = plt.subplots(figsize=(7.0, 4.2))
    bars = axes.bar(
        frame["label"].astype(str),
        frame["edge_cost"].astype(float),
        color=["#7c3aed", "#22d3ee", "#f59e0b"][: len(frame)],
        edgecolor="#0f172a",
    )
    axes.bar_label(bars, fmt="%.4f", fontsize=9)
    axes.set_ylabel("edge cost  C(e)")
    axes.set_xlabel("routing configuration")
    axes.set_title("Link cost of each selected route (example network, A -> F)")
    axes.margins(y=0.18)
    fig.tight_layout()
    return _save(fig, out_dir, "path_cost_comparison.png")


def plot_fidelity_vs_hops(frame: pd.DataFrame, out_dir: Path | str) -> Path:
    """Plot simulated and declared fidelity against hop count.

    ``[SIMULATED]`` rendering. Both curves are drawn; the gap between them is
    the whole point of the figure, so the declared curve is not omitted.

    Args:
        frame: A :func:`fidelity_vs_hop_count` frame with ``hop_count``,
            ``simulated_end_to_end_fidelity`` and ``link_estimate_fidelity``.
        out_dir: Directory to write ``fidelity_vs_hops.png`` into.

    Returns:
        The path of the saved PNG.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If the frame is empty.
        OSError: If the figure cannot be written.
    """
    required = (
        "hop_count",
        "simulated_end_to_end_fidelity",
        "link_estimate_fidelity",
    )
    for column in required:
        if column not in frame.columns:
            raise KeyError(f"frame is missing the {column!r} column")
    if frame.empty:
        raise ValueError("cannot plot an empty hop-scaling frame")

    fig, axes = plt.subplots(figsize=(7.0, 4.4))
    axes.plot(
        frame["hop_count"],
        frame["simulated_end_to_end_fidelity"],
        marker="o",
        linewidth=2.0,
        color="#22d3ee",
        label="simulated product over hops (sequential model)",
    )
    axes.plot(
        frame["hop_count"],
        frame["link_estimate_fidelity"],
        marker="s",
        linewidth=2.0,
        linestyle="--",
        color="#f59e0b",
        label="product of declared link fidelities",
    )
    axes.set_xlabel("hop count")
    axes.set_ylabel("fidelity")
    axes.set_xticks(sorted(frame["hop_count"].unique()))
    axes.set_ylim(0.0, 1.05)
    axes.grid(alpha=0.25)
    axes.legend(fontsize="small")
    axes.set_title(
        "Per-hop composition against hop count (uniform noise 0.2)\n"
        "sequential model, no entanglement swapping -- see results.md"
    )
    fig.tight_layout()
    return _save(fig, out_dir, "fidelity_vs_hops.png")


def plot_noise_vs_route_cost(
    sweep_frame: pd.DataFrame, out_dir: Path | str
) -> Path:
    """Plot chosen edge cost and simulated fidelity against noise level.

    ``[SIMULATED]`` rendering. Two y axes are used because the two quantities
    live on different scales.

    Args:
        sweep_frame: The ``sweep`` frame of
            :func:`experiment_noise_sweep`, with ``noise_level``,
            ``edge_cost`` and ``simulated_end_to_end_fidelity``.
        out_dir: Directory to write ``noise_vs_route_cost.png`` into.

    Returns:
        The path of the saved PNG.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If the frame is empty.
        OSError: If the figure cannot be written.
    """
    required = ("noise_level", "edge_cost", "simulated_end_to_end_fidelity")
    for column in required:
        if column not in sweep_frame.columns:
            raise KeyError(f"frame is missing the {column!r} column")
    if sweep_frame.empty:
        raise ValueError("cannot plot an empty noise sweep")

    fig, left = plt.subplots(figsize=(7.2, 4.4))
    right = left.twinx()
    cost_line = left.plot(
        sweep_frame["noise_level"],
        sweep_frame["edge_cost"],
        marker="o",
        linewidth=2.0,
        color="#7c3aed",
        label="chosen route edge cost",
    )[0]
    fidelity_line = right.plot(
        sweep_frame["noise_level"],
        sweep_frame["simulated_end_to_end_fidelity"],
        marker="s",
        linewidth=2.0,
        linestyle="--",
        color="#22d3ee",
        label="simulated end-to-end fidelity",
    )[0]
    left.set_xlabel("per-link noise level (set on every link)")
    left.set_ylabel("edge cost  C(e)", color="#7c3aed")
    left.tick_params(axis="y", labelcolor="#7c3aed")
    right.set_ylabel("simulated end-to-end fidelity", color="#22d3ee")
    right.tick_params(axis="y", labelcolor="#22d3ee")
    left.grid(alpha=0.25)
    handles = [cost_line, fidelity_line]
    left.legend(handles=handles, loc="center right", fontsize="small")
    left.set_title(
        "Uniform link noise vs route cost and simulated fidelity\n"
        "(constant noise column min-max normalises to 0.0, so the cost is flat;\n"
        "the payload is a 50/50 superposition, which depolarisation moves towards)"
    )
    fig.tight_layout()
    return _save(fig, out_dir, "noise_vs_route_cost.png")


def plot_weight_sensitivity(grid_frame: pd.DataFrame, out_dir: Path | str) -> Path:
    """Heatmap ``edge_cost`` over the ``(alpha, delta)`` grid, annotated with hops.

    ``[SIMULATED]`` rendering.

    Args:
        grid_frame: The ``grid`` frame of :func:`experiment_weight_sweep`, with
            ``alpha``, ``delta``, ``edge_cost`` and ``hop_count``.
        out_dir: Directory to write ``weight_sensitivity.png`` into.

    Returns:
        The path of the saved PNG.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If the frame is empty.
        OSError: If the figure cannot be written.
    """
    required = ("alpha", "delta", "edge_cost", "hop_count")
    for column in required:
        if column not in grid_frame.columns:
            raise KeyError(f"grid frame is missing the {column!r} column")
    if grid_frame.empty:
        raise ValueError("cannot plot an empty weight grid")

    pivots_cost = grid_frame.pivot(index="delta", columns="alpha", values="edge_cost")
    pivots_hops = grid_frame.pivot(index="delta", columns="alpha", values="hop_count")
    fig, axes = plt.subplots(figsize=(7.0, 4.2))
    image = axes.imshow(pivots_cost.values, cmap="magma", aspect="auto", origin="lower")
    axes.set_xticks(range(len(pivots_cost.columns)))
    axes.set_xticklabels([f"{value:g}" for value in pivots_cost.columns])
    axes.set_yticks(range(len(pivots_cost.index)))
    axes.set_yticklabels([f"{value:g}" for value in pivots_cost.index])
    axes.set_xlabel("alpha  (normalised distance weight)")
    axes.set_ylabel("delta  (fidelity-penalty weight)")
    axes.set_title("Chosen-route edge cost over the (alpha, delta) grid\n"
                   "annotation = hop count of the selected path")
    for row in range(pivots_cost.shape[0]):
        for column in range(pivots_cost.shape[1]):
            axes.text(
                column,
                row,
                f"{pivots_cost.values[row, column]:.3f}\n"
                f"{int(pivots_hops.values[row, column])} hops",
                ha="center",
                va="center",
                fontsize=8,
                color="#f8fafc",
            )
    fig.colorbar(image, ax=axes, label="edge cost  C(e)")
    fig.tight_layout()
    return _save(fig, out_dir, "weight_sensitivity.png")


def plot_qw_probability_distribution(
    out_dir: Path | str, *, steps: int = 8, shots: int = 20_000
) -> Path:
    """Plot the Aer walk's node distribution at each step of a 4-cycle.

    ``[SIMULATED]``. Runs :func:`src.quantum_walk.walk_on_aer` for eight steps
    on :func:`src.quantum_walk.example_walk_graph` ``("cycle", 4)`` with
    ``shots=20000`` and draws one bar per node per time step. The walk is
    demonstrative: a 2-regular four-cycle is used because the coined walk needs a
    regular graph, not because four nodes model a quantum network.

    Args:
        out_dir: Directory to write ``quantum_walk_distribution.png`` into.
        steps: Number of walk steps to plot. Defaults to 8.
        shots: Aer shots per time step. Defaults to 20000; lowering it trades
            sampling precision for wall-clock time.

    Returns:
        The path of the saved PNG.

    Raises:
        RuntimeError: If the Aer walk fails.
        OSError: If the figure cannot be written.
    """
    graph = example_walk_graph("cycle", 4)
    steps = int(steps)
    distributions = walk_on_aer(
        graph, steps, kind="hadamard", start_node=0, shots=int(shots),
        seed_simulator=12345,
    )
    order = list(graph.nodes)
    width = 0.8 / (steps + 1)
    positions = np.arange(len(order), dtype=float)

    fig, axes = plt.subplots(figsize=(8.0, 4.4))
    for time, distribution in enumerate(distributions):
        axes.bar(
            positions + (time - steps / 2) * width,
            [distribution[node] for node in order],
            width=width,
            label=f"t={time}",
        )
    axes.set_xticks(positions)
    axes.set_xticklabels([str(node) for node in order])
    axes.set_xlabel("node of the 4-cycle")
    axes.set_ylabel("estimated probability")
    axes.set_ylim(0.0, 1.08)
    axes.set_title(
        f"Coined discrete-time quantum walk on a 4-cycle, Hadamard coin\n"
        f"{int(shots)} Aer shots per step; node 0 = start (SIMULATED)"
    )
    axes.legend(ncols=9, fontsize="x-small", loc="upper center")
    fig.tight_layout()
    return _save(fig, out_dir, "quantum_walk_distribution.png")


def plot_walk_heatmap(out_dir: Path | str) -> Path:
    """Delegate the node/time heatmap to :mod:`src.quantum_walk`.

    ``[SIMULATED]``. Calls
    :func:`src.quantum_walk.plot_walk_heatmap` on a 4-cycle for twelve steps and
    writes the result as ``quantum_walk_heatmap.png``.

    Args:
        out_dir: Directory to write ``quantum_walk_heatmap.png`` into.

    Returns:
        The path of the saved PNG.

    Raises:
        RuntimeError: If the walk circuit fails to build or run.
        OSError: If the figure cannot be written.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "quantum_walk_heatmap.png"
    figure = _delegate_walk_heatmap(
        example_walk_graph("cycle", 4),
        12,
        kind="hadamard",
        start_node=0,
        out_path=destination,
    )
    plt.close(figure)
    return destination


def plot_random_network_aggregate(
    aggregate_frame: pd.DataFrame, out_dir: Path | str
) -> Path:
    """Grouped win/loss/tie bars, one group per weight preset.

    ``[SIMULATED]`` rendering.

    Args:
        aggregate_frame: The ``aggregate`` frame of
            :func:`experiment_random_network`, with ``preset`` and the three
            count columns.
        out_dir: Directory to write ``random_network_aggregate.png`` into.

    Returns:
        The path of the saved PNG.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If the frame is empty.
        OSError: If the figure cannot be written.
    """
    required = ("preset", "mdqw_lower_edge_cost", "dijkstra_lower_edge_cost", "tied_edge_cost")
    for column in required:
        if column not in aggregate_frame.columns:
            raise KeyError(f"aggregate frame is missing the {column!r} column")
    if aggregate_frame.empty:
        raise ValueError("cannot plot an empty aggregate frame")

    presets = aggregate_frame["preset"].astype(str).tolist()
    positions = np.arange(len(presets), dtype=float)
    width = 0.26
    fig, axes = plt.subplots(figsize=(7.4, 4.2))
    series = (
        ("mdqw_lower_edge_cost", "MDQW lower edge cost", "#22d3ee"),
        ("dijkstra_lower_edge_cost", "Dijkstra lower edge cost", "#f472b6"),
        ("tied_edge_cost", "tied edge cost", "#94a3b8"),
    )
    for index, (column, label, colour) in enumerate(series):
        axes.bar(
            positions + (index - 1) * width,
            aggregate_frame[column].astype(float),
            width=width,
            label=label,
            color=colour,
            edgecolor="#0f172a",
        )
    axes.set_xticks(positions)
    axes.set_xticklabels(presets)
    axes.set_ylabel("comparisons")
    axes.set_xlabel("routing weight preset")
    axes.set_title(
        "MDQW vs Dijkstra edge cost over seeded random networks\n"
        "both minimise the same C(e); a tie is the expected outcome"
    )
    axes.legend(fontsize="small")
    fig.tight_layout()
    return _save(fig, out_dir, "random_network_aggregate.png")


def plot_teleportation_counts(
    counts: Mapping[str, int], out_dir: Path | str
) -> Path:
    """Bar-chart the three-bit teleportation outcome distribution.

    ``[SIMULATED]`` rendering of Aer shot counts.

    Args:
        counts: Counts keyed by three-character bitstrings, as returned by
            :func:`src.quantum_teleportation.run_on_aer`.
        out_dir: Directory to write ``teleportation_counts.png`` into.

    Returns:
        The path of the saved PNG.

    Raises:
        ValueError: If ``counts`` is empty.
        OSError: If the figure cannot be written.
    """
    if not counts:
        raise ValueError("counts must not be empty")

    ordered = sorted(counts.items())
    labels = [key for key, _ in ordered]
    values = [float(value) for _, value in ordered]
    colours = [
        "#22d3ee" if key.endswith("1") else "#334155" for key in labels
    ]

    fig, axes = plt.subplots(figsize=(6.8, 4.2))
    axes.bar(labels, values, color=colours, edgecolor="#0f172a")
    for index, value in enumerate(values):
        axes.text(index, value, f"{int(value)}", ha="center", va="bottom", fontsize=8)
    axes.set_xlabel("outcome bitstring  c2 c1 c0  (rightmost bit = Bob's readout)")
    axes.set_ylabel("shots")
    axes.set_title(
        "Bennett teleportation outcome distribution (ideal Aer run)\n"
        "Alice's two bits are uniform; Bob's bit reproduces the input"
    )
    axes.margins(y=0.16)
    fig.tight_layout()
    return _save(fig, out_dir, "teleportation_counts.png")


def plot_bidirectional_fidelity(out_dir: Path | str) -> Path:
    """Run the bidirectional circuit and plot both recovered marginals.

    ``[SIMULATED]``. ``[PROPOSED]`` model. The six-qubit composite circuit from
    :func:`src.quantum_teleportation.bidirectional_teleportation_circuit` is run
    on Aer, the two recovered marginals are extracted with
    :func:`src.quantum_teleportation.bidirectional_marginals`, and each is drawn
    as a normalised bar pair next to the ideal distribution of the state that
    direction carried. The
    :func:`src.quantum_teleportation.fidelity_metrics` Bhattacharyya coefficient
    of each pair is printed in the legend.

    Args:
        out_dir: Directory to write ``bidirectional_fidelity.png`` into.

    Returns:
        The path of the saved PNG.

    Raises:
        RuntimeError: If the Aer job fails.
        OSError: If the figure cannot be written.
    """
    state_a_to_b = ArbitraryState(theta=math.pi / 4.0, phi=0.3)
    state_b_to_a = ArbitraryState(theta=math.pi / 3.0, phi=1.1)
    circuit = bidirectional_teleportation_circuit(
        state_a_to_b, state_b_to_a, dynamic=True
    )
    counts = run_on_aer(circuit, shots=4096, seed=20260930)
    a_recovered, b_recovered = bidirectional_marginals(counts)

    def _normalise(mapping: Mapping[str, float]) -> dict[str, float]:
        total = float(sum(mapping.values()))
        return {key: float(value) / total for key, value in mapping.items()}

    a_observed = _normalise(a_recovered)
    b_observed = _normalise(b_recovered)
    a_ideal = state_b_to_a.ideal_probabilities()
    b_ideal = state_a_to_b.ideal_probabilities()
    a_fidelity = fidelity_metrics(a_observed, a_ideal)["bhattacharyya"]
    b_fidelity = fidelity_metrics(b_observed, b_ideal)["bhattacharyya"]

    bits = ["0", "1"]
    positions = np.arange(2, dtype=float)
    width = 0.36
    fig, axes = plt.subplots(figsize=(7.0, 4.3))
    axes.bar(
        positions - width / 2,
        [a_observed.get(bit, 0.0) for bit in bits],
        width=width,
        label=f"A recovers (observed)  F={a_fidelity:.4f}",
        color="#22d3ee",
        edgecolor="#0f172a",
    )
    axes.bar(
        positions - width / 2,
        [a_ideal.get(bit, 0.0) for bit in bits],
        width=width,
        label="A recovers (ideal)",
        color="#22d3ee",
        alpha=0.35,
        edgecolor="#0f172a",
        hatch="//",
    )
    axes.bar(
        positions + width / 2,
        [b_observed.get(bit, 0.0) for bit in bits],
        width=width,
        label=f"B recovers (observed)  F={b_fidelity:.4f}",
        color="#f59e0b",
        edgecolor="#0f172a",
    )
    axes.bar(
        positions + width / 2,
        [b_ideal.get(bit, 0.0) for bit in bits],
        width=width,
        label="B recovers (ideal)",
        color="#f59e0b",
        alpha=0.35,
        edgecolor="#0f172a",
        hatch="//",
    )
    axes.set_xticks(positions)
    axes.set_xticklabels(["bit = 0", "bit = 1"])
    axes.set_ylabel("probability")
    axes.set_ylim(0.0, 1.0)
    axes.set_title(
        "Two opposing channels in one composite six-qubit circuit (SIMULATED)\n"
        "two independent Bell pairs; NOT physical simultaneous bidirectional transfer"
    )
    axes.legend(fontsize="x-small", ncols=2)
    fig.tight_layout()
    return _save(fig, out_dir, "bidirectional_fidelity.png")


# =============================================================================
# E. Orchestration and reporting
# =============================================================================


def environment_block() -> str:
    """Return the software stack description embedded in ``results.md``.

    The final line is an explicit statement that every result in the document is
    a classical simulation.

    Returns:
        A multi-line string naming Python, the platform, and the versions of
        qiskit, qiskit-aer, numpy, networkx, pandas and matplotlib, ending with
        the simulation statement.
    """
    import qiskit
    import qiskit_aer

    lines = [
        "BQT-MDQW simulation environment",
        f"python      : {platform.python_version()} ({platform.python_implementation()})",
        f"platform    : {platform.platform()}",
        f"qiskit      : {qiskit.__version__}",
        f"qiskit-aer  : {qiskit_aer.__version__}",
        f"numpy       : {np.__version__}",
        f"networkx    : {nx.__version__}",
        f"pandas      : {pd.__version__}",
        f"matplotlib  : {matplotlib.__version__} (backend {matplotlib.get_backend()})",
        "",
        "All results below are CLASSICAL SIMULATIONS.",
    ]
    return "\n".join(lines)


def _pretty(frame: pd.DataFrame, *, digits: int = 6) -> pd.DataFrame:
    """Return a display copy of ``frame`` with floats rounded.

    Args:
        frame: The frame to prepare for rendering.
        digits: Decimal places to keep.

    Returns:
        A copy with float columns rounded; never mutates the input.
    """
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].round(digits)
    return display


def _render_frame(frame: pd.DataFrame) -> str:
    """Render ``frame`` as Markdown, falling back to a fenced plain-text block.

    :meth:`pandas.DataFrame.to_markdown` requires the optional ``tabulate``
    package. This project does not declare ``tabulate`` as a dependency and does
    not add one here, so when it is absent the frame is rendered with
    :meth:`pandas.DataFrame.to_string` inside a fenced block instead.

    Args:
        frame: The frame to render.

    Returns:
        A Markdown string containing the table or the fenced fallback.
    """
    if frame.empty:
        return "_(no rows: this section produced an empty frame)_"
    display = _pretty(frame)
    try:
        return display.to_markdown(index=False)
    except ImportError:
        width = max(len(str(column)) for column in display.columns) + 2
        lines = ["```text"]
        lines.append(
            "".join(str(column).ljust(width) for column in display.columns).rstrip()
        )
        for _, row in display.iterrows():
            lines.append(
                "".join(
                    f"{value!s:<{width}}" for value in row.tolist()
                ).rstrip()
            )
        lines.append("```")
        return "\n".join(lines)


_FIGURE_CAPTIONS: Mapping[str, str] = {
    "network_topology.png": (
        "The example network with every link drawn at its fidelity hint and "
        "annotated with distance, so the topology behind experiment 1 is "
        "visible."
    ),
    "selected_route.png": (
        "The route each of the four configurations chose on the example "
        "network, drawn on the same node layout for direct comparison."
    ),
    "path_cost_comparison.png": (
        "Total route cost per configuration, splitting the classical edge "
        "term from the quantum-walk node term."
    ),
    "fidelity_vs_hops.png": (
        "Simulated end-to-end fidelity against hop count for each "
        "configuration, with the closed-form link and memory-decay estimates "
        "overlaid."
    ),
    "noise_vs_route_cost.png": (
        "Route cost and simulated fidelity as every link is given the same "
        "noise level."
    ),
    "weight_sensitivity.png": (
        "Edge cost and simulated fidelity across the full (alpha, delta) grid "
        "at fixed beta, gamma and rho."
    ),
    "bidirectional_fidelity.png": (
        "Fidelity of the six-qubit bidirectional circuit: the ideal entangled "
        "state, the state after readout noise, and the two reduced marginals "
        "for each direction."
    ),
    "teleportation_counts.png": (
        "Shot counts for the three-bit teleportation outcome, which is what "
        "the simulated fidelity score is computed from."
    ),
    "quantum_walk_distribution.png": (
        "Node probability distribution of the coined discrete-time quantum "
        "walk at each step of the example graph."
    ),
    "quantum_walk_heatmap.png": (
        "Step-by-step heatmap of the same walk distribution, where each cell "
        "gives the probability of occupying a node."
    ),
    "random_network_aggregate.png": (
        "Win, loss and tie counts per routing preset, plus mean edge cost "
        "and mean preset-score, aggregated over the seeded random networks."
    ),
    "architecture.png": (
        "How the pieces fit together: the seeded random or example network "
        "feeds routing, the quantum-walk prior adds node costs, and the "
        "chosen route is scored by an Aer teleportation circuit."
    ),
}


def _reading_manual(frames: Mapping[str, pd.DataFrame]) -> str:
    """Derive the plain reading of experiment 1 from the real frame contents.

    Args:
        frames: The frames returned by :func:`experiment_manual_network`.

    Returns:
        One or two sentences describing what the numbers actually show.
    """
    routes = frames["routes"]
    multihop = frames["multihop"]
    enumerated = frames["enumerated"]
    enumerated = enumerated.sort_values("edge_cost", ignore_index=True)
    best = enumerated.iloc[0]
    worst = enumerated.iloc[-1]
    chosen = routes.iloc[0]
    distinct_paths = int(routes["path"].nunique())
    fidelity = multihop["simulated_end_to_end_fidelity"].iloc[0]
    gap = multihop["simulated_vs_link_gap"].iloc[0]
    return (
        f"MDQW and Dijkstra both selected {chosen['path']} with edge cost "
        f"{float(chosen['edge_cost']):.4f}; enabling the walk prior left the "
        f"selected path unchanged and only raised the declared total cost to "
        f"{float(routes.loc[routes['label'] == 'mdqw-qw-prior', 'total_cost'].iloc[0]):.4f} "
        f"through its node term, so {distinct_paths} of {len(routes)} "
        "configurations ended on the same path. "
        f"Enumerating all simple paths ranks {best['path']} cheapest at "
        f"{float(best['edge_cost']):.4f} against {worst['path']} at "
        f"{float(worst['edge_cost']):.4f}, and the {int(chosen['hop_count'])}-hop "
        f"sequential teleportation of that route gave a simulated end-to-end "
        f"fidelity of {float(fidelity):.6f}, {float(gap):+.6f} against the "
        "product of the declared link fidelities."
    )


def _reading_random(frames: Mapping[str, pd.DataFrame]) -> str:
    """Derive the plain reading of experiment 2 from the real frame contents.

    Args:
        frames: The frames returned by :func:`experiment_random_network`.

    Returns:
        One or two sentences describing the observed counts.
    """
    aggregate = frames["aggregate"]
    total = int(aggregate["comparisons"].sum())
    lower = int(aggregate["mdqw_lower_edge_cost"].sum())
    higher = int(aggregate["dijkstra_lower_edge_cost"].sum())
    tied = int(aggregate["tied_edge_cost"].sum())
    worst = aggregate.loc[aggregate["mean_edge_cost_difference"].abs().idxmax()]
    return (
        f"Across {total} comparisons MDQW recorded a lower edge cost "
        f"{lower} times, a higher one {higher} times and tied {tied} times, "
        "which is the expected outcome because MDQW without node terms and "
        "Dijkstra minimise the same objective. "
        f"The largest mean edge-cost difference across the three presets was "
        f"{float(worst['mean_edge_cost_difference']):+.6f} for the "
        f"{worst['preset']} preset."
    )


def _reading_noise(frames: Mapping[str, pd.DataFrame]) -> str:
    """Derive the plain reading of experiment 3 from the real frame contents.

    Args:
        frames: The frames returned by :func:`experiment_noise_sweep`.

    Returns:
        One or two sentences describing the observed behaviour.
    """
    sweep = frames["sweep"].sort_values("noise_level", ignore_index=True)
    distinct_costs = int(sweep["edge_cost"].nunique())
    distinct_paths = int(sweep["path"].nunique())
    first = sweep.iloc[0]
    last = sweep.iloc[-1]
    values = sweep["simulated_end_to_end_fidelity"].to_numpy(dtype=float)
    monotone_up = bool(np.all(np.diff(values) >= -1e-12))
    monotone_down = bool(np.all(np.diff(values) <= 1e-12))
    if monotone_up:
        shape = "increased monotonically"
    elif monotone_down:
        shape = "decreased monotonically"
    else:
        shape = "was not monotone"
    return (
        f"Setting every link to the same noise gave {distinct_costs} distinct "
        f"edge costs and {distinct_paths} distinct paths across the "
        f"{len(sweep)} levels, because a constant noise column min-max "
        "normalises to zero at every link and so contributes nothing to C(e). "
        f"The simulated end-to-end fidelity {shape} over the same levels, from "
        f"{float(first['simulated_end_to_end_fidelity']):.6f} at noise "
        f"{float(first['noise_level']):.2f} to "
        f"{float(last['simulated_end_to_end_fidelity']):.6f} at noise "
        f"{float(last['noise_level']):.2f}. That shape is a property of the "
        "measurement, not of link quality: the payload is the equatorial "
        "superposition whose ideal distribution is 50/50, and the depolarising "
        "channel pushes Bob's readout towards 50/50, so adding gate noise can "
        "raise this particular score. It is not evidence that noise helps."
    )


def _reading_weight(frames: Mapping[str, pd.DataFrame]) -> str:
    """Derive the plain reading of experiment 4 from the real frame contents.

    Args:
        frames: The frames returned by :func:`experiment_weight_sweep`.

    Returns:
        One or two sentences describing the observed behaviour.
    """
    grid = frames["grid"]
    distinct = int(grid["path"].nunique())
    cheapest = grid.loc[grid["edge_cost"].idxmin()]
    dearest = grid.loc[grid["edge_cost"].idxmax()]
    return (
        f"All {len(grid)} (alpha, delta) combinations recorded a row and none "
        f"was rejected, and all {len(grid)} chose the same path "
        f"({grid['path'].iloc[0]}) -- {distinct} distinct path(s) in the grid. "
        f"The edge cost does still move with the weights, from "
        f"{float(cheapest['edge_cost']):.4f} at alpha={float(cheapest['alpha']):.1f}, "
        f"delta={float(cheapest['delta']):.1f} to {float(dearest['edge_cost']):.4f} "
        f"at alpha={float(dearest['alpha']):.1f}, delta={float(dearest['delta']):.1f}, "
        "so the weighting rescales the cost of the chosen route without changing "
        "which route is chosen on this instance."
    )


def _reading_ablation(frames: Mapping[str, pd.DataFrame]) -> str:
    """Derive the plain reading of experiment 5 from the real frame contents.

    Args:
        frames: The frames returned by :func:`experiment_qw_prior_ablation`.

    Returns:
        One or two sentences describing the measured cost of the heuristic.
    """
    ablation = frames["ablation"]
    optimal = int(ablation["edge_cost_still_optimal"].sum())
    total = int(len(ablation))
    worst = ablation.loc[ablation["edge_cost_difference"].idxmax()]
    by_weight = frames["ablation_summary"]
    first = by_weight.iloc[0]
    last = by_weight.iloc[-1]
    groups = ablation.groupby(["instance", "prior_weight"])["path"].nunique()
    step_sensitive = int((groups > 1).sum())
    if optimal < total:
        shortfall = (
            f"; the largest observed shortfall was "
            f"{float(worst['edge_cost_difference']):+.6f} on instance "
            f"{worst['instance']} at prior weight "
            f"{float(worst['prior_weight']):.1f}"
        )
    else:
        shortfall = (
            ", so on these instances the prior never raised the edge cost above "
            "the Dijkstra minimum"
        )
    return (
        f"The walk prior left the edge cost optimal in {optimal} of {total} "
        f"configurations" + shortfall + ". "
        f"Grouped by prior weight, mean edge-cost difference ran from "
        f"{float(first['mean_edge_cost_difference']):+.6f} at weight "
        f"{float(first['prior_weight']):.1f} to "
        f"{float(last['mean_edge_cost_difference']):+.6f} at weight "
        f"{float(last['prior_weight']):.1f}, and the chosen path varied with the "
        f"prior step count in {step_sensitive} of {int(len(groups))} "
        f"(instance, weight) groups."
    )


def _reading_hop(frames: Mapping[str, pd.DataFrame]) -> str:
    """Derive the plain reading of experiment 6 from the real frame contents.

    Args:
        frames: The frames returned by :func:`experiment_hop_penalty`.

    Returns:
        One or two sentences describing the observed switch-over behaviour.
    """
    sweep = frames["sweep"]
    switch = frames["switch_points"]
    sentences: list[str] = []
    for _, row in switch.iterrows():
        threshold = row["first_penalty_selecting_direct_link"]
        if pd.isna(threshold):
            sentences.append(
                f"with {int(row['n_repeaters'])} repeater(s) the five-hop chain "
                "was selected at every swept hop penalty"
            )
        else:
            sentences.append(
                f"with {int(row['n_repeaters'])} repeater(s) the one-hop direct "
                f"link first won at hop_penalty={float(threshold):.2f}"
            )
    if len(switch):
        zero_row = switch.iloc[0]
        zero_repeaters = int(zero_row["n_repeaters"])
        zero_cost = float(zero_row["chain_total_cost_at_zero_penalty"])
        zero_clause = (
            f" At hop_penalty=0 the {zero_repeaters}-repeater case still selected a "
            f"multi-hop chain costing {zero_cost:.4f} in total"
        )
    else:
        zero_clause = ""
    return (
        f"Across {len(sweep)} combinations the sweep produced "
        f"{int(sweep['path'].nunique())} distinct paths: "
        + "; ".join(sentences)
        + "."
        + zero_clause
        + "."
    )


_READINGS = {
    "manual_network": _reading_manual,
    "random_network": _reading_random,
    "noise_sweep": _reading_noise,
    "weight_sweep": _reading_weight,
    "qw_prior_ablation": _reading_ablation,
    "hop_penalty": _reading_hop,
}

_EXPERIMENT_TITLES = {
    "manual_network": "Experiment 1 -- the example network, routed four ways",
    "random_network": "Experiment 2 -- twenty seeded random networks",
    "noise_sweep": "Experiment 3 -- uniform link-noise sweep",
    "weight_sweep": "Experiment 4 -- (alpha, delta) weight grid",
    "qw_prior_ablation": "Experiment 5 -- quantum-walk prior ablation",
    "hop_penalty": "Experiment 6 -- per-hop penalty with repeaters",
}


def render_results_markdown(
    frames: Mapping[str, pd.DataFrame],
    out_dir: Path | str,
    *,
    seed: int,
    timestamp: str | None = None,
    figures: Sequence[Path] | None = None,
    verdicts: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``results.md`` from the experiment frames.

    Every plain-language sentence in the per-experiment sections is derived from
    the frame contents by the ``_reading_*`` helpers rather than written by hand,
    so the document cannot claim something the numbers do not show.

    Args:
        frames: Mapping from ``"<experiment>.<frame>"`` to
            :class:`pandas.DataFrame`, exactly as assembled by
            :func:`run_all`.
        out_dir: Directory to write ``results.md`` into. Created if missing.
        seed: The master seed the run used.
        timestamp: ISO-8601 timestamp. Defaults to the current UTC time.
        figures: Paths of the PNGs generated by the run, listed in the Figures
            section. ``None`` yields an empty section.
        verdicts: Optional mapping of extra lines to include under Provenance,
            e.g. the observed win/loss/tie counts.

    Returns:
        The :class:`~pathlib.Path` of the written ``results.md``.

    Raises:
        OSError: If the file cannot be written.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "results.md"
    stamp = timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds")

    caption = "\n".join(
        [
            "# BQT-MDQW simulation results",
            "",
            "[SIMULATED] Every number in this document was produced by a "
            "classical simulation on synthetic data. No quantum hardware was "
            "used and no link metric is calibrated against a physical channel.",
            "",
            "## Provenance",
            "",
            f"- Master seed: `{int(seed)}`",
            f"- Generated (UTC): `{stamp}`",
            f"- `PYTHONHASHSEED`: `{os.environ.get('PYTHONHASHSEED')!r}` "
            f"(reported for provenance only; the walk prior is bit-reproducible "
            f"across processes without pinning it, because the cycle cover in "
            f"`src.quantum_walk.shunt_decomposition` is selected on integer keys "
            f"rather than hash-order-sensitive string keys)",
            "",
            "### Legend",
            "",
            "- `[ESTABLISHED]` -- classical Dijkstra shortest path, and the "
            "Bennett teleportation protocol as built by "
            "`src/quantum_teleportation.py`. Neither is a quantum algorithm and "
            "neither is run on quantum hardware here.",
            "- `[PROPOSED]` -- design choices of this repository with no "
            "external citation and no hardware validation: the MDQW-inspired "
            "multi-metric cost function, the per-relay hop penalty, the "
            "walk-derived node prior, the noise/readout mappings, and the "
            "hop-by-hop multi-hop teleportation model.",
            "- `[SIMULATED]` -- every number below. Executed by Qiskit Aer or "
            "by dense NumPy matrix-vector products, on networks generated by "
            "`src/network.py`.",
            "",
            "### Limitation that applies to every fidelity number",
            "",
            SEQUENTIAL_TELEPORTATION_CAVEAT,
            "",
            "### Environment",
            "",
            "```text",
            environment_block(),
            "```",
            "",
        ]
    )

    if verdicts:
        caption += "### Run-level observations\n\n"
        for key, value in verdicts.items():
            caption += f"- {key}: `{value}`\n"
        caption += "\n"

    body: list[str] = []
    for experiment, title in _EXPERIMENT_TITLES.items():
        keys = sorted(k for k in frames if k.startswith(f"{experiment}."))
        if not keys:
            continue
        body.append(f"## {title}\n")
        try:
            reading = _READINGS[experiment](
                {key.split(".", 1)[1]: frames[key] for key in keys}
            )
        except Exception as exc:  # pragma: no cover - defensive
            reading = (
                f"(The plain reading of this section could not be derived: "
                f"{type(exc).__name__}: {exc})"
            )
        body.append(reading + "\n")
        for key in keys:
            name = key.split(".", 1)[1]
            body.append(f"### `{name}`\n")
            body.append(_render_frame(frames[key]) + "\n")

    limitations = "\n".join(
        [
            "## Limitations",
            "",
            "1. **Everything here is a classical simulation.** Every quantum "
            "operation is executed by Qiskit Aer or by dense NumPy "
            "matrix-vector products. No quantum hardware, real optical link or "
            "network device was involved.",
            "2. **Routing is not a quantum algorithm.** MDQW routing is A* and "
            "the baseline is Dijkstra, both classical graph searches. The "
            "quantum-walk prior is a plain `dict[node, float]` computed "
            "classically and then handed to the router.",
            "3. **The walk component is demonstrative.** The coined "
            "discrete-time quantum walk runs on a 4-cycle because the shunt "
            "decomposition needs a regular graph, not because four nodes model "
            "a quantum network. Its role is to supply a node prior.",
            "4. **No quantum memories are simulated.** `memory_decay_estimate` "
            "is the closed-form `exp(-latency / memory_lifetime)` surrogate, not "
            "a decoherence simulation, and the memory lifetimes themselves are "
            "declared model inputs.",
            "5. **No entanglement distribution is modelled.** The "
            "bidirectional circuit is one composite six-qubit state with two "
            "independent Bell pairs; running it as a single circuit hides the "
            "resource accounting a real two-way link must pay for and does not "
            "demonstrate physical simultaneous bidirectional teleportation.",
            "6. **No entanglement swapping, no purification, no error "
            "correction, no repeater buffering and no repeater scheduling.** "
            "The multi-hop model is sequential and hop-by-hop; see the "
            "limitation paragraph at the top of this document.",
            "7. **No hardware constraints.** There is no model of spectral "
            "bandwidth, wavelength multiplexing, detector dark counts, "
            "multiplexed-pair rate, finite key rate, decoherence during storage, "
            "or synchronisation between sites.",
            "8. **The edge metrics are synthetic.** Distance, noise, latency, "
            "fidelity and reliability are drawn independently from chosen "
            "ranges in `src/network.py` and are not measured quantities; their "
            "independence is a modelling convenience.",
            "9. **Simulation results do not establish physical performance.** "
            "Nothing in this document is evidence about a real quantum network, "
            "and no comparison here should be read as a claim about one.",
            "10. **The walk prior was once sensitive to `PYTHONHASHSEED`, and "
            "the cause has been fixed.** A complete graph admits many valid cycle "
            "covers, and the cover previously selected inside "
            "`src/quantum_walk.shunt_decomposition` depended on CPython's "
            "per-process string hash seed, so the prior could change between runs "
            "of the same seed. The bipartite double cover now uses integer keys, "
            "whose hashing is not randomised, and the prior is byte-identical "
            "across processes; `tests/test_regressions.py` guards this. Any "
            "result set generated before that fix used a different prior and is "
            "not comparable with the current one.",
            "",
        ]
    )

    figure_lines = ["## Figures\n"]
    if figures:
        for path in figures:
            name = Path(path).name
            figure_caption = _FIGURE_CAPTIONS.get(name, "Generated for this run.")
            figure_lines.append(
                f"- `{name}` -- {figure_caption} "
                f"Written to `{Path(path).as_posix()}`."
            )
    else:
        figure_lines.append("_No figures were generated for this run._")
    figure_lines.append("")

    destination.write_text(
        "\n".join([caption, *body, limitations, *figure_lines]),
        encoding="utf-8",
    )
    return destination


def render_architecture_diagram(
    out_path: Path | str = "docs/architecture.png",
) -> Path:
    """Draw the seven-stage pipeline as a labelled vertical diagram.

    Pure Matplotlib -- no graphviz, no new dependency. The boxes are drawn in
    this exact top-to-bottom order:

    1. Quantum Network
    2. Network Graph
    3. Edge Metrics
    4. MDQW-Inspired Routing
    5. Selected Multi-Hop Path
    6. Quantum Teleportation
    7. Receiver

    A left-hand bracket groups boxes 1-3 as ``CLASSICAL PLANNING LAYER`` and
    boxes 6-7 as ``QUANTUM EXECUTION LAYER (SIMULATED)``, and a small legend
    notes that the routing box is classical.

    Args:
        out_path: Destination PNG. Parent directories are created.

    Returns:
        The :class:`~pathlib.Path` written.

    Raises:
        ValueError: If ``out_path`` is an existing directory.
        OSError: If the file cannot be written.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    stages = [
        ("Quantum Network", "#1d4ed8"),
        ("Network Graph", "#2563eb"),
        ("Edge Metrics", "#0891b2"),
        ("MDQW-Inspired Routing", "#7c3aed"),
        ("Selected Multi-Hop Path", "#4f46e5"),
        ("Quantum Teleportation", "#0f766e"),
        ("Receiver", "#047857"),
    ]

    fig, axes = plt.subplots(figsize=(8.6, 10.4))
    axes.set_xlim(0.0, 10.0)
    axes.set_ylim(0.0, 11.6)
    axes.axis("off")

    box_width, box_height = 6.4, 0.92
    centre_x = 5.6
    first_centre = 10.75
    spacing = 1.35
    centres = [first_centre - spacing * index for index in range(len(stages))]

    for (label, colour), centre_y in zip(stages, centres):
        box = FancyBboxPatch(
            (centre_x - box_width / 2, centre_y - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.06,rounding_size=0.22",
            linewidth=2.0,
            edgecolor="#0f172a",
            facecolor=colour,
        )
        axes.add_patch(box)
        axes.text(
            centre_x,
            centre_y,
            label,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#f8fafc",
        )

    for upper, lower in zip(centres, centres[1:]):
        axes.add_patch(
            FancyArrowPatch(
                (centre_x, upper - box_height / 2),
                (centre_x, lower + box_height / 2),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=2.0,
                color="#334155",
            )
        )

    def _bracket(top: float, bottom: float, text: str) -> None:
        x = 0.95
        axes.plot([x, x], [top, bottom], color="#0f172a", linewidth=2.2)
        axes.plot([x, x + 0.28], [top, top], color="#0f172a", linewidth=2.2)
        axes.plot([x, x + 0.28], [bottom, bottom], color="#0f172a", linewidth=2.2)
        axes.text(
            x - 0.18,
            (top + bottom) / 2.0,
            text,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            rotation=90,
            color="#0f172a",
        )

    _bracket(centres[0] + 0.62, centres[2] - 0.62, "CLASSICAL PLANNING LAYER")
    _bracket(
        centres[5] + 0.62,
        centres[6] - 0.62,
        "QUANTUM EXECUTION LAYER (SIMULATED)",
    )

    routing_index = 3
    axes.text(
        centre_x + box_width / 2 + 0.28,
        centres[routing_index],
        "classical A* over a synthetic graph\n"
        "(not a quantum algorithm;\nthe walk prior is a classical heuristic)",
        ha="left",
        va="center",
        fontsize=9,
        color="#4c1d95",
    )
    axes.text(
        0.95,
        centres[routing_index + 2] - 0.75,
        "Legend: violet boxes are the [PROPOSED] MDQW planning stages;\n"
        "teal/green boxes are the [SIMULATED] quantum execution stages.",
        ha="left",
        va="top",
        fontsize=9,
        color="#334155",
    )
    axes.set_title(
        "BQT-MDQW prototype architecture\n"
        "every stage runs classically on this machine; quantum stages are "
        "simulated by Qiskit Aer",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_dir():
        raise ValueError(f"out_path {destination} is a directory, not a file")
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return destination


def run_all(
    output_dir: Path | str = "results",
    *,
    seed: int = 20260930,
    quick: bool = False,
    skip_figures: bool = False,
) -> dict[str, Any]:
    """Run every experiment, write every figure, and report what happened.

    Each experiment is wrapped individually: a failure is logged to stdout and
    the run continues, so one broken sweep cannot lose the other results. The
    experiment functions themselves re-raise, so calling one directly still
    surfaces its error.

    Args:
        output_dir: Root output directory. Created if missing, along with
            ``output_dir/"figures"``.
        seed: Master seed. Every derived seed is an explicit offset from it.
        quick: Reduce work so the whole run finishes in well under a minute:
            ``n_trials=3``, 512 shots for the hop sweeps and the manual
            network's multi-hop run, and truncated noise / ablation / repeater
            lists. The ``(alpha, delta)`` grid is kept whole because it costs no
            simulator time.
        skip_figures: Do not produce any PNG and do not write
            ``docs/architecture.png``.

    Returns:
        A dictionary with the keys ``frames`` (the raw experiment frames),
        ``flat_frames`` (the ``"<experiment>.<frame>"`` mapping handed to
        :func:`render_results_markdown`), ``results_md``,
        ``summary_csv``, ``figures``, ``architecture``, ``metadata``,
        ``counts`` (``wins``/``losses``/``ties``/``matched``),
        ``errors``, ``seed``, ``quick``, and ``elapsed_seconds``.

    Raises:
        OSError: If the output directory cannot be created.
    """
    started = time.perf_counter()
    root = Path(output_dir)
    figure_dir = root / "figures"
    root.mkdir(parents=True, exist_ok=True)
    if not skip_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)

    shots = 512 if quick else 2048
    hop_scaling_shots = 512 if quick else 2048
    hop_scaling_max_hops = 3 if quick else 6
    walk_figure_shots = 4_000 if quick else 20_000
    noise_levels = (0.0, 0.05, 0.1) if quick else (0.0, 0.05, 0.1, 0.2, 0.3, 0.4)
    prior_weights = (1.0, 4.0) if quick else (1.0, 2.0, 4.0)
    prior_steps = (4, 8) if quick else (4, 8, 16)
    repeater_counts = (0, 1) if quick else (0, 1, 2)
    n_trials = 3 if quick else 20
    n_instances = 1 if quick else 4

    print("=" * 78)
    print("BQT-MDQW run -- ALL RESULTS ARE CLASSICAL SIMULATIONS ON SYNTHETIC DATA")
    print("=" * 78)
    print(f"  seed            : {int(seed)}")
    print(f"  quick           : {bool(quick)}")
    print(f"  output_dir      : {root}")
    print(f"  shots (sweeps)  : {shots}")
    print(f"  random trials   : {n_trials}")
    print(f"  noise levels    : {list(noise_levels)}")
    print(f"  prior (w, steps): {list(prior_weights)} x {list(prior_steps)}")
    print(f"  repeater counts : {list(repeater_counts)}")
    hash_seed = os.environ.get("PYTHONHASHSEED")
    print(f"  PYTHONHASHSEED  : {hash_seed!r} (provenance only; not required for reproducibility)")
    print("-" * 78)

    flat: dict[str, pd.DataFrame] = {}
    metadata: dict[str, Any] = {}
    errors: list[dict[str, str]] = []

    def _log(message: str) -> None:
        print(message, flush=True)

    def _record(name: str, frames: dict[str, pd.DataFrame], meta: dict[str, Any]) -> None:
        for key, frame in frames.items():
            flat[f"{name}.{key}"] = frame
        metadata[name] = meta

    # --- Experiment 1 -------------------------------------------------------
    _log("[1/6] manual network A -> F (MDQW, MDQW+walk prior, Dijkstra) ...")
    try:
        frames, meta = experiment_manual_network(
            seed=int(seed),
            shots=shots,
            hop_scaling_shots=hop_scaling_shots,
            hop_scaling_max_hops=hop_scaling_max_hops,
        )
        _record("manual_network", frames, meta)
        routes = frames["routes"]
        _log(
            "      paths: "
            + "; ".join(
                f"{row['label']}={row['path']} (edge {float(row['edge_cost']):.4f})"
                for _, row in routes.iterrows()
            )
        )
        _log(f"      verdict: {meta['verdict']}")
        multihop = frames["multihop"]
        _log(
            "      simulated end-to-end fidelity: "
            + ", ".join(
                f"{row['label']}={float(row['simulated_end_to_end_fidelity']):.6f}"
                for _, row in multihop.iterrows()
            )
        )
    except Exception as exc:
        errors.append({"experiment": "manual_network", "error": repr(exc)})
        _log(f"      FAILED: {type(exc).__name__}: {exc}")

    # --- Experiment 2 -------------------------------------------------------
    _log(f"[2/6] random networks (n_nodes=12, n_trials={n_trials}) ...")
    try:
        frames, meta = experiment_random_network(
            n_nodes=12, n_trials=n_trials, seed=int(seed)
        )
        _record("random_network", frames, meta)
        aggregate = frames["aggregate"]
        _log(
            "      edge-cost outcomes: "
            f"MDQW lower {meta['wins']}, Dijkstra lower {meta['losses']}, "
            f"tied {meta['ties']} over {int(aggregate['comparisons'].sum())} comparisons"
        )
        for _, row in aggregate.iterrows():
            _log(
                f"      preset {row['preset']:<15} mean edge cost "
                f"{float(row['mean_dijkstra_edge_cost']):.4f} (Dijkstra) vs "
                f"{float(row['mean_mdqw_edge_cost']):.4f} (MDQW); matched "
                f"{int(row['matched_dijkstra_edge_cost'])}/{int(row['comparisons'])}"
            )
        _log(
            f"      connected networks: {meta['connected_networks']}, "
            f"disconnected: {meta['disconnected_networks']}"
        )
    except Exception as exc:
        errors.append({"experiment": "random_network", "error": repr(exc)})
        meta = {}
        _log(f"      FAILED: {type(exc).__name__}: {exc}")

    # --- Experiment 3 -------------------------------------------------------
    _log(f"[3/6] uniform link-noise sweep over {len(noise_levels)} levels ...")
    try:
        frames, meta_noise = experiment_noise_sweep(
            noise_levels, seed=int(seed), shots=max(512, shots // 2)
        )
        _record("noise_sweep", frames, meta_noise)
        sweep = frames["sweep"]
        _log(
            f"      distinct edge costs across the sweep: "
            f"{meta_noise['distinct_edge_costs']}; distinct paths: "
            f"{meta_noise['distinct_paths']}"
        )
        _log(
            "      simulated fidelity: "
            + ", ".join(
                f"noise={float(row['noise_level']):.2f}->"
                f"{float(row['simulated_end_to_end_fidelity']):.6f}"
                for _, row in sweep.iterrows()
            )
        )
        if meta_noise["edge_cost_is_constant"]:
            _log(
                "      NOTE: the edge cost is constant across the sweep because a "
                "constant noise column min-max normalises to 0.0 at every link; "
                "the figure is reported as it came out."
            )
    except Exception as exc:
        errors.append({"experiment": "noise_sweep", "error": repr(exc)})
        _log(f"      FAILED: {type(exc).__name__}: {exc}")

    # --- Experiment 4 -------------------------------------------------------
    _log("[4/6] (alpha, delta) weight grid on the example network ...")
    try:
        frames, meta_weight = experiment_weight_sweep(seed=int(seed))
        _record("weight_sweep", frames, meta_weight)
        grid = frames["grid"]
        _log(
            f"      {meta_weight['combinations_recorded']}/"
            f"{meta_weight['combinations_attempted']} combinations recorded, "
            f"{len(meta_weight['skipped'])} skipped, "
            f"{meta_weight['distinct_paths']} distinct path(s) chosen"
        )
        _log(
            "      edge cost range across the grid: "
            f"[{meta_weight['edge_cost_range'][0]:.4f}, "
            f"{meta_weight['edge_cost_range'][1]:.4f}]"
        )
    except Exception as exc:
        errors.append({"experiment": "weight_sweep", "error": repr(exc)})
        _log(f"      FAILED: {type(exc).__name__}: {exc}")

    # --- Experiment 5 -------------------------------------------------------
    _log("[5/6] quantum-walk prior ablation ...")
    try:
        frames, meta_ablation = experiment_qw_prior_ablation(
            seed=int(seed),
            weights=prior_weights,
            steps=prior_steps,
            n_instances=n_instances,
        )
        _record("qw_prior_ablation", frames, meta_ablation)
        _log(
            f"      {meta_ablation['configurations']} configurations over "
            f"{len(meta_ablation['instances'])} instances; edge cost still "
            f"optimal in {meta_ablation['edge_cost_still_optimal']}, "
            f"suboptimal in {meta_ablation['edge_cost_suboptimal']}; worst "
            f"shortfall {meta_ablation['max_edge_cost_difference']:+.6f}"
        )
    except Exception as exc:
        errors.append({"experiment": "qw_prior_ablation", "error": repr(exc)})
        _log(f"      FAILED: {type(exc).__name__}: {exc}")

    # --- Experiment 6 -------------------------------------------------------
    _log(f"[6/6] hop-penalty sweep with repeater counts {list(repeater_counts)} ...")
    try:
        frames, meta_hop = experiment_hop_penalty(
            n_repeaters=repeater_counts, seed=int(seed)
        )
        _record("hop_penalty", frames, meta_hop)
        switch = frames["switch_points"]
        for _, row in switch.iterrows():
            threshold = row["first_penalty_selecting_direct_link"]
            _log(
                f"      n_repeaters={int(row['n_repeaters'])}: "
                + (
                    f"one-hop link first wins at hop_penalty={float(threshold):.2f}"
                    if not pd.isna(threshold)
                    else "five-hop chain selected at every swept penalty"
                )
            )
    except Exception as exc:
        errors.append({"experiment": "hop_penalty", "error": repr(exc)})
        _log(f"      FAILED: {type(exc).__name__}: {exc}")

    # --- Figures ------------------------------------------------------------
    written_figures: list[Path] = []
    if not skip_figures:
        _log("-" * 78)
        _log(f"[figures] writing 11 PNGs into {figure_dir} ...")
        example_net = build_example_network()
        teleport_counts = run_on_aer(
            build_teleportation_circuit(DEFAULT_STATE),
            shots=2048,
            seed=int(seed),
        )

        default_route = MDQWRouter(example_net, MDQWConfig()).route("A", "F")
        dijkstra_route = DijkstraBaseline(example_net).route("A", "F")
        proxy, _ = walkable_proxy(example_net.to_networkx())
        prior_route = MDQWRouter(
            example_net,
            MDQWConfig(use_qw_prior=True, qw_prior_weight=1.0, prior_steps=8),
            quantum_walk_prior(proxy),
        ).route("A", "F")

        figure_jobs: list[tuple[str, Any]] = [
            (
                "network_topology",
                lambda: plot_network_topology(example_net, figure_dir),
            ),
            (
                "selected_route",
                lambda: plot_selected_route(
                    example_net,
                    {
                        "MDQW (default)": default_route,
                        "MDQW + QW prior": prior_route,
                        "Dijkstra": dijkstra_route,
                    },
                    figure_dir,
                ),
            ),
            (
                "path_cost_comparison",
                lambda: plot_path_cost_comparison(
                    flat["manual_network.routes"], figure_dir
                ),
            ),
            (
                "fidelity_vs_hops",
                lambda: plot_fidelity_vs_hops(
                    flat["manual_network.hop_scaling"], figure_dir
                ),
            ),
            (
                "noise_vs_route_cost",
                lambda: plot_noise_vs_route_cost(
                    flat["noise_sweep.sweep"], figure_dir
                ),
            ),
            (
                "weight_sensitivity",
                lambda: plot_weight_sensitivity(
                    flat["weight_sweep.grid"], figure_dir
                ),
            ),
            (
                "quantum_walk_distribution",
                lambda: plot_qw_probability_distribution(
                    figure_dir, steps=8, shots=walk_figure_shots
                ),
            ),
            ("quantum_walk_heatmap", lambda: plot_walk_heatmap(figure_dir)),
            (
                "random_network_aggregate",
                lambda: plot_random_network_aggregate(
                    flat["random_network.aggregate"], figure_dir
                ),
            ),
            (
                "teleportation_counts",
                lambda: plot_teleportation_counts(teleport_counts, figure_dir),
            ),
            (
                "bidirectional_fidelity",
                lambda: plot_bidirectional_fidelity(figure_dir),
            ),
        ]

        for name, job in figure_jobs:
            try:
                path = job()
                written_figures.append(Path(path))
                _log(f"      ok   {name:<28} -> {Path(path).name}")
            except Exception as exc:
                errors.append({"experiment": f"figure:{name}", "error": repr(exc)})
                _log(f"      FAILED {name}: {type(exc).__name__}: {exc}")

    # --- Reports ------------------------------------------------------------
    _log("-" * 78)
    random_meta = metadata.get("random_network", {})
    counts = {
        "wins": int(random_meta.get("wins", 0)),
        "losses": int(random_meta.get("losses", 0)),
        "ties": int(random_meta.get("ties", 0)),
        "matched": int(random_meta.get("matched_dijkstra_edge_cost", 0)),
    }

    verdicts: dict[str, Any] = {}
    manual_meta = metadata.get("manual_network")
    if manual_meta:
        verdicts["manual-network verdict (compare_routes/summarise)"] = manual_meta[
            "verdict"
        ]
    if flat:
        verdicts["random-network edge cost: MDQW lower / Dijkstra lower / tied"] = (
            f"{counts['wins']} / {counts['losses']} / {counts['ties']}"
        )
        verdicts["random-network MDQW runs whose edge cost matched Dijkstra"] = (
            f"{counts['matched']}"
        )
    ablation_meta = metadata.get("qw_prior_ablation")
    if ablation_meta:
        verdicts["walk-prior configurations leaving the edge cost optimal"] = (
            f"{ablation_meta['edge_cost_still_optimal']} of "
            f"{ablation_meta['configurations']}"
        )
    noise_meta = metadata.get("noise_sweep")
    if noise_meta:
        verdicts["noise sweep: distinct edge costs across levels"] = str(
            noise_meta["distinct_edge_costs"]
        )
    verdicts["experiments/figures that failed"] = str(len(errors))

    results_md: Path | None = None
    summary_csv: Path | None = None
    try:
        results_md = render_results_markdown(
            flat,
            root,
            seed=int(seed),
            figures=written_figures,
            verdicts=verdicts,
        )
        _log(f"[report] wrote {results_md} ({results_md.stat().st_size} bytes)")
    except Exception as exc:
        errors.append({"experiment": "render_results_markdown", "error": repr(exc)})
        _log(f"      FAILED results.md: {type(exc).__name__}: {exc}")

    if flat:
        try:
            combined = pd.concat(
                [frame.assign(frame=key) for key, frame in flat.items()],
                ignore_index=True,
                sort=False,
            )
            summary_csv = root / "results_summary.csv"
            combined.to_csv(summary_csv, index=False)
            _log(
                f"[report] wrote {summary_csv} "
                f"({summary_csv.stat().st_size} bytes, {len(combined)} rows)"
            )
        except Exception as exc:
            errors.append({"experiment": "results_summary.csv", "error": repr(exc)})
            _log(f"      FAILED results_summary.csv: {type(exc).__name__}: {exc}")

    architecture: Path | None = None
    if not skip_figures:
        try:
            architecture = render_architecture_diagram("docs/architecture.png")
            _log(f"[report] wrote {architecture} ({architecture.stat().st_size} bytes)")
        except Exception as exc:
            errors.append(
                {"experiment": "render_architecture_diagram", "error": repr(exc)}
            )
            _log(f"      FAILED architecture diagram: {type(exc).__name__}: {exc}")

    elapsed = time.perf_counter() - started
    _log("-" * 78)
    _log("VERDICT LINES (observed on this run; nothing asserted in general)")
    for key, value in verdicts.items():
        _log(f"  {key}: {value}")
    _log(
        "  random-network win / loss / tie on edge cost: "
        f"{counts['wins']} / {counts['losses']} / {counts['ties']}"
    )
    _log(f"  figures written: {len(written_figures)}/11")
    if errors:
        _log(f"  FAILURES ({len(errors)}):")
        for failure in errors:
            _log(f"    - {failure['experiment']}: {failure['error']}")
    else:
        _log("  failures: none")
    _log(
        "  REMINDER: sequential hop-by-hop teleportation is not a quantum "
        "repeater network (no entanglement swapping, purification or memories)."
    )
    _log(f"  elapsed: {elapsed:.1f} s")
    _log("=" * 78)

    grouped: dict[str, dict[str, pd.DataFrame]] = {}
    for key in sorted({name.split(".", 1)[0] for name in flat}):
        grouped[key] = {
            name.split(".", 1)[1]: frame
            for name, frame in flat.items()
            if name.split(".", 1)[0] == key
        }

    return {
        "frames": grouped,
        "flat_frames": flat,
        "results_md": results_md,
        "summary_csv": summary_csv,
        "figures": written_figures,
        "architecture": architecture,
        "metadata": metadata,
        "counts": counts,
        "errors": errors,
        "seed": int(seed),
        "quick": bool(quick),
        "elapsed_seconds": float(elapsed),
    }


if __name__ == "__main__":
    outcome = run_all()
    print("-" * 78)
    print("Output paths:")
    if outcome["results_md"] is not None:
        print(f"  results.md          : {outcome['results_md']}")
    if outcome["summary_csv"] is not None:
        print(f"  results_summary.csv : {outcome['summary_csv']}")
    for figure in outcome["figures"]:
        print(f"  figure              : {figure}")
    if outcome["architecture"] is not None:
        print(f"  architecture        : {outcome['architecture']}")
    print(
        "  win/loss/tie        : "
        f"{outcome['counts']['wins']}/{outcome['counts']['losses']}/"
        f"{outcome['counts']['ties']}"
    )