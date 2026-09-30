"""Discrete-time coined quantum walk over an ``m``-regular network graph.

Two concerns, deliberately kept apart
-------------------------------------
This module carries **two** different objects, and the separation is deliberate:

1. ``[ESTABLISHED]`` A discrete-time coined quantum walk (DTQW) on a regular
   graph: shunt decomposition of the adjacency matrix, coin operator, shift
   operator, and the resulting one-step unitary. This is textbook quantum
   walk theory (the coined-walk formalism of Kendon, and the shunt
   decomposition of Wong). The operators built here are genuine unitary
   matrices and :func:`walk_reference_numpy` is a faithful state-vector
   evolution of them.
2. ``[PROPOSED]`` A *node-salience prior* derived from that walk and handed to
   the router as a plain ``dict[node, float]``
   (:func:`quantum_walk_prior`). The weighting formula is a choice made in
   this prototype, not a result from the quantum-walk literature.

Provenance legend used in every docstring
-----------------------------------------
``[ESTABLISHED]``
    Standard quantum-walk formalism: shunt decomposition, coin and shift
    operators, the ``H_coin (x) H_position`` walk unitary.
``[PROPOSED]``
    The walk-derived node prior and its weighting formula.
``[SIMULATED]``
    Anything produced numerically -- by :mod:`numpy` or by Qiskit Aer's
    ``AerSimulator``. Nothing in this module executes on quantum hardware.

What this module does **not** claim
-----------------------------------
* No quantum advantage, no speed-up, no scaling property is claimed or implied
  anywhere. The walk is propagated by dense matrix-vector products and by a
  shot-based classical simulator.
* :func:`quantum_walk_prior` is a **heuristic**. It is not a routing
  guarantee, it is not a quantum algorithm, and it is *computed classically*:
  the router consumes a dictionary of floats produced by
  :func:`walk_reference_numpy`. No quantum computation is involved in
  routing.
* :func:`walk_on_aer` is a cross-validation of the NumPy reference against a
  real Qiskit circuit. The unitary it applies is a dense ``(m*n) x (m*n)``
  matrix, so :func:`build_walk_circuit` is a *demonstration* of the walk, not a
  scalable decomposition into elementary gates.

Measurement caveat (why ``walk_on_aer`` uses one circuit per time step)
-----------------------------------------------------------------------
Reading the distribution at every intermediate step would require a
mid-circuit measurement, and a measurement in the computational basis
*collapses* the walker: continuing from the collapsed state yields a mixture
over collapse outcomes, not the coherent evolution. To reproduce the reference
walk faithfully, :func:`walk_on_aer` therefore builds one circuit per time
step -- ``initialize(psi_0)`` followed by exactly ``t`` copies of the walk
unitary and a single final measurement -- and runs the whole batch on
``AerSimulator`` in one call.

Key-module references
---------------------
``src.network``
    ``[PROPOSED]`` network model. Its node identifiers (for example ``"N0"``)
    are the keys of every distribution returned here.
``src.mdqw_routing``
    ``[PROPOSED]`` consumer of :func:`quantum_walk_prior`.
"""

from __future__ import annotations

import math
import pathlib
import platform
import sys
from typing import TYPE_CHECKING, Any, Hashable, Iterable, Literal, Mapping, Sequence

import matplotlib
import networkx as nx
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator

if TYPE_CHECKING:  # pragma: no cover - import only needed for type checkers
    from matplotlib.figure import Figure

__all__ = [
    "CoinKind",
    "MAX_CIRCUIT_DIMENSION",
    "edge_index_map",
    "initial_state",
    "coin_operator",
    "shunt_decomposition",
    "shift_operator",
    "walk_reference_numpy",
    "walk_on_aer",
    "build_walk_circuit",
    "node_probability_distribution",
    "max_distribution_error",
    "quantum_walk_prior",
    "stationary_node_distribution",
    "plot_walk_evolution",
    "plot_walk_heatmap",
    "example_walk_graph",
    "environment_report",
]

#: Coin operators accepted by :func:`coin_operator`.
CoinKind = Literal["hadamard", "y", "grover"]

#: Largest walk dimension :func:`walk_on_aer` will attempt to put on a circuit.
#: The limit is a sanity guard, not a capability claim: the walk unitary is a
#: dense matrix, so the dense representation is what bounds this in practice.
MAX_CIRCUIT_DIMENSION: int = 1 << 26

#: Tolerance used for the unitarity assertions of the operators.
_UNITARITY_TOLERANCE: float = 1e-12

#: Backend names that need replacing by the non-interactive ``Agg`` backend.
_GUI_BACKENDS: frozenset[str] = frozenset(
    {"tkagg", "qtagg", "qtagg_32", "gtkagg", "gtk3agg", "gtk4agg", "wxagg", "macosx"}
)

_COIN_KINDS: tuple[str, ...] = ("hadamard", "y", "grover")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _node_order(graph: nx.Graph) -> list[Hashable]:
    """Return the graph's node identifiers in :meth:`networkx.Graph.nodes` order."""
    return list(graph.nodes)


def _require_regular(graph: nx.Graph) -> tuple[int, int]:
    """Validate that ``graph`` is a simple, non-empty, ``m``-regular graph.

    Args:
        graph: Candidate graph.

    Returns:
        ``(degree, node_count)`` where ``degree`` is the common degree ``m``.

    Raises:
        ValueError: If ``graph`` is not a :class:`networkx.Graph`, is a
            multigraph or directed graph, is empty, carries a self-loop, has no
            edges, or is not regular.
    """
    if not isinstance(graph, nx.Graph) or graph.is_multigraph() or graph.is_directed():
        raise ValueError(
            "shunt_decomposition requires a simple undirected networkx.Graph, got "
            f"{type(graph).__name__}"
        )
    node_count = graph.number_of_nodes()
    if node_count == 0:
        raise ValueError("graph is empty; a walk needs at least one node")
    if graph.number_of_edges() == 0:
        raise ValueError("graph has no edges; a walk needs at least one link")
    if nx.number_of_selfloops(graph):
        raise ValueError("graph contains self-loops, which the shift operator cannot represent")
    degrees = {int(d) for _, d in graph.degree()}
    if len(degrees) != 1:
        observed = sorted(degrees)
        raise ValueError(
            "shunt_decomposition requires a regular graph; observed degrees "
            f"{observed}. Build one with example_walk_graph() or a regular "
            "NetworkX constructor."
        )
    degree = degrees.pop()
    if node_count * degree % 2:
        raise ValueError(
            f"graph has {node_count} nodes of degree {degree}, so the number of "
            "directed arcs is odd; the adjacency cannot be split into shunt permutations"
        )
    return degree, node_count


def _edge_index(graph: nx.Graph) -> list[list[int]]:
    """Return the coin-ordered neighbour list of every node.

    Node positions follow :meth:`networkx.Graph.nodes` order. The neighbour of
    a node is placed at coin index ``j`` according to the order in which its
    edge appears in :meth:`networkx.Graph.edges`, which makes the coin labelling
    deterministic and stable for a given edge-insertion order.

    Args:
        graph: A graph whose nodes all have the same degree.

    Returns:
        A list of length ``n``, where entry ``i`` holds the neighbour *node
        indices* of node index ``i``, ordered by coin index.
    """
    order = _node_order(graph)
    position = {node: index for index, node in enumerate(order)}
    incident: list[list[int]] = [[] for _ in order]
    for u, v in graph.edges():
        incident[position[u]].append(position[v])
        incident[position[v]].append(position[u])
    return incident


def _walk_dimension(graph: nx.Graph) -> int:
    """Return the walk dimension ``m * n`` of a regular ``graph``."""
    degree, node_count = _require_regular(graph)
    return degree * node_count


def _resolve_start_node(graph: nx.Graph, start_node: Hashable | None) -> Hashable:
    """Return ``start_node``, defaulting to the first node of ``graph``.

    Args:
        graph: The graph being walked.
        start_node: Requested start node, or ``None`` for the first node.

    Returns:
        A node identifier that is present in ``graph``.

    Raises:
        ValueError: If ``start_node`` is not a node of ``graph``.
    """
    order = _node_order(graph)
    if start_node is None:
        return order[0]
    if start_node not in graph:
        raise ValueError(f"unknown start node {start_node!r}; known nodes: {order}")
    return start_node


def _walk_unitary(graph: nx.Graph, kind: CoinKind) -> np.ndarray:
    """Build the one-step DTQW unitary ``shift @ kron(coin, I_n)``.

    Args:
        graph: A regular graph.
        kind: Coin family, see :func:`coin_operator`.

    Returns:
        A complex ``(m*n, m*n)`` unitary matrix.
    """
    _, node_count = _require_regular(graph)
    coin = coin_operator(_walk_dimension(graph) // node_count, kind)
    shunts = shunt_decomposition(graph)
    shift = shift_operator(shunts)
    return shift @ np.kron(coin, np.eye(node_count, dtype=complex))


def _initial_vector(
    graph: nx.Graph,
    start_node: Hashable | None,
    initial_state_vector: np.ndarray | None,
) -> np.ndarray:
    """Return the caller's ``initial_state_vector`` or the default start state."""
    if initial_state_vector is None:
        return initial_state(graph, _resolve_start_node(graph, start_node))
    vector = np.asarray(initial_state_vector, dtype=complex).reshape(-1)
    expected = _walk_dimension(graph)
    if vector.size != expected:
        raise ValueError(f"initial_state_vector has length {vector.size}, expected {expected}")
    return vector


def _qubits_for(dimension: int) -> int:
    """Return the number of qubits needed to hold ``dimension`` amplitudes."""
    if dimension < 2:
        raise ValueError(f"a quantum circuit needs at least two basis states, got {dimension}")
    return max(1, (dimension - 1).bit_length())


def _counts_to_index_counts(counts: Mapping[Any, int]) -> dict[int, int]:
    """Collapse Qiskit count keys into a ``{basis index: occurrences}`` mapping.

    Qiskit renders count keys as bitstrings, optionally space-separated when a
    circuit declares several classical registers, and occasionally in a
    hexadecimal form. Both spellings are accepted.

    Args:
        counts: The ``result.get_counts()`` mapping of a single circuit.

    Returns:
        Mapping from the integer read of each key to its shot count.
    """
    collapsed: dict[int, int] = {}
    for key, occurrences in counts.items():
        text = str(key).replace(" ", "")
        value = int(text, 16) if text.lower().startswith("0x") else int(text, 2)
        collapsed[value] = collapsed.get(value, 0) + int(occurrences)
    return collapsed


def _ensure_headless_backend() -> str:
    """Switch Matplotlib to ``Agg`` when an interactive GUI backend is active.

    Returns:
        The backend in use afterwards.
    """
    current = matplotlib.get_backend()
    if current.lower() in _GUI_BACKENDS:
        matplotlib.use("Agg")
        current = matplotlib.get_backend()
    return current


def _finalise_figure(figure: "Figure", out_path: str | pathlib.Path | None) -> "Figure":
    """Save ``figure`` to ``out_path`` (creating parent directories) if given.

    Args:
        figure: The figure to finish.
        out_path: Destination file, or ``None`` to skip saving.

    Returns:
        The same figure, so callers can ``return _finalise_figure(fig, path)``.

    Raises:
        ValueError: If ``out_path`` is given and cannot be used as a file path.
    """
    if out_path is None:
        return figure
    destination = pathlib.Path(out_path)
    if destination.exists() and destination.is_dir():
        raise ValueError(f"out_path {destination} is a directory, not a file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    return figure


# ---------------------------------------------------------------------------
# Graph operators -- [ESTABLISHED]
# ---------------------------------------------------------------------------


def shunt_decomposition(graph: nx.Graph) -> list[np.ndarray]:
    """``[ESTABLISHED]`` Split the adjacency matrix into ``m`` permutation matrices.

    For an ``m``-regular graph with adjacency matrix ``A`` the shunt
    decomposition (Wong) writes

    .. code-block:: text

        A = P_0 + P_1 + ... + P_{m-1}

    with every ``P_i`` a permutation matrix. Each ``P_i`` is read as a disjoint
    union of directed cycles that follow graph edges: a 2-cycle ``u -> v -> u``
    is the single edge ``{u, v}``, while a longer cycle
    ``u_0 -> u_1 -> ... -> u_k -> u_0`` is a cycle of ``G`` traversed in one
    direction.

    Implementation
    --------------
    Every round extracts one **cycle cover** of the residual directed graph: a
    perfect matching between the *out*-copies and the *in*-copies of the nodes
    (the bipartite double cover), found with a deterministic Hopcroft-Karp pass
    over a double cover built in node-index order. Because every node has the
    same residual out- and in-degree, Hall's condition holds and such a perfect
    matching always exists; because a matching consumes exactly one arc per node,
    every round consumes exactly ``n`` of the ``m*n`` directed arcs, so the loop
    runs exactly ``m`` times.

    Why the double cover and not a matching of ``G`` itself
    -------------------------------------------------------
    A matching of ``G`` leaves some nodes unmatched, and an unmatched node has to
    become a *fixed point* of that shunt, i.e. a ``1`` on the diagonal. Summing
    over all rounds those fixed points add ``1`` to every diagonal entry, so the
    sum can no longer equal ``A`` (whose diagonal is zero for a simple graph).
    Requiring every shunt to be a *derangement* -- no fixed points -- is what
    makes ``sum(P_i) == A`` hold exactly. On graphs where every round admits a
    perfect matching of ``G`` (cycles, complete graphs of even order, ...) the
    two constructions coincide; on graphs without one -- for example
    ``nx.complete_graph(3)``, where the two shunts are the two orientations of
    the triangle -- the double-cover construction is the one that stays exact.
    It is likewise deterministic: the same graph always yields the same shunts.

    Args:
        graph: A simple undirected ``m``-regular :class:`networkx.Graph`.

    Returns:
        A list of ``m`` integer ``(n, n)`` permutation matrices whose integer sum
        is the adjacency matrix of ``graph`` in :meth:`networkx.Graph.nodes`
        order. The reconstruction is verified inside this function.

    Raises:
        ValueError: If ``graph`` is not a simple, non-empty, self-loop-free,
            edge-bearing regular graph (see :func:`_require_regular`).
        RuntimeError: If the internal consistency check fails, which would mean
            the decomposition is broken rather than the input being invalid.
    """
    degree, node_count = _require_regular(graph)
    order = _node_order(graph)
    position = {node: index for index, node in enumerate(order)}

    arcs: set[tuple[int, int]] = set()
    for u, v in graph.edges():
        a, b = position[u], position[v]
        arcs.add((a, b))
        arcs.add((b, a))

    shunts: list[np.ndarray] = []
    # Node keys for the bipartite double cover are INTEGERS, never tuples
    # containing strings. CPython randomises str.__hash__ per process
    # (PYTHONHASHSEED), so a string-keyed graph would make the matching -- and
    # therefore the shunts, the walk unitary and the routing prior -- depend on
    # an environment variable rather than on the fixed seed we advertise.
    # Integer hashing is not randomised, so the decomposition below is
    # byte-identical across processes.
    while arcs:
        double_cover = nx.Graph()
        double_cover.add_nodes_from(range(node_count), bipartite=0)
        double_cover.add_nodes_from(range(node_count, 2 * node_count), bipartite=1)
        double_cover.add_edges_from(
            (a, node_count + b) for a, b in sorted(arcs)
        )
        matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(
            double_cover, top_nodes=range(node_count)
        )

        shunt = np.zeros((node_count, node_count), dtype=int)
        round_arcs: set[tuple[int, int]] = set()
        for source in range(node_count):
            partner = matching.get(source)
            if partner is None:
                continue
            a, b = source, partner - node_count
            shunt[a, b] = 1
            round_arcs.add((a, b))
        if len(round_arcs) != node_count:
            raise RuntimeError(
                "shunt extraction did not yield a cycle cover "
                f"({len(round_arcs)} arcs for {node_count} nodes)"
            )
        arcs -= round_arcs
        shunts.append(shunt)

    if len(shunts) != degree:
        raise RuntimeError(
            f"expected {degree} shunt permutations, extracted {len(shunts)}"
        )

    adjacency = nx.to_numpy_array(graph, nodelist=order, dtype=int)
    if not np.array_equal(sum(shunts), adjacency):
        raise RuntimeError("shunt decomposition failed to reconstruct the adjacency matrix")

    return shunts


def coin_operator(m: int, kind: CoinKind = "hadamard") -> np.ndarray:
    """``[ESTABLISHED]`` Build the ``m x m`` unitary coin of a DTQW.

    Three coin families are provided, all exactly unitary to within
    ``1e-12``:

    ``"hadamard"``
        ``H_2 (x) I_{m/2}`` for even ``m``; for odd ``m`` the orthonormal
        discrete cosine transform (DCT-II) matrix, which is the real
        half-shifted cosine form of a normalised discrete Fourier transform.
        Taking the plain real part of the ``m``-point DFT is *not* orthogonal in
        general (``m = 3`` collapses to two identical rows), so the cosine form
        is used to keep the coin exactly real orthogonal.
    ``"y"``
        The same coin in the ``Y`` phase ladder, ``C = D C_H D`` with
        ``D = diag(1, i, -1, -i, ...)``. At ``m = 2`` this is the standard ``Y``
        coin ``(1/sqrt(2)) [[1, i], [i, 1]]``; for larger ``m`` it is the direct
        analogue.
    ``"grover"``
        The Grover diffusion ``2 |s><s| - I`` with ``|s> = (1/sqrt(m)) 1``
        uniform, i.e. the Householder reflection about the uniform state.

    Args:
        m: Coin dimension, i.e. the common degree of the graph. Must be ``>= 2``.
        kind: One of ``"hadamard"``, ``"y"``, ``"grover"``.

    Returns:
        A complex ``(m, m)`` unitary matrix.

    Raises:
        ValueError: If ``m < 2`` or ``kind`` is not a known coin family.
        RuntimeError: If the constructed matrix is not unitary, which would be a
            bug in this function rather than bad input.
    """
    if not isinstance(m, (int, np.integer)) or isinstance(m, bool):
        raise ValueError(f"coin dimension m must be an integer, got {type(m).__name__}")
    m = int(m)
    if m < 2:
        raise ValueError(f"coin dimension m must be at least 2, got {m}")
    if kind not in _COIN_KINDS:
        raise ValueError(f"unknown coin kind {kind!r}; expected one of {_COIN_KINDS}")

    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0)
    if m % 2 == 0:
        base = np.kron(hadamard, np.eye(m // 2))
    else:
        # Orthonormal DCT-II: the real, half-shifted cosine transform derived
        # from the normalised DFT. Built directly because the naive real part of
        # the DFT is only orthogonal for some m.
        rows = np.arange(m)[:, None]
        columns = np.arange(m)[None, :]
        scale = np.full(m, math.sqrt(2.0 / m))
        scale[0] = math.sqrt(1.0 / m)
        base = scale[:, None] * np.cos(math.pi * (2 * columns + 1) * rows / (2 * m))
    base = base.astype(complex)

    if kind == "hadamard":
        coin = base
    elif kind == "y":
        ladder = np.exp(1j * math.pi * np.arange(m) / 2.0).astype(complex)
        coin = ladder[:, None] * base * ladder[None, :]
    else:
        uniform = np.ones(m, dtype=complex) / math.sqrt(m)
        coin = 2.0 * np.outer(uniform, uniform.conj()) - np.eye(m, dtype=complex)

    defect = float(np.max(np.abs(coin.conj().T @ coin - np.eye(m))))
    if defect > _UNITARITY_TOLERANCE:
        raise RuntimeError(f"constructed {kind} coin for m={m} is not unitary (defect {defect:g})")
    return coin


def shift_operator(shunts: list[np.ndarray]) -> np.ndarray:
    """``[ESTABLISHED]`` Assemble the shunt shift operator ``S = sum_i |i><i| (x) P_i``.

    The walk space is ``H_coin (x) H_position`` with ``m`` coin states and ``n``
    positions, so the flat basis index of a state is ``coin_index * n +
    node_index`` and ``S`` is ``(m*n) x (m*n)``.

    Args:
        shunts: The ``m`` permutation matrices from :func:`shunt_decomposition`.

    Returns:
        A complex ``(m*n, m*n)`` unitary matrix.

    Raises:
        ValueError: If ``shunts`` is empty, a shunt is not square, the shunts do
            not all share one size, or a shunt is not a permutation matrix.
        RuntimeError: If the assembled operator is not unitary.
    """
    if not isinstance(shunts, Sequence) or len(shunts) == 0:
        raise ValueError("shift_operator needs a non-empty sequence of shunt matrices")
    size = int(np.asarray(shunts[0]).shape[0]) if np.ndim(shunts[0]) == 2 else None
    for position, shunt in enumerate(shunts):
        matrix = np.asarray(shunt)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"shunt {position} is not a square matrix: shape {matrix.shape}")
        if size is None:
            size = int(matrix.shape[0])
        elif int(matrix.shape[0]) != size:
            raise ValueError(
                f"shunt {position} has size {matrix.shape[0]}, expected {size}; all shunts "
                "must act on the same node space"
            )
        ones = np.ones(int(matrix.shape[0]))
        if not (
            np.allclose(matrix.sum(axis=0), ones)
            and np.allclose(matrix.sum(axis=1), ones)
            and np.isin(matrix, (0, 1)).all()
        ):
            raise ValueError(f"shunt {position} is not a 0/1 permutation matrix")

    node_space = int(size) if size is not None else 0
    count = len(shunts)
    shift = np.zeros((count * node_space, count * node_space), dtype=complex)
    for index, shunt in enumerate(shunts):
        block = slice(index * node_space, (index + 1) * node_space)
        shift[block, block] = np.asarray(shunt, dtype=complex)

    defect = float(np.max(np.abs(shift.conj().T @ shift - np.eye(count * node_space))))
    if defect > _UNITARITY_TOLERANCE:
        raise RuntimeError(f"assembled shift operator is not unitary (defect {defect:g})")
    return shift


def edge_index_map(graph: nx.Graph) -> dict[tuple[int, int], int]:
    """Map every ``(node_index, coin_index)`` pair to its flat state index.

    The walk state lives in ``H_coin (x) H_position``, so the flat index is
    ``coin_index * n + node_index``. Node indices follow
    :meth:`networkx.Graph.nodes` order and coin indices follow the stable
    edge order documented in :func:`_edge_index`.

    Args:
        graph: A regular graph, so that every node has all ``m`` coin states.

    Returns:
        A dictionary with all ``n * m`` ``(node_index, coin_index)`` keys mapped
        to a distinct flat index in ``[0, m*n)``.

    Raises:
        ValueError: If ``graph`` is not a valid regular graph, or if some node has
            fewer than ``m`` incident edges.
    """
    degree, node_count = _require_regular(graph)
    incident = _edge_index(graph)
    for node_index, neighbours in enumerate(incident):
        if len(neighbours) != degree:
            raise ValueError(
                f"node index {node_index} has {len(neighbours)} incident edges, expected {degree}"
            )
    return {
        (node_index, coin_index): coin_index * node_count + node_index
        for node_index in range(node_count)
        for coin_index in range(degree)
    }


def initial_state(graph: nx.Graph, start_node: Hashable, *, coin_index: int = 0) -> np.ndarray:
    """``[ESTABLISHED]`` Build the walk's starting state.

    The default (``coin_index = 0``) is the uniform superposition over the ``m``
    directed edges incident to ``start_node``,

    .. code-block:: text

        psi_0 = |start_node>_position (x) (1/sqrt(m)) sum_c |c>_coin

    which is the standard DTQW starting state used by every other function in
    this module: the ``m`` coin components at ``start_node`` are in bijection
    with its ``m`` incident directed edges.

    Any other ``coin_index = k`` selects the single-component start
    ``|start_node>_position (x) |k>_coin`` (amplitude one), which seeds the
    walker along one particular shunt. This is an interface convenience for
    callers that want a deterministic first hop, not a different formalism.

    Args:
        graph: A regular graph; its size fixes the state length ``m * n``.
        start_node: Node the walker starts from.
        coin_index: ``0`` for the uniform superposition (the default) or the
            coin component ``k in [1, m)`` for a single-component start.

    Returns:
        A complex state vector of length ``m * n`` with unit norm whose support
        lies entirely on the states of ``start_node``.

    Raises:
        ValueError: If ``graph`` is not a valid regular graph, ``start_node`` is
            unknown, or ``coin_index`` is outside ``[0, m)``.
    """
    degree, node_count = _require_regular(graph)
    if start_node not in graph:
        raise ValueError(f"unknown start node {start_node!r}; known nodes: {_node_order(graph)}")
    if not isinstance(coin_index, (int, np.integer)) or isinstance(coin_index, bool):
        raise ValueError(f"coin_index must be an integer, got {type(coin_index).__name__}")
    coin_index = int(coin_index)
    if not 0 <= coin_index < degree:
        raise ValueError(f"coin_index must lie in [0, {degree}), got {coin_index}")

    start = _node_order(graph).index(start_node)
    state = np.zeros(degree * node_count, dtype=complex)
    if coin_index == 0:
        for component in range(degree):
            state[component * node_count + start] = 1.0 / math.sqrt(degree)
    else:
        state[coin_index * node_count + start] = 1.0
    return state


# ---------------------------------------------------------------------------
# Walking -- [SIMULATED]
# ---------------------------------------------------------------------------


def walk_reference_numpy(
    graph: nx.Graph,
    steps: int,
    *,
    kind: CoinKind = "hadamard",
    start_node: Hashable,
    initial_state_vector: np.ndarray | None = None,
) -> list[np.ndarray]:
    """``[SIMULATED]`` Exact reference evolution ``psi_t = (S C)^t psi_0``.

    One step is the standard DTQW step ``U = S (x) (C (x) I_n)``: the coin acts
    first, then the shunt shift moves the walker. The evolution is a dense
    matrix-vector product per step -- a classical simulation of the walk, with
    no quantum hardware involved.

    Args:
        graph: A simple ``m``-regular graph.
        steps: Number of steps; must be non-negative.
        kind: Coin family, see :func:`coin_operator`.
        start_node: Node the walker starts from.
        initial_state_vector: Optional replacement for
            :func:`initial_state`, of length ``m * n``. A vector whose norm
            differs from one by more than ``1e-12`` is renormalised; a zero
            vector is rejected.

    Returns:
        A list of ``steps + 1`` state vectors; index ``0`` is the initial
        state and index ``t`` is the state after ``t`` steps. Every state has
        unit norm to within floating-point error.

    Raises:
        ValueError: If ``steps`` is negative or not an integer, if ``graph`` is
            not a valid regular graph, if ``start_node`` is unknown, or if
            ``initial_state_vector`` has the wrong length or zero norm.
    """
    if not isinstance(steps, (int, np.integer)) or isinstance(steps, bool):
        raise ValueError(f"steps must be an integer, got {type(steps).__name__}")
    steps = int(steps)
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")

    _resolve_start_node(graph, start_node)
    if initial_state_vector is None:
        state = initial_state(graph, _resolve_start_node(graph, start_node))
    else:
        state = np.asarray(initial_state_vector, dtype=complex).reshape(-1)
        if state.size != _walk_dimension(graph):
            raise ValueError(
                f"initial_state_vector has length {state.size}, expected {_walk_dimension(graph)}"
            )
        norm = float(np.linalg.norm(state))
        if norm == 0.0:
            raise ValueError("initial_state_vector must be non-zero")
        if abs(norm - 1.0) > 1e-12:
            state = state / norm

    unitary = _walk_unitary(graph, kind)
    states = [state]
    for _ in range(steps):
        state = unitary @ state
        states.append(state)
    return states


def build_walk_circuit(
    graph: nx.Graph,
    steps: int,
    *,
    kind: CoinKind = "hadamard",
    start_node: Hashable,
    initial_state_vector: np.ndarray | None = None,
) -> QuantumCircuit:
    """``[SIMULATED]`` Build the measurement-carrying circuit for the walk at ``t = steps``.

    The circuit is::

        initialize(psi_0) -> [ UnitaryGate(U) x steps ] -> measure_all

    on ``ceil(log2(m*n))`` qubits, where ``U`` is the dense ``(m*n) x (m*n)``
    walk unitary padded with identity rows/columns when ``m * n`` is not a
    power of two. This is exactly the circuit :func:`walk_on_aer` executes for
    ``t = steps`` (it executes one such circuit per time step and runs them as a
    batch). It is public so that the notebook can draw it.

    .. warning::
       ``U`` is a single dense unitary instruction, not a decomposition into
       elementary gates. The circuit therefore *demonstrates* the walk; it is not
       a scalable, hardware-mappable description of it, and drawing it says
       nothing about gate counts on real hardware.

    Args:
        graph: A simple ``m``-regular graph.
        steps: Number of walk steps to apply before measuring. Must be
            non-negative; ``steps = 0`` yields the initial state only.
        kind: Coin family, see :func:`coin_operator`.
        start_node: Node the walker starts from.
        initial_state_vector: Optional replacement start state of length
            ``m * n``.

    Returns:
        An untranspiled :class:`qiskit.QuantumCircuit` with one quantum register
        of ``ceil(log2(m*n))`` qubits and one classical register of the same
        width.

    Raises:
        ValueError: If ``steps`` is invalid, ``graph`` is not a valid regular
            graph, ``start_node`` is unknown, the initial vector is malformed,
            or ``m * n`` exceeds :data:`MAX_CIRCUIT_DIMENSION`.
    """
    if not isinstance(steps, (int, np.integer)) or isinstance(steps, bool):
        raise ValueError(f"steps must be an integer, got {type(steps).__name__}")
    steps = int(steps)
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")

    state = _initial_vector(graph, start_node, initial_state_vector)
    dimension = state.size
    if dimension > MAX_CIRCUIT_DIMENSION:
        raise ValueError(
            f"walk dimension {dimension} exceeds MAX_CIRCUIT_DIMENSION "
            f"({MAX_CIRCUIT_DIMENSION}); the dense unitary cannot be simulated"
        )

    qubits = _qubits_for(dimension)
    unitary = _padded_unitary(graph, kind, dimension, qubits)

    register = QuantumRegister(qubits, "q")
    measure = ClassicalRegister(qubits, "c")
    circuit = QuantumCircuit(register, measure, name=f"walk-{graph.number_of_nodes()}n-{steps}t")

    padded_state = np.zeros(1 << qubits, dtype=complex)
    padded_state[:dimension] = state
    circuit.initialize(padded_state, register)
    for _ in range(steps):
        circuit.append(UnitaryGate(unitary, label="walk-step"), register)
    circuit.measure(register, measure)
    return circuit


def _padded_unitary(
    graph: nx.Graph, kind: CoinKind, dimension: int, qubits: int
) -> np.ndarray:
    """Return the walk unitary embedded in the ``2**qubits`` basis."""
    unitary = _walk_unitary(graph, kind)
    if unitary.shape[0] != dimension:
        raise ValueError(
            f"walk unitary has dimension {unitary.shape[0]}, expected {dimension}"
        )
    padded = np.eye(1 << qubits, dtype=complex)
    padded[:dimension, :dimension] = unitary
    return padded


def walk_on_aer(
    graph: nx.Graph,
    steps: int,
    *,
    kind: CoinKind = "hadamard",
    start_node: Hashable,
    initial_state_vector: np.ndarray | None = None,
    shots: int = 20_000,
    seed_simulator: int | None = 12_345,
) -> list[dict[Hashable, float]]:
    """``[SIMULATED]`` Run the walk on ``AerSimulator`` and return node distributions.

    The same walk as :func:`walk_reference_numpy` is executed through a real
    Qiskit circuit: one circuit per time step, each of them
    ``initialize(psi_0)`` followed by exactly ``t`` applications of the walk
    unitary and a single measurement of the whole register into the
    computational basis. The ``steps + 1`` circuits are transpiled and run as one
    batch, and the counts are marginalised over the coin dimension -- a basis
    index ``c * n + v`` is read as "coin ``c`` at node ``v``" -- to give the
    probability that the walker is at each node.

    One circuit per time step, rather than a mid-circuit measurement after every
    step, is required for fidelity to the reference: a computational-basis
    measurement collapses the walker, so continuing from the collapsed state
    would sample a mixture over collapse outcomes instead of the coherent
    evolution.

    The returned distributions are estimates from ``shots`` runs, so they agree
    with :func:`walk_reference_numpy` only up to sampling error; use
    :func:`max_distribution_error` to measure the gap.

    Args:
        graph: A simple ``m``-regular graph.
        steps: Number of steps; must be non-negative.
        kind: Coin family, see :func:`coin_operator`.
        start_node: Node the walker starts from.
        initial_state_vector: Optional replacement start state of length
            ``m * n``.
        shots: Number of shots per circuit, one circuit per time step. Must be
            positive.
        seed_simulator: Seed passed to ``AerSimulator.run`` so results are
            reproducible; ``None`` disables seeding.

    Returns:
        A list of ``steps + 1`` dictionaries keyed by graph node identifier, each
        mapping a node to its estimated probability of being occupied at that
        time step. Index ``t`` is the distribution after ``t`` steps. The keys
        are the node identifiers themselves, so for a network whose nodes are
        named ``"N0"``, ``"N1"``, ... the entry reads ``{"N0": p, "N1": p, ...}``.

    Raises:
        ValueError: If ``steps`` or ``shots`` are invalid, ``graph`` is not a
            valid regular graph, ``start_node`` is unknown, the initial vector is
            malformed, or ``m * n`` exceeds :data:`MAX_CIRCUIT_DIMENSION`.
    """
    if not isinstance(steps, (int, np.integer)) or isinstance(steps, bool):
        raise ValueError(f"steps must be an integer, got {type(steps).__name__}")
    steps = int(steps)
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")
    if not isinstance(shots, (int, np.integer)) or isinstance(shots, bool):
        raise ValueError(f"shots must be an integer, got {type(shots).__name__}")
    shots = int(shots)
    if shots <= 0:
        raise ValueError(f"shots must be positive, got {shots}")

    state = _initial_vector(graph, start_node, initial_state_vector)
    dimension = state.size
    if dimension > MAX_CIRCUIT_DIMENSION:
        raise ValueError(
            f"walk dimension {dimension} exceeds MAX_CIRCUIT_DIMENSION "
            f"({MAX_CIRCUIT_DIMENSION}); the dense unitary cannot be simulated"
        )

    node_count = _require_regular(graph)[1]
    order = _node_order(graph)
    qubits = _qubits_for(dimension)
    unitary = _padded_unitary(graph, kind, dimension, qubits)

    padded_state = np.zeros(1 << qubits, dtype=complex)
    padded_state[:dimension] = state

    circuits: list[QuantumCircuit] = []
    for time in range(steps + 1):
        register = QuantumRegister(qubits, "q")
        measure = ClassicalRegister(qubits, "c")
        circuit = QuantumCircuit(register, measure, name=f"walk-t{time}")
        circuit.initialize(padded_state, register)
        for _ in range(time):
            circuit.append(UnitaryGate(unitary, label="walk-step"), register)
        circuit.measure(register, measure)
        circuits.append(circuit)

    simulator = AerSimulator()
    compiled = transpile(circuits, simulator, optimization_level=1)
    result = simulator.run(compiled, shots=shots, seed_simulator=seed_simulator).result()

    mask = (1 << qubits) - 1
    distributions: list[dict[Hashable, float]] = []
    for time in range(steps + 1):
        estimated = np.zeros(node_count)
        for basis_index, occurrences in _counts_to_index_counts(result.get_counts(time)).items():
            if basis_index < dimension:
                estimated[basis_index % node_count] += occurrences
        total = float(estimated.sum())
        if total <= 0.0:
            raise RuntimeError(f"simulator returned no counts for t={time}")
        distributions.append(
            {node: float(estimated[index] / total) for index, node in enumerate(order)}
        )
    return distributions


def node_probability_distribution(
    statevector: np.ndarray, graph: nx.Graph
) -> dict[Hashable, float]:
    """``[SIMULATED]`` Marginalise a walk state over the coin dimension.

    ``P(v) = sum_c |psi[c * n + v]|^2``, renormalised to sum to one.

    Args:
        statevector: A walk state vector of length ``m * n`` (for example one
            element of :func:`walk_reference_numpy`).
        graph: The graph whose node identifiers become the dictionary keys. Its
            node count ``n`` fixes the marginalisation stride.

    Returns:
        A dictionary with one entry per node, mapping node identifier to
        probability, summing to ``1.0`` to within floating-point error.

    Raises:
        ValueError: If ``graph`` has no nodes, if the state length is not a
            multiple of ``n``, or if the state has zero norm.
    """
    order = _node_order(graph)
    node_count = len(order)
    if node_count == 0:
        raise ValueError("graph has no nodes")
    state = np.asarray(statevector, dtype=complex).reshape(-1)
    if state.size % node_count:
        raise ValueError(
            f"state length {state.size} is not a multiple of the node count {node_count}"
        )
    coin_count = state.size // node_count
    squared = np.abs(state) ** 2
    probabilities = squared.reshape(coin_count, node_count).sum(axis=0)
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("statevector has zero norm; cannot build a probability distribution")
    return {node: float(probabilities[index] / total) for index, node in enumerate(order)}


def max_distribution_error(
    a: Mapping[Hashable, float], b: Mapping[Hashable, float]
) -> float:
    """``[SIMULATED]`` Return ``max_v |a[v] - b[v]|`` over the shared keys.

    This is the metric used to compare the Aer's sampled distributions with the
    exact NumPy reference in the test suite.

    Args:
        a: First distribution, keyed by node identifier.
        b: Second distribution, keyed by node identifier.

    Returns:
        The largest absolute per-node difference, ``0.0`` for identical inputs.

    Raises:
        ValueError: If the two mappings do not have exactly the same keys.
    """
    if set(a.keys()) != set(b.keys()):
        only_a = sorted(map(str, set(a.keys()) - set(b.keys())))
        only_b = sorted(map(str, set(b.keys()) - set(a.keys())))
        raise ValueError(
            "distributions cover different nodes; "
            f"only in a: {only_a}, only in b: {only_b}"
        )
    if not a:
        return 0.0
    return max(abs(float(a[key]) - float(b[key])) for key in a)


# ---------------------------------------------------------------------------
# Walk-derived priors -- [PROPOSED]
# ---------------------------------------------------------------------------


def quantum_walk_prior(
    graph: nx.Graph,
    *,
    steps: int = 8,
    kind: CoinKind = "hadamard",
    start_node: Hashable | None = None,
    marked_nodes: Iterable[Hashable] | None = None,
) -> dict[Hashable, float]:
    """``[PROPOSED]`` Heuristic node salience derived from the DTQW.

    **This prior is a heuristic. It is not a routing guarantee, it is not a
    quantum algorithm, and no quantum computation is involved in producing it:**
    the walk is simulated classically with :func:`walk_reference_numpy` and the
    result is handed to the router as a plain ``dict[node, float]``.

    Formula
    -------
    Let ``P_t(v)`` be the node probability of the DTQW after ``t`` steps and
    ``w(v) = 2`` if ``v`` is in ``marked_nodes`` else ``w(v) = 1``. The salience
    is the marked-node-weighted time average

    .. code-block:: text

        raw[v] = (1 / (steps + 1)) * sum_{t=0}^{steps} P_t(v) * w(v)

    followed by division by ``max_v raw[v]``, so the returned values lie in
    ``[0, 1]`` and the largest is exactly ``1.0``. Without ``marked_nodes`` the
    weighting is flat and the prior is the normalised time-averaged node
    distribution; the factor of two is the only thing ``marked_nodes`` does, and
    that factor is a parameter of this prototype, not a result from the walk
    literature.

    Args:
        graph: A simple ``m``-regular graph.
        steps: Number of walk steps to average over. Must be non-negative.
        kind: Coin family, see :func:`coin_operator`.
        start_node: Start node of the walk; defaults to the first node of
            ``graph``.
        marked_nodes: Optional nodes of interest. Must all be nodes of ``graph``.

    Returns:
        A dictionary with exactly the nodes of ``graph`` as keys, mapping each
        node to a salience in ``[0, 1]`` whose maximum is ``1.0``.

    Raises:
        ValueError: If ``steps`` is negative, ``graph`` is not a valid regular
            graph, ``start_node`` is unknown, or ``marked_nodes`` contains an
            unknown node.
    """
    if not isinstance(steps, (int, np.integer)) or isinstance(steps, bool):
        raise ValueError(f"steps must be an integer, got {type(steps).__name__}")
    steps = int(steps)
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")

    order = _node_order(graph)
    marked: set[Hashable] = set()
    if marked_nodes is not None:
        marked = set(marked_nodes)
        unknown = sorted(map(str, marked - set(order)))
        if unknown:
            raise ValueError(f"marked_nodes contains unknown node(s): {unknown}")

    origin = _resolve_start_node(graph, start_node)
    states = walk_reference_numpy(graph, steps, kind=kind, start_node=origin)
    averaged = np.zeros(len(order))
    weights = np.array([2.0 if node in marked else 1.0 for node in order])
    for state in states:
        distribution = node_probability_distribution(state, graph)
        averaged += np.array([distribution[node] for node in order])
    averaged = averaged / (steps + 1) * weights

    peak = float(averaged.max())
    if peak <= 0.0:
        raise RuntimeError(
            "walk-derived salience is identically zero, which cannot happen for a "
            "normalised probability distribution; this indicates an internal error"
        )
    return {node: float(averaged[index] / peak) for index, node in enumerate(order)}


def stationary_node_distribution(
    graph: nx.Graph,
    *,
    steps: int = 64,
    kind: CoinKind = "hadamard",
    start_node: Hashable | None = None,
) -> dict[Hashable, float]:
    """``[SIMULATED]`` Time-averaged node distribution of the walk.

    The average of ``P_t(v)`` over ``t = 0 .. steps``. Unlike
    :func:`quantum_walk_prior` this is a plain probability distribution: it sums
    to ``1.0`` and is *not* rescaled so that its maximum is one.

    For a **disconnected** graph this factorises over the connected components:
    a walker started inside one component never leaves it, so the returned
    distribution has exactly zero entries for every node outside that component
    and the average is the walk's average on the starting component alone.
    (A disconnected graph is still accepted here as long as it is
    ``m``-regular; the ``[ESTABLISHED]`` shunt decomposition does not require
    connectivity.)

    Args:
        graph: A simple ``m``-regular graph, connected or not.
        steps: Number of walk steps to average over. Must be non-negative.
        kind: Coin family, see :func:`coin_operator`.
        start_node: Start node of the walk; defaults to the first node of
            ``graph``.

    Returns:
        A dictionary with one entry per node, mapping node identifier to
        probability, summing to ``1.0`` to within floating-point error.

    Raises:
        ValueError: If ``steps`` is negative, ``graph`` is not a valid regular
            graph, or ``start_node`` is unknown.
    """
    if not isinstance(steps, (int, np.integer)) or isinstance(steps, bool):
        raise ValueError(f"steps must be an integer, got {type(steps).__name__}")
    steps = int(steps)
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")

    order = _node_order(graph)
    origin = _resolve_start_node(graph, start_node)
    states = walk_reference_numpy(graph, steps, kind=kind, start_node=origin)
    averaged = np.zeros(len(order))
    for state in states:
        distribution = node_probability_distribution(state, graph)
        averaged += np.array([distribution[node] for node in order])
    averaged /= steps + 1
    return {node: float(averaged[index]) for index, node in enumerate(order)}


# ---------------------------------------------------------------------------
# Figures -- [SIMULATED]
# ---------------------------------------------------------------------------


def plot_walk_evolution(
    graph: nx.Graph,
    steps: int = 8,
    *,
    kind: CoinKind = "hadamard",
    start_node: Hashable | None = None,
    out_path: str | pathlib.Path | None = None,
    title: str | None = None,
) -> "Figure":
    """``[SIMULATED]`` Grouped bar chart of the node distribution over time.

    The x axis is the node, and each node carries one bar per time step
    ``t = 0 .. steps``, so the figure shows how each node's share of the walker
    moves over time.

    Args:
        graph: A simple ``m``-regular graph.
        steps: Number of walk steps. Must be non-negative.
        kind: Coin family, see :func:`coin_operator`.
        start_node: Start node of the walk; defaults to the first node of
            ``graph``.
        out_path: If given, the figure is saved here and parent directories are
            created.
        title: Figure title; a descriptive default is used when ``None``.

    Returns:
        The :class:`matplotlib.figure.Figure`. The non-interactive ``Agg``
        backend is selected if an interactive GUI backend happens to be active,
        so the figure can be produced on a headless machine.

    Raises:
        ValueError: If ``steps`` is negative, ``graph`` is not a valid regular
            graph, ``start_node`` is unknown, or ``out_path`` is a directory.
    """
    _ensure_headless_backend()
    from matplotlib.figure import Figure
    from matplotlib.ticker import MultipleLocator

    if not isinstance(steps, (int, np.integer)) or isinstance(steps, bool):
        raise ValueError(f"steps must be an integer, got {type(steps).__name__}")
    steps = int(steps)
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")

    order = _node_order(graph)
    origin = _resolve_start_node(graph, start_node)
    states = walk_reference_numpy(graph, steps, kind=kind, start_node=origin)
    series = np.array(
        [
            [node_probability_distribution(state, graph)[node] for node in order]
            for state in states
        ]
    )

    figure = Figure(figsize=(max(6.0, 1.4 * len(order)), 4.0))
    axes = figure.subplots()
    positions = np.arange(len(order), dtype=float)
    width = 0.8 / max(1, steps + 1)
    for time in range(steps + 1):
        axes.bar(
            positions + (time - steps / 2) * width,
            series[time],
            width=width,
            label=f"t={time}",
        )
    axes.set_xticks(positions)
    axes.set_xticklabels([str(node) for node in order])
    axes.set_xlabel("node")
    axes.set_ylabel("probability")
    axes.xaxis.set_major_locator(MultipleLocator(1))
    axes.set_title(title or f"DTQW node probabilities ({order[0]!r} -> {order[-1]!r}, {kind})")
    if steps + 1 <= 12:
        axes.legend(ncols=min(7, steps + 1), fontsize="small")
    figure.tight_layout()
    return _finalise_figure(figure, out_path)


def plot_walk_heatmap(
    graph: nx.Graph,
    steps: int = 12,
    *,
    kind: CoinKind = "hadamard",
    start_node: Hashable | None = None,
    out_path: str | pathlib.Path | None = None,
) -> "Figure":
    """``[SIMULATED]`` Heatmap of the node distribution: nodes by time step.

    Args:
        graph: A simple ``m``-regular graph.
        steps: Number of walk steps. Must be non-negative.
        kind: Coin family, see :func:`coin_operator`.
        start_node: Start node of the walk; defaults to the first node of
            ``graph``.
        out_path: If given, the figure is saved here and parent directories are
            created.

    Returns:
        The :class:`matplotlib.figure.Figure`, with the time step on the x axis,
        the node on the y axis, and a colour bar showing the probability scale.

    Raises:
        ValueError: If ``steps`` is negative, ``graph`` is not a valid regular
            graph, ``start_node`` is unknown, or ``out_path`` is a directory.
    """
    _ensure_headless_backend()
    from matplotlib.figure import Figure

    if not isinstance(steps, (int, np.integer)) or isinstance(steps, bool):
        raise ValueError(f"steps must be an integer, got {type(steps).__name__}")
    steps = int(steps)
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")

    order = _node_order(graph)
    origin = _resolve_start_node(graph, start_node)
    states = walk_reference_numpy(graph, steps, kind=kind, start_node=origin)
    series = np.array(
        [
            [node_probability_distribution(state, graph)[node] for node in order]
            for state in states
        ]
    ).T

    figure = Figure(figsize=(max(6.0, 0.9 * (steps + 1)), max(3.0, 0.45 * len(order) + 1.5)))
    axes = figure.subplots()
    image = axes.imshow(series, aspect="auto", origin="lower", cmap="magma")
    axes.set_xticks(np.arange(steps + 1))
    axes.set_xticklabels([str(t) for t in range(steps + 1)])
    axes.set_yticks(np.arange(len(order)))
    axes.set_yticklabels([str(node) for node in order])
    axes.set_xlabel("time step t")
    axes.set_ylabel("node")
    axes.set_title(f"DTQW node probability heatmap ({kind} coin)")
    figure.colorbar(image, ax=axes, label="probability")
    figure.tight_layout()
    return _finalise_figure(figure, out_path)


# ---------------------------------------------------------------------------
# Helpers and reporting
# ---------------------------------------------------------------------------


def example_walk_graph(kind: str = "cycle", n: int = 4) -> nx.Graph:
    """Return a small regular example graph for the walk.

    Only ``m``-regular graphs are supported by :func:`shunt_decomposition`, so
    the ``n`` argument is interpreted as follows:

    ``"cycle"``
        ``nx.cycle_graph(n)``, which is ``2``-regular for ``n >= 3``.
    ``"complete"``
        ``nx.complete_graph(n)``, which is ``(n-1)``-regular for ``n >= 2``.
    ``"path"``
        ``nx.path_graph(2)`` regardless of ``n``. ``nx.path_graph(n)`` for
        ``n > 2`` has two nodes of degree one and is *not* regular, so it cannot
        be walked by this module; ``n = 2`` (a single link, which is
        ``1``-regular) is the only member of that family that qualifies.

    .. note::
       The ``"path"`` result is ``1``-regular, which is a valid input for
       :func:`shunt_decomposition` but **not walkable**: a coin space of
       dimension ``m = 1`` is rejected by :func:`coin_operator` (and would be
       trivial anyway), so the walk functions raise ``ValueError`` on it. Use
       ``"cycle"`` or ``"complete"`` for anything that actually walks.

    Args:
        kind: One of ``"cycle"``, ``"complete"``, ``"path"``.
        n: Size hint, interpreted as described above.

    Returns:
        A :class:`networkx.Graph` suitable for every function in this module.

    Raises:
        ValueError: If ``kind`` is unknown, or if ``n`` is below the minimum for
            the requested family (``3`` for a cycle, ``2`` for a complete graph).
    """
    if kind == "cycle":
        if n < 3:
            raise ValueError(f"a cycle needs at least 3 nodes, got n={n}")
        return nx.cycle_graph(n)
    if kind == "complete":
        if n < 2:
            raise ValueError(f"a complete graph needs at least 2 nodes, got n={n}")
        return nx.complete_graph(n)
    if kind == "path":
        return nx.path_graph(2)
    raise ValueError(f"unknown example graph kind {kind!r}; expected 'cycle', 'complete' or 'path'")


def environment_report() -> str:
    """Return a printable description of the simulation environment.

    The report states explicitly that everything here is ``[SIMULATED]``: the
    walk is run with dense NumPy matrix-vector products and by Qiskit Aer's
    ``AerSimulator``, never on quantum hardware.

    Returns:
        A multi-line string listing the Python version, the relevant package
        versions and the simulation statement.
    """
    import matplotlib as _matplotlib
    import networkx as _networkx
    import qiskit as _qiskit
    import qiskit_aer as _qiskit_aer

    lines = [
        "BQT-MDQW quantum walk environment",
        "=================================",
        f"python                : {platform.python_version()} ({sys.platform})",
        f"numpy                 : {np.__version__}",
        f"networkx              : {_networkx.__version__}",
        f"matplotlib            : {_matplotlib.__version__} (backend {_ensure_headless_backend()})",
        f"qiskit                : {_qiskit.__version__}",
        f"qiskit-aer            : {_qiskit_aer.__version__}",
        "",
        "SIMULATED: the discrete-time coined quantum walk in this module is",
        "propagated with dense numpy matrix-vector products and executed as a",
        "Qiskit circuit on AerSimulator. No quantum hardware is used, no quantum",
        "advantage is claimed, and the [PROPOSED] walk-derived node prior is a",
        "classical heuristic computed from that simulation.",
    ]
    return "\n".join(lines)