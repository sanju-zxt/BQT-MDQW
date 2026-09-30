"""Tests for :mod:`src.quantum_walk`.

Scope
-----
The tests fall into three groups that mirror the provenance legend of the
module:

* the ``[ESTABLISHED]`` operator algebra (shunt decomposition, coin, shift) --
  checked structurally and for unitarity;
* the ``[SIMULATED]`` walk evolution -- checked for normalisation, and, most
  importantly, the Aer circuit is cross-validated against the exact NumPy
  reference;
* the ``[PROPOSED]`` node prior and the figures -- checked for the invariants
  the router and the notebook rely on.

Nothing here asserts a performance property. The Aer tests assert agreement
with the reference up to sampling error, with a tolerance far wider than the
observed error so that they do not become flaky.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
import networkx as nx
import numpy as np
import pytest

# Make ``src`` importable no matter how pytest is invoked (``pytest`` from the
# repository root, or ``python -m pytest`` from anywhere in the tree).
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matplotlib.figure import Figure  # noqa: E402  (import after sys.path fix)

from src import quantum_walk as qw  # noqa: E402

#: Tolerance for the Aer-versus-reference cross-validation. The observed
#: maximum errors are two orders of magnitude smaller than this (see the
#: reported measurements), so the assertion is not tight; it only rejects a
#: genuinely wrong circuit.
AER_TOLERANCE = 0.06

#: Number of shots used by the cross-validation tests.
AER_SHOTS = 20_000


def _is_permutation_matrix(matrix: np.ndarray) -> bool:
    """Return whether ``matrix`` is a 0/1 matrix with exactly one 1 per row/column."""
    ones = np.ones(matrix.shape[0])
    return bool(
        np.isin(matrix, (0, 1)).all()
        and np.array_equal(matrix.sum(axis=0), ones)
        and np.array_equal(matrix.sum(axis=1), ones)
    )


# ---------------------------------------------------------------------------
# 1. shunt_decomposition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("factory", "degree"),
    [
        (lambda: nx.cycle_graph(4), 2),
        (lambda: nx.cycle_graph(6), 2),
        (lambda: nx.complete_graph(4), 3),
        (lambda: nx.complete_graph(3), 2),
    ],
    ids=["cycle4", "cycle6", "complete4", "complete3"],
)
def test_shunt_decomposition_reconstructs_adjacency_exactly(factory, degree):
    graph = factory()
    shunts = qw.shunt_decomposition(graph)

    assert len(shunts) == degree
    for shunt in shunts:
        assert shunt.shape == (graph.number_of_nodes(), graph.number_of_nodes())
        assert shunt.dtype.kind in "iu"
        assert _is_permutation_matrix(shunt)

    adjacency = nx.to_numpy_array(graph, nodelist=list(graph.nodes()), dtype=int)
    total = sum(shunts)
    assert total.dtype.kind in "iu"
    assert np.array_equal(total, adjacency)


@pytest.mark.parametrize(
    "graph_factory",
    [nx.cycle_graph, nx.complete_graph],
)
def test_shunt_decomposition_is_deterministic(graph_factory):
    graph = graph_factory(5)
    first = qw.shunt_decomposition(graph)
    second = qw.shunt_decomposition(graph)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


# ---------------------------------------------------------------------------
# 2. shunt_decomposition rejects invalid graphs
# ---------------------------------------------------------------------------


def test_shunt_decomposition_rejects_non_regular_graph():
    with pytest.raises(ValueError, match="regular"):
        qw.shunt_decomposition(nx.path_graph(4))


def test_shunt_decomposition_rejects_edge_free_and_empty_graphs():
    with pytest.raises(ValueError, match="no edges"):
        qw.shunt_decomposition(nx.empty_graph(3))
    with pytest.raises(ValueError, match="empty"):
        qw.shunt_decomposition(nx.Graph())


def test_shunt_decomposition_rejects_self_loops():
    graph = nx.cycle_graph(4)
    graph.add_edge(0, 0)
    with pytest.raises(ValueError, match="self-loops"):
        qw.shunt_decomposition(graph)


def test_shunt_decomposition_rejects_disconnected_irregular_union():
    graph = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(4))
    with pytest.raises(ValueError, match="regular"):
        qw.shunt_decomposition(graph)


# ---------------------------------------------------------------------------
# 3. coin_operator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [2, 3, 4])
@pytest.mark.parametrize("kind", ["hadamard", "y", "grover"])
def test_coin_operator_is_unitary(kind, m):
    coin = qw.coin_operator(m, kind)
    assert coin.shape == (m, m)
    defect = np.max(np.abs(coin.conj().T @ coin - np.eye(m)))
    assert defect < 1e-12


def test_coin_operator_default_is_hadamard():
    assert np.allclose(qw.coin_operator(3), qw.coin_operator(3, "hadamard"))


def test_coin_operator_rejects_unknown_kind_and_small_dimension():
    with pytest.raises(ValueError, match="unknown coin kind"):
        qw.coin_operator(3, "nonsense")
    with pytest.raises(ValueError, match="at least 2"):
        qw.coin_operator(1, "hadamard")
    with pytest.raises(ValueError, match="at least 2"):
        qw.coin_operator(0, "grover")


def test_grover_coin_reflects_the_uniform_state():
    uniform = np.ones(3) / np.sqrt(3.0)
    coin = qw.coin_operator(3, "grover")
    assert np.allclose(coin @ uniform, uniform)


# ---------------------------------------------------------------------------
# 4. shift_operator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("builder", "size"),
    [(nx.cycle_graph, 4), (nx.complete_graph, 4)],
    ids=["cycle4", "complete4"],
)
def test_shift_operator_is_unitary(builder, size):
    shunts = qw.shunt_decomposition(builder(size))
    shift = qw.shift_operator(shunts)
    dimension = len(shunts) * size
    assert shift.shape == (dimension, dimension)
    defect = np.max(np.abs(shift.conj().T @ shift - np.eye(dimension)))
    assert defect < 1e-12


def test_shift_operator_is_block_diagonal_by_coin_component():
    graph = nx.cycle_graph(4)
    shunts = qw.shunt_decomposition(graph)
    shift = qw.shift_operator(shunts)
    for index, shunt in enumerate(shunts):
        block = slice(index * 4, (index + 1) * 4)
        assert np.array_equal(shift[block, block], shunt)


def test_shift_operator_rejects_mismatched_shunt_sizes():
    shunts = qw.shunt_decomposition(nx.cycle_graph(4))
    smaller = np.zeros((3, 3), dtype=int)
    smaller[0, 1] = smaller[1, 0] = 1
    smaller[2, 2] = 1
    with pytest.raises(ValueError, match="all shunts"):
        qw.shift_operator([shunts[0], smaller])


def test_shift_operator_rejects_empty_and_non_square():
    with pytest.raises(ValueError, match="non-empty"):
        qw.shift_operator([])
    with pytest.raises(ValueError, match="square"):
        qw.shift_operator([np.ones((2, 3))])


def test_shift_operator_rejects_non_permutation_shunt():
    with pytest.raises(ValueError, match="permutation matrix"):
        qw.shift_operator([np.full((3, 3), 1)])


# ---------------------------------------------------------------------------
# 5. initial_state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("start", [0, 1, 2, 3])
def test_initial_state_is_normalised(start):
    graph = nx.cycle_graph(4)
    state = qw.initial_state(graph, start)
    assert state.shape == (8,)
    assert abs(float(np.sum(np.abs(state) ** 2)) - 1.0) < 1e-12


def test_initial_state_support_lies_on_the_start_node():
    graph = nx.cycle_graph(4)
    start_index = list(graph.nodes()).index(2)
    state = qw.initial_state(graph, 2)
    support = np.flatnonzero(np.abs(state) > 1e-12)
    assert support.size == 2  # one per incident edge, m = 2
    for flat in support:
        coin_index, node_index = divmod(int(flat), graph.number_of_nodes())
        assert node_index == start_index
        assert 0 <= coin_index < 2


def test_initial_state_single_coin_component_variant():
    graph = nx.cycle_graph(4)
    state = qw.initial_state(graph, 1, coin_index=1)
    assert abs(float(np.sum(np.abs(state) ** 2)) - 1.0) < 1e-12
    assert abs(abs(state[1 * 4 + 1]) - 1.0) < 1e-12


def test_initial_state_rejects_unknown_node_and_bad_coin_index():
    graph = nx.cycle_graph(4)
    with pytest.raises(ValueError, match="unknown start node"):
        qw.initial_state(graph, 99)
    with pytest.raises(ValueError, match=r"coin_index"):
        qw.initial_state(graph, 0, coin_index=5)


def test_edge_index_map_is_a_bijection_onto_the_flat_basis():
    graph = nx.cycle_graph(6)
    mapping = qw.edge_index_map(graph)
    assert len(mapping) == 2 * 6
    assert sorted(mapping.values()) == list(range(12))
    assert mapping[(0, 0)] == 0
    assert mapping[(0, 1)] == 6


# ---------------------------------------------------------------------------
# 6. walk_reference_numpy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("builder", "size", "kind"),
    [
        (nx.cycle_graph, 4, "hadamard"),
        (nx.cycle_graph, 6, "hadamard"),
        (nx.complete_graph, 3, "hadamard"),
        (nx.complete_graph, 4, "y"),
        (nx.cycle_graph, 5, "grover"),
    ],
    ids=["cycle4-hadamard", "cycle6-hadamard", "complete3-hadamard", "complete4-y", "cycle5-grover"],
)
def test_walk_reference_numpy_preserves_normalisation(builder, size, kind):
    graph = builder(size)
    states = qw.walk_reference_numpy(graph, 20, kind=kind, start_node=0)
    assert len(states) == 21
    for state in states:
        assert abs(float(np.sum(np.abs(state) ** 2)) - 1.0) < 1e-12


def test_walk_reference_numpy_returns_the_initial_state_first():
    graph = nx.cycle_graph(4)
    states = qw.walk_reference_numpy(graph, 3, start_node=1)
    initial = qw.initial_state(graph, 1)
    assert np.allclose(states[0], initial)
    assert not np.allclose(states[1], initial)


def test_walk_reference_numpy_rejects_bad_input():
    graph = nx.cycle_graph(4)
    with pytest.raises(ValueError, match="non-negative"):
        qw.walk_reference_numpy(graph, -1, start_node=0)
    with pytest.raises(ValueError, match="unknown start node"):
        qw.walk_reference_numpy(graph, 2, start_node=42)
    with pytest.raises(ValueError, match="regular"):
        qw.walk_reference_numpy(nx.path_graph(4), 2, start_node=0)
    with pytest.raises(ValueError, match="length"):
        qw.walk_reference_numpy(graph, 2, start_node=0, initial_state_vector=np.ones(3))
    with pytest.raises(ValueError, match="non-zero"):
        qw.walk_reference_numpy(graph, 2, start_node=0, initial_state_vector=np.zeros(8))


# ---------------------------------------------------------------------------
# 7. Aer cross-validation against the NumPy reference
# ---------------------------------------------------------------------------


def _max_aer_error(graph, steps=6, shots=AER_SHOTS, kind="hadamard", start_node=0):
    states = qw.walk_reference_numpy(graph, steps, kind=kind, start_node=start_node)
    sampled = qw.walk_on_aer(graph, steps, kind=kind, start_node=start_node, shots=shots)
    assert len(sampled) == steps + 1
    worst = 0.0
    for time, distribution in enumerate(sampled):
        reference = qw.node_probability_distribution(states[time], graph)
        assert set(distribution) == set(graph.nodes)
        worst = max(worst, qw.max_distribution_error(distribution, reference))
    return worst


def test_walk_on_aer_matches_reference_on_cycle4():
    # m = 2, n = 4 -> 8-dimensional walk, exactly 3 qubits: no padding needed.
    assert _max_aer_error(nx.cycle_graph(4)) < AER_TOLERANCE


def test_walk_on_aer_matches_reference_on_complete3():
    # m = 2, n = 3 -> 6-dimensional walk padded into 3 qubits (8 basis states).
    # Two of the eight basis states stay unpopulated; the marginalisation reads
    # the basis index modulo n, so the padding is handled correctly.
    assert _max_aer_error(nx.complete_graph(3)) < AER_TOLERANCE


def test_walk_on_aer_matches_reference_on_cycle6():
    # m = 2, n = 6 -> 12-dimensional walk padded into 4 qubits.
    assert _max_aer_error(nx.cycle_graph(6)) < AER_TOLERANCE


def test_walk_on_aer_distributions_sum_to_one():
    graph = nx.cycle_graph(4)
    distributions = qw.walk_on_aer(graph, 4, start_node=0, shots=4000)
    for distribution in distributions:
        assert abs(sum(distribution.values()) - 1.0) < 1e-9


def test_walk_on_aer_uses_graph_node_names_as_keys():
    graph = nx.relabel_nodes(nx.cycle_graph(4), {index: f"N{index}" for index in range(4)})
    distributions = qw.walk_on_aer(graph, 2, start_node="N0", shots=2000)
    assert set(distributions[0]) == {"N0", "N1", "N2", "N3"}


def test_walk_on_aer_rejects_invalid_arguments():
    graph = nx.cycle_graph(4)
    with pytest.raises(ValueError, match="non-negative"):
        qw.walk_on_aer(graph, -2, start_node=0)
    with pytest.raises(ValueError, match="shots must be positive"):
        qw.walk_on_aer(graph, 1, start_node=0, shots=0)
    with pytest.raises(ValueError, match="unknown start node"):
        qw.walk_on_aer(graph, 1, start_node="nowhere")


def test_walk_on_aer_rejects_oversized_walks(monkeypatch):
    monkeypatch.setattr(qw, "MAX_CIRCUIT_DIMENSION", 4)
    with pytest.raises(ValueError, match="MAX_CIRCUIT_DIMENSION"):
        qw.walk_on_aer(nx.cycle_graph(4), 1, start_node=0)
    with pytest.raises(ValueError, match="MAX_CIRCUIT_DIMENSION"):
        qw.build_walk_circuit(nx.cycle_graph(4), 1, start_node=0)


def test_build_walk_circuit_structure():
    graph = nx.cycle_graph(4)
    circuit = qw.build_walk_circuit(graph, 3, start_node=0)
    assert circuit.num_qubits == 3
    assert circuit.num_clbits == 3
    names = [instruction.operation.name for instruction in circuit.data]
    assert names.count("unitary") == 3
    assert names[-1] == "measure"


def test_build_walk_circuit_pads_non_power_of_two_dimension():
    graph = nx.complete_graph(3)
    circuit = qw.build_walk_circuit(graph, 1, start_node=0)
    assert circuit.num_qubits == 3  # ceil(log2(6))


# ---------------------------------------------------------------------------
# 8. node_probability_distribution / max_distribution_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("time", [0, 1, 4, 11])
def test_node_probability_distribution_sums_to_one(time):
    graph = nx.cycle_graph(6)
    state = qw.walk_reference_numpy(graph, time, start_node=0)[time]
    distribution = qw.node_probability_distribution(state, graph)
    assert len(distribution) == graph.number_of_nodes()
    assert set(distribution) == set(graph.nodes)
    assert abs(sum(distribution.values()) - 1.0) < 1e-12
    assert all(0.0 <= value <= 1.0 for value in distribution.values())


def test_node_probability_distribution_marginalises_the_coin_dimension():
    graph = nx.cycle_graph(4)
    state = qw.initial_state(graph, 0)
    distribution = qw.node_probability_distribution(state, graph)
    assert distribution[0] == pytest.approx(1.0)
    assert distribution[1] == pytest.approx(0.0)


def test_node_probability_distribution_rejects_bad_length():
    graph = nx.cycle_graph(4)
    with pytest.raises(ValueError, match="multiple of the node count"):
        qw.node_probability_distribution(np.ones(7) / np.sqrt(7), graph)


def test_max_distribution_error():
    a = {"x": 0.5, "y": 0.5}
    assert qw.max_distribution_error(a, a) == 0.0
    assert qw.max_distribution_error(a, {"x": 0.4, "y": 0.6}) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="different nodes"):
        qw.max_distribution_error(a, {"x": 1.0})


# ---------------------------------------------------------------------------
# 9. quantum_walk_prior and stationary_node_distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("builder", "size"),
    [(nx.cycle_graph, 4), (nx.cycle_graph, 6), (nx.complete_graph, 4)],
    ids=["cycle4", "cycle6", "complete4"],
)
def test_quantum_walk_prior_contract(builder, size):
    graph = builder(size)
    prior = qw.quantum_walk_prior(graph, steps=6)
    assert set(prior) == set(graph.nodes)
    assert len(prior) == graph.number_of_nodes()
    assert all(0.0 <= value <= 1.0 for value in prior.values())
    assert max(prior.values()) == pytest.approx(1.0, abs=1e-9)


def test_quantum_walk_prior_boosts_marked_nodes():
    graph = nx.cycle_graph(6)
    plain = qw.quantum_walk_prior(graph, steps=6, start_node=0)
    marked = qw.quantum_walk_prior(graph, steps=6, start_node=0, marked_nodes=[2, 3])
    for node in (2, 3):
        assert marked[node] >= plain[node]
    assert max(marked.values()) == pytest.approx(1.0, abs=1e-9)
    assert set(marked) == set(graph.nodes)


def test_quantum_walk_prior_accepts_string_node_names():
    graph = nx.relabel_nodes(nx.cycle_graph(4), {index: f"N{index}" for index in range(4)})
    prior = qw.quantum_walk_prior(graph, steps=4, start_node="N0", marked_nodes=["N2"])
    assert set(prior) == {"N0", "N1", "N2", "N3"}
    assert max(prior.values()) == pytest.approx(1.0, abs=1e-9)


def test_quantum_walk_prior_rejects_bad_input():
    graph = nx.cycle_graph(4)
    with pytest.raises(ValueError, match="non-negative"):
        qw.quantum_walk_prior(graph, steps=-1)
    with pytest.raises(ValueError, match="unknown node"):
        qw.quantum_walk_prior(graph, steps=2, marked_nodes=[99])
    with pytest.raises(ValueError, match="unknown start node"):
        qw.quantum_walk_prior(graph, steps=2, start_node="nowhere")


def test_stationary_node_distribution_is_a_distribution():
    graph = nx.cycle_graph(6)
    stationary = qw.stationary_node_distribution(graph, steps=12)
    assert set(stationary) == set(graph.nodes)
    assert abs(sum(stationary.values()) - 1.0) < 1e-12
    assert all(value >= 0.0 for value in stationary.values())


def test_stationary_distribution_factorises_over_components():
    # Documented behaviour: a walker started in one component never leaves it.
    graph = nx.disjoint_union(nx.cycle_graph(4), nx.cycle_graph(4))
    stationary = qw.stationary_node_distribution(graph, steps=12, start_node=0)
    assert set(stationary) == set(graph.nodes)
    assert stationary[4] == 0.0 and stationary[5] == 0.0
    assert stationary[6] == 0.0 and stationary[7] == 0.0
    assert abs(sum(stationary.values()) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# 10. example_walk_graph
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "n"),
    [("cycle", 4), ("cycle", 6), ("cycle", 9), ("complete", 3), ("complete", 5), ("path", 4)],
)
def test_example_walk_graph_is_regular(kind, n):
    graph = qw.example_walk_graph(kind, n)
    degrees = {int(degree) for _, degree in graph.degree()}
    assert len(degrees) == 1
    assert nx.to_numpy_array(graph, dtype=int).sum() > 0


def test_example_walk_graph_unknown_kind():
    with pytest.raises(ValueError, match="unknown example graph kind"):
        qw.example_walk_graph("hypercube", 4)


def test_example_walk_graph_rejects_too_small_n():
    with pytest.raises(ValueError, match="at least 3"):
        qw.example_walk_graph("cycle", 2)
    with pytest.raises(ValueError, match="at least 2"):
        qw.example_walk_graph("complete", 1)


def test_example_walk_graph_output_feeds_the_walk():
    graph = qw.example_walk_graph("cycle", 4)
    shunts = qw.shunt_decomposition(graph)
    assert len(shunts) == 2
    states = qw.walk_reference_numpy(graph, 4, start_node=0)
    assert len(states) == 5


# ---------------------------------------------------------------------------
# 11. Figures
# ---------------------------------------------------------------------------


def test_plot_walk_evolution_returns_figure_and_writes_file(tmp_path):
    graph = qw.example_walk_graph("cycle", 6)
    destination = tmp_path / "nested" / "evolution.png"
    figure = qw.plot_walk_evolution(graph, steps=4, out_path=destination)
    assert isinstance(figure, Figure)
    assert destination.exists()
    assert destination.stat().st_size > 0


def test_plot_walk_evolution_without_out_path(tmp_path):
    graph = qw.example_walk_graph("cycle", 4)
    figure = qw.plot_walk_evolution(graph, steps=3, title="custom title", out_path=None)
    assert isinstance(figure, Figure)
    assert list(tmp_path.iterdir()) == []


def test_plot_walk_heatmap_returns_figure_and_writes_file(tmp_path):
    graph = qw.example_walk_graph("cycle", 5)
    destination = tmp_path / "heatmap.png"
    figure = qw.plot_walk_heatmap(graph, steps=6, out_path=destination)
    assert isinstance(figure, Figure)
    assert destination.exists()
    assert destination.stat().st_size > 0


def test_plot_functions_reject_bad_input(tmp_path):
    graph = qw.example_walk_graph("cycle", 4)
    with pytest.raises(ValueError, match="non-negative"):
        qw.plot_walk_evolution(graph, steps=-1)
    with pytest.raises(ValueError, match="unknown start node"):
        qw.plot_walk_heatmap(graph, steps=2, start_node="nowhere")
    with pytest.raises(ValueError, match="directory"):
        qw.plot_walk_evolution(graph, steps=2, out_path=tmp_path)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_environment_report_states_simulation():
    report = qw.environment_report()
    assert "SIMULATED" in report
    assert "qiskit" in report
    assert "numpy" in report
    assert "quantum hardware" in report


def test_public_api_is_exported():
    for name in (
        "shunt_decomposition",
        "coin_operator",
        "shift_operator",
        "initial_state",
        "edge_index_map",
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
    ):
        assert name in qw.__all__
        assert callable(getattr(qw, name))


def test_matplotlib_backend_is_non_interactive_after_use():
    qw.plot_walk_evolution(qw.example_walk_graph("cycle", 4), steps=2)
    assert matplotlib.get_backend().lower() == "agg"