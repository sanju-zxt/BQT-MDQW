"""Tests for :mod:`src.quantum_teleportation`.

Every assertion here is either exact (algebraic identities, structural counts,
input validation) or statistical (fidelity of Aer shot counts against the ideal
distribution). Statistical assertions use tolerances sized from the binomial
standard deviation at the shot count used, so they are not tuned to one seed;
``test_reproducibility`` separately pins determinism for a fixed seed.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.quantum_info import partial_trace
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from src.quantum_teleportation import (
    ArbitraryState,
    bell_pair_entanglement_demo,
    bidirectional_marginals,
    bidirectional_teleportation_circuit,
    branch_marginal,
    build_branch_circuits,
    build_noise_model,
    build_teleportation_circuit,
    environment_report,
    fidelity_metrics,
    marginal_counts,
    run_on_aer,
    teleport_link_with_noise,
)

# Bob's readout is classical bit index 2 of the three-bit teleportation register.
BOB = 2

# (theta, phi) pairs spanning a near-basis state, an equatorial superposition, a
# nearly orthogonal state and an exact basis state.
STATES = [
    ArbitraryState(theta=math.pi / 3, phi=0.9),
    ArbitraryState(theta=0.35, phi=2.1),
    ArbitraryState(theta=1.2, phi=3.0),
]


def normalised(counts: dict[str, int]) -> dict[str, float]:
    """Turn a bit count dictionary into a probability dictionary."""
    total = sum(counts.values())
    assert total > 0, "no shots retained"
    return {key: value / total for key, value in counts.items()}


# ---------------------------------------------------------------------------
# 1. ArbitraryState validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("theta", "phi"),
    [
        (-0.01, 0.0),
        (math.pi / 2 + 0.01, 0.0),
        (math.pi, 0.0),
        (0.5, -0.1),
        (0.5, 2 * math.pi),
        (0.5, 10.0),
        (float("nan"), 0.0),
        (0.5, float("inf")),
    ],
)
def test_arbitrary_state_rejects_out_of_range_angles(theta: float, phi: float) -> None:
    with pytest.raises(ValueError):
        ArbitraryState(theta=theta, phi=phi)


@pytest.mark.parametrize(
    ("theta", "phi"),
    [(0.0, 0.0), (math.pi / 2, 0.0), (0.5, 0.0), (0.5, 2 * math.pi - 1e-12)],
)
def test_arbitrary_state_accepts_boundary_angles(theta: float, phi: float) -> None:
    state = ArbitraryState(theta=theta, phi=phi)
    assert state.theta == theta


def test_arbitrary_state_rejects_non_real_angles() -> None:
    with pytest.raises(TypeError):
        ArbitraryState(theta="0.5", phi=0.0)  # type: ignore[arg-type]


def test_arbitrary_state_is_frozen() -> None:
    state = ArbitraryState(theta=0.5, phi=1.0)
    with pytest.raises(FrozenInstanceError):
        state.theta = 0.9  # type: ignore[misc]


def test_statevector_and_probabilities_agree() -> None:
    state = ArbitraryState(theta=0.83, phi=4.2)
    vector = state.as_statevector()
    assert vector.shape == (2,)
    assert np.isclose(np.linalg.norm(vector), 1.0)
    assert np.allclose(
        vector, np.array([math.cos(0.83), np.exp(1j * 4.2) * math.sin(0.83)])
    )
    probabilities = state.ideal_probabilities()
    assert np.isclose(probabilities["0"] + probabilities["1"], 1.0)
    assert np.isclose(abs(vector[0]) ** 2, probabilities["0"])
    assert np.isclose(abs(vector[1]) ** 2, probabilities["1"])
    assert "cos(" in state.as_label() and "|1>" in state.as_label()


# ---------------------------------------------------------------------------
# 2. Ideal teleportation on Aer
# ---------------------------------------------------------------------------


def test_dynamic_circuit_teleports_the_state() -> None:
    state = ArbitraryState(theta=math.pi / 3, phi=0.9)
    shots = 8192
    counts = run_on_aer(
        build_teleportation_circuit(state, dynamic=True), shots=shots, seed=20260930
    )
    bob = normalised(marginal_counts(counts, bit_index=BOB))
    ideal = state.ideal_probabilities()

    # Binomial sigma on a probability of 0.75 at 8192 shots is about 0.0048, so
    # a 0.02 window is roughly four sigma: tight enough to be meaningful, wide
    # enough that it cannot fail by seed luck.
    assert abs(bob["0"] - ideal["0"]) < 0.02
    assert abs(bob["1"] - ideal["1"]) < 0.02

    metrics = fidelity_metrics(bob, ideal)
    assert metrics["bhattacharyya"] > 0.9
    assert metrics["total_variation"] < 0.02
    assert metrics["hellinger"] < 0.03


def test_dynamic_circuit_fidelity_is_high_across_states() -> None:
    for state in STATES:
        counts = run_on_aer(
            build_teleportation_circuit(state), shots=4096, seed=4242
        )
        bob = normalised(marginal_counts(counts, bit_index=BOB))
        metrics = fidelity_metrics(bob, state.ideal_probabilities())
        assert metrics["bhattacharyya"] > 0.9, state.as_label()


def test_alice_outcomes_are_uniform_and_bob_depends_on_correction() -> None:
    """Alice's Bell-basis outcome is uniform; Bob's readout is not.

    The second half is the regression guard for the correction convention: with
    ``dynamic=False`` Bob applies X and Z unconditionally, so his marginal must
    come out near the maximally mixed distribution rather than near the ideal
    one. If the feed-forward is wired to the wrong classical bits this test
    fails, because a swapped convention also produces a uniform marginal.
    """
    state = ArbitraryState(theta=math.pi / 3, phi=0.9)
    counts = run_on_aer(build_teleportation_circuit(state), shots=8192, seed=99)

    alice_outcomes: dict[str, int] = {}
    for key, value in counts.items():
        outcome = key[-2:]
        alice_outcomes[outcome] = alice_outcomes.get(outcome, 0) + value
    total = sum(alice_outcomes.values())
    for outcome, value in alice_outcomes.items():
        assert abs(value / total - 0.25) < 0.02, outcome

    uncontrolled = run_on_aer(
        build_teleportation_circuit(state, dynamic=False), shots=8192, seed=99
    )
    bob = normalised(marginal_counts(uncontrolled, bit_index=BOB))
    assert abs(bob["0"] - 0.5) < 0.05
    assert fidelity_metrics(bob, state.ideal_probabilities())["bhattacharyya"] < 0.95


# ---------------------------------------------------------------------------
# 3. Deterministic branch circuits
# ---------------------------------------------------------------------------


def test_branch_marginal_matches_ideal_for_all_branches() -> None:
    branches = build_branch_circuits(STATES[0])
    assert sorted(branches) == ["00", "01", "10", "11"]

    shots = 16384
    for state in (STATES[0], STATES[1]):
        for label, circuit in build_branch_circuits(state).items():
            counts = run_on_aer(circuit, shots=shots, seed=777)
            # Bob's readout is c2, i.e. bit_index 2; c0/c1 are Alice's outcome.
            selected = branch_marginal(counts, label, bit_index=BOB)
            retained = sum(selected.values())
            # Post-selection keeps about a quarter of the shots.
            assert retained > shots // 8, (label, retained)

            observed = normalised(selected)
            ideal = state.ideal_probabilities()
            # ~4000 shots are retained per branch, so the binomial sigma on p(0)
            # is about 0.008 and 0.04 is a five-sigma window. Measured worst case
            # over 15 seeds x 2 states x 4 branches was 0.021.
            assert abs(observed["0"] - ideal["0"]) < 0.04, (state.as_label(), label)
            metrics = fidelity_metrics(observed, ideal)
            assert metrics["bhattacharyya"] > 0.98, (state.as_label(), label)


def test_branches_agree_with_dynamic_circuit_marginal() -> None:
    """The four post-selected branches and the dynamic circuit share one marginal.

    Summing the branch counts reconstructs the unconditional marginal because
    every branch leaves Bob in exactly ``|psi>`` once its own correction is
    applied.
    """
    state = STATES[1]
    shots = 16384
    dynamic = run_on_aer(
        build_teleportation_circuit(state), shots=shots, seed=31337
    )
    dynamic_bob = normalised(marginal_counts(dynamic, bit_index=BOB))

    combined = {"0": 0, "1": 0}
    for label, circuit in build_branch_circuits(state).items():
        counts = run_on_aer(circuit, shots=shots, seed=31337)
        for key, value in branch_marginal(counts, label, bit_index=BOB).items():
            combined[key] += value

    branch_bob = normalised(combined)
    assert abs(branch_bob["0"] - dynamic_bob["0"]) < 0.02
    assert fidelity_metrics(branch_bob, dynamic_bob)["bhattacharyya"] > 0.99


def test_branch_corrections_are_actually_branch_specific() -> None:
    """Guard: the four branch circuits are distinct and each correction matters.

    Without this, the "all four branches reproduce the ideal distribution" test
    would also pass if all four circuits were in fact the same circuit. Here the
    identity branch circuit is post-selected on the opposite branch, where it
    must fail, because a no-op correction applied to ``XZ|psi>`` leaves
    ``XZ|psi>`` rather than ``|psi>``.
    """
    state = STATES[0]
    shots = 16384
    ideal_p0 = state.ideal_probabilities()["0"]
    circuits = build_branch_circuits(state)

    identity_counts = run_on_aer(circuits["00"], shots=shots, seed=5150)
    matched = normalised(branch_marginal(identity_counts, "00", bit_index=BOB))
    assert abs(matched["0"] - ideal_p0) < 0.04

    opposite = normalised(branch_marginal(identity_counts, "11", bit_index=BOB))
    assert abs(opposite["0"] - ideal_p0) > 0.3

    # And symmetrically for the fully-corrected branch.
    full_counts = run_on_aer(circuits["11"], shots=shots, seed=5150)
    full_matched = normalised(branch_marginal(full_counts, "11", bit_index=BOB))
    full_opposite = normalised(branch_marginal(full_counts, "00", bit_index=BOB))
    assert abs(full_matched["0"] - ideal_p0) < 0.04
    assert abs(full_opposite["0"] - ideal_p0) > 0.3


# ---------------------------------------------------------------------------
# 4. Circuit structure
# ---------------------------------------------------------------------------


def test_teleportation_circuit_structure() -> None:
    circuit = build_teleportation_circuit(STATES[0], dynamic=True)
    assert circuit.num_qubits == 3
    assert circuit.num_clbits == 3
    ops = dict(circuit.count_ops())
    assert ops["h"] == 2, ops
    assert ops["cx"] == 2, ops
    assert ops["measure"] == 3, ops
    assert ops["ry"] == 1 and ops["rz"] == 1, ops
    assert ops["if_else"] == 2, ops
    assert ops.get("x") is None and ops.get("z") is None, ops


def test_teleportation_circuit_structure_without_feedforward() -> None:
    circuit = build_teleportation_circuit(STATES[0], dynamic=False)
    assert circuit.num_qubits == 3 and circuit.num_clbits == 3
    ops = dict(circuit.count_ops())
    assert "if_else" not in ops, ops
    assert ops["x"] == 1 and ops["z"] == 1, ops
    assert ops["h"] == 2 and ops["cx"] == 2 and ops["measure"] == 3, ops


def test_branch_circuits_structure() -> None:
    for label, circuit in build_branch_circuits(STATES[0]).items():
        assert circuit.num_qubits == 3
        assert circuit.num_clbits == 3
        ops = dict(circuit.count_ops())
        assert ops["measure"] == 3, label
        assert ops["h"] == 2 and ops["cx"] == 2, label
        assert "if_else" not in ops, label
        expected_x = int(label[0])
        expected_z = int(label[1])
        assert ops.get("x", 0) == expected_x, (label, ops)
        assert ops.get("z", 0) == expected_z, (label, ops)


def test_teleportation_circuit_rejects_bad_state() -> None:
    with pytest.raises(TypeError):
        build_teleportation_circuit("not-a-state")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_branch_circuits(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5. fidelity_metrics
# ---------------------------------------------------------------------------


def test_fidelity_metrics_identical_distributions() -> None:
    metrics = fidelity_metrics({"0": 0.25, "1": 0.75}, {"0": 0.25, "1": 0.75})
    assert metrics["bhattacharyya"] == pytest.approx(1.0)
    assert metrics["total_variation"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["hellinger"] == pytest.approx(0.0, abs=1e-12)


def test_fidelity_metrics_normalises_inputs() -> None:
    """Raw counts and raw probabilities must give the same answer."""
    from_counts = fidelity_metrics({"0": 250, "1": 750}, {"0": 0.25, "1": 0.75})
    from_probs = fidelity_metrics({"0": 0.25, "1": 0.75}, {"0": 0.25, "1": 0.75})
    assert from_counts["bhattacharyya"] == pytest.approx(from_probs["bhattacharyya"])
    assert from_counts["total_variation"] == pytest.approx(from_probs["total_variation"])


def test_fidelity_metrics_disjoint_distributions() -> None:
    metrics = fidelity_metrics({"0": 1.0}, {"1": 1.0})
    assert metrics["bhattacharyya"] == pytest.approx(0.0)
    assert metrics["total_variation"] == pytest.approx(1.0)
    assert metrics["hellinger"] == pytest.approx(1.0)


def test_fidelity_metrics_known_value() -> None:
    """A 50/50 observation against a pure |0> has bhattacharyya = 0.5."""
    metrics = fidelity_metrics({"0": 0.5, "1": 0.5}, {"0": 1.0, "1": 0.0})
    assert metrics["bhattacharyya"] == pytest.approx(0.5)
    assert metrics["total_variation"] == pytest.approx(0.5)
    assert metrics["hellinger"] == pytest.approx(math.sqrt(1.0 - math.sqrt(0.5)))


def test_fidelity_metrics_rejects_invalid_distributions() -> None:
    with pytest.raises(ValueError):
        fidelity_metrics({"0": -1.0}, {"0": 1.0})
    with pytest.raises(ValueError):
        fidelity_metrics({"0": 0.0}, {"0": 0.0})
    with pytest.raises(TypeError):
        fidelity_metrics([0.5, 0.5], {"0": 1.0})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 6. build_noise_model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strength", [0.0, 0.05, 0.5, 1.0])
def test_build_noise_model_accepts_valid_strengths(strength: float) -> None:
    model = build_noise_model(strength)
    assert isinstance(model, NoiseModel)
    assert build_noise_model(strength, readout_error=0.25) is not None
    assert build_noise_model(strength, readout_error=None) is not None


@pytest.mark.parametrize("strength", [-0.01, 1.01, 2.0, float("nan")])
def test_build_noise_model_rejects_out_of_range_strength(strength: float) -> None:
    with pytest.raises(ValueError):
        build_noise_model(strength)


@pytest.mark.parametrize("readout", [-0.01, 0.51, 1.0])
def test_build_noise_model_rejects_out_of_range_readout(readout: float) -> None:
    with pytest.raises(ValueError):
        build_noise_model(0.1, readout_error=readout)


def test_build_noise_model_rejects_non_numeric() -> None:
    with pytest.raises(TypeError):
        build_noise_model("loud")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_noise_model(0.1, readout_error="noisy")  # type: ignore[arg-type]


def test_build_noise_model_empty_when_no_error_requested() -> None:
    """A zero-strength, zero-readout model must be an exact no-op."""
    ideal = build_noise_model(0.0, readout_error=0.0)
    assert not ideal.noise_instructions


def test_build_noise_model_degrades_a_teleported_transfer() -> None:
    """Sanity check that the model is wired to the right instructions."""
    state = ArbitraryState(theta=math.pi / 3, phi=0.9)
    noisy = run_on_aer(
        build_teleportation_circuit(state),
        shots=8192,
        seed=11,
        noise_model=build_noise_model(0.3, readout_error=0.1),
    )
    bob = normalised(marginal_counts(noisy, bit_index=BOB))
    metrics = fidelity_metrics(bob, state.ideal_probabilities())
    assert metrics["bhattacharyya"] < 0.99


# ---------------------------------------------------------------------------
# 7. teleport_link_with_noise ordering
# ---------------------------------------------------------------------------

#: (noise_score, fidelity_hint) triples in increasing order of link quality.
LINK_GRADIENT = [
    (1.0, 0.70),
    (0.5, 0.90),
    (0.0, 0.999),
]

#: Enough shots and enough distinct states that the *mean* fidelity is stable;
#: the per-state spread is large (near-eigenstates barely degrade) so the
#: assertion is deliberately made on the mean, not on any single state.
ORDERING_STATES = [
    ArbitraryState(theta=0.35, phi=0.4),
    ArbitraryState(theta=0.80, phi=1.7),
    ArbitraryState(theta=1.20, phi=3.0),
    ArbitraryState(theta=0.60, phi=5.2),
    ArbitraryState(theta=1.45, phi=2.2),
]


def _mean_link_fidelity(noise_score: float, fidelity_hint: float, shots: int) -> float:
    values = [
        teleport_link_with_noise(
            state,
            noise_score=noise_score,
            fidelity_hint=fidelity_hint,
            shots=shots,
            seed=20260930,
        )["measured_fidelity"]
        for state in ORDERING_STATES
    ]
    return float(np.mean(values))


@pytest.mark.parametrize("shots", [2048, 4096, 8192])
def test_link_fidelity_decreases_with_link_quality(shots: int) -> None:
    """Mean measured fidelity must be monotonically ordered by link quality.

    Run at three shot counts with a fresh seeded sample each time so that a
    single lucky seed cannot carry the assertion. The margin below is far
    smaller than the observed separation (~0.03 between the extremes) yet far
    larger than the sampling noise of a 5-state mean.
    """
    means = [
        _mean_link_fidelity(noise_score=noise, fidelity_hint=fidelity, shots=shots)
        for noise, fidelity in LINK_GRADIENT
    ]
    worst, middle, best = means
    assert worst < middle < best, means
    assert best - worst > 0.02, means


def test_teleport_link_with_noise_contract() -> None:
    state = ArbitraryState(theta=0.9, phi=1.1)
    result = teleport_link_with_noise(
        state, noise_score=0.4, fidelity_hint=0.85, shots=4096, seed=808
    )
    assert set(result) == {
        "measured_fidelity",
        "noise_score",
        "fidelity_hint",
        "shots",
        "bit_flips",
    }
    assert 0.0 <= result["measured_fidelity"] <= 1.0
    assert result["shots"] == 4096
    assert isinstance(result["shots"], int)
    assert isinstance(result["bit_flips"], int)
    assert 0 <= result["bit_flips"] <= 4096
    assert result["noise_score"] == pytest.approx(0.4)
    assert result["fidelity_hint"] == pytest.approx(0.85)


def test_teleport_link_with_noise_is_reproducible() -> None:
    state = ArbitraryState(theta=1.0, phi=2.0)
    kwargs = dict(noise_score=0.6, fidelity_hint=0.8, shots=2048, seed=606)
    assert teleport_link_with_noise(state, **kwargs) == teleport_link_with_noise(
        state, **kwargs
    )


def test_teleport_link_with_noise_validates_inputs() -> None:
    state = STATES[0]
    with pytest.raises(TypeError):
        teleport_link_with_noise(
            "nope", noise_score=0.1, fidelity_hint=0.9, shots=64, seed=1
        )
    with pytest.raises(ValueError):
        teleport_link_with_noise(
            state, noise_score=0.1, fidelity_hint=0.9, shots=0, seed=1
        )
    with pytest.raises(ValueError):
        teleport_link_with_noise(
            state, noise_score=float("nan"), fidelity_hint=0.9, shots=64, seed=1
        )


# ---------------------------------------------------------------------------
# 8. Bidirectional model
# ---------------------------------------------------------------------------


def test_bidirectional_circuit_structure() -> None:
    circuit = bidirectional_teleportation_circuit(STATES[0], STATES[2])
    assert circuit.num_qubits == 6
    assert circuit.num_clbits == 6
    ops = dict(circuit.count_ops())
    assert ops["h"] == 4, ops
    assert ops["cx"] == 4, ops
    assert ops["measure"] == 6, ops
    assert ops["if_else"] == 4, ops


def test_bidirectional_transfers_both_directions() -> None:
    state_a = ArbitraryState(theta=math.pi / 3, phi=0.9)
    state_b = ArbitraryState(theta=0.4, phi=2.6)
    counts = run_on_aer(
        bidirectional_teleportation_circuit(state_a, state_b), shots=8192, seed=4321
    )
    assert all(len(key) == 6 for key in counts)

    a_recovered, b_recovered = bidirectional_marginals(counts)
    a_observed = normalised(a_recovered)
    b_observed = normalised(b_recovered)

    # 8192 shots -> binomial sigma ~ 0.005 on either probability.
    assert abs(a_observed["0"] - state_b.ideal_probabilities()["0"]) < 0.02
    assert abs(b_observed["0"] - state_a.ideal_probabilities()["0"]) < 0.02
    assert fidelity_metrics(a_observed, state_b.ideal_probabilities())[
        "bhattacharyya"
    ] > 0.97
    assert fidelity_metrics(b_observed, state_a.ideal_probabilities())[
        "bhattacharyya"
    ] > 0.97


def test_bidirectional_marginals_validates_keys() -> None:
    with pytest.raises(ValueError):
        bidirectional_marginals({})
    with pytest.raises(ValueError):
        bidirectional_marginals({"011": 100})


def test_bidirectional_circuit_rejects_bad_states() -> None:
    with pytest.raises(TypeError):
        bidirectional_teleportation_circuit("a", STATES[0])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 9. Bell pair entanglement demo
# ---------------------------------------------------------------------------


def test_bell_pair_entanglement_demo() -> None:
    circuit, statevector = bell_pair_entanglement_demo()
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 4
    assert statevector.shape == (16,)
    assert np.isclose(np.linalg.norm(statevector), 1.0)

    # |0000>, |0011>, |1100>, |1111> in Qiskit's little-endian basis ordering.
    support = np.flatnonzero(np.abs(statevector) > 1e-9)
    assert sorted(support.tolist()) == [0, 3, 12, 15]
    assert np.allclose(np.abs(statevector)[support], 0.5)
    assert np.allclose(np.abs(statevector)[support] ** 2, 0.25)
    assert np.isclose(np.sum(np.abs(statevector) ** 2), 1.0)


def test_bell_pairs_are_pure_but_each_qubit_is_maximally_mixed() -> None:
    """The signature of two *independent* Bell pairs.

    The four-qubit state is pure (entropy 0) and each pair is separately pure
    (entropy 0), yet every *single* qubit is maximally mixed (entropy 1 bit).
    That last fact is what distinguishes a Bell pair from a classically
    correlated pair: a state like ``|0000> + |1111>`` also has zero global
    entropy but its individual qubits are not maximally mixed.
    """
    _, statevector = bell_pair_entanglement_demo()
    density = np.outer(statevector, statevector.conj())

    def entropy(reduced: np.ndarray) -> float:
        eigenvalues = np.clip(np.linalg.eigvalsh(reduced), 0.0, 1.0)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    assert entropy(density) == pytest.approx(0.0, abs=1e-9)
    # Each pair on its own is still a pure Bell state.
    assert entropy(partial_trace(density, [2, 3])) == pytest.approx(0.0, abs=1e-9)
    assert entropy(partial_trace(density, [0, 1])) == pytest.approx(0.0, abs=1e-9)
    # Any single qubit on its own is maximally mixed: 1 bit of entropy.
    for qubit in range(4):
        others = [index for index in range(4) if index != qubit]
        reduced = partial_trace(density, others)
        assert np.allclose(reduced, 0.5 * np.eye(2)), qubit
        assert entropy(reduced) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 10. Reproducibility and helper plumbing
# ---------------------------------------------------------------------------


def test_same_seed_gives_identical_counts() -> None:
    circuit = build_teleportation_circuit(STATES[1])
    first = run_on_aer(circuit, shots=4096, seed=123456)
    second = run_on_aer(circuit, shots=4096, seed=123456)
    assert first == second


def test_different_seeds_change_counts() -> None:
    """Guards against ``run_on_aer`` accidentally ignoring its seed argument."""
    circuit = build_teleportation_circuit(STATES[1])
    first = run_on_aer(circuit, shots=4096, seed=1)
    second = run_on_aer(circuit, shots=4096, seed=2)
    assert first != second


def test_marginal_counts_sums_over_the_other_bits() -> None:
    counts = {"000": 10, "001": 20, "100": 30, "101": 40}
    # bit_index 0 is the rightmost character, i.e. c0.
    assert marginal_counts(counts, bit_index=0) == {"0": 40, "1": 60}
    # bit_index 1 is c1, which is 0 in all four keys.
    assert marginal_counts(counts, bit_index=1) == {"0": 100}
    # bit_index 2 is c2, the leftmost character.
    assert marginal_counts(counts, bit_index=2) == {"0": 30, "1": 70}
    assert sum(marginal_counts(counts, bit_index=0).values()) == 100


def test_marginal_counts_validates_inputs() -> None:
    with pytest.raises(ValueError):
        marginal_counts({"000": 1}, bit_index=3)
    with pytest.raises(ValueError):
        marginal_counts({"000": 1}, bit_index=-1)
    with pytest.raises(ValueError):
        marginal_counts({"00": 1})
    with pytest.raises(ValueError):
        marginal_counts({"000": 1}, num_bits=0)


def test_branch_marginal_validates_inputs() -> None:
    # Key layout "c2c1c0", so the branch label is key[-2:]:
    # "010"/"110" are branch "10" and "001"/"101" are branch "01".
    counts = {"010": 10, "110": 20, "001": 30, "101": 40}
    assert branch_marginal(counts, "10", bit_index=BOB) == {"0": 10, "1": 20}
    assert branch_marginal(counts, "01", bit_index=BOB) == {"0": 30, "1": 40}
    assert sum(branch_marginal(counts, "00", bit_index=BOB).values()) == 0
    with pytest.raises(ValueError):
        branch_marginal(counts, "0")
    with pytest.raises(ValueError):
        branch_marginal(counts, "02")
    with pytest.raises(ValueError):
        branch_marginal(counts, "00", bit_index=9)
    with pytest.raises(ValueError):
        branch_marginal({"0": 5}, "00")


def test_run_on_aer_validates_inputs() -> None:
    with pytest.raises(TypeError):
        run_on_aer("not-a-circuit")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        run_on_aer(build_teleportation_circuit(STATES[0]), shots=0)
    with pytest.raises(TypeError):
        run_on_aer(build_teleportation_circuit(STATES[0]), shots=1.5)  # type: ignore[arg-type]


def test_run_on_aer_reports_backend_failure() -> None:
    """A circuit Aer cannot handle must surface as RuntimeError, not a raw QiskitError."""
    broken = QuantumCircuit(1, 1)
    broken.append(
        Instruction(name="not_a_real_gate", num_qubits=1, num_clbits=0, params=[]),
        [0],
    )
    with pytest.raises(RuntimeError):
        run_on_aer(broken, shots=32, seed=1)


def test_environment_report_states_simulation() -> None:
    report = environment_report()
    assert "SIMULATED" in report
    assert "python" in report
    assert "qiskit" in report
    assert "qiskit-aer" in report
    assert len(report.splitlines()) >= 5


def test_environment_report_records_installed_versions() -> None:
    import qiskit
    import qiskit_aer

    report = environment_report()
    assert qiskit.__version__ in report
    assert qiskit_aer.__version__ in report


def test_simulator_backend_is_reused_cleanly() -> None:
    """Sanity: the module's circuits run on a bare AerSimulator with no help."""
    circuit = build_teleportation_circuit(STATES[0])
    counts = run_on_aer(circuit, shots=1024, seed=7)
    assert sum(counts.values()) == 1024
    assert AerSimulator() is not None
