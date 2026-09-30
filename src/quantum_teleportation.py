"""Quantum teleportation circuits, the bidirectional *model*, and Aer execution.

Provenance legend
-----------------

Every claim made by this module falls into exactly one of three buckets, and
each public docstring repeats the relevant tag.

``[ESTABLISHED]``
    The quantum teleportation protocol itself. The circuit built by
    :func:`build_teleportation_circuit` is the Bennett et al. (1993) scheme:
    an unknown single-qubit state is destroyed at Alice, two classical bits are
    broadcast, and Bob's shared qubit is reconstructed with a Pauli correction.
    This part is textbook physics and does not depend on anything invented here.

``[PROPOSED]``
    Additions made *for this prototype* that are not claimed as physics:

    * :func:`bidirectional_teleportation_circuit` -- a composite six-qubit
      *model* of two simultaneous, opposing teleportation channels.
    * The mapping from the abstract :class:`src.network.EdgeMetrics` link
      attributes (``noise``, ``fidelity``) onto concrete Aer noise channels,
      performed by :func:`teleport_link_with_noise`.
    * :func:`build_branch_circuits` -- a set of per-outcome circuit variants
      used to unit-test the Pauli correction algebra. The *protocol* is
      established; the way this prototype forces a single branch for testing is
      a modelling choice, and is documented in detail on that function.

``[SIMULATED]``
    Everything produced by actually running a circuit: shot counts, marginal
    distributions, statevectors and fidelity numbers. All of it comes from
    Qiskit Aer, a classical simulator. **No number produced by this module was
    measured on quantum hardware**, and none of it should be read as a device
    characterisation. :func:`environment_report` prints this statement at
    runtime.

Qubit / classical-bit layout
----------------------------

For the three-qubit circuit produced by :func:`build_teleportation_circuit`::

    q0 = Alice's payload qubit       (the unknown state lives here)
    q1 = Alice's half of the Bell pair
    q2 = Bob's qubit                 (the reconstructed state lands here)

    c0 = outcome of Alice's measurement of q0
    c1 = outcome of Alice's measurement of q1
    c2 = outcome of Bob's measurement of q2

Correction convention
---------------------

After Alice's Bell-basis measurement the joint state factorises as an equal
mixture over the four outcomes ``(c0, c1)``, with Bob holding

=========  ==================  ==========================
``c0``     ``c1``               Bob's un-corrected state
=========  ==================  ==========================
0          0                   ``|psi>``
0          1                   ``X |psi>``
1          0                   ``Z |psi>``
1          1                   ``X Z |psi>``  (up to phase)
=========  ==================  ==========================

so the recovery operation is ``X**c1`` followed by ``Z**c0``: **X on Bob
whenever Alice's *Bell-pair* qubit ``q1`` reported 1, and Z on Bob whenever
Alice's *payload* qubit ``q0`` reported 1.** This is the standard
Bennett/Nielsen-Chuang convention and it is the one implemented here. Note that
this pairs each gate with the qubit that the gate is *conditioned on* in the
Bennett circuit diagram: ``q1``'s result drives ``X`` because ``q1`` holds
Alice's half of the shared pair, and ``q0``'s result drives ``Z`` because
``q0`` holds the payload. Getting this pairing backwards still produces a
perfectly running circuit that is silently *wrong* -- Bob's marginal becomes
uniformly mixed -- which is why the convention is spelled out here and asserted
in the test-suite.
"""

from __future__ import annotations

import math
import platform
from dataclasses import dataclass
from typing import Any

import numpy as np
import qiskit
import qiskit_aer
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.exceptions import QiskitError
from qiskit_aer import AerSimulator
from qiskit_aer.library import SaveStatevector
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

__all__ = [
    "ArbitraryState",
    "build_teleportation_circuit",
    "build_branch_circuits",
    "branch_marginal",
    "run_on_aer",
    "marginal_counts",
    "fidelity_metrics",
    "build_noise_model",
    "bidirectional_teleportation_circuit",
    "bidirectional_marginals",
    "bell_pair_entanglement_demo",
    "teleport_link_with_noise",
    "environment_report",
]

#: Total classical-register width of the three-qubit teleportation circuit.
TELEPORTATION_CLBITS = 3

#: Total classical-register width of the six-qubit bidirectional circuit.
BIDIRECTIONAL_CLBITS = 6

#: Single-qubit gate names that :func:`build_noise_model` decorates with a
#: depolarizing channel. The list is deliberately broad so that the model keeps
#: covering the circuit after transpilation has rewritten gates for the target
#: basis set.
_ONE_QUBIT_NOISY_GATES = (
    "id",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "sx",
    "rx",
    "ry",
    "rz",
    "p",
    "u",
    "u1",
    "u2",
    "u3",
)

#: Two-qubit gate names decorated with a depolarizing channel. ``cx`` is the
#: only one the teleportation circuits actually use; the rest are listed so the
#: model remains meaningful if a transpiler picks a different entangling gate.
_TWO_QUBIT_NOISY_GATES = (
    "cx",
    "cz",
    "swap",
    "ecr",
    "rxx",
    "ryy",
    "rzz",
    "rzx",
    "iswap",
)


def _format_angle(value: float) -> str:
    """Render an angle for a human-readable state label.

    Args:
        value: A finite, real angle in radians, already validated by
            :class:`ArbitraryState`.

    Returns:
        The shortest round-trippable decimal string for ``value``.

    Raises:
        ValueError: If ``value`` cannot be converted to a float, or is not
            finite.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"angle must be a real number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"angle must be finite, got {value!r}")
    return repr(number)


@dataclass(frozen=True)
class ArbitraryState:
    """A single-qubit pure state in the Bloch-sphere polar/azimuthal parameterisation.

    The state is

    .. math::

        |\\psi\\rangle = \\cos\\theta\\,|0\\rangle
                      + e^{i\\phi}\\sin\\theta\\,|1\\rangle ,

    so that ``theta`` is the polar angle off the ``+z`` axis and ``phi`` is the
    azimuthal phase on the equator. ``theta = 0`` is ``|0>``, ``theta = pi/2`` is
    ``|1>``, and any ``phi`` gives an equatorial ("pure superposition") state.

    Because the protocol destroys Alice's copy of the state, these are exactly
    the parameters a teleportation experiment needs: they fix the probability
    distribution that Bob must reproduce without ever learning the amplitudes.

    Args:
        theta: Polar angle in radians, in the closed interval ``[0, pi/2]``.
        phi: Azimuthal phase in radians, in the half-open interval ``[0, 2*pi)``.

    Raises:
        TypeError: If ``theta`` or ``phi`` is not a real number.
        ValueError: If either angle is non-finite or out of its allowed range.
    """

    theta: float
    phi: float

    def __post_init__(self) -> None:
        for name, value in (("theta", self.theta), ("phi", self.phi)):
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(
                    f"{name} must be a real number, got {type(value).__name__}"
                )
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if not 0.0 <= float(self.theta) <= math.pi / 2:
            raise ValueError(
                f"theta must lie in [0, pi/2] = [0, {math.pi / 2}], "
                f"got {self.theta!r}"
            )
        if not 0.0 <= float(self.phi) < 2 * math.pi:
            raise ValueError(
                f"phi must lie in [0, 2*pi) = [0, {2 * math.pi}), got {self.phi!r}"
            )

    def as_statevector(self) -> np.ndarray:
        """Return the single-qubit statevector in Qiskit's ``|0>``, ``|1>`` order.

        Returns:
            A one-dimensional ``complex128`` array of length 2 whose norm is 1
            up to floating-point error.
        """
        theta = float(self.theta)
        phi = float(self.phi)
        return np.array(
            [math.cos(theta), complex(math.sin(theta)) * np.exp(1j * phi)],
            dtype=np.complex128,
        )

    def ideal_probabilities(self) -> dict[str, float]:
        """Return the exact measurement probabilities Bob must reproduce.

        The phase ``phi`` does not appear because computational-basis
        measurement is phase-insensitive; only ``theta`` matters.

        Returns:
            ``{"0": cos(theta)**2, "1": sin(theta)**2}``. The two values always
            sum to 1 up to floating-point error.
        """
        theta = float(self.theta)
        return {"0": math.cos(theta) ** 2, "1": math.sin(theta) ** 2}

    def as_label(self) -> str:
        """Return a compact algebraic label for the state.

        Returns:
            A string of the form
            ``cos(<theta>)|0> + exp(i*<phi>)*sin(<theta>)|1>``, suitable for use
            as a plot legend or table caption.
        """
        theta = _format_angle(self.theta)
        phi = _format_angle(self.phi)
        return f"cos({theta})|0> + exp(i*{phi})*sin({theta})|1>"


def build_teleportation_circuit(
    state: ArbitraryState, *, dynamic: bool = True
) -> QuantumCircuit:
    """Build the three-qubit Bennett teleportation circuit.

    ``[ESTABLISHED]`` The protocol is the one of Bennett *et al.*, PRL **70**,
    1895 (1993). ``[SIMULATED]`` Executing it is pure Aer work.

    Circuit structure, in order:

    1. Prepare :math:`|\\psi\\rangle` on ``q0`` with ``ry(2*theta)`` then
       ``rz(phi)``. The two-argument angle is required because the Bloch
       spherical convention gives amplitude ``cos(theta)`` on ``|0>``.
    2. Distribute a Bell pair: ``h(q1)`` lifts ``q1`` out of ``|0>`` and
       ``cx(q1, q2)`` entangles it with Bob.
    3. Alice performs the Bell-basis measurement: ``cx(q0, q1)`` then ``h(q0)``
       maps ``(|psi>_q0 |phi+>_q1q2)`` into a Bell-basis product with the
       original state displaced onto Bob.
    4. Alice measures ``q0 -> c0`` and ``q1 -> c1`` and broadcasts the two
       bits. This is the step that destroys the original state.
    5. Bob applies ``X`` if ``c1 == 1`` and ``Z`` if ``c0 == 1`` (see the module
       docstring for the pairing convention).
    6. Bob measures ``q2 -> c2``.

    Args:
        state: The unknown state to teleport.
        dynamic: If ``True`` (the default) the classical feed-forward is
            expressed as two :meth:`QuantumCircuit.if_test` blocks, i.e. Bob
            reacts to the measured bits inside the circuit. If ``False`` both
            corrections are applied unconditionally instead. ``False`` is *not*
            teleportation: without the classical bit it cannot work, and Bob's
            marginal collapses towards the maximally mixed distribution. It is
            provided purely as a control experiment that demonstrates why the
            classical channel is indispensable.

    Returns:
        A :class:`~qiskit.circuit.QuantumCircuit` with exactly 3 qubits and
        3 classical bits, laid out as documented in the module docstring.

    Raises:
        TypeError: If ``state`` is not an :class:`ArbitraryState`.
    """
    if not isinstance(state, ArbitraryState):
        raise TypeError(
            f"state must be an ArbitraryState, got {type(state).__name__}"
        )

    qc = QuantumCircuit(3, TELEPORTATION_CLBITS, name="teleport")

    # (1) Prepare the unknown state on Alice's payload qubit.
    qc.ry(2.0 * float(state.theta), 0)
    qc.rz(float(state.phi), 0)

    # (2) Bell pair shared between Alice (q1) and Bob (q2).
    qc.h(1)
    qc.cx(1, 2)

    # (3) Alice's Bell-basis measurement: CNOT then Hadamard moves |psi> to Bob.
    qc.cx(0, 1)
    qc.h(0)

    # (4) Alice reads out her two Bell-basis outcome bits.
    qc.measure(0, 0)
    qc.measure(1, 1)

    # (5) Bob's classical feed-forward recovery. X is driven by the outcome of
    #     Alice's Bell-pair qubit (c1); Z is driven by the outcome of Alice's
    #     payload qubit (c0).
    if dynamic:
        with qc.if_test((qc.clbits[1], 1)):
            qc.x(2)
        with qc.if_test((qc.clbits[0], 1)):
            qc.z(2)
    else:
        # No classical information reaches Bob, so these Pauli gates are
        # applied for all four outcomes and cancel only by accident.
        qc.x(2)
        qc.z(2)

    # (6) Bob's readout.
    qc.measure(2, 2)
    return qc


def build_branch_circuits(state: ArbitraryState) -> dict[str, QuantumCircuit]:
    """Build the four per-outcome circuit variants of the teleportation protocol.

    ``[PROPOSED]`` The *protocol* these variants implement is established; the
    way a single branch is isolated for unit testing is a choice of this
    prototype, and the reasoning matters, so it is spelled out here.

    Why post-selection is unavoidable
    ---------------------------------
    A tempting construction is to omit Alice's measurement entirely, apply the
    branch's Pauli correction unconditionally to ``q2`` and read ``q2`` off.
    That does **not** work, and the reason is structural rather than a matter
    of bookkeeping. Right after Alice's Bell-basis measurement the joint state is

    .. math::

        \\tfrac12\\big(
            |00\\rangle X^0 Z^0|\\psi\\rangle
          + |01\\rangle X^1 Z^0|\\psi\\rangle
          + |10\\rangle X^0 Z^1|\\psi\\rangle
          + |11\\rangle X^1 Z^1|\\psi\\rangle \\big)

    where the kets label Alice's qubits ``q0 q1``. The four states
    ``|psi>``, ``X|psi>``, ``Z|psi>``, ``XZ|psi>`` are mutually orthogonal for any
    ``|psi>``, so the *reduced* state of ``q2`` -- with Alice's qubits traced out
    rather than measured -- is proportional to
    ``I + X^2 + Z^2 + (XZ)^2 = 4 I``, i.e. exactly the maximally mixed state.
    No unitary applied to ``q2`` can change that. Concretely: **Alice's outcome
    is uniformly distributed over the four Bell-basis states for every
    possible** ``|psi>``; it carries no information about the teleported state
    at all, so there is no preparation of ``|psi>`` that makes it deterministic.
    Forcing a branch requires discarding the other three, i.e. classical
    post-selection -- which is exactly how deterministic-branch teleportation is
    handled in linear optics.

    What these circuits therefore do
    ---------------------------------
    Each returned circuit runs the *full* protocol but replaces the two
    :meth:`~qiskit.circuit.QuantumCircuit.if_test` blocks by the corresponding
    unconditional Pauli gates, pinned to one specific outcome. Run the circuit
    and post-select on the recorded Alice bits (see :func:`branch_marginal`)
    and Bob's qubit is in exactly :math:`|\\psi\\rangle` -- no sampling noise
    from Alice's side, only binomial sampling noise from Bob's own readout.

    Args:
        state: The unknown state to teleport.

    Returns:
        A mapping from branch label to circuit, with keys ``"00"``, ``"01"``,
        ``"10"`` and ``"11"``. The label is ordered ``c1c0`` -- exactly the
        trailing two characters of a Qiskit counts key for this circuit -- so
        ``label[0]`` is the ``X``-correction exponent (Alice's Bell-pair qubit)
        and ``label[1]`` is the ``Z``-correction exponent (Alice's payload
        qubit). Every circuit has 3 qubits and 3 classical bits; ``c0`` and
        ``c1`` are written by Alice's measurement and ``c2`` by Bob's.

    Raises:
        TypeError: If ``state`` is not an :class:`ArbitraryState`.
    """
    if not isinstance(state, ArbitraryState):
        raise TypeError(
            f"state must be an ArbitraryState, got {type(state).__name__}"
        )

    branches: dict[str, QuantumCircuit] = {}
    for c1 in (0, 1):
        for c0 in (0, 1):
            label = f"{c1}{c0}"
            qc = QuantumCircuit(3, TELEPORTATION_CLBITS, name=f"teleport-branch-{label}")

            # Identical preparation and Bell measurement for every branch: the
            # branches differ only in which correction Bob applies.
            qc.ry(2.0 * float(state.theta), 0)
            qc.rz(float(state.phi), 0)
            qc.h(1)
            qc.cx(1, 2)
            qc.cx(0, 1)
            qc.h(0)
            qc.measure(0, 0)
            qc.measure(1, 1)

            # The branch's correction, applied blindly: Bob assumes the outcome
            # he was told, which is only correct on the matching shots.
            if c1:
                qc.x(2)
            if c0:
                qc.z(2)
            qc.measure(2, 2)
            branches[label] = qc
    return branches


def branch_marginal(
    counts: dict[str, int],
    branch: str,
    *,
    bit_index: int = 0,
    num_bits: int = 3,
) -> dict[str, int]:
    """Post-select a single Alice branch out of a counts dictionary.

    ``[SIMULATED]`` Pure post-processing of Aer shot counts; no circuit runs.

    The branch label is matched against the two most significant classical bits
    of every counts key, i.e. against ``key[-2:]``, which for the
    teleportation circuit is ``c1c0``. Only shots whose Alice outcome equals
    ``branch`` survive, so roughly ``num_shots / 4`` remain -- the discarded
    shots are the cost of forcing a deterministic branch.

    Args:
        counts: Shot counts keyed by ``"c2c1c0"``-style bitstrings, as returned
            by :func:`run_on_aer`.
        branch: Two-character branch label such as ``"01"``, in ``c1c0`` order.
        bit_index: Index of the classical bit whose marginal is wanted, counted
            from the least significant end, exactly as in
            :func:`marginal_counts`. Bob's readout is ``c2``, so pass
            ``bit_index=2`` for the branch circuits built here; the default of
            ``0`` addresses Alice's payload-qubit bit ``c0``, which is constant
            within a selected branch and therefore uninformative.
        num_bits: Width of the classical register, used to validate the keys.

    Returns:
        ``{"0": count, "1": count}`` over the retained shots only. Both values
        are ``0`` if no shot matched the branch.

    Raises:
        ValueError: If ``branch`` is not exactly two ``"0"``/``"1"``
            characters, if ``bit_index`` is outside ``[0, num_bits)``, or if a
            counts key does not have ``num_bits`` characters.
    """
    if len(branch) != 2 or set(branch) - {"0", "1"}:
        raise ValueError(f"branch must be a two-character c1c0 label, got {branch!r}")
    if not 0 <= bit_index < num_bits:
        raise ValueError(f"bit_index must lie in [0, {num_bits}), got {bit_index}")

    selected = {"0": 0, "1": 0}
    for raw_key, value in counts.items():
        key = raw_key.replace(" ", "")
        if len(key) != num_bits:
            raise ValueError(
                f"counts key {raw_key!r} has {len(key)} bits, expected {num_bits}"
            )
        if key[-2:] != branch:
            continue
        bit = key[-1 - bit_index]
        if bit not in selected:
            raise ValueError(
                f"bit_index {bit_index} does not address a binary bit in {raw_key!r}"
            )
        selected[bit] += value
    return selected


def run_on_aer(
    circuit: QuantumCircuit,
    *,
    shots: int = 4096,
    seed: int = 20260930,
    noise_model: NoiseModel | None = None,
) -> dict[str, int]:
    """Transpile a circuit and execute it on :class:`~qiskit_aer.AerSimulator`.

    ``[SIMULATED]`` This is a classical density-matrix/state-vector simulation.
    Nothing here touches hardware.

    Transpilation is pinned with ``optimization_level=0`` for two reasons. First,
    it keeps the gate names (``h``, ``cx``, ``ry``, ``rz``, ...) intact so that
    :func:`build_noise_model` still decorates the intended instructions after the
    backend basis mapping. Second, it makes the mapping deterministic given
    ``seed``, which is what lets :func:`teleport_link_with_noise` be
    reproducible shot-for-shot.

    Args:
        circuit: The circuit to run. Mid-circuit measurement and ``if_test``
            control flow are supported by Aer.
        shots: Number of measurement shots. Must be a positive integer.
        seed: Seed for both transpilation and the simulator, giving
            run-to-run reproducibility for a fixed circuit.
        noise_model: Optional Aer noise model; when ``None`` the run is ideal.

    Returns:
        A dictionary mapping ``"c2c1c0"``-style bitstrings (Qiskit's order: the
        leftmost character is the most significant classical bit) to shot
        counts. Whitespace separators in Aer's keys are removed.

    Raises:
        TypeError: If ``circuit`` is not a :class:`~qiskit.circuit.QuantumCircuit`.
        ValueError: If ``shots`` is not a positive integer.
        RuntimeError: If transpilation or execution fails, or if the job reports
            an unsuccessful status. The underlying Qiskit error is chained on.
    """
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(
            f"circuit must be a QuantumCircuit, got {type(circuit).__name__}"
        )
    if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)):
        raise TypeError(f"shots must be an integer, got {type(shots).__name__}")
    if shots <= 0:
        raise ValueError(f"shots must be positive, got {shots}")

    backend = AerSimulator(noise_model=noise_model)
    try:
        transpiled = transpile(
            circuit,
            backend,
            optimization_level=0,
            seed_transpiler=int(seed),
        )
    except QiskitError as exc:
        raise RuntimeError(
            f"transpilation for the AerSimulator backend failed: {exc}"
        ) from exc

    try:
        result = backend.run(transpiled, shots=int(shots), seed_simulator=int(seed)).result()
    except QiskitError as exc:
        raise RuntimeError(f"the AerSimulator job failed: {exc}") from exc

    if not result.success:
        raise RuntimeError(
            f"the AerSimulator job did not succeed (status={result.status})"
        )

    raw_counts = result.get_counts()
    if not isinstance(raw_counts, dict):
        raise RuntimeError(
            f"expected a counts mapping from the AerSimulator, got "
            f"{type(raw_counts).__name__}"
        )
    return {str(key).replace(" ", ""): int(value) for key, value in raw_counts.items()}


def marginal_counts(
    counts: dict[str, int], bit_index: int = 0, num_bits: int = 3
) -> dict[str, int]:
    """Marginalise a full counts dictionary down to a single classical bit.

    ``[SIMULATED]`` Post-processing only.

    Marginalising is *not* the same as conditioning. This function sums shot
    counts over all values of the other classical bits, so Alice's outcome
    information is discarded. Use :func:`branch_marginal` when the Alice bits
    need to be kept.

    Args:
        counts: Shot counts keyed by bitstrings of width ``num_bits``.
        bit_index: Index of the bit to keep, counted from the least significant
            end, so ``0`` is the rightmost character and ``num_bits - 1`` is the
            leftmost. For the teleportation circuit, Bob's readout ``c2`` is
            ``bit_index=2``.
        num_bits: Width of the classical register, used to validate the keys.

    Returns:
        A dictionary keyed by the single bit character (``"0"`` or ``"1"``) with
        the summed shot counts.

    Raises:
        ValueError: If ``bit_index`` is outside ``[0, num_bits)``, if ``num_bits``
            is not positive, or if a counts key does not have ``num_bits``
            characters.
    """
    if num_bits < 1:
        raise ValueError(f"num_bits must be positive, got {num_bits}")
    if not 0 <= bit_index < num_bits:
        raise ValueError(f"bit_index must lie in [0, {num_bits}), got {bit_index}")

    marginal: dict[str, int] = {}
    for raw_key, value in counts.items():
        key = str(raw_key).replace(" ", "")
        if len(key) != num_bits:
            raise ValueError(
                f"counts key {raw_key!r} has {len(key)} bits, expected {num_bits}"
            )
        bit = key[-1 - bit_index]
        marginal[bit] = marginal.get(bit, 0) + int(value)
    return marginal


def fidelity_metrics(
    observed: dict[str, float], ideal: dict[str, float]
) -> dict[str, float]:
    """Compare two discrete distributions with three standard metrics.

    ``[SIMULATED]`` Arithmetic on already-sampled distributions.

    Both inputs are renormalised to sum to one before comparison, so callers may
    pass raw counts or raw probabilities. Keys are unioned, so disjoint inputs
    are handled correctly rather than raising.

    The three returned quantities are:

    ``bhattacharyya``
        :math:`(\\sum_i \\sqrt{p_i q_i})^2`, the classical fidelity between two
        discrete distributions. It equals 1 exactly when the distributions
        coincide. (Convention note: many texts call the *un*-squared sum
        :math:`\\sum_i\\sqrt{p_i q_i}` the "fidelity" or "Bhattacharyya
        coefficient"; this module uses the squared form consistently, so that a
        perfect match is 1 and a fully disjoint pair is 0.)
    ``total_variation``
        :math:`\\tfrac12\\sum_i |p_i - q_i|`, the largest achievable advantage of
        a binary hypothesis test over guessing. Ranges from 0 to 1.
    ``hellinger``
        :math:`\\sqrt{1 - \\sum_i \\sqrt{p_i q_i}}`, the Hellinger distance. It
        is a true metric on the simplex, which the squared Bhattacharyya
        coefficient is not, so it is reported alongside for that reason.

    Args:
        observed: Measured distribution, e.g. ``{"0": 0.25, "1": 0.75}``.
        ideal: Reference distribution to compare against.

    Returns:
        ``{"bhattacharyya": ..., "total_variation": ..., "hellinger": ...}``.

    Raises:
        TypeError: If either argument is not a mapping of real numbers.
        ValueError: If a value is negative or non-finite, or if a distribution
            has zero total weight and therefore cannot be normalised.
    """
    for name, distribution in (("observed", observed), ("ideal", ideal)):
        if not isinstance(distribution, dict):
            raise TypeError(
                f"{name} must be a dict of str -> float, got "
                f"{type(distribution).__name__}"
            )

    def _normalise(distribution: dict[str, float], name: str) -> dict[str, float]:
        cleaned: dict[str, float] = {}
        for key, value in distribution.items():
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(
                    f"{name}[{key!r}] must be a real number, got "
                    f"{type(value).__name__}"
                )
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{name}[{key!r}] must be finite, got {value!r}")
            if number < 0.0:
                raise ValueError(
                    f"{name}[{key!r}] must be non-negative, got {value!r}"
                )
            cleaned[str(key)] = number
        total = sum(cleaned.values())
        if total <= 0.0:
            raise ValueError(f"{name} has zero total weight and cannot be normalised")
        return {key: value / total for key, value in cleaned.items()}

    observed_norm = _normalise(observed, "observed")
    ideal_norm = _normalise(ideal, "ideal")

    keys = sorted(set(observed_norm) | set(ideal_norm))
    overlap = sum(
        math.sqrt(observed_norm.get(key, 0.0)) * math.sqrt(ideal_norm.get(key, 0.0))
        for key in keys
    )
    # Clamp away floating-point dust: a distribution compared against itself must
    # score exactly 1 even though sqrt(p) * sqrt(p) does not round-trip exactly.
    overlap = min(max(overlap, 0.0), 1.0)
    if overlap > 1.0 - 1e-12:
        overlap = 1.0
    total_variation = 0.5 * sum(
        abs(observed_norm.get(key, 0.0) - ideal_norm.get(key, 0.0)) for key in keys
    )
    return {
        "bhattacharyya": overlap * overlap,
        "total_variation": total_variation,
        "hellinger": math.sqrt(max(0.0, 1.0 - overlap)),
    }


def build_noise_model(strength: float, *, readout_error: float | None = None) -> NoiseModel:
    """Build an Aer noise model of depolarizing gate noise plus readout error.

    ``[PROPOSED]`` The gate sets and the interpretation of ``strength`` are a
    modelling choice for this prototype, not a calibrated device
    characterisation. A real apparatus would need gate-specific infidelities and
    a measured readout confusion matrix.

    ``strength`` is used directly as the depolarizing parameter
    :math:`\\lambda`: the channel replaces an ideal gate ``G`` by
    ``(1 - 3*lambda/4) G + (lambda/4) X G X + (lambda/4) Y G Y + (lambda/4) Z G Z``
    for one qubit, and by the analogous uniform mixture over the 15 two-qubit
    Pauli products for two qubits. The validated range ``[0, 1]`` sits inside
    both the one-qubit limit ``4/3`` and the two-qubit limit ``16/15`` required by
    :class:`~qiskit_aer.noise.depolarizing_error`, so the model is always
    physical.

    Args:
        strength: Depolarizing strength in ``[0, 1]``. ``0`` adds no gate noise
            at all (an empty error is skipped so that the model stays exactly
            ideal), ``1`` is the maximally mixed single-qubit channel.
        readout_error: Optional symmetric readout confusion probability in
            ``[0, 0.5]``, applied identically to every qubit: the observed bit
            is flipped with this probability. ``None`` means perfect readout.

    Returns:
        A configured :class:`~qiskit_aer.noise.NoiseModel`.

    Raises:
        TypeError: If ``strength`` or ``readout_error`` is not a real number.
        ValueError: If ``strength`` is outside ``[0, 1]`` or ``readout_error``
            is outside ``[0, 0.5]``.
        QiskitError: If Aer rejects the requested error channels.
    """
    if isinstance(strength, bool) or not isinstance(
        strength, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"strength must be a real number, got {type(strength).__name__}")
    strength_value = float(strength)
    if not math.isfinite(strength_value):
        raise ValueError(f"strength must be finite, got {strength!r}")
    if not 0.0 <= strength_value <= 1.0:
        raise ValueError(f"strength must lie in [0, 1], got {strength!r}")

    readout_value: float | None = None
    if readout_error is not None:
        if isinstance(readout_error, bool) or not isinstance(
            readout_error, (int, float, np.integer, np.floating)
        ):
            raise TypeError(
                f"readout_error must be a real number or None, got "
                f"{type(readout_error).__name__}"
            )
        readout_value = float(readout_error)
        if not math.isfinite(readout_value):
            raise ValueError(f"readout_error must be finite, got {readout_error!r}")
        if not 0.0 <= readout_value <= 0.5:
            raise ValueError(
                f"readout_error must lie in [0, 0.5], got {readout_error!r}"
            )

    noise = NoiseModel()
    if strength_value > 0.0:
        noise.add_all_qubit_quantum_error(
            depolarizing_error(strength_value, 1), list(_ONE_QUBIT_NOISY_GATES)
        )
        noise.add_all_qubit_quantum_error(
            depolarizing_error(strength_value, 2), list(_TWO_QUBIT_NOISY_GATES)
        )
    if readout_value is not None and readout_value > 0.0:
        matrix = [
            [1.0 - readout_value, readout_value],
            [readout_value, 1.0 - readout_value],
        ]
        noise.add_all_qubit_readout_error(ReadoutError(matrix))
    return noise


def bidirectional_teleportation_circuit(
    state_a_to_b: ArbitraryState,
    state_b_to_a: ArbitraryState,
    *,
    dynamic: bool = True,
) -> QuantumCircuit:
    """Build a six-qubit model of two opposing teleportation channels.

    ``[PROPOSED]`` and ``[SIMULATED]``. Read this docstring before quoting any
    result that comes out of it.

    **What this is not.** A Bell pair is unidirectional. Two nodes cannot send
    quantum states to each other through a *single* entangled pair, because
    teleportation consumes the entanglement and leaves Alice with no copy to
    teleport from. Genuine simultaneous bidirectional teleportation therefore
    requires **two independent entangled pairs and two independent classical
    channels**. This function builds exactly that, but as one composite
    six-qubit state executed by Aer: it is a simulation-level *model* used to
    exercise the routing layer, and running it as a single composite circuit on
    Aer does **not** demonstrate physical simultaneous bidirectional
    teleportation. The composite simulation hides the resource accounting that
    a real two-way link must pay for, and it contains no decoherence, no
    repeater buffering and no finite propagation delay.

    Layout::

        q0 = A's payload                  q3 = B's payload
        q1 = A's half of shared pair AB   q4 = B's half of shared pair BA
        q2 = B's receiver                 q5 = A's receiver

        creg "a2b" (c0, c1, c2): A -> B classical channel
        creg "b2a" (c3, c4, c5): B -> A classical channel

    The two channels share no qubits and no classical bits, so they are
    independent by construction; interleaving the gates is only a layout
    convenience.

    Args:
        state_a_to_b: The state Alice teleports to Bob.
        state_b_to_a: The state Bob teleports to Alice.
        dynamic: If ``True`` (default) each channel uses ``if_test`` feed-forward
            blocks. If ``False`` both corrections are applied unconditionally in
            both channels, which is the no-classical-communication control
            experiment: the recovered state is left with a random Pauli error
            instead of being corrected. This lowers the recovered fidelity in
            both directions, but it does not drive it to zero. An unconditional
            ``X @ Z`` is a rotation of the Bloch sphere, so a payload that
            happens to lie near the pole that ``X @ Z`` maps onto itself
            survives with high apparent fidelity. In simulation the two
            directions here typically land in the 0.92-0.99 range rather than
            at 0.5, so this variant demonstrates that feed-forward *matters*,
            not that the transfer is annihilated.

    Returns:
        A :class:`~qiskit.circuit.QuantumCircuit` with exactly 6 qubits,
        6 classical bits arranged as two 3-bit registers, and name
        ``"teleport-bidirectional"``.

    Raises:
        TypeError: If either argument is not an :class:`ArbitraryState`.
    """
    for name, state in (("state_a_to_b", state_a_to_b), ("state_b_to_a", state_b_to_a)):
        if not isinstance(state, ArbitraryState):
            raise TypeError(
                f"{name} must be an ArbitraryState, got {type(state).__name__}"
            )

    # Both channels are appended directly to the six-qubit circuit rather than
    # being built as standalone sub-circuits and merged with `compose`.
    # `compose` remaps qubits and clbits for the *data* operations, but the
    # condition captured by an `if_test` block is the Clbit object created in the
    # sub-circuit's own register, which is never re-bound. The resulting
    # `IfElseOp` then references a clbit that is not a member of the outer
    # circuit, which raises CircuitError on draw() and is fragile elsewhere.
    # Building in place guarantees every condition refers to a real bit.
    qubits = QuantumRegister(6, "q")
    a2b_bits = ClassicalRegister(3, "a2b")
    b2a_bits = ClassicalRegister(3, "b2a")
    combined = QuantumCircuit(qubits, a2b_bits, b2a_bits, name="teleport-bidirectional")

    def _append_channel(
        circuit: QuantumCircuit,
        state: ArbitraryState,
        payload: int,
        bell: int,
        receiver: int,
        outcome_a: Any,
        outcome_b: Any,
        result: Any,
    ) -> None:
        """Append one complete teleportation channel to ``circuit``.

        Args:
            circuit: The six-qubit target circuit, modified in place.
            state: The single-qubit state to teleport.
            payload: Index of the sender's payload qubit.
            bell: Index of the sender's half of the shared entangled pair.
            receiver: Index of the receiver's qubit.
            outcome_a: Clbit carrying the Bell-measurement outcome of ``payload``.
            outcome_b: Clbit carrying the Bell-measurement outcome of ``bell``.
            result: Clbit that receives the receiver's readout.
        """
        # Prepare the arbitrary single-qubit state |psi> = cos(theta)|0> + e^{i phi} sin(theta)|1>.
        circuit.ry(2.0 * float(state.theta), payload)
        circuit.rz(float(state.phi), payload)
        # Entangle the shared pair between sender and receiver.
        circuit.h(bell)
        circuit.cx(bell, receiver)
        # Sender's Bell-basis measurement: CNOT then Hadamard.
        circuit.cx(payload, bell)
        circuit.h(payload)
        circuit.measure(payload, outcome_a)
        circuit.measure(bell, outcome_b)
        # Classical feed-forward corrections on the receiver.
        if dynamic:
            with circuit.if_test((outcome_b, 1)):
                circuit.x(receiver)
            with circuit.if_test((outcome_a, 1)):
                circuit.z(receiver)
        else:
            circuit.x(receiver)
            circuit.z(receiver)
        circuit.measure(receiver, result)

    _append_channel(
        combined,
        state_a_to_b,
        payload=0,
        bell=1,
        receiver=2,
        outcome_a=a2b_bits[0],
        outcome_b=a2b_bits[1],
        result=a2b_bits[2],
    )
    _append_channel(
        combined,
        state_b_to_a,
        payload=3,
        bell=4,
        receiver=5,
        outcome_a=b2a_bits[0],
        outcome_b=b2a_bits[1],
        result=b2a_bits[2],
    )
    return combined


def bidirectional_marginals(
    counts: dict[str, int],
) -> tuple[dict[str, int], dict[str, int]]:
    """Split six-bit bidirectional counts into the two recovered-state marginals.

    ``[SIMULATED]`` Post-processing only.

    With the register layout of
    :func:`bidirectional_teleportation_circuit`, a counts key reads
    ``"c5c4c3c2c1c0"``. ``c2`` is Bob's readout of the state Alice sent, and
    ``c5`` is Alice's readout of the state Bob sent. The two marginals are
    statistically independent -- no classical information is shared between the
    channels -- so their joint correlation carries no extra structure.

    Args:
        counts: Shot counts keyed by six-character bitstrings.

    Returns:
        ``(a_recovered, b_recovered)`` where ``a_recovered`` is Alice's readout
        of Bob's payload (bit ``c5``) and ``b_recovered`` is Bob's readout of
        Alice's payload (bit ``c2``). Each is a ``{"0": ..., "1": ...}`` mapping.

    Raises:
        ValueError: If ``counts`` is empty or a key is not six characters wide.
    """
    if not counts:
        raise ValueError("counts is empty; cannot compute bidirectional marginals")
    num_bits = len(str(next(iter(counts))).replace(" ", ""))
    if num_bits != BIDIRECTIONAL_CLBITS:
        raise ValueError(
            f"expected {BIDIRECTIONAL_CLBITS}-bit counts keys, got width {num_bits}"
        )
    a_recovered = marginal_counts(counts, bit_index=5, num_bits=num_bits)
    b_recovered = marginal_counts(counts, bit_index=2, num_bits=num_bits)
    return a_recovered, b_recovered


def bell_pair_entanglement_demo() -> tuple[QuantumCircuit, np.ndarray]:
    """Prepare two independent Bell pairs and return the exact statevector.

    ``[SIMULATED]`` The statevector is exact -- it is a
    :math:`16`-amplitude complex vector produced by the simulator, not a
    statistical estimate -- but it is still a *classical* simulation.

    No mid-circuit measurement is used, so appending
    :class:`~qiskit_aer.library.SaveStatevector` is safe here (unlike in the
    teleportation circuit, where the collapse from measurement would make a
    statevector meaningless).

    The resulting state is
    :math:`(|00\\rangle + |11\\rangle)\\otimes(|00\\rangle + |11\\rangle)/2`,
    so in Qiskit's little-endian convention the only non-zero amplitudes sit at
    indices ``0`` (``|0000>``), ``3`` (``|0011>``), ``12`` (``|1100>``) and
    ``15`` (``|1111>``), each of magnitude ``1/2``. Computational-basis
    measurement therefore returns one of those four strings with probability
    ``1/4`` each: the outcomes ``0011`` and ``1100`` are the visible evidence of
    the second pair's correlation, and all four outcomes agree qubit-by-qubit,
    which is the visible evidence of the first pair's.

    Returns:
        ``(circuit, statevector)``. ``circuit`` is the four-qubit circuit with
        the statevector-saving instruction already appended, so it can be
        re-executed or drawn as-is; ``statevector`` is a
        ``(16,)`` ``complex128`` array of unit norm.

    Raises:
        RuntimeError: If the Aer job fails to execute the circuit.
    """
    circuit = QuantumCircuit(4, name="two-bell-pairs")
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.h(2)
    circuit.cx(2, 3)
    circuit.append(SaveStatevector(4), circuit.qubits)

    backend = AerSimulator()
    try:
        transpiled = transpile(
            circuit, backend, optimization_level=0, seed_transpiler=20260930
        )
        result = backend.run(transpiled, shots=1, seed_simulator=20260930).result()
    except QiskitError as exc:
        raise RuntimeError(f"the two-Bell-pair demo failed to execute: {exc}") from exc
    if not result.success:
        raise RuntimeError(
            f"the two-Bell-pair demo did not succeed (status={result.status})"
        )

    statevector = np.asarray(result.get_statevector(transpiled), dtype=np.complex128)
    if statevector.shape != (16,):
        raise RuntimeError(
            f"expected a 16-element statevector, got shape {statevector.shape}"
        )
    return circuit, statevector


def teleport_link_with_noise(
    state: ArbitraryState,
    *,
    noise_score: float,
    fidelity_hint: float,
    shots: int,
    seed: int,
) -> dict[str, float]:
    """Simulate one noisy teleportation hop from this project's link model.

    This is the bridge between the abstract routing graph in
    :mod:`src.network` and the quantum simulator, and it is the function a
    multi-hop simulator calls once per selected link.

    **The noise mapping below is ``[PROPOSED]``: an illustrative mapping
    invented for this prototype, not a calibrated device model.** Concretely it
    says nothing about any real apparatus:

    * ``noise_score -> p_depol = clip(noise_score, 0, 1) * 0.05`` is capped at a
      5% per-gate depolarizing probability. The cap keeps the worst link
      distinguishable from the best one rather than destroying the transfer
      outright, so that routing comparisons stay legible.
    * ``fidelity_hint -> p_read = clip((1 - fidelity_hint) * 0.25, 0, 0.5)`` maps
      a declared link fidelity onto a symmetric readout confusion. A link
      fidelity of 0.8 gives a 5% readout error.

    The two inputs are treated as independent knobs even though a real link
    would link them; that independence is the same modelling convenience used
    throughout :mod:`src.network`.

    Args:
        state: The state being teleported over this hop.
        noise_score: Abstract per-link ``noise`` attribute in ``[0, 1]``.
        fidelity_hint: Abstract per-link ``fidelity`` attribute in ``[0, 1]``,
            used *only* to set the readout error; it is not compared against the
            measured result.
        shots: Number of measurement shots for this hop.
        seed: Seed for transpilation and the simulator.

    Returns:
        A mapping with:

        ``measured_fidelity``
            ``bhattacharyya`` from :func:`fidelity_metrics`, comparing Bob's
            normalised marginal against ``state.ideal_probabilities()``.
            Because the protocol already contains a measurement, this number is
            bounded above by roughly the classical-quantum state-discrimination
            limit of a single binary measurement, not by 1 in general; it
            approaches 1 when the state is near a computational-basis eigenstate.
        ``noise_score``, ``fidelity_hint``, ``shots``
            Echoed back unchanged. ``shots`` is carried as an exact integer in
            the same mapping, per the documented return contract.
        ``bit_flips``
            Number of retained shots whose recovered bit disagrees with the
            *majority* ideal outcome, i.e. ``shots - counts[majority]``. This is
            a disagreement count, not an error rate against a known per-shot
            truth, which is unavailable because the protocol destroys the input.

    Raises:
        TypeError: If ``state`` is not an :class:`ArbitraryState`.
        ValueError: If ``shots`` is not positive, or if ``noise_score`` or
            ``fidelity_hint`` is not a finite real number.
        RuntimeError: If the Aer job fails.
    """
    if not isinstance(state, ArbitraryState):
        raise TypeError(
            f"state must be an ArbitraryState, got {type(state).__name__}"
        )
    for name, value in (("noise_score", noise_score), ("fidelity_hint", fidelity_hint)):
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)):
        raise TypeError(f"shots must be an integer, got {type(shots).__name__}")
    if shots <= 0:
        raise ValueError(f"shots must be positive, got {shots}")

    p_depol = min(max(float(noise_score), 0.0), 1.0) * 0.05
    p_read = min(max((1.0 - float(fidelity_hint)) * 0.25, 0.0), 0.5)
    noise_model = build_noise_model(p_depol, readout_error=p_read)

    counts = run_on_aer(
        build_teleportation_circuit(state, dynamic=True),
        shots=int(shots),
        seed=int(seed),
        noise_model=noise_model,
    )
    bob = marginal_counts(counts, bit_index=2, num_bits=TELEPORTATION_CLBITS)
    total = sum(bob.values())
    observed = {key: value / total for key, value in bob.items()}

    ideal = state.ideal_probabilities()
    majority = max(ideal, key=lambda key: (ideal[key], key))
    bit_flips = int(shots) - int(bob.get(majority, 0))

    metrics = fidelity_metrics(observed, ideal)
    return {
        "measured_fidelity": metrics["bhattacharyya"],
        "noise_score": float(noise_score),
        "fidelity_hint": float(fidelity_hint),
        "shots": int(shots),
        "bit_flips": bit_flips,
    }


def environment_report() -> str:
    """Return a human-readable description of the software stack in use.

    ``[SIMULATED]`` The last line of the returned string is an explicit
    statement that every quantum result in this project is simulated.

    Returns:
        A multi-line string naming the Python, Qiskit and Qiskit-Aer versions
        and the host platform, ending with the simulation disclaimer.
    """
    lines = [
        "BQT-MDQW quantum teleportation environment",
        f"python              : {platform.python_version()}",
        f"qiskit              : {qiskit.__version__}",
        f"qiskit-aer          : {qiskit_aer.__version__}",
        f"numpy               : {np.__version__}",
        f"platform            : {platform.platform()}",
        f"machine             : {platform.machine()}",
        "results             : SIMULATED -- every quantum result produced by this",
        "                      project comes from Qiskit Aer's classical simulator.",
        "                      No result was measured on quantum hardware, and no",
        "                      link metric in this project is calibrated against a",
        "                      physical device.",
    ]
    return "\n".join(lines)
