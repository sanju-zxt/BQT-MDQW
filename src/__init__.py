"""BQT-MDQW: Bidirectional Quantum Teleportation routing research prototype.

Subpackages / modules
---------------------
``network``
    ``[PROPOSED]`` Quantum network graph model with five per-link metrics.
``mdqw_routing``
    ``[PROPOSED]`` MDQW-inspired routing plus the Dijkstra baseline.
``quantum_teleportation``
    ``[ESTABLISHED]`` teleportation circuits, ``[PROPOSED]`` bidirectional
    model, ``[SIMULATED]`` Aer execution.
``quantum_walk``
    ``[ESTABLISHED]`` discrete-time coined quantum walk, ``[PROPOSED]``
    walk-derived node prior.
``simulation``
    ``[SIMULATED]`` seeded experiments, figures and ``results/results.md``.

This package performs no experiments on quantum hardware. Every quantum
operation is executed on a classical simulator (Qiskit Aer or NumPy).
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "network",
    "mdqw_routing",
    "quantum_teleportation",
    "quantum_walk",
    "simulation",
]
