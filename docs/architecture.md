# Architecture

**BQT-MDQW — Layered architecture (classical planning + simulated quantum execution)**

This document describes how the seven stages shown in [`architecture.png`](architecture.png) map to the codebase. Every claim is tagged `[ESTABLISHED]`, `[PROPOSED]` or `[SIMULATED]` as defined in the [README](../README.md).

## 1. Overview

BQT-MDQW separates **classical planning** (stages 1–5 in the pipeline) from **simulated quantum execution** (stages 6–7). The quantum-walk component is used only to produce a heuristic node prior; it does not compute routes on a quantum device. All quantum operations are executed by Qiskit Aer or by NumPy dense matrix-vector products `[SIMULATED]`. The router is a classical graph algorithm (A* / uniform-cost) with Dijkstra as a baseline `[ESTABLISHED]`.

The seven stages are:

1. Quantum Network
2. Network Graph
3. Edge Metrics
4. MDQW-Inspired Routing (classical)
5. Selected Multi-Hop Path
6. Quantum Teleportation (simulated)
7. Receiver (simulated)

---

## 2. Stage 1 — Quantum Network `[PROPOSED]` (data model)

**What it represents:** the abstract network of quantum nodes and quantum links, along with the associated metadata (node roles, whether a node is a repeater, and per-node parameters).

**Implementation:** `src/network.py` — `QuantumNetwork` class. It wraps a `networkx.Graph`, validates link insertion, supports adding midpoint nodes (`add_midpoint_node`), exports `edge_table()` and `metric_ranges()`, and provides `copy()`.

**Key properties:**
- Node set $V$ = endpoints and repeaters (arbitrary strings).
- Edge set $E$ = candidate quantum communication links.
- Per-node flags: `is_repeater` (boolean). Source and destination are **never** charged node penalties in the routing objective.
- Synthetic bookkeeping only: no physical devices, no wavelengths, no hardware IDs `[SIMULATED]`.

---

## 3. Stage 2 — Network Graph `[PROPOSED]`

**What it represents:** the weighted graph $G = (V,E)$ passed to the routing layer. This is the graph on which A*, uniform-cost or Dijkstra operate.

**Implementation:** `src/network.py`
- `build_example_network()` constructs the canonical 6-node topology (`A…F`) with the `A–B–C–F` branch and `A–E–F` branch, so routing is not degenerate by default.
- `random_network(n_nodes, p, *, seed, ...)` constructs a reproducible Erdős–Rényi graph with explicit integer seed. Connectivity is reported and the caller only samples connected pairs in the shipped experiments.

Graph is undirected. Self-loops are not created. Duplicates rejected on insertion.

---

## 4. Stage 3 — Edge Metrics `[PROPOSED]` (synthetic inputs)

**What they represent:** five per-link quantities used to form the multi-metric cost. **They are synthetic model inputs, not measured data, and are treated as mutually uncorrelated by convenience.** Real networks would couple many of them to distance.

**Fields (per link $e=(u,v)$):**
- `distance_km` $\in \mathbb{R}_{\ge 0}$ — geographic/propagation scale
- `noise` $\in [0,1]$ — abstract link noise parameter (mapped to Aer channels later)
- `latency_ms` $\in \mathbb{R}_{\ge 0}$ — control/round-trip latency scale
- `fidelity` $\in [0,1]$ — declared link fidelity $f_e$
- `reliability` $\in [0,1]$ — declared link reliability $r_e$

**Normalisation:**
$$
\hat{x}_e = \frac{x_e - \min_{e'\in E} x_{e'}}{\max_{e'\in E} x_{e'} - \min_{e'\in E} x_{e'}},\quad \hat{x}_e = 0 \text{ if } \max x=\min x
$$
A constant metric contributes nothing to cost `[PROPOSED]`.

**Fidelity notions (three, not interchangeable):**
- Product: $F_{\text{link}}(P)=\prod f_e$ `[PROPOSED]` (model input)
- Attenuation: $F_{\text{att}}(P)=\prod (f_e e^{-d_e/L})$, $L=50$ km `[PROPOSED]`
- Simulated: per-hop Aer-measured product over the route `[SIMULATED]`

---

## 5. Stage 4 — MDQW-Inspired Routing (classical) `[ESTABLISHED]+[PROPOSED]`

**Implementation:** `src/mdqw_routing.py`. **This is a classical graph search.** No quantum computer computes any route.

### 5.1 Combined link cost `[PROPOSED]`
$$
C(e)=\alpha\,\hat{d}_e + \beta\,\hat{n}_e + \gamma\,\hat{\ell}_e + \delta\,(1-\hat{f}_e) + \rho\,(1-\hat{r}_e), \quad \alpha,\beta,\gamma,\delta,\rho \ge 0,\; \text{not all }0
$$
Defaults: $\alpha=1.0,\beta=1.0,\gamma=0.5,\delta=1.5,\rho=0.25$.

### 5.2 Node term `[PROPOSED]`
$$
N(v)=\begin{cases}
\texttt{repeater\_penalty}, & v \notin\{s,t\},\; \texttt{is\_repeater}(v)\\
\texttt{hop\_penalty}, & v \notin\{s,t\},\; \lnot \texttt{is\_repeater}(v)
\end{cases} + \lambda_{qw}\,\mathrm{prior}(v)\,.
$$
Source and destination are exempt. Both penalties live on `RoutingWeights` (not on `MDQWConfig`): `hop_penalty` defaults to `0.0`, and `repeater_penalty` defaults to `None`, which `RoutingWeights.__post_init__` resolves to the value of `hop_penalty`. So by default a repeater and an ordinary intermediate node are charged the same. Only `src/simulation.py` sets them apart, via its own module constant `REPEATER_PENALTY_FACTOR = 4.0` in the Experiment 6 sweep.

### 5.3 Total cost
$$
\mathrm{total}(P)=\sum_{e\in P}C(e)+\sum_{v\in P\setminus\{s,t\}}N(v)=\mathrm{edge\_cost} + \mathrm{node\_cost}\,.
$$
This invariant is asserted by tests `[SIMULATED]`.

### 5.4 Search modes `[ESTABLISHED]`
- `mode="astar"`: A* with admissible heuristic $h(v)=\min_{u\in \delta(v)} C(u,v)$ (for $v\ne t$), $h(t)=0$. Admissible since all $N(v)\ge 0$.
- `mode="none"`: uniform-cost expansion (Dijkstra-style over the same objective).

**Baseline:** `DijkstraBaseline` uses the identical $C(e)$ but **no node terms and no heuristic** (optimises minimum edge-cost path). Because MDQW includes node terms, `total_cost` columns are **not comparable** across algorithms; the fair comparison is `edge_cost` `[PROPOSED]`.

### 5.5 Quantum-walk prior `[PROPOSED]` (heuristic only)
`src/quantum_walk.py` produces a node prior $\mathrm{prior}(v)\in[0,1]$, $\max_v \mathrm{prior}(v)=1$:
$$
\mathrm{raw}[v]=\frac{1}{T+1}\sum_{t=0}^{T} P_t(v)\,w(v),\quad w(v)=2\ (v\in\texttt{marked}),\ else\ 1;\qquad \mathrm{prior}[v]=\frac{\mathrm{raw}[v]}{\max_u \mathrm{raw}[u]}\,.
$$
When enabled, `is_cost_optimal = False` and the route is only optimal for the prior-influenced objective. The walk runs on a **regular walkable proxy graph** (spanning-2-factor fallback to complete graph if not regular), because the shunt decomposition $A=\sum_i P_i$ requires regularity `[ESTABLISHED]`. The proxy carries no physical links/metrics.

### 5.6 Constraints
- `max_hops` — if present, routes longer than this are excluded (constrained search). Can return no route.
- Walk prior is **off by default**.

---

## 6. Stage 5 — Selected Multi-Hop Path `[SIMULATED]`

`RouteResult` returned by the router contains:
- `path`: ordered list `s = v0, ..., vk = t`
- `edges`: list of link tuples in order
- `edge_cost`, `node_cost`, `total_cost` (with invariant)
- `hop_count = k`
- `distance_km`, `noise_sum`, `latency_ms` (route sums/aggregates)
- `fidelity_est` = $F_{\text{link}}(path)$
- `attenuation_fidelity_est` = $F_{\text{att}}(path)$
- `algorithm`, `is_cost_optimal`, `notes`

`RouteComparison`/`summarise_comparison` report win/loss/tie on `edge_cost` and path equality. `enumerate_routes` lists all simple paths (used in A* correctness tests and enumeration).

---

## 7. Stage 6 — Quantum Teleportation (simulated) `[ESTABLISHED]+[SIMULATED]`

**Implementation:** `src/quantum_teleportation.py`.

### 7.1 Bennett teleportation circuit `[ESTABLISHED]`
`build_teleportation_circuit(state, *, dynamic=True)` builds the Bennett et al. (1993) three-qubit, three-clbit circuit with qubits `[q0 (data), q1 (entangled-A), q2 (entangled-B)]` and clbits `[c0, c1, c2]`. Verified instruction order: `ry(q0)`, `rz(q0)` (payload prep, dynamic mode only), `h(q1)`, `cx(q1,q2)` (Bell pair), `cx(q0,q1)`, `h(q0)`, `measure q0 -> c0`, `measure q1 -> c1`, `if_test((c1,1)): x(q2)`, `if_test((c0,1)): z(q2)`, `measure q2 -> c2`.

The correction convention is the standard one: the **X** (bit-flip) correction is conditioned on classical bit `c1` and the **Z** (phase) correction on `c0`. The third classical bit `c2` records Bob's readout of the received qubit.

### 7.2 Execution `[SIMULATED]`
- `run_on_aer(circuit, shots, seed, optimization_level=0)` executes on `AerSimulator`. `optimization_level=0` preserves intended gate names for noise-model decoration.
- `teleport_link_with_noise(...)` maps abstract metrics to Aer channels:
  $$p_{\text{depol}}=\operatorname{clip}(n_e,0,1)\times 0.05,\qquad p_{\text{read}}=\operatorname{clip}\big((1-f_e)\times 0.25,\ 0,\ 0.5\big)\,.$$
  Depolarizing applied to 1q and 2q gates; symmetric readout confusion matrix applied. These are illustrative mapping choices `[PROPOSED]`.
- `simulate_route(network, route, *, state=None, shots=2048, seed=20260930, include_link_estimate=True)` lives in **`src/simulation.py`** (not in `quantum_teleportation.py`); it walks the consecutive node pairs of `route.path` and calls `teleport_link_with_noise` once per link, in order, seeding each hop with `seed + hop_index`. It returns a `MultiHopResult` (`path`, `hops`, `source`, `target`).

  A `HopResult` carries `hop_index`, `source`, `target`, `distance_km`, `noise_score`, `fidelity_hint`, `latency_ms`, `simulated_fidelity` (the Aer-measured value for that hop), `shots`, and `memory_lifetime_ms`. The memory figure is exposed as the **property** `memory_decay_estimate = exp(-latency_ms / memory_lifetime_ms)` — a closed-form surrogate, with **no actual memory decoherence simulated**. `MultiHopResult` likewise exposes the closed-form route-level properties `ideal_fidelity` (product of declared link fidelities) and `attenuation_fidelity` (product of `f_e * exp(-d_e/50)`).

### 7.3 Bidirectional teleportation model `[PROPOSED]+[SIMULATED]`
`bidirectional_teleportation_circuit(q0_payload, q1_payload, *, label)` builds a **six-qubit** circuit representing two independent teleportation channels: two data qubits, two Bell pairs, and six classical bits total. This requires **two independent Bell pairs and two independent classical channels** (not a single shared pair). The model executes both directions as one composite circuit. It does **not** demonstrate physical simultaneous bidirectional teleportation; it is a composite simulation of two independent unidirectional transfers.

---

## 8. Stage 7 — Receiver `[SIMULATED]`

Bob's readout distribution (marginalised over classical outcomes to the computational basis for the received qubit) is compared against the ideal payload distribution. Reported metrics: Bhattacharyya (squared coefficient), total variation distance, Hellinger distance. All are computed on renormalised probability vectors `[ESTABLISHED]`.

End-to-end: for a route, the per-hop simulated fidelities multiply conceptually in the sequential model; the reported “simulated end-to-end fidelity” in experiment summaries is the product of per-hop Bhattacharyya fidelities from the executed hop chain (subject to the measurement caveat discussed in README).

---

## 9. Data flow (end-to-end)

1. Build `QuantumNetwork` (stages 1–3).
2. Create `MDQWRouter` with `MDQWConfig` (weights, penalties, prior).
3. Call `router.route(s,t)` → returns `RouteResult` (stages 4–5).
4. Define payload `ArbitraryState`.
5. Call `simulate_route(network, route, state=...)` → returns a `MultiHopResult` with a `HopResult` per hop plus aggregated metrics (stage 6).
6. Optionally run bidirectional model (stage 6 composite) and collect receiver metrics (stage 7).
7. `src/simulation.py` renders `architecture.png` from this pipeline and writes all experiment outputs to `results/`.

The diagram in `docs/architecture.png` is generated by `simulation.render_architecture_diagram()` and is byte-identical across runs.

---

## 10. Non-quantum components (explicit)

- Graph representation, min–max normalisation, cost arithmetic: pure NumPy/Python.
- Search (A*, uniform-cost, Dijkstra): classical priority queue over graph edges/nodes.
- Walk prior: computed classically from dense matrix evolution (NumPy) or from Aer circuit counts (classical post-processing). The circuit is simulated classically.

**No part of the routing decision is executed on quantum hardware.** This is by design and is stated prominently throughout.

---

## 11. Notes on reproducibility

- Explicit integer seeds for random networks and for Aer simulation.
- `PYTHONHASHSEED` does not affect the shunt decomposition (double cover uses integer keys). This is regression-tested.
- `optimization_level=0` fixed for teleportation circuits.
- Results/figures in `results/` are committed deliverables.
- Architecture diagram is generated deterministically.