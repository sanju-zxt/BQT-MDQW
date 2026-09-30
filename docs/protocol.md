# Protocol

**BQT-MDQW — Staged, implementable protocol (classical simulation only)**

This document describes the concrete, step-by-step workflow implemented in `src/*.py`. Every assumption is tagged `[ESTABLISHED]`, `[PROPOSED]` or `[SIMULATED]`. The routing layer is classical; teleportation is the Bennett protocol executed by simulation. Nothing here demonstrates physical quantum communication, entanglement swapping, purification, memory decoherence while waiting, or simultaneous bidirectional teleportation.

## 1. Assumptions

| # | Assumption | Tag |
|---|---|---|
| A1 | The network is an undirected graph $G=(V,E)$ with finite $V,E$. | `[ESTABLISHED]` |
| A2 | Each link $e\in E$ carries $(d_e,n_e,\ell_e,f_e,r_e)$ with $d_e,\ell_e\ge0$, $n_e,f_e,r_e\in[0,1]$. These are **synthetic model inputs**, not physical measurements, and treated as mutually uncorrelated by convenience. | `[PROPOSED]` |
| A3 | Bennett teleportation (Bennett et al., 1993) is the primitive for transferring a single qubit between adjacent nodes that share a Bell pair. | `[ESTABLISHED]` |
| A4 | Discrete-time coined quantum walk on an $m$-regular graph uses shunt decomposition $A=\sum_{i=0}^{m-1} P_i$ (Wong). | `[ESTABLISHED]` |
| A5 | Routing is performed by a **classical** search (A* or uniform-cost) over $G$ with cost $C(e)+N(v)$; Dijkstra is the classical baseline. No quantum algorithm computes the route. | `[ESTABLISHED]` |
| A6 | Multi-hop transfer is **sequential hop-by-hop teleportation** along the selected path. **There is no entanglement swapping, no purification, no error correction, no quantum repeater memory scheduling, and no decoherence simulated while a state waits in a memory.** | `[PROPOSED]` |
| A7 | The walk-derived node prior is a **heuristic** computed classically from a regular walkable proxy graph; it may make the route non-optimal for MDQW's objective. | `[PROPOSED]` |
| A8 | Quantum operations are executed only via Qiskit Aer (`AerSimulator`) or dense NumPy matrix-vector products. All results are classical simulations. | `[SIMULATED]` |
| A9 | The bidirectional model uses **two independent Bell pairs and two independent classical channels** (one per direction). It executes both as one composite six-qubit circuit and does **not** demonstrate physical simultaneous bidirectional teleportation. | `[PROPOSED]` |
| A10 | The noise mapping $(n_e,f_e)\mapsto (p_{\text{depol}}, p_{\text{read}})$ is illustrative, not calibrated to any device. | `[PROPOSED]` |
| A11 | Memory decay is reported as $e^{-\ell_e/T_{\text{mem}}}$ only (a closed form); **no actual memory decoherence is simulated**. | `[PROPOSED]` |
| A12 | A constant metric normalises to 0.0 for all links (no contribution). Uniform noise across all links therefore does not change $C(e)$. | `[PROPOSED]` |
| A13 | `max_hops` is a hard constraint (may return no route). | `[PROPOSED]` |
| A14 | The classical register has three bits: `c[0]` and `c[1]` carry Alice's two Bell-measurement outcomes and drive the corrections `if c[1]==1: X(q[2])` and `if c[0]==1: Z(q[2])`; `c[2]` carries Bob's final readout of the received qubit. The correction logic is therefore the standard two-bit Bennett correction, with the third bit used only for readout. | `[SIMULATED]` |
| A15 | The $(mn)\times(mn)$ walk unitary is exponential in state size and demonstrative only. | `[SIMULATED]` |

## 2. Protocol stages (0–4 implemented)

The implementation follows stages 0–4 below. Stage 4 is explicitly **not** a quantum repeater network.

### Stage 0 — Network construction and validation `[PROPOSED]+[SIMULATED]`
1. Create `QuantumNetwork` (`src/network.py`).
2. Add nodes $V$ (endpoints and optional repeaters). Mark `is_repeater(v)\in\{0,1\}$.
3. Add links $e=(u,v)$ with `EdgeMetrics(d_e,n_e,\ell_e,f_e,r_e)`. Validate ranges and non-finiteness.
4. Build min–max ranges over $E$: $x_{\min}, x_{\max}$ for each of the five metrics.
5. (Optional) Call `add_midpoint_node` to subdivide a link (creates an intermediate node/edges).

Output: validated $G=(V,E)$ with metric ranges.

### Stage 1 — Normalise metrics and form cost function `[PROPOSED]`
1. For each $x\in\{d,n,\ell,f,r\}$, $\hat x_e = (x_e - x_{\min})/(x_{\max}-x_{\min})$ if $x_{\max}>x_{\min}$ else $0.0$.
2. $C(e)=\alpha\hat d_e + \beta\hat n_e + \gamma\hat\ell_e + \delta(1-\hat f_e) + \rho(1-\hat r_e)$ with weights from `RoutingWeights`.
3. $N(v)= (\texttt{repeater\_penalty}\text{ if }v\text{ repeater else }\texttt{hop\_penalty}) + \lambda_{qw}\,\mathrm{prior}(v)$ for $v\notin\{s,t\}$, else $0$.

Weights default to $(1.0,1.0,0.5,1.5,0.25)$; penalties default `hop_penalty=0.0`, `repeater_penalty=0.0`.

### Stage 2 — Build walkable proxy and (optional) quantum-walk prior `[ESTABLISHED]+[PROPOSED]+[SIMULATED]`
1. **Proxy selection** (`quantum_walk.walkable_proxy`, `src/quantum_walk.py`):
   - Let $n=|V|$. Need $m$-regular simple graph on same vertices for shunt decomposition.
   - Try: if already $m$-regular → use $G$ (or its adjacency view). Else attempt a spanning 2-factor decomposition to form 2-regular (union of cycles) on all vertices; else fall back to **complete graph $K_n$**. The strategy used is reported.
   - In shipped results the fallback is `complete-graph-fallback` for the example and all random networks.
2. **Coin family**: `hadamard` (orthonormal for odd/even $m$), `y`, or `grover` ($2|s\rangle\langle s|-I$).
3. **Initial state**: uniform over $m$ coin directions at $v_0$: $|\psi_0\rangle = |v_0\rangle \otimes \frac{1}{\sqrt{m}}\sum_{c}|c\rangle$.
4. **Evolution**: $|\psi_t\rangle = (S (C\otimes I_n))^t |\psi_0\rangle$, $t=0,\ldots,T$, via `walk_reference_numpy` (dense matrix–vector) or `walk_on_aer` (Qiskit circuit) — cross-checked by `max_distribution_error`.
5. **Marginals**: $P_t(v)=\sum_c |\langle c,v|\psi_t\rangle|^2$, $\sum_v P_t(v)=1$.
6. **Prior**: $\mathrm{raw}[v]=\frac{1}{T+1}\sum_{t=0}^{T} P_t(v)\,w(v)$, $w(v)=2$ if marked else 1; $\mathrm{prior}[v]=\mathrm{raw}[v]/\max_u \mathrm{raw}[u]\in[0,1]$. If disabled, $\mathrm{prior}(v)=0$ for all $v$.
7. **Provenance**: decomposition uses integer keys (double cover), so deterministic across `PYTHONHASHSEED` values.

> The proxy is not the physical network and carries no metrics. The prior is a heuristic only.

### Stage 3 — Route selection (classical search) `[ESTABLISHED]+[PROPOSED]`
1. **Config**: `MDQWConfig(weights, mode, use_qw_prior, qw_prior_weight, prior_steps, max_hops)`. The five link-cost weights and the two node penalties are on `RoutingWeights`; `prior_steps` (default `8`) is the walk step count; `mode` is `"astar"` or `"none"`; `use_qw_prior` defaults to `False` and `qw_prior_weight` to `0.0`.
2. **Prior injection**: the prior is *supplied to the constructor* as a plain mapping, not computed by the router:
   `MDQWRouter(network, config, prior)` where `prior` is a `Mapping[node, float] | None`. The router validates that it is non-empty, references only known nodes, and holds finite non-negative values, then uses it only when `config.use_qw_prior` is `True`. The walk itself is computed separately by `src/quantum_walk.py` (`walkable_proxy` + `quantum_walk_prior`) and passed in. The coin family and marked-node set are therefore arguments of the *walk* call, not of `MDQWConfig`.
3. **Search**:
   - `mode="astar"`: A* with $h(v)=\min_{u\in\delta(v)} C(u,v)$, $h(t)=0$. Admissible when $\lambda_{qw}\mathrm{prior}\ge0$ in effect for remaining, but with prior term the objective is the prior-influenced one; `is_cost_optimal` becomes `False` when prior enabled.
   - `mode="none"`: uniform-cost (Dijkstra over MDQW objective including $N(v)$).
4. **Baseline**: `DijkstraBaseline.route(s,t)` uses identical $C(e)$ and **no node terms, no heuristic** (minimises $\sum C(e)$).
5. **Enforce constraints**: if `max_hops` is set, reject paths with `hop_count > max_hops`. If no route exists, `RoutingError` is raised (`"no route found from <s> to <t>"`, with a `max_hops` note when the cap was the cause).
6. **Build `RouteResult`**:
   - path sequence, edge list in order
   - `edge_cost = \sum_{e\in P} C(e)`
   - `node_cost = \sum_{v\in P\setminus\{s,t\}} N(v)`
   - `total_cost = edge_cost + node_cost` (invariant)
   - aggregates: `distance_km=\sum d_e`, `noise_sum=\sum n_e`, `latency_ms=\sum \ell_e`
   - `fidelity_est = \prod_{e\in P} f_e$ (link-product)
   - `attenuation_fidelity_est = \prod_{e\in P} (f_e e^{-d_e/L})$, $L=50$ km
   - `algorithm = "MDQW A*" | "MDQW UC" | "Dijkstra"`, `is_cost_optimal` (bool), `notes` (e.g. prior enabled)

**Fair comparison rule:** compare MDQW vs Dijkstra on **`edge_cost`**, not on `total_cost`. MDQW and Dijkstra optimise different objectives.

### Stage 4 — Sequential teleportation along the path (simulated) `[ESTABLISHED]+[PROPOSED]+[SIMULATED]`
This is **naive sequential hop-by-hop teleportation**. **No entanglement swapping, no purification, no error correction, no repeater memories, no decoherence while waiting.**

1. **Payload**: `ArbitraryState(theta, phi)` defines a single-qubit pure state $|\psi\rangle = R_z(\phi)\,R_y(\theta)\,|0\rangle$ — two parameters, validated and normalised.
2. **Per adjacent hop $(u,v)$ with metrics $(d,n,\ell,f,r)$**:
   - Assume a Bell pair $(q_{E1},q_{E2})$ is shared between $u$ (Alice side of this hop) and $v$ (Bob side) at the time of transfer (resource bookkeeping not simulated except via the composite model).
   - Build `QuantumCircuit(3, 3)` with qubits `[q0, q1, q2] = [data, entangled-A, entangled-B]` and clbits `[c0, c1, c2]`. Verified instruction order of `build_teleportation_circuit`:
     1. `ry(q0)`, `rz(q0)` — prepare the payload (dynamic mode only; the static variant omits these)
     2. `h(q1)`, `cx(q1, q2)` — prepare the shared Bell pair
     3. `cx(q0, q1)`, `h(q0)` — Alice's Bell-basis measurement basis change
     4. `measure q0 -> c0`, `measure q1 -> c1`
     5. `if_test((c1, 1)): x(q2)` — bit-flip correction
     6. `if_test((c0, 1)): z(q2)` — phase correction
     7. `measure q2 -> c2` — Bob's readout of the received qubit
   - So the correction convention is exactly the standard Bennett one: the **X** correction is conditioned on classical bit 1 and the **Z** correction on classical bit 0, and `c2` is used solely to record Bob's result.
   - **Noise injection** (`teleport_link_with_noise`):
     - $p_d = \min(\max(n,0),1) * 0.05$, passed to `build_noise_model` **directly as the depolarizing parameter** $\lambda$
     - $p_r = \min(\max((1-f)*0.25, 0), 0.5)$
     - the same depolarizing channel decorates the one-qubit and two-qubit gate sets; a symmetric readout confusion matrix of strength $p_r$ is attached to every measurement. See §4.
   - **Execute**: `run_on_aer(circ, shots=S, seed=s_e, optimization_level=0)`. Counts are returned over the three classical bits.
   - **Post-process**: marginalise the counts to Bob's received qubit, i.e. the `c[2]` bit, normalised to $p_B(0)+p_B(1)=1$, and compare against `state.ideal_probabilities()`. `fidelity_metrics` returns the Bhattacharyya coefficient $\big(\sum_b\sqrt{p_B(b)p_\psi(b)}\big)^2$, the total variation distance $\tfrac12\sum_b|p_B-p_\psi|$, and the Hellinger distance $\sqrt{1-\sum_b\sqrt{p_Bp_\psi}}$ (all inputs renormalised first).
   - **Returned mapping** (per hop): `measured_fidelity` (the Bhattacharyya value above), `noise_score`, `fidelity_hint` and `shots` echoed unchanged, plus `bit_flips` — the number of retained shots whose recovered bit disagrees with the *majority* ideal outcome. `bit_flips` is a disagreement count, **not** an error rate against a per-shot ground truth, which does not exist because the protocol destroys the input.
   - **Memory surrogate**: `HopResult.memory_decay_estimate = exp(-latency_ms / memory_lifetime_ms)`. **Not simulated** (A11).
3. **Sequential chain**: `simulate_route` (in `src/simulation.py`) iterates the consecutive node pairs of `route.path`, seeding hop $i$ with `seed + hop_index`, and records one `HopResult` per hop into a `MultiHopResult`. It also exposes the closed-form route properties `ideal_fidelity` ($\prod f_e$) and `attenuation_fidelity` ($\prod f_e e^{-d_e/50}$).
4. **End-to-end fidelity (reported)**: the experiment summaries multiply the per-hop measured fidelities. This is illustrative for sequential composition under this model; it is **not** repeater-network end-to-end fidelity.

### Stage 4b — Bidirectional teleportation model (composite) `[PROPOSED]+[SIMULATED]`
1. Define two payloads $|\psi_a\rangle, |\psi_b\rangle$ (A→B and B→A).
2. Build 6-qubit circuit: $[q_{a}, q_{e1a}, q_{e2a}, q_{b}, q_{e1b}, q_{e2b}]$ (data + two Bell pairs) and 6 classical bits.
3. Prepare Bell pair 1 for direction A→B, Bell pair 2 for direction B→A (independent). Apply Bennett circuits for both directions (order such that measurements produce outcomes on 6 clbits). Apply both corrections conditionally via `if_test` on their respective outcome bits.
4. Execute single composite Aer run (same shots/seed). Marginalise to received qubit of A and to received qubit of B. Report Bhattacharyya/TV/Hellinger for each direction independently.
5. **Explicit note**: two Bell pairs + two classical channels required; composite execution does not demonstrate physical simultaneous bidirectional teleportation (A9).

## 3. Walk decomposition details `[ESTABLISHED]+[SIMULATED]`

- **Double cover**: to have $\sum_{i=0}^{m-1} P_i = A$ exactly for any $m$-regular graph (including those with no perfect matching on $G$), the shunt extraction uses the bipartite double cover $B(G)= (V_{out}\cup V_{in}, M)$ and forms perfect matchings $M_i$ between out and in copies per round; projecting back gives $P_i$. Keys are integer node IDs (not hash strings of tuples in a way that varies), ensuring determinism across Python hash seeds.
- **Number of shunts** $m = \max_v \deg(v)$ in the proxy (regular). Each $P_i \in \{0,1\}^{n\times n}$ is a permutation matrix (a derangement on the matched set).
- **Coin unitarity**: $C^\dagger C = I_m$.
- **Shift unitarity**: $S^\dagger S = I_{mn}$.
- **Step unitary**: $U = S (C\otimes I_n)$ unitary, $|\psi_{t+1}\rangle = U|\psi_t\rangle$, $\|\psi_t\|=1$ for all $t$.

## 4. Noise/channel mapping (exact form) `[PROPOSED]+[SIMULATED]`

For a link with $(n,f)$, `teleport_link_with_noise` computes
$$
p_d = \min\!\big(\max(n, 0), 1\big) \times 0.05,\qquad p_r = \min\!\big(\max((1-f)\times 0.25, 0), 0.5\big)\,, 
$$
then calls `build_noise_model(p_d, readout_error=p_r)`.

**`p_d` is passed as the `strength` argument and is used directly as the depolarizing parameter $\lambda$ — there is no further rescaling.** `build_noise_model` documents the channel as the Pauli mixture
$$
G \longmapsto \left(1 - \tfrac{3\lambda}{4}\right) G + \tfrac{\lambda}{4}\big(X G X + Y G Y + Z G Z\big)\,,
$$
which is the standard single-qubit depolarizing form. The same channel is attached to the two-qubit gate set (`cx` is the only entangler the teleportation circuits actually use; the gate list is deliberately broad so the model still decorates the circuit after transpilation rewrites gates into the target basis).

Readout error is a symmetric confusion matrix
$$
M_r = \begin{bmatrix}
1-p_r & p_r\\
p_r & 1-p_r
\end{bmatrix},
$$
attached to every measurement. So a declared link fidelity of `0.8` yields `p_r = 0.05`, i.e. a 5% readout error, and `noise = 1.0` yields the maximum `p_d = 0.05`.

These constants (the `0.05` cap, the `0.25` scale and the `0.5` readout cap) are illustrative modelling parameters chosen for this prototype, not device calibrations. The two inputs are treated as independent knobs even though a real link would couple them.

## 5. Routing optimality rules `[PROPOSED]`

- A* with $h(v)=\min_{u\in\delta(v)} C(u,v)\ge0$ is admissible for the objective without prior. With $\lambda_{qw}\mathrm{prior}(v)>0$, the effective objective includes a node-heuristic term and `is_cost_optimal = False`.
- MDQW (with node terms $N(v)$) and Dijkstra (no node terms) optimise **different objectives**. Therefore `total_cost` values are not comparable across them; use **`edge_cost`** for any cross-algorithm comparison.
- Enabling prior never guarantees beating Dijkstra on edge cost; in general it can be worse or equal. The experiment set reports when it becomes suboptimal.
- `max_hops` is a constraint: optimality is not claimed under that constraint if it prunes the unconstrained optimum.
- If $\max x=\min x$ for some metric, that metric contributes 0 to all $C(e)$ for this network.

## 6. Outputs and reporting `[SIMULATED]`

`src/simulation.py` produces:
- `results/results.md` — full narrative with every experiment frame and the verdict lines as observed (including degenerate/null cases)
- `results/results_summary.csv` — concatenated frames (`frame` column)
- `results/figures/*.png` — 11 figures (topology, route, walk distribution/heatmap, cost comparisons, noise sweep, weight sensitivity, bidirectional fidelity, fidelity vs hops, teleportation counts, random-network aggregate)
- `docs/architecture.png` — seven-stage architecture diagram

All figures are written by Matplotlib with a non-interactive backend (`Agg`) where appropriate. The committed outputs are byte-identical to those generated by the same seed and code.

## 7. Failure modes and edge cases `[SIMULATED]`

- Source or target not a node of the network → `KeyError` (`"unknown target 'Z'"`).
- Both nodes known but no path between them → `RoutingError` (`"no path exists between 'P' and 'R'; the network is disconnected"`).
- `max_hops` smaller than the shortest path in hops → `RoutingError` (`"no route found from 'A' to 'F' within the max_hops limit of 1"`).
- All five weights zero → `RoutingError` from `RoutingWeights`.
- $f_e<0$ or $>1$, $n_e,r_e$ out of range, or a non-finite value → `MetricValidationError` from `EdgeMetrics`.
- Prior referencing an unknown node, or holding a negative/non-finite value → `RoutingError`.
- Empty graph or single node with no self-loop → no valid path for distinct endpoints.
- Proxy cannot be made regular and complete-graph fallback used → reported in notes.
- Aer execution fails (rare) → exception propagated; simulation reports `experiments/figures that failed: 0` in shipped run.

## 8. What is NOT demonstrated (explicit)

- Physical quantum communication or a physical quantum network.
- A quantum routing algorithm (routing is classical A*/Dijkstra).
- Quantum repeaters, entanglement swapping, purification, or error correction.
- Memory decoherence while waiting (only a closed-form surrogate is reported).
- Simultaneous bidirectional teleportation (two independent channels only).
- Quantum speed-up or performance advantage of any kind.
- Hardware validation against real devices.