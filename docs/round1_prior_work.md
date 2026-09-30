# Round 1 — Prior Work & Relevant Experience

**Name:** Lohith Sanju P
**GitHub:** https://github.com/sanju-zxt

---

## Submission text

I work on problems where computation has to survive contact with physics — first
classical networks and cryptography, then actual quantum mechanics. Two projects
are directly relevant to this problem space.

**Lattice — post-quantum cryptography platform.** A working security product I
built end to end: FastAPI backend with 40 REST endpoints across ten routers, a
Next.js frontend, and a proxy tier. It scans code, certificates, containers and
network exposure, then maps what it finds onto post-quantum replacements —
RSA and ECDH to ML-KEM-768 (FIPS 203), ECDSA to ML-DSA-65 (FIPS 204) — and emits
a CBOM with a harvest-now-decrypt-later risk score. It models the break-now
window explicitly (RSA-2048 and ECDSA-256 by 2029, RSA-4096 by 2031) because
that timeline is the entire reason the migration is urgent. The interesting part
for me is that this is cryptography motivated by quantum mechanics rather than
by it: Shor's and Grover's arguments are what make the current defaults unsafe,
but the mitigations are classical algorithms I can deploy today.

**BQT-MDQW — quantum teleportation and quantum-walk routing.** A research
prototype I wrote from scratch, currently in this repository. It is three layers:
the Bennett 1993 teleportation circuit built and executed in Qiskit, a
discrete-time coined quantum walk built from its shunt decomposition, and a
multi-metric router that uses the walk to bias a graph search. 337 tests pass,
and a six-experiment study writes eleven figures and a full results report.

I want to be blunt about what that project is, because the interesting part to me
is the honesty. **The router is not a quantum algorithm** — it is A* over a graph
with a cost function inspired by the walk, and the walk is simulated densely with
NumPy. Every number in the repository is a classical simulation on Qiskit Aer; no
quantum hardware was involved. I deliberately report the null results: across 60
random networks the MDQW router never once beat Dijkstra and tied 60 times, a
noise sweep collapsed to a single distinct route cost, and a walk-derived prior
left the optimal route unchanged in 37 of 45 configurations. The first draft of
that report claimed the results needed `PYTHONHASHSEED=0` to reproduce. I traced
it to CPython's randomised string hashing in the cycle-cover selection, fixed it
to use integer keys, and left the original bug documented in the limitations
section. I would rather submit a prototype that admits its limits than one that
implies an advantage it cannot show.

I also want to flag one thing I found while building it: my six-qubit
bidirectional circuit originally composed two sub-circuits, which left the
`if_test` conditions pointing at a classical register that was never attached to
the parent circuit. It ran on Aer and produced plausible fidelities, but Qiskit
could not draw it. Building the channels in place fixed it. I mention it because
"the simulation ran" and "the circuit is correct" turned out to be different
claims, and only one of them was true.

**Resonance LM (adjacent, not quantum).** An experiment treating language tokens
as coupled oscillators, where meaning emerges from phase-locking rather than
attention. It is classical Kuramoto dynamics, and I list it only because it is
where I learned to run a numerical study properly — multiple phases, fixed seeds,
results reproduced from a script. It has a paper draft and an arXiv packaging
pipeline; it has not been submitted or published, and I am not presenting it as
either.

Outside quantum, I maintain several applied AI and accessibility products
(Hack-Matrix, herald-emergency, sentinel-prototype, BharatLensAI, SignBridge2Vision)
that are on my public GitHub. They are engineering range, not quantum evidence, so
I am not counting them toward the quantum criteria.

**What I have not done:** I have not run anything on a quantum computer, I have no
published papers, and I have no quantum-computing employment or formal training.
Everything above is my own build, and the strongest claim I can make about
BQT-MDQW is that it is honest, reproducible, and tested.

---

## Evidence links

| Project | Link | What it demonstrates |
|---|---|---|
| BQT-MDQW | `C:\Projects\BQT-MDQW` (local; push to `github.com/sanju-zxt/BQT-MDQW`) | Qiskit teleportation, coined quantum walk, classical router, 337 tests, 6 experiments |
| Lattice | local only — **not yet public** | Post-quantum crypto platform, 40 endpoints, ML-KEM/ML-DSA migration, CBOM |
| Resonance LM | `C:\Projects\resonance-lm` (local) | Coupled-oscillator experiment; paper draft, unsubmitted |
| Hack-Matrix | https://github.com/sanju-zxt/Hack-Matrix | TypeScript hackathon build |
| herald-emergency | https://github.com/sanju-zxt/herald-emergency | Incident-analysis platform |
| sentinel-prototype | https://github.com/sanju-zxt/sentinel-prototype | Assistive haptic co-pilot |
| BharatLensAI | https://github.com/sanju-zxt/BharatLensAI | Multilingual image understanding |
| SignBridge2Vision | https://github.com/sanju-zxt/SignBridge2Vision | Speech-to-sign accessibility system |

## Two things to fix before submitting

1. **BQT-MDQW has no GitHub remote yet.** The Round 1 form asks for repositories.
   Push it first, then put the real URL in place of the local path above.
2. **Lattice is not a git repository and contains `backend/.env`.** It is the
   strongest quantum-adjacent item I have and it cannot currently be linked. The
   `.gitignore` does exclude `.env`, so `git init` there is safe — but confirm
   the exclusion with `git status` before the first commit, and rotate any
   credentials that were ever pasted into chat.

## Notes on what I deliberately did not claim

- No paper is described as published or peer-reviewed.
- No claim of quantum speed-up, quantum advantage, or hardware execution.
- Lattice is framed as cryptography motivated by quantum threat, which is what
  it is, rather than as a quantum algorithm.
- The 14 non-quantum repositories are named once and excluded from the quantum
  criteria, rather than presented as a list of quantum-relevant work.
