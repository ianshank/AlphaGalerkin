# Glossary

Terminology used across AlphaGalerkin's game-AI and PDE/scientific-computing code.

## Operator learning & Galerkin methods

- **Continuous operator learning** — Learning a mapping between function spaces
  (an *operator*) rather than a fixed-size vector map, so one model applies at any
  discretization/resolution.
- **Galerkin projection** — Solving `⟨Lu, v⟩ = ⟨f, v⟩ ∀ v ∈ V` by projecting onto
  finite trial/test bases. AlphaGalerkin reinterprets attention as a Petrov-Galerkin
  projection: Query = test basis, Key = trial basis, Value = the function projected.
- **Petrov-Galerkin** — A Galerkin method where the trial and test spaces differ
  (Key and Query bases need not match).
- **Galerkin attention** — O(N) attention via the projection `Context = KᵀV / n`
  (Monte-Carlo integral), `Output = Q · Context` — linear in sequence length N.
- **LBB / inf-sup condition** — The Ladyzhenskaya–Babuška–Brezzi stability
  condition `inf_u sup_v ⟨Lu, v⟩ / (‖u‖‖v‖) ≥ β > 0` guaranteeing convergence. In
  practice enforced by `dim(Key) ≥ dim(Query)`; monitored by the `StabilityGuard`.
- **Fourier features** — Positional encoding mapping coordinates to a bank of
  sinusoids, used for resolution-independent embedding.
- **FNet mixing** — Replacing attention with FFT-based token mixing (O(N log N))
  for fast MCTS leaf evaluation.
- **Fredholm integral equation** — The Green's-function formulation used to model
  influence fields.
- **Resolution invariance / zero-shot transfer** — Training at one resolution
  (e.g. 9×9) and evaluating at another (e.g. 19×19) with no retraining.

## PDE & numerical methods

- **Collocation points** — Sample points where a PDE residual is enforced.
- **Residual** — `Lu − f`; how far a candidate solution is from satisfying the PDE.
- **AMR (adaptive mesh refinement)** — Refining the discretization where error is
  largest. **h-refinement** subdivides cells; **p-refinement** raises basis order.
- **Dörfler marking** — A standard AMR strategy that marks the smallest set of
  elements carrying a fixed fraction of the total error.
- **PINN** — Physics-Informed Neural Network; a baseline that trains a net to
  minimize the PDE residual directly.
- **FDM** — Finite-Difference Method; a classical baseline solver.
- **SDF** — Signed Distance Function; implicit geometry representation (Leap 71
  / Noyron helical domains).
- **Manufactured solution** — A chosen exact `u` from which `f` is derived, giving
  a ground truth to measure error against.
- **Stochastic Galerkin / NKE** — Projecting a Kolmogorov-forward generator onto a
  Gaussian-mixture basis via operator splitting (the `src/pde/stochastic/` layer).

## Search & training (AlphaZero heritage)

- **MCTS (Monte Carlo Tree Search)** — Look-ahead search building a tree of
  simulated futures. The shared abstraction connecting the game-AI and PDE domains.
- **PUCT** — Predictor + Upper-Confidence-bound applied to Trees; the selection
  rule balancing prior policy and exploration.
- **Gumbel AlphaZero** — An MCTS variant using Gumbel-Top-k sampling and sequential
  halving for principled exploration with few simulations.
- **Evaluator** — The policy/value provider queried at MCTS leaves
  (`RandomEvaluator`, `FNetEvaluator`, `LMStudioEvaluator`, …).
- **Self-play** — Generating training data by having the agent play itself.
- **GTP** — Go Text Protocol; the interface Go GUIs use to drive an engine.
- **Elo** — Rating scale used to compare engine strength.

## Loss balancing

- **ReLoBRaLo** — Relative Loss Balancing with Random Lookback (physics-informed
  multi-term loss balancing).
- **GradNorm / SoftAdapt / Uncertainty weighting** — Alternative adaptive
  multi-task loss-weighting schemes.

## Project conventions

- **Scenario** — A configuration-driven PoC experiment run by `src/poc/`.
- **Spec** — A markdown contract (data contract + acceptance criteria +
  thresholds) written *before* code. See [specs](../specs/README.md).
- **`MetricThreshold`** — The canonical Pydantic threshold type
  (`src.poc.config.MetricThreshold`); specs reuse it rather than defining a parallel schema.
- **AQA test** — Acceptance-Quality-Assurance test asserting a spec's thresholds
  match the config's `get_default_thresholds()`.
- **Regression Surface** — The table in [`CLAUDE.md`](../CLAUDE.md) mapping each code
  path to the exact test commands that guard it.
