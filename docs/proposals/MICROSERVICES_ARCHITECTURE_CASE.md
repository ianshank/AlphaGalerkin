# A Case for Selective Microservice Decomposition of AlphaGalerkin

**Status:** Proposal / RFC
**Author:** Architecture review, 2026-05-01
**Branch:** `claude/microservices-architecture-case-oyhiw`

## TL;DR

Full microservice decomposition of AlphaGalerkin is the **wrong** target. The
hot path — Galerkin attention, FNet mixing, MCTS self-play, gradient updates —
is a tightly-coupled tensor pipeline whose latency budget is measured in
microseconds and whose correctness depends on shared autograd state. Splitting
that across the network would destroy the project's headline property
(O(N) attention, FFT-batched rollouts) and add failure modes that don't exist
today.

What **is** worth extracting are the surfaces where (a) the code already talks
to an out-of-process boundary, (b) deployment cadence diverges from the
research core, or (c) a different team / runtime would be a natural owner.
That gives us a *services-on-the-edge, monolith-at-the-core* shape, which
captures most of the maintainability win without paying the distributed-system
tax on the parts that can't afford it.

This document makes the case for that shape, names the four services worth
carving out, and lists what **must not** be split and why.

---

## 1. Why "microservices everywhere" is the wrong goal here

The repository is large (31 top-level packages under `src/`, ~380 Python
modules, ~4.6 MB of source) and the CLAUDE.md milestone log shows ~15 distinct
research surfaces shipped in the last quarter. That growth rate is the usual
trigger for a microservices conversation. It shouldn't be, for four reasons
specific to this codebase:

1. **Shared autograd graph.** `CombinedAlphaGalerkinPhysicsLoss`
   (`src/training/physics_loss.py`) computes residual / boundary / IC /
   conservation terms via `torch.autograd.grad` against the same
   `AlphaGalerkinModel` forward pass that produced the policy/value heads.
   Splitting the loss computation onto a different process means either
   shipping the whole computation graph over the wire (dead on arrival
   latency-wise) or recomputing forward passes on each side (correctness and
   cost regressions).
2. **MCTS leaf batching depends on in-process FFT.** The whole point of the
   FNet evaluator is that hundreds of leaves get evaluated in one
   `torch.fft.rfft2` call. The existing `FNetEvaluator` (`src/mcts/`) and the
   Gumbel sequential-halving search (`src/mcts/gumbel.py`) assume that the
   evaluator returns synchronously inside the search. Per-leaf RPC overhead
   (even at 100 µs) dominates the actual evaluation cost.
3. **The "modules" are research surfaces, not bounded contexts.** `src/pde/`,
   `src/games/`, `src/modeling/`, `src/math_kernel/` look like service
   candidates on the file tree. They're not — they are pieces of a single
   tensor pipeline that imports each other freely (e.g.
   `HelicalHeatOperator` extends `LShapedPoissonOperator` extends
   `PhysicsOperator`, and the operator is imported directly by the trainer).
   Drawing service lines through that graph means inventing serialization
   formats for objects that today are PyTorch modules with shared parameters.
4. **The `Regression Surface` table in CLAUDE.md is the actual coupling map.**
   Six of the seven entries cross what would be a service boundary
   (`src/alphagalerkin` ↔ `src/mcts` ↔ `src/training` ↔ `src/pde` ↔
   `src/poc`). Today those couplings are guarded by `pytest`. Across a
   network they'd be guarded by contract tests, schema versioning, and a
   deploy ordering — orders of magnitude more operational overhead for the
   same guarantee.

The correct framing is: **the research core is a library, not a system.**
Maintainability problems in libraries are solved with module boundaries,
type checking, and tests — all of which this repo already invests in
(`mypy --strict`, ≥85% per-module coverage, the regression surface table).

## 2. What *is* worth extracting

Four surfaces in this codebase already behave like services or want to.
Extracting them captures the maintainability win that motivates the
microservices conversation, without touching the hot path.

### 2.1 Inference / Serving Service (`src/deployment` → service)

**Why it's a service candidate.** ONNX export, quantization, and the
`runtime.py` ORT wrapper exist precisely because we want to deploy the model
*outside* the training process. The artifact boundary is already an `.onnx`
file. The runtime targets (CPU, edge, multi-provider ORT) are deliberately
not the training stack.

**Proposed shape.** A standalone gRPC/REST service that loads ONNX models
from a registry and serves `predict(state) -> (policy, value)`. Deployable
independently of the trainer. Versioned by checkpoint hash. Owns its own
SLOs (p99 latency, throughput) which are *different* from training SLOs
(steps/sec).

**Win.** Decouples release cadence: the team shipping a new operator family
in `src/pde/` doesn't have to coordinate with the team running production
inference. Today these are the same people; tomorrow they may not be.

**Cost.** Low. The ONNX export pipeline already exists
(`src/deployment/export_onnx.py`, `src/deployment/validate.py`). This is
mostly packaging.

### 2.2 Self-Play Worker Pool (`src/distributed/worker.py` → service)

**Why it's a service candidate.** Self-play workers are *already* designed
to run on separate processes/nodes. They consume a model checkpoint, produce
trajectories, and write to a replay buffer. The communication is naturally
asynchronous and one-directional (model → worker → buffer).

**Proposed shape.** Workers as a horizontally-scaled stateless service,
fed by a model-zoo pull and emitting trajectories to a queue (e.g.
Pub/Sub on Vertex, or Redis streams locally). The trainer is one consumer
of that queue.

**Win.** Self-play scales independently of the trainer. CPU-only or
small-GPU instances become viable for self-play (today the trainer's GPU
cohabits with self-play on the same node by default). Spot-instance
preemption, already handled in `src/vertex/`, becomes a per-worker concern
instead of a job-wide failure.

**Cost.** Medium. We need a stable trajectory schema (Protobuf or Arrow),
a queue, and replay-buffer ingestion that's idempotent under retries. The
`src/vertex/` integration already pays most of this cost.

### 2.3 Benchmark / Perf Telemetry Service (`src/video_compression/perf` → service)

**Why it's a service candidate.** The perf harness shipped on 2026-04-27
already has explicit JSON schema versioning (`PERF_BASELINE_DOCUMENT_SCHEMA_VERSION`),
a `BaselineRegistry`, and a CLI with `record-baseline` / `diff` semantics.
That is, in all but name, a service. It also has a *different consumer
profile* than the rest of the codebase: CI gates and the dual-GPU rig, not
research scripts.

**Proposed shape.** Benchmark harness emits results to a small write-only
service (HTTP `POST /baselines`) which owns the registry storage,
schema migrations, and regression queries. CI calls `diff` against the
service rather than a local file.

**Win.** Baselines stop living in branches. Cross-branch regression
detection becomes a query, not a manual diff. Schema evolution
(`PERF_BASELINE_DOCUMENT_SCHEMA_VERSION` already migrated unversioned →
versioned) is owned in one place.

**Cost.** Low-medium. The schema work is done. What's missing is the
storage backend and the CI integration.

### 2.4 Vertex / Cloud Job Orchestration (`src/vertex` → service)

**Why it's a service candidate.** `src/vertex/` already encapsulates a
foreign system (GCP). The `scripts/train_vertex.py`, `scripts/vertex_jobs.py`,
GCS checkpoint manager, spot-preemption handler, and cost-tracking
modules form a coherent control-plane that has nothing to do with the
forward pass. Today this code is imported into the same Python environment
as the trainer; tomorrow it could be a small daemon with its own auth, its
own retry policy, and its own deploy cycle.

**Proposed shape.** A thin orchestration service exposing
`submit_job(config) → job_id`, `get_status(job_id)`, `cancel(job_id)`,
`cost_estimate(config)`. The CLI talks to the service; the service talks
to Vertex.

**Win.** GCP credential surface area shrinks (only the service needs SA
keys, not every developer). Cost-tracking and quota enforcement become
centralized policy. Job submission becomes auditable.

**Cost.** Low. The Python module is already structured around these calls.

## 3. What **must not** be split (and why)

For each of these, the cost of distribution exceeds the maintainability
benefit by at least an order of magnitude.

| Surface | Why it stays in-process |
|---|---|
| `src/modeling/` (model definition) | Single PyTorch module graph. Splitting forces wire-format serialization of `nn.Module` state — there isn't a stable one. |
| `src/math_kernel/` (Galerkin attention, FFT mixing) | Hot path. Per-call latency budget is sub-millisecond. Network adds ~100×. |
| `src/mcts/` (search + evaluator) | Synchronous evaluator contract. Distributing the evaluator inside search breaks batching, which is the whole performance story. |
| `src/training/` (loss, trainer, replay buffer) | Shares autograd graph with model. `BaseTrainer` refactor on 2026-04-07 deliberately consolidated AMP/grad-clip/scheduling — undoing that across a service boundary regresses the consolidation. |
| `src/pde/operators*` | Operators are PyTorch modules with parameters that participate in autograd through `PhysicsLoss`. Cannot live in a different process from the model. |
| `src/games/` (game rules) | Pure functions called millions of times per self-play episode. RPC overhead per move is unviable. |
| `src/poc/` (scenario framework) | Already a thin orchestration layer; the value is *not* in distributing it but in keeping its surface small. |

Rule of thumb: **anything that participates in a backward pass stays in
one process.** That eliminates the majority of the tree.

## 4. Migration shape (if and when)

This is **not** a recommendation to do all four extractions now. It's a
prioritized list with rough cost/benefit, so the team can pull individual
items when the pain justifies the work.

| Order | Service | Trigger to start | Effort |
|---|---|---|---|
| 1 | Inference Serving (§2.1) | First production inference deployment that isn't the trainer process | ~1 week |
| 2 | Vertex Orchestration (§2.4) | Second team / external user submitting jobs | ~1 week |
| 3 | Perf Telemetry (§2.3) | CI baseline drift becomes a recurring problem | ~2 weeks |
| 4 | Self-Play Workers (§2.2) | Trainer GPU saturation becomes the bottleneck | ~3–4 weeks |

Each extraction is independent and reversible. None of them touch the
research core. Each one earns a clear maintainability dividend
(release-cadence decoupling, blast-radius reduction, or independent
scaling) without paying for synthetic decomposition elsewhere.

## 5. What we get from this versus full microservices

| Concern | Monolith today | Selective extraction (this proposal) | Full microservices |
|---|---|---|---|
| Release cadence decoupling for serving | ✗ | ✓ | ✓ |
| Independent scaling of self-play | partial | ✓ | ✓ |
| Cross-team ownership of cloud orchestration | ✗ | ✓ | ✓ |
| Hot-path latency preserved | ✓ | ✓ | ✗ |
| Single autograd graph | ✓ | ✓ | ✗ |
| Operational complexity | low | low-medium | high |
| Schema/contract management surface | small | medium | very large |
| Onboarding cost for a new researcher | low | low | high |

Selective extraction takes the four real wins of microservices for this
codebase and leaves the costs that don't apply on the table.

## 6. Recommendation

1. **Don't** plan a microservices migration of the research core. It would
   regress the project's headline performance properties and trade tested
   in-process coupling for untested distributed coupling.
2. **Do** plan extractions 1 and 2 (Inference Serving, Vertex Orchestration)
   opportunistically — they're small, low-risk, and the artifact boundaries
   already exist.
3. **Defer** extractions 3 and 4 until concrete operational pain
   (CI baseline drift; trainer GPU saturation) creates the trigger. Until
   then they are speculative complexity.
4. **Continue** investing in the in-process maintainability levers that are
   already working: `mypy --strict`, the per-module coverage gates, the
   `Regression Surface` table in CLAUDE.md, and the registry/Pydantic-config
   patterns codified in `src/templates/`. These are doing the actual work
   that "microservices" is usually a proxy for.
