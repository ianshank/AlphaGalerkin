# AlphaGalerkin C4 Architecture Model

This document describes the AlphaGalerkin system architecture using the C4 model
(Context, Containers, Components, Code).

---

## Level 1: System Context Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM CONTEXT                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌─────────────┐                              ┌─────────────────────┐     │
│    │  Researcher │                              │  Go Game Interface  │     │
│    │   (User)    │                              │    (GTP Client)     │     │
│    └──────┬──────┘                              └──────────┬──────────┘     │
│           │                                                │                │
│           │ Runs experiments                               │ Plays games    │
│           │ Validates claims                               │ via GTP        │
│           ▼                                                ▼                │
│    ┌──────────────────────────────────────────────────────────────┐        │
│    │                                                               │        │
│    │                     ALPHAGALERKIN                             │        │
│    │                                                               │        │
│    │   Resolution-independent Go AI using Continuous Operator      │        │
│    │   Learning (Galerkin Transformers & FNet)                     │        │
│    │                                                               │        │
│    │   Core Claims:                                                │        │
│    │   - Zero-shot transfer between board sizes (9x9 → 19x19)      │        │
│    │   - O(N) Galerkin attention (vs O(N²) softmax)                │        │
│    │   - O(N log N) FNet mixing for fast MCTS rollouts             │        │
│    │   - LBB stability guarantee for well-posed learning           │        │
│    │                                                               │        │
│    └──────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### External Entities

| Entity | Description | Interaction |
|--------|-------------|-------------|
| **Researcher** | ML researcher validating mathematical claims | Runs PoC scenarios, analyzes results |
| **GTP Client** | Go Text Protocol interface (e.g., Sabaki) | Plays games against the agent |
| **Training Data** | Self-play games, physics simulations | Input for supervised/RL training |

---

## Level 2: Container Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CONTAINER DIAGRAM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────┐    ┌──────────────────────┐                       │
│  │   CLI Entrypoints    │    │   GTP Server         │                       │
│  │   (scripts/)         │    │   (src/tools/gtp.py) │                       │
│  │                      │    │                      │                       │
│  │  - train.py          │    │  Protocol: GTP v2    │                       │
│  │  - train_physics.py  │    │  Commands: play,     │                       │
│  │  - benchmark_fnet.py │    │    genmove, etc.     │                       │
│  └──────────┬───────────┘    └──────────┬───────────┘                       │
│             │                           │                                    │
│             ▼                           ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                     CORE NEURAL OPERATOR                          │       │
│  │                     (src/modeling/)                               │       │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │       │
│  │  │ Galerkin    │  │ FNet        │  │ Stability   │               │       │
│  │  │ Attention   │  │ Mixing      │  │ Guard       │               │       │
│  │  │ (O(N))      │  │ (O(N log N))│  │ (LBB β>0)   │               │       │
│  │  └─────────────┘  └─────────────┘  └─────────────┘               │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│             │                                                                │
│             ▼                                                                │
│  ┌──────────────────────┐    ┌──────────────────────┐                       │
│  │   Math Kernel        │    │   Training Pipeline  │                       │
│  │   (src/math_kernel/) │    │   (src/training/)    │                       │
│  │                      │    │                      │                       │
│  │  - Fourier basis     │    │  - Self-play         │                       │
│  │  - Monte Carlo       │    │  - Replay buffer     │                       │
│  │  - Spectral filter   │    │  - Checkpoint mgmt   │                       │
│  └──────────────────────┘    └──────────────────────┘                       │
│             │                           │                                    │
│             ▼                           ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                     PROOF OF CONCEPT FRAMEWORK                    │       │
│  │                     (src/poc/)                                    │       │
│  │                                                                   │       │
│  │  Validates claims through reproducible, configurable scenarios    │       │
│  │  - Scenario registry    - Result persistence                      │       │
│  │  - Config-driven tests  - Comparative analysis                    │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Container Descriptions

| Container | Technology | Purpose |
|-----------|------------|---------|
| **CLI Entrypoints** | Python + Hydra | User-facing commands for training/evaluation |
| **GTP Server** | Python + socket | External interface for Go game play |
| **Core Neural Operator** | PyTorch | Galerkin-based neural network architecture |
| **Math Kernel** | PyTorch/NumPy | Mathematical primitives (basis, integrals) |
| **Training Pipeline** | PyTorch | Self-play, loss computation, checkpoints |
| **PoC Framework** | Python + Pydantic | Reproducible scenario execution |

---

## Level 3: Component Diagram

### 3.1 Core Neural Operator Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CORE NEURAL OPERATOR (src/modeling/)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────┐             │
│  │                   AlphaGalerkinOperator                     │             │
│  │                   (model.py)                                │             │
│  │                                                             │             │
│  │  Responsibilities:                                          │             │
│  │  - Compose Galerkin body + tactical head                    │             │
│  │  - Handle variable board sizes via continuous coords        │             │
│  │  - Output policy (move probabilities) + value (win prob)    │             │
│  └──────────────────────────┬─────────────────────────────────┘             │
│                             │                                                │
│          ┌──────────────────┼──────────────────┐                            │
│          ▼                  ▼                  ▼                            │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                   │
│  │ Galerkin      │  │ FNetMixing    │  │ Softmax       │                   │
│  │ Attention     │  │ Layer         │  │ Attention     │                   │
│  │ (attention.py)│  │ (fnet.py)     │  │ (attention.py)│                   │
│  │               │  │               │  │               │                   │
│  │ O(N) global   │  │ O(N log N)    │  │ O(N²) local   │                   │
│  │ influence     │  │ spectral mix  │  │ tactical      │                   │
│  └───────────────┘  └───────────────┘  └───────────────┘                   │
│          │                  │                  │                            │
│          ▼                  ▼                  ▼                            │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │                   StabilityGuard (stability.py)           │              │
│  │                                                           │              │
│  │  Monitors LBB constant β during training                  │              │
│  │  Enforces: dim(Key) >= dim(Query)                         │              │
│  │  Computes: σ_min(K^T V) for stability diagnostics         │              │
│  └──────────────────────────────────────────────────────────┘              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 PoC Framework Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    POC FRAMEWORK (src/poc/)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────┐             │
│  │                   ScenarioRunner                            │             │
│  │                   (runner.py)                               │             │
│  │                                                             │             │
│  │  Responsibilities:                                          │             │
│  │  - Load scenarios from config                               │             │
│  │  - Execute validation logic                                 │             │
│  │  - Collect and aggregate results                            │             │
│  │  - Generate reports                                         │             │
│  └──────────────────────────┬─────────────────────────────────┘             │
│                             │                                                │
│          ┌──────────────────┼──────────────────┐                            │
│          ▼                  ▼                  ▼                            │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                   │
│  │ Scenario      │  │ Scenario      │  │ Result        │                   │
│  │ Registry      │  │ Config        │  │ Collector     │                   │
│  │ (registry.py) │  │ (config.py)   │  │ (results.py)  │                   │
│  │               │  │               │  │               │                   │
│  │ @scenario     │  │ Pydantic      │  │ JSON/Parquet  │                   │
│  │ decorator     │  │ validation    │  │ persistence   │                   │
│  └───────────────┘  └───────────────┘  └───────────────┘                   │
│          │                  │                  │                            │
│          ▼                  ▼                  ▼                            │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │              Built-in Scenario Implementations            │              │
│  │                                                           │              │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │              │
│  │  │ Transfer    │ │ Complexity  │ │ Stability   │         │              │
│  │  │ Scenario    │ │ Scenario    │ │ Scenario    │         │              │
│  │  │ (transfer/) │ │ (scaling/)  │ │ (lbb/)      │         │              │
│  │  └─────────────┘ └─────────────┘ └─────────────┘         │              │
│  └──────────────────────────────────────────────────────────┘              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Math Kernel Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MATH KERNEL (src/math_kernel/)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────┐              │
│  │                   Basis Functions (basis.py)               │              │
│  │                                                            │              │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │              │
│  │  │ Fourier      │  │ Chebyshev    │  │ Legendre     │     │              │
│  │  │ Features     │  │ Polynomials  │  │ Polynomials  │     │              │
│  │  │              │  │              │  │              │     │              │
│  │  │ sin/cos      │  │ T_n(x)       │  │ P_n(x)       │     │              │
│  │  │ encoding     │  │ recurrence   │  │ orthogonal   │     │              │
│  │  └──────────────┘  └──────────────┘  └──────────────┘     │              │
│  └───────────────────────────────────────────────────────────┘              │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────┐              │
│  │               Integral Approximation (integral.py)         │              │
│  │                                                            │              │
│  │  ┌──────────────────────┐  ┌──────────────────────┐       │              │
│  │  │ Monte Carlo          │  │ Galerkin             │       │              │
│  │  │ Integral             │  │ Projection           │       │              │
│  │  │                      │  │                      │       │              │
│  │  │ 1/n normalization    │  │ Q(K^T V / n)         │       │              │
│  │  │ for O(1) outputs     │  │ projection           │       │              │
│  │  └──────────────────────┘  └──────────────────────┘       │              │
│  └───────────────────────────────────────────────────────────┘              │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────┐              │
│  │               Spectral Filtering (spectral.py)             │              │
│  │                                                            │              │
│  │  Low-pass filtering for stability                          │              │
│  │  Butterworth / Gaussian rolloff options                    │              │
│  └───────────────────────────────────────────────────────────┘              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Level 4: Code Diagram

### 4.1 Scenario Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SCENARIO EXECUTION SEQUENCE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User                ScenarioRunner      Registry      Scenario    Results   │
│   │                       │                 │             │           │      │
│   │  run("transfer")      │                 │             │           │      │
│   │──────────────────────>│                 │             │           │      │
│   │                       │                 │             │           │      │
│   │                       │  get_scenario() │             │           │      │
│   │                       │────────────────>│             │           │      │
│   │                       │                 │             │           │      │
│   │                       │  scenario_cls   │             │           │      │
│   │                       │<────────────────│             │           │      │
│   │                       │                 │             │           │      │
│   │                       │  load_config()  │             │           │      │
│   │                       │─────────────────────────────>│           │      │
│   │                       │                 │             │           │      │
│   │                       │  setup()        │             │           │      │
│   │                       │─────────────────────────────>│           │      │
│   │                       │                 │             │           │      │
│   │                       │  execute()      │             │           │      │
│   │                       │─────────────────────────────>│           │      │
│   │                       │                 │             │           │      │
│   │                       │                 │    [runs validation logic]     │
│   │                       │                 │             │           │      │
│   │                       │  ScenarioResult │             │           │      │
│   │                       │<─────────────────────────────│           │      │
│   │                       │                 │             │           │      │
│   │                       │  collect()      │             │           │      │
│   │                       │─────────────────────────────────────────>│      │
│   │                       │                 │             │           │      │
│   │  Summary Report       │                 │             │           │      │
│   │<──────────────────────│                 │             │           │      │
│   │                       │                 │             │           │      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Key Data Structures

```python
# Scenario Configuration (Pydantic)
class ScenarioConfig(BaseModel):
    """Base configuration for all scenarios."""
    name: str
    description: str
    tier: Literal["unit", "functional", "integration"]
    enabled: bool = True
    timeout_seconds: int = 3600

class TransferScenarioConfig(ScenarioConfig):
    """Zero-shot transfer scenario configuration."""
    train_resolution: int = 9
    eval_resolutions: list[int] = [9, 13, 19]
    mse_threshold: float = 0.05
    n_train_samples: int = 5000
    n_eval_samples: int = 500

# Scenario Result
class ScenarioResult(BaseModel):
    """Result of scenario execution."""
    scenario_name: str
    passed: bool
    metrics: dict[str, float]
    artifacts: dict[str, Path]
    duration_seconds: float
    timestamp: datetime
    config_hash: str  # For reproducibility
```

### 4.3 Component Interactions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPONENT DEPENDENCY GRAPH                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                           ┌───────────────┐                                 │
│                           │  CLI / API    │                                 │
│                           └───────┬───────┘                                 │
│                                   │                                          │
│                    ┌──────────────┴──────────────┐                          │
│                    ▼                             ▼                          │
│            ┌───────────────┐           ┌───────────────┐                   │
│            │ ScenarioRunner│           │ GTP Server    │                   │
│            └───────┬───────┘           └───────┬───────┘                   │
│                    │                           │                            │
│         ┌─────────┬┴─────────┐                │                            │
│         ▼         ▼          ▼                │                            │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐     │                            │
│   │ Registry │ │ Config   │ │ Results  │     │                            │
│   └──────────┘ └──────────┘ └──────────┘     │                            │
│         │                                     │                            │
│         └─────────────────┬───────────────────┘                            │
│                           ▼                                                 │
│                   ┌───────────────┐                                        │
│                   │ AlphaGalerkin │                                        │
│                   │ Operator      │                                        │
│                   └───────┬───────┘                                        │
│                           │                                                 │
│            ┌──────────────┼──────────────┐                                 │
│            ▼              ▼              ▼                                 │
│     ┌───────────┐  ┌───────────┐  ┌───────────┐                           │
│     │ Galerkin  │  │ FNet      │  │ Stability │                           │
│     │ Attention │  │ Mixing    │  │ Guard     │                           │
│     └─────┬─────┘  └─────┬─────┘  └─────┬─────┘                           │
│           │              │              │                                  │
│           └──────────────┼──────────────┘                                  │
│                          ▼                                                  │
│                   ┌───────────────┐                                        │
│                   │ Math Kernel   │                                        │
│                   │ (basis,       │                                        │
│                   │  integral,    │                                        │
│                   │  spectral)    │                                        │
│                   └───────────────┘                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Decision Records (ADRs)

### ADR-001: Galerkin Attention for Resolution Independence

**Status:** Accepted

**Context:** Standard attention is O(N²) and resolution-dependent.

**Decision:** Use Galerkin attention with 1/n Monte Carlo normalization.

**Consequences:**
- (+) O(N) complexity
- (+) Resolution-independent outputs
- (-) Requires LBB stability monitoring

### ADR-002: FNet for MCTS Rollouts

**Status:** Accepted

**Context:** MCTS requires fast leaf evaluation for high throughput.

**Decision:** Use FFT-based mixing in strategic layers.

**Consequences:**
- (+) O(N log N) mixing vs O(N²) attention
- (+) Batch parallelization of rollouts
- (-) Less expressive than full attention

### ADR-003: Physics PoC as Validation Proxy

**Status:** Accepted

**Context:** Full Go training is expensive; need fast validation.

**Decision:** Use Poisson equation as supervised learning proxy.

**Consequences:**
- (+) Fast iteration on core claims
- (+) Ground truth available (analytical solution)
- (-) Not direct Go validation (requires later integration test)

---

## Deployment View (Future)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT ARCHITECTURE (PLANNED)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         Development                                  │    │
│  │                                                                      │    │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │    │
│  │   │ Local Dev    │    │ CI/CD        │    │ Test GPU     │         │    │
│  │   │ (CPU)        │    │ (GitHub      │    │ Cluster      │         │    │
│  │   │              │    │  Actions)    │    │ (A100s)      │         │    │
│  │   │ Unit tests,  │    │ Lint, type,  │    │ Integration  │         │    │
│  │   │ fast PoC     │    │ unit tests   │    │ PoC, full    │         │    │
│  │   └──────────────┘    └──────────────┘    └──────────────┘         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         Production (Future)                          │    │
│  │                                                                      │    │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │    │
│  │   │ Inference    │    │ Training     │    │ Artifact     │         │    │
│  │   │ Service      │    │ Pipeline     │    │ Storage      │         │    │
│  │   │ (TorchServe) │    │ (K8s Jobs)   │    │ (S3/GCS)     │         │    │
│  │   └──────────────┘    └──────────────┘    └──────────────┘         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Cross-Cutting Concerns

### Logging Strategy

- **Structured logging** via `structlog`
- **Levels:** DEBUG (dev), INFO (prod), metrics as structured fields
- **Correlation IDs** for scenario execution tracing

### Configuration Management

- **Pydantic** for validation and serialization
- **YAML** files for scenario definitions
- **Environment variables** for secrets/deployment config

### Testing Strategy

| Layer | Framework | Coverage Target |
|-------|-----------|-----------------|
| Unit | pytest + hypothesis | 90% math_kernel |
| Integration | pytest | 80% poc scenarios |
| E2E | pytest + fixtures | Key happy paths |

### Observability

- **Metrics:** Training loss, LBB constant, scenario pass rates
- **Artifacts:** Model checkpoints, result JSONs, plots
- **Traceability:** Config hashes for reproducibility
