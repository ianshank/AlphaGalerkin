# AlphaGalerkin Next-Phase Implementation Plan

> **Design Principle:** This document treats prompting as constraint programming, not instruction writing. Define the feasible region, objective function, and search parameters—then let the agent solve.

---

## SECTION 1: OBJECTIVE FUNCTION

### 1.1 System Intent

```
I am building: A next-generation extensible infrastructure for AlphaGalerkin enabling
distributed training, portable deployment, multi-game support, advanced search, and
enhanced validation—all while preserving resolution independence.
```

### 1.2 Success Criteria (Mechanically Verifiable)

```
This succeeds when:
- [ ] Distributed training achieves >85% scaling efficiency on 4 nodes
- [ ] ONNX export produces models runnable on CPU/int8 with <10% accuracy loss
- [ ] Multi-game abstraction supports Go, Chess, Shogi with shared operator core
- [ ] Gumbel AlphaZero search improves win rate by >5% over vanilla MCTS
- [ ] Hyperparameter tuning finds configurations 20%+ better than defaults
- [ ] All new code has >80% test coverage
- [ ] All verification commands pass (ruff, mypy --strict, pytest)
```

### 1.3 Problem Description (The "Three Paragraphs")

**Paragraph 1: What problem does this solve?**
AlphaGalerkin's continuous operator approach enables resolution-independent learning,
but the current implementation is limited to single-node training, PyTorch-only
deployment, Go-specific logic, vanilla MCTS, and manual hyperparameter selection.
This plan extends the system to production-grade infrastructure supporting
distributed training across GPU clusters, edge deployment via ONNX/quantization,
generalization to multiple games through abstraction, state-of-the-art search
algorithms, and automated optimization.

**Paragraph 2: What are the key data flows?**
- Distributed Training: Self-play → Local Buffer → Gradient Sync (NCCL) → Parameter Server
- ONNX Export: PyTorch Model → Trace/Script → ONNX Graph → Quantization → Runtime
- Multi-Game: Game State → Abstract Interface → Continuous Embedding → Shared Operator → Game-Specific Head
- Advanced MCTS: Root → Policy Prior + Gumbel Noise → Sequential Halving → Value Update → Improved Policy
- PoC Framework: Config → Hyperparameter Sampler → Trial Runner → Statistical Analysis → Report

**Paragraph 3: What are the failure modes?**
- Distributed: Gradient divergence, stragglers, network partitions → mitigate with AllReduce, async updates
- ONNX: Dynamic shapes unsupported, operator gaps → mitigate with shape tracing, custom ops
- Multi-Game: Action space explosion, game-specific invariants → mitigate with masked heads, modular design
- Advanced MCTS: Exploration collapse, computational overhead → mitigate with Gumbel trick, batched evaluation
- PoC: False positives in significance tests, hyperparameter overfitting → mitigate with Bonferroni correction, holdout sets

---

## SECTION 2: FEASIBLE REGION (Constraints)

### 2.1 Hard Constraints (Violations = Failure)

```
- Language/Runtime: Python 3.11+, PyTorch 2.0+
- Required Dependencies: Pydantic v2, structlog, einops, hydra-core
- Security: No hardcoded secrets, all inputs validated via Pydantic
- Compatibility: Must run on CUDA 11.8+, CPU fallback required
- Architecture: All tensor operations use einops for dimension clarity
- Mathematical: LBB stability (dim(Key) >= dim(Query)) must be preserved
- Resolution Independence: No hardcoded board sizes in operator core
```

### 2.2 Soft Constraints (Preferences)

```
- Style: ruff formatting, Google-style docstrings, 100 char line limit
- Architecture: Prefer composition over inheritance, factory functions
- Performance: Prefer async I/O for network ops, minimize GPU<->CPU transfers
- Testing: pytest, >80% coverage on core logic, property-based tests for math
- Logging: structlog throughout with context binding
- Configuration: Pydantic for validation, Hydra for CLI overrides
```

### 2.3 Anti-Constraints (Explicit Freedoms)

```
You ARE permitted to:
- Restructure existing file organization for new modules
- Add PyPI dependencies for distributed training (torch.distributed, NCCL)
- Add PyPI dependencies for ONNX (onnx, onnxruntime, onnxruntime-quantization)
- Add PyPI dependencies for statistics (scipy, statsmodels)
- Choose implementation patterns not explicitly specified
- Create new configuration schemas following existing patterns
- Add new PoC scenarios for validation
```

---

## SECTION 3: PERMISSION ARCHITECTURE

### 3.1 Scope (What You Can Touch)

```
IN SCOPE:
- All files in /src
- All files in /tests
- Configuration in /config
- Documentation in /docs
- Scripts in /scripts

OUT OF SCOPE:
- External dependencies' source code
- Production deployment manifests (Kubernetes, etc.)
- Files marked with # DO NOT MODIFY
```

### 3.2 Autonomy Level

```
AUTONOMOUS (proceed without asking):
- File creation/deletion within scope
- Dependency installation (requirements.txt, pyproject.toml)
- Running tests
- Refactoring for consistency
- Adding type hints and docstrings
- Creating new configuration schemas

CONFIRM FIRST (ask before proceeding):
- Architectural changes affecting >5 modules
- Breaking API changes to existing public interfaces
- Deletions of >200 lines of established code
- Changes to mathematical operators (attention, FNet)

PROHIBITED (do not attempt):
- Commits to main branch directly
- External API calls with side effects (cloud services)
- Modifications to CLAUDE.md without explicit approval
- Removal of existing test coverage
```

### 3.3 Resource Budget

```
- Max iterations before requesting guidance: 5
- Max files to modify in single pass: 30
- Time-boxed exploration: ≤10 min on research before asking
- Max new dependencies per module: 5
```

---

## SECTION 4: FEEDBACK LOOP SPECIFICATION

### 4.1 Verification Commands

```bash
# After writing code, run in this order:
1. ruff check src/ tests/
2. ruff format src/ tests/ --check
3. mypy src/ --strict
4. pytest tests/unit -v --tb=short
5. pytest tests/integration -v --tb=short (if applicable)
```

### 4.2 Error Handling Protocol

```
ON LINT FAILURE:
→ Run `ruff check --fix` to auto-fix
→ Re-run verification

ON TYPE ERROR:
→ Analyze error message
→ Add type hints or fix type inconsistencies
→ Re-run verification

ON TEST FAILURE:
→ Read failure output carefully
→ Identify root cause (implementation bug vs test bug)
→ Fix implementation (not test, unless test is demonstrably wrong)
→ Re-run verification

ON REPEATED FAILURE (same error 3x):
→ Stop and report analysis
→ Document attempted fixes
→ Request human guidance
```

### 4.3 Success Verification

```
Before reporting completion:
1. All verification commands pass
2. New code has test coverage
3. Configuration schemas validate
4. CLAUDE.md updated with new commands/decisions
5. Generate brief summary of changes made
```

---

## SECTION 5: CONTEXT PERSISTENCE

### 5.1 Session Memory (CLAUDE.md Updates)

```markdown
## New Verification Commands

# Distributed Training
torchrun --nproc_per_node=4 scripts/train_distributed.py

# ONNX Export
python -m src.deployment.export_onnx --checkpoint path/to/model.pt

# Multi-Game
python -m scripts.train --config-name=train_chess
python -m scripts.train --config-name=train_shogi

# Advanced MCTS
python -m src.experiments.benchmark_mcts --search-type gumbel

# PoC Framework Enhanced
python -m src.poc.cli tune --scenario transfer --n-trials 100
python -m src.poc.cli significance --baseline run_a --treatment run_b
```

### 5.2 Information to Preserve

```
- Build/test commands that work
- Non-obvious environment setup steps
- Architectural decisions and their rationale
- Gotchas discovered during implementation
- Performance benchmarks and baselines
```

---

## SECTION 6: EXECUTION PROTOCOL

### 6.1 Initial Actions (Always Do First)

```
1. Read CLAUDE.md if exists
2. Scan project structure to understand existing patterns
3. Identify entry points and test commands
4. Run existing tests to establish baseline
5. Review existing configuration schemas
```

### 6.2 Implementation Order

```
For each major module:
1. Design configuration schema (Pydantic)
2. Implement core abstractions/interfaces
3. Implement concrete implementations
4. Write unit tests
5. Write integration tests
6. Run verification loop
7. Update documentation
8. Update CLAUDE.md
```

### 6.3 Completion Checklist

```
□ All success criteria met
□ All verification commands pass
□ New tests achieve >80% coverage
□ CLAUDE.md updated with new commands/decisions
□ Summary of changes provided
□ Known limitations documented
```

---

## SECTION 7: MODULE SPECIFICATIONS

### 7.1 Distributed Training (`src/distributed/`)

**Files:**
- `config.py` - DistributedConfig schema
- `launcher.py` - Multi-node launch utilities
- `worker.py` - Self-play worker for distributed generation
- `gradient_sync.py` - NCCL-based gradient aggregation
- `model_zoo.py` - Model checkpointing and curriculum

**Key Interfaces:**
```python
class DistributedConfig(BaseModel):
    world_size: int = 1
    backend: Literal["nccl", "gloo"] = "nccl"
    gradient_accumulation_steps: int = 1
    sync_batch_norm: bool = True
    find_unused_parameters: bool = False

class DistributedTrainer:
    def __init__(self, config: DistributedConfig): ...
    def setup(self, rank: int, world_size: int) -> None: ...
    def train_step(self, batch: TrainingBatch) -> TrainingMetrics: ...
    def all_reduce_gradients(self) -> None: ...
    def save_checkpoint(self, rank: int) -> Path | None: ...
```

### 7.2 ONNX Export (`src/deployment/`)

**Files:**
- `config.py` - ExportConfig schema
- `export_onnx.py` - PyTorch to ONNX conversion
- `quantize.py` - INT8 quantization utilities
- `runtime.py` - ONNX Runtime inference wrapper
- `validate.py` - Export validation against PyTorch

**Key Interfaces:**
```python
class ExportConfig(BaseModel):
    opset_version: int = 17
    dynamic_axes: dict[str, dict[int, str]] = {}
    input_names: list[str] = ["board_state"]
    output_names: list[str] = ["policy", "value"]
    quantization: QuantizationConfig | None = None

class QuantizationConfig(BaseModel):
    mode: Literal["static", "dynamic"] = "dynamic"
    per_channel: bool = True
    reduce_range: bool = False
    calibration_samples: int = 100

class ONNXExporter:
    def export(self, model: AlphaGalerkinModel, config: ExportConfig) -> Path: ...
    def quantize(self, onnx_path: Path, config: QuantizationConfig) -> Path: ...
    def validate(self, pytorch_model, onnx_path, tolerance: float = 1e-5) -> bool: ...
```

### 7.3 Multi-Game Support (`src/games/`)

**Files:**
- `interface.py` - Abstract game interface
- `state.py` - Generic game state representation
- `go.py` - Go-specific implementation
- `chess.py` - Chess implementation (stub/full)
- `shogi.py` - Shogi implementation (stub/full)
- `registry.py` - Game registration and discovery

**Key Interfaces:**
```python
class GameInterface(ABC):
    @property
    @abstractmethod
    def action_space_size(self) -> int: ...

    @property
    @abstractmethod
    def state_channels(self) -> int: ...

    @abstractmethod
    def get_legal_actions(self, state: GameState) -> list[int]: ...

    @abstractmethod
    def apply_action(self, state: GameState, action: int) -> GameState: ...

    @abstractmethod
    def is_terminal(self, state: GameState) -> bool: ...

    @abstractmethod
    def get_winner(self, state: GameState) -> int | None: ...

    @abstractmethod
    def to_tensor(self, state: GameState) -> Tensor: ...

    @abstractmethod
    def get_symmetries(self, state: GameState, policy: Tensor) -> list[tuple[GameState, Tensor]]: ...

@dataclass
class GameState:
    board: np.ndarray
    current_player: int
    move_history: list[int]
    metadata: dict[str, Any]
```

### 7.4 Advanced MCTS (`src/mcts/`)

**Files:**
- `config.py` - MCTSConfig extensions
- `gumbel.py` - Gumbel AlphaZero search
- `exploration.py` - Value-based exploration strategies
- `policy_improvement.py` - Policy improvement operators
- `batched_evaluator.py` - Efficient batched leaf evaluation

**Key Interfaces:**
```python
class GumbelMCTSConfig(BaseModel):
    n_simulations: int = 800
    max_num_considered_actions: int = 16
    c_visit: float = 50.0
    c_scale: float = 1.0
    use_mixed_value: bool = True

class GumbelMCTS:
    def __init__(self, config: GumbelMCTSConfig): ...
    def search(self, root_state: GameState, model: AlphaGalerkinModel) -> PolicyOutput: ...
    def _sequential_halving(self, actions: list[int], logits: Tensor) -> list[int]: ...
    def _compute_completed_q(self, node: Node) -> Tensor: ...
```

### 7.5 Enhanced PoC Framework (`src/poc/`)

**Files (additions):**
- `tuning/config.py` - Hyperparameter tuning config
- `tuning/sampler.py` - Hyperparameter samplers (grid, random, Bayesian)
- `tuning/trial.py` - Trial execution and tracking
- `statistics/significance.py` - Statistical significance testing
- `statistics/effect_size.py` - Effect size calculations
- `visualization/plots.py` - Comparative visualizations
- `visualization/reports.py` - HTML/PDF report generation

**Key Interfaces:**
```python
class TuningConfig(BaseModel):
    search_space: dict[str, SearchSpace]
    n_trials: int = 100
    sampler: Literal["grid", "random", "tpe"] = "tpe"
    pruner: Literal["none", "median", "hyperband"] = "median"
    objective_metric: str = "mse"
    direction: Literal["minimize", "maximize"] = "minimize"

class SearchSpace(BaseModel):
    type: Literal["float", "int", "categorical"]
    low: float | None = None
    high: float | None = None
    choices: list[Any] | None = None
    log_scale: bool = False

class SignificanceTest(BaseModel):
    test_type: Literal["t_test", "mann_whitney", "bootstrap"]
    alpha: float = 0.05
    correction: Literal["none", "bonferroni", "holm"] = "bonferroni"

class HyperparameterTuner:
    def __init__(self, config: TuningConfig, scenario: BaseScenario): ...
    def tune(self) -> TuningResult: ...
    def get_best_params(self) -> dict[str, Any]: ...

class StatisticalAnalyzer:
    def compare_runs(self, baseline: list[float], treatment: list[float], test: SignificanceTest) -> ComparisonResult: ...
    def effect_size(self, baseline: list[float], treatment: list[float]) -> EffectSizeResult: ...
```

---

## SECTION 8: CONFIGURATION EXAMPLES

### 8.1 Distributed Training Config

```yaml
# config/train_distributed.yaml
distributed:
  enabled: true
  world_size: 4
  backend: nccl
  gradient_accumulation_steps: 4
  sync_batch_norm: true

training:
  batch_size: 64  # Per-GPU batch size
  learning_rate: 0.001  # Will be scaled by world_size

self_play:
  workers_per_node: 2
  games_per_worker: 50
```

### 8.2 ONNX Export Config

```yaml
# config/export.yaml
export:
  opset_version: 17
  input_names: ["board_state"]
  output_names: ["policy", "value"]
  dynamic_axes:
    board_state: {0: "batch", 2: "height", 3: "width"}
    policy: {0: "batch"}
    value: {0: "batch"}

quantization:
  enabled: true
  mode: dynamic
  per_channel: true
```

### 8.3 Multi-Game Config

```yaml
# config/train_chess.yaml
game:
  name: chess
  action_space: 4672  # All possible moves
  state_channels: 119  # AlphaZero representation

operator:
  # Shared continuous operator config
  d_model: 256
  n_galerkin_layers: 6
```

### 8.4 Enhanced PoC Config

```yaml
# config/scenarios/transfer_tuning.yaml
tuning:
  n_trials: 100
  sampler: tpe
  pruner: hyperband
  objective_metric: mse_19x19
  direction: minimize

  search_space:
    d_model:
      type: int
      low: 64
      high: 512
      log_scale: true
    learning_rate:
      type: float
      low: 1e-5
      high: 1e-2
      log_scale: true
    n_fourier_features:
      type: categorical
      choices: [32, 64, 128, 256]

statistics:
  test_type: bootstrap
  n_bootstrap: 10000
  alpha: 0.05
  correction: bonferroni
```

---

## SECTION 9: TESTING STRATEGY

### 9.1 Unit Tests

```
tests/
  distributed/
    test_config.py      - Config validation
    test_gradient_sync.py - Mock gradient synchronization
    test_worker.py      - Worker isolation tests

  deployment/
    test_export.py      - ONNX export with small models
    test_quantize.py    - Quantization pipeline
    test_runtime.py     - Inference accuracy

  games/
    test_interface.py   - Interface contract tests
    test_go.py          - Go-specific logic
    test_chess.py       - Chess move generation
    test_registry.py    - Game discovery

  mcts/
    test_gumbel.py      - Gumbel sampling
    test_exploration.py - Exploration strategies
    test_batched.py     - Batched evaluation

  poc/
    test_tuning.py      - Hyperparameter sampler
    test_significance.py - Statistical tests
    test_visualization.py - Plot generation
```

### 9.2 Integration Tests

```
tests/integration/
  test_distributed_training.py  - Multi-process training (mocked NCCL)
  test_onnx_e2e.py             - Full export->quantize->inference
  test_multigame_training.py   - Training loop with different games
  test_gumbel_mcts_game.py     - Full game with Gumbel MCTS
  test_tuning_workflow.py      - Hyperparameter tuning pipeline
```

### 9.3 Property-Based Tests

```
tests/property/
  test_game_symmetries.py      - Symmetry preservation
  test_onnx_equivalence.py     - PyTorch ↔ ONNX output equivalence
  test_gumbel_distribution.py  - Gumbel sampling properties
```

---

## SECTION 10: MIGRATION PATH

### 10.1 Backwards Compatibility

```
- Existing configs continue to work (new fields have defaults)
- Single-node training unchanged when distributed.enabled=false
- Existing PoC scenarios work with new framework
- Go remains default game when game.name not specified
```

### 10.2 Deprecation Strategy

```
- Old MCTS config fields marked deprecated in v1.1
- Migration warnings in logs for 2 release cycles
- Full removal in v2.0
```

---

## SECTION 11: DEPENDENCIES

### 11.1 New Required Dependencies

```toml
# pyproject.toml additions
[project.dependencies]
# Distributed
torch >= 2.0.0  # Already present

# ONNX
onnx >= 1.14.0
onnxruntime >= 1.15.0
onnxruntime-tools >= 1.7.0  # For quantization

# Statistics
scipy >= 1.11.0
statsmodels >= 0.14.0

# Visualization
plotly >= 5.15.0
kaleido >= 0.2.1  # For static image export

[project.optional-dependencies]
distributed = [
    "torch >= 2.0.0",
]
onnx = [
    "onnx >= 1.14.0",
    "onnxruntime >= 1.15.0",
]
tuning = [
    "optuna >= 3.3.0",
]
```

---

## SECTION 12: IMPLEMENTATION PRIORITY

### Phase 1: Foundation (Week 1-2)
1. Configuration schemas for all modules
2. Abstract game interface + Go implementation
3. Basic ONNX export (no quantization)
4. Unit tests for new schemas

### Phase 2: Core Features (Week 3-4)
5. Distributed training infrastructure
6. ONNX quantization support
7. Gumbel MCTS implementation
8. Integration tests

### Phase 3: Enhanced PoC (Week 5-6)
9. Hyperparameter tuning with Optuna
10. Statistical significance testing
11. Visualization and reporting
12. Documentation updates

### Phase 4: Multi-Game (Week 7-8)
13. Chess implementation
14. Shogi implementation
15. Cross-game validation
16. Performance benchmarks

---

## SECTION 13: SUCCESS METRICS

| Module | Metric | Target | Measurement |
|--------|--------|--------|-------------|
| Distributed | Scaling efficiency | >85% on 4 nodes | Time per step ratio |
| ONNX | Accuracy retention | <10% loss | MSE vs PyTorch |
| Multi-Game | Interface coverage | 100% | Game tests pass |
| Gumbel MCTS | Win rate improvement | >5% | vs vanilla MCTS |
| Tuning | Best config improvement | >20% | vs default config |
| Coverage | Test coverage | >80% | pytest-cov |

---

*Last Updated: 2026-01-26*
*Version: 1.0.0*
