# AGENT.md - PoC Scenario Framework Module (`src/poc/`)

## Persona

**Name**: Validation Scientist
**Expertise**: Experimental design, statistical significance testing, hyperparameter optimization, reproducibility, benchmark orchestration
**Mindset**: You design and execute rigorous validation experiments. Every claim (zero-shot transfer, O(N) complexity, LBB stability) must be backed by reproducible scenarios with statistical significance testing. Configuration drives everything — no hardcoded values.

## Module Overview

This module provides a configuration-driven framework for running Proof-of-Concept validation scenarios. It includes three built-in scenarios (transfer, complexity, stability), a scenario runner with sequential/parallel execution and retry logic, result persistence with run comparison, hyperparameter tuning (TPE/grid/random), and statistical analysis (t-test, Mann-Whitney, bootstrap, effect sizes, multiple comparison correction).

## Design Patterns

### 1. Registry + Decorator (Scenario Discovery)
```python
@scenario("transfer")
class TransferScenario(BaseScenario):
    config_class = TransferScenarioConfig
    def execute(self) -> ScenarioResult: ...
```
Auto-registration via decorator. No central registration file. Thread-safe singleton registry.

### 2. Template Method (Scenario Lifecycle)
`BaseScenario.run()` provides the lifecycle skeleton:
```
setup() → execute() → teardown()
```
Subclasses implement `execute()` which returns a `ScenarioResult` directly. Helper methods `_evaluate_thresholds()` and `_create_result()` are available for subclasses to call from within `execute()` but are not part of the base `run()` skeleton. Setup/teardown are optional overrides.

### 3. Strategy Pattern (Execution Mode)
`ScenarioRunner` supports sequential and parallel (ThreadPoolExecutor) execution strategies, switchable via config.

### 4. Strategy Pattern (Statistical Tests)
`StatisticalAnalyzer` dispatches to different test implementations:
- Parametric: t-test
- Non-parametric: Mann-Whitney U
- Resampling: Bootstrap, Permutation

### 5. Factory Pattern (Tuning)
```python
tuner = HyperparameterTuner(config, TransferScenario)
result = tuner.tune()  # Runs N trials with TPE/grid/random sampling
```

### 6. Configuration as Code (Pydantic)
Every scenario parameter flows through validated Pydantic models. Configs support deterministic SHA256 hashing for reproducibility.

### 7. Observer Pattern (Results)
`ResultCollector` persists results immediately after each scenario completes. Fault-tolerant — partial results survive crashes.

## Skills Required

- **Experimental design**: Control variables, significance thresholds, effect sizes
- **Statistical testing**: t-test, Mann-Whitney, bootstrap CI, Bonferroni/Holm/FDR correction
- **Hyperparameter optimization**: TPE (Tree-structured Parzen Estimators), grid search, random search
- **Scenario design**: Defining metrics, thresholds, pass/fail criteria
- **Parallel execution**: ThreadPoolExecutor, timeout handling, retry with backoff
- **Configuration management**: Pydantic schemas, YAML loading, hash-based reproducibility

## Sub-Agents

| Sub-Agent | Scope | When to Invoke |
|-----------|-------|----------------|
| **Transfer Scenario Expert** | `scenarios/transfer.py` | Zero-shot transfer validation experiments |
| **Complexity Scenario Expert** | `scenarios/complexity.py` | O(N) scaling benchmarks |
| **Stability Scenario Expert** | `scenarios/stability.py` | LBB condition monitoring |
| **Statistical Analyst** | `statistics/significance.py` | Significance testing, effect sizes, corrections |
| **Tuning Specialist** | `tuning/` | Hyperparameter search configuration |
| **Runner Engineer** | `runner.py` | Execution orchestration, retry logic |
| **Results Manager** | `results.py` | Persistence, comparison, summary generation |

## Tools & Commands

```bash
# Run PoC framework tests
pytest tests/poc/ -v

# CLI commands
python -m src.poc.cli list
python -m src.poc.cli info transfer
python -m src.poc.cli run --scenario transfer
python -m src.poc.cli run --config config/scenarios/poc_full.yaml
python -m src.poc.cli run --parallel 4
python -m src.poc.cli compare run_a run_b
```

## Key Files

| File | Purpose | Key Classes |
|------|---------|-------------|
| `config.py` | Pydantic configuration | `BaseScenarioConfig`, `TransferScenarioConfig`, `ComplexityScenarioConfig`, `StabilityScenarioConfig`, `ScenarioResult`, `MetricThreshold` |
| `registry.py` | Scenario registration | `ScenarioRegistry`, `BaseScenario`, `@scenario()` |
| `runner.py` | Execution engine | `ScenarioRunner` |
| `results.py` | Result persistence | `ResultCollector` |
| `logging.py` | Structured logging | `ScenarioLogger`, `DebugContext` |
| `cli.py` | CLI entry point | `cmd_run()`, `cmd_list()`, `cmd_info()`, `cmd_compare()` |
| `scenarios/transfer.py` | Zero-shot transfer | `TransferScenario` |
| `scenarios/complexity.py` | O(N) complexity | `ComplexityScenario` |
| `scenarios/stability.py` | LBB stability | `StabilityScenario` |
| `tuning/config.py` | Tuning configuration | `TuningConfig`, `SearchSpace` |
| `tuning/sampler.py` | Parameter sampling | `BaseSampler` (ABC), `RandomSampler`, `GridSampler`, `TPESampler` |
| `tuning/tuner.py` | Tuning orchestrator | `HyperparameterTuner`, `TuningResult`, `TrialResult` |
| `statistics/significance.py` | Statistical analysis | `SignificanceTest`, `StatisticalAnalyzer`, `ComparisonResult`, `EffectSizeResult` |

## Dependencies

**Internal**: `src.modeling` (attention/FNet for complexity scenario), `src.math_kernel` (GalerkinProjection for stability scenario), `src.physics.poisson` (data for transfer scenario), `src.experiments.physics_model` (PhysicsOperator for transfer scenario). Note: `src.training` and `src.templates` are NOT imported — the PoC module implements its own config, registry, and logging independently.
**External**: `pydantic`, `structlog`, `numpy`, `scipy` (statistical tests), `torch`, `yaml`

## Conventions & Constraints

1. **No Hardcoded Values**: Every parameter must flow through Pydantic config. Use `Field()` with constraints.
2. **Deterministic Hashing**: Configs produce SHA256 hashes for reproducibility. Same config = same hash.
3. **Immediate Persistence**: Results save to JSON after each scenario. Never batch result persistence.
4. **Metric Thresholds**: Define pass/fail via `MetricThreshold(metric, operator, value)`. Evaluated automatically.
5. **Tier System**: `ScenarioTier` classifies runtime: UNIT (~seconds), FUNCTIONAL (~minutes), INTEGRATION (~hours).
6. **Retry Logic**: Runner supports exponential backoff retries with fresh scenario instances per attempt.
7. **Structured Logging**: `ScenarioLogger` is available for scenario implementations. Note: core framework files (`runner.py`, `results.py`, `registry.py`, `cli.py`) use raw `structlog.get_logger()` directly.

## Execution Flow

```
CLI (python -m src.poc.cli run --scenario transfer)
  └→ ScenarioRunner.run("transfer")
       └→ ScenarioRegistry.get("transfer")
            └→ TransferScenario(config)
                 ├→ setup()           # Initialize model, data
                 ├→ execute()         # Train, evaluate, build ScenarioResult
                 └→ teardown()        # Cleanup
       └→ ResultCollector.collect(result)
            └→ Save JSON to output_dir/results/{run_id}/
```

## Adding a New Scenario

1. Create `src/poc/scenarios/my_scenario.py`
2. Define `MyScenarioConfig(BaseScenarioConfig)` with Pydantic validation
3. Implement `@scenario("my_scenario") class MyScenario(BaseScenario)`
4. Override `execute()` with validation logic, call `self.record_metric()` for results
5. Add config entry in `config/scenarios/` YAML files
6. Add tests in `tests/poc/`
