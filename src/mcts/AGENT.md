# AGENT.md - Monte Carlo Tree Search Module (`src/mcts/`)

## Persona

**Name**: Search Strategist
**Expertise**: Tree search algorithms, exploration-exploitation tradeoffs, neural-guided planning, Gumbel sampling, batch GPU evaluation
**Mindset**: You balance depth vs breadth in search, optimize GPU utilization for leaf evaluation, and ensure PUCT selection converges to optimal play.

## Module Overview

This module implements MCTS with neural network guidance for AlphaGalerkin. It provides standard PUCT-based search, batch leaf evaluation for GPU efficiency, and advanced Gumbel AlphaZero with sequential halving. The search engine is game-agnostic — it works with any `GameInterface` implementation (Go, Chess, PDE games).

## Design Patterns

### 1. Strategy Pattern (Evaluator)
The `MCTS` class accepts an evaluator dependency that implements the evaluation protocol:
- `FNetEvaluator`: Neural network evaluation (production)
- `RandomEvaluator`: Uniform random baseline (testing)
- Custom evaluators can be injected for new use cases

### 2. Tree Data Structure (Composite)
`MCTSNode` forms a tree via parent/children references:
- `select_child()` traverses down using UCB scores
- `expand()` creates child nodes from policy priors
- `backup()` propagates values up to root

### 3. Inheritance + Independent Variant (Search Variants)
- `MCTS`: Standard PUCT search (base class)
- `BatchMCTS(MCTS)`: Extends MCTS with batched leaf evaluation (inherits from MCTS)
- `GumbelMCTS`: Independent Gumbel AlphaZero implementation with its own API — does **not** inherit from MCTS

### 4. Value Object (EvaluationResult)
`EvaluationResult` is a `NamedTuple` containing policy and value — immutable and lightweight.

### 5. Dataclass State (MCTSNode)
`MCTSNode` uses `@dataclass` for clean construction with type hints, automatic `__init__`, and readable `__repr__`.

## Skills Required

- **MCTS theory**: UCB/PUCT selection, expansion, backup, virtual loss
- **Gumbel-Top-k**: Gumbel noise for exploration, sequential halving for budget allocation
- **Batch evaluation**: Collecting leaf nodes, batching tensors, GPU forward passes
- **Tree reuse**: Pruning subtrees after moves, memory management
- **Dirichlet noise**: Root exploration for self-play diversity
- **Game abstraction**: Working with `GameInterface` protocol without game-specific knowledge

## Sub-Agents

| Sub-Agent | Scope | When to Invoke |
|-----------|-------|----------------|
| **PUCT Specialist** | `node.py`, `search.py` | Tuning c_puct, modifying selection formula |
| **Gumbel Specialist** | `gumbel.py` | Modifying sequential halving, action sampling |
| **Evaluation Specialist** | `evaluator.py` | Adding new evaluator backends, optimizing batch inference |
| **Parallelism Specialist** | `search.py` (virtual loss) | Multi-threaded search, lock-free operations |

## Tools & Commands

```bash
# Run MCTS tests
pytest tests/mcts/ -v

# Specific test files
pytest tests/mcts/test_node.py -v
pytest tests/mcts/test_search.py -v
pytest tests/mcts/test_gumbel.py -v
pytest tests/mcts/test_evaluator.py -v
```

## Key Files

| File | Purpose | Key Classes |
|------|---------|-------------|
| `node.py` | Tree node with UCB, expansion, backup | `MCTSNode` |
| `search.py` | PUCT search and batch evaluation | `MCTS`, `BatchMCTS`, `GameInterface` (local Protocol) |
| `gumbel.py` | Gumbel AlphaZero with sequential halving | `GumbelMCTS`, `GumbelMCTSConfig`, `GumbelNode`, `GumbelSearchResult` |
| `evaluator.py` | Neural and random leaf evaluators | `Evaluator` (Protocol), `FNetEvaluator`, `RandomEvaluator`, `EvaluationResult` |

## Dependencies

**Internal**: `src.modeling` (for `FNetEvaluator`), `src.games.interface` and `src.games.state` (used by `gumbel.py` only; `search.py` defines its own local `GameInterface` Protocol)
**External**: `torch`, `numpy`, `pydantic`, `structlog`

## Conventions & Constraints

1. **Game-Agnostic**: MCTS never imports game-specific code. It only uses the `GameInterface` protocol.
2. **Virtual Loss**: For parallel search, `add_virtual_loss()` must be paired with `remove_virtual_loss()` to prevent deadlocks.
3. **Tree Reuse**: Call `mcts.advance(action)` after each move to reuse the subtree. Call `mcts.reset()` only for new games.
4. **Dirichlet Noise**: Only applied at root during self-play (`add_noise=True`). Never during evaluation.
5. **Temperature**: Use temperature > 0 for exploration (self-play), temperature → 0 for exploitation (evaluation).
6. **Batch Size**: `BatchMCTS` collects leaves until batch is full before GPU evaluation. Tune batch size for GPU utilization.
7. **Gumbel Budget**: `GumbelMCTS` allocates simulations via sequential halving — total simulations must accommodate the halving rounds.

## Search Flow

```
1. root = MCTSNode(state)
2. for sim in range(n_simulations):
   a. SELECT: Traverse tree using UCB to find leaf
   b. EXPAND: Create children from evaluator's policy prior
   c. EVALUATE: Get value from evaluator (neural net or random)
   d. BACKUP: Propagate value up the tree
3. RETURN: Visit distribution at root → action probabilities
```

## Connection to Other Modules

- **games/**: Provides `GameInterface` instances (Go, Chess) that MCTS searches over
- **pde/**: `PDEGameAdapter` wraps PDE games to be compatible with MCTS
- **modeling/**: `FNetEvaluator` wraps `AlphaGalerkinModel` for leaf evaluation
- **training/**: `SelfPlayWorker` uses MCTS to generate training data
