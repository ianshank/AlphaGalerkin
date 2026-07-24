# Use Cases

Illustrative applications of AlphaGalerkin across its two domains. Snippets are
schematic — see the API docs and [`ARCHITECTURE.md`](../ARCHITECTURE.md) for exact
signatures.

## 1. Research: resolution-independent learning

Study how knowledge transfers across board sizes — train on small boards (faster
iteration), evaluate on unseen larger ones. Research questions: does influence
understanding transfer 9×9 → 19×19? What is the minimum training size for
effective 19×19 play? How does spectral filtering affect transfer quality?

## 2. Education: learn on small boards

A teaching tool that demonstrates concepts learned on any board size — start on
9×9, move to 13×13, then full 19×19 with the same model, e.g.

```bash
python -m src.tools.cli gtp --board-size 9 --model teacher_model.pt
```

## 3. Fast prototyping: accelerated MCTS

Use FNet-only mixing for fast rollouts, fitting more simulations into the same
time budget for rapid game analysis.

## 4. Hybrid systems

Use AlphaGalerkin for global strategy/influence assessment and a traditional
engine for local tactical verification, combining the two scores.

## 5. Tournament play: GTP-compatible engine

Compete via the Go Text Protocol with standard GUIs (Sabaki, GoGui, Lizzie, KaTrain):

```bash
python -m src.tools.cli gtp --model tournament_model.pt --board-size 19 --device cuda
```

## 6. Analysis: position evaluation

Analyze games with policy (likely moves), value (win probability), and the LBB
constant (model confidence) per position.

## 7. Curriculum learning

Train efficiently by progressing through board sizes (5 → 7 → 9 → 13 → 19), so the
model learns progressively more complex positions. See `src/curriculum/`.

## 8. Embedded / edge inference

Use a minimal configuration with FNet mixing and dynamic quantization for
deployment on edge devices (Raspberry Pi, Jetson).

## 9. LLM-prior MCTS for out-of-distribution PDEs

Guide MCTS basis selection with a generalist LLM (e.g. Qwen-14B via LM Studio) so
search survives PDE families a domain-trained evaluator has never seen. The
`llm_prior_ablation` scenario benchmarks random / trained / LLM arms on in- and
out-of-distribution PDEs:

```bash
pip install -e '.[lm-studio]'
python -m src.poc.cli run --config config/scenarios/llm_prior_demo.yaml
```

The scenario is GPU-only and gates the LLM arm gracefully when LM Studio's
preflight fails. See the LLM-prior rows in the
[Regression Surface](../CLAUDE.md#regression-surface) for the exact acceptance metrics.
