# Spec: <Feature Name>

> **Status:** Draft | Accepted | Implemented | Runbook
> **Owner:** <name/role>
> **Primary module(s):** `src/<pkg>/<module>.py`
> **Config class:** `src.<pkg>.<ConfigClass>` (the executable data contract)
> **Tracking:** <PR / milestone reference>

## Context

Why this feature exists — the problem or gap it addresses, what prompted it, and the intended
outcome. Link the roadmap source (CLAUDE.md Next-Steps row, ROI plan tier, milestone).

## User Story

**As a** <role>,
**I want** <capability>,
**so that** <value>.

## Data Contract

The feature is configured by `<ConfigClass>` (a `BaseModuleConfig` / `BaseScenarioConfig`
subclass). Every tunable is a typed Pydantic `Field(default=…, <bounds>, description=…)` — **no
hardcoded values**. List the fields that matter to this contract:

| Field | Type | Default | Bounds | Meaning |
|---|---|---|---|---|
| `<field>` | `<type>` | `<default>` | `<ge/le/gt/lt>` | <description> |

Named module-level constants for numerical-stability literals (mirror
`DEFAULT_TRANSFER_RATIO_FLOOR`, `EVAL_SEED_STRIDE`): list any here.

## Acceptance Criteria

Concrete, testable Given/When/Then statements. Each AC maps to at least one test.

### AC1: <name>
- **Given** <precondition>
- **When** <action>
- **Then** <observable outcome>

### AC2: <name>
- **Given** …
- **When** …
- **Then** …

## Thresholds

Pass/fail criteria expressed as `src.poc.config.MetricThreshold(name, operator, value)` tuples.
These must be exactly what the feature's `get_default_thresholds()` (or equivalent) returns — the
config is the single source of truth; this table documents it, and the AQA test asserts agreement.

| Metric | Operator | Value | Meaning |
|---|---|---|---|
| `<metric_name>` | `>=` / `<=` / `<` / `>` / `==` | `<value>` | <what passing means> |

## Regression Surface

The test command(s) that must stay green for this feature (added to the CLAUDE.md
*Regression Surface* table on implementation):

```bash
pytest tests/<pkg>/test_<feature>.py -v -m "not gpu_required"
# per-module coverage gate
pytest tests/<pkg>/test_<feature>.py --cov=src/<pkg>/<module> --cov-branch --cov-fail-under=85
```

## Out of Scope

What this spec deliberately does **not** cover (deferred items, hardware-gated runs, follow-on
phases). Name the follow-up spec/milestone where deferred work lands.
