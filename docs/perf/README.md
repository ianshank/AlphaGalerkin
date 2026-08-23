# Performance Baselines

This directory contains versioned performance baselines and regression-tracking ground truth for AlphaGalerkin.

## Baseline Files

- `baseline_v1.json`: Version 1 performance baseline document tracking latency, throughput, VRAM usage, and accuracy across benchmarks and resolution transfers.

## Recording Baselines

To record a new performance baseline:

```bash
python -m src.poc.cli record-baseline \
    --config config/scenarios/poc_full.yaml \
    --output docs/perf/baseline_v1.json
```

## Comparing Against Baselines

To diff a run against recorded baselines:

```bash
python -m src.poc.cli diff \
    --baseline docs/perf/baseline_v1.json \
    --run outputs/poc_latest/
```

Baselines are JSON documents with explicit schema versioning (`PERF_BASELINE_DOCUMENT_SCHEMA_VERSION`). Unversioned or legacy files migrate cleanly via `_migrate_baseline_document`.
