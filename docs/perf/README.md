# Performance Baselines

This directory holds versioned baselines for the codec performance benchmark.

## Files

| File | What it is | Schema |
|---|---|---|
| `baseline_v1.json` | Headline GPU baseline (per `config/perf/default.yaml`) | `BaselineDocument` v1 |
| `baseline_smoke_v1.json` | CPU smoke baseline (per `config/perf/smoke.yaml`) | `BaselineDocument` v1 |

## Recording a baseline

```bash
python -m scripts.benchmark_codec record-baseline \
    --config config/perf/default.yaml \
    --output docs/perf/baseline_v1.json \
    --hardware-tag <gpu-model> \
    --description "Recorded on <date> against commit <sha>"
```

## Regression-gating an existing baseline

```bash
python -m scripts.benchmark_codec run \
    --config config/perf/default.yaml \
    --baseline docs/perf/baseline_v1.json \
    --output reports/perf_$(git rev-parse --short HEAD).json \
    --tolerance 5.0
```

A non-zero exit code means at least one regression was detected. Inspect
the `regression` artifact in the output report for details.

## Schema migration

Old baselines remain readable under `BaselineRegistry.load()`:

* unversioned files are migrated to v1 on read
* unknown fields are ignored (forward-compat)
* baselines newer than the running binary trigger a `ValueError`

When the schema changes, add a migration step in
`src/video_compression/perf/baseline.py::_migrate_baseline_document` and
bump `PERF_BASELINE_DOCUMENT_SCHEMA_VERSION` in `config.py`.
