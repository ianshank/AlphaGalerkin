---
name: run-provenance
description: Write or verify the .run.json provenance sidecar for a committed benchmark artifact. Use whenever a results/ file is created or regenerated, or when auditing whether an existing artifact can be traced to the code that produced it.
---

# Run provenance for committed artifacts

Artifact *existence* is not provenance. `results/lshape_mcts_vs_dorfler.csv` carries exactly one
provenance column — `seed`. Not the search mode, not the marking fraction, not a git SHA. It
cannot be dated against the 2026-08-16 backup fix, and the harness that produced it still
exposes the mode that generated the retracted number. That is the gap this closes.

## Writing one

```python
from src.research.run_manifest import (
    ArmProvenance, RunManifest, collect_git_provenance, collect_package_versions,
    manifest_path_for, write_run_manifest,
)

write_run_manifest(
    RunManifest(
        run_id=...,                       # stable, derived from the config
        harness="scripts.run_<name>",     # what produced it
        config=vars(args),                # the full resolved config
        git=collect_git_provenance(),     # never raises
        packages=collect_package_versions(),
        seeds=[...],
        arms=[ArmProvenance(name=..., parameters={...}, counters={...})],
        metrics={...},
        artifacts={"csv": str(output)},
        notes="what these numbers do and do NOT establish",
    ),
    manifest_path_for(output),            # results/<stem>.run.json
)
```

`notes` is not decoration. It is where a reader learns that a ratio above 1 means the adaptive
arm is *worse*, or that a figure predates a caching change and is not comparable.

## Rules that are load-bearing

- **Collectors never raise.** `collect_git_provenance` and `collect_package_versions` degrade to
  `unknown` on any failure. A provenance collector that throws destroys the run it documents.
- **`dirty` is tri-state.** `None` means "could not be determined" and is deliberately distinct
  from `False`.
- **Do not invent provenance.** For an artifact predating the module, add it to
  `_ARTIFACTS_WITHOUT_PROVENANCE` in `tests/docs/test_charter_alignment.py` with a reason it
  cannot be reconstructed. A meta-test asserts the entry is still needed, so regenerating the
  artifact forces the list to shrink.
- **`stable_fields()` for comparisons.** It drops timestamps, hardware tags and counters, which
  differ between two runs of identical code.
- **Pin CSV line endings to LF.** `csv.writer` defaults to CRLF.

## The trap

`.gitignore` carries a blanket `*.json`. Without the `!results/**/*.json` negation the sidecar is
**silently never committed** — no error, the artifact simply lands alone as though the module did
not exist. Guarded by `tests/research/test_run_manifest.py`; if you add a new artifact directory,
check its JSON is committable before trusting that a file you wrote is in the tree.

## Verify

```bash
pytest tests/research/test_run_manifest.py -v
pytest tests/docs/test_charter_alignment.py -v     # cited CSVs must carry a sidecar
git status --short results/                        # the sidecar must actually stage
```
