---
name: claims-ledger
description: Add, correct or retract a numeric claim in the charter's evidence register end-to-end — artifact, provenance sidecar, register row, and the guard that enforces it. Use whenever a headline number is introduced, changed or withdrawn.
---

# Add, correct or retract a claim

The charter's evidence register is this project's claims ledger. It already exists; do **not**
build a parallel `CLAIMS.yaml`. Extend the register and its guard in
`tests/docs/test_charter_alignment.py`.

Three headline numbers have been retracted here. In each case the number was written down
before, or instead of, the artifact that would support it.

## Adding or correcting a claim

1. **Produce the artifact first.** A committed `results/*.csv` from a script under `scripts/`,
   not a notebook cell and not a local run. The number follows the artifact.
2. **Write the provenance sidecar** — see the `run-provenance` skill.
3. **Check the artifact contains what the claim asserts.** A comparison claim needs an artifact
   holding *both arms*. The charter's adaptive-vs-uniform row cited a CSV whose `method` column
   held only `{dorfler, mcts}`; a correct number traced to prose, and the existence guard could
   not see it because the file existed.
4. **Regenerate the artifact-freeze manifest.** `results/MANIFEST.sha256` freezes the bytes of
   every committed `results/*.csv`, `results/*.run.json` and `config/baselines/*.json` (PNGs
   presence-only) and is verified by CI's `lint` job, `make artifact-manifest`, and
   `tests/docs/test_artifact_manifest.py`. It is regenerated **only** here and in
   `run-provenance` — i.e. when an artifact is deliberately produced or replaced:

   ```bash
   python -m scripts.artifact_manifest write     # rewrites results/MANIFEST.sha256
   python -m scripts.artifact_manifest check     # exit 0; sha256sum -c results/MANIFEST.sha256 also works
   git add results/MANIFEST.sha256               # commit it WITH the artifact and its sidecar
   ```

   Any other change to a frozen artifact is a defect; the guard names the file that moved.
5. **Add the register row** between the `<!-- charter:evidence -->` markers. State the measured
   range, not a remembered band — "1.5× at 56 DOF rising to 10.5× at 2847" beats "5–9×" and is
   harder to drift.
6. **Prefer a rate to a ratio** where one exists. A convergence exponent does not depend on
   where the reader takes the reading; a ratio does.
7. **Verify**: `pytest tests/docs/test_charter_alignment.py tests/docs/test_artifact_manifest.py -v`

## Live runs dirty the tree by design

The shipped scenario YAMLs under `config/scenarios/` default to `output_dir: results`, so the
documented live-run commands (`python -m src.poc.cli run --config ...` and the
`scripts/run_*.py` harnesses without `--output-dir`) **overwrite the committed artifacts in
place**. That is deliberate — it is how a headline artifact gets replaced — but it means an
exploratory run leaves `results/` modified and the manifest check red. CI never writes
`results/` (`transfer-baseline-regression` writes and uploads `outputs/transfer_ci`; E2E
tests write `tmp_path`), and `tests/docs/test_artifact_manifest.py` asserts that.

- **Exploring / reproducing**: prefer `--output-dir outputs/<name>` (gitignored), e.g.
  `python -m scripts.run_transfer_baseline_compare --config config/scenarios/transfer_baseline_compare_ci.yaml --output-dir outputs/transfer_ci`.
  `poc.cli run` has no such flag; copy the YAML and set `output_dir` instead.
- **Replacing the headline artifact**: run into `results/` on purpose, then do step 4 above.
  A modified `results/` file with an unchanged `results/MANIFEST.sha256` is the guard doing
  its job, not a flaky test.

## Retracting a claim

1. **Add the constant** to `tests/support/cut_modules.py` — one definition, several guards.
2. **Correct every live statement.** Guards scan different surfaces: the charter, `README.md`
   + `docs/related-work.md`, `dashboard/` + `hf_space/`, and `docs/business/` +
   `docs/doe_genesis/`. A retraction that propagates to some and not others is how one survived
   in `PRIOR_ART_REVIEW.md` after the charter had already corrected it.
3. **Leave the retraction visible.** House style is to record it *in place* — see
   `specs/lshape_amr_compare.spec.md` — not to edit history away. Guards allow a retracted
   string when the surrounding block carries a marker word.
4. **Add the guard**, then **mutation-test it**: restore the defect and confirm a *named* test
   fails.

## Two ways a guard silently does nothing

Both have happened here; check for both.

- **It scans an empty set.** A guard whose roots or vocabulary match nothing passes everything.
  Add a meta-test asserting it examines at least one real subject — and prefer *coverage* of the
  vocabulary over "something matched", because one arm still matching can mask another going
  unexamined.
- **The mutation never applied.** A mutation lost to shell quoting (an umlaut, say) produces a
  false negative indistinguishable from a passing guard. Assert the file actually changed before
  trusting the result.

## Exemptions

Any allowlist entry needs a reason **and** a meta-test asserting it is still required, so a stale
exemption fails rather than rotting into a permanent blind spot. When the drafted exemption turns
out to be unnecessary, that is the meta-test working — delete it.
