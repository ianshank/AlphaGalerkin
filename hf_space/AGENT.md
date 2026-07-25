# `hf_space/` — HuggingFace Space deploy bundle (developer notes)

This directory is the self-contained bundle deployed to the HuggingFace Space
[`ianshank/alphagalerkin-demo`](https://huggingface.co/spaces/ianshank/alphagalerkin-demo).
`deploy_space.py` (repo root) uploads the whole folder to the Space via
`HfApi.upload_folder`. The public-facing description + Space configuration live in
[`README.md`](README.md) (its YAML frontmatter sets the Gradio SDK version, app
file, etc.). **This `AGENT.md` is for maintainers, not Space visitors.**

## `hf_space/src/` and `hf_space/config/` are a *manual, partial* mirror

`hf_space/src/` and `hf_space/config/` are a hand-maintained copy of the
repository's top-level `src/` and `config/` trees so the Space is importable
without installing the package. **Treat them as a mirror that drifts, not as a
second source of truth:**

- **Partial.** The mirror intentionally omits packages the demo does not need.
  As of this writing `hf_space/src/` is missing `agents/`, `alphagalerkin/`,
  `backend/`, `engines/`, `integrations/`, and `refinement/` relative to `src/`.
- **Independently formatted.** The mirror was formatted with an older `ruff`, so
  files differ from `src/` cosmetically even where logic matches.
- **Already diverged in content.** Some files have drifted materially from their
  `src/` counterparts (e.g. `mcts/search.py`). **Do not assume parity** — if you
  fix a bug in `src/`, it is *not* automatically reflected here.
- **Has Space-only modules.** The mirror also contains modules that do **not**
  exist in the main `src/` at all — `game_manager.py`, `endgame.py`,
  `rendering/` — which back the interactive Go demo.

### What keeps it honest (and what doesn't)

`tests/hf_space/test_mirror_guard.py` is a **floor** guard, run on the normal CPU
CI surface. It asserts only that:

1. every `src.*` / `config.*` module `app.py` imports resolves to a file in the
   mirror (so the Space can't `ImportError` on launch),
2. every `.py` under `hf_space/` parses, and
3. the mirror stays scrubbed of the 2026-07-22 "cut to the core" modules and the
   retracted `0.000209` transfer figure.

It deliberately does **not** assert byte/logic parity with `src/` — the trees
have already diverged, so a parity check would fail today and force a full
reconciliation. **Fully single-sourcing the mirror** (a build-time sync step, or
pruning it to exactly what `app.py` imports) is a tracked follow-up, intentionally
out of scope for the guard.

## `checkpoint.pt`

`checkpoint.pt` (~7 MB) is the demo model weights: `app.py` loads it
(`MODEL_PATH = Path("checkpoint.pt")`) and `deploy_space.py` ships it with the
folder. Notes for maintainers:

- It is routed through **HuggingFace Xet** on deploy (`hf_space/.gitattributes`:
  `*.pt filter=xet …`), *not* Git LFS. In the GitHub repo it is stored as a plain
  binary blob, which is why it is tracked despite the root `.gitignore` `*.pt`
  rule (and why it slips past the `check-added-large-files --maxkb=1000`
  pre-commit hook). Do **not** convert it to Git LFS — that would fight the Xet
  filter used for the actual Space deploy.
- `app.py` can also fetch the checkpoint from the Hub at runtime
  (`_ensure_checkpoint` → `hf_hub_download`). A future cleanup could rely on that
  path to drop the committed blob from the GitHub mirror entirely; that is a
  deploy-behavior change and is left as a follow-up, not done here.

## `requirements.txt`

Pinned to the **HuggingFace Space runtime**, independently of `pyproject.toml`:
`gradio==4.44.1` matches the `sdk_version` declared in `README.md`'s frontmatter,
and `pydantic==2.10.6` tracks the Space runtime. These pins **intentionally
diverge** from the main project's (`gradio>=6.0.0`, `pydantic<2.10`); bumping them
is a breaking change against `app.py` and the Space SDK, so align only when
re-testing the deployed Space.
