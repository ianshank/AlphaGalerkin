# Extracting `src/video_compression/` into a Standalone Repository

Ready-to-execute guide for carving the neural video codec out of AlphaGalerkin
into its own repo **with history preserved**. This round is documentation only —
no code is moved or deleted. Every claim below was verified against the tree at
extraction time; `file:line` citations are provided so a reviewer can re-check.

---

## 1. Summary

Extraction is **low-risk** because the codec is a leaf in the dependency graph:

- **Size / shape.** 60 Python files, **18,384 LOC**, 12 subpackages
  (`codec`, `runtime`, `training`, `metrics`, `zoo`, `models`, `data`, `mcts`,
  `perf`, `demo`, `utils`, plus top-level `config.py`).
  Verified: `find src/video_compression -name '*.py' | wc -l` → 60;
  `find src/video_compression -name '*.py' -exec cat {} + | wc -l` → 18384;
  `find src/video_compression -maxdepth 1 -type d` → 11 dirs + `config.py`.
- **Zero inbound coupling.** No module under `src/` outside the package imports
  `src.video_compression`. A repo-wide grep for `src.video_compression` returns
  only files *inside* the package itself, plus `scripts/*.py`, `tests/**`, and
  `pyproject.toml` — never a core `src/pde|mcts|poc|training|modeling` module.
- **Already CI-isolated.** The package is `--ignore`d in the global unit pass
  (`.github/workflows/ci.yml:128,260`), fully excluded from the global coverage
  gate (`pyproject.toml:412-431`), carries per-module mypy relaxations
  (`pyproject.toml:355-363`), and is validated by a dedicated workflow
  (`.github/workflows/codec-perf-coverage.yml`).
- **Carries no thesis weight.** AlphaGalerkin's thesis is resolution-independent
  **PDE solving** via Galerkin/FNet operators (see `CLAUDE.md` Project Overview).
  The codec reuses the same operator ideas but is the single largest package in
  the repo while contributing nothing to the PDE/MCTS headline results. It is the
  cleanest candidate for extraction.

Only **three** thin outward dependency areas exist, all trivially vendorable
(Section 3).

---

## 2. Coupling Analysis

### 2a. Inbound imports (rest-of-repo → codec): **ZERO from `src/`**

Command run:

```bash
grep -rn "src\.video_compression\|from video_compression\|import video_compression" \
  --include='*.py' . | grep -v "^\./src/video_compression/"
```

Every hit lives in one of three **auxiliary** (non-core) locations that travel
with the codec or are handled explicitly during the split — **no core solver
module appears**:

| Consumer | Files | Disposition |
|---|---|---|
| Codec CLI scripts | `scripts/benchmark_codec.py`, `scripts/train_compression.py`, `scripts/train_compression_zoo.py`, `scripts/train_compression_zoo_entry.py`, `scripts/encode_video.py`, `scripts/decode_video.py`, `scripts/demo_compression.py` | Move to new repo (Section 4/5) |
| Codec tests | `tests/video_compression/**`, `tests/scripts/test_train_compression_zoo*.py`, `tests/integration/test_video_workflow.py` | Move to new repo |
| Build/CI config | `pyproject.toml`, `.github/workflows/*` | Trim (Section 6) |

No `src/pde/*`, `src/mcts/*`, `src/poc/*`, `src/training/*`, `src/modeling/*`
imports the package. Inbound coupling into the core solver is genuinely zero.

### 2b. Outward imports (codec → rest-of-repo): exactly **3 areas**

Command run:

```bash
grep -rn "from src\.\|import src\." src/video_compression | \
  grep -v "src\.video_compression"
```

All non-`src.video_compression` `src.*` imports (17 sites) fall into three areas:

| # | Dependency | Symbols used | Sites (`file:line`) |
|---|---|---|---|
| 1 | `src.constants` | `CHECKPOINT_BEST` (= `"best.pt"`, `src/constants.py:113`) | `training/trainer.py:24` (1 usage) |
| 2 | `src.poc.device` | `resolve_device` (aliased `_resolve_bare`) | `perf/device.py:21` (1 usage) |
| 3a | `src.templates.config` | `BaseModuleConfig` (all), `TrainableModuleConfig` (only at `config.py:14`) | `config.py:14`, `perf/config.py:20`, `runtime/metadata.py:29`, `runtime/protocol.py:29`, `metrics/baselines.py:38`, `zoo/config.py:24`, `zoo/dataset_spec.py:30`, `zoo/bdrate.py:26`, `zoo/rdcurve.py:38`, `zoo/h265_baseline.py:26` (**10 usages**) |
| 3b | `src.templates.logging` | `create_logger_class` | `perf/benchmark.py:30`, `zoo/sweep.py:35`, `training/zoo_trainer.py:19` (**3 usages**) |
| 3c | `src.templates.base` | `BaseExecutable`, `ExecutionResult`, `ExecutionStatus` | `perf/benchmark.py:29` (**1 usage**, 3 symbols) |
| 3d | `src.templates.registry` | `create_registry` | `runtime/registry.py:34` (**1 usage**) |

Total outward surface: **1 constant, 1 function, 6 classes/factories** — the
entire external footprint. Symbol definitions verified: `BaseModuleConfig`
(`src/templates/config.py:121`), `TrainableModuleConfig` (`:226`),
`ExecutionStatus` (`src/templates/base.py:43`), `ExecutionResult` (`:70`),
`BaseExecutable` (`:192`), `create_logger_class` (`src/templates/logging.py:313`),
`create_registry` (`src/templates/registry.py:240`).

---

## 3. Shared-Deps Shim Plan

Two options per area: **vendor** (copy the exact symbols into the new repo) or
**publish `alphagalerkin-common`** (a tiny shared PyPI/GitHub lib both repos
depend on).

### Recommendation: **VENDOR** into `video_codec/_vendor/`

Rationale:
- The surface is tiny and stable (7 symbols total). A shared lib adds a release
  cadence, version-pinning, and a second CI target for near-zero payoff.
- `src/templates/*` is generic module-infra (Pydantic base config, registry
  factory, structlog wrapper, executable base) with **no PDE/Go/codec coupling**
  — safe to copy verbatim.
- `resolve_device` and `CHECKPOINT_BEST` are leaf utilities with trivial bodies.
- Vendoring keeps the codec repo installable with **zero dependency on
  AlphaGalerkin**, which is the entire point of extraction.

Revisit `alphagalerkin-common` only if a *third* consumer of `src/templates`
ever needs to be extracted; until then vendoring is strictly simpler.

### What to copy

Create `video_codec/_vendor/` and copy:

1. **`src/templates/config.py`** → `_vendor/templates_config.py`
   Export `BaseModuleConfig`, `TrainableModuleConfig`.
2. **`src/templates/base.py`** → `_vendor/templates_base.py`
   Export `BaseExecutable`, `ExecutionResult`, `ExecutionStatus`.
3. **`src/templates/logging.py`** → `_vendor/templates_logging.py`
   Export `create_logger_class` (+ its `BaseModuleLogger` return type).
4. **`src/templates/registry.py`** → `_vendor/templates_registry.py`
   Export `create_registry`.
5. **`resolve_device`** from `src/poc/device.py` → `_vendor/device.py`.
   `perf/device.py` already wraps it as `_resolve_bare` to add `cuda:N`
   parsing (`perf/device.py:21`), so only the *bare* `resolve_device`
   (`auto`/`cuda`/`cpu`) body is needed.
6. **`CHECKPOINT_BEST = "best.pt"`** (`src/constants.py:113`) → a one-line
   constant in `_vendor/constants.py` (the only `src.constants` name the codec
   references — confirmed by grep: `training/trainer.py:24` is the sole usage).

> Note: copy `src/templates/config.py` wholesale rather than hand-trimming —
> `BaseModuleConfig`/`TrainableModuleConfig` may pull in helpers within the same
> module; a wholesale copy avoids a partial-symbol break. Same for `logging.py`
> (`create_logger_class` returns `BaseModuleLogger`).

---

## 4. History-Preserving Split Commands

### Option A (preferred): `git filter-repo`

`git filter-repo` rewrites a fresh clone, keeping only codec-relevant paths and
their full commit history. Install: `pip install git-filter-repo`.

```bash
# 0. Fresh mirror clone (never run filter-repo on your working checkout)
git clone https://github.com/<org>/AlphaGalerkin.git video_codec-extract
cd video_codec-extract

# 1. Keep only codec source, tests, scripts, configs, and its workflow.
git filter-repo \
  --path src/video_compression/ \
  --path tests/video_compression/ \
  --path tests/scripts/test_train_compression_zoo.py \
  --path tests/scripts/test_train_compression_zoo_entry.py \
  --path tests/scripts/test_train_compression_zoo_report.py \
  --path tests/integration/test_video_workflow.py \
  --path scripts/benchmark_codec.py \
  --path scripts/train_compression.py \
  --path scripts/train_compression_zoo.py \
  --path scripts/train_compression_zoo_entry.py \
  --path scripts/encode_video.py \
  --path scripts/decode_video.py \
  --path scripts/demo_compression.py \
  --path config/perf/ \
  --path config/video_compression/ \
  --path .github/workflows/codec-perf-coverage.yml

# 2. Relocate the package to the new top-level import name `video_codec`.
git filter-repo \
  --path-rename src/video_compression/:video_codec/ \
  --path-rename tests/video_compression/:tests/

# 3. Point origin at the new empty GitHub repo and push.
git remote add origin https://github.com/<org>/video-codec.git
git push -u origin --all
git push -u origin --tags
```

### Option B (alternative): `git subtree split`

Preserves history for a single subtree; you then graft tests/scripts manually.

```bash
cd AlphaGalerkin

# 1. Split src/video_compression/ history onto a detached branch.
git subtree split --prefix=src/video_compression -b codec-only

# 2. Create the new repo from that branch.
mkdir ../video-codec && cd ../video-codec
git init
git pull ../AlphaGalerkin codec-only     # lands package at repo root

# 3. Repeat the split for tests/scripts and merge them in, OR cherry-pick.
#    (subtree splits one prefix at a time — filter-repo is less fiddly for the
#    multi-path case, hence Option A is preferred.)
```

### New-repo bootstrap: `pyproject.toml`

Ship a self-contained `pyproject.toml` with a **dedicated codec extra** (these
deps currently live in AlphaGalerkin's shared `test-extras`,
`pyproject.toml:63-69`, with no codec-specific extra):

```toml
[project]
name = "video-codec"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.1",
    "numpy>=1.24",
    "pydantic>=2.0",
    "structlog>=24.1",
    "einops>=0.7",
]

[project.optional-dependencies]
# Was folded into AlphaGalerkin's shared `test-extras`; now a first-class codec extra.
codec = [
    "onnx>=1.15",
    "onnxruntime>=1.17",
    "onnxscript>=0.1",   # required by torch>=2.x torch.onnx.export dynamo path
    "torchvision>=0.15", # VGG perceptual loss
]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.5", "mypy>=1.10"]

[project.scripts]
# AlphaGalerkin only exposes `alphagalerkin = "src.tools.cli:main"` (pyproject.toml:132-133);
# there is no codec console entry point today. Add one here.
video-codec = "video_codec.perf.benchmark:main"   # or a new video_codec/cli.py

[tool.setuptools.packages.find]
include = ["video_codec*"]

[tool.coverage.run]
branch = true
source = ["video_codec"]
```

Move `.github/workflows/codec-perf-coverage.yml` unchanged except path prefixes
(`src/video_compression/perf/**` → `video_codec/perf/**`, drop the
`--ignore=tests/video_compression/` rationale comment). The zoo/phase coverage
gates are enforced today only via the per-module `pytest --cov-fail-under`
commands documented in `CLAUDE.md`'s *Regression Surface* rows — promote those
into new workflow jobs (`codec-zoo-coverage.yml`, `codec-runtime-coverage.yml`)
in the standalone repo.

> **Verification note / discrepancy:** the AlphaGalerkin `pyproject.toml:416-417`
> omit comment references an *"upcoming `phase2-zoo-validation.yml`"*, but no such
> workflow file exists yet — `.github/workflows/` contains only `ci.yml` and
> `codec-perf-coverage.yml`. The zoo/runtime coverage gates are currently CLI-only
> (`CLAUDE.md` Regression Surface). Create the missing zoo/runtime workflow jobs
> fresh in the new repo; do not expect to `git mv` an existing file.

---

## 5. New-Repo Build Checklist

Run inside the extracted repo:

1. **Vendored shims in place.** Create `video_codec/_vendor/` and copy the 6
   items from Section 3. Add `video_codec/_vendor/__init__.py` re-exporting the
   public names.
2. **Rewrite outward imports** (the only 17 sites from Section 2b):
   ```bash
   # templates.config  →  _vendor
   grep -rl "from src.templates.config import" video_codec | xargs \
     sed -i 's/from src\.templates\.config import/from video_codec._vendor.templates_config import/'
   # templates.logging / base / registry, poc.device, constants — repeat per area:
   sed -i 's/from src\.templates\.logging import/from video_codec._vendor.templates_logging import/'   ...
   sed -i 's/from src\.templates\.base import/from video_codec._vendor.templates_base import/'         ...
   sed -i 's/from src\.templates\.registry import/from video_codec._vendor.templates_registry import/' ...
   sed -i 's/from src\.poc\.device import resolve_device/from video_codec._vendor.device import resolve_device/' video_codec/perf/device.py
   sed -i 's/from src\.constants import CHECKPOINT_BEST/from video_codec._vendor.constants import CHECKPOINT_BEST/' video_codec/training/trainer.py
   ```
3. **Rewrite internal package prefix** `src.video_compression` → `video_codec`
   across all moved source, tests, and scripts:
   ```bash
   grep -rl "src\.video_compression" video_codec tests scripts | xargs \
     sed -i 's/src\.video_compression/video_codec/g'
   ```
4. **Declare extras** — the `[codec]` extra (Section 4) so `onnx*` / `torchvision`
   install; base install stays lean.
5. **Move/author workflows** — `codec-perf-coverage.yml` with rewritten paths;
   add zoo/runtime coverage-gate jobs.
6. **Verify green:**
   ```bash
   pip install -e '.[dev,codec]'
   ruff check video_codec/
   mypy video_codec/ --strict          # relaxations from pyproject.toml:355-363 no longer apply — expect to fix or re-add per-module overrides
   pytest tests/ -m "not gpu_required" -q
   ```
7. **Port the codec section of `CLAUDE.md`** into the new repo's `AGENT.md` /
   `README.md` (Phase 0–2-D milestones + Regression Surface rows).

---

## 6. What Stays Behind in AlphaGalerkin

After the codec repo is live, remove the now-dead codec scaffolding from
AlphaGalerkin (a **separate** follow-up PR — not this round):

### `pyproject.toml`
- **Coverage omit** — delete the 13 codec lines in the `[tool.coverage.run] omit`
  block (`pyproject.toml:419-431`) and their preamble comment (`:413-418`).
- **mypy relaxations** — delete the 9 `src.video_compression.*` module names from
  the per-module override list (`pyproject.toml:355-363`).
- **`test-extras`** — `onnx`, `onnxruntime`, `onnxscript`, `torchvision`
  (`pyproject.toml:65-69`) exist *only* for the codec's ONNX/TensorRT/perceptual
  paths (`scikit-fem`/`pettingzoo` in the same extra serve FEM/PettingZoo and
  stay). Drop the four codec deps unless another consumer is confirmed —
  `src/deployment/export_onnx.py` also uses `onnx*`, so **keep `onnx*`** if the
  deployment ONNX path is retained; `torchvision` is codec-only and can go.
  Verify before deleting: `grep -rn "torchvision\|import onnx" src/ | grep -v video_compression`.
- **mypy `[[overrides]]` ignore-missing** for `onnx*`/`torchvision`
  (`pyproject.toml:219-243`) — trim entries no longer needed after the above.

### `.github/workflows/`
- Delete `codec-perf-coverage.yml` (moved to the new repo).
- Remove the `test-video-compression` job from `ci.yml` (steps around
  `ci.yml:456-461`) and the two `--ignore=tests/video_compression/` lines
  (`ci.yml:128,260`) once `tests/video_compression/` is gone.
- Drop the codec line from the `test-extras` job (`ci.yml:551`).

### `CLAUDE.md`
- Trim the *Neural Video Compression* section and its *Key Architecture
  Decisions* subsection, the six Self-Hosted Transcoder milestone entries
  (Phase 0 / 1 / 2-B / 2-D), the codec *Known Issues* entry (MCTS rate-control
  skips), and the five codec *Regression Surface* rows (Codec perf, Decoder
  runtime, Codec model zoo, Codec sweep orchestrator). Leave a one-line pointer
  to the new repo.
- Remove the *Video Compression Commands* block.

### Blast-radius confirmation
No core `pde` / `mcts` / `poc` / `training` / `modeling` path imports the
package (Section 2a), so deleting `src/video_compression/` and
`tests/video_compression/` breaks **nothing** in the PDE/MCTS/PoC surfaces. The
only follow-on edits are the config/CI/doc trims above and removing the seven
codec `scripts/*.py` entry points.

### Integration-engineer confirmation (`src/integrations/` orthogonality)
Verified independently as part of this extraction prep:

- `grep -rn "src.integrations" src/video_compression/` → **no matches**: the
  codec does not import the LM Studio / OpenAI-compatible client, the
  backend-profile registry, or any `src/integrations` symbol.
- `grep -rn "video_compression" src/integrations/` → **no matches**: the
  integrations package does not depend on the codec either.
- `grep -rn "lm_studio\|openai" src/video_compression/` → **no matches**: the
  `[lm-studio]` optional extra (and the lazily-imported `openai` SDK) is fully
  orthogonal to the codec; extracting the codec neither pulls nor drops it.

Conclusion: the codec and the third-party-integration layer are decoupled in
both directions, so the extraction does not touch `src/integrations/` or its
`[lm-studio]` extra.
