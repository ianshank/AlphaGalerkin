#!/usr/bin/env bash
# SessionStart hook for AlphaGalerkin (Claude Code on the web / CLI).
#
# Purpose: make a fresh, ephemeral session able to lint, type-check, and run the
# CPU test surface. The container clones the repo but does not install the
# package; the biggest first-run risk is that PyTorch (a heavy dependency) is
# absent, so nothing under src/ imports. This hook installs the dev extra and
# reports the resulting toolchain state without failing the session if the
# install is slow or partially unavailable behind a proxy.
#
# Environment is aligned with .github/workflows/ci.yml so local runs match CI.
set -uo pipefail

export MPLBACKEND="${MPLBACKEND:-Agg}"          # headless matplotlib
export WANDB_MODE="${WANDB_MODE:-disabled}"     # no W&B API key needed
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"    # reproducible builds

# antlr4-python3-runtime==4.9.3 (transitive via hydra-core -> omegaconf) ships
# sdist-only and its setup.py trips the Debian setuptools 'install_layout' bug
# on some base images. Forcing stdlib distutils lets its wheel build. Harmless
# where the bug is absent.
export SETUPTOOLS_USE_DISTUTILS="${SETUPTOOLS_USE_DISTUTILS:-stdlib}"

echo "[session-start] AlphaGalerkin environment bootstrap"

if python -c "import torch" >/dev/null 2>&1; then
  echo "[session-start] torch present — skipping reinstall"
else
  echo "[session-start] installing dev extra (pip install -e '.[dev]') ..."
  if pip install -e '.[dev]' >/tmp/alphagalerkin_bootstrap.log 2>&1; then
    echo "[session-start] dev extra installed"
  else
    echo "[session-start] WARNING: 'pip install -e .[dev]' did not complete;"
    echo "[session-start]          see /tmp/alphagalerkin_bootstrap.log."
    echo "[session-start]          Markdown/.claude deliverables can proceed;"
    echo "[session-start]          torch-dependent tests will be unavailable."
  fi
fi

# Report toolchain availability (non-fatal).
for tool in ruff mypy pytest; do
  if command -v "$tool" >/dev/null 2>&1; then
    echo "[session-start] $tool: $($tool --version 2>&1 | head -1)"
  else
    echo "[session-start] $tool: NOT AVAILABLE"
  fi
done

echo "[session-start] CPU test surface: pytest -m 'not gpu_required'"
echo "[session-start] done"
