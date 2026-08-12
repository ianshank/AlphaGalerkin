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
# on some base images. Forcing stdlib distutils lets its wheel build --- BUT
# Python 3.12 removed the stdlib distutils module entirely (PEP 632), so the
# override becomes fatal: setuptools.monkey imports distutils.filelist on load
# and the whole `pip install -e '.[dev]'` fails before torch or anything else
# can install. Only apply the override on interpreters that still ship distutils
# (< 3.12). Callers can force either behaviour by exporting the variable
# themselves before invoking the hook.
if [[ -z "${SETUPTOOLS_USE_DISTUTILS-}" ]]; then
  if python -c 'import sys; sys.exit(0 if sys.version_info < (3, 12) else 1)' >/dev/null 2>&1; then
    export SETUPTOOLS_USE_DISTUTILS=stdlib
    echo "[session-start] SETUPTOOLS_USE_DISTUTILS=stdlib (Python < 3.12)"
  else
    # On Python >= 3.12 rely on the modern setuptools build path
    # (setuptools >= 68 vendors what distutils used to provide).
    echo "[session-start] SETUPTOOLS_USE_DISTUTILS unset (Python >= 3.12; stdlib distutils removed by PEP 632)"
  fi
else
  echo "[session-start] SETUPTOOLS_USE_DISTUTILS=${SETUPTOOLS_USE_DISTUTILS} (honouring caller override)"
fi

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

# Install the pre-commit git hook so local commits run the same ruff lint/format
# that CI enforces (config: .pre-commit-config.yaml). Non-fatal; pre-commit ships
# in the dev extra installed above.
if command -v pre-commit >/dev/null 2>&1; then
  if pre-commit install >/dev/null 2>&1; then
    echo "[session-start] pre-commit hook installed"
  else
    echo "[session-start] WARNING: 'pre-commit install' failed (non-fatal)"
  fi
else
  echo "[session-start] pre-commit: NOT AVAILABLE (provided by pip install -e '.[dev]')"
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
