#!/usr/bin/env bash
# PostToolUse guard: when a build/CI config file is edited, run the hermetic
# checks that decide whether anything else in this repo is enforced.
#
# Why this exists: every one of this repo's seven recorded invisibility defects
# lived in one of the five paths below, and every one was found by a person
# reading the file rather than by a check. The most recent -- Make's `-` prefix
# silencing half the E2E tier -- shipped in one commit and was caught by an
# external review bot in the next.
#
# Cost is the reason this is path-gated rather than universal. The suite takes
# ~13 s (most of it the torch import), so firing it on every edit would be
# intolerable; firing it only on these five paths is roughly six times per
# branch. It runs no subprocess tests, needs no network and no GPU.
#
# Never blocks: this reports, it does not gate. CI is the gate. A hook that can
# stop work gets disabled, and a disabled hook checks nothing.
set -uo pipefail

payload="$(cat)"
path="$(printf '%s' "$payload" | python -c '
import json, sys
try:
    print(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))
except Exception:
    print("")
' 2>/dev/null)"

case "$path" in
  *.github/workflows/*.yml|*.github/workflows/*.yaml) ;;
  */Makefile|Makefile) ;;
  *pyproject.toml) ;;
  *.pre-commit-config.yaml) ;;
  */conftest.py|conftest.py) ;;
  *) exit 0 ;;
esac

# Dry-run: report the gating decision and skip the work. Exists so the
# decision itself is testable in milliseconds -- the real command below
# costs seconds, and a test that pays that on every parametrised path is a
# test someone deletes. Also lets a developer check "would this fire?"
# without waiting.
if [ "${ALPHAGALERKIN_HOOK_DRY_RUN:-}" = "1" ]; then
  echo "[hook] would run: enforcement checks for $path"
  exit 0
fi

echo "[guard] build config touched: ${path##*/} -- running enforcement checks"
python -m pytest -q -p no:randomly --no-header \
  tests/docs/test_e2e_visibility.py \
  tests/docs/test_marker_vocabulary.py \
  tests/docs/test_coverage_gate_integrity.py \
  tests/claude/test_harness_validation.py 2>&1 | tail -12
exit 0
