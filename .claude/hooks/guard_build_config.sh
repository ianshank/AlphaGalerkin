#!/usr/bin/env bash
# PostToolUse guard: when a build/CI config file (or a file the guards read as
# data) is edited, run the hermetic guard modules that READ that file.
#
# Why this exists: every one of this repo's recorded invisibility defects lived
# in one of the paths below, and every one was found by a person reading the
# file rather than by a check. The 2026-09-11 red build (CLAUDE.md's eval-harness
# row said `--cov-fail-under=1` while ci.yml enforced 96) was caught by
# tests/docs/test_claude_coverage_gates.py in CI -- a guard that existed, read
# CLAUDE.md as data, and never ran locally because this hook (a) did not fire
# on CLAUDE.md and (b) ran a hand-maintained list of four guard modules that
# predated every guard added that day.
#
# So the guard set is DERIVED, not listed: every module under tests/docs/ and
# tests/claude/ whose source mentions the edited file (by the keyword table in
# select_guards below) is selected. A new guard that reads ci.yml is picked up
# the moment it lands; a renamed file changes the keyword table, not a list.
#
# Cost is the reason this is path-gated rather than universal: the selected
# modules take ~10-60 s (torch import + a few subprocess probes), so firing on
# every edit would be intolerable; these paths change a handful of times per
# branch. No network, no GPU.
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
  */CLAUDE.md|CLAUDE.md) ;;
  *.claude/settings.json|*.claude/hooks/*|*.claude/skills/*|*.claude/agents/*|*.claude/commands/*) ;;
  *) exit 0 ;;
esac

# Guard-module selection. Printed in dry-run so the DECISION is testable in
# milliseconds (tests/claude/test_harness_validation.py drives this script
# with synthetic payloads and ALPHAGALERKIN_HOOK_DRY_RUN=1).
selected="$(python - "$path" <<'PY'
import re
import sys
from pathlib import Path

path = sys.argv[1]
name = Path(path).name
is_workflow = ".github/workflows/" in path
is_harness = "/.claude/" in path or path.startswith(".claude/")

# Keyword table: what a guard module that READS the edited file must mention.
# Regexes, matched against module source. Workflow YAMLs are read through the
# shared parser (tests/support/workflows.py), so importing it counts too.
KEYWORDS: dict[str, list[str]] = {
    "workflow": [r"\bci\.yml\b", r"tests\.support\.workflows", r"WORKFLOW_DIR", r"CI_WORKFLOW"],
    "Makefile": [r"\bMakefile\b", r"makefile_"],
    "pyproject.toml": [r"pyproject"],
    "conftest.py": [r"\bconftest\b"],
    "CLAUDE.md": [r"CLAUDE\.md", r"CLAUDE_MD"],
    ".pre-commit-config.yaml": [r"pre-commit"],
}
key = "workflow" if is_workflow else name
patterns = [re.compile(p) for p in KEYWORDS.get(key, [])]

candidates = sorted(Path("tests/docs").glob("test_*.py")) + sorted(Path("tests/claude").glob("test_*.py"))
chosen: list[str] = []
for module in candidates:
    text = module.read_text(encoding="utf-8", errors="replace")
    if is_harness and module.parent.name == "claude":
        chosen.append(module.as_posix())
    elif any(p.search(text) for p in patterns):
        chosen.append(module.as_posix())
print(" ".join(dict.fromkeys(chosen)))
PY
)"

if [ -z "$selected" ]; then
  echo "[guard] ${path##*/}: no guard module reads this file -- nothing to run"
  exit 0
fi

if [ "${ALPHAGALERKIN_HOOK_DRY_RUN:-}" = "1" ]; then
  echo "[hook] would run: enforcement checks for $path"
  echo "[hook] selected: $selected"
  exit 0
fi

echo "[guard] ${path##*/} touched -- running $(printf '%s\n' $selected | wc -l) guard module(s) that read it"
# shellcheck disable=SC2086
python -m pytest -q -p no:randomly --no-header $selected 2>&1 | tail -12
exit 0
