#!/usr/bin/env bash
# PostToolUse guard: resolve internal Markdown links when a .md file is edited.
#
# The same check runs as a pre-commit hook, so this adds no new enforcement --
# it moves the feedback from commit time to authorship time, typically ~20 edits
# earlier. It costs 0.10 s, which is why that is worth doing.
#
# Known limit, stated so it is not oversold: `check_doc_links.py` resolves
# `[](...)` links only, never backticked inline paths. It would NOT have caught
# the Regression Surface rows that documented a command CI does not run, nor an
# orphan doc that nothing links to.
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
  *.md) ;;
  *) exit 0 ;;
esac

# Dry-run: report the gating decision and skip the work. Exists so the
# decision itself is testable in milliseconds -- the real command below
# costs seconds, and a test that pays that on every parametrised path is a
# test someone deletes. Also lets a developer check "would this fire?"
# without waiting.
if [ "${ALPHAGALERKIN_HOOK_DRY_RUN:-}" = "1" ]; then
  echo "[hook] would run: doc-link check for $path"
  exit 0
fi

python scripts/check_doc_links.py 2>&1 | tail -5
exit 0
