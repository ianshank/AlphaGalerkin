"""Print the top-level structure of a checkpoint file.

Deserializes through :func:`src.training.checkpoint.load_torch_checkpoint`, so a
file is inspected under ``weights_only=True`` rather than unpickled. That matters
more here than in a training entry point: the reason to reach for this script is
usually that a checkpoint is unfamiliar or suspect, which is exactly when running
its contents is least acceptable.

Usage:
    python -m scripts.inspect_checkpoint <path>
    python -m scripts.inspect_checkpoint <path> --allow-unsafe-pickle
"""

from __future__ import annotations

import argparse
import sys
from typing import Final

from src.training.checkpoint import load_torch_checkpoint

#: Process exit codes. Named because the failure one was previously absent
#: entirely: `inspect()` caught every exception, printed "Error:", and returned
#: `None`, and `main()` returned `None` too -- so the script exited **0 whether
#: the checkpoint deserialized or not**. A caller scripting around it (or a CI
#: step, or an E2E journey asserting that a hostile payload is refused) could not
#: tell success from failure by the only signal a shell reads.
EXIT_OK: Final[int] = 0
EXIT_LOAD_FAILED: Final[int] = 1


def inspect(path: str, *, allow_unsafe_pickle: bool = False) -> int:
    """Inspect a PyTorch checkpoint file and print its structure.

    Args:
        path: Checkpoint file to read.
        allow_unsafe_pickle: Deserialize with ``weights_only=False``. Executes
            arbitrary code if the file is malicious; only for a file whose
            provenance you have established.

    Returns:
        :data:`EXIT_OK` when the file deserialized, else :data:`EXIT_LOAD_FAILED`.
        The error is still *reported* rather than raised -- this remains a
        reporting tool, and a traceback is not more useful here than the message
        -- but the outcome is now visible to the caller.

    """
    print(f"Loading {path}")
    try:
        data = load_torch_checkpoint(path, allow_unsafe_pickle=allow_unsafe_pickle)
    # Broad by intent: this is a reporting tool whose whole job is to say what a
    # file contains, including when the answer is 'it will not deserialize'.
    except Exception as e:
        print("Error:", e)
        return EXIT_LOAD_FAILED

    if isinstance(data, dict):
        print("Keys:", data.keys())
    else:
        print("Not a dict, type:", type(data))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Split out of ``main`` -- matching the ``build_parser()`` convention six other
    scripts here already follow -- so a test can assert the *real* parser's
    defaults. The test that covers this used to rebuild an equivalent parser of
    its own and assert against that, which passes no matter what this script
    does: it would still have been green with the flag deleted from here.
    """
    parser = argparse.ArgumentParser(description="Inspect a checkpoint file's structure")
    parser.add_argument("path", help="Path to the checkpoint file")
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help=(
            "Load with weights_only=False. Executes arbitrary code if the "
            "checkpoint is malicious; use only for a file whose provenance you "
            "have established."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and inspect the named checkpoint.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``. Accepting it makes
            the entry point testable in-process, matching the ``main(argv)``
            convention the other scripts here follow.

    Returns:
        The process exit code.

    """
    args = build_parser().parse_args(argv)
    return inspect(args.path, allow_unsafe_pickle=args.allow_unsafe_pickle)


if __name__ == "__main__":
    sys.exit(main())
