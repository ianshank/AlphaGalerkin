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

from src.training.checkpoint import load_torch_checkpoint


def inspect(path: str, *, allow_unsafe_pickle: bool = False) -> None:
    """Inspect a PyTorch checkpoint file and print its structure.

    Args:
        path: Checkpoint file to read.
        allow_unsafe_pickle: Deserialize with ``weights_only=False``. Executes
            arbitrary code if the file is malicious; only for a file whose
            provenance you have established.

    """
    print(f"Loading {path}")
    try:
        data = load_torch_checkpoint(path, allow_unsafe_pickle=allow_unsafe_pickle)
    # Broad by intent: this is a reporting tool whose whole job is to say what a
    # file contains, including when the answer is 'it will not deserialize'.
    except Exception as e:
        print("Error:", e)
        return

    if isinstance(data, dict):
        print("Keys:", data.keys())
    else:
        print("Not a dict, type:", type(data))


def main() -> None:
    """Parse arguments and inspect the named checkpoint."""
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
    args = parser.parse_args()
    inspect(args.path, allow_unsafe_pickle=args.allow_unsafe_pickle)


if __name__ == "__main__":
    main()
