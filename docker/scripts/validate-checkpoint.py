#!/usr/bin/env python3
"""CLI entrypoint for sandboxed checkpoint validation.

This script is designed to run inside the validator container with
maximum security restrictions (no network, read-only filesystem, etc.).

Usage:
    python validate.py /path/to/checkpoint.pt
    python validate.py /path/to/checkpoint.pt --expected-hash abc123...
    python validate.py /path/to/checkpoint.pt --strict
    python validate.py /path/to/checkpoint.pt --output json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, "/app")

from src.safety.config import (
    ValidationConfig,
    ValidationLevel,
    get_permissive_config,
    get_standard_config,
    get_strict_config,
)
from src.safety.validator import CheckpointValidator, ValidationResult


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate PyTorch checkpoints for security issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic validation
    python validate.py model.pt

    # Strict validation with hash verification
    python validate.py model.pt --strict --expected-hash abc123...

    # Output as JSON
    python validate.py model.pt --output json

Security Note:
    This validator should be run in an isolated container with:
    - No network access (--network none)
    - Read-only filesystem (--read-only)
    - No capabilities (--cap-drop=ALL)
    - Memory limits (--memory=1g)
        """,
    )

    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to checkpoint file to validate",
    )

    parser.add_argument(
        "--expected-hash",
        type=str,
        default=None,
        help="Expected SHA256 hash for verification",
    )

    parser.add_argument(
        "--level",
        type=str,
        choices=["permissive", "standard", "strict"],
        default="standard",
        help="Validation level (default: standard)",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Shortcut for --level strict",
    )

    parser.add_argument(
        "--permissive",
        action="store_true",
        help="Shortcut for --level permissive",
    )

    parser.add_argument(
        "--max-size-gb",
        type=float,
        default=None,
        help="Maximum file size in GB (overrides level default)",
    )

    parser.add_argument(
        "--output",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output result (valid/invalid)",
    )

    return parser.parse_args()


def get_config(args: argparse.Namespace) -> ValidationConfig:
    """Build validation config from arguments."""
    # Determine level
    if args.strict:
        config = get_strict_config()
    elif args.permissive:
        config = get_permissive_config()
    elif args.level == "strict":
        config = get_strict_config()
    elif args.level == "permissive":
        config = get_permissive_config()
    else:
        config = get_standard_config()

    # Apply overrides
    if args.max_size_gb is not None:
        config = config.with_overrides(max_file_size_gb=args.max_size_gb)

    if args.expected_hash:
        config = config.with_overrides(require_hash_verification=True)

    return config


def format_text_output(result: ValidationResult, args: argparse.Namespace) -> str:
    """Format result as human-readable text."""
    lines = []

    # Status line
    status = "VALID" if result.valid else "INVALID"
    lines.append(f"Status: {status}")

    if args.quiet:
        return lines[0]

    lines.append(f"Hash: {result.checkpoint_hash}")
    lines.append(f"Level: {result.validation_level.value}")

    # Errors
    if result.errors:
        lines.append("\nErrors:")
        for error in result.errors:
            lines.append(f"  - {error}")

    # Warnings
    if result.warnings:
        lines.append("\nWarnings:")
        for warning in result.warnings:
            lines.append(f"  - {warning}")

    # Metadata
    if result.metadata:
        lines.append("\nMetadata:")
        for key, value in result.metadata.items():
            if key != "path":  # Skip redundant path
                lines.append(f"  {key}: {value}")

    return "\n".join(lines)


def format_json_output(result: ValidationResult) -> str:
    """Format result as JSON."""
    return json.dumps(result.to_dict(), indent=2)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Check file exists
    if not args.checkpoint.exists():
        if args.output == "json":
            print(json.dumps({"valid": False, "error": "File not found"}))
        else:
            print(f"Error: File not found: {args.checkpoint}", file=sys.stderr)
        return 1

    # Build config and validate
    config = get_config(args)
    validator = CheckpointValidator(config)

    try:
        result = validator.validate(args.checkpoint, args.expected_hash)
    except Exception as e:
        if args.output == "json":
            print(json.dumps({"valid": False, "error": str(e)}))
        else:
            print(f"Validation error: {e}", file=sys.stderr)
        return 2

    # Output result
    if args.output == "json":
        print(format_json_output(result))
    else:
        print(format_text_output(result, args))

    # Exit code: 0 for valid, 1 for invalid
    return 0 if result.valid else 1


if __name__ == "__main__":
    sys.exit(main())
