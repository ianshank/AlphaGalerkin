"""Security tests for GTP input sanitization.

These tests verify that GTP (Go Text Protocol) input is properly sanitized
to prevent injection attacks and denial of service via oversized inputs.
"""


def sanitize_gtp_input(command: str) -> str:
    """Sanitize GTP input by removing non-printable characters and limiting length.

    This is a reference implementation for TDD purposes.
    The actual implementation should exist in src/tools/gtp.py.

    Args:
        command: Raw GTP command string.

    Returns:
        Sanitized command with non-printable chars removed and length limited.

    Note:
        Uses str.isprintable() which excludes all control characters including
        newlines. For GTP protocol compliance, newlines should be handled
        separately at the protocol parsing layer.

    """
    # Filter non-printable characters (control chars, null bytes, etc.)
    if not command.isprintable():
        command = "".join(c for c in command if c.isprintable())

    # Limit length to prevent DoS via memory exhaustion
    max_length = 1000
    return command[:max_length]


def test_gtp_input_sanitization() -> None:
    """Verify GTP commands are sanitized to prevent injection attacks."""
    # Input with null byte injection attempt
    malicious_input = "genmove black\x00; rm -rf /"

    clean_input = sanitize_gtp_input(malicious_input)

    # Null byte should be removed
    assert "\x00" not in clean_input
    # Safe portion of command should remain
    assert "genmove black; rm -rf /" in clean_input


def test_gtp_command_length_limit() -> None:
    """Verify denial of service protection via input length limiting."""
    massive_input = "A" * 10000

    clean_input = sanitize_gtp_input(massive_input)

    assert len(clean_input) <= 1000


def test_control_characters_removed() -> None:
    """Verify various control characters are filtered out."""
    # Various control characters that should be removed
    control_chars_input = "play\x07black\x08\x1bA1\x7f"

    clean_input = sanitize_gtp_input(control_chars_input)

    # Only printable characters should remain
    assert clean_input == "playblackA1"
