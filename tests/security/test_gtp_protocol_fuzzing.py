"""Property-based fuzzing tests for GTP protocol engine.

This file uses Hypothesis to fuzz the GTP engine with various malicious
and unexpected inputs to ensure it never crashes (returns gracefully handled errors).
"""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.tools.gtp import GTPEngine

pytestmark = pytest.mark.security

@pytest.fixture
def engine() -> GTPEngine:
    """Create a mock GTPEngine instance for fuzzing.

    We pass dummy/None dependencies since we only test the parser.
    """
    # GTPEngine typically expects a model, board_size, device
    return GTPEngine(model=None, board_size=19, device="cpu")

@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    input_str=st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)),  # avoid surrogates
        min_size=0,
        max_size=10000,
    )
)
def test_gtp_engine_text_fuzzing(engine: GTPEngine, input_str: str) -> None:
    """Fuzz the GTP engine with random text strings to check for unhandled exceptions."""
    try:
        response = engine.process_command(input_str)
        # Any response is valid as long as it didn't crash
        assert isinstance(response, str)
    except Exception as e:
        pytest.fail(f"GTP engine crashed on text input: {e}")

@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    binary_data=st.binary(min_size=1, max_size=10000)
)
def test_gtp_engine_binary_fuzzing(engine: GTPEngine, binary_data: bytes) -> None:
    """Fuzz the GTP engine with random binary data.

    Checks for buffer overflow, null byte injections, and unicode decoding issues.
    """
    try:
        # Since GTP takes strings, we decode with errors='replace' to simulate
        # garbage data being passed into the parser layer.
        input_str = binary_data.decode(errors="replace")
        response = engine.process_command(input_str)
        assert isinstance(response, str)
    except Exception as e:
        pytest.fail(f"GTP engine crashed on binary input: {e}")

@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    command=st.sampled_from(["play", "genmove", "known_command", "boardsize", "clear_board"]),
    args=st.lists(st.text(), min_size=0, max_size=50)
)
def test_gtp_engine_command_injection_fuzzing(
    engine: GTPEngine, command: str, args: list[str]
) -> None:
    """Fuzz known commands with massive or unexpected arguments."""
    # Build a command string with random arguments
    input_str = f"{command} {' '.join(args)}"
    try:
        response = engine.process_command(input_str)
        assert isinstance(response, str)
    except Exception as e:
        # The engine should handle bad arguments with GTP error responses, not crashes
        pytest.fail(f"GTP engine crashed on command injection: {e}")
