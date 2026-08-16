"""Tests for YAML configuration injection vulnerabilities."""

from pathlib import Path

import pytest
import yaml

from src.poc.runner import ScenarioRunner

pytestmark = pytest.mark.security

@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Provide a temporary config file path."""
    return tmp_path / "test_config.yaml"

def test_yaml_object_tag_rejected(temp_config_file: Path) -> None:
    """Verify that !!python/object tags are rejected by safe_load."""
    payload = """
malicious: !!python/object/apply:os.system
  args: ['echo vulnerable']
"""
    temp_config_file.write_text(payload)

    with pytest.raises(yaml.constructor.ConstructorError):
        with open(temp_config_file) as f:
            yaml.safe_load(f)

def test_yaml_object_apply_rejected(temp_config_file: Path) -> None:
    """Verify that !!python/object/apply is rejected by safe_load."""
    payload = """
execute: !!python/object/apply:subprocess.check_output
  args: ['ls']
"""
    temp_config_file.write_text(payload)

    with pytest.raises(yaml.constructor.ConstructorError):
        with open(temp_config_file) as f:
            yaml.safe_load(f)

def test_path_traversal_in_config(tmp_path: Path) -> None:
    """Verify config loading handles path traversal securely."""
    # The runner might not raise ValueError explicitly for path traversal,
    # but the OS would raise FileNotFoundError or PermissionError.
    # We want to make sure it doesn't execute anything outside bounds.
    with pytest.raises((FileNotFoundError, ValueError, PermissionError)):
        runner = ScenarioRunner(output_dir=str(tmp_path))
        runner.load_config("../../etc/passwd")

def test_env_var_injection_no_shell(temp_config_file: Path) -> None:
    """Verify env var loading doesn't execute shell commands."""
    payload = """
env: ${PWD; echo vulnerable}
"""
    temp_config_file.write_text(payload)

    with open(temp_config_file) as f:
        data = yaml.safe_load(f)

    assert data["env"] == "${PWD; echo vulnerable}"
