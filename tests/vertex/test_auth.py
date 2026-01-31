"""Tests for GCP authentication utilities."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.vertex.auth import (
    AuthConfig,
    AuthenticationError,
    AuthMethod,
    CommandResult,
    GCPAuthenticator,
    PlatformInfo,
    ValidationResult,
    create_authenticator,
    detect_platform,
    find_gcloud_path,
    run_gcloud_command,
)


class TestPlatformDetection:
    """Tests for platform detection."""

    def test_detect_windows_powershell(self) -> None:
        """Test Windows PowerShell detection."""
        with patch.object(sys, "platform", "win32"):
            with patch.dict(os.environ, {"PSModulePath": "C:\\Windows\\..."}):
                platform = detect_platform()
                assert platform.is_windows is True
                assert platform.shell_type == "powershell"
                assert platform.use_cmd_wrapper is True
                assert platform.gcloud_executable == "gcloud.cmd"

    def test_detect_windows_cmd(self) -> None:
        """Test Windows CMD detection."""
        with patch.object(sys, "platform", "win32"):
            # Simulate CMD environment without PSModulePath
            cmd_env = {"COMSPEC": "C:\\Windows\\System32\\cmd.exe"}
            with patch.dict(os.environ, cmd_env, clear=True):
                platform = detect_platform()
                assert platform.is_windows is True
                assert platform.shell_type == "cmd"
                assert platform.use_cmd_wrapper is False

    def test_detect_linux_bash(self) -> None:
        """Test Linux bash detection."""
        with patch.object(sys, "platform", "linux"):
            with patch.dict(os.environ, {"SHELL": "/bin/bash"}):
                platform = detect_platform()
                assert platform.is_windows is False
                assert platform.shell_type == "bash"
                assert platform.use_cmd_wrapper is False
                assert platform.gcloud_executable == "gcloud"

    def test_detect_macos_zsh(self) -> None:
        """Test macOS zsh detection."""
        with patch.object(sys, "platform", "darwin"):
            with patch.dict(os.environ, {"SHELL": "/bin/zsh"}):
                platform = detect_platform()
                assert platform.is_windows is False
                assert platform.shell_type == "zsh"


class TestFindGcloudPath:
    """Tests for gcloud path discovery."""

    def test_finds_in_path(self) -> None:
        """Test finding gcloud in PATH."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/gcloud"
            with patch.object(sys, "platform", "linux"):
                path = find_gcloud_path()
                assert path == Path("/usr/bin/gcloud")

    def test_custom_path_preferred(self) -> None:
        """Test custom path takes precedence."""
        custom = Path("/custom/gcloud")
        with patch.object(Path, "exists", return_value=True):
            path = find_gcloud_path(custom)
            assert path == custom

    def test_not_found_returns_none(self) -> None:
        """Test returns None when gcloud not found."""
        with patch("shutil.which", return_value=None):
            with patch.object(Path, "exists", return_value=False):
                path = find_gcloud_path()
                assert path is None


class TestRunGcloudCommand:
    """Tests for gcloud command execution."""

    def test_successful_command(self) -> None:
        """Test successful command execution."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='[{"account": "test@example.com", "status": "ACTIVE"}]',
                stderr="",
            )
            with patch(
                "src.vertex.auth.find_gcloud_path", return_value=Path("/usr/bin/gcloud")
            ):
                result = run_gcloud_command(["auth", "list", "--format=json"])
                assert result.success is True
                assert result.return_code == 0
                assert "test@example.com" in result.stdout

    def test_failed_command(self) -> None:
        """Test failed command execution."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="ERROR: Not authenticated",
            )
            with patch(
                "src.vertex.auth.find_gcloud_path", return_value=Path("/usr/bin/gcloud")
            ):
                result = run_gcloud_command(["auth", "list"])
                assert result.success is False
                assert "Not authenticated" in result.stderr

    def test_timeout_handling(self) -> None:
        """Test command timeout handling."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["gcloud"], timeout=30)
            with patch(
                "src.vertex.auth.find_gcloud_path", return_value=Path("/usr/bin/gcloud")
            ):
                result = run_gcloud_command(["auth", "list"], timeout=30)
                assert result.success is False
                assert "timed out" in result.stderr

    def test_gcloud_not_found(self) -> None:
        """Test handling when gcloud is not found."""
        with patch("src.vertex.auth.find_gcloud_path", return_value=None):
            result = run_gcloud_command(["auth", "list"])
            assert result.success is False
            assert "not found" in result.stderr.lower()

    def test_windows_powershell_direct_execution(self) -> None:
        """Test direct execution on Windows PowerShell.

        Note: subprocess.run with shell=False bypasses PowerShell execution
        policy issues because it doesn't go through the shell.
        """
        platform = PlatformInfo(
            is_windows=True,
            shell_type="powershell",
            use_cmd_wrapper=True,  # This is for reference but not used
            gcloud_executable="gcloud.cmd",
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            with patch(
                "src.vertex.auth.find_gcloud_path", return_value=Path("C:\\gcloud.cmd")
            ):
                run_gcloud_command(["auth", "list"], platform=platform)

                # Verify gcloud is called directly (subprocess with shell=False
                # bypasses PowerShell entirely)
                call_args = mock_run.call_args[0][0]
                assert call_args[0] == "C:\\gcloud.cmd"
                assert call_args[1] == "auth"
                assert call_args[2] == "list"

    def test_file_not_found_error(self) -> None:
        """Test handling of FileNotFoundError."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("gcloud not found")
            with patch(
                "src.vertex.auth.find_gcloud_path", return_value=Path("/usr/bin/gcloud")
            ):
                result = run_gcloud_command(["auth", "list"])
                assert result.success is False
                assert "not found" in result.stderr.lower()


class TestAuthConfig:
    """Tests for AuthConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = AuthConfig()
        assert config.auth_method == AuthMethod.APPLICATION_DEFAULT
        assert config.validate_before_launch is True
        assert config.timeout_seconds == 30
        assert config.retry_count == 2

    def test_service_account_requires_key_path(self) -> None:
        """Test service account method requires key path."""
        with pytest.raises(
            ValidationError, match="service_account_key_path is required"
        ):
            AuthConfig(auth_method=AuthMethod.SERVICE_ACCOUNT_KEY)

    def test_service_account_with_key_path(self, tmp_path: Path) -> None:
        """Test service account with key path is valid."""
        key_path = tmp_path / "key.json"
        key_path.touch()  # Create file so resolve works consistently
        config = AuthConfig(
            auth_method=AuthMethod.SERVICE_ACCOUNT_KEY,
            service_account_key_path=key_path,
        )
        assert config.service_account_key_path is not None
        assert config.service_account_key_path.name == "key.json"

    def test_key_path_normalization(self) -> None:
        """Test key path is normalized."""
        config = AuthConfig(
            auth_method=AuthMethod.SERVICE_ACCOUNT_KEY,
            service_account_key_path="~/keys/sa.json",
        )
        assert config.service_account_key_path is not None
        assert "~" not in str(config.service_account_key_path)

    def test_timeout_minimum_validation(self) -> None:
        """Test timeout minimum validation."""
        with pytest.raises(ValidationError):
            AuthConfig(timeout_seconds=2)  # Below minimum of 5

    def test_timeout_maximum_validation(self) -> None:
        """Test timeout maximum validation."""
        with pytest.raises(ValidationError):
            AuthConfig(timeout_seconds=500)  # Above maximum of 300

    def test_retry_count_validation(self) -> None:
        """Test retry count validation."""
        with pytest.raises(ValidationError):
            AuthConfig(retry_count=10)  # Above maximum of 5

    def test_extra_fields_forbidden(self) -> None:
        """Test extra fields are not allowed."""
        with pytest.raises(ValidationError):
            AuthConfig(unknown_field="value")  # type: ignore[call-arg]


class TestGCPAuthenticator:
    """Tests for GCPAuthenticator."""

    @pytest.fixture
    def authenticator(self) -> GCPAuthenticator:
        """Create authenticator for testing."""
        config = AuthConfig(auth_method=AuthMethod.GCLOUD_CLI)
        return GCPAuthenticator(config)

    def test_validate_gcloud_success(self, authenticator: GCPAuthenticator) -> None:
        """Test successful gcloud validation."""
        with patch("src.vertex.auth.run_gcloud_command") as mock_run:
            mock_run.return_value = CommandResult(
                success=True,
                return_code=0,
                stdout='[{"account": "test@example.com", "status": "ACTIVE"}]',
                stderr="",
                command=["gcloud", "auth", "list"],
            )
            with patch(
                "src.vertex.auth.find_gcloud_path", return_value=Path("/usr/bin/gcloud")
            ):
                result = authenticator.validate_credentials()
                assert result.is_valid is True
                assert result.account == "test@example.com"

    def test_validate_gcloud_no_active_account(
        self, authenticator: GCPAuthenticator
    ) -> None:
        """Test gcloud validation with no active account."""
        with patch("src.vertex.auth.run_gcloud_command") as mock_run:
            mock_run.return_value = CommandResult(
                success=True,
                return_code=0,
                stdout="[]",
                stderr="",
                command=["gcloud", "auth", "list"],
            )
            with patch(
                "src.vertex.auth.find_gcloud_path", return_value=Path("/usr/bin/gcloud")
            ):
                result = authenticator.validate_credentials()
                assert result.is_valid is False
                assert result.error_code == "NO_ACTIVE_ACCOUNT"

    def test_validate_gcloud_not_found(self) -> None:
        """Test gcloud validation when gcloud is not found."""
        config = AuthConfig(auth_method=AuthMethod.GCLOUD_CLI)
        with patch("src.vertex.auth.find_gcloud_path", return_value=None):
            auth = GCPAuthenticator(config)
            result = auth.validate_credentials()
            assert result.is_valid is False
            assert result.error_code == "GCLOUD_NOT_FOUND"

    def test_validate_service_account_success(self, tmp_path: Path) -> None:
        """Test service account validation success."""
        key_file = tmp_path / "key.json"
        key_data = {
            "type": "service_account",
            "client_email": "sa@project.iam.gserviceaccount.com",
            "project_id": "test-project",
        }
        key_file.write_text(json.dumps(key_data))

        config = AuthConfig(
            auth_method=AuthMethod.SERVICE_ACCOUNT_KEY,
            service_account_key_path=key_file,
        )
        auth = GCPAuthenticator(config)
        result = auth.validate_credentials()

        assert result.is_valid is True
        assert result.account == "sa@project.iam.gserviceaccount.com"
        assert result.project == "test-project"

    def test_validate_service_account_file_not_found(self) -> None:
        """Test service account validation with missing file."""
        config = AuthConfig(
            auth_method=AuthMethod.SERVICE_ACCOUNT_KEY,
            service_account_key_path=Path("/nonexistent/key.json"),
        )
        auth = GCPAuthenticator(config)
        result = auth.validate_credentials()

        assert result.is_valid is False
        assert result.error_code == "KEY_NOT_FOUND"

    def test_validate_service_account_invalid_json(self, tmp_path: Path) -> None:
        """Test service account validation with invalid JSON."""
        key_file = tmp_path / "key.json"
        key_file.write_text("not valid json")

        config = AuthConfig(
            auth_method=AuthMethod.SERVICE_ACCOUNT_KEY,
            service_account_key_path=key_file,
        )
        auth = GCPAuthenticator(config)
        result = auth.validate_credentials()

        assert result.is_valid is False
        assert result.error_code == "INVALID_JSON"

    def test_validate_service_account_missing_email(self, tmp_path: Path) -> None:
        """Test service account validation with missing client_email."""
        key_file = tmp_path / "key.json"
        key_data = {"type": "service_account", "project_id": "test-project"}
        key_file.write_text(json.dumps(key_data))

        config = AuthConfig(
            auth_method=AuthMethod.SERVICE_ACCOUNT_KEY,
            service_account_key_path=key_file,
        )
        auth = GCPAuthenticator(config)
        result = auth.validate_credentials()

        assert result.is_valid is False
        assert result.error_code == "INVALID_KEY"

    def test_validate_adc_success(self) -> None:
        """Test ADC validation success."""
        config = AuthConfig(auth_method=AuthMethod.APPLICATION_DEFAULT)
        auth = GCPAuthenticator(config)

        mock_credentials = MagicMock()
        mock_credentials.service_account_email = "sa@project.iam.gserviceaccount.com"

        # Patch at the google.auth module level since import is done inside method
        with patch(
            "google.auth.default",
            return_value=(mock_credentials, "project"),
        ):
            result = auth.validate_credentials()
            assert result.is_valid is True

    def test_validate_adc_not_found(self) -> None:
        """Test ADC validation when no credentials found."""
        config = AuthConfig(auth_method=AuthMethod.APPLICATION_DEFAULT)
        auth = GCPAuthenticator(config)

        # Simulate DefaultCredentialsError
        from google.auth.exceptions import DefaultCredentialsError

        # Patch at the google.auth module level since import is done inside method
        with patch("google.auth.default") as mock_default:
            mock_default.side_effect = DefaultCredentialsError("No credentials")
            result = auth.validate_credentials()
            assert result.is_valid is False
            assert result.error_code == "NO_ADC"

    def test_ps_security_exception_handling(
        self, authenticator: GCPAuthenticator
    ) -> None:
        """Test PSSecurityException is properly handled."""
        with patch("src.vertex.auth.run_gcloud_command") as mock_run:
            mock_run.return_value = CommandResult(
                success=False,
                return_code=1,
                stdout="",
                stderr="PSSecurityException: execution policy",
                command=["gcloud", "auth", "list"],
            )
            with patch(
                "src.vertex.auth.find_gcloud_path", return_value=Path("C:\\gcloud.cmd")
            ):
                result = authenticator.validate_credentials()
                assert result.is_valid is False
                assert result.error_code == "PS_SECURITY_EXCEPTION"
                assert any("cmd /c" in s.lower() for s in result.suggestions)

    def test_expired_credentials_handling(
        self, authenticator: GCPAuthenticator
    ) -> None:
        """Test expired credentials error handling."""
        with patch("src.vertex.auth.run_gcloud_command") as mock_run:
            mock_run.return_value = CommandResult(
                success=False,
                return_code=1,
                stdout="",
                stderr="ERROR: credentials have expired",
                command=["gcloud", "auth", "list"],
            )
            with patch(
                "src.vertex.auth.find_gcloud_path", return_value=Path("/usr/bin/gcloud")
            ):
                result = authenticator.validate_credentials()
                assert result.is_valid is False
                assert result.error_code == "CREDENTIALS_EXPIRED"

    def test_permission_denied_handling(self, authenticator: GCPAuthenticator) -> None:
        """Test permission denied error handling."""
        with patch("src.vertex.auth.run_gcloud_command") as mock_run:
            mock_run.return_value = CommandResult(
                success=False,
                return_code=1,
                stdout="",
                stderr="ERROR: Permission denied",
                command=["gcloud", "auth", "list"],
            )
            with patch(
                "src.vertex.auth.find_gcloud_path", return_value=Path("/usr/bin/gcloud")
            ):
                result = authenticator.validate_credentials()
                assert result.is_valid is False
                assert result.error_code == "PERMISSION_DENIED"

    def test_get_access_token_success(self, authenticator: GCPAuthenticator) -> None:
        """Test getting access token."""
        with patch("src.vertex.auth.run_gcloud_command") as mock_run:
            mock_run.return_value = CommandResult(
                success=True,
                return_code=0,
                stdout="ya29.token123\n",
                stderr="",
                command=["gcloud", "auth", "print-access-token"],
            )
            with patch(
                "src.vertex.auth.find_gcloud_path", return_value=Path("/usr/bin/gcloud")
            ):
                token = authenticator.get_access_token()
                assert token == "ya29.token123"

    def test_get_access_token_failure(self, authenticator: GCPAuthenticator) -> None:
        """Test getting access token when not authenticated."""
        with patch("src.vertex.auth.run_gcloud_command") as mock_run:
            mock_run.return_value = CommandResult(
                success=False,
                return_code=1,
                stdout="",
                stderr="ERROR: not authenticated",
                command=["gcloud", "auth", "print-access-token"],
            )
            with patch(
                "src.vertex.auth.find_gcloud_path", return_value=Path("/usr/bin/gcloud")
            ):
                token = authenticator.get_access_token()
                assert token is None


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = ValidationResult(
            is_valid=True,
            account="test@example.com",
            project="test-project",
        )
        d = result.to_dict()
        assert d["is_valid"] is True
        assert d["account"] == "test@example.com"
        assert d["project"] == "test-project"

    def test_suggestions_default_empty(self) -> None:
        """Test suggestions default to empty list."""
        result = ValidationResult(is_valid=True, account=None, project=None)
        assert result.suggestions == []

    def test_with_error_info(self) -> None:
        """Test result with error information."""
        result = ValidationResult(
            is_valid=False,
            account=None,
            project=None,
            error_message="Test error",
            error_code="TEST_ERROR",
            suggestions=["Try this", "Or that"],
        )
        d = result.to_dict()
        assert d["error_message"] == "Test error"
        assert d["error_code"] == "TEST_ERROR"
        assert len(d["suggestions"]) == 2


class TestCommandResult:
    """Tests for CommandResult."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = CommandResult(
            success=True,
            return_code=0,
            stdout="output",
            stderr="",
            command=["gcloud", "auth", "list"],
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["return_code"] == 0
        assert d["stdout"] == "output"

    def test_to_dict_truncates_long_output(self) -> None:
        """Test long output is truncated in to_dict."""
        long_output = "x" * 1000
        result = CommandResult(
            success=True,
            return_code=0,
            stdout=long_output,
            stderr=long_output,
            command=["test"],
        )
        d = result.to_dict()
        assert len(d["stdout"]) == 500
        assert len(d["stderr"]) == 500


class TestAuthenticationError:
    """Tests for AuthenticationError."""

    def test_basic_error(self) -> None:
        """Test basic error creation."""
        error = AuthenticationError("Test error")
        assert str(error) == "Test error"
        assert error.validation_result is None

    def test_with_validation_result(self) -> None:
        """Test error with validation result."""
        result = ValidationResult(
            is_valid=False,
            account=None,
            project=None,
            error_message="Auth failed",
        )
        error = AuthenticationError("Auth failed", result)
        assert error.validation_result is result


class TestCreateAuthenticator:
    """Tests for factory function."""

    def test_create_with_defaults(self) -> None:
        """Test creating authenticator with defaults."""
        auth = create_authenticator()
        assert auth.config.auth_method == AuthMethod.APPLICATION_DEFAULT

    def test_create_with_gcloud(self) -> None:
        """Test creating with gcloud auth method."""
        auth = create_authenticator(auth_method="gcloud")
        assert auth.config.auth_method == AuthMethod.GCLOUD_CLI

    def test_create_with_service_account(self, tmp_path: Path) -> None:
        """Test creating with service account."""
        key_file = tmp_path / "key.json"
        key_file.write_text('{"type": "service_account"}')

        auth = create_authenticator(
            auth_method="service_account",
            service_account_key_path=str(key_file),
        )
        assert auth.config.auth_method == AuthMethod.SERVICE_ACCOUNT_KEY

    def test_create_with_project_id(self) -> None:
        """Test creating with project ID."""
        auth = create_authenticator(project_id="my-project")
        assert auth.config.project_id == "my-project"

    def test_create_with_custom_timeout(self) -> None:
        """Test creating with custom timeout."""
        auth = create_authenticator(timeout_seconds=60)
        assert auth.config.timeout_seconds == 60


class TestPlatformInfo:
    """Tests for PlatformInfo dataclass."""

    def test_windows_powershell_info(self) -> None:
        """Test Windows PowerShell platform info."""
        info = PlatformInfo(
            is_windows=True,
            shell_type="powershell",
            use_cmd_wrapper=True,
            gcloud_executable="gcloud.cmd",
        )
        assert info.is_windows is True
        assert info.use_cmd_wrapper is True

    def test_linux_bash_info(self) -> None:
        """Test Linux bash platform info."""
        info = PlatformInfo(
            is_windows=False,
            shell_type="bash",
            use_cmd_wrapper=False,
            gcloud_executable="gcloud",
        )
        assert info.is_windows is False
        assert info.use_cmd_wrapper is False
