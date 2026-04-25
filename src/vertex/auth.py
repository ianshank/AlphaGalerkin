"""GCP authentication utilities for Vertex AI training.

This module provides platform-aware authentication validation and
gcloud command execution, ensuring cross-platform compatibility
with Windows (PowerShell/cmd) and Unix systems.

Key features:
- Platform detection (Windows PowerShell, CMD, Unix shells)
- cmd /c wrapper to bypass PowerShell PSSecurityException
- Multiple auth methods (ADC, service account, gcloud CLI)
- Friendly error messages with actionable suggestions

Example:
    from src.vertex.auth import GCPAuthenticator, AuthConfig

    config = AuthConfig(
        auth_method=AuthMethod.APPLICATION_DEFAULT,
        validate_before_launch=True,
    )
    auth = GCPAuthenticator(config)

    # Validate credentials before job submission
    result = auth.validate_credentials()
    if not result.is_valid:
        print(f"Auth failed: {result.error_message}")
        for suggestion in result.suggestions:
            print(f"  - {suggestion}")

"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = structlog.get_logger(__name__)


class AuthMethod(str, Enum):
    """Authentication methods for GCP."""

    APPLICATION_DEFAULT = "adc"  # Default credentials (gcloud auth, metadata, etc.)
    SERVICE_ACCOUNT_KEY = "service_account"  # JSON key file
    GCLOUD_CLI = "gcloud"  # Explicit gcloud auth login


class AuthConfig(BaseModel):
    """Configuration for GCP authentication.

    Attributes:
        auth_method: Preferred authentication method.
        service_account_key_path: Path to service account JSON key file.
        project_id: GCP project ID for validation.
        validate_before_launch: Validate credentials before job submission.
        gcloud_path: Custom path to gcloud CLI (auto-detected if None).
        timeout_seconds: Timeout for auth commands.
        retry_count: Number of retries for transient failures.

    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    auth_method: AuthMethod = Field(
        default=AuthMethod.APPLICATION_DEFAULT,
        description="Preferred authentication method",
    )
    service_account_key_path: Path | None = Field(
        default=None,
        description="Path to service account JSON key file",
    )
    project_id: str | None = Field(
        default=None,
        description="GCP project ID for validation (uses default if None)",
    )
    validate_before_launch: bool = Field(
        default=True,
        description="Validate credentials before job submission",
    )
    gcloud_path: Path | None = Field(
        default=None,
        description="Custom path to gcloud CLI (auto-detected if None)",
    )
    timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Timeout for auth commands in seconds",
    )
    retry_count: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Number of retries for transient failures",
    )

    @field_validator("service_account_key_path", mode="before")
    @classmethod
    def validate_key_path(cls, v: str | Path | None) -> Path | None:
        """Validate and normalize key path."""
        if v is None:
            return None
        path = Path(v).expanduser().resolve()
        return path

    @model_validator(mode="after")
    def validate_service_account_config(self) -> AuthConfig:
        """Ensure service account key is provided when using SA auth."""
        if (
            self.auth_method == AuthMethod.SERVICE_ACCOUNT_KEY
            and self.service_account_key_path is None
        ):
            raise ValueError(
                "service_account_key_path is required when auth_method is SERVICE_ACCOUNT_KEY"
            )
        return self


@dataclass
class PlatformInfo:
    """Platform information for command execution."""

    is_windows: bool
    shell_type: str  # "cmd", "powershell", "bash", "zsh", etc.
    use_cmd_wrapper: bool  # Whether to wrap with cmd /c on Windows
    gcloud_executable: str  # "gcloud" or "gcloud.cmd"


def detect_platform() -> PlatformInfo:
    """Detect current platform and shell configuration.

    Returns:
        PlatformInfo with platform-specific settings.

    """
    is_windows = sys.platform == "win32"

    if is_windows:
        # Detect if running in PowerShell or CMD
        # PSModulePath is set in PowerShell environments
        ps_module_path = os.environ.get("PSModulePath", "")
        is_powershell = bool(ps_module_path)

        return PlatformInfo(
            is_windows=True,
            shell_type="powershell" if is_powershell else "cmd",
            use_cmd_wrapper=is_powershell,  # Use cmd /c to bypass PS restrictions
            gcloud_executable="gcloud.cmd",
        )
    else:
        # Unix-like system
        shell = os.environ.get("SHELL", "/bin/bash")
        shell_type = Path(shell).name

        return PlatformInfo(
            is_windows=False,
            shell_type=shell_type,
            use_cmd_wrapper=False,
            gcloud_executable="gcloud",
        )


def find_gcloud_path(custom_path: Path | None = None) -> Path | None:
    """Find gcloud CLI executable.

    Args:
        custom_path: Custom path to check first.

    Returns:
        Path to gcloud executable, or None if not found.

    """
    platform = detect_platform()

    # Check custom path first
    if custom_path is not None:
        if custom_path.exists():
            return custom_path
        logger.warning("custom_gcloud_path_not_found", path=str(custom_path))

    # Try to find in PATH
    gcloud_name = platform.gcloud_executable
    gcloud_path = shutil.which(gcloud_name)

    if gcloud_path:
        return Path(gcloud_path)

    # Common installation locations
    common_paths: list[Path] = []

    if platform.is_windows:
        # Windows installation paths
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        program_files = os.environ.get("PROGRAMFILES", "")

        if local_app_data:
            common_paths.append(
                Path(local_app_data)
                / "Google"
                / "Cloud SDK"
                / "google-cloud-sdk"
                / "bin"
                / "gcloud.cmd"
            )
        if program_files:
            common_paths.append(
                Path(program_files)
                / "Google"
                / "Cloud SDK"
                / "google-cloud-sdk"
                / "bin"
                / "gcloud.cmd"
            )
        common_paths.append(
            Path.home()
            / "AppData"
            / "Local"
            / "Google"
            / "Cloud SDK"
            / "google-cloud-sdk"
            / "bin"
            / "gcloud.cmd"
        )
    else:
        # Unix installation paths
        common_paths = [
            Path("/usr/bin/gcloud"),
            Path("/usr/local/bin/gcloud"),
            Path.home() / "google-cloud-sdk" / "bin" / "gcloud",
            Path("/snap/bin/gcloud"),
        ]

    for path in common_paths:
        if path.exists():
            return path

    return None


@dataclass
class CommandResult:
    """Result of a command execution."""

    success: bool
    return_code: int
    stdout: str
    stderr: str
    command: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "success": self.success,
            "return_code": self.return_code,
            "stdout": self.stdout[:500] if self.stdout else "",
            "stderr": self.stderr[:500] if self.stderr else "",
        }


def run_gcloud_command(
    args: list[str],
    timeout: int = 30,
    platform: PlatformInfo | None = None,
    gcloud_path: Path | None = None,
    capture_output: bool = True,
) -> CommandResult:
    """Execute gcloud command with platform-aware shell handling.

    On Windows PowerShell, wraps command with 'cmd /c' to bypass
    execution policy restrictions (PSSecurityException).

    Args:
        args: gcloud command arguments (without 'gcloud' prefix).
        timeout: Command timeout in seconds.
        platform: Platform info (auto-detected if None).
        gcloud_path: Path to gcloud (auto-detected if None).
        capture_output: Whether to capture stdout/stderr.

    Returns:
        CommandResult with execution results.

    Example:
        result = run_gcloud_command(["auth", "list", "--format=json"])
        if result.success:
            accounts = json.loads(result.stdout)

    """
    if platform is None:
        platform = detect_platform()

    if gcloud_path is None:
        gcloud_path = find_gcloud_path()
        if gcloud_path is None:
            return CommandResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=(
                    "gcloud CLI not found. Install from https://cloud.google.com/sdk/docs/install"
                ),
                command=args,
            )

    # Build command
    # Note: subprocess.run with shell=False bypasses PowerShell execution policy
    # issues because it doesn't go through the shell. The gcloud.cmd is executed
    # directly by the Windows API, avoiding PSSecurityException.
    gcloud_cmd = str(gcloud_path)
    full_command = [gcloud_cmd, *args]

    logger.debug(
        "executing_gcloud_command",
        command=full_command,
        platform=platform.shell_type,
    )

    try:
        result = subprocess.run(
            full_command,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            shell=False,  # Never use shell=True for security
        )

        return CommandResult(
            success=result.returncode == 0,
            return_code=result.returncode,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            command=full_command,
        )

    except subprocess.TimeoutExpired:
        logger.warning("gcloud_command_timeout", command=full_command, timeout=timeout)
        return CommandResult(
            success=False,
            return_code=-1,
            stdout="",
            stderr=f"Command timed out after {timeout} seconds",
            command=full_command,
        )

    except FileNotFoundError as e:
        logger.error("gcloud_not_found", error=str(e))
        return CommandResult(
            success=False,
            return_code=-1,
            stdout="",
            stderr=f"gcloud CLI not found: {e}",
            command=full_command,
        )

    except Exception as e:
        logger.exception("gcloud_command_error", error=str(e))
        return CommandResult(
            success=False,
            return_code=-1,
            stdout="",
            stderr=str(e),
            command=full_command,
        )


@dataclass
class ValidationResult:
    """Result of credential validation."""

    is_valid: bool
    account: str | None
    project: str | None
    error_message: str | None = None
    error_code: str | None = None  # e.g., "EXPIRED", "NOT_FOUND", "PERMISSION_DENIED"
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "account": self.account,
            "project": self.project,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "suggestions": self.suggestions,
        }


class AuthenticationError(Exception):
    """Raised when GCP authentication fails."""

    def __init__(self, message: str, validation_result: ValidationResult | None = None) -> None:
        """Initialize authentication error.

        Args:
            message: Error message.
            validation_result: Optional validation result with details.

        """
        super().__init__(message)
        self.validation_result = validation_result


class GCPAuthenticator:
    """GCP authentication validation and management.

    This class provides methods to validate GCP credentials before
    launching Vertex AI jobs, with friendly error messages and
    suggestions for common issues.

    Example:
        auth = GCPAuthenticator(AuthConfig())
        result = auth.validate_credentials()
        if not result.is_valid:
            for suggestion in result.suggestions:
                print(f"Try: {suggestion}")

    """

    def __init__(self, config: AuthConfig) -> None:
        """Initialize authenticator.

        Args:
            config: Authentication configuration.

        """
        self.config = config
        self._platform = detect_platform()
        self._gcloud_path = find_gcloud_path(config.gcloud_path)
        self._logger = structlog.get_logger(__name__).bind(
            auth_method=config.auth_method.value,
            platform=self._platform.shell_type,
        )

    def validate_credentials(self) -> ValidationResult:
        """Validate GCP credentials based on configured method.

        Returns:
            ValidationResult with status and any errors.

        """
        if self.config.auth_method == AuthMethod.SERVICE_ACCOUNT_KEY:
            return self._validate_service_account()
        elif self.config.auth_method == AuthMethod.GCLOUD_CLI:
            return self._validate_gcloud_auth()
        else:  # APPLICATION_DEFAULT
            return self._validate_adc()

    def _validate_gcloud_auth(self) -> ValidationResult:
        """Validate gcloud CLI authentication."""
        if self._gcloud_path is None:
            return ValidationResult(
                is_valid=False,
                account=None,
                project=None,
                error_message="gcloud CLI not found",
                error_code="GCLOUD_NOT_FOUND",
                suggestions=[
                    "Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install",
                    "Add gcloud to PATH or set gcloud_path in config",
                ],
            )

        # Get current account
        result = run_gcloud_command(
            ["auth", "list", "--format=json"],
            timeout=self.config.timeout_seconds,
            platform=self._platform,
            gcloud_path=self._gcloud_path,
        )

        if not result.success:
            return self._parse_gcloud_error(result)

        try:
            accounts = json.loads(result.stdout)
            active_account = next(
                (acc["account"] for acc in accounts if acc.get("status") == "ACTIVE"),
                None,
            )
        except (json.JSONDecodeError, KeyError):
            active_account = None

        if not active_account:
            return ValidationResult(
                is_valid=False,
                account=None,
                project=None,
                error_message="No active gcloud account found",
                error_code="NO_ACTIVE_ACCOUNT",
                suggestions=[
                    "Run: gcloud auth login",
                    "Or set up Application Default Credentials: "
                    "gcloud auth application-default login",
                ],
            )

        # Get current project
        project = self._get_current_project()

        self._logger.info(
            "gcloud_auth_validated",
            account=active_account,
            project=project,
        )

        return ValidationResult(
            is_valid=True,
            account=active_account,
            project=project,
        )

    def _validate_adc(self) -> ValidationResult:
        """Validate Application Default Credentials."""
        try:
            from google.auth import default as google_auth_default
            from google.auth.exceptions import DefaultCredentialsError

            credentials, project = google_auth_default()

            # Try to get the service account email if available
            account = getattr(credentials, "service_account_email", None)
            if account is None:
                account = getattr(credentials, "_principal", "ADC")

            return ValidationResult(
                is_valid=True,
                account=account,
                project=project or self.config.project_id,
            )

        except DefaultCredentialsError as e:
            return ValidationResult(
                is_valid=False,
                account=None,
                project=None,
                error_message=f"No Application Default Credentials found: {e}",
                error_code="NO_ADC",
                suggestions=[
                    "Run: gcloud auth application-default login",
                    "Set GOOGLE_APPLICATION_CREDENTIALS environment variable",
                    "Use a service account key file",
                ],
            )
        except ImportError:
            return ValidationResult(
                is_valid=False,
                account=None,
                project=None,
                error_message="google-auth library not installed",
                error_code="MISSING_LIBRARY",
                suggestions=[
                    "Install: pip install google-auth",
                ],
            )

    def _validate_service_account(self) -> ValidationResult:
        """Validate service account key file."""
        if self.config.service_account_key_path is None:
            return ValidationResult(
                is_valid=False,
                account=None,
                project=None,
                error_message="Service account key path not configured",
                error_code="NO_KEY_PATH",
                suggestions=[
                    "Set service_account_key_path in AuthConfig",
                ],
            )

        key_path = self.config.service_account_key_path

        if not key_path.exists():
            return ValidationResult(
                is_valid=False,
                account=None,
                project=None,
                error_message=f"Service account key file not found: {key_path}",
                error_code="KEY_NOT_FOUND",
                suggestions=[
                    f"Create or download the key file to: {key_path}",
                    "Generate a new key in GCP Console: IAM & Admin > Service Accounts",
                ],
            )

        try:
            with open(key_path) as f:
                key_data = json.load(f)

            account = key_data.get("client_email")
            project = key_data.get("project_id")

            if not account:
                return ValidationResult(
                    is_valid=False,
                    account=None,
                    project=None,
                    error_message="Invalid service account key: missing client_email",
                    error_code="INVALID_KEY",
                    suggestions=[
                        "Download a new key from GCP Console",
                    ],
                )

            # Set environment variable for google-auth
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)

            return ValidationResult(
                is_valid=True,
                account=account,
                project=project or self.config.project_id,
            )

        except json.JSONDecodeError:
            return ValidationResult(
                is_valid=False,
                account=None,
                project=None,
                error_message=f"Invalid JSON in service account key file: {key_path}",
                error_code="INVALID_JSON",
                suggestions=[
                    "Download a new key from GCP Console",
                ],
            )

    def _get_current_project(self) -> str | None:
        """Get current GCP project from gcloud config."""
        result = run_gcloud_command(
            ["config", "get-value", "project"],
            timeout=self.config.timeout_seconds,
            platform=self._platform,
            gcloud_path=self._gcloud_path,
        )

        if result.success and result.stdout.strip():
            return result.stdout.strip()

        return self.config.project_id

    def _parse_gcloud_error(self, result: CommandResult) -> ValidationResult:
        """Parse gcloud error and return friendly validation result."""
        stderr = result.stderr.lower()

        if "not recognized" in stderr or "not found" in stderr:
            return ValidationResult(
                is_valid=False,
                account=None,
                project=None,
                error_message="gcloud command not found",
                error_code="GCLOUD_NOT_FOUND",
                suggestions=[
                    "Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install",
                    "Restart your terminal after installation",
                ],
            )

        if "expired" in stderr:
            return ValidationResult(
                is_valid=False,
                account=None,
                project=None,
                error_message="GCP credentials have expired",
                error_code="CREDENTIALS_EXPIRED",
                suggestions=[
                    "Run: gcloud auth login",
                    "Or: gcloud auth application-default login",
                ],
            )

        if "permission" in stderr or "forbidden" in stderr:
            return ValidationResult(
                is_valid=False,
                account=None,
                project=None,
                error_message="Permission denied - check IAM roles",
                error_code="PERMISSION_DENIED",
                suggestions=[
                    "Ensure your account has required Vertex AI permissions",
                    "Required roles: roles/aiplatform.user, roles/storage.objectAdmin",
                ],
            )

        # PSSecurityException handling (Windows PowerShell)
        if "pssecurityexception" in stderr or "execution policy" in stderr:
            return ValidationResult(
                is_valid=False,
                account=None,
                project=None,
                error_message="PowerShell execution policy blocked gcloud",
                error_code="PS_SECURITY_EXCEPTION",
                suggestions=[
                    "The auth module will use cmd /c wrapper automatically",
                    "Or run from cmd.exe instead of PowerShell",
                    "Or set PowerShell execution policy: "
                    "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser",
                ],
            )

        # Generic error
        return ValidationResult(
            is_valid=False,
            account=None,
            project=None,
            error_message=f"gcloud command failed: {result.stderr[:200]}",
            error_code="GCLOUD_ERROR",
            suggestions=[
                "Check gcloud CLI installation",
                "Run: gcloud auth login",
            ],
        )

    def refresh_credentials(self) -> ValidationResult:
        """Attempt to refresh credentials.

        Returns:
            ValidationResult after refresh attempt.

        """
        self._logger.info("refreshing_credentials")

        if self.config.auth_method == AuthMethod.SERVICE_ACCOUNT_KEY:
            # Service account keys don't need refresh
            return self._validate_service_account()

        # Try to refresh ADC first
        try:
            from google.auth import default as google_auth_default
            from google.auth.transport.requests import Request

            credentials, _ = google_auth_default()
            if hasattr(credentials, "refresh"):
                credentials.refresh(Request())
                return self._validate_adc()
        except Exception as e:
            self._logger.warning("adc_refresh_failed", error=str(e))

        # Fall back to gcloud auth
        result = run_gcloud_command(
            ["auth", "login", "--update-adc"],
            timeout=120,  # Login may be interactive
            platform=self._platform,
            gcloud_path=self._gcloud_path,
        )

        if result.success:
            return self._validate_gcloud_auth()

        return self._parse_gcloud_error(result)

    def get_access_token(self) -> str | None:
        """Get access token for API calls.

        Returns:
            Access token string, or None if unavailable.

        """
        result = run_gcloud_command(
            ["auth", "print-access-token"],
            timeout=self.config.timeout_seconds,
            platform=self._platform,
            gcloud_path=self._gcloud_path,
        )

        if result.success:
            return result.stdout.strip()

        return None


def create_authenticator(
    auth_method: str = "adc",
    service_account_key_path: str | Path | None = None,
    project_id: str | None = None,
    **kwargs: Any,
) -> GCPAuthenticator:
    """Create a GCP authenticator with sensible defaults.

    Args:
        auth_method: One of "adc", "service_account", "gcloud".
        service_account_key_path: Path to service account key.
        project_id: GCP project ID.
        **kwargs: Additional AuthConfig options.

    Returns:
        Configured GCPAuthenticator instance.

    """
    config = AuthConfig(
        auth_method=AuthMethod(auth_method),
        service_account_key_path=(
            Path(service_account_key_path) if service_account_key_path else None
        ),
        project_id=project_id,
        **kwargs,
    )
    return GCPAuthenticator(config)
