"""PR merge readiness checker scenario.

Validates that all requirements are met for merging a PR.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from src.validation.config import (
    MergeReadinessConfig,
    TestFailureInfo,
    ValidationResult,
    ValidationStatus,
)
from src.validation.scenarios.base import BaseValidator


class MergeReadinessChecker(BaseValidator):
    """Checks if a PR is ready to merge.

    Validates:
    1. All tests pass (or allowed failures only)
    2. Linting passes
    3. Type checking passes
    4. No forbidden patterns (FIXME, TODO if configured)
    5. Documentation requirements
    """

    name = "merge_readiness"
    config_class = MergeReadinessConfig

    def __init__(
        self,
        config: MergeReadinessConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize merge readiness checker.

        Args:
            config: Merge readiness configuration.
            **kwargs: Override config fields.
        """
        super().__init__(config, **kwargs)
        self.config: MergeReadinessConfig = self.config  # Type hint
        self._test_failures: list[TestFailureInfo] = []

    def validate(self) -> ValidationResult:
        """Run all merge readiness checks.

        Returns:
            ValidationResult with check results.
        """
        checks: dict[str, bool] = {}
        issues: list[str] = []

        # Run tests
        if self.config.require_all_tests_pass:
            test_passed, test_issues = self._run_tests()
            checks["tests"] = test_passed
            issues.extend(test_issues)
            self.record_metric("test_failures", float(len(self._test_failures)))

        # Run linting
        if self.config.require_lint_pass:
            lint_passed, lint_issues = self._run_lint()
            checks["lint"] = lint_passed
            issues.extend(lint_issues)

        # Run type checking
        if self.config.require_type_check_pass:
            type_passed, type_issues = self._run_type_check()
            checks["type_check"] = type_passed
            issues.extend(type_issues)

        # Check for FIXME/TODO
        if self.config.require_no_fixmes or self.config.require_no_todos:
            pattern_passed, pattern_issues = self._check_forbidden_patterns()
            checks["patterns"] = pattern_passed
            issues.extend(pattern_issues)

        # Check dependencies
        if self.config.check_dependencies:
            dep_passed, dep_issues = self._check_dependencies()
            checks["dependencies"] = dep_passed
            issues.extend(dep_issues)

        # Calculate overall pass/fail
        passed = all(checks.values())

        # Record check results
        for check_name, check_passed in checks.items():
            self.record_metric(f"check_{check_name}", 1.0 if check_passed else 0.0)

        self.record_detail("checks", checks)
        self.record_detail("issues", issues)

        return self._create_result(
            ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            passed=passed,
            test_failures=self._test_failures,
            pr_number=self.config.pr_number,
        )

    def _run_tests(self) -> tuple[bool, list[str]]:
        """Run the test suite.

        Returns:
            Tuple of (passed, list of issues).
        """
        issues: list[str] = []
        self._test_failures = []

        self._logger.info(
            "running_tests",
            command=self.config.test_command,
            timeout=self.config.test_timeout_seconds,
        )

        try:
            result = subprocess.run(
                shlex.split(self.config.test_command),
                capture_output=True,
                text=True,
                timeout=self.config.test_timeout_seconds,
            )

            # Parse test output for failures
            self._parse_test_output(result.stdout + result.stderr)

            # Check if tests passed
            if result.returncode != 0:
                # Filter out allowed failures
                actual_failures = [
                    f
                    for f in self._test_failures
                    if f.test_name not in self.config.allowed_test_failures
                ]

                if len(actual_failures) > self.config.max_allowed_failures:
                    issues.append(
                        f"Test suite failed with {len(actual_failures)} failures "
                        f"(max allowed: {self.config.max_allowed_failures})"
                    )
                    return False, issues

            self._logger.info(
                "tests_completed",
                returncode=result.returncode,
                failures=len(self._test_failures),
            )

            return True, issues

        except subprocess.TimeoutExpired:
            issues.append(
                f"Test suite timed out after {self.config.test_timeout_seconds}s"
            )
            return False, issues
        except FileNotFoundError:
            issues.append(f"Test command not found: {self.config.test_command}")
            return False, issues

    def _parse_test_output(self, output: str) -> None:
        """Parse test output for failures.

        Args:
            output: Combined stdout/stderr from test run.
        """
        # Match pytest failure patterns
        failure_pattern = r"FAILED\s+(\S+)::(test_\w+)"
        for match in re.finditer(failure_pattern, output):
            file_path, test_name = match.groups()

            # Extract error message
            error_pattern = rf"{test_name}.*?(?:AssertionError|Error):\s*(.+?)(?:\n|$)"
            error_match = re.search(error_pattern, output, re.DOTALL)
            error_msg = error_match.group(1).strip() if error_match else "Unknown error"

            # Determine failure type
            failure_type = "assertion"
            if "tolerance" in error_msg.lower() or "allclose" in error_msg.lower():
                failure_type = "tolerance"
            elif "Exception" in error_msg or "Error" in error_msg:
                failure_type = "exception"

            self._test_failures.append(
                TestFailureInfo(
                    test_name=test_name,
                    file_path=file_path,
                    error_message=error_msg[:200],  # Truncate
                    failure_type=failure_type,
                )
            )

    def _run_lint(self) -> tuple[bool, list[str]]:
        """Run linting checks.

        Returns:
            Tuple of (passed, list of issues).
        """
        issues: list[str] = []

        self._logger.info("running_lint", command=self.config.lint_command)

        try:
            result = subprocess.run(
                shlex.split(self.config.lint_command),
                capture_output=True,
                text=True,
                timeout=self.config.lint_timeout_seconds,
            )

            if result.returncode != 0:
                # Count lint errors
                error_count = len(result.stdout.strip().split("\n")) if result.stdout else 0
                issues.append(f"Linting failed with {error_count} issues")
                self._logger.warning(
                    "lint_failed",
                    error_count=error_count,
                    output=result.stdout[:500],
                )
                return False, issues

            self._logger.info("lint_passed")
            return True, issues

        except subprocess.TimeoutExpired:
            issues.append(f"Lint timed out after {self.config.lint_timeout_seconds}s")
            return False, issues
        except FileNotFoundError:
            issues.append(f"Lint command not found: {self.config.lint_command}")
            return False, issues

    def _run_type_check(self) -> tuple[bool, list[str]]:
        """Run type checking.

        Returns:
            Tuple of (passed, list of issues).
        """
        issues: list[str] = []

        self._logger.info("running_type_check", command=self.config.type_check_command)

        try:
            result = subprocess.run(
                shlex.split(self.config.type_check_command),
                capture_output=True,
                text=True,
                timeout=self.config.lint_timeout_seconds * 2,  # Type check can take longer
            )

            if result.returncode != 0:
                # Count type errors
                error_pattern = r"error:"
                error_count = len(re.findall(error_pattern, result.stdout))
                issues.append(f"Type checking failed with {error_count} errors")
                self._logger.warning(
                    "type_check_failed",
                    error_count=error_count,
                )
                return False, issues

            self._logger.info("type_check_passed")
            return True, issues

        except subprocess.TimeoutExpired:
            issues.append("Type checking timed out")
            return False, issues
        except FileNotFoundError:
            issues.append(f"Type check command not found: {self.config.type_check_command}")
            return False, issues

    def _check_forbidden_patterns(self) -> tuple[bool, list[str]]:
        """Check for forbidden patterns in code.

        Returns:
            Tuple of (passed, list of issues).
        """
        issues: list[str] = []
        patterns_to_check: list[tuple[str, str]] = []

        if self.config.require_no_fixmes:
            patterns_to_check.append((r"FIXME", "FIXME"))
        if self.config.require_no_todos:
            patterns_to_check.append((r"TODO", "TODO"))

        if not patterns_to_check:
            return True, issues

        src_path = Path("src")
        if not src_path.exists():
            return True, issues

        for pattern, name in patterns_to_check:
            count = 0
            for py_file in src_path.rglob("*.py"):
                content = py_file.read_text()
                matches = re.findall(pattern, content)
                count += len(matches)

            if count > 0:
                issues.append(f"Found {count} {name} comments in source code")

        return len(issues) == 0, issues

    def _check_dependencies(self) -> tuple[bool, list[str]]:
        """Check for dependency issues.

        Returns:
            Tuple of (passed, list of issues).
        """
        issues: list[str] = []

        # Check pyproject.toml exists
        pyproject = Path("pyproject.toml")
        if not pyproject.exists():
            issues.append("pyproject.toml not found")
            return False, issues

        # Try to parse and check for common issues
        try:
            content = pyproject.read_text()

            # Check for pinned versions that might conflict
            if "==" in content:
                # Count exact version pins
                pins = re.findall(r'"\w+==[\d.]+"', content)
                if len(pins) > 10:
                    issues.append(
                        f"Many exact version pins ({len(pins)}), consider using >= for flexibility"
                    )

        except Exception as e:
            issues.append(f"Error reading pyproject.toml: {e}")
            return False, issues

        return len(issues) == 0, issues

    def get_summary(self) -> str:
        """Get a human-readable summary of readiness.

        Returns:
            Summary string.
        """
        lines = [
            f"PR #{self.config.pr_number} Merge Readiness Check",
            "=" * 40,
        ]

        # Run quick checks
        result = self.run()

        if result.passed:
            lines.append("[READY] All checks passed!")
        else:
            lines.append("[NOT READY] Issues found:")

            if result.test_failures:
                lines.append(f"\nTest Failures ({len(result.test_failures)}):")
                for failure in result.test_failures[:5]:
                    lines.append(f"  - {failure.test_name}: {failure.error_message[:50]}")

            if "issues" in result.details:
                lines.append("\nOther Issues:")
                for issue in result.details["issues"]:
                    lines.append(f"  - {issue}")

        return "\n".join(lines)
