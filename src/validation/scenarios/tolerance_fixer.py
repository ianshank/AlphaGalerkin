"""Tolerance test fixer scenario.

Identifies and suggests fixes for tests with tolerance/precision issues.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.validation.config import (
    TestFailureInfo,
    ToleranceConfig,
    ToleranceLevel,
    ValidationResult,
    ValidationStatus,
)
from src.validation.scenarios.base import BaseValidator
from src.validation.tolerance import get_tolerance_for_dtype


@dataclass
class ToleranceIssue:
    """Represents a tolerance issue in a test."""

    file_path: str
    line_number: int
    test_name: str
    current_rtol: float | None
    current_atol: float | None
    suggested_rtol: float
    suggested_atol: float
    reason: str
    code_snippet: str


class ToleranceTestFixer(BaseValidator):
    """Identifies and fixes tolerance-related test failures.

    Analyzes test files for:
    1. Hardcoded tolerance values that are too strict
    2. Missing dtype-aware tolerance adjustment
    3. Inconsistent tolerance usage
    """

    name = "tolerance_fixer"
    config_class = ToleranceConfig

    def __init__(
        self,
        config: ToleranceConfig | None = None,
        test_dirs: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize tolerance fixer.

        Args:
            config: Tolerance configuration.
            test_dirs: Directories to scan for tests.
            **kwargs: Override config fields.
        """
        super().__init__(config, **kwargs)
        self.config: ToleranceConfig = self.config  # Type hint
        self.test_dirs = test_dirs or ["tests/"]
        self._issues: list[ToleranceIssue] = []

    def validate(self) -> ValidationResult:
        """Analyze tests and suggest tolerance fixes.

        Returns:
            ValidationResult with issues and suggestions.
        """
        self._issues = []

        # Scan test files
        for test_dir in self.test_dirs:
            test_path = Path(test_dir)
            if test_path.exists():
                self._scan_directory(test_path)

        # Record metrics
        self.record_metric("total_issues", float(len(self._issues)))
        self.record_metric("files_scanned", float(len(self._get_scanned_files())))

        # Group issues by type
        rtol_issues = [i for i in self._issues if "rtol" in i.reason.lower()]
        atol_issues = [i for i in self._issues if "atol" in i.reason.lower()]
        dtype_issues = [i for i in self._issues if "dtype" in i.reason.lower()]

        self.record_metric("rtol_issues", float(len(rtol_issues)))
        self.record_metric("atol_issues", float(len(atol_issues)))
        self.record_metric("dtype_issues", float(len(dtype_issues)))

        # Generate suggestions
        suggestions = self._generate_suggestions()
        self.record_detail("suggestions", suggestions)

        # Convert to test failures for report
        test_failures = [
            TestFailureInfo(
                test_name=issue.test_name,
                file_path=issue.file_path,
                line_number=issue.line_number,
                error_message=issue.reason,
                failure_type="tolerance",
                suggested_fix=f"Use rtol={issue.suggested_rtol}, atol={issue.suggested_atol}",
            )
            for issue in self._issues
        ]

        # Determine pass/fail
        # For this validator, we pass if we successfully analyzed the tests
        # Actual fix implementation is separate
        passed = True

        return self._create_result(
            ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            passed=passed,
            issues=[self._issue_to_dict(i) for i in self._issues],
            test_failures=test_failures,
        )

    def _scan_directory(self, path: Path) -> None:
        """Scan a directory for test files.

        Args:
            path: Directory to scan.
        """
        for file_path in path.rglob("test_*.py"):
            self._analyze_file(file_path)

    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a test file for tolerance issues.

        Args:
            file_path: Path to test file.
        """
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError) as e:
            self._logger.warning(
                "file_parse_failed",
                file=str(file_path),
                error=str(e),
            )
            return

        lines = content.split("\n")

        for node in ast.walk(tree):
            # Look for function definitions (tests)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                self._analyze_test_function(node, lines, str(file_path))

    def _analyze_test_function(
        self,
        node: ast.FunctionDef,
        lines: list[str],
        file_path: str,
    ) -> None:
        """Analyze a test function for tolerance issues.

        Args:
            node: AST node for the function.
            lines: Source lines.
            file_path: Path to file.
        """
        # Look for tolerance-related calls
        patterns = [
            (r"np\.allclose\s*\(", self._check_numpy_allclose),
            (r"torch\.allclose\s*\(", self._check_torch_allclose),
            (r"assert_allclose\s*\(", self._check_custom_allclose),
            (r"np\.testing\.assert_allclose\s*\(", self._check_numpy_testing),
            (r"rtol\s*=\s*([\d.e+-]+)", self._check_rtol_value),
            (r"atol\s*=\s*([\d.e+-]+)", self._check_atol_value),
        ]

        for i, line in enumerate(lines[node.lineno - 1 : node.end_lineno or len(lines)], start=node.lineno):
            for pattern, checker in patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    issue = checker(match, line, i, node.name, file_path)
                    if issue:
                        self._issues.append(issue)

    def _check_numpy_allclose(
        self,
        match: re.Match[str],
        line: str,
        line_num: int,
        test_name: str,
        file_path: str,
    ) -> ToleranceIssue | None:
        """Check numpy.allclose calls."""
        # Check if rtol/atol are specified
        rtol_match = re.search(r"rtol\s*=\s*([\d.e+-]+)", line)
        atol_match = re.search(r"atol\s*=\s*([\d.e+-]+)", line)

        current_rtol = float(rtol_match.group(1)) if rtol_match else 1e-5
        current_atol = float(atol_match.group(1)) if atol_match else 1e-8

        # Check if tolerances are too strict for float32
        if current_rtol < 1e-6 or current_atol < 1e-7:
            return ToleranceIssue(
                file_path=file_path,
                line_number=line_num,
                test_name=test_name,
                current_rtol=current_rtol,
                current_atol=current_atol,
                suggested_rtol=max(current_rtol, 1e-5),
                suggested_atol=max(current_atol, 1e-6),
                reason="Tolerance may be too strict for float32 tensors",
                code_snippet=line.strip(),
            )

        return None

    def _check_torch_allclose(
        self,
        match: re.Match[str],
        line: str,
        line_num: int,
        test_name: str,
        file_path: str,
    ) -> ToleranceIssue | None:
        """Check torch.allclose calls."""
        # Similar logic to numpy
        rtol_match = re.search(r"rtol\s*=\s*([\d.e+-]+)", line)
        atol_match = re.search(r"atol\s*=\s*([\d.e+-]+)", line)

        current_rtol = float(rtol_match.group(1)) if rtol_match else 1e-5
        current_atol = float(atol_match.group(1)) if atol_match else 1e-8

        if current_rtol < 1e-6 or current_atol < 1e-7:
            return ToleranceIssue(
                file_path=file_path,
                line_number=line_num,
                test_name=test_name,
                current_rtol=current_rtol,
                current_atol=current_atol,
                suggested_rtol=max(current_rtol, 1e-5),
                suggested_atol=max(current_atol, 1e-6),
                reason="Tolerance may be too strict for PyTorch default dtype",
                code_snippet=line.strip(),
            )

        return None

    def _check_custom_allclose(
        self,
        match: re.Match[str],
        line: str,
        line_num: int,
        test_name: str,
        file_path: str,
    ) -> ToleranceIssue | None:
        """Check custom assert_allclose calls."""
        # Check for missing tolerance configuration
        if "config=" not in line and "rtol=" not in line:
            return ToleranceIssue(
                file_path=file_path,
                line_number=line_num,
                test_name=test_name,
                current_rtol=None,
                current_atol=None,
                suggested_rtol=1e-5,
                suggested_atol=1e-6,
                reason="assert_allclose missing explicit tolerance config",
                code_snippet=line.strip(),
            )

        return None

    def _check_numpy_testing(
        self,
        match: re.Match[str],
        line: str,
        line_num: int,
        test_name: str,
        file_path: str,
    ) -> ToleranceIssue | None:
        """Check numpy.testing.assert_allclose calls."""
        rtol_match = re.search(r"rtol\s*=\s*([\d.e+-]+)", line)

        if not rtol_match:
            return ToleranceIssue(
                file_path=file_path,
                line_number=line_num,
                test_name=test_name,
                current_rtol=None,
                current_atol=None,
                suggested_rtol=1e-5,
                suggested_atol=1e-6,
                reason="Missing explicit rtol in np.testing.assert_allclose",
                code_snippet=line.strip(),
            )

        return None

    def _check_rtol_value(
        self,
        match: re.Match[str],
        line: str,
        line_num: int,
        test_name: str,
        file_path: str,
    ) -> ToleranceIssue | None:
        """Check rtol values for reasonableness."""
        try:
            rtol = float(match.group(1))
            if rtol < 1e-7:
                return ToleranceIssue(
                    file_path=file_path,
                    line_number=line_num,
                    test_name=test_name,
                    current_rtol=rtol,
                    current_atol=None,
                    suggested_rtol=1e-5,
                    suggested_atol=1e-6,
                    reason=f"rtol={rtol} is very strict, may cause float32 failures",
                    code_snippet=line.strip(),
                )
        except ValueError:
            pass  # Regex matched but value isn't a valid float, skip this check

        return None

    def _check_atol_value(
        self,
        match: re.Match[str],
        line: str,
        line_num: int,
        test_name: str,
        file_path: str,
    ) -> ToleranceIssue | None:
        """Check atol values for reasonableness."""
        try:
            atol = float(match.group(1))
            if atol < 1e-9:
                return ToleranceIssue(
                    file_path=file_path,
                    line_number=line_num,
                    test_name=test_name,
                    current_rtol=None,
                    current_atol=atol,
                    suggested_rtol=1e-5,
                    suggested_atol=1e-6,
                    reason=f"atol={atol} is very strict, may cause precision issues",
                    code_snippet=line.strip(),
                )
        except ValueError:
            pass  # Regex matched but value isn't a valid float, skip this check

        return None

    def _get_scanned_files(self) -> list[str]:
        """Get list of scanned files."""
        return list(set(i.file_path for i in self._issues))

    def _generate_suggestions(self) -> list[dict[str, Any]]:
        """Generate fix suggestions for issues.

        Returns:
            List of suggestion dictionaries.
        """
        suggestions = []

        for issue in self._issues:
            suggestion = {
                "file": issue.file_path,
                "line": issue.line_number,
                "test": issue.test_name,
                "current": f"rtol={issue.current_rtol}, atol={issue.current_atol}",
                "suggested": f"rtol={issue.suggested_rtol}, atol={issue.suggested_atol}",
                "reason": issue.reason,
                "fix_code": self._generate_fix_code(issue),
            }
            suggestions.append(suggestion)

        return suggestions

    def _generate_fix_code(self, issue: ToleranceIssue) -> str:
        """Generate fix code for an issue.

        Args:
            issue: Tolerance issue.

        Returns:
            Code snippet with suggested fix.
        """
        # Replace tolerance values in the code snippet
        fixed = issue.code_snippet

        if issue.current_rtol is not None:
            fixed = re.sub(
                r"rtol\s*=\s*[\d.e+-]+",
                f"rtol={issue.suggested_rtol}",
                fixed,
            )
        elif "rtol" not in fixed and "allclose" in fixed:
            # Add rtol parameter
            fixed = fixed.replace(")", f", rtol={issue.suggested_rtol})")

        if issue.current_atol is not None:
            fixed = re.sub(
                r"atol\s*=\s*[\d.e+-]+",
                f"atol={issue.suggested_atol}",
                fixed,
            )
        elif "atol" not in fixed and "allclose" in fixed:
            # Add atol parameter
            fixed = fixed.replace(")", f", atol={issue.suggested_atol})")

        return fixed

    def _issue_to_dict(self, issue: ToleranceIssue) -> dict[str, Any]:
        """Convert issue to dictionary.

        Args:
            issue: Tolerance issue.

        Returns:
            Dictionary representation.
        """
        return {
            "file_path": issue.file_path,
            "line_number": issue.line_number,
            "test_name": issue.test_name,
            "current_rtol": issue.current_rtol,
            "current_atol": issue.current_atol,
            "suggested_rtol": issue.suggested_rtol,
            "suggested_atol": issue.suggested_atol,
            "reason": issue.reason,
            "code_snippet": issue.code_snippet,
        }
