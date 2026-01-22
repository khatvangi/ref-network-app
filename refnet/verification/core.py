"""
verification core - data structures and base classes.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class Severity(Enum):
    """check severity levels."""
    ERROR = "error"      # critical, must pass
    WARNING = "warning"  # suspicious, investigate
    INFO = "info"        # nice to know


@dataclass
class Check:
    """single verification check result."""
    name: str               # e.g., "doi_matches", "author_in_papers"
    passed: bool
    severity: Severity
    message: str            # human explanation
    expected: Any = None    # what we expected
    actual: Any = None      # what we got

    def __str__(self) -> str:
        status = "✓" if self.passed else "✗"
        return f"[{status}] {self.name}: {self.message}"


@dataclass
class VerificationResult:
    """result of verifying an agent's output."""
    agent_name: str
    input_summary: str          # brief description of input
    timestamp: datetime = field(default_factory=datetime.now)

    checks: List[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """true if all critical (error-level) checks passed."""
        return all(
            c.passed for c in self.checks
            if c.severity == Severity.ERROR
        )

    @property
    def confidence(self) -> float:
        """overall confidence score 0.0-1.0."""
        if not self.checks:
            return 1.0

        # weight by severity
        weights = {
            Severity.ERROR: 1.0,
            Severity.WARNING: 0.5,
            Severity.INFO: 0.1
        }

        total_weight = sum(weights[c.severity] for c in self.checks)
        passed_weight = sum(
            weights[c.severity] for c in self.checks if c.passed
        )

        return passed_weight / total_weight if total_weight > 0 else 1.0

    @property
    def errors(self) -> List[Check]:
        """failed checks with severity=ERROR."""
        return [c for c in self.checks if not c.passed and c.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[Check]:
        """failed checks with severity=WARNING."""
        return [c for c in self.checks if not c.passed and c.severity == Severity.WARNING]

    @property
    def infos(self) -> List[Check]:
        """failed checks with severity=INFO."""
        return [c for c in self.checks if not c.passed and c.severity == Severity.INFO]

    def add_check(self, check: Check):
        """add a check result."""
        self.checks.append(check)

    def summary(self) -> str:
        """brief summary string."""
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        status = "PASSED" if self.passed else "FAILED"

        lines = [
            f"Verification [{self.agent_name}]: {status}",
            f"  Confidence: {self.confidence:.0%}",
            f"  Checks: {passed}/{total} passed",
        ]

        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")

        return "\n".join(lines)

    def details(self) -> str:
        """detailed check results."""
        lines = [self.summary(), ""]

        if self.errors:
            lines.append("ERRORS:")
            for c in self.errors:
                lines.append(f"  {c}")

        if self.warnings:
            lines.append("WARNINGS:")
            for c in self.warnings:
                lines.append(f"  {c}")

        return "\n".join(lines)


@dataclass
class VerificationReport:
    """overall verification report for a pipeline run."""
    created_at: datetime = field(default_factory=datetime.now)

    # per-agent results
    agent_results: Dict[str, VerificationResult] = field(default_factory=dict)

    # suggestions for fixing issues
    suggestions: List[str] = field(default_factory=list)

    @property
    def overall_passed(self) -> bool:
        """true if all agents passed verification."""
        return all(r.passed for r in self.agent_results.values())

    @property
    def overall_confidence(self) -> float:
        """average confidence across all agents."""
        if not self.agent_results:
            return 1.0
        return sum(r.confidence for r in self.agent_results.values()) / len(self.agent_results)

    @property
    def total_checks(self) -> int:
        """total number of checks run."""
        return sum(len(r.checks) for r in self.agent_results.values())

    @property
    def passed_checks(self) -> int:
        """number of checks that passed."""
        return sum(
            sum(1 for c in r.checks if c.passed)
            for r in self.agent_results.values()
        )

    @property
    def failed_errors(self) -> int:
        """number of failed error-level checks."""
        return sum(len(r.errors) for r in self.agent_results.values())

    @property
    def failed_warnings(self) -> int:
        """number of failed warning-level checks."""
        return sum(len(r.warnings) for r in self.agent_results.values())

    @property
    def critical_issues(self) -> List[str]:
        """all critical issues across agents."""
        issues = []
        for agent_name, result in self.agent_results.items():
            for check in result.errors:
                issues.append(f"[{agent_name}] {check.name}: {check.message}")
        return issues

    def add_result(self, result: VerificationResult):
        """add an agent verification result."""
        self.agent_results[result.agent_name] = result

    def add_suggestion(self, suggestion: str):
        """add a suggestion for fixing issues."""
        self.suggestions.append(suggestion)

    def summary(self) -> str:
        """brief summary."""
        status = "PASSED" if self.overall_passed else "FAILED"
        return (
            f"Verification Report: {status}\n"
            f"  Confidence: {self.overall_confidence:.0%}\n"
            f"  Checks: {self.passed_checks}/{self.total_checks} passed\n"
            f"  Errors: {self.failed_errors}, Warnings: {self.failed_warnings}\n"
            f"  Agents: {len(self.agent_results)}"
        )

    def details(self) -> str:
        """detailed report."""
        lines = [self.summary(), ""]

        for agent_name, result in self.agent_results.items():
            lines.append(f"--- {agent_name} ---")
            lines.append(result.details())
            lines.append("")

        if self.critical_issues:
            lines.append("CRITICAL ISSUES:")
            for issue in self.critical_issues:
                lines.append(f"  ! {issue}")
            lines.append("")

        if self.suggestions:
            lines.append("SUGGESTIONS:")
            for s in self.suggestions:
                lines.append(f"  * {s}")

        return "\n".join(lines)


def check(
    name: str,
    passed: bool,
    message: str,
    severity: Severity = Severity.WARNING,
    expected: Any = None,
    actual: Any = None
) -> Check:
    """convenience function to create a Check."""
    return Check(
        name=name,
        passed=passed,
        severity=severity,
        message=message,
        expected=expected,
        actual=actual
    )


def error_check(
    name: str,
    passed: bool,
    message: str,
    expected: Any = None,
    actual: Any = None
) -> Check:
    """convenience function to create an error-level Check."""
    return check(name, passed, message, Severity.ERROR, expected, actual)


def warning_check(
    name: str,
    passed: bool,
    message: str,
    expected: Any = None,
    actual: Any = None
) -> Check:
    """convenience function to create a warning-level Check."""
    return check(name, passed, message, Severity.WARNING, expected, actual)


def info_check(
    name: str,
    passed: bool,
    message: str,
    expected: Any = None,
    actual: Any = None
) -> Check:
    """convenience function to create an info-level Check."""
    return check(name, passed, message, Severity.INFO, expected, actual)
