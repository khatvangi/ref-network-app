"""
base agent - foundation for all refnet agents.

principles:
- single responsibility (one task, one agent)
- composable (agents can call other agents)
- traceable (every decision logged with reason)
- fallible (handle failures gracefully, never crash pipeline)
- testable (each agent can be tested in isolation)
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypeVar, Generic
from enum import Enum


class AgentStatus(Enum):
    """status of agent execution."""
    SUCCESS = "success"
    PARTIAL = "partial"  # some data retrieved, some failed
    FAILED = "failed"
    SKIPPED = "skipped"  # preconditions not met


@dataclass
class AgentError:
    """structured error from agent execution."""
    code: str  # machine-readable error code
    message: str  # human-readable message
    recoverable: bool = True  # can pipeline continue?
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"[{self.code}] {self.message}"


T = TypeVar('T')


@dataclass
class AgentResult(Generic[T]):
    """
    result from agent execution.

    always returns a result, even on failure.
    check status and errors before using data.
    """
    status: AgentStatus
    data: Optional[T] = None
    errors: List[AgentError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # execution metadata
    duration_ms: float = 0.0
    api_calls: int = 0
    cache_hits: int = 0

    # tracing
    trace: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """true if status is SUCCESS or PARTIAL with data."""
        return self.status in (AgentStatus.SUCCESS, AgentStatus.PARTIAL) and self.data is not None

    @property
    def failed(self) -> bool:
        """true if status is FAILED."""
        return self.status == AgentStatus.FAILED

    def add_trace(self, message: str):
        """add trace message for debugging."""
        self.trace.append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def add_warning(self, message: str):
        """add non-fatal warning."""
        self.warnings.append(message)

    def add_error(self, code: str, message: str, recoverable: bool = True, **details):
        """add structured error."""
        self.errors.append(AgentError(
            code=code,
            message=message,
            recoverable=recoverable,
            details=details
        ))


class Agent(ABC):
    """
    base class for all agents.

    subclasses must implement:
    - name: str property
    - execute(): the main logic

    provides:
    - logging
    - timing
    - error handling
    - tracing
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self._logger = logger or logging.getLogger(f"refnet.agents.{self.name}")

    @property
    @abstractmethod
    def name(self) -> str:
        """agent name for logging and tracing."""
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> AgentResult:
        """
        execute the agent's task.

        must return AgentResult, never raise exceptions.
        all errors should be captured in result.errors.
        """
        pass

    def run(self, *args, **kwargs) -> AgentResult:
        """
        run agent with timing and error handling.

        wraps execute() with:
        - timing measurement
        - exception catching
        - logging
        """
        start = time.time()

        try:
            self._logger.debug(f"[{self.name}] starting execution")
            result = self.execute(*args, **kwargs)

        except Exception as e:
            # catch any uncaught exception and wrap in result
            self._logger.exception(f"[{self.name}] uncaught exception")
            result = AgentResult(
                status=AgentStatus.FAILED,
                errors=[AgentError(
                    code="UNCAUGHT_EXCEPTION",
                    message=str(e),
                    recoverable=False,
                    details={"exception_type": type(e).__name__}
                )]
            )

        # record timing
        result.duration_ms = (time.time() - start) * 1000

        # log result
        if result.ok:
            self._logger.info(f"[{self.name}] completed in {result.duration_ms:.1f}ms")
        elif result.status == AgentStatus.PARTIAL:
            self._logger.warning(f"[{self.name}] partial success in {result.duration_ms:.1f}ms: {len(result.errors)} errors")
        else:
            self._logger.error(f"[{self.name}] failed in {result.duration_ms:.1f}ms: {result.errors}")

        return result

    def log(self, message: str, level: str = "info"):
        """log with agent context."""
        getattr(self._logger, level)(f"[{self.name}] {message}")
