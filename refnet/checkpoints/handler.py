"""
checkpoint handlers - present checkpoints to users.

handlers are responsible for:
- displaying checkpoint information
- collecting user responses
- returning structured responses

usage:
    handler = ConsoleCheckpointHandler(auto_confirm=False)
    response = handler.present(checkpoint)

    # or auto-confirm all checkpoints
    handler = ConsoleCheckpointHandler(auto_confirm=True)
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Callable

from .core import (
    Checkpoint, CheckpointResponse, CheckpointType,
    FieldCheckpoint, SeedCheckpoint, DirectionCheckpoint
)

logger = logging.getLogger("refnet.checkpoints.handler")


class CheckpointHandler(ABC):
    """abstract base class for checkpoint handlers."""

    @abstractmethod
    def present(self, checkpoint: Checkpoint) -> CheckpointResponse:
        """present a checkpoint and get user response."""
        pass

    @abstractmethod
    def should_checkpoint(self, checkpoint_type: CheckpointType) -> bool:
        """check if we should present this type of checkpoint."""
        pass


class ConsoleCheckpointHandler(CheckpointHandler):
    """
    console-based checkpoint handler.

    presents checkpoints to the user via console and collects responses.
    """

    def __init__(
        self,
        auto_confirm: bool = False,
        confidence_threshold: float = 0.8,
        skip_types: Optional[set] = None,
        input_fn: Optional[Callable[[str], str]] = None
    ):
        """
        initialize console handler.

        args:
            auto_confirm: automatically confirm all checkpoints
            confidence_threshold: auto-confirm if confidence >= threshold
            skip_types: checkpoint types to skip entirely
            input_fn: custom input function (for testing)
        """
        self.auto_confirm = auto_confirm
        self.confidence_threshold = confidence_threshold
        self.skip_types = skip_types or set()
        self.input_fn = input_fn or input

    def should_checkpoint(self, checkpoint_type: CheckpointType) -> bool:
        """check if we should present this type of checkpoint."""
        if self.auto_confirm:
            return False
        return checkpoint_type not in self.skip_types

    def present(self, checkpoint: Checkpoint) -> CheckpointResponse:
        """present a checkpoint and get user response."""
        # auto-confirm if configured
        if self.auto_confirm:
            logger.info(f"[checkpoint] auto-confirming {checkpoint.checkpoint_type.value}")
            return CheckpointResponse(confirmed=True)

        # auto-confirm high confidence
        if checkpoint.confidence >= self.confidence_threshold:
            logger.info(
                f"[checkpoint] auto-confirming {checkpoint.checkpoint_type.value} "
                f"(confidence: {checkpoint.confidence:.0%})"
            )
            return CheckpointResponse(confirmed=True)

        # skip certain types
        if checkpoint.checkpoint_type in self.skip_types:
            logger.info(f"[checkpoint] skipping {checkpoint.checkpoint_type.value}")
            return CheckpointResponse(confirmed=True, skip_future=True)

        # present to user
        return self._present_checkpoint(checkpoint)

    def _present_checkpoint(self, checkpoint: Checkpoint) -> CheckpointResponse:
        """present checkpoint based on type."""
        if isinstance(checkpoint, FieldCheckpoint):
            return self._present_field_checkpoint(checkpoint)
        elif isinstance(checkpoint, SeedCheckpoint):
            return self._present_seed_checkpoint(checkpoint)
        elif isinstance(checkpoint, DirectionCheckpoint):
            return self._present_direction_checkpoint(checkpoint)
        else:
            return self._present_generic_checkpoint(checkpoint)

    def _present_field_checkpoint(self, checkpoint: FieldCheckpoint) -> CheckpointResponse:
        """present field checkpoint."""
        print("\n" + "=" * 60)
        print("FIELD VERIFICATION")
        print("=" * 60)
        print(f"\nIdentified field: {checkpoint.identified_field}")
        print(f"Confidence: {checkpoint.confidence:.0%}")
        print(f"\nContext: {checkpoint.context}")

        if checkpoint.evidence:
            print(f"\nEvidence:")
            for e in checkpoint.evidence[:5]:
                print(f"  • {e}")

        if checkpoint.suggested_journals:
            print(f"\nKey journals:")
            for j in checkpoint.suggested_journals[:5]:
                print(f"  • {j}")

        if checkpoint.suggested_authors:
            print(f"\nKnown leaders:")
            for a in checkpoint.suggested_authors[:5]:
                print(f"  • {a}")

        print(f"\nQuestion: {checkpoint.question}")
        print("\nOptions:")
        for i, opt in enumerate(checkpoint.options, 1):
            print(f"  {i}. {opt}")

        if checkpoint.alternative_fields:
            print(f"\nAlternative fields: {', '.join(checkpoint.alternative_fields)}")

        # get response
        try:
            response = self.input_fn("\nYour choice (1 to confirm, or type field name): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[auto-confirming due to non-interactive mode]")
            return CheckpointResponse(confirmed=True)

        if response in ("1", "y", "yes", ""):
            return CheckpointResponse(confirmed=True)
        elif response in ("2", "3"):
            correction = self.input_fn("Enter the correct field: ").strip()
            return CheckpointResponse(confirmed=False, correction=correction)
        else:
            # treat as field name
            return CheckpointResponse(confirmed=False, correction=response)

    def _present_seed_checkpoint(self, checkpoint: SeedCheckpoint) -> CheckpointResponse:
        """present seed checkpoint."""
        print("\n" + "=" * 60)
        print("SEED PAPER VERIFICATION")
        print("=" * 60)
        print(f"\n{checkpoint.question}")
        print(f"\n{checkpoint.context}")

        print("\nPapers found:")
        for summary in checkpoint.paper_summaries:
            print(f"  {summary}")

        print("\nOptions:")
        for i, opt in enumerate(checkpoint.options, 1):
            print(f"  {i}. {opt}")

        # get response
        try:
            response = self.input_fn("\nYour choice (1 to confirm): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[auto-confirming due to non-interactive mode]")
            return CheckpointResponse(confirmed=True)

        if response in ("1", "y", "yes", ""):
            return CheckpointResponse(confirmed=True)
        elif response == "2":
            # select specific papers
            indices = self.input_fn("Enter paper numbers to keep (e.g., 1,3,5): ").strip()
            return CheckpointResponse(
                confirmed=True,
                metadata={"selected_indices": indices}
            )
        else:
            # search again
            new_query = self.input_fn("Enter new search query: ").strip()
            return CheckpointResponse(confirmed=False, correction=new_query)

    def _present_direction_checkpoint(self, checkpoint: DirectionCheckpoint) -> CheckpointResponse:
        """present direction checkpoint."""
        print("\n" + "=" * 60)
        print("EXPANSION DIRECTION")
        print("=" * 60)
        print(f"\nCurrent topics: {', '.join(checkpoint.current_topics)}")
        print(f"Proposed directions: {', '.join(checkpoint.proposed_directions)}")
        print(f"Papers in direction: {checkpoint.papers_in_direction}")
        print(f"Confidence: {checkpoint.confidence:.0%}")

        print(f"\n{checkpoint.question}")
        print("\nOptions:")
        for i, opt in enumerate(checkpoint.options, 1):
            print(f"  {i}. {opt}")

        # get response
        try:
            response = self.input_fn("\nYour choice (1 to proceed): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[auto-confirming due to non-interactive mode]")
            return CheckpointResponse(confirmed=True)

        if response in ("1", "y", "yes", ""):
            return CheckpointResponse(confirmed=True)
        elif response == "2":
            direction = self.input_fn("Enter preferred direction: ").strip()
            return CheckpointResponse(confirmed=False, correction=direction)
        else:
            return CheckpointResponse(
                confirmed=True,
                metadata={"stay_focused": True}
            )

    def _present_generic_checkpoint(self, checkpoint: Checkpoint) -> CheckpointResponse:
        """present generic checkpoint."""
        print("\n" + "=" * 60)
        print(f"CHECKPOINT: {checkpoint.checkpoint_type.value.upper()}")
        print("=" * 60)
        print(f"\n{checkpoint.question}")
        print(f"\nSuggestion: {checkpoint.suggestion}")
        print(f"Confidence: {checkpoint.confidence:.0%}")

        if checkpoint.context:
            print(f"\nContext: {checkpoint.context}")

        if checkpoint.evidence:
            print(f"\nEvidence:")
            for e in checkpoint.evidence[:5]:
                print(f"  • {e}")

        # get response
        try:
            response = self.input_fn("\nConfirm? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[auto-confirming due to non-interactive mode]")
            return CheckpointResponse(confirmed=True)

        return CheckpointResponse(confirmed=response in ("y", "yes", ""))


class AutoConfirmHandler(CheckpointHandler):
    """handler that auto-confirms all checkpoints (for batch/CI use)."""

    def present(self, checkpoint: Checkpoint) -> CheckpointResponse:
        """auto-confirm everything."""
        logger.info(f"[checkpoint] auto-confirming {checkpoint.checkpoint_type.value}")
        return CheckpointResponse(confirmed=True)

    def should_checkpoint(self, checkpoint_type: CheckpointType) -> bool:
        """never checkpoint - auto-confirm everything."""
        return False


class CallbackCheckpointHandler(CheckpointHandler):
    """handler that calls a callback function for each checkpoint."""

    def __init__(self, callback: Callable[[Checkpoint], CheckpointResponse]):
        """
        initialize with callback.

        args:
            callback: function that takes Checkpoint and returns CheckpointResponse
        """
        self.callback = callback

    def present(self, checkpoint: Checkpoint) -> CheckpointResponse:
        """call the callback."""
        return self.callback(checkpoint)

    def should_checkpoint(self, checkpoint_type: CheckpointType) -> bool:
        """always checkpoint."""
        return True
