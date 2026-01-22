#!/usr/bin/env python3
"""
test verification, field resolver, tiered search, and checkpoints.

run with: pytest test_verification.py -v
or: python test_verification.py (for manual run)
"""

import sys
sys.path.insert(0, '/storage/kiran-stuff/ref-network-app')

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from refnet.core.models import Paper
from refnet.agents.base import AgentResult, AgentStatus

# verification imports
from refnet.verification import (
    Check, VerificationResult, VerificationReport, Verifier, Severity,
    verify_seed_resolver, verify_field_resolver
)
from refnet.agents.seed_resolver import ResolvedSeed
from refnet.agents.field_resolver import FieldResolver, FieldProfile, FieldResolution

# search imports
from refnet.search import TieredSearchStrategy, TieredSearchResult

# checkpoint imports
from refnet.checkpoints import (
    CheckpointType, CheckpointResponse, FieldCheckpoint,
    create_field_checkpoint, ConsoleCheckpointHandler
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_paper():
    """a sample paper for testing."""
    return Paper(
        id="test-paper-1",
        title="Palladium-Catalyzed Cross-Coupling Reactions",
        year=2020,
        venue="Nature Chemistry",
        authors=["John Hartwig", "Jane Doe"],
        citation_count=150,
        concepts=[
            {"name": "Palladium", "score": 0.9},
            {"name": "Catalysis", "score": 0.8},
            {"name": "Cross-coupling", "score": 0.7}
        ]
    )


@pytest.fixture
def organic_chemistry_profile():
    """organic chemistry field profile."""
    return FieldProfile(
        name="Organic Chemistry",
        aliases=["orgo", "synthetic chemistry"],
        parent_field="Chemistry",
        tier1_journals=["Nature", "Science", "Nature Chemistry", "JACS"],
        tier2_journals=["J. Org. Chem.", "Organic Letters"],
        tier3_sources=["any"],
        known_leaders=["John Hartwig", "Phil Baran"],
        key_concepts=["catalysis", "synthesis", "reaction", "palladium"]
    )


@pytest.fixture
def sample_papers():
    """list of sample papers with different venues."""
    return [
        Paper(id="p1", title="Paper 1", venue="Nature", year=2020, citation_count=100),
        Paper(id="p2", title="Paper 2", venue="Science", year=2020, citation_count=90),
        Paper(id="p3", title="Paper 3", venue="J. Org. Chem.", year=2020, citation_count=50),
        Paper(id="p4", title="Paper 4", venue="Random Journal", year=2020, citation_count=20),
        Paper(id="p5", title="Paper 5", venue="Organic Letters", year=2020, citation_count=40),
        Paper(id="p6", title="Paper 6", venue="Another Journal", year=2020, citation_count=10),
    ]


# =============================================================================
# Verification Tests
# =============================================================================

class TestVerificationCore:
    """test core verification classes."""

    def test_check_creation(self):
        """test creating a check."""
        check = Check(
            name="test_check",
            passed=True,
            severity=Severity.ERROR,
            message="Test passed"
        )
        assert check.passed
        assert check.severity == Severity.ERROR
        assert check.name == "test_check"

    def test_verification_result(self):
        """test verification result aggregation."""
        result = VerificationResult(
            agent_name="TestAgent",
            input_summary="test input"
        )

        result.add_check(Check("c1", True, Severity.ERROR, "ok"))
        result.add_check(Check("c2", False, Severity.WARNING, "warn"))
        result.add_check(Check("c3", True, Severity.INFO, "info"))

        assert result.passed  # no ERROR failures
        assert len(result.warnings) == 1
        assert len(result.checks) == 3

    def test_verification_result_fails_on_error(self):
        """test that ERROR failures cause overall failure."""
        result = VerificationResult(
            agent_name="TestAgent",
            input_summary="test input"
        )

        result.add_check(Check("c1", False, Severity.ERROR, "error!"))
        result.add_check(Check("c2", True, Severity.INFO, "ok"))

        assert not result.passed
        assert len(result.errors) == 1

    def test_verification_report(self):
        """test verification report aggregation."""
        report = VerificationReport()

        r1 = VerificationResult("Agent1", "input1")
        r1.add_check(Check("c1", True, Severity.ERROR, "ok"))
        report.add_result(r1)

        r2 = VerificationResult("Agent2", "input2")
        r2.add_check(Check("c2", False, Severity.WARNING, "warn"))
        report.add_result(r2)

        assert report.overall_passed  # no ERROR failures
        assert report.failed_warnings == 1
        assert len(report.agent_results) == 2


class TestSeedResolverVerification:
    """test seed resolver verification."""

    def test_valid_seed_result(self, sample_paper):
        """test verification of valid seed result."""
        resolved = ResolvedSeed(
            paper=sample_paper,
            input_type="title",
            input_value="palladium catalysis"
        )

        checks = verify_seed_resolver("palladium catalysis", resolved)

        # should pass all checks
        errors = [c for c in checks if c.severity == Severity.ERROR and not c.passed]
        assert len(errors) == 0, f"Unexpected errors: {[c.name for c in errors]}"

    def test_null_seed_result(self):
        """test verification of null seed result."""
        checks = verify_seed_resolver("test query", None)

        # should fail result_exists check
        result_check = next((c for c in checks if c.name == "result_exists"), None)
        assert result_check is not None
        assert not result_check.passed

    def test_seed_without_title(self):
        """test verification catches missing title."""
        paper = Paper(id="test-1", title="")  # empty title
        resolved = ResolvedSeed(paper=paper, input_type="title", input_value="test")

        checks = verify_seed_resolver("test", resolved)

        required_check = next((c for c in checks if c.name == "has_required_fields"), None)
        assert required_check is not None
        assert not required_check.passed


class TestFieldResolverVerification:
    """test field resolver verification."""

    def test_valid_field_result(self, sample_paper, organic_chemistry_profile):
        """test verification of valid field result."""
        resolution = FieldResolution(
            primary_field=organic_chemistry_profile,
            confidence=0.8,
            evidence=["matched concepts"],
            suggested_strategy="tier1_first"
        )

        checks = verify_field_resolver(sample_paper, None, resolution)

        errors = [c for c in checks if c.severity == Severity.ERROR and not c.passed]
        assert len(errors) == 0, f"Unexpected errors: {[c.name for c in errors]}"

    def test_null_field_result(self, sample_paper):
        """test verification of null field result."""
        checks = verify_field_resolver(sample_paper, None, None)

        result_check = next((c for c in checks if c.name == "result_exists"), None)
        assert result_check is not None
        assert not result_check.passed

    def test_ocean_strategy_valid(self, sample_paper):
        """test that ocean strategy is considered valid."""
        profile = FieldProfile(name="General Science", tier1_journals=["Nature"])
        resolution = FieldResolution(
            primary_field=profile,
            confidence=0.0,
            evidence=["ocean mode"],
            suggested_strategy="ocean"
        )

        checks = verify_field_resolver(sample_paper, None, resolution)

        strategy_check = next((c for c in checks if c.name == "strategy_valid"), None)
        assert strategy_check is not None
        assert strategy_check.passed


# =============================================================================
# Field Resolver Tests
# =============================================================================

class TestFieldResolver:
    """test field resolver agent."""

    def test_resolve_organic_chemistry(self, sample_paper):
        """test resolving organic chemistry field."""
        resolver = FieldResolver()
        result = resolver.execute(seed_paper=sample_paper)

        assert result.ok
        assert result.data.primary_field.name == "Organic Chemistry"

    def test_resolve_from_query(self):
        """test resolving field from query."""
        resolver = FieldResolver()
        result = resolver.execute(query="CRISPR gene editing genome")

        assert result.ok
        # should match genomics or biochemistry
        assert result.data.primary_field.name in ["Genomics", "Biochemistry"]

    def test_fallback_to_ocean_mode(self):
        """test fallback to ocean mode for unknown field."""
        resolver = FieldResolver()
        result = resolver.execute(query="quantum entanglement teleportation")

        assert result.ok
        assert result.data.primary_field.name == "General Science"
        assert result.data.suggested_strategy == "ocean"
        assert result.data.confidence == 0.0

    def test_get_profile_by_name(self):
        """test getting profile by name."""
        resolver = FieldResolver()

        profile = resolver.get_profile("Origins of Life")
        assert profile is not None
        assert profile.name == "Origins of Life"

        # test alias
        profile2 = resolver.get_profile("prebiotic chemistry")
        assert profile2 is not None
        assert profile2.name == "Origins of Life"

    def test_list_fields(self):
        """test listing all fields."""
        resolver = FieldResolver()
        fields = resolver.list_fields()

        assert len(fields) >= 9
        assert "Organic Chemistry" in fields
        assert "Machine Learning" in fields


# =============================================================================
# Tiered Search Tests
# =============================================================================

class TestTieredSearch:
    """test tiered search strategy."""

    def test_prioritize_by_tier(self, organic_chemistry_profile, sample_papers):
        """test paper prioritization by tier."""
        strategy = TieredSearchStrategy(organic_chemistry_profile)
        result = strategy.prioritize_papers(sample_papers, limit=10)

        # first papers should be tier 1 (Nature, Science)
        assert result.papers[0].venue == "Nature"
        assert result.papers[1].venue == "Science"
        assert result.tier1_count == 2

    def test_ocean_mode(self, organic_chemistry_profile, sample_papers):
        """test ocean mode (no filtering)."""
        strategy = TieredSearchStrategy(organic_chemistry_profile)
        result = strategy.prioritize_papers(sample_papers, limit=10, strategy="ocean")

        # should return papers in original order
        assert len(result.papers) == len(sample_papers)
        assert result.tier3_count == len(sample_papers)
        assert "ocean mode" in result.search_summary

    def test_tier1_only(self, organic_chemistry_profile, sample_papers):
        """test tier1-only filtering."""
        strategy = TieredSearchStrategy(organic_chemistry_profile)
        result = strategy.prioritize_papers(sample_papers, limit=10, strategy="tier1_only")

        # should only have tier 1 papers
        assert result.tier1_count == 2
        assert len(result.papers) == 2

    def test_get_tier(self, organic_chemistry_profile):
        """test getting tier for individual paper."""
        strategy = TieredSearchStrategy(organic_chemistry_profile)

        p1 = Paper(id="1", title="Test", venue="Nature")
        assert strategy.get_tier(p1) == 1

        p2 = Paper(id="2", title="Test", venue="J. Org. Chem.")
        assert strategy.get_tier(p2) == 2

        p3 = Paper(id="3", title="Test", venue="Unknown Journal")
        assert strategy.get_tier(p3) == 3

    def test_is_high_impact(self, organic_chemistry_profile):
        """test high impact detection."""
        strategy = TieredSearchStrategy(organic_chemistry_profile)

        p1 = Paper(id="1", title="Test", venue="Nature")
        assert strategy.is_high_impact(p1)

        p2 = Paper(id="2", title="Test", venue="Random Journal")
        assert not strategy.is_high_impact(p2)


# =============================================================================
# Checkpoint Tests
# =============================================================================

class TestCheckpoints:
    """test checkpoint system."""

    def test_create_field_checkpoint(self, organic_chemistry_profile):
        """test creating field checkpoint."""
        resolution = FieldResolution(
            primary_field=organic_chemistry_profile,
            confidence=0.8,
            evidence=["matched concepts"],
            suggested_strategy="tier1_first"
        )

        checkpoint = create_field_checkpoint(resolution)

        assert checkpoint.checkpoint_type == CheckpointType.FIELD
        assert checkpoint.identified_field == "Organic Chemistry"
        assert checkpoint.confidence == 0.8
        assert len(checkpoint.suggested_journals) > 0

    def test_auto_confirm_handler(self, organic_chemistry_profile):
        """test auto-confirm handler."""
        handler = ConsoleCheckpointHandler(auto_confirm=True)

        resolution = FieldResolution(
            primary_field=organic_chemistry_profile,
            confidence=0.5,
            evidence=["test"],
            suggested_strategy="balanced"
        )
        checkpoint = create_field_checkpoint(resolution)

        response = handler.present(checkpoint)
        assert response.confirmed

    def test_confidence_threshold_handler(self, organic_chemistry_profile):
        """test confidence threshold auto-confirm."""
        # high threshold - should NOT auto-confirm low confidence
        handler = ConsoleCheckpointHandler(
            auto_confirm=False,
            confidence_threshold=0.9,
            input_fn=lambda x: "1"  # mock user input: confirm
        )

        resolution = FieldResolution(
            primary_field=organic_chemistry_profile,
            confidence=0.5,  # below threshold
            evidence=["test"],
            suggested_strategy="balanced"
        )
        checkpoint = create_field_checkpoint(resolution)

        # should ask user (mocked to confirm)
        response = handler.present(checkpoint)
        assert response.confirmed

    def test_checkpoint_response(self):
        """test checkpoint response structure."""
        response = CheckpointResponse(
            confirmed=False,
            correction="Machine Learning",
            metadata={"reason": "user specified"}
        )

        assert not response.confirmed
        assert response.correction == "Machine Learning"
        assert response.metadata["reason"] == "user specified"


# =============================================================================
# Integration Tests
# =============================================================================

class TestVerifierIntegration:
    """test full verifier workflow."""

    def test_verifier_workflow(self, sample_paper, organic_chemistry_profile):
        """test complete verification workflow."""
        verifier = Verifier()

        # verify seed resolver
        resolved_seed = ResolvedSeed(
            paper=sample_paper,
            input_type="title",
            input_value="palladium catalysis"
        )
        seed_result = AgentResult[ResolvedSeed](status=AgentStatus.SUCCESS)
        seed_result.data = resolved_seed

        vr1 = verifier.verify_seed_resolver("palladium catalysis", seed_result)
        assert vr1.passed

        # verify field resolver
        resolution = FieldResolution(
            primary_field=organic_chemistry_profile,
            confidence=0.8,
            evidence=["matched"],
            suggested_strategy="tier1_first"
        )
        field_result = AgentResult[FieldResolution](status=AgentStatus.SUCCESS)
        field_result.data = resolution

        vr2 = verifier.verify_field_resolver(sample_paper, None, field_result)
        assert vr2.passed

        # finalize report
        report = verifier.finalize()
        assert report.overall_passed
        assert len(report.agent_results) == 2


# =============================================================================
# Run Tests
# =============================================================================

def run_manual_tests():
    """run tests manually without pytest."""
    print("=" * 60)
    print("Running Verification Tests")
    print("=" * 60)

    # create fixtures
    paper = Paper(
        id="test-paper-1",
        title="Palladium-Catalyzed Cross-Coupling Reactions",
        year=2020,
        venue="Nature Chemistry",
        authors=["John Hartwig", "Jane Doe"],
        citation_count=150,
        concepts=[
            {"name": "Palladium", "score": 0.9},
            {"name": "Catalysis", "score": 0.8}
        ]
    )

    profile = FieldProfile(
        name="Organic Chemistry",
        aliases=["orgo"],
        parent_field="Chemistry",
        tier1_journals=["Nature", "Science", "Nature Chemistry"],
        tier2_journals=["J. Org. Chem."],
        known_leaders=["John Hartwig"],
        key_concepts=["catalysis", "palladium"]
    )

    # test 1: field resolver
    print("\n1. Testing FieldResolver...")
    resolver = FieldResolver()
    result = resolver.execute(seed_paper=paper)
    print(f"   Field: {result.data.primary_field.name}")
    print(f"   Confidence: {result.data.confidence:.0%}")
    assert result.ok, "FieldResolver failed"
    print("   PASSED")

    # test 2: tiered search
    print("\n2. Testing TieredSearch...")
    papers = [
        Paper(id="1", title="P1", venue="Nature", citation_count=100),
        Paper(id="2", title="P2", venue="Random", citation_count=50),
        Paper(id="3", title="P3", venue="Science", citation_count=80),
    ]
    strategy = TieredSearchStrategy(profile)
    result = strategy.prioritize_papers(papers, limit=10)
    print(f"   Tier1: {result.tier1_count}, Tier2: {result.tier2_count}, Tier3: {result.tier3_count}")
    assert result.papers[0].venue == "Nature", "Tier prioritization failed"
    print("   PASSED")

    # test 3: checkpoints
    print("\n3. Testing Checkpoints...")
    resolution = FieldResolution(
        primary_field=profile,
        confidence=0.8,
        evidence=["test"],
        suggested_strategy="tier1_first"
    )
    checkpoint = create_field_checkpoint(resolution)
    print(f"   Checkpoint type: {checkpoint.checkpoint_type.value}")
    print(f"   Question: {checkpoint.question[:50]}...")
    assert checkpoint.identified_field == "Organic Chemistry"
    print("   PASSED")

    # test 4: verifier
    print("\n4. Testing Verifier...")
    verifier = Verifier()
    resolved = ResolvedSeed(paper=paper, input_type="title", input_value="test")
    seed_result = AgentResult[ResolvedSeed](status=AgentStatus.SUCCESS)
    seed_result.data = resolved

    vr = verifier.verify_seed_resolver("test", seed_result)
    print(f"   Passed: {vr.passed}, Confidence: {vr.confidence:.0%}")
    assert vr.passed, "Verification failed"
    print("   PASSED")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    # check if pytest is available
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not available, running manual tests...")
        run_manual_tests()
