"""Tests for the M0 capability matrix."""

import os
import sys

# Allow this test file to import from the src/ folder next to it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from capability_matrix import (
    AccessTier,
    CapabilityMatrix,
    FragileWatermarkDetector,
    OutputSignatureVerifier,
    inspect_remote_api,
)
from schema import TaskType


def test_tier_ranking():
    """T0 should be lowest, T2 should be highest."""
    assert AccessTier.T0_LABELS.rank < AccessTier.T1_LOGITS.rank
    assert AccessTier.T1_LOGITS.rank < AccessTier.T2_WEIGHTS.rank


def test_matrix_supports_equal_tier():
    matrix = CapabilityMatrix(access_tier=AccessTier.T1_LOGITS)
    assert matrix.is_tier_supported(AccessTier.T1_LOGITS) is True


def test_matrix_supports_lower_tier():
    matrix = CapabilityMatrix(access_tier=AccessTier.T2_WEIGHTS)
    assert matrix.is_tier_supported(AccessTier.T0_LABELS) is True


def test_matrix_rejects_higher_tier():
    matrix = CapabilityMatrix(access_tier=AccessTier.T0_LABELS)
    assert matrix.is_tier_supported(AccessTier.T2_WEIGHTS) is False


def test_t0_detector_runs_on_t1_matrix():
    matrix = CapabilityMatrix(access_tier=AccessTier.T1_LOGITS)
    detector = OutputSignatureVerifier()
    result = detector.run([], matrix)
    assert result["status"] == "SUCCESS"


def test_t2_detector_blocked_on_t1_matrix():
    matrix = CapabilityMatrix(access_tier=AccessTier.T1_LOGITS)
    detector = FragileWatermarkDetector()
    result = detector.run([], matrix)
    assert result["status"] == "UNAVAILABLE"
    assert "UNAVAILABLE_AT_TIER" in result["reason"]


def test_t2_detector_runs_on_t2_matrix():
    matrix = CapabilityMatrix(access_tier=AccessTier.T2_WEIGHTS)
    detector = FragileWatermarkDetector()
    result = detector.run([], matrix)
    assert result["status"] == "SUCCESS"


def test_remote_api_detects_t1_from_confidence_field():
    def fake_api_call(sample_input):
        return {"label": "car", "confidence": 0.91}

    matrix = inspect_remote_api(fake_api_call, sample_input="test.jpg")
    assert matrix.access_tier == AccessTier.T1_LOGITS
    assert matrix.source == "remote_api"
    assert matrix.task_type == TaskType.DETECTION


def test_remote_api_detects_t0_when_only_label_returned():
    def fake_api_call(sample_input):
        return {"label": "car"}

    matrix = inspect_remote_api(fake_api_call, sample_input="test.jpg")
    assert matrix.access_tier == AccessTier.T0_LABELS
