"""Tests for the M0 capability matrix."""

import sys
import os

# Allow this test file to import from the src/ folder next to it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from capability_matrix import (
    AccessTier,
    CapabilityMatrix,
    BaseDetector,
    OutputSignatureVerifier,
    FragileWatermarkDetector,
    inspect_remote_api,
)


def test_tier_ranking():
    """T0 should be lowest, T2 should be highest."""
    assert AccessTier.T0_BLACK_BOX.rank < AccessTier.T1_GREY_BOX.rank
    assert AccessTier.T1_GREY_BOX.rank < AccessTier.T2_WHITE_BOX.rank


def test_matrix_supports_equal_tier():
    matrix = CapabilityMatrix(access_tier=AccessTier.T1_GREY_BOX)
    assert matrix.is_tier_supported(AccessTier.T1_GREY_BOX) is True


def test_matrix_supports_lower_tier():
    matrix = CapabilityMatrix(access_tier=AccessTier.T2_WHITE_BOX)
    assert matrix.is_tier_supported(AccessTier.T0_BLACK_BOX) is True


def test_matrix_rejects_higher_tier():
    matrix = CapabilityMatrix(access_tier=AccessTier.T0_BLACK_BOX)
    assert matrix.is_tier_supported(AccessTier.T2_WHITE_BOX) is False


def test_t0_detector_runs_on_t1_matrix():
    matrix = CapabilityMatrix(access_tier=AccessTier.T1_GREY_BOX)
    detector = OutputSignatureVerifier()
    result = detector.run(matrix, input_data="sample.png")
    assert result["status"] == "SUCCESS"


def test_t2_detector_blocked_on_t1_matrix():
    matrix = CapabilityMatrix(access_tier=AccessTier.T1_GREY_BOX)
    detector = FragileWatermarkDetector()
    result = detector.run(matrix, input_data="sample.png")
    assert result["status"] == "UNAVAILABLE_AT_TIER"


def test_t2_detector_runs_on_t2_matrix():
    matrix = CapabilityMatrix(access_tier=AccessTier.T2_WHITE_BOX)
    detector = FragileWatermarkDetector()
    result = detector.run(matrix, input_data="sample.png")
    assert result["status"] == "SUCCESS"


def test_remote_api_detects_t1_from_confidence_field():
    def fake_api_call(sample_input):
        return {"label": "car", "confidence": 0.91}

    matrix = inspect_remote_api(fake_api_call, sample_input="test.jpg")
    assert matrix.access_tier == AccessTier.T1_GREY_BOX
    assert matrix.source == "remote_api"


def test_remote_api_detects_t0_when_only_label_returned():
    def fake_api_call(sample_input):
        return {"label": "car"}

    matrix = inspect_remote_api(fake_api_call, sample_input="test.jpg")
    assert matrix.access_tier == AccessTier.T0_BLACK_BOX