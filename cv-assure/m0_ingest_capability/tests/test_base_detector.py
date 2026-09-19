"""Tests for M0 BaseDetector capability gating."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from base_detector import BaseDetector, DetectorResult  # noqa: E402
from schema import AccessTier, AssetRecord, CapabilityMatrix, ReferenceMode, TaskType  # noqa: E402


class NeuralCleanseDetector(BaseDetector):
    def __init__(self) -> None:
        super().__init__(
            detector_id="neural_cleanse",
            required_tier=AccessTier.T2_WEIGHTS,
            supported_tasks=[TaskType.CLASSIFICATION],
            requires_reference=True,
        )

    def _run(
        self,
        dataset: List[AssetRecord],
        capability_matrix: CapabilityMatrix,
        **kwargs: Any,
    ) -> DetectorResult:
        return DetectorResult(detector_id=self.detector_id, status="OK", score=0.1)


def test_unavailable_at_tier_does_not_raise():
    detector = NeuralCleanseDetector()
    matrix = CapabilityMatrix(
        access_tier=AccessTier.T0_LABELS,
        task_type=TaskType.CLASSIFICATION,
        reference_mode=ReferenceMode.ATTESTED,
    )
    result = detector.run([], matrix)
    assert result["status"] == "UNAVAILABLE"
    assert result["score"] is None
    assert result["reason"] == "UNAVAILABLE_AT_TIER: requires T2_WEIGHTS, found T0_LABELS"


def test_unavailable_for_task_type():
    detector = NeuralCleanseDetector()
    matrix = CapabilityMatrix(
        access_tier=AccessTier.T2_WEIGHTS,
        task_type=TaskType.DETECTION,
        reference_mode=ReferenceMode.ATTESTED,
    )
    ok, reason = detector.check_availability(matrix)
    assert ok is False
    assert reason == "NEURAL_CLEANSE_UNAVAILABLE_FOR_TASK_TYPE"
    result = detector.run([], matrix)
    assert result.status == "UNAVAILABLE"
    assert result.reason == reason


def test_unavailable_without_reference():
    detector = NeuralCleanseDetector()
    matrix = CapabilityMatrix(
        access_tier=AccessTier.T2_WEIGHTS,
        task_type=TaskType.CLASSIFICATION,
        reference_mode=ReferenceMode.UNREFERENCED,
    )
    ok, reason = detector.check_availability(matrix)
    assert ok is False
    assert reason == "NEURAL_CLEANSE_UNAVAILABLE_WITHOUT_REFERENCE"


def test_run_executes_when_matrix_matches():
    detector = NeuralCleanseDetector()
    matrix = CapabilityMatrix(
        access_tier=AccessTier.T2_WEIGHTS,
        task_type=TaskType.CLASSIFICATION,
        reference_mode=ReferenceMode.BOOTSTRAPPED,
    )
    result = detector.run([], matrix)
    assert result["status"] == "OK"
    assert result["score"] == 0.1
