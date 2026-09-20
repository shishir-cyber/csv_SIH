"""Tests for M0 ingest / capability schemas (CV-ASSURE v2)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schema import (  # noqa: E402
    AccessTier,
    Annotation,
    AssetRecord,
    CapabilityMatrix,
    ReferenceMode,
    TaskType,
)


SAMPLE_SHA256 = "a" * 64


def _asset(**overrides) -> AssetRecord:
    payload = {
        "asset_id": "img-001",
        "file_path": "/data/images/001.jpg",
        "sha256": SAMPLE_SHA256,
        "format": "COCO",
        "image_metadata": {"width": 640, "height": 480, "channels": 3},
        "annotations": [
            {"bbox": [10.0, 20.0, 30.0, 40.0], "category_id": 1, "class_name": "car"}
        ],
        "contributor_id": "contrib-7",
        "timestamp": "2026-09-19T09:00:00Z",
    }
    payload.update(overrides)
    return AssetRecord.from_dict(payload)


def test_access_tier_ranking():
    assert AccessTier.T0_LABELS.rank < AccessTier.T1_LOGITS.rank
    assert AccessTier.T1_LOGITS.rank < AccessTier.T2_WEIGHTS.rank
    assert AccessTier.T2_WEIGHTS.satisfies(AccessTier.T0_LABELS)
    assert not AccessTier.T0_LABELS.satisfies(AccessTier.T2_WEIGHTS)


def test_asset_record_json_roundtrip(tmp_path: Path):
    record = _asset(file_path=tmp_path / "001.jpg")
    restored = AssetRecord.from_json(record.to_json())
    assert restored.asset_id == "img-001"
    assert restored.format == "COCO"
    assert restored.file_path == tmp_path / "001.jpg"
    assert restored.annotations[0].class_name == "car"
    assert json.loads(record.to_json())["sha256"] == SAMPLE_SHA256


def test_asset_record_rejects_bad_sha256():
    with pytest.raises(Exception):
        _asset(sha256="not-a-digest")


def test_asset_record_rejects_unknown_format():
    with pytest.raises(Exception):
        _asset(format="PASCAL")


def test_annotation_bbox_must_be_xywh():
    with pytest.raises(Exception):
        Annotation(bbox=[1, 2, 3], category_id=0, class_name="x")


def test_supports_detector_ok_at_matching_tier():
    matrix = CapabilityMatrix(
        access_tier=AccessTier.T2_WEIGHTS,
        task_type=TaskType.DETECTION,
        reference_mode=ReferenceMode.ATTESTED,
        model_digest=SAMPLE_SHA256,
    )
    ok, reason = matrix.supports_detector(
        "weight_digest_check",
        AccessTier.T2_WEIGHTS,
        [TaskType.DETECTION, TaskType.CLASSIFICATION],
    )
    assert ok is True
    assert reason.startswith("SUPPORTED")


def test_supports_detector_unavailable_at_tier():
    matrix = CapabilityMatrix(access_tier=AccessTier.T0_LABELS)
    ok, reason = matrix.supports_detector(
        "weight_digest_check",
        AccessTier.T2_WEIGHTS,
        [TaskType.DETECTION],
    )
    assert ok is False
    assert reason == "UNAVAILABLE_AT_TIER: requires T2, found T0"


def test_supports_detector_task_mismatch():
    matrix = CapabilityMatrix(
        access_tier=AccessTier.T2_WEIGHTS,
        task_type=TaskType.SEGMENTATION,
    )
    ok, reason = matrix.supports_detector(
        "bbox_poison_detector",
        AccessTier.T0_LABELS,
        [TaskType.DETECTION],
    )
    assert ok is False
    assert reason == "TASK_UNSUPPORTED: requires one of [DETECTION], found SEGMENTATION"


def test_capability_matrix_json_roundtrip():
    matrix = CapabilityMatrix(
        access_tier=AccessTier.T1_LOGITS,
        task_type=TaskType.CLASSIFICATION,
        reference_mode=ReferenceMode.UNREFERENCED,
        probe_timestamp="2026-09-19T09:00:00Z",
    )
    restored = CapabilityMatrix.from_json(matrix.to_json())
    assert restored.access_tier is AccessTier.T1_LOGITS
    assert restored.to_dict()["access_tier"] == "T1"
    assert restored.to_dict()["task_type"] == "CLASSIFICATION"
