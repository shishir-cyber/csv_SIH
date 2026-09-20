"""Comprehensive pytest coverage for Module M0 (Ingest & Capability Probe)."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any, List

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "m0_ingest_capability" / "src"))

from m0_ingest.adapters import COCOAdapter, YOLOAdapter, sha256_file  # noqa: E402
from m0_ingest.base_detector import BaseDetector, DetectorResult  # noqa: E402
from m0_ingest.encoder_binding import (  # noqa: E402
    BUNDLED_E_REF_PATH,
    DualEncoderBinding,
    IntegrityError,
    PINNED_E_REF_DIGEST,
)
from m0_ingest.probe import probe_capabilities  # noqa: E402
from schema import AccessTier, AssetRecord, CapabilityMatrix, ReferenceMode, TaskType  # noqa: E402


def _write_png(path: Path, width: int = 64, height: int = 48, color=(30, 90, 150)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color).save(path)
    return path


def _write_safetensors(path: Path) -> Path:
    header = {
        "embed.weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
        "__metadata__": {"format": "pt"},
    }
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\x00" * 16)
    return path


class _T2OnlyDetector(BaseDetector):
    def __init__(self) -> None:
        super().__init__(
            detector_id="weight_digest_check",
            required_tier=AccessTier.T2_WEIGHTS,
            supported_tasks=list(TaskType),
            requires_reference=False,
        )

    def _run(
        self,
        dataset: List[AssetRecord],
        capability_matrix: CapabilityMatrix,
        **kwargs: Any,
    ) -> DetectorResult:
        return DetectorResult(detector_id=self.detector_id, status="OK", score=0.0)


def test_coco_ingest_records_image_sha256(tmp_path: Path):
    images = tmp_path / "images"
    png = _write_png(images / "cat.jpg", 80, 60)
    coco = {
        "images": [{"id": 11, "file_name": "cat.jpg", "width": 80, "height": 60}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [{"id": 99, "image_id": 11, "category_id": 1, "bbox": [2, 4, 16, 20]}],
    }
    ann = tmp_path / "annotations" / "instances.json"
    ann.parent.mkdir()
    ann.write_text(json.dumps(coco), encoding="utf-8")

    records = COCOAdapter().ingest(ann, images, "contrib-coco", ledger=tmp_path / "ledger.jsonl")
    assert len(records) == 1
    record = records[0]
    assert record.format == "COCO"
    assert record.sha256 == sha256_file(png)
    assert len(record.sha256) == 64
    assert record.annotations[0].bbox == [2.0, 4.0, 16.0, 20.0]
    assert record.image_metadata.width == 80


def test_yolo_ingest_records_image_and_label_hashes(tmp_path: Path):
    png = _write_png(tmp_path / "images" / "obj.png", 100, 80)
    labels = tmp_path / "labels"
    labels.mkdir()
    label = labels / "obj.txt"
    label.write_text("0 0.5 0.5 0.2 0.5\n", encoding="utf-8")
    (tmp_path / "data.yaml").write_text("names:\n  0: car\ntrain: images\n", encoding="utf-8")

    records = YOLOAdapter().ingest(tmp_path, "contrib-yolo", ledger=tmp_path / "ledger.jsonl")
    assert len(records) == 1
    record = records[0]
    assert record.format == "YOLO"
    assert record.sha256 == sha256_file(png)
    assert record.annotations[0].class_name == "car"
    ledger = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    assert sha256_file(png) in ledger
    assert sha256_file(label) in ledger


def test_probe_valid_weight_file_is_t2(tmp_path: Path):
    model = _write_safetensors(tmp_path / "subject.safetensors")
    matrix = probe_capabilities(model, [], None)
    assert matrix.access_tier is AccessTier.T2_WEIGHTS
    assert matrix.model_digest == sha256_file(model)


def test_probe_none_model_is_t0():
    matrix = probe_capabilities(None, [], None)
    assert matrix.access_tier is AccessTier.T0_LABELS
    assert matrix.model_digest is None


def test_probe_detection_dataset_sets_task_type():
    records = [
        AssetRecord.from_dict(
            {
                "asset_id": "det-1",
                "file_path": "/tmp/det.jpg",
                "sha256": "c" * 64,
                "format": "COCO",
                "image_metadata": {"width": 32, "height": 32, "channels": 3},
                "annotations": [{"bbox": [1, 2, 8, 8], "category_id": 0, "class_name": "car"}],
                "contributor_id": "c1",
                "timestamp": "2026-09-19T10:00:00Z",
            }
        )
    ]
    matrix = probe_capabilities(None, records, None)
    assert matrix.task_type is TaskType.DETECTION
    assert matrix.access_tier is AccessTier.T0_LABELS


def test_base_detector_unavailable_at_t0_when_t2_required():
    detector = _T2OnlyDetector()
    matrix = CapabilityMatrix(
        access_tier=AccessTier.T0_LABELS,
        task_type=TaskType.DETECTION,
        reference_mode=ReferenceMode.UNREFERENCED,
    )
    result = detector.run([], matrix)
    assert result["status"] == "UNAVAILABLE"
    assert result["score"] is None
    assert result["reason"].startswith("UNAVAILABLE_AT_TIER")
    assert "T2_WEIGHTS" in result["reason"]
    assert "T0_LABELS" in result["reason"]


def test_dinov2_pinning_accepts_bundled_weights():
    assert BUNDLED_E_REF_PATH.is_file()
    assert sha256_file(BUNDLED_E_REF_PATH) == PINNED_E_REF_DIGEST
    binding = DualEncoderBinding()
    loaded = binding.verify_and_load_ref_encoder(BUNDLED_E_REF_PATH)
    assert loaded is not None
    assert binding.e_ref_digest == PINNED_E_REF_DIGEST


def test_dinov2_pinning_rejects_tampered_weights(tmp_path: Path):
    tampered = tmp_path / "dinov2_vits14.onnx"
    tampered.write_bytes(BUNDLED_E_REF_PATH.read_bytes() + b"\xff")
    with pytest.raises(IntegrityError, match="digest mismatch"):
        DualEncoderBinding().verify_and_load_ref_encoder(tampered)
