"""Tests for M0 COCO / YOLO ingest adapters."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from adapters import (  # noqa: E402
    COCOAdapter,
    CorruptAnnotationError,
    MissingAssetError,
    YOLOAdapter,
    normalize_dataset,
    sha256_file,
    yolo_norm_to_xywh,
)
from shared.audit_ledger.ledger_writer import LedgerWriter  # noqa: E402


def _write_png(path: Path, width: int = 100, height: int = 80, color=(20, 40, 60)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color).save(path)
    return path


def test_sha256_file_matches_hashlib(tmp_path: Path):
    target = tmp_path / "blob.bin"
    target.write_bytes(b"cv-assure" * 4096)
    expected = hashlib.sha256(target.read_bytes()).hexdigest()
    assert sha256_file(target, chunk_size=64) == expected


def test_yolo_norm_to_xywh_center_box():
    # 100x80 image, box centered with w=0.2 h=0.5 → 20x40 box at (40, 20)
    bbox = yolo_norm_to_xywh(0.5, 0.5, 0.2, 0.5, 100, 80)
    assert bbox == pytest.approx([40.0, 20.0, 20.0, 40.0])


def test_coco_adapter_builds_asset_records(tmp_path: Path):
    images = tmp_path / "images"
    png = _write_png(images / "dog.jpg", 64, 48)
    coco = {
        "images": [{"id": 7, "file_name": "dog.jpg", "width": 64, "height": 48}],
        "categories": [{"id": 2, "name": "dog"}],
        "annotations": [{"id": 1, "image_id": 7, "category_id": 2, "bbox": [4, 6, 10, 12]}],
    }
    ann = tmp_path / "annotations" / "instances.json"
    ann.parent.mkdir()
    ann.write_text(json.dumps(coco), encoding="utf-8")
    ledger_path = tmp_path / "ledger.jsonl"

    records = COCOAdapter().ingest(ann, images, "contrib-1", ledger=ledger_path)
    assert len(records) == 1
    record = records[0]
    assert record.asset_id == "coco-7"
    assert record.format == "COCO"
    assert record.sha256 == sha256_file(png)
    assert record.image_metadata.width == 64
    assert record.image_metadata.channels == 3
    assert record.annotations[0].class_name == "dog"
    assert record.annotations[0].bbox == [4.0, 6.0, 10.0, 12.0]
    lines = ledger_path.read_text(encoding="utf-8").strip().splitlines()
    assert any("image_sha256" in line and record.sha256 in line for line in lines)


def test_yolo_adapter_converts_boxes_and_hashes_labels(tmp_path: Path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    png = _write_png(images / "a.png", 100, 80)
    labels.mkdir()
    label = labels / "a.txt"
    label.write_text("0 0.5 0.5 0.2 0.5\n", encoding="utf-8")
    (tmp_path / "data.yaml").write_text(
        "names:\n  0: car\nnc: 1\ntrain: images\n",
        encoding="utf-8",
    )
    ledger = LedgerWriter(tmp_path / "ledger.jsonl")

    records = YOLOAdapter().ingest(tmp_path, "contrib-y", ledger=ledger)
    assert len(records) == 1
    record = records[0]
    assert record.format == "YOLO"
    assert record.sha256 == sha256_file(png)
    assert record.annotations[0].class_name == "car"
    assert record.annotations[0].bbox == pytest.approx([40.0, 20.0, 20.0, 40.0])
    events = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    assert sha256_file(label) in events
    assert record.sha256 in events


def test_normalize_dataset_autodetects_coco(tmp_path: Path):
    images = tmp_path / "images"
    _write_png(images / "x.png", 32, 32)
    coco = {
        "images": [{"id": 1, "file_name": "x.png", "width": 32, "height": 32}],
        "categories": [{"id": 0, "name": "n"}],
        "annotations": [],
    }
    ann = tmp_path / "annotations" / "inst.json"
    ann.parent.mkdir()
    ann.write_text(json.dumps(coco), encoding="utf-8")

    records = normalize_dataset(tmp_path, "auto", "c1", ledger=tmp_path / "l.jsonl")
    assert len(records) == 1
    assert records[0].format == "COCO"


def test_normalize_dataset_autodetects_yolo(tmp_path: Path):
    _write_png(tmp_path / "images" / "z.png", 16, 16)
    (tmp_path / "labels").mkdir()
    (tmp_path / "data.yaml").write_text("names: [cat]\n", encoding="utf-8")
    records = normalize_dataset(tmp_path, "auto", "c2", ledger=tmp_path / "l.jsonl")
    assert len(records) == 1
    assert records[0].format == "YOLO"


def test_missing_dataset_raises(tmp_path: Path):
    with pytest.raises(MissingAssetError):
        normalize_dataset(tmp_path / "nope", "auto", "c3")


def test_corrupt_coco_json_raises(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    ann = tmp_path / "ann.json"
    ann.write_text("{not-json", encoding="utf-8")
    with pytest.raises(CorruptAnnotationError):
        COCOAdapter().ingest(ann, images, "c", ledger=tmp_path / "l.jsonl", strict=True)


def test_bad_image_header_skipped_unless_strict(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    bad = images / "bad.png"
    bad.write_bytes(b"this is not a png")
    coco = {
        "images": [{"id": 1, "file_name": "bad.png", "width": 1, "height": 1}],
        "categories": [],
        "annotations": [],
    }
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps(coco), encoding="utf-8")

    skipped = COCOAdapter().ingest(ann, images, "c", ledger=tmp_path / "l.jsonl", strict=False)
    assert skipped == []

    with pytest.raises(Exception):
        COCOAdapter().ingest(ann, images, "c", ledger=tmp_path / "l2.jsonl", strict=True)
