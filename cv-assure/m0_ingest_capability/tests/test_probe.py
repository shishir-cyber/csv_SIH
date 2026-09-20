"""Tests for the M0 capability probe engine."""

from __future__ import annotations

import json
import struct
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from probe import probe_capabilities  # noqa: E402
from schema import AccessTier, AssetRecord, ReferenceMode, TaskType  # noqa: E402
from shared.crypto.signer import sign_hex, write_keypair  # noqa: E402
from shared.crypto.hashing import sha256_file  # noqa: E402
from shared.reference_profile.profile_loader import canonical_profile_bytes  # noqa: E402

SAMPLE_SHA = "b" * 64


def _record(**overrides) -> AssetRecord:
    payload = {
        "asset_id": "a1",
        "file_path": "/tmp/a.jpg",
        "sha256": SAMPLE_SHA,
        "format": "COCO",
        "image_metadata": {"width": 32, "height": 32, "channels": 3},
        "annotations": [],
        "contributor_id": "c1",
        "timestamp": "2026-09-19T10:00:00Z",
    }
    payload.update(overrides)
    return AssetRecord.from_dict(payload)


def _write_safetensors(path: Path) -> Path:
    header = {
        "embed.weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
        "__metadata__": {"format": "pt"},
    }
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\x00" * 16)
    return path


def test_no_model_empty_records_is_t0_classification_unreferenced():
    matrix = probe_capabilities(None, [], None)
    assert matrix.access_tier is AccessTier.T0_LABELS
    assert matrix.task_type is TaskType.CLASSIFICATION
    assert matrix.reference_mode is ReferenceMode.UNREFERENCED
    assert matrix.model_digest is None
    assert matrix.reference_profile_digest is None


def test_safetensors_weights_are_t2_and_hashed(tmp_path: Path):
    model = _write_safetensors(tmp_path / "model.safetensors")
    records = [
        _record(
            annotations=[{"bbox": [1, 2, 3, 4], "category_id": 0, "class_name": "car"}],
            contributor_id="lab-a",
        )
    ]
    matrix = probe_capabilities(model, records, None)
    assert matrix.access_tier is AccessTier.T2_WEIGHTS
    assert matrix.source == "local_file"
    assert matrix.model_digest == sha256_file(model)
    assert matrix.task_type is TaskType.DETECTION
    assert matrix.reference_mode is ReferenceMode.BOOTSTRAPPED


def test_pytorch_zip_checkpoint_is_t2(tmp_path: Path):
    model = tmp_path / "weights.pt"
    with zipfile.ZipFile(model, "w") as archive:
        archive.writestr("data.pkl", b"\x80\x04stub")
    matrix = probe_capabilities(model, [], None)
    assert matrix.access_tier is AccessTier.T2_WEIGHTS
    assert matrix.model_digest == sha256_file(model)


def test_api_url_is_t1_logits():
    matrix = probe_capabilities(Path("https://api.example.test/v1/predict"), [], None)
    assert matrix.access_tier is AccessTier.T1_LOGITS
    assert matrix.source == "remote_api"
    assert matrix.model_digest is None


def test_api_url_string_is_t1_logits():
    matrix = probe_capabilities("https://api.example.test/v1/predict", [], None)  # type: ignore[arg-type]
    assert matrix.access_tier is AccessTier.T1_LOGITS


def test_api_descriptor_json_logits(tmp_path: Path):
    spec = tmp_path / "api.json"
    spec.write_text(json.dumps({"endpoint": "https://x.test/infer", "outputs": ["logits"]}), encoding="utf-8")
    matrix = probe_capabilities(spec, [], None)
    assert matrix.access_tier is AccessTier.T1_LOGITS
    assert matrix.model_digest == sha256_file(spec)


def test_segmentation_polygons_win_over_boxes():
    records = [
        _record(
            annotations=[
                {
                    "bbox": [0, 0, 10, 10],
                    "category_id": 1,
                    "class_name": "person",
                    "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
                }
            ]
        )
    ]
    matrix = probe_capabilities(None, records, None)
    assert matrix.task_type is TaskType.SEGMENTATION


def test_attested_reference_profile(tmp_path: Path):
    private_key = tmp_path / "private.pem"
    public_key = tmp_path / "public.pem"
    write_keypair(private_key, public_key)

    profile_path = tmp_path / "reference_profile.yaml"
    unsigned = {
        "name": "unit-test-profile",
        "status": "attested",
        "attestation": {"algorithm": "Ed25519", "public_key_path": str(public_key)},
    }
    profile_path.write_text(yaml.safe_dump(unsigned), encoding="utf-8")
    signature = sign_hex(canonical_profile_bytes(unsigned), private_key)
    unsigned["attestation"]["signature"] = signature
    profile_path.write_text(yaml.safe_dump(unsigned), encoding="utf-8")

    matrix = probe_capabilities(None, [], profile_path)
    assert matrix.reference_mode is ReferenceMode.ATTESTED
    assert matrix.reference_profile_digest == sha256_file(profile_path)


def test_unsigned_profile_with_cohorts_is_bootstrapped(tmp_path: Path):
    profile_path = tmp_path / "reference_profile.yaml"
    profile_path.write_text("name: boot\nstatus: scaffold\nclean_set: []\n", encoding="utf-8")
    records = [_record(contributor_id="c1"), _record(asset_id="a2", contributor_id="c2")]
    matrix = probe_capabilities(None, records, profile_path)
    assert matrix.reference_mode is ReferenceMode.BOOTSTRAPPED
    assert matrix.reference_profile_digest == sha256_file(profile_path)
