"""Tests for M0 dual encoder binding."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from dual_encoder_binding import (  # noqa: E402
    BUNDLED_E_REF_PATH,
    DualEncoderBinding,
    IntegrityError,
    PINNED_E_REF_DIGEST,
)
from schema import AccessTier  # noqa: E402
from shared.crypto.hashing import sha256_file  # noqa: E402


def _write_png(path: Path, color=(10, 80, 160)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 32), color).save(path)
    return path


def test_bundled_weights_match_pinned_digest():
    assert BUNDLED_E_REF_PATH.is_file()
    assert sha256_file(BUNDLED_E_REF_PATH) == PINNED_E_REF_DIGEST


def test_verify_rejects_tampered_ref_encoder(tmp_path: Path):
    decoy = tmp_path / "tampered.onnx"
    decoy.write_bytes(BUNDLED_E_REF_PATH.read_bytes() + b"\x00")
    binding = DualEncoderBinding()
    with pytest.raises(IntegrityError, match="digest mismatch"):
        binding.verify_and_load_ref_encoder(decoy)


def test_extract_features_ref_only_is_deterministic(tmp_path: Path):
    image = _write_png(tmp_path / "a.png")
    binding = DualEncoderBinding(access_tier=AccessTier.T0_LABELS)
    binding.verify_and_load_ref_encoder(BUNDLED_E_REF_PATH)

    first = binding.extract_features(image)
    second = binding.extract_features(image)

    assert first["e_sub"] is None
    assert len(first["e_ref"]) == 384
    assert first["e_ref"] == second["e_ref"]
    assert binding.e_ref_runtime == "numpy-cpu-fallback"


def test_extract_features_runs_subject_at_t1(tmp_path: Path):
    image = _write_png(tmp_path / "b.png", color=(200, 10, 10))

    def fake_sub(tensor: np.ndarray):
        return tensor.mean(axis=(2, 3)).reshape(-1)

    binding = DualEncoderBinding()
    binding.verify_and_load_ref_encoder(BUNDLED_E_REF_PATH)
    binding.bind_subject(fake_sub, AccessTier.T1_LOGITS)

    result = binding.extract_features(image)
    assert result["e_ref"] is not None
    assert result["e_sub"] is not None
    assert len(result["e_sub"]) == 3


def test_subject_skipped_at_t0_even_if_bound(tmp_path: Path):
    image = _write_png(tmp_path / "c.png")
    binding = DualEncoderBinding(e_sub=lambda tensor: np.ones(4), access_tier=AccessTier.T0_LABELS)
    binding.verify_and_load_ref_encoder(BUNDLED_E_REF_PATH)
    result = binding.extract_features(image)
    assert result["e_sub"] is None
