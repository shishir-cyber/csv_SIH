"""Registration for reference and subject encoders."""

from __future__ import annotations

import sys
from pathlib import Path

_CV_ASSURE = Path(__file__).resolve().parents[2]
if str(_CV_ASSURE) not in sys.path:
    sys.path.insert(0, str(_CV_ASSURE))

from m0_ingest.encoder_binding import (  # noqa: E402
    BUNDLED_E_REF_PATH,
    DualEncoderBinding,
    IntegrityError,
    PINNED_E_REF_DIGEST,
    configure_cpu_determinism,
)

__all__ = [
    "BUNDLED_E_REF_PATH",
    "DualEncoderBinding",
    "IntegrityError",
    "PINNED_E_REF_DIGEST",
    "configure_cpu_determinism",
]
