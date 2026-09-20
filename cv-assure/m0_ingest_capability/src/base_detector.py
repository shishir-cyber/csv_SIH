"""Base detector ABC (re-exported from ``m0_ingest.base_detector``)."""

from __future__ import annotations

import sys
from pathlib import Path

_CV_ASSURE = Path(__file__).resolve().parents[2]
if str(_CV_ASSURE) not in sys.path:
    sys.path.insert(0, str(_CV_ASSURE))

from m0_ingest.base_detector import (  # noqa: E402
    BaseDetector,
    DetectorResult,
    capability_gated,
    unavailable_result,
)

__all__ = [
    "BaseDetector",
    "DetectorResult",
    "capability_gated",
    "unavailable_result",
]
