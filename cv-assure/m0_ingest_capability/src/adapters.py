"""COCO and YOLO ingest adapters (re-exported from ``m0_ingest.adapters``)."""

from __future__ import annotations

import sys
from pathlib import Path

_CV_ASSURE = Path(__file__).resolve().parents[2]
if str(_CV_ASSURE) not in sys.path:
    sys.path.insert(0, str(_CV_ASSURE))

from m0_ingest.adapters import (  # noqa: E402
    BadImageHeaderError,
    COCOAdapter,
    CorruptAnnotationError,
    IngestError,
    MissingAssetError,
    YOLOAdapter,
    detect_format,
    normalize_dataset,
    sha256_file,
    yolo_norm_to_xywh,
)

__all__ = [
    "BadImageHeaderError",
    "COCOAdapter",
    "CorruptAnnotationError",
    "IngestError",
    "MissingAssetError",
    "YOLOAdapter",
    "detect_format",
    "normalize_dataset",
    "sha256_file",
    "yolo_norm_to_xywh",
]
