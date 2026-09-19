"""Capability probe (re-exported from ``m0_ingest.probe``)."""

from __future__ import annotations

import sys
from pathlib import Path

_CV_ASSURE = Path(__file__).resolve().parents[2]
if str(_CV_ASSURE) not in sys.path:
    sys.path.insert(0, str(_CV_ASSURE))

from m0_ingest.probe import probe_capabilities  # noqa: E402

__all__ = ["probe_capabilities"]
