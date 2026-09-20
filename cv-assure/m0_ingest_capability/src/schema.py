"""Core data schemas for Module M0 (Ingest & Capability Probe).

These types are the contract between ingest adapters, the capability probe,
and downstream detectors (M1–M6). Values are JSON-serializable so they can
be persisted on the audit ledger and included in assurance reports.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


def _utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp suitable for probe records."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


class AccessTier(str, Enum):
    """Model-access level discovered by the M0 capability probe.

    Higher tiers are supersets of lower ones: labels ⊂ logits ⊂ weights.
    """

    T0_LABELS = "T0"
    """Final class labels / detections only (black-box outputs)."""

    T1_LOGITS = "T1"
    """Labels plus logits, scores, or other pre-argmax tensors."""

    T2_WEIGHTS = "T2"
    """Full parameter access (local weights / ONNX initializers)."""

    @property
    def rank(self) -> int:
        """Numeric order used to compare tiers: T0 < T1 < T2."""
        return {AccessTier.T0_LABELS: 0, AccessTier.T1_LOGITS: 1, AccessTier.T2_WEIGHTS: 2}[self]

    def satisfies(self, required: "AccessTier") -> bool:
        """Return True if this tier is at least as privileged as ``required``."""
        return self.rank >= required.rank


class TaskType(str, Enum):
    """Computer-vision task the subject model or dataset is bound to."""

    CLASSIFICATION = "CLASSIFICATION"
    DETECTION = "DETECTION"
    SEGMENTATION = "SEGMENTATION"


class ReferenceMode(str, Enum):
    """How a trusted reference profile was obtained for this asset."""

    ATTESTED = "ATTESTED"
    """Signed vendor/reference profile is available and verified."""

    BOOTSTRAPPED = "BOOTSTRAPPED"
    """Reference statistics were estimated from the ingested corpus."""

    UNREFERENCED = "UNREFERENCED"
    """No reference profile; detectors must run in absolute-threshold mode."""


class ImageMetadata(BaseModel):
    """Pixel geometry recorded at ingest time."""

    model_config = ConfigDict(extra="forbid")

    width: int = Field(..., ge=1, description="Image width in pixels.")
    height: int = Field(..., ge=1, description="Image height in pixels.")
    channels: int = Field(..., ge=1, description="Number of color channels (e.g. 1, 3, 4).")


class Annotation(BaseModel):
    """A single detection-style annotation in XYWH pixel coordinates."""

    model_config = ConfigDict(extra="forbid")

    bbox: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box as [x, y, w, h] in pixel coordinates.",
    )
    category_id: int = Field(..., description="Numeric class id from the source taxonomy.")
    class_name: str = Field(..., min_length=1, description="Human-readable class label.")
    segmentation: Optional[List[List[float]]] = Field(
        default=None,
        description="Optional polygon(s), each a flat [x1, y1, x2, y2, ...] list.",
    )

    @field_validator("bbox")
    @classmethod
    def _bbox_non_negative(cls, value: List[float]) -> List[float]:
        if any(component < 0 for component in value):
            raise ValueError("bbox values must be non-negative [x, y, w, h]")
        return value


class AssetRecord(BaseModel):
    """Normalized ingest record for one image (or image-like) asset.

    Adapters (COCO / YOLO) map native annotations onto this schema so
    downstream integrity detectors share a single representation.
    """

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(..., min_length=1, description="Stable identifier for this asset.")
    file_path: Path = Field(..., description="Filesystem path to the source image.")
    sha256: str = Field(..., description="Hex-encoded SHA-256 digest of the file bytes.")
    format: Literal["COCO", "YOLO"] = Field(..., description="Source annotation format.")
    image_metadata: ImageMetadata = Field(..., description="Width, height, and channel count.")
    annotations: List[Annotation] = Field(
        default_factory=list,
        description="Object annotations as dicts with bbox, category_id, class_name.",
    )
    contributor_id: str = Field(..., min_length=1, description="Uploader or dataset contributor.")
    timestamp: str = Field(..., description="ISO-8601 ingest timestamp.")

    @field_validator("sha256")
    @classmethod
    def _sha256_hex(cls, value: str) -> str:
        digest = value.lower().strip()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("sha256 must be a 64-character hexadecimal string")
        return digest

    @field_serializer("file_path")
    def _serialize_path(self, value: Path) -> str:
        return str(value)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dict (Path rendered as a string)."""
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """Serialize this record to a JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssetRecord":
        """Build an ``AssetRecord`` from a plain dict."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, payload: str) -> "AssetRecord":
        """Build an ``AssetRecord`` from a JSON string."""
        return cls.model_validate_json(payload)


class CapabilityMatrix(BaseModel):
    """Probe result describing what detectors are allowed to run.

    Every detector consults ``supports_detector`` before executing so that
    T2-only checks are never claimed on a T0 API.
    """

    model_config = ConfigDict(extra="forbid")

    access_tier: AccessTier = Field(..., description="Highest access tier confirmed by the probe.")
    task_type: TaskType = Field(
        default=TaskType.DETECTION,
        description="Vision task bound to this model/dataset.",
    )
    reference_mode: ReferenceMode = Field(
        default=ReferenceMode.BOOTSTRAPPED,
        description="Availability of a trusted reference.",
    )
    model_digest: Optional[str] = Field(
        default=None,
        description="SHA-256 of model weights when T2 access is available.",
    )
    reference_profile_digest: Optional[str] = Field(
        default=None,
        description="SHA-256 of the reference profile used for attested/bootstrapped mode.",
    )
    probe_timestamp: str = Field(
        default_factory=_utc_now_iso,
        description="ISO-8601 UTC time the capability probe ran.",
    )
    source: Optional[Literal["local_file", "remote_api", "unknown"]] = Field(
        default=None,
        description="Optional probe origin (local ONNX file vs remote API).",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional probe diagnostics (output node names, API keys, etc.).",
    )

    def is_tier_supported(self, required_tier: AccessTier) -> bool:
        """Return True if the probed tier meets or exceeds ``required_tier``."""
        return self.access_tier.satisfies(required_tier)

    def supports_detector(
        self,
        detector_id: str,
        required_tier: AccessTier,
        supported_tasks: List[TaskType],
    ) -> Tuple[bool, str]:
        """Decide whether ``detector_id`` may run against this matrix.

        Returns:
            A ``(allowed, reason)`` pair. ``reason`` is machine-readable, e.g.
            ``UNAVAILABLE_AT_TIER: requires T2, found T0``.
        """
        if not detector_id:
            return False, "INVALID_DETECTOR_ID: detector_id must be non-empty"

        if self.task_type not in supported_tasks:
            supported = ", ".join(task.value for task in supported_tasks) or "<none>"
            return (
                False,
                f"TASK_UNSUPPORTED: requires one of [{supported}], found {self.task_type.value}",
            )

        if not self.access_tier.satisfies(required_tier):
            return (
                False,
                (
                    "UNAVAILABLE_AT_TIER: "
                    f"requires {required_tier.value}, found {self.access_tier.value}"
                ),
            )

        return True, f"SUPPORTED: detector={detector_id}"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dict with enum values as strings."""
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """Serialize this matrix to a JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapabilityMatrix":
        """Build a ``CapabilityMatrix`` from a plain dict."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, payload: str) -> "CapabilityMatrix":
        """Build a ``CapabilityMatrix`` from a JSON string."""
        return cls.model_validate_json(payload)
