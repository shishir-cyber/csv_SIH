"""Base detector ABC and capability-gating for Module M0.

Downstream modules (M1–M4) subclass ``BaseDetector``. The public ``run``
entry point never raises for a capability miss: it returns a standard
``UNAVAILABLE`` result instead.
"""

from __future__ import annotations

import functools
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

_CV_ASSURE = Path(__file__).resolve().parent.parent
_M0_SRC = _CV_ASSURE / "m0_ingest_capability" / "src"
for _path in (str(_CV_ASSURE), str(_M0_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from schema import AccessTier, AssetRecord, CapabilityMatrix, ReferenceMode, TaskType  # noqa: E402

F = TypeVar("F", bound=Callable[..., Any])


class DetectorResult(BaseModel):
    """Standard detector payload (JSON-serializable, dict-accessible)."""

    model_config = ConfigDict(extra="allow")

    detector_id: str
    status: str
    reason: Optional[str] = None
    score: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict suitable for the assurance report."""
        return self.model_dump()

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def unavailable_result(detector_id: str, reason: str) -> DetectorResult:
    """Build the graceful-fallback payload for a gated-out detector."""
    return DetectorResult(
        detector_id=detector_id,
        status="UNAVAILABLE",
        reason=reason,
        score=None,
    )


def capability_gated(method: F) -> F:
    """Decorator that skips detector body when the capability matrix is insufficient.

    Applied automatically to ``BaseDetector.run``. Subclasses that expose extra
    entry points can decorate those methods as well.
    """

    @functools.wraps(method)
    def wrapper(
        self: "BaseDetector",
        dataset: List[AssetRecord],
        capability_matrix: CapabilityMatrix,
        *args: Any,
        **kwargs: Any,
    ) -> DetectorResult:
        allowed, reason = self.check_availability(capability_matrix)
        if not allowed:
            return unavailable_result(self.detector_id, reason)
        return method(self, dataset, capability_matrix, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


class BaseDetector(ABC):
    """Abstract integrity detector with capability gating.

    Subclasses implement ``_run``. Callers always use ``run``, which returns
    ``status="UNAVAILABLE"`` instead of raising when the matrix cannot support
    this detector.
    """

    def __init__(
        self,
        detector_id: str,
        required_tier: AccessTier,
        supported_tasks: List[TaskType],
        requires_reference: bool = False,
    ) -> None:
        if not detector_id:
            raise ValueError("detector_id must be a non-empty string")
        self.detector_id = detector_id
        self.required_tier = required_tier
        self.supported_tasks = list(supported_tasks)
        self.requires_reference = requires_reference

    def check_availability(self, capability_matrix: CapabilityMatrix) -> tuple[bool, str]:
        """Return whether ``capability_matrix`` satisfies this detector's contract.

        Reasons are machine-readable, for example::

            UNAVAILABLE_AT_TIER: requires T2_WEIGHTS, found T0_LABELS
            NEURAL_CLEANSE_UNAVAILABLE_FOR_TASK_TYPE
        """
        if capability_matrix.task_type not in self.supported_tasks:
            return False, f"{self._reason_id()}_UNAVAILABLE_FOR_TASK_TYPE"

        if not capability_matrix.access_tier.satisfies(self.required_tier):
            return (
                False,
                (
                    "UNAVAILABLE_AT_TIER: "
                    f"requires {self.required_tier.name}, found {capability_matrix.access_tier.name}"
                ),
            )

        if self.requires_reference and capability_matrix.reference_mode is ReferenceMode.UNREFERENCED:
            return False, f"{self._reason_id()}_UNAVAILABLE_WITHOUT_REFERENCE"

        return True, f"SUPPORTED: detector={self.detector_id}"

    def _reason_id(self) -> str:
        return self.detector_id.upper().replace("-", "_").replace(" ", "_")

    @capability_gated
    def run(
        self,
        dataset: List[AssetRecord],
        capability_matrix: CapabilityMatrix,
        **kwargs: Any,
    ) -> DetectorResult:
        """Gate on capabilities, then execute ``_run`` (never raises on a miss)."""
        return self._run(dataset, capability_matrix, **kwargs)

    @abstractmethod
    def _run(
        self,
        dataset: List[AssetRecord],
        capability_matrix: CapabilityMatrix,
        **kwargs: Any,
    ) -> DetectorResult:
        """Detector-specific logic. Invoked only after ``check_availability`` passes."""
        raise NotImplementedError
