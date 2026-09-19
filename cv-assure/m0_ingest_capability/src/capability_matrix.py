"""Access-tier, task-type, and reference-mode capability probing."""

from enum import Enum
from typing import Any, Dict, Optional, Callable
import onnx
import onnxruntime as ort


# =====================================================================
# 1. Access Tier definition
# =====================================================================

class AccessTier(str, Enum):
    T0_BLACK_BOX = "T0"   # Only final label/answer is visible
    T1_GREY_BOX = "T1"    # Label + confidence/logits visible
    T2_WHITE_BOX = "T2"   # Full model file / weights visible

    @property
    def rank(self) -> int:
        """Used to compare tiers: T0 < T1 < T2"""
        ranks = {"T0": 0, "T1": 1, "T2": 2}
        return ranks[self.value]


# =====================================================================
# 2. Capability Matrix - the "board" every detector reads from
# =====================================================================

class CapabilityMatrix:
    """Holds what access level we currently have, plus basic info."""

    def __init__(
        self,
        access_tier: AccessTier,
        task_type: str = "vision_detection",
        reference_mode: str = "bootstrapped",
        source: str = "unknown",          # "local_file" or "remote_api"
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.access_tier = access_tier
        self.task_type = task_type
        self.reference_mode = reference_mode
        self.source = source
        self.metadata = metadata or {}

    def is_tier_supported(self, required_tier: AccessTier) -> bool:
        """True if our current tier is enough for what a detector needs."""
        return self.access_tier.rank >= required_tier.rank

    def to_dict(self) -> Dict[str, Any]:
        """Turns this object into a plain dict, for API responses / UI."""
        return {
            "access_tier": self.access_tier.value,
            "task_type": self.task_type,
            "reference_mode": self.reference_mode,
            "source": self.source,
            "metadata": self.metadata,
        }


# =====================================================================
# 3A. PATH A - We have the actual model FILE
#     (this case is ALWAYS T2, because the file already has the weights)
# =====================================================================

def inspect_local_onnx_file(onnx_path: str) -> CapabilityMatrix:
    """
    Use this when a contributor gives us the actual .onnx file.

    If we have any weight values stored in the file at all, that means
    we already have full access -> T2. This works no matter which tool
    (PyTorch, TensorFlow, etc.) originally exported the ONNX file.
    """
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        has_weights = len(onnx_model.graph.initializer) > 0

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        outputs = session.get_outputs()

        if has_weights:
            detected_tier = AccessTier.T2_WHITE_BOX
        else:
            output_names = [out.name.lower() for out in outputs]
            has_rich_outputs = any(
                term in name for name in output_names
                for term in ["logit", "raw", "embedding", "feature"]
            )
            detected_tier = AccessTier.T1_GREY_BOX if has_rich_outputs else AccessTier.T0_BLACK_BOX

        return CapabilityMatrix(
            access_tier=detected_tier,
            task_type="computer_vision",
            reference_mode="bootstrapped",
            source="local_file",
            metadata={
                "num_inputs": len(session.get_inputs()),
                "output_nodes": [out.name for out in outputs],
                "num_initializers": len(onnx_model.graph.initializer),
            },
        )

    except Exception as err:
        print(f"[WARNING] ONNX file inspection failed, falling back to T0. Reason: {err}")
        return CapabilityMatrix(access_tier=AccessTier.T0_BLACK_BOX, source="local_file")


# =====================================================================
# 3B. PATH B - We only have an API / endpoint (no file at all)
# =====================================================================

def inspect_remote_api(
    call_api_fn: Callable[[Any], Dict[str, Any]],
    sample_input: Any,
) -> CapabilityMatrix:
    """
    Use this when a contributor only gives us an API endpoint, not a file.
    This is the real black-box case.
    """
    try:
        response = call_api_fn(sample_input)
        keys = [k.lower() for k in response.keys()]

        if any(k in keys for k in ["weights", "state_dict", "parameters"]):
            detected_tier = AccessTier.T2_WHITE_BOX
        elif any(k in keys for k in ["logits", "confidence", "probabilities", "scores"]):
            detected_tier = AccessTier.T1_GREY_BOX
        else:
            detected_tier = AccessTier.T0_BLACK_BOX

        return CapabilityMatrix(
            access_tier=detected_tier,
            task_type="computer_vision",
            reference_mode="bootstrapped",
            source="remote_api",
            metadata={"raw_response_keys": list(response.keys())},
        )

    except Exception as err:
        print(f"[WARNING] Remote API probe failed, falling back to T0. Reason: {err}")
        return CapabilityMatrix(access_tier=AccessTier.T0_BLACK_BOX, source="remote_api")


# =====================================================================
# 4. Base Detector class - every detector inherits from this
# =====================================================================

class BaseDetector:
    """Parent class for every integrity detector (poison, watermark, etc.)."""

    def __init__(self, name: str, min_required_tier: AccessTier):
        self.name = name
        self.min_required_tier = min_required_tier

    def run(self, matrix: CapabilityMatrix, input_data: Any) -> Dict[str, Any]:
        """Checks the capability matrix BEFORE running the real detection logic."""
        if not matrix.is_tier_supported(self.min_required_tier):
            return {
                "detector": self.name,
                "status": "UNAVAILABLE_AT_TIER",
                "message": (
                    f"Detector '{self.name}' needs tier {self.min_required_tier.value}, "
                    f"but current access is only {matrix.access_tier.value}."
                ),
                "confidence": None,
            }
        return self._execute_detection(input_data)

    def _execute_detection(self, input_data: Any) -> Dict[str, Any]:
        raise NotImplementedError("Each detector must implement its own _execute_detection method.")


# =====================================================================
# 5. Example detectors (other teammates will replace these with real logic)
# =====================================================================

class OutputSignatureVerifier(BaseDetector):
    """Only needs T0 - works even with just the final label."""

    def __init__(self):
        super().__init__(name="Output Signature Verifier", min_required_tier=AccessTier.T0_BLACK_BOX)

    def _execute_detection(self, input_data: Any) -> Dict[str, Any]:
        return {
            "detector": self.name,
            "status": "SUCCESS",
            "verified": True,
            "confidence": 0.99,
        }


class FragileWatermarkDetector(BaseDetector):
    """Needs T2 - only runs if we have full model weights."""

    def __init__(self):
        super().__init__(name="Fragile Watermark Detector", min_required_tier=AccessTier.T2_WHITE_BOX)

    def _execute_detection(self, input_data: Any) -> Dict[str, Any]:
        return {
            "detector": self.name,
            "status": "SUCCESS",
            "tamper_detected": False,
            "confidence": 0.95,
        }