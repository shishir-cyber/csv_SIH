"""Access-tier, task-type, and reference-mode capability probing."""

from typing import Any, Callable, Dict

from base_detector import BaseDetector, DetectorResult
from schema import AccessTier, CapabilityMatrix, ReferenceMode, TaskType

try:
    from probe import probe_capabilities
except ImportError:  # pragma: no cover - path layout fallback
    probe_capabilities = None  # type: ignore[assignment]

# Re-export v2 schema types so existing imports keep working.
__all__ = [
    "AccessTier",
    "CapabilityMatrix",
    "ReferenceMode",
    "TaskType",
    "inspect_local_onnx_file",
    "inspect_remote_api",
    "probe_capabilities",
    "BaseDetector",
    "DetectorResult",
    "OutputSignatureVerifier",
    "FragileWatermarkDetector",
]


# =====================================================================
# 3A. PATH A - We have the actual model FILE
#     (this case is ALWAYS T2, because the file already has the weights)
# =====================================================================

def inspect_local_onnx_file(onnx_path: str) -> CapabilityMatrix:
    """
    Use this when a contributor gives us the actual .onnx file.

    If we have any weight values stored in the file at all, that means
    we already have full access -> T2. This works no matter which tool
    (PyTorch, TensorFlow, etc.)     originally exported the ONNX file.
    """
    try:
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        has_weights = len(onnx_model.graph.initializer) > 0

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        outputs = session.get_outputs()

        if has_weights:
            detected_tier = AccessTier.T2_WEIGHTS
        else:
            output_names = [out.name.lower() for out in outputs]
            has_rich_outputs = any(
                term in name for name in output_names
                for term in ["logit", "raw", "embedding", "feature"]
            )
            detected_tier = AccessTier.T1_LOGITS if has_rich_outputs else AccessTier.T0_LABELS

        return CapabilityMatrix(
            access_tier=detected_tier,
            task_type=TaskType.DETECTION,
            reference_mode=ReferenceMode.BOOTSTRAPPED,
            source="local_file",
            metadata={
                "num_inputs": len(session.get_inputs()),
                "output_nodes": [out.name for out in outputs],
                "num_initializers": len(onnx_model.graph.initializer),
            },
        )

    except Exception as err:
        print(f"[WARNING] ONNX file inspection failed, falling back to T0. Reason: {err}")
        return CapabilityMatrix(
            access_tier=AccessTier.T0_LABELS,
            task_type=TaskType.DETECTION,
            reference_mode=ReferenceMode.UNREFERENCED,
            source="local_file",
        )


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
            detected_tier = AccessTier.T2_WEIGHTS
        elif any(k in keys for k in ["logits", "confidence", "probabilities", "scores"]):
            detected_tier = AccessTier.T1_LOGITS
        else:
            detected_tier = AccessTier.T0_LABELS

        return CapabilityMatrix(
            access_tier=detected_tier,
            task_type=TaskType.DETECTION,
            reference_mode=ReferenceMode.BOOTSTRAPPED,
            source="remote_api",
            metadata={"raw_response_keys": list(response.keys())},
        )

    except Exception as err:
        print(f"[WARNING] Remote API probe failed, falling back to T0. Reason: {err}")
        return CapabilityMatrix(
            access_tier=AccessTier.T0_LABELS,
            task_type=TaskType.DETECTION,
            reference_mode=ReferenceMode.UNREFERENCED,
            source="remote_api",
        )


# =====================================================================
# Example detectors (other teammates will replace these with real logic)
# =====================================================================

class OutputSignatureVerifier(BaseDetector):
    """Only needs T0 - works even with just the final label."""

    def __init__(self) -> None:
        super().__init__(
            detector_id="output_signature_verifier",
            required_tier=AccessTier.T0_LABELS,
            supported_tasks=list(TaskType),
            requires_reference=False,
        )

    def _run(self, dataset, capability_matrix, **kwargs) -> DetectorResult:
        return DetectorResult(
            detector_id=self.detector_id,
            status="SUCCESS",
            score=0.99,
            details={"verified": True},
        )


class FragileWatermarkDetector(BaseDetector):
    """Needs T2 - only runs if we have full model weights."""

    def __init__(self) -> None:
        super().__init__(
            detector_id="fragile_watermark_detector",
            required_tier=AccessTier.T2_WEIGHTS,
            supported_tasks=list(TaskType),
            requires_reference=False,
        )

    def _run(self, dataset, capability_matrix, **kwargs) -> DetectorResult:
        return DetectorResult(
            detector_id=self.detector_id,
            status="SUCCESS",
            score=0.95,
            details={"tamper_detected": False},
        )