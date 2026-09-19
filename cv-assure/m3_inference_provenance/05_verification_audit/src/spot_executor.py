"""
spot_executor.py
----------------
Spot Re-Execution Verification Module for CV-ASSURE M3 Inference Provenance.

Re-runs local inference on a randomized sample (e.g. 1%) of receipts using the
attested model to compare generated output digests against recorded receipts.
Establishes a statistical honesty bound to detect host-side fabricated results.
"""

import hashlib
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

from audit_engine import Finding


class SpotReExecutor:
    """
    Local ONNX Spot Re-Executor for Statistical Honesty Audit.
    """

    def __init__(
        self,
        sample_rate: float = 0.05,
        min_samples: int = 1,
        seed: Optional[int] = 42,
    ):
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.min_samples = min_samples
        self.seed = seed

    def select_sample_receipts(self, receipts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select randomized subset of receipts for spot re-execution."""
        if not receipts:
            return []

        if self.seed is not None:
            random.seed(self.seed)

        sample_count = max(self.min_samples, int(len(receipts) * self.sample_rate))
        sample_count = min(sample_count, len(receipts))

        return random.sample(receipts, sample_count)

    def verify_spot_samples(
        self,
        receipts: List[Dict[str, Any]],
        model_path: Optional[str | Path] = None,
        config_path: Optional[str | Path] = None,
        mock_output_evaluator: Optional[Any] = None,
    ) -> List[Finding]:
        """
        Spot re-execute inference on sample receipts and verify output hash matching.

        Args:
            receipts: Complete list of receipt dicts from ledger.
            model_path: Path to attested ONNX model file.
            config_path: Path to preprocess config file.
            mock_output_evaluator: Callable(image_path, model_path, config_path) -> bytes for testing.

        Returns:
            List of Finding objects for any spot check failures.
        """
        findings: List[Finding] = []
        samples = self.select_sample_receipts(receipts)

        if not samples:
            return findings

        for rec in samples:
            payload = rec.get("payload", {})
            seq = payload.get("sequence_number", 0)
            expected_output_sha256 = payload.get("output_payload_sha256", "")
            image_path = payload.get("image_path", "")
            asset_ref = payload.get("input_image_sha256", f"seq_{seq}")

            # Calculate actual output bytes (via mock or real ONNX runtime if available)
            actual_output_sha256 = None

            if mock_output_evaluator is not None:
                try:
                    out_bytes = mock_output_evaluator(image_path, str(model_path), str(config_path))
                    actual_output_sha256 = hashlib.sha256(out_bytes).hexdigest()
                except Exception as e:
                    findings.append(
                        Finding(
                            finding_id=f"FINDING-M3-SPOT-EXEC-ERROR-SEQ-{seq}",
                            detector="spot_re_executor",
                            asset_ref=asset_ref,
                            score=0.5,
                            confidence="medium",
                            severity="warning",
                            reason_human_readable=f"Spot re-execution error at sequence {seq}: {e}",
                            recommended_disposition="review",
                        )
                    )
                    continue

            elif model_path and Path(model_path).exists() and image_path and Path(image_path).exists():
                try:
                    # Optional import of 01_input_side runner if environment permits
                    import sys
                    root_m3 = Path(__file__).resolve().parents[2]
                    input_src = root_m3 / "01_input_side" / "src"
                    if str(input_src) not in sys.path:
                        sys.path.insert(0, str(input_src))

                    from run_inference import run_single_inference
                    result = run_single_inference(image_path, str(model_path), str(config_path))
                    actual_output_sha256 = hashlib.sha256(result["output_bytes"]).hexdigest()
                except Exception as e:
                    findings.append(
                        Finding(
                            finding_id=f"FINDING-M3-SPOT-EXEC-FAILED-SEQ-{seq}",
                            detector="spot_re_executor",
                            asset_ref=asset_ref,
                            score=0.5,
                            confidence="medium",
                            severity="warning",
                            reason_human_readable=f"Spot re-execution failed to run at sequence {seq}: {e}",
                            recommended_disposition="review",
                        )
                    )
                    continue
            else:
                # If model or image not accessible on disk, record graceful fallback
                findings.append(
                    Finding(
                        finding_id=f"FINDING-M3-SPOT-UNAVAILABLE-SEQ-{seq}",
                        detector="spot_re_executor",
                        asset_ref=asset_ref,
                        score=0.0,
                        confidence="medium",
                        severity="info",
                        reason_human_readable=(
                            f"Spot re-execution skipped for sequence {seq}: model file or input image "
                            "not available locally."
                        ),
                        recommended_disposition="unavailable",
                    )
                )
                continue

            # Verify hash match
            if actual_output_sha256 and actual_output_sha256.lower() != expected_output_sha256.lower():
                findings.append(
                    Finding(
                        finding_id=f"FINDING-M3-SPOT-FABRICATION-SEQ-{seq}",
                        detector="spot_re_executor",
                        asset_ref=asset_ref,
                        score=1.0,
                        confidence="high",
                        severity="critical",
                        reason_human_readable=(
                            f"Host fabrication detected during spot re-execution at sequence {seq}! "
                            f"Auditor re-executed output hash '{actual_output_sha256[:12]}...' "
                            f"does not match host receipt output hash '{expected_output_sha256[:12]}...'."
                        ),
                        evidence_artifacts=[f"spot_reexec_seq_{seq}.log"],
                        recommended_disposition="quarantine",
                    )
                )

        return findings
