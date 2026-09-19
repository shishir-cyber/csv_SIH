"""
test_m3_pipeline.py
-------------------
Master Test Suite for M3 (Inference Provenance - Clause 2.2.3 & 2.2.5).
Executes all 6 stages of M3 in a single run:

  Stage 1: Input Side - Image Preprocessing & ONNX Model Execution
  Stage 2: Receipt Building - Clause 2.2.3 Canonical RFC 8785 Schema Assembly
  Stage 3: Cryptographic Signing - Ed25519 Signature Generation & Verification
  Stage 4: Ledger Chaining - Append-Only JSONL Ledger & Merkle Checkpoints
  Stage 5: Verification Audit - 6-Point Audit Engine & ONNX Spot Re-Execution
  Stage 6: Governance Reporting - JSON Report, HTML Dashboard, C2PA & Coverage

Usage:
    py scripts/test_m3_pipeline.py
"""

import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

# Add module source paths
ROOT_M3 = Path(__file__).resolve().parents[1]
SRC_01 = ROOT_M3 / "01_input_side" / "src"
SRC_02 = ROOT_M3 / "02_receipt_building" / "src"
SRC_03 = ROOT_M3 / "03_crypto_signing" / "src"
SRC_04 = ROOT_M3 / "04_ledger_chaining" / "src"
SRC_05 = ROOT_M3 / "05_verification_audit" / "src"
SRC_06 = ROOT_M3 / "06_reporting" / "src"

for p in (SRC_01, SRC_02, SRC_03, SRC_04, SRC_05, SRC_06):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from preprocess import load_preprocess_config
from run_inference import run_batch_inference
from receipt_builder import ReceiptBuilder
from signer import ReceiptSigner
from verifier import ReceiptVerifier
from ledger_chain import LedgerWriter, LedgerReader, compute_receipt_hash
from audit_engine import AuditEngine
from spot_executor import SpotReExecutor
from report_generator import ReportGenerator
from c2pa_exporter import C2PAExporter
from coverage_generator import CoverageGenerator

CONFIG_PATH = ROOT_M3 / "00_setup" / "config" / "preprocess_config.json"
MODEL_PATH = ROOT_M3 / "01_input_side" / "data" / "model" / "demo_model.onnx"
IMAGE_DIR = ROOT_M3 / "01_input_side" / "data" / "sample_images"
PRIV_KEY_PATH = ROOT_M3 / "00_setup" / "keys" / "private_key.pem"
PUB_KEY_PATH = ROOT_M3 / "00_setup" / "keys" / "public_key.pem"

OUTPUT_DIR = ROOT_M3 / "m3_test_output"


def main():
    print("==================================================================")
    print("      CV-ASSURE v2 · M3 INFERENCE PROVENANCE MASTER TEST         ")
    print("      Executing All 6 Stages in Sequence                         ")
    print("==================================================================\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path = OUTPUT_DIR / "inference_ledger.jsonl"
    chk_path = OUTPUT_DIR / "checkpoints.jsonl"

    if ledger_path.exists():
        ledger_path.unlink()
    if chk_path.exists():
        chk_path.unlink()

    # STAGE 1: Input Side (Preprocessing & ONNX Inference)
    print("[1/6] STAGE 1: Input Side (Image Preprocessing & ONNX Model Loading)...")
    config = load_preprocess_config(str(CONFIG_PATH))
    assert CONFIG_PATH.exists(), "preprocess_config.json missing!"
    assert MODEL_PATH.exists(), "demo_model.onnx missing!"

    image_paths = sorted(
        str(p) for p in IMAGE_DIR.glob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    inference_results = list(run_batch_inference(image_paths, str(MODEL_PATH), str(CONFIG_PATH)))
    assert len(inference_results) > 0, "No inference results produced!"
    print(f"      [OK] Preprocessed & executed ResNet-18 ONNX inference on {len(inference_results)} sample image(s)")

    # STAGE 2: Receipt Building (Schema & Canonical Serialization)
    print("\n[2/6] STAGE 2: Receipt Building (Clause 2.2.3 Canonical Schema)...")
    builder = ReceiptBuilder(runtime_version="onnxruntime-1.30.0")
    built_receipts = []
    prev_hash = "0" * 64

    for idx, res in enumerate(inference_results, start=1):
        payload = builder.build_receipt(
            input_bytes=res["input_bytes"],
            output_bytes=res["output_bytes"],
            model_bytes=res["model_bytes"],
            preprocess_config=res["preprocess_config"],
            sequence_number=idx,
            prev_receipt_hash=prev_hash,
            image_path=Path(res["image_path"]).name,
        )
        rec_hash = builder.get_receipt_hash(payload, prev_hash)
        built_receipts.append((payload, rec_hash))
        prev_hash = rec_hash

    assert len(built_receipts) == len(inference_results), "Receipt count mismatch!"
    print(f"      [OK] Built & validated {len(built_receipts)} canonical RFC 8785 receipts")

    # STAGE 3: Cryptographic Signing & Verification
    print("\n[3/6] STAGE 3: Cryptographic Signing (Ed25519 Private Key Signing)...")
    signer = ReceiptSigner(PRIV_KEY_PATH)
    verifier = ReceiptVerifier(PUB_KEY_PATH)
    signed_records = []

    for payload, rec_hash in built_receipts:
        sig_hex = signer.sign_hash(rec_hash)
        assert verifier.verify_signature(rec_hash, sig_hex), f"Ed25519 signature verification failed for sequence {payload['sequence_number']}!"
        signed_record = signer.sign_receipt(rec_hash, payload)
        signed_records.append(signed_record)

    print(f"      [OK] Generated & verified Ed25519 signatures for all {len(signed_records)} receipts")

    # STAGE 4: Ledger Chaining & Merkle Checkpoints
    print("\n[4/6] STAGE 4: Ledger Chaining (Hash-Chained JSONL & Merkle Roots)...")
    ledger_writer = LedgerWriter(ledger_path=ledger_path, checkpoint_path=chk_path, checkpoint_interval=2)

    for rec in signed_records:
        ledger_writer.append_receipt(rec["payload"], rec["signature"])

    ledger_writer.emit_checkpoint()

    reader = LedgerReader(ledger_path, chk_path)
    receipts = reader.get_all_receipts()
    checkpoints = reader.get_all_checkpoints()

    assert len(receipts) == len(signed_records), "Ledger receipt count mismatch!"
    assert len(checkpoints) >= 1, "Merkle checkpoint count mismatch!"
    print(f"      [OK] Appended {len(receipts)} receipts & {len(checkpoints)} Merkle checkpoints to ledger")

    # STAGE 5: Verification Audit Engine & Local ONNX Spot Re-Execution
    print("\n[5/6] STAGE 5: Verification Audit Engine & Local ONNX Spot Re-Execution...")
    model_digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    audit_engine = AuditEngine(
        expected_model_digest=model_digest,
        public_key_pem=PUB_KEY_PATH.read_bytes(),
    )
    findings = audit_engine.audit_ledger(receipts, checkpoints)

    spot_executor = SpotReExecutor(sample_rate=0.4, seed=42)
    spot_findings = spot_executor.verify_spot_samples(
        receipts,
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
    )
    findings.extend(spot_findings)

    critical_findings = [f for f in findings if f.severity in ("critical", "high")]
    assert len(critical_findings) == 0, f"Unexpected critical findings on clean pipeline: {critical_findings}"
    print(f"      [OK] Completed 6-point cryptographic audit & ONNX spot check (0 critical findings)")

    # STAGE 6: Governance & Report Generation
    print("\n[6/6] STAGE 6: Governance Reporting (JSON, HTML Dashboard & C2PA)...")
    report_gen = ReportGenerator(findings, total_receipts=len(receipts), total_checkpoints=len(checkpoints))

    json_report_path = OUTPUT_DIR / "assurance_report.json"
    html_report_path = OUTPUT_DIR / "assurance_report.html"
    c2pa_path = OUTPUT_DIR / "c2pa_manifest.json"
    cov_path = OUTPUT_DIR / "coverage_statement.json"

    report_gen.save_json_report(json_report_path)
    report_gen.save_html_report(html_report_path)

    c2pa_exp = C2PAExporter()
    manifest = c2pa_exp.generate_manifest(
        title="CV-ASSURE M3 Master Provenance Audit",
        asset_sha256=receipts[0]["payload"]["input_image_sha256"],
        model_digest=model_digest,
        signature_hex=receipts[0]["signature"],
    )
    c2pa_exp.save_manifest(manifest, c2pa_path)

    cov_gen = CoverageGenerator()
    cov_gen.save_coverage_statement(cov_path)

    assert json_report_path.exists(), "assurance_report.json missing!"
    assert html_report_path.exists(), "assurance_report.html missing!"
    assert c2pa_path.exists(), "c2pa_manifest.json missing!"
    assert cov_path.exists(), "coverage_statement.json missing!"

    disposition = report_gen.determine_overall_disposition()

    print("\n==================================================================")
    print(f" SUCCESS: ALL 6 STAGES PASSED FLAWLESSLY | DISPOSITION: {disposition}")
    print(" Output directory: m3_inference_provenance/m3_test_output")
    print("==================================================================")


if __name__ == "__main__":
    main()
