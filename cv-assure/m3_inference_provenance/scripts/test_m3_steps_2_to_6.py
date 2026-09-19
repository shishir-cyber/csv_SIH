"""
test_m3_steps_2_to_6.py
-----------------------
Complete End-to-End Test for M3 Inference Provenance:
  - Step 1: Input Side Inference Execution (01_input_side)
  - Step 2: Receipt Building & Canonical Serialization (02_receipt_building)
  - Step 3: Ed25519 Cryptographic Signing & Verification (03_crypto_signing)
  - Step 4: Hash-Chained Ledger & Merkle Checkpoints (04_ledger_chaining)
  - Step 5: Audit Engine & Spot Re-Execution Audit (05_verification_audit)
  - Step 6: JSON/HTML Assurance Reports, C2PA & Coverage (06_reporting)

Usage:
    py scripts/test_m3_steps_2_to_6.py
"""

import sys
import hashlib
from pathlib import Path

# Add all M3 stage source directories to sys.path
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

from run_inference import run_batch_inference
from receipt_builder import ReceiptBuilder
from signer import ReceiptSigner
from verifier import ReceiptVerifier
from ledger_chain import LedgerWriter, LedgerReader
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

OUTPUT_DIR = ROOT_M3 / "m3_full_pipeline_output"


def main():
    print("==================================================================")
    print("      CV-ASSURE v2 · FULL M3 PIPELINE (STEPS 1 -> 6)             ")
    print("==================================================================\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path = OUTPUT_DIR / "inference_ledger.jsonl"
    chk_path = OUTPUT_DIR / "checkpoints.jsonl"

    if ledger_path.exists():
        ledger_path.unlink()
    if chk_path.exists():
        chk_path.unlink()

    # Step 0: Setup keys & model digests
    signer = ReceiptSigner(PRIV_KEY_PATH)
    verifier = ReceiptVerifier(PUB_KEY_PATH)

    model_bytes = MODEL_PATH.read_bytes()
    model_digest = hashlib.sha256(model_bytes).hexdigest()
    print(f"[STAGE 0/6] Initialized keys & loaded ONNX model digest: {model_digest[:16]}...")

    # Step 1: Real Inference Execution
    image_paths = sorted(
        str(p) for p in IMAGE_DIR.glob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print(f"\n[STAGE 1/6] Running ONNX inference on {len(image_paths)} sample image(s)...")
    inference_results = list(run_batch_inference(image_paths, str(MODEL_PATH), str(CONFIG_PATH)))
    assert len(inference_results) > 0, "No inference results returned!"
    print(f"      [OK] Completed inference passes on {len(inference_results)} image(s)")

    # Step 2: Receipt Building & Canonical Serialization
    print("\n[STAGE 2/6] Building Clause 2.2.3 canonical receipts...")
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

        # Step 3: Cryptographic Signing
        sig_hex = signer.sign_hash(rec_hash)
        assert verifier.verify_signature(rec_hash, sig_hex), f"Signature verification failed for receipt {idx}!"

        signed_record = signer.sign_receipt(rec_hash, payload)
        built_receipts.append((payload, sig_hex, rec_hash))
        prev_hash = rec_hash

        print(f"      [OK] Receipt {idx}: seq={payload['sequence_number']} | digest={rec_hash[:12]}... | sig_verified=True")

    # Step 4: Ledger Chaining & Merkle Checkpoints
    print("\n[STAGE 4/6] Appending receipts to hash-chained ledger and writing Merkle checkpoints...")
    ledger_writer = LedgerWriter(ledger_path=ledger_path, checkpoint_path=chk_path, checkpoint_interval=2)

    for payload, sig_hex, _ in built_receipts:
        ledger_writer.append_receipt(payload, sig_hex)

    ledger_writer.emit_checkpoint()

    reader = LedgerReader(ledger_path, chk_path)
    stored_receipts = reader.get_all_receipts()
    stored_checkpoints = reader.get_all_checkpoints()
    print(f"      [OK] Wrote {len(stored_receipts)} receipts & {len(stored_checkpoints)} Merkle checkpoints to ledger")

    # Step 5: Verification Audit Engine & Spot Execution
    print("\n[STAGE 5/6] Running 6-Point Cryptographic Audit Engine & Local ONNX Spot Re-Execution...")
    audit_engine = AuditEngine(
        expected_model_digest=model_digest,
        public_key_pem=PUB_KEY_PATH.read_bytes(),
    )
    findings = audit_engine.audit_ledger(stored_receipts, stored_checkpoints)

    spot_executor = SpotReExecutor(sample_rate=0.4, seed=42)
    spot_findings = spot_executor.verify_spot_samples(
        stored_receipts,
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
    )
    findings.extend(spot_findings)

    critical_count = sum(1 for f in findings if f.severity in ("critical", "high"))
    print(f"      [OK] Audit finished: {len(findings)} total finding(s) ({critical_count} critical)")

    # Step 6: Reporting & Governance Exports
    print("\n[STAGE 6/6] Compiling Clause 2.2.5 JSON report, HTML Audit Dashboard & C2PA manifest...")
    report_gen = ReportGenerator(findings, total_receipts=len(stored_receipts), total_checkpoints=len(stored_checkpoints))
    json_path = OUTPUT_DIR / "assurance_report.json"
    html_path = OUTPUT_DIR / "assurance_report.html"
    c2pa_path = OUTPUT_DIR / "c2pa_manifest.json"
    cov_path = OUTPUT_DIR / "coverage_statement.json"

    report_gen.save_json_report(json_path)
    report_gen.save_html_report(html_path)

    c2pa_exp = C2PAExporter()
    manifest = c2pa_exp.generate_manifest(
        title="CV-ASSURE v2 Full M3 Provenance Audit",
        asset_sha256=stored_receipts[0]["payload"]["input_image_sha256"],
        model_digest=model_digest,
        signature_hex=stored_receipts[0]["signature"],
    )
    c2pa_exp.save_manifest(manifest, c2pa_path)

    cov_gen = CoverageGenerator()
    cov_gen.save_coverage_statement(cov_path)

    disposition = report_gen.determine_overall_disposition()

    print("\n==================================================================")
    print(f" SUCCESS: ALL STAGES 1 TO 6 COMPLETED WITH DISPOSITION: {disposition}")
    print(" Output directory: m3_inference_provenance/m3_full_pipeline_output")
    print("==================================================================")


if __name__ == "__main__":
    main()
