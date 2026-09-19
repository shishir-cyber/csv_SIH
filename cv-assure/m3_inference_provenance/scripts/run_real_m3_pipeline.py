"""
run_real_m3_pipeline.py
------------------------
End-to-End Real Pipeline Execution for M3 (Inference Provenance).

Executes REAL ONNX inference on REAL sample images (01_input_side),
writes real hash-chained receipts and Merkle checkpoints (04_ledger_chaining),
runs real cryptographic verification & ONNX spot auditing (05_verification_audit),
and generates production JSON/HTML assurance reports (06_reporting).

Usage:
    py scripts/run_real_m3_pipeline.py
"""

import sys
import json
import hashlib
from pathlib import Path

# Resolve M3 source directories
ROOT_M3 = Path(__file__).resolve().parents[1]
SRC_01 = ROOT_M3 / "01_input_side" / "src"
SRC_04 = ROOT_M3 / "04_ledger_chaining" / "src"
SRC_05 = ROOT_M3 / "05_verification_audit" / "src"
SRC_06 = ROOT_M3 / "06_reporting" / "src"

for p in (SRC_01, SRC_04, SRC_05, SRC_06):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Imports from 01, 04, 05, 06
from run_inference import run_batch_inference, run_single_inference
from ledger_chain import LedgerWriter, LedgerReader, compute_receipt_hash
from audit_engine import AuditEngine
from spot_executor import SpotReExecutor
from report_generator import ReportGenerator
from c2pa_exporter import C2PAExporter
from coverage_generator import CoverageGenerator

from cryptography.hazmat.primitives import serialization

CONFIG_PATH = ROOT_M3 / "00_setup" / "config" / "preprocess_config.json"
MODEL_PATH = ROOT_M3 / "01_input_side" / "data" / "model" / "demo_model.onnx"
IMAGE_DIR = ROOT_M3 / "01_input_side" / "data" / "sample_images"
PRIV_KEY_PATH = ROOT_M3 / "00_setup" / "keys" / "private_key.pem"
PUB_KEY_PATH = ROOT_M3 / "00_setup" / "keys" / "public_key.pem"

OUTPUT_DIR = ROOT_M3 / "real_output"


def main():
    print("==================================================================")
    print("      CV-ASSURE v2 · REAL M3 INFERENCE PROVENANCE PIPELINE        ")
    print("==================================================================\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path = OUTPUT_DIR / "real_inference_ledger.jsonl"
    chk_path = OUTPUT_DIR / "real_checkpoints.jsonl"

    if ledger_path.exists():
        ledger_path.unlink()
    if chk_path.exists():
        chk_path.unlink()

    # Load cryptographic key pair
    priv_pem = PRIV_KEY_PATH.read_bytes()
    pub_pem = PUB_KEY_PATH.read_bytes()
    private_key = serialization.load_pem_private_key(priv_pem, password=None)

    # Calculate real model digest
    model_bytes = MODEL_PATH.read_bytes()
    model_digest = hashlib.sha256(model_bytes).hexdigest()
    print(f"[STAGE 0] Loaded attested ONNX model: {MODEL_PATH.name} (digest: {model_digest[:16]}...)")
    print(f"[STAGE 0] Loaded Ed25519 signing keys from 00_setup/keys")

    # STAGE 1: Real ONNX Inference
    image_paths = sorted(
        str(p) for p in IMAGE_DIR.glob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print(f"\n[STAGE 1] Running real ONNX inference on {len(image_paths)} image(s)...")

    inference_results = list(run_batch_inference(image_paths, str(MODEL_PATH), str(CONFIG_PATH)))

    # STAGE 4: Real Ledger Chaining & Merkle Checkpoints
    print("\n[STAGE 4] Building real hash-chained receipts and Merkle checkpoints...")
    writer = LedgerWriter(ledger_path=ledger_path, checkpoint_path=chk_path, checkpoint_interval=2)

    for idx, res in enumerate(inference_results, start=1):
        input_sha256 = hashlib.sha256(res["input_bytes"]).hexdigest()
        output_sha256 = hashlib.sha256(res["output_bytes"]).hexdigest()
        config_hash = hashlib.sha256(json.dumps(res["preprocess_config"], sort_keys=True).encode()).hexdigest()

        payload = {
            "input_image_sha256": input_sha256,
            "preprocess_config_hash": config_hash,
            "model_weight_digest": model_digest,
            "model_arch_hash": hashlib.sha256(b"resnet18_onnx").hexdigest(),
            "runtime_version": "onnxruntime-1.30.0",
            "output_payload_sha256": output_sha256,
            "nonce": f"nonce-real-{idx:04d}-{input_sha256[:8]}",
            "sequence_number": idx,
            "image_path": res["image_path"],
        }

        # Sign receipt payload using real Ed25519 key
        prev_hash = writer.last_receipt_hash
        rec_hash = compute_receipt_hash(payload, prev_hash)
        sig_bytes = private_key.sign(rec_hash.encode("utf-8"))

        writer.append_receipt(payload, sig_bytes.hex())
        print(f"  Receipt {idx}: image={Path(res['image_path']).name} | output_hash={output_sha256[:12]}...")

    writer.emit_checkpoint()

    # STAGE 5: Real Cryptographic & Spot Audit
    print("\n[STAGE 5] Running real cryptographic audit and local ONNX spot re-execution...")
    reader = LedgerReader(ledger_path, chk_path)
    receipts = reader.get_all_receipts()
    checkpoints = reader.get_all_checkpoints()

    audit_engine = AuditEngine(
        expected_model_digest=model_digest,
        public_key_pem=pub_pem,
    )
    findings = audit_engine.audit_ledger(receipts, checkpoints)

    # Run real ONNX spot re-execution verification
    spot_executor = SpotReExecutor(sample_rate=0.4, seed=42)
    spot_findings = spot_executor.verify_spot_samples(
        receipts,
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
    )
    findings.extend(spot_findings)

    critical_count = sum(1 for f in findings if f.severity in ("critical", "high"))
    print(f"  Audited {len(receipts)} receipts & {len(checkpoints)} checkpoints. Findings: {len(findings)} ({critical_count} critical)")

    # STAGE 6: Real Governance & Report Generation
    print("\n[STAGE 6] Generating production JSON/HTML assurance reports and C2PA manifest...")
    report_gen = ReportGenerator(findings, total_receipts=len(receipts), total_checkpoints=len(checkpoints))

    json_report_path = OUTPUT_DIR / "real_assurance_report.json"
    html_report_path = OUTPUT_DIR / "real_assurance_report.html"
    c2pa_path = OUTPUT_DIR / "real_c2pa_manifest.json"
    cov_path = OUTPUT_DIR / "real_coverage_statement.json"

    report_gen.save_json_report(json_report_path)
    report_gen.save_html_report(html_report_path)

    c2pa_exp = C2PAExporter()
    manifest = c2pa_exp.generate_manifest(
        title="CV-ASSURE v2 Real Inference Audit",
        asset_sha256=receipts[0]["payload"]["input_image_sha256"],
        model_digest=model_digest,
        signature_hex=receipts[0]["signature"],
    )
    c2pa_exp.save_manifest(manifest, c2pa_path)

    cov_gen = CoverageGenerator()
    cov_gen.save_coverage_statement(cov_path)

    print("\n==================================================================")
    print(f" SUCCESS: REAL M3 PIPELINE EXECUTED WITH DISPOSITION: {report_gen.determine_overall_disposition()}")
    print(" Output directory: m3_inference_provenance/real_output")
    print("==================================================================")


if __name__ == "__main__":
    main()
