"""
test_m3_steps_4_5_6.py
-----------------------
Comprehensive integration test suite for M3 Steps 4, 5, and 6:
  - Step 4: Ledger Chaining & Merkle Checkpoints
  - Step 5: Verification Audit Engine & Attack Detection (Chain Break, Replay, Model Swap, Spot Re-exec)
  - Step 6: JSON/HTML Assurance Reports, C2PA Manifest Export, and Coverage Generator

Run from command line:
    python cv-assure/m3_inference_provenance/scripts/test_m3_steps_4_5_6.py
"""

import sys
import os
import shutil
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

# Resolve paths
ROOT_M3 = Path(__file__).resolve().parents[1]
SRC_4 = ROOT_M3 / "04_ledger_chaining" / "src"
SRC_5 = ROOT_M3 / "05_verification_audit" / "src"
SRC_6 = ROOT_M3 / "06_reporting" / "src"

for p in (SRC_4, SRC_5, SRC_6):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from merkle import MerkleTree, sha256_hex
from ledger_chain import LedgerWriter, LedgerReader, compute_receipt_hash, GENESIS_PREV_HASH
from audit_engine import AuditEngine, Finding
from spot_executor import SpotReExecutor
from report_generator import ReportGenerator
from c2pa_exporter import C2PAExporter
from coverage_generator import CoverageGenerator

# Cryptography helpers
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

TEST_DIR = ROOT_M3 / "test_output"


def setup_keys() -> tuple[bytes, bytes, ed25519.Ed25519PrivateKey]:
    """Generate Ed25519 private/public key PEM bytes for testing."""
    priv_key = ed25519.Ed25519PrivateKey.generate()
    pub_key = priv_key.public_key()

    priv_pem = priv_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = pub_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv_pem, pub_pem, priv_key


def generate_mock_payload(seq: int, model_digest: str, nonce: str = "") -> dict:
    """Generate mock inference receipt payload."""
    if not nonce:
        nonce = f"nonce-{seq:04d}-{sha256_hex(str(seq).encode())[:8]}"

    return {
        "input_image_sha256": sha256_hex(f"image_data_{seq}".encode()),
        "preprocess_config_hash": sha256_hex(b"config_v1"),
        "model_weight_digest": model_digest,
        "model_arch_hash": sha256_hex(b"yolov8n_onnx"),
        "runtime_version": "onnxruntime-1.16.0",
        "output_payload_sha256": sha256_hex(f"output_data_{seq}".encode()),
        "nonce": nonce,
        "sequence_number": seq,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "image_path": f"sample_images/sample_{seq:02d}.jpg",
    }


def test_step4_ledger_chaining(priv_key: ed25519.Ed25519PrivateKey, model_digest: str):
    """Test Step 4: Hash-Chained Ledger Writer and Merkle Checkpoints."""
    print("[1/5] Testing Step 4: Ledger Chaining & Merkle Checkpoints...")

    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    ledger_path = TEST_DIR / "inference_ledger.jsonl"
    chk_path = TEST_DIR / "checkpoints.jsonl"

    writer = LedgerWriter(ledger_path=ledger_path, checkpoint_path=chk_path, checkpoint_interval=5)

    # Write 12 receipts (triggers 2 Merkle checkpoints at interval=5)
    for seq in range(1, 13):
        payload = generate_mock_payload(seq, model_digest)
        prev_hash = writer.last_receipt_hash
        receipt_hash = compute_receipt_hash(payload, prev_hash)

        # Sign receipt_hash
        sig_bytes = priv_key.sign(receipt_hash.encode("utf-8"))
        writer.append_receipt(payload, sig_bytes.hex())

    # Flush remaining uncheckpointed receipts
    writer.emit_checkpoint()

    assert ledger_path.exists(), "Ledger file was not created!"
    assert chk_path.exists(), "Checkpoint file was not created!"

    reader = LedgerReader(ledger_path, chk_path)
    receipts = reader.get_all_receipts()
    checkpoints = reader.get_all_checkpoints()

    assert len(receipts) == 12, f"Expected 12 receipts, got {len(receipts)}"
    assert len(checkpoints) >= 2, f"Expected at least 2 checkpoints, got {len(checkpoints)}"

    # Check Merkle tree calculation
    tree_leaves = [r["receipt_hash"] for r in receipts[:5]]
    tree = MerkleTree(tree_leaves)
    assert tree.root == checkpoints[0]["merkle_root"], "Merkle root mismatch on checkpoint 0!"

    print("      [OK] Hash chaining, prev_receipt_hash linking, and Merkle checkpoints OK")
    return receipts, checkpoints


def test_step5_clean_audit(receipts: list, checkpoints: list, pub_pem: bytes, model_digest: str):
    """Test Step 5: Verification Audit on clean ledger."""
    print("[2/5] Testing Step 5: Verification Audit on Clean Ledger...")

    engine = AuditEngine(
        expected_model_digest=model_digest,
        public_key_pem=pub_pem,
    )
    findings = engine.audit_ledger(receipts, checkpoints)

    # Filter critical/high findings
    critical_findings = [f for f in findings if f.severity in ("critical", "high")]
    assert len(critical_findings) == 0, f"Expected 0 critical findings on clean ledger, got: {critical_findings}"

    print("      [OK] Clean ledger passed all cryptographic audit checks (0 critical findings)")


def test_step5_attack_detection(receipts: list, checkpoints: list, pub_pem: bytes, model_digest: str):
    """Test Step 5: Detection of attack scenarios (chain break, replay, model swap, tampering)."""
    print("[3/5] Testing Step 5: Attack Detection Suite...")

    engine = AuditEngine(
        expected_model_digest=model_digest,
        public_key_pem=pub_pem,
    )

    # Attack Scenario 1: Hash Chain Break (delete 5th receipt)
    tampered_receipts_1 = [r for idx, r in enumerate(receipts) if idx != 4]
    findings_1 = engine.audit_ledger(tampered_receipts_1, checkpoints)
    chain_breaks = [f for f in findings_1 if "CHAIN-BREAK" in f.finding_id or f.detector == "hash_chain_verifier"]
    assert len(chain_breaks) > 0, "Failed to detect chain break / record deletion!"
    assert chain_breaks[0].recommended_disposition == "quarantine"
    print("      [OK] Attack 1 Detected: Chain break / record deletion flagged as QUARANTINE")

    # Attack Scenario 2: Replay Attack (reuse nonce from seq 1 in seq 6)
    import copy
    tampered_receipts_2 = copy.deepcopy(receipts)
    tampered_receipts_2[5]["payload"]["nonce"] = tampered_receipts_2[0]["payload"]["nonce"]
    findings_2 = engine.audit_ledger(tampered_receipts_2, checkpoints)
    replays = [f for f in findings_2 if f.detector == "replay_attack_detector"]
    assert len(replays) > 0, "Failed to detect replay attack!"
    assert replays[0].recommended_disposition == "quarantine"
    print("      [OK] Attack 2 Detected: Replay attack flagged as QUARANTINE")

    # Attack Scenario 3: Model Substitution Attack
    tampered_receipts_3 = copy.deepcopy(receipts)
    tampered_receipts_3[2]["payload"]["model_weight_digest"] = sha256_hex(b"malicious_substituted_model")
    findings_3 = engine.audit_ledger(tampered_receipts_3, checkpoints)
    model_swaps = [f for f in findings_3 if f.detector == "model_integrity_attestation"]
    assert len(model_swaps) > 0, "Failed to detect model substitution attack!"
    assert model_swaps[0].recommended_disposition == "quarantine"
    print("      [OK] Attack 3 Detected: Model substitution attack flagged as QUARANTINE")


def test_step5_spot_execution(receipts: list):
    """Test Step 5: Spot Re-Execution Statistical Honesty Check."""
    print("[4/5] Testing Step 5: Spot Re-Execution Verification...")

    spot_executor = SpotReExecutor(sample_rate=0.5, seed=42)

    # Mock evaluator returning expected output for seq 1, but altered output for others
    def mock_evaluator(image_path: str, model_path: str, config_path: str) -> bytes:
        if "sample_01" in image_path or "sample_1" in image_path:
            return f"output_data_1".encode()
        return b"fabricated_fake_output"

    findings = spot_executor.verify_spot_samples(
        receipts,
        mock_output_evaluator=mock_evaluator,
    )
    fabrication_findings = [f for f in findings if f.detector == "spot_re_executor" and f.severity == "critical"]
    assert len(fabrication_findings) > 0, "Failed to detect host fabrication via spot re-execution!"
    print("      [OK] Spot Re-Execution successfully caught host-side result fabrication")


def test_step6_reporting(receipts: list, checkpoints: list, pub_pem: bytes, model_digest: str):
    """Test Step 6: JSON/HTML Report Generation, C2PA Export, and Coverage Generator."""
    print("[5/5] Testing Step 6: Reporting & Governance Exports...")

    engine = AuditEngine(expected_model_digest=model_digest, public_key_pem=pub_pem)
    clean_findings = engine.audit_ledger(receipts, checkpoints)

    # 1. Report Generator
    report_gen = ReportGenerator(
        findings=clean_findings,
        total_receipts=len(receipts),
        total_checkpoints=len(checkpoints),
    )

    json_path = TEST_DIR / "assurance_report.json"
    html_path = TEST_DIR / "assurance_report.html"

    report_gen.save_json_report(json_path)
    report_gen.save_html_report(html_path)

    assert json_path.exists(), "JSON assurance report missing!"
    assert html_path.exists(), "HTML assurance report missing!"

    # 2. C2PA Exporter
    c2pa_exp = C2PAExporter()
    c2pa_manifest = c2pa_exp.generate_manifest(
        title="SIH CV-ASSURE Provenance Verification",
        asset_sha256=receipts[0]["payload"]["input_image_sha256"],
        model_digest=model_digest,
        signature_hex=receipts[0]["signature"],
    )
    c2pa_path = TEST_DIR / "c2pa_manifest.json"
    c2pa_exp.save_manifest(c2pa_manifest, c2pa_path)
    assert c2pa_path.exists(), "C2PA manifest JSON missing!"

    # 3. Coverage Statement Generator
    cov_gen = CoverageGenerator()
    cov_path = TEST_DIR / "coverage_statement.json"
    cov_gen.save_coverage_statement(cov_path)
    assert cov_path.exists(), "Coverage statement JSON missing!"

    print("      [OK] Report Generator (JSON & HTML), C2PA Exporter, and Coverage Statement OK")


def main():
    print("==================================================================")
    print("      CV-ASSURE v2 · M3 Inference Provenance Smoke Test           ")
    print("      Verifying Steps 4 (Ledger), 5 (Audit), & 6 (Reporting)      ")
    print("==================================================================\n")

    priv_pem, pub_pem, priv_key = setup_keys()
    model_digest = sha256_hex(b"attested_yolov8n_demo_weights")

    receipts, checkpoints = test_step4_ledger_chaining(priv_key, model_digest)
    test_step5_clean_audit(receipts, checkpoints, pub_pem, model_digest)
    test_step5_attack_detection(receipts, checkpoints, pub_pem, model_digest)
    test_step5_spot_execution(receipts)
    test_step6_reporting(receipts, checkpoints, pub_pem, model_digest)

    print("\n==================================================================")
    print(" SUCCESS: All M3 Steps 4, 5, and 6 tests passed flawlessly!")
    print(" Output files written to: m3_inference_provenance/test_output")
    print("==================================================================")


if __name__ == "__main__":
    main()
