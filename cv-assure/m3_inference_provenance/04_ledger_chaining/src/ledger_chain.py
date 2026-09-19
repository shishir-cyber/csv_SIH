"""
ledger_chain.py
----------------
Hash-Chained Append-Only Ledger and Dual-Custody Checkpoint Manager
for CV-ASSURE M3 Inference Provenance.

Maintains cryptographic chain continuity (prev_receipt_hash) across receipts,
computes canonical receipt digests, and emits signed Merkle checkpoints.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator

from merkle import MerkleTree, sha256_hex

GENESIS_PREV_HASH = "0" * 64


def canonical_json_bytes(data: Dict[str, Any]) -> bytes:
    """Return deterministically formatted, UTF-8 encoded JSON bytes (sorted keys, compact)."""
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_receipt_hash(payload: Dict[str, Any], prev_receipt_hash: str) -> str:
    """
    Compute SHA-256 hash over receipt payload + prev_receipt_hash.
    Ensures deterministic, canonical representation.
    """
    binding_obj = {
        "payload": payload,
        "prev_receipt_hash": prev_receipt_hash,
    }
    return sha256_hex(canonical_json_bytes(binding_obj))


class LedgerWriter:
    """
    Append-only manager for inference_ledger.jsonl and checkpoints.jsonl.
    """

    def __init__(
        self,
        ledger_path: str | Path,
        checkpoint_path: Optional[str | Path] = None,
        checkpoint_interval: int = 10,
        verifier_private_key_pem: Optional[bytes] = None,
    ):
        self.ledger_path = Path(ledger_path)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else self.ledger_path.parent / "checkpoints.jsonl"
        self.checkpoint_interval = checkpoint_interval
        self.verifier_private_key_pem = verifier_private_key_pem

        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.last_receipt_hash: str = GENESIS_PREV_HASH
        self.current_sequence: int = 0
        self.uncheckpointed_hashes: List[str] = []
        self.checkpoint_counter: int = 0

        # Resume state if ledger file already exists
        self._recover_state()

    def _recover_state(self) -> None:
        """Scan existing ledger file to recover sequence count and last receipt hash."""
        if not self.ledger_path.exists():
            return

        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("record_type") == "receipt":
                        self.last_receipt_hash = record["receipt_hash"]
                        self.current_sequence = record["payload"]["sequence_number"]
                        self.uncheckpointed_hashes.append(record["receipt_hash"])
                except (json.JSONDecodeError, KeyError):
                    continue

    def append_receipt(self, payload: Dict[str, Any], signature: str) -> Dict[str, Any]:
        """
        Append a signed receipt payload to the ledger with hash chaining.

        Args:
            payload: Dict containing input_image_sha256, model_weight_digest, etc.
            signature: Hex Ed25519 signature of receipt digest.

        Returns:
            The complete ledger entry record dict.
        """
        self.current_sequence += 1
        payload["sequence_number"] = self.current_sequence

        prev_hash = self.last_receipt_hash
        receipt_hash = compute_receipt_hash(payload, prev_hash)

        record = {
            "record_type": "receipt",
            "payload": payload,
            "prev_receipt_hash": prev_hash,
            "receipt_hash": receipt_hash,
            "signature": signature,
        }

        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        self.last_receipt_hash = receipt_hash
        self.uncheckpointed_hashes.append(receipt_hash)

        # Check if Merkle Checkpoint threshold reached
        if len(self.uncheckpointed_hashes) >= self.checkpoint_interval:
            self.emit_checkpoint()

        return record

    def emit_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Compute Merkle root over pending receipts and emit signed checkpoint record."""
        if not self.uncheckpointed_hashes:
            return None

        self.checkpoint_counter += 1
        tree = MerkleTree(self.uncheckpointed_hashes)
        merkle_root = tree.root

        end_seq = self.current_sequence
        start_seq = end_seq - len(self.uncheckpointed_hashes) + 1

        checkpoint_sig = self._sign_checkpoint_root(merkle_root)

        checkpoint_record = {
            "record_type": "checkpoint",
            "checkpoint_id": f"CHK-{self.checkpoint_counter:04d}",
            "start_sequence": start_seq,
            "end_sequence": end_seq,
            "receipt_count": len(self.uncheckpointed_hashes),
            "merkle_root": merkle_root,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "verifier_signature": checkpoint_sig,
        }

        with open(self.checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(checkpoint_record) + "\n")

        # Clear batch
        self.uncheckpointed_hashes.clear()
        return checkpoint_record

    def _sign_checkpoint_root(self, merkle_root: str) -> str:
        """Sign Merkle root with verifier private key if available, else return fallback signature."""
        if not self.verifier_private_key_pem:
            return f"UNSIGNED_CHECKPOINT_ROOT_{merkle_root[:16]}"

        try:
            from cryptography.hazmat.primitives import serialization
            private_key = serialization.load_pem_private_key(
                self.verifier_private_key_pem, password=None
            )
            sig_bytes = private_key.sign(merkle_root.encode("utf-8"))
            return sig_bytes.hex()
        except Exception:
            return f"SIGNATURE_ERROR_{merkle_root[:16]}"


class LedgerReader:
    """
    Reader and parser for hash-chained ledger files and checkpoint stores.
    """

    def __init__(self, ledger_path: str | Path, checkpoint_path: Optional[str | Path] = None):
        self.ledger_path = Path(ledger_path)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else self.ledger_path.parent / "checkpoints.jsonl"

    def iter_receipts(self) -> Generator[Dict[str, Any], None, None]:
        """Yield each receipt record from the ledger."""
        if not self.ledger_path.exists():
            return

        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("record_type") == "receipt":
                        yield record
                except json.JSONDecodeError:
                    continue

    def iter_checkpoints(self) -> Generator[Dict[str, Any], None, None]:
        """Yield each checkpoint record from the checkpoint file."""
        if not self.checkpoint_path.exists():
            return

        with open(self.checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("record_type") == "checkpoint":
                        yield record
                except json.JSONDecodeError:
                    continue

    def get_all_receipts(self) -> List[Dict[str, Any]]:
        """Return all receipt records as a list."""
        return list(self.iter_receipts())

    def get_all_checkpoints(self) -> List[Dict[str, Any]]:
        """Return all checkpoint records as a list."""
        return list(self.iter_checkpoints())
