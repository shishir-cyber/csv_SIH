"""
merkle.py
---------
Binary Merkle Tree implementation for CV-ASSURE M3 Inference Provenance.
Computes cryptographic Merkle roots over receipt hashes and provides
inclusion proof generation and verification.
"""

import hashlib
from typing import List, Dict, Optional


def sha256_hex(data: bytes) -> str:
    """Return SHA-256 hex digest for given bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_pair(left_hex: str, right_hex: str) -> str:
    """Hash two hex-encoded child hashes together (left + right)."""
    combined = bytes.fromhex(left_hex) + bytes.fromhex(right_hex)
    return sha256_hex(combined)


class MerkleTree:
    """
    Binary Merkle Tree constructed over leaf hashes (hex strings).
    If leaf count is odd at any level, the last leaf is duplicated.
    """

    def __init__(self, leaves: Optional[List[str]] = None):
        self.leaves: List[str] = list(leaves) if leaves else []
        self.levels: List[List[str]] = []
        if self.leaves:
            self._build_tree()

    def _build_tree(self) -> None:
        """Construct all levels of the Merkle Tree up to the root."""
        current_level = list(self.leaves)
        self.levels = [current_level]

        while len(current_level) > 1:
            next_level: List[str] = []
            if len(current_level) % 2 == 1:
                current_level.append(current_level[-1])

            for i in range(0, len(current_level), 2):
                parent = hash_pair(current_level[i], current_level[i + 1])
                next_level.append(parent)

            self.levels.append(next_level)
            current_level = next_level

    @property
    def root(self) -> str:
        """Return the hex Merkle root of the tree, or empty string if no leaves."""
        if not self.levels or not self.levels[-1]:
            return ""
        return self.levels[-1][0]

    def get_inclusion_proof(self, index: int) -> List[Dict[str, str]]:
        """
        Generate Merkle inclusion proof for the leaf at `index`.
        Returns a list of dicts: [{'position': 'left'|'right', 'hash': <hex>}, ...].
        """
        if index < 0 or index >= len(self.leaves):
            raise IndexError(f"Leaf index {index} out of bounds (total leaves: {len(self.leaves)})")

        proof: List[Dict[str, str]] = []
        curr_idx = index

        for level in self.levels[:-1]:
            level_copy = list(level)
            if len(level_copy) % 2 == 1:
                level_copy.append(level_copy[-1])

            is_even = (curr_idx % 2 == 0)
            sibling_idx = curr_idx + 1 if is_even else curr_idx - 1

            if sibling_idx < len(level_copy):
                sibling_hash = level_copy[sibling_idx]
                position = "right" if is_even else "left"
                proof.append({"position": position, "hash": sibling_hash})

            curr_idx //= 2

        return proof

    @staticmethod
    def verify_inclusion_proof(leaf_hash: str, proof: List[Dict[str, str]], expected_root: str) -> bool:
        """
        Verify that a leaf hash belongs to a Merkle Tree with `expected_root`
        using the provided inclusion proof.
        """
        current = leaf_hash
        for step in proof:
            sibling = step["hash"]
            position = step["position"]

            if position == "right":
                current = hash_pair(current, sibling)
            else:
                current = hash_pair(sibling, current)

        return current.lower() == expected_root.lower()
