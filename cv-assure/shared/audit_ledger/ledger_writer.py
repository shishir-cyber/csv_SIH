"""Append-only hash-chained JSONL ledger writer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

PathLike = Union[str, Path]
GENESIS_HASH = "0" * 64


class LedgerWriter:
    """Minimal append-only JSONL audit ledger with a previous-hash chain.

    Each line is a JSON object. ``entry_hash`` is SHA-256 of the canonical
    record (including ``prev_hash``, excluding ``entry_hash`` itself).
    """

    def __init__(self, ledger_path: PathLike) -> None:
        self.path = Path(ledger_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._prev_hash = self._read_tip() or GENESIS_HASH

    def _read_tip(self) -> Optional[str]:
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return None
        last_line = ""
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last_line = line
        if not last_line:
            return None
        try:
            record = json.loads(last_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt ledger tip in {self.path}: {exc}") from exc
        tip = record.get("entry_hash")
        if not isinstance(tip, str) or len(tip) != 64:
            raise ValueError(f"Ledger tip is missing a valid entry_hash: {self.path}")
        return tip

    def append(self, event_type: str, payload: Dict[str, Any]) -> str:
        """Append ``payload`` as ``event_type`` and return the new entry hash."""
        record: Dict[str, Any] = {
            "event_type": event_type,
            "payload": payload,
            "prev_hash": self._prev_hash,
        }
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
        entry_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        record["entry_hash"] = entry_hash
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self._prev_hash = entry_hash
        return entry_hash
