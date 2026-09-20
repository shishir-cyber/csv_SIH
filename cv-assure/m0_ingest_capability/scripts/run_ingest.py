"""M0 ingest entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_CV_ASSURE = Path(__file__).resolve().parents[2]
_M0_SRC = Path(__file__).resolve().parents[1] / "src"
for _path in (str(_CV_ASSURE), str(_M0_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from adapters import normalize_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize a COCO or YOLO dataset into AssetRecords.")
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--format", dest="format_hint", default="auto")
    parser.add_argument("--contributor-id", required=True)
    parser.add_argument("--ledger", type=Path, default=None)
    args = parser.parse_args(argv)

    records = normalize_dataset(
        args.dataset_path,
        args.format_hint,
        args.contributor_id,
        ledger=args.ledger,
    )
    print(f"ingested {len(records)} assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
