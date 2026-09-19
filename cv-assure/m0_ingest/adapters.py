"""COCO and YOLO format adapters for Module M0 (Ingest & Capability Probe).

Converts native dataset layouts into standardized ``AssetRecord`` lists,
hashes source files in chunks, and appends those digests to the audit ledger.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import yaml
from PIL import Image, UnidentifiedImageError

_CV_ASSURE = Path(__file__).resolve().parent.parent
_M0_SRC = _CV_ASSURE / "m0_ingest_capability" / "src"
for _path in (str(_CV_ASSURE), str(_M0_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from schema import Annotation, AssetRecord, ImageMetadata, _utc_now_iso  # noqa: E402
from shared.audit_ledger.ledger_writer import LedgerWriter  # noqa: E402
from shared.crypto.hashing import sha256_file as _shared_sha256_file  # noqa: E402

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
COCO_HINTS = {"coco", "ms-coco", "mscoco"}
YOLO_HINTS = {"yolo", "yolov5", "yolov8", "ultralytics"}

PathLike = Union[str, Path]


class IngestError(Exception):
    """Base error for dataset ingest failures."""


class MissingAssetError(IngestError):
    """A required dataset file or directory is missing."""


class CorruptAnnotationError(IngestError):
    """Annotation JSON, YAML, or label file could not be parsed."""


class BadImageHeaderError(IngestError):
    """Image bytes could not be decoded as a valid raster header."""


def sha256_file(path: PathLike, chunk_size: int = 1024 * 1024) -> str:
    """Calculate the SHA-256 digest of ``path`` by reading the file in chunks.

    This is the M0 ingest hashing primitive used for images and YOLO label files.
    """
    return _shared_sha256_file(path, chunk_size=chunk_size)


def _as_path(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()


def _require_file(path: Path, what: str) -> Path:
    if not path.is_file():
        raise MissingAssetError(f"{what} not found: {path}")
    return path


def _require_dir(path: Path, what: str) -> Path:
    if not path.is_dir():
        raise MissingAssetError(f"{what} not found: {path}")
    return path


def _read_image_metadata(image_path: Path) -> ImageMetadata:
    """Load width/height/channels, rejecting truncated or non-image files."""
    try:
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            channels = len(image.getbands())
    except UnidentifiedImageError as exc:
        raise BadImageHeaderError(f"Unrecognized image header: {image_path}") from exc
    except OSError as exc:
        raise BadImageHeaderError(f"Corrupt or unreadable image: {image_path}") from exc

    if width < 1 or height < 1 or channels < 1:
        raise BadImageHeaderError(f"Invalid image geometry {width}x{height}x{channels}: {image_path}")
    return ImageMetadata(width=width, height=height, channels=channels)


def _ledger_from(ledger: Optional[Union[LedgerWriter, PathLike]], fallback: Path) -> LedgerWriter:
    if isinstance(ledger, LedgerWriter):
        return ledger
    if ledger is not None:
        return LedgerWriter(ledger)
    return LedgerWriter(fallback)


def _log_asset_hashes(
    ledger: LedgerWriter,
    record: AssetRecord,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "asset_id": record.asset_id,
        "file_path": str(record.file_path),
        "format": record.format,
        "contributor_id": record.contributor_id,
        "image_sha256": record.sha256,
        "annotation_count": len(record.annotations),
    }
    if extra:
        payload.update(extra)
    ledger.append("m0.ingest.asset", payload)


def yolo_norm_to_xywh(
    cx: float,
    cy: float,
    nw: float,
    nh: float,
    width: int,
    height: int,
) -> List[float]:
    """Convert YOLO normalized ``[cx, cy, w, h]`` to pixel ``[x_min, y_min, w, h]``."""
    box_w = nw * width
    box_h = nh * height
    x_min = (cx - nw / 2.0) * width
    y_min = (cy - nh / 2.0) * height
    x_min = max(0.0, min(x_min, float(width)))
    y_min = max(0.0, min(y_min, float(height)))
    box_w = max(0.0, min(box_w, float(width) - x_min))
    box_h = max(0.0, min(box_h, float(height) - y_min))
    return [x_min, y_min, box_w, box_h]


class COCOAdapter:
    """Ingest a COCO detection dataset (annotations JSON + image folder)."""

    def ingest(
        self,
        annotations_json: PathLike,
        image_root: PathLike,
        contributor_id: str,
        *,
        ledger: Optional[Union[LedgerWriter, PathLike]] = None,
        strict: bool = False,
    ) -> List[AssetRecord]:
        """Parse COCO ``images`` / ``annotations`` / ``categories`` into AssetRecords."""
        ann_path = _require_file(_as_path(annotations_json), "COCO annotations JSON")
        img_root = _require_dir(_as_path(image_root), "COCO image folder")
        writer = _ledger_from(ledger, ann_path.parent / ".cv_assure_ingest_ledger.jsonl")

        try:
            with ann_path.open("r", encoding="utf-8") as handle:
                coco = json.load(handle)
        except json.JSONDecodeError as exc:
            raise CorruptAnnotationError(f"Corrupt COCO JSON {ann_path}: {exc}") from exc
        except OSError as exc:
            raise MissingAssetError(f"Cannot read COCO JSON {ann_path}: {exc}") from exc

        if not isinstance(coco, dict):
            raise CorruptAnnotationError(f"COCO JSON must be an object: {ann_path}")

        categories = {int(cat["id"]): str(cat.get("name") or f"class_{cat['id']}") for cat in coco.get("categories") or []}
        by_image: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
        for annotation in coco.get("annotations") or []:
            try:
                image_id = int(annotation["image_id"])
            except (KeyError, TypeError, ValueError) as exc:
                if strict:
                    raise CorruptAnnotationError(f"Annotation missing image_id in {ann_path}") from exc
                logger.warning("Skipping COCO annotation without image_id in %s", ann_path)
                continue
            by_image[image_id].append(annotation)

        records: List[AssetRecord] = []
        timestamp = _utc_now_iso()
        images = coco.get("images") or []
        if not images:
            raise CorruptAnnotationError(f"COCO JSON has no images: {ann_path}")

        for image_info in images:
            try:
                record = self._record_for_image(
                    image_info,
                    img_root,
                    by_image,
                    categories,
                    contributor_id,
                    timestamp,
                )
            except (MissingAssetError, BadImageHeaderError, CorruptAnnotationError) as exc:
                if strict:
                    raise
                logger.warning("Skipping COCO image: %s", exc)
                continue
            _log_asset_hashes(writer, record)
            records.append(record)

        writer.append(
            "m0.ingest.dataset",
            {
                "format": "COCO",
                "annotations_json": str(ann_path),
                "annotations_sha256": sha256_file(ann_path),
                "image_root": str(img_root),
                "asset_count": len(records),
                "contributor_id": contributor_id,
            },
        )
        return records

    def _record_for_image(
        self,
        image_info: Mapping[str, Any],
        image_root: Path,
        by_image: Mapping[int, Sequence[Mapping[str, Any]]],
        categories: Mapping[int, str],
        contributor_id: str,
        timestamp: str,
    ) -> AssetRecord:
        try:
            image_id = image_info["id"]
            file_name = image_info["file_name"]
        except KeyError as exc:
            raise CorruptAnnotationError(f"COCO image entry missing {exc}") from exc

        image_path = image_root / str(file_name)
        if not image_path.is_file():
            raise MissingAssetError(f"COCO image listed but missing on disk: {image_path}")

        metadata = _read_image_metadata(image_path)
        annotations: List[Annotation] = []
        for raw in by_image.get(int(image_id), []):
            parsed = self._parse_annotation(raw, categories)
            if parsed is not None:
                annotations.append(parsed)

        return AssetRecord(
            asset_id=f"coco-{image_id}",
            file_path=image_path,
            sha256=sha256_file(image_path),
            format="COCO",
            image_metadata=metadata,
            annotations=annotations,
            contributor_id=contributor_id,
            timestamp=timestamp,
        )

    @staticmethod
    def _parse_annotation(
        raw: Mapping[str, Any],
        categories: Mapping[int, str],
    ) -> Optional[Annotation]:
        bbox = raw.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            logger.warning("Skipping COCO annotation with invalid bbox: %s", raw.get("id"))
            return None
        try:
            x, y, w, h = (float(part) for part in bbox)
            category_id = int(raw["category_id"])
        except (KeyError, TypeError, ValueError):
            logger.warning("Skipping COCO annotation with non-numeric bbox/category: %s", raw.get("id"))
            return None
        if w < 0 or h < 0:
            logger.warning("Skipping COCO annotation with negative box size: %s", raw.get("id"))
            return None
        x = max(0.0, x)
        y = max(0.0, y)
        class_name = categories.get(category_id, f"class_{category_id}")
        return Annotation(bbox=[x, y, w, h], category_id=category_id, class_name=class_name)


class YOLOAdapter:
    """Ingest a YOLO dataset (``images/``, ``labels/``, and ``data.yaml``)."""

    def ingest(
        self,
        dataset_root: PathLike,
        contributor_id: str,
        *,
        data_yaml: Optional[PathLike] = None,
        ledger: Optional[Union[LedgerWriter, PathLike]] = None,
        strict: bool = False,
    ) -> List[AssetRecord]:
        """Walk YOLO images, convert normalized boxes, and hash images plus labels."""
        root = _as_path(dataset_root)
        if root.is_file() and root.suffix.lower() in {".yaml", ".yml"}:
            yaml_path = root
            root = root.parent
        else:
            _require_dir(root, "YOLO dataset root")
            yaml_path = _as_path(data_yaml) if data_yaml is not None else self._find_data_yaml(root)

        yaml_path = _require_file(yaml_path, "YOLO data.yaml")
        writer = _ledger_from(ledger, root / ".cv_assure_ingest_ledger.jsonl")
        spec = self._load_yaml(yaml_path)
        class_names = self._class_names(spec)
        image_dirs = self._image_dirs(root, yaml_path, spec)

        records: List[AssetRecord] = []
        timestamp = _utc_now_iso()
        seen: set[Path] = set()

        for images_dir in image_dirs:
            labels_dir = self._labels_dir_for(images_dir)
            for image_path in self._iter_images(images_dir):
                if image_path in seen:
                    continue
                seen.add(image_path)
                try:
                    record, label_sha = self._record_for_image(
                        image_path,
                        images_dir,
                        labels_dir,
                        class_names,
                        contributor_id,
                        timestamp,
                    )
                except (BadImageHeaderError, CorruptAnnotationError, MissingAssetError) as exc:
                    if strict:
                        raise
                    logger.warning("Skipping YOLO image: %s", exc)
                    continue
                extra = {"label_sha256": label_sha} if label_sha else {"label_sha256": None}
                _log_asset_hashes(writer, record, extra=extra)
                records.append(record)

        writer.append(
            "m0.ingest.dataset",
            {
                "format": "YOLO",
                "dataset_root": str(root),
                "data_yaml": str(yaml_path),
                "data_yaml_sha256": sha256_file(yaml_path),
                "asset_count": len(records),
                "contributor_id": contributor_id,
            },
        )
        return records

    @staticmethod
    def _find_data_yaml(root: Path) -> Path:
        for name in ("data.yaml", "data.yml", "dataset.yaml"):
            candidate = root / name
            if candidate.is_file():
                return candidate
        raise MissingAssetError(f"No data.yaml found under {root}")

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                spec = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:
            raise CorruptAnnotationError(f"Corrupt YOLO YAML {path}: {exc}") from exc
        except OSError as exc:
            raise MissingAssetError(f"Cannot read YOLO YAML {path}: {exc}") from exc
        if not isinstance(spec, dict):
            raise CorruptAnnotationError(f"YOLO YAML must be a mapping: {path}")
        return spec

    @staticmethod
    def _class_names(spec: Mapping[str, Any]) -> Dict[int, str]:
        names = spec.get("names", spec.get("nc_names"))
        if names is None:
            count = spec.get("nc")
            if isinstance(count, int) and count >= 0:
                return {index: f"class_{index}" for index in range(count)}
            return {}
        if isinstance(names, dict):
            mapped: Dict[int, str] = {}
            for key, value in names.items():
                mapped[int(key)] = str(value)
            return mapped
        if isinstance(names, list):
            return {index: str(name) for index, name in enumerate(names)}
        raise CorruptAnnotationError("YOLO names must be a list or id→name mapping")

    def _image_dirs(self, root: Path, yaml_path: Path, spec: Mapping[str, Any]) -> List[Path]:
        base = Path(str(spec.get("path") or root))
        if not base.is_absolute():
            base = (yaml_path.parent / base).resolve()

        splits: List[Path] = []
        for key in ("train", "val", "test", "images"):
            value = spec.get(key)
            if not value or not isinstance(value, str):
                continue
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = (base / candidate).resolve()
            if candidate.is_file():
                candidate = candidate.parent
            # YAML split paths often point at an image list or an images/ subtree.
            if candidate.name != "images" and (candidate / "images").is_dir():
                candidate = candidate / "images"
            if candidate.is_dir():
                splits.append(candidate)

        default_images = root / "images"
        if default_images.is_dir():
            splits.append(default_images.resolve())

        unique: List[Path] = []
        seen: set[Path] = set()
        for directory in splits:
            if directory not in seen and directory.is_dir():
                seen.add(directory)
                unique.append(directory)
        if not unique:
            raise MissingAssetError(f"No YOLO images/ directory found under {root}")
        return unique

    @staticmethod
    def _labels_dir_for(images_dir: Path) -> Path:
        if images_dir.name == "images":
            return images_dir.parent / "labels"
        if "images" in images_dir.parts:
            parts = ["labels" if part == "images" else part for part in images_dir.parts]
            return Path(*parts)
        return images_dir.parent / "labels"

    @staticmethod
    def _iter_images(images_dir: Path) -> Iterable[Path]:
        yield from sorted(
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _record_for_image(
        self,
        image_path: Path,
        images_dir: Path,
        labels_dir: Path,
        class_names: Mapping[int, str],
        contributor_id: str,
        timestamp: str,
    ) -> tuple[AssetRecord, Optional[str]]:
        metadata = _read_image_metadata(image_path)
        try:
            relative = image_path.relative_to(images_dir)
        except ValueError:
            relative = Path(image_path.name)
        label_path = (labels_dir / relative).with_suffix(".txt")

        annotations: List[Annotation] = []
        label_sha: Optional[str] = None
        if label_path.is_file():
            label_sha = sha256_file(label_path)
            annotations = self._parse_label_file(
                label_path,
                metadata.width,
                metadata.height,
                class_names,
            )
        elif labels_dir.is_dir():
            logger.info("No YOLO label file for %s", image_path)

        return (
            AssetRecord(
                asset_id=f"yolo-{image_path.stem}",
                file_path=image_path,
                sha256=sha256_file(image_path),
                format="YOLO",
                image_metadata=metadata,
                annotations=annotations,
                contributor_id=contributor_id,
                timestamp=timestamp,
            ),
            label_sha,
        )

    def _parse_label_file(
        self,
        label_path: Path,
        width: int,
        height: int,
        class_names: Mapping[int, str],
    ) -> List[Annotation]:
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise CorruptAnnotationError(f"Cannot read YOLO label file {label_path}: {exc}") from exc

        annotations: List[Annotation] = []
        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                raise CorruptAnnotationError(
                    f"YOLO label {label_path}:{line_number} expected 5 fields, found {len(parts)}"
                )
            try:
                class_id = int(float(parts[0]))
                cx, cy, nw, nh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            except ValueError as exc:
                raise CorruptAnnotationError(
                    f"YOLO label {label_path}:{line_number} is not numeric"
                ) from exc
            bbox = yolo_norm_to_xywh(cx, cy, nw, nh, width, height)
            if bbox[2] <= 0 or bbox[3] <= 0:
                logger.warning("Skipping zero-area YOLO box in %s:%s", label_path, line_number)
                continue
            class_name = class_names.get(class_id, f"class_{class_id}")
            annotations.append(
                Annotation(bbox=bbox, category_id=class_id, class_name=class_name)
            )
        return annotations


def detect_format(dataset_path: Path, format_hint: str = "auto") -> str:
    """Return ``COCO`` or ``YOLO`` from ``format_hint`` or directory layout."""
    hint = (format_hint or "auto").strip().lower()
    if hint in COCO_HINTS:
        return "COCO"
    if hint in YOLO_HINTS:
        return "YOLO"
    if hint not in {"", "auto", "detect"}:
        raise IngestError(f"Unknown format_hint {format_hint!r}; expected COCO, YOLO, or auto")

    path = dataset_path
    if path.is_file() and path.suffix.lower() == ".json":
        return "COCO"
    if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
        return "YOLO"

    if (path / "data.yaml").is_file() or (path / "data.yml").is_file():
        return "YOLO"
    if (path / "labels").is_dir() and (path / "images").is_dir():
        return "YOLO"
    if (path / "annotations").is_dir() and any((path / "annotations").glob("*.json")):
        return "COCO"
    if any(path.glob("*.json")):
        return "COCO"
    raise IngestError(f"Unable to auto-detect COCO or YOLO layout at {path}")


def _resolve_coco_inputs(dataset_path: Path) -> tuple[Path, Path]:
    if dataset_path.is_file() and dataset_path.suffix.lower() == ".json":
        json_path = dataset_path
        for candidate in (json_path.parent / "images", json_path.parent.parent / "images", json_path.parent):
            if not candidate.is_dir():
                continue
            if any(p.suffix.lower() in IMAGE_EXTENSIONS for p in candidate.rglob("*") if p.is_file()):
                return json_path, candidate
        return json_path, json_path.parent

    annotations_dir = dataset_path / "annotations"
    json_candidates: List[Path] = []
    if annotations_dir.is_dir():
        json_candidates.extend(sorted(annotations_dir.glob("*.json")))
    json_candidates.extend(sorted(dataset_path.glob("*.json")))
    if not json_candidates:
        raise MissingAssetError(f"No COCO annotations JSON under {dataset_path}")
    json_path = json_candidates[0]

    for name in ("images", "train", "val", "test"):
        candidate = dataset_path / name
        if candidate.is_dir():
            return json_path, candidate
    return json_path, dataset_path


def normalize_dataset(
    dataset_path: Path,
    format_hint: str,
    contributor_id: str,
    *,
    ledger: Optional[Union[LedgerWriter, PathLike]] = None,
    strict: bool = False,
) -> List[AssetRecord]:
    """Ingest ``dataset_path`` as COCO or YOLO and return normalized AssetRecords.

    Hashes of images (and YOLO label files / COCO JSON) are appended to the
    audit ledger skeleton.
    """
    if not contributor_id:
        raise IngestError("contributor_id must be a non-empty string")

    path = _as_path(dataset_path)
    if not path.exists():
        raise MissingAssetError(f"Dataset path does not exist: {path}")

    detected = detect_format(path, format_hint)
    writer = _ledger_from(ledger, path / ".cv_assure_ingest_ledger.jsonl" if path.is_dir() else path.parent / ".cv_assure_ingest_ledger.jsonl")

    if detected == "COCO":
        annotations_json, image_root = _resolve_coco_inputs(path)
        records = COCOAdapter().ingest(
            annotations_json,
            image_root,
            contributor_id,
            ledger=writer,
            strict=strict,
        )
    else:
        records = YOLOAdapter().ingest(
            path,
            contributor_id,
            ledger=writer,
            strict=strict,
        )

    logger.info(
        "Normalized %s assets from %s (%s) for contributor %s",
        len(records),
        path,
        detected,
        contributor_id,
    )
    return records
