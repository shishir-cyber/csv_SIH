"""Capability Probe Engine for Module M0 (Ingest & Capability Probe).

Inspects the subject model, ingested AssetRecords, and optional reference
profile to produce a ``CapabilityMatrix`` every downstream detector consults.
"""

from __future__ import annotations

import json
import logging
import struct
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

_CV_ASSURE = Path(__file__).resolve().parent.parent
_M0_SRC = _CV_ASSURE / "m0_ingest_capability" / "src"
for _path in (str(_CV_ASSURE), str(_M0_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from schema import (  # noqa: E402
    AccessTier,
    Annotation,
    AssetRecord,
    CapabilityMatrix,
    ReferenceMode,
    TaskType,
    _utc_now_iso,
)
from shared.crypto.hashing import sha256_file  # noqa: E402
from shared.reference_profile.profile_loader import (  # noqa: E402
    load_reference_profile,
    profile_digest,
    verify_profile_attestation,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

WEIGHT_SUFFIXES = {".onnx", ".pt", ".pth", ".safetensors"}
LOGIT_KEYS = ("logit", "logits", "score", "scores", "confidence", "probabilities", "prob")
WEIGHT_KEYS = ("weights", "state_dict", "parameters", "tensors")
LABEL_KEYS = ("label", "labels", "class", "classes", "pred", "prediction")
API_SCHEMES = {"http", "https"}
SEGMENTATION_KEYS = ("segmentation", "polygon", "polygons", "mask", "rle", "counts")


def probe_capabilities(
    model_path: Optional[Path],
    dataset_records: List[AssetRecord],
    reference_profile_path: Optional[Path],
) -> CapabilityMatrix:
    """Probe access tier, task type, and reference mode for an ingest session.

    Args:
        model_path: Local weight file, API descriptor JSON, or URL-like path.
            ``None`` means labels-only (T0).
        dataset_records: Normalized assets from COCO/YOLO ingest.
        reference_profile_path: Optional YAML/JSON reference profile to attest
            or bootstrap.

    Returns:
        A populated ``CapabilityMatrix`` including SHA-256 digests when the
        model and/or reference profile exist as files.
    """
    records = list(dataset_records or [])
    model_spec = _normalize_model_spec(model_path)

    access_tier, source, tier_meta, model_digest = _determine_access_tier(model_spec)
    task_type, task_meta = _determine_task_type(records)
    reference_mode, profile_digest_hex, ref_meta = _determine_reference_mode(
        reference_profile_path,
        records,
    )

    metadata: Dict[str, Any] = {
        "tier": tier_meta,
        "task": task_meta,
        "reference": ref_meta,
        "asset_count": len(records),
        "contributor_ids": sorted({record.contributor_id for record in records}),
    }

    return CapabilityMatrix(
        access_tier=access_tier,
        task_type=task_type,
        reference_mode=reference_mode,
        model_digest=model_digest,
        reference_profile_digest=profile_digest_hex,
        probe_timestamp=_utc_now_iso(),
        source=source,
        metadata=metadata,
    )


def _normalize_model_spec(model_path: Optional[PathLike]) -> Optional[str]:
    if model_path is None:
        return None
    if isinstance(model_path, str):
        text = model_path.strip()
    else:
        text = str(model_path).strip()
        # pathlib collapses "https://host" into "https:/host".
        if text.startswith("https:/") and not text.startswith("https://"):
            text = "https://" + text[len("https:/") :]
        elif text.startswith("http:/") and not text.startswith("http://"):
            text = "http://" + text[len("http:/") :]
    return text or None


def _determine_access_tier(
    model_spec: Optional[str],
) -> Tuple[AccessTier, Optional[str], Dict[str, Any], Optional[str]]:
    """Classify T2 / T1 / T0 and hash a local model file when present."""
    if not model_spec:
        return (
            AccessTier.T0_LABELS,
            "unknown",
            {"reason": "NO_MODEL: labels-only ingest"},
            None,
        )

    if _looks_like_url(model_spec):
        return (
            AccessTier.T1_LOGITS,
            "remote_api",
            {"reason": "API_ENDPOINT: remote URL assumed to expose scores/logits", "endpoint": model_spec},
            None,
        )

    path = Path(model_spec).expanduser()
    if not path.exists():
        return (
            AccessTier.T0_LABELS,
            "unknown",
            {"reason": f"MISSING_MODEL: path does not exist ({path})"},
            None,
        )

    if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml"}:
        tier, meta = _tier_from_api_descriptor(path)
        digest = _safe_sha256(path)
        source = "remote_api" if tier != AccessTier.T2_WEIGHTS else "local_file"
        return tier, source, meta, digest

    if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES:
        readable, detail = _contains_readable_weights(path)
        digest = _safe_sha256(path)
        if readable:
            return (
                AccessTier.T2_WEIGHTS,
                "local_file",
                {"reason": "WEIGHTS_READABLE", **detail},
                digest,
            )
        if _onnx_exposes_logits(path, detail):
            return (
                AccessTier.T1_LOGITS,
                "local_file",
                {"reason": "ONNX_LOGITS_WITHOUT_WEIGHTS", **detail},
                digest,
            )
        return (
            AccessTier.T0_LABELS,
            "local_file",
            {"reason": "WEIGHTS_UNREADABLE", **detail},
            digest,
        )

    if path.is_file():
        digest = _safe_sha256(path)
        return (
            AccessTier.T0_LABELS,
            "local_file",
            {"reason": f"UNSUPPORTED_ARTIFACT: suffix {path.suffix!r} is not a weight format"},
            digest,
        )

    return (
        AccessTier.T0_LABELS,
        "unknown",
        {"reason": f"NOT_A_FILE: {path}"},
        None,
    )


def _looks_like_url(spec: str) -> bool:
    parsed = urlparse(spec)
    if parsed.scheme.lower() in API_SCHEMES and parsed.netloc:
        return True
    return spec.startswith("//")


def _tier_from_api_descriptor(path: Path) -> Tuple[AccessTier, Dict[str, Any]]:
    """Interpret a JSON/YAML sidecar that describes an inference API."""
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix.lower() == ".json" else __import__("yaml").safe_load(text)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return AccessTier.T0_LABELS, {"reason": f"API_DESCRIPTOR_UNREADABLE: {exc}"}

    if not isinstance(data, dict):
        return AccessTier.T0_LABELS, {"reason": "API_DESCRIPTOR_NOT_AN_OBJECT"}

    blob = json.dumps(data).lower()
    keys = [str(key).lower() for key in data.keys()]
    if any(token in blob for token in WEIGHT_KEYS) or any(key in WEIGHT_KEYS for key in keys):
        return AccessTier.T2_WEIGHTS, {"reason": "API_RETURNS_WEIGHTS", "descriptor": str(path)}
    if any(token in blob for token in LOGIT_KEYS):
        return AccessTier.T1_LOGITS, {"reason": "API_RETURNS_LOGITS", "descriptor": str(path)}
    endpoint = data.get("endpoint") or data.get("url") or data.get("api")
    if isinstance(endpoint, str) and _looks_like_url(endpoint):
        return AccessTier.T1_LOGITS, {"reason": "API_ENDPOINT", "endpoint": endpoint}
    return AccessTier.T0_LABELS, {"reason": "API_LABELS_ONLY", "descriptor": str(path)}


def _contains_readable_weights(path: Path) -> Tuple[bool, Dict[str, Any]]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".onnx":
            return _onnx_has_weights(path)
        if suffix in {".pt", ".pth"}:
            return _pytorch_has_weights(path)
        if suffix == ".safetensors":
            return _safetensors_has_weights(path)
    except OSError as exc:
        return False, {"error": str(exc), "path": str(path)}
    return False, {"path": str(path)}


def _onnx_has_weights(path: Path) -> Tuple[bool, Dict[str, Any]]:
    try:
        import onnx
    except ImportError:
        size = path.stat().st_size
        return size > 0, {"format": "onnx", "onnx_available": False, "bytes": size}

    try:
        model = onnx.load(str(path))
        count = len(model.graph.initializer)
        outputs = [tensor.name for tensor in model.graph.output]
        return count > 0, {
            "format": "onnx",
            "initializer_count": count,
            "outputs": outputs,
        }
    except Exception as exc:  # noqa: BLE001 - probe must never crash the ingest
        return False, {"format": "onnx", "error": str(exc)}


def _onnx_exposes_logits(path: Path, detail: Mapping[str, Any]) -> bool:
    outputs = [str(name).lower() for name in detail.get("outputs") or []]
    if any(any(token in name for token in LOGIT_KEYS) for name in outputs):
        return True
    if path.suffix.lower() != ".onnx":
        return False
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        names = [out.name.lower() for out in session.get_outputs()]
        return any(any(token in name for token in LOGIT_KEYS) for name in names)
    except Exception:  # noqa: BLE001
        return False


def _pytorch_has_weights(path: Path) -> Tuple[bool, Dict[str, Any]]:
    header = path.read_bytes()[:8]
    # Torch zip serialization (modern .pt) or pickle (legacy).
    is_zip = header.startswith(b"PK")
    is_pickle = header[:1] == b"\x80" and len(header) > 1 and header[1] in b"\x02\x03\x04\x05"
    readable = (is_zip or is_pickle) and path.stat().st_size > 0
    return readable, {
        "format": "pytorch",
        "container": "zip" if is_zip else "pickle" if is_pickle else "unknown",
        "bytes": path.stat().st_size,
    }


def _safetensors_has_weights(path: Path) -> Tuple[bool, Dict[str, Any]]:
    with path.open("rb") as handle:
        size_bytes = handle.read(8)
        if len(size_bytes) < 8:
            return False, {"format": "safetensors", "error": "truncated_header"}
        header_len = struct.unpack("<Q", size_bytes)[0]
        if header_len <= 0 or header_len > 100_000_000:
            return False, {"format": "safetensors", "error": "invalid_header_length"}
        raw_header = handle.read(header_len)
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, {"format": "safetensors", "error": f"header_json: {exc}"}
    if not isinstance(header, dict):
        return False, {"format": "safetensors", "error": "header_not_object"}
    tensors = [key for key in header.keys() if key != "__metadata__" and isinstance(header.get(key), dict)]
    return len(tensors) > 0, {"format": "safetensors", "tensor_count": len(tensors)}


def _safe_sha256(path: Path) -> Optional[str]:
    try:
        return sha256_file(path)
    except OSError as exc:
        logger.warning("Unable to hash %s: %s", path, exc)
        return None


def _determine_task_type(records: Sequence[AssetRecord]) -> Tuple[TaskType, Dict[str, Any]]:
    """Infer DETECTION / SEGMENTATION / CLASSIFICATION from annotations."""
    saw_boxes = False
    saw_polygons = False
    for payload in _iter_annotation_payloads(records):
        if _has_segmentation(payload):
            saw_polygons = True
        if _has_bbox(payload):
            saw_boxes = True

    if saw_polygons:
        return TaskType.SEGMENTATION, {"reason": "SEGMENTATION_POLYGONS_PRESENT", "boxes_present": saw_boxes}
    if saw_boxes:
        return TaskType.DETECTION, {"reason": "BBOX_XYWH_PRESENT"}
    return TaskType.CLASSIFICATION, {"reason": "NO_BOXES_OR_POLYGONS"}


def _iter_annotation_payloads(records: Sequence[AssetRecord]) -> Iterable[Dict[str, Any]]:
    for record in records:
        for annotation in record.annotations or []:
            yield _annotation_payload(annotation)


def _annotation_payload(annotation: Union[Annotation, Mapping[str, Any], Any]) -> Dict[str, Any]:
    if isinstance(annotation, Annotation):
        return annotation.model_dump()
    if isinstance(annotation, Mapping):
        return dict(annotation)
    if hasattr(annotation, "model_dump"):
        return dict(annotation.model_dump())
    return {"value": annotation}


def _has_segmentation(payload: Mapping[str, Any]) -> bool:
    for key in SEGMENTATION_KEYS:
        value = payload.get(key)
        if value:
            return True
    return False


def _has_bbox(payload: Mapping[str, Any]) -> bool:
    bbox = payload.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        _, _, width, height = (float(part) for part in bbox)
    except (TypeError, ValueError):
        return False
    return width > 0 and height > 0


def _determine_reference_mode(
    reference_profile_path: Optional[PathLike],
    records: Sequence[AssetRecord],
) -> Tuple[ReferenceMode, Optional[str], Dict[str, Any]]:
    """ATTESTED if the profile verifies; else BOOTSTRAPPED from cohorts; else UNREFERENCED."""
    contributors = {record.contributor_id for record in records if record.contributor_id}
    digest: Optional[str] = None
    profile_path = Path(reference_profile_path).expanduser() if reference_profile_path is not None else None

    if profile_path is not None and profile_path.is_file():
        digest = profile_digest(profile_path)
        if verify_profile_attestation(profile_path):
            return (
                ReferenceMode.ATTESTED,
                digest,
                {"reason": "ATTESTATION_VALID", "profile": str(profile_path)},
            )

        if _profile_is_bootstrap(profile_path) or contributors:
            return (
                ReferenceMode.BOOTSTRAPPED,
                digest,
                {
                    "reason": "PROFILE_UNSIGNED_BOOTSTRAP_FROM_COHORTS",
                    "profile": str(profile_path),
                    "contributor_count": len(contributors),
                },
            )
        return (
            ReferenceMode.UNREFERENCED,
            digest,
            {"reason": "PROFILE_PRESENT_BUT_NOT_ATTESTED_OR_BOOTSTRAPPABLE", "profile": str(profile_path)},
        )

    if profile_path is not None and str(profile_path).strip() and not profile_path.is_file():
        logger.warning("Reference profile path does not exist: %s", profile_path)

    if contributors:
        return (
            ReferenceMode.BOOTSTRAPPED,
            None,
            {
                "reason": "BOOTSTRAPPED_FROM_CONTRIBUTOR_COHORTS",
                "contributor_count": len(contributors),
            },
        )

    return ReferenceMode.UNREFERENCED, None, {"reason": "NO_REFERENCE_PROFILE"}


def _profile_is_bootstrap(profile_path: Path) -> bool:
    try:
        profile = load_reference_profile(profile_path)
    except (OSError, ValueError):
        return False
    status = str(profile.get("status") or profile.get("mode") or "").lower()
    if status in {"bootstrap", "bootstrapped", "scaffold"}:
        return True
    if profile.get("bootstrap") or profile.get("cohorts") or profile.get("contributor_cohorts"):
        return True
    clean_set = profile.get("clean_set")
    return isinstance(clean_set, list) and len(clean_set) > 0
