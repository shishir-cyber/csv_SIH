"""Dual encoder binding for Module M0 (reference DINOv2-S + subject model).

``E_ref`` is the bundled, digest-pinned DINOv2-S feature extractor. ``E_sub``
is the contributed model (or a wrapper around an API) and is only invoked
when the capability probe reported T1 or T2 access.
"""

from __future__ import annotations

import logging
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

_CV_ASSURE = Path(__file__).resolve().parent.parent
_M0_SRC = _CV_ASSURE / "m0_ingest_capability" / "src"
for _path in (str(_CV_ASSURE), str(_M0_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from schema import AccessTier  # noqa: E402
from shared.crypto.hashing import sha256_file  # noqa: E402

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# SHA-256 of the bundled DINOv2-S artifact at ``BUNDLED_E_REF_PATH``.
# Replace this pin when the stub is swapped for the production ViT-S/14 weights.
PINNED_E_REF_DIGEST = "22f2072a8441979523cb99b474680196775131986b5ad559ac8eedcf97a2e409"

BUNDLED_E_REF_PATH = Path(__file__).resolve().parent / "weights" / "dinov2_vits14.onnx"
STUB_MAGIC = b"CVASSURE-DINOV2S-STUB"
REF_EMBED_DIM = 384
REF_INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


class IntegrityError(Exception):
    """Raised when the reference encoder fails its pinned-digest check."""


class DualEncoderBinding:
    """Holds ``E_ref`` (pinned DINOv2-S) and optional ``E_sub`` (contributed model)."""

    def __init__(
        self,
        e_sub: Any = None,
        access_tier: AccessTier = AccessTier.T0_LABELS,
        e_ref: Any = None,
    ) -> None:
        self.e_ref: Any = e_ref
        self.e_sub: Any = e_sub
        self.access_tier = AccessTier(access_tier) if not isinstance(access_tier, AccessTier) else access_tier
        self.e_ref_digest: Optional[str] = None
        self.e_ref_runtime: Optional[str] = None
        self.e_sub_runtime: Optional[str] = None

    def bind_subject(self, e_sub: Any, access_tier: AccessTier) -> None:
        """Attach the contributed encoder/wrapper and the probed access tier."""
        self.access_tier = access_tier
        if isinstance(e_sub, (str, Path)):
            self.e_sub, self.e_sub_runtime = load_encoder(Path(e_sub), role="e_sub")
        else:
            self.e_sub = e_sub
            self.e_sub_runtime = "callable" if e_sub is not None else None

    def verify_and_load_ref_encoder(self, model_path: Path) -> Any:
        """Load ``E_ref`` only if ``model_path`` matches ``PINNED_E_REF_DIGEST``.

        Raises:
            IntegrityError: File is missing or the SHA-256 digest does not match
                the hardcoded pin.
        """
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise IntegrityError(f"Reference encoder not found: {path}")

        digest = sha256_file(path)
        if digest.lower() != PINNED_E_REF_DIGEST.lower():
            raise IntegrityError(
                "E_ref digest mismatch: "
                f"expected {PINNED_E_REF_DIGEST}, found {digest}"
            )

        self.e_ref, self.e_ref_runtime = load_encoder(path, role="e_ref")
        self.e_ref_digest = digest
        logger.info("Loaded pinned E_ref via %s (%s)", self.e_ref_runtime, path)
        return self.e_ref

    def load_bundled_ref_encoder(self) -> Any:
        """Verify and load the in-repo DINOv2-S artifact."""
        return self.verify_and_load_ref_encoder(BUNDLED_E_REF_PATH)

    def extract_features(self, image_path: Path) -> Dict[str, Optional[List[float]]]:
        """Run ``E_ref`` (and ``E_sub`` when T1/T2) on ``image_path``.

        Returns:
            ``{"e_ref": ref_emb, "e_sub": sub_emb}`` where embeddings are
            float lists. ``e_sub`` is ``None`` when the subject encoder is
            unavailable (T0 or unbound).
        """
        if self.e_ref is None:
            raise RuntimeError("E_ref is not loaded; call verify_and_load_ref_encoder first")

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        tensor = preprocess_image(path)
        ref_emb = _as_vector(infer_encoder(self.e_ref, tensor, image_path=path))

        sub_emb: Optional[List[float]] = None
        if self._subject_available():
            raw_sub = infer_encoder(self.e_sub, tensor, image_path=path)
            sub_emb = _as_vector(raw_sub)

        return {"e_ref": ref_emb, "e_sub": sub_emb}

    def _subject_available(self) -> bool:
        return self.e_sub is not None and self.access_tier.satisfies(AccessTier.T1_LOGITS)


def preprocess_image(image_path: PathLike, size: int = REF_INPUT_SIZE) -> np.ndarray:
    """Load an image as a deterministic NCHW float32 batch (ImageNet-normalized)."""
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize((size, size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Cannot decode image {image_path}: {exc}") from exc

    tensor = np.transpose(array, (2, 0, 1))[None, ...]
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(tensor, dtype=np.float32)


def load_encoder(path: Path, role: str = "encoder") -> tuple[Any, str]:
    """Load an ONNX/Torch encoder on CPU, with a deterministic numpy fallback."""
    configure_cpu_determinism()

    ort_encoder = _try_onnxruntime(path)
    if ort_encoder is not None:
        return ort_encoder, "onnxruntime-cpu"

    torch_encoder = _try_torch(path)
    if torch_encoder is not None:
        return torch_encoder, "pytorch-cpu"

    stub = NumpyDinoStub.from_file(path)
    if stub is not None:
        logger.warning("%s: using deterministic CPU numpy fallback for %s", role, path)
        return stub, "numpy-cpu-fallback"

    raise IntegrityError(f"Unable to load {role} from {path} (no ONNX/Torch runtime and not a stub)")


def configure_cpu_determinism() -> None:
    """Force CPU-only, single-thread, deterministic kernels when runtimes exist."""
    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            logger.info("CUDA is available; DualEncoderBinding still pins E_ref/E_sub to CPU")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (TypeError, RuntimeError):
            pass
    except ImportError:
        pass

    np.random.seed(0)


def _try_onnxruntime(path: Path) -> Any:
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    try:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.enable_mem_pattern = False
        return ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:  # noqa: BLE001 - probe fallback is intentional
        logger.debug("ONNX Runtime rejected %s: %s", path, exc)
        return None


def _try_torch(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix not in {".pt", ".pth"}:
        return None
    try:
        import torch
    except ImportError:
        return None
    try:
        model = torch.load(str(path), map_location="cpu", weights_only=False)
        if hasattr(model, "eval"):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        return model
    except Exception as exc:  # noqa: BLE001
        logger.debug("PyTorch rejected %s: %s", path, exc)
        return None


class NumpyDinoStub:
    """Deterministic CPU stand-in: GAP(RGB) @ W + B → 384-d embedding."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray) -> None:
        self.weight = weight.astype(np.float32, copy=False)
        self.bias = bias.astype(np.float32, copy=False)

    @classmethod
    def from_file(cls, path: Path) -> Optional["NumpyDinoStub"]:
        payload = path.read_bytes()
        if not payload.startswith(STUB_MAGIC):
            return None
        floats = payload[32:]
        expected = (3 * REF_EMBED_DIM + REF_EMBED_DIM) * 4
        if len(floats) < expected:
            return None
        values = struct.unpack("<" + "f" * (3 * REF_EMBED_DIM + REF_EMBED_DIM), floats[:expected])
        weight = np.asarray(values[: 3 * REF_EMBED_DIM], dtype=np.float32).reshape(3, REF_EMBED_DIM)
        bias = np.asarray(values[3 * REF_EMBED_DIM :], dtype=np.float32)
        return cls(weight, bias)

    def embed(self, tensor: np.ndarray) -> np.ndarray:
        pooled = tensor.mean(axis=(2, 3)).reshape(3)
        return pooled @ self.weight + self.bias


def infer_encoder(encoder: Any, tensor: np.ndarray, image_path: Optional[Path] = None) -> np.ndarray:
    """Run a loaded encoder (ORT / Torch / stub / callable) on a preprocessed batch."""
    if isinstance(encoder, NumpyDinoStub):
        return encoder.embed(tensor)

    if hasattr(encoder, "get_inputs") and hasattr(encoder, "run"):
        input_name = encoder.get_inputs()[0].name
        outputs = encoder.run(None, {input_name: tensor})
        return np.asarray(outputs[0], dtype=np.float32)

    if hasattr(encoder, "eval") and callable(getattr(encoder, "__call__", None)):
        try:
            import torch

            with torch.no_grad():
                output = encoder(torch.from_numpy(tensor))
            if isinstance(output, (tuple, list)):
                output = output[0]
            if hasattr(output, "detach"):
                output = output.detach().cpu().numpy()
            return np.asarray(output, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Torch forward failed, trying callable protocol: %s", exc)

    if callable(encoder):
        try:
            output = encoder(tensor)
        except TypeError:
            if image_path is None:
                raise
            output = encoder(image_path)
        return np.asarray(output, dtype=np.float32)

    raise TypeError(f"Unsupported encoder type: {type(encoder)!r}")


def _as_vector(value: Any) -> List[float]:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return [float(x) for x in array.tolist()]
