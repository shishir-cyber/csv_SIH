"""
test_input_side.py
-------------------
Full smoke test for Step 1 (input side). Confirms that:
  1. preprocess_config.json loads and has expected keys
  2. the ONNX model loads correctly
  3. each sample image can be preprocessed and run through the model
  4. we get back input_bytes, output_bytes, model_bytes -- the raw
     material Step 2 (receipt_building) will hash

Run from anywhere -- paths are resolved relative to this file.

Usage:
    python test_input_side.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]          # cv-assure-m3/  (or m3_inference_provenance/)
SRC = ROOT / "01_input_side" / "src"
sys.path.insert(0, str(SRC))

from preprocess import load_preprocess_config        # noqa: E402
from run_inference import run_batch_inference         # noqa: E402

CONFIG_PATH = ROOT / "00_setup" / "config" / "preprocess_config.json"
IMAGE_DIR = ROOT / "01_input_side" / "data" / "sample_images"
MODEL_PATH = ROOT / "01_input_side" / "data" / "model" / "demo_model.onnx"


def check_config():
    config = load_preprocess_config(str(CONFIG_PATH))
    assert config["color_format"] == "RGB"
    assert tuple(config["image_size"]) == (224, 224)
    print("[1/2] preprocess config: OK")
    return config


def check_inference():
    if not MODEL_PATH.exists():
        print(f"[2/2] SKIPPED - model not found at: {MODEL_PATH}")
        print("      Run export_demo_model.py and copy demo_model.onnx there first.")
        return

    image_paths = sorted(
        str(p) for p in IMAGE_DIR.glob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    if not image_paths:
        print(f"[2/2] SKIPPED - no images found in: {IMAGE_DIR}")
        return

    print(f"[2/2] Running inference on {len(image_paths)} image(s)...\n")

    for result in run_batch_inference(image_paths, str(MODEL_PATH), str(CONFIG_PATH)):
        print(f"  Image: {Path(result['image_path']).name}")
        print(f"    input_bytes length : {len(result['input_bytes'])}")
        print(f"    output_bytes length: {len(result['output_bytes'])}")
        print(f"    model_bytes length : {len(result['model_bytes'])}")
        print(f"    output shape        : {result['output_array'].shape}")
        print()

    print("Input side pipeline: OK")


def main():
    print("Running input-side smoke test...\n")
    check_config()
    check_inference()


if __name__ == "__main__":
    main()