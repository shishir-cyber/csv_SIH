"""
model_loader.py
----------------
Loads an ONNX model and exposes an inference session.
"""

import os
import onnxruntime as ort


def load_model(model_path: str) -> ort.InferenceSession:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session


def get_model_input_name(session: ort.InferenceSession) -> str:
    return session.get_inputs()[0].name


def get_model_output_names(session: ort.InferenceSession) -> list:
    return [o.name for o in session.get_outputs()]


def read_model_bytes(model_path: str) -> bytes:
    with open(model_path, "rb") as f:
        return f.read()