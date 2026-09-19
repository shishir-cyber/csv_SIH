"""
run_inference.py
-----------------
Ties model_loader.py and preprocess.py together.
NOTE: uses plain (absolute) imports, NOT relative imports (no dots),
because this file is run as a standalone module, not as part of a package.
"""

import numpy as np

from model_loader import load_model, get_model_input_name, read_model_bytes
from preprocess import load_preprocess_config, preprocess_image, get_raw_image_bytes


def run_single_inference(image_path: str, model_path: str, config_path: str):
    config = load_preprocess_config(config_path)
    session = load_model(model_path)
    input_name = get_model_input_name(session)

    input_tensor = preprocess_image(image_path, config)
    outputs = session.run(None, {input_name: input_tensor})
    output_array = outputs[0]

    input_bytes = get_raw_image_bytes(image_path)
    output_bytes = np.ascontiguousarray(output_array).tobytes()
    model_bytes = read_model_bytes(model_path)

    return {
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "model_bytes": model_bytes,
        "preprocess_config": config,
        "output_array": output_array,
    }


def run_batch_inference(image_paths: list, model_path: str, config_path: str):
    config = load_preprocess_config(config_path)
    session = load_model(model_path)
    input_name = get_model_input_name(session)
    model_bytes = read_model_bytes(model_path)

    for image_path in image_paths:
        input_tensor = preprocess_image(image_path, config)
        outputs = session.run(None, {input_name: input_tensor})
        output_array = outputs[0]

        yield {
            "image_path": image_path,
            "input_bytes": get_raw_image_bytes(image_path),
            "output_bytes": np.ascontiguousarray(output_array).tobytes(),
            "model_bytes": model_bytes,
            "preprocess_config": config,
            "output_array": output_array,
        }