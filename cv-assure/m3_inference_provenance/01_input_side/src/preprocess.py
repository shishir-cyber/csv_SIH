"""
preprocess.py
-------------
Turns a raw image file into a normalized tensor.
"""

import json
import numpy as np
from PIL import Image


def load_preprocess_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def preprocess_image(image_path: str, config: dict) -> np.ndarray:
    color_format = config.get("color_format", config.get("channel_order", "RGB"))
    image_size = config.get("image_size")
    if image_size is None:
        image_size = (config["width"], config["height"])

    img = Image.open(image_path).convert(color_format)
    img = img.resize(tuple(image_size))

    arr = np.array(img).astype("float32") * config.get("scale", 1.0 / 255.0)

    if config.get("normalize", "mean" in config and "std" in config):
        mean = np.array(config["mean"], dtype="float32")
        std = np.array(config["std"], dtype="float32")
        arr = (arr - mean) / std

    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)

    return arr.astype("float32")


def get_raw_image_bytes(image_path: str) -> bytes:
    with open(image_path, "rb") as f:
        return f.read()