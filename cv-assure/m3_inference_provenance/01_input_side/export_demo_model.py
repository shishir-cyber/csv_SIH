"""Export a small deterministic torchvision model for local M3 demos."""

from pathlib import Path

import torch
from torchvision.models import resnet18


def main() -> None:
    output_path = Path(__file__).parent / "data" / "model" / "demo_model.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    model = resnet18(weights=None).eval()
    sample = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        sample,
        output_path,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported {output_path}")


if __name__ == "__main__":
    main()