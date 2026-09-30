"""Build WFD-YOLO and verify parallel gates plus the native YOLOv12 loss."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo_config"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.nn.modules import ParallelFusion, WFCAParallel  # noqa: E402


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN or Inf")


def assert_tree_finite(name: str, value) -> None:
    if isinstance(value, torch.Tensor):
        assert_finite(name, value)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            assert_tree_finite(f"{name}[{index}]", item)


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    x = torch.randn(2, 32, 31, 47, device=device, requires_grad=True)
    wfca = WFCAParallel(32).to(device)
    custom = wfca(x)
    fusion = ParallelFusion([48, 32]).to(device)
    fused = fusion((torch.randn(2, 48, 31, 47, device=device), custom))
    assert fused.shape == (2, 80, 31, 47)
    assert_finite("custom module output", fused)
    fused.mean().backward()
    assert_finite("custom module gradient", x.grad)
    for module in (wfca, fusion):
        baseline, innovation = module.branch_weights()
        if abs(baseline - 0.95) > 1e-6 or abs(innovation - 0.05) > 1e-6:
            raise RuntimeError(f"Unexpected initial branch weights: {(baseline, innovation)}")

    cfg = ROOT / "ultralytics" / "cfg" / "models" / "v12" / "wfd-yolov12n.yaml"
    wrapper = YOLO(str(cfg))
    model = wrapper.model.to(device).train()
    model.args = get_cfg()
    batch = {
        "img": torch.rand(2, 3, 128, 128, device=device),
        "batch_idx": torch.tensor([0, 1], device=device),
        "cls": torch.tensor([[0.0], [25.0]], device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.30], [0.4, 0.6, 0.20, 0.18]], device=device),
    }
    loss, components = model.loss(batch)
    assert_finite("model loss", loss)
    assert_finite("loss components", components)
    loss.backward()
    if device.type == "cuda":
        model.eval().half()
        with torch.no_grad():
            half_predictions = model(batch["img"].half())
        assert_tree_finite("FP16 validation output", half_predictions)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    criterion = model.criterion
    if hasattr(criterion, "use_atfl") or hasattr(criterion, "atfl"):
        raise RuntimeError("ATFL must not be present; WFD-YOLO should use the native YOLOv12 loss")
    branch_weights = {
        name: module.branch_weights()
        for name, module in model.named_modules()
        if isinstance(module, (WFCAParallel, ParallelFusion))
    }
    report = {
        "device": str(device),
        "torch": torch.__version__,
        "parameters": parameters,
        "trainable_parameters": trainable,
        "loss": float(loss.detach().cpu()),
        "loss_components": [float(value) for value in components.detach().cpu()],
        "native_yolov12_loss": True,
        "branch_weights": branch_weights,
        "output_shape": list(fused.shape),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
