"""Train the parallel-gated WFD-YOLO-N model on OSI26."""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo_config"))
os.environ.setdefault("YOLO_DATASET_CACHE_DIR", str(ROOT / ".dataset_cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402


DEFAULT_MODEL = ROOT / "ultralytics" / "cfg" / "models" / "v12" / "wfd-yolov12n.yaml"
DEFAULT_DATA = ROOT / "ultralytics" / "cfg" / "datasets" / "OSI26.yaml"


def transfer_yolov12n_weights(target: YOLO, weights: str) -> int:
    """
    Transfer baseline YOLOv12n weights across WFD-YOLO's inserted layers.

    All original YOLOv12n layers are retained unchanged. This map accounts for
    the two inserted WFCAParallel layers and the four Concat replacements
    (Concat itself has no parameters), preserving every shape-compatible
    backbone, neck, regression-head and DFL tensor.
    """
    source = YOLO(weights)
    # Reproduce Ultralytics' normal transfer-learning path first: construct the
    # baseline detector directly for the dataset's 26 classes with seed 1, then
    # load every compatible official COCO tensor. This also gives WFD the exact
    # same initialized 26-class prediction head as a baseline YOLOv12n run.
    # Ultralytics intentionally calls init_seeds(args.seed + 1 + RANK), so
    # training with seed=0 constructs the model with torch seed 1.
    baseline_yaml = deepcopy(source.model.yaml)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        baseline = DetectionModel(baseline_yaml, ch=3, nc=int(target.model.yaml["nc"]), verbose=False)
    baseline.load(source.model, verbose=False)

    source_layers = baseline.model
    target_layers = target.model.model
    layer_map = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 6,
        6: 7,
        7: 9,
        8: 10,
        9: 11,
        10: 12,
        11: 13,
        12: 14,
        13: 15,
        14: 16,
        15: 17,
        16: 18,
        17: 19,
        18: 20,
        19: 21,
        20: 22,
        21: 23,
    }
    transferred = 0
    for source_index, target_index in layer_map.items():
        source_state = source_layers[source_index].state_dict()
        target_module = target_layers[target_index]
        target_state = target_module.state_dict()
        compatible = {
            key: value
            for key, value in source_state.items()
            if key in target_state and value.shape == target_state[key].shape
        }
        target_module.load_state_dict(compatible, strict=False)
        transferred += len(compatible)
    print(
        f"Smart-transferred {transferred} tensors from the 26-class adapted "
        f"{weights} baseline into WFD-YOLO."
    )
    return transferred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=str, default="", help="Optional YOLOv12n checkpoint to partially transfer")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a WFD-YOLO last.pt checkpoint")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--name", type=str, default="wfd_yolov12n_parallel_pretrained_osi26_200e")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "wfd-yolo")
    parser.add_argument("--fraction", type=float, default=1.0, help="Dataset fraction; use a small value for smoke tests")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume:
        model = YOLO(str(args.resume))
        model.train(resume=True)
        return

    model = YOLO(str(args.model))
    if args.weights:
        transfer_yolov12n_weights(model, args.weights)
        # YOLO was created from YAML, so Ultralytics sets ``ckpt=None``.
        # Model.train() otherwise builds another YAML model and silently drops
        # every manually transferred tensor. Mark this model as checkpoint
        # backed so DetectionTrainer receives and reloads the transferred WFD
        # state instead of starting from scratch.
        model.ckpt = {"model": model.model}
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=args.name,
        fraction=args.fraction,
        patience=args.patience,
        cache=args.cache,
        amp=not args.no_amp,
        seed=0,
        deterministic=True,
        optimizer="auto",
        val=True,
        plots=True,
        verbose=True,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
