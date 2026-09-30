"""Evaluate an Ultralytics detection checkpoint on an OSI26 split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo_config"))
os.environ.setdefault("YOLO_DATASET_CACHE_DIR", str(ROOT / ".dataset_cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import ParallelFusion, WFCAParallel  # noqa: E402


DEFAULT_DATA = ROOT / "ultralytics" / "cfg" / "datasets" / "OSI26.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "evaluation")
    parser.add_argument("--name", type=str, default="test_best")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    branch_weights = {}
    for module_name, module in model.model.named_modules():
        if isinstance(module, (WFCAParallel, ParallelFusion)):
            baseline, innovation = module.branch_weights()
            branch_weights[module_name] = {
                "module": type(module).__name__,
                "baseline": baseline,
                "innovation": innovation,
                "adapter_gain": module.adapter_gain(),
            }
    summary = {
        "weights": str(args.weights.resolve()),
        "split": args.split,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
        "branch_weights": branch_weights,
    }
    output = Path(metrics.save_dir) / "metrics_summary.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("METRICS_JSON=" + json.dumps(summary, ensure_ascii=False))
    print(f"METRICS_FILE={output}")


if __name__ == "__main__":
    main()
