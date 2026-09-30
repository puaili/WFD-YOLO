"""Train and evaluate the final no-ATFL protected-parallel ablation matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo_config"))
os.environ.setdefault("YOLO_DATASET_CACHE_DIR", str(ROOT / ".dataset_cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


DATA = ROOT / "ultralytics/cfg/datasets/OSI26.yaml"
PRETRAINED = ROOT / "yolov12n.pt"
FULL_BEST = ROOT / "runs/wfd-yolo/wfd_yolov12n_parallel_pretrained_osi26_200e/weights/best.pt"
DEFAULT_PROJECT = ROOT / "runs/reviewer_ablation"
DEFAULT_OUTPUT = ROOT / "runs/reviewer_ablation"

VARIANTS = {
    "baseline": None,
    "wfca_only": ROOT / "ultralytics/cfg/models/v12/wfd-yolov12n-wfca-only.yaml",
    "fusion_only": ROOT / "ultralytics/cfg/models/v12/wfd-yolov12n-fusion-only.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker", choices=[*VARIANTS], default=None)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def completed_epochs(run_dir: Path) -> int:
    path = run_dir / "results.csv"
    if not path.exists():
        return 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return len(list(csv.DictReader(stream)))


def train_worker(variant: str, project: Path, device: str) -> None:
    run_name = f"wfd_ablation_{variant}_200e"
    run_dir = project / run_name
    if completed_epochs(run_dir) >= 200 and (run_dir / "weights/best.pt").exists():
        print(f"SKIP {variant}: already complete", flush=True)
        return
    if variant == "baseline":
        model = YOLO(str(PRETRAINED))
        model.train(
            data=str(DATA),
            epochs=200,
            batch=16,
            imgsz=640,
            workers=1,
            device=device,
            project=str(project),
            name=run_name,
            patience=200,
            amp=True,
            seed=0,
            deterministic=True,
            optimizer="auto",
            val=True,
            plots=True,
            verbose=True,
            exist_ok=True,
        )
        return
    command = [
        sys.executable,
        "-u",
        str(ROOT / "train_wfd_yolo.py"),
        "--model",
        str(VARIANTS[variant]),
        "--data",
        str(DATA),
        "--weights",
        str(PRETRAINED),
        "--epochs",
        "200",
        "--batch",
        "16",
        "--imgsz",
        "640",
        "--workers",
        "1",
        "--device",
        device,
        "--patience",
        "200",
        "--project",
        str(project),
        "--name",
        run_name,
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def metrics_payload(metrics: Any) -> dict[str, float]:
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }


def evaluate(project: Path, output: Path, device: str) -> list[dict[str, Any]]:
    weights = {
        name: project / f"wfd_ablation_{name}_200e/weights/best.pt" for name in VARIANTS
    }
    weights["full_protected_parallel"] = FULL_BEST
    rows: list[dict[str, Any]] = []
    for name, checkpoint in weights.items():
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = YOLO(str(checkpoint))
        base = {
            "variant": name,
            "WFCA_parallel": name in {"wfca_only", "full_protected_parallel"},
            "SFPA_DGFA_parallel": name in {"fusion_only", "full_protected_parallel"},
            "protected_gate": name != "baseline",
            "ATFL": False,
            "parameters_M": sum(parameter.numel() for parameter in model.model.parameters()) / 1e6,
            "checkpoint_MiB": checkpoint.stat().st_size / 2**20,
            "best_epoch": None,
        }
        results_path = checkpoint.parents[1] / "results.csv"
        if results_path.exists():
            with results_path.open(encoding="utf-8-sig", newline="") as stream:
                history = list(csv.DictReader(stream))
            if history:
                best_row = max(history, key=lambda row: float(row["metrics/mAP50-95(B)"]))
                base["best_epoch"] = int(float(best_row["epoch"]))
        for split in ("val", "test"):
            metrics = model.val(
                data=str(DATA),
                split=split,
                imgsz=640,
                batch=1,
                device=device,
                workers=0,
                plots=False,
                verbose=False,
                project=str(output / "evaluation"),
                name=f"{name}_{split}",
                exist_ok=True,
            )
            for key, value in metrics_payload(metrics).items():
                base[f"{split}_{key}"] = value
        rows.append(base)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "protected_parallel_ablation.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "protected_parallel_ablation.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "epochs": 200,
                    "batch": 16,
                    "imgsz": 640,
                    "optimizer": "Ultralytics auto (AdamW in completed full run)",
                    "seed": 0,
                    "loss": "native YOLOv12 BCE + CIoU/DFL; ATFL disabled in every variant",
                    "pretraining": "same official YOLOv12n checkpoint and shape-aware transfer",
                },
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows


def main() -> None:
    args = parse_args()
    if args.worker:
        train_worker(args.worker, args.project, args.device)
        return
    args.project.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    log_dir = args.output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        run_dir = args.project / f"wfd_ablation_{variant}_200e"
        if completed_epochs(run_dir) >= 200 and (run_dir / "weights/best.pt").exists():
            continue
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            variant,
            "--project",
            str(args.project),
            "--output",
            str(args.output),
            "--device",
            args.device,
        ]
        with (log_dir / f"{variant}.log").open("w", encoding="utf-8") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    rows = evaluate(args.project, args.output, args.device)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
