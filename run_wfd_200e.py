"""Run the complete parallel-gated WFD-YOLO OSI26 200-epoch experiment."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ultralytics" / "cfg" / "datasets" / "OSI26.yaml"
MODEL = ROOT / "ultralytics" / "cfg" / "models" / "v12" / "wfd-yolov12n.yaml"
PRETRAINED = ROOT / "yolov12n.pt"
RUN_NAME = "wfd_yolov12n_parallel_pretrained_osi26_200e"
TRAIN_PROJECT = ROOT / "runs" / "wfd-yolo"
TRAIN_DIR = TRAIN_PROJECT / RUN_NAME
EVAL_PROJECT = ROOT / "runs" / "evaluation"
DATASET = ROOT / "data/OSI26_Dataset"
EXPECTED_IMAGES = {"train": 380, "val": 41, "test": 31}
EXPECTED_BOXES = {"train": 1404, "val": 139, "test": 173}


def run(*arguments: object) -> None:
    command = [sys.executable, "-u", *map(str, arguments)]
    print("RUNNING:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def completed_epochs() -> int:
    results = TRAIN_DIR / "results.csv"
    if not results.is_file():
        return 0
    with results.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def verify_dataset() -> dict:
    class_counts: Counter[int] = Counter()
    payload = {"splits": {}, "total_images": 0, "total_boxes": 0}
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    for split in ("train", "val", "test"):
        image_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split
        images = [path for path in image_dir.iterdir() if path.suffix.lower() in image_suffixes]
        labels = list(label_dir.glob("*.txt"))
        boxes = 0
        for label in labels:
            for line in label.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                values = line.split()
                if len(values) != 5:
                    raise ValueError(f"Invalid YOLO label row in {label}: {line}")
                class_id = int(values[0])
                if not 0 <= class_id < 26:
                    raise ValueError(f"Invalid class {class_id} in {label}")
                class_counts[class_id] += 1
                boxes += 1
        actual = {"images": len(images), "labels": len(labels), "boxes": boxes}
        expected = {"images": EXPECTED_IMAGES[split], "labels": EXPECTED_IMAGES[split], "boxes": EXPECTED_BOXES[split]}
        if actual != expected:
            raise RuntimeError(f"Dataset mismatch for {split}: expected={expected}, actual={actual}")
        payload["splits"][split] = actual
        payload["total_images"] += len(images)
        payload["total_boxes"] += boxes
    if set(class_counts) != set(range(26)):
        raise RuntimeError(f"Expected classes 0..25, found {sorted(class_counts)}")
    payload["class_counts"] = {str(key): class_counts[key] for key in range(26)}
    return payload


def train() -> None:
    epochs = completed_epochs()
    best = TRAIN_DIR / "weights" / "best.pt"
    last = TRAIN_DIR / "weights" / "last.pt"
    if epochs >= 200 and best.is_file():
        print(f"SKIP TRAIN: completed epochs={epochs}", flush=True)
        return
    if epochs and last.is_file():
        run("train_wfd_yolo.py", "--resume", last)
        return
    if not PRETRAINED.is_file():
        raise FileNotFoundError(f"Missing official YOLOv12n checkpoint: {PRETRAINED}")
    run(
        "train_wfd_yolo.py",
        "--model",
        MODEL,
        "--data",
        DATA,
        "--weights",
        PRETRAINED,
        "--epochs",
        200,
        "--batch",
        16,
        "--imgsz",
        640,
        "--workers",
        1,
        "--device",
        0,
        "--patience",
        200,
        "--project",
        TRAIN_PROJECT,
        "--name",
        RUN_NAME,
    )


def evaluate(split: str) -> Path:
    output_name = f"{RUN_NAME}_best_{split}"
    run(
        "evaluate_osi26.py",
        "--weights",
        TRAIN_DIR / "weights" / "best.pt",
        "--data",
        DATA,
        "--split",
        split,
        "--imgsz",
        640,
        "--batch",
        16,
        "--workers",
        1,
        "--device",
        0,
        "--project",
        EVAL_PROJECT,
        "--name",
        output_name,
    )
    return EVAL_PROJECT / output_name / "metrics_summary.json"


def summarize(dataset: dict, metric_paths: dict[str, Path]) -> None:
    metrics = {split: json.loads(path.read_text(encoding="utf-8")) for split, path in metric_paths.items()}
    summary = {
        "model": "WFD-YOLOv12n zero-initialized parallel gated (WFCA || baseline, SFPA+DGFA || Concat)",
        "loss": "YOLOv12n native detection loss (BCE + CIoU/DFL); ATFL disabled",
        "epochs": completed_epochs(),
        "dataset": dataset,
        "best_weights": str(TRAIN_DIR / "weights" / "best.pt"),
        "last_weights": str(TRAIN_DIR / "weights" / "last.pt"),
        "metrics": metrics,
    }
    summary_path = ROOT / "WFD_YOLO_OSI26_200e_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    val, test = metrics["val"], metrics["test"]
    weights = val["branch_weights"]
    weight_lines = "\n".join(
        f"- `{name}` ({values['module']}): baseline {values['baseline']:.6f}, "
        f"innovation {values['innovation']:.6f}, adapter gain {values['adapter_gain']:.6f}"
        for name, values in weights.items()
    )
    report = (
        "# Parallel-gated WFD-YOLOv12n on OSI26 (200 epochs)\n\n"
        f"- Dataset: 380 train / 41 val / 31 test images; {dataset['total_boxes']} boxes; 26 classes\n"
        "- Architecture: original YOLOv12n computation retained at every modified node; "
        "WFCA and SFPA+DGFA are learnable parallel alternatives\n"
        "- Loss: native YOLOv12n BCE classification + CIoU/DFL regression (ATFL disabled)\n"
        "- Training: 640 px, batch 16, 200 epochs, official YOLOv12n transfer weights\n"
        f"- Validation: P {val['precision']:.6f}, R {val['recall']:.6f}, F1 {val['f1']:.6f}, "
        f"mAP50 {val['mAP50']:.6f}, mAP50-95 {val['mAP50-95']:.6f}\n"
        f"- Test: P {test['precision']:.6f}, R {test['recall']:.6f}, F1 {test['f1']:.6f}, "
        f"mAP50 {test['mAP50']:.6f}, mAP50-95 {test['mAP50-95']:.6f}\n\n"
        "## Learned branch weights\n\n"
        f"{weight_lines}\n"
    )
    (ROOT / "WFD_YOLO_OSI26_200e_report.md").write_text(report, encoding="utf-8")
    print(f"SUMMARY={summary_path}", flush=True)


def main() -> None:
    dataset = verify_dataset()
    print("DATASET=" + json.dumps(dataset, ensure_ascii=False), flush=True)
    train()
    paths = {split: evaluate(split) for split in ("val", "test")}
    summarize(dataset, paths)


if __name__ == "__main__":
    main()

