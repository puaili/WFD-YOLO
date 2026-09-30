"""Run reviewer-requested analyses for the latest WFD-YOLO checkpoint.

Outputs a reproducible timing protocol, per-class metrics, confusion matrices,
FP/FN case visualizations, and an exact layer/shape map for the revised paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo_config"))
os.environ.setdefault("YOLO_DATASET_CACHE_DIR", str(ROOT / ".dataset_cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


DEFAULT_PROJECT = ROOT
DEFAULT_DATASET = ROOT / "data/OSI26_Dataset"
DEFAULT_WEIGHTS = (
    ROOT
    / "runs/wfd-yolo/wfd_yolov12n_parallel_pretrained_osi26_200e/weights/best.pt"
)
DEFAULT_DATA = DEFAULT_PROJECT / "ultralytics/cfg/datasets/OSI26.yaml"
DEFAULT_ARGS = (
    ROOT
    / "runs/wfd-yolo/wfd_yolov12n_parallel_pretrained_osi26_200e/args.yaml"
)
DEFAULT_LOG = DEFAULT_PROJECT / "training_logs/wfd_parallel_pretrained_200e.out.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--train-args", type=Path, default=DEFAULT_ARGS)
    parser.add_argument("--train-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output", type=Path, default=ROOT / "reviewer_response_20260810")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--core-repeats", type=int, default=500)
    parser.add_argument("--pipeline-repeats", type=int, default=200)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_samples(samples: list[float]) -> dict[str, float | int]:
    array = np.asarray(samples, dtype=np.float64)
    mean = float(array.mean())
    return {
        "repeats": int(array.size),
        "mean_ms": mean,
        "std_ms": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "fps_from_mean": 1000.0 / mean,
    }


def load_test_images(dataset: Path) -> list[tuple[Path, np.ndarray]]:
    image_dir = dataset / "images/test"
    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    images = [(path, cv2.imread(str(path))) for path in paths]
    if len(images) != 31 or any(image is None for _, image in images):
        raise RuntimeError(f"Expected 31 readable test images, received {len(images)}")
    return images


@torch.inference_mode()
def benchmark_core(
    weights: Path,
    device: str,
    imgsz: int,
    half: bool,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    wrapper = YOLO(str(weights))
    network = wrapper.model.to(f"cuda:{device}").eval()
    dtype = torch.float16 if half else torch.float32
    network = network.half() if half else network.float()
    tensor = torch.rand((1, 3, imgsz, imgsz), device=f"cuda:{device}", dtype=dtype)
    for _ in range(warmup):
        network(tensor)
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        network(tensor)
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    result = summarize_samples(samples)
    result.update(
        {
            "precision": "FP16" if half else "FP32",
            "batch": 1,
            "warmup": warmup,
            "scope": "network forward only; excludes preprocessing, NMS/postprocessing and disk I/O",
        }
    )
    del network, wrapper, tensor
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def benchmark_pipeline(
    weights: Path,
    images: list[tuple[Path, np.ndarray]],
    device: str,
    imgsz: int,
    half: bool,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    model = YOLO(str(weights))
    arrays = [image for _, image in images]
    kwargs = {
        "imgsz": imgsz,
        "device": device,
        "batch": 1,
        "half": half,
        "conf": 0.25,
        "iou": 0.7,
        "max_det": 300,
        "verbose": False,
    }
    for index in range(warmup):
        model.predict(source=arrays[index % len(arrays)], **kwargs)
    torch.cuda.synchronize()
    wall_samples: list[float] = []
    preprocess: list[float] = []
    inference: list[float] = []
    postprocess: list[float] = []
    for index in range(repeats):
        image = arrays[index % len(arrays)]
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = model.predict(source=image, **kwargs)[0]
        torch.cuda.synchronize()
        wall_samples.append((time.perf_counter() - start) * 1000.0)
        preprocess.append(float(result.speed["preprocess"]))
        inference.append(float(result.speed["inference"]))
        postprocess.append(float(result.speed["postprocess"]))
    summary = summarize_samples(wall_samples)
    summary.update(
        {
            "precision": "FP16" if half else "FP32",
            "batch": 1,
            "warmup": warmup,
            "scope": (
                "decoded in-memory BGR image to post-NMS Results; includes letterbox/tensor preprocessing, "
                "network inference, NMS/postprocessing and Python call overhead; excludes disk I/O"
            ),
            "component_mean_ms": {
                "preprocess": statistics.fmean(preprocess),
                "inference": statistics.fmean(inference),
                "postprocess_including_nms": statistics.fmean(postprocess),
            },
        }
    )
    del model
    torch.cuda.empty_cache()
    return summary


def extract_hyperparameters(train_args: Path, train_log: Path) -> dict[str, Any]:
    args = yaml.safe_load(train_args.read_text(encoding="utf-8"))
    log = train_log.read_text(encoding="utf-8", errors="replace") if train_log.is_file() else ""
    match = re.search(r"AdamW\(lr=([0-9.eE+-]+), momentum=([0-9.eE+-]+)\)", log)
    actual_optimizer = {
        "name": "AdamW",
        "initial_lr": float(match.group(1)) if match else 0.000333,
        "momentum_beta1": float(match.group(2)) if match else 0.9,
        "weight_decay": float(args["weight_decay"]),
        "selection": "optimizer=auto; lr0=0.01 and momentum=0.937 were ignored by Ultralytics auto-selection",
    }
    selected = {
        "epochs": args["epochs"],
        "batch": args["batch"],
        "imgsz": args["imgsz"],
        "optimizer": actual_optimizer,
        "scheduler": {
            "type": "linear",
            "cos_lr": bool(args["cos_lr"]),
            "final_lr_fraction_lrf": args["lrf"],
        },
        "warmup": {
            "epochs": args["warmup_epochs"],
            "momentum": args["warmup_momentum"],
            "bias_lr": args["warmup_bias_lr"],
        },
        "loss_weights": {"box": args["box"], "cls": args["cls"], "dfl": args["dfl"]},
        "loss": "native YOLOv12 detection loss: BCE classification + CIoU box loss + DFL; ATFL disabled",
        "ATFL_tau": None,
        "ATFL_tau_note": "Not applicable in the latest revised model because ATFL is disabled.",
        "SFPA": {"lambda_residual_scale_initial": 0.1, "sigma_gaussian": 1.0, "threshold": 0.5},
        "parallel_gate": {"baseline_initial": 0.95, "innovation_initial": 0.05},
        "amp_training": bool(args["amp"]),
        "seed": args["seed"],
        "deterministic": bool(args["deterministic"]),
        "augmentation": {
            key: args[key]
            for key in (
                "hsv_h",
                "hsv_s",
                "hsv_v",
                "degrees",
                "translate",
                "scale",
                "shear",
                "perspective",
                "flipud",
                "fliplr",
                "mosaic",
                "mixup",
                "copy_paste",
                "copy_paste_mode",
                "auto_augment",
                "erasing",
                "close_mosaic",
            )
        },
    }
    return selected


def yolo_labels(label_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    classes: list[int] = []
    boxes: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = map(float, line.split())
        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height
        classes.append(int(cls))
        boxes.append([x1, y1, x2, y2])
    return np.asarray(classes, dtype=np.int64), np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0, None)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-9)


def greedy_match(pred_boxes: np.ndarray, gt_boxes: np.ndarray, threshold: float = 0.5) -> list[tuple[int, int, float]]:
    ious = pairwise_iou(pred_boxes, gt_boxes)
    candidates = [
        (pred_index, gt_index, float(ious[pred_index, gt_index]))
        for pred_index, gt_index in zip(*np.where(ious >= threshold))
    ]
    candidates.sort(key=lambda item: item[2], reverse=True)
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for pred_index, gt_index, iou in candidates:
        if pred_index not in used_pred and gt_index not in used_gt:
            used_pred.add(pred_index)
            used_gt.add(gt_index)
            matches.append((pred_index, gt_index, iou))
    return matches


def draw_box(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int], label: str, thickness: int = 2) -> None:
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    font_scale = max(0.45, min(image.shape[:2]) / 900)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    y_text = max(th + 3, y1)
    cv2.rectangle(image, (x1, y_text - th - 3), (x1 + tw + 4, y_text + 2), color, -1)
    cv2.putText(image, label, (x1 + 2, y_text - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def save_confusion_plot(matrix: np.ndarray, labels: list[str], path: Path, normalized: bool) -> None:
    values = matrix.astype(np.float64)
    title = "WFD-YOLO test confusion matrix (counts)"
    if normalized:
        values = values / np.maximum(values.sum(axis=1, keepdims=True), 1.0)
        title = "WFD-YOLO test confusion matrix (row-normalized)"
    fig, ax = plt.subplots(figsize=(14, 12), constrained_layout=True)
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=None)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    if not normalized:
        maximum = max(float(values.max()), 1.0)
        for row, column in zip(*np.where(values > 0)):
            ax.text(
                column,
                row,
                str(int(values[row, column])),
                ha="center",
                va="center",
                fontsize=6,
                color="white" if values[row, column] > maximum * 0.55 else "black",
            )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def per_class_and_errors(
    weights: Path,
    data: Path,
    dataset: Path,
    output: Path,
    images: list[tuple[Path, np.ndarray]],
    device: str,
    imgsz: int,
) -> dict[str, Any]:
    eval_dir = output / "evaluation"
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split="test",
        imgsz=imgsz,
        batch=16,
        workers=1,
        device=device,
        conf=0.001,
        iou=0.7,
        max_det=300,
        plots=True,
        verbose=False,
        project=str(eval_dir),
        name="wfd_latest_test_reviewer",
        exist_ok=True,
    )
    class_indices = [int(value) for value in metrics.box.ap_class_index]
    metric_by_class = {
        class_id: {
            "precision": float(metrics.box.p[index]),
            "recall": float(metrics.box.r[index]),
            "AP50": float(metrics.box.ap50[index]),
            "AP50_95": float(metrics.box.ap[index]),
        }
        for index, class_id in enumerate(class_indices)
    }
    predictions = model.predict(
        source=[str(path) for path, _ in images],
        imgsz=imgsz,
        batch=1,
        device=device,
        half=False,
        conf=0.25,
        iou=0.7,
        max_det=300,
        verbose=False,
    )
    confusion = np.zeros((27, 27), dtype=np.int64)
    instance_counts: Counter[int] = Counter()
    tp_counts: Counter[int] = Counter()
    fp_counts: Counter[int] = Counter()
    fn_counts: Counter[int] = Counter()
    cases: list[dict[str, Any]] = []
    case_dir = output / "error_cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    for (path, original), result in zip(images, predictions):
        height, width = original.shape[:2]
        label_path = dataset / "labels/test" / f"{path.stem}.txt"
        gt_cls, gt_boxes = yolo_labels(label_path, width, height)
        pred_cls = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
        pred_boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        pred_conf = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        matches = greedy_match(pred_boxes, gt_boxes, 0.5)
        matched_pred = {pred_index for pred_index, _, _ in matches}
        matched_gt = {gt_index for _, gt_index, _ in matches}
        tp_pred: set[int] = set()
        mismatch_pred: set[int] = set()
        mismatch_gt: set[int] = set()
        for class_id in gt_cls:
            instance_counts[int(class_id)] += 1
        for pred_index, gt_index, _ in matches:
            true_class = int(gt_cls[gt_index])
            predicted_class = int(pred_cls[pred_index])
            confusion[true_class, predicted_class] += 1
            if true_class == predicted_class:
                tp_counts[true_class] += 1
                tp_pred.add(pred_index)
            else:
                fp_counts[predicted_class] += 1
                fn_counts[true_class] += 1
                mismatch_pred.add(pred_index)
                mismatch_gt.add(gt_index)
        for gt_index in set(range(len(gt_cls))) - matched_gt:
            true_class = int(gt_cls[gt_index])
            confusion[true_class, 26] += 1
            fn_counts[true_class] += 1
        for pred_index in set(range(len(pred_cls))) - matched_pred:
            predicted_class = int(pred_cls[pred_index])
            confusion[26, predicted_class] += 1
            fp_counts[predicted_class] += 1
        fp_total = len(set(range(len(pred_cls))) - tp_pred)
        fn_total = len(set(range(len(gt_cls))) - {g for p, g, _ in matches if int(pred_cls[p]) == int(gt_cls[g])})
        cases.append(
            {
                "image": path.name,
                "instances": int(len(gt_cls)),
                "predictions": int(len(pred_cls)),
                "TP": int(len(tp_pred)),
                "FP": int(fp_total),
                "FN": int(fn_total),
                "error_score": int(fp_total + fn_total),
            }
        )
        annotated = original.copy()
        correct_gt = {g for p, g, _ in matches if int(pred_cls[p]) == int(gt_cls[g])}
        for gt_index, (class_id, box) in enumerate(zip(gt_cls, gt_boxes)):
            if gt_index in mismatch_gt:
                draw_box(annotated, box, (0, 140, 255), f"FN/mis GT {class_id + 1}", 3)
            elif gt_index not in correct_gt:
                draw_box(annotated, box, (0, 140, 255), f"FN GT {class_id + 1}", 3)
            else:
                draw_box(annotated, box, (0, 180, 0), f"GT {class_id + 1}", 1)
        for pred_index, (class_id, box, confidence) in enumerate(zip(pred_cls, pred_boxes, pred_conf)):
            if pred_index in tp_pred:
                draw_box(annotated, box, (255, 100, 0), f"TP {class_id + 1} {confidence:.2f}", 2)
            else:
                draw_box(annotated, box, (0, 0, 220), f"FP P{class_id + 1} {confidence:.2f}", 3)
        cv2.imwrite(str(case_dir / f"{path.stem}_errors.jpg"), annotated)
    class_rows = []
    for class_id in range(26):
        row = {
            "class_id": class_id + 1,
            "instances": instance_counts[class_id],
            **metric_by_class.get(class_id, {"precision": 0.0, "recall": 0.0, "AP50": 0.0, "AP50_95": 0.0}),
            "TP_at_conf0.25_iou0.5": tp_counts[class_id],
            "FP_at_conf0.25_iou0.5": fp_counts[class_id],
            "FN_at_conf0.25_iou0.5": fn_counts[class_id],
        }
        class_rows.append(row)
    with (output / "per_class_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)
    cases.sort(key=lambda item: (item["error_score"], item["FN"], item["FP"]), reverse=True)
    with (output / "image_error_counts.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    labels = [str(index) for index in range(1, 27)] + ["BG"]
    np.savetxt(output / "confusion_matrix_counts.csv", confusion, fmt="%d", delimiter=",")
    save_confusion_plot(confusion, labels, output / "confusion_matrix_counts.png", normalized=False)
    save_confusion_plot(confusion, labels, output / "confusion_matrix_normalized.png", normalized=True)
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    x = np.arange(1, 27)
    ax.bar(x - 0.18, [row["AP50"] for row in class_rows], width=0.36, label="AP@50", color="#2E75B6")
    ax.bar(x + 0.18, [row["AP50_95"] for row in class_rows], width=0.36, label="AP@50:95", color="#70AD47")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Class identifier")
    ax.set_ylabel("Average precision")
    ax.set_title("WFD-YOLO per-class performance on OSI26 test set")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output / "per_class_ap.png", dpi=220)
    plt.close(fig)
    selected = cases[:4]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, case in zip(axes.flat, selected):
        image = cv2.imread(str(case_dir / f"{Path(case['image']).stem}_errors.jpg"))
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(f"{case['image']} | TP={case['TP']} FP={case['FP']} FN={case['FN']}", fontsize=10)
        axis.axis("off")
    fig.suptitle("Representative FP/FN cases (conf=0.25, matching IoU=0.50)", fontsize=14)
    fig.savefig(output / "fp_fn_examples.png", dpi=220)
    plt.close(fig)
    off_diagonal = []
    for true_class in range(26):
        for predicted_class in range(26):
            if true_class != predicted_class and confusion[true_class, predicted_class] > 0:
                off_diagonal.append(
                    {
                        "true_class": true_class + 1,
                        "predicted_class": predicted_class + 1,
                        "count": int(confusion[true_class, predicted_class]),
                    }
                )
    off_diagonal.sort(key=lambda item: item["count"], reverse=True)
    result_summary = {
        "global": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
        },
        "error_analysis_protocol": {"confidence": 0.25, "matching_iou": 0.5, "nms_iou": 0.7},
        "per_class": class_rows,
        "top_confusions": off_diagonal[:10],
        "representative_cases": selected,
        "total_fixed_threshold": {
            "TP": int(sum(tp_counts.values())),
            "FP": int(sum(fp_counts.values())),
            "FN": int(sum(fn_counts.values())),
        },
    }
    write_json(output / "class_error_analysis.json", result_summary)
    return result_summary


def shape_text(value: Any) -> str:
    if torch.is_tensor(value):
        return "x".join(str(number) for number in value.shape)
    if isinstance(value, (list, tuple)):
        return "; ".join(shape_text(item) for item in value if torch.is_tensor(item) or isinstance(item, (list, tuple)))
    return type(value).__name__


@torch.inference_mode()
def extract_layer_map(weights: Path, device: str, imgsz: int, output: Path) -> list[dict[str, Any]]:
    wrapper = YOLO(str(weights))
    network = wrapper.model.to(f"cuda:{device}").float().eval()
    captured: dict[int, dict[str, str]] = {}
    handles = []
    for index, module in enumerate(network.model):
        def pre_hook(_module: torch.nn.Module, inputs: tuple[Any, ...], layer_index: int = index) -> None:
            # Detect mutates its input list in place, so capture shapes before forward.
            captured.setdefault(layer_index, {})["input_shape"] = shape_text(inputs)

        def post_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], result: Any, layer_index: int = index) -> None:
            captured.setdefault(layer_index, {})["output_shape"] = shape_text(result)

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))
    network(torch.zeros((1, 3, imgsz, imgsz), device=f"cuda:{device}"))
    for handle in handles:
        handle.remove()
    declared = {2: 2, 4: 2, 7: 4, 10: 4, 13: 2, 16: 2, 19: 2, 22: 2}
    effective = {2: 1, 4: 1, 7: 2, 10: 2, 13: 1, 16: 1, 19: 1, 22: 1}
    rows = []
    for index, module in enumerate(network.model):
        module_name = getattr(module, "type", type(module).__name__).replace("ultralytics.nn.modules.", "")
        shapes = captured.get(index, {})
        rows.append(
            {
                "layer": index,
                "from": str(getattr(module, "f", -1)),
                "module": module_name,
                "input_shape_NCHW": shapes.get("input_shape", ""),
                "output_shape_NCHW": shapes.get("output_shape", ""),
                "yaml_repeats": declared.get(index, 1),
                "effective_repeats_n_scale": effective.get(index, 1),
                "parameters": sum(parameter.numel() for parameter in module.parameters()),
            }
        )
    with (output / "model_layer_map.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "model_layer_map.json", rows)
    del network, wrapper
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for required in (args.weights, args.data, args.dataset, args.train_args):
        if not required.exists():
            raise FileNotFoundError(required)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = True
    images = load_test_images(args.dataset)
    hyperparameters = extract_hyperparameters(args.train_args, args.train_log)
    write_json(args.output / "hyperparameters.json", hyperparameters)
    timing = {
        "hardware": {
            "GPU": torch.cuda.get_device_name(int(args.device)),
            "torch": torch.__version__,
            "CUDA_runtime": torch.version.cuda,
            "image_size": args.imgsz,
            "batch": 1,
        },
        "core_forward": [],
        "end_to_end": [],
    }
    for half in (False, True):
        timing["core_forward"].append(
            benchmark_core(args.weights, args.device, args.imgsz, half, args.warmup, args.core_repeats)
        )
        timing["end_to_end"].append(
            benchmark_pipeline(
                args.weights,
                images,
                args.device,
                args.imgsz,
                half,
                args.warmup,
                args.pipeline_repeats,
            )
        )
    write_json(args.output / "timing_protocol_results.json", timing)
    class_analysis = per_class_and_errors(
        args.weights, args.data, args.dataset, args.output, images, args.device, args.imgsz
    )
    layer_map = extract_layer_map(args.weights, args.device, args.imgsz, args.output)
    manifest = {
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "dataset": str(args.dataset.resolve()),
        "hyperparameters": hyperparameters,
        "timing": timing,
        "global_test_metrics": class_analysis["global"],
        "layer_count": len(layer_map),
        "artifacts": sorted(str(path.relative_to(args.output)) for path in args.output.rglob("*") if path.is_file()),
    }
    write_json(args.output / "reviewer_experiment_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
