"""Additional reviewer experiments for WFD-YOLO latest.

This script adds five reproducible analyses requested during revision:

1. Synthetic corruption robustness on the unchanged OSI26 test split.
2. One-factor-at-a-time inference sensitivity for SFPA lambda/sigma/tau and
   the protected parallel gate.
3. Training/validation curves and the validation-selected best epoch.
4. Checkpoint/model size accounting.
5. Peak CUDA memory at batch=1 for FP32 and FP16 inference.

The corruption suite is a deterministic stress test, not a substitute for a
prospectively collected clinical corruption dataset.  The sensitivity sweep is
post-training and changes one inference-time control at a time; it must not be
described as independent retraining of every parameter setting.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

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
from ultralytics.nn.modules.wfd import SFPA, _AdaptiveBranchMixer  # noqa: E402


DEFAULT_PROJECT = ROOT
DEFAULT_DATASET = ROOT / "data/OSI26_Dataset"
DEFAULT_WEIGHTS = (
    ROOT
    / "runs/wfd-yolo/wfd_yolov12n_parallel_pretrained_osi26_200e/weights/best.pt"
)
DEFAULT_DATA = DEFAULT_PROJECT / "ultralytics/cfg/datasets/OSI26.yaml"
DEFAULT_RESULTS = (
    DEFAULT_PROJECT / "runs/wfd-yolo/wfd_yolov12n_parallel_pretrained_osi26_200e/results.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=ROOT / "reviewer_additional_20260811")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--memory-warmup", type=int, default=50)
    parser.add_argument("--memory-repeats", type=int, default=100)
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_payload(metrics: Any) -> dict[str, float]:
    box = metrics.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
    }


def validate(
    model: YOLO,
    data: Path,
    split: str,
    output: Path,
    name: str,
    device: str,
    imgsz: int,
) -> dict[str, float]:
    result = model.val(
        data=str(data),
        split=split,
        imgsz=imgsz,
        batch=1,
        device=device,
        workers=0,
        plots=False,
        save_json=False,
        verbose=False,
        project=str(output / "evaluation_runs"),
        name=name,
        exist_ok=True,
    )
    return metric_payload(result)


def deterministic_rng(*parts: str) -> np.random.Generator:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def corruption_brightness(image: np.ndarray, severity: int, _: np.random.Generator) -> np.ndarray:
    factor = {1: 0.80, 2: 0.60, 3: 0.40}[severity]
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def corruption_contrast(image: np.ndarray, severity: int, _: np.random.Generator) -> np.ndarray:
    factor = {1: 0.80, 2: 0.60, 3: 0.40}[severity]
    array = image.astype(np.float32)
    mean = array.mean(axis=(0, 1), keepdims=True)
    return np.clip((array - mean) * factor + mean, 0, 255).astype(np.uint8)


def corruption_blur(image: np.ndarray, severity: int, _: np.random.Generator) -> np.ndarray:
    kernel, sigma = {1: (3, 0.8), 2: (7, 1.8), 3: (11, 3.0)}[severity]
    return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)


def corruption_smoke(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    alpha = {1: 0.15, 2: 0.30, 3: 0.45}[severity]
    h, w = image.shape[:2]
    noise = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    sigma = max(h, w) / 18.0
    field = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma)
    field -= field.min()
    field /= max(float(field.max()), 1e-6)
    field = 0.35 + 0.65 * field
    opacity = (alpha * field)[..., None]
    out = image.astype(np.float32) * (1.0 - opacity) + 235.0 * opacity
    return np.clip(out, 0, 255).astype(np.uint8)


def corruption_specular(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    count = {1: 2, 2: 4, 3: 6}[severity]
    peak = {1: 0.40, 2: 0.60, 3: 0.80}[severity]
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(count):
        center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        axes = (
            max(4, int(rng.uniform(0.025, 0.075) * w)),
            max(3, int(rng.uniform(0.012, 0.045) * h)),
        )
        angle = float(rng.uniform(0, 180))
        cv2.ellipse(mask, center, axes, angle, 0, 360, 1.0, thickness=-1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(h, w) / 180.0)
    opacity = np.clip(mask * peak, 0.0, peak)[..., None]
    out = image.astype(np.float32) * (1.0 - opacity) + 255.0 * opacity
    return np.clip(out, 0, 255).astype(np.uint8)


def corruption_occlusion(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    target_fraction = {1: 0.05, 2: 0.10, 3: 0.20}[severity]
    h, w = image.shape[:2]
    out = image.copy()
    covered = 0
    target = int(h * w * target_fraction)
    fill = tuple(int(value) for value in np.median(image.reshape(-1, 3), axis=0))
    while covered < target:
        rw = int(rng.uniform(0.08, 0.22) * w)
        rh = int(rng.uniform(0.06, 0.18) * h)
        x1 = int(rng.integers(0, max(1, w - rw)))
        y1 = int(rng.integers(0, max(1, h - rh)))
        cv2.rectangle(out, (x1, y1), (x1 + rw, y1 + rh), fill, thickness=-1)
        covered += rw * rh
    return out


def corruption_clutter(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    count = {1: 8, 2: 16, 3: 32}[severity]
    alpha = {1: 0.20, 2: 0.30, 3: 0.40}[severity]
    overlay = image.copy()
    for _ in range(count):
        color = tuple(int(value) for value in rng.integers(20, 236, size=3))
        if rng.random() < 0.5:
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.line(overlay, p1, p2, color, thickness=int(rng.integers(2, 8)))
        else:
            center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            radius = int(rng.uniform(0.01, 0.04) * min(h, w))
            cv2.circle(overlay, center, max(radius, 2), color, thickness=-1)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)


CORRUPTIONS: dict[str, Callable[[np.ndarray, int, np.random.Generator], np.ndarray]] = {
    "brightness": corruption_brightness,
    "contrast": corruption_contrast,
    "blur": corruption_blur,
    "smoke": corruption_smoke,
    "specular": corruption_specular,
    "occlusion": corruption_occlusion,
    "clutter": corruption_clutter,
}


def build_corruption_dataset(
    dataset: Path,
    output: Path,
    corruption: str,
    severity: int,
) -> tuple[Path, list[tuple[Path, Path]]]:
    root = output / "corruption_datasets" / f"{corruption}_s{severity}"
    image_output = root / "images" / "test"
    label_output = root / "labels" / "test"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[Path, Path]] = []
    operation = CORRUPTIONS[corruption]
    for source in sorted((dataset / "images" / "test").iterdir()):
        if source.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        image = cv2.imread(str(source))
        if image is None:
            raise RuntimeError(f"Unable to read {source}")
        rng = deterministic_rng(corruption, str(severity), source.name)
        changed = operation(image, severity, rng)
        destination = image_output / source.name
        if not cv2.imwrite(str(destination), changed):
            raise RuntimeError(f"Unable to write {destination}")
        label = dataset / "labels" / "test" / f"{source.stem}.txt"
        shutil.copy2(label, label_output / label.name)
        pairs.append((source, destination))
    if len(pairs) != 31:
        raise RuntimeError(f"Expected 31 test images, built {len(pairs)} for {corruption} severity {severity}")
    data_yaml = root / "dataset.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": root.as_posix(),
                "train": "images/test",
                "val": "images/test",
                "test": "images/test",
                "names": {index: str(index + 1) for index in range(26)},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return data_yaml, pairs


def create_corruption_montage(
    dataset: Path,
    output: Path,
) -> None:
    source = sorted((dataset / "images" / "test").glob("*"))[0]
    original = cv2.imread(str(source))
    panels = [("clean", original)]
    for name, operation in CORRUPTIONS.items():
        panels.append((name, operation(original, 3, deterministic_rng(name, "3", source.name))))
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2))
    for axis, (name, image) in zip(axes.ravel(), panels):
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(name.capitalize(), fontsize=11)
        axis.axis("off")
    fig.suptitle("Deterministic OSI26 corruption stress test (severity 3 examples)", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output / "corruption_examples.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_robustness(
    weights: Path,
    data: Path,
    dataset: Path,
    output: Path,
    device: str,
    imgsz: int,
) -> dict[str, Any]:
    model = YOLO(str(weights))
    clean = validate(model, data, "test", output, "robust_clean", device, imgsz)
    rows: list[dict[str, Any]] = [
        {"corruption": "clean", "severity": 0, **clean, "relative_mAP50_95": 1.0, "drop_mAP50_95": 0.0}
    ]
    for corruption in CORRUPTIONS:
        for severity in (1, 2, 3):
            corruption_yaml, _ = build_corruption_dataset(dataset, output, corruption, severity)
            metrics = validate(
                model,
                corruption_yaml,
                "test",
                output,
                f"robust_{corruption}_s{severity}",
                device,
                imgsz,
            )
            rows.append(
                {
                    "corruption": corruption,
                    "severity": severity,
                    **metrics,
                    "relative_mAP50_95": metrics["mAP50_95"] / clean["mAP50_95"],
                    "drop_mAP50_95": clean["mAP50_95"] - metrics["mAP50_95"],
                }
            )
    csv_path = output / "robustness_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    severity3 = [row for row in rows if row["severity"] == 3]
    mean_corruption = float(np.mean([row["mAP50_95"] for row in severity3]))
    summary = {
        "protocol": {
            "split": "unchanged OSI26 test split (31 images, 173 instances)",
            "severities": [1, 2, 3],
            "seed": "SHA256-derived deterministic per-image seed",
            "labels": "unchanged from clean images",
            "warning": "synthetic stress test; not a clinical-domain corruption benchmark",
        },
        "clean": clean,
        "mean_corruption_mAP50_95_at_severity3": mean_corruption,
        "relative_mean_corruption_performance_at_severity3": mean_corruption / clean["mAP50_95"],
        "rows": rows,
    }
    write_json(output / "robustness_summary.json", summary)
    create_corruption_montage(dataset, output)

    fig, axis = plt.subplots(figsize=(10.2, 5.5))
    for corruption in CORRUPTIONS:
        selected = [row for row in rows if row["corruption"] == corruption]
        selected = [{"severity": 0, "mAP50_95": clean["mAP50_95"]}, *selected]
        axis.plot(
            [row["severity"] for row in selected],
            [row["mAP50_95"] for row in selected],
            marker="o",
            linewidth=2,
            label=corruption,
        )
    axis.set_xticks([0, 1, 2, 3], ["clean", "mild", "moderate", "severe"])
    axis.set_ylabel("mAP@50:95")
    axis.set_xlabel("Corruption severity")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(ncol=4, fontsize=9, frameon=False)
    axis.set_title("WFD-YOLO robustness on deterministic synthetic corruptions")
    fig.tight_layout()
    fig.savefig(output / "robustness_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


def collect_sfpa_and_mixers(model: YOLO) -> tuple[list[SFPA], list[_AdaptiveBranchMixer]]:
    sfpa = [module for module in model.model.modules() if isinstance(module, SFPA)]
    mixers = [module for module in model.model.modules() if isinstance(module, _AdaptiveBranchMixer)]
    if len(sfpa) != 4 or len(mixers) != 6:
        raise RuntimeError(f"Expected 4 SFPA modules and 6 protected mixers, received {len(sfpa)} and {len(mixers)}")
    return sfpa, mixers


def run_sensitivity(
    weights: Path,
    data: Path,
    output: Path,
    device: str,
    imgsz: int,
) -> dict[str, Any]:
    model = YOLO(str(weights))
    sfpa, mixers = collect_sfpa_and_mixers(model)
    original_lambda = [module.residual_scale.detach().clone() for module in sfpa]
    original_threshold = [module.threshold for module in sfpa]
    original_kernel = [module.gaussian_kernel.detach().clone() for module in sfpa]
    original_logits = [module.innovation_logit.detach().clone() for module in mixers]
    learned = {
        "lambda_per_sfpa": [float(value) for value in original_lambda],
        "tau_per_sfpa": original_threshold,
        "innovation_weight_per_gate": [float(value.sigmoid()) for value in original_logits],
    }
    checkpoint = validate(model, data, "val", output, "sensitivity_checkpoint", device, imgsz)
    rows: list[dict[str, Any]] = []

    def record(parameter: str, value: float, run_name: str) -> None:
        metrics = validate(model, data, "val", output, run_name, device, imgsz)
        rows.append({"parameter": parameter, "value": value, **metrics})

    for value in (0.0, 0.05, 0.10, 0.20, 0.30):
        for module in sfpa:
            module.residual_scale.data.fill_(value)
        record("lambda", value, f"sensitivity_lambda_{value:.2f}")
    for module, value in zip(sfpa, original_lambda):
        module.residual_scale.data.copy_(value)

    for value in (0.50, 0.75, 1.00, 1.50, 2.00):
        for module in sfpa:
            kernel = module._gaussian_kernel(size=5, sigma=value)[None, None]
            module.gaussian_kernel = kernel.to(module.gaussian_kernel.device)
        record("sigma", value, f"sensitivity_sigma_{value:.2f}")
    for module, value in zip(sfpa, original_kernel):
        module.gaussian_kernel = value

    for value in (0.25, 0.50, 0.75, 1.00):
        for module in sfpa:
            module.threshold = value
        record("tau", value, f"sensitivity_tau_{value:.2f}")
    for module, value in zip(sfpa, original_threshold):
        module.threshold = value

    for value in (0.0, 0.025, 0.05, 0.10, 0.20):
        clipped = min(max(value, 1e-8), 1.0 - 1e-8)
        logit = math.log(clipped / (1.0 - clipped))
        for module in mixers:
            module.innovation_logit.data.fill_(logit)
        record("protected_gate_innovation", value, f"sensitivity_gate_{value:.3f}")
    for module, value in zip(mixers, original_logits):
        module.innovation_logit.data.copy_(value)

    with (output / "sensitivity_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = {}
    for parameter in ("lambda", "sigma", "tau", "protected_gate_innovation"):
        selected = [row for row in rows if row["parameter"] == parameter]
        best[parameter] = max(selected, key=lambda row: row["mAP50_95"])
    summary = {
        "protocol": (
            "Post-training one-factor-at-a-time sensitivity on the fixed OSI26 validation split. "
            "All other trained weights are frozen; this is not independent retraining."
        ),
        "checkpoint_learned_controls": learned,
        "checkpoint_validation": checkpoint,
        "best_static_setting_by_validation_mAP50_95": best,
        "rows": rows,
    }
    write_json(output / "sensitivity_summary.json", summary)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for axis, parameter, title in zip(
        axes.ravel(),
        ("lambda", "sigma", "tau", "protected_gate_innovation"),
        ("SFPA residual lambda", "SFPA Gaussian sigma", "SFPA threshold tau", "Protected innovation weight"),
    ):
        selected = [row for row in rows if row["parameter"] == parameter]
        axis.plot([row["value"] for row in selected], [row["mAP50_95"] for row in selected], marker="o")
        axis.axhline(checkpoint["mAP50_95"], color="#E76F51", linestyle="--", linewidth=1.2, label="learned checkpoint")
        axis.set_title(title)
        axis.set_xlabel("Value")
        axis.set_ylabel("Validation mAP@50:95")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, frameon=False)
    fig.suptitle("WFD-YOLO one-factor post-training sensitivity", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output / "sensitivity_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


def analyze_training_curves(results_csv: Path, output: Path) -> dict[str, Any]:
    with results_csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    numeric = [{key.strip(): float(value) for key, value in row.items()} for row in rows]
    best = max(numeric, key=lambda row: row["metrics/mAP50-95(B)"])
    final = numeric[-1]
    last20 = numeric[-20:]
    x = np.asarray([row["epoch"] for row in last20], dtype=np.float64)
    y = np.asarray([row["metrics/mAP50-95(B)"] for row in last20], dtype=np.float64)
    slope = float(np.polyfit(x, y, 1)[0]) if len(last20) >= 2 else 0.0
    summary = {
        "epochs_recorded": len(numeric),
        "best_epoch_1_based": int(best["epoch"]),
        "best_validation": {
            "precision": best["metrics/precision(B)"],
            "recall": best["metrics/recall(B)"],
            "mAP50": best["metrics/mAP50(B)"],
            "mAP50_95": best["metrics/mAP50-95(B)"],
        },
        "final_epoch_1_based": int(final["epoch"]),
        "final_validation_mAP50_95": final["metrics/mAP50-95(B)"],
        "best_to_final_mAP50_95_change": final["metrics/mAP50-95(B)"] - best["metrics/mAP50-95(B)"],
        "last20_mAP50_95_linear_slope_per_epoch": slope,
        "final_train_losses": {
            "box": final["train/box_loss"],
            "cls": final["train/cls_loss"],
            "dfl": final["train/dfl_loss"],
        },
        "final_validation_losses": {
            "box": final["val/box_loss"],
            "cls": final["val/cls_loss"],
            "dfl": final["val/dfl_loss"],
        },
    }
    write_json(output / "training_curve_analysis.json", summary)

    epochs = [row["epoch"] for row in numeric]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), sharex=True)
    pairs = [
        ("train/box_loss", "val/box_loss", "Box loss"),
        ("train/cls_loss", "val/cls_loss", "Classification loss"),
        ("train/dfl_loss", "val/dfl_loss", "DFL loss"),
    ]
    for axis, (train_key, val_key, title) in zip(axes[0], pairs):
        axis.plot(epochs, [row[train_key] for row in numeric], label="train")
        axis.plot(epochs, [row[val_key] for row in numeric], label="val")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    metrics = [
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
        ("metrics/mAP50-95(B)", "mAP@50:95"),
    ]
    for axis, (key, title) in zip(axes[1], metrics):
        axis.plot(epochs, [row[key] for row in numeric], color="#2A9D8F")
        axis.axvline(best["epoch"], color="#E76F51", linestyle="--", linewidth=1.2)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    fig.suptitle(f"WFD-YOLO training/validation history (best epoch = {int(best['epoch'])})", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output / "training_validation_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


@torch.inference_mode()
def memory_for_precision(
    weights: Path,
    device: str,
    imgsz: int,
    half: bool,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    device_obj = torch.device(f"cuda:{device}")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_obj)
    process_before = int(torch.cuda.memory_allocated(device_obj))
    wrapper = YOLO(str(weights))
    network = wrapper.model.to(device_obj).eval()
    network = network.half() if half else network.float()
    dtype = torch.float16 if half else torch.float32
    tensor = torch.rand((1, 3, imgsz, imgsz), device=device_obj, dtype=dtype)
    for _ in range(warmup):
        output = network(tensor)
        del output
    torch.cuda.synchronize(device_obj)
    resident_allocated = int(torch.cuda.memory_allocated(device_obj))
    resident_reserved = int(torch.cuda.memory_reserved(device_obj))
    torch.cuda.reset_peak_memory_stats(device_obj)
    for _ in range(repeats):
        output = network(tensor)
        del output
    torch.cuda.synchronize(device_obj)
    peak_allocated = int(torch.cuda.max_memory_allocated(device_obj))
    peak_reserved = int(torch.cuda.max_memory_reserved(device_obj))
    result = {
        "precision": "FP16" if half else "FP32",
        "batch": 1,
        "input_shape": [1, 3, imgsz, imgsz],
        "warmup": warmup,
        "measured_forwards": repeats,
        "process_before_model_MiB": process_before / 2**20,
        "resident_model_plus_input_allocated_MiB": resident_allocated / 2**20,
        "resident_reserved_MiB": resident_reserved / 2**20,
        "peak_allocated_MiB": peak_allocated / 2**20,
        "peak_reserved_MiB": peak_reserved / 2**20,
        "incremental_forward_workspace_MiB": (peak_allocated - resident_allocated) / 2**20,
        "scope": "PyTorch CUDA allocator; model + one input + forward activations; excludes disk I/O and plotting",
    }
    del tensor, network, wrapper
    gc.collect()
    torch.cuda.empty_cache()
    return result


def memory_worker(
    queue: Any,
    weights: str,
    device: str,
    imgsz: int,
    half: bool,
    warmup: int,
    repeats: int,
) -> None:
    """Measure one precision in a fresh process so CUDA/FFT caches cannot leak across settings."""
    try:
        queue.put(
            {
                "ok": True,
                "result": memory_for_precision(Path(weights), device, imgsz, half, warmup, repeats),
            }
        )
    except Exception as error:  # pragma: no cover - returned to parent for a useful traceback message
        queue.put({"ok": False, "error": repr(error)})


def isolated_memory_measurement(
    weights: Path,
    device: str,
    imgsz: int,
    half: bool,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=memory_worker,
        args=(queue, str(weights), device, imgsz, half, warmup, repeats),
    )
    process.start()
    message = queue.get()
    process.join()
    if process.exitcode != 0 or not message.get("ok"):
        raise RuntimeError(
            f"Isolated {'FP16' if half else 'FP32'} memory measurement failed: "
            f"exit={process.exitcode}, message={message}"
        )
    return message["result"]


def model_size_and_memory(
    weights: Path,
    output: Path,
    device: str,
    imgsz: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    wrapper = YOLO(str(weights))
    network = wrapper.model.cpu().eval().float()
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    checkpoint_requires_grad_count = sum(
        parameter.numel() for parameter in network.parameters() if parameter.requires_grad
    )
    buffer_count = sum(buffer.numel() for buffer in network.buffers())
    state = io.BytesIO()
    torch.save(network.state_dict(), state)
    state_dict_fp32_bytes = state.tell()
    state.close()
    theoretical_parameter_fp32_bytes = parameter_count * 4
    del network, wrapper
    gc.collect()
    size = {
        "checkpoint_path": str(weights.resolve()),
        "checkpoint_bytes": weights.stat().st_size,
        "checkpoint_MiB": weights.stat().st_size / 2**20,
        "parameters": parameter_count,
        "architecture_trainable_parameters": parameter_count,
        "checkpoint_requires_grad_parameters_after_inference_load": checkpoint_requires_grad_count,
        "buffers": buffer_count,
        "parameter_count_M": parameter_count / 1e6,
        "theoretical_parameter_storage_FP32_MiB": theoretical_parameter_fp32_bytes / 2**20,
        "serialized_state_dict_FP32_MiB": state_dict_fp32_bytes / 2**20,
        "note": "Checkpoint size includes Ultralytics metadata; theoretical storage excludes metadata and buffers.",
    }
    memory = [
        isolated_memory_measurement(weights, device, imgsz, False, warmup, repeats),
        isolated_memory_measurement(weights, device, imgsz, True, warmup, repeats),
    ]
    payload = {
        "hardware": {
            "GPU": torch.cuda.get_device_name(int(device)),
            "GPU_total_MiB": torch.cuda.get_device_properties(int(device)).total_memory / 2**20,
            "torch": torch.__version__,
            "CUDA_runtime": torch.version.cuda,
        },
        "model_size": size,
        "peak_inference_memory": memory,
    }
    write_json(output / "model_size_and_gpu_memory.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for path in (args.weights, args.data, args.dataset, args.results_csv):
        if not path.exists():
            raise FileNotFoundError(path)
    torch.manual_seed(0)
    np.random.seed(0)
    training = analyze_training_curves(args.results_csv, args.output)
    efficiency = model_size_and_memory(
        args.weights,
        args.output,
        args.device,
        args.imgsz,
        args.memory_warmup,
        args.memory_repeats,
    )
    robustness = None
    if not args.skip_robustness:
        robustness = run_robustness(
            args.weights,
            args.data,
            args.dataset,
            args.output,
            args.device,
            args.imgsz,
        )
    sensitivity = None
    if not args.skip_sensitivity:
        sensitivity = run_sensitivity(
            args.weights,
            args.data,
            args.output,
            args.device,
            args.imgsz,
        )
    manifest = {
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "dataset": str(args.dataset.resolve()),
        "training_curve_analysis": training,
        "model_size_and_gpu_memory": efficiency,
        "robustness_summary": robustness,
        "sensitivity_summary": sensitivity,
        "artifacts": sorted(
            str(path.relative_to(args.output)) for path in args.output.rglob("*") if path.is_file()
        ),
    }
    write_json(args.output / "additional_experiment_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
