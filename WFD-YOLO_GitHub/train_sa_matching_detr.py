"""Train the paper-driven full SA-Matching DETR PartialViT-T reproduction on OSI26."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo_config"))
os.environ.setdefault("YOLO_DATASET_CACHE_DIR", str(ROOT / ".dataset_cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import RTDETR  # noqa: E402
from ultralytics.nn.modules.samatching import PartialViTBlock  # noqa: E402


DEFAULT_MODEL = ROOT / "ultralytics/cfg/models/rt-detr/sa-matching-detr-partialvit-t.yaml"
DEFAULT_DATA = ROOT / "ultralytics/cfg/datasets/OSI26.yaml"
DEFAULT_OUTPUT = ROOT / "runs/sa_matching_detr"
DEFAULT_RUNS = DEFAULT_OUTPUT / "runs"
RUN_NAME = "sa_matching_detr_full_partialvit_t_osi26_200e_batch16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--project", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--name", default=RUN_NAME)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16, help="Physical batch; nbs=16 preserves the paper's total batch")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-imagenet-pretrained", action="store_true")
    return parser.parse_args()


def transfer_deit_tiny(model: RTDETR) -> dict[str, Any]:
    """Trim ImageNet-1K DeiT-T weights into the partial-channel backbone."""
    import timm

    source = timm.create_model("deit_tiny_patch16_224", pretrained=True)
    target = model.model.model
    copied: list[str] = []
    with torch.no_grad():
        target[0].proj.weight.copy_(source.patch_embed.proj.weight)
        target[0].proj.bias.copy_(source.patch_embed.proj.bias)
        copied.extend(["patch_embed.proj.weight", "patch_embed.proj.bias"])
        blocks = list(target[1]) if isinstance(target[1], torch.nn.Sequential) else [target[1]]
        if len(blocks) != len(source.blocks):
            raise RuntimeError(f"Expected {len(source.blocks)} target blocks, received {len(blocks)}")
        for index, (destination, origin) in enumerate(zip(blocks, source.blocks)):
            if not isinstance(destination, PartialViTBlock):
                raise TypeError(type(destination))
            active = destination.active_channels
            channels = destination.channels
            destination.norm1.weight.copy_(origin.norm1.weight)
            destination.norm1.bias.copy_(origin.norm1.bias)
            qkv_weight = origin.attn.qkv.weight
            qkv_bias = origin.attn.qkv.bias
            rows = torch.cat(
                (
                    torch.arange(0, active),
                    torch.arange(channels, channels + active),
                    torch.arange(2 * channels, 2 * channels + active),
                )
            )
            destination.qkv.weight.copy_(qkv_weight[rows, :active])
            destination.qkv.bias.copy_(qkv_bias[rows])
            destination.proj.weight.copy_(origin.attn.proj.weight[:active, :active])
            destination.proj.bias.copy_(origin.attn.proj.bias[:active])
            destination.norm2.weight.copy_(origin.norm2.weight)
            destination.norm2.bias.copy_(origin.norm2.bias)
            destination.mlp[0].weight.copy_(origin.mlp.fc1.weight)
            destination.mlp[0].bias.copy_(origin.mlp.fc1.bias)
            destination.mlp[3].weight.copy_(origin.mlp.fc2.weight)
            destination.mlp[3].bias.copy_(origin.mlp.fc2.bias)
            copied.extend(
                [
                    f"blocks.{index}.norm1",
                    f"blocks.{index}.partial_qkv",
                    f"blocks.{index}.partial_proj",
                    f"blocks.{index}.norm2",
                    f"blocks.{index}.mlp",
                ]
            )
        target[2].norm.weight.copy_(source.norm.weight)
        target[2].norm.bias.copy_(source.norm.bias)
        copied.append("final_norm")
    return {
        "source": "timm deit_tiny_patch16_224 ImageNet-1K",
        "source_parameter_count": sum(parameter.numel() for parameter in source.parameters()),
        "target_partialvit_parameter_count": sum(parameter.numel() for layer in target[:3] for parameter in layer.parameters()),
        "copied_groups": copied,
        "partial_channel_rule": "Q/K/V and output projection top-left active-channel submatrices; full norms/MLP",
    }


def completed_epochs(run_dir: Path) -> int:
    path = run_dir / "results.csv"
    if not path.exists():
        return 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return len(list(csv.DictReader(stream)))


def metrics_payload(metrics: Any) -> dict[str, float]:
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }


def evaluate(run_dir: Path, data: Path, output: Path, device: str, imgsz: int) -> dict[str, Any]:
    best = run_dir / "weights/best.pt"
    if not best.exists():
        raise FileNotFoundError(best)
    model = RTDETR(str(best))
    payload: dict[str, Any] = {
        "checkpoint": str(best.resolve()),
        "checkpoint_MiB": best.stat().st_size / 2**20,
        "parameters_M": sum(parameter.numel() for parameter in model.model.parameters()) / 1e6,
        "queries": model.model.model[-1].num_queries,
        "decoder_layers": model.model.model[-1].num_decoder_layers,
    }
    results_path = run_dir / "results.csv"
    with results_path.open(encoding="utf-8-sig", newline="") as stream:
        history = list(csv.DictReader(stream))
    best_row = max(history, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    payload["best_epoch"] = int(float(best_row["epoch"]))
    for split in ("val", "test"):
        metrics = model.val(
            data=str(data),
            split=split,
            imgsz=imgsz,
            batch=1,
            device=device,
            workers=0,
            plots=True,
            verbose=False,
            project=str(output / "evaluation"),
            name=f"sa_matching_detr_{split}",
            exist_ok=True,
        )
        payload[split] = metrics_payload(metrics)
    (output / "sa_matching_detr_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.project.mkdir(parents=True, exist_ok=True)
    run_dir = args.project / args.name
    if args.resume:
        model = RTDETR(str(args.resume))
        model.train(resume=True)
    elif completed_epochs(run_dir) < args.epochs or not (run_dir / "weights/best.pt").exists():
        model = RTDETR(str(args.model))
        transfer = None
        if not args.no_imagenet_pretrained:
            transfer = transfer_deit_tiny(model)
            (args.output / "imagenet_pretraining_transfer.json").write_text(
                json.dumps(transfer, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        model.ckpt = {"model": model.model}
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            batch=args.batch,
            nbs=16,
            imgsz=args.imgsz,
            workers=args.workers,
            device=args.device,
            project=str(args.project),
            name=args.name,
            patience=args.epochs,
            optimizer="AdamW",
            lr0=2e-4,
            lrf=0.1,
            weight_decay=1e-4,
            warmup_epochs=250 / 24,
            warmup_bias_lr=0.0,
            cos_lr=False,
            amp=True,
            seed=0,
            deterministic=True,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            val=True,
            plots=True,
            verbose=True,
            exist_ok=True,
        )
    result = evaluate(run_dir, args.data, args.output, args.device, args.imgsz)
    protocol = {
        "reference_paper_declared": {
            "model": "full SA-Matching DETR + Partial ViT-T + 3 decoder layers",
            "queries": 900,
            "pretraining": "ImageNet-1K",
            "optimizer": "AdamW",
            "paper_total_batch": 16,
            "paper_input": 1024,
            "paper_schedules": "12 or 24 epochs; LR x0.1 at epoch 11 or 20; 250-iteration warm-up",
            "s_m_l": [1, 7, 9],
        },
        "osi26_fair_comparison_adaptation": {
            "epochs": args.epochs,
            "input": args.imgsz,
            "physical_batch": args.batch,
            "gradient_accumulation_target_nbs": 16,
            "optimizer": "AdamW",
            "lr0": 2e-4,
            "lrf": 0.1,
            "warmup_epochs_equivalent_to_250_OSI26_iterations": 250 / 24,
            "seed": 0,
        },
        "official_code_status": "not released as of 2026-08-11; paper-driven reconstruction",
        "results": result,
    }
    (args.output / "sa_matching_detr_reproduction_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
