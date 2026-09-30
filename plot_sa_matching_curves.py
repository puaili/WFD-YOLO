"""Create a compact SA-Matching DETR training/validation curve figure."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent / "runs/sa_matching_detr"
CSV = ROOT / "runs/sa_matching_detr_full_partialvit_t_osi26_200e_batch16/results.csv"
OUT = ROOT / "sa_matching_training_curves.png"

with CSV.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

epochs = [int(float(r["epoch"])) for r in rows]
ap50 = [float(r["metrics/mAP50(B)"]) for r in rows]
ap95 = [float(r["metrics/mAP50-95(B)"]) for r in rows]
train_giou = [float(r["train/giou_loss"]) for r in rows]
val_giou = [float(r["val/giou_loss"]) for r in rows]
train_cls = [float(r["train/cls_loss"]) for r in rows]
val_cls = [float(r["val/cls_loss"]) for r in rows]
best = max(range(len(rows)), key=lambda i: ap95[i])

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), dpi=180)
axes[0].plot(epochs, ap50, label="mAP@50", color="#2E74B5")
axes[0].plot(epochs, ap95, label="mAP@50:95", color="#ED7D31")
axes[0].scatter([epochs[best]], [ap95[best]], color="#C00000", s=24, zorder=4)
axes[0].annotate(f"best e{epochs[best]}: {ap95[best]:.3f}", (epochs[best], ap95[best]), xytext=(6, -18), textcoords="offset points")
axes[0].set(title="Validation metrics", xlabel="Epoch", ylabel="AP", ylim=(0, 0.55))
axes[0].legend(frameon=False)
axes[1].plot(epochs, train_giou, label="Train", color="#2E74B5")
axes[1].plot(epochs, val_giou, label="Validation", color="#ED7D31")
axes[1].set(title="GIoU loss", xlabel="Epoch", ylabel="Loss")
axes[1].legend(frameon=False)
axes[2].plot(epochs, train_cls, label="Train", color="#2E74B5")
axes[2].plot(epochs, val_cls, label="Validation", color="#ED7D31")
axes[2].set(title="Classification loss", xlabel="Epoch", ylabel="Loss")
axes[2].legend(frameon=False)
for ax in axes:
    ax.grid(alpha=0.2)
fig.suptitle("SA-Matching DETR PartialViT-T on OSI26 (200 epochs, batch 16)", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(OUT)
