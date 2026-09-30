# WFD-YOLO parallel-gating optimization

## Why the first reproduction underperformed

On the fixed OSI26 split, the first WFD reproduction reached only Precision
0.3814 and Recall 0.5014 on the test set, while the YOLOv12n baseline reached
0.9432 and 0.9179. The gap was caused by the implementation, not the converted
dataset:

1. `WFCA` replaced a normal feature with `x + wavelet + Fourier` from the first
   iteration. Randomly initialized spectral branches therefore changed the
   activation scale seen by all later pretrained layers.
2. `SFPA` returned a purified feature instead of retaining the lateral feature.
   Its initial spatial sigmoid is near 0.5, so an important pretrained route was
   attenuated before the network had learned a useful attention map.
3. `DGFA` replaced every lossless `Concat` with a narrow aligned bottleneck.
   The baseline keeps all channels; the replacement compressed both inputs into
   a much smaller shared representation and permanently discarded information.
4. ATFL replaced YOLOv12's calibrated BCE classification objective. On OSI26,
   the old training classification loss remained around 1.9, compared with
   roughly 0.3-0.5 for the converged baseline.
5. Inserted/replaced layers reduced the benefit of COCO transfer learning and
   changed the feature distribution received by pretrained downstream layers.

## Optimized design

The optimized model treats YOLOv12n as a protected baseline route:

- `WFCAParallel`: `output = w_base * x + w_wfca * WFCA(x)`.
- `ParallelFusion`: `output = w_base * Concat(inputs) + w_wfd *
  DGFA(SFPA(lateral), main)`.
- Each innovation candidate first passes through a zero-initialized depthwise
  residual adapter. The complete network is therefore mathematically identical
  to transferred YOLOv12n at step zero, while the adapter receives gradients
  immediately and can progressively activate a useful innovation.
- Each pair of route weights is a sigmoid-constrained convex mixture, so the two
  weights are non-negative and sum to one.
- Every gate starts at 95% baseline and 5% innovation. Training can increase an
  effective innovation, or reduce it towards zero and recover the baseline.
- All original weighted YOLOv12n layers receive their shape-compatible official
  pretrained tensors. A seeded 26-class baseline is constructed first, matching
  Ultralytics' standard `seed + 1` fine-tuning path, so the adapted classification head is
  also transferred and all 739 baseline state tensors are preserved.
- The YAML-created `YOLO` wrapper is explicitly marked checkpoint-backed after
  transfer. Without this flag, Ultralytics silently constructs a second model
  inside `train()` and discards all manually transferred weights.
- The detector uses the unmodified YOLOv12 detection loss: BCE classification,
  CIoU box regression and DFL. ATFL is not used.

The learned branch weights are exported with validation and test metrics so the
experiment records which route the optimizer preferred at each modified node.

## Full experiment

```powershell
conda activate yolov8
python run_wfd_200e.py
```

The runner verifies the exact 380/41/31 split and 1716 labels, trains for 200
epochs, evaluates `best.pt` independently on validation and test data, and
writes:

- `WFD_YOLO_OSI26_200e_summary.json`
- `WFD_YOLO_OSI26_200e_report.md`
- `runs/wfd-yolo/wfd_yolov12n_parallel_pretrained_osi26_200e/weights/best.pt`
- separate validation and test evaluation directories
