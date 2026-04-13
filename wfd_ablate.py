
"""Ablation runner for WFD-YOLO variants."""
from __future__ import annotations
import argparse
from ultralytics import YOLO

ABLATIONS = {
    'baseline': 'ultralytics/cfg/models/12/yolo12.yaml',
    'full': 'Lib/wfd_yolo12n.yaml',
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=ABLATIONS.keys(), default='full')
    parser.add_argument('--data', default='dataset/sid_ras.yaml')
    parser.add_argument('--weights', default='yolo12n.pt')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='0')
    parser.add_argument('--project', default='runs/ablation')
    return parser.parse_args()

def main():
    args = parse_args()
    model = YOLO(ABLATIONS[args.variant])
    if args.weights:
        model = model.load(args.weights)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.variant,
    )

if __name__ == '__main__':
    main()
