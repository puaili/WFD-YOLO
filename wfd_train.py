
"""Train WFD-YOLO on SID-RAS or a custom surgical instrument dataset."""
from __future__ import annotations
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Lib/wfd_yolo12n.yaml', help='Model YAML path.')
    parser.add_argument('--data', default='dataset/sid_ras.yaml', help='Dataset YAML path.')
    parser.add_argument('--weights', default='yolo12n.pt', help='Optional pretrained checkpoint.')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--device', default='0')
    parser.add_argument('--project', default='runs/wfd')
    parser.add_argument('--name', default='wfd_yolo12n_sid_ras')
    return parser.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)
    if args.weights:
        model = model.load(args.weights)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
    )

if __name__ == '__main__':
    main()
