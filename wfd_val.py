
"""Validate a trained WFD-YOLO checkpoint."""
from __future__ import annotations
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='Path to trained weights.')
    parser.add_argument('--data', default='dataset/sid_ras.yaml')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--device', default='0')
    return parser.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, split=args.split, device=args.device)

if __name__ == '__main__':
    main()
