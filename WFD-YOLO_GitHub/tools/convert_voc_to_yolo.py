"""Convert OSI26 Pascal VOC XML annotations to YOLO TXT labels without deleting XML files."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SPLITS = ("train", "val", "test")


def _sort_class_names(names: set[str]) -> list[str]:
    if names and all(name.isdigit() for name in names):
        return sorted(names, key=int)
    return sorted(names)


def _find_image(images_dir: Path, filename: str, stem: str) -> Path | None:
    direct = images_dir / filename
    if filename and direct.is_file():
        return direct
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _read_xml(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError("missing <size>")
    width = int(float(size.findtext("width", "0")))
    height = int(float(size.findtext("height", "0")))
    objects = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        objects.append(
            {
                "name": obj.findtext("name", "").strip(),
                "xmin": float(box.findtext("xmin", "nan")),
                "ymin": float(box.findtext("ymin", "nan")),
                "xmax": float(box.findtext("xmax", "nan")),
                "ymax": float(box.findtext("ymax", "nan")),
            }
        )
    return {
        "filename": root.findtext("filename", "").strip(),
        "width": width,
        "height": height,
        "objects": objects,
    }


def discover_classes(labels_root: Path) -> tuple[list[str], Counter]:
    counts: Counter = Counter()
    for split in SPLITS:
        for xml_path in sorted((labels_root / split).glob("*.xml")):
            annotation = _read_xml(xml_path)
            counts.update(obj["name"] for obj in annotation["objects"] if obj["name"])
    return _sort_class_names(set(counts)), counts


def convert_split(
    dataset_root: Path,
    split: str,
    class_to_index: dict[str, int],
    dry_run: bool,
) -> dict:
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    xml_files = sorted(labels_dir.glob("*.xml"))
    image_files = {p.stem: p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    report = {
        "xml_files": len(xml_files),
        "image_files": len(image_files),
        "labels_written": 0,
        "objects_written": 0,
        "empty_annotations": 0,
        "clamped_boxes": 0,
        "invalid_boxes": [],
        "missing_images": [],
        "image_size_mismatches": [],
        "images_without_xml": [],
        "class_counts": Counter(),
    }
    xml_stems: set[str] = set()

    for xml_path in xml_files:
        xml_stems.add(xml_path.stem)
        try:
            annotation = _read_xml(xml_path)
        except Exception as exc:
            report["invalid_boxes"].append({"xml": str(xml_path), "error": str(exc)})
            continue

        image_path = _find_image(images_dir, annotation["filename"], xml_path.stem)
        if image_path is None:
            report["missing_images"].append(str(xml_path))
            continue
        width, height = annotation["width"], annotation["height"]
        if width <= 0 or height <= 0:
            report["invalid_boxes"].append({"xml": str(xml_path), "error": f"invalid size {width}x{height}"})
            continue
        with Image.open(image_path) as image:
            actual_size = image.size
        if actual_size != (width, height):
            report["image_size_mismatches"].append(
                {
                    "xml": str(xml_path),
                    "declared": [width, height],
                    "actual": list(actual_size),
                }
            )
            width, height = actual_size

        lines: list[str] = []
        for obj in annotation["objects"]:
            name = obj["name"]
            if name not in class_to_index:
                report["invalid_boxes"].append({"xml": str(xml_path), "error": f"unknown class {name!r}"})
                continue
            raw = (obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"])
            xmin = min(max(raw[0], 0.0), float(width))
            ymin = min(max(raw[1], 0.0), float(height))
            xmax = min(max(raw[2], 0.0), float(width))
            ymax = min(max(raw[3], 0.0), float(height))
            if (xmin, ymin, xmax, ymax) != raw:
                report["clamped_boxes"] += 1
            if xmax <= xmin or ymax <= ymin:
                report["invalid_boxes"].append({"xml": str(xml_path), "class": name, "box": list(raw)})
                continue
            x_center = ((xmin + xmax) / 2.0) / width
            y_center = ((ymin + ymax) / 2.0) / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height
            lines.append(
                f"{class_to_index[name]} {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}"
            )
            report["objects_written"] += 1
            report["class_counts"][name] += 1

        if not lines:
            report["empty_annotations"] += 1
        if not dry_run:
            output_path = labels_dir / f"{xml_path.stem}.txt"
            output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        report["labels_written"] += 1

    report["images_without_xml"] = [str(path) for stem, path in image_files.items() if stem not in xml_stems]
    report["class_counts"] = dict(sorted(report["class_counts"].items(), key=lambda item: class_to_index[item[0]]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Dataset root containing images/ and labels/")
    parser.add_argument("--expected-classes", type=int, default=26)
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing TXT labels")
    args = parser.parse_args()

    dataset_root = args.dataset.resolve()
    labels_root = dataset_root / "labels"
    classes, source_counts = discover_classes(labels_root)
    if len(classes) != args.expected_classes:
        raise SystemExit(
            f"Expected {args.expected_classes} classes but found {len(classes)}: {classes}. "
            "No labels were converted."
        )
    class_to_index = {name: index for index, name in enumerate(classes)}
    split_reports = {
        split: convert_split(dataset_root, split, class_to_index, args.dry_run)
        for split in SPLITS
    }
    report = {
        "dataset": str(dataset_root),
        "dry_run": args.dry_run,
        "classes": classes,
        "class_to_index": class_to_index,
        "source_class_counts": dict(source_counts),
        "splits": split_reports,
    }
    report_path = dataset_root / ("conversion_report_dry_run.json" if args.dry_run else "conversion_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    critical = sum(
        len(split_reports[split][key])
        for split in SPLITS
        for key in ("invalid_boxes", "missing_images", "images_without_xml")
    )
    if critical:
        raise SystemExit(f"Conversion completed with {critical} critical validation issue(s); see {report_path}")


if __name__ == "__main__":
    main()
