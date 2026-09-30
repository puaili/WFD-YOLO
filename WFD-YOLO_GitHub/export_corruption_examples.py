"""Export the eight standalone images used by the severity-3 corruption montage."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image


WORKSPACE = Path(__file__).resolve().parent
DATASET = WORKSPACE / "data/OSI26_Dataset"
ADDITIONAL = WORKSPACE / "reviewer_additional_20260811"
OUTPUT = ADDITIONAL / "corruption_example_images"
SOURCE_NAME = "a_11_8_26_12_16_4_23_7.jpg"
CORRUPTIONS = ("brightness", "contrast", "blur", "smoke", "specular", "occlusion", "clutter")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = {
        "clean": DATASET / "images/test" / SOURCE_NAME,
        **{
            name: ADDITIONAL / "corruption_datasets" / f"{name}_s3" / "images/test" / SOURCE_NAME
            for name in CORRUPTIONS
        },
    }
    records = []
    for index, (condition, source) in enumerate(sources.items(), start=1):
        if not source.exists():
            raise FileNotFoundError(source)
        destination = OUTPUT / f"{index:02d}_{condition}_severity3.jpg"
        shutil.copy2(source, destination)
        with Image.open(destination) as image:
            width, height = image.size
        records.append(
            {
                "index": index,
                "condition": condition,
                "severity": 0 if condition == "clean" else 3,
                "source_image": SOURCE_NAME,
                "file": destination.name,
                "width": width,
                "height": height,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    manifest = {
        "description": "Standalone source-resolution images used in the severity-3 synthetic-corruption montage.",
        "source_image": SOURCE_NAME,
        "protocol": "clean plus seven deterministic severity-3 corruptions; no crop, resize, annotation, or recompression during export",
        "images": records,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

