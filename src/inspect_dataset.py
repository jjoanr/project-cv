"""
Run this first to check the dataset classes and distribution.
Usage: python src/inspect_dataset.py
"""

import json
from pathlib import Path
from collections import Counter


def inspect(annotation_path: str):
    with open(annotation_path) as f:
        data = json.load(f)

    categories = {c["id"]: c["name"] for c in data["categories"]}
    print("=== Classes found ===")
    for cid, name in categories.items():
        print(f"  {cid}: {name}")

    counts = Counter(ann["category_id"] for ann in data["annotations"])
    print("\n=== Examples per class ===")
    for cid, count in sorted(counts.items()):
        print(f"  {categories[cid]:<20} {count}")

    print(f"\nTotal images : {len(data['images'])}")
    print(f"Total annotations: {len(data['annotations'])}")


if __name__ == "__main__":
    base = Path("dataset/annotations")
    for split in ("instances_train.json", "instances_val.json"):
        path = base / split
        if path.exists():
            print(f"\n{'='*40}")
            print(f"File: {split}")
            print('='*40)
            inspect(str(path))
        else:
            print(f"[!] Not found: {path}")
