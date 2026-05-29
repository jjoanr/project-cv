from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "xy": (255, 60, 60),
    "yx": (40, 120, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw JSON corner annotations on the first dataset images."
    )
    parser.add_argument("--data", default="Dataset/data", help="Folder with .jpg/.json pairs.")
    parser.add_argument(
        "--out",
        default="Dataset/json_corners_debug",
        help="Output folder for annotated images.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="How many numeric image/json pairs to process.",
    )
    return parser.parse_args()


def numeric_stem(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def first_json_files(data_dir: Path, count: int) -> list[Path]:
    json_files = []
    for path in data_dir.glob("*.json"):
        number = numeric_stem(path)
        if number is not None:
            json_files.append((number, path))
    return [path for _, path in sorted(json_files)[:count]]


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def as_pixels(point: list[float], width: int, height: int, mode: str) -> tuple[int, int]:
    first, second = point
    if mode == "xy":
        x_norm, y_norm = first, second
    elif mode == "yx":
        y_norm, x_norm = first, second
    else:
        raise ValueError(f"Unknown corner mode: {mode}")

    return round(x_norm * width), round(y_norm * height)


def draw_corners(image_path: Path, corners: list[list[float]], out_path: Path, mode: str) -> None:
    with Image.open(image_path) as image:
        annotated = image.convert("RGB")

    width, height = annotated.size
    draw = ImageDraw.Draw(annotated)
    color = COLORS[mode]
    radius = max(6, round(min(width, height) * 0.012))
    line_width = max(3, radius // 3)
    font = load_font(max(14, radius * 2))

    points = [as_pixels(point, width, height, mode) for point in corners]
    if len(points) > 1:
        draw.line(points + [points[0]], fill=color, width=line_width)

    for index, (x, y) in enumerate(points, start=1):
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline=(255, 255, 255),
            width=max(2, line_width // 2),
        )
        label = str(index)
        bbox = draw.textbbox((0, 0), label, font=font)
        label_w = bbox[2] - bbox[0]
        label_h = bbox[3] - bbox[1]
        label_x = x + radius + 4
        label_y = y - radius - 4
        draw.rectangle(
            (label_x - 2, label_y - 2, label_x + label_w + 4, label_y + label_h + 4),
            fill=(0, 0, 0),
        )
        draw.text((label_x, label_y), label, fill=(255, 255, 255), font=font)

    title = f"{image_path.name} corners as ({mode[0]},{mode[1]})"
    title_font = load_font(max(18, radius * 2))
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    draw.rectangle(
        (8, 8, title_bbox[2] + 22, title_bbox[3] + 22),
        fill=(0, 0, 0),
    )
    draw.text((15, 13), title, fill=color, font=title_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(out_path, quality=95)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data)
    out_dir = Path(args.out)

    for json_path in first_json_files(data_dir, args.count):
        image_path = json_path.with_suffix(".jpg")
        if not image_path.exists():
            print(f"Skipping {json_path}: missing {image_path.name}")
            continue

        data = json.loads(json_path.read_text(encoding="utf-8"))
        corners = data.get("corners")
        if not isinstance(corners, list) or len(corners) != 4:
            print(f"Skipping {json_path}: expected 4 corners")
            continue

        for mode in ("xy", "yx"):
            out_path = out_dir / f"{json_path.stem}_corners_{mode}.jpg"
            draw_corners(image_path, corners, out_path, mode)
            print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
