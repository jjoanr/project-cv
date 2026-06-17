from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from refine_graella import (
    color_valid_notations,
    choose_orientation,
    detect_top_left_is_dark,
    load_json_config,
    load_json_corners,
    refine_corners,
    square_name,
    warp_board,
    warp_board_with_margin,
    board_side,
)


RED = (0, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="DATASET/data")
    parser.add_argument("--out-dir", default="DATASET/piece_crops")
    parser.add_argument("--images", default="all", help="Comma-separated numeric image stems, or 'all'.")
    parser.add_argument("--layout", choices=("by-piece", "by-image"), default="by-piece")
    parser.add_argument("--save-debug", action="store_true", help="Also save annotated crops and contact sheets.")
    parser.add_argument("--occupied-only", action="store_true", help="Export only squares labeled with pieces in JSON.")
    parser.add_argument("--empty-label", default="empty")
    parser.add_argument("--max-empty-per-board", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corner-format", choices=("yx", "xy"), default="yx")
    parser.add_argument("--warp-size", type=int, default=640)
    parser.add_argument("--crop-scale", type=float, default=2.2, help="Crop size as a multiple of one square.")
    parser.add_argument(
        "--warp-margin-scale",
        type=float,
        default=0.8,
        help="Extra warped context around the board, measured in square widths.",
    )
    parser.add_argument("--out-size", type=int, default=192)
    parser.add_argument("--max-move-ratio", type=float, default=0.08)
    parser.add_argument("--steps", default="32,16,8,4,2")
    return parser.parse_args()


def image_stems(data_dir: Path, images_arg: str) -> list[str]:
    if images_arg.strip().lower() != "all":
        return [item.strip() for item in images_arg.split(",") if item.strip()]

    stems = []
    for image_path in data_dir.glob("*.jpg"):
        if (data_dir / f"{image_path.stem}.json").exists():
            stems.append(image_path.stem)

    return sorted(stems, key=lambda stem: (not stem.isdigit(), int(stem) if stem.isdigit() else stem))


def safe_class_name(label: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in label)


def square_to_row_col(target: str, notation: str) -> tuple[int, int]:
    target = target.upper()
    for row in range(8):
        for col in range(8):
            if square_name(row, col, notation) == target:
                return row, col
    raise ValueError(f"Could not map square {target} using notation {notation}")


def crop_with_padding(image: np.ndarray, cx: float, cy: float, size: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    half = size / 2.0
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1 = int(round(cx + half))
    y1 = int(round(cy + half))

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - image.shape[1])
    pad_bottom = max(0, y1 - image.shape[0])

    if any((pad_left, pad_top, pad_right, pad_bottom)):
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top

    return image[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def draw_debug_crop(crop: np.ndarray, square_size: float, crop_scale: float, label: str) -> np.ndarray:
    debug = crop.copy()
    h, w = debug.shape[:2]
    central = square_size / crop_scale
    x0 = int(round((w - central) / 2.0))
    y0 = int(round((h - central) / 2.0))
    x1 = int(round((w + central) / 2.0))
    y1 = int(round((h + central) / 2.0))
    cv2.rectangle(debug, (x0, y0), (x1, y1), RED, 2, cv2.LINE_AA)
    cv2.putText(debug, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2, cv2.LINE_AA)
    return debug


def make_contact_sheet(images: list[np.ndarray], labels: list[str], out_path: Path, thumb_size: int = 128, cols: int = 8) -> None:
    if not images:
        return

    rows = int(np.ceil(len(images) / cols))
    sheet = np.full((rows * thumb_size, cols * thumb_size, 3), 245, dtype=np.uint8)
    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        thumb = cv2.resize(image, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        x0 = col * thumb_size
        y0 = row * thumb_size
        sheet[y0 : y0 + thumb_size, x0 : x0 + thumb_size] = thumb
        cv2.putText(sheet, labels[idx], (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 2, cv2.LINE_AA)
        cv2.putText(sheet, labels[idx], (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def process_image(stem: str, args: argparse.Namespace) -> None:
    data_dir = Path(args.data)
    out_dir = Path(args.out_dir)
    image_path = data_dir / f"{stem}.jpg"
    json_path = data_dir / f"{stem}.json"

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)

    initial = load_json_corners(json_path, image.shape, args.corner_format)
    config = load_json_config(json_path)
    config = {name.upper(): label for name, label in config.items()}
    occupied_names = set(config)

    steps = [int(item) for item in args.steps.split(",") if item.strip()]
    refined, _score = refine_corners(
        image,
        initial,
        args.warp_size,
        steps,
        board_side(initial) * args.max_move_ratio,
    )
    warp_for_orientation = warp_board(image, refined, args.warp_size)
    top_left_is_dark = detect_top_left_is_dark(warp_for_orientation)
    candidates = color_valid_notations(top_left_is_dark)
    notation, orientation_scores = choose_orientation(warp_for_orientation, occupied_names, candidates)

    square = args.warp_size / 8.0
    crop_size = square * args.crop_scale
    warp_margin = int(round(square * args.warp_margin_scale))
    warp = warp_board_with_margin(image, refined, args.warp_size, warp_margin)
    image_dir = out_dir / stem / "piece_crop"
    debug_dir = out_dir / stem / "debug"
    if args.layout == "by-image":
        image_dir.mkdir(parents=True, exist_ok=True)
    if args.save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    debug_images = []
    debug_labels = []
    occupied_labels = sorted(occupied_names)
    empty_labels = [
        square_name(row, col, notation)
        for row in range(8)
        for col in range(8)
        if square_name(row, col, notation) not in occupied_names
    ]
    if args.occupied_only:
        square_labels = occupied_labels
    else:
        rng = random.Random(f"{args.seed}:{stem}")
        rng.shuffle(empty_labels)
        selected_empty = sorted(empty_labels[: max(0, args.max_empty_per_board)])
        square_labels = sorted(occupied_labels + selected_empty)

    for square_label in square_labels:
        row, col = square_to_row_col(square_label, notation)
        cx = warp_margin + (col + 0.5) * square
        cy = warp_margin + (row + 0.5) * square
        crop, _bounds = crop_with_padding(warp, cx, cy, crop_size)
        crop_resized = cv2.resize(crop, (args.out_size, args.out_size), interpolation=cv2.INTER_AREA)

        piece_label = config.get(square_label, args.empty_label)
        file_stem = f"{stem}_{square_label}_{piece_label}"
        if args.layout == "by-piece":
            raw_dir = out_dir / safe_class_name(piece_label)
            raw_dir.mkdir(parents=True, exist_ok=True)
        else:
            raw_dir = image_dir
        cv2.imwrite(str(raw_dir / f"{file_stem}.jpg"), crop_resized)

        if args.save_debug:
            debug = draw_debug_crop(crop_resized, args.out_size, args.crop_scale, f"{square_label} {piece_label}")
            cv2.imwrite(str(debug_dir / f"{file_stem}_debug.jpg"), debug)
            debug_images.append(debug)
            debug_labels.append(square_label)

    if args.save_debug:
        make_contact_sheet(debug_images, debug_labels, out_dir / stem / f"{stem}_contact_sheet.jpg")

    print(f"image={stem}")
    print(f"notation={notation}")
    print("orientation_scores=" + ",".join(f"{name}:{score:.3f}" for name, score in sorted(orientation_scores.items())))
    print(f"occupied_crops={len(occupied_names)}")
    print(f"empty_crops={sum(1 for square_label in square_labels if square_label not in occupied_names)}")
    print(f"exported_crops={len(square_labels)}")
    print(out_dir.resolve())


def main() -> None:
    args = parse_args()
    stems = image_stems(Path(args.data), args.images)
    for stem in stems:
        process_image(stem, args)


if __name__ == "__main__":
    main()
