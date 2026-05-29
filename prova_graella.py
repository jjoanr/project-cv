"""
Draw a chessboard grid over an original dataset image.

This script is only for diagnostics. It does not crop pieces.

Examples:
  python prova_graella.py --image Dataset/data/0.jpg --json Dataset/data/0.json
  python prova_graella.py --image Dataset/data/0.jpg --json Dataset/data/0.json --mode all
  python prova_graella.py --image Dataset/data/0.jpg --json Dataset/data/0.json --inner-margin 0.08 --labels
  python prova_graella.py --image Dataset/data/0.jpg --json Dataset/data/0.json --labels --notation all
  python prova_graella.py --image Dataset/data/0.jpg --json Dataset/data/0.json --corner-format xy
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


BOARD_CORNERS = np.array(
    [
        [0.0, 0.0],  # top-left
        [1.0, 0.0],  # top-right
        [1.0, 1.0],  # bottom-right
        [0.0, 1.0],  # bottom-left
    ],
    dtype=np.float64,
)


def homography(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Compute H such that dst ~= H @ src."""
    a = []
    for (sx, sy), (dx, dy) in zip(src_pts, dst_pts):
        a.append([-sx, -sy, -1, 0, 0, 0, dx * sx, dx * sy, dx])
        a.append([0, 0, 0, -sx, -sy, -1, dy * sx, dy * sy, dy])
    _, _, vt = np.linalg.svd(np.array(a, dtype=np.float64))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def project(h: np.ndarray, xy) -> tuple[float, float]:
    p = np.array([xy[0], xy[1], 1.0], dtype=np.float64)
    q = h @ p
    return float(q[0] / q[2]), float(q[1] / q[2])


def sort_corners_by_image_position(pts: np.ndarray) -> np.ndarray:
    """Return corners as [top-left, top-right, bottom-right, bottom-left]."""
    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    return np.array(
        [
            pts[np.argmin(s)],
            pts[np.argmax(diff)],
            pts[np.argmax(s)],
            pts[np.argmin(diff)],
        ],
        dtype=np.float64,
    )


def corners_for_mode(pts: np.ndarray, mode: str) -> np.ndarray:
    """
    Return image points corresponding to normalized board corners:
    [top-left, top-right, bottom-right, bottom-left].
    """
    modes = {
        "sorted": sort_corners_by_image_position,
        "raw_0123": lambda p: p[[0, 1, 2, 3]],
        "raw_2301": lambda p: p[[2, 3, 0, 1]],
        "chessrender360": lambda p: p[[2, 3, 1, 0]],
        "chessrender360_alt": lambda p: p[[3, 2, 1, 0]],
    }
    if mode not in modes:
        raise ValueError(f"Unknown mode: {mode}")
    return np.array(modes[mode](pts), dtype=np.float64)


def normalized_corners_to_xy(corners_norm: np.ndarray, width: int, height: int, corner_format: str) -> np.ndarray:
    if corner_format == "yx":
        corners_norm = corners_norm[:, [1, 0]]
    elif corner_format != "xy":
        raise ValueError(f"Unknown corner format: {corner_format}")
    return corners_norm * np.array([width, height], dtype=np.float64)


def draw_polyline(draw: ImageDraw.ImageDraw, points, fill, width: int):
    draw.line(list(points) + [points[0]], fill=fill, width=width)


def draw_sampled_line(draw: ImageDraw.ImageDraw, h: np.ndarray, p0, p1, fill, width: int):
    points = []
    for t in np.linspace(0.0, 1.0, 50):
        x = p0[0] * (1.0 - t) + p1[0] * t
        y = p0[1] * (1.0 - t) + p1[1] * t
        points.append(project(h, (x, y)))
    draw.line(points, fill=fill, width=width)


def square_name(row: int, col: int, notation: str) -> str:
    """
    Convert grid row/col to chess notation.

    These modes are independent from the geometric corner mode. They let us
    test where A1/H1 actually fall in the rendered board.
    """
    if notation == "a8_tl":
        file_idx = col
        rank = 8 - row
    elif notation == "h8_tl":
        file_idx = 7 - col
        rank = 8 - row
    elif notation == "a1_tl":
        file_idx = col
        rank = row + 1
    elif notation == "h1_tl":
        file_idx = 7 - col
        rank = row + 1
    elif notation == "a8_tr":
        file_idx = row
        rank = 8 - col
    elif notation == "h8_tr":
        file_idx = 7 - row
        rank = 8 - col
    elif notation == "a1_tr":
        file_idx = row
        rank = col + 1
    elif notation == "h1_tr":
        file_idx = 7 - row
        rank = col + 1
    else:
        raise ValueError(f"Unknown notation: {notation}")

    return f"{chr(ord('A') + file_idx)}{rank}"


def draw_grid(
    image: Image.Image,
    corners_norm: np.ndarray,
    corner_format: str,
    mode: str,
    inner_margin: float,
    labels: bool,
    notation: str,
) -> Image.Image:
    if not 0.0 <= inner_margin < 0.45:
        raise ValueError("--inner-margin must be in [0, 0.45)")

    w, h_img = image.size
    pts_px = normalized_corners_to_xy(corners_norm, w, h_img, corner_format)
    corners = corners_for_mode(pts_px, mode)

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    board_to_image = homography(BOARD_CORNERS, corners)

    outer = [project(board_to_image, p) for p in BOARD_CORNERS]
    draw_polyline(draw, outer, fill=(255, 220, 0), width=4)

    for idx, (x, y) in enumerate(corners):
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(0, 255, 255), outline=(0, 0, 0), width=2)
        draw.text((x + 10, y + 10), str(idx), fill=(0, 255, 255))

    lo = inner_margin
    hi = 1.0 - inner_margin

    inner = [
        project(board_to_image, (lo, lo)),
        project(board_to_image, (hi, lo)),
        project(board_to_image, (hi, hi)),
        project(board_to_image, (lo, hi)),
    ]
    draw_polyline(draw, inner, fill=(0, 255, 0), width=3)

    # The board has 9 boundary lines in each direction for 8 squares.
    for i in range(9):
        u = lo + (hi - lo) * i / 8.0
        draw_sampled_line(draw, board_to_image, (u, lo), (u, hi), fill=(255, 0, 0), width=3)
        draw_sampled_line(draw, board_to_image, (lo, u), (hi, u), fill=(255, 0, 0), width=3)

    if labels:
        for row in range(8):
            for col in range(8):
                x = lo + (hi - lo) * (col + 0.5) / 8.0
                y = lo + (hi - lo) * (row + 0.5) / 8.0
                px, py = project(board_to_image, (x, y))
                draw.text((px - 10, py - 7), square_name(row, col, notation), fill=(0, 255, 255))

    draw.text(
        (16, 16),
        f"corner_format={corner_format} mode={mode} inner_margin={inner_margin:.3f} notation={notation}",
        fill=(255, 255, 255),
    )
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="Dataset/data/0.jpg")
    parser.add_argument("--json", default="Dataset/data/0.json")
    parser.add_argument("--out-dir", default="Dataset/grid_debug")
    parser.add_argument(
        "--corner-format",
        choices=("yx", "xy"),
        default="yx",
        help="Coordinate order stored in JSON corners. Dataset is y,x by default.",
    )
    parser.add_argument(
        "--mode",
        choices=("sorted", "raw_0123", "raw_2301", "chessrender360", "chessrender360_alt", "all"),
        default="all",
    )
    parser.add_argument(
        "--inner-margin",
        type=float,
        default=0.0,
        help="Perspective-correct inward margin. Use this if corners mark the outer board frame.",
    )
    parser.add_argument("--labels", action="store_true", help="Draw A8..H1 labels at square centers.")
    parser.add_argument(
        "--notation",
        choices=(
            "a8_tl",
            "h8_tl",
            "a1_tl",
            "h1_tl",
            "a8_tr",
            "h8_tr",
            "a1_tr",
            "h1_tr",
            "all",
        ),
        default="a8_tl",
        help="How to map grid cells to chess notation. Suffix tl/tr says where that named corner is.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    json_path = Path(args.json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    with open(json_path, encoding="utf-8") as f:
        ann = json.load(f)
    corners_norm = np.array(ann["corners"], dtype=np.float64)

    modes = ["sorted", "raw_0123", "raw_2301", "chessrender360", "chessrender360_alt"]
    if args.mode != "all":
        modes = [args.mode]

    notations = ["a8_tl", "h8_tl", "a1_tl", "h1_tl", "a8_tr", "h8_tr", "a1_tr", "h1_tr"]
    if args.notation != "all":
        notations = [args.notation]

    stem = image_path.stem
    for mode in modes:
        for notation in notations:
            out = draw_grid(image, corners_norm, args.corner_format, mode, args.inner_margin, args.labels, notation)
            out_path = out_dir / f"{stem}_{mode}_{notation}_margin{args.inner_margin:.3f}.jpg"
            out.save(out_path, quality=95)
            print(out_path)


if __name__ == "__main__":
    main()
