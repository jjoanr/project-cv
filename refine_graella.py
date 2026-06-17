from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


YELLOW = (0, 255, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)


BOARD_UNIT = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ],
    dtype=np.float32,
)

NOTATIONS = ["a8_tl", "h8_tl", "a1_tl", "h1_tl", "a8_tr", "h8_tr", "a1_tr", "h1_tr"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="Dataset/data/0.jpg")
    parser.add_argument("--json", default="Dataset/data/0.json")
    parser.add_argument("--out-dir", default="Dataset/refined_grid_debug")
    parser.add_argument(
        "--corner-format",
        choices=("yx", "xy"),
        default="yx",
        help="Coordinate order stored in JSON corners. Dataset is y,x by default.",
    )
    parser.add_argument("--warp-size", type=int, default=640)
    parser.add_argument(
        "--max-move-ratio",
        type=float,
        default=0.08,
        help="Maximum corner movement as a fraction of the estimated board side.",
    )
    parser.add_argument(
        "--steps",
        default="32,16,8,4,2",
        help="Comma-separated local-search steps in original-image pixels.",
    )
    parser.add_argument("--labels", action="store_true")
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
            "color_valid",
            "auto",
        ),
        default="a8_tl",
        help=(
            "How to map grid cells to chess notation. Suffix tl/tr says where that named corner is. "
            "Use color_valid to keep only orientations where A1 is on a dark square. "
            "Use auto to choose the best color-valid orientation from JSON occupancy."
        ),
    )
    return parser.parse_args()


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
        dtype=np.float32,
    )


def load_json_corners(json_path: Path, image_shape, corner_format: str) -> np.ndarray:
    with json_path.open(encoding="utf-8") as f:
        ann = json.load(f)

    h, w = image_shape[:2]
    corners = np.array(ann["corners"], dtype=np.float32)
    if corner_format == "yx":
        corners = corners[:, [1, 0]]
    elif corner_format != "xy":
        raise ValueError(f"Unknown corner format: {corner_format}")

    return sort_corners_by_image_position(corners * np.array([w, h], dtype=np.float32))


def load_json_config(json_path: Path) -> dict[str, str]:
    with json_path.open(encoding="utf-8") as f:
        ann = json.load(f)
    return dict(ann.get("config", {}))


def board_side(corners: np.ndarray) -> float:
    lengths = [
        np.linalg.norm(corners[1] - corners[0]),
        np.linalg.norm(corners[2] - corners[1]),
        np.linalg.norm(corners[2] - corners[3]),
        np.linalg.norm(corners[3] - corners[0]),
    ]
    return float(np.mean(lengths))


def destination_square(warp_size: int) -> np.ndarray:
    last = float(warp_size - 1)
    return np.array([[0, 0], [last, 0], [last, last], [0, last]], dtype=np.float32)


def warp_board(image: np.ndarray, corners: np.ndarray, warp_size: int) -> np.ndarray:
    h = cv2.getPerspectiveTransform(corners.astype(np.float32), destination_square(warp_size))
    return cv2.warpPerspective(image, h, (warp_size, warp_size), flags=cv2.INTER_LINEAR)


def warp_board_with_margin(image: np.ndarray, corners: np.ndarray, warp_size: int, margin: int) -> np.ndarray:
    output_size = warp_size + 2 * margin
    last = float(warp_size - 1)
    dst = np.array(
        [
            [margin, margin],
            [margin + last, margin],
            [margin + last, margin + last],
            [margin, margin + last],
        ],
        dtype=np.float32,
    )
    h = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(image, h, (output_size, output_size), flags=cv2.INTER_LINEAR)


def edge_maps(warped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.abs(grad_x), np.abs(grad_y)


def line_band_score(profile: np.ndarray, center: float, radius: int) -> float:
    lo = max(0, int(round(center)) - radius)
    hi = min(len(profile), int(round(center)) + radius + 1)
    if hi <= lo:
        return 0.0
    return float(profile[lo:hi].mean())


def score_corners(image: np.ndarray, corners: np.ndarray, initial: np.ndarray, warp_size: int, max_move: float) -> float:
    movement = np.linalg.norm(corners - initial, axis=1)
    if float(movement.max()) > max_move:
        return -1e9

    area = cv2.contourArea(corners.astype(np.float32))
    if area <= 100:
        return -1e9

    warped = warp_board(image, corners, warp_size)
    grad_x, grad_y = edge_maps(warped)

    # Ignore a little of the outside border (often contains the wooden frame)
    margin = max(4, warp_size // 80)
    gx = grad_x[margin : warp_size - margin, :]
    gy = grad_y[:, margin : warp_size - margin]
    vertical_profile = gx.mean(axis=0)
    horizontal_profile = gy.mean(axis=1)

    radius = max(1, warp_size // 220)
    positions = np.linspace(0, warp_size - 1, 9)
    vertical_scores = [line_band_score(vertical_profile, pos, radius) for pos in positions]
    horizontal_scores = [line_band_score(horizontal_profile, pos, radius) for pos in positions]

    weights = np.array([0.65, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.65], dtype=np.float32)
    data_score = float(np.dot(vertical_scores, weights) + np.dot(horizontal_scores, weights))
    close_penalty = float(np.mean((movement / max_move) ** 2)) if max_move > 0 else 0.0
    return data_score - 12.0 * close_penalty


def refine_corners(image: np.ndarray, initial: np.ndarray, warp_size: int, steps: list[int], max_move: float) -> tuple[np.ndarray, float]:
    best = initial.copy()
    best_score = score_corners(image, best, initial, warp_size, max_move)
    directions = [
        (0, 0),
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    for step in steps:
        improved = True
        while improved:
            improved = False
            for corner_idx in range(4):
                local_best = best.copy()
                local_score = best_score
                for dx, dy in directions:
                    candidate = best.copy()
                    candidate[corner_idx] += np.array([dx * step, dy * step], dtype=np.float32)
                    candidate_score = score_corners(image, candidate, initial, warp_size, max_move)
                    if candidate_score > local_score:
                        local_best = candidate
                        local_score = candidate_score
                if local_score > best_score:
                    best = local_best
                    best_score = local_score
                    improved = True

    return best, best_score


def project_grid_points(corners: np.ndarray, warp_size: int) -> list[list[tuple[float, float]]]:
    src = destination_square(warp_size)
    h_inv = cv2.getPerspectiveTransform(src, corners.astype(np.float32))
    points = []
    positions = np.linspace(0, warp_size - 1, 9)
    for y in positions:
        row = []
        for x in positions:
            p = np.array([x, y, 1.0], dtype=np.float32)
            q = h_inv @ p
            row.append((float(q[0] / q[2]), float(q[1] / q[2])))
        points.append(row)
    return points


def square_name(row: int, col: int, notation: str) -> str:
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


def square_is_dark(name: str) -> bool:
    file_idx = ord(name[0].upper()) - ord("A") + 1
    rank = int(name[1])
    return (file_idx + rank) % 2 == 0


def notation_top_left_is_dark(notation: str) -> bool:
    return square_is_dark(square_name(0, 0, notation))


def detect_top_left_is_dark(warped: np.ndarray) -> bool:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    size = warped.shape[0]
    cell = size / 8.0
    even_values = []
    odd_values = []

    # Sample away from the square center because pieces often occupy it.
    offsets = ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))
    patch_radius = max(2, round(cell * 0.055))
    for row in range(8):
        for col in range(8):
            samples = []
            for ox, oy in offsets:
                cx = round((col + ox) * cell)
                cy = round((row + oy) * cell)
                x0 = max(0, cx - patch_radius)
                x1 = min(size, cx + patch_radius + 1)
                y0 = max(0, cy - patch_radius)
                y1 = min(size, cy + patch_radius + 1)
                samples.append(float(np.median(gray[y0:y1, x0:x1])))
            value = float(np.median(samples))
            if (row + col) % 2 == 0:
                even_values.append(value)
            else:
                odd_values.append(value)

    even_median = float(np.median(even_values))
    odd_median = float(np.median(odd_values))
    return even_median < odd_median


def color_valid_notations(top_left_is_dark: bool) -> list[str]:
    return [notation for notation in NOTATIONS if notation_top_left_is_dark(notation) == top_left_is_dark]


def cell_occupancy_scores(warped: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)

    size = warped.shape[0]
    cell = size / 8.0
    scores = np.zeros((8, 8), dtype=np.float32)

    for row in range(8):
        for col in range(8):
            x0 = round((col + 0.18) * cell)
            x1 = round((col + 0.82) * cell)
            y0 = round((row + 0.18) * cell)
            y1 = round((row + 0.82) * cell)
            cx0 = round((col + 0.28) * cell)
            cx1 = round((col + 0.72) * cell)
            cy0 = round((row + 0.28) * cell)
            cy1 = round((row + 0.72) * cell)

            inner_grad = grad_mag[y0:y1, x0:x1]
            center_gray = gray[cy0:cy1, cx0:cx1]
            center_sat = sat[cy0:cy1, cx0:cx1]

            edge_score = float(np.percentile(inner_grad, 82))
            texture_score = float(np.std(center_gray))
            sat_score = float(np.std(center_sat))
            scores[row, col] = edge_score + 0.55 * texture_score + 0.2 * sat_score

    return scores


def orientation_score(scores: np.ndarray, occupied_names: set[str], notation: str) -> float:
    occupied_scores = []
    empty_scores = []
    for row in range(8):
        for col in range(8):
            name = square_name(row, col, notation)
            if name in occupied_names:
                occupied_scores.append(float(scores[row, col]))
            else:
                empty_scores.append(float(scores[row, col]))

    if not occupied_scores or not empty_scores:
        return -1e9

    occ = np.array(occupied_scores, dtype=np.float32)
    empty = np.array(empty_scores, dtype=np.float32)
    separation = float(occ.mean() - empty.mean())

    # A rank-like pairwise term is less sensitive to a few very textured empty squares.
    pairwise = float((occ[:, None] > empty[None, :]).mean())
    return separation + 35.0 * (pairwise - 0.5)


def choose_orientation(warped: np.ndarray, occupied_names: set[str], candidates: list[str]) -> tuple[str, dict[str, float]]:
    scores = cell_occupancy_scores(warped)
    orientation_scores = {
        notation: orientation_score(scores, occupied_names, notation)
        for notation in candidates
    }
    best = max(orientation_scores, key=orientation_scores.get)
    return best, orientation_scores


def draw_grid(
    image: np.ndarray,
    corners: np.ndarray,
    color,
    thickness: int,
    labels: bool = False,
    notation: str = "a8_tl",
) -> np.ndarray:
    out = image.copy()
    grid = project_grid_points(corners, 640)

    for row in grid:
        pts = np.round(row).astype(np.int32)
        cv2.polylines(out, [pts], False, color, thickness, cv2.LINE_AA)
    for col_idx in range(9):
        pts = np.round([grid[row_idx][col_idx] for row_idx in range(9)]).astype(np.int32)
        cv2.polylines(out, [pts], False, color, thickness, cv2.LINE_AA)

    corner_pts = np.round(corners).astype(np.int32)
    cv2.polylines(out, [corner_pts], True, color, max(thickness + 1, 3), cv2.LINE_AA)
    for idx, (x, y) in enumerate(corner_pts):
        cv2.circle(out, (int(x), int(y)), 7, color, -1, cv2.LINE_AA)
        cv2.putText(out, str(idx), (int(x) + 9, int(y) + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if labels:
        for row in range(8):
            for col in range(8):
                x0, y0 = grid[row][col]
                x1, y1 = grid[row + 1][col + 1]
                name = square_name(row, col, notation)
                cv2.putText(
                    out,
                    name,
                    (round((x0 + x1) * 0.5) - 10, round((y0 + y1) * 0.5) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    CYAN,
                    1,
                    cv2.LINE_AA,
                )
    return out


def draw_warped_grid(warped: np.ndarray, labels: bool = False, notation: str = "a8_tl") -> np.ndarray:
    out = warped.copy()
    size = warped.shape[0]
    positions = np.linspace(0, size - 1, 9).round().astype(int)
    for pos in positions:
        cv2.line(out, (int(pos), 0), (int(pos), size - 1), RED, 2, cv2.LINE_AA)
        cv2.line(out, (0, int(pos)), (size - 1, int(pos)), RED, 2, cv2.LINE_AA)

    if labels:
        cell = size / 8.0
        for row in range(8):
            for col in range(8):
                name = square_name(row, col, notation)
                center_x = int(round((col + 0.5) * cell))
                center_y = int(round((row + 0.5) * cell))
                cv2.putText(
                    out,
                    name,
                    (center_x - 13, center_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    CYAN,
                    2,
                    cv2.LINE_AA,
                )
    return out


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    json_path = Path(args.json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)

    initial = load_json_corners(json_path, image.shape, args.corner_format)
    config = load_json_config(json_path)
    occupied_names = {name.upper() for name in config}
    side = board_side(initial)
    max_move = side * args.max_move_ratio
    steps = [int(item) for item in args.steps.split(",") if item.strip()]

    refined, refined_score = refine_corners(image, initial, args.warp_size, steps, max_move)
    initial_score = score_corners(image, initial, initial, args.warp_size, max_move)

    stem = image_path.stem
    refined_warp = warp_board(image, refined, args.warp_size)
    top_left_is_dark = detect_top_left_is_dark(refined_warp)

    if args.notation in ("color_valid", "auto"):
        notations = color_valid_notations(top_left_is_dark)
    else:
        notations = NOTATIONS.copy()

    orientation_scores = {}
    selected_notation = None
    if args.notation == "auto":
        selected_notation, orientation_scores = choose_orientation(refined_warp, occupied_names, notations)
        notations = [selected_notation]

    if args.notation not in ("all", "color_valid", "auto"):
        notations = [args.notation]
    primary_notation = notations[0]

    initial_overlay = draw_grid(image, initial, YELLOW, 2, labels=False)
    refined_overlay = draw_grid(image, refined, RED, 2, labels=args.labels, notation=primary_notation)
    comparison = draw_grid(initial_overlay, refined, RED, 2, labels=args.labels, notation=primary_notation)
    cv2.putText(comparison, "yellow=json init  red=refined", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, WHITE, 2)

    cv2.imwrite(str(out_dir / f"{stem}_01_initial_grid.jpg"), initial_overlay)
    cv2.imwrite(str(out_dir / f"{stem}_02_refined_grid.jpg"), refined_overlay)
    cv2.imwrite(str(out_dir / f"{stem}_03_compare.jpg"), comparison)
    cv2.imwrite(
        str(out_dir / f"{stem}_04_initial_warp.jpg"),
        draw_warped_grid(warp_board(image, initial, args.warp_size), labels=args.labels, notation=primary_notation),
    )
    cv2.imwrite(
        str(out_dir / f"{stem}_05_refined_warp.jpg"),
        draw_warped_grid(warp_board(image, refined, args.warp_size), labels=args.labels, notation=primary_notation),
    )

    if args.labels and args.notation in ("all", "color_valid"):
        for notation in notations:
            cv2.imwrite(
                str(out_dir / f"{stem}_05_refined_warp_{notation}.jpg"),
                draw_warped_grid(refined_warp, labels=True, notation=notation),
            )

    movement = np.linalg.norm(refined - initial, axis=1)
    print(f"image={image_path}")
    print(f"initial_score={initial_score:.3f} refined_score={refined_score:.3f}")
    print(f"top_left_square_color={'dark' if top_left_is_dark else 'light'}")
    print("notation_candidates=" + ",".join(notations))
    if selected_notation:
        print(f"selected_notation={selected_notation}")
        print(
            "orientation_scores="
            + ",".join(f"{notation}:{score:.3f}" for notation, score in sorted(orientation_scores.items()))
        )
    print("corner_movement_px=" + ",".join(f"{value:.1f}" for value in movement))
    print("refined_corners_xy=" + json.dumps(refined.tolist()))
    print(out_dir.resolve())


if __name__ == "__main__":
    main()
