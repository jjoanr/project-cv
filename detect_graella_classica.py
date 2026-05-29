"""
Detect chessboard grid candidates with classical computer vision.

Diagnostic script only: it does not crop pieces and it does not trust the JSON
corners as the final grid. It uses Canny + probabilistic Hough, separates the
two dominant line directions, clusters parallel lines, and writes debug images.

Example:
  python detect_graella_classica.py --image Dataset/data/0.jpg --json Dataset/data/0.json
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


RED = (0, 0, 255)
GREEN = (0, 255, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
MAGENTA = (255, 0, 255)


def load_image(path: Path, max_side: int):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)

    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return image, scale


def json_corners_px(json_path: Path | None, image_shape, corner_format: str = "yx"):
    if json_path is None:
        return None

    with open(json_path, encoding="utf-8") as f:
        ann = json.load(f)

    h, w = image_shape[:2]
    corners = np.array(ann["corners"], dtype=np.float32)
    if corner_format == "yx":
        corners = corners[:, [1, 0]]
    elif corner_format != "xy":
        raise ValueError(f"Unknown corner format: {corner_format}")
    return corners * np.array([w, h], dtype=np.float32)


def draw_json_points(image, json_path: Path | None, corner_format: str = "yx"):
    pts = json_corners_px(json_path, image.shape, corner_format)
    if pts is None:
        return

    for idx, (x, y) in enumerate(pts):
        cv2.circle(image, (round(x), round(y)), 8, CYAN, -1)
        cv2.putText(image, str(idx), (round(x) + 10, round(y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)


def sort_corners_by_image_position(pts: np.ndarray) -> np.ndarray:
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


def mask_edges_to_json_roi(edges, json_path: Path | None, image_shape, expand: float, corner_format: str = "yx"):
    pts = json_corners_px(json_path, image_shape, corner_format)
    if pts is None:
        return edges

    pts = sort_corners_by_image_position(pts)
    center = pts.mean(axis=0)
    expanded = center + (pts - center) * expand
    mask = np.zeros(edges.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(expanded).astype(np.int32), 255)
    return cv2.bitwise_and(edges, mask)


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160, apertureSize=3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    return gray, edges


def polygon_area(pts: np.ndarray) -> float:
    return float(abs(cv2.contourArea(pts.astype(np.float32))))


def score_quad(quad: np.ndarray, image_shape) -> float:
    h, w = image_shape[:2]
    image_area = h * w
    area = polygon_area(quad)
    if area <= image_area * 0.04:
        return -1.0

    ordered = sort_corners_by_image_position(quad.reshape(4, 2))
    widths = [
        np.linalg.norm(ordered[1] - ordered[0]),
        np.linalg.norm(ordered[2] - ordered[3]),
    ]
    heights = [
        np.linalg.norm(ordered[3] - ordered[0]),
        np.linalg.norm(ordered[2] - ordered[1]),
    ]
    mean_w = float(np.mean(widths))
    mean_h = float(np.mean(heights))
    if mean_w <= 1 or mean_h <= 1:
        return -1.0

    aspect = mean_w / mean_h
    aspect_penalty = abs(math.log(max(aspect, 1e-6)))
    center = ordered.mean(axis=0)
    image_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    center_penalty = float(np.linalg.norm(center - image_center) / max(w, h))

    # Chess boards are large, roughly square quadrilaterals, usually near the image center.
    return area / image_area - 0.35 * aspect_penalty - 0.25 * center_penalty


def detect_board_quads(image, edges, max_candidates=12):
    contours, _hier = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    image_area = image.shape[0] * image.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.03:
            continue

        peri = cv2.arcLength(contour, True)
        for eps_factor in (0.015, 0.025, 0.04, 0.06):
            approx = cv2.approxPolyDP(contour, eps_factor * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            quad = approx.reshape(4, 2).astype(np.float32)
            score = score_quad(quad, image.shape)
            if score > 0:
                candidates.append((score, sort_corners_by_image_position(quad)))

    # Remove near-duplicates by center and area.
    candidates.sort(key=lambda item: item[0], reverse=True)
    unique = []
    for score, quad in candidates:
        center = quad.mean(axis=0)
        area = polygon_area(quad)
        duplicate = False
        for _prev_score, prev_quad in unique:
            prev_center = prev_quad.mean(axis=0)
            prev_area = polygon_area(prev_quad)
            if np.linalg.norm(center - prev_center) < 25 and abs(area - prev_area) / max(area, prev_area) < 0.15:
                duplicate = True
                break
        if not duplicate:
            unique.append((score, quad))
        if len(unique) >= max_candidates:
            break
    return unique


def draw_quad(image, quad, color, thickness=3, label=None):
    pts = np.round(quad).astype(np.int32)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    for idx, (x, y) in enumerate(pts):
        cv2.circle(image, (int(x), int(y)), 6, color, -1)
        cv2.putText(image, str(idx), (int(x) + 8, int(y) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if label:
        x, y = pts[0]
        cv2.putText(image, label, (int(x), int(y) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def make_board_detection_diagnostics(args):
    image_path = Path(args.image)
    json_path = Path(args.json) if args.json else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image, _scale = load_image(image_path, args.max_side)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, args.canny_low, args.canny_high, apertureSize=3)

    # Closing helps recover the continuous outer board frame.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=args.close_iterations)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.dilate(closed, dilate_kernel, iterations=args.dilate_iterations)

    quads = detect_board_quads(image, closed, max_candidates=args.max_candidates)

    stem = image_path.stem
    cv2.imwrite(str(out_dir / f"{stem}_board_01_edges.jpg"), edges)
    cv2.imwrite(str(out_dir / f"{stem}_board_02_closed.jpg"), closed)

    candidates_img = image.copy()
    draw_json_points(candidates_img, json_path, args.corner_format)
    colors = [MAGENTA, YELLOW, RED, GREEN, CYAN, WHITE]
    for idx, (score, quad) in enumerate(quads):
        draw_quad(
            candidates_img,
            quad,
            colors[idx % len(colors)],
            thickness=2 if idx else 4,
            label=f"#{idx} score={score:.3f}",
        )
    cv2.imwrite(str(out_dir / f"{stem}_board_03_candidates.jpg"), candidates_img)

    best_img = image.copy()
    draw_json_points(best_img, json_path, args.corner_format)
    if quads:
        score, best = quads[0]
        draw_quad(best_img, best, MAGENTA, thickness=5, label=f"best score={score:.3f}")
    else:
        cv2.putText(best_img, "No board quad found", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2)
    cv2.imwrite(str(out_dir / f"{stem}_board_04_best.jpg"), best_img)

    print(f"board_candidates={len(quads)}")
    if quads:
        print(f"best_score={quads[0][0]:.4f}")
    print(out_dir.resolve())


def line_angle_deg(x1, y1, x2, y2):
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if angle < 0:
        angle += 180.0
    return angle


def detect_segments(edges, min_line_length, max_line_gap):
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=70,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return []

    segments = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, line)
        length = math.hypot(x2 - x1, y2 - y1)
        angle = line_angle_deg(x1, y1, x2, y2)
        segments.append((x1, y1, x2, y2, length, angle))
    return segments


def dominant_angle_peaks(segments, bins=180):
    hist = np.zeros(bins, dtype=np.float64)
    for *_coords, length, angle in segments:
        hist[int(round(angle)) % bins] += length

    if hist.max() == 0:
        return 0.0, 90.0

    first = int(hist.argmax())
    masked = hist.copy()
    for offset in range(-25, 26):
        masked[(first + offset) % bins] = 0
    second = int(masked.argmax())
    return float(first), float(second)


def angle_distance(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def split_families(segments, peak_a, peak_b, tolerance):
    family_a = []
    family_b = []
    for segment in segments:
        da = angle_distance(segment[5], peak_a)
        db = angle_distance(segment[5], peak_b)
        if da <= tolerance and da <= db:
            family_a.append(segment)
        elif db <= tolerance:
            family_b.append(segment)
    return family_a, family_b


def draw_segments(image, segments, color, thickness=2):
    for x1, y1, x2, y2, *_rest in segments:
        cv2.line(image, (round(x1), round(y1)), (round(x2), round(y2)), color, thickness, cv2.LINE_AA)


def line_from_segment(segment):
    x1, y1, x2, y2, *_rest = segment
    dx = x2 - x1
    dy = y2 - y1
    theta = math.atan2(dy, dx) + math.pi / 2.0
    rho = x1 * math.cos(theta) + y1 * math.sin(theta)
    if rho < 0:
        rho = -rho
        theta += math.pi
    return rho, theta % math.pi


def cluster_parallel_lines(segments, rho_gap):
    if not segments:
        return []

    lines = []
    for segment in segments:
        rho, theta = line_from_segment(segment)
        lines.append((rho, theta, segment[4]))
    lines.sort(key=lambda item: item[0])

    clusters = []
    current = [lines[0]]
    for item in lines[1:]:
        center = float(np.mean([x[0] for x in current]))
        if abs(item[0] - center) <= rho_gap:
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
    clusters.append(current)

    merged = []
    for cluster in clusters:
        weights = np.array([item[2] for item in cluster], dtype=np.float64)
        rhos = np.array([item[0] for item in cluster], dtype=np.float64)
        thetas = np.array([item[1] for item in cluster], dtype=np.float64)
        merged.append((
            float(np.average(rhos, weights=weights)),
            float(np.average(thetas, weights=weights)),
            float(weights.sum()),
        ))
    return sorted(merged, key=lambda item: item[0])


def draw_infinite_line(image, rho, theta, color, thickness=2):
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x0 = cos_t * rho
    y0 = sin_t * rho
    dx = -sin_t
    dy = cos_t
    scale = max(image.shape[:2]) * 2
    pt1 = (round(x0 + dx * scale), round(y0 + dy * scale))
    pt2 = (round(x0 - dx * scale), round(y0 - dy * scale))
    cv2.line(image, pt1, pt2, color, thickness, cv2.LINE_AA)


def draw_clustered_lines(image, lines, color):
    for rho, theta, score in lines:
        draw_infinite_line(image, rho, theta, color, 1 if score < 300 else 2)


def make_diagnostics(args):
    image_path = Path(args.image)
    json_path = Path(args.json) if args.json else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image, _scale = load_image(image_path, args.max_side)
    _gray, edges = preprocess(image)
    edges = mask_edges_to_json_roi(edges, json_path, image.shape, args.roi_expand, args.corner_format)

    min_len = args.min_line_length or int(min(image.shape[:2]) * 0.13)
    max_gap = args.max_line_gap or int(min(image.shape[:2]) * 0.035)
    segments = detect_segments(edges, min_len, max_gap)
    peak_a, peak_b = dominant_angle_peaks(segments)
    family_a, family_b = split_families(segments, peak_a, peak_b, args.angle_tolerance)

    rho_gap = args.rho_gap or int(min(image.shape[:2]) * 0.025)
    lines_a = cluster_parallel_lines(family_a, rho_gap)
    lines_b = cluster_parallel_lines(family_b, rho_gap)

    stem = image_path.stem
    cv2.imwrite(str(out_dir / f"{stem}_01_edges.jpg"), edges)

    all_segments = image.copy()
    draw_segments(all_segments, segments, WHITE, 1)
    draw_json_points(all_segments, json_path, args.corner_format)
    cv2.imwrite(str(out_dir / f"{stem}_02_all_hough_segments.jpg"), all_segments)

    families = image.copy()
    draw_segments(families, family_a, RED, 2)
    draw_segments(families, family_b, GREEN, 2)
    draw_json_points(families, json_path, args.corner_format)
    cv2.putText(
        families,
        f"peaks: {peak_a:.0f} deg / {peak_b:.0f} deg",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        YELLOW,
        2,
    )
    cv2.imwrite(str(out_dir / f"{stem}_03_line_families.jpg"), families)

    clustered = image.copy()
    draw_clustered_lines(clustered, lines_a, RED)
    draw_clustered_lines(clustered, lines_b, GREEN)
    draw_json_points(clustered, json_path, args.corner_format)
    cv2.putText(
        clustered,
        f"clusters: red={len(lines_a)} green={len(lines_b)}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        YELLOW,
        2,
    )
    cv2.imwrite(str(out_dir / f"{stem}_04_clustered_lines.jpg"), clustered)

    print(f"segments={len(segments)} family_a={len(family_a)} family_b={len(family_b)}")
    print(f"peaks={peak_a:.1f},{peak_b:.1f} clustered={len(lines_a)},{len(lines_b)}")
    print(out_dir.resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="Dataset/data/0.jpg")
    parser.add_argument("--json", default=None, help="Optional JSON, only drawn as reference points.")
    parser.add_argument("--out-dir", default="Dataset/classical_grid_debug")
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument(
        "--corner-format",
        choices=("yx", "xy"),
        default="yx",
        help="Coordinate order stored in JSON corners. Dataset is y,x by default.",
    )
    parser.add_argument("--min-line-length", type=int, default=None)
    parser.add_argument("--max-line-gap", type=int, default=None)
    parser.add_argument("--angle-tolerance", type=float, default=18.0)
    parser.add_argument("--rho-gap", type=int, default=None)
    parser.add_argument("--detect-board", action="store_true", help="Ignore JSON geometry and detect board quadrilaterals.")
    parser.add_argument("--canny-low", type=int, default=45)
    parser.add_argument("--canny-high", type=int, default=130)
    parser.add_argument("--close-iterations", type=int, default=2)
    parser.add_argument("--dilate-iterations", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument(
        "--roi-expand",
        type=float,
        default=1.08,
        help="If --json is provided, expand its quadrilateral by this factor and ignore edges outside it.",
    )
    args = parser.parse_args()
    if args.detect_board:
        make_board_detection_diagnostics(args)
    else:
        make_diagnostics(args)


if __name__ == "__main__":
    main()
