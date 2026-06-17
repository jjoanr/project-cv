from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from refine_graella import board_side, refine_corners, warp_board_with_margin

# Config

IMAGES_DIR  = Path("Dataset/Figures_reis")
OUT_DIR     = Path("Dataset/Figures_reis/segmentades")
WARP_SIZE   = 640
CROP_SIZE   = 192
CROP_SCALE  = 1.4
# MAX_EMPTY   = 5   # màxim de caselles buides guardades per imatge (per no desbalancejar el dataset)

# Posició inicial d'escacs: fila 0 = rank 8, fila 7 = rank 1, col 0 = A
INITIAL_POSITION: list[list[str]] = [
    ["rook_b",   "knight_b", "bishop_b", "queen_b", "king_b",  "bishop_b", "knight_b", "rook_b"],   # rank 8
    ["pawn_b"] * 8,                                                                                    # rank 7
    ["empty"]  * 8,                                                                                    # rank 6
    ["empty"]  * 8,                                                                                    # rank 5
    ["empty"]  * 8,                                                                                    # rank 4
    ["empty"]  * 8,                                                                                    # rank 3
    ["pawn_w"] * 8,                                                                                    # rank 2
    ["rook_w",   "knight_w", "bishop_w", "queen_w", "king_w",  "bishop_w", "knight_w", "rook_w"],   # rank 1
]

CORNER_LABELS = [
    "A1  (baix-esquerra)",
    "H1  (baix-dreta)",
    "H8  (dalt-dreta)",
    "A8  (dalt-esquerra)",
]


# Selecció interactiva de cantonades

def pick_corners_interactive(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    max_display = 900
    scale = min(1.0, max_display / max(h, w))
    disp_w, disp_h = int(w * scale), int(h * scale)
    display_base = cv2.resize(image, (disp_w, disp_h)) if scale < 1.0 else image.copy()

    clicks: list[tuple[int, int]] = []
    COLORS = [(0, 255, 255), (0, 200, 255), (255, 100, 0), (0, 255, 0)]
    WIN = "Cantonades"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((int(x / scale), int(y / scale)))

    cv2.setMouseCallback(WIN, on_click)

    while True:
        frame = display_base.copy()

        for i, (cx, cy) in enumerate(clicks):
            dx, dy = int(cx * scale), int(cy * scale)
            cv2.circle(frame, (dx, dy), 9, COLORS[i], -1)
            cv2.circle(frame, (dx, dy), 9, (0, 0, 0), 2)
            cv2.putText(frame, CORNER_LABELS[i].split()[0], (dx + 12, dy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS[i], 2, cv2.LINE_AA)

        for i in range(1, len(clicks)):
            p1 = (int(clicks[i-1][0] * scale), int(clicks[i-1][1] * scale))
            p2 = (int(clicks[i][0]   * scale), int(clicks[i][1]   * scale))
            cv2.line(frame, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)
        if len(clicks) == 4:
            p1 = (int(clicks[3][0] * scale), int(clicks[3][1] * scale))
            p2 = (int(clicks[0][0] * scale), int(clicks[0][1] * scale))
            cv2.line(frame, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)

        if len(clicks) < 4:
            msg = f"Clic {len(clicks)+1}/4:  {CORNER_LABELS[len(clicks)]}"
        else:
            msg = "Fet! ENTER/ESPAI per continuar."
        cv2.putText(frame, msg, (14, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, msg, (14, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC — saltar imatge
            cv2.destroyAllWindows()
            return None
        if key in (ord('r'), ord('R')):
            clicks.clear()
        if key in (13, 32) and len(clicks) == 4:
            cv2.destroyAllWindows()
            break

    # [A1, H1, H8, A8] -> [A8, H8, H1, A1] = [TL, TR, BR, BL]
    a1 = np.array(clicks[0], dtype=np.float32)
    h1 = np.array(clicks[1], dtype=np.float32)
    h8 = np.array(clicks[2], dtype=np.float32)
    a8 = np.array(clicks[3], dtype=np.float32)
    return np.array([a8, h8, h1, a1], dtype=np.float32)


# Extracció de crops (caselles)

def extract_square_crops(warped: np.ndarray, warp_size: int, crop_scale: float, margin: int) -> list[list[np.ndarray]]:
    square = warp_size / 8.0
    half = square * crop_scale / 2.0
    crops = []
    for row in range(8):
        row_crops = []
        for col in range(8):
            cx = margin + (col + 0.5) * square
            cy = margin + (row + 0.5) * square
            x0 = max(0, int(round(cx - half)))
            y0 = max(0, int(round(cy - half)))
            x1 = min(warped.shape[1], int(round(cx + half)))
            y1 = min(warped.shape[0], int(round(cy + half)))
            crop = cv2.resize(warped[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE),
                              interpolation=cv2.INTER_AREA)
            row_crops.append(crop)
        crops.append(row_crops)
    return crops


# Main

def main() -> None:
    image_paths = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()
    )

    if not image_paths:
        print(f"No s'han trobat imatges a {IMAGES_DIR}")
        return

    # Imatges ja processades: busca qualsevol crop amb el mateix stem a peces_segmentades
    already_done = {p.stem.rsplit("_r", 1)[0] for p in OUT_DIR.rglob("*.jpg")}

    pending = [p for p in image_paths if p.stem not in already_done]
    skipped = len(image_paths) - len(pending)

    print(f"Trobades {len(image_paths)} imatges ({skipped} ja processades, {len(pending)} pendents).")
    if not pending:
        print("Totes les imatges ja estan processades.")
        return
    print("Ordre de clics: A1 (baix-esquerra) -> H1 -> H8 -> A8 (dalt-esquerra)")
    print("ESC per saltar una imatge. R per reiniciar els clics.")
    print()

    # Comptar quants crops tenim per classe al final
    class_counts: dict[str, int] = {}

    for img_idx, img_path in enumerate(pending):
        print(f"[{img_idx+1}/{len(pending)}] {img_path.name}")

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  No s'ha pogut llegir la imatge, saltant.")
            continue

        # Clic de les cantonades
        corners = pick_corners_interactive(image)
        if corners is None:
            print(f"  Saltada per l'usuari.")
            continue

        # Refinament de cantonades
        max_move = board_side(corners) * 0.08
        corners, score = refine_corners(image, corners, WARP_SIZE, [32, 16, 8, 4, 2], max_move)
        print(f"  Refinament: score={score:.1f}")

        # Warp amb marge
        margin = int(round(WARP_SIZE / 8.0 * 0.8))
        warped = warp_board_with_margin(image, corners, WARP_SIZE, margin)

        # Extracció dels 64 crops
        crops = extract_square_crops(warped, WARP_SIZE, CROP_SCALE, margin)

        """
        # 5. Seleccionar quines caselles buides guardar (màx MAX_EMPTY)
        empty_positions = [
            (row, col)
            for row in range(8)
            for col in range(8)
            if INITIAL_POSITION[row][col] == "empty"
        ]
        rng = random.Random(img_idx)
        selected_empty = set(rng.sample(empty_positions, min(MAX_EMPTY, len(empty_positions))))
        """

        # Guardar crops
        saved = 0
        for row in range(8):
            for col in range(8):
                piece = INITIAL_POSITION[row][col]
                class_dir = OUT_DIR / piece
                class_dir.mkdir(parents=True, exist_ok=True)

                fname = f"{img_path.stem}_r{row}c{col}.jpg"
                out_path = class_dir / fname
                cv2.imwrite(str(out_path), crops[row][col])
                saved += 1
                class_counts[piece] = class_counts.get(piece, 0) + 1

                """
                if piece == "empty" and (row, col) not in selected_empty:
                    continue

                class_dir = OUT_DIR / piece
                class_dir.mkdir(parents=True, exist_ok=True)

                fname = f"{img_path.stem}_r{row}c{col}.jpg"
                out_path = class_dir / fname
                cv2.imwrite(str(out_path), crops[row][col])
                saved += 1
                class_counts[piece] = class_counts.get(piece, 0) + 1
                """

        print(f"  Guardats {saved} crops.")

    print()
    print("=== Resum ===")
    print(f"Sortida: {OUT_DIR.resolve()}")
    print()
    total = 0
    for cls in sorted(class_counts):
        print(f"  {cls:<14}: {class_counts[cls]:>4} imatges")
        total += class_counts[cls]
    print(f"  {'TOTAL':<14}: {total:>4} imatges")


if __name__ == "__main__":
    main()
