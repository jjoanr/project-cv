from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms

from refine_graella import (
    board_side,
    refine_corners,
    square_name,
    warp_board,
    warp_board_with_margin,
)


# Constants

PIECE_TO_FEN = {
    "king_w": "K", "queen_w": "Q", "rook_w": "R",
    "bishop_w": "B", "knight_w": "N", "pawn_w": "P",
    "king_b": "k", "queen_b": "q", "rook_b": "r",
    "bishop_b": "b", "knight_b": "n", "pawn_b": "p",
    "empty": ".",
}

PIECE_UNICODE = {
    "king_w": "♔", "queen_w": "♕", "rook_w": "♖",
    "bishop_w": "♗", "knight_w": "♘", "pawn_w": "♙",
    "king_b": "♚", "queen_b": "♛", "rook_b": "♜",
    "bishop_b": "♝", "knight_b": "♞", "pawn_b": "♟",
}

SQ_LIGHT    = (240, 217, 181)
SQ_DARK     = (119, 149,  86)
BOARD_CELL  = 80
BOARD_MARGIN = 28

# Corner labels shown during interactive selection
CORNER_LABELS = ["A1  (baix-esquerra)", "H1  (baix-dreta)", "H8  (dalt-dreta)", "A8  (dalt-esquerra)"]

# Model

class PrototypicalNet(nn.Module):
    def __init__(self, backbone_name: str = "resnet18"):
        super().__init__()
        if backbone_name == "resnet18":
            backbone = models.resnet18(weights=None)
        elif backbone_name == "resnet50":
            backbone = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# Interactive corner selection

def pick_corners_interactive(image: np.ndarray) -> np.ndarray:
    """
    Open a window and ask the user to click 4 corners in order:
    A1, H1, H8, A8.

    Returns corners as (4, 2) float32 array ordered [A8, H8, H1, A1],
    which is [TL, TR, BR, BL] in the warped output    
    """

    h, w = image.shape[:2]
    max_display = 900
    scale = min(1.0, max_display / max(h, w))
    disp_w, disp_h = int(w * scale), int(h * scale)
    display_base = cv2.resize(image, (disp_w, disp_h)) if scale < 1.0 else image.copy()

    clicks: list[tuple[int, int]] = []  # original-image pixel coords

    COLORS = [(0, 255, 255), (0, 200, 255), (255, 100, 0), (0, 255, 0)]
    WIN = "Seleccio de cantonades - R: reiniciar  |  ESC: cancellar"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((int(x / scale), int(y / scale)))

    cv2.setMouseCallback(WIN, on_click)

    while True:
        frame = display_base.copy()

        # Draw completed clicks
        for i, (cx, cy) in enumerate(clicks):
            dx, dy = int(cx * scale), int(cy * scale)
            cv2.circle(frame, (dx, dy), 9, COLORS[i], -1)
            cv2.circle(frame, (dx, dy), 9, (0, 0, 0), 2)
            cv2.putText(frame, CORNER_LABELS[i].split()[0], (dx + 12, dy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS[i], 2, cv2.LINE_AA)

        # Draw lines connecting clicks in order
        for i in range(1, len(clicks)):
            p1 = (int(clicks[i-1][0] * scale), int(clicks[i-1][1] * scale))
            p2 = (int(clicks[i][0]   * scale), int(clicks[i][1]   * scale))
            cv2.line(frame, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)
        if len(clicks) == 4:
            p1 = (int(clicks[3][0] * scale), int(clicks[3][1] * scale))
            p2 = (int(clicks[0][0] * scale), int(clicks[0][1] * scale))
            cv2.line(frame, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)

        # Instruction
        if len(clicks) < 4:
            msg = f"Clic {len(clicks)+1}/4:  {CORNER_LABELS[len(clicks)]}"
        else:
            msg = "Fet! Prem ENTER o ESPAI per continuar."
        cv2.putText(frame, msg, (14, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, msg, (14, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC — cancel
            cv2.destroyAllWindows()
            raise RuntimeError("Selecció de cantonades cancel·lada.")
        if key in (ord('r'), ord('R')):
            clicks.clear()
        if key in (13, 32) and len(clicks) == 4:  # ENTER o ESPAI — continuar
            cv2.destroyAllWindows()
            break

    # clicks = [A1, H1, H8, A8]
    # Rearrange to [A8, H8, H1, A1] = [TL, TR, BR, BL] for "a8_tl" warp
    a1 = np.array(clicks[0], dtype=np.float32)
    h1 = np.array(clicks[1], dtype=np.float32)
    h8 = np.array(clicks[2], dtype=np.float32)
    a8 = np.array(clicks[3], dtype=np.float32)
    return np.array([a8, h8, h1, a1], dtype=np.float32)


# Support set (k-shot learning)

INFER_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_support_images(support_dir: Path, k_shot: int, seed: int) -> dict[str, list[np.ndarray]]:
    rng = random.Random(seed)
    support: dict[str, list[np.ndarray]] = {}
    for class_dir in sorted(support_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        paths = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if not paths:
            continue
        rng.shuffle(paths)
        
        imgs = []
        for p in paths[:k_shot]:
            img = cv2.imread(str(p))
            if img is not None:
                # Convert BGR to RGB so it matches training data format
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(img)
                
        if imgs:
            support[class_dir.name] = imgs
    return support


@torch.no_grad()
def compute_prototypes(
    model: PrototypicalNet,
    support: dict[str, list[np.ndarray]],
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    class_names = sorted(support.keys())
    protos = []
    for cls in class_names:
        embeddings = [model(INFER_TRANSFORM(img).unsqueeze(0).to(device)) for img in support[cls]]
        protos.append(torch.stack(embeddings).mean(0))
    return torch.cat(protos, dim=0), class_names


# Square extraction

def extract_square_crops(
    warped: np.ndarray,
    warp_size: int,
    crop_scale: float,
    warp_margin: int,
) -> list[list[np.ndarray]]:
    square = warp_size / 8.0
    half = square * crop_scale / 2.0
    crops = []
    for row in range(8):
        row_crops = []
        for col in range(8):
            cx = warp_margin + (col + 0.5) * square
            cy = warp_margin + (row + 0.5) * square
            x0 = max(0, int(round(cx - half)))
            y0 = max(0, int(round(cy - half)))
            x1 = min(warped.shape[1], int(round(cx + half)))
            y1 = min(warped.shape[0], int(round(cy + half)))
            
            crop = warped[y0:y1, x0:x1]
            # Convert crop to RGB for the model
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            crop_resized = cv2.resize(crop_rgb, (192, 192), interpolation=cv2.INTER_AREA)
            row_crops.append(crop_resized)
        crops.append(row_crops)
    return crops


# Classification

@torch.no_grad()
def classify_board(
    model: PrototypicalNet,
    prototypes: torch.Tensor,
    class_names: list[str],
    crops: list[list[np.ndarray]],
    device: torch.device,
    empty_bias: float = 0.85,
) -> list[list[str]]:
    empty_idx = class_names.index("empty") if "empty" in class_names else None
    preds = []
    for row_crops in crops:
        row_preds = []
        for crop in row_crops:
            emb = model(INFER_TRANSFORM(crop).unsqueeze(0).to(device))
            dists = torch.pow(emb - prototypes, 2).sum(dim=1)
            if empty_idx is not None:
                dist_empty = dists[empty_idx].item()
                piece_dists = dists.clone()
                piece_dists[empty_idx] = float("inf")
                best_piece_idx = piece_dists.argmin().item()
                if piece_dists[best_piece_idx].item() >= empty_bias * dist_empty:
                    row_preds.append("empty")
                else:
                    row_preds.append(class_names[best_piece_idx])
            else:
                row_preds.append(class_names[dists.argmin().item()])
        preds.append(row_preds)
    return preds


# FEN (chess notation)

def predictions_to_fen(grid: list[list[str]], notation: str) -> str:
    board: dict[str, str] = {}
    for row in range(8):
        for col in range(8):
            board[square_name(row, col, notation)] = grid[row][col]

    rows = []
    for rank in range(8, 0, -1):
        empty = 0
        s = ""
        for file_idx in range(8):
            sq = f"{chr(ord('A') + file_idx)}{rank}"
            ch = PIECE_TO_FEN.get(board.get(sq, "empty"), ".")
            if ch == ".":
                empty += 1
            else:
                if empty:
                    s += str(empty)
                    empty = 0
                s += ch
        if empty:
            s += str(empty)
        rows.append(s)
    return "/".join(rows)


# Board gui generation

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in [
        # Linux paths
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        # Windows paths
        "C:/Windows/Fonts/seguisym.ttf", 
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        # macOS paths
        "/Library/Fonts/Arial Unicode.ttf"
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    
    print("WARNING: No suitable Unicode font found. Pieces may not render.")
    return ImageFont.load_default()


def render_board(grid: list[list[str]], notation: str, out_path: Path) -> None:
    C, M = BOARD_CELL, BOARD_MARGIN
    TOTAL = 8 * C + 2 * M

    img = Image.new("RGB", (TOTAL, TOTAL), (49, 46, 43))
    draw = ImageDraw.Draw(img)
    piece_font = _load_font(int(C * 0.78))
    label_font = _load_font(13)

    board: dict[str, str] = {}
    for row in range(8):
        for col in range(8):
            board[square_name(row, col, notation)] = grid[row][col]

    for rank in range(8, 0, -1):
        for fi in range(8):
            sq = f"{chr(ord('A') + fi)}{rank}"
            piece = board.get(sq, "empty")
            x, y = M + fi * C, M + (8 - rank) * C
            is_light = (fi + rank) % 2 == 1
            draw.rectangle([x, y, x + C, y + C], fill=SQ_LIGHT if is_light else SQ_DARK)

            symbol = PIECE_UNICODE.get(piece)
            if symbol:
                is_white = piece.endswith("_w")
                fg      = (255, 255, 255) if is_white else (30,  30,  30)
                outline = (60,  60,  60)  if is_white else (210, 210, 210)
                bb = draw.textbbox((0, 0), symbol, font=piece_font)
                tw, th = bb[2] - bb[0], bb[3] - bb[1]
                tx = x + (C - tw) // 2 - bb[0]
                ty = y + (C - th) // 2 - bb[1]
                for dx, dy in [(-2,0),(2,0),(0,-2),(0,2),(-1,-1),(1,-1),(-1,1),(1,1)]:
                    draw.text((tx+dx, ty+dy), symbol, font=piece_font, fill=outline)
                draw.text((tx, ty), symbol, font=piece_font, fill=fg)

    for rank in range(8, 0, -1):
        draw.text((M - 18, M + (8 - rank) * C + (C - 13) // 2),
                  str(rank), font=label_font, fill=(180, 180, 180))
    for fi in range(8):
        draw.text((M + fi * C + (C - 8) // 2, M + 8 * C + 6),
                  chr(ord("a") + fi), font=label_font, fill=(180, 180, 180))

    img.save(str(out_path))
    print(f"Tauler digital guardat → {out_path}")


# Main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chess inference: photo → FEN + digital board")
    p.add_argument("--image", required=True, help="Foto del tauler d'escacs")
    p.add_argument("--weights", default="model/proto_resnet18_best.pth")
    p.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet18")
    p.add_argument("--support-dir", default="Dataset/Imatges_propies/peces_segmentades_recrop")
    p.add_argument("--k-shot", type=int, default=120)
    p.add_argument("--out-dir", default="inference_output")
    p.add_argument("--warp-size", type=int, default=640)
    p.add_argument("--crop-scale", type=float, default=1.4)
    p.add_argument("--empty-bias", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-refine", action="store_true",
                   help="Desactiva el refinament de cantonades després del clic manual")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Carregar imatge
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    print(f"Imatge carregada: {args.image}  ({image.shape[1]}×{image.shape[0]})")

    # Selecció interactiva de cantonades
    print("Obre finestra per seleccionar les 4 cantonades...")
    corners = pick_corners_interactive(image)
    notation = "a8_tl"

    if not args.no_refine:
        max_move = board_side(corners) * 0.08
        corners, score = refine_corners(image, corners, args.warp_size, [32, 16, 8, 4, 2], max_move)
        print(f"Refinament de cantonades: score={score:.3f}")

    # Warp del tauler
    warp_margin = int(round(args.warp_size / 8.0 * 0.8))
    warped = warp_board_with_margin(image, corners, args.warp_size, warp_margin)
    warped_plain = warp_board(image, corners, args.warp_size)

    stem = Path(args.image).stem
    cv2.imwrite(str(out_dir / f"{stem}_warped.jpg"), warped_plain)
    print(f"Warp guardat → {out_dir / f'{stem}_warped.jpg'}")

    # Carregar model
    print(f"Carregant model: {args.weights}")
    model = PrototypicalNet(backbone_name=args.backbone).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # Prototips del support set (fewshot learning)
    support_dir = Path(args.support_dir)
    print(f"Support set: {support_dir}  (k={args.k_shot})")
    support = load_support_images(support_dir, args.k_shot, args.seed)
    if not support:
        raise RuntimeError(f"No s'han trobat classes a: {support_dir}")
    prototypes, class_names = compute_prototypes(model, support, device)
    print(f"Prototips calculats per {len(class_names)} classes.")

    # Classificacio de les 64 caselles individualment
    print("Classificant caselles...")
    crops = extract_square_crops(warped, args.warp_size, args.crop_scale, warp_margin)
    predictions = classify_board(model, prototypes, class_names, crops, device,
                                 empty_bias=args.empty_bias)

    # notacio FEN
    fen = predictions_to_fen(predictions, notation)
    print(f"\nFEN: {fen}\n")
    (out_dir / "position.fen").write_text(fen)

    # Render digital
    render_board(predictions, notation, out_dir / f"{stem}_board.jpg")

    # Guardar prediccions per casella (per avaluació posterior)
    pred_dict = {}
    for row_idx, row in enumerate(predictions):
        for col_idx, piece in enumerate(row):
            sq = square_name(row_idx, col_idx, notation)
            pred_dict[sq] = piece
    import json as _json
    pred_path = out_dir / f"{stem}_predictions.json"
    pred_path.write_text(_json.dumps(pred_dict, indent=2))
    print(f"Prediccions guardades → {pred_path}")

    # print de logs
    print("\nPredicció per rank:")
    for row_idx, row in enumerate(predictions):
        rank = 8 - row_idx
        print(f"  rank {rank}: " + "  ".join(f"{p:<12}" for p in row))


if __name__ == "__main__":
    main()
