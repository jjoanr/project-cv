"""
Inference usant com a support set les peces pròpies segmentades.

Igual que inference.py però:
  - --support-dir per defecte: Dataset/Imatges_propies/peces_segmentades
  - --k-shot per defecte: 9999 (totes les imatges disponibles)

Ús:
  python inference_propies.py --image foto.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch

from inference import (
    PrototypicalNet,
    classify_board,
    compute_prototypes,
    extract_square_crops,
    load_support_images,
    pick_corners_interactive,
    predictions_to_fen,
    render_board,
)
from refine_graella import board_side, refine_corners, square_name, warp_board, warp_board_with_margin


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference amb peces pròpies segmentades")
    p.add_argument("--image",       required=True)
    p.add_argument("--weights",     default="model/proto_resnet18_best.pth")
    p.add_argument("--backbone",    choices=["resnet18", "resnet50"], default="resnet18")
    p.add_argument("--support-dir", default="Dataset/Imatges_propies/peces_segmentades")
    p.add_argument("--k-shot",      type=int, default=9999)
    p.add_argument("--out-dir",     default="Dataset/Imatges_propies/imatges_avaluacio/inference_output")
    p.add_argument("--warp-size",   type=int, default=640)
    p.add_argument("--crop-scale",  type=float, default=1.8)
    p.add_argument("--empty-bias",  type=float, default=0.85)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--no-refine",   action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar imatge
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    print(f"Imatge: {args.image}  ({image.shape[1]}x{image.shape[0]})")

    # 2. Selecció interactiva de cantonades (A1 -> H1 -> H8 -> A8)
    corners = pick_corners_interactive(image)
    notation = "a8_tl"

    # 3. Refinament
    if not args.no_refine:
        max_move = board_side(corners) * 0.08
        corners, score = refine_corners(image, corners, args.warp_size, [32, 16, 8, 4, 2], max_move)
        print(f"Refinament: score={score:.3f}")

    # 4. Warp
    warp_margin = int(round(args.warp_size / 8.0 * 0.8))
    warped      = warp_board_with_margin(image, corners, args.warp_size, warp_margin)
    warped_plain = warp_board(image, corners, args.warp_size)

    stem = Path(args.image).stem
    cv2.imwrite(str(out_dir / f"{stem}_warped.jpg"), warped_plain)

    # 5. Carregar model
    print(f"Model: {args.weights}")
    model = PrototypicalNet(backbone_name=args.backbone).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # 6. Support set — totes les imatges de peces_segmentades
    support_dir = Path(args.support_dir)
    support = load_support_images(support_dir, args.k_shot, args.seed)
    if not support:
        raise RuntimeError(f"No s'han trobat classes a: {support_dir}")

    print(f"Support set: {support_dir}")
    for cls in sorted(support):
        print(f"  {cls:<14}: {len(support[cls])} imatges")

    prototypes, class_names = compute_prototypes(model, support, device)
    print(f"Prototips calculats per {len(class_names)} classes.")

    # 7. Classificació
    crops = extract_square_crops(warped, args.warp_size, args.crop_scale, warp_margin)
    predictions = classify_board(model, prototypes, class_names, crops, device,
                                 empty_bias=args.empty_bias)

    # 8. FEN
    fen = predictions_to_fen(predictions, notation)
    print(f"\nFEN: {fen}\n")
    (out_dir / "position.fen").write_text(fen)

    # 9. Render
    render_board(predictions, notation, out_dir / f"{stem}_board.jpg")

    # 10. Guardar prediccions JSON
    pred_dict = {}
    for row_idx, row in enumerate(predictions):
        for col_idx, piece in enumerate(row):
            pred_dict[square_name(row_idx, col_idx, notation)] = piece
    pred_path = out_dir / f"{stem}_predictions.json"
    pred_path.write_text(json.dumps(pred_dict, indent=2))
    print(f"Prediccions: {pred_path}")

    # 11. Resum
    print("\nPrediccio per rank:")
    for row_idx, row in enumerate(predictions):
        print(f"  rank {8 - row_idx}: " + "  ".join(f"{p:<12}" for p in row))


if __name__ == "__main__":
    main()
