from pathlib import Path
import cv2
import numpy as np

PECES_DIR    = Path("Dataset/Imatges_propies/peces_segmentades_recrop")
CROP_SIZE    = 192
OLD_SCALE    = 1.8
NEW_SCALE    = 1.4

def recrop(img: np.ndarray, old_scale: float, new_scale: float, out_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    factor = new_scale / old_scale          # 1.8/2.2 ≈ 0.818
    inner_h = int(round(h * factor))
    inner_w = int(round(w * factor))
    y0 = (h - inner_h) // 2
    x0 = (w - inner_w) // 2
    cropped = img[y0:y0+inner_h, x0:x0+inner_w]
    return cv2.resize(cropped, (out_size, out_size), interpolation=cv2.INTER_AREA)

def main() -> None:
    image_paths = sorted(PECES_DIR.rglob("*.jpg")) + sorted(PECES_DIR.rglob("*.png"))

    if not image_paths:
        print(f"No s'han trobat imatges a {PECES_DIR}")
        return

    print(f"Processant {len(image_paths)} imatges ({OLD_SCALE} -> {NEW_SCALE} scale)...")

    for i, path in enumerate(image_paths):
        img = cv2.imread(str(path))
        if img is None:
            print(f"  Error llegint {path.name}, saltant.")
            continue
        out = recrop(img, OLD_SCALE, NEW_SCALE, CROP_SIZE)
        cv2.imwrite(str(path), out)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(image_paths)}...")

    print(f"Fet. {len(image_paths)} imatges actualitzades.")

if __name__ == "__main__":
    main()
