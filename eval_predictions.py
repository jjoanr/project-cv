"""
Compara les prediccions d'inference_output amb els GT del dataset.
Prova les 4 rotacions possibles i tria la millor per cada tauler.
"""

import json
from pathlib import Path

INFERENCE_DIR = Path("inference_output")
DATASET_DIR   = Path("Dataset/data")
IDS = [1851, 1852, 1853, 1854, 1855, 1856]

ALL_SQUARES = [f"{f}{r}" for r in range(8, 0, -1) for f in "ABCDEFGH"]
FILES = list("ABCDEFGH")
RANKS = list(range(8, 0, -1))  # 8..1

def sq(file_idx: int, rank_idx: int) -> str:
    """file_idx 0=A..7=H, rank_idx 0=8..7=1"""
    return f"{FILES[file_idx]}{RANKS[rank_idx]}"

def rotate_square_0(f, r):   return sq(f, r)
def rotate_square_90cw(f, r): return sq(7 - r, f)
def rotate_square_180(f, r):  return sq(7 - f, 7 - r)
def rotate_square_90ccw(f, r):return sq(r, 7 - f)

ROTATIONS = {
    "0°":      rotate_square_0,
    "90° CW":  rotate_square_90cw,
    "180°":    rotate_square_180,
    "90° CCW": rotate_square_90ccw,
}

def load_gt(board_id: int) -> dict[str, str]:
    p = DATASET_DIR / f"{board_id}.json"
    data = json.loads(p.read_text())
    gt = {sq: "empty" for sq in ALL_SQUARES}
    for sq_name, piece in data["config"].items():
        gt[sq_name] = piece
    return gt

def load_pred(board_id: int) -> dict[str, str]:
    p = INFERENCE_DIR / f"{board_id}_predictions.json"
    return json.loads(p.read_text())

def compare(pred: dict, gt: dict, rot_fn) -> dict:
    correct = 0
    correct_type = 0  # right type, wrong color
    wrong = 0
    errors = []
    for f_idx in range(8):
        for r_idx in range(8):
            pred_sq  = sq(f_idx, r_idx)
            gt_sq    = rot_fn(f_idx, r_idx)
            p = pred.get(pred_sq, "empty")
            g = gt.get(gt_sq, "empty")
            if p == g:
                correct += 1
            else:
                # check if same piece type, different color
                p_type = p.rsplit("_", 1)[0] if "_" in p else p
                g_type = g.rsplit("_", 1)[0] if "_" in g else g
                if p_type == g_type and p_type != "empty":
                    correct_type += 1
                else:
                    wrong += 1
                errors.append((pred_sq, gt_sq, p, g))
    return {
        "correct": correct,
        "correct_type": correct_type,
        "wrong": wrong,
        "total": 64,
        "errors": errors,
    }

print("=" * 70)
print(f"{'Board':<8} {'Rotació':<12} {'Encerts':>8} {'Tipus OK':>9} {'Errors':>7} {'Acc':>7}")
print("=" * 70)

best_results = {}
for board_id in IDS:
    gt   = load_gt(board_id)
    pred = load_pred(board_id)

    best_rot = None
    best_score = -1
    best_res = None

    for rot_name, rot_fn in ROTATIONS.items():
        res = compare(pred, gt, rot_fn)
        score = res["correct"] + 0.5 * res["correct_type"]
        if score > best_score:
            best_score = score
            best_rot   = rot_name
            best_res   = res

    best_results[board_id] = (best_rot, best_res)

    for rot_name, rot_fn in ROTATIONS.items():
        res = compare(pred, gt, rot_fn)
        marker = " <--" if rot_name == best_rot else ""
        print(f"{board_id:<8} {rot_name:<12} {res['correct']:>8} {res['correct_type']:>9} {res['wrong']:>7} {res['correct']/64*100:>6.1f}%{marker}")
    print("-" * 70)

print("\n" + "=" * 70)
print("RESUM — millor rotació per cada tauler")
print("=" * 70)
total_correct = total_correct_type = total_wrong = 0
for board_id in IDS:
    rot, res = best_results[board_id]
    print(f"\n  Tauler {board_id}  [{rot}]")
    print(f"    Encerts exactes : {res['correct']}/64 ({res['correct']/64*100:.1f}%)")
    print(f"    Tipus correcte  : {res['correct_type']} (peca ok, color diferent)")
    print(f"    Errors complets : {res['wrong']}")
    total_correct      += res['correct']
    total_correct_type += res['correct_type']
    total_wrong        += res['wrong']
    if res['errors']:
        print(f"    Errors (pred_sq → gt_sq): pred  ≠  gt")
        for pred_sq, gt_sq, p, g in res['errors'][:20]:  # max 20
            print(f"      {pred_sq}→{gt_sq}:  {p:<14}  ≠  {g}")
        if len(res['errors']) > 20:
            print(f"      ... i {len(res['errors'])-20} errors més")

print()
print(f"TOTAL  Encerts: {total_correct}/384  ({total_correct/384*100:.1f}%)")
print(f"       Tipus correcte (color ≠): {total_correct_type}")
print(f"       Errors complets: {total_wrong}")
