"""Self-test with known ground truth (the task data has no true coordinates).

1. Unit test of drop_jumps() on hand-made routes: wrong places (single and in a row),
   a hovering drone, sharp turns - no images needed, runs in a second.
2. End-to-end test: cuts synthetic UAV photos out of the map at known places (random
   rotation, the usual zoom), breaks some of them, runs main.py and compares the route
   with the truth.

    python selftest.py --map data/global_map.png
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from main import drop_jumps

SIDE, SIZE = 220, 512   # photo footprint on the map (px) and photo size (px), like the task


def box_at(c, side=SIDE):
    """Footprint of a photo centred at c (only its size matters for drop_jumps)."""
    return np.float32([[c[0], c[1]], [c[0] + side, c[1]], [c[0] + side, c[1] + side], [c[0], c[1] + side]])


def check_drop_jumps():
    t = np.linspace(0, 1, 50)
    route = np.stack([300 + 3000 * t, 800 + 400 * np.sin(6 * t)], axis=1)
    cases = {}
    pts = list(route.copy())
    pts[10] = np.array([3200.0, 100.0])                       # one wrong place
    cases["single wrong place"] = (pts, {10})
    pts = list(route.copy())
    for k in (20, 21, 22):                                    # three wrong photos in a row
        pts[k] = np.array([200.0 + 20 * k, 1500.0])
    cases["3 wrong in a row"] = (pts, {20, 21, 22})
    pts = list(route.copy())
    for k in range(5, 35):                                    # drone hovers, then flies on
        pts[k] = route[5] + np.random.default_rng(k).normal(0, 1.5, 2)
    for k in range(35, 50):                                   # ... from where it hovered
        pts[k] = route[5] + 2 * (route[k] - route[35])
    cases["hovering drone"] = (pts, set())
    zig = np.cumsum(np.array([[100, 0] if (i // 6) % 2 else [0, 100] for i in range(50)], float), axis=0)
    cases["sharp turns"] = (list(zig + 300), set())

    ok = True
    for name, (pts, expected) in cases.items():
        centres = [np.asarray(p, float) for p in pts]
        dropped = set(drop_jumps(centres, [box_at(c) for c in centres]))
        good = dropped == expected
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  drop_jumps: {name:20s} dropped={sorted(dropped)}")
    return ok


def photo(map_bgr, centre, angle):
    """Cut a rotated, zoomed-in photo of the map around `centre`."""
    k, a = SIDE / SIZE, np.deg2rad(angle)
    A = k * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    M = np.hstack([A, (np.asarray(centre) - A @ [SIZE / 2, SIZE / 2])[:, None]])
    return cv2.warpAffine(map_bgr, M, (SIZE, SIZE), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)


def check_flight(map_path, n=50):
    map_bgr = cv2.imread(map_path)
    h, w = map_bgr.shape[:2]
    m = SIDE + 50
    s = np.linspace(0, 1, n)
    truth = np.stack([m + s * (w - 2 * m), h / 2 + (h / 2 - m) * 0.7 * np.sin(2 * np.pi * 1.2 * s)], axis=1)
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        crops = Path(tmp) / "crops"
        crops.mkdir()
        for i, c in enumerate(truth):
            img = photo(map_bgr, c, rng.uniform(0, 360))
            if i in (8, 30):                                   # unusable photo (noise)
                img = rng.integers(0, 256, img.shape, dtype=np.uint8)
            if i in (15, 16):                                  # two photos of a wrong place
                img = photo(map_bgr, (w - m, m), 33)
            cv2.imwrite(str(crops / f"crop_{i}.png"), img)
        subprocess.run([sys.executable, str(Path(__file__).with_name("main.py")), "--map", map_path,
                        "--frames", str(crops), "--out", str(Path(tmp) / "out")],
                       check=True, capture_output=True)
        route = np.loadtxt(Path(tmp) / "out" / "trajectory.csv", delimiter=",", skiprows=1)
    err = np.linalg.norm(route - truth, axis=1)
    good = np.median(err) < 1.0 and err.max() < 0.15 * SIDE
    print(f"  {'PASS' if good else 'FAIL'}  flight with 4 broken photos: median error {np.median(err):.2f} px, "
          f"max {err.max():.1f} px (limit {0.15 * SIDE:.0f})")
    return good


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="data/global_map.png")
    args = ap.parse_args()
    print("Unit test:")
    ok = check_drop_jumps()
    if Path(args.map).exists():
        print("End-to-end test:")
        ok &= check_flight(args.map)
    else:
        print(f"  (map {args.map} not found - end-to-end test skipped)")
    print("ALL PASSED" if ok else "SOME TESTS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
