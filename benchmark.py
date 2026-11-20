"""Accuracy / robustness benchmark with known ground truth.

The real task has no ground-truth coordinates, so this script makes its own: it cuts
synthetic "UAV frames" out of the map with a known centre, rotation, scale and (optionally)
perspective tilt, spoils them in different ways (lighting, blur, noise, clouds, colour shift)
and measures how far the localiser's answer is from the truth.

    python benchmark.py --map data/global_map.png --n 40
    python benchmark.py --map data/global_map.png --demo-flight data/demo   # frames + gt.csv

Error is reported in map pixels; a frame counts as "correct" if its centre is within
10 % of the frame footprint from the truth, and as "wrong" if it was accepted but is farther
(a wrong answer is worse than no answer, because it silently corrupts the route).
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from localizer import MapLocalizer

OUT_SIZE = 512             # synthetic frame size, px
SIDE_RANGE = (180, 260)    # frame footprint on the map, map px (frames are "zoomed in" ~2-3x)


def crop_homography(center, side, angle_deg, tilt=0.0, rng=None):
    """Homography that maps synthetic-frame pixels to map pixels."""
    k = side / OUT_SIZE
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    A = k * R
    t = np.asarray(center) - A @ np.array([OUT_SIZE / 2, OUT_SIZE / 2])
    H = np.eye(3)
    H[:2, :2], H[:2, 2] = A, t
    if tilt:
        # Small projective term = camera not looking straight down.
        g = rng.uniform(-tilt, tilt, 2) / OUT_SIZE
        P = np.array([[1, 0, 0], [0, 1, 0], [g[0], g[1], 1.0]])
        # keep the frame centre fixed under the tilt
        c = np.array([OUT_SIZE / 2, OUT_SIZE / 2, 1.0])
        pc = P @ c
        pc = pc[:2] / pc[2]
        T = np.array([[1, 0, OUT_SIZE / 2 - pc[0]], [0, 1, OUT_SIZE / 2 - pc[1]], [0, 0, 1]])
        H = H @ np.linalg.inv(T @ P)
    return H


def render(map_bgr, H):
    return cv2.warpPerspective(map_bgr, H, (OUT_SIZE, OUT_SIZE),
                               flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REFLECT)


# --- ways to spoil a frame -------------------------------------------------------------
def c_clean(img, rng):
    return img


def c_light(img, rng):
    alpha, beta = rng.uniform(0.6, 1.4), rng.uniform(-50, 50)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def c_blur(img, rng):
    return cv2.GaussianBlur(img, (0, 0), rng.uniform(1.5, 3.0))


def c_noise(img, rng):
    return np.clip(img + rng.normal(0, 15, img.shape), 0, 255).astype(np.uint8)


def c_clouds(img, rng):
    """Soft white blobs covering roughly a third of the frame."""
    m = np.zeros(img.shape[:2], np.float32)
    for _ in range(3):
        c = rng.integers(0, OUT_SIZE, 2)
        cv2.circle(m, (int(c[0]), int(c[1])), int(rng.integers(70, 130)), 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), 25)[..., None]
    return (img * (1 - 0.9 * m) + 255 * 0.9 * m).astype(np.uint8)


def c_season(img, rng):
    """Other season / camera: hue and saturation shift, lower contrast, slight blur."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(rng.integers(8, 20))) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.4, 0.7), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    out = cv2.convertScaleAbs(out, alpha=0.7, beta=40)
    return cv2.GaussianBlur(out, (0, 0), 1.2)


def c_combo(img, rng):
    """Worst case: other season + clouds + blur + sensor noise at once."""
    return c_noise(c_blur(c_clouds(c_season(img, rng), rng), rng), rng)


CONDITIONS = {"clean": c_clean, "lighting": c_light, "blur": c_blur, "noise": c_noise,
              "clouds": c_clouds, "season": c_season, "all-combined": c_combo,
              "low-altitude": c_clean, "high-altitude": c_clean}
# Conditions that change the flight altitude (= frame footprint on the map, map px).
SIDE_OVERRIDE = {"low-altitude": (70, 100), "high-altitude": (450, 600)}


def sample_poses(map_shape, n, rng, tilt, side_range=SIDE_RANGE):
    h, w = map_shape[:2]
    m = side_range[1]
    for _ in range(n):
        yield dict(center=(rng.uniform(m, w - m), rng.uniform(m, h - m)),
                   side=rng.uniform(*side_range), angle=rng.uniform(0, 360),
                   tilt=tilt)


def evaluate(loc, map_bgr, poses, cond, rng, scale, fallback):
    rows = []
    for p in poses:
        H = crop_homography(p["center"], p["side"], p["angle"], p["tilt"], rng)
        frame = CONDITIONS[cond](render(map_bgr, H), rng)
        t = time.perf_counter()
        res = loc.locate(frame, scale=scale, fallback_scales=fallback)
        dt = time.perf_counter() - t
        if res is None:
            rows.append(dict(found=False, err=np.nan, side=p["side"], t=dt, inliers=0))
        else:
            err = float(np.linalg.norm(res.center - np.asarray(p["center"])))
            rows.append(dict(found=True, err=err, side=p["side"], t=dt, inliers=res.inliers))
    return rows


def summarize(rows):
    n = len(rows)
    found = [r for r in rows if r["found"]]
    correct = [r for r in found if r["err"] <= 0.1 * r["side"]]
    errs = np.array([r["err"] for r in correct]) if correct else np.array([np.nan])
    return dict(n=n, correct=len(correct), wrong=len(found) - len(correct),
                lost=n - len(found), median_err=float(np.median(errs)),
                p95_err=float(np.percentile(errs, 95)),
                ms=1000 * float(np.mean([r["t"] for r in rows])))


def demo_flight(map_bgr, out_dir, n=50, seed=0):
    """Write a synthetic flight (smooth curve) as frames + gt.csv, to run main.py end-to-end."""
    rng = np.random.default_rng(seed)
    h, w = map_bgr.shape[:2]
    out_dir = Path(out_dir)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    m = SIDE_RANGE[1]
    s = np.linspace(0, 1, n)
    xs = m + s * (w - 2 * m)
    ys = h / 2 + (h / 2 - m) * 0.8 * np.sin(2 * np.pi * s * 1.2 + rng.uniform(0, 3))
    heading = np.degrees(np.arctan2(np.gradient(ys), np.gradient(xs)))
    with open(out_dir / "gt.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "x", "y"])
        for i in range(n):
            H = crop_homography((xs[i], ys[i]), 220, heading[i] + 90, tilt=0.0, rng=rng)
            cv2.imwrite(str(out_dir / "crops" / f"crop_{i}.png"), render(map_bgr, H))
            wr.writerow([i, f"{xs[i]:.1f}", f"{ys[i]:.1f}"])
    cv2.imwrite(str(out_dir / "global_map.png"), map_bgr)
    print(f"Demo flight written to {out_dir} ({n} frames, ground truth in gt.csv)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="data/global_map.png")
    ap.add_argument("--n", type=int, default=40, help="frames per condition")
    ap.add_argument("--tilt", type=float, default=0.15, help="max perspective tilt (0 = nadir)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/benchmark.csv")
    ap.add_argument("--demo-flight", metavar="DIR", help="only write a synthetic flight and exit")
    args = ap.parse_args()

    map_bgr = cv2.imread(args.map)
    if map_bgr is None:
        raise SystemExit(f"Cannot read map: {args.map}")
    if args.demo_flight:
        demo_flight(map_bgr, args.demo_flight, seed=args.seed)
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    loc = MapLocalizer(map_bgr, cache_path=Path(args.out).parent / "map_features.npz")
    configs = {"baseline (scale 0.5)": (0.5, ()),
               "+ multi-scale fallback": (0.5, (1.0, 0.25))}

    table = []
    for cond in CONDITIONS:
        rng = np.random.default_rng(args.seed)
        poses = list(sample_poses(map_bgr.shape, args.n, rng, args.tilt,
                                  SIDE_OVERRIDE.get(cond, SIDE_RANGE)))
        for name, (scale, fb) in configs.items():
            s = summarize(evaluate(loc, map_bgr, poses, cond, np.random.default_rng(args.seed + 1),
                                   scale, fb))
            table.append(dict(condition=cond, config=name, **s))
            print(f"{cond:13s} | {name:24s} | correct {s['correct']:3d}/{s['n']}  wrong {s['wrong']:2d}"
                  f"  lost {s['lost']:2d} | median err {s['median_err']:5.2f} px"
                  f"  p95 {s['p95_err']:5.2f} px | {s['ms']:5.0f} ms/frame")

    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(table[0]))
        wr.writeheader()
        for row in table:
            wr.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
