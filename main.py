"""Reconstruct a UAV route on a satellite map from a series of frames.

Usage:
    python main.py --map data/global_map.png --frames data/crops --out results
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from localizer import MapLocalizer
from trajectory import interpolate_missing, reject_outliers, smooth
from visualize import draw_route, make_video


def natural_key(path):
    """Sort crop_2.png before crop_10.png (plain string sort would not)."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", default="data/global_map.png", help="path to the satellite map")
    p.add_argument("--frames", default="data/crops", help="directory with UAV frames")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--scale", type=float, default=0.5, help="frame downscale factor before SIFT")
    p.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test threshold")
    p.add_argument("--min-inliers", type=int, default=15, help="min RANSAC inliers to accept a frame")
    p.add_argument("--smooth", type=int, default=1, help="moving-average window (1 = off)")
    p.add_argument("--video", action="store_true", help="also write route.mp4")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    map_bgr = cv2.imread(args.map)
    if map_bgr is None:
        raise SystemExit(f"Cannot read map: {args.map}")

    frame_files = sorted(Path(args.frames).glob("*.png"), key=natural_key)
    if not frame_files:
        raise SystemExit(f"No .png frames in {args.frames}")

    loc = MapLocalizer(map_bgr, cache_path=out / "map_features.npz",
                       ratio=args.ratio, min_inliers=args.min_inliers)

    centers, boxes = [], []
    for i, f in enumerate(frame_files):
        res = loc.locate(cv2.imread(str(f)), scale=args.scale)
        if res is None:
            print(f"[{i:2d}] {f.name}: FAILED")
            centers.append(None)
            boxes.append(None)
        else:
            print(f"[{i:2d}] {f.name}: inliers={res.inliers:4d} centre=({res.center[0]:.0f}, {res.center[1]:.0f})")
            centers.append(res.center)
            boxes.append(res.corners)

    # Remove implausible jumps, then fill the gaps and (optionally) smooth.
    cleaned = reject_outliers(centers)
    for i, (a, b) in enumerate(zip(centers, cleaned)):
        if a is not None and b is None:
            print(f"[{i:2d}] rejected as outlier")
            boxes[i] = None
    path, measured = interpolate_missing(cleaned)
    path = smooth(path, args.smooth)

    print(f"Matched {measured.sum()}/{len(frame_files)} frames")
    np.savetxt(out / "trajectory.csv", path, delimiter=",", header="x,y", comments="", fmt="%.1f")
    draw_route(map_bgr, path, measured, boxes, out / "route.png")
    if args.video:
        make_video(map_bgr, path, boxes, out / "route.mp4")
    print(f"Saved results to {out.resolve()}")


if __name__ == "__main__":
    main()
