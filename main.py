"""Reconstruct a UAV route on a satellite map from a series of frames.

Usage:
    python main.py --map data/global_map.png --frames data/crops --out results
"""
import argparse
import csv
import re
import time
from pathlib import Path

import cv2
import numpy as np

from localizer import MapLocalizer
from trajectory import interpolate_missing, reject_outliers, smooth
from visualize import draw_route, draw_verification, make_video

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def natural_key(path):
    """Sort crop_2.png before crop_10.png (plain string sort would not)."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def parse_scales(text):
    return tuple(float(s) for s in text.split(",") if s.strip())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", default="data/global_map.png", help="path to the satellite map")
    p.add_argument("--frames", default="data/crops", help="directory with UAV frames")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--scale", type=float, default=0.5, help="frame downscale factor before SIFT")
    p.add_argument("--fallback-scales", type=parse_scales, default=(1.0, 0.25),
                   help="scales tried if --scale fails, comma separated ('' = off)")
    p.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test threshold")
    p.add_argument("--min-inliers", type=int, default=15, help="min RANSAC inliers to accept a frame")
    p.add_argument("--smooth", type=int, default=1, help="moving-average window (1 = off)")
    p.add_argument("--video", action="store_true", help="also write route.mp4")
    p.add_argument("--verify", action="store_true",
                   help="also write verification.png (frame vs. map warped into the frame view)")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    map_bgr = cv2.imread(args.map)
    if map_bgr is None:
        raise SystemExit(f"Cannot read map: {args.map}")

    frame_files = sorted((f for f in Path(args.frames).iterdir() if f.suffix.lower() in IMAGE_EXT),
                         key=natural_key) if Path(args.frames).is_dir() else []
    if not frame_files:
        raise SystemExit(f"No image frames in {args.frames}")

    t0 = time.perf_counter()
    loc = MapLocalizer(map_bgr, cache_path=out / "map_features.npz",
                       ratio=args.ratio, min_inliers=args.min_inliers)
    t_map = time.perf_counter() - t0

    t0 = time.perf_counter()
    results, centers, boxes = [], [], []
    for i, f in enumerate(frame_files):
        frame = cv2.imread(str(f))
        res = None if frame is None else loc.locate(frame, scale=args.scale,
                                                    fallback_scales=args.fallback_scales)
        results.append(res)
        if res is None:
            print(f"[{i:2d}] {f.name}: FAILED")
            centers.append(None)
            boxes.append(None)
        else:
            note = "" if res.scale == args.scale else f"  (fallback scale {res.scale})"
            print(f"[{i:2d}] {f.name}: inliers={res.inliers:4d} err={res.reproj_err:4.2f}px "
                  f"centre=({res.center[0]:.0f}, {res.center[1]:.0f}){note}")
            centers.append(res.center)
            boxes.append(res.corners)
    t_frames = time.perf_counter() - t0

    # Remove implausible jumps, then fill the gaps and (optionally) smooth.
    cleaned = reject_outliers(centers)
    status = ["ok" if c is not None else "failed" for c in centers]
    for i, (a, b) in enumerate(zip(centers, cleaned)):
        if a is not None and b is None:
            print(f"[{i:2d}] rejected as outlier")
            boxes[i] = None
            status[i] = "outlier"
    path, measured = interpolate_missing(cleaned)
    path = smooth(path, args.smooth)

    # Outputs.
    np.savetxt(out / "trajectory.csv", path, delimiter=",", header="x,y", comments="", fmt="%.1f")
    with open(out / "frames_report.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "file", "status", "x", "y", "inliers", "reproj_err_px", "scale"])
        for i, (f, r) in enumerate(zip(frame_files, results)):
            st = status[i] if measured[i] else f"{status[i]}->interpolated"
            w.writerow([i, f.name, st, f"{path[i, 0]:.1f}", f"{path[i, 1]:.1f}",
                        r.inliers if r else "", f"{r.reproj_err:.2f}" if r else "",
                        r.scale if r else ""])

    draw_route(map_bgr, path, measured, boxes, out / "route.png")
    if args.video:
        make_video(map_bgr, path, boxes, out / "route.mp4")
    if args.verify:
        draw_verification(map_bgr, frame_files, results, out / "verification.png")

    ok = [r for r, m in zip(results, measured) if r is not None and m]
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    print("-" * 60)
    print(f"Matched {measured.sum()}/{len(frame_files)} frames")
    if ok:
        inl = np.array([r.inliers for r in ok])
        err = np.array([r.reproj_err for r in ok])
        print(f"Inliers: min {inl.min()}, mean {inl.mean():.0f}; "
              f"median reprojection error {np.median(err):.2f} px")
    print(f"Route length {steps.sum():.0f} px, step {np.median(steps):.1f} px (median)")
    print(f"Time: map features {t_map:.1f} s, frames {t_frames:.1f} s "
          f"({1000 * t_frames / len(frame_files):.0f} ms/frame)")
    print(f"Saved results to {out.resolve()}")


if __name__ == "__main__":
    main()
