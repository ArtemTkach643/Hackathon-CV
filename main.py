"""Reconstruct the route of a UAV on a satellite map.

Idea: for every UAV photo we find where it lies on the big map, take the centre of
that place as one point of the route, and finally draw all the points on the map.

How one photo is placed on the map:
    1. SIFT finds "interesting points" (corners, crossroads...) on the photo and on the map.
    2. Points are matched by their descriptors (a description of the surroundings).
    3. RANSAC (inside findHomography) throws away wrong matches and gives a matrix H
       that converts photo coordinates to map coordinates.
    4. With H we convert the photo centre and its corners to the map.

Run:
    python main.py --map data/global_map.png --frames data/crops --out results
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

SCALE = 0.5        # shrink photos before SIFT: they are zoomed in ~4x compared to the map
RATIO = 0.75       # ratio test: a match is kept only if it is clearly better than the 2nd best
MIN_INLIERS = 15   # fewer RANSAC "agreeing" matches than this -> we do not trust the result


def find_features(gray, sift):
    """Return coordinates of the keypoints (N x 2) and their descriptors (N x 128)."""
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    points = np.float32([k.pt for k in keypoints])
    return points, descriptors


def find_on_map(photo_gray, sift, matcher, map_points):
    """Find the place of one photo on the map.

    Returns (centre, corners, inliers) in map pixels, or None if the photo was not found.
    """
    photo = cv2.resize(photo_gray, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    photo_points, photo_desc = find_features(photo, sift)
    if photo_desc is None or len(photo_desc) < 2:
        return None

    # For each photo descriptor take the 2 most similar map descriptors.
    matches = matcher.knnMatch(photo_desc, k=2)
    good = [m for m, n in matches if m.distance < RATIO * n.distance]
    if len(good) < 4:                      # a homography needs at least 4 point pairs
        return None

    src = np.float32([photo_points[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([map_points[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask.sum() < MIN_INLIERS:
        return None

    h, w = photo.shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_on_map = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    centre = cv2.perspectiveTransform(np.float32([[[w / 2, h / 2]]]), H)[0, 0]
    return centre, corners_on_map, int(mask.sum())


def fill_gaps(centres):
    """Photos that were not found (None) get a point on the straight line between neighbours."""
    found = [i for i, c in enumerate(centres) if c is not None]
    x = np.interp(range(len(centres)), found, [centres[i][0] for i in found])
    y = np.interp(range(len(centres)), found, [centres[i][1] for i in found])
    return np.stack([x, y], axis=1)


def draw_route(map_bgr, path, boxes, was_found, out_path):
    """Draw the route on the map and save the part of the map around it."""
    img = map_bgr.copy()
    points = np.int32(np.round(path))

    for i, box in enumerate(boxes):                # footprint of every 5th photo + the last one
        if box is not None and (i % 5 == 0 or i == len(boxes) - 1):
            cv2.polylines(img, [np.int32(box)], True, (255, 200, 0), 3, cv2.LINE_AA)

    cv2.polylines(img, [points.reshape(-1, 1, 2)], False, (0, 220, 255), 4, cv2.LINE_AA)

    for i, (x, y) in enumerate(points):
        colour = (0, 0, 255) if was_found[i] else (0, 140, 255)   # orange = interpolated
        cv2.circle(img, (x, y), 7, colour, -1, cv2.LINE_AA)
        if i % 2 == 0:                                            # number of the point
            cv2.putText(img, str(i), (x + 9, y - 9), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)

    start, finish = points[0], points[-1]
    cv2.circle(img, tuple(int(v) for v in start), 16, (0, 200, 0), -1, cv2.LINE_AA)
    cv2.circle(img, tuple(int(v) for v in finish), 16, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.putText(img, "START", (start[0] + 22, start[1] + 8), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, "FINISH", (finish[0] + 22, finish[1] + 8), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (255, 255, 255), 3, cv2.LINE_AA)

    # Cut the map to the area of the route (+ a margin) so the details are visible.
    everything = np.vstack([points] + [np.int32(b) for b in boxes if b is not None])
    x0, y0 = np.maximum(everything.min(axis=0) - 250, 0)
    x1, y1 = everything.max(axis=0) + 250
    cv2.imwrite(str(out_path), img[y0:y1, x0:x1])


def main():
    parser = argparse.ArgumentParser(description="Reconstruct a UAV route on a satellite map")
    parser.add_argument("--map", default="data/global_map.png", help="satellite map")
    parser.add_argument("--frames", default="data/crops", help="folder with UAV photos")
    parser.add_argument("--out", default="results", help="folder for the results")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    map_bgr = cv2.imread(args.map)
    if map_bgr is None:
        raise SystemExit(f"Cannot read the map: {args.map}")

    # Photos are named crop_0.png, crop_1.png ... crop_10.png: sort them by the NUMBER,
    # otherwise crop_10 would come before crop_2.
    files = [f for f in Path(args.frames).iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
    files.sort(key=lambda f: int("".join(ch for ch in f.stem if ch.isdigit()) or 0))
    if not files:
        raise SystemExit(f"No photos in {args.frames}")

    # Features of the map are found once; FLANN is a fast search of similar descriptors.
    sift = cv2.SIFT_create()
    map_points, map_desc = find_features(cv2.cvtColor(map_bgr, cv2.COLOR_BGR2GRAY), sift)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matcher.add([map_desc])
    matcher.train()

    centres, boxes = [], []
    for i, f in enumerate(files):
        photo_gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        result = None if photo_gray is None else find_on_map(photo_gray, sift, matcher, map_points)
        if result is None:
            print(f"[{i:2d}] {f.name}: NOT FOUND")
            centres.append(None)
            boxes.append(None)
        else:
            centre, corners, inliers = result
            print(f"[{i:2d}] {f.name}: inliers={inliers:3d} centre=({centre[0]:.0f}, {centre[1]:.0f})")
            centres.append(centre)
            boxes.append(corners)

    found_count = sum(c is not None for c in centres)
    print(f"Found {found_count} of {len(files)} photos")
    if found_count < 2:
        raise SystemExit("Too few photos were found to build a route")

    path = fill_gaps(centres)
    was_found = [c is not None for c in centres]

    np.savetxt(out / "trajectory.csv", path, delimiter=",", header="x,y", comments="", fmt="%.1f")
    draw_route(map_bgr, path, boxes, was_found, out / "route.png")
    print(f"Results saved to {out.resolve()}")


if __name__ == "__main__":
    main()
