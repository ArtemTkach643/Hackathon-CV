"""Drawing of the reconstructed route on the map (image and optional video)."""
from pathlib import Path

import cv2
import numpy as np

GREEN, RED, YELLOW, CYAN, WHITE = (0, 200, 0), (0, 0, 255), (0, 220, 255), (255, 200, 0), (255, 255, 255)


def _draw_frame_box(img, corners, color, thickness):
    cv2.polylines(img, [np.int32(corners).reshape(-1, 1, 2)], True, color, thickness, cv2.LINE_AA)


def draw_route(map_bgr, path, measured, boxes, out_path, margin=250, label_every=2, box_every=5,
               preview_width=1600):
    """Save an image with the route, numbered points, start/finish and frame footprints.

    path     : (N, 2) route points in map pixels (after interpolation/smoothing)
    measured : bool array, False for points that were interpolated (drawn hollow)
    boxes    : list of (4, 2) corner arrays or None, footprints of the frames
    """
    img = map_bgr.copy()
    n = len(path)

    # Footprints of the "photos" with their real size and rotation.
    # Only every `box_every`-th footprint (plus the last one): drawing all of them
    # overlaps heavily because consecutive frames cover almost the same area.
    for i, b in enumerate(boxes):
        if b is not None and (i % box_every == 0 or i == n - 1):
            _draw_frame_box(img, b, CYAN, 3)

    pts = np.int32(np.round(path))
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, YELLOW, 4, cv2.LINE_AA)

    for i, (x, y) in enumerate(pts):
        if measured[i]:
            cv2.circle(img, (x, y), 7, RED, -1, cv2.LINE_AA)
        else:  # interpolated point: hollow circle
            cv2.circle(img, (x, y), 7, RED, 2, cv2.LINE_AA)
        if i % label_every == 0 or i == n - 1:
            cv2.putText(img, str(i), (x + 9, y - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, str(i), (x + 9, y - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)

    # Start / finish markers.
    for p, color, text in ((pts[0], GREEN, "START"), (pts[-1], (0, 0, 0), "FINISH")):
        cv2.circle(img, tuple(p), 16, color, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(p), 16, WHITE, 3, cv2.LINE_AA)
        cv2.putText(img, text, (p[0] + 22, p[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, text, (p[0] + 22, p[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 3, cv2.LINE_AA)

    # Crop the picture to the route area so details are visible.
    allpts = np.vstack([pts] + [np.int32(b) for b in boxes if b is not None])
    x0, y0 = np.maximum(allpts.min(0) - margin, 0)
    x1, y1 = allpts.max(0) + margin
    crop = img[y0:y1, x0:x1]
    cv2.imwrite(str(out_path), crop)
    # Light-weight preview (the full-resolution PNG is ~15 MB) for README / slides.
    k = min(1.0, preview_width / crop.shape[1])
    preview = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(Path(out_path).with_name("route_preview.jpg")), preview,
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    return img


def draw_verification(map_bgr, frame_files, results, out_path, n_show=4, tile=384):
    """Sanity-check image: each row pair = UAV frame (left) and the map warped into
    the frame's view with the found homography (right). If the localisation is right,
    buildings, roads and bridges coincide."""
    ok = [i for i, r in enumerate(results) if r is not None]
    if not ok:
        return None
    picks = sorted(set(ok[int(round(k))] for k in np.linspace(0, len(ok) - 1, min(n_show, len(ok)))))
    tiles = []
    for i in picks:
        frame = cv2.imread(str(frame_files[i]))
        r = results[i]
        h, w = frame.shape[:2]
        # H maps the *resized* frame to the map; add the resize to map the full frame.
        H_full = r.H @ np.diag([r.scale, r.scale, 1.0])
        warped = cv2.warpPerspective(map_bgr, np.linalg.inv(H_full), (w, h),
                                     flags=cv2.INTER_LINEAR)
        a = cv2.resize(frame, (tile, tile), interpolation=cv2.INTER_AREA)
        b = cv2.resize(warped, (tile, tile), interpolation=cv2.INTER_LINEAR)
        for img, text in ((a, f"frame {i}"), (b, "map")):
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        tiles.append(np.hstack([a, b]))
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]
    grid = np.vstack(rows)
    cv2.imwrite(str(out_path), grid)
    return grid


def make_video(map_bgr, path, boxes, out_path, fps=4, margin=250, repeat=6):
    """Video with a moving UAV position, its footprint and the trail travelled so far."""
    allpts = np.vstack([np.int32(path)] + [np.int32(b) for b in boxes if b is not None])
    x0, y0 = np.maximum(allpts.min(0) - margin, 0)
    x1, y1 = allpts.max(0) + margin
    view = map_bgr[y0:y1, x0:x1]
    h, w = view.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    offset = np.array([x0, y0])
    for i in range(len(path)):
        frame = view.copy()
        trail = np.int32(np.round(path[: i + 1] - offset)).reshape(-1, 1, 2)
        if len(trail) > 1:
            cv2.polylines(frame, [trail], False, YELLOW, 4, cv2.LINE_AA)
        if boxes[i] is not None:
            _draw_frame_box(frame, boxes[i] - offset, CYAN, 3)
        cv2.circle(frame, tuple(int(v) for v in trail[-1, 0]), 10, RED, -1, cv2.LINE_AA)
        cv2.putText(frame, f"frame {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3, cv2.LINE_AA)
        for _ in range(repeat if i == len(path) - 1 else 1):
            writer.write(frame)
    writer.release()
