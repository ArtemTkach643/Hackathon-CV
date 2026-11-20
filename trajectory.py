"""Post-processing of the raw per-frame positions: outlier removal, gap filling, smoothing."""
import numpy as np


def reject_outliers(points, max_jump_factor=4.0):
    """Mark as None the points that jump much farther than the typical step.

    A wrong homography usually places the frame far from where the drone really is,
    so its distance to the neighbours is much larger than the median step length.
    """
    pts = [None if p is None else np.asarray(p, float) for p in points]
    valid = [i for i, p in enumerate(pts) if p is not None]
    if len(valid) < 5:
        return pts
    steps = [np.linalg.norm(pts[b] - pts[a]) for a, b in zip(valid[:-1], valid[1:])]
    limit = max_jump_factor * max(np.median(steps), 1.0)
    out = list(pts)
    for k in range(1, len(valid) - 1):
        i, prev, nxt = valid[k], valid[k - 1], valid[k + 1]
        # Bad only if it is far from BOTH neighbours (a real fast turn is near one of them).
        if (np.linalg.norm(pts[i] - pts[prev]) > limit and
                np.linalg.norm(pts[i] - pts[nxt]) > limit):
            out[i] = None
    return out


def interpolate_missing(points):
    """Fill None entries by linear interpolation between neighbouring valid points.

    Returns (array (N, 2), bool array `measured` telling which points are real).
    """
    n = len(points)
    measured = np.array([p is not None for p in points])
    if measured.sum() < 2:
        raise ValueError("Not enough successfully matched frames to build a trajectory")
    idx = np.arange(n)
    xs = np.array([p[0] for p in points if p is not None])
    ys = np.array([p[1] for p in points if p is not None])
    x = np.interp(idx, idx[measured], xs)
    y = np.interp(idx, idx[measured], ys)
    return np.stack([x, y], axis=1), measured


def smooth(path, window=3):
    """Moving average with edge padding. `window` must be odd; 1 disables smoothing."""
    if window <= 1:
        return path
    pad = window // 2
    padded = np.pad(path, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.stack([np.convolve(padded[:, c], kernel, mode="valid") for c in range(2)], axis=1)
