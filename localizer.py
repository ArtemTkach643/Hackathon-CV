"""Localisation of a single UAV frame on a big satellite map.

Pipeline for one frame:
    SIFT keypoints/descriptors -> FLANN nearest-neighbour matching -> Lowe ratio test
    -> homography with RANSAC -> projected frame corners / centre on the map.

If a frame cannot be matched at the main scale, a few fallback scales are tried
before the frame is declared lost (multi-scale fallback).
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Localization:
    """Result of locating one frame on the map (all coordinates are map pixels)."""
    H: np.ndarray          # 3x3 homography: (resized) frame -> map
    corners: np.ndarray    # (4, 2) frame corners projected on the map (TL, TR, BR, BL)
    center: np.ndarray     # (2,) frame centre projected on the map
    inliers: int           # number of RANSAC inliers (confidence of the match)
    reproj_err: float      # median reprojection error of the inliers, map pixels
    scale: float           # frame scale at which the match succeeded


class MapLocalizer:
    def __init__(self, map_bgr, cache_path=None, ratio=0.75, min_inliers=15,
                 ransac_thresh=5.0):
        self.ratio = ratio
        self.min_inliers = min_inliers
        self.ransac_thresh = ransac_thresh
        self.sift = cv2.SIFT_create()

        # Map features are computed once (it is the slow part: ~150k keypoints)
        # and cached on disk, because they never change between runs.
        gray = cv2.cvtColor(map_bgr, cv2.COLOR_BGR2GRAY)
        self.map_pts, self.map_desc = self._map_features(gray, cache_path)

        # FLANN = fast approximate nearest neighbour search (KD-trees).
        # Much faster than brute force on 150k descriptors.
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        self.matcher.add([self.map_desc])
        self.matcher.train()

    @staticmethod
    def _fingerprint(gray):
        """Cheap map signature, so a cache built for another map is never reused."""
        small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        return np.array([*gray.shape, int(small.astype(np.int64).sum())], dtype=np.int64)

    def _map_features(self, gray, cache_path):
        fp = self._fingerprint(gray)
        if cache_path and Path(cache_path).exists():
            data = np.load(cache_path)
            if "fp" in data and np.array_equal(data["fp"], fp):
                return data["pts"], data["desc"]
            print("Map changed since the cache was built - recomputing map features")
        kps, desc = self.sift.detectAndCompute(gray, None)
        if desc is None:
            raise ValueError("No SIFT features found on the map")
        pts = np.float32([k.pt for k in kps])
        if cache_path:
            np.savez(cache_path, pts=pts, desc=desc, fp=fp)
        return pts, desc

    def locate(self, frame_bgr, scale=0.5, fallback_scales=(1.0, 0.25)):
        """Return the Localization of a frame, or None if it could not be matched reliably.

        The main `scale` is tried first; if it fails, the `fallback_scales` are tried in order.
        """
        for s in (scale, *[f for f in fallback_scales if f != scale]):
            res = self._locate_at(frame_bgr, s)
            if res is not None:
                return res
        return None

    def _locate_at(self, frame_bgr, scale):
        """One matching attempt at a fixed frame scale.

        `scale` resizes the frame before SIFT. Frames are ~4x "zoomed in" compared
        to the map, so shrinking them can bring features closer to the map's scale.
        Result coordinates are still in map pixels (the homography absorbs the scale).
        """
        gray = frame_bgr if frame_bgr.ndim == 2 else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if scale != 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        kps, desc = self.sift.detectAndCompute(gray, None)
        if desc is None or len(kps) < 4:
            return None

        # For every frame descriptor take 2 nearest map descriptors, keep the match only
        # if the best one is clearly better than the second (Lowe's ratio test).
        pairs = self.matcher.knnMatch(desc, k=2)
        good = [p[0] for p in pairs
                if len(p) == 2 and p[0].distance < self.ratio * p[1].distance]
        if len(good) < 4:
            return None

        src = np.float32([kps[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = self.map_pts[[m.trainIdx for m in good]].reshape(-1, 1, 2)

        # RANSAC throws away wrong matches and keeps the largest consistent group.
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
        if H is None:
            return None
        inl = mask.ravel().astype(bool)
        inliers = int(inl.sum())
        if inliers < self.min_inliers:
            return None

        # How well the inliers fit the homography (lower = more precise localisation).
        proj = cv2.perspectiveTransform(src[inl], H)
        reproj_err = float(np.median(np.linalg.norm(proj - dst[inl], axis=2)))

        h, w = gray.shape
        corners = cv2.perspectiveTransform(
            np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), H).reshape(-1, 2)
        center = cv2.perspectiveTransform(
            np.float32([[w / 2, h / 2]]).reshape(-1, 1, 2), H).reshape(2)

        # Sanity check: the projected frame must be a proper convex quadrilateral,
        # otherwise the homography is degenerate (a "bow-tie" = wrong match).
        if not cv2.isContourConvex(np.int32(np.round(corners)).reshape(-1, 1, 2)):
            return None

        return Localization(H=H, corners=corners, center=center, inliers=inliers,
                            reproj_err=reproj_err, scale=scale)
