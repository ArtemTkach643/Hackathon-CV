# UAV route reconstruction from satellite imagery (classic CV)

Given a large satellite map and 50 consecutive "photos" from a UAV (satellite crops taken at
another scale and rotation), the program finds where each photo was taken on the map and
draws the resulting route.

![route](results/route_preview.jpg)

## How it works

Every photo is placed **directly on the big map** (not relative to the previous photo), so
errors do not pile up along the flight.

1. **SIFT** finds keypoints (corners, crossroads, roof edges) on the map and on the photo.
   Each keypoint gets a descriptor - 128 numbers describing its surroundings. SIFT works even
   if the photo is rotated or has another scale, which is exactly our case.
2. **Matching** - for each photo descriptor we look for the two most similar map descriptors
   (FLANN, a fast approximate search). The *ratio test* keeps a match only if the best
   candidate is clearly better than the second one (`0.75`), otherwise it is ambiguous.
3. **Homography + RANSAC** - `cv2.findHomography` finds a 3x3 matrix that converts photo
   coordinates to map coordinates. RANSAC ignores wrong matches and keeps the biggest
   consistent group; the photo is accepted if at least 15 matches agree (*inliers*).
4. **Route point** - the photo centre converted with this matrix is one point of the route;
   the four converted corners give the footprint of the photo (size and rotation are kept).
5. **Missing photos** - if a photo is not found, its point is placed on the straight line
   between the neighbouring found photos (drawn in orange).

## Results

* 50 / 50 photos found, 109-359 inliers per photo (mean ~260).
* The route is smooth from photo 0 (START) to photo 49 (FINISH).
* `results/verification.png` - sanity check: photo (left) vs. the map warped into the photo's
  view (right); buildings, bridges and rails coincide.

### Downscaling experiment (photo scale before SIFT)

| scale | photos found | mean inliers | min inliers | time (50 photos) |
|-------|--------------|--------------|-------------|------------------|
| 1.0   | 50/50        | 261          | 112         | 22 s             |
| 0.5   | 50/50        | 263          | 109         | 6 s              |
| 0.35  | 50/50        | 261          | 108         | 3 s              |
| 0.25  | 50/50        | 256          | 108         | 2 s              |

The quality is practically the same, but smaller photos are much faster, so `SCALE = 0.5`
(constant at the top of `main.py`).

## Usage

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python main.py --map data/global_map.png --frames data/crops --out results
```

The task data goes into `data/` (it is not stored in the repository). Output in `results/`:
`route.png` (map with the route, numbered points, START/FINISH, footprint of every 5th photo)
and `trajectory.csv` (route points in map pixels). The run takes about 12 seconds.

## Files

| File | Purpose |
|------|---------|
| `main.py` | the whole solution (about 160 lines, every function is commented) |
| `explore_single.py` | step-by-step prototype for a single photo |
| `pitch/` | pitch slides (EN and UA) |

## Other versions of the code (git branches)

This branch (`main`) is the simple version, written to be easy to read and explain.
Nothing was thrown away:

* `complex-version` - the same idea split into modules (`localizer.py`, `trajectory.py`,
  `visualize.py`) with a map-features cache, outlier rejection, smoothing and a route video.
* `benchmark-and-pitch` (pull request #1) - on top of that: accuracy benchmark on synthetic
  frames with known ground truth (`benchmark.py`), multi-scale fallback, per-frame CSV report,
  `--verify`. **The slides in `pitch/` quote the benchmark numbers from this branch** (e.g.
  "0 wrong answers in 540 tests"), they are not produced by the code in `main`.

## Limitations

* Photo order is taken from the file numbers (`crop_0` ... `crop_49`).
* A photo that is not found is replaced by a point between its neighbours; there is no check
  for wrong-but-confident matches (see `complex-version` for outlier rejection).
* Works because photos and map are the same kind of imagery. Real UAV photos with another
  season or lighting would need a stronger matcher (SuperPoint + LightGlue, LoFTR).
* Coordinates are in map pixels; GPS would need the georeference of the map.
