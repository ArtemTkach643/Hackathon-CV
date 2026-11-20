# UAV route reconstruction from satellite imagery (classic CV)

Given a large satellite map and 50 consecutive "photos" from a UAV (satellite crops taken
at another scale and rotation), the program finds where each photo was taken on the map and
draws the resulting route.

![route](results/route.png)

## Approach

Every frame is localised **independently on the global map** (not frame-to-frame), so errors
do not accumulate over the flight.

1. **Features** - SIFT keypoints and 128-d descriptors on the grayscale map (computed once
   and cached in `results/map_features.npz`) and on each frame. SIFT is invariant to scale
   and rotation, which is exactly the difference between the frames and the map.
2. **Matching** - FLANN (KD-tree) finds the 2 nearest map descriptors for every frame descriptor;
   Lowe's ratio test (`best < 0.75 * second best`) keeps only unambiguous matches.
3. **Geometry** - `cv2.findHomography` with RANSAC (5 px threshold) estimates the frame -> map
   homography and rejects remaining wrong matches. Ground is almost planar, so a homography
   describes the view change well.
4. **Projection** - the frame centre gives a route point; the four projected corners give the
   footprint (size and rotation preserved).
5. **Validation** - a frame is accepted only with >= 15 RANSAC inliers and a convex projected
   footprint.
6. **Post-processing** (`trajectory.py`) - jump outliers are removed, gaps are linearly
   interpolated (shown as hollow points), optional moving-average smoothing (`--smooth`).
   In this dataset none of them were needed: all frames matched and no outliers were found.

## Results

* 50 / 50 frames localised, 112-359 RANSAC inliers per frame (mean ~260).
* The route is smooth (step between frames ~64 px) from frame 0 (START) to frame 49 (FINISH).
* `results/verification.png` - sanity check: frame (left) vs. the map warped into the frame's
  view (right); buildings, bridges and rails coincide.

### Downscaling experiment (frame scale before SIFT)

| scale | frames matched | mean inliers | min inliers | time (50 frames) |
|-------|----------------|--------------|-------------|------------------|
| 1.0   | 50/50          | 261          | 112         | 22 s             |
| 0.5   | 50/50          | 263          | 109         | 6 s              |
| 0.35  | 50/50          | 261          | 108         | 3 s              |
| 0.25  | 50/50          | 256          | 108         | 2 s              |

Quality is practically unchanged down to 0.25 while speed improves ~10x, so the default is
`--scale 0.5` (a safe middle ground). Time excludes the one-off map feature extraction.

## Usage

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python main.py --map data/global_map.png --frames data/crops --out results
python main.py --video          # additionally writes results/route.mp4
```

Options: `--scale`, `--ratio`, `--min-inliers`, `--smooth`, `--video` (see `python main.py -h`).

Outputs in `results/`: `route.png` (map with route, numbered points, START/FINISH and every
5th frame footprint), `trajectory.csv` (route in map pixels), `route.mp4` (optional),
`map_features.npz` (cache; safe to delete).

## Project layout

| File | Purpose |
|------|---------|
| `main.py` | CLI, runs the pipeline over all frames |
| `localizer.py` | SIFT + FLANN + ratio test + RANSAC homography for one frame |
| `trajectory.py` | outlier rejection, gap interpolation, smoothing |
| `visualize.py` | route image and video |
| `explore_single.py` | step-by-step prototype for a single frame |

## Limitations

* Frame order is taken from the file numbering (`crop_0` ... `crop_49`).
* Matching relies on the frames and the map being the same kind of imagery (here: crops of the
  same map). Real UAV photos with other lighting/season would need a stronger matcher
  (e.g. SuperPoint + LightGlue or LoFTR).
* Accuracy was checked visually; no ground-truth coordinates were available.
