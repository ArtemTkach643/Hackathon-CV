# UAV route reconstruction from satellite imagery (classic CV)

Given a large satellite map and 50 consecutive "photos" from a UAV (satellite crops taken
at another scale and rotation), the program finds where each photo was taken on the map and
draws the resulting route.

![route](results/route_preview.jpg)

Full-resolution image: [`results/route.png`](results/route.png).

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
   footprint. If the frame fails at the main scale (0.5), it is retried at scales 1.0 and 0.25
   (**multi-scale fallback**) before it is declared lost.
6. **Post-processing** (`trajectory.py`) - jump outliers are removed, gaps are linearly
   interpolated (shown as hollow points), optional moving-average smoothing (`--smooth`).
   In this dataset none of them were needed: all frames matched and no outliers were found.

## Results

* 50 / 50 frames localised, 112-359 RANSAC inliers per frame (mean ~260).
* The route is smooth (step between frames 63-90 px, total ~3 770 px) from frame 0 (START) to frame 49 (FINISH).
* `results/verification.png` - sanity check: frame (left) vs. the map warped into the frame's
  view (right); buildings, bridges and rails coincide.

### Accuracy benchmark with ground truth (`benchmark.py`)

The task data has no ground-truth coordinates, so `benchmark.py` creates its own: it cuts synthetic
frames out of the map with a known centre, random rotation (0-360 deg), random footprint and a small
perspective tilt, spoils them and measures the distance between the found and the true centre.
A frame is *correct* if the error is below 10 % of its footprint, *wrong* if it was accepted but is
farther (the dangerous case), *lost* if it was rejected.

60 frames per condition, multi-scale fallback on:

| condition | what is simulated | correct | wrong | lost | median error |
|-----------|-------------------|---------|-------|------|--------------|
| clean | rotation + scale + tilt only | 60/60 | 0 | 0 | 0.35 px |
| lighting | brightness x0.6-1.4, +/-50 | 60/60 | 0 | 0 | 0.35 px |
| blur | Gaussian sigma 1.5-3 | 60/60 | 0 | 0 | 0.35 px |
| noise | sensor noise sigma 15 | 60/60 | 0 | 0 | 0.36 px |
| clouds | ~1/3 of the frame covered | 60/60 | 0 | 0 | 0.35 px |
| season | hue/saturation shift, low contrast | 60/60 | 0 | 0 | 0.35 px |
| all-combined | season + clouds + blur + noise | 59/60 (baseline 58/60) | 0 | 1 | 0.36 px |
| low-altitude | footprint 70-100 px (5-7x zoom) | 60/60 | 0 | 0 | 0.36 px |
| high-altitude | footprint 450-600 px | 60/60 | 0 | 0 | 0.35 px |

**537 / 540 frames correct, 0 wrong answers, sub-pixel error.** End-to-end check on a synthetic
50-frame flight (`--demo-flight`): 50/50 frames, mean route error 0.36 px, max 0.42 px.
Full table: `results/benchmark.csv`.

Caveat: synthetic frames are cut from the same map, so this measures robustness to geometry and
image degradations, not to really different imagery (other satellite, other year). The run above
used the map area shown in `route.png` (overlays removed); re-run it on the real map with
`python benchmark.py --map data/global_map.png`.

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
python main.py --video --verify # also results/route.mp4 and results/verification.png

python benchmark.py --map data/global_map.png --n 60               # accuracy benchmark
python benchmark.py --map data/global_map.png --demo-flight data/demo  # synthetic flight + gt.csv
```

Options: `--scale`, `--fallback-scales`, `--ratio`, `--min-inliers`, `--smooth`, `--video`,
`--verify` (see `python main.py -h`).

Outputs in `results/`: `route.png` (map with route, numbered points, START/FINISH and every
5th frame footprint), `route_preview.jpg` (light copy for README/slides), `trajectory.csv`
(route in map pixels), `frames_report.csv` (per frame: status, position, inliers, reprojection
error, scale used), `verification.png` (optional), `route.mp4` (optional), `map_features.npz`
(cache, rebuilt automatically if the map changes; safe to delete).

## Pitch

Slides for the hackathon pitch (11 slides, speaker notes included):

* English: [`pitch/UAV_Route_Pitch.pptx`](pitch/UAV_Route_Pitch.pptx)
* Ukrainian: [`pitch/UAV_Route_Pitch_UA.pptx`](pitch/UAV_Route_Pitch_UA.pptx)

## Project layout

| File | Purpose |
|------|---------|
| `main.py` | CLI, runs the pipeline over all frames |
| `localizer.py` | SIFT + FLANN + ratio test + RANSAC homography for one frame |
| `trajectory.py` | outlier rejection, gap interpolation, smoothing |
| `visualize.py` | route image and video |
| `benchmark.py` | accuracy/robustness benchmark with synthetic ground truth |
| `explore_single.py` | step-by-step prototype for a single frame |
| `pitch/` | pitch slides (EN and UA) |

## Limitations

* Frame order is taken from the file numbering (`crop_0` ... `crop_49`).
* Matching relies on the frames and the map being the same kind of imagery (here: crops of the
  same map). Real UAV photos with other lighting/season would need a stronger matcher
  (e.g. SuperPoint + LightGlue or LoFTR).
* On the task data accuracy is checked visually (`verification.png`) and by the reprojection error;
  the numeric accuracy comes from the synthetic benchmark.
* Coordinates are in map pixels; turning them into GPS needs the map's georeference (GeoTIFF).
