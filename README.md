# 3D LiDAR Perception for Autonomous Driving

Frame-by-frame detection and tracking of vehicles in 3D LiDAR point clouds.
Each frame is reduced from a raw sweep to a set of vehicle detections with
position, axis-aligned bounding box, and a stable track ID carried across
frames.

Built for CSCI 431 (Introduction to Computer Vision), Rochester Institute of
Technology, Spring 2026.

## Pipeline

`src/perception_3d.py` processes each frame through five stages:

| Stage | Method | Key parameters |
| --- | --- | --- |
| Downsample | Voxel grid | `voxel_size=0.15` |
| Crop | Rectangular ROI | x,y ∈ ±25 m, z ∈ (−3, 4) m |
| Ground removal | RANSAC plane segmentation | `distance_threshold=0.18`, 100 iterations |
| Cluster | DBSCAN | `eps=1.2`, `min_points=12` |
| Classify | Size heuristics | see below |

### Vehicle filter

A cluster is accepted as a vehicle when its axis-aligned extent falls inside
passenger-car bounds — long side 2.0–5.5 m, short side 1.2–3.0 m, height
0.8–2.5 m — with at least 15 points, and its base sits below 2.0 m so objects
floating above the road surface are rejected.

### Tracking

`SimpleTracker` associates detections between frames by nearest neighbour.
Each existing track claims the closest unmatched detection within 3.0 m;
unmatched detections start new tracks, and a track survives 2 consecutive
missed frames before being dropped. This is what gives each vehicle a stable
ID and, from successive centers, its motion vector.

## Layout

```
src/perception_3d.py    The pipeline — detection, tracking, CSV output, visualization
src/3d_demo.py          Minimal Open3D point-cloud viewer
tune_offset.py          Sweeps a center-estimation offset to tune position accuracy
perception_results/     Per-frame output, one CSV per frame (100 frames)
screenshots/            Rendered frames plus their Open3D camera viewpoints
```

## Running it

```bash
pip install -r requirements.txt
python src/perception_3d.py --data_path /path/to/pcd/frames
```

The input frames are numbered `.pcd` files (`0.pcd`, `1.pcd`, …). Useful flags:
`--start_index` / `--end_index` to process a subset, `--visualize` to open an
Open3D window, `--visualize_every_frame` to step through all of them, and
`--output_path` to redirect the CSVs.

**The dataset is not included here** — the point clouds are course-provided
and not mine to redistribute. Point `--data_path` at your own directory of
`.pcd` frames.

## Notes on accuracy

Position estimation was the hard part. How a vehicle's center is defined —
cluster centroid, bounding-box center, or a blend — moves the error more than
any other single choice, and `tune_offset.py` exists to sweep that offset
against known positions. The other persistent tension is between detecting
every vehicle in every frame and not accumulating false positives from
roadside structure; the size filter above is where that balance is set.
