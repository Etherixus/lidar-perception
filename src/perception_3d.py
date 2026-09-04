from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d
from tqdm.auto import trange


# ============================================================
# Data model
# ============================================================

@dataclass
class Vehicle:
    """Represents one tracked vehicle for a single frame.

    Attributes:
        vehicle_id: Unique ID assigned by the tracker.
        position_x/y/z: Estimated vehicle center position.
        mvec_x/y/z: Motion vector since the previous frame.
        bbox_*: Axis-aligned 3D bounding box coordinates.
    """

    vehicle_id: int = -1

    # Vehicle center position
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0

    # Motion vector (current center - previous center)
    mvec_x: float = 0.0
    mvec_y: float = 0.0
    mvec_z: float = 0.0

    # Bounding box coordinates
    bbox_x_min: float = 0.0
    bbox_x_max: float = 1.0
    bbox_y_min: float = 0.0
    bbox_y_max: float = 1.0
    bbox_z_min: float = 0.0
    bbox_z_max: float = 1.0

    @classmethod
    def csv_header(cls) -> str:
        """Return a CSV header row using dataclass field names."""
        return ",".join(cls.__annotations__.keys())

    def csv_row(self) -> str:
        """Return this vehicle as a CSV row in field order."""
        return ",".join(str(self.__dict__[field]) for field in self.__annotations__.keys())


# ============================================================
# File I/O helpers
# ============================================================


def write_csv_helper(file: Path, vehicles: Iterable[Vehicle]) -> None:
    """Write tracked vehicle data for one frame to a CSV file."""
    with open(file, "w", encoding="utf-8") as f:
        f.write(Vehicle.csv_header() + "\n")
        for vehicle in vehicles:
            f.write(vehicle.csv_row() + "\n")



def load_point_cloud(path_to_cloud: Path) -> o3d.geometry.PointCloud:
    """Load a point cloud from a .pcd file."""
    return o3d.io.read_point_cloud(path_to_cloud)


# ============================================================
# Visualization helpers
# ============================================================


def save_visualization_frame(
    display_pcd: o3d.geometry.PointCloud,
    detections: list[dict],
    save_path: Path,
) -> None:
    """Render a point cloud with detection boxes and save it as an image."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1200, height=800)

    # Use a copy so the original point cloud is not modified.
    pcd_copy = o3d.geometry.PointCloud(display_pcd)
    pcd_copy.paint_uniform_color([0.65, 0.65, 0.65])
    vis.add_geometry(pcd_copy)

    # Draw every detection as a red bounding box.
    for det in detections:
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=det["min_bound"],
            max_bound=det["max_bound"],
        )
        bbox.color = (1, 0, 0)
        vis.add_geometry(bbox)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(save_path))
    vis.destroy_window()



def detections_to_bboxes(detections: list[dict]) -> list[o3d.geometry.AxisAlignedBoundingBox]:
    """Convert detection dictionaries into Open3D bounding box objects."""
    bboxes = []
    for det in detections:
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=det["min_bound"],
            max_bound=det["max_bound"],
        )
        bbox.color = (1, 0, 0)
        bboxes.append(bbox)
    return bboxes



def centers_to_pointcloud(detections: list[dict]) -> o3d.geometry.PointCloud:
    """Create a point cloud containing detection centers for display."""
    centers = [det["center"] for det in detections]
    pc = o3d.geometry.PointCloud()

    if centers:
        center_array = np.array(centers, dtype=float)
        pc.points = o3d.utility.Vector3dVector(center_array)

        # Color all centers green.
        colors = np.tile(np.array([[0.0, 1.0, 0.0]]), (len(center_array), 1))
        pc.colors = o3d.utility.Vector3dVector(colors)

    return pc



def visualize_frame(
    display_pcd: o3d.geometry.PointCloud,
    detections: list[dict],
    window_name: str = "LiDAR Detections",
) -> None:
    """Display one frame interactively with boxes and detection centers."""
    geometries = [display_pcd]

    bboxes = detections_to_bboxes(detections)
    centers = centers_to_pointcloud(detections)

    geometries.extend(bboxes)
    if len(centers.points) > 0:
        geometries.append(centers)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=1200,
        height=800,
    )


# ============================================================
# Tracking helpers
# ============================================================


class SimpleTracker:
    """A nearest-neighbor tracker for associating detections across frames.

    Each existing track is matched to the closest unmatched detection within
    `max_match_distance`. Tracks that go unmatched for too many frames are
    removed.
    """

    def __init__(self, max_match_distance: float = 3.0, max_missed_frames: int = 2):
        self.max_match_distance = max_match_distance
        self.max_missed_frames = max_missed_frames
        self.next_id = 0
        self.tracks: dict[int, dict] = {}

    def update(self, detections: list[dict]) -> list[Vehicle]:
        """Match detections to tracks and return frame-level Vehicle objects."""
        vehicles: list[Vehicle] = []
        unmatched_detection_indices = set(range(len(detections)))

        # Try to match every current track to the nearest available detection.
        for track_id, track in list(self.tracks.items()):
            prev_center = track["center"]
            best_idx = None
            best_dist = float("inf")

            for det_idx in unmatched_detection_indices:
                det_center = detections[det_idx]["center"]
                dist = np.linalg.norm(det_center - prev_center)

                if dist < best_dist and dist <= self.max_match_distance:
                    best_dist = dist
                    best_idx = det_idx

            if best_idx is not None:
                # Existing track matched to a detection.
                det = detections[best_idx]
                curr_center = det["center"]
                velocity = curr_center - prev_center

                self.tracks[track_id] = {
                    "center": curr_center,
                    "missed": 0,
                }
                unmatched_detection_indices.remove(best_idx)

                vehicles.append(
                    Vehicle(
                        vehicle_id=track_id,
                        position_x=float(curr_center[0]),
                        position_y=float(curr_center[1]),
                        position_z=float(curr_center[2]),
                        mvec_x=float(velocity[0]),
                        mvec_y=float(velocity[1]),
                        mvec_z=float(velocity[2]),
                        bbox_x_min=float(det["min_bound"][0]),
                        bbox_x_max=float(det["max_bound"][0]),
                        bbox_y_min=float(det["min_bound"][1]),
                        bbox_y_max=float(det["max_bound"][1]),
                        bbox_z_min=float(det["min_bound"][2]),
                        bbox_z_max=float(det["max_bound"][2]),
                    )
                )
            else:
                # No match found for this track in the current frame.
                self.tracks[track_id]["missed"] += 1

        # Remove tracks that have been missing for too long.
        dead_track_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track["missed"] > self.max_missed_frames
        ]
        for track_id in dead_track_ids:
            del self.tracks[track_id]

        # Create new tracks for detections that were not matched.
        for det_idx in unmatched_detection_indices:
            det = detections[det_idx]
            center = det["center"]

            track_id = self.next_id
            self.next_id += 1

            self.tracks[track_id] = {
                "center": center,
                "missed": 0,
            }

            vehicles.append(
                Vehicle(
                    vehicle_id=track_id,
                    position_x=float(center[0]),
                    position_y=float(center[1]),
                    position_z=float(center[2]),
                    mvec_x=0.0,
                    mvec_y=0.0,
                    mvec_z=0.0,
                    bbox_x_min=float(det["min_bound"][0]),
                    bbox_x_max=float(det["max_bound"][0]),
                    bbox_y_min=float(det["min_bound"][1]),
                    bbox_y_max=float(det["max_bound"][1]),
                    bbox_z_min=float(det["min_bound"][2]),
                    bbox_z_max=float(det["max_bound"][2]),
                )
            )

        return vehicles


# ============================================================
# Point cloud processing helpers
# ============================================================


def crop_roi(
    pcd: o3d.geometry.PointCloud,
    x_range: tuple[float, float] = (-25.0, 25.0),
    y_range: tuple[float, float] = (-25.0, 25.0),
    z_range: tuple[float, float] = (-3.0, 4.0),
) -> o3d.geometry.PointCloud:
    """Crop the point cloud to a region of interest (ROI)."""
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return pcd

    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
        & (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
        & (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )

    cropped = o3d.geometry.PointCloud()
    cropped.points = o3d.utility.Vector3dVector(points[mask])

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        cropped.colors = o3d.utility.Vector3dVector(colors[mask])

    return cropped



def remove_ground_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.18,
    ransac_n: int = 3,
    num_iterations: int = 100,
) -> o3d.geometry.PointCloud:
    """Remove the ground plane using RANSAC plane segmentation."""
    if len(pcd.points) < 50:
        return pcd

    _, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    return pcd.select_by_index(inliers, invert=True)



def cluster_objects(
    pcd: o3d.geometry.PointCloud,
    eps: float = 1.2,
    min_points: int = 12,
) -> list[o3d.geometry.PointCloud]:
    """Cluster non-ground points into candidate objects using DBSCAN."""
    if len(pcd.points) == 0:
        return []

    labels = np.array(
        pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points,
            print_progress=False,
        )
    )

    if labels.size == 0:
        return []

    max_label = labels.max()
    if max_label < 0:
        return []

    clusters = []
    for label in range(max_label + 1):
        indices = np.where(labels == label)[0]
        if len(indices) == 0:
            continue
        clusters.append(pcd.select_by_index(indices))

    return clusters



def is_likely_vehicle(cluster: o3d.geometry.PointCloud) -> bool:
    """Filter a cluster based on simple vehicle-like size heuristics."""
    points = np.asarray(cluster.points)
    if len(points) < 15:
        return False

    aabb = cluster.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    min_bound = aabb.get_min_bound()

    dx, dy, dz = extent
    long_side = max(dx, dy)
    short_side = min(dx, dy)

    # Rough size checks for common passenger vehicles.
    if long_side < 2.0 or long_side > 5.5:
        return False
    if short_side < 1.2 or short_side > 3.0:
        return False
    if dz < 0.8 or dz > 2.5:
        return False

    # Reject objects that appear unrealistically high above the ground.
    if min_bound[2] > 2.0:
        return False

    return True



def extract_vehicle_detections(pcd: o3d.geometry.PointCloud) -> list[dict]:
    """Convert clustered objects into vehicle detections.

    Each detection stores:
        - center: representative center point used for tracking
        - min_bound / max_bound: AABB corners used for visualization/output
    """
    clusters = cluster_objects(pcd, eps=1.2, min_points=12)
    detections: list[dict] = []

    for cluster in clusters:
        if not is_likely_vehicle(cluster):
            continue

        points = np.asarray(cluster.points)
        aabb = cluster.get_axis_aligned_bounding_box()

        min_bound = np.array(aabb.get_min_bound(), dtype=float)
        max_bound = np.array(aabb.get_max_bound(), dtype=float)

        # Use mean X/Y as the horizontal center.
        # Use a low Z anchor so the tracked center stays near the base of the car.
        center = np.array(
            [
                np.mean(points[:, 0]),
                np.mean(points[:, 1]),
                min_bound[2] + 0.3,
            ],
            dtype=float,
        )

        detections.append(
            {
                "center": center,
                "min_bound": min_bound,
                "max_bound": max_bound,
            }
        )

    return detections



def process_frame(
    pcd: o3d.geometry.PointCloud,
) -> tuple[o3d.geometry.PointCloud, list[dict]]:
    """Run the full perception pipeline on one point cloud frame.

    Pipeline steps:
        1. Downsample for speed.
        2. Crop to the region of interest.
        3. Remove the ground plane.
        4. Extract vehicle detections from remaining clusters.
    """
    pcd = pcd.voxel_down_sample(voxel_size=0.15)

    pcd = crop_roi(
        pcd,
        x_range=(-50.0, 100.0),
        y_range=(-50.0, 50.0),
        z_range=(-3.0, 4.0),
    )

    pcd = remove_ground_plane(
        pcd,
        distance_threshold=0.18,
        ransac_n=3,
        num_iterations=100,
    )

    detections = extract_vehicle_detections(pcd)
    return pcd, detections


# ============================================================
# Main frame processing loop
# ============================================================


def main(
    data_path: Path,
    output_path: Path = Path("perception_results"),
    start_index: int = 0,
    end_index: int = -1,
    visualize: bool = False,
    visualize_every_frame: bool = False,
    visualize_processed: bool = True,
) -> None:
    """Process a sequence of .pcd frames and save tracking results.

    For each frame:
        - load point cloud
        - run vehicle detection
        - update tracker
        - save visualization image
        - optionally display an interactive visualization
        - save CSV output
    """
    frame_files = sorted(data_path.glob("*.pcd"), key=lambda p: int(p.stem))

    if len(frame_files) == 0:
        print(f"No .pcd files found in: {data_path.resolve()}")
        return

    # If end_index is negative, interpret it relative to the end of the list.
    if end_index < 0:
        end_index = len(frame_files) + end_index

    tracker = SimpleTracker(max_match_distance=3.0, max_missed_frames=2)

    viz_folder = output_path / "frames"
    viz_folder.mkdir(exist_ok=True)

    for frame_number in trange(start_index, end_index + 1, desc="Processing Frames"):
        frame_path = data_path / f"{frame_number}.pcd"
        if not frame_path.exists():
            print(f"Skipping missing frame: {frame_path}")
            continue

        raw_pcd = load_point_cloud(frame_path)
        processed_pcd, detections = process_frame(raw_pcd)
        vehicles = tracker.update(detections)

        # Save a rendered PNG for this frame.
        pcd_to_save = processed_pcd if visualize_processed else raw_pcd
        save_visualization_frame(
            pcd_to_save,
            detections,
            viz_folder / f"frame_{frame_number:04d}.png",
        )

        # Optionally show the frame interactively.
        if visualize:
            should_show = visualize_every_frame or frame_number == start_index
            if should_show:
                pcd_to_show = processed_pcd if visualize_processed else raw_pcd
                visualize_frame(
                    pcd_to_show,
                    detections,
                    window_name=f"Frame {frame_number}",
                )

        # Save tracking output for the frame as CSV.
        write_csv_helper(output_path / f"{frame_number}.csv", vehicles)


# ============================================================
# Command-line entry point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect and track vehicles in LiDAR .pcd frames.")
    parser.add_argument("data_path", type=Path, help="Directory containing .pcd files")
    parser.add_argument(
        "-o",
        "--output_path",
        type=Path,
        default=Path("perception_results"),
        help="Directory where CSV outputs and visualizations will be saved",
    )
    parser.add_argument(
        "-s",
        "--start_index",
        type=int,
        default=0,
        help="Index of the first frame to process",
    )
    parser.add_argument(
        "-e",
        "--end_index",
        type=int,
        default=-1,
        help="Index of the last frame to process (-1 means final frame)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show an Open3D visualization window while processing",
    )
    parser.add_argument(
        "--visualize_every_frame",
        action="store_true",
        help="Visualize every frame instead of only the first displayed frame",
    )
    parser.add_argument(
        "--visualize_raw",
        action="store_true",
        help="Visualize/save the raw point cloud instead of the processed one",
    )

    args = parser.parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)

    main(
        data_path=args.data_path,
        output_path=args.output_path,
        start_index=args.start_index,
        end_index=args.end_index,
        visualize=args.visualize,
        visualize_every_frame=args.visualize_every_frame,
        visualize_processed=not args.visualize_raw,
    )
