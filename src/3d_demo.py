import open3d as o3d
import numpy as np
import os
import argparse

def Read_Point_Cloud(path_to_cloud):
    pcd = o3d.io.read_point_cloud(path_to_cloud)
    print("{} points loaded from {}".format(np.asarray(pcd.points).shape[0], path_to_cloud))
    return pcd

def main():
    # Read one point clouds from the arguments
    parser = argparse.ArgumentParser(description="3D Point Cloud Viewer")
    parser.add_argument("pcd1", type=str, help="Path to the first point cloud")
    args = parser.parse_args()

    # Reading point clouds from disk
    pcd1 = Read_Point_Cloud(args.pcd1)

    # Print some statistics
    print("Point Cloud 1: {} points".format(np.asarray(pcd1.points).shape[0]))
    print("Point Cloud 1 - Min Bound: {}, Max Bound: {}".format(pcd1.get_min_bound(), pcd1.get_max_bound()))

    # Visualization
    o3d.visualization.draw_geometries([pcd1])

if __name__ == "__main__":
    main()