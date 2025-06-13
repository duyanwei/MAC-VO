#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@file vis_map.py
@author Yanwei Du (yanwei.du@gatech.edu)
@date 06-12-2025
@version 1.0
@license Copyright (c) 2025
@desc None
'''

import argparse
import numpy as np
from pathlib import Path
import open3d as o3d


def vis_map(map_dir: str):
    # Load Map
    data = dict(np.load(Path(map_dir, "pointcloud.npz"), allow_pickle=True))
    # print(data)

    # Convert to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data["xyz"])  # Convert to NumPy first
    pcd.colors = o3d.utility.Vector3dVector(data["rgb"].astype(np.float64) / 255.0)

    # downpcd = pcd.voxel_down_sample(voxel_size=0.05)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.8)
    vis.run()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dir", type=str, default=None, required=True)
    args = args.parse_args()

    if args.dir is None or not Path(args.dir).exists():
        exit(-1)

    vis_map(args.dir)