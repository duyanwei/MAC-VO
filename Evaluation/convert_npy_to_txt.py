#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@file convert_npy_to_txt.py
@author Yanwei Du (yanwei.du@gatech.edu)
@date 06-06-2025
@version 1.0
@license Copyright (c) 2025
@desc None
'''

import argparse
import os
import glob
import numpy as np


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--dir", type=str, default=None)
    args = args.parse_args()
    
    if args.dir is None:
        return
    
    assert(os.path.exists(args.dir))
    spaces = glob.glob(os.path.join(args.dir, "**/config.yaml"), recursive=True)
    for space in spaces:
        dirname = os.path.dirname(space)
        print(dirname)
        for filename in ["poses", "ref_poses"]:
            filepath = os.path.join(dirname, filename + ".npy")
            assert(os.path.exists(filepath))
            poses = np.load(filepath)
            poses[:, 0] /= 1e9  # sec to ms
            np.savetxt(os.path.join(dirname, filename + ".txt"), poses, fmt="%.6f", header="stamp tx ty tz qx qy qz qw")
    print("Done")

if __name__ == "__main__":
    main()