#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@file TSRB.py
@author Yanwei Du (yanwei.du@gatech.edu)
@date 06-12-2025
@version 1.0
@license Copyright (c) 2025
@desc None
'''


from __future__ import annotations
import cv2
import torch
import numpy as np
import pypose as pp

from types import SimpleNamespace
from typing import cast, Any
from pathlib import Path
from torch.utils.data import Dataset

from Utility.PrettyPrint import Logger
from Utility.Config import load_config
from Utility.Math import qinterp, interpolate_pose

from ..Interface import StereoData, IMUData, StereoFrame, StereoInertialFrame, AttitudeData
from ..SequenceBase import SequenceBase


class TSRB_StereoSequence(SequenceBase[StereoFrame]):
    @classmethod
    def name(cls) -> str: return "TSRB_Stereo"

    def __init__(self, config: SimpleNamespace | dict[str, Any]) -> None:
        cfg = self.config_dict2ns(config)
        self.root = Path(cfg.root)
        self.sequence_name = self.root.name

        self.imageL = TSRBMonocularDataset(Path(self.root, "cam0"))
        self.imageR = TSRBMonocularDataset(Path(self.root, "cam1"))
        assert len(self.imageL) == len(self.imageR)
        
        self.baseline = 0.05  # m
        self.width = 640
        self.height = 480
        K_np = np.array([
            [378.7210693359375, 0., 324.4961853027344],
            [0., 378.7210693359375, 239.1063690185547],
            [0., 0., 1.],
        ])
        self.K = torch.tensor(K_np).float().unsqueeze(0)
       
        # Setup metadata
        self.T_BS = pp.identity_SE3(1)
        # T_BS_ext      = np.eye(4)[np.newaxis, ...]
        # T_BS_ext[0, :3, :3] = 
        # T_BS_ext[0, :3,  3] = self.cam2_t[..., 0]
        # self.T_BS = pp.from_matrix(T_BS_ext, pp.SE3_type).float() @ NED2EDN.unsqueeze(0)
        # self.T_BS_lcam = pp.from_matrix(
        #     torch.tensor(T_BS_lcam, dtype=torch.float32).unsqueeze(0), pp.SE3_type
        # ) @ NED2EDN.unsqueeze(0)
        
        # Load ground truth pose
        self.gt_pose_data = None
        
        super().__init__(len(self.imageL))

    def __getitem__(self, local_index: int) -> StereoFrame:
        index = self.get_index(local_index)

        return StereoFrame(
            idx=[local_index],
            time_ns=[self.imageL.cam_timestamps[index]], 
            stereo=StereoData(
                T_BS=self.T_BS,
                K   =self.K,
                baseline=torch.tensor([self.baseline]),
                width=self.width,
                height=self.height,
                time_ns=[self.imageL.cam_timestamps[index]],
                imageL=self.imageL[index],
                imageR=self.imageR[index],
            ),
            gt_pose= None if self.gt_pose_data is None else cast(pp.LieTensor, self.gt_pose_data[index].unsqueeze(0))
        )
   
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "root"   : lambda v: isinstance(v, str),
            "gt_pose": lambda b: isinstance(b, bool)
        })


class TSRBMonocularDataset(Dataset):
    def __init__(self, image_path: Path) -> None:
        super().__init__()
        self.file_names = sorted([file for file in image_path.glob("*.png")])
        self.length = len(self.file_names)
        self.cam_timestamps = (np.loadtxt(
            Path(image_path, "..", "times.txt"), delimiter=" ", dtype=np.float64
        ) * 1_000_000_000).astype(np.int64)
    
    def __len__(self) -> int: return self.length
    
    def __getitem__(self, index) -> torch.Tensor:
        image = cv2.imread(str(self.file_names[index]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
        image_tensor /= 255.
        return image_tensor