"""
    Script for launching training process
"""
import os
import sys
import numpy as np
from easydict import EasyDict
from tqdm import tqdm
from fire import Fire
import coloredlogs
import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from _path_init import *
from visualDet3D.networks.utils.registry import DETECTOR_DICT, DATASET_DICT, PIPELINE_DICT
from visualDet3D.networks.utils.utils import BackProjection, BBox3dProjector, get_num_parameters
from visualDet3D.evaluator.kitti.evaluate import evaluate
import visualDet3D.data.kitti.dataset
from visualDet3D.utils.timer import Timer
from visualDet3D.utils.utils import LossLogger, cfg_from_file
from visualDet3D.networks.optimizers import optimizers, schedulers

def main(config="../config/config.py", experiment_name="default", world_size=1, local_rank=-1):
    """Main function for the training script.

    KeywordArgs:
        config (str): Path to config file.
        experiment_name (str): Custom name for the experitment, only used in tensorboard.
        world_size (int): Number of total subprocesses in distributed training. 
        local_rank: Rank of the process. Should not be manually assigned. 0-N for ranks in distributed training (only process 0 will print info and perform testing). -1 for single training. 
    """

    ## Get config
    cfg = cfg_from_file(config)

    ## Collect distributed(or not) information
    cfg.dist = EasyDict()
    cfg.dist.world_size = world_size
    cfg.dist.local_rank = local_rank
    is_distributed = local_rank >= 0 # local_rank < 0 -> single training
    is_logging     = local_rank <= 0 # only log and test with main process
    is_evaluating  = local_rank <= 0

    ## Setup writer if local_rank > 0
    dataset_val = DATASET_DICT[cfg.data.val_dataset](cfg)
    # print(dataset_val[0])
    res = dataset_val[0]
    print(f"res['calib'] = {res['calib']}")
    # print(f"res['image'].shape = {res['image'].shape}")
    # print(f"res['label'] = {res['label']}")
    print(f"len(res['label']) = {len(res['label'])}")
    # print(f"res['bbox2d'] = {res['bbox2d']}")
    print(f"len(res['bbox2d']) = {len(res['bbox2d'])}")

    # print(f"res['bbox3d'] = {res['bbox3d']}")
    print(f"len(res['bbox3d']) = {len(res['bbox3d'])}")

    # print(f"res['depth'] = {res['depth']}")
    print(f"len(res['depth']) = {len(res['depth'])}")

    print(f"res['original_P'] = {res['original_P']}")
    print(f"len(res['original_P']) = {len(res['original_P'])}")

if __name__ == '__main__':
    main()
