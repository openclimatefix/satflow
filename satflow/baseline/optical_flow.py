import cv2
from satflow.data.datasets import OpticalFlowDataset
import webdataset as wds
import yaml
import torch.nn.functional as F
import numpy as np


def load_config(config_file):
    with open(config_file, "r") as cfg:
        return yaml.load(cfg, Loader=yaml.FullLoader)["config"]


config = load_config(
    "/home/jacob/Development/satflow/satflow/configs/datamodule/optical_flow_datamodule.yaml"
)
dset = wds.WebDataset("/run/media/jacob/data/satflow-flow-144-tiled-{00001..00149}.tar")

dataset = OpticalFlowDataset([dset], config=config)

for data in dataset:
    prev_frame, curr_frame, next_frames = data
    prev_frame = np.moveaxis(prev_frame, [0], [2])
    curr_frame = np.moveaxis(curr_frame, [0], [2])
    print(prev_frame.shape)
    print(curr_frame.shape)
    print(next_frames.shape)
    flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    print(flow.shape)
    exit()

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
