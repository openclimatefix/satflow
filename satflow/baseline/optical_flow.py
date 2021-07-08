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

import matplotlib.pyplot as plt


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


for data in dataset:
    prev_frame, curr_frame, next_frames, image, prev_image = data
    prev_frame = np.moveaxis(prev_frame, [0], [2])
    curr_frame = np.moveaxis(curr_frame, [0], [2])
    flow = cv2.calcOpticalFlowFarneback(prev_image, image, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    plt.imshow(prev_frame)
    plt.imshow(image)
    plt.title("Image")
    plt.show()
    plt.imshow(prev_image)
    plt.title("Prev Image")
    plt.show()
    plt.imshow(image - prev_image)
    plt.title("Image Diff")
    plt.show()
    plt.imshow(curr_frame - prev_frame)
    plt.title("Difference")
    plt.show()
    warped_frame = warp_flow(curr_frame.astype(np.float32), flow)
    warped_frame = np.expand_dims(warped_frame, axis=-1)
    plt.imshow(curr_frame - warped_frame)
    plt.title("Curr - Warped After 1")
    plt.show()
    diff = curr_frame - warped_frame
    for i in range(47):
        warped_frame = warp_flow(warped_frame.astype(np.float32), flow)
    warped_frame = np.expand_dims(warped_frame, axis=-1)
    plt.imshow(curr_frame - warped_frame)
    plt.title("Curr - Warped after 48")
    plt.show()
    diff2 = curr_frame - warped_frame
    plt.imshow(diff - diff2)
    plt.title("Warped 1 minus Warped 2")
    plt.show()
    plt.imshow(
        prev_frame - np.expand_dims(warp_flow(prev_frame.astype(np.float32), flow), axis=-1)
    )
    plt.title("Prev - Warped")
    plt.show()
    exit()
