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
import torch


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


debug = False
total_losses = np.array([0.0 for _ in range(48)])  # Want to break down loss by future timestep
count = 0
baseline_losses = np.array([0.0 for _ in range(48)])  # Want to break down loss by future timestep
overall_loss = 0.0
overall_baseline = 0.0

for data in dataset:
    tmp_loss = 0
    tmp_base = 0
    count += 1
    prev_frame, curr_frame, next_frames, image, prev_image = data
    prev_frame = np.moveaxis(prev_frame, [0], [2])
    curr_frame = np.moveaxis(curr_frame, [0], [2])
    flow = cv2.calcOpticalFlowFarneback(prev_image, image, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    warped_frame = warp_flow(curr_frame.astype(np.float32), flow)
    warped_frame = np.expand_dims(warped_frame, axis=-1)
    loss = F.mse_loss(
        torch.from_numpy(warped_frame), torch.from_numpy(np.expand_dims(next_frames[0], axis=-1))
    )
    total_losses[0] += loss.item()
    tmp_loss += loss.item()
    loss = F.mse_loss(
        torch.from_numpy(curr_frame.astype(np.float32)),
        torch.from_numpy(np.expand_dims(next_frames[0], axis=-1)),
    )
    baseline_losses[0] += loss.item()
    tmp_base += loss.item()

    for i in range(1, 48):
        warped_frame = warp_flow(warped_frame.astype(np.float32), flow)
        warped_frame = np.expand_dims(warped_frame, axis=-1)
        loss = F.mse_loss(
            torch.from_numpy(warped_frame),
            torch.from_numpy(np.expand_dims(next_frames[i], axis=-1)),
        )
        total_losses[i] += loss.item()
        tmp_loss += loss.item()
        loss = F.mse_loss(
            torch.from_numpy(curr_frame.astype(np.float32)),
            torch.from_numpy(np.expand_dims(next_frames[i], axis=-1)),
        )
        baseline_losses[i] += loss.item()
        tmp_base += loss.item()
    tmp_base /= 48
    tmp_loss /= 48
    overall_loss += tmp_loss
    overall_baseline += tmp_base
    print(
        f"Avg Total Loss: {np.mean(total_losses) / count} Avg Baseline Loss: {np.mean(baseline_losses) / count} \n Overall Loss: {overall_loss / count} Baseline: {overall_baseline / count}"
    )
np.save("optical_flow_mse_loss.npy", total_losses / count)
np.save("baseline_current_image_mse_loss.npy", baseline_losses / count)
