import cv2
import numpy as np
import torch.nn.functional as F
import webdataset as wds
import yaml

from satflow.data.datasets import OpticalFlowDataset, SatFlowDataset


def load_config(config_file):
    with open(config_file, "r") as cfg:
        return yaml.load(cfg, Loader=yaml.FullLoader)["config"]


config = load_config("/satflow/configs/datamodule/optical_flow.yaml")
dset = wds.WebDataset("/run/media/jacob/data/satflow-flow-144-tiled-{00001..00149}.tar")

dataset = SatFlowDataset([dset], config=config)

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
channel_total_losses = np.array([total_losses for _ in range(12)])
count = 0
baseline_losses = np.array([0.0 for _ in range(48)])  # Want to break down loss by future timestep
channel_baseline_losses = np.array([baseline_losses for _ in range(12)])

for data in dataset:
    tmp_loss = 0
    tmp_base = 0
    count += 1
    past_frames, next_frames = data
    prev_frame = past_frames[1]
    curr_frame = past_frames[0]
    # Do it for each of the 12 channels
    for ch in range(12):
        # prev_frame = np.moveaxis(prev_frame, [0], [2])
        # curr_frame = np.moveaxis(curr_frame, [0], [2])
        flow = cv2.calcOpticalFlowFarneback(
            past_frames[1][ch], past_frames[0][ch], None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        warped_frame = warp_flow(curr_frame[ch].astype(np.float32), flow)
        warped_frame = np.expand_dims(warped_frame, axis=-1)
        loss = F.mse_loss(
            torch.from_numpy(warped_frame),
            torch.from_numpy(np.expand_dims(next_frames[0][ch], axis=-1)),
        )
        channel_total_losses[ch][0] += loss.item()
        loss = F.mse_loss(
            torch.from_numpy(curr_frame[ch].astype(np.float32)),
            torch.from_numpy(next_frames[0][ch]),
        )
        channel_baseline_losses[ch][0] += loss.item()

        for i in range(1, 48):
            warped_frame = warp_flow(warped_frame.astype(np.float32), flow)
            warped_frame = np.expand_dims(warped_frame, axis=-1)
            loss = F.mse_loss(
                torch.from_numpy(warped_frame),
                torch.from_numpy(np.expand_dims(next_frames[i][ch], axis=-1)),
            )
            channel_total_losses[ch][i] += loss.item()
            tmp_loss += loss.item()
            loss = F.mse_loss(
                torch.from_numpy(curr_frame[ch].astype(np.float32)),
                torch.from_numpy(next_frames[i][ch]),
            )
            channel_baseline_losses[ch][i] += loss.item()
    print(
        f"Avg Total Loss: {np.mean(channel_total_losses) / count} Avg Baseline Loss: {np.mean(channel_baseline_losses) / count}"
    )
    if count % 100 == 0:
        np.save("optical_flow_mse_loss_channels_reverse.npy", channel_total_losses / count)
        np.save(
            "baseline_current_image_mse_loss_channels_reverse.npy", channel_baseline_losses / count
        )
np.save("optical_flow_mse_loss_reverse.npy", channel_total_losses / count)
np.save("baseline_current_image_mse_loss_reverse.npy", channel_baseline_losses / count)
