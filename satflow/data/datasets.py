from typing import Iterator, Dict, List, Any, Union, Optional

import datetime
import torch
import torch.utils.data as thd
from torch.utils.data.dataset import T_co

import numpy as np
import webdataset as wds

import albumentations as A
import pickle
import numpy.lib.format
import io

from typing import Dict, Any, Type, List, Sequence

REGISTERED_DATASET_CLASSES = {}


def register_dataset(cls: Type[thd.IterableDataset]):
    global REGISTERED_DATASET_CLASSES
    name = cls.__name__
    assert (
        name not in REGISTERED_DATASET_CLASSES
    ), f"exists class: {REGISTERED_DATASET_CLASSES}"
    REGISTERED_DATASET_CLASSES[name] = cls
    return cls


def get_dataset(name: str) -> Type[thd.IterableDataset]:
    global REGISTERED_DATASET_CLASSES
    assert (
        name in REGISTERED_DATASET_CLASSES
    ), f"available class: {REGISTERED_DATASET_CLASSES}"
    return REGISTERED_DATASET_CLASSES[name]


def create_time_layer(dt: datetime.datetime, shape):
    """Create 3 layer for current time of observation"""
    month = dt.month / 12
    day = dt.day / 31
    hour = dt.hour / 24
    # minute = dt.minute / 60
    return np.stack(
        [np.full(shape, month), np.full(shape, day), np.full(shape, hour)], axis=-1
    )


def binarize_mask(mask):
    """Binarize mask, taking max value as the data, and setting everything else to 0"""
    mask[
        mask == 255
    ] = 0  # hardcoded incase missing any of the cloud mask background problem
    mask[mask < np.max(mask)] = 0
    mask[mask > 0] = 1
    return mask


# Taken from OCF Zarr file min and max for all the channels
MSG_MIN = np.array(
    [
        -1.2278595,
        -2.5118103,
        -64.83977,
        63.404694,
        2.844452,
        199.10002,
        -17.254883,
        -26.29155,
        -1.1009827,
        -2.4184198,
        199.57048,
        198.95093,
    ]
)
MSG_MAX = np.array(
    [
        103.90016,
        69.60857,
        339.15588,
        340.26526,
        317.86752,
        313.2767,
        315.99194,
        274.82297,
        93.786545,
        101.34922,
        249.91806,
        286.96323,
    ]
)


def load_np(data):
    return numpy.lib.format.read_array(io.BytesIO(data))


@register_dataset
class SatFlowDataset(thd.IterableDataset, wds.Shorthands, wds.Composable):
    def __init__(self, datasets, config, train=True):
        super().__init__()
        self.config = config
        self.datasets = datasets
        self.train = train
        self.num_timesteps = config["num_timesteps"]
        self.forecast_times = config.get(
            "forecast_times", 48
        )  # Max timesteps to predict ahead (minutes / 5) default 4 hours

        # Defined output sizes, etc.
        self.output_shape = config["output_shape"]
        self.target_type = config.get("target", "cloudmask")
        # Should load the common data here
        self.bands = config.get(
            "bands",
            (
                "HRV",
                "IR016",
                "IR039",
                "IR087",
                "IR097",
                "IR108",
                "IR120",
                "IR134",
                "VIS006",
                "VIS008",
                "WV062",
                "WV073",
            ),
        )
        self.use_topo = config.get("use_topo", False)
        self.use_latlon = config.get("use_latlon", False)
        self.use_time = config.get("use_time", True)
        self.use_mask = config.get("use_mask", True)

        self.topo = None
        self.location = None

        self.num_crops = config.get("num_crops", 5)

        transforms = []
        if self.train:
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ]
        transforms.append(A.RandomCrop(self.output_shape, self.output_shape))
        self.aug = A.ReplayCompose(transforms,)

    def create_target_time_layer(self, target_timestep):
        """Create target time layer"""
        time_cube = np.zeros(
            (self.output_shape, self.output_shape, self.forecast_times), dtype=np.int8
        )
        time_cube[:, :, target_timestep] = 1
        return time_cube

    def get_timestep(self, sample, idx, return_target=False):
        """
        Gets the image stack of the given timestep, if return_target is true, only returns teh mask and satellite channels
        as the model does not need to predict the time, topogrpahic data, etc.
        :param sample:
        :param idx:
        :param return_target:
        :return:
        """
        image = np.stack(
            [load_np(sample[f"{b.lower()}.{idx:03d}.npy"]) for b in self.bands], axis=-1
        )

        # Regularize here
        image = (image - MSG_MIN) / (MSG_MAX - MSG_MIN)

        target = load_np(sample[f"{self.target_type}.{idx:03d}.npy"])
        if "mask" in self.target_type:
            target = binarize_mask(
                target
            )  # Not actual target, but for now, should be good

        if return_target:
            return image, target

        if self.use_topo:
            image = np.concatenate([image, self.topo], axis=-1)
        if self.use_latlon:
            image = np.concatenate([image, self.location], axis=-1)
        if self.use_time:
            t = create_time_layer(
                pickle.loads(sample["time.pyd"])[idx],
                shape=(image.shape[0], image.shape[1]),
            )
            image = np.concatenate([image, t,], axis=-1,)
        if self.use_mask:
            image = np.concatenate(
                [
                    image,
                    np.expand_dims(
                        binarize_mask(load_np(sample[f"cloudmask.{idx:03d}.npy"])),
                        axis=-1,
                    ),
                ],
                axis=-1,
            )

        return image, target

    def __iter__(self) -> Iterator[T_co]:
        # Need to make sure same time step for all of them.
        # As its all from rapid scan, should be fairly easy.
        # Main missing one is the regional and rapid weather ones, which are every 15 minutes,
        # but could be interpolated between the previous step and next one by weighting by time difference
        # Topographic is same of course, just need to resize to 1km x 1km?
        # grid by taking the mean value of the interior ones
        sources = [iter(ds) for ds in self.datasets]
        while True:
            for source in sources:
                sample = next(source)
                timesteps = pickle.loads(sample["time.pyd"])
                available_steps = len(timesteps)  # number of available timesteps
                # Check to make sure all timesteps exist
                sample_keys = [
                    key for key in sample.keys() if self.bands[0].lower() in key
                ]
                key_checker = [
                    f"{self.bands[0].lower()}.{idx:03d}.npy"
                    for idx in range(1, available_steps)
                ]
                if not all(e in sample_keys for e in key_checker):
                    continue  # Skip this sample as it is missing timesteps
                # Times that have enough previous timesteps and post timesteps for training
                # pick one at random
                # To reduce having to load as much data again and again, take 10% of available timesteps to train on with different future time periods

                idxs = np.random.randint(
                    self.num_timesteps,
                    available_steps - self.forecast_times,
                    size=self.num_crops,
                )
                if self.use_topo:
                    topo = load_np(sample["topo.npy"])
                    self.topo = topo - np.min(topo) / (np.max(topo) - np.min(topo))
                    self.topo = np.expand_dims(self.topo, axis=-1)
                if self.use_latlon:
                    self.location = load_np(sample["location.npy"])
                for idx in idxs:
                    target_timesteps = (
                        np.random.randint(
                            idx + 1, idx + self.forecast_times, size=self.num_crops
                        )
                        - idx
                    )
                    for target_timestep in target_timesteps:
                        time_cube = self.create_target_time_layer(target_timestep)
                        for _ in range(
                            self.num_crops
                        ):  # Do 5 random crops as well for training
                            image, _ = self.get_timestep(
                                sample, idx - self.num_timesteps
                            )  # First timestep considered
                            data = self.aug(image=image)
                            replay = data["replay"]
                            image = data["image"]
                            image = np.concatenate([image, time_cube], axis=-1)
                            image = np.expand_dims(image, axis=0)
                            for i in range(idx - self.num_timesteps + 1, idx + 1):
                                t_image, _ = self.get_timestep(sample, i)
                                t_image = self.aug.replay(replay, image=t_image)[
                                    "image"
                                ]
                                t_image = np.concatenate([t_image, time_cube], axis=-1)
                                image = np.concatenate(
                                    [image, np.expand_dims(t_image, axis=0)]
                                )
                            # Now in a Time x W x H x Channel order
                            target_image, target_mask = self.get_timestep(
                                sample, target_timestep, return_target=True
                            )
                            target_image = self.aug.replay(replay, image=target_image)[
                                "image"
                            ]
                            target_mask = self.aug.replay(replay, image=target_mask)[
                                "image"
                            ]
                            # Convert to Channel x Time x W x H
                            image = np.moveaxis(image, [3], [1])
                            target_image = np.moveaxis(target_image, [2], [0])
                            target_mask = np.expand_dims(target_mask, axis=0)

                            # return as a PyTorch thing
                            yield image, target_image, target_mask
