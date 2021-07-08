import datetime
from typing import Any, Dict, Iterator, List, Optional, Sequence, Type, Union

import albumentations as A
import numpy as np
import torch.utils.data as thd
import webdataset as wds
from torch.utils.data.dataset import T_co
import logging
import pickle
import io

logger = logging.getLogger("satflow.dataset")
logger.setLevel(logging.INFO)

REGISTERED_DATASET_CLASSES = {}


def register_dataset(cls: Type[thd.IterableDataset]):
    global REGISTERED_DATASET_CLASSES
    name = cls.__name__
    assert name not in REGISTERED_DATASET_CLASSES, f"exists class: {REGISTERED_DATASET_CLASSES}"
    REGISTERED_DATASET_CLASSES[name] = cls
    return cls


def get_dataset(name: str) -> Type[thd.IterableDataset]:
    global REGISTERED_DATASET_CLASSES
    assert name in REGISTERED_DATASET_CLASSES, f"available class: {REGISTERED_DATASET_CLASSES}"
    return REGISTERED_DATASET_CLASSES[name]


def create_time_layer(dt: datetime.datetime, shape):
    """Create 3 layer for current time of observation"""
    month = dt.month / 12
    day = dt.day / 31
    hour = dt.hour / 24
    # minute = dt.minute / 60
    return np.stack([np.full(shape, month), np.full(shape, day), np.full(shape, hour)], axis=-1)


def load_np(data):
    import numpy.lib.format

    stream = io.BytesIO(data)
    return numpy.lib.format.read_array(stream)


def binarize_mask(mask):
    """Binarize mask, taking max value as the data, and setting everything else to 0"""
    mask[mask == 255] = 0  # hardcoded incase missing any of the cloud mask background problem
    mask[mask < 2] = 0  # 2 is cloud, others are clear over land (1) and clear over water (0)
    mask[mask > 0] = 1
    return mask


# Taken from training set
MSG_MEAN = {
    "HRV": 14.04300588,
    "IR016": 12.08545261,
    "IR039": 277.3749233,
    "IR087": 269.09229239,
    "IR097": 246.08281192,
    "IR108": 271.22961027,
    "IR120": 269.87252372,
    "IR134": 251.17556403,
    "VIS006": 12.23808318,
    "VIS008": 14.80151262,
    "WV062": 232.57341978,
    "WV073": 248.14469363,
}

MSG_STD = {
    "HRV": 7.6144786,
    "IR016": 6.70064364,
    "IR039": 10.4374892,
    "IR087": 13.27530427,
    "IR097": 6.9411872,
    "IR108": 14.14880209,
    "IR120": 14.15595176,
    "IR134": 8.41474376,
    "VIS006": 7.16105213,
    "VIS008": 8.04250388,
    "WV062": 4.20345723,
    "WV073": 6.93812301,
}

TOPO_MEAN = 224.3065682349895
TOPO_STD = 441.7514422990341


def create_pixel_coord_layers(x_dim: int, y_dim: int, with_r: bool = False) -> np.ndarray:
    """
    Creates Coord layer for CoordConv model

    :param x_dim: size of x dimension for output
    :param y_dim: size of y dimension for output
    :param with_r: Whether to include polar coordinates from center
    :return: (2, x_dim, y_dim) or (3, x_dim, y_dim) array of the pixel coordinates
    """
    xx_ones = np.ones([1, x_dim], dtype=np.int32)
    xx_ones = np.expand_dims(xx_ones, -1)

    xx_range = np.expand_dims(np.arange(x_dim), 0)
    xx_range = np.expand_dims(xx_range, 1)

    xx_channel = np.matmul(xx_ones, xx_range)
    xx_channel = np.expand_dims(xx_channel, -1)

    yy_ones = np.ones([1, y_dim], dtype=np.int32)
    yy_ones = np.expand_dims(yy_ones, 1)

    yy_range = np.expand_dims(np.arange(y_dim), 0)
    yy_range = np.expand_dims(yy_range, -1)

    yy_channel = np.matmul(yy_range, yy_ones)
    yy_channel = np.expand_dims(yy_channel, -1)

    xx_channel = xx_channel.astype("float32") / (x_dim - 1)
    yy_channel = yy_channel.astype("float32") / (y_dim - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1
    ret = np.stack([xx_channel, yy_channel], axis=0)

    if with_r:
        rr = np.sqrt(np.square(xx_channel - 0.5) + np.square(yy_channel - 0.5))
        ret = np.concatenate([ret, np.expand_dims(rr, axis=0)], axis=0)
    ret = np.moveaxis(ret, [1], [0])
    return ret


@register_dataset
class SatFlowDataset(thd.IterableDataset, wds.Shorthands, wds.Composable):
    def __init__(self, datasets: List[wds.WebDataset], config: dict, train: bool = True):
        super().__init__()
        self.config = config
        self.datasets = datasets
        self.train = train
        self.num_timesteps = config["num_timesteps"]
        self.skip_timesteps = config.get(
            "skip_timesteps", 1
        )  # Every nth historical timestep to take
        self.forecast_times = config.get(
            "forecast_times", 48
        )  # Max timesteps to predict ahead (minutes / 5) default 4 hours

        # Defined output sizes, etc.
        self.output_shape = config["output_shape"]
        self.output_target = config.get("output_target", config["output_shape"])
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

        self.mean = np.array([MSG_MEAN[b] for b in self.bands])
        self.std = np.array([MSG_STD[b] for b in self.bands])

        self.use_topo = config.get("use_topo", False)
        self.use_latlon = config.get("use_latlon", False)
        self.use_time = config.get("use_time", True)
        self.time_aux = config.get("time_aux", False)  # Time as an auxiliary input as 1D array
        self.use_mask = config.get("use_mask", True)
        self.use_image = config.get("use_image", False)
        self.return_target_stack = config.get("stack_targets", False)
        self.time_as_channels = config.get("time_as_channels", False)
        self.add_pixel_coords = config.get("add_pixel_coords", False)

        self.pixel_coords = create_pixel_coord_layers(
            self.output_shape, self.output_shape, with_r=config.get("add_polar_coords", False)
        )
        if self.time_as_channels:
            # Only want one copy, so don't have extra ones
            self.pixel_coords = np.squeeze(self.pixel_coords)
        else:
            self.pixel_coords = np.repeat(
                self.pixel_coords, repeats=self.num_timesteps + 1, axis=0
            )  # (timesteps, H, W, Ch)
            self.pixel_coords = self.pixel_coords.squeeze(axis=4)

        self.topo = None
        self.location = None

        self.num_crops = config.get("num_crops", 5)
        self.num_times = config.get("num_times", 10)

        self.vis = config.get("visualize", False)

        transforms = []
        if self.train and False:
            # TODO Change if we want to actually flip things
            # Pointed out that flips might mess up learning dominant winds, etc. physical phenomena, disable for now
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ]
        transforms.append(A.RandomCrop(self.output_shape, self.output_shape))
        self.aug = A.ReplayCompose(
            transforms,
        )

    def visualize(self, image, target_image, mask):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(
            len(self.bands) + 1, self.num_timesteps + 1 + self.forecast_times, figsize=(15, 15)
        )
        for t_step, img in enumerate(image):
            for channel, channel_img in enumerate(img):
                if channel >= len(self.bands):
                    break
                # Now should be 2D array
                axs[channel, t_step].imshow(channel_img)
                # axs[channel,t_step].set_title(f"{self.bands[channel]} T{'+' if t_step-self.num_timesteps >= 0 else '-'}{t_step - self.num_timesteps}")
        for t_step, img in enumerate(target_image):
            for channel, channel_img in enumerate(img):
                if channel >= len(self.bands):
                    break
                # Now should be 2D array
                axs[channel, t_step + self.num_timesteps + 1].imshow(channel_img)
                # axs[channel,t_step+self.num_timesteps+1].set_title(f"{self.bands[channel]} T{'+' if t_step+1 >= 0 else '-'}{t_step+1}")
        for t_step, img in enumerate(mask):
            # Now should be 2D array
            axs[-1, t_step + self.num_timesteps + 1].imshow(img[0])
            # axs[-1,t_step+self.num_timesteps+1].set_title(f"Mask T{'+' if t_step+1 >= 0 else '-'}{t_step+1}")
        for ax in axs.flat:
            ax.label_outer()
        plt.show()
        plt.close()

    def create_target_time_cube(self, target_timestep):
        """Create target time layer"""
        time_cube = np.zeros(
            (self.output_shape, self.output_shape, self.forecast_times), dtype=np.int8
        )
        time_cube[:, :, target_timestep] = 1
        return time_cube

    def create_target_time_layer(self, target_timestep):
        """
        Creates a one-hot encoded layer, a lot more space efficient than timecube, but needs to be an aux layer
        Args:
            target_timestep: The target timestep for predicting

        Returns:
            1-D numpy array where there is a 1 for the target timestep
        """

        time_layer = np.zeros((self.output_shape, self.forecast_times), dtype=np.int8)
        time_layer[target_timestep] = 1
        return time_layer

    def get_timestep(self, sample, idx, return_target=False, return_image=True):
        """
        Gets the image stack of the given timestep, if return_target is true, only returns teh mask and satellite channels
        as the model does not need to predict the time, topogrpahic data, etc.
        :param sample:
        :param idx:
        :param return_target:
        :return:
        """
        target = load_np(sample[f"{self.target_type}.{idx:03d}.npy"])
        if "mask" in self.target_type:
            target = binarize_mask(target)  # Not actual target, but for now, should be good

        if return_target and not return_image:
            return None, target

        image = np.stack(
            [load_np(sample[f"{b.lower()}.{idx:03d}.npy"]) for b in self.bands], axis=-1
        )

        # Regularize here
        image = (image - self.mean) / self.std

        if return_target and return_image:
            return image, target

        if self.use_topo:
            image = np.concatenate([image, self.topo], axis=-1)
        if self.use_latlon:
            image = np.concatenate([image, self.location], axis=-1)
        if self.use_time and not self.time_aux:
            t = create_time_layer(
                pickle.loads(sample["time.pyd"])[idx],
                shape=(image.shape[0], image.shape[1]),
            )
            image = np.concatenate(
                [
                    image,
                    t,
                ],
                axis=-1,
            )
        if self.use_mask:
            image = np.concatenate(
                [
                    image,
                    np.expand_dims(
                        binarize_mask(load_np(sample[f"cloudmask.{idx:03d}.npy"]).astype(np.int8)),
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
        while True:
            sources = [iter(ds) for ds in self.datasets]
            for source in sources:
                try:
                    sample = next(source)
                except StopIteration:
                    continue
                timesteps = pickle.loads(sample["time.pyd"])
                available_steps = len(timesteps)  # number of available timesteps
                # Check to make sure all timesteps exist
                sample_keys = [key for key in sample.keys() if self.bands[0].lower() in key]
                key_checker = [
                    f"{self.bands[0].lower()}.{idx:03d}.npy" for idx in range(1, available_steps)
                ]
                if (
                    not all(e in sample_keys for e in key_checker)
                    or len(sample_keys)
                    <= self.num_timesteps * self.skip_timesteps + self.forecast_times
                ):
                    continue  # Skip this sample as it is missing timesteps, or has none
                # Times that have enough previous timesteps and post timesteps for training
                # pick one at random

                idxs = np.random.randint(
                    self.num_timesteps * self.skip_timesteps + 1,
                    available_steps - self.forecast_times,
                    size=self.num_times,
                )
                if self.use_topo:
                    topo = load_np(sample["topo.npy"])
                    topo[
                        topo < 100
                    ] = 0  # Elevation shouldn't really be below 0 here (ocean mostly)
                    self.topo = (topo - TOPO_MEAN) / TOPO_STD
                    self.topo = np.expand_dims(self.topo, axis=-1)
                if self.use_latlon:
                    self.location = load_np(sample["location.npy"])
                for idx in idxs:
                    target_timesteps = np.full(self.num_crops, idx + self.forecast_times)
                    for _ in range(self.num_crops):  # Do random crops as well for training
                        for target_timestep in target_timesteps:
                            time_cube = self.create_target_time_cube(
                                target_timestep - idx - 1
                            )  # Want relative tiemstep forward
                            image, _ = self.get_timestep(
                                sample, idx - (self.num_timesteps * self.skip_timesteps)
                            )  # First timestep considered
                            data = self.aug(image=image)
                            replay = data["replay"]
                            image = data["image"]
                            if self.use_time and not self.time_aux:
                                image = np.concatenate([image, time_cube], axis=-1)
                            image = self.create_stack(idx, image, replay, sample, time_cube)
                            # Now in a Time x W x H x Channel order
                            target_image, target_mask = self.get_timestep(
                                sample,
                                target_timestep,
                                return_target=True,
                                return_image=self.use_image,
                            )
                            # Only keep is target also
                            if target_image is not None:
                                target_image = self.aug.replay(replay, image=target_image)["image"]
                                target_image = np.expand_dims(target_image, axis=0)
                            target_mask = self.aug.replay(replay, image=target_mask)["image"]
                            target_mask = np.expand_dims(target_mask, axis=0)

                            if np.isclose(np.min(target_mask), np.max(target_mask)):
                                continue  # Ignore if target timestep has no clouds, or only clouds
                            # Now create stack here
                            for i in range(idx + 1, target_timestep):
                                t_image, t_mask = self.get_timestep(
                                    sample,
                                    i,
                                    return_target=True,
                                    return_image=self.use_image,
                                )
                                t_mask = self.aug.replay(replay, image=t_mask)["image"]
                                target_mask = np.concatenate(
                                    [np.expand_dims(t_mask, axis=0), target_mask]
                                )
                                if self.use_image:
                                    t_image = self.aug.replay(replay, image=t_image)["image"]
                                    target_image = np.concatenate(
                                        [np.expand_dims(t_image, axis=0), target_image]
                                    )
                            # Ensure last target mask is also different than previous ones -> only want ones where things change
                            if np.allclose(target_mask[0], target_mask[-1]):
                                continue
                            # Convert to Time x Channel x W x H
                            # target_mask = np.expand_dims(target_mask, axis=1)
                            # One timestep as well
                            if not self.return_target_stack:
                                target_mask = np.expand_dims(target_mask, axis=0)
                            target_mask = target_mask.astype(np.float32)

                            # Convert to float/half-precision
                            image = image.astype(np.float32)
                            # Move channel to Time x Channel x W x H
                            image = np.moveaxis(image, [3], [1])
                            target_mask = np.moveaxis(target_mask, [1], [0])
                            if target_image is not None:
                                target_image = np.moveaxis(target_image, [3], [1])
                                target_image = target_image.astype(np.float32)
                            if self.time_as_channels:
                                images = image[0]
                                for m in image[1:]:
                                    images = np.concatenate([images, m], axis=0)
                                image = images
                                ts = target_mask[0]
                                for t in target_mask[1:]:
                                    ts = np.concatenate([ts, t], axis=0)
                                target_mask = ts
                            if self.output_target != self.output_shape:
                                if self.use_image:
                                    target_image = crop_center(
                                        target_image, self.output_target, self.output_target
                                    )
                                target_mask = crop_center(
                                    target_mask, self.output_target, self.output_target
                                )
                            if self.add_pixel_coords:
                                # Add channels for pixel_coords, once per channel, or once per stack, dependent
                                if self.time_as_channels:
                                    image = np.concatenate([image, self.pixel_coords], axis=0)
                                else:
                                    image = np.concatenate([image, self.pixel_coords], axis=1)
                            # Ensure there is no NaN here
                            image = np.nan_to_num(image, posinf=0.0, neginf=0.0)
                            target_mask = np.nan_to_num(target_mask, posinf=0, neginf=0)
                            if self.use_image:
                                target_image = np.nan_to_num(target_image, posinf=0.0, neginf=0.0)
                            if self.vis:
                                self.visualize(image, target_image, target_mask)
                            if self.use_time and self.time_aux:
                                time_layer = create_time_layer(
                                    target_timestep - idx - 1, self.output_shape
                                )
                                yield image, time_layer, target_image, target_mask
                            if not self.use_image:
                                yield image, target_mask
                            else:
                                yield image, target_image, target_mask

    def create_stack(self, idx, image, replay, sample, time_cube):
        image = np.expand_dims(image, axis=0)
        for i in range(
            idx - (self.num_timesteps * self.skip_timesteps) + self.skip_timesteps,
            idx + 1,
            self.skip_timesteps,
        ):
            t_image, _ = self.get_timestep(sample, i)
            t_image = self.aug.replay(replay, image=t_image)["image"]
            if self.use_time and not self.time_aux:
                t_image = np.concatenate([t_image, time_cube], axis=-1)
            image = np.concatenate([image, np.expand_dims(t_image, axis=0)])
        return image


class CloudFlowDataset(SatFlowDataset):
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
                sample_keys = [key for key in sample.keys() if self.bands[0].lower() in key]
                key_checker = [
                    f"{self.bands[0].lower()}.{idx:03d}.npy" for idx in range(1, available_steps)
                ]
                if (
                    not all(e in sample_keys for e in key_checker)
                    or len(sample_keys)
                    <= self.num_timesteps * self.skip_timesteps + self.forecast_times
                ):
                    continue  # Skip this sample as it is missing timesteps, or has none
                # Times that have enough previous timesteps and post timesteps for training
                # pick one at random

                idxs = np.random.randint(
                    self.num_timesteps * self.skip_timesteps + 1,
                    available_steps - self.forecast_times,
                    size=self.num_times,
                )
                if self.use_topo:
                    topo = load_np(sample["topo.npy"])
                    topo[
                        topo < 100
                    ] = 0  # Elevation shouldn't really be below 0 here (ocean mostly)
                    self.topo = (topo - TOPO_MEAN) / TOPO_STD
                    self.topo = np.expand_dims(self.topo, axis=-1)
                if self.use_latlon:
                    self.location = load_np(sample["location.npy"])
                for idx in idxs:
                    target_timesteps = np.full(self.num_crops, idx + self.forecast_times)
                    for _ in range(self.num_crops):  # Do random crops as well for training
                        for target_timestep in target_timesteps:
                            logger.debug(
                                f"Timestep: {target_timestep} IDX: {idx} Timecube idx: {target_timestep - idx} Future Timesteps: {self.forecast_times}"
                            )
                            time_cube = self.create_target_time_cube(target_timestep - idx - 1)
                            _, mask = self.get_timestep(
                                sample,
                                idx - (self.num_timesteps * self.skip_timesteps),
                                return_target=True,
                                return_image=False,
                            )  # First timestep considered
                            data = self.aug(image=mask)
                            replay = data["replay"]
                            mask = data["image"]
                            if self.use_time and not self.time_aux:
                                mask = np.concatenate([mask, time_cube], axis=-1)
                            mask = np.expand_dims(mask, axis=0)
                            for i in range(
                                idx
                                - (self.num_timesteps * self.skip_timesteps)
                                + self.skip_timesteps,
                                idx + 1,
                                self.skip_timesteps,
                            ):
                                _, t_mask = self.get_timestep(
                                    sample, i, return_target=True, return_image=False
                                )
                                t_mask = self.aug.replay(replay, image=t_mask)["image"]
                                if self.use_time and not self.time_aux:
                                    t_mask = np.concatenate([t_mask, time_cube], axis=-1)
                                mask = np.concatenate([mask, np.expand_dims(t_mask, axis=0)])
                            # Now in a Time x W x H x Channel order
                            _, target_mask = self.get_timestep(
                                sample,
                                target_timestep,
                                return_target=True,
                                return_image=False,
                            )
                            target_mask = self.aug.replay(replay, image=target_mask)["image"]
                            target_mask = np.expand_dims(target_mask, axis=0)

                            if np.isclose(np.min(target_mask), np.max(target_mask)):
                                continue  # Ignore if target timestep has no clouds, or only clouds
                            # Now create stack here
                            for i in range(idx + 1, target_timestep):
                                _, t_mask = self.get_timestep(
                                    sample,
                                    i,
                                    return_target=True,
                                    return_image=False,
                                )
                                t_mask = self.aug.replay(replay, image=t_mask)["image"]
                                target_mask = np.concatenate(
                                    [np.expand_dims(t_mask, axis=0), target_mask]
                                )
                            # Ensure last target mask is also different than previous ones -> only want ones where things change
                            if np.allclose(target_mask[0], target_mask[-1]):
                                continue
                            # Convert to Time x Channel x W x H
                            # target_mask = np.expand_dims(target_mask, axis=1)
                            # One timestep as well
                            if not self.return_target_stack:
                                target_mask = np.expand_dims(target_mask, axis=0)
                            target_mask = target_mask.astype(np.float32)

                            # Convert to float/half-precision
                            mask = mask.astype(np.float32)
                            # Move channel to Time x Channel x W x H
                            # mask = np.moveaxis(mask, [2], [1])
                            mask = np.expand_dims(mask, axis=1)
                            target_mask = np.moveaxis(target_mask, [1], [0])
                            if self.time_as_channels:
                                images = mask[0]
                                for m in mask[1:]:
                                    images = np.concatenate([images, m], axis=0)
                                mask = images
                                ts = target_mask[0]
                                for t in target_mask[1:]:
                                    ts = np.concatenate([ts, t], axis=0)
                                target_mask = ts
                            if self.output_target != self.output_shape:
                                target_mask = crop_center(
                                    target_mask, self.output_target, self.output_target
                                )
                            # Ensure there is no NaN here
                            mask = np.nan_to_num(mask, posinf=0.0, neginf=0.0)
                            target_mask = np.nan_to_num(target_mask, posinf=0, neginf=0)
                            if self.use_time and self.time_aux:
                                time_layer = create_time_layer(
                                    target_timestep - idx - 1, self.output_shape
                                )
                                yield mask, time_layer, target_mask
                            if self.add_pixel_coords:
                                # Add channels for pixel_coords, once per channel, or once per stack, dependent
                                if self.time_as_channels:
                                    mask = np.concatenate([mask, self.pixel_coords], axis=0)
                                else:
                                    mask = np.concatenate([mask, self.pixel_coords], axis=1)
                            logger.debug(f"Mask: {mask.shape} Target: {target_mask.shape}")
                            yield mask, target_mask


class OpticalFlowDataset(SatFlowDataset):
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
                sample_keys = [key for key in sample.keys() if self.bands[0].lower() in key]
                key_checker = [
                    f"{self.bands[0].lower()}.{idx:03d}.npy" for idx in range(1, available_steps)
                ]
                if (
                    not all(e in sample_keys for e in key_checker)
                    or len(sample_keys)
                    <= self.num_timesteps * self.skip_timesteps + self.forecast_times
                ):
                    continue  # Skip this sample as it is missing timesteps, or has none
                idxs = list(range(20, available_steps - 1))
                for idx in idxs:
                    for _ in range(self.num_crops):  # Do random crops as well for training
                        logger.debug(f"IDX: {idx}")
                        image, mask = self.get_timestep(
                            sample,
                            idx,
                            return_target=True,
                            return_image=True,
                        )  # First timestep considered
                        data = self.aug(image=mask)
                        replay = data["replay"]
                        mask = data["image"]
                        image = self.aug.replay(replay, image=image)["image"]
                        prev_image, prev_mask = self.get_timestep(
                            sample,
                            idx - 1,
                            return_target=True,
                            return_image=True,
                        )  # First timestep considered
                        prev_mask = self.aug.replay(replay, image=prev_mask)["image"]
                        prev_image = self.aug.replay(replay, image=prev_image)["image"]
                        # Now in a Time x W x H x Channel order
                        _, target_mask = self.get_timestep(
                            sample,
                            idx + self.forecast_times,
                            return_target=True,
                            return_image=False,
                        )
                        target_mask = self.aug.replay(replay, image=target_mask)["image"]
                        target_mask = np.expand_dims(target_mask, axis=0)

                        if np.isclose(np.min(target_mask), np.max(target_mask)):
                            continue  # Ignore if target timestep has no clouds, or only clouds
                        # Now create stack here
                        for i in range(idx + 1, idx + self.forecast_times):
                            _, t_mask = self.get_timestep(
                                sample,
                                i,
                                return_target=True,
                                return_image=False,
                            )
                            t_mask = self.aug.replay(replay, image=t_mask)["image"]
                            target_mask = np.concatenate(
                                [np.expand_dims(t_mask, axis=0), target_mask]
                            )
                        target_mask = np.round(target_mask).astype(np.int8)
                        # Convert to float/half-precision
                        mask = np.round(mask).astype(np.int8)
                        prev_mask = np.round(prev_mask).astype(np.int8)
                        # Move channel to Time x Channel x W x H
                        mask = np.expand_dims(mask, axis=0)
                        prev_mask = np.expand_dims(prev_mask, axis=0)
                        mask = np.nan_to_num(mask, posinf=0.0, neginf=0.0)
                        prev_mask = np.nan_to_num(prev_mask, posinf=0.0, neginf=0.0)
                        target_mask = np.nan_to_num(target_mask, posinf=0, neginf=0)
                        logger.debug(f"Mask: {mask.shape} Target: {target_mask.shape}")
                        yield prev_mask, mask, target_mask, image, prev_image


def crop_center(img, cropx, cropy):
    """Crops center of image through timestack, fails if all the images are concatenated as channels"""
    t, c, y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:, :, starty : starty + cropy, startx : startx + cropx]
