import datetime
import io
import pickle
from typing import Any, Dict, Iterator, List, Optional, Sequence, Type, Union

import albumentations as A
import braceexpand
import numpy as np
import numpy.lib.format
import torch
import torch.utils.data as thd
import webdataset as wds
from torch.utils.data.dataset import T_co

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


def binarize_mask(mask):
    """Binarize mask, taking max value as the data, and setting everything else to 0"""
    mask[mask == 255] = 0  # hardcoded incase missing any of the cloud mask background problem
    mask[mask < 2] = 0  # 2 is cloud, others are clear over land (1) and clear over water (0)
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
        self.skip_timesteps = config.get(
            "skip_timesteps", 1
        )  # Every nth historical timestep to take
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
        self.time_aux = config.get("time_aux", False)  # Time as an auxiliary input as 1D array
        self.use_mask = config.get("use_mask", True)
        self.use_image = config.get("use_image", False)
        self.return_target_stack = config.get("stack_targets", False)
        self.time_as_chennels = config.get("time_as_channels", False)

        self.topo = None
        self.location = None

        self.num_crops = config.get("num_crops", 5)

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
        image = (image - MSG_MIN) / (MSG_MAX - MSG_MIN)

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
                    size=self.num_crops,
                )
                if self.use_topo:
                    topo = load_np(sample["topo.npy"])
                    topo[topo < 0] = 0  # Elevation shouldn't really be below 0 here (ocean mostly)
                    self.topo = topo - np.min(topo) / (np.max(topo) - np.min(topo))
                    self.topo = np.expand_dims(self.topo, axis=-1)
                if self.use_latlon:
                    self.location = load_np(sample["location.npy"])
                for idx in idxs:
                    target_timesteps = (
                        np.arange(start=idx + 1, stop=idx + self.forecast_times) - idx
                    )
                    if not self.return_target_stack:
                        target_timesteps = (
                            np.random.randint(
                                idx + 1, idx + self.forecast_times, size=self.num_crops
                            )
                            - idx
                        )
                    else:
                        # Same for all the crops TODO Change this to work for all setups/split to differnt datasets
                        target_timesteps = np.full(self.num_crops, self.forecast_times)
                    for _ in range(self.num_crops):  # Do random crops as well for training
                        for target_timestep in target_timesteps:
                            time_cube = self.create_target_time_cube(target_timestep)
                            image, _ = self.get_timestep(
                                sample, idx - self.num_timesteps
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
                            if target_image is not None:
                                target_image = self.aug.replay(replay, image=target_image)["image"]
                                target_image = np.expand_dims(target_image, axis=0)
                            target_mask = self.aug.replay(replay, image=target_mask)["image"]
                            target_mask = np.expand_dims(target_mask, axis=0)
                            # Now create stack here
                            for i in range(idx + 1, idx + self.forecast_times):
                                t_image, t_mask = self.get_timestep(
                                    sample,
                                    target_timestep,
                                    return_target=True,
                                    return_image=self.use_image,
                                )
                                t_mask = self.aug.replay(replay, image=t_mask)["image"]
                                target_mask = np.concatenate(
                                    [target_mask, np.expand_dims(t_mask, axis=0)]
                                )
                                if self.use_image:
                                    t_image = self.aug.replay(replay, image=t_image)["image"]
                                    target_image = np.concatenate(
                                        [target_image, np.expand_dims(t_image, axis=0)]
                                    )
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
                            if self.time_as_chennels:
                                images = image[0]
                                for m in image[1:]:
                                    images = np.concatenate([images, m], axis=0)
                                image = images
                                ts = target_mask[0]
                                for t in target_mask[1:]:
                                    ts = np.concatenate([ts, t], axis=0)
                                target_mask = ts
                            if self.vis:
                                self.visualize(image, target_image, target_mask)
                            if self.use_time and self.time_aux:
                                time_layer = create_time_layer(target_timestep, self.output_shape)
                                yield image, time_layer, target_image, target_mask
                            if not self.use_image:
                                yield image, target_mask
                            else:
                                yield image, target_image, target_mask

    def create_stack(self, idx, image, replay, sample, time_cube):
        image = np.expand_dims(image, axis=0)
        for i in range(
            idx - (self.num_timesteps * self.skip_timesteps) + 1,
            idx + 1,
            self.skip_timesteps,
        ):
            t_image, _ = self.get_timestep(sample, i)
            t_image = self.aug.replay(replay, image=t_image)["image"]
            if self.use_time and not self.time_aux:
                t_image = np.concatenate([t_image, time_cube], axis=-1)
            image = np.concatenate([image, np.expand_dims(t_image, axis=0)])
        return image


@register_dataset
class SatFlowIterDataset(SatFlowDataset):
    """
    SatFlow dataset that works with WebDataset format where each tar file is 1 day, and each frame is a sample, iterated
    in order. So after getting a sample, iterate through it all, building up the samples from earliest to latest
    """

    def __init__(self, datasets, config, train=True):
        urls = list(braceexpand.braceexpand(datasets))
        datasets = [
            wds.WebDataset(src) for src in urls
        ]  # This is to ensure that each day is kept together
        super(SatFlowIterDataset, self).__init__(datasets, config, train)
        self.topo = np.load("../resources/cutdown_europe_dem.npy")
        self.topo = self.topo - np.min(self.topo) / (np.max(self.topo) - np.min(self.topo))
        self.topo = np.expand_dims(self.topo, axis=-1)
        self.location = np.load("../resources/location_array.npy")
        # Need to create topo and location arrays here fpr use later

    def get_timestep(self, sample, idx, return_target=False):
        """
        Gets the image stack of the given timestep, if return_target is true, only returns teh mask and satellite channels
        as the model does not need to predict the time, topogrpahic data, etc.
        :param sample:
        :param idx:
        :param return_target:
        :return:
        """
        image = np.stack([load_np(sample[f"{b.lower()}.npy"]) for b in self.bands], axis=-1)

        # Regularize here
        image = (image - MSG_MIN) / (MSG_MAX - MSG_MIN)

        target = load_np(sample[f"{self.target_type}.npy"])
        if "mask" in self.target_type:
            target = binarize_mask(target)  # Not actual target, but for now, should be good

        if return_target:
            return image, target

        if self.use_topo:
            image = np.concatenate([image, self.topo], axis=-1)
        if self.use_latlon:
            image = np.concatenate([image, self.location], axis=-1)
        if self.use_time and not self.time_aux:
            t = create_time_layer(
                pickle.loads(sample["time.pyd"]),
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
                        binarize_mask(load_np(sample[f"cloudmask.npy"])),
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
            np.random.shuffle(sources)  # Shuffle order of the days
            for source in sources:
                # Check to make sure all timesteps exist
                # Times that have enough previous timesteps and post timesteps for training
                # pick one at random
                # TODO Get better way of doing this rather than hardcoding 144, but for now, lets do it
                idxs = np.random.randint(
                    self.num_timesteps * self.skip_timesteps,
                    144 - self.forecast_times,
                    size=1,  # TODO Change back to num crops when better
                )
                # Need to get lowest to largest to keep iteration going
                idxs = np.sort(idxs)
                for idx in idxs:
                    # Get current and prev images
                    target_timesteps = (
                        np.random.randint(idx + 1, idx + self.forecast_times, size=self.num_crops)
                        - idx
                    )
                    target_timesteps = np.sort(target_timesteps)
                    s_timesteps = target_timesteps + idx
                    first_sample_idx = idx - (self.num_timesteps * self.skip_timesteps)
                    future_targets = []
                    future_images = []
                    curr_images = []
                    for sample_idx, sample in source:  # Go through one day
                        if sample_idx == first_sample_idx:  # First one
                            image, _ = self.get_timestep(
                                sample, idx - self.num_timesteps
                            )  # First timestep considered
                            image = np.expand_dims(image, axis=0)
                            curr_images.append(image)
                        elif sample_idx in np.arange(
                            idx - ((self.num_timesteps - 1) * self.skip_timesteps),
                            idx + self.skip_timesteps,
                            self.skip_timesteps,
                        ):
                            t_image, _ = self.get_timestep(sample, idx)
                            curr_images.append(t_image)
                        elif sample_idx in s_timesteps:  # In the future ones
                            target_image, target_mask = self.get_timestep(
                                sample, sample_idx, return_target=True
                            )
                            future_images.append(target_image)
                            future_targets.append(target_mask)
                    # Now do the post/pre processing
                    # First check if enough samples exist
                    if len(curr_images) != (self.num_timesteps + 1) or len(future_images) != len(
                        target_timesteps
                    ):
                        print(
                            f"Curr Wanted: {(self.num_timesteps + 1)} Curr Got: {len(curr_images)} "
                            f"Future Wanted: {len(target_timesteps)} Future Got: {len(future_images)}"
                        )
                        continue

                    # Now do the cropping, but keep the original, adding the timecube, etc.
                    for i, target_timestep in enumerate(s_timesteps):
                        time_cube = self.create_target_time_cube(target_timestep)
                        for _ in self.num_crops:
                            data = self.aug(image=curr_images[0])
                            replay = data["replay"]
                            image = data["image"]
                            if self.use_time and not self.time_aux:
                                image = np.concatenate([image, time_cube], axis=-1)
                            image = np.expand_dims(image, axis=0)
                            for t_image in curr_images[1:]:
                                t_image = self.aug.replay(replay, image=t_image)["image"]
                                if self.use_time and not self.time_aux:
                                    t_image = np.concatenate([t_image, time_cube], axis=-1)
                                image = np.concatenate([image, np.expand_dims(t_image, axis=0)])
                            # Now in a Time x W x H x Channel order
                            target_image = self.aug.replay(replay, image=future_images[i])["image"]
                            target_mask = self.aug.replay(replay, image=future_targets[i])["image"]
                            # Convert to Channel x Time x W x H
                            image = np.moveaxis(image, [3], [1])
                            target_image = np.moveaxis(target_image, [2], [0])
                            target_mask = np.expand_dims(target_mask, axis=0)

                            if self.use_time and self.time_aux:
                                time_layer = create_time_layer(target_timestep, self.output_shape)
                                yield image, time_layer, target_image, target_mask
                            yield image, target_image, target_mask
