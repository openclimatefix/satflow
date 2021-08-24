import glob
from typing import Iterator, List, Type, Union, Tuple

import albumentations as A
import numpy as np
import torch.utils.data as thd
import webdataset as wds
from torch.utils.data.dataset import T_co
import logging
import pickle
import random
import os
from satflow.data.utils.normalization import metnet_normalization, standard_normalization
from satflow.data.utils.utils import (
    create_time_layer,
    load_np,
    binarize_mask,
    create_pixel_coord_layers,
    check_channels,
    crop_center,
    load_config,
)

logger = logging.getLogger("satflow.dataset")
logger.setLevel(logging.WARN)

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
                "HRV",  # mask[mask == 255] = 0  # hardcoded incase missing any of the cloud mask background problem
                # mask[mask < 2] = 0  # 2 is cloud, others are clear over land (1) and clear over water (0)
                # mask[mask > 0] = 1
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
        # Make it match the dimensions of the output so that it can be broadcasted
        self.mean = np.expand_dims(self.mean, (0, 2, 3))
        self.std = np.expand_dims(self.std, (0, 2, 3))

        self.use_topo = config.get("use_topo", False)
        self.use_latlon = config.get("use_latlon", False)
        self.use_time = config.get("use_time", True)
        self.time_aux = config.get("time_aux", False)  # Time as an auxiliary input as 1D array
        self.use_mask = config.get("use_mask", True)
        self.use_image = config.get("use_image", False)
        self.return_target_stack = config.get("stack_targets", False)
        self.time_as_channels = config.get("time_as_channels", False)
        self.add_pixel_coords = config.get("add_pixel_coords", False)
        self.metnet_norm = config.get("metnet_normalization", False)

        # Number of channels in final one
        self.num_channels = check_channels(config)
        self.num_bands = len(self.bands)
        self.total_per_timestep_channels = (
            self.num_bands + 3 if self.use_time and not self.time_aux else self.num_bands
        )
        self.total_per_timestep_channels = (
            self.total_per_timestep_channels + 1
            if self.use_mask
            else self.total_per_timestep_channels
        )
        self.input_cube = np.empty(
            (self.num_timesteps + 1, self.num_channels, self.output_shape, self.output_shape)
        )
        self.input_mask_cube = np.empty(
            (self.num_timesteps + 1, 1, self.output_shape, self.output_shape)
        )
        self.target_cube = np.empty((self.forecast_times, 1, self.output_shape, self.output_shape))
        self.target_image_cube = np.empty(
            (self.forecast_times, self.num_bands, self.output_shape, self.output_shape)
        )
        self.image_input = True
        self.pixel_coords = create_pixel_coord_layers(
            self.output_shape, self.output_shape, with_r=config.get("add_polar_coords", False)
        )
        self.pixel_coords = np.squeeze(self.pixel_coords)

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
        if self.train:
            transforms.append(A.RandomCrop(self.output_shape, self.output_shape))
        else:
            transforms.append(
                A.RandomCrop(self.output_shape, self.output_shape)
            )  # TODO Make sure is reproducible
        self.aug = A.ReplayCompose(
            transforms,
        )
        self.replay = None

    def visualize(self, image: np.ndarray, target_image: np.ndarray, mask: np.ndarray):
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

    def create_target_time_cube(self, target_timestep: int) -> np.ndarray:
        """Create target time layer"""
        time_cube = np.zeros(
            (1, self.forecast_times, self.output_shape, self.output_shape), dtype=np.int8
        )
        time_cube[:, target_timestep, :, :] = 1
        return time_cube

    def create_target_time_layer(self, target_timestep: int) -> np.ndarray:
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

    def get_timestep(
        self, sample: dict, idx: int, return_target=False, return_image=True
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, np.ndarray]]:
        """
        Gets the image stack of the given timestep, if return_target is true, only returns the mask and satellite channels
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
        img_cube = np.empty(
            shape=(target.shape[0], target.shape[1], self.total_per_timestep_channels)
        )
        bands = np.array([load_np(sample[f"{b.lower()}.{idx:03d}.npy"]) for b in self.bands])
        img_cube[:, :, : self.num_bands] = bands.transpose((1, 2, 0))

        if self.use_time and not self.time_aux:
            t = create_time_layer(
                pickle.loads(sample["time.pyd"])[idx],
                shape=(img_cube.shape[0], img_cube.shape[1]),
            )
            img_cube[:, :, self.num_bands : self.num_bands + 3] = t
        if self.use_mask:
            mask = binarize_mask(load_np(sample[f"cloudmask.{idx:03d}.npy"]))
            # Resample between -1 and 1 like the rest
            mask = (mask * 2) - 1.0
            img_cube[:, :, -1] = mask
        return img_cube, target

    def __iter__(self):
        # Need to make sure same time step for all of them.
        # As its all from rapid scan, should be fairly easy.
        # Main missing one is the regional and rapid weather ones, which are every 15 minutes,
        # but could be interpolated between the previous step and next one by weighting by time difference
        # Topographic is same of course, just need to resize to 1km x 1km?
        # grid by taking the mean value of the interior ones
        while True:
            sources = [iter(ds) for ds in self.datasets]
            if not self.train:  # Same for validation each time for each source
                np.random.seed(42)
                # Have to set Python random seed for Albumentations
                random.seed(a=42)
                # Reduce number of repeat examples from each source, gives more variety in the inputs
                self.num_crops = 5
                self.num_times = 4
            for source in sources:
                try:
                    sample = next(source)
                except StopIteration:
                    continue
                timesteps = pickle.loads(sample["time.pyd"])
                logger.debug(f"Timesteps: {timesteps}")
                available_steps = len(timesteps)  # number of available timesteps
                logger.debug(f"Available Timesteps: {available_steps}")
                # Check to make sure all timesteps exist
                sample_keys = [key for key in sample.keys() if self.bands[0].lower() in key]
                key_checker = [
                    f"{self.bands[0].lower()}.{idx:03d}.npy" for idx in range(1, available_steps)
                ]
                if (
                    not all(e in sample_keys for e in key_checker)
                    or self.num_timesteps * self.skip_timesteps + 1
                    >= available_steps - self.forecast_times
                ):
                    logger.debug(
                        f"Issue with {self.num_timesteps * self.skip_timesteps + 1} >= {available_steps - self.forecast_times} with {available_steps} available timesteps"
                    )
                    continue  # Skip this sample as it is missing timesteps, or has none
                # Times that have enough previous timesteps and post timesteps for training
                # pick one at random
                idxs = np.random.randint(
                    self.num_timesteps * self.skip_timesteps + 1,
                    available_steps - self.forecast_times,
                    size=self.num_times,
                )
                self.get_topo_latlon(sample)
                for idx in idxs:
                    target_timesteps = np.full(self.num_crops, idx + self.forecast_times)
                    for _ in range(self.num_crops):  # Do random crops as well for training
                        for target_timestep in target_timesteps:
                            input_idxs = list(
                                range(
                                    idx - (self.num_timesteps * self.skip_timesteps),
                                    idx + self.skip_timesteps,
                                    self.skip_timesteps,
                                )
                            )
                            logger.debug(f"Input IDXs: {input_idxs}")
                            if not self.create_stack(
                                input_idxs, sample, is_input=self.image_input
                            ):
                                self.replay = None
                                continue
                            # Now in a Time x W x H x Channel order
                            target_idxs = list(range(idx + 1, target_timestep + 1))
                            logger.debug(f"Target IDXs: {target_idxs}")
                            logger.debug(
                                f"Timesteps: Current: {timesteps[input_idxs[-1]]} Prev: {timesteps[input_idxs[0]]} Next: {timesteps[target_idxs[0]]} Final: {timesteps[target_idxs[-1]]} "
                                f"Timedelta: Next - Curr: {timesteps[target_idxs[0]] - timesteps[input_idxs[-1]] } End - Curr: {timesteps[target_idxs[-1]] - timesteps[input_idxs[-1]]}"
                            )
                            if not self.create_stack(target_idxs, sample, is_input=False):
                                self.replay = None
                                continue
                            logger.debug(f"Target Masks Shape: {self.target_cube.shape}")
                            logger.debug(
                                f"After Time Changes Image/Masks Shape: {self.input_cube.shape} {self.target_cube.shape}"
                            )
                            self.input_cube[:, : self.num_bands, :, :] = (
                                metnet_normalization(self.input_cube[:, : self.num_bands, :, :])
                                if self.metnet_norm
                                else standard_normalization(
                                    self.input_cube[:, : self.num_bands, :, :],
                                    std=self.std,
                                    mean=self.mean,
                                )
                            )
                            if self.use_image:
                                self.target_image_cube[:, : self.num_bands, :, :] = (
                                    metnet_normalization(
                                        self.target_image_cube[:, : self.num_bands, :, :]
                                    )
                                    if self.metnet_norm
                                    else standard_normalization(
                                        self.target_image_cube[:, : self.num_bands, :, :],
                                        std=self.std,
                                        mean=self.mean,
                                    )
                                )
                            start_channel = self.num_bands + 3 if self.use_time else self.num_bands
                            start_channel = start_channel + 1 if self.use_mask else start_channel
                            logger.info(f"Start Channel: {start_channel}")
                            self.add_aux_layers(start_channel=start_channel)
                            logger.debug(
                                f"After Aux Layers Image/Masks Shape: {self.input_cube.shape} {self.target_cube.shape}"
                            )
                            # Reset the replay
                            self.replay = None
                            if self.output_target != self.output_shape:
                                if self.use_image:
                                    target_image = crop_center(
                                        self.target_image_cube,
                                        self.output_target,
                                        self.output_target,
                                    )
                                target_mask = crop_center(
                                    self.target_cube, self.output_target, self.output_target
                                )
                            else:
                                if self.use_image:
                                    target_image = self.target_image_cube
                                target_mask = self.target_cube
                            # Now convert to channels if time_as_channels
                            if self.time_as_channels:
                                image, target_mask, target_image = self.time_changes(
                                    self.input_cube if self.image_input else self.input_mask_cube,
                                    self.target_cube,
                                    self.target_image_cube,
                                )
                            else:
                                image = (
                                    self.input_cube if self.image_input else self.input_mask_cube
                                )
                            # Need to make sure no NaNs and change dtype
                            image = np.nan_to_num(
                                image, copy=False, neginf=0.0, posinf=0.0
                            ).astype(np.float32)
                            target_mask = np.nan_to_num(
                                target_mask, copy=False, neginf=0.0, posinf=0.0
                            ).astype(np.float32)
                            if self.use_image:
                                target_image = np.nan_to_num(
                                    target_image, copy=False, neginf=0.0, posinf=0.0
                                ).astype(np.float32)
                            if self.vis:
                                self.visualize(image, target_image=target_image, mask=target_mask)
                            if self.use_time and self.time_aux:
                                time_layer = create_time_layer(
                                    target_timestep - idx - 1, self.output_shape
                                )
                                yield image, time_layer, target_image, target_mask
                            if not self.use_image:
                                yield image, target_mask
                            else:
                                yield image, target_image

    def get_topo_latlon(self, sample: dict) -> None:
        if self.use_topo:
            topo = load_np(sample["topo.npy"])
            topo[topo < 100] = 0  # Elevation shouldn't really be below 0 here (ocean mostly)
            self.topo = (topo - TOPO_MEAN) / TOPO_STD
            self.topo = np.expand_dims(self.topo, axis=0)
        if self.use_latlon:
            self.location = load_np(sample["location.npy"])
            self.location = np.moveaxis(self.location, [2], [0])

    def create_stack(self, idxs: list, sample: dict, is_input: bool = False) -> bool:
        # Now create stack here
        time_idx = 0
        for i in idxs:
            t_image, t_mask = self.get_timestep(
                sample,
                i,
                return_target=True,
                return_image=self.use_image or is_input,
            )
            if self.replay is None:
                data = self.aug(image=t_mask)
                self.replay = data["replay"]
                logger.debug(self.replay)
            t_mask = self.aug.replay(self.replay, image=t_mask)["image"]
            if t_image is not None:
                t_image = self.aug.replay(self.replay, image=t_image)["image"]
            if is_input:
                self.input_mask_cube[time_idx, :, :, :] = t_mask
            else:
                self.target_cube[time_idx, :, :, :] = t_mask
            if is_input:
                self.input_cube[time_idx, : t_image.shape[2], :, :] = t_image.transpose((2, 1, 0))
            elif t_image is not None:
                remove_last_channels = 3 if self.use_time else 0
                remove_last_channels = (
                    remove_last_channels + 1 if self.use_mask else remove_last_channels
                )
                t_image = (
                    t_image[:, :, :-remove_last_channels] if remove_last_channels > 0 else t_image
                )
                self.target_image_cube[time_idx, : t_image.shape[2], :, :] = t_image.transpose(
                    (2, 1, 0)
                )

            time_idx += 1
        # Convert to Time x Channel x W x H
        # target_mask = np.expand_dims(target_mask, axis=1)
        # One timestep as well
        # Ensure last target mask is also different than previous ones -> only want ones where things change
        if np.allclose(self.target_cube[0], self.target_cube[-1]) and not is_input:
            return False
        return True

    def add_aux_layers(self, start_channel: int) -> None:
        if self.use_topo:
            self.apply_aug_to_time(self.topo, start_channel=start_channel)
            start_channel += 1  # Topo is single channel
        if self.use_latlon:
            self.apply_aug_to_time(self.location, start_channel=start_channel)
            start_channel += 3  # Location is triple channel
        if self.add_pixel_coords:
            self.input_cube[
                :, start_channel : start_channel + self.pixel_coords.shape[0], :, :
            ] = self.pixel_coords

    def apply_aug_to_time(self, data: np.ndarray, start_channel: int = 1) -> None:
        data = np.moveaxis(
            data,
            [0],
            [2],
        )
        data = self.aug.replay(self.replay, image=data)["image"]
        data = np.moveaxis(
            data,
            [2],
            [0],
        )
        self.input_cube[:, start_channel : start_channel + data.shape[0], :, :] = data

    def time_changes(
        self, inputs: np.ndarray, target: np.ndarray, target_image: np.ndarray
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, None]]:
        # Time changes has to be different make new array and copy into that one
        if self.image_input and self.num_channels - self.total_per_timestep_channels != 0:
            time_flattened_input = np.empty(
                (
                    self.total_per_timestep_channels * inputs.shape[0]
                    + (self.num_channels - self.total_per_timestep_channels),
                    inputs.shape[2],
                    inputs.shape[3],
                )
            )
            time_flattened_input[
                : self.total_per_timestep_channels * inputs.shape[0], :, :
            ] = inputs[:, : self.total_per_timestep_channels, :, :].reshape(
                -1, inputs.shape[2], inputs.shape[3]
            )
            time_flattened_input[
                -(self.num_channels - self.total_per_timestep_channels) :, :, :
            ] = inputs[0, self.total_per_timestep_channels :, :, :]
            inputs = time_flattened_input
        else:
            inputs = inputs.reshape((-1, inputs.shape[2], inputs.shape[3]))
        targets = target.reshape((-1, target.shape[2], target.shape[3]))
        target_image = target_image.reshape((-1, target_image.shape[2], target_image.shape[3]))
        return inputs, targets, target_image


class CloudFlowDataset(SatFlowDataset):
    def __init__(self, datasets: List[wds.WebDataset], config: dict, train: bool = True):
        super(CloudFlowDataset, self).__init__(datasets, config, train)
        self.image_input = False  # Only want Mask -> Mask training


class PerceiverDataset(SatFlowDataset):
    def __init__(self, datasets: List[wds.WebDataset], config: dict, train: bool = True):
        super(PerceiverDataset, self).__init__(datasets, config, train)
        self.image_input = True

    def add_position_encoding(self):
        pass

    def encode_images(self):
        pass


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
                idxs = list(range(2, available_steps - self.forecast_times))
                for idx in idxs:
                    for _ in range(self.num_crops):  # Do random crops as well for training
                        logger.debug(f"IDX: {idx}")
                        logger.debug(
                            f"Timesteps: Current: {timesteps[idx]} Prev: {timesteps[idx - 1]} Next: {timesteps[idx + 1]} Final: {timesteps[idx + self.forecast_times - 1]} "
                            f"Timedelta: Next - Curr: {timesteps[idx + 1] - timesteps[idx] } End - Curr: {timesteps[idx + self.forecast_times - 1] - timesteps[idx]}"
                        )
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
                            idx + 1,
                            return_target=True,
                            return_image=False,
                        )
                        target_mask = self.aug.replay(replay, image=target_mask)["image"]
                        target_mask = np.expand_dims(target_mask, axis=0)

                        if np.isclose(np.min(target_mask), np.max(target_mask)):
                            continue  # Ignore if target timestep has no clouds, or only clouds
                        # Now create stack here
                        for i in range(idx + 2, idx + self.forecast_times + 1):
                            _, t_mask = self.get_timestep(
                                sample,
                                i,
                                return_target=True,
                                return_image=False,
                            )
                            t_mask = self.aug.replay(replay, image=t_mask)["image"]
                            target_mask = np.concatenate(
                                [target_mask, np.expand_dims(t_mask, axis=0)]
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


class FileDataset(thd.Dataset):
    def __getitem__(self, index) -> T_co:
        try:
            if index in self.bad_indexes:
                index = np.random.randint(0, len(self.files))
                while index in self.bad_indexes:
                    index = np.random.randint(0, len(self.files))
            arrays = np.load(self.files[index])
            return arrays["images"], arrays["future_images" if self.use_image else "masks"][:24]
        except:
            self.bad_indexes.append(index)
            replacement_index = np.random.randint(0, len(self.files))
            while replacement_index in self.bad_indexes:
                replacement_index = np.random.randint(0, len(self.files))
            arrays = np.load(self.files[replacement_index])
            return arrays["images"], arrays["future_images" if self.use_image else "masks"][:24]

    def __init__(self, directory: str = "./", train: bool = True, use_image: bool = True):
        super().__init__()
        search_space = os.path.join(directory, f"{'train' if train else 'val'}_*.npz")
        self.files = glob.glob(search_space)
        self.use_image = use_image
        self.bad_indexes = []

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    # NIR1.6, VIS0.8 and VIS0.6 RGB for near normal view
    import numpy as np
    import webdataset as wds

    dataset = wds.WebDataset("../../datasets/satflow-test.tar")
    config = load_config("../tests/configs/satflow.yaml")
    cloudflow = SatFlowDataset([dataset], config)
    datas = iter(cloudflow)
    i = 0
    print("Starting Data")
    for data in datas:
        print(f"Data {i}")
        i += 1
        if i + 1 > 50:
            break
    config = load_config("../tests/configs/satflow_all.yaml")
    cloudflow = SatFlowDataset([dataset], config)
    datas = iter(cloudflow)
    i = 0
    for data in datas:
        i += 1
        if i + 1 > 50:
            exit()
