from typing import List, Optional, Tuple, Union

import numpy as np
from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.consts import (
    DATETIME_FEATURE_NAMES,
    NWP_DATA,
    NWP_X_COORDS,
    NWP_Y_COORDS,
    SATELLITE_DATA,
    SATELLITE_DATETIME_INDEX,
    SATELLITE_X_COORDS,
    SATELLITE_Y_COORDS,
    TOPOGRAPHIC_DATA,
)
from nowcasting_dataset.dataset.datasets import NetCDFDataset


class SatFlowDataset(NetCDFDataset):
    """Loads data saved by the `prepare_ml_training_data.py` script.
    Adapted from predict_pv_yield
    """

    def __init__(
        self,
        n_batches: int,
        src_path: str,
        tmp_path: str,
        configuration: Configuration,
        cloud: str = "gcp",
        required_keys: Union[Tuple[str], List[str]] = [
            NWP_DATA,
            NWP_X_COORDS,
            NWP_Y_COORDS,
            SATELLITE_DATA,
            SATELLITE_X_COORDS,
            SATELLITE_Y_COORDS,
            SATELLITE_DATETIME_INDEX,
            TOPOGRAPHIC_DATA,
        ]
        + list(DATETIME_FEATURE_NAMES),
        history_minutes: int = 30,
        forecast_minutes: int = 60,
        combine_inputs: bool = False,
    ):
        """
        Args:
          n_batches: Number of batches available on disk.
          src_path: The full path (including 'gs://') to the data on
            Google Cloud storage.
          tmp_path: The full path to the local temporary directory
            (on a local filesystem).
        batch_size: Batch size, if requested, will subset data along batch dimension
        """
        super().__init__(
            n_batches,
            src_path,
            tmp_path,
            configuration,
            cloud,
            required_keys,
            history_minutes,
            forecast_minutes,
        )
        # SatFlow specific changes, i.e. which timestep to split on
        self.required_keys = list(required_keys)
        self.combine_inputs = combine_inputs
        self.current_timestep_index = (history_minutes // 5) + 1

    def __getitem__(self, batch_idx: int):
        batch = super().__getitem__(batch_idx)

        # Need to partition out past and future sat images here, along with the rest of the data
        past_satellite_data = batch[SATELLITE_DATA][:, : self.current_timestep_index]
        future_sat_data = batch[SATELLITE_DATA][:, self.current_timestep_index :]
        x = {
            SATELLITE_DATA: past_satellite_data,
            SATELLITE_X_COORDS: batch.get(SATELLITE_X_COORDS, None),
            SATELLITE_Y_COORDS: batch.get(SATELLITE_Y_COORDS, None),
            SATELLITE_DATETIME_INDEX: batch[SATELLITE_DATETIME_INDEX][
                :, : self.current_timestep_index
            ],
        }
        y = {
            SATELLITE_DATA: future_sat_data,
            SATELLITE_DATETIME_INDEX: batch[SATELLITE_DATETIME_INDEX][
                :, self.current_timestep_index :
            ],
        }

        for k in list(DATETIME_FEATURE_NAMES):
            if k in self.required_keys:
                x[k] = batch[k][:, : self.current_timestep_index]

        if NWP_DATA in self.required_keys:
            past_nwp_data = batch[NWP_DATA][:, :, : self.current_timestep_index]
            x[NWP_DATA] = past_nwp_data
            x[NWP_X_COORDS] = batch.get(NWP_X_COORDS, None)
            x[NWP_Y_COORDS] = batch.get(NWP_Y_COORDS, None)

        if TOPOGRAPHIC_DATA in self.required_keys:
            # Need to expand dims to get a single channel one
            # Results in topographic maps with [Batch, Channel, H, W]
            x[TOPOGRAPHIC_DATA] = np.expand_dims(batch[TOPOGRAPHIC_DATA], axis=1)

        return x, y
