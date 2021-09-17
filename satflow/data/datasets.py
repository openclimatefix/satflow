from typing import Tuple, Union, List
from nowcasting_dataset.dataset.datasets import NetCDFDataset
from nowcasting_dataset.consts import (
    SATELLITE_DATA,
    SATELLITE_X_COORDS,
    SATELLITE_Y_COORDS,
    SATELLITE_DATETIME_INDEX,
    NWP_DATA,
    NWP_Y_COORDS,
    NWP_X_COORDS,
    NWP_TARGET_TIME,
    DATETIME_FEATURE_NAMES,
)


class SatFlowDataset(NetCDFDataset):
    """Loads data saved by the `prepare_ml_training_data.py` script.
    Adapted from predict_pv_yield
    """

    def __init__(
        self,
        n_batches: int,
        src_path: str,
        tmp_path: str,
        cloud: str = "gcp",
        required_keys: Union[Tuple[str], List[str]] = [
            NWP_DATA,
            NWP_X_COORDS,
            NWP_Y_COORDS,
            SATELLITE_DATA,
            SATELLITE_X_COORDS,
            SATELLITE_Y_COORDS,
            SATELLITE_DATETIME_INDEX,
        ]
        + list(DATETIME_FEATURE_NAMES),
        history_minutes: int = 30,
        forecast_minutes: int = 60,
        current_timestep_index: int = 7,
    ):
        """
        Args:
          n_batches: Number of batches available on disk.
          src_path: The full path (including 'gs://') to the data on
            Google Cloud storage.
          tmp_path: The full path to the local temporary directory
            (on a local filesystem).
        """
        super().__init__(
            n_batches,
            src_path,
            tmp_path,
            cloud,
            required_keys,
            history_minutes,
            forecast_minutes,
            current_timestep_index,
        )
        # SatFlow specific changes, i.e. which timestep to split on
        self.history_steps = history_minutes // 5
        self.forecast_steps = forecast_minutes // 5
        self.current_index = (
            self.history_steps + 1
        )  # +2 as indexing does not include this index, so need to go one beyond
        self.required_keys = list(required_keys)

    def __getitem__(self, batch_idx: int):
        batch = super().__getitem__(batch_idx)

        # Need to partition out past and future sat images here, along with the rest of the data
        past_satellite_data = batch[SATELLITE_DATA][:, : self.current_index]
        future_sat_data = batch[SATELLITE_DATA][:, self.current_index :]
        x = {
            SATELLITE_DATA: past_satellite_data,
            SATELLITE_X_COORDS: batch.get(SATELLITE_X_COORDS, None),
            SATELLITE_Y_COORDS: batch.get(SATELLITE_Y_COORDS, None),
            SATELLITE_DATETIME_INDEX: batch[SATELLITE_DATETIME_INDEX][:, : self.current_index],
        }
        y = {
            SATELLITE_DATA: future_sat_data,
            SATELLITE_DATETIME_INDEX: batch[SATELLITE_DATETIME_INDEX][:, self.current_index :],
        }

        for k in list(DATETIME_FEATURE_NAMES):
            if k in self.required_keys:
                x[k] = batch[k][:, : self.current_index]

        if NWP_DATA in self.required_keys:
            past_nwp_data = batch[NWP_DATA][:, :, : self.current_index]
            x[NWP_DATA] = past_nwp_data
            x[NWP_X_COORDS] = batch.get(NWP_X_COORDS, None)
            x[NWP_Y_COORDS] = batch.get(NWP_Y_COORDS, None)

        return x, y
