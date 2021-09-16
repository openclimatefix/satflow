from typing import Tuple, Union, List
from nowcasting_dataset.dataset.datasets import NetCDFDataset
from nowcasting_dataset.dataset.example import Example


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
        required_keys: Union[Tuple[str], List[str]] = (
            "nwp",
            "nwp_x_coords",
            "nwp_y_coords",
            "sat_data",
            "sat_x_coords",
            "sat_y_coords",
        ),
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
        self.current_index = self.history_steps + 1
        self.required_keys = list(required_keys)

    def __getitem__(self, batch_idx: int):
        """Returns a whole batch at once.

        Args:
          batch_idx: The integer index of the batch. Must be in the range
          [0, self.n_batches).

        Returns:
            NamedDict where each value is a numpy array. The size of this
            array's first dimension is the batch size.
        """
        batch = super().__getitem__(batch_idx)

        # Need to partition out past and future sat images here, along with the rest of the data

        return batch
