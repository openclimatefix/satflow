from typing import Tuple, Union, List
from nowcasting_dataset.dataset import NetCDFDataset


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
    ):
        """
        Args:
          n_batches: Number of batches available on disk.
          src_path: The full path (including 'gs://') to the data on
            Google Cloud storage.
          tmp_path: The full path to the local temporary directory
            (on a local filesystem).
        """
        super().__init__(n_batches, src_path, tmp_path, cloud, required_keys)

        # SatFlow specific changes to it

    def __getitem__(self, batch_idx: int) -> example.Example:
        """Returns a whole batch at once.

        Args:
          batch_idx: The integer index of the batch. Must be in the range
          [0, self.n_batches).

        Returns:
            NamedDict where each value is a numpy array. The size of this
            array's first dimension is the batch size.
        """
        if not 0 <= batch_idx < self.n_batches:
            raise IndexError(
                "batch_idx must be in the range" f" [0, {self.n_batches}), not {batch_idx}!"
            )
        netcdf_filename = nd_utils.get_netcdf_filename(batch_idx)
        remote_netcdf_filename = os.path.join(self.src_path, netcdf_filename)
        local_netcdf_filename = os.path.join(self.tmp_path, netcdf_filename)

        if self.cloud == "gcp":
            gcp_download_to_local(
                remote_filename=remote_netcdf_filename,
                local_filename=local_netcdf_filename,
                gcs=self.gcs,
            )
        else:
            aws_download_to_local(
                remote_filename=remote_netcdf_filename,
                local_filename=local_netcdf_filename,
                s3_resource=self.s3_resource,
            )

        netcdf_batch = xr.load_dataset(local_netcdf_filename)
        os.remove(local_netcdf_filename)

        batch = example.Example(
            sat_datetime_index=netcdf_batch.sat_time_coords,
            nwp_target_time=netcdf_batch.nwp_time_coords,
        )
        for key in self.required_keys + list(example.DATETIME_FEATURE_NAMES):
            try:
                batch[key] = netcdf_batch[key]
            except KeyError:
                pass

        sat_data = batch["sat_data"]
        if sat_data.dtype == np.int16:
            sat_data = sat_data.astype(np.float32)
            sat_data = sat_data - SAT_MEAN
            sat_data /= SAT_STD
            batch["sat_data"] = sat_data

        batch = example.to_numpy(batch)

        return batch
