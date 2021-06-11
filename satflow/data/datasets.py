from typing import Iterator, Dict, List, Any, Union, Optional

import torch
import torch.utils.data as thd
from torch.utils.data.dataset import T_co

import numpy as np
import webdataset as wds

"""

Need the data to be aligned by the same time and place, so need to project things to the globe and use that. 

Datasets to include: 
Topographic maps (elevation), 
Satellite Imagery (all channels or subset),
Rapid Scan products (atmopsheric variation, winds, although maybe not as good),
Radar if we can get it,
Cloud Masks

Want the targets to be the new cloud mask? Or actually, just the new image, more just needs to predict the cloud mask though
So have two possible targets, just cloud mask at time T in the future, and entire image at time T in future

"""

def create_time_layer(shape, index):
    """Create tiem prediction layer, e.g. 256x256x240 where one hot encoding of the time period to predict"""
    return NotImplementedError


class SatFlowDataset(thd.IterableDataset, wds.Shorthands, wds.Composable):

    def __init__(self, datasets, config):
        super().__init__()
        self.config = config
        self.datasets = datasets
        self.num_timesteps = config["num_timesteps"]
        self.forecast_times = config["forecast_times"]

        # Defined output sizes, etc.
        self.output_shape = config['output_shape']
        self.target_type = config.get("target_type", "mask")
        # Should load the common data here
        self.use_topo = config.get('use_topo', False)
        self.use_latlon = config.get('use_latlon', False)
        self.topo = np.load("satflow/resources/cutdown_europe_dem.npy") if self.use_topo else None
        self.location = np.load("satflow/resources/location_array.npy") if self.use_latlon else None

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
                # Add topo and location data, if we want
                if self.use_topo:

            yield NotImplementedError
