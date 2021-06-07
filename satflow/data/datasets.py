from typing import Iterator, Dict, List, Any, Union, Optional

import torch
import torch.utils.data as thd
from torch.utils.data.dataset import T_co

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
        

class SatFlowDataset(thd.IterableDataset):

    def __init__(self, config):
        self.config = config
        self.datasets = config["datasets"]
        self.num_timesteps = config["num_timesteps"]
        # Should set the data types here

    def __iter__(self) -> Iterator[T_co]:
        sample = {}
        # Need to make sure same time step for all of them.
        # As its all from rapid scan, should be fairly easy.
        # Main missing one is the regional and rapid weather ones, which are every 15 minutes,
        # but could be interpolated between the previous step and next one by weighting by time difference
        # Topographic is same of course, just need to resize to 1km x 1km?
        # grid by taking the mean value of the interior ones

        for d in self.datasets:
            yield NotImplementedError
