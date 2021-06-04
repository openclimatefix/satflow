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

    def __iter__(self) -> Iterator[T_co]:
        pass