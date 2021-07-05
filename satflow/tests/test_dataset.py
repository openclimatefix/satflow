# NIR1.6, VIS0.8 and VIS0.6 RGB for near normal view
import numpy as np
import matplotlib.pyplot as plt
from satflow.data.datasets import SatFlowDataset, CloudFlowDataset
import webdataset as wds


def test_satflow():
    dataset = wds.WebDataset("../../datasets/satflow-flow-4-tiled-00001.tar").decode()
    sources = iter(dataset)

    pass
