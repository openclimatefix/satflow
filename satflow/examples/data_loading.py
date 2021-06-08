#import os
#os.environ["PYTROLL_CHUNK_SIZE"] = "512"
#import dask
#from multiprocessing.pool import ThreadPool
#dask.config.set(pool=ThreadPool(1))

import rasterio
import rasterio
from rasterio.plot import show
import datetime
import glob
import subprocess
import os
import numpy as np

from satpy import Scene
import cartopy.crs as ccrs
import pandas as pd

import matplotlib.pyplot as plt


import rasterio
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling, calculate_default_transform, transform
from rasterio.control import GroundControlPoint
from rasterio.transform import xy

import xarray as xr
from collections import OrderedDict
from itertools import product
import satpy
from pyresample import load_area

scene = Scene(
    filenames={"seviri_l1b_native": ["/run/media/jacob/Round1/EUMETSAT/2021/06/01/12/29/MSG3-SEVI-MSG15-0100-NA-20210601122916.506000000Z-NA.nat"],
               "seviri_l2_grib": ["/run/media/jacob/Round1/EUMETSAT/2021/06/01/12/29/MSG3-SEVI-MSGCLMK-0100-0100-20210601123000.000000000Z-NA.grb"]},)

print(scene.available_dataset_names())
scene.load(['HRV', 'IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073', 'cloud_mask'], upper_right_corner='NE')
tmerc_areas = load_area("areas.yaml")
print(tmerc_areas)
new_scene = scene.resample(tmerc_areas[0])
new_scene.show('cloud_mask')
exit()
new_scene.save_datasets(compute=True, writer="png")
exit()

def decompress(full_bzip_filename: str) -> str:
    base_bzip_filename = os.path.basename(full_bzip_filename)
    base_nat_filename = os.path.splitext(base_bzip_filename)[0]
    full_nat_filename = os.path.join("./", base_nat_filename)
    if os.path.exists(full_nat_filename):
        os.remove(full_nat_filename)
    with open(full_nat_filename, 'wb') as nat_file_handler:
        process = subprocess.run(
            ['pbzip2', '--decompress', '--keep', '--stdout', full_bzip_filename],
            stdout=nat_file_handler)
    process.check_returncode()
    return full_nat_filename

