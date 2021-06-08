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
    filenames=["/run/media/jacob/Round1/EUMETSAT/2021/06/01/12/29/MSG3-SEVI-MSG15-0100-NA-20210601122916.506000000Z-NA.nat"],
    reader='seviri_l1b_native')

scene.load(['HRV', 'IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073'], upper_right_corner='NE')
tmerc_areas = load_area("areas.yaml")
print(tmerc_areas)
new_scene = scene.resample(tmerc_areas[0])
new_scene.show('IR_016')
exit()
new_scene.save_datasets(compute=True)
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



#topo_loc = "/home/jacob/Datasets/UK_DEM_GeoTIFF/SRTM 1 Arc-Second Global/"

#fp = "/home/jacob/Datasets/UK_DEM_GeoTIFF/SRTM 1 Arc-Second Global/n51_w001_1arc_v3.tif"
#img = rasterio.open(fp)
#show(img)

import satpy
# Now go and open GRIB one
from satpy import Scene
#from satpy import available_readers
#print(available_readers())
#scn = Scene(filenames=["/run/media/jacob/Round1/EUMETSAT/2021/01/01/08/04/MSG3-SEVI-MSGCLMK-0100-0100-20210101080500.000000000Z-NA.grb"])
#import cfgrib
import xarray as xr
import cfgrib
grib_data = cfgrib.open_datasets('/run/media/jacob/Round1/EUMETSAT/2021/01/01/08/04/MSG3-SEVI-MSGCLMK-0100-0100-20210101080500.000000000Z-NA.grb')
print(grib_data)
time, lat, lon = grib_data[0].indexes.values()
print(lat)
print(len(grib_data))
for i in range(len(grib_data)):
    for var_name, values in grib_data[i].items():
        print(var_name)
#ds = xr.open_dataset('/run/media/jacob/Round1/EUMETSAT/2021/01/01/08/04/MSG3-SEVI-MSGCLMK-0100-0100-20210101080500.000000000Z-NA.grb', engine='cfgrib')

#print(ds)
#print(ds)

#grbs = cfgrib.open_dataset("/run/media/jacob/Round1/EUMETSAT/2021/01/01/08/04/MSG3-SEVI-MSGCLMK-0100-0100-20210101080500.000000000Z-NA.grb")


#scn.available_dataset_ids()

