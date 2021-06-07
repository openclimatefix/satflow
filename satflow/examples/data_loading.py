import torch
import rasterio
import rasterio
from rasterio.plot import show
from satflow.data.utils import eumetsat_name_to_datetime
import datetime
import glob



print(eumetsat_name_to_datetime("/run/media/jacob/Round1/EUMETSAT/MSG3-SEVI-MSGCLMK-0100-0100-20200101080000.000000000Z-NA.grb"))
base_pth = "/run/media/jacob/Round1/EUMETSAT/"
filenames = glob.glob(base_pth + "*.grb")
import os
import shutil

for f in filenames:
    date_time = eumetsat_name_to_datetime(f) - datetime.timedelta(minutes=1)
    base_native_filename = os.path.basename(f)

    new_dst_path = os.path.join(base_pth, date_time.strftime("%Y/%m/%d/%H/%M"))
    if not os.path.exists(new_dst_path):
        os.makedirs(new_dst_path)
    EXTENSION = ".grb"
    new_dst_full_filename = os.path.join(new_dst_path, base_native_filename + EXTENSION)

    shutil.move(src=f, dst=new_dst_path)



topo_loc = "/home/jacob/Datasets/UK_DEM_GeoTIFF/SRTM 1 Arc-Second Global/"

fp = "/home/jacob/Datasets/UK_DEM_GeoTIFF/SRTM 1 Arc-Second Global/n51_w001_1arc_v3.tif"
img = rasterio.open(fp)
show(img)
