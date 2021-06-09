import webdataset
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import glob
import os

dem_path = "/run/media/bieker/T7/SRTM 1 Arc-Second Global/*.tif"

#src = rasterio.open("/run/media/bieker/Round1/EUMETSAT/europe_dem.tif")
#show(src, cmap='terrain')
#exit()

dem_files = glob.glob(dem_path)
import numpy as np
src_files_to_mosaic = []
for fp in dem_files:
    src = rasterio.open(fp)
    #src.astype(np.int8)
    src_files_to_mosaic.append(src)

from tempfile import mkdtemp
import rasterio
from rasterio import Affine
from rasterio import windows
from rasterio.enums import Resampling
import math
import numpy as np
import os

INPUT_FILES = dem_files

sources = [rasterio.open(raster) for raster in INPUT_FILES]
print(sources[0].res)

mosaic, out_trans = merge(src_files_to_mosaic, nodata=0, res=(0.03, 0.03))
print(mosaic.shape)
show(mosaic, cmap='terrain')
out_meta = src.meta.copy()

out_meta.update({"driver": "GTiff",
                 "height": mosaic.shape[1],
                 "width": mosaic.shape[2],
                 "transform": out_trans,
                 "crs":  "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "})
out_fp = "/run/media/bieker/data/EUMETSAT/europe_all_dem_300m_nearest.tif"
with rasterio.open(out_fp, "w", **out_meta) as dest:
    dest.write(mosaic)

