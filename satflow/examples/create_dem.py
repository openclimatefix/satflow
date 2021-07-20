import glob

import numpy as np
import rasterio
from pyresample import load_area
from rasterio.merge import merge
from rasterio.plot import show
from satpy import Scene


def create_pixel_coord_layers(x_dim: int, y_dim: int, with_r: bool = False) -> np.ndarray:
    """
    Creates Coord layer for CoordConv model

    :param x_dim: size of x dimension for output
    :param y_dim: size of y dimension for output
    :param with_r: Whether to include polar coordinates from center
    :return: (2, x_dim, y_dim) or (3, x_dim, y_dim) array of the pixel coordinates
    """
    xx_ones = np.ones([1, x_dim], dtype=np.int32)
    xx_ones = np.expand_dims(xx_ones, -1)

    xx_range = np.expand_dims(np.arange(x_dim), 0)
    xx_range = np.expand_dims(xx_range, 1)

    xx_channel = np.matmul(xx_ones, xx_range)
    xx_channel = np.expand_dims(xx_channel, -1)

    yy_ones = np.ones([1, y_dim], dtype=np.int32)
    yy_ones = np.expand_dims(yy_ones, 1)

    yy_range = np.expand_dims(np.arange(y_dim), 0)
    yy_range = np.expand_dims(yy_range, -1)

    yy_channel = np.matmul(yy_range, yy_ones)
    yy_channel = np.expand_dims(yy_channel, -1)

    xx_channel = xx_channel.astype("float32") / (x_dim - 1)
    yy_channel = yy_channel.astype("float32") / (y_dim - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1
    ret = np.concatenate((xx_channel, yy_channel), axis=0)
    if with_r:
        rr = np.sqrt(np.square(xx_channel - 0.5) + np.square(yy_channel - 0.5))
        ret = np.concatenate([ret, rr], axis=0)
    ret = ret.squeeze(axis=-1)
    ret = np.expand_dims(ret, axis=0)
    return ret


ret = create_pixel_coord_layers(x_dim=5, y_dim=5, with_r=False)
img = np.random.random((6, 13, 128, 128))
np.concatenate([img, ret], axis=1)
exit()

dem_path = "/home/bieker/bda/Bulk Order 20210611_041625/SRTM 1 Arc-Second Global/*.tif"
dem_files = glob.glob(dem_path)

src_files_to_mosaic = []
for fp in dem_files:
    src = rasterio.open(fp)
    # src.astype(np.int8)
    src_files_to_mosaic.append(src)

areas = load_area("/home/bieker/Development/satflow/satflow/examples/areas.yaml")
for f in dem_files:
    scene = Scene(filenames={"generic_image": [f]})
    scene.load(["image"])
    scene = scene.resample(areas[0])
    fname = f.split("/")[-1]
    scene.save_datasets(
        writer="geotiff",
        base_dir="/run/media/bieker/T7/ResampledElevation/",
        filename=fname,
        enhance=False,
    )

dem_path = "/run/media/bieker/T7/ResampledElevation/*.tif"
dem_files = glob.glob(dem_path)

src_files_to_mosaic = []
for fp in dem_files:
    src = rasterio.open(fp)
    # src.astype(np.int8)
    src_files_to_mosaic.append(src)
mosaic, out_trans = merge(src_files_to_mosaic, nodata=0, method="max")
print(mosaic.shape)
show(mosaic[0], cmap="terrain")
show(mosaic[0][515:-641, 603:], cmap="terrain")
out_meta = src.meta.copy()

out_meta.update(
    {
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": "+proj=utm +zone=35 +ellps=WGS84 +units=m +no_defs ",
    }
)
out_fp = "europe_all_dem_3km_satproj.tif"
with rasterio.open(out_fp, "w", **out_meta) as dest:
    dest.write(mosaic)

out = rasterio.open(out_fp)
show(out, cmap="terrain")
# Save the numpy array, we don't care about other geotspatial transforms here
np.save("../resources/cutdown_europe_dem.npy", mosaic[0][515:-641, 603:])
np.save("europe_dem.npy", mosaic[0])
