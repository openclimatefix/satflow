import datetime
import io
import re

import affine
import numpy as np
import yaml

try:
    from pyresample import load_area
    from satpy import Scene

    _SAT_LIBS = True
except:
    print("No pyresample or satpy")
    _SAT_LIBS = False


def eumetsat_filename_to_datetime(inner_tar_name):
    """Takes a file from the EUMETSAT API and returns
    the date and time part of the filename"""

    p = re.compile("^MSG[23]-SEVI-MSG15-0100-NA-(\d*)\.")
    title_match = p.match(inner_tar_name)
    date_str = title_match.group(1)
    return datetime.datetime.strptime(date_str, "%Y%m%d%H%M%S")


def eumetsat_name_to_datetime(filename: str):
    date_str = filename.split("0100-0100-")[-1].split(".")[0]
    return datetime.datetime.strptime(date_str, "%Y%m%d%H%M%S")


def retrieve_pixel_value(geo_coord, data_source):
    """Return floating-point value that corresponds to given point.
    Taken from https://gis.stackexchange.com/questions/221292/retrieve-pixel-value-with-geographic-coordinate-as-input-with-gdal"""
    x, y = geo_coord[0], geo_coord[1]
    forward_transform = affine.Affine.from_gdal(*data_source.GetGeoTransform())
    reverse_transform = ~forward_transform
    px, py = reverse_transform * (x, y)
    px, py = int(px + 0.5), int(py + 0.5)
    pixel_coord = px, py

    data_array = np.array(data_source.GetRasterBand(1).ReadAsArray())
    return data_array[pixel_coord[0]][pixel_coord[1]]


def map_satellite_to_mercator(
    native_satellite=None,
    grib_files=None,
    bufr_files=None,
    bands=(
        "HRV",
        "IR_016",
        "IR_039",
        "IR_087",
        "IR_097",
        "IR_108",
        "IR_120",
        "IR_134",
        "VIS006",
        "VIS008",
        "WV_062",
        "WV_073",
    ),
    save_scene="geotiff",
    save_loc=None,
):
    """
    Opens, transforms to Transverse Mercator over Europe, and optionally saves it to files on disk.
    :param native_satellite:
    :param grib_files:
    :param bufr_files:
    :param bands:
    :param save_scene:
    :param save_loc: Save location
    :return:
    """
    if not _SAT_LIBS:
        raise EnvironmentError("Pyresample or Satpy are not installed, please install them first")
    areas = load_area("/home/bieker/Development/satflow/satflow/resources/areas.yaml")
    filenames = {}
    if native_satellite is not None:
        filenames["seviri_l1b_native"] = [native_satellite]
    if grib_files is not None:
        filenames["seviri_l2_grib"] = [grib_files]
    if bufr_files is not None:
        filenames["seviri_l2_bufr"] = [bufr_files]
    scene = Scene(filenames=filenames)
    scene.load(bands)
    # By default resamples to 3km, as thats the native resolution of all bands other than HRV
    scene = scene.resample(areas[0])
    if save_loc is not None:
        # Now the relvant data is all together, just need to save it somehow, or return it to the calling process
        scene.save_datasets(writer=save_scene, base_dir=save_loc, enhance=False)
    return scene


def create_time_layer(dt: datetime.datetime, shape):
    """Create 3 layer for current time of observation"""
    month = dt.month / 12
    day = dt.day / 31
    hour = dt.hour / 24
    # minute = dt.minute / 60
    return np.stack([np.full(shape, month), np.full(shape, day), np.full(shape, hour)], axis=-1)


def load_np(data):
    import numpy.lib.format

    stream = io.BytesIO(data)
    return numpy.lib.format.read_array(stream)


def binarize_mask(mask):
    """Binarize mask, taking max value as the data, and setting everything else to 0"""
    tmp_mask = np.zeros_like(mask)
    tmp_mask[np.isclose(np.round(mask), 2)] = 1
    return tmp_mask


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
    ret = np.stack([xx_channel, yy_channel], axis=0)

    if with_r:
        rr = np.sqrt(np.square(xx_channel - 0.5) + np.square(yy_channel - 0.5))
        ret = np.concatenate([ret, np.expand_dims(rr, axis=0)], axis=0)
    ret = np.moveaxis(ret, [1], [0])
    return ret


def check_channels(config: dict) -> int:
    """
    Checks the number of channels needed per timestep, to use for preallocating the numpy array
    Is not the same as the one for training, as that includes the number of channels after the array is partly
    flattened
    Args:
        config:

    Returns:

    """
    channels = len(config.get("bands", []))
    channels = channels + 1 if config.get("use_mask", False) else channels
    channels = (
        channels + 3
        if config.get("use_time", False) and not config.get("time_aux", False)
        else channels
    )
    # if config.get("time_as_channels", False):
    # Calc number of channels + inital ones
    #    channels = channels * (config["num_timesteps"] + 1)
    channels = channels + 1 if config.get("use_topo", False) else channels
    channels = channels + 3 if config.get("use_latlon", False) else channels
    channels = channels + 2 if config.get("add_pixel_coords", False) else channels
    channels = channels + 1 if config.get("add_polar_coords", False) else channels
    return channels


def crop_center(img: np.ndarray, cropx: int, cropy: int) -> np.ndarray:
    """Crops center of image through timestack, fails if all the images are concatenated as channels"""
    t, c, y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:, :, starty : starty + cropy, startx : startx + cropx]


def load_config(config_file):
    with open(config_file, "r") as cfg:
        return yaml.load(cfg, Loader=yaml.FullLoader)["config"]
