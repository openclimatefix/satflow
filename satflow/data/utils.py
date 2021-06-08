import affine
import numpy as np
import re
import datetime
from satpy import Scene
from pyresample import load_area


def eumetsat_filename_to_datetime(inner_tar_name):
    """Takes a file from the EUMETSAT API and returns
    the date and time part of the filename"""

    p = re.compile('^MSG[23]-SEVI-MSG15-0100-NA-(\d*)\.')
    title_match = p.match(inner_tar_name)
    date_str = title_match.group(1)
    return datetime.datetime.strptime(date_str, "%Y%m%d%H%M%S")


def eumetsat_name_to_datetime(filename: str):
    date_str = filename.split("0100-0100-")[-1].split(".")[0]
    return datetime.datetime.strptime(date_str, "%Y%m%d%H%M%S")


def retrieve_pixel_value(geo_coord, data_source):
    """Return floating-point value that corresponds to given point.
    Taken from https://gis.stackexchange.com/questions/221292/retrieve-pixel-value-with-geographic-coordinate-as-input-with-gdal """
    x, y = geo_coord[0], geo_coord[1]
    forward_transform = \
        affine.Affine.from_gdal(*data_source.GetGeoTransform())
    reverse_transform = ~forward_transform
    px, py = reverse_transform * (x, y)
    px, py = int(px + 0.5), int(py + 0.5)
    pixel_coord = px, py

    data_array = np.array(data_source.GetRasterBand(1).ReadAsArray())
    return data_array[pixel_coord[0]][pixel_coord[1]]



def map_satellite_to_mercator(native_satellite, grib_files=None, bufr_files=None, bands=('HRV', 'IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073'), save_scene="geotiff"):
    areas = load_area("/home/bieker/Development/satflow/satflow/examples/areas.yaml")
    scene = Scene(
        filenames={"seviri_l1b_native": [native_satellite],
                   "seviri_l2_grib": [grib_files],
                   "seviri_l2_bufr": [bufr_files]})
    scene.load(bands)
    # By default resamples to 3km, as thats the native resolution of all bands other than HRV
    scene = scene.resample(areas[0])

    # Now the relvant data is all together, just need to save it somehow, or return it to the calling process
    scene.save_datasets(writer=save_scene)
    return scene



