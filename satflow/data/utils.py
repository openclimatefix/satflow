import osgeo
import affine
import numpy as np


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

