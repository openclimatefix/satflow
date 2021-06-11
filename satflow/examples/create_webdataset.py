import satpy
import webdataset as wds
import rasterio
import numpy as np
import os
from datetime import datetime
from rasterio.windows import Window, from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from satpy import Scene
from pyresample import load_area
from rasterio.plot import show


HRV_data = "/run/media/bieker/Round1/EUMETSAT/2020/02/21/13/14/HRV_20200221_131010.tif"
cloud_data = "/run/media/bieker/Round1/EUMETSAT/2020/02/21/13/14/cloud_mask_20200221_131500.tif"
dem_data = "/run/media/bieker/data/EUMETSAT/europe_all_dem_300m_nearest.tif"
test_data = "/home/bieker/Development/satflow/satflow/examples/test_crop.tif"
reproject_data = "/home/bieker/Development/satflow/satflow/examples/test_reproject.tif"

dem_d = rasterio.open(dem_data)
print(dem_d.bounds)
c_d = rasterio.open(cloud_data)
print(c_d.bounds)
print(c_d.read(1)[515:-641,603:].shape)
exit()

areas = load_area("/home/bieker/Development/satflow/satflow/examples/areas.yaml")
filenames = {"seviri_l1b_native": ["/run/media/bieker/Round1/EUMETSAT/2021/03/14/10/19/MSG3-SEVI-MSG15-0100-NA-20210314101915.098000000Z-NA.nat"]}
filenames["seviri_l2_grib"] = ["/run/media/bieker/Round1/EUMETSAT/2021/03/14/10/19/MSG3-SEVI-MSGCLMK-0100-0100-20210314102000.000000000Z-NA.grb"]
scene = Scene(filenames=filenames)
scene.load(('HRV', 'IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073', 'cloud_mask'))
# By default resamples to 3km, as thats the native resolution of all bands other than HRV
scene = scene.resample(areas[0])

hrv_long, hrv_lat = scene['HRV'].attrs['area'].get_lonlats()
c_long, c_lat = scene['cloud_mask'].attrs['area'].get_lonlats()
ir_long, ir_lat = scene['IR_097'].attrs['area'].get_lonlats()

hrv_long = hrv_long[515:-641,603:]
hrv_lat = hrv_lat[515:-641,603:]

data = scene['HRV'].values
print(data)
data = data[515:-641,603:]

import matplotlib.pyplot as plt
plt.imshow(data)
plt.show()

out_fp = "europe_all_dem_3km_cropbounds.tif"
out = rasterio.open(out_fp)
show(out, cmap='terrain')
dem_map = np.zeros(shape=hrv_long.shape)
data = out.read(1)
for i in range(hrv_long.shape[0]):
    map_dem_row, map_dem_col = rasterio.transform.rowcol(out.transform, hrv_long[i], hrv_lat[i])
    for x, y in zip(map_dem_row, map_dem_col):
        if dem_map.shape[0] > x >= 0 and dem_map.shape[1] > y >= 0:
            dem_map[x][y] = data[x][y]

plt.imshow(dem_map)
plt.show()

print(f"Long Lats HRV: {hrv_long.shape}, {hrv_lat.shape}, Max lat: {np.max(hrv_lat)} Max Long: {np.max(hrv_long)} Min lat: {np.min(hrv_lat)} Min Long: {np.min(hrv_long)}")
print(f"Long Lats Cloud: {c_long.shape}, {c_lat.shape}, Max Lat: {np.max(c_lat)} Max Long: {np.max(c_long)} Min Lat: {np.min(c_lat)} Min Lon: {np.min(c_long)}")
print(f"Long Lats IR: {ir_long.shape}, {ir_lat.shape}")

#cropped_scene = scene.crop(ll_bbox=(-45, 33, 65, 65))
#scene = scene.resample(areas[0])
scene.save_datasets(writer="geotiff")
exit()



"""
#scn = Scene(reader='generic_image', filenames=[dem_data])
#scn.load(['image'])
#areas = load_area("/home/bieker/Development/satflow/satflow/examples/areas.yaml")
#scene = scn.resample(areas[0])
#scene.save_datasets(writer='geotiff', filename="dem.tif")
#exit()

dem_data = "/home/bieker/Development/satflow/satflow/examples/dem.tif"
dem_src = rasterio.open(dem_data)
dem_window = dem_src.read(1)
import matplotlib.pyplot as plt
plt.imshow(dem_window)
plt.show()
exit()
reproj = rasterio.open(reproject_data)
re_d = reproj.read(1)
re_d = np.flipud(np.fliplr(re_d))[:,:1957]
plt.imshow(re_d, cmap='terrain')
plt.show()
exit()
"""
# Create WebDataset with each shard being a single day and all the images for that day
eumetsat_dir = "/run/media/bieker/Round1/EUMETSAT/"


shard_num = 0
prev_datetime = datetime.now()
sink = wds.TarWriter(f"satflow-first.tar", compress=True)
for root, dirs, files in os.walk(eumetsat_dir):
    # As its stored in year, month, day, hour, minute, format, once we are at the day level, create a new shard
    # Also cut down everything to only the square of all value data
    if files:
        # At lowest one, so have everything in it
        if len(files) < 11: # Not everything worked
            continue

        date_data = root.split("/")
        date_str = os.path.join(date_data[-5],date_data[-4],date_data[-3],date_data[-2],date_data[-1])
        datetime_object = datetime.strptime(date_str, "%Y/%m/%d/%H/%M")
        if prev_datetime.date() != datetime_object.date():
            # New shard
            sink.close()
            sink = wds.TarWriter(f"/run/media/bieker/data/EUMETSAT/satflow-{shard_num:05d}.tar", compress=True)
            shard_num += 1
            prev_datetime = datetime_object
        sample = {"__key__": datetime_object.strftime("%Y/%m/%d/%H/%M"),
                  }
        try:
            for f in files:
                if ".tif" in f:
                    # Only need geotiff ones
                    src = rasterio.open(os.path.join(root, f))
                    data = src.read(1)
                    cropped_data = data[515:-641,603:] # Done by handish to exclude all NODATA and invalid masks for clouds and images (clouds have a smaller extant)
                    channel = f.split("_")
                    if channel[0] in ["IR", "WV", "cloud"]:
                        channel = channel[0] + channel[1] # These are split across multiple
                    else:
                        channel = channel[0]
                    sample[channel + ".npy"] = cropped_data
            # Now all channels added to thing, write to shard
            sink.write(sample)
        except Exception as e:
            print(e)



