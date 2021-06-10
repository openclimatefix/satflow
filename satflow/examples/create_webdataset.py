import webdataset as wds
import rasterio
import numpy as np
import os
from datetime import datetime
from rasterio.windows import Window, from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling

HRV_data = "/run/media/bieker/Round1/EUMETSAT/2020/02/21/13/14/HRV_20200221_131010.tif"
cloud_data = "/run/media/bieker/Round1/EUMETSAT/2020/02/21/13/14/cloud_mask_20200221_131500.tif"
dem_data = "/run/media/bieker/data/EUMETSAT/europe_all_dem_300m_nearest.tif"
test_data = "/home/bieker/Development/satflow/satflow/examples/test_crop.tif"
reproject_data = "/home/bieker/Development/satflow/satflow/examples/test_reproject.tif"
t_src = rasterio.open(test_data)
print(t_src.bounds)
print(t_src.height)
print(t_src.width)

dst_crs = 'EPSG:26915'
dem_src = rasterio.open(dem_data)
transform, width, height = calculate_default_transform(
    dem_src.crs, dst_crs, dem_src.width, dem_src.height, *dem_src.bounds)
kwargs = dem_src.meta.copy()
kwargs.update({
    'crs': dst_crs,
    'transform': transform,
    'width': width,
    'height': height
})
print(width)
print(height)
print(transform)
dem_window = from_bounds(left=t_src.bounds.left, right=t_src.bounds.right,
                         bottom=t_src.bounds.bottom, top=t_src.bounds.top, transform=transform,
                         height=height, width=width)

dem_window = dem_src.read(window=dem_window)
import matplotlib.pyplot as plt
plt.imshow(dem_window[0])
plt.show()
reproj = rasterio.open(reproject_data)
re_d = reproj.read(1)
re_d = np.flipud(np.fliplr(re_d))[:,:1957]
plt.imshow(re_d, cmap='terrain')
plt.show()
exit()

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



