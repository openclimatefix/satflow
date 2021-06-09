import webdataset as wds
import rasterio
import numpy as np
import os
from datetime import datetime

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



