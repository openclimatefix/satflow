import logging
import os
from datetime import datetime

import numpy as np
import rasterio
import satpy
import webdataset as wds
from pyresample import load_area
from rasterio.plot import show
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import Window, from_bounds
from satpy import Scene

logger = logging.getLogger("satflow")
logger.setLevel(logging.DEBUG)

#areas = load_area("/home/bieker/Development/satflow/satflow/resources/areas.yaml")
#filenames = {
#    "seviri_l1b_native": [
#        "/run/media/bieker/Round1/EUMETSAT/2021/03/14/10/19/MSG3-SEVI-MSG15-0100-NA-20210314101915.098000000Z-NA.nat"
#    ]
#}
#scene = Scene(filenames=filenames)
#scene.load(("HRV",))
# By default resamples to 3km, as thats the native resolution of all bands other than HRV
#scene = scene.resample(areas[0])
#hrv_long, hrv_lat = scene["HRV"].attrs["area"].get_lonlats()
#hrv_long = hrv_long[515:-641, 603:]
#hrv_lat = hrv_lat[515:-641, 603:]
#loc_x = np.cos(hrv_lat) * np.cos(hrv_long)
#loc_y = np.cos(hrv_lat) * np.sin(hrv_long)
#loc_z = np.sin(hrv_lat)

#location_array = np.stack([loc_x, loc_y, loc_z], axis=-1).astype(np.float32) # We aren't using float64 anyway, so some small overlap of locations should be fine?
#print(location_array.dtype)
#print(len(np.unique(location_array)))
#print(len(np.unique(location_array.astype(np.float32))))
#print(len(np.unique(location_array.astype(np.float16))))
#np.save("../resources/location_array.npy", location_array)
topo_data = np.load("../resources/cutdown_europe_dem.npy")
location_array = np.load("../resources/location_array.npy")
#print(topo_data.dtype)
#print(topo_data.shape)
#print(location_array.shape)
box_1 = (0, 446, 0, 978)
box_2 = (446, -1, 978, -1)
# Create WebDataset with each shard being a single day and all the images for that day
eumetsat_dir = "/run/media/bieker/Round1/EUMETSAT/"


def make_day(data, flow=True, batch=144, tile=True):
    root_dir, shard_num = data
    # reset optical flow samples
    flow_sample = {}
    # Initialize new flow sample
    date_data = root_dir.split("/")
    date_str = os.path.join(date_data[-3], date_data[-2], date_data[-1])
    overall_datetime = datetime.strptime(date_str, "%Y/%m/%d")
    batch_num = 0
    if batch < 1:
        flow_sample["__key__"] = overall_datetime.strftime("%Y/%m/%d")
    else:
        flow_sample["__key__"] = overall_datetime.strftime("%Y/%m/%d") + f"{batch_num}"
    if tile:
        # Split data into 8 chunks? 446x489 squareish areas
        samples = [{"__key__": overall_datetime.strftime("%Y/%m/%d") + f"_{i}", "time.pyd": []} for i in range(8)]
    flow_sample["time.pyd"] = []
    if flow:
        flow_sample["topo.npy"] = topo_data
        flow_sample["location.npy"] = location_array
    print(f"On shard: {shard_num} Date: {overall_datetime.strftime('%Y/%m/%d')}")
    shard_num += 1
    interday_frame = 0
    if os.path.exists(
        f"/run/media/bieker/data/EUMETSAT/satflow{'-' if not flow else '-flow'}{'-' if not flow and batch > 0 else f'-{batch}-'}{'tiled-' if tile else ''}{shard_num:05d}.tar"
    ):
        return
    sink_flow = wds.TarWriter(
        f"/run/media/bieker/data/EUMETSAT/satflow{'-' if not flow else '-flow'}{'-' if not flow and batch > 0 else f'-{batch}-'}{'tiled-' if tile else ''}{shard_num:05d}.tar",
        compress=True,
    )
    for root, dirs, files in os.walk(root_dir):
        dirs.sort(
            key=int
        )  # Ensure this is done in sorted order, so its always from earliest to latest
        # As its stored in year, month, day, hour, minute, format, once we are at the day level, create a new shard
        # Also cut down everything to only the square of all value data
        if files:
            # At lowest one, so have everything in it
            if len(files) < 11:  # Not everything worked
                continue

            date_data = root.split("/")
            date_str = os.path.join(
                date_data[-5], date_data[-4], date_data[-3], date_data[-2], date_data[-1]
            )
            datetime_object = datetime.strptime(date_str, "%Y/%m/%d/%H/%M")
            if not flow:
                flow_sample = {}
                flow_sample["__key__"] = datetime_object.strftime("%Y/%m/%d/%H/%M")

            try:
                # Want this at the top, as if this extraction fails, we still then know if there is a missing frame
                interday_frame += 1
                for f in files:
                    if ".tif" in f:
                        # Only need geotiff ones
                        src = rasterio.open(os.path.join(root, f))
                        data = src.read(1)
                        cropped_data = data[
                            515:-641, 603:
                        ]  # Done by handish to exclude all NODATA and invalid masks for clouds and images (clouds have a smaller extant)
                        channel = f.split("_")
                        from scipy import stats
                        print(f"Channel: {channel}")
                        print(stats.describe(cropped_data))
                        if channel[0] in ["IR", "WV", "cloud"]:
                            channel = channel[0] + channel[1]  # These are split across multiple
                        else:
                            channel = channel[0]
                        if tile:
                            height = (0,446,-1)
                            width = (0,490,980,1470,-1)
                            s_num = 0
                            for j in range(4):
                                for i in range(2):
                                    tile_cropped_data = cropped_data[height[i]:height[i+1], width[j]:width[j+1]]
                                    tiled_topo = topo_data[height[i]:height[i+1], width[j]:width[j+1]]
                                    tiled_loc = location_array[height[i]:height[i+1], width[j]:width[j+1]]
                                    samples[s_num][channel + f".{interday_frame:03d}.npy"] = tile_cropped_data
                                    samples[s_num]['location.npy'] = tiled_loc
                                    samples[s_num]['topo.npy'] = tiled_topo
                                    s_num += 1
                        if not flow:
                            flow_sample[channel + ".npy"] = cropped_data
                        else:
                            flow_sample[channel + f".{interday_frame:03d}.npy"] = cropped_data
                # Now all channels added to thing, write to shard
                if flow:
                    flow_sample["time.pyd"].append(datetime_object)
                    samples = [s["time.pyd"].append(datetime_object) for s in samples]
                    if batch > 0:
                        print(
                            f"In Batch: {len(flow_sample['time.pyd'])} == {batch} Shard: {shard_num}"
                        )
                        if len(flow_sample["time.pyd"]) == batch:
                            if tile:
                                for s in samples:
                                    sink_flow.write(s)
                            else:
                                sink_flow.write(flow_sample)
                                flow_sample["__key__"] = datetime_object.strftime("%Y/%m/%d/%H/%M")
                                flow_sample["topo.npy"] = topo_data
                                flow_sample["location.npy"] = location_array
                                flow_sample["time.pyd"] = []
                            interday_frame = 0
                            flow_sample = {}
                            batch_num += 1
                else:
                    flow_sample["time.pyd"] = datetime_object
                    sink_flow.write(flow_sample)
            except Exception as e:
                print(e)
    # print(f"Write Sample: {flow_sample.keys()}")
    if flow and batch <= 0:
        sink_flow.write(flow_sample)
    # Close old shard
    # sink.close()
    sink_flow.close()


import multiprocessing

pool = multiprocessing.Pool(4)

old = os.listdir(os.path.join(eumetsat_dir, "2020"))
old.sort(key=int)
old = [os.path.join(eumetsat_dir, "2020", d) for d in old]
tmp_old = []
for d in old:
    days = os.listdir(os.path.join(d))
    days.sort(key=int)
    tmp_old += [os.path.join(d, day) for day in days]
old = tmp_old
new = os.listdir(os.path.join(eumetsat_dir, "2021"))
new.sort(key=int)
new = [os.path.join(eumetsat_dir, "2021", d) for d in new]
tmp_new = []
for d in new:
    days = os.listdir(os.path.join(d))
    days.sort(key=int)
    tmp_new += [os.path.join(d, day) for day in days]
new = tmp_new
all_dates = old + new
print(all_dates)
all_dates = zip(all_dates, range(len(all_dates)))
pool.map(make_day, all_dates)
exit()
for data in all_dates:
    make_day(data, flow=True, batch=12)
    exit()
