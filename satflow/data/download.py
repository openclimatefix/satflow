import datetime
import os

from satip import eumetsat
from satip.eumetsat import compress_downloaded_files

from satflow.data.utils.utils import eumetsat_name_to_datetime, eumetsat_filename_to_datetime
import logging
import os
from datetime import datetime, timedelta

import numpy as np
import rasterio
import satpy
import webdataset as wds
from pyresample import load_area

logger = logging.getLogger("satflow")
logger.setLevel(logging.DEBUG)

#
areas = load_area("/home/bieker/Development/satflow/satflow/resources/areas.yaml")
topo_data = np.load("../resources/cutdown_europe_dem.npy")
topo_data[topo_data < -100] = 0  # Ocean
location_array = np.load("../resources/location_array.npy")
box_1 = (0, 446, 0, 978)
box_2 = (446, -1, 978, -1)

data_dir = "/run/media/bieker/Round1/EUMETSAT/"
debug_fp = "../logs/EUMETSAT_download.txt"
metadata_db_fp = "../data/EUMETSAT_metadata.db"
from glob import glob

grb_files = glob(data_dir + "*.grb")

dm = eumetsat.DownloadManager(user_key, user_secret, data_dir, metadata_db_fp, debug_fp)

start_date = "2020-01-01 08:00"
end_date = "2020-01-01 20:00"
from satflow.data.utils.utils import map_satellite_to_mercator


def make_day(data, flow=True, batch=146, tile=True):
    root_dir, sat_date, shard_num = data
    shard_num += 150  # Start after the current ones
    # reset optical flow samples
    flow_sample = {}
    # Initialize new flow sample
    overall_datetime = sat_date
    shard_num = overall_datetime.strftime("%Y%m%d")
    batch_num = 0
    if batch < 1:
        flow_sample["__key__"] = overall_datetime.strftime("%Y/%m/%d")
    else:
        flow_sample["__key__"] = overall_datetime.strftime("%Y/%m/%d") + f"{batch_num}"
    # Split data into 8 chunks? 446x489 squareish areas
    samples = [
        {"__key__": overall_datetime.strftime("%Y/%m/%d") + f"_{i}", "time.pyd": []}
        for i in range(8)
    ]
    flow_sample["time.pyd"] = []
    if flow:
        flow_sample["topo.npy"] = topo_data
        flow_sample["location.npy"] = location_array
    print(f"On shard: {shard_num} Date: {overall_datetime.strftime('%Y/%m/%d')}")
    interday_frame = 0
    if os.path.exists(
        f"/run/media/bieker/data/EUMETSAT/satflow{'-' if not flow else '-flow'}{'-' if not flow and batch > 0 else f'-{batch}-'}{'tiled-' if tile else ''}{shard_num}.tar"
    ):
        return
    sink_flow = wds.TarWriter(
        f"/run/media/bieker/data/EUMETSAT/satflow{'-' if not flow else '-flow'}{'-' if not flow and batch > 0 else f'-{batch}-'}{'tiled-' if tile else ''}{shard_num}.tar",
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

            # try:
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
                    cropped_data = cropped_data.astype(np.float16)
                    if channel[0] in ["IR", "WV", "cloud"]:
                        channel = channel[0] + channel[1]  # These are split across multiple
                    else:
                        channel = channel[0]
                    if tile:
                        height = (0, 446, -1)
                        width = (0, 490, 980, 1470, -1)
                        s_num = 0
                        for j in range(4):
                            for i in range(2):
                                tile_cropped_data = cropped_data[
                                    height[i] : height[i + 1], width[j] : width[j + 1]
                                ]
                                tiled_topo = topo_data[
                                    height[i] : height[i + 1], width[j] : width[j + 1]
                                ]
                                tiled_loc = location_array[
                                    height[i] : height[i + 1], width[j] : width[j + 1]
                                ]
                                samples[s_num][
                                    channel + f".{interday_frame:03d}.npy"
                                ] = tile_cropped_data.astype(np.float16)
                                samples[s_num]["location.npy"] = tiled_loc
                                samples[s_num]["topo.npy"] = tiled_topo
                                s_num += 1

                    elif not flow:
                        flow_sample[channel + ".npy"] = cropped_data
                    else:
                        flow_sample[channel + f".{interday_frame:03d}.npy"] = cropped_data.astype(
                            np.float16
                        )
            # Now all channels added to thing, write to shard
            if flow:
                if not tile:
                    flow_sample["time.pyd"].append(datetime_object)
                for i, s in enumerate(samples):
                    samples[i]["time.pyd"].append(datetime_object)
                if batch > 0:
                    print(f"In Batch: {len(samples[0]['time.pyd'])} == {batch} Shard: {shard_num}")
                    if len(samples[0]["time.pyd"]) == batch:
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
                        samples = [
                            {
                                "__key__": overall_datetime.strftime("%Y/%m/%d") + f"_{i}",
                                "time.pyd": [],
                            }
                            for i in range(8)
                        ]
                        batch_num += 1
            else:
                flow_sample["time.pyd"] = datetime_object
                sink_flow.write(flow_sample)
    if tile:
        for s in samples:
            sink_flow.write(s)
    sink_flow.close()


import shutil

for day in range(15, 28):
    for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        for year in [2020]:
            curr_date = f"{year}/{month}/{day}"
            curr_date = datetime.strptime(curr_date, "%Y/%m/%d")
            # make_day((os.path.join(data_dir, curr_date.strftime("%Y/%m/%d/")), curr_date, 0))
            # continue
            dm.download_date_range(
                f"{year}-{month}-{day} 07:59",
                f"{year}-{month}-{day} 20:05",
                product_id="EO:EUM:DAT:MSG:RSS-CLM",
            )
            dm.download_date_range(
                f"{year}-{month}-{day} 07:59",
                f"{year}-{month}-{day} 20:05",
                product_id="EO:EUM:DAT:MSG:RII",
            )
            dm.download_date_range(
                f"{year}-{month}-{day} 07:59",
                f"{year}-{month}-{day} 20:05",
                product_id="EO:EUM:DAT:MSG:RSS-AMV",
            )
            dm.download_date_range(
                f"{year}-{month}-{day} 07:59",
                f"{year}-{month}-{day} 20:05",
                product_id="EO:EUM:DAT:MSG:RSS-MPE-GRIB",
            )
            dm.download_date_range(f"{year}-{month}-{day} 07:59", f"{year}-{month}-{day} 20:05")
            # Now go through and create the webdataset for the day
            nat_files = sorted(glob(data_dir + "*.nat"))

            for n in nat_files:
                base_native_filename = os.path.basename(n)
                dt = eumetsat_filename_to_datetime(base_native_filename)
                if dt.date() != curr_date.date():
                    # New date, so theoretically all done for the day
                    continue
                grb_files = sorted(glob(data_dir + "*.grb"))
                for grb in grb_files:
                    base_grb_filename = os.path.basename(grb)
                    grb_dt = eumetsat_name_to_datetime(base_grb_filename) - timedelta(minutes=1)
                    if (
                        grb_dt.date() == dt.date()
                        and grb_dt.date() == curr_date.date()
                        and grb_dt.hour == dt.hour
                        and grb_dt.minute == dt.minute
                    ):
                        # Have the data to create webdataset
                        map_satellite_to_mercator(
                            native_satellite=n,
                            grib_files=grb,
                            save_loc=os.path.join(data_dir, dt.strftime("%Y/%m/%d/%H/%M")),
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
                                "cloud_mask",
                            ),
                        )
                        os.remove(grb)
                        os.remove(n)
            make_day((os.path.join(data_dir, curr_date.strftime("%Y/%m/%d/")), curr_date, 0))
            # Clean up directory that isn't needed anymore
            # shutil.rmtree(os.path.join(data_dir, dt.strftime("%Y/%m/%d/")))
