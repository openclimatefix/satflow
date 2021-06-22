import os

from satip import eumetsat
from satip.eumetsat import compress_downloaded_files

from satflow.data.utils import eumetsat_name_to_datetime

data_dir = "/run/media/bieker/Round1/EUMETSAT/"
debug_fp = "../logs/EUMETSAT_download.txt"
metadata_db_fp = "../data/EUMETSAT_metadata.db"


import shutil
from glob import glob

grb_files = glob(data_dir + "*.grb")

for f in grb_files:
    base_native_filename = os.path.basename(f)
    dt = eumetsat_name_to_datetime(base_native_filename)

    new_dst_path = os.path.join(data_dir, dt.strftime("%Y/%m/%d/%H/%M"))
    if not os.path.exists(new_dst_path):
        os.makedirs(new_dst_path)

    new_dst_full_filename = os.path.join(new_dst_path, base_native_filename + ".grb")

    if os.path.exists(new_dst_full_filename):
        os.remove(new_dst_full_filename)
    shutil.move(src=f, dst=new_dst_path)

exit()

dm = eumetsat.DownloadManager(user_key, user_secret, data_dir, metadata_db_fp, debug_fp)

start_date = "2020-01-01 08:00"
end_date = "2020-01-01 20:00"

for day in [2, 8, 15, 22, 27]:
    for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        for year in [2020, 2021]:
            dm.download_date_range(
                f"{year}-{month}-{day} 07:59",
                f"{year}-{month}-{day} 20:04",
                product_id="EO:EUM:DAT:MSG:RSS-CLM",
            )
            dm.download_date_range(
                f"{year}-{month}-{day} 08:00",
                f"{year}-{month}-{day} 20:00",
                product_id="EO:EUM:DAT:MSG:RII",
            )
            dm.download_date_range(
                f"{year}-{month}-{day} 07:59",
                f"{year}-{month}-{day} 20:04",
                product_id="EO:EUM:DAT:MSG:RSS-AMV",
            )
            dm.download_date_range(
                f"{year}-{month}-{day} 07:59",
                f"{year}-{month}-{day} 20:04",
                product_id="EO:EUM:DAT:MSG:RSS-MPE-GRIB",
            )
            dm.download_date_range(f"{year}-{month}-{day} 07:59", f"{year}-{month}-{day} 20:04")
            compress_downloaded_files(data_dir=data_dir, compressed_dir=data_dir)
