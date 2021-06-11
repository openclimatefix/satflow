from satflow.data.utils import map_satellite_to_mercator
from glob import glob
import os
import subprocess

eumetsat_dir = "/run/media/bieker/Round1/EUMETSAT/"

def decompress(full_bzip_filename: str) -> str:
    base_bzip_filename = os.path.basename(full_bzip_filename)
    base_nat_filename = os.path.splitext(base_bzip_filename)[0]
    full_nat_filename = os.path.join("./", base_nat_filename)
    if os.path.exists(full_nat_filename):
        os.remove(full_nat_filename)
    with open(full_nat_filename, 'wb') as nat_file_handler:
        process = subprocess.run(
            ['pbzip2', '--decompress', '--keep', '--stdout', full_bzip_filename],
            stdout=nat_file_handler)
    process.check_returncode()
    return full_nat_filename

for root, dirs, files in os.walk(eumetsat_dir):
    #For each of these, load cloud mask, etc. and save to GeoTIFF
    sat_file = ""
    cloud_mask = ""
    for f in files:
        if ".bz2" in f:
            sat_file = decompress(os.path.join(root, f))
        elif ".nat" in f:
            sat_file = os.path.join(root, f)
        if ".grb" in f:
            cloud_mask = os.path.join(root, f)
        if ".tif" in f: # Already processed
            break
        if sat_file and cloud_mask:
            map_satellite_to_mercator(sat_file, cloud_mask, save_loc=root, bands=('HRV', 'IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073', 'cloud_mask'))
            os.remove(sat_file)