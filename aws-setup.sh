#!/bin/bash
# Setup Miniconda
mkdir -p ~/miniconda3
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda3.sh
bash miniconda3.sh -b -p /conda
echo export PATH=/conda/bin:$PATH >> .bashrc
export PATH="/conda/bin:${PATH}"
rm miniconda3.sh
~/miniconda3/bin/conda init bash


# Get and setup SatFlow
git clone https://github.com/openclimatefix/satflow.git
cd satflow
conda env create -f environment.yml
conda activate satflow
pip install -r requirements.txt

echo "source activate satflow" >> ~/.bashrc
