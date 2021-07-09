# SatFlow
***Sat***ellite Optical ***Flow*** with machine learning models.

The goal of this repo is to improve upon optical flow models for predicting
future satellite images from current and past ones, focused primarily on EUMETSAT data.

## Installation

Clone the repository, then run
```shell
conda env create -f environment.yml
conda activate satflow
pip install -e .
````

Alternatively, you can also install a usually older version through ```pip install satflow```

## Data

The data used here is a combination of the UK Met Office's rainfall radar data, EUMETSAT MSG
satellite data (12 channels), derived data from the MSG satellites (cloud masks, etc.), and
numerical weather prediction data. Currently, some example transformed EUMETSAT data can be downloaded
from the tagged release, as well as included under ```datasets/```.
