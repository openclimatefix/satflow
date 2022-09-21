# SatFlow
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
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

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center"><a href="https://www.jacobbieker.com"><img src="https://avatars.githubusercontent.com/u/7170359?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jacob Bieker</b></sub></a><br /><a href="https://github.com/openclimatefix/satflow/commits?author=jacobbieker" title="Code">💻</a></td>
      <td align="center"><a href="https://lewtun.github.io/blog/"><img src="https://avatars.githubusercontent.com/u/26859204?v=4?s=100" width="100px;" alt=""/><br /><sub><b>lewtun</b></sub></a><br /><a href="https://github.com/openclimatefix/satflow/commits?author=lewtun" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!