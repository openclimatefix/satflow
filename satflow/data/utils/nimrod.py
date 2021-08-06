import iris
import os
from glob import glob


# Load the nimrod file
data_cube = iris.fileformats.nimrod.load_cubes("test.nimrod")
