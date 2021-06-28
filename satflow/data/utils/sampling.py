import numpy as np
OCEAN = 0
MOUNTAIN = 2000
HILLY = 100


def calc_geo(topo_map: np.ndarray):
    ocean_frac = len(topo_map[topo_map <= OCEAN])
    hill_frac = len(topo_map[(topo_map > OCEAN) & (topo_map <= HILLY)])
    mt_frac = len(topo_map[(topo_map > HILLY) & (topo_map <= MOUNTAIN)])
