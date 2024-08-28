#!/usr/bin/env python3

"""
This module implement the calculation of cross correlation and stacking.
"""

import numpy as np
import geomath as geomath
from numba import jit, float64

class StackBins:
    def __init__(self, start, end, step):
        self.centers = np.arange(start, end+step*0.001, step)
        self.step = step
        self.edges = np.arange(start-step*0.5, end+step*0.501, step)
    def get_idx(self, v):
        return StackBins.stack_bin_index(v, self.centers[0], self.step)
    @staticmethod
    @jit(nopython=True, nogil=True)
    def stack_bin_index(distance, dist_min, dist_step):
        """
        Return the index given stacking oarameters
        """
        idx = int( round((distance-dist_min) / dist_step) )
        return idx

if __name__ == "__main__":
    bins = StackBins(0, 10, 1)
    print(bins.centers)
    print(bins.edges)
    print(bins.get_idx(2.5))
    print(bins.get_idx(3.5))
    print(bins.get_idx(4.5))
    print(bins.get_idx(5.5))
    print(bins.get_idx(6.5))