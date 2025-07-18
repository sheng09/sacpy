#!/usr/bin/env python3

"""
This module implement the calculation of cross correlation and stacking.
"""
from mpi4py import MPI
import numpy as np
from numba import jit
FLAG_PYFFTW_USED = False
try:
    import pyfftw
    import pyfftw.interfaces.cache as pyffftw_interfaces_cache
    from pyfftw.interfaces.scipy_fft import rfft, irfft
    FLAG_PYFFTW_USED = True
except:
    from scipy.fft import rfft, irfft
    FLAG_PYFFTW_USED = False
from time import time as time_time
from h5py import File as h5_File
import matplotlib.pyplot as plt
import sys, traceback, os, os.path

from .geomath import haversine_rad, azimuth_rad, point_distance_to_great_circle_plane_rad, great_circle_plane_center_triple_rad
from .utils import Logger, Timer, TimeSummary
from .processing import fwhiten_f32, tnorm_f32, iirfilter_f32, taper, get_rfft_spectra_bound

class StackBins:
    """
    Class to define the evenly distributed and continuous stacking bins w.r.t. distances.
    #
    Parameters:
        `centers`, `step` and `edges`: for the stacking bins.
    #
    Methods:
        `get_idx`: get the index of the bin given the distance.
    #
    Example:
    >>> bins = StackBins(0, 180, 1)
    >>> print(bins.centers)
    >>> print(bins.edges)
    >>> idx = bins.get_idx(2.5))
    """
    def __init__(self, start, end, step):
        """
        Init the stacking bins.
        start, end, step: the start, end and step of the stacking bins.
                          the start declare the center of the first bin.
                          the end will be adjusted if (end-start) cannot be exactly divisible by step.
        """
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
        idx = np.round((distance-dist_min) / dist_step).astype(np.int32)
        return idx


HALF_PI = np.pi*0.5 # 90 degree
PI = np.pi          # 180 degree
TWO_PI = np.pi*2.0  # 360 degree
@jit(nopython=True, nogil=True)
def zne2znert(zne_mat, znert_mat, stlo_rad, stla_rad, evlo_rad, evla_rad):
    """
    Convert the ZNE matrix to ZNERT matrix. The RT here refers to event-receiver R and T.
    In other words, the input matrix has rows of ZNEZNEZNE... and the output matrix has rows of ZNERTZNERTZNERT...
    #
    Note: (1) the `znert_mat` should be allocated by users before running this function.
          (2) the `znert_mat` should have the number of rows: zne_mat's nrow/3*5.
                znert_mat.nrow = zne_mat.nrow/3*5
          (3) the `znert_mat` should have the number of columns equal or greater than zne_mat's ncol.
                znert_mat.ncol >= zne_mat.ncol
              zeros will be padded to the right for `znert_mat` if znert_mat.ncol > zne_mat.ncol.
                znert_mat[:, znert_mat.ncol: ] = 0
    #
    Parameters:
        zne_mat:   the input ZNE matrix with rows of ZNEZNEZNE...
        znert_mat: the ouput ZNERT matrix with rows of ZNERTZNERTZNERT...
        stlo_rad: an array for the station longitudes in radian
        stla_rad: an array for the station latitudes in radian
        evlo_rad: an array or a single value for the event longitude in radian
        evla_rad: an array or a single value for the event latitude in radian
    """
    baz = azimuth_rad(stlo_rad, stla_rad, evlo_rad, evla_rad)
    ang = -HALF_PI-baz # baz to anti-clockwise angle from east
    cs, ss = np.cos(ang), np.sin(ang)
    nsta = zne_mat.shape[0]//3
    ncol = zne_mat.shape[1]
    for ista in range(nsta):
        c, s = cs[ista], ss[ista]
        z = zne_mat[ista*3]
        n = zne_mat[ista*3+1]
        e = zne_mat[ista*3+2]
        #
        znert_mat[ista*5,   :ncol] = z
        znert_mat[ista*5+1, :ncol] = n
        znert_mat[ista*5+2, :ncol] = e
        znert_mat[ista*5+3, :ncol] = e*c + n*s
        znert_mat[ista*5+4, :ncol] = e*s - n*c

@jit(nopython=True, nogil=True)
def gcd_selection(n, lo_rad, la_rad, ptlo_rad, ptla_rad, gcd_range_rad, gcd_selection_mat_nn):
    """
    Compute a NxN matrix with values of 1 or 0 for selecting the pairs of stations or not. 1 for selecting and 0 for discarding.
    The ijth component of the matrix is 1 or 0 to select or discard the pair formed by the ith and jth records.
    The returned matrix is symmetric.
    #
    Parameters:
        n:        the length of lo_rad or la_rad used for computation.
        lo_rad:   an ndarray for the longitudes in radian.
                  This means the station longitudes for inter-station correlations,
                  and the event longitudes for inter-source correlations.
        la_rad:   an ndarray for the latitudes in radian.
                  This means the station latitudes for inter-station correlations,
                  and the event latitudes for inter-source correlations.
        ptlo_rad: a single value for the longitude of the point in radian.
                  This means the event longitude for inter-station correlations,
                  and the station longitude for inter-source correlations.
        ptla_rad: a single value for the latitude of the point in radian
                  This means the event latitude for inter-station correlations,
                  and the station latitude for inter-source correlations.
        gcd_range_rad: the minimum and maximum great circle distance in radian
        gcd_selection_mat_nn: the output selection matrix for gcd-based selection
    """
    dont_select_gcd = (gcd_range_rad.size == 0)
    if dont_select_gcd:
        gcd_selection_mat_nn.fill(1)
        return
    #### select gcd
    gcd_selection_mat_nn.fill(0)
    single_row = np.zeros(n, dtype=gcd_selection_mat_nn.dtype)
    ####
    gcd_range_rad = np.reshape(gcd_range_rad, (-1, 2) )
    for ista1 in range(n-1):
        lo1, la1 = lo_rad[ista1], la_rad[ista1]
        ista2 = ista1+1
        gcd = np.abs( point_distance_to_great_circle_plane_rad(ptlo_rad, ptla_rad, lo1, la1, lo_rad[ista2:], la_rad[ista2:] ) )
        single_row.fill(0)
        for (gcd_min, gcd_max) in gcd_range_rad:
            inside = (gcd_min<=gcd) & (gcd<=gcd_max)
            single_row[ista2:] |= inside      # the value before ista2 is always zero!
        gcd_selection_mat_nn[ista1] = single_row # second, modify each row. The lower triangle including the diagonal is always zero
    np.fill_diagonal(gcd_selection_mat_nn, 1) # the diagonal (auto correlation) is always zero
    gcd_selection_mat_nn |= gcd_selection_mat_nn.T # make it symmetric
@jit(nopython=True, nogil=True)
def daz_selection(n, lo_rad, la_rad, ptlo_rad, ptla_rad, daz_range_rad, daz_selection_mat_nn):
    """
    Compute a NxN matrix with values of 1 or 0 for selecting the pairs of stations or not. 1 for selecting and 0 for discarding.
    The ijth component of the matrix is 1 or 0 to select or discard the pair formed by the ith and jth records.
    The returned matrix is symmetric.
    #
    Parameters:
        n:        the length of lo_rad or la_rad used for computation.
        lo_rad:   an ndarray for the longitudes in radian.
                  This means the station longitudes for inter-station correlations,
                  and the event longitudes for inter-source correlations.
        la_rad:   an ndarray for the latitudes in radian.
                  This means the station latitudes for inter-station correlations,
                  and the event latitudes for inter-source correlations.
        ptlo_rad: a single value for the longitude of the point in radian.
                  This means the event longitude for inter-station correlations,
                  and the station longitude for inter-source correlations.
        ptla_rad: a single value for the latitude of the point in radian
                  This means the event latitude for inter-station correlations,
                  and the station latitude for inter-source correlations.
        daz_range_rad: the minimum and maximum azimuth difference in radian
        daz_selection_mat_nn: the output selection matrix
    """
    dont_select_daz = (daz_range_rad.size == 0)
    if dont_select_daz:
        daz_selection_mat_nn.fill(1)
        return
    #### select daz
    daz_selection_mat_nn.fill(0)
    single_row = np.zeros(n, dtype=daz_selection_mat_nn.dtype)
    ####
    daz_range_rad = np.reshape(daz_range_rad, (-1, 2) )
    az = azimuth_rad(ptlo_rad, ptla_rad, lo_rad, la_rad)
    for ista1 in range(n-1):
        az1 = az[ista1]
        ista2 = ista1+1
        daz = np.abs( az1-az[ista2:] ) % PI
        daz = np.minimum(daz, PI-daz)
        single_row.fill(0)
        for (daz_min, daz_max) in daz_range_rad:
            inside = (daz_min<=daz) & (daz<=daz_max)
            single_row[ista2:] |= inside      # the value before ista2 is always zero!
        daz_selection_mat_nn[ista1] = single_row # third, modify each row. The lower triangle including the diagonal is always zero
    ###
    np.fill_diagonal(daz_selection_mat_nn, 1) # the diagonal (auto correlation) is always zero
    daz_selection_mat_nn |= daz_selection_mat_nn.T # make it symmetric
@jit(nopython=True, nogil=True)
def gc_center_selection(n, lo_rad, la_rad, ptlo_rad, ptla_rad,
                        gc_center_rect_boxes_rad, gc_center_circles_rad,
                        gc_center_selection_mat_nn, critical_distance_rad=0.00174):
    """
    Compute a NxN matrix with values of 1 or 0 for selecting the pairs of stations or not given
    rectangle boxes and circle areas for the selection.
    A correlation pair will be discarded if it is discarded through either the rectangle-box
    selection or the circle-area selection.
    In other word, a pair is only selected if it is selected through both the rectangle-box
    selection and the circle-area selection!!!
    #
    Parameters:
        n:        the length of lo_rad or la_rad used for computation.
        lo_rad:   an ndarray for the longitudes in radian.
                  This means the station longitudes for inter-station correlations,
                  and the event longitudes for inter-source correlations.
        la_rad:   an ndarray for the latitudes in radian.
                  This means the station latitudes for inter-station correlations,
                  and the event latitudes for inter-source correlations.
        ptlo_rad: a single value for the longitude of the point in radian.
                  This means the event longitude for inter-station correlations,
                  and the station longitude for inter-source correlations.
        ptla_rad: a single value for the latitude of the point in radian
                  This means the event latitude for inter-station correlations,
                  and the station latitude for inter-source correlations.
        gc_center_rect_boxes_rad: a 2D matrix, each row of which corresponds to a rectangle boxes for selecting correlation
                                  pairs. Each row is (lon_min_rad, lon_max_rad, lat_min_rad, lat_max_rad).
                                  #
                                  Also, a flattend 1D array can be used so that every 4 elements define a rectangle box.
                                  In other words, `x = np.reshape(x, (-1, 4))` will be called to reshape the input.
                                  #
                                  Also, an zero-length array can be used to disable the rectangle box selection.
                                  `gc_center_rect_boxes_rad = np.array([] )` or
                                  `gc_center_rect_boxes_rad = np.empty((0, 4))` works.
                                  #
                                  We will compute the center of the great circle path for each pair of stations.
                                  For a pair, if the center is not inside any rectangle boxes, then we discard it.
                                  #
                                  The center of the great circle path is defined by a (longitude, latitude) on the
                                  northern sphere surface.
                                  For example, the center for the equator is the north pole (long=0, lat=90).
                                               the center for the Prime meridian is (lon=90, lat=0).
                                  #
                                  A rectangle box is on the sphere surface, and it is defined by
                                  (lon_min_rad, lon_max_rad, lat_min_rad, lat_max_rad).
                                  Please make sure the lon_min_rad is in the range of [0, 2PI].
        gc_center_circles_rad:    a 2D matrix, each row if which corresponds to a circle areas for selecting correlation
                                  pairs. Each row is (lon_center_rad, lat_center_rad, radius_rad).
                                  #
                                  Also, a flattend 1D array can be used so that every 3 elements define a circle area.
                                  In other words, `x = np.reshape(x, (-1, 3))` will be called to reshape the input.
                                  #
                                  Also, an zero-length array can be used to disable the circle area selection.
                                  `gc_center_circles_rad = np.array([] )` or
                                  `gc_center_circles_rad = np.empty((0, 3))` works.
                                  #
                                  We will compute the center of the great circle path for each pair of stations.
                                  For a pair, if the center is not inside any circle areas, then we discard it.
                                  #
                                  The center of the great circle path is defined by a (longitude, latitude) on the
                                  northern sphere surface.
                                  For example, the center for the equator is the north pole (long=0, lat=90).
                                               the center for the Prime meridian is (lon=90, lat=0).
                                  #
                                  A circle area is on the sphere surface, and it is defined by
                                  a center point (lon_center_rad, lat_center_rad) and a radius_rad.
                                  Please make sure the lon_center_rad is in the range of [0, 2PI].
        gc_center_selection_mat_nn: the buffer to output selection matrix. The matrix will be modified in-place.
                                    It is an NxN with values of 1 or 0 for selecting the pairs of stations or not.
                                    1 for selecting and 0 for discarding. The ijth component of the matrix is
                                    1 or 0 to select or discard the pair formed by the ith and jth records.
                                    The matrix is symmetric!
        critical_distance_rad:  the critical gc distance in radian to determine if two locations are the same.
                                # 0.00174 rad is about 0.1 degree
    """
    ####
    dont_select_rect   = (gc_center_rect_boxes_rad.size == 0)
    dont_select_circle = (gc_center_circles_rad.size    == 0)
    if dont_select_rect and dont_select_circle:
        gc_center_selection_mat_nn.fill(1)
        return
    #### in case of flattend arrays or zero-length arrays
    gc_center_rect_boxes_rad = np.reshape(gc_center_rect_boxes_rad, (-1, 4) )
    gc_center_circles_rad    = np.reshape(gc_center_circles_rad,    (-1, 3) )
    ####
    gc_center_selection_mat_nn.fill(0)
    rect_mat_row   = np.zeros(n, dtype=gc_center_selection_mat_nn.dtype)
    circle_mat_row = np.zeros(n, dtype=gc_center_selection_mat_nn.dtype)
    for ista1 in range(n):
        lo1, la1 = lo_rad[ista1], la_rad[ista1]
        ##### compute the clo, cla for a list of correlation pairs
        (clos, clas), junk = great_circle_plane_center_triple_rad(lo1, la1, lo_rad[ista1:], la_rad[ista1:],
                                                                  ptlo_rad, ptla_rad, critical_distance=critical_distance_rad)
                                                                  # 0.00174 rad is about 0.1 degree
        #### check with the rectangle boxes
        rect_mat_row.fill(0)
        for (lon_min, lon_max, lat_min, lat_max) in gc_center_rect_boxes_rad:
            inside = ~( (clos < lon_min) | (lon_max < clos) | (clas < lat_min) | (lat_max < clas) )
            #inside = ((lon_min <= clos) & (clos <= lon_max) & (lat_min <= clas) & (clas <= lat_max))
            rect_mat_row[ista1:] |= inside
        #### check with the circle areas
        circle_mat_row.fill(0)
        for (lon_center, lat_center, radius) in gc_center_circles_rad:
            tmp = haversine_rad(lon_center, lat_center, clos, clas)
            inside = (tmp <= radius)
            circle_mat_row[ista1:] |= inside
        #### don't commit select if no rectangle boxes or circle areas are provided
        if dont_select_rect:
            rect_mat_row.fill(1)
        if dont_select_circle:
            circle_mat_row.fill(1)
        ####
        rect_mat_row &= circle_mat_row
        gc_center_selection_mat_nn[ista1, ista1:] = rect_mat_row[ista1:] # only modify the upper triangle including the diagonal
    #### make it symmetric
    gc_center_selection_mat_nn |= gc_center_selection_mat_nn.T
__stack_bin_index = StackBins.stack_bin_index
@jit(nopython=True, nogil=True)
def get_stack_index_mat(n, lo_rad, la_rad, dist_min_deg, dist_step_deg, stack_index_mat_nn):
    """
    Compute  a NxN matrix which contains the index for each correlation pair.
    The ijth component of the matrix is the stacking index for the pair formed by the ith and jth records.
    The returen matrix is symmetric.
    #
    Parameters:
        n: the length of lo_rad or la_rad used for computation.
        lo_rad:        an ndarray for the longitudes in radian.
                       This means the station longitudes for inter-station correlations,
                       and the event longitudes for inter-source correlations.
        la_rad:        an ndarray for the latitudes in radian.
                       This means the station latitudes for inter-station correlations,
                       and the event latitudes for inter-source correlations.
        dist_min_deg:  the minimum distance in degree.
                       This corresponds to the inter-station distance for inter-station correlations,
                       and the inter-source distance for inter-source correlations.
        dist_step_deg: the step of the distance in degree.
                       This corresponds to the inter-station distance for inter-station correlations,
                       and the inter-source distance for inter-source correlations.
        stack_index_mat_nn: the output stack index matrix
    """
    for ista1 in range(n):
        lo1, la1 = lo_rad[ista1], la_rad[ista1]
        inter_rcv_dist = np.rad2deg( haversine_rad(lo1, la1, lo_rad[ista1:], la_rad[ista1:]) )
        istack = __stack_bin_index(inter_rcv_dist, dist_min_deg, dist_step_deg)
        stack_index_mat_nn[ista1, ista1:] = istack
        stack_index_mat_nn[ista1:, ista1] = istack
@jit(nopython=True, nogil=True)
def cc_stack(ch1, ch2, znert_mat_spec_c64, lo_rad, la_rad, stack_mat_spec_c64, stack_count, stack_index_mat_nn, selection_mat_nn):
    """
    Compute the cross-correlation and stack in spectral domain.
    #
    ch1, ch2: 0 for Z
              1 for inter-receiver R IRR
              2 for inter-receiver T IRT
              3 for N
              4 for E
              5 for event-receiver R ERR
              6 for event-receiver T ERT
    znert_mat_spec_c64: spectra for rows of Z,N,E,ERR,ERT,Z,N,E,ERR,ERT,Z,N,E,ERR,ERT,... in complex64 dtype.
    lo_rad: an ndarray for the longitudes in radian.
            This means the station longitudes for inter-station correlations,
            and the event longitudes for inter-source correlations.
    la_rad: an ndarray for the latitudes in radian.
            This means the station latitudes for inter-station correlations,
            and the event latitudes for inter-source correlations.
    stack_mat_spec_c64: the output stack matrix in spectral domain in complex64 dtype.
    stack_count:        the output stack count matrix in int32/int64 dtype.
    stack_index_mat_nn: the stack index matrix. It can be returned from `get_stack_index_mat(...)`.
    selection_mat_nn:   the selection matrix. It can be returned from `gcd_selection_rad(...)` and `daz_selection_rad(...)` .
    """
    mat_spec = znert_mat_spec_c64
    flag_IR = (ch1==1 or ch1==2 or ch2==1 or ch2==2)
    ncmp = 5
    n    = mat_spec.shape[0]//ncmp
    ncol = mat_spec.shape[1]
    ####
    irr1 = np.zeros(ncol, dtype=np.complex64) # useless if ch1 and ch2 are only Z,N,E,ERR,ERT
    irt1 = np.zeros(ncol, dtype=np.complex64) # useless if ch1 and ch2 are only Z,N,E,ERR,ERT
    irr2 = np.zeros(ncol, dtype=np.complex64) # useless if ch1 and ch2 are only Z,N,E,ERR,ERT
    irt2 = np.zeros(ncol, dtype=np.complex64) # useless if ch1 and ch2 are only Z,N,E,ERR,ERT
    ####
    for ista1 in range(n):
        lo1, la1 = lo_rad[ista1], la_rad[ista1]
        z1 = mat_spec[ista1*ncmp]
        n1 = mat_spec[ista1*ncmp+1]
        e1 = mat_spec[ista1*ncmp+2]
        err1 = mat_spec[ista1*ncmp+3]
        ert1 = mat_spec[ista1*ncmp+4]
        #### compute the angle from 1->2 and 2->1
        angle12 =  HALF_PI - azimuth_rad(lo1, la1, lo_rad, la_rad)  # convert azimuth (clockwise from north) to angle (counter-clockwise from east)
        angle21 = -HALF_PI - azimuth_rad(lo_rad, la_rad, lo1, la1)
        cos12, sin12 = np.cos(angle12).astype(np.float32), np.sin(angle12).astype(np.float32)
        cos21, sin21 = np.cos(angle21).astype(np.float32), np.sin(angle21).astype(np.float32)
        ####
        start = 0 if ch1 != ch2 else ista1
        for ista2 in range(start, n):
            #lo2, la2 = lo_rad[ista2], la_rad[ista2]
            z2 = mat_spec[ista2*ncmp]
            n2 = mat_spec[ista2*ncmp+1]
            e2 = mat_spec[ista2*ncmp+2]
            err2 = mat_spec[ista2*ncmp+3]
            ert2 = mat_spec[ista2*ncmp+4]
            ####
            if selection_mat_nn[ista1, ista2] > 0:
                if flag_IR:
                    # rotate to get IRR and IRT along the great-circle-path between the two stations
                    # virtual event -> station #1 -> station #2
                    #
                    #        N2  /
                    #         | /         ang2 = azimuth(st2 -> st1)
                    #         |/ R2       R2 = E2*cos(ang2) + N2*sin(ang2)
                    #    N1   2-----E2    T2 = E2*sin(ang2) - N2*sin(ang2)
                    #     |  / sta #2
                    #     | /             ang1 = azimuth(st1 -> st2)
                    #     |/ R1           R1 = E1*cos(ang1) + N1*sin(ang1)
                    #     1-------E1      T1 = E1*sin(ang1) - N1*sin(ang1)
                    #    / sta #1
                    #   /
                    #  * virtual event
                    #irr1 = e1*c1 + n1*s1
                    #irt1 = e1*s1 - n1*c1
                    #irr2 = e2*c2 + n2*s2
                    #irt2 = e2*s2 - n2*c2
                    c1, s1 = cos12[ista2], sin12[ista2]
                    c2, s2 = cos21[ista2], sin21[ista2]
                    if ch1 == 1:             #1 for inter-receiver R IRR
                        irr1 = e1*c1 + n1*s1 #
                    elif ch1 == 2:           #2 for inter-receiver T IRT
                        irt1 = e1*s1 - n1*c1 #
                    if ch2 == 1:             #1 for inter-receiver R IRR
                        irr2 = e2*c2 + n2*s2 #
                    elif ch2 == 2:           #2 for inter-receiver T IRT
                        irt2 = e2*s2 - n2*c2 #
                #### cc
                #dtypes = dtypes.union( set( [str(it) for it in (z1.dtype, irr1.dtype, irt1.dtype, n1.dtype, e1.dtype, err1.dtype, ert1.dtype)] ) )
                #print(z1.dtype, irr1.dtype, irt1.dtype, n1.dtype, e1.dtype, err1.dtype, ert1.dtype, flush=True)
                s1 = [z1, irr1, irt1, n1, e1, err1, ert1]
                s2 = [z2, irr2, irt2, n2, e2, err2, ert2]
                ss = s1[ch1] * s2[ch2].conj() # of complex64 (8 bytes)
                ss *= selection_mat_nn[ista1, ista2]
                #ss.view(dtype=np.uint64)[:] #&= selection_mat_nn[ista1, ista2] # select or discard for 0xFFFFFFFFFFFFFFFF or 0
                #### stack
                istack = stack_index_mat_nn[ista1, ista2]
                stack_mat_spec_c64[istack] += ss
                stack_count[istack] += selection_mat_nn[ista1, ista2] # select or discard for 1 or 0, respectively
    #return(dtypes)


def rd_proc_correlogram(h5file, filter_band=(0.02, 0.06666), taper_sec=50, taper_dist_range=(-1.0, 180.0), cut_sec=(0, 3000), norm=True ):
    """
    Read a correlogram from the h5file, preprocess it, and return the processed correlogram.

    Conduct the following preprocessing steps. The order is important.
    1. Apply a band-pass filter.
    2. Apply a taper given the distance range.
    3. Cut
    4. Normalize
    """
    with h5_File(h5file, 'r') as h5:
        stack_mat   = h5['dat'][:]
        stack_count = h5['stack_count'][:]
        stack_bin_centers   = h5.attrs['stack_bin_centers'][:]
        cc_t1, cc_t2, delta = h5.attrs['cc_time'][:]
        ######
        f1, f2 = filter_band
        for xs in stack_mat:
            iirfilter_f32(xs, delta, 0, 2, f1, f2, 2, 2)
        ######
        dmin, dmax = taper_dist_range
        taper_halfsize = int(taper_sec/delta)
        for xs, d in zip(stack_mat, stack_bin_centers):
            if dmin <= d <= dmax:
                taper(xs, taper_halfsize, delta)
        ######
        if cut_sec is not None:
            cut1, cut2 = cut_sec
            cut1 = max(cut1, cc_t1)
            cut2 = min(cut2, cc_t2)
            cc_t1, cc_t2 = cut1, cut2
            cut1, cut2 = int(cut1/delta), int(cut2/delta)
            stack_mat = stack_mat[:, cut1:cut2]
        ######
        if norm:
            for xs in stack_mat:
                for xs in stack_mat:
                    v = xs.max()
                    if v> 0.0:
                        xs *= (1.0/v)
        ######
        return stack_mat, stack_count, stack_bin_centers, (cc_t1, cc_t2, delta)
    pass
def plot_correlogram(stack_mat, stack_count, stack_bin_centers, cc_time_range,
                     plot_im=True, plot_waveforms=False,
                     waveform_color='k', waveform_time_shifts=None, waveform_alpha=1.0, waveform_ls='-', waveform_lw=0.5,
                     vmin=-0.6, vmax=0.6, cmap='gray', bar_color='#999999',
                     ax=None, hax=None, **kwargs):
    """
    Plot correlogram
    """
    if ax is None:
        fig, (ax, hax) = plt.subplots(2, 1, figsize=(3, 6), gridspec_kw={'height_ratios': [5, 1] } )
    cc_t1, cc_t2 = cc_time_range
    x0, x1 = stack_bin_centers[0], stack_bin_centers[-1]
    dx     = stack_bin_centers[1]-stack_bin_centers[0]
    if plot_im:
        ax.imshow(stack_mat.transpose(), aspect='auto', extent=[x0-dx*0.5, x1+dx*0.5, cc_t1, cc_t2],
                origin='lower', vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
    if plot_waveforms:
        nt = stack_mat.shape[1]
        delta = (cc_t2-cc_t1)/(nt-1)
        ts = np.arange(nt)*delta + cc_t1
        for irow, (time_series) in enumerate(stack_mat):
            loc = stack_bin_centers[irow]
            xs = time_series
            v  = xs.max()
            if v > 0.0:
                xs = xs * ((vmax*dx)/v)
            xs = xs+loc
            ys = ts
            if waveform_time_shifts is not None:
                shift = waveform_time_shifts[irow]
                ys = ys + shift
            ax.plot(xs, ys, color=waveform_color, ls=waveform_ls, lw=waveform_lw, alpha=waveform_alpha)
        pass
    if hax is not None:
        hax.bar(stack_bin_centers, stack_count, dx, color=bar_color)
        hax.set_xlim( (x0, x1) )
        hax.set_ylim((0, stack_count[1:].max()*1.1 ) )
        hax.set_xlabel('Distance ($\degree$)')
    ax.set_ylabel('Time (s)')
    return ax, hax
class CS_InterRcv:
    """
    Correlation Stack for inter receiver settings.
    Currently, we support the pair combinations for Z, N, E, ERR, ERT, IRR, IRT components/channels.

    To use, we can do:
    ```
    >>> cs = CS_InterRcv(...) # init
    >>> for it in a_list_of_events:
    >>>     cs.add(...)       # stack for each event
    ```
    """
    Z, IRR, IRT, N, E, ERR, ERT = 0, 1, 2, 3, 4, 5, 6
    def __init__(self, ch_pair_type,
                 delta, nt, cut_time_range=None,
                 dist_range=(0.0, 180.0), dist_step=1.0,
                 wtlen=None, wt_f1=None, wt_f2=None, wflen=None, whiten_taper_halfsize=50,
                 daz_range=None, gcd_range=None,
                 gc_center_rect_boxes=None, gc_center_circles=None, critical_distance_deg=0.1,
                 speedup_critical_frequency_band=None, speedup_critical_level=0.01,
                 intermediate_outfnm_prefix=None,
                 log_print=None, mpi_comm=None):
        """
        ch_pair_type: a pair of int.
                      E.g., (0, 0) for ZZ, (1, 1) for RR, (0, 3) for ZN...
                        0 (CS_InterRcv.Z)   for Z
                        1 (CS_InterRcv.IRR) for inter-receiver R (IRR)
                        2 (CS_InterRcv.IRT) for inter-receiver T (IRT)
                        3 (CS_InterRcv.N)   for N
                        4 (CS_InterRcv.E)   for E
                        5 (CS_InterRcv.ERR) for event-receiver R (ERR)
                        6 (CS_InterRcv.ERT) for event-receiver T (ERT)
        """
        ###
        self.mpi_comm = mpi_comm
        ############################################################################################################
        if log_print is None:
            if mpi_comm is  None:
                log_print = print
            else:
                log_print = Logger(log_dir='./', mpi_comm=mpi_comm)
        self.logger = log_print
        ############################################################################################################
        log_print(-1, "Start")
        if mpi_comm is not None:
            log_print(-1, 'mpi_rank=%d, mpi_size=%d' % (mpi_comm.Get_rank(), mpi_comm.Get_size() ) )
        ############################################################################################################
        # Stacking settings
        self.ch_pair_type = ch_pair_type
        log_print(-1, 'Init' )
        log_print( 1, 'ch_pair_type:       ', self.ch_pair_type, '(Note: 0:Z; 1:IRR; 2:IRT; 3:N; 4:E; 5:ERR; 6:ERT)')
        #
        self.stack_bins = StackBins(dist_range[0], dist_range[1], dist_step)
        log_print( 1, 'Stacking settings:' )
        log_print( 2, 'stack_bins:         ', dist_range, dist_step )
        log_print( 2, 'stack_bins.centers: ', self.stack_bins.centers[0], '...', self.stack_bins.centers[-1])
        log_print( 2, 'stack_bins.edges:   ', self.stack_bins.edges[0],   '...', self.stack_bins.edges[-1])
        ### daz and gcd selections
        daz_range = np.zeros(0) if (daz_range is None) else daz_range
        gcd_range = np.zeros(0) if (gcd_range is None) else gcd_range
        self.daz_range_rad = np.deg2rad(daz_range)
        self.gcd_range_rad = np.deg2rad(gcd_range)
        log_print( 1, 'Select correlation pairs:' )
        log_print( 2, 'daz_range (deg and rad):', daz_range, self.daz_range_rad)
        log_print( 2, 'gcd_range (deg and rad):', gcd_range, self.gcd_range_rad)
        ### great circle center selections
        #Don't worry if gc_center_rect_boxes/gc_center_circles is None, 1d or 2d array. They are handled well latter.
        gc_center_rect_boxes = np.zeros(0) if (gc_center_rect_boxes is None) else gc_center_rect_boxes
        gc_center_circles    = np.zeros(0) if (gc_center_circles is  None)   else gc_center_circles
        self.gc_center_rect_boxes_rad = np.deg2rad(gc_center_rect_boxes)
        self.gc_center_circles_rad    = np.deg2rad(gc_center_circles)
        self.critical_distance_rad    = np.deg2rad(critical_distance_deg)
        log_print( 1, 'Select correlation pairs by the center of the great circle path:' )
        log_print( 2, 'gc_center_rect_boxes:     ', gc_center_rect_boxes)
        log_print( 2, 'gc_center_rect_boxes(rad):', self.gc_center_rect_boxes_rad)
        log_print( 2, 'gc_center_circles:        ', gc_center_circles)
        log_print( 2, 'gc_center_circles(rad):   ', self.gc_center_circles_rad)
        log_print( 2, 'critical_distance:        ', critical_distance_deg)
        log_print( 2, 'critical_distance(rad):   ', self.critical_distance_rad)
        ############################################################################################################
        # time and spectral settings
        #
        # There are multiple tyoes of NT:
        # 1. nt4cut, this corresponds the expected time series to be cut from the input data.
        # 2. nt4mem, this corresponds to the memory allocated for the time series.
        #            nt4mem is multiples of 1024 bytes, and it is also used for address alignment.
        #.           nt4mem >=nt4cut
        #.           nt4mem is ncol of znert_mat_f32 created in add(...).
        # 3. nt4fft, this corresponds to the FFT size for the time series
        #            Here we choose nt4fft = nt4mem, which is a close to 2^n but not 2^n.
        #            nt4fft is used for cross-correlation computation. (use in fft and ifft: znert_mat_f32 <--> spectra_c64 )
        # 4. the local nt within add(...) when input data.
        #            local nt <= nt4cut
        #
        #
        idx0 = 0
        if cut_time_range is not None: # derive nt from cut_time_range
            nt = int(np.ceil((cut_time_range[1]-cut_time_range[0])/delta) )+1
            idx0 = int(cut_time_range[0]/delta)
        #
        self.nt4cut = nt
        self.cut_idx_range = (idx0, idx0 + nt) # the effective one for cutting, which is related to `nt4cut` only
        effective_cut_time_range = (idx0*delta, (idx0+nt-1)*delta) # useless parameter, but for logging only
        #
        # padding for (1) address alignment, and also (2) the length more like 2^n which is good for FFT algorithm
        func_align = lambda n, b: n + ( (b - n%b ) % b)
        alignment_bytes = 2048  # pyfftw.simd_alignment #  or just use 64 or 128 or big 2^n bytes
        n_bytes = nt*4          # 4 bytes corresponds to float32
        n_bytes = func_align(n_bytes, alignment_bytes) # align the number for columns in order
        self.nt4mem = n_bytes // 4  # only used for storing time series data
        #
        ####
        self.delta = delta
        self.nt4fft  = self.nt4mem
        self.cc_fftsize = self.nt4fft*2
        self.stack_mat = None
        self.cc_time_range = None
        log_print( 1, 'Time domain settings:' )
        log_print( 2, 'delta:                   ', self.delta)
        log_print( 2, 'input cut_time_range:     [%.2f, %.2f]' % (cut_time_range[0], cut_time_range[1]) )
        log_print( 2, 'effective cut_time_range: [%.2f, %.2f]' % (effective_cut_time_range[0], effective_cut_time_range[1]) )
        log_print( 2, 'effective cut_idx_range:  [%d, %d)' % (self.cut_idx_range[0], self.cut_idx_range[1]) )
        log_print( 2, 'nt4cut:   ', self.nt4cut)
        log_print( 2, 'nt4mem:   ', self.nt4mem)
        log_print( 2, 'nt4fft:   ', self.nt4fft)
        log_print( 2, 'alignment_bytes:         ', alignment_bytes)
        log_print( 2, 'cc_fftsize:              ', self.cc_fftsize)
        ############################################################################################################
        # obtain cc_nf for cross correlatino with optional speed up
        if (speedup_critical_frequency_band is not None) and (speedup_critical_level is not None):
            i1, i2 = get_rfft_spectra_bound(self.cc_fftsize, self.delta,
                                            speedup_critical_frequency_band, speedup_critical_level)
            ####
            self.cc_speedup_spectra_index_range = (i1, i2)
            cc_nf = i2-i1
            log_print( 1, 'Correlation speedup ON')
            log_print( 2, 'critical_frequency_band:       ', speedup_critical_frequency_band)
            log_print( 2, 'critical_level:                ', speedup_critical_level)
            log_print( 2, 'cc_speedup_spectra_index_range:', self.cc_speedup_spectra_index_range)
            log_print( 2, 'cc_nf:                         ', cc_nf)
        else:
            cc_nf = self.cc_fftsize//2 + 1
            self.cc_speedup_spectra_index_range = (0, cc_nf)
            log_print( 1, 'Correlation speed up OFF')
            log_print( 2, 'cc_nf:                         ', cc_nf)
        ############################################################################################################
        # stack buffers
        self.stack_mat_spec = np.zeros( (self.stack_bins.centers.size, cc_nf ), dtype=np.complex64)
        self.stack_count    = np.zeros(  self.stack_bins.centers.size, dtype=np.int32)
        log_print( 1, 'Correlation spectra stack buffer:')
        log_print( 2, 'stack_mat_spec: ', self.stack_mat_spec.shape, self.stack_mat_spec.dtype)
        log_print( 2, 'stack_count:    ', self.stack_count.shape, self.stack_count.dtype)
        ############################################################################################################
        # whiten settings
        self.wtlen = wtlen
        self.wt_f1 = wt_f1
        self.wt_f2 = wt_f2
        self.wflen = wflen
        self.whiten_taper_halfsize = whiten_taper_halfsize
        log_print( 1, 'Temporal normalization and spectral whitening settings:')
        log_print( 2, 'wtlen:                 ', self.wtlen, self.wt_f1, self.wt_f2)
        log_print( 2, 'wflen:                 ', self.wflen)
        log_print( 2, 'whiten_taper_halfsize: ', self.whiten_taper_halfsize, flush=True)
        ############################################################################################################
        # Optional: turn on spectral whitening speedup
        self.wf_fftsize = func_align(self.nt4cut, 2) # this may need more tests
        if (speedup_critical_frequency_band is not None) and (speedup_critical_level is not None) and (self.wtlen is not None):
            f1, f2 = speedup_critical_frequency_band
            f1, f2 = f1-wflen, f2+wflen
            if f1 < 0:
                f1 = 0
            if f2 > 0.5/self.delta:
                f2 = 0.5/self.delta
            i1, i2 = get_rfft_spectra_bound(self.wf_fftsize, self.delta, (f1, f2), speedup_critical_level)
            ####
            self.whiten_speedup_spectra_index_range = (i1, i2)
            log_print( 1, 'Spectral whitening speedup ON')
            log_print( 2, 'wf_fftsize:                        ', self.wf_fftsize )
            log_print( 2, 'critical_frequency_band:           ', (f1, f2) )
            log_print( 2, 'critical_level:                    ', speedup_critical_level)
            log_print( 2, 'whiten_speedup_spectra_index_range:', self.whiten_speedup_spectra_index_range)
        else:
            self.whiten_speedup_spectra_index_range = (-1, -1) # -1 in fwhiten_f32 for disabling speedup
            log_print( 1, 'Spectral whitening speedup OFF or not applicable')
            log_print( 2, 'wf_fftsize:                        ', self.wf_fftsize )
            log_print( 2, 'whiten_speedup_spectra_index_range:', self.whiten_speedup_spectra_index_range)
        ############################################################################################################
        # pyFFTW interface buffers and FFTW obj
        if FLAG_PYFFTW_USED:
            self.pyfftw_buf_time = pyfftw.zeros_aligned(self.nt4fft, dtype=np.float32) # the `self.nt4fft` is already aligned as in __init__(...)
            self.pyfftw_buf_spec = pyfftw.zeros_aligned(self.cc_fftsize//2+1, dtype=np.complex64)
            self.pyfftw_rfft_obj = pyfftw.builders.rfft(self.pyfftw_buf_time, n=self.cc_fftsize,
                                                        planner_effort='FFTW_PATIENT',
                                                        overwrite_input=True, auto_align_input=True, avoid_copy=False) #)
            log_print( 1, 'pyFFTW ON')
            log_print( 2, 'pyFFTW interface buffers and FFTW obj:')
            log_print( 2, 'pyfftw_buf_time: ', self.pyfftw_buf_time.shape, self.pyfftw_buf_time.dtype)
            log_print( 2, 'pyfftw_buf_spec: ', self.pyfftw_buf_spec.shape, self.pyfftw_buf_spec.dtype)
            log_print( 2, 'pyfftw_rfft_obj: ', self.pyfftw_rfft_obj, flush=True)
        ############################################################################################################
        # Optional: turn on intermediate output
        self.intermediate_out_h5fid = None
        if intermediate_outfnm_prefix is not None:
            h5_ofnm = '%s%03d.h5' % (intermediate_outfnm_prefix, mpi_comm.Get_rank() )
            self.intermediate_out_h5fid = h5_File(h5_ofnm, 'w')
            log_print( 1, 'Turn on intermediate output.')
            log_print( 2, 'Filename on this rank:             ', h5_ofnm)
        ############################################################################################################
        self.time_summary  = TimeSummary(accumulative=True)
    def __del__(self):
        if self.intermediate_out_h5fid is not None:
            self.intermediate_out_h5fid.close()
    def add(self, zne_mat_f32, stlo_rad, stla_rad, evlo_rad, evla_rad, event_name=None, local_time_summary=None):
        """
        local_time_summary: a TimeSummary object to record the time consumption for this function.
                            If `None`, a internal empty TimeSummary object will be created.
                            #
                            Note, all time summary by the `local_time_summary` will be added
                            to `self.time_summary`.  So be careful for avoiding repeatedly
                            adding the same time summary multiple times.
        """
        if local_time_summary is None:
            local_time_summary = TimeSummary(accumulative=True)
        ############################################################################################################
        #### log
        log_print = self.logger
        log_print(-1, 'Adding... %s' % event_name, flush=True)
        log_print( 1, 'Function variables:')
        log_print( 2, 'zne_mat_f32: ', zne_mat_f32.shape, zne_mat_f32.dtype)
        log_print( 2, 'stlo:        ', stlo_rad.shape, stlo_rad.dtype)
        log_print( 2, 'stla:        ', stla_rad.shape, stla_rad.dtype)
        log_print( 2, 'evlo:        ', evlo_rad.shape, evlo_rad.dtype)
        log_print( 2, 'evla:        ', evla_rad.shape, evla_rad.dtype, flush=True)
        ############################################################################################################
        #### get ZNERT if necessary, the RT here refers to event-receiver R and T (ERR, ERT)
        with Timer(tag='zne2znert', verbose=False, summary=local_time_summary):
            log_print(1, 'Cutting and making ZNERT matrix from the ZNE...')
            idx0, idx1 = self.cut_idx_range
            idx0 = max(idx0, 0)
            idx1 = min(idx1, zne_mat_f32.shape[1])
            local_sz = idx1-idx0
            if FLAG_PYFFTW_USED:
                znert_mat_f32  = pyfftw.zeros_aligned((zne_mat_f32.shape[0]//3*5, self.nt4mem), dtype=np.float32)
            else:
                znert_mat_f32  = np.zeros(            (zne_mat_f32.shape[0]//3*5, self.nt4mem), dtype=np.float32)
            ### Note: the nt4mem is different from nt4cut.
            ### nt4mem is used for cc_fftsize and address-aligned memory allocation
            ### nt4cut is for cutting the input time series.
            ### nt4mem >= nt4cut, and zeros are padded for those additional slots in nt4mem compared to nt4cut.
            ### nt4cut == self.cut_idx_range[1]-self.cut_idx_range[0]
            ### nt4mem != self.cut_idx_range[1]-self.cut_idx_range[0]
            zne2znert(zne_mat_f32[:, idx0:idx1], znert_mat_f32, stlo_rad, stla_rad, evlo_rad, evla_rad)
            del zne_mat_f32
            log_print(2, 'cut_idx here:  [%d, %d)' % (idx0, idx1) )
            log_print(2, 'nt here:       ', local_sz)
            log_print(2, 'cut_idx_range: [%d, %d)' % (self.cut_idx_range[0], self.cut_idx_range[1]) )
            log_print(2, 'nt4cut:        ', self.nt4cut)
            log_print(2, 'nt4fft:        ', self.nt4fft)
            log_print(2, 'Finished.')
            log_print(2, 'znert_mat_f32: ', znert_mat_f32.shape, znert_mat_f32.dtype, flush=True)
        ############################################################################################################
        #### whiten the input data in time series
        if self.wtlen:
            with Timer(tag='wt', verbose=False, summary=local_time_summary):
                log_print(1, 'Temporal normalization for the ZNERT matrix...')
                for xs in znert_mat_f32:
                    if xs.max() > xs.min():
                        tnorm_f32(xs[:local_sz],   self.delta, self.wtlen, self.wt_f1, self.wt_f2, 1.0e-5, self.whiten_taper_halfsize)
                log_print(2, 'Finished.', flush=True)
        if self.wflen:
            with Timer(tag='wf', verbose=False, summary=local_time_summary):
                su_wf_i1, su_wf_i2 = self.whiten_speedup_spectra_index_range # default is (-1, -1) to disable speedup
                log_print(1, 'Spectral whitening for the ZNERT matrix...')
                log_print(2, 'wf_fftsize:                        ', self.wf_fftsize )
                log_print(2, 'whiten_speedup_spectra_index_range:', (su_wf_i1, su_wf_i2) )
                wf_fftsize = self.wf_fftsize
                for xs in znert_mat_f32:
                    if xs.max() > xs.min():
                        fwhiten_f32(xs[:local_sz], self.delta, self.wflen, 1.0e-5, self.whiten_taper_halfsize,
                                    su_wf_i1, su_wf_i2, wf_fftsize )
                log_print(2, 'Finished.', flush=True)
        ############################################################################################################
        #### remove Nan of Inf
        not_valid = lambda xs: np.any( np.isnan(xs) ) or np.any( np.isinf(xs) ) or (np.max(xs) == np.min(xs) )
        with Timer(tag='fix_nan', verbose=False, summary=local_time_summary):
            log_print(1, 'Zeroing the stations with any Nan of Inf in any of ZNERT...')
            invalid_stas = [idx//5 for idx, xs in enumerate(znert_mat_f32) if not_valid(xs) ]
            invalid_stas = list(set(invalid_stas) )
            for ista in invalid_stas:
                znert_mat_f32[ista*5,   :] = 0.0
                znert_mat_f32[ista*5+1, :] = 0.0
                znert_mat_f32[ista*5+2, :] = 0.0
                znert_mat_f32[ista*5+3, :] = 0.0
                znert_mat_f32[ista*5+4, :] = 0.0
            log_print(2, 'Finished.', flush=True)
        ############################################################################################################
        #### fft
        with Timer(tag='fft', verbose=False, summary=local_time_summary):
            log_print(1, 'FFTing...')
            nrow = znert_mat_f32.shape[0]
            i1, i2 = self.cc_speedup_spectra_index_range
            if FLAG_PYFFTW_USED:
                spectra_c64 = pyfftw.zeros_aligned((nrow, i2-i1), dtype=np.complex64)
            else:
                spectra_c64 = np.zeros(            (nrow, i2-i1), dtype=np.complex64)
            ####
            if FLAG_PYFFTW_USED:
                ####method: pyfftw interface fft
                ###pyffftw_interfaces_cache.enable()
                ###for irow in range(nrow): # do fft one by one may save some memory
                ###    spectra_c64[irow] = rfft(znert_mat_f32[irow], self.cc_fftsize)[i1:i2]
                #method: pyfftw builder fft
                buf_spec = self.pyfftw_buf_spec
                rfft_obj = self.pyfftw_rfft_obj
                for irow in range(nrow):
                    rfft_obj(znert_mat_f32[irow], buf_spec, normalise_idft=True)
                    spectra_c64[irow] = buf_spec[i1:i2]
            else:
                #method: scipy fft
                for irow in range(nrow):
                    spectra_c64[irow] = rfft(znert_mat_f32[irow], self.cc_fftsize)[i1:i2]# method: scipy.fft.rfft
            del znert_mat_f32
            log_print(2, 'Finished.')
            log_print(2, 'spectra_c64: ', spectra_c64.shape, spectra_c64.dtype, flush=True)
        ############################################################################################################
        ccpairs_selection_mat = np.ones( (stlo_rad.size, stlo_rad.size), dtype=np.int8)
        gcd_selection_mat = None
        daz_selection_mat = None
        gc_center_selection_mat = None
        dict_output_inter_mediate_data = dict()
        ############################################################################################################
        #### selection w.r.t. to gcd and daz
        with Timer(tag='gcd_daz_select', verbose=False, summary=local_time_summary):
            if (self.gcd_range_rad.size>0):
                log_print(1, 'Selecting wr.t. gcd ranges...')
                gcd_selection_mat = np.zeros( (stlo_rad.size, stlo_rad.size), dtype=np.int8)
                gcd_selection(stlo_rad.size, stlo_rad, stla_rad, evlo_rad[0], evla_rad[0], self.gcd_range_rad, gcd_selection_mat)
                ccpairs_selection_mat &= gcd_selection_mat
                #
                dict_output_inter_mediate_data['gcd_selection_mat'] = gcd_selection_mat
                log_print(2, 'gcd_selection_mat: ', gcd_selection_mat.shape, gcd_selection_mat.dtype)
                log_print(2, 'Finished.', flush=True)
            if (self.daz_range_rad.size>0):
                log_print(1, 'Selecting wr.t. daz ranges...')
                daz_selection_mat = np.zeros( (stlo_rad.size, stlo_rad.size), dtype=np.int8)
                daz_selection(stlo_rad.size, stlo_rad, stla_rad, evlo_rad[0], evla_rad[0], self.daz_range_rad, daz_selection_mat)
                ccpairs_selection_mat &= daz_selection_mat
                #
                dict_output_inter_mediate_data['daz_selection_mat'] = daz_selection_mat
                log_print(2, 'daz_selection_mat: ', daz_selection_mat.shape, daz_selection_mat.dtype)
                log_print(2, 'Finished.', flush=True)
        ############################################################################################################
        #### selection w.r.t. great circle center locations
        with Timer(tag='gc_center_select', verbose=False, summary=local_time_summary):
            if (self.gc_center_rect_boxes_rad.size>0) or (self.gc_center_circles_rad.size>0):
                log_print(1, 'Selecting w.r.t. to great circle center locations...')
                gc_center_selection_mat = np.zeros( (stlo_rad.size, stlo_rad.size), dtype=np.int8)
                gc_center_selection(stlo_rad.size, stlo_rad, stla_rad, evlo_rad[0], evla_rad[0],
                                    self.gc_center_rect_boxes_rad, self.gc_center_circles_rad, gc_center_selection_mat,
                                    self.critical_distance_rad)
                dict_output_inter_mediate_data['gc_center_selection_mat'] = gc_center_selection_mat
                ccpairs_selection_mat &= gc_center_selection_mat ##### merge this selection to the functional selection matrix
                log_print(2, 'gc_center_selection_mat: ', gc_center_selection_mat.shape, gc_center_selection_mat.dtype)
                log_print(2, 'Finished.', flush=True)
        ############################################################################################################
        #### Finish the correlation selection
        with Timer(tag='cc_pair_select', verbose=False, summary=local_time_summary):
            log_print(1, 'Finish obtaining cc pair selections matrix...')
            for ista in invalid_stas:
                ccpairs_selection_mat[ista,:] = 0
                ccpairs_selection_mat[:,ista] = 0
            dict_output_inter_mediate_data['ccpairs_selection_mat'] = ccpairs_selection_mat
            log_print(2, 'ccpairs_selection_mat: ', ccpairs_selection_mat.shape, ccpairs_selection_mat.dtype)
            log_print(2, 'Finished.', flush=True)
        ############################################################################################################
        #### stack index
        with Timer(tag='get_stack_index', verbose=False, summary=local_time_summary):
            log_print(1, 'Computing the stack index...')
            stack_index_mat_nn = np.zeros( (stlo_rad.size, stlo_rad.size), dtype=np.uint16)
            dist_min, dist_step = self.stack_bins.centers[0], self.stack_bins.step
            get_stack_index_mat(stlo_rad.size, stlo_rad, stla_rad, dist_min, dist_step, stack_index_mat_nn)
            dict_output_inter_mediate_data['stack_index_mat_nn'] = stack_index_mat_nn
            log_print(2, 'stack_index_mat_nn: ', stack_index_mat_nn.shape, stack_index_mat_nn.dtype)
            log_print(2, 'Finished.', flush=True)
        ############################################################################################################
        #### Output the intermediate data
        with Timer(tag='o_inter', verbose=False, summary=local_time_summary):
            if self.intermediate_out_h5fid is not None:
                grp = self.intermediate_out_h5fid.create_group(event_name)
                grp.create_dataset('stlo_rad', data=stlo_rad, dtype=np.float32)
                grp.create_dataset('stla_rad', data=stla_rad, dtype=np.float32)
                grp.create_dataset('evlo_rad', data=evlo_rad, dtype=np.float32)
                grp.create_dataset('evla_rad', data=evla_rad, dtype=np.float32)
                if self.gcd_range_rad.size>0:
                    grp.create_dataset('gcd_selection_mat', data=np.packbits(gcd_selection_mat.flatten() ) )
                    grp['gcd_selection_mat'].attrs['shape'] = gcd_selection_mat.shape
                    grp['gcd_selection_mat'].attrs['gcd_range_rad'] = np.reshape(self.gcd_range_rad, (-1, 2) )
                if self.daz_range_rad.size>0:
                    grp.create_dataset('daz_selection_mat', data=np.packbits(daz_selection_mat.flatten() ) )
                    grp['daz_selection_mat'].attrs['shape'] = daz_selection_mat.shape
                    grp['daz_selection_mat'].attrs['daz_range_rad'] = np.reshape(self.daz_range_rad, (-1, 2) )
                if (self.gc_center_rect_boxes_rad.size>0) or (self.gc_center_circles_rad.size>0):
                    grp.create_dataset('gc_center_selection_mat', data=np.packbits(gc_center_selection_mat.flatten() ) )
                    grp['gc_center_selection_mat'].attrs['shape'] = gc_center_selection_mat.shape
                    grp['gc_center_selection_mat'].attrs['gc_center_rect_boxes_rad'] = np.reshape(self.gc_center_rect_boxes_rad, (-1, 4) )
                    grp['gc_center_selection_mat'].attrs['gc_center_circles_rad']    = np.reshape(self.gc_center_circles_rad,    (-1, 3) )
                if np.sum(ccpairs_selection_mat==0) > 0:
                    grp.create_dataset('ccpairs_selection_mat', data=np.packbits(ccpairs_selection_mat.flatten() ) )
                    grp['ccpairs_selection_mat'].attrs['shape'] = ccpairs_selection_mat.shape
                # stack_index_mat_nn values are not binary, hence CANNOT support packbits(...)
                grp.create_dataset('stack_index_mat_nn', data=stack_index_mat_nn)
                log_print(1, 'Intermediate data saved!', flush=True)
            del gcd_selection_mat
            del daz_selection_mat
            del gc_center_selection_mat
        ############################################################################################################
        #### cc & stack
        with Timer(tag='ccstack', verbose=False, summary=local_time_summary):
            log_print(1, 'cc&stack...')
            ch1, ch2 = self.ch_pair_type
            stack_count = self.stack_count
            stack_mat_spec = self.stack_mat_spec
            #print(spectra_c64.dtype, stack_mat_spec.dtype, stack_count.dtype, stack_index_mat_nn.dtype, selection_mat.dtype, flush=True)
            cc_stack(ch1, ch2, spectra_c64, stlo_rad, stla_rad, stack_mat_spec, stack_count, stack_index_mat_nn, ccpairs_selection_mat)
            log_print(2, 'Finished.', flush=True)
        ############################################################################################################
        #t_total = time_time() - t0
        #t_other = t_total*1000 - local_time_summary.total_t() # miliseconds
        #local_time_summary.push('other', t_other, color='k')
        log_print(1, 'Time consumption:', flush=True)
        local_time_summary.plot_rough(file=log_print.log_fid, prefix_str='        ')
        ############################################################################################################
        self.time_summary += local_time_summary
        ############################################################################################################
    def finish(self, filter_band=None, fold=True, cut=None, taper_sec=None, norm=False, ofnm=None):
        """
        """
        log_print = self.logger
        log_print(-1, 'To finish...')
        ############################################################################################################
        with Timer(tag='finish.ifft', verbose=False, summary=self.time_summary):
            log_print(1, 'IFFT...')
            cc_fftsize = self.cc_fftsize
            stack_count = self.stack_count
            stack_mat_spec = self.stack_mat_spec
            ####
            i1, i2 = self.cc_speedup_spectra_index_range
            full_stack_mat_spec = np.zeros( (stack_mat_spec.shape[0], i2), dtype=np.complex64)
            full_stack_mat_spec[:, i1:i2] = stack_mat_spec
            stack_mat = irfft(full_stack_mat_spec, cc_fftsize, axis=1)
            log_print(2, 'stack_mat:    ', stack_mat.shape, stack_mat.dtype, 'Finished.')
        ############################################################################################################
        with Timer(tag='finish.roll', verbose=False, summary=self.time_summary):
            log_print(1, 'Roll...')
            delta = self.delta
            rollsize = self.nt4fft-1
            stack_mat = np.roll(stack_mat, rollsize, axis=1)
            stack_mat = stack_mat[:,:-1] #get rid of the zero point
            cc_t1, cc_t2 = -rollsize*delta, rollsize*delta
            log_print(2, 'stack_mat:    ', stack_mat.shape, stack_mat.dtype )
            log_print(2, 'cc time range:', (cc_t1, cc_t2), 'Finished.' )
        ############################################################################################################
        if filter_band is not None:
            with Timer(tag='finish.filter', verbose=False, summary=self.time_summary):
                log_print(1, 'Filter...', filter_band)
                f1, f2 = filter_band
                for xs in stack_mat:
                    iirfilter_f32(xs, delta, 0, 2, f1, f2, 2, 2)
                log_print(2, 'Finished.')
        ############################################################################################################
        if fold:
            with Timer(tag='finish.fold', verbose=False, summary=self.time_summary):
                log_print(1, 'Fold...')
                stack_mat += stack_mat[:, ::-1]
                stack_mat = stack_mat[:, rollsize:]
                cc_t1, cc_t2 = 0, rollsize*delta
                log_print(2, 'stack_mat:    ', stack_mat.shape, stack_mat.dtype )
                log_print(2, 'cc time range:', (cc_t1, cc_t2), 'Finished.')
        ############################################################################################################
        if cut is not None:
            with Timer(tag='finish.cut', verbose=False, summary=self.time_summary):
                log_print(1, 'Cut...', cut)
                post_t1, post_t2 = cut
                post_t1 = cc_t1 if post_t1 < cc_t1 else post_t1
                post_t2 = cc_t2 if post_t2 > cc_t2 else post_t2
                #
                post_i1 = int( round((post_t1-cc_t1)/delta) )
                post_i2 = int( round((post_t2-cc_t1)/delta) ) + 1
                #
                cc_t1 = cc_t1 + post_i1 *delta
                cc_t2 = cc_t1 + (post_i2-post_i1-1)*delta
                stack_mat = stack_mat[:, post_i1:post_i2 ]
                log_print(2, 'stack_mat:    ', stack_mat.shape, stack_mat.dtype)
                log_print(2, 'cc time range:', (cc_t1, cc_t2), 'Finished.')
        ############################################################################################################
        if taper_sec:
            with Timer(tag='finish.taper', verbose=False, summary=self.time_summary):
                log_print(1, 'Taper...', taper_sec)
                post_taper_halfsize = int(taper_sec/delta)
                for xs in stack_mat:
                    taper(xs, post_taper_halfsize)
                log_print(2, 'Finished.')
        ############################################################################################################
        if norm:
            with Timer(tag='finish.norm', verbose=False, summary=self.time_summary):
                log_print(1, 'Norm...')
                for xs in stack_mat:
                    v = xs.max()
                    if v> 0.0:
                        xs *= (1.0/v)
                log_print(2, 'Finished.')
        ############################################################################################################
        if ofnm:
            with Timer(tag='finish.output', verbose=False, summary=self.time_summary):
                log_print(1, 'Write...', ofnm)
                CS_InterRcv.mkdirs(ofnm)
                with h5_File(ofnm, 'w') as fid:
                    fid.attrs['cc_time'] = (cc_t1, cc_t2, delta)
                    fid.attrs['stack_bin_centers'] = self.stack_bins.centers
                    fid.attrs['stack_bin_edges'] = self.stack_bins.edges
                    if filter_band is not None:
                        fid.attrs['filter_band'] = filter_band
                    fid.attrs['flag_filter_band']= 1 if (filter_band is not None) else 0
                    fid.attrs['flag_fold']       = 1 if fold else 0
                    fid.attrs['flag_cut']        = 1 if cut  else 0
                    fid.attrs['taper_sec']       = taper_sec if taper_sec else 0
                    fid.attrs['flag_norm']       = 1 if norm else 0
                    fid.create_dataset('dat', data=stack_mat, dtype=np.float32)
                    fid.create_dataset('stack_count', data=stack_count, dtype=np.int32)
                log_print(2, 'Finished.')
        ############################################################################################################
        self.stack_mat = stack_mat
        self.cc_time_range = (cc_t1, cc_t2)
        ############################################################################################################
        result = {'stack_mat': stack_mat, 'stack_count': stack_count, 
                  'cc_time': (cc_t1, cc_t2, delta),
                  'cc_dist': (self.stack_bins.centers, self.stack_bins.edges) }
        return result
    def mpi_finish(self, filter_band=None, fold=True, cut=None, taper_sec=None, norm=False, ofnm=None):
        """
        """
        mpi_comm = self.mpi_comm
        if mpi_comm is None:
            raise ValueError('mpi_comm is None!')
        ####
        log_print = self.logger
        ####
        mpi_comm.Barrier()
        mpi_rank = mpi_comm.Get_rank()
        mpi_ncpu = mpi_comm.Get_size()
        ########################################################################
        # gather stack_mat_spec and stack_count
        log_print(-1, 'MPI reduce stack results to rank0...', flush=True)
        stack_mat_spec = self.stack_mat_spec
        stack_count    = self.stack_count
        global_stack_mat_spec = stack_mat_spec
        global_stack_count    = stack_count
        if mpi_rank == 0:
            global_stack_mat_spec = np.zeros(stack_mat_spec.shape, dtype=stack_mat_spec.dtype )
            global_stack_count    = np.zeros(stack_count.shape,    dtype=stack_count.dtype )
        mpi_comm.Reduce([stack_mat_spec, MPI.C_FLOAT_COMPLEX], [global_stack_mat_spec, MPI.C_FLOAT_COMPLEX], MPI.SUM, root= 0)
        mpi_comm.Reduce([stack_count, MPI.INT32_T], [global_stack_count, MPI.INT32_T], MPI.SUM, root= 0 )
        log_print(1, 'Finished.')
        mpi_comm.Barrier()
        ########################################################################
        # finish on rank 0
        if mpi_rank == 0:
            self.stack_mat_spec = global_stack_mat_spec
            self.stack_count    = global_stack_count
            to_return = self.finish(filter_band=filter_band, fold=fold, cut=cut, taper_sec=taper_sec, norm=norm, ofnm=ofnm)
        else:
            to_return = None
        ########################################################################
        # gather and print time consumption information
        mpi_comm.Barrier()
        log_print(-1, 'Done!')
        log_print( 1, 'Time summary for all operations on this rank')
        self.time_summary.plot_rough(file=log_print.log_fid, prefix_str='        ')
        self.mpi_gather_time_summary()
        if mpi_rank == 0:
            log_print( 1, 'Summed on all ranks')
            self.global_time_summary.plot_rough(file=log_print.log_fid, prefix_str='        ')
        ########################################################################
        return to_return
    def mpi_gather_time_summary(self):
        """
        Gather all `time_summary` from all ranks, and sum them up
        into a `self.global_time_summary` which is only meaningful
        on rank 0.
        """
        global_time_summary = TimeSummary(accumulative=True)
        mpi_comm = self.mpi_comm
        mpi_rank = mpi_comm.Get_rank()
        mpi_size = mpi_comm.Get_size()
        if mpi_rank != 0:
            # Send the object to rank 0
            mpi_comm.send(self.time_summary, dest=0, tag=mpi_rank)
        else:
            global_time_summary += self.time_summary
            # Rank 0: Receive objects from all other ranks
            for i in range(1, mpi_size):
                it_time_summary = mpi_comm.recv(source=i, tag=i)
                global_time_summary += it_time_summary
        self.global_time_summary = global_time_summary
    def plot(self, fignm='test.png', figsize=(4, 8), vmin=-0.6, vmax=0.6, cmap='gray', color='#999999', title=''):
        """
        """
        log_print = self.logger
        log_print(-1, 'Plot correlogram...', fignm)
        stack_mat = self.stack_mat
        if stack_mat is None:
            raise ValueError('!')
        stack_count = self.stack_count
        cc_t1, cc_t2 = self.cc_time_range
        bin_centers, bin_edges, bin_step = self.stack_bins.centers, self.stack_bins.edges, self.stack_bins.step
        ####
        try:
            CS_InterRcv.mkdirs(fignm)
            extent = (bin_edges[0], bin_edges[-1], cc_t1, cc_t2)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [6, 1]})
            ax1.imshow(stack_mat.transpose(), aspect='auto', extent=extent, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
            ax2.bar(bin_centers, stack_count, bin_step, color=color)
            ax2.set_xlim( (bin_edges[0], bin_edges[-1]) )
            ax2.set_ylim((0, stack_count[1:].max()*1.1 ) )
            ax2.set_xlabel('Distance ($\degree$)')
            ax1.set_ylabel('Time (s)')
            ax1.set_title(title)
            plt.savefig(fignm, dpi=200, bbox_inches = 'tight', pad_inches = 0.02)
            plt.close()
        except Exception as err:
            print(err, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        log_print(1, 'Finished.')
    @staticmethod
    def mkdirs(filename):
        if filename:
            folder = '/'.join(filename.split('/')[:-1])
            if not os.path.exists(folder):
                os.makedirs(folder)

if __name__ == "__main__":
    if False:
        bins = StackBins(0, 10, 1)
        print(bins.centers)
        print(bins.edges)
        print(bins.get_idx(2.5))
        print(bins.get_idx(3.5))
        print(bins.get_idx(4.5))
        print(bins.get_idx(5.5))
        print(bins.get_idx(6.5))