#!/usr/bin/env python3

"""
This module implement the calculation of cross correlation and stacking.
"""
from mpi4py import MPI
import numpy as np
from numba import jit
from scipy.fft import rfft, irfft
from time import time as time_time
from h5py import File as h5_File
import matplotlib.pyplot as plt
import sys, traceback, os, os.path

from .geomath import haversine_rad, azimuth_rad, point_distance_to_great_circle_plane_rad
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
    n = zne_mat.shape[0]//3
    for ista in range(n):
        c, s = cs[ista], ss[ista]
        z = zne_mat[ista*3]
        n = zne_mat[ista*3+1]
        e = zne_mat[ista*3+2]
        #
        znert_mat[ista*5]   = z
        znert_mat[ista*5+1] = n
        znert_mat[ista*5+2] = e
        znert_mat[ista*5+3] = e*c + n*s
        znert_mat[ista*5+4] = e*s - n*c

@jit(nopython=True, nogil=True)
def gcd_daz_selection(n, lo_rad, la_rad, ptlo_rad, ptla_rad, gcd_min_rad, gcd_max_rad, daz_min_rad, daz_max_rad, selection_mat_nn):
    """
    Compute a NxN matrix with values of 1 or 0 for selecting the pairs of stations or not.
    1 for selecting and 0 for discarding.
    The ijth component of the matrix is 1 or 0 to select or discard the pair formed by the ith and jth records.
    The returen matrix is symmetric.
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
        gcd_min_rad: the minimum great circle distance in radian
        gcd_max_rad: the maximum great circle distance in radian
        daz_min_rad: the minimum azimuth difference in radian
        daz_max_rad: the maximum azimuth difference in radian
        selection_mat_nn: the output selection matrix
    """
    az = azimuth_rad(ptlo_rad, ptla_rad, lo_rad, la_rad)
    np.fill_diagonal(selection_mat_nn, 1)
    for ista1 in range(n-1):
        lo1, la1, az1 = lo_rad[ista1], la_rad[ista1], az[ista1]
        ista2 = ista1+1
        gcd = np.abs( point_distance_to_great_circle_plane_rad(ptlo_rad, ptla_rad, lo1, la1, lo_rad[ista2:], la_rad[ista2:] ) )
        daz = np.abs( az1-az[ista2:] ) % PI
        daz = np.minimum(daz, PI-daz)
        v = np.where((gcd_min_rad<=gcd) & (gcd<=gcd_max_rad), 1, 0) & np.where((daz_min_rad<=daz) & (daz<=daz_max_rad), 1, 0)
        selection_mat_nn[ista1, ista2:] = v
        selection_mat_nn[ista2:, ista1] = v
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
    selection_mat_nn:   the selection matrix. It can be returned from `gcd_daz_selection_rad(...)`.
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
#def plot_stack_mat(stack_mat, stack_count, stack_bins, freq_band, title=None, fig=None, ax=None, figname=None):
#    """
#    Plot the stack matrix
#    """
#    if fig is None:
#        fig = plt.figure()
#    if ax is None:
#        ax = fig.add_subplot(111)
#    #
#    nf = stack_mat_spec.shape[1]
#    freqs = np.fft.rfftfreq(nf*2, d=1.0)
#    idxs = np.where( (freqs>=freq_band[0]) & (freqs<=freq_band[1]) )[0]
#    freqs = freqs[idxs]
#    spec = np.abs(stack_mat_spec).sum(axis=0)
#    spec = spec[idxs]
#    ax.plot(freqs, spec)
#    ax.set_xlabel('Frequency (Hz)')
#    ax.set_ylabel('Amplitude')
#    if title:
#        ax.set_title(title)
#    return fig, ax

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
                 daz_range=None, gcd_range=None, log_print=None,
                 speedup_critical_frequency_band=None, speedup_critical_level=0.01,
                 mpi_comm=None):
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
        #
        self.daz_range_rad = np.deg2rad(daz_range) if daz_range else None
        self.gcd_range_rad = np.deg2rad(gcd_range) if gcd_range else None
        log_print( 1, 'Select correlation pairs:' )
        log_print( 2, 'daz_range:          ', daz_range, self.daz_range_rad)
        log_print( 2, 'gcd_range:          ', gcd_range, self.gcd_range_rad)
        ############################################################################################################
        # time and spectral settings
        idx0 = 0
        if cut_time_range: # derive nt from cut_time_range
            nt = int(np.ceil((cut_time_range[1]-cut_time_range[0])/delta) )+1
            idx0 = int(cut_time_range[0]/delta)
        nt = nt+nt%2
        self.cut_idx_range  = (idx0, idx0+nt)
        self.cut_time_range = ( idx0*delta, (idx0+nt-1)*delta )
        ####
        self.nt = nt
        self.delta = delta
        self.cc_fftsize = self.nt*2
        self.stack_mat = None
        self.cc_time_range = None
        log_print( 1, 'Time domain settings:' )
        log_print( 2, 'delta:                   ', self.delta)
        log_print( 2, 'original cut_time_range: [%.2f, %.2f]' % (cut_time_range[0], cut_time_range[1]) )
        log_print( 2, 'adjusted cut_time_range: [%.2f, %.2f]' % (self.cut_time_range[0], self.cut_time_range[1]) )
        log_print( 2, 'adjusted cut_idx_range:  [%d, %d)' % (self.cut_idx_range[0], self.cut_idx_range[1]) )
        log_print( 2, 'nt:                      ', self.nt)
        log_print( 2, 'cc_fftsize:              ', self.cc_fftsize)
        ############################################################################################################
        # obtain cc_nf for cross correlatino with optional speed up
        if (speedup_critical_frequency_band is not None) and (speedup_critical_level is not None):
            i1, i2 = get_rfft_spectra_bound(self.cc_fftsize, self.delta,
                                            speedup_critical_frequency_band, speedup_critical_level)
            ####
            self.cc_speedup_spectra_index_range = (i1, i2)
            cc_nf = i2-i1
            log_print( 1, 'Correlation speed up ON')
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
        #self.stack_mat      = np.zeros( (self.stack_bins.centers.size, self.cc_fftsize-1),  dtype=np.float32)
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
        if (speedup_critical_frequency_band is not None) and (speedup_critical_level is not None) and (self.wtlen is not None):
            f1, f2 = speedup_critical_frequency_band
            f1, f2 = f1-wflen, f2+wflen
            if f1 < 0:
                f1 = 0
            if f2 > 0.5/self.delta:
                f2 = 0.5/self.delta
            i1, i2 = get_rfft_spectra_bound(self.nt, self.delta, (f1, f2), speedup_critical_level)
            ####
            self.whiten_speedup_spectra_index_range = (i1, i2)
            log_print( 1, 'Spectral whitening speedup ON')
            log_print( 2, 'critical_frequency_band:           ', (f1, f2) )
            log_print( 2, 'critical_level:                    ', speedup_critical_level)
            log_print( 2, 'whiten_speedup_spectra_index_range:', self.whiten_speedup_spectra_index_range)
        else:
            self.whiten_speedup_spectra_index_range = (-1, -1) # -1 in fwhiten_f32 for disabling speedup
            log_print( 1, 'Spectral whitening speedup OFF or not applicable')
        ############################################################################################################
        self.time_summary  = TimeSummary(accumulative=True)
    def add(self, zne_mat_f32, stlo_rad, stla_rad, evlo_rad, evla_rad, infostr=None, local_time_summary=None):
        """
        local_time_summary: a TimeSummary object to record the time consumption for this function.
                            If `None`, a internal empty TimeSummary object will be created.
                            #
                            Note, all time summary by the `local_time_summary` will be added
                            to `time.time_summary`.  So be careful for avoiding repeatedly
                            adding the same time summary multiple times.
        """
        if local_time_summary is None:
            local_time_summary = TimeSummary(accumulative=True)
        ############################################################################################################
        #### log
        log_print = self.logger
        log_print(-1, 'Adding... %s' % infostr, flush=True)
        log_print( 1, 'Function variables:')
        log_print( 2, 'zne_mat_f32: ', zne_mat_f32.shape, zne_mat_f32.dtype)
        log_print( 2, 'stlo:        ', stlo_rad.shape, stlo_rad.dtype)
        log_print( 2, 'stla:        ', stla_rad.shape, stla_rad.dtype)
        log_print( 2, 'evlo:        ', evlo_rad.shape, evlo_rad.dtype)
        log_print( 2, 'evla:        ', evla_rad.shape, evla_rad.dtype)
        ############################################################################################################
        #### get ZNERT if necessary, the RT here refers to event-receiver R and T (ERR, ERT)
        with Timer(tag='zne2znert', verbose=False, summary=local_time_summary):
            log_print(1, 'Cutting and making ZNERT matrix from the ZNE...')
            idx0, idx1 = self.cut_idx_range
            idx0 = max(idx0, 0)
            idx1 = min(idx1, zne_mat_f32.shape[1])
            znert_mat_f32 = np.zeros( (zne_mat_f32.shape[0]//3*5, idx1-idx0), dtype=np.float32)
            zne2znert(zne_mat_f32[:, idx0:idx1], znert_mat_f32, stlo_rad, stla_rad, evlo_rad, evla_rad)
            log_print(2, 'Finished.')
            log_print(2, 'znert_mat_f32: ', znert_mat_f32.shape, znert_mat_f32.dtype)
            del zne_mat_f32
        ############################################################################################################
        #### whiten the input data in time series
        if self.wtlen:
            with Timer(tag='wt', verbose=False, summary=local_time_summary):
                log_print(1, 'Temporal normalization for the ZNERT matrix...')
                for xs in znert_mat_f32:
                    if xs.max() > xs.min():
                        tnorm_f32(xs,   self.delta, self.wtlen, self.wt_f1, self.wt_f2, 1.0e-5, self.whiten_taper_halfsize)
                log_print(2, 'Finished.')
        if self.wflen:
            with Timer(tag='wf', verbose=False, summary=local_time_summary):
                su_wf_i1, su_wf_i2 = self.whiten_speedup_spectra_index_range # default is (-1, -1) to disable speedup
                log_print(1, 'Spectral whitening for the ZNERT matrix...')
                for xs in znert_mat_f32:
                    if xs.max() > xs.min():
                        fwhiten_f32(xs, self.delta, self.wflen, 1.0e-5, self.whiten_taper_halfsize, su_wf_i1, su_wf_i2)
                log_print(2, 'Finished.')
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
            log_print(2, 'Finished.')
        ############################################################################################################
        #### fft
        with Timer(tag='fft', verbose=False, summary=local_time_summary):
            log_print(1, 'FFTing...')
            nrow = znert_mat_f32.shape[0]
            i1, i2 = self.cc_speedup_spectra_index_range
            spectra_c64 = np.zeros( (nrow, i2-i1), dtype=np.complex64)
            for irow in range(nrow): # do fft one by one may save some memory
                spectra_c64[irow] = rfft(znert_mat_f32[irow], self.cc_fftsize)[i1:i2]
            log_print(2, 'Finished.')
            log_print(2, 'spectra_c64: ', spectra_c64.shape, spectra_c64.dtype)
            del znert_mat_f32
        ############################################################################################################
        #### selection
        with Timer(tag='select', verbose=False, summary=local_time_summary):
            log_print(1, 'Selecting and weighting(not implemented)...')
            selection_mat = np.zeros( (stlo_rad.size, stlo_rad.size), dtype=np.int8) + 1
            if (self.gcd_range_rad is not None) or (self.daz_range_rad is not None):
                gcd_min_rad, gcd_max_rad = self.gcd_range_rad if (self.gcd_range_rad is not None) else (-1, 1000) # (-1, 1000) means to select all
                daz_min_rad, daz_max_rad = self.daz_range_rad if (self.daz_range_rad is not None) else (-1, 1000) # (-1, 1000) means to select all
                #print(type(stlo_rad), type(stla_rad), type(evlo_rad[0]), type(evla_rad[0]) )
                gcd_daz_selection(stlo_rad.size, stlo_rad, stla_rad, evlo_rad[0], evla_rad[0],
                                  gcd_min_rad, gcd_max_rad, daz_min_rad, daz_max_rad, selection_mat)
            for ista in invalid_stas:
                selection_mat[ista,:] = 0
                selection_mat[:,ista] = 0
            log_print(2, 'selection_mat: ', selection_mat.shape, selection_mat.dtype)
            log_print(2, 'Finished.')
        ############################################################################################################
        #### stack index
        with Timer(tag='get_stack_index', verbose=False, summary=local_time_summary):
            log_print(1, 'Computing the stack index...')
            stack_index_mat_nn = np.zeros( (stlo_rad.size, stlo_rad.size), dtype=np.uint32)
            dist_min, dist_step = self.stack_bins.centers[0], self.stack_bins.step
            get_stack_index_mat(stlo_rad.size, stlo_rad, stla_rad, dist_min, dist_step, stack_index_mat_nn)
            log_print(2, 'stack_index_mat_nn: ', stack_index_mat_nn.shape, stack_index_mat_nn.dtype)
            log_print(2, 'Finished.')
        ############################################################################################################
        #### cc & stack
        with Timer(tag='ccstack', verbose=False, summary=local_time_summary):
            log_print(1, 'cc&stack...')
            ch1, ch2 = self.ch_pair_type
            stack_count = self.stack_count
            stack_mat_spec = self.stack_mat_spec
            #print(spectra_c64.dtype, stack_mat_spec.dtype, stack_count.dtype, stack_index_mat_nn.dtype, selection_mat.dtype, flush=True)
            cc_stack(ch1, ch2, spectra_c64, stlo_rad, stla_rad, stack_mat_spec, stack_count, stack_index_mat_nn, selection_mat)
            log_print(2, 'Finished.')
        ############################################################################################################
        #t_total = time_time() - t0
        #t_other = t_total*1000 - local_time_summary.total_t() # miliseconds
        #local_time_summary.push('other', t_other, color='k')
        log_print(1, 'Time consumption:', flush=True)
        local_time_summary.plot_rough(file=log_print.log_fid, prefix_str='        ')
        ############################################################################################################
        self.time_summary += local_time_summary
        ############################################################################################################
    def finish(self, filter_band=(0.02, 0.066), fold=True, cut=None, taper_sec=None, norm=True, ofnm=None):
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
            #stack_mat = np.fft.irfft(stack_mat_spec, cc_fftsize, axis=1).astype(np.float32)
            #stack_mat = irfft(stack_mat_spec, cc_fftsize, axis=1)
            i1, i2 = self.cc_speedup_spectra_index_range
            full_stack_mat_spec = np.zeros( (stack_mat_spec.shape[0], i2), dtype=np.complex64)
            full_stack_mat_spec[:, i1:i2] = stack_mat_spec
            stack_mat = irfft(full_stack_mat_spec, cc_fftsize, axis=1)
            log_print(2, 'stack_mat:    ', stack_mat.shape, stack_mat.dtype, 'Finished.')
        ############################################################################################################
        with Timer(tag='finish.roll', verbose=False, summary=self.time_summary):
            log_print(1, 'Roll...')
            nt = self.nt
            delta = self.delta
            rollsize = nt-1
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
                    fid.attrs['filter_band'] = filter_band
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
    def mpi_finish(self, filter_band=(0.02, 0.066), fold=True, cut=None, taper_sec=None, norm=True, ofnm=None):
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