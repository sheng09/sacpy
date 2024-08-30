#!/usr/bin/env python3

"""
This module implement the calculation of cross correlation and stacking.
"""

import numpy as np
import geomath as geomath
from numba import jit, float64

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
        idx = int( round((distance-dist_min) / dist_step) )
        return idx


@jit(nopython=True, nogil=True)
def zne2znert(zne_mat, znert_mat, stlo, stla, evlo, evla):
    """
    Convert the ZNE matrix to ZNERT matrix. The RT here refers to event-receiver R and T.
    In other words, the input matrix has rows of ZNEZNEZNE... and the output matrix has rows of ZNERTZNERTZNERT...
    """
    baz = azimuth(stlo, stla, evlo, evla)
    ang = np.deg2rad( 90- (baz+180) )
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
def gcd_daz_selection(n, stlo, stla, evlo, evla, gcd_min, gcd_max, daz_min, daz_max, selection_mat_nn):
    """
    """
    #### obtain the a NxN matrix with values of 1 or 0 for selecting the pairs of stations or not
    az = azimuth(evlo, evla, stlo, stla)
    for ista1 in range(n):
        stlo1, stla1, az1 = stlo[ista1], stla[ista1], az[ista1]
        for ista2 in range(ista1, n):
            stlo2, stla2, az2 = stlo[ista2], stla[ista2], az[ista2]
            gcd = abs(point_distance_to_great_circle_plane(evlo, evla, stlo1, stla1, stlo2, stla2))
            # the below makes:
            # daz = daz (0<=daz<90), 180-daz (90<=daz<180), daz-180(180<=daz<270), 360-daz(270<=daz<360)
            daz = abs(az1-az2)%180
            daz = min(daz, 180-daz)
            # avoid if ... else ...
            #if gcd_min <= gcd <= gcd_max and daz_min <= daz <= daz_max else const_uint64_zero
            v = (gcd_min <= gcd <= gcd_max) and (daz_min <= daz <= daz_max)
            selection_mat_nn[ista1, ista2] = v
            selection_mat_nn[ista2, ista1] = v
@jit(nopython=True, nogil=True)
def get_stack_index_mat(n, stlo, stla, dist_min, dist_step, stack_index_mat_nn):
    """
    """
    for ista1 in range(n):
        stlo1, stla1 = stlo[ista1], stla[ista1]
        for ista2 in range(ista1, n):
            stlo2, stla2 = stlo[ista2], stla[ista2]
            inter_rcv_dist = haversine(stlo1, stla1, stlo2, stla2)
            istack = stack_bin_index(inter_rcv_dist, dist_min, dist_step)
            stack_index_mat_nn[ista1, ista2] = istack
            stack_index_mat_nn[ista2, ista1] = istack
@jit(nopython=True, nogil=True)
def cc_stack(ch1, ch2, mat_spec, stlo, stla, stack_mat_spec, stack_count, stack_index_mat_nn, selection_mat_nn):
    """
    ch1, ch2: 0 for Z
              1 for inter-receiver R IRR
              2 for inter-receiver T IRT
              3 for N
              4 for E
              5 for event-receiver R ERR
              6 for event-receiver T ERT
    mat_spec: spectra for rows of Z,N,E,ERR,ERT,Z,N,E,ERR,ERT,Z,N,E,ERR,ERT,...
    """
    flag_IR = (ch1==1 or ch1==2 or ch2==1 or ch2==2)
    ncmp = 5
    n = mat_spec.shape[0]//ncmp
    nt = mat_spec.shape[1]
    ####
    irr1 = np.zeros(nt, dtype=np.complex64) # useless for ZNERT in fact
    irt1 = np.zeros(nt, dtype=np.complex64) # useless for ZNERT in fact
    irr2 = np.zeros(nt, dtype=np.complex64) # useless for ZNERT in fact
    irt2 = np.zeros(nt, dtype=np.complex64) # useless for ZNERT in fact
    ####
    #dtypes = set()
    for ista1 in range(n):
        stlo1, stla1 = stlo[ista1], stla[ista1]
        z1 = mat_spec[ista1*ncmp]
        n1 = mat_spec[ista1*ncmp+1]
        e1 = mat_spec[ista1*ncmp+2]
        err1 = mat_spec[ista1*ncmp+3]
        ert1 = mat_spec[ista1*ncmp+4]
        start = 0 if ch1 != ch2 else ista1
        for ista2 in range(start, n):
            stlo2, stla2 = stlo[ista2], stla[ista2]
            z2 = mat_spec[ista2*ncmp]
            n2 = mat_spec[ista2*ncmp+1]
            e2 = mat_spec[ista2*ncmp+2]
            err2 = mat_spec[ista2*ncmp+3]
            ert2 = mat_spec[ista2*ncmp+4]
            ####
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
                ang1 = np.deg2rad(90 - azimuth(stlo1, stla1, stlo2, stla2) )  # convert azimuth (clockwise from north) to angle (counter-clockwise from east)
                ang2 = np.deg2rad(90 - ( (azimuth(stlo2, stla2, stlo1, stla1) + 180.0) )  )
                c1, s1 = np.array(np.cos(ang1), dtype=np.float32), np.array(np.sin(ang1), dtype=np.float32)
                c2, s2 = np.array(np.cos(ang2), dtype=np.float32), np.array(np.sin(ang2), dtype=np.float32)
                irr1 = e1*c1 + n1*s1
                irt1 = e1*s1 - n1*c1
                irr2 = e2*c2 + n2*s2
                irt2 = e2*s2 - n2*c2
            #### cc
            #dtypes = dtypes.union( set( [str(it) for it in (z1.dtype, irr1.dtype, irt1.dtype, n1.dtype, e1.dtype, err1.dtype, ert1.dtype)] ) )
            s1 = [z1, irr1, irt1, n1, e1, err1, ert1]
            s2 = [z2, irr2, irt2, n2, e2, err2, ert2]
            ss = s1[ch1] * s2[ch2].conj() # of complex64 (8 bytes)
            ss *= selection_mat_nn[ista1, ista2]
            #ss.view(dtype=np.uint64)[:] #&= selection_mat_nn[ista1, ista2] # select or discard for 0xFFFFFFFFFFFFFFFF or 0
            #### stack
            istack = stack_index_mat_nn[ista1, ista2]
            stack_mat_spec[istack] += ss
            stack_count[istack] += selection_mat_nn[ista1, ista2] # select or discard for 1 or 0, respectively
    #return(dtypes)
def plot_stack_mat(stack_mat, stack_count, stack_bins, freq_band, title=None, fig=None, ax=None, figname=None):
    """
    Plot the stack matrix
    """
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111)
    #
    nf = stack_mat_spec.shape[1]
    freqs = np.fft.rfftfreq(nf*2, d=1.0)
    idxs = np.where( (freqs>=freq_band[0]) & (freqs<=freq_band[1]) )[0]
    freqs = freqs[idxs]
    spec = np.abs(stack_mat_spec).sum(axis=0)
    spec = spec[idxs]
    ax.plot(freqs, spec)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    if title:
        ax.set_title(title)
    return fig, ax

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
                 dist_range=(0.5, 179.5), dist_step=1.0,
                 wtlen=None, wt_f1=None, wt_f2=None, wflen=None, whiten_taper_halfsize=50,
                 daz_range=None, gcd_range=None, log_print=None,
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
        if log_print is None:
            if mpi_comm is  None:
                log_print = print
            else:
                log_print = MPILogger(mpi_comm, './mpi_log')
        ###
        log_print(-1, "Start")
        if mpi_comm is not None:
            log_print(-1, 'mpi_rank=%d, mpi_size=%d' % (mpi_comm.Get_rank(), mpi_comm.Get_size() ) )
        ### 
        self.ch_pair_type = ch_pair_type
        log_print(-1, 'Init settings' )
        log_print( 1, 'ch_pair_type:       ', self.ch_pair_type, '(Note: 0:Z; 1:IRR; 2:IRT; 3:N; 4:E; 5:ERR; 6:ERT)')
        #
        self.stack_bins = StackBins(dist_range[0], dist_range[1], dist_step)
        log_print( 1, 'stack_bins:         ', dist_range, dist_step )
        log_print( 1, 'stack_bins.centers: ', self.stack_bins.centers[0], '...', self.stack_bins.centers[-1])
        log_print( 1, 'stack_bins.edges:   ', self.stack_bins.edges[0],   '...', self.stack_bins.edges[-1])
        ####
        idx0 = 0
        if cut_time_range: # derive nt from cut_time_range
            nt = int(np.ceil((cut_time_range[1]-cut_time_range[0])/delta) )+1
            idx0 = int(cut_time_range[0]/delta)
        nt = nt+nt%2
        self.cut_idx_range  = (idx0, idx0+nt)
        self.cut_time_range = ( idx0*delta, (idx0+nt)*delta )
        ####
        self.nt = nt
        self.delta = delta
        self.cc_fftsize = self.nt*2
        self.nf = self.cc_fftsize//2 + 1
        self.stack_mat = None
        self.cc_time_range = None
        log_print( 1, 'delta:      ', self.delta)
        log_print( 1, 'original cut_time_range: ', cut_time_range)
        log_print( 1, 'adjusted cut_time_range: ', self.cut_time_range)
        log_print( 1, 'adjusted cut_idx_range:  ', self.cut_idx_range)
        log_print( 1, 'nt:         ', self.nt)
        log_print( 1, 'cc_fftsize: ', self.cc_fftsize)
        log_print( 1, 'nf:         ', self.nf)
        ###
        #self.stack_mat      = np.zeros( (self.stack_bins.centers.size, self.cc_fftsize-1),  dtype=np.float32)
        self.stack_mat_spec = np.zeros( (self.stack_bins.centers.size, self.nf ), dtype=np.complex64)
        self.stack_count    = np.zeros(  self.stack_bins.centers.size, dtype=np.int32)
        log_print( 1, 'stack_mat_spec: ', self.stack_mat_spec.shape, self.stack_mat_spec.dtype)
        log_print( 1, 'stack_count:    ', self.stack_count.shape, self.stack_count.dtype)
        ###
        self.wtlen = wtlen
        self.wt_f1 = wt_f1
        self.wt_f2 = wt_f2
        self.wflen = wflen
        self.whiten_taper_halfsize = whiten_taper_halfsize
        log_print( 1, 'wtlen:                 ', self.wtlen, self.wt_f1, self.wt_f2)
        log_print( 1, 'wflen:                 ', self.wflen)
        log_print( 1, 'whiten_taper_halfsize: ', self.whiten_taper_halfsize)
        ###
        self.daz_range = daz_range
        self.gcd_range = gcd_range
        log_print( 1, 'daz_range: ', self.daz_range)
        log_print( 1, 'gcd_range: ', self.gcd_range, flush=True)
        ###
        self.init_timepoint = time_time()
        self.log_print = log_print
    def add(self, zne_mat_f32, stlo, stla, evlo, evla, infostr=None):
        """
        """
        t0 = time_time()
        ####
        log_print = self.log_print
        log_print(-1, 'Adding... %s' % infostr)
        log_print( 1, 'Function variables:')
        log_print( 2, 'zne_mat_f32: ', zne_mat_f32.shape, zne_mat_f32.dtype)
        log_print( 2, 'stlo:        ', stlo.shape, stlo.dtype)
        log_print( 2, 'stla:        ', stla.shape, stla.dtype)
        log_print( 2, 'evlo:        ', evlo.shape, evlo.dtype)
        log_print( 2, 'evla:        ', evla.shape, evla.dtype)
        #### get ZNERT if necessary, the RT here refers to event-receiver R and T (ERR, ERT)
        log_print(1, 'Cutting and making ZNERT matrix from the ZNE...')
        tmp = time_time()
        idx0, idx1 = self.cut_idx_range
        idx0 = max(idx0, 0)
        idx1 = min(idx1, zne_mat_f32.shape[1])
        znert_mat_f32 = np.zeros( (zne_mat_f32.shape[0]//3*5, idx1-idx0), dtype=np.float32)
        zne2znert(zne_mat_f32[:, idx0:idx1], znert_mat_f32, stlo, stla, evlo, evla)
        t_zne2rt = time_time() - tmp
        log_print(2, 'Finished.')
        log_print(2, 'znert_mat_f32: ', znert_mat_f32.shape, znert_mat_f32.dtype)
        #### whiten the input data in time series
        log_print(1, 'Whitening the ZNERT matrix...')
        tmp = time_time()
        if self.wtlen:
            for xs in znert_mat_f32:
                if xs.max() > xs.min():
                    tnorm_f32(xs,   self.delta, self.wtlen, self.wt_f1, self.wt_f2, 1.0e-5, self.whiten_taper_halfsize)
        t_wt = time_time() - tmp
        tmp = time_time()
        if self.wflen:
            for xs in znert_mat_f32:
                if xs.max() > xs.min():
                    fwhiten_f32(xs, self.delta, self.wflen, 1.0e-5, self.whiten_taper_halfsize)
        t_wf = time_time() - tmp
        log_print(2, 'Finished.')
        #### remove Nan of Inf
        log_print(1, 'Zeroing the stations with any Nan of Inf in any of ZNERT...')
        not_valid = lambda xs: np.any( np.isnan(xs) ) or np.any( np.isinf(xs) ) or (np.max(xs) == np.min(xs) )
        invalid_stas = [idx//5 for idx, xs in enumerate(znert_mat_f32) if not_valid(xs) ]
        invalid_stas = list(set(invalid_stas) )
        for ista in invalid_stas:
            znert_mat_f32[ista*5,   :] = 0.0
            znert_mat_f32[ista*5+1, :] = 0.0
            znert_mat_f32[ista*5+2, :] = 0.0
            znert_mat_f32[ista*5+3, :] = 0.0
            znert_mat_f32[ista*5+4, :] = 0.0
        log_print(2, 'Finished.')
        #### fft
        log_print(1, 'FFTing...')
        tmp = time_time()
        #spectra_c64 = np.fft.rfft(znert_mat_f32, self.cc_fftsize, axis=1) # spectra has (n, cc_fftsize//2+1)
        spectra_c64 = rfft(znert_mat_f32, self.cc_fftsize, axis=1) # spectra has (n, cc_fftsize//2+1)
        t_fft = time_time() - tmp
        log_print(2, 'Finished.')
        log_print(2, 'spectra_c64: ', spectra_c64.shape, spectra_c64.dtype)
        #### selection
        log_print(1, 'Selecting and weighting(not implemented)...')
        tmp = time_time()
        selection_mat = np.zeros( (stlo.size, stlo.size), dtype=np.int8) + 1
        if self.gcd_range or self.daz_range:
            gcd_min, gcd_max = self.gcd_range if self.gcd_range else (-1, 361) # (-1, 361) means to select all
            daz_min, daz_max = self.daz_range if self.daz_range else (-1, 361) # (-1, 361) means to select all
            gcd_daz_selection(stlo.size, stlo, stla, evlo[0], evla[0], gcd_min, gcd_max, daz_min, daz_max, selection_mat)
        for ista in invalid_stas:
            selection_mat[ista,:] = 0
            selection_mat[:,ista] = 0
        t_selection_weighting = time_time() - tmp
        log_print(2, 'Finished.')
        log_print(2, 'selection_mat: ', selection_mat.shape, selection_mat.dtype)
        #### stack index
        log_print(1, 'Computing the stack index...')
        stack_index_mat_nn = np.zeros( (stlo.size, stlo.size), dtype=np.uint32)
        dist_min, dist_step = self.stack_bins.centers[0], self.stack_bins.step
        get_stack_index_mat(stlo.size, stlo, stla, dist_min, dist_step, stack_index_mat_nn)
        log_print(2, 'Finished.')
        log_print(2, 'stack_index_mat_nn: ', stack_index_mat_nn.shape, stack_index_mat_nn.dtype)
        #### cc & stack
        tmp = time_time()
        log_print(1, 'cc&stack...')
        ch1, ch2 = self.ch_pair_type
        stack_count = self.stack_count
        stack_mat_spec = self.stack_mat_spec
        #print(spectra_c64.dtype, stack_mat_spec.dtype, stack_count.dtype, stack_index_mat_nn.dtype, selection_mat.dtype, flush=True)
        cc_stack(ch1, ch2, spectra_c64, stlo, stla, stack_mat_spec, stack_count, stack_index_mat_nn, selection_mat)
        t_ccstack = time_time() - tmp
        log_print(2, 'Finished.')
        t_total = time_time() - t0
        t_other = t_total - t_wt - t_wf - t_ccstack - t_selection_weighting - t_zne2rt - t_fft
        log_print(1, 'Total time lapse(%.2fs) zne2rt(%.2fs)  wt(%.2fs) wf(%.2fs) fft(%.2fs) selection&weight(%.2fs) cc&stack(%.2fs) other(%.2fs)' % (t_total, t_zne2rt, t_wt, t_wf,  t_fft, t_selection_weighting, t_ccstack, t_other),  flush=True)
        ####
    def finish(self, filter_band=(0.02, 0.066), fold=True, cut=None, taper_sec=None, norm=True, ofnm=None):
        """
        """
        cc_fftsize = self.cc_fftsize
        stack_count = self.stack_count
        stack_mat_spec = self.stack_mat_spec
        ####
        #stack_mat = np.fft.irfft(stack_mat_spec, cc_fftsize, axis=1).astype(np.float32)
        stack_mat = irfft(stack_mat_spec, cc_fftsize, axis=1)
        ####
        nt = self.nt
        delta = self.delta
        rollsize = nt-1
        stack_mat = np.roll(stack_mat, rollsize, axis=1)
        stack_mat = stack_mat[:,:-1] #get rid of the zero point
        cc_t1, cc_t2 = -rollsize*delta, rollsize*delta
        ####
        if filter_band is not None:
            f1, f2 = filter_band
            for xs in stack_mat:
                iirfilter_f32(xs, delta, 0, 2, f1, f2, 2, 2)
        ####
        if fold:
            stack_mat += stack_mat[:, ::-1]
            stack_mat = stack_mat[:, rollsize:]
            cc_t1, cc_t2 = 0, rollsize*delta
        ####
        if cut is not None:
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
        ####
        if taper_sec:
            post_taper_halfsize = int(taper_sec/delta)
            for xs in stack_mat:
                taper(xs, post_taper_halfsize)
        ####
        if norm:
            for xs in stack_mat:
                v = xs.max()
                if v> 0.0:
                    xs *= (1.0/v)
        ####
        if ofnm:
            CS_InterRcv.mkdirs(ofnm)
            with h5_File(ofnm, 'w') as fid:
                fid.attrs['cc_time'] = (cc_t1, cc_t2, delta)
                fid.attrs['stack_bin_centers'] = self.stack_bins.centers
                fid.attrs['stack_bin_edges'] = self.stack_bins.edges
                fid.attrs['filter_band'] = filter_band
                fid.create_dataset('dat', data=stack_mat, dtype=np.float32)
                fid.create_dataset('stack_count', data=stack_count, dtype=np.int32)
        ####
        self.stack_mat = stack_mat
        self.cc_time_range = (cc_t1, cc_t2)
        ####
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
        log_print = self.log_print
        ####
        mpi_comm.Barrier()
        mpi_rank = mpi_comm.Get_rank()
        mpi_ncpu = mpi_comm.Get_size()
        ####
        log_print(-1, 'Time lapse for all adding (%.2fs)' % (time_time()-self.init_timepoint) )
        log_print(-1, 'MPI finish on rank0...')
        stack_mat_spec = self.stack_mat_spec
        stack_count    = self.stack_count
        global_stack_mat_spec = stack_mat_spec
        global_stack_count    = stack_count
        if mpi_rank == 0:
            global_stack_mat_spec = np.zeros(stack_mat_spec.shape, dtype=stack_mat_spec.dtype )
            global_stack_count    = np.zeros(stack_count.shape,    dtype=stack_count.dtype )
        mpi_comm.Reduce([stack_mat_spec, MPI.C_FLOAT_COMPLEX], [global_stack_mat_spec, MPI.C_FLOAT_COMPLEX], MPI.SUM, root= 0)
        mpi_comm.Reduce([stack_count, MPI.INT32_T], [global_stack_count, MPI.INT32_T], MPI.SUM, root= 0 )
        mpi_comm.Barrier()
        ####
        log_print(1, 'Finished and return results')
        ####
        if mpi_rank == 0:
            self.stack_mat_spec = global_stack_mat_spec
            self.stack_count    = global_stack_count
            return self.finish(filter_band=filter_band, fold=fold, cut=cut, taper_sec=taper_sec, norm=norm, ofnm=ofnm)
        else:
            return None
    def plot(self, fignm='test.png', figsize=(4, 8), vmin=-0.6, vmax=0.6, cmap='gray', color='#999999', title=''):
        """
        """
        log_print = self.log_print
        log_print(-1, 'Plot')
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
    @staticmethod
    def mkdirs(filename):
        if filename:
            folder = '/'.join(filename.split('/')[:-1])
            if not os.path.exists(folder):
                os.makedirs(folder)

if __name__ == "__main__":
    bins = StackBins(0, 10, 1)
    print(bins.centers)
    print(bins.edges)
    print(bins.get_idx(2.5))
    print(bins.get_idx(3.5))
    print(bins.get_idx(4.5))
    print(bins.get_idx(5.5))
    print(bins.get_idx(6.5))