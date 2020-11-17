#!/usr/bin/env python3
"""
Executable files for cross-correlation and stacking operations
"""
from numpy.core.numeric import roll
from operator import pos
from sacpy.processing import filter, taper
from sacpy.sac import rd_sac_2
import time
import sacpy.sac as sac
import sacpy.processing as processing
import sacpy.geomath as geomath
import mpi4py.MPI
import h5py
import pyfftw
import numpy as np
import sys
import getpass
import getopt
from glob import glob
import scipy.signal as signal
from numba import jit
import pyfftw


def main():
    """
    The main function.

    The run consists of
    - 1. reading sac files and pre-processing;
    - 2. optional whitening;
    - 3. cross-correlation and stacking;
    - 4. post-processing of correlation stacks.

    """
    ### 1. parameters for reading sac files and pre-processing
    fnm_wildcard = 'test_vol/20100509_055941.a/processed/*.BHZ'
    tmark, t1, t2 = '0', 10800, 32400
    delta = 0.1
    pre_detrend, pre_taper_ratio = True, 100.0/(6*3600)
    pre_filter = None
    ### dependent parameters
    sampling_rate = 1.0/delta
    npts = int( np.round((t2-t1)/delta) ) + 1
    fftsize = npts * 2
    nrfft = fftsize // 2 + 1
    df = 1.0/(delta*fftsize)
    taper_size = int(npts*pre_taper_ratio)
    ### 2. parameters for optional whitening
    wt = 128.0 # sec
    wt_size = (int(np.round(wt / delta) ) //2 ) * 2 + 1
    wt_f1, wt_f2 = 0.02, 0.06666
    wf = 0.02  # Hz
    wf_size = (int(np.round(wf / df) ) //2 ) * 2 + 1
    ### 3. cross-correlation stack parameters
    dist_range = (0.0, 180.0)
    dist_step  = 1.0
    dist = np.arange(dist_range[0], dist_range[1] + dist_step )
    spec_stack_mat = np.zeros( (dist.size, nrfft), dtype= np.complex64 )
    stack_mat = np.zeros((dist.size, npts), dtype=np.float32 )
    ### 4. post-processing
    post_filter = ('bandpass', 0.02, 0.066666)
    cc_index_range = get_bound(fftsize, sampling_rate, post_filter[1], post_filter[2], 0.01)
    #######################################################################################################################################
    ### run
    #######################################################################################################################################
    directories  = sorted( glob('/'.join(fnm_wildcard.split('/')[:-1] ) ) )
    sac_wildcards = fnm_wildcard.split('/')[-1]
    for it in directories:
        wildcards = '%s/%s' % (it, sac_wildcards)
        ### 1. read and pre-processing
        mat, sampling_rate, stlo, stla, evlo, evla, az, baz = rd_preproc_single(wildcards, tmark, t1, t2, pre_detrend, pre_taper_ratio, pre_filter)
        ### 2. whitening
        whitened_spectra_mat = whiten_spec(mat, sampling_rate, wt_size, wt_f1, wt_f2, wf_size, fftsize, taper_size)
        ### 3.1 cc and stack
        ccstack(whitened_spectra_mat, stlo, stla, spec_stack_mat, dist_step, cc_index_range, dist_range[0] )
    ### 3.2 ifft
    rollsize = npts-1
    for irow in range(dist.size):
        x = pyfftw.interfaces.numpy_fft.irfft(spec_stack_mat[irow], fftsize)
        x.roll(rollsize )


    ###
    #
    ###
    
    pass

def mpi_print_log(msg, n_pre, file=sys.stdout, flush=False):
    print('%s%s' % ('\t'*n_pre, msg), file= file, flush= flush )


def run_single(sacfnm_template, tmark, t1, t2, delta, wt= 128.0, f1=0.02, f2=0.06666, wf=0.02 ):
    ###
    npts, fftsize, pre_proc_window = init_preproc_parameter(t1, t2, delta, 0.005)
    ###
    mat, sampling_rate, stlo, stla, evlo, evla, az, baz = rd_preproc_single(sacfnm_template, tmark, t1, t2, True, 0.005, None)
    ###
    wnd_size_t = (int(np.round(wt/delta) )//2) * 2 + 1
    df = 1.0/(delta * fftsize)
    wnd_size_freq = (int(wf/df) // 2) * 2 + 1
    spec_mat = whiten_spec(mat, sampling_rate, wnd_size_t, f1, f2, wnd_size_freq, fftsize)
    ###

def init_preproc_parameter(t1, t2, delta, taper_ratio= 0.005):
    """
    Init parameter for preprocessing, cross-correlation, stacking, and postprocessing.

    t1, t2, delta: parametering for reading sac files.
    taper_ratio:   taper ratio between 0.0 and 0.5.

    Return (npts, fftsize, pre_proc_window).
    npts: npts of input time series
    fftsize: fftsize will be used in fft for cross-correlation.
    pre_proc_window: the time window will be used in and after pre-processing.
    """
    npts = np.round((t2-t1)/delta) + 1
    fftsize = npts * 2 # in fact it should be `npts*2-1`. However here we use EVEN fftsize to speed-up fft.
    #
    pre_proc_window = processing.tukey_jit(npts, taper_ratio)
    return npts, fftsize, pre_proc_window

def init_ccstack(dist_range, dist_step, fftsize):
    """
    Init cross-correlation and stacking.
    dist_range : inter-distance range.
    dist_step  : inter-distance step.
    fftsize    : fft size.

    Return (dist, spec_stack_mat).
    dist: a list of inter-distances.
    spec_stack_mat: the matrix for stacking spectra.
    """
    nrfft = fftsize // 2 + 1
    dist = np.arange(dist_range[0], dist_range[1]+dist_step, dist_step )
    nrow = dist.size
    spec_stack_mat = np.zeros((nrow, nrfft), dtype=np.complex64 )
    return dist, spec_stack_mat

def rd_preproc_single(sacfnm_template, tmark, t1, t2, detrend= True, taper_ratio= 0.005, filter=None ):
    """
    Read and preproc many sacfiles given a filename template `sacfnm_template`.
    Those sac files can be recorded at many receivers for the same event, or they
    can be recorded at the same receiver from different events.

    sacfnm_template: The filename template that contain wildcards. The function with use
                     `glob(sacfnm_template)` to obtain all sac filenames.
    tmark, t1, t2:   The cut time window to read sac time series.
    detrend:         Set True (default) to enable detrend after reading.
    taper_ratio:     Set taper ratio (0~0.5) to apply taper after the optional detrend.
                     Default taper ratui is 0.005.
    filter:          Filter all the time series. Set `None` to disable filter.
                     The filter can be ('bandpass', f1, f2), ('lowpass', fc, meaningless_f) or
                     ('highpass', fc, meaningless_f).

    Return: (mat, sampling_rate, stlo, stla, evlo, evla, az, baz)
    """
    ### read
    fnms = sorted( glob(sacfnm_template) )
    tr  = [rd_sac_2(it, tmark, t1, t2, True) for it in fnms]
    ### important geomeotry values
    stlo = np.array( [it['stlo'] for it in tr] )
    stla = np.array( [it['stla'] for it in tr] )
    evlo = np.array( [it['evlo'] for it in tr] )
    evla = np.array( [it['evla'] for it in tr] )
    az   = np.array( [it['az']  for it in tr] )
    baz  = np.array( [it['baz'] for it in tr] )
    ### important parameters
    delta = tr[0]['delta']
    sampling_rate = 1.0/delta
    npts = tr[0].dat.size
    nsac = len(tr)
    ### preprocessing
    if detrend:
        for it in tr:
            it.detrend()
    if taper_ratio > 1.0e-5:
        w = processing.tukey_jit(npts, taper_ratio)
        for it in tr:
            it.dat *= w
    if filter != None:
        btype, f1, f2 = filter
        for it in tr:
            processing.filter(it.dat, sampling_rate, btype, (f1, f2), 2, 2)
    ### form the mat
    mat  = np.zeros((nsac, npts), dtype=np.float32 )
    for irow in range(nsac):
        mat[irow] = tr[irow].dat
    ###
    return mat, sampling_rate, stlo, stla, evlo, evla, az, baz

def whiten_spec(mat, sampling_rate, wnd_size_t, f1, f2, wnd_size_freq, fftsize, taper_length= 0):
    """
    Apply #1 temporal normalizatin and #2 spectral whitening to time series.

    mat:           The time series to be processed. Each row is a single time series.
    sampling_rate: Sampling rate in Hz.
    wnd_size_t:    An odd integer that declares the moving time window size in temporal normalization.
                   Set `None` to disable temporal normalization
    f1, f2:        The frequency band used in temporal normalization. The input time series will be
                   bandpassed in the band (f1, f2) to compute the weight.
    wnd_size_freq: And odd integer that delcares the moving window size in frequency domain in
                   spectral whitening.
                   Set `None` to disable spectral whitening.
    fftsize:       Since this function would return spectra of the whitened time series. The `fftsize`
                   will be used in generating the spectra. The spectra length is `fftsize//2+1` since
                   we use `rfft`.

    Return:        mat_spec

    mat_spec:      The spectra of whitened time series.
    """
    #### The return mat
    nrow  = mat.shape[0]
    nrfft = fftsize // 2 + 1
    spec_mat = np.zeros((nrow, nrfft), dtype=np.complex64 )
    #### temporal normlization
    if wnd_size_t != None:
        for irow in range(nrow):
            mat[irow] = processing.temporal_normalize(mat[irow], sampling_rate, wnd_size_t, f1, f2, 1.0e-5, taper_length)
    #### spectral whitening
    if wnd_size_freq != None:
        for irow in range(nrow):
            spec_mat[irow] = processing.frequency_whiten_spec(mat[irow], wnd_size_freq, fftsize, 1.0e-5)
    else:
        for irow in range(nrow):
            spec_mat[irow] = pyfftw.interfaces.numpy_fft.rfft(mat[irow], fftsize)
    ####
    return spec_mat

def get_bound(fftsize, sampling_rate, f1, f2, critical_level= 0.01):
    x = np.zeros(fftsize)
    x[0] = 1.0
    x = processing.filter(x, sampling_rate, 'bandpass', (f1, f2), 2, 2)
    s = pyfftw.interfaces.numpy_fft.rfft(x, fftsize)
    amp = np.abs(s)
    c = np.max(amp) * critical_level
    i1 = np.argmax(amp>c)
    i2 = np.argmin(amp[i1:]>c) + i1 + 1
    return i1, i2

@jit(nopython=True, nogil=True)
def ccstack(spec_mat, stlo_lst, stla_lst, stack_mat, dist_step, index_range, dist_start=0.0):
    """
    """
    i1, i2 = index_range
    nsac = spec_mat.shape[0]
    for isac1 in range(nsac):
        stlo1, stla1 = stlo_lst[isac1], stla_lst[isac1]
        spec1 = spec_mat[isac1]
        for isac2 in range(isac1, nsac):
            stlo2, stla2 = stlo_lst[isac2], stla_lst[isac2]
            spec2 = spec_mat[isac2]
            ###
            dist = geomath.haversine(stlo1, stla1, stlo2, stla2)
            idx = int( np.round((dist-dist_start) / dist_step) )
            w   = np.conj(spec1[i1:i2]) * spec2[i1:i2]
            stack_mat[idx][i1:i2] += w

def ifft(spec_mat, fftsize, bp_range=None ):
    """
    """
    nrow = spec_mat.shape[0]
    ncol = (fftsize-1)//2 + 1
    mat = np.zeros((nrow, ncol), dtype=np.float32 )
    rollsize = fftsize//2-1

    w = processing.tukey_jit(ncol, 0.02)

    for irow in range(nrow):
        tmp = pyfftw.interfaces.numpy_fft.irfft( spec_mat[irow], fftsize )
        tmp = np.roll(tmp, rollsize)[:-1]
        tmp += tmp[::-1]
        mat[irow] = tmp[ncol-1:] * w
        ####
        v = np.max(mat[irow] )
        if v > 0.0:
            mat[irow] *= (1.0/ v)
    if bp_range != None:
        mat[irow] = processing.filter(mat[irow], 10, 'bandpass', (0.02, 0.06), 2, 2)
    return mat

def run_test():
    ###
    #
    ###
    delta = 0.1
    t1, t2 = 10800, 32400
    start_time = time.time()
    mat, sampling_rate, stlo, stla, evlo, evla, az, baz = rd_preproc_single('test_vol/20100509_055941.a/processed/*.BHZ', '0', t1, t2)
    elapsed_time = time.time() - start_time
    print('rd', elapsed_time, 'sec', mat.shape )

    ###
    #
    ###
    start_time = time.time()
    npts = int((t2-t1)/delta) + 1
    fftsize = npts * 2

    
    elapsed_time = time.time() - start_time
    print('whiten', elapsed_time, 'sec', spec_mat.shape )

    ###
    #
    ###
    dist_range = (0, 180)
    dist_step = 1.0
    dist, spec_stack_mat = init_ccstack_spec(dist_range, dist_step, fftsize)

    ###
    #
    ###
    #print(spec_mat.shape, type(spec_mat), type(spec_mat[0,0]),  spec_stack_mat.shape, type(spec_stack_mat), type(spec_stack_mat[0,0]) )
    start_time = time.time()
    ccstack_same_event(spec_mat, stlo, stla, spec_stack_mat, dist_step, 0.01, 0.2, delta, fftsize, dist[0] )
    elapsed_time = time.time() - start_time
    print('ccstack', elapsed_time, 'sec', spec_stack_mat.shape )

    ###
    #
    ###
    start_time = time.time()
    ccmat = ifft(spec_stack_mat, fftsize)
    elapsed_time = time.time() - start_time
    print('ifft', elapsed_time, 'sec', ccmat.shape )


    ###
    #
    ###
    print( np.max(ccmat), np.min(ccmat) )
    import matplotlib.pyplot as plt
    plt.imshow(ccmat, aspect='auto', extent=(0, 21600, 0, 180), origin='lower', cmap='gray', vmin=-0.9, vmax=0.9 )
    plt.xlim([100, 3000])
    plt.show()

run_test()


class cc_stack_sac_worker:
    """
    The input includes 
    1) a list of sac files, from which each pair has a cross-correlation;
    2) whitening parameters;
    3) stack distance range and stack distance step;
    4) optional selection range for daz;
    """
    def __init__(self, fnmlst, new_npts, new_dt=0.1, log_fid=sys.stdout, tmark=None, 
        t1=None, t2=None, temporal_window_len=128, temporal_f1_Hz=0.02, temporal_f2_Hz=0.066667, spectral_window_len=0.02,
        dist_min=0.0, dist_max=180.0, dist_step=1.0, daz_min=-999, daz_max=999 ):
        """
        fnmlst: a list of sac filenames
        new_npts: all time series will be aligned to `npts`. 
            This pad zeros if input a sac time series has insufficient npts.
            Please nots, this is different from zero padding in fft.
        new_dt: dt for resampling.
        log_fid: file id for logging.
        """
        ###
        #self.select_stack_msg_template = '%2s cc ev(%.2f %.2f) rcv(%.2f %.2f)x(%.2f %.2f) dist(%.2f) stack_cc(%d,%.2f) az(%f,%f) daz(%f), gc(%f)'
        ###
        self.d_log = log_fid
        self.__obtain_aligned_zero_padded_time_seires(fnmlst, new_npts, new_dt, tmark, t1, t2)
        self.__compute_whitened_spectra__(temporal_window_len, temporal_f1_Hz, temporal_f2_Hz, spectral_window_len)
        self.__cc_stack_rcvpair__(dist_min, dist_max, dist_step, daz_min, daz_max)
    def get_cc_stacked(self, type='f'):
        """
        type: 'f' for stacked spectra, and 't' for stacked time series.
        """
        if type == 't':
            self.__ifft__()
            return self.cc_stacked, self.d_cc_stacked_count
        else:
            return self.d_cc_stacked_spectra, self.d_cc_stacked_count
    def __obtain_aligned_zero_padded_time_seires(self, fnmlst, aligned_npts, new_dt= 0.1, tmark=None, t1=None, t2=None):
        """
        Read sac files, check if values are valid, and pad zeros if a `npts` is not big enough.
        """
        ### read
        st = None
        if (tmark!=None and t1!=None and t2!=None):
            st = [sac.rd_sac_2(it, tmark, t1, t2) for it in fnmlst]
        else:
            st = [sac.rd_sac(it) for it in fnmlst]
        dt = st[0]['delta']
        ### check
        for it in st:
            try:
                it.resample(new_dt)
            except:
                it.set_zero()
            if it.is_nan_inf():
                print("Warning. Nan or Inf values in %s. Time series set to ZEROs." % it['filename'], file=self.d_log, flush=True )
                it.set_zero()
        ### pad zeros or cut points
        nrow = len(st)
        mat = np.zeros((nrow, aligned_npts), dtype=np.float32 )
        for irow in range(nrow):
            min_npts = st[irow].dat.size if aligned_npts > st[irow].dat.size else aligned_npts
            mat[irow, :min_npts] = st[irow].dat[:min_npts]
        ### cross-correlation parameters
        self.d_dt         = dt
        self.d_cc_npts    = aligned_npts*2-1
        #self.d_nrfft = self.d_cc_npts // 2 + 1
        self.d_df = 1.0 / (self.d_cc_npts* self.d_dt)
        self.d_mat_in = mat
        ### compute event-station geometry paramters
        self.d_evst = {
            'stlo': np.array([it['stlo'] for it in st], dtype=np.float32 ),
            'stla': np.array([it['stla'] for it in st], dtype=np.float32 ),
            'evlo': np.array([it['stlo'] for it in st], dtype=np.float32 ),
            'evla': np.array([it['stlo'] for it in st], dtype=np.float32 )  }
        self.d_evst['gcarc'] = [ geomath.haversine(evlo, evla, stlo, stla) for stlo, stla, evlo, evla in 
                                    zip(self.d_evst['stlo'], self.d_evst['stla'], self.d_evst['evlo'], self.d_evst['evla'] ) ]
        self.d_evst['az']    = [ geomath.azimuth(evlo, evla, stlo, stla) for stlo, stla, evlo, evla in 
                                    zip(self.d_evst['stlo'], self.d_evst['stla'], self.d_evst['evlo'], self.d_evst['evla'] ) ]
        ###
    def __compute_whitened_spectra__(self, temporal_window_len=128, temporal_f1_Hz=0.02, temporal_f2_Hz=0.066667, spectral_window_len=0.02 ):
        """
        Compute whitened spectra for all time-series. Zero padding are used in computing the spectra to easa cross-correlation.
        """
        twin_len = int( np.ceil(temporal_window_len / self.d_dt) )
        # fwin_len = int( np.ceil(spectral_window_len / self.d_df) )
        df = 1.0/ (self.d_dt * self.d_mat_i.shape[1])
        fwin_len = int( np.ceil(spectral_window_len / self.df) )
        
        
        sampling_rate = 1.0/self.d_dt
        nsac, junk = self.d_mat_in.shape
        self.d_nrfft = self.d_cc_npts // 2 + 1
        self.d_spectra_in = np.zeros((nsac, self.d_nrfft), dtype=np.complex64)
        for irow in range(nsac):
            tr = processing.temporal_normalization(self.d_mat_in[irow], sampling_rate, twin_len, temporal_f1_Hz, temporal_f2_Hz )
            tr = processing.frequency_whiten(tr, sampling_rate, fwin_len)
            self.d_spectra_in[irow] = pyfftw.interfaces.numpy_fft.rfft(tr, self.d_cc_npts) # zero padded for fft for cross-correlation
            # self.d_spectra_in[irow] = processing.frequency_whiten_spec(tr, sampling_rate, fwin_len, self.d_cc_npts) # zero padded for fft
        ###
    def __cc_stack_rcvpair__(self, d0, d1, dstep=1.0, daz_min=-999, daz_max=999):
        """
        Cross-correlation, and stacked with respect to inter-receiver distance.
        """
        ### stack methods
        nbins = int( np.ceil((d1-d0)/dstep) ) + 1
        self.d_bins  = np.array( [d0+it*dstep for it in range(nbins) ], dtype=np.float32 )
        bin0 = self.d_bins - dstep*0.5
        bin1 = self.d_bins + dstep*0.5
        min_interdist = bin0[0]
        max_interdist = bin1[-1]
        ### stacked spectra
        self.d_cc_stacked_spectra = np.zeros((nbins, self.d_nrfft), dtype=np.complex64 )
        self.d_cc_stacked_count = np.zeros(nbins, dtype=np.int32)
        func_get_ibin = lambda x: int(round((x-d0)/dstep) )
        ### 
        nsac, junk = self.d_spectra_in.shape
        for first in range(nsac):
            stlo1, stla1, az1 = self.d_evst['stlo'][first], self.d_evst['stla'][first], self.d_evst['az'][first]
            for second in range(first, nsac):
                stlo2, stla2, az2 = self.d_evst['stlo'][second], self.d_evst['stla'][second], self.d_evst['az'][second]
                ### select interdist
                interdist = geomath.haversine(stlo1, stla1, stlo2, stla2)
                if (interdist < min_interdist or interdist >= max_interdist):
                    # mpi_print_log('ND %.2f' % (interdist), 2, self.local_log_fp, False )
                    continue
                # select daz
                daz = np.abs(az1-az2)
                daz = 180.0-daz if daz>90.0 else daz
                if (daz < daz_min or daz > daz_max):
                    # mpi_print_log('NA %.2f' % (interdist), 2, self.local_log_fp, False )
                    continue
                ###
                ibin = func_get_ibin(interdist)
                self.d_cc_stacked_spectra[ibin] += self.d_spectra_in[first] * np.conj(self.d_spectra_in[second] )
                self.d_cc_stacked_count[ibin] = self.d_cc_stacked_count[ibin] + 1
        ###
    def __ifft__(self):
        ###
        self.cc_stacked = np.zeros((nbins, self.d_cc_npts), dtype=np.float32)
        for ibin in range(nbins):
            self.cc_stacked[ibi] = pyfftw.interfaces.numpy_fft.irfft(self.d_cc_stacked_spectra[ibin], self.d_cc_npts)
        ###

class cc_stack_sac:
    def __init__(self, fnm_template, output_fnm_prefix, log_fnm_prefix="mpi_log",
            tmark= 'b', t0= 10800, t1= 32400, delta= 0.1,
            temporal_normalization_flag= True, temporal_window_sec= 128.0, temporal_bandpass_Hz= (0.02, 0.066667),
            spectral_whitening_flag= True, spectral_window_Hz= 0.02,
            inter_dist_range_deg=(0.0, 180.0), dist_step_deg=1.0,
            daz_range= (-0.01, 99.01),
            gcd_range= (-0.01, 99.01)    ):
        """
        fnm_template: The filename template for all sac files stored with respect to event.
            e.g. '/rootdir/selected_event_20*/processed/*.00.BHZ.sac'
            in which '/rootdir/selected_event_20*/processed' will be used to distinguish between events.
        output_fnm_prefix : Output filename prefix.
        tmark, t0, t1: time window to read the sac.
        temporal_window_sec, temporal_bandpass_Hz:
        spectral_window_Hz:
        inter_dist_range_deg, dist_step_deg: this will generate a list of dist by `np.arange(d0, d1+step, step)`.
        daz_range:
        gcd_range:
        """
        ### Init MPI
        self.comm = mpi4py.MPI.COMM_WORLD.Dup()
        self.rank = self.comm.Get_rank()
        self.ncpu = self.comm.Get_size()
        ### Obtain all events
        event_folder_name_template = '/'.join(fnm_template.split('/')[:-1])
        event_folder_lst = sorted(glob(event_folder_name_template) )
        global_njobs = len(event_folder_lst)
        self.single_sac_fnm_template = fnm_template.split('/')[-1]
        ### Distribute events among all procs
        local_njobs = (global_njobs // self.ncpu) if (global_njobs % self.ncpu != 0) else (global_njobs // self.ncpu + 1)
        idx0 = self.rank*local_njobs
        idx1 = idx0 + local_njobs
        idx1 = idx1 if idx1 < global_njobs else global_njobs
        local_njobs = idx1 - idx0
        self.local_event_folder_lst = event_folder_lst[idx0: idx1]
        ### Init reading parameters
        self.rd_tmark, self.rd_t0, self.rd_t1, self.rd_delta = tmark, t0, t1, delta
        self.rd_npts = np.round((t1-t0)/delta) + 1
        self.rd_sampling_rate = 1.0/delta
        self.df = 1.0/(1.0*self.rd_npts)

        ### Init ouput parameters
        self.output_fnm_prefix = output_fnm_prefix
        self.local_log_fnm     = '%s_%03d.txt' % (log_fnm_prefix, self.rank)
        self.local_log_fid     = open(self.local_log_fnm, 'w')


        ### Init whitening parameters
        self.temporal_normalization_flag = temporal_normalization_flag
        self.temporal_window_sec   = temporal_window_sec
        self.temporal_window_npts  = (np.round(temporal_window_sec/delta) // 2 ) * 2 + 1 # ODD
        self.temporal_bandpass_Hz  = temporal_bandpass_Hz

        self.spectral_whitening_flag = spectral_whitening_flag
        self.spectral_window_Hz    = spectral_window_Hz
        self.spectral_window_npts  = (np.round(spectral_window_Hz/df) // 2 ) * 2 +1 # ODD

        ### Init stacking list
        self.inter_dist_range_deg  = inter_dist_range_deg
        self.dist_step_deg         = dist_step_deg
        self.dist_lst              = np.arange(inter_dist_range_deg[0], inter_dist_range_deg[1]+dist_step_deg, dist_step_deg)

        ### Init selection method
        self.daz_range = daz_range
        self.gcd_range = gcd_range

        #msg = '>>> '
        #mpi_print_log(msg, 0, local_log_fid, True)
        ###
        #self.run(self.d_local_fnm_template_lst, new_npts,  output_fnm_prefix, new_dt, local_log_fid, 
        #    tmark, t1, t2, temporal_window_len, temporal_f1_Hz, temporal_f2_Hz,
        #    spectral_window_len, dist_min, dist_max, dist_step, daz_min, daz_max)
        ###

    def run(self):
        """
        """
        fftsize = self.rd_npts * 2
        nrfft = fftsize // 2 + 1
        nstack= len(self.dist_lst)
        self.local_spec_mat = np.zeros((nstack, nrfft), dtype= np.complex64 )
        for it in self.local_event_folder_lst:
            sacfnm_template = '%s/%s' % (it, self.single_sac_fnm_template)
            self.run_single_event(sacfnm_template)
    def run_single_event(self, sacfnm_template):
        """
        """
        mat, (evlo, evla), (stlo_lst, stla_lst, az_lst) = rd_preproc_sac_same_event(sacfnm_template, self.rd_tmark, self.rd_t0, self.rd_t1)
        spec_mat = whitening(mat, self.rd_sampling_rate, self.temporal_window_npts, self.temporal_bandpass_Hz[0], self.temporal_bandpass_Hz[1], self.spectral_window_npts, 
                    self.temporal_normalization_flag, self.spectral_whitening_flag)
        ccstack_same_event(spec_mat, stlo_lst, stla_lst, self.local_spec_mat, self.dist_step_deg)
    
    def run(self, fnm_template_lst, new_npts, output_fnm_prefix, new_dt=0.1, log_fid=sys.stdout,
            tmark=None, t1=None, t2=None, temporal_window_len=128, temporal_f1_Hz=0.02, temporal_f2_Hz=0.066667, spectral_window_len=0.02,
            dist_min=0.0, dist_max=180.0, dist_step=1.0, daz_min=-999, daz_max=999):
        """
        """
        single_cc_spectra = None
        single_cc_count   = None
        for fnm_template in fnm_template_lst:
            fnm_lst = sorted(glob(fnm_template) )
            app = cc_stack_sac_worker(fnm_lst, new_npts, new_dt, log_fid, tmark, t1, t2, 
                temporal_window_len, temporal_f1_Hz, temporal_f2_Hz, spectral_window_len,
                dist_min, dist_max, dist_step, daz_min, daz_max)
            if single_cc_spectra == None:
                single_cc_spectra, single_cc_count = app.get_cc_stacked('f')
            else:
                tmp1, tmp2 = app.get_cc_stacked('f')
                single_cc_spectra += tmp1
                single_cc_count   += tmp2
                del tmp1
                del tmp2
        ### reduce stack
        nbins = single_cc_count.size
        cc_npts = new_npts*2-1
        nrfft = cc_npts // 2 + 1
        if (self.rank == 0):
            self.rank0_stacked_cc_spectra = np.zeros((nbins, nrfft),   dtype= np.complex64 )
            self.rank0_stacked_cc_count   = np.zeros(nbins, dtype=np.int32)
            self.rank0_stacked_cc_time    = np.zeros((nbins, cc_npts), dtype= np.float32 )
        self.comm.Reduce([single_cc_spectra, mpi4py.MPI.C_FLOAT_COMPLEX],
                    [self.rank0_stacked_cc_spectra, mpi4py.MPI.C_FLOAT_COMPLEX], mpi4py.MPI.SUM, root= 0)
        self.comm.Reduce([single_cc_count, mpi4py.MPI.INT32_T],
                    [self.rank0_stacked_cc_count, mpi4py.MPI.INT32_T], mpi4py.MPI.SUM, root= 0 )
        ###
        ###
        ### ifft and output on rank0
        if (self.rank == 0):
            roll_size = new_npts-1
            for ibin in range(nbins):
                self.rank0_stacked_cc_time[ibin] = pyfftw.interfaces.numpy_fft.irfft(self.rank0_stacked_cc_spectra[ibin], cc_npts)
                #self.rank0_stacked_cc_time[ibin] = np.roll(self.rank0_stacked_cc_time[ibin], roll_size)
            #self.t1 = roll_size*new_dt
            #self.t0 = -self.t1

            ### output to hdf5
            fnm = '%s.h5' % (output_fnm_prefix)
            fid = h5py.File(fnm, 'w')
            fid.create_dataset('cc_stack', data=self.rank0_stacked_cc_time)
            fid.create_dataset('cc_stack_count', data=self.rank0_stacked_cc_count)
            fid.attrs['dt'] = new_dt
            fid.attrs['dist_min'] = dist_min
            fid.attrs['dist_max'] = dist_max
            fid.attrs['dist_step'] = dist_step
            fid.close()
            ### output to sac
            for ibin in range(nbins):
                fnm = '%s/cc_%03d.sac' % (output_fnm_prefix, ibin)
                sac.wrt_sac_2(fnm, self.rank0_stacked_cc_time[ibin], delta=new_dt, b=0)
    ###




#if __name__ == "__main__":
if False:
    ###
    root_fnm_template = ''
    log_prefnm = ''
    username = getpass.getuser()
    d1, d2, dstep = 0.0, 180.0, 1.0
    cut_marker, cut_t1, cut_t2 = 'o', 10800.0, 32400.0
    new_dt = 0.1
    new_npts = 216000
    twin, f1, f2 = 128.0, 0.02, 0.0666666
    fwin = 0.02
    output_fnm_prefix = 'junk'
    #
    daz_min, daz_max =-999, 999
    gc_range_deg = None
    gc_reverse = False
    ###
    HMSG = '%s -I fnm_template -L log.txt -T o/10800/32400 -D 0.1 -M 216000 -S 0/180/1 -N 128.0/0.02/0.0666666 -W 0.02 -A -5/15 -O out_fnm' % (sys.argv[0] )
    if len(sys.argv) < 2:
        print(HMSG)
        sys.exit(0)
    options, remainder = getopt.getopt(sys.argv[1:], 'I:L:T:D:M:S:N:W:A:O:' )
    for opt, arg in options:
        if opt in ('-I'):
            root_fnm_template = arg
        elif opt in ('-L'):
            log_prefnm = arg
        elif opt in ('-T'):
            cut_marker, cut_t1, cut_t2 = arg.split('/')
            cut_t1 = float(cut_t1)
            cut_t2 = float(cut_t2)
        elif opt in ('-D'):
            new_dt = float(arg)
        elif opt in ('-M'):
            new_npts = int(arg)
        elif opt in ('-S'):
            d1, d2, dstep = [float(it) for it in arg.split('/')]
        elif opt in ('-N'):
            twin, f1, f2 = [float(it) for it in arg.split('/')]
        elif opt in ('-W'):
            fwin = float(arg)
        elif opt in ('-O'):
            output_fnm_prefix = arg
        elif opt in ('-A'): ## example: -A10/20 
            daz_min, daz_max  = [float(it) for it in arg.split('/') ]
        else:
            print('invalid options: %s' % (opt) )
            print(HMSG)
            sys.exit(0)
    ###
    ###
    print('>>> cmd: ', ' '.join(sys.argv ) )
    app = cc_stack_sac(root_fnm_template, new_npts, output_fnm_prefix, new_dt, log_prefnm, cut_marker, cut_t1, cut_t2, 
            twin, f1, f2, fwin, d1, d2, dstep, daz_min, daz_max)
