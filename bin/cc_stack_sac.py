#!/usr/bin/env python3
"""
Executable files for cross-correlation and stacking operations
"""
from numpy.core.numeric import roll
from operator import pos
from sacpy.processing import filter, taper
from sacpy.sac import rd_sac_2
import timeit
import sacpy.sac as sac
import sacpy.processing as processing
import sacpy.geomath as geomath
from mpi4py import MPI
import h5py
import pyfftw
import numpy as np
import sys
import getpass
import getopt
from glob import glob
import scipy.signal as signal
from numba import jit

def main(   fnm_wildcard, tmark, t1, t2, delta, pre_detrend=True, pre_taper_ratio= 0.005, pre_filter= None,
            temporal_normalization = (128.0, 0.02, 0.066666), spectral_whiten= 0.02,
            dist_range = (0.0, 180.0), dist_step= 1.0,
            post_folding = True, post_taper_ratio = 0.005, post_filter = ('bandpass', 0.02, 0.066666), post_norm = True,
            log_prefnm= 'cc_mpi_log',
            output_pre_fnm= 'junk', output_format='hdf5' ):
    """
    The main function.

    The run consists of
    - 1. reading sac files and pre-processing;
    - 2. optional whitening;
    - 3. cross-correlation and stacking;
    - 4. post-processing of correlation stacks.

    The parameters for each step are:
    - 1. reading sac files and pre-processing:
        fnm_wildcard:    file name wildcards. (e.g., fnm_wildcard= download/event_*/processed/*.00.BHZ ) 
        tmark, t1, t2:   the cut window to read.
        delta:           sampling time interval of the sac time series.
        pre_detrend:     True or False
        pre_taper_ratio: a ratio value between 0 and 0.5
        pre_filter:      None to disable pre-filter or a tuple (btype, f1, f2) to enable the filter.
                         (e.g.,  pre_filter= ('lowpass', 1.0, 0.0), or pre_filter= ('bandpass', 0.02, 0.5) )
    - 2. optional whitening:
        temporal_normalization: temporal normalization parameter. (e.g., 128.0, 0.02, 0.066666)
                                set None to disable temporal normalization.
        spectral_whiten:        spectral whitening parameter in Hz. (e.g., spectral_whiten= 0.02)
                                set None to disable spectral whitening.
    - 3. cross-correlation and stacking:
        dist_range: inter-distance range (e.g., dist_range=(0.0, 180.0) ).
        dist_step:  inter-distance step. The generated dist list will be `np.arange(dist_range[0], dist_range[1] + dist_step )`.
    - 4. post-processing of correlation stacks:
        post_folding:     True to enable folding the positive and the negative cross-correlation lags. False to disable.
        post_taper_ratio: taper ratio between 0 and 0.5 when enable post filter.
        post_filter:      None to disable post-filter or a tuple (btype, f1, f2) to enable the filter.
                          (e.g.,  post_filter= ('lowpass', 1.0, 0.0), or post_filter= ('bandpass', 0.02, 0.5) )
        post_norm:        Normalize each cross-correlation stack with the max positive amplitude.
    """
    ### Init MPI
    mpi_comm = MPI.COMM_WORLD.Dup()
    mpi_rank = mpi_comm.Get_rank()
    mpi_ncpu = mpi_comm.Get_size()
    mpi_log_fid = open('%s_%03d.txt' % (log_prefnm, mpi_rank), 'w' )
    
    mpi_print_log('Start!', 0, mpi_log_fid, True)
    t_main_start = timeit.timeit()

    ### 1. read sac files and pre-processing
    # dependent parameters
    sampling_rate = 1.0/delta
    npts    = int( np.round((t2-t1)/delta) ) + 1
    fftsize = npts * 2
    nrfft   = fftsize // 2 + 1
    df      = 1.0/(delta*fftsize)
    taper_size = int(npts*pre_taper_ratio)
    # logging
    if True: # user-defined parameters
        mpi_print_log('Set reading parameters', 0, mpi_log_fid, True)
        mpi_print_log('fnm_wildcard:    %s ' % (fnm_wildcard)          , 1, mpi_log_fid, True)
        mpi_print_log('tmark, t1, t2:   %s, %f, %f ' % (tmark, t1, t2) , 1, mpi_log_fid, True)
        mpi_print_log('delta:           %f ' % (delta)                 , 1, mpi_log_fid, True)
        mpi_print_log('pre_detrend:     %d ' % (pre_detrend)           , 1, mpi_log_fid, True)
        mpi_print_log('pre_taper_ratio: %f ' % (pre_taper_ratio)       , 1, mpi_log_fid, True)
        if pre_filter != None:
            mpi_print_log('pre_filter:      (%s, %f, %f)' % (pre_filter[0], pre_filter[1], pre_filter[2]), 1, mpi_log_fid, True)
        else:
            mpi_print_log('pre_filter:  None (disabled)', 1, mpi_log_fid, True)
        #
        mpi_print_log('', 1, mpi_log_fid, False)
        mpi_print_log('sampling_rate: %f ' % (sampling_rate) , 1, mpi_log_fid, True)
        mpi_print_log('npts:          %d ' % (npts) , 1, mpi_log_fid, True)
        mpi_print_log('fftsize:       %d ' % (fftsize) , 1, mpi_log_fid, True)
        mpi_print_log('df:            %f ' % (df) , 1, mpi_log_fid, True)
        mpi_print_log('pre_taper_size:%d ' % (taper_size) , 1, mpi_log_fid, True)
    
    ### 2. optional whitening
    wt_size, wt_f1, wt_f2, wf_size = None, None, None, None
    if temporal_normalization != None:
        wt, wt_f1, wt_f2 = temporal_normalization
        wt_size = (int(np.round(wt / delta) ) //2 ) * 2 + 1
    if spectral_whiten != None:
        wf = spectral_whiten  # Hz
        wf_size = (int(np.round(wf / df) ) //2 ) * 2 + 1
    # logging
    if True: # user-defined parameters
        mpi_print_log('Set whitening parameters', 0, mpi_log_fid, True)
        if temporal_normalization != None:
            mpi_print_log('temporal normalization:   (%f, %f, %f), size= %d' % (temporal_normalization[0], 
                    temporal_normalization[1], temporal_normalization[2], wt_size), 1, mpi_log_fid, True)
        else:
            mpi_print_log('temporal normalization:   None (disabled)', 1, mpi_log_fid, True)
        if spectral_whiten != None:
            mpi_print_log('spetral whitening:        %f, size=%d' % (spectral_whiten, wf_size) , 1, mpi_log_fid, True)
        else:
            mpi_print_log('spetral whitening:        None (disabled)', 1, mpi_log_fid, True)
    
    ### 3. cross-correlation stack
    dist = np.arange(dist_range[0], dist_range[1] + dist_step )
    spec_stack_mat = np.zeros( (dist.size, nrfft), dtype= np.complex64 )
    stack_count = np.zeros( dist.size, dtype=np.int32 )

    global_spec_stack_mat, global_stack_count, stack_mat= None, None, None
    if mpi_rank == 0:
        global_spec_stack_mat = np.zeros( (dist.size, nrfft), dtype= np.complex64 )
        global_stack_count = np.zeros( dist.size, dtype=np.int32 )
        stack_mat = np.zeros((dist.size, npts), dtype=np.float32 )

    # logging
    if True: # user-defined parameters
        mpi_print_log('Set ccstack parameters', 0, mpi_log_fid, True)
        mpi_print_log('dist_range: [%f,%f]' % (dist_range[0], dist_range[1]), 1, mpi_log_fid, True)
        mpi_print_log('dist_step:   %f' % (dist_step), 1, mpi_log_fid, True)
        msg = '[%s...%s] (size=%d)' % (', '.join(['%.1f' % it for it in dist[:3] ] ),  
                                       ', '.join(['%.1f' % it for it in dist[-1:] ]), 
                                       len(dist) )
        mpi_print_log(msg, 1, mpi_log_fid, True)
    
    ### 4. post-processing
    # dependent parameters
    critical_parameter= 0.01
    cc_index_range = get_bound(fftsize, sampling_rate, post_filter[1], post_filter[2], critical_parameter)
    # logging
    if True:
        mpi_print_log('Enable accelerating cross-correlation', 0, mpi_log_fid, True)
        mpi_print_log('Level parameter: %f (will abandon the spectra outside (%f,%f)Hz.)' % (critical_parameter, cc_index_range[0]*df, cc_index_range[1]*df ), 1, mpi_log_fid, True )
    
    #######################################################################################################################################
    ### run
    #######################################################################################################################################
    directories  = sorted( glob('/'.join(fnm_wildcard.split('/')[:-1] ) ) )
    sac_wildcards = fnm_wildcard.split('/')[-1]
    njobs  = len(directories) 
    nchunk = (njobs // mpi_ncpu) if (njobs % mpi_ncpu == 0) else (njobs // mpi_ncpu + 1)
    mpi_i1 = mpi_rank * nchunk
    mpi_i2 = mpi_i1 + nchunk
    # logging
    if True:
        mpi_print_log('Distribute MPI jobs...', 0, mpi_log_fid, True)
        mpi_print_log('total jobs: %d' % (njobs), 1, mpi_log_fid, True)
        mpi_print_log('job ranges on this node: [%d, %d)' % (mpi_i1, mpi_i2), 1, mpi_log_fid, True)
    ###
    if True:
        mpi_print_log('Running...', 0, mpi_log_fid, True)
    t_rd, t_whiten, t_ccstack = 0.0, 0.0, 0.0
    for it in directories[mpi_i1: mpi_i2]:
        wildcards = '%s/%s' % (it, sac_wildcards)
        if True:
            mpi_print_log(wildcards, 1, mpi_log_fid, True)
        ### 1. read and pre-processing
        t_start = timeit.timeit()
        mat, sampling_rate, stlo, stla, evlo, evla, az, baz = rd_preproc_single(wildcards, tmark, t1, t2, pre_detrend, pre_taper_ratio, pre_filter)
        t_end = timeit.timeit()
        t_rd = t_rd + (t_end-t_start)
        
        ### 2. whitening
        t_start = timeit.timeit()
        whitened_spectra_mat = whiten_spec(mat, sampling_rate, wt_size, wt_f1, wt_f2, wf_size, fftsize, taper_size)
        t_end = timeit.timeit()
        t_whiten = t_whiten + (t_end-t_start)
        
        ### 3.1 cc and stack
        t_start = timeit.timeit()
        ccstack(whitened_spectra_mat, stack_count, stlo, stla, spec_stack_mat, dist_step, cc_index_range, dist_range[0] )
        t_end = timeit.timeit()
        t_ccstack = t_ccstack + (t_end-t_start)
    if True:
        total_loop_time = t_rd + t_whiten + t_ccstack
        msg = 'Time consumption summation: (rd: %.1f(%d%%)， whiten: %.1f(%d%%), ccstack %.1f(%d%%))' % (
                        t_rd, t_rd/total_loop_time*100, 
                        t_whiten, t_whiten/total_loop_time*100, 
                        t_ccstack, t_ccstack/total_loop_time*100 )
        mpi_print_log(msg, 1, mpi_log_fid, True)
    
    ### 3.1.2 MPI collecting
    if True:
        mpi_print_log('MPI collecting...', 0, mpi_log_fid, True)
    mpi_comm.Reduce([spec_stack_mat, MPI.C_FLOAT_COMPLEX], [global_spec_stack_mat, MPI.C_FLOAT_COMPLEX], MPI.SUM, root= 0)
    mpi_comm.Reduce([stack_count, MPI.INT32_T], [global_stack_count, MPI.INT32_T], MPI.SUM, root= 0 )
    
    ### 3.2 ifft
    if mpi_rank == 0: 
        if True:
            mpi_print_log('Ifft', 0, mpi_log_fid, True)
        rollsize = npts-1
        for irow in range(dist.size):
            x = pyfftw.interfaces.numpy_fft.irfft(spec_stack_mat[irow], fftsize)
            x = np.roll(x)
            stack_mat[irow] = x[:-1]
    
    ### 4. post processing
    if mpi_rank == 0:
        if True:
            mpi_print_log('Post processing', 0, mpi_log_fid, True)
        cc_t0, cc_t1 = -rollsize*delta, rollsize*delta
        # 4.1 post folding
        if post_folding:
            for irow in range(dist.size):
                stack_mat[irow] += stack_mat[irow][:-1]
            stack_mat = stack_mat[:,rollsize:]
            cc_t0, cc_t1 = 0, rollsize*delta
        # 4.2 post filtering
        if post_filter:
            w = processing.tukey_jit( stack_mat.shape[1], post_taper_ratio )
            btype, f1, f2 = post_filter
            for irow in range(dist.size):
                stack_mat[irow] = signal.detrend( stack_mat[irow] )
                stack_mat[irow] *= w
                stack_mat[irow] = processing.filter(stack_mat[irow], 'bandpass', (f1, f2), 2, 2 )
        # 4.3 post norm
        if post_norm:
            for irow in range(dist.size):
                v = np.max(stack_mat[irow])
                if v> 0.0:
                    stack_mat[irow] *= (1.0/v)
        if True:
            mpi_print_log('correlation time range (%f, %f)' % (cc_t0, cc_t1), 1, mpi_log_fid, True)
    
    ### 5. output
    if mpi_rank == 0:
        if True:
            mpi_print_log('Output format(%s)' % (output_format) , 0, mpi_log_fid, True)
        if output_format == 'hdf5' or output_format == 'h5':
            h5_fnm = '%s.h5' % (output_pre_fnm)
            if True:
                mpi_print_log(h5_fnm, 1, mpi_log_fid, True)
            f = h5py.File(h5_fnm, 'w' )
            dset = f.create_dataset('ccstack', data=stack_mat )
            dset.attrs['cc_t0'] = cc_t0
            dset.attrs['cc_t1'] = cc_t1
            f.create_dataset('stack_count', data= stack_count )
            f.close()
        elif output_format == 'sac' or output_format == 'SAC':
            if True:
                mpi_print_log('%s_..._sac' % (output_pre_fnm), 1, mpi_log_fid, True)
            for irow, it in enumerate(dist):
                fnm = '%s_%05.1f_.sac' % (output_pre_fnm, it)
                sac.wrt_sac_2(fnm, stack_mat[irow], delta, cc_t0, user3= stack_count[irow] )
    
    ##### Done
    t_end = timeit.timeit()
    if True:
        mpi_print_log('Done (%.1f sec)' % (output_format, t_end-t_main_start) , 0, mpi_log_fid, True)
    mpi_log_fid.close()

def mpi_print_log(msg, n_pre, file=sys.stdout, flush=False):
    if n_pre <= 0:
        print('>>> %s' % (msg), file= file, flush= flush )
    else:
        print('%s%s' % ('    '*n_pre, msg), file= file, flush= flush )

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
    if i1 < 0:
        i1 = 0
    if i2 >= fftsize:
        i2 = fftsize
    return i1, i2

@jit(nopython=True, nogil=True)
def ccstack(spec_mat, stack_count, stlo_lst, stla_lst, stack_mat, dist_step, index_range, dist_start=0.0):
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
            stack_count[idx] += 1


if __name__ == "__main__":
    fnm_wildcard = ''
    tmark, t1, t2 = '0', 10800, 32400
    delta = 0.1
    pre_detrend=True
    pre_taper_ratio= 0.005
    pre_filter= None

    temporal_normalization = None #(128.0, 0.02, 0.066666)
    spectral_whiten= None #0.02

    dist_range = (0.0, 180.0)
    dist_step= 1.0

    post_folding = False
    post_taper_ratio = 0.005
    post_filter = None #('bandpass', 0.02, 0.066666)
    post_norm = False

    log_prefnm= 'cc_mpi_log'
    output_pre_fnm= 'junk'
    output_format='hdf5'

    ######################
    HMSG = """\n
    %s  -I "in*/*.sac" -T 0/10800/32400 -D 0.1 -O cc_stack --out_format hdf5
        [--pre_detrend] [--pre_taper 0.005] [--pre_filter bandpass/0.005/0.1] 
        [--stack_dist 0/180/1]
        [--w_temporal 128.0/0.02/0.06667] [--w_spec 0.02] 
        [--post_fold] [--post_taper 0.05] [--post_filter bandpass/0.02/0.0666] [--post_norm] 
         --log cc_log  

Args:
    #1. I/O and pre-processing args:
    -I  : filename wildcards.
    -T  : cut time window for reading.
    -D  : delta.
    -O  : output filename prefix. 
    --out_format:  output formate (can be 'hdf5' or 'sac').
    --pre_detrend: use this to enable detrend.
    --pre_taper 0.005 : set taper ratio (default in 0.005)
    [--pre_filter bandpass/0.005/0.1]: set pre filter.

    #2. whitening parameter.
    [--w_temporal] : temporal normalization. (default is disabled)
    [--w_spec] : spectral whitening. (default is disabled)

    #3. stacking method
    --stack_dist :

    #4. post-processing parameters:
    [--post_fold]
    [--post_taper]
    [--post_filter]
    [--post_norm]

    #5. log:
    --log : log filename prefix.

E.g.,
    %s -I "in*/*.sac" -T 0/10800/32400 -D 0.1 -O cc --out_format hdf5
        --pre_detrend --pre_taper 0.005 
        --w_temporal 128.0/0.02/0.06667 --w_spec 0.02
        --stack_dist 0/180/1
        --post_fold --post_taper 0.005 --post_filter bandpass/0.001/0.06667 --post_norm 
        --log cc_log  
    
    """
    
    if len(sys.argv) <= 1:
        print(HMSG % (sys.argv[0], sys.argv[0]), flush=True)
        sys.exit(0)
    ######################
    ######################
    options, remainder = getopt.getopt(sys.argv[1:], 'I:T:D:O:', 
                            ['pre_detrend', 'pre_taper=', 'pre_filter=',
                             'w_temporal=', 'w_spec=',
                             'stack_dist=',
                             'post_fold', 'post_taper=', 'post_filter=', 'post_norm',
                             'log=', 'out_format='] )
    for opt, arg in options:
        if opt in ('-I'):
            fnm_wildcard = arg
        elif opt in ('-T'):
            tmark, t1, t2 = arg.split('/')
            t1 = float(t1)
            t2 = float(t2)
        elif opt in ('--pre_detrend'):
            pre_detrend = True
        elif opt in ('--pre_taper'):
            pre_taper_ratio = float(arg)
        elif opt in ('--pre_filter'):
            junk1, junk2, junk3 = arg.split('/')
            pre_filter = (junk1, float(junk2), float(junk3) )
        elif opt in ('--w_temporal'):
            temporal_normalization = [float(it) for it in arg.split('/') ]
        elif opt in ('--w_spec'):
            spectral_whiten = float(arg)
        elif opt in ('--stack_dist'):
            x, y, z = [float(it) for it in arg.split('/') ]
            dist_range, dist_step = (x, y), z
        elif opt in ('--post_fold'):
            post_folding = True
        elif opt in ('--post_taper'):
            post_taper_ratio = float(arg)
        elif opt in ('--post_filter'):
            junk1, junk2, junk3 = arg.split('/')
            post_filter = (junk1, float(junk2), float(junk3) )
        elif opt in ('--post_norm'):
            post_norm = True
        elif opt in ('--log'):
            log_prefnm = arg
        elif opt in ('-O'):
            output_pre_fnm = arg
        elif opt in ('--out_format'):
            output_format = arg
    #######
    main(fnm_wildcard, tmark, t1, t2, delta, pre_detrend, pre_taper_ratio, pre_filter,
            temporal_normalization, spectral_whiten,
            dist_range, dist_step,
            post_folding, post_taper_ratio, post_filter, post_norm,
            log_prefnm, output_pre_fnm, output_format )
    ########


