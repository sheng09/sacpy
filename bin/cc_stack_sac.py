#!/usr/bin/env python3
"""
Executable files for cross-correlation and stacking operations
"""
from sacpy.processing import filter, taper, tukey_jit, temporal_normalize, frequency_whiten
from sacpy.sac import c_rd_sac, c_wrt_sac, c_mk_sachdr_time
import time
import sacpy.geomath as geomath
import h5py
import pyfftw
import numpy as np
import sys
import getopt
from glob import glob
from numba import jit

from mpi4py import MPI

def main(mode, 
            fnm_wildcard, tmark, t1, t2, delta, input_format='sac', pre_detrend=True, pre_taper_ratio= 0.005, pre_filter= None,
            tnorm = (128.0, 0.02, 0.066666), swht= 0.02,
            stack_dist_range = (0.0, 180.0), stadist_step= 1.0, 
            daz_range= None, gcd_range= None, gc_center_rect= None, gc_center_circle=None, min_recordings=0, epdd=False,
            post_fold = False, post_taper_ratio = 0.005, post_filter=None, post_norm = False, post_cut= None,
            output_pre_fnm= 'junk', output_format= ['hdf5'],
            log_prefnm= 'cc_mpi_log', log_mode=None, spec_acc_threshold = 0.001 ):
    """
    The main function.

    The run consists of
    - 0. mode for rcv-to-rcv or src-to-src correlation stacks.
    - 1. reading sac files and pre-processing;
    - 2. optional whitening;
    - 3. cross-correlation and stacking;
    - 4. post-processing of correlation stacks;
    - 5. output.

    The parameters for each step are:
    - 0. mode:

    - 1. reading sac files and pre-processing:
        fnm_wildcard:    file name wildcards. (e.g., fnm_wildcard= download/event_*/processed/*.00.BHZ ) 
        tmark, t1, t2:   the cut window to read.
        delta:           sampling time interval of the sac time series.
        pre_detrend:     True or False
        pre_taper_ratio: a ratio value between 0 and 0.5
        pre_filter:      None to disable pre-filter or a tuple (btype, f1, f2) to enable the filter.
                         (e.g.,  pre_filter= ('lowpass', 1.0, 0.0), or pre_filter= ('bandpass', 0.02, 0.5) )
    - 2. optional whitening:
        tnorm: temporal normalization parameter. (e.g., 128.0, 0.02, 0.066666)
               set None to disable temporal normalization.
        swht:  spectral whitening parameter in Hz. (e.g., spectral_whiten_parameter= 0.02)
               set None to disable spectral whitening.
    - 3. cross-correlation and stacking:
        dist_range:   inter-distance range (e.g., dist_range=(0.0, 180.0) ).
        dist_step:    inter-distance step. The generated dist list will be `np.arange(dist_range[0], dist_range[1] + dist_step )`.
        daz_range:    selection criterion for daz range. Set `None` to disable. (e.g., daz_range=(-0.1, 15.0) )
        gcd_range:    selection criterion for the distance from the event to two receivers great-circle path.
                      Set `None` to disable. (e.g., gcd_range= (-0.1, 20) )
        gc_center_rect: selection criterion for the center of the great-circle plane formed by the two receivers.
                        If the center is outside the area, then it is not used in the cross-correlation stacks.
                        Can be a list of rect (lo1, lo2, la1, la2).
                        E.g., gc_center_rect= [ (120, 180, 0, 35), (180, 190, 0, 10) ]
    - 4. post-processing of correlation stacks:
        post_folding:     True to enable folding the positive and the negative cross-correlation lags. False to disable.
        post_taper_ratio: taper ratio between 0 and 0.5 when enable post filter.
                          The ratio is with respect to the full correlation time series instead of the cutted one declared by `post_cut`.
        post_filter:      None to disable post-filter or a tuple (btype, f1, f2) to enable the filter.
                          (e.g.,  post_filter= ('lowpass', 1.0, 0.0), or post_filter= ('bandpass', 0.02, 0.5) )
        post_norm:        Normalize each cross-correlation stack with the max positive amplitude.
        post_cut:         A tuple to cut the correlation stacks before outputting. Set None to disable.
    - 5. output:
        log_prefnm     : mpi log filename prefix;
        output_pre_fnm : output filename prefix;
        output_format  : a list of output format. (e.g., output_format=['sac'], output_format=['sac', 'h5'] )
    """
    mpi_comm = MPI.COMM_WORLD.Dup()
    mpi_rank = mpi_comm.Get_rank()
    mpi_ncpu = mpi_comm.Get_size()

    mpi_log_fid = None # Set None to enable MPI logging on each PROC
    if log_mode == None or mpi_rank in log_mode:
        mpi_log_fid = open('%s_%03d.txt' % (log_prefnm, mpi_rank), 'w' )
        mpi_print_log(mpi_log_fid, 0, True, 'Start! rank(%d) nproc(%d)' % (mpi_rank, mpi_ncpu) )
    tc_main = time.time()
    ############################################################################################################################################
    ### 0. Init mode
    ############################################################################################################################################
    flag_mode = 0 # rcv-to-rcv
    if mode in ('r2r', 'rcv', 'rcv2rcv', 'rcv-to-rcv'):
        flag_mode = 0
    elif mode in ('s2s', 'src', 'src2src', 'src-to-src'):
        flag_mode = 1
    if True:
        mpi_print_log(mpi_log_fid, 0, True, 'mode: ', mode  )
    ############################################################################################################################################
    ### 1. Init parameters for reading sac files and pre-processing
    ############################################################################################################################################
    # dependent parameters
    sampling_rate = 1.0/delta
    npts          = int( np.round((t2-t1)/delta) ) + 1
    fftsize       = npts * 2
    df            = 1.0/(delta*fftsize)
    if True: # user-defined parameters
        mpi_print_log(mpi_log_fid, 0, False, 'Set reading and pre-processing parameters')
        mpi_print_log(mpi_log_fid, 1, False, 'fnm_wildcard:   ', fnm_wildcard)
        mpi_print_log(mpi_log_fid, 1, False, 'input formate:  ', input_format)
        mpi_print_log(mpi_log_fid, 1, False, 'tmark, t1, t2:  ', tmark, t1, t2)
        mpi_print_log(mpi_log_fid, 1, False, 'delta:          ', delta)
        mpi_print_log(mpi_log_fid, 1, False, 'pre_detrend:    ', pre_detrend)
        mpi_print_log(mpi_log_fid, 1, False, 'pre_taper_ratio:', pre_taper_ratio)
        mpi_print_log(mpi_log_fid, 1, False, 'pre_filter:     ', pre_filter, '\n')
        # dependent parameters
        mpi_print_log(mpi_log_fid, 0, False, 'Dependent parameters:')
        mpi_print_log(mpi_log_fid, 1, False, 'sampling_rate:  ', sampling_rate)
        mpi_print_log(mpi_log_fid, 1, False, 'npts:           ', npts)
        mpi_print_log(mpi_log_fid, 1, False, 'fftsize:        ', fftsize)
        mpi_print_log(mpi_log_fid, 1, False, 'df:             ', df)

    ############################################################################################################################################
    ### 2. Init parameters for optional whitening
    ############################################################################################################################################
    wt_size, wt_f1, wt_f2, wf_size = None, None, None, None
    if tnorm != None:
        junk, wt_f1, wt_f2 = tnorm
        wt_size = (int(np.round(junk / delta) ) //2 ) * 2 + 1
    if swht != None:
        junk = swht  # Hz
        wf_size = (int(np.round(junk / df) ) //2 ) * 2 + 1
    # logging
    if True: # user-defined parameters
        mpi_print_log(mpi_log_fid, 0, False, 'Set whitening parameters')
        mpi_print_log(mpi_log_fid, 1, False, 'Temporal normalization: ', tnorm, 'size:', wt_size)
        mpi_print_log(mpi_log_fid, 1, True,  'Spetral whitening:      ', swht, 'size:', wf_size)
    ############################################################################################################################################
    ### 2.1 Optional acceleration for spectral whitening and spectral cross-correlation
    ############################################################################################################################################
    acc_range = None, None
    if post_filter != None:
        junk1, junk2 = post_filter[1], post_filter[2]
        if swht != None:
            junk2 = junk2 + swht
        acc_range = acc_bound(fftsize, sampling_rate, junk1, junk2, spec_acc_threshold)

        if True:
            mpi_print_log(mpi_log_fid, 0, False, 'Enable spectral acceleration')
            mpi_print_log(mpi_log_fid, 1, True,  'Threshold parameter: %f (will abandon the spectra outside (%f, %f) Hz.)' % (
                            spec_acc_threshold, acc_range[0]*df, acc_range[1]*df ) )
    ############################################################################################################################################
    #### 3. Init selection criteria
    ############################################################################################################################################
    if daz_range != None or gcd_range != None or gc_center_rect != None or gc_center_circle != None:
        daz_range      = (-0.1, 90.1) if daz_range == None else daz_range
        gcd_range      = (-0.1, 90.1) if gcd_range == None else gcd_range
        gc_center_rect = [(-9999, 9999, -9999, 9999)] if gc_center_rect == None else gc_center_rect
        gc_center_circle = [(0, 0, 9999)] if gc_center_circle == None else gc_center_circle
    # logging
    if True:
        mpi_print_log(mpi_log_fid, 0, False, 'Set selection criteria')
        mpi_print_log(mpi_log_fid, 1, False, 'daz: ', daz_range )
        mpi_print_log(mpi_log_fid, 1, False, 'gcd: ', gcd_range )
        mpi_print_log(mpi_log_fid, 1, False, 'selection of gc-center rect: ', gc_center_rect )
        mpi_print_log(mpi_log_fid, 1, False, 'selection of gc-center circle: ', gc_center_circle )
        mpi_print_log(mpi_log_fid, 1, False, 'minmal number of recordings: ', min_recordings )
    ############################################################################################################################################
    ### 4. Init post-processing parameters
    ############################################################################################################################################
    # logging
    if True:
        mpi_print_log(mpi_log_fid, 0, False, 'Set post-processing parameters' )
        mpi_print_log(mpi_log_fid, 1, False, 'Post folding:      ', post_fold )
        mpi_print_log(mpi_log_fid, 1, False, 'Post taper taio:   ', post_taper_ratio )
        mpi_print_log(mpi_log_fid, 1, False, 'Post filter:       ', post_filter )
        mpi_print_log(mpi_log_fid, 1, True,  'Post normalizaion: ', post_norm )
    ############################################################################################################################################
    ### 5. Init cross-correlation stack buf
    ############################################################################################################################################
    dist           = np.arange(dist_range[0], dist_range[1]+dist_step, dist_step )
    nrfft_valid    = acc_range[1] # spectra bigger than this are not useful given the post-filter processing
    spec_stack_mat = np.zeros( (dist.size, nrfft_valid), dtype= np.complex64 )
    stack_count    = np.zeros( dist.size, dtype=np.int32 )

    global_spec_stack_mat, global_stack_count= None, None
    if mpi_rank == 0:
        global_spec_stack_mat = np.zeros( (dist.size, nrfft_valid), dtype= np.complex64 )
        global_stack_count    = np.zeros( dist.size, dtype=np.int32 )
    # logging
    if True: # user-defined parameters
        mpi_print_log(mpi_log_fid, 0, False, 'Set ccstack parameters' )
        mpi_print_log(mpi_log_fid, 1, False, 'dist range: ', dist_range)
        mpi_print_log(mpi_log_fid, 1, False, 'dist step:  ', dist_step)
        mpi_print_log(mpi_log_fid, 1, True,  'dist:       ',
                            ', '.join(['%.1f' % it for it in dist[:3] ] ), '...',
                            ', '.join(['%.1f' % it for it in dist[-1:] ]), )
    ############################################################################################################################################
    ### 6. Distribut MPI
    ############################################################################################################################################
    global_wildcards, local_wildcards, flag_input_format = list(), list(), 0
    if input_format in ('sac', 'SAC'):
        global_wildcards  = sorted( glob('/'.join(fnm_wildcard.split('/')[:-1] ) ) )
        remainders = fnm_wildcard.split('/')[-1]
        local_wildcards = ['%s/%s' % (it, remainders) for it in  global_wildcards[mpi_rank::mpi_ncpu]]
        flag_input_format = 0
    # logging
    if True:
        mpi_print_log(mpi_log_fid, 0, False, 'Distribute MPI jobs')
        mpi_print_log(mpi_log_fid, 1, False, 'Total job size:    ', len(global_wildcards) )
        mpi_print_log(mpi_log_fid, 1, True,  'Jobs on this node: ', len(local_wildcards) )
    ############################################################################################################################################
    ### 7. start running
    ############################################################################################################################################
    if True:
        mpi_print_log(mpi_log_fid, 0, False, 'Start running...')
    tc_rdw, tc_cc = 0.0, 0.0
    for idx_job, it in enumerate(local_wildcards):
        mpi_print_log(mpi_log_fid, 1, True, '-',idx_job+1, it)
        ### (1). read and pre-processing and whitening
        tc_junk = time.time()
        whitened_spectra_mat, stlo, stla, evlo, evla, az, baz, gcarc= None, None, None, None, None, None, None, None
        if flag_input_format == 0: # input sac
            tmp = rd_wh_sac(it, delta, tmark, t1, t2, npts,
                            pre_detrend, pre_taper_ratio, pre_filter,
                            wt_size, wt_f1, wt_f2, wf_size, acc_range[0], acc_range[1], pre_taper_ratio,
                            fftsize, nrfft_valid, mpi_log_fid)
            whitened_spectra_mat, stlo, stla, evlo, evla, az, baz, gcarc = tmp
        ### (2) check if the obtained whitened_spectra_mat is valid
        if whitened_spectra_mat is None:
            mpi_print_log(mpi_log_fid, 2, True, '+ None of sac time series are valid, and hence we jump over to the next.' )
            continue
        ###
        nsac = stlo.size
        if nsac < min_recordings:
            mpi_print_log(mpi_log_fid, 2, True, '+ Insufficient number of recordings (%d), and hence we jump over to the next.' % (nsac) )
            continue
        ###
        local_tc_rdw = time.time()-tc_junk

        ### (3) r2r or s2s mode
        lon, lat, azimuth = (stlo, stla, az) if flag_mode == 0 else (evlo, evla, baz)
        pt_lon, pt_lat = (evlo[0], evla[0]) if flag_mode == 0 else (stlo[0], stla[0])

        ### (4) cc and stack
        tc_junk = time.time()
        if daz_range != None or gcd_range != None or gc_center_rect != None or gc_center_circle != None:
            center_clo1 = np.array( [rect[0] for rect in gc_center_rect] )
            center_clo2 = np.array( [rect[1] for rect in gc_center_rect] )
            center_cla1 = np.array( [rect[2] for rect in gc_center_rect] )
            center_cla2 = np.array( [rect[3] for rect in gc_center_rect] )

            circle_center_clo = np.array( [it[0] for it in gc_center_circle] )
            circle_center_cla = np.array( [it[1] for it in gc_center_circle] )
            circle_center_radius = np.array( [it[2] for it in gc_center_circle] )

            local_ncc = spec_ccstack2(whitened_spectra_mat, lon, lat, gcarc, epdd,
                                        spec_stack_mat, stack_count,
                                        dist_range[0], dist_range[1], dist_step, acc_range[0], acc_range[1],
                                        azimuth, pt_lon, pt_lat, daz_range[0], daz_range[1], gcd_range[0], gcd_range[1],
                                        center_clo1, center_clo2, center_cla1, center_cla2,
                                        circle_center_clo, circle_center_cla, circle_center_radius)
        else:
            local_ncc = spec_ccstack(whitened_spectra_mat, lon, lat, gcarc, epdd,
                                        spec_stack_mat, stack_count,
                                        dist_range[0], dist_range[1], dist_step, acc_range[0], acc_range[1])
        local_tc_cc = time.time()-tc_junk

        ### time summary
        tc_rdw = tc_rdw + local_tc_rdw
        tc_cc  = tc_cc  + local_tc_cc
        if True:
            local_tc = local_tc_rdw+ local_tc_cc + 0.001
            mpi_print_log(mpi_log_fid, 2, False,
                            '+ rd&whiten: %.1fs(%d%%)(%d)，ccstack %.1fs(%d%%) (%d)' % (
                            local_tc_rdw, local_tc_rdw/local_tc*100, nsac, local_tc_cc, local_tc_cc/local_tc*100, local_ncc ) )
    if True:
        tc = tc_rdw + tc_cc + 1.0e-3 # in case of zeros
        mpi_print_log(mpi_log_fid, 1, True, 'Time consumption summary: (rd&whiten: %.1fs(%d%%)， ccstack %.1fs(%d%%) )' % (
                        tc_rdw, tc_rdw/tc*100, tc_cc, tc_cc/tc*100 ) )
    ############################################################################################################################################
    ### 8. MPI collecting
    ############################################################################################################################################
    if True:
        mpi_print_log(mpi_log_fid, 0, True, 'MPI collecting(reducing) to RANK0...')
    mpi_comm.Reduce([spec_stack_mat, MPI.C_FLOAT_COMPLEX], [global_spec_stack_mat, MPI.C_FLOAT_COMPLEX], MPI.SUM, root= 0)
    mpi_comm.Reduce([stack_count, MPI.INT32_T], [global_stack_count, MPI.INT32_T], MPI.SUM, root= 0 )
    mpi_comm.Barrier()

    ############################################################################################################################################
    ### On RANK0
    ### 9 ifft and
    ### 10. post processing
    ### 11. output
    ############################################################################################################################################
    if mpi_rank == 0:
        absolute_amp = np.ones(dist.size, dtype=np.float32)
        cc_t1, cc_t2, cc_mat = post_proc(global_spec_stack_mat, absolute_amp, fftsize, npts, delta,
                                    post_fold, post_taper_ratio, post_filter, post_norm, post_cut, mpi_log_fid)
        
        output(cc_mat, global_stack_count, dist, absolute_amp, cc_t1, cc_t2, delta, output_pre_fnm, output_format, mpi_log_fid)
    ########################################################################################################################################
    ##### Done
    ########################################################################################################################################
    if True:
        mpi_print_log(mpi_log_fid, 0, True,  'Done (%.1f sec)' % (time.time()-tc_main) )
    if mpi_log_fid != None:
        mpi_log_fid.close()

def mpi_print_log(file, n_pre, flush, *objects, end='\n'):
    """
    Set file=None to disable logging.
    """
    if file != None:
        tmp = '>>> ' if n_pre<=0 else '    '*n_pre
        print(tmp, file=file, end='' )
        print(*objects, file=file, flush=flush, end=end )
def rd_wh_sac(fnm_wildcard, delta, tmark, t1, t2, npts,
              pre_detrend, pre_taper_ratio , pre_filter, # preproc args
              wnd_size_t, w_f1, w_f2, wnd_size_freq, speedup_i1, speedup_i2, whiten_taper_ratio, # whitening args
              fftsize, nrfft_valid, mpi_log_fid):
    """
    Read, preproc, and whiten many sacfiles given a filename template `sacfnm_template`.
    Those sac files can be recorded at many receivers for the same event, or they
    can be recorded at the same receiver from different events.

    fnm_wildcard:    The filename template that contain wildcards. The function with use
                     `glob(sacfnm_template)` to obtain all sac filenames.
    tmark, t1, t2:   The cut time window to read sac time series.

    pre_detrend:     Set True to enable detrend after reading.
    pre_taper_ratio: Set taper ratio (0~-0.5) to apply taper after the optional detrend.
    pre_filter:      Filter all the time series. Set `None` to disable filter.
                     The filter can be ('bandpass', f1, f2), ('lowpass', fc, meaningless_f) or
                     ('highpass', fc, meaningless_f).

    Return: (spectra, stlo, stla, evlo, evla, az, baz)
    """
    ###
    fnms = sorted( glob(fnm_wildcard) )
    nsac = len(fnms)
    ###
    sampling_rate = 1.0/delta
    btype, f1, f2 = pre_filter if pre_filter != None else (None, None, None)
    whiten_taper_length = int(npts * whiten_taper_ratio)
    ### buffer to return
    spectra = np.zeros((nsac, nrfft_valid), dtype=np.complex64  )
    stlo = np.zeros(nsac, dtype=np.float32 )
    stla = np.zeros(nsac, dtype=np.float32 )
    evlo = np.zeros(nsac, dtype=np.float32 )
    evla = np.zeros(nsac, dtype=np.float32 )
    az   = np.zeros(nsac, dtype=np.float32 )
    baz  = np.zeros(nsac, dtype=np.float32 )
    gcarc= np.zeros(nsac, dtype=np.float32 )
    ###
    index = 0
    for it in fnms:
        st = c_rd_sac(it, tmark, t1, t2, True, False)
        ## Check if Nan or all zeros.
        if (st is None) or (not st.dat.any() ):
            mpi_print_log(mpi_log_fid, 2, True, '+ Failure reading:', it)
            continue
        ## preproc
        try:
            if pre_detrend:
                st.detrend()
            if pre_taper_ratio > 1.0e-5:
                st.taper(pre_taper_ratio)
            if pre_filter != None:
                st.dat = filter(st.dat, sampling_rate, btype, (f1, f2), 2, 2)
        except:
            mpi_print_log(mpi_log_fid, 2, True, '+ Failure preproc:', it)
            continue
        ## whiten
        try:
            if wnd_size_t != None:
                st.dat = temporal_normalize(st.dat, sampling_rate, wnd_size_t, w_f1, w_f2, 1.0e-5, whiten_taper_length)
            if wnd_size_freq != None:
                st.dat = frequency_whiten(st.dat, wnd_size_freq, 1.0e-5, speedup_i1, speedup_i2, whiten_taper_length)
        except:
            mpi_print_log(mpi_log_fid, 2, True, '+ Failure whitening:', it)
            continue
        ## check if the whitened data is valid
        if (not st.dat.any()) or (np.isnan(st.dat).any() ):
            mpi_print_log(mpi_log_fid, 2, True, '+ Failure whitening:', it)
            continue
        ## fft and obtain valid values
        spectra[index] = pyfftw.interfaces.numpy_fft.rfft(st.dat, fftsize)[:nrfft_valid]
        hdr = st.hdr
        stlo[index] = hdr.stlo
        stla[index] = hdr.stla
        evlo[index] = hdr.evlo
        evla[index] = hdr.evla
        az[index]   = hdr.az
        baz[index]  = hdr.baz
        gcarc[index]= hdr.gcarc
        index = index + 1
    ###
    if 0 < index < nsac:
        spectra = spectra[:index, :]
        stlo = stlo[:index]
        stla = stla[:index]
        evlo = evlo[:index]
        evla = evla[:index]
        az   = az[:index]
        baz  = baz[:index]
        gcarc= gcarc[:index]
    ###
    if index > 0:
        return spectra, stlo, stla, evlo, evla, az, baz, gcarc
    else:
        return None, None, None, None, None, None, None, None
def acc_bound(fftsize, sampling_rate, f1, f2, critical_level= 0.001):
    """
    Return the acceleration bound (i1, i2) for spetral computation.
    Data outside the [i1, i2) can be useless given specific frequency band (f1, f2).
    """
    x = np.zeros(fftsize)
    x[0] = 1.0
    x = filter(x, sampling_rate, 'bandpass', (f1, f2), 2, 2)
    s = pyfftw.interfaces.numpy_fft.rfft(x, fftsize)
    amp = np.abs(s)
    c = amp.max() * critical_level
    i1 = np.argmax(amp>c)
    if np.min(amp[i1:]) >= c:
        i2 = fftsize
    else:
        i2 = np.argmin(amp[i1:]>c) + i1 + 1
    if i1 < 0:
        i1 = 0
    if i2 >= fftsize:
        i2 = fftsize
    return i1, i2
@jit(nopython=True, nogil=True)
def spec_ccstack(spec_mat, lon, lat, gcarc, epdd, stack_mat, stack_count, 
                    dist_min, dist_max, dist_step, acc_idx_min, acc_idx_max):
    """
    spec_mat: Input spectra that is a 2D matrix. Each row is for a single time series.
    lon, lat: A list of longitude/latitude. Can be stlo/stla for rcv-to-rcv correlations, 
              or evlo/evla for src-to-src correlations.
    stack_mat:   The correlation spectra for stacking.
    stack_count: A 1D array to store number of stacks for each stack bin.
    dist_min, dist_max, dist_step: Stack distance settings.
    acc_idx_min, acc_idx_max:

    Return:
        count: The number of correlation functions that are stacked.
    """
    count = 0
    i1, i2 = acc_idx_min, acc_idx_max
    nrow = stack_mat.shape[0]
    d1, d2 = dist_min-dist_step*0.5, dist_max+dist_step*0.5
    nsac = spec_mat.shape[0]
    for isac1 in range(nsac):
        lo1, la1, gcarc1 = lon[isac1], lat[isac1], gcarc[isac1]
        spec1 = spec_mat[isac1]
        for isac2 in range(isac1, nsac):
            lo2, la2, gcarc2 = lon[isac2], lat[isac2], gcarc[isac2]
            spec2 = spec_mat[isac2]
            ###
            dist = abs(gcarc1-gcarc2)
            if not epdd:
                dist = geomath.haversine(lo1, la1, lo2, la2)
            if dist > d2 or dist < d1:
                continue
            idx = int( np.round((dist-dist_min) / dist_step) )
            if idx < 0 or idx >= nrow:
                continue
            w   = np.conj(spec1[i1:i2]) * spec2[i1:i2]
            stack_mat[idx][i1:i2] += w
            stack_count[idx] += 1
            count += 1
    return count
@jit(nopython=True, nogil=True)
def sph_center_triple_pts(pt_lon, pt_lat, lo1, la1, lo2, la2):
    """
    """
    clo, cla, clo1, cla1 = 0., 0., 0., 0.
    if abs(lo1-lo2)>1.0e-3 or abs(la1-la2)>1.0e-3:
        (clo, cla), (clo1, cla1) = geomath.great_circle_plane_center(lo1, la1, lo2, la2)
    else:
        (clo, cla), (clo1, cla1) = geomath.great_circle_plane_center(pt_lon, pt_lat, lo1, la1)
    if cla<0:
        clo, cla = clo1, cla1
    return clo%360, cla
@jit(nopython=True, nogil=True)
def round_daz(daz):
    daz = daz % 360
    if daz > 180.0:
        daz = 360.0-daz
    if daz > 90.0:
        daz = 180.0-daz
    return daz
@jit(nopython=True, nogil=True)
def spec_ccstack2(spec_mat, lon, lat, gcarc, epdd, stack_mat, stack_count, 
                    dist_min, dist_max, dist_step, acc_idx_min, acc_idx_max,
                    azimuth, pt_lon, pt_lat, daz_min, daz_max, gcd_min, gcd_max,
                    rect_clo1, rect_clo2, rect_cla1, rect_cla2,
                    circle_center_clo, circle_center_cla, circle_center_radius):
    """
    spec_mat: Input spectra that is a 2D matrix. Each row is for a single time series.
    lon, lat: A list of longitude/latitude. Can be stlo/stla for rcv-to-rcv correlations, 
              or evlo/evla for src-to-src correlations.
    stack_mat:   The correlation spectra for stacking.
    stack_count: A 1D array to store number of stacks for each stack bin.
    dist_min, dist_max, dist_step: Stack distance settings.
    acc_idx_min, acc_idx_max:

    azimuth: Azimuth (az or baz) for each row of the `spec_mat`.
    pt_lon, pt_lat: The same point (event or source) for each correlation pair.
    daz_min, daz_max: Azimuth (az or baz) difference range for selecting correlation pairs.
    gcd_min, gcd_max: Great-circle distance range for selecting correlation pairs.
    rect_clo1, rect_clo2, rect_cla1, rect_cla2: Rectangels/boxes area that contain the spherical center of correlation pairs for selecting correlation pairs.
    circle_center_clo, circle_center_cla, circle_center_radius: Circle area that contain the spherical center of correlation pairs for selecting correlation pairs.
    Return:
        count: The number of correlation functions that are stacked.
    """
    nrow = stack_mat.shape[0]
    dist1, dist2 = dist_min-dist_step*0.5, dist_max+dist_step*0.5
    count = 0
    i1, i2 = acc_idx_min, acc_idx_max
    nsac = spec_mat.shape[0]
    for isac1 in range(nsac):
        lo1, la1, az1, gcarc1 = lon[isac1], lat[isac1], azimuth[isac1], gcarc[isac1]
        spec1 = spec_mat[isac1]
        for isac2 in range(isac1, nsac):
            lo2, la2, az2, gcarc2 = lon[isac2], lat[isac2], azimuth[isac2], gcarc[isac2]
            ### dist selection
            dist = abs(gcarc1-gcarc2)
            if not epdd:
                dist = geomath.haversine(lo1, la1, lo2, la2)
            if dist < dist1 or dist > dist2:
                continue
            idx = int( np.round((dist-dist_min) / dist_step) )
            if idx<0 or idx>nrow:
                continue
            ### rect and circle selection
            flag = 1 # 1 means the (clo, cla) is out of any rectangles
            clo, cla = sph_center_triple_pts(pt_lon, pt_lat, lo1, la1, lo2, la2)
            for clo1, clo2, cla1, cla2 in zip(rect_clo1, rect_clo2, rect_cla1, rect_cla2):
                if (cla1<=cla<=cla2) and ( clo1<=clo<=clo2 or (clo1>clo2 and (clo>=clo1 or clo<=clo2)) ):
                    flag = 0
                    break
            if flag == 1:
                continue

            flag = 1 # 1 means the (clo, cla) is out of any circles
            for cclo, ccla, cradius in zip(circle_center_clo, circle_center_cla, circle_center_radius):
                if geomath.haversine(clo, cla, cclo, ccla) <= cradius:
                    flag = 0
                    break
            if flag == 1:
                continue
            ### daz selection
            daz = round_daz(az1-az2)
            if daz < daz_min or daz > daz_max:
                continue
            ### gcd selection
            gcd = abs( geomath.point_distance_to_great_circle_plane(pt_lon, pt_lat, lo1, la1, lo2, la2) )
            if gcd < gcd_min or gcd > gcd_max:
                continue
            ###
            spec2 = spec_mat[isac2]
            ###
            w   = np.conj(spec1[i1:i2]) * spec2[i1:i2]
            stack_mat[idx][i1:i2] += w
            stack_count[idx] += 1
            count += 1
    return count
def post_proc(spec_stack_mat, absolute_amp, fftsize, npts, delta,
                post_fold, post_taper_ratio, post_filter, post_norm, post_cut,
                mpi_log_fid):
    """
    """
    if True:
        mpi_print_log(mpi_log_fid, 0, False, 'Ifft', )
        mpi_print_log(mpi_log_fid, 1, True,  'fftsize: ', fftsize, '(we will later get rid of the useless ZERO point in the cross-correlation.)')
    rollsize = npts-1
    nbin = spec_stack_mat.shape[0]
    time_mat = np.zeros( (nbin, fftsize-1), dtype=np.float32 )
    ### ifft
    for irow in range(nbin):
        spec_stack_mat[irow][0] = 0.0 # set the DC component to zero
        junk = pyfftw.interfaces.numpy_fft.irfft(spec_stack_mat[irow], fftsize)
        junk = np.roll(junk, rollsize)
        time_mat[irow] = junk[:-1] # get rid of the zero point
    ### post processing
    if True:
        mpi_print_log(mpi_log_fid, 0, True, 'Post processing...' )
    # 4.1 post processing fold
    cc_t1, cc_t2 = -rollsize*delta, rollsize*delta
    if post_fold:
        mpi_print_log(mpi_log_fid, 1, True, 'Folding...' )
        for irow in range(nbin):
            time_mat[irow] += time_mat[irow][::-1]
        time_mat = time_mat[:,rollsize:]
        cc_t1, cc_t2 = 0, rollsize*delta
    # 4.2 post filtering
    if post_filter:
        sampling_rate = 1.0/delta
        junk = int(post_taper_ratio * time_mat.shape[1])
        mpi_print_log(mpi_log_fid, 1, False, 'Filtering... ', post_filter )
        mpi_print_log(mpi_log_fid, 1, True,  'Tapering ... ', post_taper_ratio, 'size:', junk )
        btype, f1, f2 = post_filter
        for irow in range(nbin):
            time_mat[irow] = filter(time_mat[irow], sampling_rate, btype, (f1, f2), 2, 2 )
            taper(time_mat[irow], junk)
    # 4.3 post norm
    if post_norm:
        mpi_print_log(mpi_log_fid, 1, True, 'Post normalizing... ' )
        for irow in range(nbin):
            v = time_mat[irow].max()
            if v> 0.0:
                time_mat[irow] *= (1.0/v)
                absolute_amp[irow] = v
    # 4.4 post cut
    if post_cut != None:
        mpi_print_log(mpi_log_fid, 1, True, 'Post cutting... ', post_cut )
        post_t1, post_t2 = post_cut
        post_t1 = cc_t1 if post_t1 < cc_t1 else post_t1
        post_t2 = cc_t2 if post_t2 > cc_t2 else post_t2

        post_i1 = int( round((post_t1-cc_t1)/delta) )
        post_i2 = int( round((post_t2-cc_t1)/delta) ) + 1

        cc_t1 = cc_t1 + post_i1 *delta
        cc_t2 = cc_t1 + (post_i2-post_i1-1)*delta
        time_mat = time_mat[:, post_i1:post_i2 ]
    if True:
        mpi_print_log(mpi_log_fid, 1, True, 'Correlation time range: ', (cc_t1, cc_t2) )
    return cc_t1, cc_t2, time_mat
def output( stack_mat, stack_count, dist, absolute_amp,
            cc_t1, cc_t2, delta, 
            output_fnm_prefix, output_format, 
            mpi_log_fid):
    """
    """
    if True:
        mpi_print_log(mpi_log_fid, 0, False, 'Outputting...', output_format)
    if 'hdf5' in output_format or 'h5' in output_format:
        h5_fnm = '%s.h5' % (output_fnm_prefix)
        if True:
            mpi_print_log(mpi_log_fid, 1, True, 'hdf5: ', h5_fnm)
        f = h5py.File(h5_fnm, 'w' )
        dset = f.create_dataset('ccstack', data=stack_mat )
        dset.attrs['cc_t0'] = cc_t1
        dset.attrs['cc_t1'] = cc_t2
        dset.attrs['delta'] = delta
        f.create_dataset('stack_count', data= stack_count )
        f.create_dataset('absolute_amp', data= absolute_amp )
        f.create_dataset('dist', data= dist)
        f.close()
    if 'sac' in output_format or 'SAC' in output_format:
        out_hdr = c_mk_sachdr_time(cc_t1, delta, 1)
        if True:
            mpi_print_log(mpi_log_fid, 1, True, 'sac:  ', '%s_..._sac' % (output_fnm_prefix))
        for irow, it in enumerate(dist):
            fnm = '%s_%05.1f_.sac' % (output_fnm_prefix, it)
            out_hdr.user2 = dist[irow]
            out_hdr.user3 = stack_count[irow]
            out_hdr.user4 = absolute_amp[irow]
            c_wrt_sac(fnm, stack_mat[irow], out_hdr, True)
    pass

HMSG = """%s  -I "in*/*.sac" -T -5/10800/32400 -D 0.1 -O cc_stack --out_format hdf5
    [--pre_detrend] [--pre_taper 0.005] [--pre_filter bandpass/0.005/0.1] 
    --stack_dist 0/180/1 [--daz -0.1/15] [--gcd -0.1/20] [--gc_center_rect 120/180/0/40,180/190/0/10]
    [--gc_center_circle 100/-20/10,90/0/15] [--min_recordings 10] [--epdd]
    [--w_temporal 128.0/0.02/0.06667] [--w_spec 0.02] 
    [--post_fold] [--post_taper 0.05] [--post_filter bandpass/0.02/0.0666] [--post_norm] [--post_cut]
     --log cc_log  [--log_mode 0] [--acc=0.001]

Args:
    #0. Mode:
    --mode: 'r2r' or 's2s'.

    #1. I/O and pre-processing args:
    -I  : filename wildcards.
    -T  : cut time window for reading.
    -D  : delta.
    --pre_detrend: use this to enable detrend.
    --pre_taper 0.005 : set taper ratio (default in 0.005)
    [--pre_filter bandpass/0.005/0.1]: set pre filter.
    
    -O  : output filename prefix.
    --out_format:  output formate (can be 'hdf5' or 'sac', or 'hdf5,sac' ).

    #2. whitening parameter.
    [--w_temporal] : temporal normalization. (default is disabled)
    [--w_spec] : spectral whitening. (default is disabled)

    #3. stacking method
    --stack_dist  :
    [--daz]/[--dbaz]                    :
    [--gcd_ev]/[--gcd_ev]/[--gcd_rcv]   :
    [--gc_center_rect] : a list of rect (lo1, lo2, la1, la2) to exclude some receiver pairs.
                         The receiver pairs with great-circle center outside those area are excluded.
    [--gc_center_circle]:a list of (clo, cla, radius) to excluse some receiver pairs.
                         The receiver pairs with great-circle center outside those area are excluded.
    [--min_recordings]/[--min_ev_per_rcv]/[--min_rcv_per_ev] : minmal number of recordings for a single event or for a single receiver.
    [--epdd] : Use epi-distance difference instead of inter-receiver or inter-source distance.

    #4. post-processing parameters:
    [--post_fold]
    [--post_taper]
    [--post_filter]
    [--post_norm]
    [--post_cut]

    #5. log:
    --log : log filename prefix.
    --log_mode :
                E.g.: `--log_mode=all`, `--log_mode=0`, `--log_mode=0,1,2,3,10`.

    #6. Other options:
    --acc : acceleration threshold. (default is 0.001)

E.g.,
    %s --mode r2r -I "in*/*.sac" -T -5/10800/32400 -D 0.1 --pre_detrend --pre_taper 0.005
        -O cc --out_format hdf5,sac
        --w_temporal 128.0/0.02/0.06667 --w_spec 0.02
        --stack_dist 0/180/1
        --daz -0.1/20 --gcd -0.1/30 --gc_center_rect 120/180/0/40 --gc_center_circle 100/-20/15,90/0/15
        --post_fold --post_taper 0.005 --post_filter bandpass/0.001/0.06667 --post_norm  --post_cut 0/5000
        --log cc_log --log_mode=0 --acc 0.01

    """
if __name__ == "__main__":
    mode = 'r2r'

    fnm_wildcard = ''
    input_format = 'sac'
    tmark, t1, t2 = -5, 10800, 32400
    delta = 0.1
    pre_detrend=True
    pre_taper_ratio= 0.005
    pre_filter= None

    output_pre_fnm= 'junk'
    output_format='hdf5'

    tnorm = None #(128.0, 0.02, 0.066666)
    swht  = None #0.02

    dist_range = (0.0, 180.0)
    dist_step= 1.0
    daz_range = None
    gcd_range = None
    gc_center_rect = None
    gc_center_circle = None
    min_recordings = 0
    epdd = False

    post_fold = False
    post_taper_ratio = 0.005
    post_filter = None #('bandpass', 0.02, 0.066666)
    post_norm = False
    post_cut = None

    log_prefnm= 'cc_mpi_log'
    log_mode  = None

    spec_acc_threshold = 0.001
    ######################
    if len(sys.argv) <= 1:
        print(HMSG % (sys.argv[0], sys.argv[0]), flush=True)
        sys.exit(0)
    ######################
    ######################
    options, remainder = getopt.getopt(sys.argv[1:], 'I:T:D:O:',
                            ['mode=',
                             'in_format=', 'pre_detrend', 'pre_taper=', 'pre_filter=',
                             'out_format=',
                             'w_temporal=', 'w_spec=',
                             'stack_dist=', 'daz=', 'dbaz=', 'gcd=', 'gcd_ev=', 'gcd_rcv=', 'gc_center_rect=', 'gc_center_circle=', 'min_recordings=', 'min_ev_per_rcv=', 'min_rcv_per_ev=', 'epdd=',
                             'post_fold', 'post_taper=', 'post_filter=', 'post_norm', 'post_cut=',
                             'log=', 'log_mode=', 'acc='] )
    for opt, arg in options:
        if opt in ('--mode'):
            mode = arg
        elif opt in ('-I'):
            fnm_wildcard = arg
        elif opt in ('-T'):
            tmark, t1, t2 = arg.split('/')
            tmark = int(tmark)
            t1 = float(t1)
            t2 = float(t2)
        elif opt in ('-D'):
            delta = float(arg)
        elif opt in ('--pre_detrend'):
            pre_detrend = True
        elif opt in ('--pre_taper'):
            pre_taper_ratio = float(arg)
        elif opt in ('--pre_filter'):
            junk1, junk2, junk3 = arg.split('/')
            pre_filter = (junk1, float(junk2), float(junk3) )
        elif opt in ('--in_format'):
            input_format = arg # need dev
        elif opt in ('-O'):
            output_pre_fnm = arg
        elif opt in ('--out_format'):
            output_format = arg.split(',')
        elif opt in ('--w_temporal'):
            tnorm = tuple( [float(it) for it in arg.split('/') ] )
        elif opt in ('--w_spec'):
            swht = float(arg)
        elif opt in ('--stack_dist'):
            x, y, z = [float(it) for it in arg.split('/') ]
            dist_range, dist_step = (x, y), z
        elif opt in ('--daz', '-dbaz'):
            daz_range = [float(it) for it in arg.split('/') ]
        elif opt in ('--gcd_ev', '--gcd', '--gcd_rcv'):
            gcd_range = [float(it) for it in arg.split('/') ]
        elif opt in ('--gc_center_rect'):
            gc_center_rect = []
            for rect in arg.split(','):
                tmp = [float(it) for it in rect.split('/') ]
                tmp[0] = tmp[0] % 360.0
                tmp[1] = tmp[1] % 360.0
                gc_center_rect.append( tmp )
        elif opt in ('--gc_center_circle'):
            gc_center_circle = []
            for rect in arg.split(','):
                tmp = [float(it) for it in rect.split('/') ]
                tmp[0] = tmp[0] % 360.0
                gc_center_circle.append( tmp )
        elif opt in ('--min_recordings', '--min_ev_per_rcv', '--min_rcv_per_ev'):
            min_recordings = int(arg)
        elif opt in ('--epdd'):
            epdd = True
        elif opt in ('--post_fold'):
            post_fold = True
        elif opt in ('--post_taper'):
            post_taper_ratio = float(arg)
        elif opt in ('--post_filter'):
            junk1, junk2, junk3 = arg.split('/')
            post_filter = (junk1, float(junk2), float(junk3) )
        elif opt in ('--post_norm'):
            post_norm = True
        elif opt in ('--post_cut'):
            post_cut = tuple( [float(it) for it in arg.split('/') ] )
        elif opt in ('--log'):
            log_prefnm = arg
        elif opt in ('--log_mode'):
            if arg == 'all':
                log_mode = None
            elif arg in ('rank0', 'RANK0'):
                log_mode = [0]
            else:
                log_mode = [int(it) for it in arg.split(',') ]
        elif opt in ('--acc'):
            spec_acc_threshold = float(arg)
    #######
    main(mode, fnm_wildcard, tmark, t1, t2, delta, input_format,
                pre_detrend, pre_taper_ratio, pre_filter, 
                tnorm, swht, dist_range, dist_step, 
                daz_range, gcd_range, gc_center_rect, gc_center_circle, min_recordings, epdd,
                post_fold, post_taper_ratio, post_filter, post_norm, post_cut, 
                output_pre_fnm, output_format, 
                log_prefnm, log_mode, spec_acc_threshold)
    ########


