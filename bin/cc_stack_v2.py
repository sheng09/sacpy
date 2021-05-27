#!/usr/bin/env python3
"""
Executable files for cross-correlation and stacking operations
"""
from glob import glob
from sacpy.sac import c_rd_sac, c_mk_sachdr_time, c_rd_sachdr, c_wrt_sac
from sacpy.geomath import haversine, azimuth, point_distance_to_great_circle_plane, great_circle_plane_center_triple
from sacpy.processing import iirfilter_f32, taper, rmean, detrend, cut, tnorm_f32, fwhiten_f32

from numba import jit
from h5py import File as h5_File

from numpy import array, roll, zeros, ones, argmin, argmax, abs, float32, complex64, int32, arange, isnan, conj
from numpy.random import random
from pyfftw.interfaces.numpy_fft import rfft, irfft

import sys
from time import time
from mpi4py import MPI
from getopt import getopt
from h5py import File as h5_File
##################################################################################################################
def init_mpi(log_prefnm, log_mode):
    mpi_comm = MPI.COMM_WORLD.Dup()
    mpi_rank = mpi_comm.Get_rank()
    mpi_ncpu = mpi_comm.Get_size()

    mpi_log_fid = None # Set None disable mpi log on this PROC
    if log_mode == None or mpi_rank in log_mode:
        mpi_log_fid = open('%s_%03d.txt' % (log_prefnm, mpi_rank), 'w' )
        mpi_print_log(mpi_log_fid, 0, True, 'Start! rank(%d) nproc(%d)' % (mpi_rank, mpi_ncpu) )
    return mpi_comm, mpi_rank, mpi_ncpu, mpi_log_fid
def init_parameter(mode,
    tmark, t1, t2, delta, input_format='sac',                   # input param
    pre_detrend=True, pre_taper_ratio= 0.005, pre_filter= None, # pre-proc param)
    speedup_fs=None, speedup_level=0.01,
    mpi_log_fid=None):
    """
    Return critical parameters for cross-correlation computations.
    t1, t2: the time window for cutting time series.
    delta:  sampling time interval of input time series.
    fs:     frequency band (f1, f2) of interest (default is None).
    """
    flag_mode = 0 # rcv-to-rcv
    if mode in ('r2r', 'rcv', 'rcv2rcv', 'rcv-to-rcv'):
        flag_mode = 0
    elif mode in ('s2s', 'src', 'src2src', 'src-to-src'):
        flag_mode = 1

    npts = int( round((t2-t1)/delta) ) + 1 # make even for acceleration
    fftsize = ((npts+1)//2) * 2  # fft without pending zeros
    df = 1.0 / (delta*fftsize)
    speedup = init_speedup(fftsize, delta, speedup_fs, speedup_level)
    speedup_fs_valid = [it*df for it in speedup]
    pre_taper_halfsize = int(npts*pre_taper_ratio)

    cc_fftsize = npts*2 # fft by zero zeros of same size for cross-correlation computation
    cc_df = 1.0/(delta*cc_fftsize)
    cc_npts = cc_fftsize-1
    cc_speedup = init_speedup(cc_fftsize, delta, speedup_fs, speedup_level)
    cc_speedup_fs_valid = [it*cc_df for it in cc_speedup]

    mpi_print_log(mpi_log_fid, 0, False, 'mode: ', mode)
    mpi_print_log(mpi_log_fid, 0, False, 'Set reading and pre-processing parameters')
    mpi_print_log(mpi_log_fid, 1, False, 'tmark, t1, t2:  ', tmark, t1, t2)
    mpi_print_log(mpi_log_fid, 1, False, 'delta:          ', delta)
    mpi_print_log(mpi_log_fid, 1, False, 'input format:   ', input_format)
    mpi_print_log(mpi_log_fid, 1, False, 'pre_detrend:    ', pre_detrend)
    mpi_print_log(mpi_log_fid, 1, False, 'pre_taper_halfratio:', pre_taper_ratio, pre_taper_halfsize)
    mpi_print_log(mpi_log_fid, 1, False, 'pre_filter:     ', pre_filter)
    mpi_print_log(mpi_log_fid, 1, False, 'speedup:        (%f, %f, %f) (%d %d) (%f %f)' % (speedup_fs[0], speedup_fs[1], speedup_level, speedup[0], speedup[1], speedup_fs_valid[0], speedup_fs_valid[1]) )
    mpi_print_log(mpi_log_fid, 1, False, 'cc_speedup:     (%f, %f, %f) (%d %d) (%f %f)\n' % (speedup_fs[0], speedup_fs[1], speedup_level, cc_speedup[0], cc_speedup[1], cc_speedup_fs_valid[0], cc_speedup_fs_valid[1]) )

    # dependent parameters
    mpi_print_log(mpi_log_fid, 0, False, 'Dependent parameters:')
    mpi_print_log(mpi_log_fid, 1, False, 'npts:           ', npts)
    mpi_print_log(mpi_log_fid, 1, False, 'fftsize:        ', fftsize)
    mpi_print_log(mpi_log_fid, 1, True,  'df:             ', df)
    mpi_print_log(mpi_log_fid, 1, False, 'cc_npts:        ', cc_npts)
    mpi_print_log(mpi_log_fid, 1, False, 'cc_fftsize:     ', cc_fftsize)
    mpi_print_log(mpi_log_fid, 1, True,  'cc_df:          ', cc_df)

    return flag_mode, (npts, fftsize, df, speedup, pre_taper_halfsize), (cc_npts, cc_fftsize, cc_df, cc_speedup)
def init_ccstack_spec_buf(dist_min, dist_max, dist_step, nrfft_valid, mpi_log_fid):
    """
    Return the buffer for cross-correlation stack in frequency domain.
    """
    dist = arange(dist_min, dist_max+dist_step, dist_step, dtype=float32)
    spec_stack_mat = zeros( (dist.size, nrfft_valid), dtype= complex64 )
    stack_count    = zeros( dist.size, dtype=int32 )

    mpi_print_log(mpi_log_fid, 0, False, 'Set ccstack parameters' )
    mpi_print_log(mpi_log_fid, 1, False, 'dist range: (%f %f)' % (dist_min, dist_max) )
    mpi_print_log(mpi_log_fid, 1, False, 'dist step:  ', dist_step)
    mpi_print_log(mpi_log_fid, 1, True,  'dist:       ',
                            ', '.join(['%.1f' % it for it in dist[:3] ] ), '...',
                            ', '.join(['%.1f' % it for it in dist[-1:] ]), )

    return dist, spec_stack_mat, stack_count
def update_selection(daz_range, gcd_range, gc_center_rect, gc_center_circle, min_recordings, mpi_log_fid):
    flag_selection = False
    if daz_range != None or gcd_range != None or gc_center_rect != None or gc_center_circle != None:
        daz_range      = (-0.1, 90.1) if daz_range == None else daz_range
        gcd_range      = (-0.1, 90.1) if gcd_range == None else gcd_range
        gc_center_rect = [(-9999, 9999, -9999, 9999)] if gc_center_rect == None else gc_center_rect
        gc_center_circle = [(0, 0, 9999)] if gc_center_circle == None else gc_center_circle
        flag_selection = True


    center_clo1 = array( [rect[0] for rect in gc_center_rect] ) if gc_center_rect != None else None
    center_clo2 = array( [rect[1] for rect in gc_center_rect] ) if gc_center_rect != None else None
    center_cla1 = array( [rect[2] for rect in gc_center_rect] ) if gc_center_rect != None else None
    center_cla2 = array( [rect[3] for rect in gc_center_rect] ) if gc_center_rect != None else None

    circle_center_clo = array( [it[0] for it in gc_center_circle] )    if gc_center_circle != None else None
    circle_center_cla = array( [it[1] for it in gc_center_circle] )    if gc_center_circle != None else None
    circle_center_radius = array( [it[2] for it in gc_center_circle] ) if gc_center_circle != None else None


    mpi_print_log(mpi_log_fid, 0, False, 'Set selection criteria')
    mpi_print_log(mpi_log_fid, 1, False, 'daz: ', daz_range )
    mpi_print_log(mpi_log_fid, 1, False, 'gcd: ', gcd_range )
    mpi_print_log(mpi_log_fid, 1, False, 'selection of gc-center rect:   ', gc_center_rect )
    mpi_print_log(mpi_log_fid, 1, False, 'selection of gc-center circle: ', gc_center_circle )
    mpi_print_log(mpi_log_fid, 1, False, 'minmal number of recordings:   ', min_recordings )

    return ( (daz_range, gcd_range, gc_center_rect, gc_center_circle),
             (center_clo1, center_clo2, center_cla1, center_cla2),
             (circle_center_clo, circle_center_cla, circle_center_radius),
             flag_selection )
def distribute_jobs(mpi_rank, mpi_ncpu, fnm_wildcard, input_format, mpi_log_fid):
    """
    """
    global_jobs, local_jobs = list(), list()
    if input_format in ('sac', 'SAC'):
        global_jobs  = sorted( glob('/'.join(fnm_wildcard.split('/')[:-1] ) ) )
        remainders = fnm_wildcard.split('/')[-1]
        local_jobs = ['%s/%s' % (it, remainders) for it in  global_jobs[mpi_rank::mpi_ncpu]]
    elif input_format in ('hdf5', 'HDF5', 'h5', 'H5'):
        global_jobs = sorted(glob(fnm_wildcard) )
        local_jobs = global_jobs[mpi_rank::mpi_ncpu]

    mpi_print_log(mpi_log_fid, 0, False, 'Distribute MPI jobs')
    mpi_print_log(mpi_log_fid, 1, False, 'Total job size:    ', len(global_jobs) )
    mpi_print_log(mpi_log_fid, 1, True,  'Jobs on this node: ', len(local_jobs) )

    return local_jobs
def init_speedup(fftsize, delta, fs, critical_level=0.01):
    """
    Return the acceleration bound (i1, i2) for spetral computation.
    Data outside the [i1, i2) can be useless given specific frequency band fs=(f1, f2).
    """
    if fs == None:
        return 0, fftsize

    df = 1.0/(fftsize*delta)
    fmin, fmax = df, 0.5/delta-df # safe value
    f1, f2 = fs
    if f1 >= f2 or (f1<=fmin and f2>=fmax) or f1>=fmax or f2 <=fmin:
        return 0, fftsize

    i1, i2 = 0, fftsize
    x = zeros(fftsize, dtype=float32)
    x[fftsize//2] = 1.0 # Note, cannot use x[0] = 1.0
    if f1 <=fmin:
        iirfilter_f32(x, delta, 0, 0, f1, f2, 2, 2)
        amp = abs( rfft(x, fftsize) )
        c = amp.max() * critical_level
        i2 = argmax(amp<c)
    elif f2 >= fmax:
        iirfilter_f32(x, delta, 0, 1, f1, f2, 2, 2)
        amp = abs( rfft(x, fftsize) )
        c = amp.max() * critical_level
        i1 = argmax(amp>c)
    else:
        iirfilter_f32(x, delta, 0, 2, f1, f2, 2, 2)
        amp = abs( rfft(x, fftsize) )
        c = amp.max() * critical_level
        i1 = argmax(amp>c)
        i2 = i1 + argmax(amp[i1:]<c)
    return i1, i2
def mpi_print_log(file, n_pre, flush, *objects, end='\n'):
    """
    Set file=None to disable logging.
    """
    if file == None:
        return
    tmp = '>>> ' if n_pre<=0 else '    '*n_pre
    print(tmp, file=file, end='' )
    print(*objects, file=file, flush=flush, end=end )


def rd_wh_sac(fnm_wildcard, tmark, t1, t2, year_range, force_time_window, delta, npts,
    pre_detrend, pre_taper_halfsize, pre_filter,
    wtlen, wt_f1, wt_f2, wflen, speedup_i1, speedup_i2, whiten_taper_halfsize,
    fftsize, nrfft_valid,
    min_recordings,
    mpi_log_fid):
    """
    Read, whiten, and compute the spectra given sac filenames.
    The input sac files should have same `delta`.

    fnms, tmark, t1, t2, delta, npts:
        sac filenames and parameters for reading.
    pre_detrend, pre_taper_halfsize, pre_filter:
        Parameters for pre-processing.
    wtlen, wt_f1, wt_f2, wflen, speedup_i1, speedup_i2, whiten_taper_halfsize:
        Parameters for whitening.
    fftsize, nrfft_valid:
        Parameters for optaining spectra.
    """
    fnms = sorted(glob(fnm_wildcard))
    nsac = len(fnms)
    if nsac < min_recordings:
        mpi_print_log(mpi_log_fid, 2, False, '(n:%d) insufficent number of recordings' % (nsac) )
        return None, None, None, None, None, None, None, None, (0.0, 0.0, 0.0, 0.0)
    btype, f1, f2 = pre_filter if pre_filter != None else (None, None, None) # 0 for LP, 1 for HP, 2 for BP, 3 for BR

    spectra = zeros((nsac, nrfft_valid), dtype=complex64) #buffer to return
    stlo = zeros(nsac, dtype=float32 )
    stla = zeros(nsac, dtype=float32 )
    evlo = zeros(nsac, dtype=float32 )
    evla = zeros(nsac, dtype=float32 )
    az   = zeros(nsac, dtype=float32 )
    baz  = zeros(nsac, dtype=float32 )
    gcarc= zeros(nsac, dtype=float32 )

    ftw_tmark, ftw_t1, ftw_t2= force_time_window
    pre_taper_win = ones(npts, dtype=float32)
    taper(pre_taper_win, pre_taper_halfsize)
    time_rd, time_preproc, time_whiten, time_fft = 0.0, 0.0, 0.0, 0.0
    index = 0
    for it in fnms:
        ttmp = time()

        hdr = c_rd_sachdr(it)
        b, e = hdr.b, hdr.e
        tmp = (hdr.t0, hdr.t1, hdr.t2, hdr.t3, hdr.t4, hdr.t5, hdr.t6, hdr.t7, hdr.t8, hdr.t9, 0.0, 0.0, hdr.b, hdr.e, hdr.o, hdr.a, 0.0)
        ftw_t1 = tmp[ftw_tmark] + ftw_t1
        ftw_t2 = tmp[ftw_tmark] + ftw_t2
        if ftw_t1 < b or ftw_t2 > e:
            mpi_print_log(mpi_log_fid, 2, True, '+ Insufficient data within the forced time window:', it)
            continue

        st = c_rd_sac(it, tmark, t1, t2, True, False, False) # zeros will be used if gaps exist in sac recordings
        if (st is None) or (not st.dat.any()): # Check if Nan or all zeros.
            mpi_print_log(mpi_log_fid, 2, True, '+ Failure opening:', it)
            continue
        time_rd = time_rd + time()-ttmp

        ttmp = time()
        xs = st.dat
        try: # preproc
            if pre_detrend:
                detrend(xs)
            if pre_taper_halfsize > 0:
                xs[:pre_taper_halfsize] *= pre_taper_win[:pre_taper_halfsize]
                xs[-pre_taper_halfsize:] *= pre_taper_win[-pre_taper_halfsize:]
            if pre_filter:
                iirfilter_f32(xs, delta, 0, btype, f1, f2, 2, 2)
        except:
            mpi_print_log(mpi_log_fid, 2, True, '+ Failure preproc:', it)
            continue
        time_preproc = time_preproc + time()-ttmp

        ttmp = time()
        try: # whiten
            if wtlen != None:
                tnorm_f32(xs, delta, wtlen, wt_f1, wt_f2, 1.0e-5, whiten_taper_halfsize)
            if wflen != None:
                fwhiten_f32(xs, delta, wflen, 1.0e-5, whiten_taper_halfsize, speedup_i1, speedup_i2)
        except:
            mpi_print_log(mpi_log_fid, 2, True, '+ Failure whitening:', it)
            continue
        if (not xs.any()) or (isnan(xs).any() ): # check if the whitened data is valid
            mpi_print_log(mpi_log_fid, 2, True, '+ Failure whitening:', it)
            continue
        time_whiten = time_whiten + time()-ttmp

        ttmp = time()
        spectra[index] = rfft(xs, fftsize)[:nrfft_valid]
        hdr = st.hdr
        stlo[index] = hdr.stlo
        stla[index] = hdr.stla
        evlo[index] = hdr.evlo
        evla[index] = hdr.evla
        az[index]   = hdr.az
        baz[index]  = hdr.baz
        gcarc[index]= hdr.gcarc
        index = index + 1
        time_fft = time_fft + time()-ttmp

    if 0 < index < nsac: # just keep the correct data
        spectra = spectra[:index, :]
        stlo = stlo[:index]
        stla = stla[:index]
        evlo = evlo[:index]
        evla = evla[:index]
        az   = az[:index]
        baz  = baz[:index]
        gcarc= gcarc[:index]

    mpi_print_log(mpi_log_fid, 2, False, '(n:%d) rd(%.1fs) preproc(%.1fs) whiten(%.1fs) fft(%.1fs)' % (index, time_rd, time_preproc, time_whiten, time_fft), end='; ' )

    if index > 0: # return
        return spectra, stlo, stla, evlo, evla, az, baz, gcarc, (time_rd, time_preproc, time_whiten, time_fft)
    else:
        return None, None, None, None, None, None, None, None, (time_rd, time_preproc, time_whiten, time_fft)
def rd_wh_h5(fnm, tmark, t1, t2, year_range, force_time_window, delta, npts,
    pre_detrend, pre_taper_halfsize, pre_filter,
    wtlen, wt_f1, wt_f2, wflen, speedup_i1, speedup_i2, whiten_taper_halfsize,
    fftsize, nrfft_valid,
    min_recordings,
    mpi_log_fid):
    """
    Read, whiten, and compute the spectra given a hdf5 file.

    fnm, tmark, t1, t2, delta, npts:
        sac filenames and parameters for reading.
    pre_detrend, pre_taper_halfsize, pre_filter:
        Parameters for pre-processing.
    wtlen, wt_f1, wt_f2, wflen, speedup_i1, speedup_i2, whiten_taper_halfsize:
        Parameters for whitening.
    fftsize, nrfft_valid:
        Parameters for optaining spectra.
    """

    fid = h5_File(fnm, 'r')
    nsac = fid.attrs['nfile']
    if nsac < min_recordings:
        mpi_print_log(mpi_log_fid, 2, False, '(n:%d) insufficent number of recordings' % (nsac) )
        return None, None, None, None, None, None, None, None, (0.0, 0.0, 0.0, 0.0)

    btype, f1, f2 = pre_filter if pre_filter != None else (None, None, None)
    fnms = fid['filename'][:]

    spectra = zeros((nsac, nrfft_valid), dtype=complex64) #buffer to return
    stlo = zeros(nsac, dtype=float32 )
    stla = zeros(nsac, dtype=float32 )
    evlo = zeros(nsac, dtype=float32 )
    evla = zeros(nsac, dtype=float32 )
    az   = zeros(nsac, dtype=float32 )
    baz  = zeros(nsac, dtype=float32 )
    gcarc= zeros(nsac, dtype=float32 )

    time_rd, time_preproc, time_whiten, time_fft = 0.0, 0.0, 0.0, 0.0

    raw_mat = fid['dat']
    raw_stlo  = fid['hdr/stlo'][:]
    raw_stla  = fid['hdr/stla'][:]
    raw_evlo  = fid['hdr/evlo'][:]
    raw_evla  = fid['hdr/evla'][:]
    raw_az    = fid['hdr/az'][:]
    raw_baz   = fid['hdr/baz'][:]
    raw_gcarc = fid['hdr/gcarc'][:]
    raw_b     = fid['hdr/b'][:]
    raw_year  = fid['hdr/nzyear'][:]

    ftw_tmark, ftw_t1, ftw_t2= force_time_window
    tmp = ('t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 0, 'b', 'e', 'o', 'a', 0) # -3 is 'o' and -5 is 'b'
    ftw_t1 = fid['hdr'][tmp[ftw_tmark]][:] + ftw_t1
    ftw_t2 = fid['hdr'][tmp[ftw_tmark]][:] + ftw_t2
    b = fid['hdr/b'][:]
    e = fid['hdr/e'][:]

    tref = fid['hdr'][tmp[tmark]][:]
    t1 = tref - raw_b + t1
    t2 = tref - raw_b + t2

    min_year, max_year = year_range

    pre_taper_win = ones(npts, dtype=float32)
    taper(pre_taper_win, pre_taper_halfsize)
    index = 0
    for isac in range(nsac):
        it = fnms[isac]

        ttmp = time()

        if raw_year[isac] < min_year or raw_year[isac] > max_year:
            continue

        if ftw_t1[isac] < b[isac] or ftw_t2[isac] > e[isac]:
            mpi_print_log(mpi_log_fid, 2, True, '+ Insufficient data within the forced time window:', it)
            continue

        xs = raw_mat[isac][:]
        #print(delta, raw_b[isac], t1[isac], t2[isac] )
        xs, new_b = cut(xs, delta, raw_b[isac], t1[isac], t2[isac])
        if (not xs.any()) or (isnan(xs).any() ): # Check if Nan or all zeros.
            mpi_print_log(mpi_log_fid, 2, True, '+ Failure Nan:', it)
            continue
        time_rd = time_rd + time()-ttmp

        ttmp = time()
        try: # preproc
            if pre_detrend:
                detrend(xs)
            if pre_taper_halfsize > 0:
                xs[:pre_taper_halfsize] *= pre_taper_win[:pre_taper_halfsize]
                xs[-pre_taper_halfsize:] *= pre_taper_win[-pre_taper_halfsize:]
            if pre_filter:
                iirfilter_f32(xs, delta, 0, btype, f1, f2, 2, 2)
        except:
            mpi_print_log(mpi_log_fid, 2, True, '+ Failure preproc:', it)
            continue
        time_preproc = time_preproc + time() - ttmp

        ttmp = time()
        try: # whiten
            if wtlen != None:
                tnorm_f32(xs, delta, wtlen, wt_f1, wt_f2, 1.0e-5, whiten_taper_halfsize)
            if wflen != None:
                fwhiten_f32(xs, delta, wflen, 1.0e-5, whiten_taper_halfsize, speedup_i1, speedup_i2)
        except:
            mpi_print_log(mpi_log_fid, 2, True, '+ Failure whitening:', it)
            continue
        if (not xs.any()) or (isnan(xs).any() ): # check if the whitened data is valid
            mpi_print_log(mpi_log_fid, 2, True, '+ Failure whitening:', it)
            continue
        time_whiten = time_whiten + time() - ttmp

        ttmp = time()
        spectra[index] = rfft(xs, fftsize)[:nrfft_valid]
        stlo[index] = raw_stlo[isac]
        stla[index] = raw_stla[isac]
        evlo[index] = raw_evlo[isac]
        evla[index] = raw_evla[isac]
        az[index]   = raw_az[isac]
        baz[index]  = raw_baz[isac]
        gcarc[index]= raw_gcarc[isac]
        index = index + 1
        time_fft = time_fft + time() - ttmp

    fid.close()

    if 0 < index < nsac: # just keep the correct data
        spectra = spectra[:index, :]
        stlo = stlo[:index]
        stla = stla[:index]
        evlo = evlo[:index]
        evla = evla[:index]
        az   = az[:index]
        baz  = baz[:index]
        gcarc= gcarc[:index]

    mpi_print_log(mpi_log_fid, 2, False, '(n:%d) rd(%.1fs) preproc(%.1fs) whiten(%.1fs) fft(%.1fs)' % (index, time_rd, time_preproc, time_whiten, time_fft), end='; ' )

    if index > 0: # return
        return spectra, stlo, stla, evlo, evla, az, baz, gcarc, (time_rd, time_preproc, time_whiten, time_fft)
    else:
        return None, None, None, None, None, None, None, None, (time_rd, time_preproc, time_whiten, time_fft)

@jit(nopython=True, nogil=True)
def stack_bin_index(distance, dist_min, dist_step):
    """
    Return the index given stacking oarameters
    """
    idx = int( round((distance-dist_min) / dist_step) )
    return idx
@jit(nopython=True, nogil=True)
def round_daz(daz):
    daz = daz % 360
    if daz > 180.0:
        daz = 360.0-daz
    if daz > 90.0:
        daz = 180.0-daz
    return daz
@jit(nopython=True, nogil=True)
def spec_ccstack(spec_mat, lon, lat, gcarc, epdd,
    spec_stack_mat, stack_count,
    dist_min, dist_max, dist_step, speedup_i1, speedup_i2, random_sample):
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
    i1, i2 = speedup_i1, speedup_i2
    nstacks = stack_count.size
    nsac = spec_mat.shape[0]

    for isac1 in range(nsac):
        lo1, la1, gcarc1 = lon[isac1], lat[isac1], gcarc[isac1]
        spec1 = spec_mat[isac1]
        for isac2 in range(isac1, nsac):
            lo2, la2, gcarc2 = lon[isac2], lat[isac2], gcarc[isac2]

            dist = abs(gcarc1-gcarc2)
            if not epdd:
                dist = haversine(lo1, la1, lo2, la2)
            if dist > dist_max or dist < dist_min:
                continue
            idx = stack_bin_index(dist, dist_min, dist_step)
            if idx < 0 or idx >= nstacks:
                continue

            if 0 < random_sample and  random_sample < random(): # randomly resample
                continue

            spec2 = spec_mat[isac2]
            w   = conj(spec1[i1:i2]) * spec2[i1:i2]
            spec_stack_mat[idx][i1:i2] += w
            stack_count[idx] += 1
            count += 1

    return count
@jit(nopython=True, nogil=True)
def spec_ccstack2(spec_mat, lon, lat, gcarc, epdd,
    stack_mat, stack_count,
    dist_min, dist_max, dist_step, speedup_i1, speedup_i2, random_sample,
    azimuth, daz_min, daz_max,
    pt_lon, pt_lat, gcd_min, gcd_max,
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
    count = 0
    i1, i2 = speedup_i1, speedup_i2
    nstacks = stack_count.size
    nsac = spec_mat.shape[0]

    for isac1 in range(nsac):
        lo1, la1, az1, gcarc1 = lon[isac1], lat[isac1], azimuth[isac1], gcarc[isac1]
        spec1 = spec_mat[isac1]
        for isac2 in range(isac1, nsac):
            lo2, la2, az2, gcarc2 = lon[isac2], lat[isac2], azimuth[isac2], gcarc[isac2]

            dist = abs(gcarc1-gcarc2)
            if not epdd:
                dist = haversine(lo1, la1, lo2, la2)
            if dist < dist_min or dist > dist_max: # dist selection
                continue
            idx = stack_bin_index(dist, dist_min, dist_step)
            if idx<0 or idx>=nstacks:
                continue

            flag = 1 # 1 means the (clo, cla) is out of any rectangles
            (clo, cla), junk = great_circle_plane_center_triple(lo1, la1, lo2, la2, pt_lon, pt_lat, 1.0e-3)
            for clo1, clo2, cla1, cla2 in zip(rect_clo1, rect_clo2, rect_cla1, rect_cla2): # rect and circle selection
                if (cla1<=cla<=cla2) and ( clo1<=clo<=clo2 or (clo1>clo2 and (clo>=clo1 or clo<=clo2)) ):
                    flag = 0
                    break
            if flag == 1:
                continue

            flag = 1 # 1 means the (clo, cla) is out of any circles
            for cclo, ccla, cradius in zip(circle_center_clo, circle_center_cla, circle_center_radius):
                if haversine(clo, cla, cclo, ccla) <= cradius:
                    flag = 0
                    break
            if flag == 1:
                continue

            daz = round_daz(az1-az2)
            if daz < daz_min or daz > daz_max: # daz selection
                continue

            gcd = abs( point_distance_to_great_circle_plane(pt_lon, pt_lat, lo1, la1, lo2, la2) )
            if gcd < gcd_min or gcd > gcd_max: # gcd selection
                continue

            if 0 < random_sample and  random_sample < random(): # randomly resample
                continue

            spec2 = spec_mat[isac2]
            w   = conj(spec1[i1:i2]) * spec2[i1:i2]
            stack_mat[idx][i1:i2] += w
            stack_count[idx] += 1
            count += 1

    return count

#@jit(nopython=True, nogil=True)
def post_proc(spec_stack_mat, npts, delta,
    post_fold, post_detrend, post_taper_halfratio, post_filter, post_cut, post_scale,
    mpi_log_fid):
    """
    """
    fftsize = npts*2
    mpi_print_log(mpi_log_fid, 0, False, 'Ifft', )
    mpi_print_log(mpi_log_fid, 1, True,  'fftsize: ', fftsize, '(we will later get rid of the useless ZERO point in the cross-correlation.)')
    rollsize = npts-1
    nbin = spec_stack_mat.shape[0]
    time_mat = zeros( (nbin, fftsize-1), dtype=float32 )

    for irow in range(nbin): # ifft
        junk = irfft(spec_stack_mat[irow], fftsize)
        junk = roll(junk, rollsize)
        time_mat[irow] = junk[:-1] # get rid of the zero point


    mpi_print_log(mpi_log_fid, 0, True, 'Post processing...' )
    cc_t1, cc_t2 = -rollsize*delta, rollsize*delta # fold
    if post_fold:
        mpi_print_log(mpi_log_fid, 1, True, 'Folding...' )
        for irow in range(nbin):
            time_mat[irow] += time_mat[irow][::-1]
        time_mat = time_mat[:,rollsize:]
        cc_t1, cc_t2 = 0, rollsize*delta

    if post_detrend:
        mpi_print_log(mpi_log_fid, 1, True,  'Post detrend... ')
        for irow in range(nbin):
            time_mat[irow] = detrend(time_mat[irow])

    post_taper_halfsize = int(post_taper_halfratio * time_mat.shape[1])
    if post_taper_halfsize>0:
        mpi_print_log(mpi_log_fid, 1, True,  'Post taper...', post_taper_halfsize)
        for irow in range(nbin):
            taper(time_mat[irow], post_taper_halfsize)

    if post_filter:
        mpi_print_log(mpi_log_fid, 1, True, 'Post filtering... ', post_filter )
        btype, f1, f2 = post_filter
        for irow in range(nbin):
            iirfilter_f32(time_mat[irow], delta, 0, btype, f1, f2, 2, 2)

    if post_cut != None:
        post_t1, post_t2 = post_cut
        post_t1 = cc_t1 if post_t1 < cc_t1 else post_t1
        post_t2 = cc_t2 if post_t2 > cc_t2 else post_t2

        post_i1 = int( round((post_t1-cc_t1)/delta) )
        post_i2 = int( round((post_t2-cc_t1)/delta) ) + 1

        cc_t1 = cc_t1 + post_i1 *delta
        cc_t2 = cc_t1 + (post_i2-post_i1-1)*delta
        mpi_print_log(mpi_log_fid, 1, True, 'Post cutting... Correlation time range: ', (cc_t1, cc_t2) )
        time_mat = time_mat[:, post_i1:post_i2 ]

    absolute_amp = ones(nbin, dtype=float32 )
    if post_scale:
        mpi_print_log(mpi_log_fid, 1, True, 'Post scaling... ' )
        for irow in range(nbin):
            v = time_mat[irow].max()
            if v> 0.0:
                time_mat[irow] *= (1.0/v)
                absolute_amp[irow] = v

    return cc_t1, cc_t2, time_mat, absolute_amp
def output( stack_mat, stack_count, dist, absolute_amp,
            cc_t1, cc_t2, delta,
            output_fnm_prefix, output_format,
            mpi_log_fid):
    """
    """
    mpi_print_log(mpi_log_fid, 0, False, 'Outputting...', output_format)

    if 'hdf5' in output_format or 'h5' in output_format:
        h5_fnm = '%s.h5' % (output_fnm_prefix)
        mpi_print_log(mpi_log_fid, 1, True, 'hdf5: ', h5_fnm)

        f = h5_File(h5_fnm, 'w' )
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
        mpi_print_log(mpi_log_fid, 1, True, 'sac:  ', '%s_..._sac' % (output_fnm_prefix))
        for irow, it in enumerate(dist):
            fnm = '%s_%05.1f_.sac' % (output_fnm_prefix, it)
            out_hdr.user2 = dist[irow]
            out_hdr.user3 = stack_count[irow]
            out_hdr.user4 = absolute_amp[irow]
            c_wrt_sac(fnm, stack_mat[irow], out_hdr, False, False)
    pass



def main(mode,
    fnm_wildcard, tmark, t1, t2, delta, input_format='sac', year_range=(1000, 9999), force_time_window= None, # input param
    pre_detrend=True, pre_taper_halfratio= 0.005, pre_filter= None, # pre-proc param
    tnorm = (128.0, 0.02, 0.066666), swht= 0.02,                # whitening param
    stack_dist_range = (0.0, 180.0), stack_dist_step= 1.0,         # stack param
    daz_range= None, gcd_range= None, gc_center_rect= None, gc_center_circle=None, min_recordings=0, epdd=False, # selection param
    post_fold = False, post_detrend=False, post_taper_halfratio = 0.005, post_filter=None, post_cut= None, post_scale = False,           # post-proc param
    output_fnm_prefix= 'junk', output_format= ['hdf5'],                                                             # output param
    log_prefnm= 'cc_mpi_log', log_mode=None,           # log param
    spec_acc_threshold = 0.01, random_sample=-1.0 ):  # other param
    """
    """
    tc_main = time()
    mpi_comm, mpi_rank, mpi_ncpu, mpi_log_fid = init_mpi(log_prefnm, log_mode) # mpi

    speedup_fs = None
    if post_filter != None:
        speedup_f1, speedup_f2 = post_filter[1], post_filter[2]
        if swht != None:
            speedup_f1 = speedup_f1 - swht
            speedup_f2 = speedup_f2 + swht
            speedup_fs = (speedup_f1, speedup_f2) # dont worry if speedup_fs is invalid as init_speedup(...) process whatever cases
    tmp = init_parameter(mode, tmark, t1, t2, delta, input_format, pre_detrend, pre_taper_halfratio, pre_filter, speedup_fs, spec_acc_threshold, mpi_log_fid)
    flag_mode, (npts, fftsize, df, speedup, pre_taper_halfsize), (cc_npts, cc_fftsize, cc_df, cc_speedup) = tmp


    dist, spec_stack_mat, stack_count = init_ccstack_spec_buf(stack_dist_range[0], stack_dist_range[1], stack_dist_step, cc_speedup[1], mpi_log_fid ) # init buf
    global_dist, global_spec_stack_mat, global_stack_count = dist, spec_stack_mat, stack_count
    if mpi_rank == 0 and mpi_ncpu > 1:
        global_dist, global_spec_stack_mat, global_stack_count = init_ccstack_spec_buf(stack_dist_range[0], stack_dist_range[1], stack_dist_step, cc_speedup[1], mpi_log_fid )


    tmp = update_selection(daz_range, gcd_range, gc_center_rect, gc_center_circle, min_recordings, mpi_log_fid)
    (daz_range, gcd_range, gc_center_rect, gc_center_circle), (rect_clo1, rect_clo2, rect_cla1, rect_cla2), (circle_center_clo, circle_center_cla, circle_center_radius), flag_selection  = tmp

    local_jobs = distribute_jobs(mpi_rank, mpi_ncpu, fnm_wildcard, input_format, mpi_log_fid)
    func_rd = rd_wh_sac if input_format in ('sac', 'SAC') else rd_wh_h5
    wtlen, wt_f1, wt_f2 = tnorm if tnorm != None else (None, None, None)
    wflen = swht
    mpi_print_log(mpi_log_fid, 0, True, 'Start running...' )
    time_rd_sum, time_preproc_sum, time_whiten_sum, time_fft_sum, time_cc_sum, time_sum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for idx, it in enumerate(local_jobs):
        mpi_print_log(mpi_log_fid, 1, True, '- %d %s' % (idx+1, it) )
        tmp = func_rd(  it, tmark, t1, t2, year_range, force_time_window, delta, npts,
                        pre_detrend, pre_taper_halfsize, pre_filter,
                        wtlen, wt_f1, wt_f2, wflen, 0, speedup[1], pre_taper_halfsize,
                        cc_fftsize, cc_speedup[1],
                        min_recordings,
                        mpi_log_fid)

        spec_mat, stlo, stla, evlo, evla, az, baz, gcarc, (time_rd, time_preproc, time_whiten, time_fft) = tmp
        time_rd_sum      = time_rd_sum       + time_rd
        time_preproc_sum = time_preproc_sum  + time_preproc
        time_whiten_sum  = time_whiten_sum   + time_whiten
        time_fft_sum     = time_fft_sum      + time_fft
        if stlo is None:
            continue
        if stlo.size < min_recordings:
            continue
        lon, lat, azimuth = (stlo, stla, az) if flag_mode == 0 else (evlo, evla, baz)
        pt_lon, pt_lat = (evlo[0], evla[0])  if flag_mode == 0 else (stlo[0], stla[0])


        ttmp = time()
        if flag_selection:
            ncc = spec_ccstack2(  spec_mat, lon, lat, gcarc, epdd,
                                  spec_stack_mat, stack_count,
                                  stack_dist_range[0], stack_dist_range[1], stack_dist_step, cc_speedup[0], cc_speedup[1], random_sample,
                                  azimuth, daz_range[0], daz_range[1],
                                  pt_lon, pt_lat, gcd_range[0], gcd_range[1],
                                  rect_clo1, rect_clo2, rect_cla1, rect_cla2,
                                  circle_center_clo, circle_center_cla, circle_center_radius)
        else:
            ncc = spec_ccstack(   spec_mat, lon, lat, gcarc, epdd,
                                  spec_stack_mat, stack_count,
                                  stack_dist_range[0], stack_dist_range[1], stack_dist_step, cc_speedup[0], cc_speedup[1], random_sample)
        tmp = time()-ttmp
        time_cc_sum = time_cc_sum + tmp
        mpi_print_log(mpi_log_fid, 2, True, '(ncc: %d) ccstack(%.1fs)' % (ncc, tmp) )

    time_sum = time_rd_sum + time_preproc_sum + time_whiten_sum + time_fft_sum + time_cc_sum + 0.01
    mpi_print_log(mpi_log_fid, 1, True, 'Time consumption summary: rd: %.1fs(%d%%), preproc: %.1fs(%d%%)), whiten: %.1fs(%d%%)), fft: %.1fs(%d%%)), cc: %.1fs(%d%%)) ' % (
                time_rd_sum,      time_rd_sum/time_sum*100,
                time_preproc_sum, time_preproc_sum/time_sum*100,
                time_whiten_sum,  time_whiten_sum/time_sum*100,
                time_fft_sum,     time_fft_sum/time_sum*100,
                time_cc_sum,      time_cc_sum/time_sum*100,
            )
    )


    if mpi_ncpu > 1:
        mpi_print_log(mpi_log_fid, 0, True, 'MPI collecting(reducing) to RANK0...')
        mpi_comm.Reduce([spec_stack_mat, MPI.C_FLOAT_COMPLEX], [global_spec_stack_mat, MPI.C_FLOAT_COMPLEX], MPI.SUM, root= 0)
        mpi_comm.Reduce([stack_count, MPI.INT32_T], [global_stack_count, MPI.INT32_T], MPI.SUM, root= 0 )
        mpi_comm.Barrier()

    if mpi_rank == 0:
        tmp = post_proc(global_spec_stack_mat, npts, delta,
                        post_fold, post_detrend, post_taper_halfratio, post_filter, post_cut, post_scale,
                        mpi_log_fid)
        cc_t1, cc_t2, time_mat, absolute_amp = tmp
        output( time_mat, global_stack_count, dist, absolute_amp,
                cc_t1, cc_t2, delta,
                output_fnm_prefix, output_format,
                mpi_log_fid)

    mpi_print_log(mpi_log_fid, 0, True,  'Done (%.1f sec)' % (time()-tc_main) )
    if mpi_log_fid != None:
        mpi_log_fid.close()




HMSG = """%s --mode r2r -I "in*/*.sac" -T -5/10800/32400 -D 0.1 --input_format sac
    [--year_range 2010/2020]
    [--force_time_window -5/10800/32000] -O cc_stack --out_format hdf5
    [--pre_detrend] [--pre_taper 0.005] [--pre_filter bandpass/0.005/0.1]
    --stack_dist 0/180/1 [--daz -0.1/15] [--gcd -0.1/20] [--gc_center_rect 120/180/0/40,180/190/0/10]
    [--gc_center_circle 100/-20/10,90/0/15] [--min_recordings 10] [--epdd]
    [--w_temporal 128.0/0.02/0.06667] [--w_spec 0.02]
    [--post_fold] [--post_taper 0.05] [--post_filter bandpass/0.02/0.0666] [--post_norm] [--post_cut]
     --log cc_log  [--log_mode 0] [--acc=0.01] [--random_sample 0.6]

Args:
    #0. Mode:
    --mode: 'r2r' or 's2s'.

    #1. I/O and pre-processing args:
    -I  : filename wildcards.
    -T  : cut time window for reading.
        For some time series that have gaps in the window, the gaps are filled with zeros.
    -D  : delta.
    --year_range : A year range to select input data.
                   (Only effective when  using `--input_format h5`).
    --force_time_window: force a time window to select input time series.
        Any time serie that has gaps in the window are excluded.
        This option is off in default.
    --input_format : 'sac' pr 'h5;.
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
    --acc : acceleration threshold. (default is 0.01)
    --random_sample : random resample the cross-correlation functions in the stacking.  (a value between 0 and 1)

E.g.,
    %s --mode r2r -I "in*/*.sac" -T -5/10800/32400 -D 0.1 --input_format sac \
--year_range 2010/2020 \
--force_time_window -5/10800/32000 \
-O cc --out_format hdf5,sac \
--pre_detrend --pre_taper 0.005 \
--w_temporal 128.0/0.02/0.06667 --w_spec 0.02 \
--stack_dist 0/180/1 \
--daz -0.1/20 --gcd -0.1/30 --gc_center_rect 120/180/0/40 --gc_center_circle 100/-20/15,90/0/15 \
--post_fold --post_taper 0.005 --post_filter bandpass/0.01/0.06667 --post_norm  --post_cut 0/5000 \
--log cc_log --log_mode=0 --acc 0.01 --random_sample 0.6

    """
if __name__ == "__main__":
    mode = 'r2r'

    fnm_wildcard = ''
    input_format = 'sac'
    tmark, t1, t2 = -5, 10800, 32400
    year_range = (1000, 9999)
    force_time_window = None
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
    post_detrend = False
    post_filter = None #('bandpass', 0.02, 0.066666)
    post_norm = False
    post_cut = None

    log_prefnm= 'cc_mpi_log'
    log_mode  = None

    spec_acc_threshold = 0.01
    random_sample = -1.0
    ######################
    if len(sys.argv) <= 1:
        print(HMSG % (sys.argv[0], sys.argv[0]), flush=True)
        sys.exit(0)
    ######################
    ######################
    options, remainder = getopt(sys.argv[1:], 'I:T:D:O:',
                            ['mode=',
                             'in_format=', 'input_format=', 'year_range=',
                             'force_time_window=',
                             'pre_detrend', 'pre_taper=', 'pre_filter=',
                             'out_format=',
                             'w_temporal=', 'w_spec=',
                             'stack_dist=', 'daz=', 'dbaz=', 'gcd=', 'gcd_ev=', 'gcd_rcv=', 'gc_center_rect=', 'gc_center_circle=', 'min_recordings=', 'min_ev_per_rcv=', 'min_rcv_per_ev=', 'epdd=',
                             'post_fold', 'post_taper=', 'post_filter=', 'post_norm', 'post_cut=',
                             'log=', 'log_mode=', 'acc=', 'random_sample='] )
    for opt, arg in options:
        if opt in ('--mode'):
            mode = arg
        elif opt in ('-I'):
            fnm_wildcard = arg
        elif opt in ('--input_format'):
            input_format = arg
        elif opt in ('-T'):
            tmark, t1, t2 = arg.split('/')
            tmark = int(tmark)
            t1 = float(t1)
            t2 = float(t2)
        elif opt in ('--year_range'):
            year_range = [int(it) for it in arg.split('/')]
        elif opt in ('--force_time_window'):
            v1, v2, v3 = arg.split('/')
            force_time_window = (int(v1), float(v2), float(v3) )
        elif opt in ('-D'):
            delta = float(arg)
        elif opt in ('--pre_detrend'):
            pre_detrend = True
        elif opt in ('--pre_taper'):
            pre_taper_ratio = float(arg)
        elif opt in ('--pre_filter'):
            vol = {'LP':0, 'HP':1, 'BP': 2, 'BR': 3, 'lowpass':0, 'highpass':1, 'bandpass':2}
            btype, f1, f2 = arg.split('/')
            pre_filter = (vol[btype], float(f1), float(f2) )
        elif opt in ('--in_format', '--input_format'):
            input_format = arg
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
            vol = {'LP':0, 'HP':1, 'BP': 2, 'BR': 3, 'lowpass':0, 'highpass':1, 'bandpass':2}
            btype, f1, f2 = arg.split('/')
            post_filter = (vol[btype], float(f1), float(f2) )
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
        elif opt in ('--random_sample'):
            random_sample = float(arg)
    #######
    main(mode,
        fnm_wildcard, tmark, t1, t2, delta, input_format, year_range,
        force_time_window,
        pre_detrend, pre_taper_ratio, pre_filter,
        tnorm, swht,
        dist_range, dist_step,
        daz_range, gcd_range, gc_center_rect, gc_center_circle, min_recordings, epdd,
        post_fold, post_detrend, post_taper_ratio, post_filter, post_cut, post_norm,
        output_pre_fnm, output_format,
        log_prefnm, log_mode,
        spec_acc_threshold, random_sample)


