#!/usr/bin/env python3
"""
Executable files for cross-correlation and stacking operations
"""
from time import gmtime, strftime
import time
import sacpy.processing as processing
import sacpy.sac_hdf5 as sac_hdf5
import sacpy.geomath as geomath
import mpi4py
import h5py
import pyfftw
import numpy as np 
import sys
import getpass
import getopt

def mpi_print_log(msg, n_pre, file=sys.stdout, flush=False):
    print('%s%s' % ('\t'*n_pre, msg), file= file, flush= flush )

class cc_stack_rcv_pairs:
    def __init__(self, fnm_lst_alignedSac2Hdf5, log_prefnm = '', username= ''):
        self.all_fnm_lst_alignedSac2Hdf5 = sorted( list(fnm_lst_alignedSac2Hdf5 ) )
        self.username = username
        ### MPI settings
        self.comm = mpi4py.MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.ncpu = self.comm.Get_size()
        ### local variables
        self.local_stacked_cc_spectra = None
        self.local_stacked_cc_count = None
        self.local_log_fnm = '%s_mpi_log_%03d.txt' % (log_prefnm, self.rank)
        self.local_log_fp = open(self.local_log_fnm, 'w')
        self.local_fnm_lst_alignedSac2Hdf5 = []
        ### global variables that should be same
        self.global_cut_marker = 'o'
        self.global_cut_t1 = 0.0
        self.global_cut_t2 = 0.0
        self.global_npts = 0
        self.global_npts_cc = 0
        self.global_nrfft = 0
        self.global_ncc = 0
        self.global_dt = 1.0
        self.global_sampling_rate = 1.0 # sampling rate
        self.global_df = 1.0
        #
        self.global_twin = 0.0
        self.global_twin_len = 0
        self.global_f1, self.global_f2 = 0.0, 1.0
        self.global_fwin = 0.0
        self.global_fwin_len = 0.0
        #
        self.global_select_stack = False
        self.global_select_stack_az_diff_deg_max = 720.0 # that means all cc functions are selected
        self.global_select_stack_az_reverse = False      # set True to turn include into exclude
        self.global_select_stack_dis_to_greate_circle_plane_deg = 720.0 # that means all cc functions are selected
        self.global_select_stack_gc_reverse = False                     # set True to turn include into exclude
        #
        self.global_inter_rcv_distance_deg_min = 0.0
        self.global_inter_rcv_distance_deg_max = 180.0
        self.global_inter_rcv_distance_step_deg = 1.0
        self.global_inter_rcv_distance_deg_lst = None
        ### variable_at_RANK0
        self.rank0_stacked_cc_spectra = None
        self.rank0_stacked_cc_count = None
        self.rank0_stacked_cc_time = None
    def init(self, inter_rcv_distance_range_deg, distance_step_deg, cut_marker, cut_t1, cut_t2, twin, f1, f2, fwin,
        az_diff_max_deg= None, az_reverse= None, gc_dis_max_deg= None, gc_reverse= None):
        self.set_stack_inter_rcv_distance(inter_rcv_distance_range_deg, distance_step_deg)
        if az_diff_max_deg != None and az_reverse != None and gc_dis_max_deg != None and gc_reverse != None:
            self.set_select_stack(az_diff_deg_max, az_reverse, gc_dis_max_deg, gc_reverse)
        self.set_local_stacked_cc_spectra(cut_marker, cut_t1, cut_t2)
        self.set_whitening_parameter(twin, f1, f2, fwin)
        self.set_local_work_h5_lst()
    def run(self):
        self.local_run()
        self.comm.Barrier()
        self.collect_stack()
    def output2hdf5(self, fnm_hdf5):
        """
        Output stacked cc to hdf5 file
        """
        if self.rank == 0:
            ###
            mpi_print_log('>>> output (in hdf5 formate) to %s' % (fnm_hdf5), 0, self.local_log_fp, True)
            ### info
            app = h5py.File(fnm_hdf5, 'w')
            app.attrs['user'] = self.username
            app.attrs['timestamp'] = strftime("%Y-%m-%d %H:%M:%S", gmtime() )
            if self.global_select_stack:
                az_operater = '<=' if self.global_select_stack_az_reverse else '>'
                gc_operater = '<=' if self.global_select_stack_gc_reverse else '>'
                app.attrs['info'] = 'selective stacking (az %s %f) (gc %s %f)' % (
                        az_operater, self.global_select_stack_az_diff_deg_max,
                        gc_operater, self.global_select_stack_gc_reverse)
            else:
                app.attrs['info'] = 'stacking using all cross-correlation functions'
            ### stacked spectrum
            dset = app.create_dataset('cc_stacked_spectra', data= self.rank0_stacked_cc_spectra)
            dset.attrs['df'] = self.global_df
            ### stacked time series
            dset = app.create_dataset('cc_stacked_time', data= self.rank0_stacked_cc_time)
            dset.attrs['info'] = 'untappered and unrolled cc time-series'
            dset.attrs['dt'] = self.global_dt
            ### number of cc for each stacked bin
            dset = app.create_dataset('cc_stacked_count', data= self.rank0_stacked_cc_count)
            ### inter-receiver distance bin
            dset = app.create_dataset('distance_bin', data= self.global_inter_rcv_distance_deg_lst)
            dset.attrs['info'] = 'inter-receiver distance (degree) at the middle of each stacked bin.  '
            app.close()
    def set_stack_inter_rcv_distance(self, inter_rcv_distance_range_deg= [0.0, 180.0], distance_step_deg= 1.0):
        """
        Set stack parameters
        """
        if self.rank == 0:
            d1, d2 = inter_rcv_distance_range_deg
            dd = distance_step_deg
            self.global_ncc = int( np.floor((d2-d1)/dd) )
            self.global_inter_rcv_distance_deg_min = d1
            self.global_inter_rcv_distance_deg_max = d1 + dd * self.global_ncc
            self.global_inter_rcv_distance_step_deg = dd
            self.global_inter_rcv_distance_deg_lst = np.arange(0.0, self.global_ncc, 1.0) * dd + d1 + dd*0.5
            ### msg out
            msg = '>>> bcast send from RANK 0 for stack routine: inter_rcv_distance [%d, %d] step(%f)' % (
                    self.global_inter_rcv_distance_deg_min,
                    self.global_inter_rcv_distance_deg_max,
                    self.global_inter_rcv_distance_step_deg )
            mpi_print_log(msg, 0, self.local_log_fp, True)
        ###
        self.comm.Barrier()
        ###
        self.global_ncc                         = self.comm.bcast(self.global_ncc                        , root= 0)
        self.global_inter_rcv_distance_deg_min  = self.comm.bcast(self.global_inter_rcv_distance_deg_min , root= 0)
        self.global_inter_rcv_distance_deg_max  = self.comm.bcast(self.global_inter_rcv_distance_deg_max , root= 0)
        self.global_inter_rcv_distance_step_deg = self.comm.bcast(self.global_inter_rcv_distance_step_deg, root= 0)
        self.global_inter_rcv_distance_deg_lst  = self.comm.bcase(self.global_inter_rcv_distance_deg_lst,  root= 0)
        ###
        self.comm.Barrier()
        if self.rank != 0:
            msg = '>>> bcast receive from RANK 0 for stack routine: inter_rcv_distance [%d, %d] step(%f)' % (
                    self.global_inter_rcv_distance_deg_min,
                    self.global_inter_rcv_distance_deg_max,
                    self.global_inter_rcv_distance_step_deg )
            mpi_print_log(msg, 0, self.local_log_fp, True)
    def set_select_stack(self, az_diff_max_deg=10, az_reverse=False, dis_to_greate_circle_plane_deg=10, gc_reverse=False):
        """
        Call this function to set `select` mode in stacking cross-correlation functions
        """
        if self.rank == 0:
            self.global_select_stack = True
            self.global_select_stack_az_diff_deg_max = az_diff_max_deg
            self.global_select_stack_az_reverse = az_reverse
            self.global_select_stack_dis_to_greate_circle_plane_deg = dis_to_greate_circle_plane_deg
            self.global_select_stack_gc_reverse = gc_reverse
            ###
            az_operater = '<=' if az_reverse else '>'
            gc_operater = '<=' if gc_reverse else '>'
            msg = '>>> bcast send from RANK 0. Turn on `select` mode in cross-correlation stacking. (az %s %f) (dis %s %f)' % (
                az_operater, az_diff_max_deg, gc_operater, dis_to_greate_circle_plane_deg)
            mpi_print_log(msg, 0, self.local_log_fp, True)
        ###
        self.comm.Barrier()
        ###
        self.global_select_stack = self.comm.bcast(self.global_select_stack, root= 0)
        self.global_select_stack_az_diff_deg_max                = self.comm.bcast(self.global_select_stack_az_diff_deg_max, 0)
        self.global_select_stack_az_reverse                     = self.comm.bcast(self.global_select_stack_az_reverse                    , root= 0)
        self.global_select_stack_dis_to_greate_circle_plane_deg = self.comm.bcast(self.global_select_stack_dis_to_greate_circle_plane_deg, root= 0)
        self.global_select_stack_gc_reverse                     = self.comm.bcast(self.global_select_stack_gc_reverse                    , root= 0)
        ###
        self.comm.Barrier()
        ###
        if self.rank != 0:
            az_operater = '<=' if az_reverse else '>'
            gc_operater = '<=' if gc_reverse else '>'
            msg = '>>> bcast receive from RANK 0. Turn on `select` mode in cross-correlation stacking. (az %s %f) (dis %s %f)' % (
                az_operater, az_diff_max_deg, gc_operater, dis_to_greate_circle_plane_deg)
            mpi_print_log(msg, 0, self.local_log_fp, True)
    def set_local_stacked_cc_spectra(self, cut_marker, cut_t1, cut_t2):
        """
        Make local stacked crosscorrelation spectra mat.
        """
        if self.rank == 0: #
            app = sac_hdf5.alignedSac2Hdf5(self.username)
            app.fromH5(self.all_fnm_lst_alignedSac2Hdf5[0], self.username, open_mode='r' )
            junk = app.get_raw_sac(cut_marker=cut_marker, cut_range= [cut_t1, cut_t2] )
            self.global_cut_marker = cut_marker
            self.global_cut_t1 = cut_t1
            self.global_cut_t2 = cut_t2
            self.global_npts = junk.shape[1] # npts are same in all *.h5 file
            self.global_dt = app.h5['raw_sac/data'].attrs['dt']
            self.global_sampling_rate = 1.0/self.global_dt
            del app # delete useless object
            self.global_npts_cc = 2*self.global_npts-1       # the npts for cross-correlation functions (zero-padding is considered)
            self.global_nrfft = self.global_npts_cc // 2 + 1 # the size of spectrum generated by rfft
            self.global_df = 1.0 / (self.global_npts_cc* self.global_dt)
            ### msg out on rank 0
            msg = '>>> bcast send from RANK 0: dt(%f) fs(%f) df(%f) cut(%c,%f,%f) npts(%d), npts_cc(%d), nrfft(%d)' % (
                self.global_dt, self.global_sampling_rate, self.global_df,
                self.global_cut_marker, self.global_cut_t1, self.global_cut_t2, self.global_npts, self.global_npts_cc, self.global_nrfft)
            mpi_print_log(msg, 0, self.local_log_fp, flush=True)
        ###
        self.comm.Barrier()
        ###
        self.global_cut_marker    = self.comm.bcast(self.global_cut_marker   , root= 0)
        self.global_cut_t1        = self.comm.bcast(self.global_cut_t1       , root= 0)
        self.global_cut_t2        = self.comm.bcast(self.global_cut_t2       , root= 0)
        self.global_dt            = self.comm.bcast(self.global_dt           , root= 0)
        self.global_sampling_rate = self.comm.bcast(self.global_sampling_rate, root= 0)
        self.global_df            = self.comm.bcast(self.global_df           , root= 0)
        self.global_npts          = self.comm.bcast(self.global_npts         , root= 0)
        self.global_npts_cc       = self.comm.bcast(self.global_npts_cc      , root= 0)
        self.global_nrfft         = self.comm.bcast(self.global_nrfft        , root= 0)
        ###
        self.comm.Barrier()
        ### msg out on other ranks
        if self.rank != 0:
            msg = '>>> bcast receive from RANK 0: dt(%f) fs(%f) df(%f) cut(%c,%f,%f) npts(%d), npts_cc(%d), nrfft(%d)' % (
                self.global_dt, self.global_sampling_rate, self.global_df,
                self.global_cut_marker, self.global_cut_t1, self.global_cut_t2, self.global_npts, self.global_npts_cc, self.global_nrfft)
            mpi_print_log(msg, 0, self.local_log_fp, flush=True)
        ### make local stack cc spectra pool
        self.local_stacked_cc_spectra = np.zeros( (self.global_ncc, self.global_nrfft), dtype=np.complex64 )
        self.local_stacked_cc_count   = np.zeros( self.global_ncc, dtype=np.int32 )
        ### msg out
        msg = '>>> Init spectral pool for stacked crosscorrelations. (%d,%d) by np.complex64' % (self.global_ncc, self.global_nrfft)
        mpi_print_log(msg, 0, self.local_log_fp, True)
    def set_whitening_parameter(self, twin, f1, f2, fwin):
        """
        twin (sec), f1(hz), f2(hz): parameters for temporal normalization
        f_win(hz): parameter for spectral whitening
        """
        if self.rank == 0:
            self.global_twin = twin
            self.global_f1 = f1
            self.global_f2 = f2
            self.global_fwin = fwin
            self.global_twin_len = int( np.ceil(twin/self.global_dt) )
            self.global_fwin_len = int( np.ceil(fwin/self.global_df) )
            ###
            msg = '>>> Init and bcast whitening parameter sent from RANK0. twin(%f, %d) f1f2(%f,%f) fwin(%f,%d)' % (
                    self.global_twin, self.global_twin_len, self.global_f1, self.global_f2,
                    self.global_fwin, self.global_fwin_len)
            mpi_print_log(msg, 0, self.local_log_fp, True)
        ###
        self.comm.Barrier()
        ###
        self.global_twin     = self.comm.bcast(self.global_twin    , root=0 )
        self.global_twin_len = self.comm.bcast(self.global_twin_len, root=0 )
        self.global_f1       = self.comm.bcast(self.global_f1      , root=0 )
        self.global_f2       = self.comm.bcast(self.global_f2      , root=0 )
        self.global_fwin     = self.comm.bcast(self.global_fwin    , root=0 )
        self.global_fwin_len = self.comm.bcast(self.global_fwin_len, root=0 )
        ###
        self.comm.Barrier()
        if self.rank != 0:
            msg = '>>> Init and bcast whitening parameter receive from RANK0. twin(%f, %d) f1f2(%f,%f) fwin(%f,%d)' % (
                    self.global_twin, self.global_twin_len, self.global_f1, self.global_f2,
                    self.global_fwin, self.global_fwin_len)
            mpi_print_log(msg, 0, self.local_log_fp, True)
    def set_local_work_h5_lst(self):
        """
        set local list of sac_hdf5 files that are used for latter fft, cc, and stacking
        """
        chunksize = 0
        if self.rank == 0:
            chunksize = (len(self.all_fnm_lst_alignedSac2Hdf5) // self.ncpu) + 1
        ###
        self.comm.Barrier()
        ###
        chunksize = self.comm.bcast(chunksize, root= 0)
        ###
        self.comm.Barrier()
        ###
        i1 = chunksize * self.rank
        i2 = i1 + chunksize
        self.local_fnm_lst_alignedSac2Hdf5 = self.all_fnm_lst_alignedSac2Hdf5[i1:i2]
        msg = '>>> Init sac_hdf5 files to process (%d): [%d,%d) :\n\t' % (len(self.local_fnm_lst_alignedSac2Hdf5), i1, i2)
        msg += '\n\t'.join(self.local_fnm_lst_alignedSac2Hdf5)
        mpi_print_log(msg, 0, self.local_log_fp, True)
    def local_run(self):
        worksize= len(self.local_fnm_lst_alignedSac2Hdf5)
        if not self.global_select_stack: ### stack all
            mpi_print_log('>>> Run spectral cc and all stack...', 0, self.local_log_fp, True)
            for idx, fnm in enumerate(self.local_fnm_lst_alignedSac2Hdf5):
                mpi_print_log('%s index(%d/%d)'% (fnm, idx+1, worksize), 1, self.local_log_fp, True)
                self.run_all_cc_spectra_from_single_hdf5_file(fnm)
        else: ### select stack
            mpi_print_log('>>> Run spectral cc and selective stack...', 0, self.local_log_fp, True)
            for idx, fnm in enumerate(self.local_fnm_lst_alignedSac2Hdf5):
                mpi_print_log('%s index(%d/%d)'% (fnm, idx+1, worksize), 1, self.local_log_fp, True)
                self.run_select_cc_spectra_from_single_hdf5_file(fnm, 
                        self.global_select_stack_az_reverse, self.global_select_stack_az_reverse,
                        self.global_select_stack_dis_to_greate_circle_plane_deg, self.global_select_stack_gc_reverse)
    def run_all_cc_spectra_from_single_hdf5_file(self, fnm_alignedSac2Hdf5):
        """
        Use all time-series to get stacked cross-correlations with respect to inter-receiver distance
        """
        app = sac_hdf5.alignedSac2Hdf5(self.username)
        app.fromH5(fnm_alignedSac2Hdf5, self.username, open_mode='r')
        stlo = app.h5['raw_sac/hdr/stlo'][:]
        stla = app.h5['raw_sac/hdr/stla'][:]
        evlo = app.h5['raw_sac/hdr/evlo'][0]
        evla = app.h5['raw_sac/hdr/evla'][0]
        time_series = app.get_raw_sac(cut_marker= self.global_cut_marker, cut_range= [self.global_cut_t1, self.global_cut_t2] )
        ntr = time_series.shape[0]
        ### get whitened spectra
        spectra = np.zeros((ntr, self.global_nrfft), dtype=np.complex64)
        for idx, row in enumerate(time_series):
            tr = processing.temporal_normalization(row, self.global_sampling_rate, self.global_twin_len, self.global_f1, self.global_f2)
            spectra[idx] = processing.frequency_whiten_spec(tr, self.global_sampling_rate, self.global_fwin_len, self.global_npts_cc)
        ### cross correlation
        msg_template = 'cc ev (%.2f %.2f)' % (evlo, evla)  + 'rcv(%.2f %.2f)x(%.2f %.2f) dist(%.2f) stack_cc(%.2f,%d) %s' 
        for idx1 in range(ntr):
            stlo1, stla1 = stlo[idx1], stla[idx1]
            for idx2 in range(ntr):
                stlo2, stla2 = stlo[idx2], stla[idx2]
                distance = geomath.haversine(stlo1, stla1, stlo2, stla2) # return degree
                if distance > self.global_inter_rcv_distance_deg_max or distance < self.global_inter_rcv_distance_deg_min:
                    mpi_print_log(msg_template % (stlo1, stla1, stlo2, stla2, distance, np.nan, np.nan, 'N'), 2, self.local_log_fp, False )
                    continue
                ### cc and stack
                tmp = spectra[idx1] * np.conj(spectra[idx2])
                if True not in (np.isnan(tmp) ):
                    idx_cc = int( (distance - self.global_inter_rcv_distance_deg_min)/self.global_inter_rcv_distance_step_deg)
                    mpi_print_log(msg_template % (stlo1, stla1, stlo2, stla2, distance, 
                            self.global_inter_rcv_distance_deg_lst[idx_cc], idx_cc, 'Y'), 2, self.local_log_fp, False )
                    self.local_stacked_cc_spectra[idx_cc, :] += tmp
                    self.local_stacked_cc_count[idx_cc] += 1
        app.h5.close()
        del spectra
        del time_series
        del app
    def run_select_cc_spectra_from_single_hdf5_file(self, fnm_alignedSac2Hdf5, 
            az_diff_max_deg=10, az_reverse=False,
            dis_to_greate_circle_plane_deg=10, gc_reverse=False):
        """
        Use selected time-series to get stacked cross-correlations with respect to inter-receiver distance.
        (1) Azimuth difference and (2) distance from event to inter-receiver great-circle-plane are used to 
            select cross-correlation functions.
        """
        app = sac_hdf5.alignedSac2Hdf5(self.username)
        app.fromH5(fnm_alignedSac2Hdf5, self.username, open_mode='r')
        stlo = app.h5['raw_sac/hdr/stlo'][:]
        stla = app.h5['raw_sac/hdr/stla'][:]
        evlo = app.h5['raw_sac/hdr/evlo'][0]
        evla = app.h5['raw_sac/hdr/evla'][0]
        az   = [geomath(evlo, evla, lo, la) for (lo, la) in zip(stlo, stla) ] # return degree
        ###
        time_series = app.get_raw_sac(cut_marker= self.global_cut_marker, cut_range= [self.global_cut_t1, self.global_cut_t2] )
        ntr = time_series.shape[0]
        ### get whitened spectra
        spectra = np.zeros((ntr, self.global_nrfft), dtype=np.complex64)
        for idx, row in enumerate(time_series):
            tr = processing.temporal_normalization(row, self.global_sampling_rate, self.global_twin_len, self.global_f1, self.global_f2)
            spectra[idx] = processing.frequency_whiten_spec(tr, self.global_sampling_rate, self.global_fwin_len, self.global_npts_cc)
        ### cross correlation
        msg_template = 'cc ev (%.2f %.2f)' % (evlo, evla)  + 'rcv(%.2f %.2f)x(%.2f %.2f) dist(%.2f) stack_cc(%.2f,%d) %s' 
        for idx1 in range(ntr):
            stlo1, stla1 = stlo[idx1], stla[idx1]
            az1 = az[idx1]
            for idx2 in range(ntr):
                stlo2, stla2 = stlo[idx2], stla[idx2]
                ### select distance
                distance = geomath.haversine(stlo1, stla1, stlo2, stla2) # return degree
                if distance > self.global_inter_rcv_distance_deg_max or distance < self.global_inter_rcv_distance_deg_min:
                    mpi_print_log(msg_template % (stlo1, stla1, stlo2, stla2, distance, np.nan, np.nan, 'N'), 2, self.local_log_fp, False )
                    continue
                ### select max az difference
                az2 = az[idx2]
                if (np.abs(az1-az2) <= az_diff_max_deg) == az_reverse:
                    continue
                ### select max distance from the event to the great circle plane
                junk = geomath.point_distance_to_great_circle_plane(evlo, evla, stlo1, stla1, stlo2, stla2) # return degree
                if (junk <= dis_to_greate_circle_plane_deg) == gc_reverse:
                    continue
                ### cc and stack
                tmp = spectra[idx1] * np.conj(spectra[idx2])
                if True not in (np.isnan(tmp) ):
                    idx_cc = int( (distance - self.global_inter_rcv_distance_deg_min)/self.global_inter_rcv_distance_step_deg)
                    mpi_print_log(msg_template % (stlo1, stla1, stlo2, stla2, distance, 
                            self.global_inter_rcv_distance_deg_lst[idx_cc], idx_cc, 'Y'), 2, self.local_log_fp, False )
                    self.local_stacked_cc_spectra[idx_cc, :] += tmp
                    self.local_stacked_cc_count[idx_cc] += 1
        app.h5.close()
        del spectra
        del time_series
        del app
    def collect_stack(self):
        if self.rank == 0:
            self.rank0_stacked_cc_spectra = np.zeros( self.local_stacked_cc_spectra.shape, dtype= np.complex64 )
            self.rank0_stacked_cc_count = np.zeros(self.local_stacked_cc_count.size, dtype=np.int32)
            self.rank0_stacked_cc_time  = np.zeros((self.global_ncc, self.global_npts_cc), dtype='float32' )
        ### stack cc
        mpi_print_log('>>> Reduce to RANK0 for stacking...', 0, self.local_log_fp, True)
        self.comm.Reduce([self.local_stacked_cc_spectra, mpi4py.MPI.C_FLOAT_COMPLEX], 
                    [self.rank0_stacked_cc_spectra, mpi4py.MPI.C_FLOAT_COMPLEX], mpi4py.MPI.SUM, root= 0)
        self.comm.Reduce([self.local_stacked_cc_count, mpi4py.MPI.INT32_T], 
                    [self.rank0_stacked_cc_count, mpi4py.MPI.INT32_T], mpi4py.MPI.SUM, root= 0 )
        ### ifft and average
        if self.rank == 0:
            mpi_print_log('>>> ifft and average...', 0, self.local_log_fp, True)
            for idx, (row, count) in enumerate(zip(self.rank0_stacked_cc_spectra, self.rank0_stacked_cc_count) ):
                if count < 1:
                    continue
                self.rank0_stacked_cc_time[idx]  = pyfftw.interfaces.numpy_fft.irfft(row, self.global_npts_cc)
                self.rank0_stacked_cc_time[idx] *= (1.0/count)
            mpi_print_log('>>> Computation done', 0, self.local_log_fp, True)
            

            
if __name__ == "__main__":
    ###
    fnm_lst_alignedSac2Hdf5 = None
    log_prefnm = ''
    username = getpass.getuser()
    d1, d2, dstep = 0.0, 180.0, 1.0
    cut_marker, cut_t1, cut_t2 = 'o', 10800.0, 32400.0
    twin, f1, f2 = 128.0, 0.02, 0.0666666
    fwin = 0.02
    fnm_hdf5 = 'junk.h5'
    #
    az_switch = False
    az_diff_deg_max = 720.0
    az_reverse = False
    gc_switch = False
    gc_dist = 720.0
    gc_reverse = False
    ###
    options, remainder = getopt.getopt(sys.argv[1:], 'I:L:T:S:N:W:O:A:G:' )
    for opt, arg in options:
        if opt in ('-I'):
            fnm_lst_alignedSac2Hdf5 = [line.strip() for line in open(arg, 'r')]
            pass
        elif opt in ('-L'):
            log_prefnm = arg
        elif opt in ('-T'):
            cut_marker, cut_t1, cut_t2 = arg.split('/')
            cut_t1 = float(cut_t1)
            cut_t2 = float(cut_t2)
        elif opt in ('-S'):
            d1, d2, dstep = [float(it) for it in arg.split('/')]
        elif opt in ('-N'):
            twin, f1, f2 = [float(it) for it in arg.split('/')]
        elif opt in ('-W'):
            fwin = float(arg)
        elif opt in ('-O'):
            fnm_hdf5 = arg
        elif opt in ('-A'): ## example: -A10 or -A10r
            az_switch = True
            if arg[-1] == 'r':
                az_reverse = True
                arg = arg[:-1]
            az_diff_deg_max = float(arg)
        elif opt in ('-G'): ## exmaple -G10 or -G10r
            gc_switch = True
            if arg[-1] == 'r':
                gc_reverse = True
                arg = arg[:-1]
            gc_dist = float(arg)
        else:
            print('invalid options: %s' % (opt) )
    ###
    if az_switch == False and gc_switch == False: # to disable selectived stacking 
        az_diff_deg_max = None
        az_reverse = None
        gc_dist = None
        gc_reverse = None
    ###
    print('>>> cmd: ', ' '.join(sys.argv ) )
    app = cc_stack_rcv_pairs(fnm_lst_alignedSac2Hdf5, log_prefnm=log_prefnm, username= username )
    app.init([d1, d2], dstep, cut_marker, cut_t1, cut_t2, twin, f1, f2, fwin, az_diff_deg_max, az_reverse, gc_dist, gc_reverse)
    app.run()
    app.output2hdf5(fnm_hdf5)
