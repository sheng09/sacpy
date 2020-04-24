#!/usr/bin/env python3
"""
Executable files for STCC
"""
from time import gmtime, strftime
import time
import sacpy.processing as processing
import sacpy.sac_hdf5 as sac_hdf5
import sacpy.sac as sac
import sacpy.geomath as geomath
import mpi4py.MPI
import h5py
import pyfftw
import numpy as np 
import sys
import getpass
import getopt
import scipy.signal
import pickle
import obspy.taup as taup
import getpass
import os, os.path

def mpi_print_log(msg, n_pre, file=sys.stdout, flush=False):
    print('%s%s' % ('\t'*n_pre, msg), file= file, flush= flush )

global_earth_model = taup.TauPyModel('ak135')
def get_synthetic_travel_time_and_slowness(seismic_phase, evdp, gcarc_deg):
    tmp = global_earth_model.get_travel_times(evdp*0.001, gcarc_deg, [seismic_phase] )
    if len(tmp) == 1:
        return tmp[0].time, tmp[0].ray_param
    elif len(tmp) < 1:
        return None, None
    else:
        slowness = [it.ray_param for it in tmp]
        idx = np.argmin(slowness)
        return tmp[idx].time, tmp[idx].ray_param

def cut_short_timeseries(tr, dt, b, cut_start, cut_size):
    idx1 = int(np.floor((cut_start-b)/dt) )
    idx2 = idx1+cut_size
    return (idx1, idx2), b+idx1*dt, tr[idx1:idx2 ]

class cc_stcc:
    """
    The input will be a list of `alignedSac2Hdf5` hdf5 files, either whitend or not.
    
    The program cut out many body waves from each time series, and calculate their cross-correlations.
    
    The output will be in `hdf5` format. Inside the file is a matrix, and each row represents
    a cross-correlation between two body waves at two receivers. Rows are distinguished according
    to 1) station pairs with increasing inter-receiver distances, and 2) cross-term with increasing 
    absolute traveltime.

    Here, We do not care cross-correlation across events. 
    Please note, we distinguish between events. All cross-terms or cross-correlations are conducted for
    recordings from the same event. Therefore, the input has many `hdf5` files, each of which corresponds 
    to a single event. The output has many `hdf5` files, each of which correspond to a single event. 

    We do not do any stacking here, instead we reserve the cross-term between two body waves.
    """
    def __init__(self, fnm_lst_alignedSac2Hdf5, out_put_prenm, log_prefnm = '' ):
        """
        """
        ###
        if len(fnm_lst_alignedSac2Hdf5) < 2:
            self.comm = mpi4py.MPI.COMM_SELF
            self.rank = self.comm.Get_rank()
            self.ncpu = self.comm.Get_size()
            self.local_log_fnm = '%s_serial_log.txt' % (log_prefnm)
        else:
            self.comm = mpi4py.MPI.COMM_WORLD.Dup()
            self.rank = self.comm.Get_rank()
            self.ncpu = self.comm.Get_size()
            self.local_log_fnm = '%s_mpi_log_%03d.txt' % (log_prefnm, self.rank)
        ### global variables
        self.global_username = getpass.getuser()
        self.global_fnm_lst_alignedSac2Hdf5 = sorted(list(fnm_lst_alignedSac2Hdf5) )
        self.global_out_put_prenm  = out_put_prenm
        ### local variables
        self.local_log_fid = open(self.local_log_fnm, 'w')
        chunk = (len(self.global_fnm_lst_alignedSac2Hdf5)//self.ncpu)+1
        self.local_mpi_i1 = chunk * self.rank
        self.local_mpi_i2 = self.local_mpi_i1 + chunk
        self.local_fnm_lst_alignedSac2Hdf5 = self.global_fnm_lst_alignedSac2Hdf5[self.local_mpi_i1: self.local_mpi_i2]
        ###
        mpi_print_log('>>> Allocated. rank: %d; nproc: %d; range: [%d, %d); total: %d' % (self.rank, self.ncpu, 
                    self.local_mpi_i1, self.local_mpi_i2, 
                    len(self.global_fnm_lst_alignedSac2Hdf5) ), 
                    0, self.local_log_fid, True )
        ###
    def init(self,  lst_cross_term= [ ('I4', 'I2', 'PKIKP'*4, 'PKIKP'*2), ], 
                    seismic_phase_time_window_sec = (-50, 50),
                    inter_rcv_distance_range_deg=(-1, 181),
                    az_diff_range_deg = (-10, 10),
                    bandpass_hz = None,
                    h5_group='raw_sac',
                    ftcc_time_window=[2300, 2600], 
                    flag_output_sac= False ):
        """
        Use `lst_cross_term` to tell the program the cross-terms.
            The program will jump over the cross-terms that do not exist.
            The program will choose the cross-terms with the smallest slowness if there are multiple seismic phases.

        Use `seismic_phase_time_window_sec` to cut out seismic waves.
            e.g., (-50, 50) means a time window from 50 sec before to 50 sec after the synthetic traveltime.

        Use `inter_rcv_distance_range_deg` to limit inter-receiver distance to get rid of some station pairs.
        """
        ###
        self.global_lst_cross_term = lst_cross_term
        # 'I4', 'I2', 'PKIKP'*4, 'PKIKP'*2
        self.global_all_seismic_phases_dict = dict()
        for ph1_user, ph2_user, ph1, ph2 in lst_cross_term:
            self.global_all_seismic_phases_dict[ph1] = ph1_user
            self.global_all_seismic_phases_dict[ph2] = ph2_user
        self.global_all_seismic_phases = sorted( list(self.global_all_seismic_phases_dict.keys() ) )
        ###
        self.global_seismic_phase_time_window_sec = seismic_phase_time_window_sec         
        ###
        self.global_inter_rcv_distance_range_deg = inter_rcv_distance_range_deg
        self.global_az_diff_range_deg = az_diff_range_deg
        self.global_bandpass_hz = bandpass_hz
        self.global_h5_group = h5_group
        ###
        self.global_ftcc_time_window = ftcc_time_window
        ###
        self.global_flag_output_sac = flag_output_sac
        ###
        mpi_print_log('>>> Initialized', 0, self.local_log_fid, False)
        mpi_print_log('inter-rcv-dist(%f, %f)'% (self.global_inter_rcv_distance_range_deg[0], self.global_inter_rcv_distance_range_deg[1]), 1, self.local_log_fid, False)
        mpi_print_log('time-window (%f, %f)' % (self.global_seismic_phase_time_window_sec[0], self.global_seismic_phase_time_window_sec[1]) , 1, self.local_log_fid, False)
        mpi_print_log('cross-terms: %s' % (',  '.join(['-'.join(it[:2]) for it in self.global_lst_cross_term]) ), 1, self.local_log_fid, True )
    def run(self):
        mpi_print_log('>>> Running... [%d] ' % (len(self.local_fnm_lst_alignedSac2Hdf5) ) , 0, self.local_log_fid, True)
        for idx, fnm in enumerate(self.local_fnm_lst_alignedSac2Hdf5):
            outfnm = '%s_%s.pkl' % (self.global_out_put_prenm, fnm.split('/')[-1].replace('.h5', '') )
            ###
            mpi_print_log('[%d] %s ==> %s' % (idx, fnm, outfnm) , 1, self.local_log_fid, True)
            ###
            self.single_file_run(fnm, outfnm)
    def release(self):
        mpi_print_log('>>> Releasing resources ...', 0, self.local_log_fid, True )
        mpi_print_log('>>> Stop mpi logging and safe exit!', 0, self.local_log_fid, True )
        self.local_log_fid.close()
        del self.comm
    def single_file_run(self, h5_fnm, out_pkl_fnm):
        """
        Run for a single h5_fnm, the output is performed by pickle.dump(...)
        """
        ### obtain sactrace
        st = sac_hdf5.get_sactrace_from_H5(h5_fnm, h5_grp_name= self.global_h5_group )
        if self.global_bandpass_hz != None:
            f1, f2 = self.global_bandpass_hz 
            for it in st:
                it.rmean()
                it.detrend()
                it.taper(ratio= 0.01) # 
                it.bandpass(f1, f2, 2, 2) # order=2, npass=2
        ### cut seismic waves from sactrace
        vol_seismic_waves = self.__cut_seismic_waves__(st)
        ### compute stcc
        vol_correlations = self.__run_correlations__(st, vol_seismic_waves) # check `__run_stcc__(...)`for the data structure
        ### output
        self.__output_pkl__(out_pkl_fnm, st, vol_seismic_waves, vol_correlations)
        ###
    def __cut_seismic_waves__(self, st):
        """
        Internal functions
        ### structure of `vol_seismic_waves`. It is a list of dict()
            #  +--------------------------+---------+---------+--------------+
            #  | #idx=0#                  | #idx=1# | #idx=2# | ... ... ...  |
            #  | wave_name1 --> SACTRACE1 | ... ... | ... ... |              |
            #  | wave_name2 --> SACTRACE2 | ... ... | ... ... |              |
            #  | ... ... ...              | ... ... | ... ... |              |
            #  |                          | ... ... | ... ... |              |
            #  +--------------------------+---------+---------+--------------+
        """
        dt1, dt2 = self.global_seismic_phase_time_window_sec
        gcarc= [it['gcarc'] for it in st] 
        evdp = [it['evdp']*0.001 for it in st] 
        nsac = len(st)
        ###
        mpi_print_log('cutting seismic waves... nsac(%d)' % (nsac) , 2, self.local_log_fid, True )
        ###
        vol_seismic_waves = [dict() for it in range(nsac) ]
        for isac, tr in enumerate(st):
            for wave_name in self.global_all_seismic_phases:
                traveltime, slowness = get_synthetic_travel_time_and_slowness(wave_name, evdp[isac], gcarc[isac] )
                if traveltime == None or slowness == None: # the wave does not exist for that evdp and gcarc
                    continue
                ###
                msg = '[%d/%d] %s %f %f' % (isac+1, nsac, self.global_all_seismic_phases_dict[wave_name], traveltime, slowness)
                mpi_print_log(msg , 3, self.local_log_fid, True )
                ###
                seismic_wave = sac.truncate_sac(tr, 'o', traveltime+dt1, traveltime+dt2)
                seismic_wave['slowness'] = slowness
                seismic_wave['traveltime']  = traveltime 
                seismic_wave['phase'] = self.global_all_seismic_phases_dict[wave_name]  
                seismic_wave['t8' ]  = traveltime  # use t8 to allow sachdr info
                seismic_wave['kt8'] = self.global_all_seismic_phases_dict[wave_name]  
                vol_seismic_waves[isac][wave_name] = seismic_wave
        return vol_seismic_waves
    def __run_correlations__(self, st, vol_seismic_waves):
        """
        Internal functions
        ### structure of `vol_correlations`
               *******************************************************************
               * #LEVEL 0#                                                       *
               * (0, 0) --> +-------------------------------------------------+  *
               *            | #LEVEL 1#                                       |  *
               *            | 'ftcc' --> SACTRACE(ST)                         |  *
               *            |                                                 |  *
               *            | 'stcc' --> +---------------------------------+  |  *
               *            |            | #LEVEL 2#                       |  |  *
               *            |            | 'PKIKPPKIKP-PKIKP' --> (ST1,ST2)|  |  *
               *            |            | 'PKIKPP****-P****' --> (ST1,ST2)|  |  *
               *            |            | 'PKIKPP****-P****' --> (ST1,ST2)|  |  *
               *            |            | ... ... ...                     |  |  *
               *            |            | ... ... ...                     |  |  *
               *            |            |                                 |  |  *
               *            |            +---------------------------------+  |  *
               *            +-------------------------------------------------+  *
               *                                                                 *
               * (0, 1) --> +-------------------------------------------------+  *
               *            | #LEVEL 1#                                       |  *
               *            | ... ... ...                                     |  *
               *            |                                                 |  *
               *            +-------------------------------------------------+  *
               *  ... ...                                                        *
               *  ... ...                                                        *
               *                                                                 *
               *******************************************************************
            
        """
        ###
        stlo = [it['stlo'] for it in st]
        stla = [it['stla'] for it in st]
        az   = [it['az'] for it in st]
        gcarc= [it['gcarc'] for it in st]
        kstnm   = [it['kstnm'] for it in st]
        knetwk  = [it['knetwk'] for it in st]
        full_stnm = ['%s.%s' % (it1, it2) for it1, it2 in zip(knetwk, kstnm) ]
        ###
        d1, d2 = self.global_inter_rcv_distance_range_deg
        a1, a2 = self.global_az_diff_range_deg
        cc_t1, cc_t2 = self.global_ftcc_time_window
        nsac = len(st)
        ###
        mpi_print_log('generating correlations... (%dx%d)' % (nsac, nsac), 2, self.local_log_fid, True )
        ###
        ###
        vol_correlations = dict()
        for isac1 in range(nsac):
            for isac2 in range(nsac):
                inter_dist = geomath.haversine(stlo[isac1], stla[isac1], stlo[isac2], stla[isac2]) # return degree
                if inter_dist < d1 or inter_dist > d2: # skip for meaningless station pairs
                    continue 
                az_diff = az[isac1]-az[isac2] # degree
                if az_diff < a1 or az_diff > a2:
                    continue
                gcarc_diff = gcarc[isac1]-gcarc[isac2] # degree
                ###
                vol_correlations[(isac1, isac2)] = dict()
                ### tags
                tag = 'auto' if isac1 == isac2 else 'cros'
                ###
                msg = '[%dx%d] %s, %sx%s, inter-dist(%.2f), az_diff(%.2f) gcarc_diff(%.2f)' % (isac1+1, isac2+1, tag, full_stnm[isac1], full_stnm[isac2], 
                                                                                        inter_dist, az_diff, gcarc_diff)
                mpi_print_log(msg, 3, self.local_log_fid, True )
                ### ftcc
                mpi_print_log('ftcc...', 4, self.local_log_fid, True )
                ftcc = sac.correlation_sac(st[isac1], st[isac2])
                ftcc.truncate('0', cc_t1, cc_t2)
                ftcc['tag']        = tag
                ftcc['type']       = 'ftcc'
                ftcc['stnm1']      = full_stnm[isac1]
                ftcc['stnm2']      = full_stnm[isac2]
                ftcc['inter-dist'] = inter_dist
                ftcc['az-diff']    = az_diff
                ftcc['gcarc-diff'] = gcarc_diff
                ftcc['gcarc1']     = gcarc[isac1]
                ftcc['gcarc2']     = gcarc[isac2]
                vol_correlations[(isac1, isac2)]['ftcc'] = ftcc
                ### stcc
                vol_correlations[(isac1, isac2)]['stcc'] = dict()
                for junk1, junk2, wave1, wave2 in self.global_lst_cross_term:
                    if (wave1 not in vol_seismic_waves[isac1]) or (wave2 not in vol_seismic_waves[isac2]):
                        continue
                    mpi_print_log('stcc... %sx%s' % (self.global_all_seismic_phases_dict[wave1], self.global_all_seismic_phases_dict[wave2]) , 4, self.local_log_fid, True )
                    seismic_wave1, seismic_wave2 = vol_seismic_waves[isac1][wave1], vol_seismic_waves[isac2][wave2]
                    reference_time = seismic_wave1['traveltime'] - seismic_wave2['traveltime']
                    stcc1 = sac.correlation_sac(seismic_wave1, seismic_wave2)
                    stcc2 = sac.correlation_sac(st[isac1], seismic_wave2)
                    stcc1.truncate('0', cc_t1, cc_t2)
                    stcc2.truncate('0', cc_t1, cc_t2)
                    stcc1['flag'] = 'zero padding'
                    stcc2['flag'] = 'values padding'
                    stcc1['kevnm'] = 'zero padding'
                    stcc2['kevnm'] = 'values padding'
                    #
                    for stcc in [stcc1, stcc2]:
                        stcc['reference-time'] = reference_time
                        stcc['t9'] = reference_time
                        stcc['tag']        = tag
                        stcc['type']       = 'stcc'
                        stcc['stnm1']      = full_stnm[isac1]
                        stcc['stnm2']      = full_stnm[isac2]
                        stcc['inter-dist'] = inter_dist
                        stcc['az-diff']    = az_diff
                        stcc['gcarc-diff'] = gcarc_diff
                        stcc['gcarc1']     = gcarc[isac1]
                        stcc['gcarc2']     = gcarc[isac2]
                        stcc['cross_term'] =  '%s-%s' % (self.global_all_seismic_phases_dict[wave1], self.global_all_seismic_phases_dict[wave2])
                    cross_term         = stcc1['cross_term']
                    vol_correlations[(isac1, isac2)]['stcc'][cross_term] = (stcc1, stcc2)
        return vol_correlations
    def __output_pkl__(self, out_pkl_fnm, st, vol_seismic_waves, vol_correlations):
        """
        Internal functions
        """
        all_vol = dict()
        # sta info
        kstnm   = [it['kstnm'] for it in st]
        knetwk  = [it['knetwk'] for it in st]
        all_vol['full_stnm'] = ['%s.%s' % (it1, it2) for it1, it2 in zip( knetwk, kstnm) ]
        for key in ['stlo', 'stla', 'evlo', 'evla', 'evdp', 'az', 'gcarc', 'kstnm', 'knetwk']:
            all_vol[key] = [it[key] for it in st]
        # time series
        all_vol['seismic-waves'] = vol_seismic_waves
        all_vol['correlations']  = vol_correlations
        #
        msg = 'outputing... %s' % (out_pkl_fnm)
        mpi_print_log(msg, 2, self.local_log_fid, True )
        #
        with open(out_pkl_fnm, 'wb') as fid_out:
            pickle.dump(all_vol, fid_out)
        ### out to sac
        if self.global_flag_output_sac:
            dir_nm = out_pkl_fnm.replace('.pkl', '_sacvol')
            if not os.path.exists(dir_nm):
                os.mkdir(dir_nm)
            ### seismic waves
            mpi_print_log('outputing seismic waves... to sac files', 2, self.local_log_fid, True )
            for it in vol_seismic_waves:
                for wave_name, st in it.items():
                    sacfnm = '%s/seismic-wave_%s.%s_%s.sac' % (dir_nm, st['knetwk'], st['kstnm'], self.global_all_seismic_phases_dict[wave_name])
                    st.write(sacfnm)
            ### correlations
            mpi_print_log('outputing correlations... to sac files', 2, self.local_log_fid, True )
            for (isac1, isac2), content in vol_correlations.items():
                ftcc = content['ftcc']
                ftcc_fnm = '%s/%sx%s_ftcc.sac' % (dir_nm, all_vol['full_stnm'][isac1], all_vol['full_stnm'][isac2] )
                ftcc.write(ftcc_fnm)
                for nm, (stcc1, stcc2) in content['stcc'].items():
                    st1_fnm = '%s/%sx%s_stcc1_%s.sac' % (dir_nm, all_vol['full_stnm'][isac1], all_vol['full_stnm'][isac2], nm )
                    stcc1.write(st1_fnm)
                    st2_fnm = '%s/%sx%s_stcc2_%s.sac' % (dir_nm, all_vol['full_stnm'][isac1], all_vol['full_stnm'][isac2], nm )
                    stcc2.write(st2_fnm)
            ###
        

    def single_file_run_old(self, h5_fnm, out_fnm):
        """
        Run for a single h5_fnm
        """
        mpi_print_log('%s ==> %s...' % (h5_fnm, out_fnm), 1, self.local_log_fid, True )
        ###
        d1, d2 = self.global_inter_rcv_distance_range_deg
        ###
        fid_h5 = h5py.File(h5_fnm, 'r')
        grp = fid_h5[self.global_h5_group ]
        ### obtain event and station info. All recording are for the same event in a single H5 file.
        evlo = grp['hdr/evlo'][0]
        evla = grp['hdr/evla'][0]
        evdp = grp['hdr/evdp'][0]
        dt   = grp['hdr/delta'][0] # dt is the same for all time series
        gcarc= grp['hdr/gcarc'][:] 
        b    = grp['hdr/b'   ][:] # b can be different
        stlo = grp['hdr/stlo'][:]
        stla = grp['hdr/stla'][:]
        az   = grp['hdr/az'][:]
        stnm = [it.decode('utf8').strip() for it in grp['hdr/kstnm' ][:] ]
        netwk= [it.decode('utf8').strip() for it in grp['hdr/knetwk'][:] ]
        mat = grp['data'][:]
        fid_h5.close()
        ### much more useful values for network and station names
        full_stnm = [ '.'.join(it) for it in zip (netwk, stnm) ]
        nsac = len(stlo)
        ### construct station pairs
        mpi_print_log('Constructing station pairs...', 2, self.local_log_fid, True )
        mat_sta_pairs = [[None for j in range(nsac)] for i in range(nsac)] # meaningless station pairs will be None
        for isac1, (stlo1, stla1, az1, gcarc1, full_nm1) in enumerate( zip( stlo, stla, az, gcarc, full_stnm )  ):
            for isac2, (stlo2, stla2, az2, gcarc2, full_nm2) in enumerate(zip( stlo, stla, az, gcarc, full_stnm ) ):
                inter_dist = geomath.haversine(stlo1, stla1, stlo2, stla2) # return degree
                if inter_dist < d1 or inter_dist > d2: # skip for meaningless station pairs
                    continue 
                az_difference = az1 - az2 # degree
                gcarc_difference = gcarc1 - gcarc2 # degree
                #mat_sta_pairs[isac1][isac2] = {   'evlo': evlo, 'evla': evla, 'evdp': evdp,
                #                        'stlo1': stlo1, 'stla1': stla1, 'netwk1': netwk1, 'stnm1': stnm1, 'az1': az1, 'gcarc1': gcarc1, 
                #                        'stlo2': stlo2, 'stla2': stla2, 'netwk2': netwk2, 'stnm2': stnm2, 'az2': az2, 'gcarc2': gcarc2, 
                #                        'inter_rcv_distance': inter_dist,  'az_difference': az_difference }
                # most of informations were redundant
                mat_sta_pairs[isac1][isac2] = {'inter_rcv_distance': inter_dist,  
                                                'az_difference': az_difference, 
                                                'gcarc_difference': gcarc_difference }
                flag = 'auto' if isac1==isac2 else 'cros'
                msg = '(%s)%17s x %17s; inter-dist: %.2f; daz %.2f;' % (flag, full_nm1, full_nm2, inter_dist, az_difference)
                mpi_print_log(msg, 3, self.local_log_fid, False)
        ### data and bandpass filter
        if self.global_bandpass_hz != None: # 
            msg = 'Filtering... bp(%f,%f)HZ, taper(0.05x2)' % ( self.global_bandpass_hz[0], 
                                                                        self.global_bandpass_hz[1] )
            mpi_print_log(msg , 2, self.local_log_fid, True )
            sampling_rate = 1.0/dt
            tmp_win = scipy.signal.tukey(mat[0,:].size, 0.05) # hardcoded 0.05 at each end of the time-series
            for isac in range(nsac):
                mat[isac,:] = scipy.signal.detrend( mat[isac,:] ) * tmp_win
                mat[isac,:] = processing.filter(mat[isac,:], sampling_rate, 'bandpass', self.global_bandpass_hz, 2, 2)
        ### obtain several short time-series
        mpi_print_log('Getting short time-series...', 2, self.local_log_fid, True )
        ###
        msg = 'phase list: %s' % ( ', '.join(self.global_all_seismic_phases) )
        mpi_print_log(msg, 3, self.local_log_fid, True)
        ###
        t1, t2 = self.global_seismic_phase_time_window_sec
        cut_size = int(np.round( (t2-t1)/dt) ) + 1
        vol_short_time_series = [dict() for it in range(nsac)] # an archive to store short time series for all recordings
        for isac, (single_gcarc, single_b) in enumerate( zip(gcarc, b) ):
            for idx_phase, seismic_phase in enumerate(self.global_all_seismic_phases):
                synthetic_travel_time, slowness = get_synthetic_travel_time_and_slowness(seismic_phase, evdp, single_gcarc) # always choose the one of the smallest slowness
                ###
                msg = 'idx_sac (%03d); idx_phase(%02d); gcarc(%6.2f) synthetic_time(%8.2f) rp(%f) %s' % (isac, idx_phase, single_gcarc, synthetic_travel_time, slowness, seismic_phase)
                mpi_print_log(msg, 3, self.local_log_fid, False)
                ###
                if synthetic_travel_time is not None: # some seismic phases do not exist given certain distance
                    (idx1, idx2), start, xs= cut_short_timeseries(mat[isac,:], dt, single_b, synthetic_travel_time+t1, cut_size)
                    vol_short_time_series[isac][seismic_phase] = {  'idx1': idx1, 'idx2': idx2,
                                                                    'start': start,
                                                                    'time-series': xs, 
                                                                    'slowness': slowness,
                                                                    'traveltime': synthetic_travel_time }
        ### cross-correlation
        mpi_print_log('Conducting STCC...', 2, self.local_log_fid, True )
        ftcc_cut_t1, ftcc_cut_t2 = self.global_ftcc_time_window
        ftcc_cutsize = int(np.round( (ftcc_cut_t2-ftcc_cut_t1)/dt) ) + 1
        for isac1 in range(nsac):
            for isac2 in range(nsac):
                if mat_sta_pairs[isac1][isac2] == None: # skip becuase being out of ranges of (inter-distance,  ,,,)
                    continue
                # ftcc
                ftcc = scipy.signal.correlate( mat[isac1,:], mat[isac2,:], 'full', 'fft' )
                ftcc_start = (1-mat[isac1,:].size)*dt
                # cut ftcc to reduce storage
                (junk, junk), ftcc_start, ftcc = cut_short_timeseries(ftcc, dt, ftcc_start, ftcc_cut_t1, ftcc_cutsize)
                mat_sta_pairs[isac1][isac2]['ftcc'] = ftcc
                mat_sta_pairs[isac1][isac2]['ftcc_start'] = ftcc_start
                # stcc
                mat_sta_pairs[isac1][isac2]['stcc'] = dict()
                tmp1 = vol_short_time_series[isac1]
                tmp2 = vol_short_time_series[isac2]
                for phase1, phase2 in self.global_lst_cross_term:
                    if (phase1 not in tmp1) or (phase2 not in tmp2): # skip becuase the phase does not exist
                        continue
                    ### two types of STCC
                    # 1. cc between two short time-series
                    # 2. cc between a short time-series and a full time-series
                    # The first can result in decreasing amplitude when approaching two ends of the cross-correlation functions.
                    #
                    ### e.g. phase1 and phase2 can be I6 and I4 respectively. That means I6 in tmp1, and I4 in tmp2
                    seismic_wave1 = tmp1[phase1]
                    seismic_wave2 = tmp2[phase2]
                    #
                    stcc_start = seismic_wave1['start']-seismic_wave2['start']-seismic_wave1['time-series'].size*dt
                    #
                    stcc1 = scipy.signal.correlate(seismic_wave1['time-series'], seismic_wave2['time-series'], 'full', 'auto')
                    #
                    stcc2  = scipy.signal.correlate(mat[isac1,:], seismic_wave2['time-series'], 'full', 'auto')
                    idx1, idx2 = seismic_wave1['idx1'], seismic_wave1['idx2']
                    stcc2 = stcc2[idx1: idx2+cut_size-1]
                    #
                    mat_sta_pairs[isac1][isac2]['stcc'][(phase1, phase2)] = {
                                        'seismic_wave1': seismic_wave1,
                                        'seismic_wave2': seismic_wave2,
                                        'stcc_start': stcc_start, 
                                        'stcc1' : stcc1,
                                        'stcc2': stcc2   }
                    flag = 'auto' if isac1==isac2 else 'cros'
                    msg = '(%s) (%d)%s x (%d)%s' % (flag, isac1, phase1, isac2, phase2) 
                    mpi_print_log(msg, 3, self.local_log_fid, False)
                    ###
        ### save to file. save 1) cutted time-series, 2) stcc,
        mpi_print_log('Outputing ... (%s)' % (out_fnm), 2, self.local_log_fid, True )
        all_vol = dict()
        # sta info
        all_vol['full_stnm'] = full_stnm
        all_vol['stlo'] = stlo
        all_vol['stla'] = stla
        all_vol['az'] = az
        all_vol['gcarc'] = gcarc
        # ev info
        all_vol['evlo'] = evlo
        all_vol['evla'] = evla
        all_vol['dt']   = dt
        # time series
        all_vol['short_timeseries'] = vol_short_time_series
        all_vol['stcc_vol'] = mat_sta_pairs
        with open(out_fnm, 'wb') as fid_out:
            pickle.dump(all_vol, fid_out)
        ###
        del ftcc
        del stcc1
        del stcc2
        del mat
        del gcarc
        del b    
        del stlo 
        del stla 
        del az   
        del stnm 
        del netwk
        ###
        mpi_print_log('Done', 2, self.local_log_fid, True )



if __name__ == "__main__":
    ###
    fnm_lst_alignedSac2Hdf5 = None
    log_prefnm = ''
    output_prenm = ''
    wave_pairs = None
    t1, t2 = -50, 50
    d1, d2 = -1, 30
    log_prenm = None
    bandpass_hz = None
    h5_grp = 'raw_sac'
    ftcc_time_window= [2300, 2600]
    flag_output_sac = False
    HMSG = '%s -I fnm_lst_alignedSac2Hdf5.txt  -O output_prenm -P wave_pairs.txt -C 2300/2600 -W -50/50 -D 0/30  -L log_prenm -G raw_sac [-B 0.02/0.0666] [-S]  ' % (sys.argv[0] )
    HMSG2 = """
    -I fnm_lst_alignedSac2Hdf5.txt : a text file that list all HDF5 files.
    -O output_prenm : the pre-name for output files.
    -P wave_pairs.txt : a text file that each line is : 
           `user-defined-phase-name1  user-defined-phase-name2  phase-name1  phase-name2`

    -C ct1/ct2 : the time window [ct1, ct2] to cut cross-correlation functions.
    -W t1/t2 : the time window [t1, t2] with respect to synthetic traveltime of a seismic phases will used to cut out the seismic wave.
    -D d1/d2 : inter-receiver distance range to jump over some receiver pairs.
    -L log_prenm : pre-name for log files in mpi-parallel mode, or the log filename in serial mode.
    -G raw_sac : the group name in the HDF5 files.
    -B f1/f2   : bandpass filter cutoff frequency in Hz.
    -S : use `-S` to turn on outputing sac files.
    """
    HMSG = HMSG + HMSG2
    if len(sys.argv) < 2:
        print(HMSG)
        sys.exit(0)
    options, remainder = getopt.getopt(sys.argv[1:], 'I:B:O:P:C:W:D:L:G:SH' )
    for opt, arg in options:
        if opt in ('-I'):
            fnm_lst_alignedSac2Hdf5 = [line.strip() for line in open(arg, 'r')]
        elif opt in ('-B'):
            bandpass_hz = [float(it) for it in arg.split('/')]
        elif opt in ('-O'):
            output_prenm = arg
        elif opt in ('-P'):
            wave_pairs = [line.strip().split() for line in open(arg, 'r') ]
        elif opt in ('-C'):
            ftcc_time_window = [float(it) for it in arg.split('/')]
        elif opt in ('-W'):
            t1, t2 = [float(it) for it in arg.split('/')]
        elif opt in ('-D'):
            d1, d2 = [float(it) for it in arg.split('/')]
        elif opt in ('-L'):
            log_prefnm = arg
        elif opt in ('-G'):
            h5_grp = arg
        elif opt in ('-S'):
            flag_output_sac = True
        else:
            print('invalid options: %s' % (opt) )
            print(HMSG)
            sys.exit(0)
    ###
    print('>>> cmd: ', ' '.join(sys.argv ) )
    ###
    if (not fnm_lst_alignedSac2Hdf5) or (not wave_pairs) or (not log_prefnm):
        #print(fnm_lst_alignedSac2Hdf5, wave_pairs, log_prefnm)
        print(HMSG)
        sys.exit(-1)
    app = cc_stcc(fnm_lst_alignedSac2Hdf5, output_prenm, log_prefnm)
    app.init(   lst_cross_term= wave_pairs, 
                seismic_phase_time_window_sec= (t1, t2), 
                inter_rcv_distance_range_deg= (d1, d2), 
                bandpass_hz= bandpass_hz,
                h5_group=h5_grp,
                ftcc_time_window= ftcc_time_window, 
                flag_output_sac= flag_output_sac )
    app.run()
    app.release()
    del app