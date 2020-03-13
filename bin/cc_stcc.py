#!/usr/bin/env python3
"""
Executable files for STCC
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
import scipy.signal
import pickle
import obspy.taup as taup

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
    return tr[idx1:idx1+cut_size ]

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
    def __init__(self, fnm_lst_alignedSac2Hdf5, out_put_prenm, log_prefnm = '', username= ''):
        """
        """
        ###
        if len(fnm_lst_alignedSac2Hdf5) < 2:
            self.comm = mpi4py.MPI.COMM_SELF
            self.rank = self.comm.Get_rank()
            self.ncpu = self.comm.Get_size()
            self.local_log_fnm = log_prefnm
        else:
            self.comm = mpi4py.MPI.COMM_WORLD.Dup()
            self.rank = self.comm.Get_rank()
            self.ncpu = self.comm.Get_size()
            self.local_log_fnm = '%s_mpi_log_%03d.txt' % (log_prefnm, self.rank)
        ### global variables
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
    def init(self,  lst_cross_term= [ ('PKIKP'*4, 'PKIKP'*2), ('PKIKP'*5, 'PKIKP'*4) ], 
                    seismic_phase_time_window_sec = (-50, 50),
                    inter_rcv_distance_range_deg=(-1, 181) ):
        """
        Use `lst_cross_term` to tell the program the cross-terms.
            The program will jump over the cross-terms that do not exist.
            The program will choose the cross-terms with the smallest slowness if there are multiple seismic phases.

        Use `seismic_phase_time_window_sec` to cut out seismic waves.
            e.g., (-50, 50) means a time window from 50 sec before to 50 sec after the synthetic traveltime.

        Use `inter_rcv_distance_range_deg` to limit inter-receiver distance to get rid of some station pairs.
        """
        ###
        self.global_inter_rcv_distance_range_deg = inter_rcv_distance_range_deg
        ###
        self.global_lst_cross_term = lst_cross_term
        self.global_all_seismic_phases = set([it1 for (it1, it2) in lst_cross_term] )
        self.global_all_seismic_phases.update( set( [it1 for (it1, it2) in lst_cross_term] ) )
        self.global_all_seismic_phases = sorted(list(self.global_all_seismic_phases) )
        self.global_seismic_phase_time_window_sec = seismic_phase_time_window_sec         
        ###
        mpi_print_log('>>> Initialized', 0, self.local_log_fid, False)
        mpi_print_log('inter-rcv-dist(%f, %f)'% (self.global_inter_rcv_distance_range_deg[0], self.global_inter_rcv_distance_range_deg[1]), 1, self.local_log_fid, False)
        mpi_print_log('time-window (%f, %f)' % (self.global_seismic_phase_time_window_sec[0], self.global_seismic_phase_time_window_sec[1]) , 1, self.local_log_fid, False)
        mpi_print_log('cross-terms: %s' % (',  '.join(['.'.join(it) for it in self.global_lst_cross_term]) ), 1, self.local_log_fid, True )
    def run(self):
        mpi_print_log('>>> Running...', 0, self.local_log_fid, True)
        for fnm in self.local_fnm_lst_alignedSac2Hdf5:
            outfnm = '%s_%s.pkl' % (self.global_out_put_prenm, fnm.split('/')[-1].replace('.h5', '') )
            self.single_file_run(fnm, outfnm)
    def release(self):
        mpi_print_log('>>> Releasing resources ...', 0, self.local_log_fid, True )
        self.local_log_fid.close()
        del self.comm
        mpi_print_log('>>> Safe exit!', 0, self.local_log_fid, True )
    def single_file_run(self, h5_fnm, out_fnm):
        """
        Run for a single h5_fnm
        """
        mpi_print_log('%s ==> %s...' % (h5_fnm, out_fnm), 1, self.local_log_fid, True )
        ###
        d1, d2 = self.global_inter_rcv_distance_range_deg
        ###
        fid_h5 = h5py.File(h5_fnm, 'r')
        ### obtain station info
        evlo = fid_h5['raw_sac/hdr/evlo'][0]
        evla = fid_h5['raw_sac/hdr/evla'][0]
        evdp = fid_h5['raw_sac/hdr/evdp'][0]
        dt   = fid_h5['raw_sac/hdr/delta'][0] # dt is the same for all time series
        gcarc= fid_h5['raw_sac/hdr/gcarc'][:] 
        b    = fid_h5['raw_sac/hdr/b'   ][:] # b can be different
        stlo = fid_h5['raw_sac/hdr/stlo'][:]
        stla = fid_h5['raw_sac/hdr/stla'][:]
        az   = fid_h5['raw_sac/hdr/az'][:]
        stnm = [it.decode('utf8') for it in fid_h5.h5['raw_sac/hdr/kstnm' ] ]
        netwk= [it.decode('utf8') for it in fid_h5.h5['raw_sac/hdr/knetwk'] ]
        ### much more useful values
        full_stnm = [ '.'.join(it) for it in zip (netwk, stnm) ]
        nsac = len(evlo)
        ### construct station pairs
        mpi_print_log('Constructing station pairs...', 2, self.local_log_fid, True )
        vol_sta_pairs = [[None for j in range(nsac)] for i in range(nsac)]
        for isac1, (stlo1, stla1, az1, gcarc1, stnm1, netwk1) in enumerate( zip( [stlo, stla, az, gcarc, stnm, netwk] )  ):
            for isac2, (stlo2, stla2, az2, gcarc2, stnm2, netwk2) in enumerate(zip( [stlo, stla, az, gcarc, stnm, netwk] ) ):
                inter_dist = geomath.haversine(stlo1, stla1, stlo2, stla2) # return degree
                if inter_dist < d1 or inter_dist > d2: # skip
                    continue 
                az_difference = az1 - az2 # degree
                vol_sta_pairs[isac1][isac2] = {   'evlo': evlo, 'evla': evla, 'evdp': evdp,
                                        'stlo1': stlo1, 'stla1': stla1, 'netwk1': netwk1, 'stnm1': stnm1, 'az1': az1, 'gcarc1': gcarc1, 
                                        'stlo2': stlo2, 'stla2': stla2, 'netwk2': netwk2, 'stnm2': stnm2, 'az2': az2, 'gcarc2': gcarc2, 
                                        'inter_rcv_distance': inter_dist,  'az_difference': az_difference }
        ### obtain several short time-series
        mpi_print_log('Getting short time-series...', 2, self.local_log_fid, True )
        mat = fid_h5['raw_sac/data'][:]
        t1, t2 = self.global_seismic_phase_time_window_sec
        cut_size = int( (t2-t1)/dt)+1
        vol_short_time_series = [dict() for it in range(nsac)]
        for seismic_phase in self.global_all_seismic_phases:
            for isac, single_gcarc, single_b in enumerate( zip(gcarc, b) ):
                synthetic_travel_time, slowness = get_synthetic_travel_time_and_slowness(seismic_phase, evdp, single_gcarc) # always choose the one of the smallest slowness
                if synthetic_travel_time is not None: # some seismic phases do not exist given certain distance
                    vol_short_time_series[isac][seismic_phase] = {  'start': synthetic_travel_time+t1,
                                                                    'time-series': cut_short_timeseries(mat[isac,:], dt, single_b, synthetic_travel_time+t1, cut_size),
                                                                    'slowness': slowness}
                    
        ### cross-correlation
        mpi_print_log('Conducting STCC...', 2, self.local_log_fid, True )
        for isac1 in range(nsac):
            for isac2 in range(nsac):
                tmp1 = vol_short_time_series[isac1]
                tmp2 = vol_short_time_series[isac2]
                for phase1, phase2 in self.global_lst_cross_term:
                    if (phase1 not in tmp1) or (phase2 not in tmp2):
                        continue
                    if vol_sta_pairs[isac1][isac2] == None: # skip becuase being out of ranges of (inter-distance,  ,,,)
                        continue
                    vol_sta_pairs[isac1][isac2][(phase1, phase2)] = {
                                        '%s@rcv1' % phase1 : tmp1[phase1],
                                        '%s@rcv2' % phase2 : tmp2[phase2],
                                        'cc_start': tmp1[phase1]['start'] - tmp2[phase2]['start'], # should this be inverted?
                                        'stcc' : scipy.signal.correlate(tmp1[phase1]['time-series'], tmp2[phase2]['time-series'], 'full', 'auto') 
                                        # FFT mode may not be efficient here for short time-series. 
                                    }
                    ###
        ### save to file. save 1) cutted time-series, 2) stcc,
        mpi_print_log('Outputing ...', 2, self.local_log_fid, True )
        all_vol = dict()
        all_vol['full_stnm'] = full_stnm
        all_vol['short_timeseries'] = vol_short_time_series
        all_vol['stcc'] = vol_sta_pairs
        with open(out_fnm, 'wb') as fid_out:
            pickle.dump(all_vol, fid_out)
        ###
        fid_h5.close()
        del mat
        del gcarc
        del b    
        del stlo 
        del stla 
        del az   
        del stnm 
        del netwk
        ###
        mpi_print_log('Succeed', 2, self.local_log_fid, True )



if __name__ == "__main__":
    ###
    fnm_lst_alignedSac2Hdf5 = None
    log_prefnm = ''
    username = getpass.getuser()
    output_prenm = ''
    wave_pairs = None
    t1, t2 = -50, 50
    d1, d2 = -1, 30
    log_prenm = None
    HMSG = '%s -I fnm_lst_alignedSac2Hdf5.txt -O output_prenm -P wave_pairs.txt -W -50/50 -D 0/30  -L log_prenm  ' % (sys.argv[0] )
    if len(sys.argv) < 2:
        print(HMSG)
        sys.exit(0)
    options, remainder = getopt.getopt(sys.argv[1:], 'I:O:P:W:D:L:H' )
    for opt, arg in options:
        if opt in ('-I'):
            fnm_lst_alignedSac2Hdf5 = [line.strip() for line in open(arg, 'r')]
        elif opt in ('-O'):
            output_prenm = arg
        elif opt in ('-P'):
            wave_pairs = [line.strip().split() for line in open(arg, 'r') ]
        elif opt in ('-W'):
            t1, t2 = [float(it) for it in arg.split('/')]
        elif opt in ('-D'):
            d1, d2 = [float(it) for it in arg.split('/')]
        elif opt in ('-L'):
            log_prefnm = arg
        else:
            print('invalid options: %s' % (opt) )
            print(HMSG)
            sys.exit(0)
    ###
    print('>>> cmd: ', ' '.join(sys.argv ) )
    ###
    if (not fnm_lst_alignedSac2Hdf5) or (not wave_pairs) or (not log_prenm):
        print(HMSG)
        sys.exit(-1)
    app = cc_stcc(fnm_lst_alignedSac2Hdf5, output_prenm, log_prefnm, username)
    app.init(wave_pairs, (t1, t2), (d1, d2) )
    app.run()
    app.release()
    del app