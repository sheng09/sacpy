#!/usr/bin/env python3
"""
Executable files for transferring .pkl to .sac files.
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

class cc_stcc_pkl2sac:
    """
    This class is for transferring many `.pkl` files into many `.sac` files.
    The `.pkl` is the output of `cc_stcc.py`.
    The `.sac` files include 1) ftcc, 2) seismic-wave cross-term (stcc), 
    3) stcc plain stacking, 4) stcc normalized stacking.
    """
    def __init__(self, pkl_fnmlst, output_prenm, log_prefnm = ''):
        ### MPI settings
        if len(pkl_fnmlst) < 2:
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
        self.global_pkl_fnmlst = sorted( pkl_fnmlst )
        self.global_output_prenm = output_prenm
        self.global_log_prefnm = log_prefnm
        self.global_username = getpass.getuser()
        ### local variables
        chunk = (len(self.global_pkl_fnmlst)//self.ncpu)+1
        self.local_mpi_i1 = chunk * self.rank
        self.local_mpi_i2 = self.local_mpi_i1 + chunk
        self.local_pkl_fnmlst = self.global_pkl_fnmlst[self.local_mpi_i1: self.local_mpi_i2]
        self.local_log_fid = open(self.local_log_fnm, 'w')
    def release(self):
        mpi_print_log('>>> Releasing resources ...', 0, self.local_log_fid, True )
        mpi_print_log('>>> Stop mpi logging and safe exit!', 0, self.local_log_fid, True )
        self.local_log_fid.close()
        del self.comm
    def init(self, flag_output_ftcc=True, flag_output_cross_term=True, flag_output_seismic_waves=True,
                exclude_rcv_lst=None, include_rcv_lst=None, 
                exclude_cross_term_lst=None, include_cross_term_lst=None,
                exclude_seismic_wave_lst=None, include_seismic_wave_lst=None,
            ):
        """
        ##################  
        flag_output_ftcc            : True of False to output ftcc.
        flag_output_cross_term      : True of False to output cross_term.
        flag_output_seismic_waves   : True of False to output seismic_waves.

        exclude_rcv_lst:    a list of receivers to be excluded. Set `None` to disable this variable.
                            The receiver name should be `KNETWK.KSTNM` (e.g., II.TAU)
        include_rcv_lst:    a list of receivers to be included. Receviers out of this list would be
                            excluded. Set `None` to disable this variable.

        exclude_cross_term_lst: a list of cross-terms to be excluded. Set `None` to disable this variable.
                                (e.g., ['I06-I04', 'I22-_20'])
        include_cross_term_lst: a list of cross-terms to be included. Cross-terms out of this list would be
                                excluded.Set `None` to disable this variable.

        exclude_seismic_wave_lst: a list of seismic waves to be excluded. Set `None` to disable this variable.
                                (e.g., ['I06', 'I04'])
        include_seismic_wave_lst: a list of seismic waves to be included. Seismic waves out of this list would be
                                excluded.Set `None` to disable this variable.
        
        ##################
        In default, the program would output all receivers, cross-terms, and seismic-waves.
        ##################
        Example:
            1. to exclude some of receivers: use `exclude_rcv_lst = ['R1', 'R2'] `
            2. to output only for a few receivers: use `include_rcv_lst = ['R1', 'R2'] `

            3. to exclude some of cross-terms: use `exclude_rcv_lst = ['C1', 'C2'] `
            4. to output only for a few cross-terms: use `include_rcv_lst = ['C1', 'C2'] `
            5. do not output any cross-terms: use `flag_output_cross_term = False`

            6. to excluse some seismic waves: use `exclude_rcv_lst = ['W1', 'W2'] `
            7. to output only for a few seismic waves: use `include_seismic_wave_lst = ['W1', 'W2'] `
            8. do not output any seismic wavess: `flag_output_seismic_waves = False` 
        """
        self.global_flag_output_ftcc            = flag_output_ftcc            
        self.global_flag_output_cross_term      = flag_output_cross_term      
        self.global_flag_output_seismic_waves   = flag_output_seismic_waves   
        self.global_exclude_rcv_lst             = exclude_rcv_lst
        self.global_include_rcv_lst             = include_rcv_lst
        self.global_exclude_cross_term_lst      = exclude_cross_term_lst
        self.global_include_cross_term_lst      = include_cross_term_lst
        self.global_exclude_seismic_wave_lst    = exclude_seismic_wave_lst
        self.global_include_seismic_wave_lst    = include_seismic_wave_lst
        ####
        mpi_print_log('>>> Initialized', 0, self.local_log_fid, False)
        mpi_print_log('Output FTCC: %s' % (self.global_flag_output_ftcc            ), 1, self.local_log_fid, False )
        mpi_print_log('Output Cross-terms: %s' % (self.global_flag_output_cross_term      ), 1, self.local_log_fid, False )
        mpi_print_log('Output Seimic waves: %s' % (self.global_flag_output_seismic_waves   ), 1, self.local_log_fid, False )
        ###
        if self.global_exclude_rcv_lst          != None:
            mpi_print_log('exclude rcv         :', 1, self.local_log_fid, False)
            mpi_print_log(', '.join(self.global_exclude_rcv_lst         ), 2, self.local_log_fid, False )
        if self.global_include_rcv_lst          != None:
            mpi_print_log('include rcv         :', 1, self.local_log_fid, False)
            mpi_print_log(', '.join(self.global_include_rcv_lst         ), 2, self.local_log_fid, False )
        if self.global_exclude_cross_term_lst   != None:
            mpi_print_log('exclude cross-terms  :', 1, self.local_log_fid, False)
            mpi_print_log(', '.join(self.global_exclude_cross_term_lst  ), 2, self.local_log_fid, False )
        if self.global_include_cross_term_lst   != None:
            mpi_print_log('include cross-terms  :', 1, self.local_log_fid, False)
            mpi_print_log(', '.join(self.global_include_cross_term_lst  ), 2, self.local_log_fid, False )
        if self.global_exclude_seismic_wave_lst != None:
            mpi_print_log('exclude seismic waves:', 1, self.local_log_fid, False)
            mpi_print_log(', '.join(self.global_exclude_seismic_wave_lst), 2, self.local_log_fid, False )
        if self.global_include_seismic_wave_lst != None:
            mpi_print_log('include seismic waves:', 1, self.local_log_fid, False)
            mpi_print_log(', '.join(self.global_include_seismic_wave_lst), 2, self.local_log_fid, False )
        ###
    def run(self):
        for idx, pkl_fnm in enumerate(self.local_pkl_fnmlst):
            #print('run')
            self.run_single_pkl(pkl_fnm, idx, len(self.local_pkl_fnmlst) )
    def run_single_pkl(self, pkl_fnm, idx, total_number):
        """
        """
        ###
        dir_nm = self.global_output_prenm + '/' + pkl_fnm.split('/')[-1].replace('.pkl', '_sacvol')
        if not os.path.exists(dir_nm):
            os.mkdir(dir_nm)
        ###
        msg = '[%d/%d] %s ==> %s' % (idx+1, total_number, pkl_fnm, dir_nm)
        mpi_print_log(msg , 1, self.local_log_fid, True )
        ###
        with open(pkl_fnm, 'rb') as fid:
            all_vol = pickle.load(fid)
            full_stnm = all_vol['full_stnm'] 
            vol_seismic_waves = all_vol['seismic-waves']
            vol_correlations  = all_vol['correlations']
            ### output ftcc 
            if self.global_flag_output_ftcc:
                self.__output_ftcc__(vol_correlations, dir_nm)
            ### output cross-term
            if self.global_flag_output_cross_term:
                self.__output_stcc__(vol_correlations, dir_nm)
            ### output seismic waves
            if self.global_flag_output_seismic_waves:
                self.__output_seismic_wave(vol_seismic_waves, dir_nm)
    def __output_ftcc__(self, vol_correlations, dir_nm):
        if not self.global_flag_output_ftcc:
            return
        mpi_print_log('outputing ftcc... to sac files', 2, self.local_log_fid, True )
        for (isac1, isac2), content in vol_correlations.items():
            ftcc = content['ftcc']
            stnm1, stnm2 = ftcc['stnm1'], ftcc['stnm2']
            if self.global_include_rcv_lst != None: ### include options
                if (stnm1 not in self.global_include_rcv_lst) or (stnm2 not in self.global_include_rcv_lst):
                    continue
            if self.global_exclude_rcv_lst != None: ### exclude options
                if (stnm1 in self.global_exclude_rcv_lst) or (stnm2 in self.global_exclude_rcv_lst):
                    continue
            ###
            ftcc_fnm = '%s/ftcc_%sx%s.sac' % (dir_nm, stnm1, stnm2)
            mpi_print_log(ftcc_fnm, 3, self.local_log_fid, True )
            ###
            ftcc['user3'] = ftcc['inter-dist']
            ftcc['user4'] = ftcc['az-diff']   
            ftcc['user5'] = ftcc['gcarc-diff']
            ftcc['user6'] = ftcc['gcarc1']    
            ftcc['user7'] = ftcc['gcarc2']    
            ftcc['kuser3'] = 'interD'
            ftcc['kuser4'] = 'daz'
            ftcc['kuser5'] = 'dgcarc'
            ftcc['kuser6'] = 'gcarc1'
            ftcc['kuser7'] = 'gcarc2'
            ftcc.write(ftcc_fnm)
    def __output_stcc__(self, vol_correlations, dir_nm):
        if not self.global_flag_output_cross_term:
            return
        mpi_print_log('outputing cross-terms... to sac files', 2, self.local_log_fid, True )
        ###
        for (isac1, isac2), content in vol_correlations.items():
            sac_pair_lst = list(content['stcc'].values())
            stnm1, stnm2 = sac_pair_lst[0][0]['stnm1'], sac_pair_lst[0][0]['stnm2']
            ###
            if self.global_include_rcv_lst != None: ### include receivers
                if (stnm1 not in self.global_include_rcv_lst) or (stnm2 not in self.global_include_rcv_lst):
                    continue
            if self.global_exclude_rcv_lst != None: ### exclude receivers
                if (stnm1 in self.global_exclude_rcv_lst) or (stnm2 in self.global_exclude_rcv_lst):
                    continue
            ###
            for stcc1, stcc2 in  sac_pair_lst:
                ###
                cross_term = stcc1['cross_term']
                if self.global_include_cross_term_lst != None: ### include cross-terms
                    if cross_term not in self.global_include_cross_term_lst:
                        continue
                if self.global_exclude_cross_term_lst != None: ### exclude cross-terms
                    if cross_term in self.global_exclude_cross_term_lst:
                        continue
                ###
                st1_fnm = '%s/stcc1_%sx%s_%s.sac' % (dir_nm, stnm1, stnm2, cross_term )
                st2_fnm = '%s/stcc2_%sx%s_%s.sac' % (dir_nm, stnm1, stnm2, cross_term )
                mpi_print_log(st1_fnm, 3, self.local_log_fid, True )
                mpi_print_log(st2_fnm, 3, self.local_log_fid, True )
                ###
                for it in [stcc1, stcc2]:
                    it['user3']  = it['inter-dist']
                    it['user4']  = it['az-diff']   
                    it['user5']  = it['gcarc-diff']
                    it['user6']  = it['gcarc1']    
                    it['user7']  = it['gcarc2']    
                    it['kuser3'] = 'interD'
                    it['kuser4'] = 'daz'
                    it['kuser5'] = 'dgcarc'
                    it['kuser6'] = 'gcarc1'
                    it['kuser7'] = 'gcarc2'
                ###
                stcc1.write(st1_fnm)
                stcc2.write(st2_fnm)
    def __output_seismic_wave(self, vol_seismic_waves, dir_nm):
        if not self.global_flag_output_seismic_waves:
            return
        mpi_print_log('outputing seismic waves... to sac files', 2, self.local_log_fid, True )
        ###
        for it in vol_seismic_waves:
            saclst= list(it.values())
            stnm = '%s.%s' % (saclst[0]['knetwk'], saclst[0]['kstnm'] )
            ###
            if self.global_include_rcv_lst != None: ### include receivers
                if stnm not in self.global_include_rcv_lst:
                    continue
            if self.global_exclude_rcv_lst != None: ### exclude receivers
                if stnm in self.global_exclude_rcv_lst:
                    continue
            ###
            for st in saclst:
                wave_name =  st['phase']
                ###
                if self.global_include_seismic_wave_lst != None: ### include seismic waves
                    if wave_name not in self.global_include_seismic_wave_lst:
                        continue
                if self.global_exclude_seismic_wave_lst != None: ### exclude seismic waves
                    if wave_name in global_exclude_seismic_wave_lst:
                        continue
                ###
                sacfnm = '%s/seismic-wave_%s.%s_%s.sac' % (dir_nm, st['knetwk'], st['kstnm'], st['phase'])
                mpi_print_log(sacfnm, 3, self.local_log_fid, True )
                ###
                st['user8']  =  st['slowness']
                st['kuser8'] = 'rp'
                st.write(sacfnm)


if __name__ == "__main__":
    ###
    pkl_fnmlst = []
    out_put_prenm = ''
    log_prefnm = ''
    flag_output_ftcc            = False
    flag_output_cross_term      = False
    flag_output_seismic_waves   = False
    exclude_rcv_lst         = None
    include_rcv_lst         = None
    exclude_cross_term_lst  = None
    include_cross_term_lst  = None
    exclude_seismic_wave_lst= None
    include_seismic_wave_lst= None
    ###
    HMSG = '%s -I pklfnm_lst.txt  -O output_prenm -L log_prenm  [--ftcc] [--stcc] [--seismicwave] ' % (sys.argv[0] )
    HMSG2 = """
    [--exc_rcv  exc.txt] [--inc_rcv  inc.txt] [--exc_stcc exc.txt] [--inc_stcc inc.txt] [--exc_wave exc.txt] [--inc_wave inc.txt] 

    
    -I pklfnm_lst.txt : a text file that list all `/pkl` files.
    -O output_prenm : the pre-name for output files.
    -L log_prenm : pre-name for log files in mpi-parallel mode, or the log filename in serial mode.
    
    [--ftcc]        : to output ftcc.
    [--stcc]        : to output stcc.
    [--seismicwave] : to output seimic waves.
    [--exc_rcv  exc.txt] [--inc_rcv  inc.txt]  : a file of which contain the included or excluded receivers. 
                                                (default is to output all receivers.)
    [--exc_stcc exc.txt] [--inc_stcc inc.txt] : a file of which contain the included or excluded cross-terms.
                                                (defualt it to output all cross-terms.)
    [--exc_wave exc.txt] [--inc_wave inc.txt] : a file of which contain the included or excluded seismic waves.
                                                (defualt it to output all seismic waves.) 
    """
    HMSG = HMSG + HMSG2
    ###
    if len(sys.argv) < 2:
        print(HMSG)
        sys.exit(0)
    ###
    options, remainder = getopt.getopt(sys.argv[1:], 'I:O:L:H', ['ftcc', 'stcc', 'seismicwave', 'exc_rcv=', 'inc_rcv=', 'exc_stcc=', 'inc_stcc=', 'exc_wave=', 'inc_wave='] )
    for opt, arg in options:
        if opt in ('-I'):
            pkl_fnmlst = [line.strip() for line in open(arg, 'r')]
        elif opt in ('-O'):
            output_prenm = arg
        elif opt in ('-L'):
            log_prefnm = arg
        elif opt in ('-H'):
            print(HMSG)
            sys.exit(0)
        elif opt in ('--ftcc'):
            flag_output_ftcc = True
        elif opt in ('--stcc'):
            flag_output_cross_term = True
        elif opt in ('--seismicwave'):
            flag_output_seismic_waves = True
        elif opt in ('--exc_rcv' ):
            with open(arg,'r') as fid:
                tmp = fid.readlines()
                vol = [it for it in line.strip().split() for line in tmp]
                exclude_rcv_lst         = vol
        elif opt in ('--inc_rcv' ):
            with open(arg,'r') as fid:
                tmp = fid.readlines()
                vol = [it for it in line.strip().split() for line in tmp]
                include_rcv_lst         = vol
        elif opt in ('--exc_stcc'):
            with open(arg,'r') as fid:
                tmp = fid.readlines()
                vol = [it for it in line.strip().split() for line in tmp]
                exclude_cross_term_lst  = vol
        elif opt in ('--inc_stcc'):
            with open(arg,'r') as fid:
                tmp = fid.readlines()
                vol = [it for it in line.strip().split() for line in tmp]
                include_cross_term_lst  = vol
        elif opt in ('--exc_wave'):
            with open(arg,'r') as fid:
                tmp = fid.readlines()
                vol = [it for it in line.strip().split() for line in tmp]
                exclude_seismic_wave_lst= vol
        elif opt in ('--inc_wave'):
            with open(arg,'r') as fid:
                tmp = fid.readlines()
                vol = [it for it in line.strip().split() for line in tmp]
                include_seismic_wave_lst= vol
        else:
            print('invalid options: %s' % (opt) )
            print(HMSG)
            sys.exit(-1)
    ###
    print('>>> cmd: ', ' '.join(sys.argv ) )
    ###
    if (not flag_output_ftcc) and (not flag_output_cross_term) and (not flag_output_seismic_waves): # nothing to output
        sys.exit(0) 
    if not flag_output_cross_term:
        exclude_cross_term_lst = None
        include_cross_term_lst = None
    if not flag_output_seismic_waves:
        exclude_seismic_wave_lst = None
        include_seismic_wave_lst = None
    ###
    app = cc_stcc_pkl2sac(pkl_fnmlst, output_prenm, log_prefnm)
    app.init(   flag_output_ftcc            =flag_output_ftcc, 
                flag_output_cross_term      =flag_output_cross_term, 
                flag_output_seismic_waves   =flag_output_seismic_waves,    ####
                exclude_rcv_lst         =exclude_rcv_lst,           include_rcv_lst         =include_rcv_lst, 
                exclude_cross_term_lst  =exclude_cross_term_lst,    include_cross_term_lst  =include_cross_term_lst,
                exclude_seismic_wave_lst=exclude_seismic_wave_lst,  include_seismic_wave_lst=include_seismic_wave_lst)
    app.run()