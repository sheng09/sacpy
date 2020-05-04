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
from copy import deepcopy
import matplotlib.pyplot as plt

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
    def init(self,  cc_time_range = [2300, 2600], cc_reference_time_for_shift_sec = 2425,
                exclude_rcv_lst=None, include_rcv_lst=None, 
                exclude_cross_term_lst=None, include_cross_term_lst=None
            ):
        """
        ##################  
        exclude_rcv_lst:    a list of receivers to be excluded. Set `None` to disable this variable.
                            The receiver name should be `KNETWK.KSTNM` (e.g., II.TAU)
        include_rcv_lst:    a list of receivers to be included. Receviers out of this list would be
                            excluded. Set `None` to disable this variable.

        exclude_cross_term_lst: a list of cross-terms to be excluded. Set `None` to disable this variable.
                                (e.g., ['I06-I04', 'I22-_20'])
        include_cross_term_lst: a list of cross-terms to be included. Cross-terms out of this list would be
                                excluded.Set `None` to disable this variable.

        ##################
        In default, the program would plot for all receivers and cross-terms.
        ##################
        Example:
            1. to exclude some of receivers: use `exclude_rcv_lst = ['R1', 'R2'] `
            2. to output only for a few receivers: use `include_rcv_lst = ['R1', 'R2'] `

            3. to exclude some of cross-terms: use `exclude_rcv_lst = ['C1', 'C2'] `
            4. to output only for a few cross-terms: use `include_rcv_lst = ['C1', 'C2'] `
            5. do not output any cross-terms: use `flag_output_cross_term = False`
        """
        self.global_cc_time_range               = cc_time_range
        self.global_cc_reference_time_for_shift_sec = cc_reference_time_for_shift_sec
        self.global_exclude_rcv_lst             = sorted( exclude_rcv_lst        ) if exclude_rcv_lst         != None else None
        self.global_include_rcv_lst             = sorted( include_rcv_lst        ) if include_rcv_lst         != None else None
        self.global_exclude_cross_term_lst      = sorted( exclude_cross_term_lst ) if exclude_cross_term_lst  != None else None
        self.global_include_cross_term_lst      = sorted( include_cross_term_lst ) if include_cross_term_lst  != None else None
        ####
        mpi_print_log('>>> Initialized', 0, self.local_log_fid, False)
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
        ###
    def run(self):
        mpi_print_log('>>> Plotting... ', 0, self.local_log_fid, True )
        for idx, pkl_fnm in enumerate(self.local_pkl_fnmlst):
            self.plot_single_pkl(pkl_fnm, idx, len(self.local_pkl_fnmlst) )
    def plot_single_pkl(self, pkl_fnm, idx, total_number):
        """
        """
        ###
        dir_nm = self.global_output_prenm + '/' + pkl_fnm.split('/')[-1].replace('.pkl', '_imgs/')
        if not os.path.exists(dir_nm):
            os.mkdir(dir_nm)
        ###
        msg = '[%d/%d] %s ==> %s' % (idx+1, total_number, pkl_fnm, dir_nm)
        mpi_print_log(msg, 1, self.local_log_fid, True )
        ###
        with open(pkl_fnm, 'rb') as fid:
            all_vol = pickle.load(fid)
            full_stnm = all_vol['full_stnm'] 
            vol_correlations  = all_vol['correlations']
            ###
            n_total = len(vol_correlations)
            for idx, content in enumerate(vol_correlations.values() ):
                self.__plot_single_correlation(content, dir_nm, idx, n_total)
    def __plot_single_correlation(self, content, dir_nm, idx, n_total):
        ftcc = content['ftcc']
        stnm1, stnm2 = ftcc['stnm1'], ftcc['stnm2']
        ### check receivers
        if self.global_include_rcv_lst != None: ### include options
            if (stnm1 not in self.global_include_rcv_lst) or (stnm2 not in self.global_include_rcv_lst):
                return
        if self.global_exclude_rcv_lst != None: ### exclude options
            if (stnm1 in self.global_exclude_rcv_lst) or (stnm2 in self.global_exclude_rcv_lst):
                return
        ###
        stcc_dict = content['stcc'] ## cross-term --> (stcc1, stcc2)
        cross_term_set = set(stcc_dict.keys() )
        ### check cross-terms
        if self.global_exclude_cross_term_lst != None:
            cross_term_set.difference_update( self.global_exclude_cross_term_lst )
        if self.global_include_cross_term_lst != None:
            cross_term_set.intersection_update( self.global_include_cross_term_lst ) 
        cross_term_lst = sorted(list(cross_term_set) )
        ### cross-terms
        stcc_lst = [stcc_dict[cross_term] for cross_term in cross_term_lst]
        ### stack
        stack_p, stack_n, stack_sp, stack_sn, (shifted_stcc1_lst, shifted_stcc2_lst) = self.__run_stack(stcc_dict, cross_term_lst) # plain, norm, shift-plain, shift-norm
        ### plot
        figname = '%s/%sx%s.png' % (dir_nm, stnm1, stnm2 )
        mpi_print_log('[%d/%d] %s' % (idx+1, n_total, figname), 2, self.local_log_fid, True )
        ###
        self.__atom_plot(figname, stcc_lst, shifted_stcc1_lst, shifted_stcc2_lst, ftcc, stack_p, stack_n, stack_sp, stack_sn, cross_term_lst)
    def __run_stack(self, stcc_dict, cross_term_lst):
        cc_t1, cc_t2 = self.global_cc_time_range
        return_values = {   'noshift-plain': [None, None],
                            'noshift-norm' : [None, None],
                            'shift-plain'  : [None, None],
                            'shift-norm'   : [None, None], 
                            'shifted-stcc-lst' : [None, None] }
        #### no shift, plain and norm stacking
        for idx in [0, 1]:
            stcc_lst = [sac.truncate_sac(stcc_dict[cross_term][idx], '0', cc_t1, cc_t2, clean_sachdr=False ) for cross_term in cross_term_lst]
            return_values['noshift-plain'][idx] = sac.stack_sac(stcc_lst, amp_norm=False)
            return_values['noshift-norm' ][idx] = sac.stack_sac(stcc_lst, amp_norm=True)
        #### shift, plain and norm stacking
        t_ref = self.global_cc_reference_time_for_shift_sec
        for idx in [0, 1]:
            tmp = [sac.time_shift_all_sac(stcc_dict[cross_term][idx], t_ref - stcc_dict[cross_term][idx]['reference-time']) for cross_term in cross_term_lst]
            return_values['shifted-stcc-lst'][idx] = tmp
            stcc_lst = [sac.truncate_sac(it, '0', cc_t1, cc_t2, clean_sachdr=False) for it in tmp]
            return_values['shift-plain'][idx] = sac.stack_sac(stcc_lst, amp_norm=False)
            return_values['shift-norm' ][idx] = sac.stack_sac(stcc_lst, amp_norm=True)
        ###
        return  return_values['noshift-plain'], return_values['noshift-norm'], return_values['shift-plain'  ], return_values['shift-norm'   ], return_values['shifted-stcc-lst']
    def __atom_plot(self, figname, stcc_lst, shifted_stcc1_lst, shifted_stcc2_lst, 
                ftcc, stack_p, stack_n, 
                stack_sp, stack_sn,
                cross_term_lst):
        ####
        cc_t1, cc_t2 = self.global_cc_time_range
        ####
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 12) )
        scale = 0.4
        #################################################################################################
        # ax1 is for not shifted 
        #################################################################################################
        # cross-terms
        for isac, v in enumerate(stcc_lst ): 
            for stcc, color, alpha, linestyle in zip(v, ['k', 'gray'], [0.9, 0.8], ['-', '--'] ):
                st_tmp = sac.truncate_sac(stcc, '0', cc_t1, cc_t2)
                st_tmp.norm()
                ts, ys = st_tmp.get_time_axis(), st_tmp['dat']*scale + isac
                ax1.plot(ts, ys, color=color, alpha=alpha, linestyle=linestyle, linewidth= 0.8 )
            reference_t_cross_term = v[0]['reference-time']
            ax1.plot([reference_t_cross_term, reference_t_cross_term], [isac-0.4, isac+0.4], 'C3', linewidth= 2)
        # plain stack, norm stack
        for zshift, v in zip( [-1, -2], [stack_p, stack_n] ):
            for st, color, alpha, linestyle in zip(v, ['b', 'b'], [0.9, 0.8], ['-', '--'] ):
                st_tmp = sac.truncate_sac(st, '0', cc_t1, cc_t2)
                st_tmp.norm()
                ts, ys = st_tmp.get_time_axis(), st_tmp['dat']*scale + zshift
                ax1.plot(ts, ys, color=color, alpha=alpha, linestyle=linestyle, linewidth= 0.9 )
        # ftcc
        st_tmp = sac.truncate_sac(ftcc, '0', cc_t1, cc_t2)
        st_tmp.norm()
        ts, ys = st_tmp.get_time_axis(), st_tmp['dat']*scale-3
        ax1.plot(ts, ys, color='b', linewidth= 0.9 )
        #################################################################################################
        #  ax2 is for shifted 
        #################################################################################################
        # cross-terms
        for isac, v in enumerate(zip(shifted_stcc1_lst, shifted_stcc2_lst) ): 
            for stcc, color, alpha, linestyle in zip(v, ['k', 'gray'], [0.9, 0.8], ['-', '--'] ):
                st_tmp = sac.truncate_sac(stcc, '0', cc_t1, cc_t2)
                st_tmp.norm()
                ts, ys = st_tmp.get_time_axis(), st_tmp['dat']*scale + isac
                ax2.plot(ts, ys, color=color, alpha=alpha, linestyle=linestyle, linewidth= 0.8 )
            reference_t_cross_term = v[0]['t9']
            ax2.plot([reference_t_cross_term, reference_t_cross_term], [isac-0.4, isac+0.4], 'C3', linewidth= 2)
        # shifted plain stack, shifted norm stack
        for zshift, v in zip( [-1, -2], [stack_sp, stack_sn] ):
            for st, color, alpha, linestyle in zip(v, ['b', 'b'], [0.9, 0.8], ['-', '--'] ):
                st_tmp = sac.truncate_sac(st, '0', cc_t1, cc_t2)
                st_tmp.norm()
                ts, ys = st_tmp.get_time_axis(), st_tmp['dat']*scale + zshift
                ax2.plot(ts, ys, color=color, alpha=alpha, linestyle=linestyle, linewidth= 0.9 )
        # ftcc
        st_tmp = sac.truncate_sac(ftcc, '0', cc_t1, cc_t2)
        st_tmp.norm()
        ts, ys = st_tmp.get_time_axis(), st_tmp['dat']*scale-3
        ax2.plot(ts, ys, color='b', linewidth= 0.9 )
        #################################################################################################
        #  adjust basemap for ax1 and ax2
        #################################################################################################
        ticklabels = ['ftcc', 'norm-stack', 'plain-stack']
        ticklabels.extend(cross_term_lst)
        stnm1, stnm2 = ftcc['stnm1'], ftcc['stnm2']
        gcarc1, gcarc2 = ftcc['gcarc1'], ftcc['gcarc2']
        for ax in [ax1, ax2]:
            ax.plot([self.global_cc_reference_time_for_shift_sec, self.global_cc_reference_time_for_shift_sec], [-3.5, len(cross_term_lst)-0.5], 'r--', alpha= 0.5, linewidth= 2)
            ax.set_xlim([cc_t1, cc_t2])
            ax.set_xlabel('Time (s)')
            ax.set_yticks(np.arange(-3, len(cross_term_lst) ) )
            ax.set_yticklabels( ticklabels )
            ax.set_ylim([-3.5, len(cross_term_lst)-0.5] )
            ax.set_title('%s(%.1f) x %s(%.1f)' % (stnm1, gcarc1, stnm2, gcarc2) )
        #################################################################################################
        #  output to pdf
        #################################################################################################
        plt.tight_layout()
        plt.savefig(figname, bbox_inches = 'tight', pad_inches = 0.5)
        plt.close()
    
if __name__ == "__main__":
    ###
    pkl_fnmlst = []
    out_put_prenm = ''
    log_prefnm = ''
    cc_time_range = [2300, 2600]
    cc_reference_time_for_shift_sec = 2425
    exclude_rcv_lst         = None
    include_rcv_lst         = None
    exclude_cross_term_lst  = None
    include_cross_term_lst  = None
    ###
    HMSG = '%s -I pklfnm_lst.txt  -O output_prenm -L log_prenm -W t1/t2 -T cc_feature_t [--exc_rcv  exc.txt] [--inc_rcv  inc.txt] [--exc_stcc exc.txt] [--inc_stcc inc.txt]' % (sys.argv[0] )
    HMSG2 = """
    
    This program is for plotting stcc given `.pkl` files generated by `cc_stcc.py`.

    
    -I pklfnm_lst.txt : a text file that list all `.pkl` files.
    -O output_prenm : the pre-name for output files.
    -L log_prenm : pre-name for log files in mpi-parallel mode, or the log filename in serial mode.
    -W t1/t2 : correlation time window to focus on.
    -T cc_feature_t: the reference time for the correlation feature.
    
    [--exc_rcv  exc.txt] [--inc_rcv  inc.txt]  : a file of which contain the included or excluded receivers. 
                                                (default is to output all receivers.)
    [--exc_stcc exc.txt] [--inc_stcc inc.txt] : a file of which contain the included or excluded cross-terms.
                                                (defualt it to output all cross-terms.)
    """
    HMSG = HMSG + HMSG2
    ###
    if len(sys.argv) < 2:
        print(HMSG)
        sys.exit(0)
    ###
    options, remainder = getopt.getopt(sys.argv[1:], 'I:O:L:W:T:H', ['exc_rcv=', 'inc_rcv=', 'exc_stcc=', 'inc_stcc=' ] )
    for opt, arg in options:
        if opt in ('-I'):
            pkl_fnmlst = [line.strip() for line in open(arg, 'r')]
        elif opt in ('-O'):
            output_prenm = arg
        elif opt in ('-L'):
            log_prefnm = arg
        elif opt in ('-W'):
            cc_time_range = [float(it) for it in arg.split('/')]
        elif opt in ('-T'):
            cc_reference_time_for_shift_sec = float(arg)
        elif opt in ('-H'):
            print(HMSG)
            sys.exit(0)
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
        else:
            print('invalid options: %s' % (opt) )
            print(HMSG)
            sys.exit(-1)
    ###
    print('>>> cmd: ', ' '.join(sys.argv ) )
    ###
    app = cc_stcc_pkl2sac(pkl_fnmlst, output_prenm, log_prefnm)
    app.init(   cc_time_range = cc_time_range, cc_reference_time_for_shift_sec= cc_reference_time_for_shift_sec, 
                exclude_rcv_lst         =exclude_rcv_lst,           include_rcv_lst         =include_rcv_lst, 
                exclude_cross_term_lst  =exclude_cross_term_lst,    include_cross_term_lst  =include_cross_term_lst )
    app.run()