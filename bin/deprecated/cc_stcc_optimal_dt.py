#!/usr/bin/env python3
"""
Executable file to optimal optimal `time shift` of each cross-term given 
a set of cross-terms. 
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
import sys
from random import random

def get_stack(st_lst_tmp, dt, norm_stack=False):
    st_lst = deepcopy(st_lst_tmp)
    for it, idt in zip(st_lst, dt):
        it.shift_time_all(idt)
    stack = sac.stack_sac(st_lst)
    if norm_stack:
        stack.norm()
    return stack
def run_direct(st_lst_tmp, time_window, niter= 5):
    """
    """
    ### pre-norm
    st_lst = deepcopy(st_lst_tmp)
    for it in st_lst:
        it.norm()
        #print(it['t9'])
    ###
    old_time_shift = [2425-it['reference-time'] for it in st_lst]
    old_stack = get_stack(st_lst, old_time_shift, norm_stack=True)
    for iter in range(niter):
        ### get the stack
        ### get the time shift for each st
        optimal_timeshift = np.zeros(len(st_lst), dtype=np.float32 )
        for isac, it in enumerate(st_lst):
            search_range = (2425-it['reference-time']-25, 2425-it['reference-time']+25 )
            optimal_timeshift[isac] = optimal_dt_direct(it, old_stack, time_window, search_range)
            #print(optimal_timeshift[isac] )
        ### obtain the new stack
        new_stack = get_stack(st_lst, optimal_timeshift, norm_stack=True)
        ### plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 12), gridspec_kw={'height_ratios':[5, 1] }  )
        sac.plot_sac_lst(st_lst, ax1)
        junk = [sac.time_shift_all_sac(it, dt) for it,dt in zip(st_lst, optimal_timeshift) ]
        sac.plot_sac_lst(junk, ax2)
        tmp_old_stack = get_stack(st_lst, old_time_shift, norm_stack=False)
        tmp_new_stack = get_stack(st_lst, optimal_timeshift, norm_stack=False)
        tmp_old_stack.plot_ax(ax3, color='b')
        tmp_new_stack.plot_ax(ax3, color='r')
        tmp_new_stack.plot_ax(ax4, color='r')
        plt.show()
        old_stack = new_stack


def run():
    pkl_fnm = '/home/catfly/workspace/cc_tomo_first_trial/05_cross_terms_nci/01_workspace/stcc_2009_015_17_49_39_+0000.pkl'
    with open(pkl_fnm, 'rb') as fid:
        all_vol = pickle.load(fid)
        full_stnm = all_vol['full_stnm'] 
        vol_correlations  = all_vol['correlations']
        for content in vol_correlations.values():
            ftcc = content['ftcc']
            stnm1, stnm2 = ftcc['stnm1'], ftcc['stnm2']
            stcc_dict = content['stcc'] ## cross-term --> (stcc1, stcc2)
            cross_term_lst = sorted(stcc_dict.keys())
            ###
            stcc1_lst = [stcc_dict[ct][0] for ct in cross_term_lst]
            #stcc2_lst = [stcc_dict[ct][1] for ct in cross_term_lst]
            run_direct(stcc1_lst, [2375, 2475], 5)
        

def get_stack(st_lst, dt_lst, norm_return=False):
    """
    """
    junk = deepcopy(st_lst)
    for dt, it in zip(dt_lst, junk):
        it.shift_time_all(dt)
    stack = sac.sac.stack_sac(junk)
    if norm_return:
        stack.norm()
    return stack
def get_dt(st_lst, stack):
    """
    """
    dt = np.zeros(len(st_lst), dtype=np.float32 )
    for isac, st in enumerate( st_lst):
        t0 = 2425-st['t9']
        dt[isac] = optimal_dt_direct(st, stack, [2375, 2475], [t0-25, t0+25] )
    return dt


class adaptive_stack:
    def __init__(self, st_lst, reference_time_shift_sec, relative_search_range_sec, focus_time_window_sec, niter= 3):
        self.st_lst = st_lst
        self.reference_time_shift = reference_time_shift_sec
        self.relative_search_range = relative_search_range_sec
        self.absolute_search_range = [ [it+self.relative_search_range[0], it+self.relative_search_range[1] ] for it in self.reference_time_shift]
        self.focus_time_window = focus_time_window_sec
        self.niter = niter
        ### pre-norm
        for it in self.st_lst:
            it.norm()
        ###
    def run(self):
        dt = np.array( deepcopy(self.reference_time_shift) )
        #dt *= 0.0
        dt_iter = [dt]
        for iter in range(1, self.niter+1):
            print(iter)
            print(dt)
            ## update stack
            stack = self.shift_and_stack(self.st_lst, dt)
            ## update dt
            dt = adaptive_stack.get_dt(self.st_lst, stack, self.absolute_search_range, self.focus_time_window)
            dt_iter.append(dt)
        self.dt_iter= dt_iter
    def plot(self):
        ncol = len(self.dt_iter) + 1
        fig, (ax_row1, ax_row2) = plt.subplots(2, ncol, figsize= (5*ncol, 12), gridspec_kw={'height_ratios':[5, 1] } )
        ### raw
        sac.plot_sac_lst(self.st_lst, ax= ax_row1[0] )
        sac.stack_sac(self.st_lst).plot_ax(ax_row2[0], color='k' )
        ### shift
        for idx, (dt, clr) in enumerate(zip(self.dt_iter, ['C1', 'C2', 'C3', 'C4'] ) ):
            ax0, ax1 = ax_row1[idx+1], ax_row2[idx+1]
            junk = [sac.time_shift_all_sac(it, idt) for it, idt in zip(self.st_lst, dt) ]
            junk_stack = sac.stack_sac(junk)
            sac.plot_sac_lst(junk, ax= ax0 )
            junk_stack.plot_ax(ax=ax_row2[0], color= clr)
            junk_stack.plot_ax(ax=ax1, color= clr )
        for ax1, ax2 in zip(ax_row1, ax_row2):
            ax1.set_xlim([2350, 2500])
            ax2.set_xlim([2350, 2500])
            ax1.grid(True, linestyle=':')
            ax2.grid(True, linestyle=':')
        ###
        plt.show()
    @staticmethod
    def shift_and_stack(st_lst, dt_lst):
        """
        """
        junk = [sac.time_shift_all_sac(st, dt) for dt, st in zip(dt_lst, st_lst) ]
        s = sac.stack_sac(junk)
        s['dat'] *= (1.0/len(st_lst) )
        return s
    @staticmethod
    def get_dt(st_lst, template, search_range_lst, focus_tw):
        dt = np.zeros(len(st_lst), dtype=np.float32 )
        for idx, (st, search_range) in enumerate( zip(st_lst, search_range_lst) ):
            dt[idx] = adaptive_stack.get_dt_single(st, template, search_range, focus_tw)
        return dt
    @staticmethod
    def get_dt_single(single_st, template, search_range, focus_tw):
        """
        """
        delta = single_st['delta']
        dt1, dt2 = search_range
        idx_t1 = int(dt1//delta-1)
        idx_t2 = int(dt2//delta+2)
        minerr = 1.0e20
        optimal_timeshift = dt1
        for idx in range(idx_t1, idx_t2):
            timeshift = idx*delta
            shifted_st = sac.time_shift_all_sac(single_st, timeshift)
            err = adaptive_stack.misfit(shifted_st, template, focus_tw[0], focus_tw[1] )
            #print(shifted_st['b'], template['b'], timeshift, err)
            if err < minerr:
                minerr = err
                optimal_timeshift = timeshift
        return optimal_timeshift
    @staticmethod
    def misfit(st1, st2, t1, t2, order= 2):
        """
        calculate the misfit between two SACTRACEs.
        """
        x1 = sac.truncate_sac(st1, '0', t1, t2)['dat']
        x2 = sac.truncate_sac(st2, '0', t1, t2)['dat']
        #print(x1.size, x2.size, t1, t2)
        if x1.size != x2.size:
            print('Warning: unmatched size (%d, %d) in misfit(...)' % (x1.size, x2.size) )
            sz = min(x1.size, x2.size)
            v =  np.abs(x1[:sz] - x2[:sz] )
            return np.sum( v**order )
        v =  np.abs(x1- x2)
        return np.sum( v**order )


class sa_stack:
    def __init__(self, st_lst, reference_time_shift_sec, relative_search_range_sec, focus_time_window_sec, niter= 10000):
        self.st_lst = st_lst
        self.reference_time_shift = reference_time_shift_sec
        self.relative_search_range = relative_search_range_sec
        self.absolute_search_range = [ [it+self.relative_search_range[0], it+self.relative_search_range[1] ] for it in self.reference_time_shift]
        self.focus_time_window = focus_time_window_sec
        self.niter = niter
        ### pre-norm
        for it in self.st_lst:
            it.norm()
        ###
    def plot(self):
        fig, (ax_row1, ax_row2) = plt.subplots(2, 3, figsize= (10, 12), gridspec_kw={'height_ratios':[5, 1] } )
        ### raw
        sac.plot_sac_lst(self.st_lst, ax= ax_row1[0] )
        sac.stack_sac(self.st_lst).plot_ax(ax_row2[0], color='k' )
        ### shift
        ax1, ax2 = ax_row1[1], ax_row2[1]
        junk = [sac.time_shift_all_sac(it, idt) for it, idt in zip(self.st_lst, self.optimal_dt ) ]
        junk_stack = sac.stack_sac(junk)
        sac.plot_sac_lst(junk, ax= ax1 )
        junk_stack.plot_ax(ax=ax_row2[0], color= 'r')
        junk_stack.plot_ax(ax=ax2, color= 'r' )
        for ax1, ax2 in zip(ax_row1[:2], ax_row2[:2] ):
            ax1.set_xlim([2350, 2500])
            ax2.set_xlim([2350, 2500])
            ax1.grid(True, linestyle=':')
            ax2.grid(True, linestyle=':')
        ###
        ax_row1[2].plot(self.err_lst )
        ax = ax_row1[2].twinx()
        ax.plot(self.temp_lst, 'r')
        plt.show()
    def run(self, dstep_ratio= 0.1):
        temp = 2000
        alpha = 0.98
        ###
        self.err_lst  = np.zeros(self.niter, dtype=np.float32 )
        self.temp_lst = np.zeros(self.niter, dtype=np.float32 )
        ###
        dstep_ratio = dstep_ratio
        old_dt = self.init_dt()
        old_err = self.err(old_dt)
        idecrease = 0
        for it in range(self.niter):
            ### generate dt
            new_dt = self.update_dt(old_dt, dstep_ratio)
            ### new temprature
            new_err = self.err(new_dt)
            derr = new_err - old_err
            if derr < 0.0:
                ### accept
                old_err = new_err
                old_dt = new_dt
                self.err_lst[idecrease] = old_err
                self.temp_lst[idecrease] = temp
                idecrease = idecrease + 1
            elif random() < np.exp(-derr/temp):
                ### 
                old_err = new_err
                old_dt = new_dt
                self.err_lst[idecrease] = old_err
                self.temp_lst[idecrease] = temp
                idecrease = idecrease + 1
            ### decrease temprature
            if derr < 0.0:
                temp = temp * alpha
        self.err_lst= self.err_lst[:idecrease]
        self.temp_lst = self.temp_lst[:idecrease]
        print(self.temp_lst )
        self.optimal_dt = old_dt
    def init_dt(self):
        """
        """
        dt = np.array( [random()*(dt2-dt1)+dt1 for (dt1, dt2) in self.absolute_search_range] )
        return dt
    def update_dt(self, dt, update_ratio= 0.1):
        update_ratio = update_ratio * 2.0
        dt_step = np.array( [ (random()-0.5)*update_ratio*(dt2-dt1) for (dt1, dt2) in self.absolute_search_range ] )
        dt += dt_step
        dt = [idt if dt1<=idt<=dt2 else dt1+(dt2-dt1)*random()*update_ratio*0.5  for (dt1, dt2), idt in zip(self.absolute_search_range, dt)]
        return dt
    def err(self, dt_lst):
        stack = adaptive_stack.shift_and_stack(self.st_lst, dt_lst)
        t1, t2 = self.focus_time_window
        misfit = np.array( [adaptive_stack.misfit(st, stack, t1, t2, order=1) for st in self.st_lst] )
        return np.sum( misfit**2 )

if __name__ == "__main__":
    pkl_fnm = '/home/catfly/workspace/cc_tomo_first_trial/05_cross_terms_nci/01_workspace/stcc_2009_015_17_49_39_+0000.pkl'
    with open(pkl_fnm, 'rb') as fid:
        all_vol = pickle.load(fid)
        full_stnm = all_vol['full_stnm'] 
        vol_correlations  = all_vol['correlations']
        for content in vol_correlations.values():
            ftcc = content['ftcc']
            stnm1, stnm2 = ftcc['stnm1'], ftcc['stnm2']
            stcc_dict = content['stcc'] ## cross-term --> (stcc1, stcc2)
            cross_term_lst = sorted(stcc_dict.keys())
            ###
            for idx in [0]:
                stcc_lst = [stcc_dict[ct][idx] for ct in cross_term_lst]
                ref = [2425-it['t9'] for it in stcc_lst]
                search = [-15, 15]
                focus_tw = [2375, 2475]
                
                #app = adaptive_stack(stcc_lst, ref, search, focus_tw)
                #app.run()
                #app.plot()
                #sys.exit()
                app = sa_stack(stcc_lst, ref, search, focus_tw)
                app.run()
                app.plot()
            #run_direct(stcc1_lst, [2375, 2475], 5)
        


        