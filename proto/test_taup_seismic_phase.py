#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100
import obspy.taup as taup
from obspy.taup.seismic_phase import SeismicPhase
from obspy.taup.tau import Arrivals
from blist import blist
import cProfile
from time import time as compute_time
import sys
from copy import copy, deepcopy
import functools
from sacpy.profiler_tools import Timer, TimeSummary

if False:
    mod = taup.TauPyModel('ak135')
    arrs = mod.get_ray_paths(30.0, 130,  ['PKIKP'], receiver_depth_in_km=0 )
    PKIKP = arrs[0]
    print('%.8f %.8f %.8f %.8f' % (PKIKP.distance, PKIKP.purist_distance, PKIKP.ray_param, PKIKP.time))
    #arrs.plot()

if False:
    mod = taup.TauPyModel('ak135')
    evdp_km = 30
    rcvdp_km = 0
    phase = 'PKIKP'
    ray_param = 108.78406294*1.000004175
    # 1. correct for evdp and rcvdp
    tau_model = mod.model
    tau_model_depth_corrected = tau_model.depth_correct(evdp_km)
    tau_model_depth_corrected.split_branch(rcvdp_km)


    # 2. create a seismic phase
    phase = SeismicPhase(phase, tau_model_depth_corrected, rcvdp_km)
    #print('phase ray_param range:', phase.min_ray_param, phase.max_ray_param)
    arr = phase.shoot_ray(130.0, ray_param)
    print('%.8f %.8f %.8f %.8f' % (arr.distance, arr.purist_distance, arr.ray_param, arr.time))
    arr.distance = np.rad2deg(arr.purist_dist)
    #print(arr.path)
    phase.calc_path_from_arrival(arr)
    arrs = Arrivals([arr], tau_model_depth_corrected)
    #arrs.plot()
    #print(arr.path)



if True:
    x = np.abs(1.0)
    tsummary = TimeSummary()
    def get_arrivals(tau_model_depth_corrected, phase_name, rcvdp_km, ray_param_range=None, max_dist_step_deg=1.0, cal_ray_path=True, recursion_limit=None):
        try:
            min_ray_param, max_ray_param = ray_param_range
        except Exception:
            junk_phase = SeismicPhase(phase_name, tau_model_depth_corrected, rcvdp_km)
            min_ray_param, max_ray_param = junk_phase.min_ray_param, junk_phase.max_ray_param
        min_ray_param, max_ray_param = float(min_ray_param), float(max_ray_param)
        left = get_arrival(tau_model_depth_corrected, phase_name, min_ray_param)
        right = get_arrival(tau_model_depth_corrected, phase_name, max_ray_param)
        arrs = [left, right]
        with Timer(message='denser_arrivals_recur', color='C2', verbose=True, summary=tsummary):
            denser_arrivals_recur(arrs, tau_model_depth_corrected, phase_name, max_dist_step_deg, recursion_limit=recursion_limit)
        arrs = [left, right]
        with Timer(message='denser_arrivals_loop', color='C2', verbose=True, summary=tsummary):
            denser_arrivals_loop(arrs, tau_model_depth_corrected, phase_name, max_dist_step_deg, recursion_limit=recursion_limit)
        if cal_ray_path:
            with Timer(message='cal ray paths', color='C4', verbose=True, summary=tsummary):
                for it in arrs:
                    it.phase.calc_path_from_arrival(it)
        return Arrivals(arrs, tau_model_depth_corrected)
    #static_phase_dict = dict()
    def get_arrival(tau_model_depth_corrected, phase_name, ray_param, useless_distance=0):
        #if phase_name not in static_phase_dict:
        #    #print('new')
        #    phase = SeismicPhase(phase_name, tau_model_depth_corrected, rcvdp_km)
        #    static_phase_dict[phase_name] = phase
        #else:
        #    #print('copy')
        #    phase = copy( static_phase_dict[phase_name] )
        phase = SeismicPhase(phase_name, tau_model_depth_corrected, rcvdp_km)
        arr = phase.shoot_ray(useless_distance, ray_param) # Arrival not Arrivals
        arr.distance = arr.purist_distance
        return arr
    def denser_arrivals_loop(arrs, tau_model_depth_corrected, phase_name, max_dist_step_deg=1.0, recursion_limit=None):
        """
        Loop method.
        """
        def keep_going(temp_arrs, temp_max_dist_step_deg):
            purist_distance = [it.purist_distance for it in temp_arrs]
            abs_diff = np.abs( np.diff( purist_distance ) )
            return abs_diff.max() > temp_max_dist_step_deg
        while keep_going(arrs, max_dist_step_deg):
            ray_params = np.array([it.ray_param for it in arrs])
            mid_ray_params = 0.5*( ray_params[:-1] + ray_params[1:] )
            purist_distances = [it.purist_distance for it in arrs]
            diff = np.abs(np.diff(purist_distances) )
            idxs = np.where(diff>max_dist_step_deg)[0] + 1
            for idx_right, rp_mid in zip( idxs[::-1], mid_ray_params[::-1] ):
                arr_mid = get_arrival(tau_model_depth_corrected, phase_name, rp_mid, 0)
                arrs.insert(idx_right, arr_mid)
            if recursion_limit:
                recursion_limit = recursion_limit-1
                if recursion_limit<=0:
                    break
    def denser_arrivals_recur(arrs, tau_model_depth_corrected, phase_name, max_dist_step_deg=1.0, recursion_limit=None):
        """
        Recursion method to get a list of arrivals sorted with respect to the slowness (ray parameters).
        """
        purist_distance = [it.purist_distance for it in arrs]
        abs_diff = np.abs( np.diff( purist_distance ) )
        actual_max_dist_step = abs_diff.max()
        if actual_max_dist_step <= max_dist_step_deg:
            return
        else:
            if recursion_limit != None:
                if recursion_limit <=0:
                    return arrs
                recursion_limit = recursion_limit -1
            ray_params = np.array([it.ray_param for it in arrs])
            mid_ray_params = 0.5*( ray_params[:-1] + ray_params[1:] )
            purist_distances = [it.purist_distance for it in arrs]
            diff = np.abs(np.diff(purist_distances) )
            idxs = np.where(diff>max_dist_step_deg)[0] + 1
            for idx_right, rp_mid in zip( idxs[::-1], mid_ray_params[::-1] ):
                arr_mid = get_arrival(tau_model_depth_corrected, phase_name, rp_mid, 0)
                arrs.insert(idx_right, arr_mid)
            return denser_arrivals_recur(arrs, tau_model_depth_corrected, phase_name, max_dist_step_deg, recursion_limit)
    #####
    with Timer(message='run all', color='gray', verbose=True, summary=tsummary):
        mod = taup.TauPyModel('ak135')
        evdp_km = 100
        rcvdp_km = 0.1
        phase_name = 'PKIKP'
        # correct model
        with Timer(message='correct model', color='C0', verbose=True, summary=tsummary):
            tau_model = mod.model
            tau_model_depth_corrected = tau_model.depth_correct(evdp_km)
            tau_model_depth_corrected.split_branch(rcvdp_km)
        arrs = get_arrivals(tau_model_depth_corrected, phase_name, rcvdp_km, None, 1, recursion_limit=None)
    tsummary.plot_rough()
    #tsummary.plot(show=True, plot_percentage=False)
        #print([it.distance for it in arrs])
    arrs.plot()