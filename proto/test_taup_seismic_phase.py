#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100
import obspy.taup as taup
from obspy.taup.seismic_phase import SeismicPhase
from obspy.taup.tau import Arrivals


if True:
    mod = taup.TauPyModel('ak135')
    arrs = mod.get_ray_paths(30.0, 130,  ['PKIKP'], receiver_depth_in_km=0 )
    PKIKP = arrs[0]
    print('%.8f %.8f %.8f %.8f' % (PKIKP.distance, PKIKP.purist_distance, PKIKP.ray_param, PKIKP.time))
    #arrs.plot()

if True:
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
    print('%.8f %.8f %.8f %.8f' % (arr.distance, np.rad2deg(arr.purist_dist), arr.ray_param, arr.time))
    arr.distance = np.rad2deg(arr.purist_dist)
    #print(arr.path)
    phase.calc_path_from_arrival(arr)
    arrs = Arrivals([arr], tau_model_depth_corrected)
    #arrs.plot()
    #print(arr.path)


if True:
    mod = taup.TauPyModel('ak135')
    evdp_km = 100
    rcvdp_km = 0.1
    phase = 'PKIKP'
    # correct model
    tau_model = mod.model
    tau_model_depth_corrected = tau_model.depth_correct(evdp_km)
    tau_model_depth_corrected.split_branch(rcvdp_km)
    def get_arrival(tau_model_depth_corrected, phase, ray_param, useless_distance=None):
        phase = SeismicPhase(phase, tau_model_depth_corrected, rcvdp_km)
        arr = phase.shoot_ray(useless_distance, ray_param) # Arrival not Arrivals
        arr.distance = np.rad2deg(arr.purist_dist)
        return arr
    def get_arrivals(tau_model_depth_corrected, phase, ray_min, ray_max, max_dist_step=1.0):
        left = get_arrival(tau_model_depth_corrected, phase, min_ray_param)
        right = get_arrival(tau_model_depth_corrected, phase, min_ray_param)
        arrs = [left, right]
        arrs = recur(arrs, tau_model_depth_corrected, max_dist_step)
        return arrs
        pass
        
        #if np.abs(left.purist_dist-right.purist_dist) > max_dist_step:
        #    pass
        #else:
        #    return arrs
    def recur(arrs, tau_model_depth_corrected, max_dist_step):
        
        pass




    