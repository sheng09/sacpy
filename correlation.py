#!/usr/bin/env python3

"""
This module provides functions to compute the ray parameters, correlation times, inter-receiver distances for cross-correlation features.

Example:
    >>>
    >>> cc_features = get_all_interrcv_ccs(['P*', 'PcP*'], evdp_km=50, model_name='ak135', log=sys.stdout, rcvdp1_km=0.0, rcvdp2_km=0.0)
    >>>
    >>> # Now, plot
    >>> for it in cc_features:
    >>>     name            = it['name']
    >>>     ray_param       = it['ray_param']
    >>>     time            = it['time']
    >>>     purist_distance = it['purist_distance']
    >>>     distance        = it['distance']
    >>>     plt.plot(distance, time, label=name)
    >>> plt.show()
    >>>
"""
from obspy import taup
from numpy import arange, array, linspace, interp, where, abs, diff, where, round, sum
from sacpy.utils import Timer, CacheRun
from obspy.taup.seismic_phase import SeismicPhase
from matplotlib import pyplot as plt
import sacpy
from copy import deepcopy
import sys, os, os.path, pickle, warnings, traceback
import matplotlib
from sacpy.processing import round_degree_180, round_degree_360

__global_verbose= False

"""
Here, we rely on the `taup` method to compute ray_param-distance-time relations for cross-correlation features.
So, it is better to understand how obspy's `get_travel_times(...)` works.
Below shows the workflow of the `get_travel_times(...)` method in `obspy.taup`.

First, generate a initial object of `TauModel` (or `TauPyModel.model`)
Then, correct the `TauModel` object with source depth and receiver depth.
Then, form the `SeismicPhase` object with the corrected `TauModel` object and receiver depth.
Then, using shooting methods `SeismicPhase.shoot_ray(...)` by trying different ray parameters
      to get the arrival with its epicentral distance close to the targeted distance.

        In `obspy.taup`, when running `TauPyModel.get_travel_times(....)` the workflow is as follows:
        1. `mod = obspy.taup.TauPyModel('ak135')` will run:
            +--> `mod.model.from_file(model, cache=cache)` where the `model` is either a pre-existed model name or a file name.
        2. `mod.get_travel_times(source_depth_in_km, distance_in_degree, phase_list, receiver_depth_in_km)` will run:
            +--> `TauPTime(TauPyModel.model, phase_list, source_depth_in_km, distance_in_degree, receiver_depth_in_km, ray_param_tol)`
            |     where `TauPyModel.model` is a `TauModel` object
            +--> `TauPTime.run()`
                  +--> `TauPTime.depth_correct(source_depth_in_km, receiver_depth_in_km)`
                  |     will correct and modify `TauPTime.depth_corrected_model` which is a `TauModel` object
                  |     +--> `TauModel.depth_correct(source_depth_in_km)`
                  |     +--> `TauModel.split_branch(receiver_depth_in_km)`
                  |
                  +--> `TauPTime.calculate(distance_in_degree)` the arrivals
                        +--> `TauPTime.depth_correct(...)`
                        |    will correct `TauPTime.depth_corrected_model` if necessary if the previous call change the phases
                        +--> `TauPTime.recalc_phases(...)`
                        |    +--> `SeismicPhase(temp_phase_name, TauPTime.depth_corrected_model, TauPTime.receiver_depth)`
                        |          |  This will take the `TauPTime.depth_corrected_model`, which corrected from the `TauPyModel.model`.
                        |          |  This will do sth...
                        |          +-->  TauPTime.phases =  a list of `SeismicPhase` objects
                        |
                        +--> TauPTime.calc_time(...)
                             +--> TauPTime.phases[idx].calc_time(degrees, ray_param_tolerance)
                                  which in fact is `SeismicPhase.calc_time(...)`
                                                    +--> `SeismicPhase.refine_arrival(...)`
                                                          +--> `SeismicPhase.shoot_ray(degrees, ray_param)` which returns an `Arrival` object
                                                                and where only the `ray_param` is useful in the computation, and
                                                                the `degrees` is only used to force change the distance parameter in the `Arrival` object

From the workflow of obspy's `get_travel_times(...)`, using the `SeismicPhase` and its `shoot_ray(...)` is very powerful!
In other words, we can compute epicentral distance and travel time given ray parameter and phase name!
"""
def __example_use_SeismicPhase_shoot_ray():
    """
    This function showcase how to use the `SeismicPhase.shoot_ray(...)` method by providing a ray parameter.
    """
    evdp_km  = 0
    rcvdp_km = 300
    phase = 'PKIKP'
    rp = 1.3
    # invokering the `SeismicPhase` method
    tau_mod = taup.TauPyModel('ak135').model                     # get TauModel object
    tau_mod_corr = tau_mod.depth_correct(evdp_km)                # correct/split the source and receiver depth
    tau_mod_corr = tau_mod_corr.split_branch(rcvdp_km)
    seismic_phase = SeismicPhase(phase, tau_mod_corr, rcvdp_km)  # generate a seismic phase object
    arr2 = seismic_phase.shoot_ray(0, rp)                        # shoot the ray, please note, the first parameter is assigned to returned `Arrival.distance`!
    print(arr2.purist_distance, arr2.time, arr2.ray_param)       # use `Arrival.purist_distance` to access the real epicentral distance!
    # now, we compare with obspy's get_travel_times(...)         # Note, the `Arrival.distance` can be above 360 degree, which is the real traveled distance!
    mod = taup.TauPyModel('ak135')
    arr = mod.get_travel_times(evdp_km, arr2.purist_distance, [phase], receiver_depth_in_km=rcvdp_km, ray_param_tol=0.000001)[0]
    print(arr.purist_distance, arr.time, arr.ray_param)
    rp = arr.ray_param



######################################################################################################################################################################
# Correct a `TauModel` object with source and receiver depth
######################################################################################################################################################################
def get_corrected_model(tau_model, evdp_km, rcvdp_km):
    """
    Correct a model (an object of `TauModel`) with certain source and receiver depth, and
    return the corrected model (also an object of `TauModel`).
    #
    Parameters:
        tau_model: an object a `obspy.taup.TauModel`.
        evdp_km:   the source depth in km.
        rcvdp_km:  the receiver depth in km.
    Returm:
        an object of `TauModel` for the corrected model.
    #
    E.g.,
        >>> mod       = taup.TauPyModel('ak135')
        >>> tau_model = mod.model # Note! `mod` is a `TauPyModel` object
        >>> mod_corr = get_corrected_model(tau_model, evdp_km=310, rcvdp_km=0.3)
    """
    tau_model_corrected = tau_model.depth_correct(evdp_km)
    tau_model_corrected = tau_model_corrected.split_branch(rcvdp_km)
    return tau_model_corrected



######################################################################################################################################################################
# Dealing with correlation features
######################################################################################################################################################################
class CorrelationFeatureName:
    def __init__(self):
        pass
    @staticmethod
    def __find_n_character(a_string, char):
        """
        Find the maximal number of continuous character `char` in a string `a_string`.
        #
        Parameters:
            a_string: a string.
            char:     a character.
        #
        Return:
            the maximal number of continuous character `char` in the string `a_string`.
        #
        E.g.,
        'PKKKP' should return 3 for K, PKKSKP should return 2 for K.
        """
        n = len(a_string)
        for i in range(n, -1, -1):
            template = char*i
            if template in a_string:
                break
        return i
    @staticmethod
    def __replace_K_seismic_phase(original_string, max_nchar):
        """
        Replace seismic phases from S[_i K]P to P[_i K]S, where i could be [1, max_nchar].
        """
        result = original_string
        seg = ''
        for i in range(max_nchar):
            seg += 'K'
            ph1 = 'S%sP' % seg
            ph2 = 'P%sS' % seg
            result = result.replace(ph1, ph2)
        return result
    @staticmethod
    def __replace_I_seismic_phase(original_string, max_nchar):
        """
        Replace seismic phases from SK[_i I]KP to PK[_i I]KS, where i could be [1, max_nchar].
        """
        result = original_string
        seg = ''
        for i in range(max_nchar):
            seg += 'I'
            ph1 = 'SK%sKP' % seg
            ph2 = 'PK%sKS' % seg
            result = result.replace(ph1, ph2)
        return result
    @staticmethod
    def decode(encoded_feature_name):
        """
        Decode the encoded feature name to the normal feature name.
        #
        Parameters:
            encoded_feature_name: a string of the encoded feature name.
                                  Its has the format of '_encoded_NAME1@N1+NAME2+NAME3@N3+...',
                                  It must started with the '_encoded_'.
                                  The 'NAME1', 'NAME2'... can be
                                  1. seismic phase name, such as P, PKIKP, PcP, ScP, PcS,...
                                  2. normal correlation feature name which are two seismic
                                     phase names seperated by a '-',
                                     such as 'PcS-PcP', 'PKIKP-PKJKP'...
                                     Simplified names are not allows, such as 'cS-cP'.
                                  The 'N1', 'N2'... are numbers which means the repeat times of
                                  the 'NAME1', 'NAME2'...
                                  If 'Ni' is 1, then 'NAMEi@1' can be simplified as 'NAMEi'.
        """
        if encoded_feature_name[:9] != '_encoded_':
            raise ValueError('The input encoded_feature_name must start with "_encoded_"!')
        try:
            ####
            local_encoded_feature_name = encoded_feature_name[9:].replace('ScP', 'PcS')
            # find the maximum number of contiuous K in a string
            nK                         = CorrelationFeatureName.__find_n_character(local_encoded_feature_name, 'K')
            local_encoded_feature_name = CorrelationFeatureName.__replace_K_seismic_phase(local_encoded_feature_name, nK)
            # find the maximum number of contiuous I in a string
            nI                         = CorrelationFeatureName.__find_n_character(local_encoded_feature_name, 'I')
            local_encoded_feature_name = CorrelationFeatureName.__replace_I_seismic_phase(local_encoded_feature_name, nI)
            ####
            names, repeats = list(), list()
            for temp in local_encoded_feature_name.split('+'):
                if '@' in temp:
                    name, repeat = temp.split('@')
                    repeat = int(repeat)
                else:
                    name, repeat = temp, 1
                names.append(name)
                repeats.append(repeat)
            ####
            vol_dict = { it2:0 for it1 in names for it2 in it1.split('-') }
            for name, repeat in zip(names, repeats):
                if '-' not in name:
                    vol_dict[name] = vol_dict[name] + repeat
                else:
                    name1, name2 = name.split('-')
                    vol_dict[name1] = vol_dict[name1] + repeat
                    vol_dict[name2] = vol_dict[name2] - repeat
            ####
            phase1, phase2 = '', ''
            for ph, n in sorted(vol_dict.items() ):
                if n > 0:
                    phase1 = phase1 + ph*n
                elif n < 0:
                    phase2 = phase2 + ph*(-n)
            ####
            if phase1 == '':
                cc_feature_name = '%s*' % phase2
            elif phase2 == '':
                cc_feature_name = '%s*' % phase1
            else:
                cc_feature_name = '%s-%s' % (phase1, phase2)
            return cc_feature_name
        except Exception as err:
            raise ValueError('The input encoded_feature_name is invalid!', encoded_feature_name)
            traceback.print_exc()
    @staticmethod
    def encode(normal_feature_name):
        """
        Encode the normal feature name to an encoded feature name.
        #
        Parameters:
            normal_feature_name: a string of the normal feature name.
                                  It has the format of 'NAME1*' or 'NAME1-NAME2',
                                  where 'NAME1' and 'NAME2' are seismic phase names.
        """
        try:
            local_feature_name = normal_feature_name.replace('ScP', 'PcS')
            # find the maximum number of contiuous K in a string
            nK                         = CorrelationFeatureName.__find_n_character(local_feature_name, 'K')
            local_feature_name = CorrelationFeatureName.__replace_K_seismic_phase(local_feature_name, nK)
            # find the maximum number of contiuous I in a string
            nI                         = CorrelationFeatureName.__find_n_character(local_feature_name, 'I')
            local_feature_name = CorrelationFeatureName.__replace_I_seismic_phase(local_feature_name, nI)
            ####
            seismic_phase_names = list()
            seismic_phase_names.extend( ['PK%sKP' % ('I'*n) for n in range(nI, 0, -1)] )
            seismic_phase_names.extend( ['PK%sKS' % ('I'*n) for n in range(nI, 0, -1)] )
            seismic_phase_names.extend( ['P%sP'   % ('K'*n) for n in range(nK, 0, -1)] )
            seismic_phase_names.extend( ['P%sS'   % ('K'*n) for n in range(nK, 0, -1)] )
            seismic_phase_names.extend( ['PcP', 'PcS', 'ScS'] )
            seismic_phase_names.extend( ['P', 'S'] )  # this must be at the end
            ####
            vol_dict = dict()
            if '-' in local_feature_name:
                phase1, phase2 = local_feature_name.split('-')
            else:
                phase1, phase2 = local_feature_name, ''
            #
            for sph in seismic_phase_names:
                if sph not in vol_dict:
                    vol_dict[sph] = 0
                # get the number of repeat times of sph in phase1
                vol_dict[sph] += phase1.count(sph)
                vol_dict[sph] -= phase2.count(sph)
                phase1 = phase1.replace(sph, '')
                phase2 = phase2.replace(sph, '')
            ####
            temp = list() #'_encoded_'
            for sph, n in sorted(vol_dict.items()):
                if n != 0:
                    temp.append( '%s@%d' % (sph, n) )
            result = '_encoded_' + '+'.join(temp)
            return result
        except Exception as err:
            raise ValueError('The input normal_feature_name is invalid!', normal_feature_name)
            traceback.print_exc()
        pass

@CacheRun('%s/bin/dataset/cc_feature_time.h5' % sacpy.__path__[0] )
# Compute a list of all possible ray parameters, correlation times, inter-receiver distance,... for a correlation feature.
def get_ccs(tau_model1_corrected, tau_model2_corrected, phase1_name, phase2_name, rcvdp1_km, rcvdp2_km,
            threshold_distance=0.1, max_interation=None, enable_h5update=True):
    """
    Compute all possible ray parameters, correlation times, and inter-receiver distance for a correlation feature.
    #
    Parameters:
        tau_model1_corrected: an object a tau_model after correction for the depths of the same source and the 1st receiver.
        tau_model2_corrected: an object a tau_model after correction for the depths of the same source and the 2nd receiver.
                              Can be obtained from `get_corrected_model(...)` by providing source and receiver depths.
        phase1_name:        name of the seismic phase travel from the same source to the 1st receiver.
        phase2_name:        name of the seismic phase travel from the same source to the 2nd receiver.
                            So that the correlation is `phase1_name-phase2_name`!
        rcvdp1_km:          the receiver depth of the 1st receiver in km.
        rcvdp2_km:          the receiver depth of the 2nd receiver in km.
        threshold_distance: the threshold distance for the denser arrivals, so that the successive two arrivals for
                            the 1st seismic phase have a distance difference not larger than this threshold.
                            This may not be the case if `max_interation` is used and the `max_interation` is reached
                            in iterations before the threshold distance is satisfied.
        max_interation:     the maximum number of iterations for the denser arrivals.
        enable_h5update:    A Boolean value to enable the update of the cache file, as this function use `CacheRun` for acceleration.
                            Set `True` (default) to append the results in the cache file, and `False` to not.
    #
    Return: (rps, cc_time, trvt1, trvt2, cc_purist_distance, cc_distance, purist_distance1, purist_distance2)
            NOTE, all the arrays are sorted with respect to the ray parameters!
        rps:                an array of ray parameters in sec/radian.
        cc_time:            an array of correlation time in second.
        trvt1:              an array of travel times for the 1st seismic phase in second.
        trvt2:              an array of travel times for the 2nd seismic phase in second.
        cc_purist_distance: an array of purist distance differences in degree.
        cc_distance:        an array of distance differences in degree (which equals `round_degree_180(cc_purist_distance)`)
        purist_distance1:   an array of purist distances for the 1st seismic phase in degree.
        purist_distance2:   an array of purist distances for the 2nd seismic phase in degree.
    """
    #### load from memory volume
    evdp1_meter = int(round(tau_model1_corrected.source_depth * 1000.0))
    evdp2_meter = int(round(tau_model2_corrected.source_depth * 1000.0))
    rdp1_meter = int(round(rcvdp1_km*1000.0) )
    rdp2_meter = int(round(rcvdp2_km*1000.0) )
    cache_key = 'CC_%s-%s_EV%d-%dRCV%d-%d' % (phase1_name, phase2_name, evdp1_meter, evdp2_meter, rdp1_meter, rdp2_meter)
    # The load_from_cache(...) is added by the CacheRun decorator
    temp = get_ccs.load_from_cache(cache_key) # return None if cannot find anything
    if temp:
        if 'threshold_distance' in temp['attrs']:
            cached_threshold_distance = temp['attrs']['threshold_distance']
            if cached_threshold_distance <= threshold_distance:
                try:
                    rps                = temp['ray_param'][:]
                    cc_time            = temp['cc_time'][:]
                    cc_purist_distance = temp['cc_purist_distance'][:]
                    cc_distance        = temp['cc_distance'][:]
                    trvt1              = temp['trvt1'][:]
                    trvt2              = temp['trvt2'][:]
                    purist_distance1 = temp['purist_distance1'][:]
                    purist_distance2 = temp['purist_distance2'][:]
                    junk = (rps.size, cc_time.size, cc_purist_distance.size, cc_distance.size, trvt1.size, trvt2.size, purist_distance1.size, purist_distance2.size)
                    if len(set(junk)) == 1:
                        return rps, cc_time, trvt1, trvt2, cc_purist_distance, cc_distance, purist_distance1, purist_distance2
                except Exception:
                    pass
    #### Does not exist in the cache. Hence we need to compute the new
    # Step 0 ###################################################################
    phase1 = SeismicPhase(phase1_name, tau_model1_corrected, rcvdp1_km)
    phase2 = SeismicPhase(phase2_name, tau_model2_corrected, rcvdp2_km)
    min_rp = max(phase1.min_ray_param, phase2.min_ray_param)
    max_rp = min(phase1.max_ray_param, phase2.max_ray_param)
    #print(min_rp, max_rp, )
    if min_rp > max_rp:
        return [array([]) for it in range(8) ]
    # Step 1 Init all arrivals #################################################
    rps = list( linspace(min_rp, max_rp, 20) )
    arrs1 = [phase1.shoot_ray(0.0, it) for it in rps]
    arrs2 = [phase2.shoot_ray(0.0, it) for it in rps]

    purist_distance1 = [it.purist_distance for it in arrs1]
    purist_distance2 = [it.purist_distance for it in arrs2]
    trvt1 = [it.time for it in arrs1]
    trvt2 = [it.time for it in arrs2]
    # Step 2 Denser arrivals ###################################################
    while True:
        with Timer(tag='denser arrvials in get_interrcv_ccs(...)', color='red', verbose=__global_verbose ):
            if max_interation:
                max_interation = max_interation-1
                if max_interation < 0:
                    break
            temp = abs( diff(purist_distance1) )
            # +1 to make them the index of right one (not the left one)
            idxs = set( where(temp>threshold_distance)[0] +1 )
            temp = abs( diff(purist_distance2) )
            idxs = idxs.union( where(temp>threshold_distance)[0] +1 )
            if idxs:
                idxs = sorted(idxs, reverse=True) #descending order
                for idx in idxs:
                    mid_rp = ( rps[idx]+rps[idx-1] )*0.5
                    arr1 = phase1.shoot_ray(0.0, mid_rp)
                    arr2 = phase2.shoot_ray(0.0, mid_rp)
                    rps.insert(idx, mid_rp)
                    purist_distance1.insert(idx, arr1.purist_distance)
                    purist_distance2.insert(idx, arr2.purist_distance)
                    trvt1.insert(idx, arr1.time)
                    trvt2.insert(idx, arr2.time)
            else:
                break
    # Step 3 form correlation features #########################################
    # must use the `.purist_distance` which are of the same anti-clockwise direction
    cc_purist_distance = array( [pd1-pd2 for pd1, pd2 in zip(purist_distance1, purist_distance2)] )
    trvt1, trvt2 = array(trvt1), array(trvt2)
    purist_distance1, purist_distance2 = array(purist_distance1), array(purist_distance2)
    cc_time = trvt1-trvt2
    cc_distance = round_degree_180(cc_purist_distance)
    rps = array(rps)
    ## Step 4 update cache #####################################################
    if enable_h5update:
        data_dict = {  'ray_param': rps, 'trvt1': trvt1, 'trvt2': trvt2, 'cc_time': cc_time, 'cc_purist_distance': cc_purist_distance, 'cc_distance': cc_distance, 'purist_distance1': purist_distance1, 'purist_distance2':purist_distance2 }
        attrs_dict = { 'threshold_distance': threshold_distance }
        get_ccs.dump_to_cache(cache_key, data_dict, attrs_dict)
    return rps, cc_time, trvt1, trvt2, cc_purist_distance, cc_distance, purist_distance1, purist_distance2
# Given a list of inter-receiver distances, compute all possible ray parameters and correlation times,... for a correlation feature.
def get_ccs_from_cc_dist(cc_distances_deg, tau_model1_corrected, tau_model2_corrected,
    phase1_name, phase2_name, rcvdp1_km, rcvdp2_km, threshold_distance=0.1, max_interation=None):
    """
    Compute all possible ray parameters and correlation times given a list of inter-receiver distances for a correlation feature.
    This cannot obtain the correlation feature to obtain the result that exactly matches the input `cc_distances_deg`.
    Also, some values within the `cc_distances_deg` could be invalid, and those will be ignored.
    #
    Parameters:
        cc_distances_deg:     a list of inter-receiver distances in degree to search for the correlation features.
        tau_model1_corrected: an object a tau_model after correction for the depths of the same source and the 1st receiver.
        tau_model2_corrected: an object a tau_model after correction for the depths of the same source and the 2nd receiver.
                              Can be obtained from `get_corrected_model(...)` by providing source and receiver depths.
        phase1_name:        name of the seismic phase travel from the same source to the 1st receiver.
        phase2_name:        name of the seismic phase travel from the same source to the 2nd receiver.
                            So that the correlation is `phase1_name-phase2_name`!
        rcvdp1_km:          the receiver depth of the 1st receiver in km.
        rcvdp2_km:          the receiver depth of the 2nd receiver in km.
        threshold_distance: the threshold distance for the denser arrivals, so that the successive two arrivals for
                            the 1st seismic phase have a distance difference not larger than this threshold.
                            This may not be the case if `max_interation` is used and the `max_interation` is reached
                            in iterations before the threshold distance is satisfied.
        max_interation:     the maximum number of iterations for the denser arrivals.
    #
    Return: (rps_found, cct_found, trvt1_found, trvt2_found, cc_purist_distance_found, cc_distance_found, pd1_found, pd2_found)
            Note: the arrays are not sorted, but kind of sorted with respect to the valid values in `cc_distances_deg`.
        rps_found:                an array of ray parameters in sec/radian.
        cct_found:                an array of correlation time in second.
        trvt1_found:              an array of travel times for the 1st seismic phase in second.
        trvt2_found:              an array of travel times for the 2nd seismic phase in second.
        cc_purist_distance_found: an array of purist distance differences in degree.
        cc_distance_found:        an array of distance differences in degree (which equals `round_degree_180(cc_purist_distance)`)
        pd1_found:                an array of purist distances for the 1st seismic phase in degree.
        pd2_found:                an array of purist distances for the 2nd seismic phase in degree.
    """
    rp1, trvt1, purist_distance1, junk = get_arrivals(tau_model1_corrected, phase1_name, rcvdp1_km, None, 0.5)
    rp2, trvt2, purist_distance2, junk = get_arrivals(tau_model2_corrected, phase2_name, rcvdp2_km, None, 0.5)
    min_rp = max(rp1.min(), rp2.min() )
    max_rp = min(rp1.max(), rp2.max() )
    denser_rps = linspace(min_rp, max_rp, 181)
    denser_trvt1, denser_trvt2 = interp(denser_rps, rp1, trvt1), interp(denser_rps, rp2, trvt2)
    denser_pd1, denser_pd2 = interp(denser_rps, rp1, purist_distance1), interp(denser_rps, rp2, purist_distance2)
    denser_pd_diff = denser_pd1-denser_pd2
    n_round360_min, n_round360_max = int(denser_pd_diff.min()//360)-1, int(denser_pd_diff.max()//360)+2
    if n_round360_min > n_round360_max:
        n_round360_min, n_round360_max = n_round360_max, n_round360_min
    ph1 = SeismicPhase(phase1_name, tau_model1_corrected, rcvdp1_km)
    ph2 = SeismicPhase(phase2_name, tau_model2_corrected, rcvdp2_km)
    rps_found, cc_distance_found, cc_purist_distance_found, trvt1_found, trvt2_found = list(), list(), list(), list(), list()
    pd1_found, pd2_found = list(), list()
    for cc_distance in cc_distances_deg:
        for _value in (-cc_distance, cc_distance):
            for overlap in range(n_round360_min, n_round360_max):
                constant_diff = -_value-overlap*360
                temp_dist = denser_pd_diff + constant_diff
                temp_product = temp_dist[:-1]*temp_dist[1:]
                #
                idxs = where(temp_product==0.0)[0]
                if idxs.size>0:
                    if idxs[0]==0:
                        idxs = idxs[::2]
                    else:
                        idxs = idxs[1::2]
                    rps_found.extend( denser_rps[idxs] )
                    cc_distance_found.extend( [cc_distance for it in idxs] )
                    cc_purist_distance_found.extend( denser_pd_diff[idxs] )
                    trvt1_found.extend( denser_trvt1[idxs] )
                    trvt2_found.extend( denser_trvt2[idxs] )
                    pd1_found.extend(denser_pd1)
                    pd2_found.extend(denser_pd2)

                idxs_left = where(temp_product<0.0)[0]
                for idx in idxs_left:
                    rp_left, rp_right = denser_rps[idx-1], denser_rps[idx+1]
                    pd1_left = denser_pd1[idx]
                    pd2_left = denser_pd2[idx]
                    diff_left = pd1_left-pd2_left+constant_diff
                    local_interation = -1
                    # shoot and get better results
                    while True:
                        local_interation = local_interation + 1
                        if max_interation:
                            if local_interation>=max_interation:
                                #print(diff_mid, pd1_mid-pd2_mid, local_interation)
                                break
                        rp_mid = (rp_left+rp_right)*0.5
                        arr = ph1.shoot_ray(0.0, rp_mid); pd1_mid = arr.purist_distance; trvt1_mid = arr.time
                        arr = ph2.shoot_ray(0.0, rp_mid); pd2_mid = arr.purist_distance; trvt2_mid = arr.time
                        diff_mid  = pd1_mid-pd2_mid+constant_diff
                        if abs(diff_mid)<threshold_distance:
                            #print(diff_mid, pd1_mid-pd2_mid, local_interation)
                            break
                        #print(local_interation, (pd2_mid-pd1_mid, diff_mid), (rp_left, rp_right) )
                        if diff_left*diff_mid < 0:
                            rp_right = rp_mid
                        else:
                            rp_left, pd1_left, pd2_left = rp_mid, pd1_mid, pd2_mid
                    rps_found.append(rp_mid)
                    cc_distance_found.append( cc_distance )
                    cc_purist_distance_found.append( pd1_mid-pd2_mid )
                    trvt1_found.append(trvt1_mid)
                    trvt2_found.append(trvt2_mid)
                    pd1_found.append(pd1_mid)
                    pd2_found.append(pd2_mid)
    cc_distance_found = array(cc_distance_found)
    cc_purist_distance_found = array(cc_purist_distance_found)
    rps_found = array(rps_found)
    trvt1_found = array(trvt1_found)
    trvt2_found = array(trvt2_found)
    pd1_found, pd2_found = array(pd1_found), array(pd2_found)
    cct_found = trvt1_found-trvt2_found
    #cct_found = interp(rps_found, rp1, trvt1) - interp(rps_found, rp2, trvt2)
    return rps_found, cct_found, trvt1_found, trvt2_found, cc_purist_distance_found, cc_distance_found, pd1_found, pd2_found
#Return a lot of built-in correlation feature names.
def get_all_cc_names():
    """
    Return a lot of built-in correlation feature names.
    Return:
        a list of strings of the built-in correlation feature
    """
    bw = dict()
    ##### All first order body waves
    bw['direct']         = ('P', 'S', 'PKP', 'PKS', 'SKS', 'PKIKP', 'PKIKS', 'SKIKS', 'PKJKP', 'PKJKS', 'SKJKS', )
    bw['cmb_reflection'] = ('PcP', 'PcS', 'ScS', 'PKKP', 'PKKKP', 'PKKKKP', 'PKKS', 'PKKKS', 'PKKKKS', 'SKKS', 'SKKKS', 'SKKKKS', )
    bw['icb_reflection'] = ('PKiKP', 'PKIIKP', 'PKIIKS', 'SKIIKS')

    ##### All reverberation from the surface for twice
    temp = list(bw['direct'])
    temp.extend(bw['cmb_reflection'] )
    temp.extend(bw['icb_reflection'] )
    bw['reverberation2'] = ['%s%s' %(it, it) for it in temp]
    bw['reverberation3'] = ['%s%s%s' %(it, it, it) for it in temp]

    ##### All cross-terms
    temp.extend(bw['reverberation2'])
    temp.extend(bw['reverberation3'])
    cc_set1 = set( temp )# type X* where X is a seismic phase
    cc_set2 = set()      # type Z* where X-Y=Z*
    cc_set3 = set()      # type X-Y where X and Y are seismic phases
    for idx1, ph1 in enumerate(temp):
        for idx2, ph2 in enumerate(temp):
            if ph1 == ph2:
                continue
            if ph1 in ph2:
                cc_set2.add( ph2.replace(ph1, '', 1) )
            elif ph2 in ph1:
                cc_set2.add( ph1.replace(ph2, '', 1) )
            else:
                cc_set3.add( '%s-%s' % (ph1, ph2) )
    ###### remove 'cc', 'ccc', 'cccc', ....
    for n in range(2, 5):
        junk = 'c'*n
        cc_set2 = set([it.replace(junk, 'c') for it in cc_set2])
    cc_set2 = set(valid_seismic_phases(cc_set2) )
    ###### remove the inverted duplications, e.g., SKP and PKS
    temp = set()
    for it in cc_set2:
        if (it not in temp) and (it[::-1] not in temp):
            temp.add(it)
    cc_set2 = temp
    ###### The result
    ccs = sorted( ['%s*' % it for it in cc_set1.union(cc_set2) ] )
    ccs.extend(cc_set3)
    ccs.extend( ('SSSS', 'PcPPcS', 'ScSScP', 'PKPPKS', 'PKPSKS', 'PKSSKS', 'PKPScS',
                 'ScSPcS-PcPPcP', 'PKPPKS-ScS', 'PKPPKS-PcS', 'PKPPKS-PcP') )
    return sorted(ccs)
#Compute a list of inter-receiver correlation feature dataset for their ray parameters, correlation times, inter-receiver distances and ...
def get_all_interrcv_ccs(cc_feature_names=None, evdp_km=25.0, model_name='ak135', log=sys.stdout, selection_ratio=None, save_to_pickle_file=None,
                         rcvdp1_km=0.0, rcvdp2_km=0.0):
    """
    Return a list of inter-receiver correlation feature dataset for their ray parameters, correlation times, inter-receiver distances and ...
    #
    Parameters:
        cc_feature_names:    a list of strings of the correlation feature names.
                             If `None` (default), all the built-in correlation features will be obtained from `get_all_cc_names(...)`.
                             A correlation feature name usually is of two types"
                                1. 'X*' or 'X', where 'X' is a seismic phase name,
                                    where the 'X' is from a virtual source at the first receiver (at depth `rcvdp1_km`)
                                    and to a virtual receiver at the second receiver (at depth `rcvdp2_km`).
                                    In this case, the provided `evdp_km` in fact is not useful!
                                2. 'X-Y', where 'X' and 'Y' are different seismic phase names,
                                    where the 'X' is from the same source (at depth `evdp_km`) and to the first
                                    receiver (at depth `rcvdp1_km`), and the 'Y' is from the same source (at
                                    depth `evdp_km`) and to the second receiver (at depth `rcvdp2_km`).
                                    In this case, the provided `evdp_km` is useful!
        evdp_km:             the depth (in km) of the same source for forming the inter-receiver correlation features.
        model_name:          the name of the seismic model.
        log:                 the file object to write the log information.
        selection_ratio:     a float number between 0 and 1 to select a subset of the `cc_feature_names`.
        save_to_pickle_file: the file path to save the results in a pickle file.
        rcvdp1_km:           the receiver depth of the 1st receiver in km (default is 0.0).
        rcvdp2_km:           the receiver depth of the 2nd receiver in km (default is 0.0).

        #Note: currently, the `rcvdp1_km` and `rcvdp2_km` must be 0.0.
    #
    Return: results
        results: a list of dictionary objects, each dictionary object contains the following keys:
                 'name':             the name of the correlation feature.
                 'ray_param':        an array of ray parameters in sec/radian.
                 'time':             an array of correlation times in second.
                 'purist_distance':  an array of purist distances in degree.
                 'distance':         an array of distances in degree (the `round_degree_180(purist_distances
    """
    #############################################################################################
    if rcvdp1_km!=0.0 or rcvdp2_km!=0.0:
        msg = 'Currently, the `rcvdp1_km` and `rcvdp2_km` must be 0.0. They are forced to be 0.0 from {}, {}'.format(rcvdp1_km, rcvdp2_km)
        warnings.warn(msg)
    verbose = True if log else False
    modc_seismic_phase = get_corrected_model( taup.TauPyModel(model_name).model, evdp_km=rcvdp1_km, rcvdp_km=rcvdp2_km) # for 'X*'
    modc1 = get_corrected_model( taup.TauPyModel(model_name).model, evdp_km=evdp_km, rcvdp_km=rcvdp1_km) # for 'X-Y'
    modc2 = get_corrected_model( taup.TauPyModel(model_name).model, evdp_km=evdp_km, rcvdp_km=rcvdp2_km)
    if cc_feature_names == None:
        cc_feature_names = get_all_cc_names()
    local_cc_feature_names = cc_feature_names
    if selection_ratio:
        step = int(len(cc_feature_names)*selection_ratio)
        local_cc_feature_names = cc_feature_names[::step]
    results = list()
    n = len(local_cc_feature_names)
    for idx, feature_name in enumerate(local_cc_feature_names):
        tag = '[%d/%d]%s' % (idx+1, n, feature_name)
        with Timer(tag=tag, verbose=verbose, file=log):
            ak = None
            if '*' in feature_name: # for 'X*'
                phase = feature_name[:-1]
                rps, trvt, purist_distance, distance = get_arrivals(modc_seismic_phase, phase, rcvdp2_km, None,
                                                                    0.5, 20, enable_h5update=True)
                if rps.size>0:
                    ak = rps, trvt, purist_distance, distance
            elif '-' in feature_name: # for 'X-Y'
                phase1, phase2= feature_name.split('-')
                tmp = get_ccs(modc1, modc2, phase1, phase2, rcvdp1_km, rcvdp2_km,
                              0.5, 20, True)
                rps, cc_time, trvt1, trvt2, cc_purist_distance, cc_distance, purist_distance1, purist_distance2  = tmp
                if rps.size>0:
                    ak = rps, cc_time, cc_purist_distance, cc_distance 
            if ak:
                rps, time, pdist, dist = ak
                results.append( {   'name': feature_name, 'ray_param': rps, 'time': time, 
                                    'purist_distance': pdist, 'distance': dist } )
    #############################################################################################
    if save_to_pickle_file:
        direc = '/'.join(save_to_pickle_file.split('/')[:-1])
        if not os.path.exists(direc):
            os.makedirs(direc)
        with open(save_to_pickle_file, 'wb') as fid:
            pickle.dump(results, fid)
    #############################################################################################
    #if True:
    #    n = len(results)
    #    cmap = matplotlib.cm.get_cmap('Spectral')
    #    fig, ax = plt.subplots(1, 1, figsize=(6, 11) )
    #    for idx, (tmp, nm) in enumerate(zip(results, local_cc_feature_names)):
    #        (rps, cc_time, cc_purist_distance, cc_distance) = tmp['ray_param'], tmp['time'], tmp['purist_distance'], tmp['distance']
    #        clr = cmap(idx/n)
    #        ax.plot(cc_distance, abs(cc_time), label=nm, color=clr)
    #    ax.set_ylim((0, 3000))
    #    plt.show()
    return results



######################################################################################################################################################################
# Dealing with arrivals
######################################################################################################################################################################
# Get a list of dense-distance arrivals sorted with respect to ray parameter for a given ray parameter range
@CacheRun('%s/bin/dataset/cc_feature_time.h5' % sacpy.__path__[0] )
def get_arrivals(tau_model_corrected, phase_name, rcvdp_km, ray_param_range=None, threshold_distance=0.3, max_interation=None,
                 enable_h5update=True):
    """
    Compute a list of seismic arrival data sorted with respect to ray parameter for a given ray parameter range.
    #
    Parameters:
        tau_model_corrected: an object a tau_model after correction for certain source and receiver depth,
                             which could be obtained from `get_corrected_model(...)`.
        phase_name:          the seismic phase name.
        rcvdp_km:            the receiver depth in km.
        ray_param_range:     the range of ray parameter in sec/radians. A tuple of two floating numbers.
                             If the range is `None` (default) or outside the valid range,
                             the valid ray parameter minimum and maximum for this phase will be used.
        threshold_distance:  the threshold distance for the denser arrivals, so that the successive
                             two arrivals (sorted with respect to ray parameter) will have a distance
                             difference not larger than this threshold.
                             This may not be the case if `max_interation` is used and the `max_interation`
                             is reached in iterations before the threshold distance is satisfied.
        max_interation:      the maximum number of iterations for the denser arrivals.
        enable_h5update:     A Boolean value to enable the update of the cache file, as this function
                             use `CacheRun` for acceleration. Set `True` (default) to append the results
                             in the cache file, and `False` to not.
    #
    Return: (ray_params, trvts, purist_distances, distances)
        ray_params:         a ndarray of ray parameters in sec/radian.
        trvts:              a ndarray of travel times in second.
        purist_distances:   a ndarray of purist distances in degree.
        distances:          a ndarray of distances in degree. (the `round_degree_180(purist_distances)` )
                            All the arrays are sorted with respect to the ray parameters.
    """
    #### load from memory volume
    evdp_meter = int(round(tau_model_corrected.source_depth * 1000.0))
    rdp_meter = int(round(rcvdp_km*1000.0) )
    cache_key = 'SP%s_EV%d_RCV%d' % (phase_name, evdp_meter, rdp_meter)
    # The load_from_cache(...) is added by the CacheRun decorator
    temp = get_arrivals.load_from_cache(cache_key) # return None if cannot find anything
    if temp:
        cached_threshold_distance = temp['attrs']['threshold_distance']
        if cached_threshold_distance <= threshold_distance:
            rps             = temp['ray_param'][:]
            trvt            = temp['time'][:]
            purist_distance = temp['purist_distance'][:]
            distance        = temp['distance'][:]
            return rps, trvt, purist_distance, distance
    #### Does not exist in the cache. Hence we need to compute the new
    # Step 0 ###################################################################
    phase = SeismicPhase(phase_name, tau_model_corrected, rcvdp_km)
    try:
        min_ray_param, max_ray_param = ray_param_range
    except Exception:
        min_ray_param, max_ray_param = phase.min_ray_param, phase.max_ray_param
    min_ray_param, max_ray_param = float(min_ray_param), float(max_ray_param)
    # Step 1 Init all arrivals #################################################
    arrs = [ phase.shoot_ray(0.0, min_ray_param), phase.shoot_ray(0.0, max_ray_param) ]
    purist_distance = [it.purist_distance for it in arrs]
    trvt = [it.time for it in arrs]
    rps = [min_ray_param, max_ray_param]
    # Step 2 Denser arrivals ###################################################
    while True:
        with Timer(tag='denser arrivals in get_arrivals(...)', color='red', verbose=__global_verbose ):
            if max_interation:
                max_interation = max_interation-1
                if max_interation < 0:
                    break
            temp = abs( diff(purist_distance) )
            # +1 to make them the index of right one (not the left one)
            idxs = where(temp>threshold_distance)[0][::-1] +1
            if idxs.size>0:
                for idx in idxs:
                    mid_rp = ( rps[idx]+rps[idx-1] )*0.5
                    mid_arr = phase.shoot_ray(0.0, mid_rp)
                    rps.insert(idx, mid_rp)
                    purist_distance.insert(idx, mid_arr.purist_distance)
                    trvt.insert(idx, mid_arr.time)
            else:
                break
    # Step 3 post jobs ########################################################
    purist_distance = array(purist_distance)
    distance = round_degree_180(purist_distance)
    trvt = array(trvt)
    rps = array(rps)
    # Step 4 update cache #####################################################
    # The dump_to_cache(...) is added by the CacheRun decorator
    if enable_h5update:
        data_dict  = { 'ray_param': rps, 'time': trvt, 'purist_distance': purist_distance, 'distance': distance }
        attrs_dict = { 'threshold_distance': threshold_distance }
        get_arrivals.dump_to_cache(cache_key, data_dict, attrs_dict)
    ###########################################################################
    return deepcopy((rps, trvt, purist_distance, distance))
# Get an arrival given a ray parameter
def get_arrival_from_ray_param(tau_model_corrected, phase_name, ray_param, rcvdp_km):
    """
    Return an `Arrival` object given a ray parameter.
    #
    Parameters:
        tau_model_corrected: an object a tau_model after correction for certain source and receiver depth,
                             which could be obtained from `get_corrected_model(...)`.
        phase_name:          the seismic phase name.
        ray_param:           the ray parameter in sec/radians.
        rcvdp_km:            the receiver depth in km.
    #
    Return:
        An `Arrival` object. Its `distance` attribute is assigned to `round_degree_180(purist_distance)`
    """
    phase = SeismicPhase(phase_name, tau_model_corrected, rcvdp_km)
    arr = phase.shoot_ray(0.0, ray_param) # Arrival not Arrivals
    arr.distance = round_degree_180(arr.purist_distance)
    return arr
# Return a list of unduplicated valid seismic phases
@CacheRun(None)
def valid_seismic_phases(phase_names):
    """
    Check if a set of seismic `phase_name` are valid or not.

    Return the valid set of seismic phase.
    """
    cache_key = 'Valid_seismic_phases'
    temp = valid_seismic_phases.load_from_cache(cache_key)
    temp = temp if temp else dict()
    ######################################################
    valid_phases = [it for it in phase_names if it in temp]
    ######################################################
    reminders = [it for it in phase_names if it not in valid_phases]
    if len(reminders)<=0:
        return sorted(set(valid_phases) ) # no need to update the cache
    else:
        mod = taup.TauPyModel('ak135')
        new_valid_phase_set = set()
        for it in reminders:
            exist = True
            # ask obspy
            try:
                junk = SeismicPhase(it, mod.model)
            except Exception:
                exist = False
            # test if name start/end with P or S
            if exist:
                if (it[0] not in 'psPS') or (it[-1] not in 'PS'):
                    exist = False
            # test
            if exist:
                if ('KPK' in it) or ('KSK' in it) or ('IKI' in it):
                    exist = False
            if exist:
                new_valid_phase_set.add(it)
    ###################################################### 
    if new_valid_phase_set:
        valid_phases.extend( new_valid_phase_set )
        for it in new_valid_phase_set:
            temp[it] = 1
        valid_seismic_phases.dump_to_cache(cache_key, temp)
    ######################################################
    return sorted(set(valid_phases))


if __name__ == "__main__":
    __global_verbose= False
    if False:
        with Timer(tag='main', verbose=__global_verbose):
            evdp_km = 50
            r1dp_km, r2dp_km = 0.1, 0.2
            ph1, ph2 = 'PKIKPPKIKP', 'PKJKP'
            with Timer(tag='correct model', verbose=__global_verbose):
                mod = taup.TauPyModel('ak135').model
                modc1 = get_corrected_model(mod, evdp_km, r1dp_km)
                modc2 = get_corrected_model(mod, evdp_km, r2dp_km)
            with Timer(tag='test1 get_ccs(...)', verbose=__global_verbose):
                rps, cc_time, cc_purist_distance, cc_distance = get_ccs(modc1, modc2, ph1, ph2, r1dp_km, r2dp_km, 1.0, None)
            with Timer(tag='test2 get_ccs(...)', verbose=__global_verbose):
                rps, cc_time, cc_purist_distance, cc_distance = get_ccs(modc1, modc2, ph1, ph2, r1dp_km, r2dp_km, 1.0, None)
            plt.plot(cc_distance, cc_time)
            plt.show()
    if False:
        with Timer(tag='main', verbose=__global_verbose):
            ph1 = 'P'
            with Timer(tag='test1 get_arrivals(...)', verbose=__global_verbose):
                ray_params, trvts, purist_distances, distances = get_arrivals(modc1, ph1, r1dp_km, None, 0.3, None)
                print(sum(distances))
                distances += 10
                print(sum(distances))
            with Timer(tag='test2 get_arrivals(...)', verbose=__global_verbose):
                ray_params, trvts, purist_distances, distances = get_arrivals(modc1, ph1, r1dp_km, None, 0.3, max_interation=None)
                print(sum(distances))
            #plt.plot(distances, trvts)
            #plt.show()
            with CacheRun('junssssk.h5') as cache:
                cache.dump_to_cache('test', {'a':[1,2,3,4]}, {'who': 'sw'})
                pass
    if False:
        with Timer(tag='main', verbose=__global_verbose):
            evdp_km = 50
            r1dp_km, r2dp_km = 0.1, 0.2
            ph1, ph2 = 'PcS', 'PcP'
            with Timer(tag='correct model', verbose=__global_verbose):
                mod = taup.TauPyModel('ak135').model
                modc1 = get_corrected_model(mod, evdp_km, r1dp_km)
                modc2 = get_corrected_model(mod, evdp_km, r2dp_km)
            with Timer(tag='test1 get_ccs(...)', verbose=__global_verbose):
                rps, cc_time, trv1, trv2, cc_purist_distance, cc_distance, purist_distance1, purist_distance2 = get_ccs(modc1, modc2, ph1, ph2, r1dp_km, r2dp_km, 1.0, None)
            with Timer(tag='test2 get_ccs(...)', verbose=__global_verbose):
                rps, cc_time, trv1, trv2, cc_purist_distance, cc_distance, purist_distance1, purist_distance2 = get_ccs(modc1, modc2, ph1, ph2, r1dp_km, r2dp_km, 1.0, None)
            plt.plot(cc_distance, cc_time)
            with Timer(tag='test1 get_ccs_from_cc_dist', verbose=__global_verbose):
                cc_distances = arange(5)
                rps, cc_time, trv1, trv2, cc_purist_distance, cc_distance, purist_distance1, purist_distance2 = get_ccs_from_cc_dist(cc_distances, modc1, modc2, ph1, ph2, r1dp_km, r2dp_km, 0.01, 50)

            plt.plot(cc_distance, cc_time, 'o')
            plt.plot(round_degree_180(cc_purist_distance), cc_time, 'x')
            print(cc_distance)
            print(round_degree_180(cc_purist_distance))
            plt.show()
    if False:
        tmp = valid_seismic_phases(['Pc'])
        print(tmp)
        ccs = get_cc_names()
        print(ccs, len(ccs) )
    if False:
        get_all_interrcv_ccs(log=sys.stdout, selection_ratio=0.005)
    if False:
        __example_use_SeismicPhase_shoot_ray()
        get_corrected_model(taup.TauPyModel('ak135').model, 0, 0)
    if False:
        results = get_all_interrcv_ccs(['P*', 'PcP*'], evdp_km=50, model_name='ak135', log=sys.stdout, rcvdp1_km=0.0, rcvdp2_km=0.0)
        for it in results:
            name            = it['name']
            ray_param       = it['ray_param']
            time            = it['time']
            purist_distance = it['purist_distance']
            distance        = it['distance']
            plt.plot(distance, time, label=name)
        plt.show()
    if True:
        x = 'PcP-PcSSSSPPPPPKKKSSKKP'
        print(x)
        x = CorrelationFeatureName.encode(x)
        print(x)
        x = CorrelationFeatureName.decode(x)
        print(x)
    pass