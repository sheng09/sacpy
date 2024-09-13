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
from .utils import Timer, CacheRun
from obspy.taup.seismic_phase import SeismicPhase
from obspy.taup import taup_create
from matplotlib import pyplot as plt
import sacpy
from copy import deepcopy
import sys, os, os.path, pickle, warnings, traceback
import matplotlib
from sacpy.processing import round_degree_180, round_degree_360
import numpy as np

__global_verbose= False
__default_model_name = 'ak135'

"""
**HOW TO USE obspy's SeismicPhase class**
Here, we rely on the `taup` method to compute ray_param-distance-time relations for cross-correlation features.
So, it is better to understand how obspy's `get_travel_times(...)` works.
Below shows the workflow of the `get_travel_times(...)` method in `obspy.taup`.

First, generate a initial object of `TauModel` (or `TauPyModel.model`)
Then, correct the `TauModel` object with source depth and receiver depth.
Then, form the `SeismicPhase` object with the corrected `TauModel` object and receiver depth.
Then, using shooting methods `SeismicPhase.shoot_ray(...)` by trying different ray parameters
      to get the arrival with its epicentral distance close to the targeted distance.

The code workflow is as follows:
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

"""
Here we show how to create a model by ourself:
>>> import obspy.taup.taup_create as taup_create
>>> taup_create.build_taup_model('path_to/filename.nd', output_folder='out_folder/', verbose=False)
>>> my_mod = taup.TauPyModel('out_folder/filename.npz')
>>> # where the `path_to/filename.nd` is the path to the model file, and `out_folder/` is the output folder.
>>> # Can check `...../lib/python3.10/site-packages/obspy/taup/data/prem.nd` for the format of input `.nd` file
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
# Please use AppCCFeatureNameEncoder.encode(...), AppCCFeatureNameEncoder.decode(...), and AppCCFeatureNameEncoder.encode_leg_count_dict(...)
class AppCCFeatureNameEncoder:
    def __init__(self, debug=False):
        self.debug = debug
    def encode(self, human_feature_name):
        """
        Encode the human readable feature name to a unique string.
        #
        Parameters:
            human_feature_name: a string of the human feature name.
            Note! Currently, do not allow the simplified feature name, such as 'cS-cP', 'K*', 'I2*',...
        #
        Return:
            a string of the encoded feature name.
        """
        #1. Determine if the feature-related seismic phases are
        #   IC: inner-core related (which has I,J,i in the name),
        #   OC: or only outer-core (which has K,c in the name but no I,J,i in the name),
        #   MANTLE: or only mantle related (which has P,S in the name but no I,J,i,K,c in the name).
        feature_type = self.__determine_feature_type(human_feature_name)
        #2. separate all the legs
        leg_count_dict = self.__separate_legs(human_feature_name)
        #3. valid if the legs are valid
        flag_valid = self.__valid_legs(human_feature_name, feature_type, leg_count_dict)
        if not flag_valid:
            raise ValueError('The input human_feature_name is invalid!', human_feature_name)
        #4. encode the leg_count_dict to a string
        encoded = self.__encode_dict_2_str(feature_type, leg_count_dict)
        return encoded
    def __determine_feature_type(self, human_feature_name):
        """
        Return 'IC', 'OC, or 'MANTLE' if all the feature-related seismic phases
        are inner-core related, outer-core related, or mantle related, repsectively.
        """
        feature_type_dict = {'IC': ('IJi', ''), 'OC': ('Kc', 'IJi'), 'MANTLE': ('PS', 'IJiKc') }
        ####
        result = None
        for feature_type, (must_included_legs, must_not_included_legs) in feature_type_dict.items():
            tmp1 = any( [leg in human_feature_name for leg in must_included_legs] )
            tmp2 = all( [(leg not in human_feature_name) for leg in must_not_included_legs] )
            if tmp1 and tmp2:
                result = feature_type
                break
        if result is None:
            raise ValueError('The input human_feature_name is invalid!', human_feature_name)
        if self.debug:
            print('The feature type is:', result, 'for', human_feature_name)
        return result
    def __separate_legs(self, human_feature_name):
        """
        Separate the seismic phase names into different legs.
        Return:
            leg_count_dict:      a dict of leg -> number of legs in the feature name (considering the minus sign)
        """
        if human_feature_name[-1] == '*':
            human_feature_name = human_feature_name[:-1]
        leg_count_dict = {leg:0 for leg in 'IJKPS'} # this count the number of legs in the feature name considering the minus sign
        part1, part2   = human_feature_name.split('-') if ('-' in human_feature_name) else (human_feature_name, '')
        for leg in 'IJKPS':
            leg_count_dict[leg] += part1.count(leg)
            leg_count_dict[leg] -= part2.count(leg)
            if self.debug:
                part1 = part1.replace(leg, '')
                part2 = part2.replace(leg, '')
        ## remove empty items
        #to_remove = [leg for leg, n in leg_count_dict.items() if n == 0]
        #for leg in to_remove:
        #    leg_count_dict.pop(leg)
        if self.debug:
            part1 = part1.replace('i', '').replace('c', '')
            part2 = part2.replace('i', '').replace('c', '')
            print('After separating all `IJKPS` legs and removing `ic`:', part1, part2)
        return leg_count_dict
    def __valid_legs(self, human_feature_name, feature_type, leg_count_dict):
        """
        Check if the legs are valid.
        """
        flag_valid = True
        nP, nS =    leg_count_dict['P'], leg_count_dict['S']
        nK, nI, nJ = leg_count_dict['K'], leg_count_dict['I'], leg_count_dict['J']
        # For IC-related features
        # 1. the number of P+S legs must be even
        # 2. the number of K legs must be even
        if feature_type == 'IC':
            if (nP+nS)%2 != 0 or (nK%2 != 0):
                flag_valid = False
        # For OC-related features
        # 1. the number of P+S legs must be even
        # 2. there should not be any I, J legs and i letters in the feature name
        if feature_type == 'OC':
            if (nP+nS)%2 != 0:
                flag_valid = False
            for letter in 'IJi':
                if letter in human_feature_name:
                    flag_valid = False
                    break
        # For MANTLE-related features
        # 1. there should not be any I, J, K legs and i, c letters in the feature name
        if feature_type == 'MANTLE':
            for letter in 'IJKic':
                if letter in human_feature_name:
                    flag_valid = False
                    break
        ####
        if self.debug:
            msg = 'The feature name is valid' if flag_valid else 'The feature name is invalid'
            print(msg)
        return flag_valid
    def __encode_dict_2_str(self, feature_type, leg_count_dict):
        """
        Encode the feature name based on the feature type and the leg count dict.
        Return a encoded string.
        """
        # if all the number are negative, then we change all of them to positive
        if all([n<=0 for n in leg_count_dict.values()]):
            leg_count_dict = {leg:-n for leg, n in leg_count_dict.items() }
        #
        temp = ['%s@%d' % (leg, n) for leg,n in sorted(leg_count_dict.items() ) if n != 0]
        encoded = '_encoded_%s_%s' % (feature_type, '+'.join(temp) )
        return encoded
    def decode(self, encoded_str):
        """
        Decode the encoded feature name to the human readable feature name.
        #
        Parameters:
            encoded_str: a string of the encoded feature name.
        #
        Return:
            a string of the human readable feature name.
        """
        #1. extract the feature type and leg count dict
        feature_type, leg_count_dict = self.__interpret_encoded_str_2_dict(encoded_str)
        #2. Valid the leg count dict
        effective_legs = [leg for leg, n in leg_count_dict.items() if n!=0]
        flag_valid = self.__valid_legs( ''.join(effective_legs), feature_type, leg_count_dict)
        if not flag_valid:
            raise ValueError('The input encoded string is invalid!', encoded_str)
        #3. decode the leg count dict to the human readable feature name
        decoded = self.__decode_dict_2_str(feature_type, leg_count_dict)
        return decoded
    def __interpret_encoded_str_2_dict(self, encoded_str):
        """
        Valid and interpret the encoded string to the feature type and leg count dict.
        Return a tuple of (feature_type, leg_count_dict).
        """
        if '_encoded_' not in encoded_str:
            raise ValueError('The input encoded_str is invalid! There must be a leading `_encoded_` keyword!', encoded_str)
        feature_type, temp = encoded_str[9:].split('_')
        if feature_type not in ['IC', 'OC', 'MANTLE']:
            raise ValueError('The input encoded_str is invalid! The feature type must be `IC`, `OC` or `MANTLE`!', encoded_str)
        leg_count_dict = {leg:0 for leg in 'IJKPS'} # this count the number of legs in the feature name considering the minus sign
        for seg in temp.split('+'):
            leg, n = seg.split('@') if ('@' in seg) else (seg, 1)
            if leg not in 'IJKPS':
                raise ValueError('The input encoded_str is invalid! The leg name must be `I`, `J`, `K`, `P`, `S`!', encoded_str)
            leg_count_dict[leg] += int(n)
        return feature_type, leg_count_dict
    def __decode_dict_2_str(self, feature_type, leg_count_dict):
        """
        Decode the feature name based on the feature type and the leg count dict.
        Return a human readable string.
        """
        # if all the number are negative, then we change all of them to positive
        if all([n<=0 for n in leg_count_dict.values()]):
            leg_count_dict = {leg:-n for leg, n in leg_count_dict.items() }
        #
        sps = ('PKIKP', 'SKIKS', 'PKIKS', 'SKIKP',
               'PKJKP', 'SKJKS', 'PKJKS', 'SKJKP',
               'PKiKP', 'SKiKS', 'PKiKS', 'SKiKP',
               'PKP', 'SKS', 'PKS', 'SKP',
               'PcP', 'ScS', 'PcS', 'ScP',
               'P', 'S')
        seismic_phase_count_dict = {it:0 for it in sps }
        for leg in 'IJKPS':
            n = leg_count_dict[leg]
            if n == 0:
                continue
            if leg == 'I':  # handle I
                if abs(leg_count_dict['P']) > abs(leg_count_dict['S']):
                    seismic_phase_count_dict['PKIKP'] += n
                    leg_count_dict['I'] -=  n
                    leg_count_dict['K'] -= (n*2)
                    leg_count_dict['P'] -= (n*2)
                else:
                    seismic_phase_count_dict['SKIKS'] += n
                    leg_count_dict['I'] -=  n
                    leg_count_dict['K'] -= (n*2)
                    leg_count_dict['S'] -= (n*2)
            elif leg == 'J': # handle J
                if abs(leg_count_dict['P']) > abs(leg_count_dict['S']):
                    seismic_phase_count_dict['PKJKP'] += n
                    leg_count_dict['J'] -=  n
                    leg_count_dict['K'] -= (n*2)
                    leg_count_dict['P'] -= (n*2)
                else:
                    seismic_phase_count_dict['SKJKS'] += n
                    leg_count_dict['J'] -=  n
                    leg_count_dict['K'] -= (n*2)
                    leg_count_dict['S'] -= (n*2)
            elif leg == 'K': # handle K
                if feature_type == 'IC':
                    if abs(leg_count_dict['P']) > abs(leg_count_dict['S']):
                        seismic_phase_count_dict['PKiKP'] += n//2 # n must be a even as it passed the valid check
                        leg_count_dict['K'] -= ((n//2)*2)
                        leg_count_dict['P'] -= ((n//2)*2)
                    else:
                        seismic_phase_count_dict['SKiKS'] += n//2 # n must be a even as it passed the valid check
                        leg_count_dict['K'] -= ((n//2)*2)
                        leg_count_dict['S'] -= ((n//2)*2)
                elif feature_type == 'OC':
                    if abs(leg_count_dict['P']) > abs(leg_count_dict['S']):
                        seismic_phase_count_dict['PKP'] += n
                        leg_count_dict['K'] -= n
                        leg_count_dict['P'] -= (n*2)
                    else:
                        seismic_phase_count_dict['SKS'] += n
                        leg_count_dict['K'] -= n
                        leg_count_dict['S'] -= (n*2)
                else:
                    raise ValueError('The input feature_type or leg_count_dict are invalid!', feature_type, leg_count_dict)
            elif leg in 'P': # handle P
                if feature_type in ('IC', 'OC'):
                    if n%2 != 0:
                        if n > 0:
                            seismic_phase_count_dict['PcS'] += 1
                            leg_count_dict['P'] -= 1
                            leg_count_dict['S'] -= 1
                        else:
                            seismic_phase_count_dict['PcS'] -= 1
                            leg_count_dict['P'] += 1
                            leg_count_dict['S'] += 1
                        n = leg_count_dict['P'] # updated to even
                    seismic_phase_count_dict['PcP'] += n//2
                    leg_count_dict['P'] -= ((n//2)*2)
                elif feature_type == 'MANTLE':
                    seismic_phase_count_dict['P'] += n
                    leg_count_dict['P'] -= n
                else:
                    raise ValueError('The input feature_type or leg_count_dict are invalid!', feature_type, leg_count_dict)
            elif leg in 'S': # handle S
                if feature_type in ('IC', 'OC'):
                    if n%2 != 0:
                        raise ValueError('The input feature_type or leg_count_dict are invalid!', feature_type, leg_count_dict)
                    seismic_phase_count_dict['ScS'] += n//2
                    leg_count_dict['S'] -= ((n//2)*2)
                elif feature_type == 'MANTLE':
                    seismic_phase_count_dict['S'] += n
                    leg_count_dict['S'] -= n
                else:
                    raise ValueError('The input feature_type or leg_count_dict are invalid!', feature_type, leg_count_dict)
            else:
                raise ValueError('The leg is invalid!', leg)
        ################################################################################################
        part1, part2 = '', ''
        for sp, count in sorted(seismic_phase_count_dict.items() ):
            if count > 0:
                part1 += sp*count
            elif count < 0:
                part2 += sp*(-count)
        result = '%s-%s' % (part1, part2)
        if part2 == '':
            result = '%s*' % part1
        if part1 == '':
            result = '%s*' % part2
        return result
    def encode_leg_count_dict(self, feature_type, leg_count_dict):
        """
        Encode the leg count dict to a string.
        """
        effective_legs = [leg for leg, n in leg_count_dict.items() if n!=0]
        if not self.__valid_legs(effective_legs, feature_type, leg_count_dict):
            raise ValueError('The input leg_count_dict is invalid!', feature_type, leg_count_dict)
        return self.__encode_dict_2_str(feature_type, leg_count_dict)
# Encode a human feature name to a unique string. A wrapper of AppCCFeatureNameEncoder.encode(...)
def encode_cc_feature_name(human_feature_name, debug=False):
    """
    Encode the human feature name to a unique string
    #
    Parameters:
        human_feature_name: a string of the human feature name.
        debug:              a Boolean value to enable the debug information.
    #
    Return:
        a string of the encoded feature name.
    """
    app = AppCCFeatureNameEncoder(debug=debug)
    return app.encode(human_feature_name)
# Encode a leg count dict to a unique string. A wrapper of AppCCFeatureNameEncoder.encode_leg_count_dict(...)
def encode_leg_count_dict(feature_type, leg_count_dict, debug=False):
    """
    Encode the leg count dict to a unique string.
    #
    Parameters:
        feature_type:    a string of the feature type, which is either 'IC', 'OC', or 'MANTLE'.
        leg_count_dict:  a dict of leg -> number of legs in the feature name (considering the minus sign).
        debug:           a Boolean value to enable the debug information.
    #
    Return:
        a string of the encoded feature name.
    """
    app = AppCCFeatureNameEncoder(debug=debug)
    return app.encode_leg_count_dict(feature_type, leg_count_dict)
# Eecode the encoded string to a unique human readable feature name. A wrapper of AppCCFeatureNameEncoder.decode(...)
def decode_cc_feature_name(encoded_string, debug=False):
    """
    Decode the encoded string to the human readable feature name.
    #
    Parameters:
        encoded_feature_name: a string of the encoded feature name.
    #
    Return:
        a string of the human readable feature name.
    """
    app = AppCCFeatureNameEncoder(debug=debug)
    return app.decode(encoded_string)


@CacheRun('%s/bin/dataset/cc_feature_time.h5' % sacpy.__path__[0] )
# Compute a list of all possible ray parameters, correlation times, inter-receiver distance,... for a correlation feature.
def get_ccs(tau_model1_corrected, tau_model2_corrected, phase1_name, phase2_name, rcvdp1_km, rcvdp2_km,
            threshold_distance=0.1, max_interation=None, enable_h5update=True, model_tag=__default_model_name):
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
        model_tag:          The model name tag for the originial model that is used for generating
                            the `tau_model_corrected`. Default is `ak135`.
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
    # cache_key in default is for built-in ak135 model
    if model_tag != __default_model_name:
        cache_key = '%s@%s' % (cache_key, model_tag)
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
    bw['direct']         = ('P', 'S', 'PP', 'SS', 'PPP', 'SSS',
                            'PKP', 'PKS', 'SKS', 'PKPPKP', 'PKPPKS', 'PKPSKS', 'PKSSKS', 'SKSSKS',
                            'PKIKP', 'PKIKS', 'SKIKS', 'PKIKPPKIKP', 'PKIKPPKIKS', 'PKIKPSKIKS', 'PKIKSSKIKS', 'SKIKSSKIKS',
                            'PKJKP', 'PKJKS', 'SKJKS', 'PKJKPPKJKP', 'PKJKPPKJKS', 'PKJKPSKJKS', 'PKJKSSKJKS', 'SKJKSSKJKS', )
    bw['cmb_reflection'] = ('PcP', 'PcS', 'ScS',
                            'PcPPcP', 'PcPPcS', 'PcPScS', 'PcSScS', 'ScSScS',
                            'PcPPcPPcP', 'PcPPcPPcS', 'PcPPcPScS', 'PcPPcSScS', 'PcPScSScS', 'PcSScSScS', 'ScSScSScS',
                            'PcPPcPPcPPcP', 'PcPPcPPcPPcS', 'PcPPcPPcPScS', 'PcPPcPPcSScS', 'PcPPcPScSScS', 'PcPPcSScSScS', 'PcPScSScSScS', 'PcSScSScSScS', 'ScSScSScSScS',
                            'PKKP', 'PKKKP', 'PKKKKP', 'PKKKKKP', #'PKKKKKKP', 'PKKKKKKP',
                            'PKKS', 'PKKKS', 'PKKKKS', 'PKKKKKS', #'PKKKKKKS', 'PKKKKKKS',
                            'SKKS', 'SKKKS', 'SKKKKS', 'SKKKKKS', #'SKKKKKKS', 'SKKKKKKS',
                            )
    bw['icb_reflection'] = ('PKiKP', #'PKiKS', 'SKiKS',
                            #'PKiKPPKiKP', 'PKiKPPKiKS', 'PKiKPSPKiKS', 'PKiKSSPKiKS', 'SKiKSSPKiKS',
                            #'PKiKPPKiKPPKiKP', 'PKiKPPKiKPPKiKS', 'PKiKPPKiKPSKiKS', 'PKiKPPKiKSSPKiKS', 'PKiKPSKiKSSPKiKS', 'PKiKSSPKiKSSPKiKS', 'SKiKSSPKiKSSPKiKS',
                            #'PKIIKP', 'PKIIKS', 'SKIIKS', 'PKIIIKP', 'PKIIIKS', 'SKIIIKS',
                            #'PKJJKP', 'PKJJKS', 'SKJJKS', 'PKJJJKP', 'PKJJJKS', 'SKJJJKS',
                            )

    ##### All reverberation from the surface for twice
    temp = list(bw['direct'])
    temp.extend(bw['cmb_reflection'] )
    temp.extend(bw['icb_reflection'] )
    bw['reverberation2'] = ['%s%s' %(it, it) for it in temp]
    bw['reverberation3'] = ['%s%s%s' %(it, it, it) for it in temp]
    bw['reverberation4'] = ['%s%s%s%s' %(it, it, it, it) for it in temp]
    bw['reverberation5'] = ['%s%s%s%s' %(it, it, it, it) for it in temp]

    ##### All cross-terms
    temp.extend(bw['reverberation2'])
    temp.extend(bw['reverberation3'])
    temp.extend(bw['reverberation4'])
    temp.extend(bw['reverberation5'])
    temp = sorted( set(temp) )

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
def get_all_interrcv_ccs(cc_feature_names=None, evdp_km=25.0, model_name=__default_model_name, log=sys.stdout, selection_ratio=None, save_to_pickle_file=None,
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
        model_name:          the name of the seismic model (default is 'ak135'), or the path/filename of a user-defined model.
                             The user-defined model can be in the format of `.nd` or `.npz`.
                             The `.nd` file is a pure text file, and an example can be `..../site-packages/obspy/taup/data/prem.nd`
                             The `.npz` file is a binary file, which can be generate by `obspy.taup.taup_create.build_taup_model(nd_file)`.
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
    #############################################################################################
    if model_name[-3:] == '.nd':
        taup_create.build_taup_model(model_name, output_folder='./', verbose=False)
        model_tag = model_name.split('/')[-1].replace('.nd', '')
        temp = './%s.npz' % model_tag
        original_tau_model = taup.TauPyModel(temp).model
    elif model_name[-4:] == '.npz':
        original_tau_model = taup.TauPyModel(model_name).model
        model_tag = model_name.split('/')[-1].replace('.npz', '')
    else:
        original_tau_model = taup.TauPyModel(model_name).model
        model_tag = model_name
    #############################################################################################
    modc_seismic_phase = deepcopy(original_tau_model)
    modc1              = deepcopy(original_tau_model)
    modc2              = deepcopy(original_tau_model)
    modc_seismic_phase = get_corrected_model( modc_seismic_phase, evdp_km=rcvdp1_km, rcvdp_km=rcvdp2_km) # for 'X*'
    modc1              = get_corrected_model( modc1,              evdp_km=evdp_km,   rcvdp_km=rcvdp1_km) # for 'X-Y'
    modc2              = get_corrected_model( modc2,              evdp_km=evdp_km,   rcvdp_km=rcvdp2_km)
    if cc_feature_names == None:
        cc_feature_names = get_all_cc_names()
    local_cc_feature_names = cc_feature_names
    if selection_ratio:
        step = int(len(local_cc_feature_names)*selection_ratio)
        local_cc_feature_names = local_cc_feature_names[::step]
    #############################################################################################
    # valid&encode&decode the feature names
    app = AppCCFeatureNameEncoder()
    temp = set()
    for it in local_cc_feature_names:
        try:
            temp.add( app.decode( app.encode(it) ) )
        except:
            pass
    local_cc_feature_names = sorted(temp)
    #############################################################################################
    results = list()
    n = len(local_cc_feature_names)
    for idx, feature_name in enumerate(local_cc_feature_names):
        tag = '[%d/%d]%s' % (idx+1, n, feature_name)
        with Timer(tag=tag, verbose=verbose, file=log):
            ak = None
            if '*' in feature_name: # for 'X*'
                phase = feature_name[:-1]
                rps, trvt, purist_distance, distance = get_arrivals(modc_seismic_phase, phase, rcvdp2_km, None,
                                                                    0.5, 20, enable_h5update=True,
                                                                    model_tag=model_tag)
                if rps.size>0:
                    ak = rps, trvt, purist_distance, distance
            elif '-' in feature_name: # for 'X-Y'
                phase1, phase2= feature_name.split('-')
                tmp = get_ccs(modc1, modc2, phase1, phase2, rcvdp1_km, rcvdp2_km,
                              0.5, 20, True,
                              model_tag=model_tag)
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
                 enable_h5update=True, model_tag = __default_model_name):
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
        model_tag:           the model name tag for the originial model that is used for generating
                             the `tau_model_corrected`. Default is `ak135`.
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
    # cache_key in default is for built-in ak135 model
    if model_tag != __default_model_name:
        cache_key = '%s@%s' % (cache_key, model_tag)
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
        #tmp = get_all_cc_names()
        #print(tmp[:2])
        get_all_interrcv_ccs(log=sys.stdout)
        x = '-PKIKPPKIKPPKIKPPKIKPPKIKPPKIKPPKIKPPcPPcS'
        y = encode_cc_feature_name(x)
        z = decode_cc_feature_name(y)
        print(x)
        print(z)
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
    if False:
        #get_arrival_from_ray_param(taup.TauPyModel('ak135').model, 'P', 0.01, 0)
        x = 'PcSPcSPcSPcSPcS-PcSScSScPPcPPcPPKKKSSKP'
        x = 'SKS-ScS'
        y = encode_cc_feature_name(x)
        z = decode_cc_feature_name(y)
        print(x)
        print(y)
        print(z)
        #
        x = 'PKIKPPKIKP-PKiKP'
        y = encode_cc_feature_name(x)
        z = decode_cc_feature_name(y)
        print(x)
        print(y)
        print(z)
        #####
        lst = get_all_cc_names()
        print(len(lst))
        encodes = set()
        for it in lst:
            try:
                encodes.add( encode_cc_feature_name(it) )
            except Exception:
                pass
        print(len(encodes))
        #####
        all_encodes = set()
        leg_count_dict = {leg:0 for leg in 'IJKPS'}
        nmin, nmax = -5, 6
        for feature_name in ('IC', 'OC', 'MANTLE'):
            for nI in range(nmin, nmax):
                leg_count_dict['I'] = nI
                for nJ in range(nmin, nmax):
                    leg_count_dict['J'] = nJ
                    for nK in range(nmin, nmax):
                        leg_count_dict['K'] = nK
                        for nP in range(nmin, nmax):
                            leg_count_dict['P'] = nP
                            for nS in range(nmin, nmax):
                                leg_count_dict['S'] = nS
                                try:
                                    encoded_str = encode_leg_count_dict(feature_name, leg_count_dict)
                                    all_encodes.add(encoded_str)
                                except Exception:
                                    pass
        print(len(all_encodes))
        for it in sorted(all_encodes):
            try:
                x = decode_cc_feature_name(it)
                #print(it, x )
            except Exception:
                print(it)
                pass
        #####
        x = '_encoded_MANTLE_P@1'
        y = decode_cc_feature_name(x)
        print(y)
    if True:
        x = '_encoded_OC_P@-1+S@1'
        y = decode_cc_feature_name(x)
        print(y)
    pass
    pass