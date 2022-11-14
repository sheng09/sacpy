#!/usr/bin/env python3

from obspy import taup
from numpy import arange, array, linspace, interp, where, abs, diff, where, round, sum
from sacpy.utils import Timer, CacheRun
from obspy.taup.seismic_phase import SeismicPhase
from matplotlib import pyplot as plt
import sacpy
from copy import deepcopy
import sys, os, os.path, pickle
import matplotlib

__global_verbose= False

def round_degree_360(deg):
    """
    Round degress to be in [0, 360)
    """
    return deg % 360
def round_degree_180(deg):
    """
    Round degress to be in [0, 360), and then angles in [80, 360) will be rounded to 360-angles.
    """
    x = round_degree_360(deg)
    return round_degree_360((x//180)*(-2*x) + x)
def get_corrected_model(reference_model, evdp_km, rcvdp_km):
    """
    reference_model: an object a tau_model.

    e.g.,
        >>> mod = taup.TauPyModel('ak135')
        >>> mod_corr = get_corrected_model(mod.model, evdp_km=310, rcvdp_km=0.3)
    """
    tau_model = reference_model
    tau_model_corrected = tau_model.depth_correct(evdp_km)
    tau_model_corrected.split_branch(rcvdp_km)
    return tau_model_corrected

######################################################################################################################################################################
# Dealing with correlation features
######################################################################################################################################################################
mpi_cache_tag = ''
try:
    mpi_cache_tag = '_mpi_%4d' % int(os.environ['OMPI_COMM_WORLD_RANK'])
except:
    pass
@CacheRun('%s/bin/dataset/cc_feature_time%s.h5' % (sacpy.__path__[0], mpi_cache_tag) )
def get_ccs(tau_model1_corrected, tau_model2_corrected, phase1_name, phase2_name, rcvdp1_km, rcvdp2_km, 
            threshold_distance=0.1, max_interation=None, enable_h5update=True):
    """
    Compute all possible correlation features.

    reference_model: an object a tau_model. (e.g., an object of taup.TauPyModel('ak135') )

    return: new_rps, cc_distance, cc_time
        ew_rps: a list of ray parameters selected from the input ray_params and being
                valid for the two phases.
        cc_distance: a list of correlation distance in degree.
        cc_time: a list of correlation feature time.
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
        with Timer(message='denser arrvials in get_interrcv_ccs(...)', color='red', verbose=__global_verbose ):
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
def get_ccs_from_cc_dist(cc_distances_deg, tau_model1_corrected, tau_model2_corrected,
    phase1_name, phase2_name, rcvdp1_km, rcvdp2_km, threshold_distance=0.1, max_interation=None):
    """
    Get cross-correlation features given interrceiver or intersource distances
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
def get_all_cc_names():
    """
    Get all possible correlation features.
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

    ##### All cross-terms
    temp.extend(bw['reverberation2'])
    cc_set1 = set( temp )# type X* where X is a seismic phase
    cc_set2 = set()      # type Z* where X-Y=Z*
    cc_set3 = set()      # type X-Y where X and Y are seismic phases
    for idx1, ph1 in enumerate(temp):
        for idx2, ph2 in enumerate(temp[idx1+1:]):
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
    return sorted(ccs)
def get_all_interrcv_ccs(cc_feature_names=None, evdp_km=25.0, model_name='ak135', log=sys.stdout, selection_ratio=None, save_to_pickle_file=None):
    """
    """
    #############################################################################################
    verbose = True if log else False
    modc = get_corrected_model( taup.TauPyModel(model_name).model, evdp_km=evdp_km, rcvdp_km=0.0)
    if cc_feature_names == None:
        cc_feature_names = get_all_cc_names()
    local_cc_feature_names = cc_feature_names
    if selection_ratio:
        step = int(len(cc_feature_names)*selection_ratio)
        local_cc_feature_names = cc_feature_names[::step]
    results = list()
    n = len(local_cc_feature_names)
    for idx, feature_name in enumerate(local_cc_feature_names):
        message = '[%d/%d]%s' % (idx+1, n, feature_name)
        with Timer(message=message, verbose=verbose, file=log):
            ak = None
            if '*' in feature_name:
                phase = feature_name[:-1]
                rps, trvt, purist_distance, distance = get_arrivals(modc, phase, 0.0, None, 0.5, 20, True)
                if rps.size>0:
                    ak = rps, trvt, purist_distance, distance
            elif '-' in feature_name:
                phase1, phase2= feature_name.split('-')
                tmp = get_ccs(modc, modc, phase1, phase2, 0.0, 0.0, 0.5, 20)
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
    #if plot:
    #    n = len(results)
    #    cmap = matplotlib.cm.get_cmap('Spectral')
    #    fig, ax = plt.subplots(1, 1, figsize=(6, 11) )
    #    for idx, ((rps, cc_time, cc_purist_distance, cc_distance), nm) in enumerate(zip(results, local_cc_feature_names)):
    #        clr = cmap(idx/n)
    #        ax.plot(cc_distance, abs(cc_time), label=nm, color=clr)
    #    plt.show()
    return results
######################################################################################################################################################################
# Dealing with arrivals
######################################################################################################################################################################
mpi_cache_tag = ''
try:
    mpi_cache_tag = '_mpi_%4d' % int(os.environ['OMPI_COMM_WORLD_RANK'])
except:
    pass
@CacheRun('%s/bin/dataset/cc_feature_time%s.h5' % (sacpy.__path__[0], mpi_cache_tag) )
def get_arrivals(tau_model_corrected, phase_name, rcvdp_km, ray_param_range=None, threshold_distance=0.3, max_interation=None,
                 enable_h5update=True):
    """
    return: ray_params, trvts, purist_distances, distances
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
        with Timer(message='denser arrivals in get_arrivals(...)', color='red', verbose=__global_verbose ):
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
def get_arrival_from_ray_param(ray_param, tau_model_corrected, phase_name, rcvdp_km):
    """
    """
    phase = SeismicPhase(phase_name, tau_model_corrected, rcvdp_km)
    arr = phase.shoot_ray(0.0, ray_param) # Arrival not Arrivals
    arr.distance = round_degree_180(arr.purist_distance)
    return arr
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
        with Timer(message='main', verbose=__global_verbose):
            evdp_km = 50
            r1dp_km, r2dp_km = 0.1, 0.2
            ph1, ph2 = 'PKIKPPKIKP', 'PKJKP'
            with Timer(message='correct model', verbose=__global_verbose):
                mod = taup.TauPyModel('ak135').model
                modc1 = get_corrected_model(mod, evdp_km, r1dp_km)
                modc2 = get_corrected_model(mod, evdp_km, r2dp_km)
            with Timer(message='test1 get_ccs(...)', verbose=__global_verbose):
                rps, cc_time, cc_purist_distance, cc_distance = get_ccs(modc1, modc2, ph1, ph2, r1dp_km, r2dp_km, 1.0, None)
            with Timer(message='test2 get_ccs(...)', verbose=__global_verbose):
                rps, cc_time, cc_purist_distance, cc_distance = get_ccs(modc1, modc2, ph1, ph2, r1dp_km, r2dp_km, 1.0, None)
            plt.plot(cc_distance, cc_time)
            plt.show()
    if False:
        with Timer(message='main', verbose=__global_verbose):
            ph1 = 'P'
            with Timer(message='test1 get_arrivals(...)', verbose=__global_verbose):
                ray_params, trvts, purist_distances, distances = get_arrivals(modc1, ph1, r1dp_km, None, 0.3, None)
                print(sum(distances))
                distances += 10
                print(sum(distances))
            with Timer(message='test2 get_arrivals(...)', verbose=__global_verbose):
                ray_params, trvts, purist_distances, distances = get_arrivals(modc1, ph1, r1dp_km, None, 0.3, max_interation=None)
                print(sum(distances))
            #plt.plot(distances, trvts)
            #plt.show()
            with CacheRun('junssssk.h5') as cache:
                cache.dump_to_cache('test', {'a':[1,2,3,4]}, {'who': 'sw'})
                pass
    if False:
        with Timer(message='main', verbose=__global_verbose):
            evdp_km = 50
            r1dp_km, r2dp_km = 0.1, 0.2
            ph1, ph2 = 'PcS', 'PcP'
            with Timer(message='correct model', verbose=__global_verbose):
                mod = taup.TauPyModel('ak135').model
                modc1 = get_corrected_model(mod, evdp_km, r1dp_km)
                modc2 = get_corrected_model(mod, evdp_km, r2dp_km)
            with Timer(message='test1 get_ccs(...)', verbose=__global_verbose):
                rps, cc_time, trv1, trv2, cc_purist_distance, cc_distance, purist_distance1, purist_distance2 = get_ccs(modc1, modc2, ph1, ph2, r1dp_km, r2dp_km, 1.0, None)
            with Timer(message='test2 get_ccs(...)', verbose=__global_verbose):
                rps, cc_time, trv1, trv2, cc_purist_distance, cc_distance, purist_distance1, purist_distance2 = get_ccs(modc1, modc2, ph1, ph2, r1dp_km, r2dp_km, 1.0, None)
            plt.plot(cc_distance, cc_time)
            with Timer(message='test1 get_ccs_from_cc_dist', verbose=__global_verbose):
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
    if True:
        get_all_interrcv_ccs(log=sys.stdout)
    pass