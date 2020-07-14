#!/usr/bin/env python3
import string
import numpy as np
import obspy.taup as taup
import matplotlib.pyplot as plt
import os.path
import pickle

mod_ak135 = taup.TauPyModel('ak135')


def run(phase, verbose= True):
    dist = []
    traveltime = []
    slowness = []
    dist_smallest_rp       = [] # the one of the smallest slowness
    traveltime_smallest_rp = []
    slowness_smallest_rp   = []
    print('Running for %s' % (phase) )
    for epi_dist in range(181):
        arrs = mod_ak135.get_travel_times(0.0, epi_dist, [phase] )
        if len(arrs) > 0:
            for it in arrs:
                dist.append(epi_dist)
                traveltime.append(it.time)
                slowness.append(it.ray_param_sec_degree) # s/degree
            ###
            idx = 0
            if len(arrs) > 1:
                idx = np.argmin( [it.ray_param for it in arrs] )
            it = arrs[idx]
            dist_smallest_rp.append(epi_dist)
            traveltime_smallest_rp.append(it.time)
            slowness_smallest_rp.append(it.ray_param_sec_degree)


    junk = np.argsort(slowness)
    arr_dist        = np.array([dist[it] for it in junk ])
    arr_traveltime  = np.array([traveltime[it] for it in junk ])
    arr_slowness    = np.array([slowness[it] for it in junk ])
    return (arr_dist, arr_traveltime, arr_slowness), (dist_smallest_rp, traveltime_smallest_rp, slowness_smallest_rp)
###
#  Generate dataset for severl body waves, and save into pkl files
#  I_{n}
###
vol = dict()
pkl_fnm = '02_workspace/body-waves.pkl'

if os.path.exists(pkl_fnm): # load existed things
    with open(pkl_fnm, 'rb')  as fid:
        vol = pickle.load(fid)
#del[vol['PKIKP'*27] ]
for nrever in range(4, 30): # generate new things
    #  I_{n}
    phase, phase_name = 'PKIKP'*nrever, '$I_{%02d}$' % (nrever)
    if phase not in vol:
        (dist, time, rp), (dist_v, time_v, rp_v) = run(phase) # xx_v means the most vertical (the smallest slowness) one if there are multiple paths
        vol[phase] = { 'name': phase_name, 'dist': dist, 'time': time, 'slowness': rp,
                                           'dist_v': dist_v, 'time_v': time_v, 'slowness_v': rp_v,  }

with open(pkl_fnm, 'wb') as fid: # save new dataset into file
    pickle.dump(vol, fid)

###
#  plot
###
for hr1, hr2 in zip([1,3,5,7], [3,5,7,9] ):
    sec1, sec2 = hr1*3600.0, hr2*3600.0
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 12) )
    for key in sorted(vol.keys())[::-1]:
        value = vol[key]
        nm = value['name']
        dist, time, rp = value['dist'], value['time'], value['slowness']
        dist_v, time_v, rp_v = value['dist_v'], value['time_v'], value['slowness_v']
        #
        if np.max(time) < sec1 or np.min(time) > sec2:
            continue
        #
        line, = ax1.plot(dist, time, '-', linewidth= 0.5)
        ax1.plot(dist_v, time_v, label=nm, linewidth= 2, color=line.get_color() )
        #
        ax2.plot(dist, rp, '-', linewidth= 0.5, color=line.get_color() )
        ax2.plot(dist_v, rp_v, label=nm, linewidth= 2, color=line.get_color() )
        ax3.plot(dist_v, rp_v, label=nm, linewidth= 2, color=line.get_color() )
    ###
    time_ticks = np.arange(sec1, sec2+900, 900)
    time_ticklabels = time_ticks/3600
    ax1.set_yticks(time_ticks)
    ax1.set_yticklabels(time_ticklabels)
    ax1.set_ylim(sec1, sec2)
    ax2.set_ylim([0, 2.0] )
    ax2.plot([0.0, 180.0], [0.5, 0.5], 'k--')
    ax3.plot([0.0, 180.0], [0.5, 0.5], 'k--')
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim([0, 180])
        ax.set_xlabel('Epicentral distance $(\degree)$')
        ax.legend(loc='lower left')
    ax1.set_ylabel('Travel time (h)')
    ax2.set_ylabel('Slowness (s/$\degree$)')
    ax3.set_ylabel('Slowness (s/$\degree$)')

    for n, ax in enumerate([ax1, ax2, ax3]):
        ax.text(-0.1, 1.1, string.ascii_uppercase[n], transform=ax.transAxes, 
                size=20, weight='bold')

    fignm = '02_workspace/cross-term_%d-%dhr.png' % (hr1, hr2)
    plt.savefig(fignm, bbox_inches = 'tight', pad_inches = 0.02 )
    plt.close()

    
###
#  selected cross-terms
###
fnm = '02_workspace/cross-terms.txt'
with open(fnm, 'w') as fid:
    for nrever in range(6, 20):
        ph1 = 'PKIKP' * (nrever-1) + 'PKIKS'
        ph2 = 'PKIKP' * nrever
        ph1_nm = '$I_{%02d}$PKIKS' % (nrever-1)
        ph2_nm = '$I_{%02d}$' % (nrever)
        print('%s-%s' % (ph1_nm, ph2_nm), ph1, ph2, file= fid )