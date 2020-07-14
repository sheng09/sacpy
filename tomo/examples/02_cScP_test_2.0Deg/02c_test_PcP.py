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
arrs = None

for epi_dist in np.arange(0.1, 21, 1):
    tmp = mod_ak135.get_ray_paths(0.0, epi_dist, ['PcP'] )
    if arrs == None:
        arrs = tmp
    else:
        arrs.extend(tmp )
arrs_ScS = None
for epi_dist in np.arange(0.1, 21, 1):
    tmp = mod_ak135.get_ray_paths(0.0, epi_dist, ['ScS'] )
    if arrs_ScS == None:
        arrs_ScS = tmp
    else:
        arrs_ScS.extend(tmp )
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios':[2, 1]} )

###
#  Plot ray-paths
###
# ax0.axis('off')
# ax1.axis('off')
# ax1 = fig.add_axes([0.05, 0.1, 0.45, 0.85] )
for it in arrs:
    dist = np.rad2deg(it.path['dist'])
    depth = it.path['depth']
    size = dist.size // 2; dist0 = dist[size]
    dist = dist[size:]-dist0
    depth = depth[size:]
    ax1.plot(dist, depth, 'k', alpha= 1.0)
for it in arrs_ScS:
    dist = np.rad2deg(it.path['dist'])
    depth = it.path['depth']
    size = dist.size // 2; dist0 = dist[size]
    dist = dist[size:]-dist0
    depth = depth[size:]
    ax1.plot(dist, depth, 'r--', alpha= 0.6)

ax1.set_xlabel('Distance ($\degree$)')
ax1.set_ylabel('Depth (km)')
ax1.set_ylim([3000, 0])
###
#  Plot ray-parameters
###
dist = [it.distance*0.5 for it in arrs]
rp   = [it.ray_param_sec_degree for it in arrs]
ax2.plot(dist, rp, 'k', label='P leg')

dist = [it.distance*0.5 for it in arrs_ScS]
rp   = [it.ray_param_sec_degree for it in arrs_ScS]
ax2.plot(dist, rp, 'r--', label='S leg')

ax2.plot([5.0, 5.0], [0.0, 3.0], ':', linewidth=2.0, alpha= 0.6)
for ax in [ax2]:
    ax.legend(loc='upper left')
    ax.set_ylabel('Slowness (s/$\degree$)')
    ax.set_xlabel('Distance $(\degree)$')
ax2.set_xlim([0, 10] )
ax2.set_ylim([0, 2.0] )

for n, ax in enumerate([ax1, ax2]):
        ax.text(-0.1, 1.1, string.ascii_uppercase[n], transform=ax.transAxes, 
                size=20, weight='bold')

plt.tight_layout()

fignm = '02_workspace/PcP_ScP_slowness.png'
plt.savefig(fignm, bbox_inches = 'tight', pad_inches = 0.02 )
plt.close()
