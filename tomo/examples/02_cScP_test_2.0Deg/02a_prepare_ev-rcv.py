#!/usr/bin/env python3

from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import sacpy.geomath as geomath
###
#  The plume
###

dlon, dlat = 2.0, 2.0
lon0, lat0 = 30.0, 0.0
fnm_plume_grd = '01_workspace/plume.grd.txt'
tmp = np.array( [it for it in  np.loadtxt(fnm_plume_grd, comments='#') if it[3] < 661.0 ] )
plume_lons, plume_lats, plume_deps, plume_vp_ratio, plume_vs_ratio = tmp[:,1], tmp[:,2], tmp[:,3], tmp[:, 10], tmp[:, 11]
plume_dvp = plume_vp_ratio-1.0
plume_dvs = plume_vs_ratio-1.0

###
#  The event 
###
evlo0 = lon0
evlo_dev = 40
step_evlo = 2.0
evlo = np.arange(evlo0, evlo0+evlo_dev+step_evlo, step_evlo) % 360.0
evlo = np.concatenate( [evlo, evlo+180.0] ) % 360.0
evla = np.zeros(evlo.size )

with open('02_workspace/event.txt', 'w')  as fid:
    for x, y in zip(evlo, evla):
        evid = '%03d%03d' % (x, y)
        print(evid, x, y, file=fid)

###
#  The receivers
###
ngroup = 3
group_size = 10
netwk_diameter= 5
tmp_lon = np.random.uniform(-0.5, 0.5, ngroup*group_size)*netwk_diameter
tmp_lat = np.random.uniform(-0.5, 0.5, ngroup*group_size)*netwk_diameter
func_lon = lambda inet, stlo0 :  tmp_lon[inet*group_size: (inet+1)*group_size] + stlo0
func_lat = lambda inet, stla0 :  tmp_lat[inet*group_size: (inet+1)*group_size] + stla0
# group1 inside plume region
stlo0, stla0 = 30.0, 0.0
stlo = func_lon(0, stlo0)
stla = func_lat(0, stla0)
net1 = (stlo, stla)
# group2
stlo0, stla0 = 50.0, 0.0
stlo = func_lon(1, stlo0)
stla = func_lat(1, stla0)
net2 = (stlo, stla)
# group3
stlo0, stla0 = 10.0, 0.0
stlo = func_lon(2, stlo0)
stla = func_lat(2, stla0)
net3 = (stlo, stla)

with open('02_workspace/rcv.txt', 'w') as fid:
    for net_id, net in enumerate( [net1, net2, net3] ):
        for rcv_id, (lo, la) in enumerate(zip(net[0], net[1] )):
            rcv_idx = '%02d%04d' % (net_id, rcv_id)
            print(rcv_idx, lo, la, file=fid)
###
#  Plot basemap
###
fig, ax = plt.subplots(1, 1, figsize=(12, 8) )
m = Basemap(projection='robin',lon_0=30, ax=ax, resolution='c')
m.drawcoastlines(linewidth= 0.2, color='gray')
#m.fillcontinents(color='white',lake_color='white')
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,120.,30.), linewidth= 0.1, color='gray', labels=[1,0,0,1]  )
m.drawmeridians(np.arange(0.,360.,60.),   linewidth= 0.1, color='gray', labels=[1,0,1,0] )
m.drawmapboundary(fill_color='white')
###
#  plot plume
###
x,y = m(plume_lons, plume_lats)
sc = m.scatter(x, y, c=plume_dvs, s=4, marker='o', vmin=-0.02, vmax=0.0, cmap='YlOrRd_r', zorder=10)
cax = fig.add_axes([0.3, 0.15, 0.43, 0.02] )
cbar = plt.colorbar(sc, cax=cax, orientation='horizontal')
cbar.set_ticks([-0.02,-0.015, -0.01, -0.005, 0.0] )
cbar.ax.set_xlabel('$\Delta V_S / V_S$')
###
#  plot event
###
x,y = m(evlo, evla)
ax.plot(x, y, 'x', color='b', markersize= 2, linewidth=0, alpha= 0.9, zorder= 11)
###
#  plot networks
###
for net in [net1, net2, net3]:
    lo, la = net
    x, y = m(lo, la)
    ax.plot(x, y, '^', color='g', markersize= 2, linewidth=0, alpha= 0.9, zorder= 11)

fignm = '02_workspace/geo.png'
plt.savefig(fignm, bbox_inches = 'tight', pad_inches = 0.02, dpi=600)
plt.close()