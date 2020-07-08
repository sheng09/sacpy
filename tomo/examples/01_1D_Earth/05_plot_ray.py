#!/usr/bin/env python3

from sacpy.tomo import plot_tomo as PT
from mayavi import mlab
import h5py
import matplotlib.cm
import numpy as np

cmap = matplotlib.cm.get_cmap('gist_earth_r')
##
# plot raypath
##
fid = h5py.File("03_workspace/raypath.h5", "r")
nray = fid['raypath'].attrs['size']
for idx, it in enumerate(fid['raypath']):
    if idx % 5 != 0:
        continue
    grp = fid['raypath'][it]
    id  = grp.attrs['id']
    lon = grp['lon'][:]
    lat = grp['lat'][:]
    depth = grp['depth'][:]
    r, g, b, a = cmap(idx/nray )
    PT.plot_ray_path(lon, lat, depth, color=(r, g, b) )
##
# plot ev and sta
##
evdp = fid['settings/evdp'][:]
evla = fid['settings/evla'][:]
evlo = fid['settings/evlo'][:]
stla = fid['settings/stla'][:]
stlo = fid['settings/stlo'][:]

ev =list(set( [(dp, la, lo) for dp, la, lo in zip(evdp, evla, evlo)] ))
st =list(set( [(la, lo) for la, lo in zip(stla, stlo)] ))

evdp = np.array( [it[0] for it in ev] )
evla = np.array( [it[1] for it in ev] )
evlo = np.array( [it[2] for it in ev] )
stla = np.array( [it[0] for it in st] )
stlo = np.array( [it[1] for it in st] )
stdp = np.zeros(stlo.size)

PT.plot_point_sphere(evlo, evla, evdp, color=(1, 0, 0 ) )
PT.plot_point_sphere(stlo, stla, stdp, color=(0, 1, 0 ) )

fid.close()

PT.plot_Earth_pretty()

mlab.show()

#fnm = 'earth1d.txt'
#PT.plot_absolute_earth_model(fnm)
#mlab.view(0.0, 55.0 )
#mlab.show()


