#!/usr/bin/env python3

import obspy.taup as taup
from copy import deepcopy
import numpy as np
import bisect

class earthmod3d:
    def __init__(self, mod1d_nm = 'ak135', thete_start = 0.0):
        """
        """
        self.mod1d = taup.TauPyModel(mod1d_nm)
        self.max_radius = self.mod1d.model.radius_of_planet
        self.theta_start = thete_start
        self.raypath = []
    ### 
    def init_grd(self, lons, lats, depth):
        # all angles are in degree, and distances are in km
        ### theta-phi-r coordinate
        rs, thetas, phis  = self.geo2sph(lons, lats, depth)
        ### grid
        self.grd = { 'lon':   np.array(lons), 
                     'lat':   np.array(lats), 
                     'depth': np.array(depth),
                     'r':     rs,
                     'theta': thetas,
                     'phi':   phis, 
                     'nlon': len(lons), 'nlat': len(lats), 'ndepth': len(depth) }
        ### grid points in lons, lats, rs
        self.pts = {'lon': None, 'lat':   None, 'depth': None,
                    'r':   None, 'theta': None, 'phi':   None,
                    'x':   None, 'y':     None, 'z':     None,
                    'pts': None, 
                    'vp1d':None, 'up1d': None,
                    'vs1d':None, 'us1d': None,
                    'dvp': None, 'dup':  None,
                    'dvs': None, 'dus':  None,
                    'rho1d': None  }
        ###
        self.pts['pts'] = [(lo, la, d) for d in depth for  la in lats for lo in lons ]
        self.pts['lon']  = np.array( [it[0] for it in self.pts['pts'] ], dtype=np.float32 )
        self.pts['lat']  = np.array( [it[1] for it in self.pts['pts'] ], dtype=np.float32 )
        self.pts['depth']= np.array( [it[2] for it in self.pts['pts'] ], dtype=np.float32 )
        self.pts['r'], self.pts['theta'], self.pts['phi'] = self.geo2sph(self.pts['lon'], self.pts['lat'], self.pts['depth'] )
        self.pts['x'], self.pts['y'], self.pts['z'] = self.sph2xyz(self.pts['r'], self.pts['theta'], self.pts['phi'] )
        for key in ['vp1d', 'vs1d', 'up1d', 'us1d', 'dvp', 'dvs', 'dup', 'dus', 'rho1d']:
            self.pts[key] = np.zeros(len(self.pts['pts']), dtype=np.float32)
        ### init 1d model
        for idx, d in enumerate( self.pts['depth'] ):
            for key, key2 in zip('psd', ['vp1d', 'vs1d', 'rho1d'] ):
                ############################
                vel_above, vel_below = None, None
                if d<self.mod1d.model.s_mod.v_mod.min_radius:
                    d = self.mod1d.model.s_mod.v_mod.min_radius
                elif d>self.mod1d.model.s_mod.v_mod.max_radius:
                    d = self.mod1d.model.s_mod.v_mod.max_radius
                try:
                    vel_above = self.mod1d.model.s_mod.v_mod.evaluate_below(d, key)
                except:
                    pass
                try:
                    vel_below = self.mod1d.model.s_mod.v_mod.evaluate_above(d, key)
                except:
                    pass
                ############################
                if vel_above == vel_below:
                    self.pts[key2][idx] = vel_above
                elif vel_above != None:
                    self.pts[key2][idx] = vel_above
                elif vel_below != None:
                    self.pts[key2][idx] = vel_below
        self.pts['up1d'] = 1.0/self.pts['vp1d']
        self.pts['us1d'] = np.array( [1.0/it if it>0.01 else 0.0 for it in  self.pts['vs1d']] )
        np.savetxt('tmp.txt', self.pts['us1d'] )
        ###
    def plot(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)
        xs, ys, zs = [self.pts[key] for key in 'xyz']
        vp = self.pts['vp1d']
        sc = ax.scatter(xs, ys, zs, s=1, c=vp, marker='.',  alpha= 0.3, cmap='rainbow' )
        plt.colorbar(sc)
        ###
        for it in self.raypath:
            ax.plot(it['x'], it['y'], it['z'])
            ax.plot(it['x'][:1], it['y'][:1], it['z'][:1], 'o')
            ax.plot(it['x'][-1:], it['y'][-1:], it['z'][-1:], 'o')
        ###
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.axis('off')
        plt.show()
    ###
    def add_single_raypath(self, raypath3d_obj, tags):
        lons, lats, depth = raypath3d_obj.raypath['lon'], raypath3d_obj.raypath['lat'], raypath3d_obj.raypath['depth']
        rs, thetas, phis  = self.geo2sph(lons, lats, depth)
        x, y, z = self.sph2xyz(rs, thetas, phis)
        print(lons[0], lats[0], depth[0], rs[0], thetas[0], phis[0], x[0], y[0], z[0])
        #print(lons[-1], lats[-1], depth[-1], rs[-1], thetas[-1], phis[-1], x[-1], y[-1], z[-1])
        self.raypath.append(
            {'lon': lons, 'lat': lats, 'depth': depth, 
             'r': rs, 'theta': thetas, 'phi': phis, 
             'x': x, 'y': y, 'z': z,
             'tag': tags }
        )
    ###
    def travel_time(self, raypath3d_obj):
        """
        """
        lons, lats, depth = raypath3d_obj.raypath['lon'], raypath3d_obj.raypath['lat'], raypath3d_obj.raypath['depth']
        for lo, la, d in zip(lons, lats, depth):
            i0  = bisect.bisect_left(self.grd['lon'], lo)
            j0  = bisect.bisect_left(self.grd['lat'], la)
            k0  = bisect.bisect_left(self.grd['depth'], d)
            i1, j1, k1 = i0+1, j0+1, k0+1
            ###
            func = lambda i, j, k: k*self.grd['nlon']*self.grd['nlat']+j*self.grd['nlon']+k
            idx000 = func(i0, j0, k0)
            idx001 = func(i0, j0, k1)
            idx010 = func(i0, j1, k0)
            idx011 = func(i0, j1, k1)
            idx100 = func(i1, j0, k0)
            idx101 = func(i1, j0, k1)
            idx110 = func(i1, j1, k0)
            idx111 = func(i1, j1, k1)
            
        pass
    ###
    def geo2sph(self, lons, lats, depth ):
        rs = self.max_radius - np.array(depth)
        thetas = np.array( deepcopy(lons) ) - self.theta_start
        phis = 90.0 - np.array(lats)
        return rs, thetas, phis
    @staticmethod
    def sph2xyz(rs, thetas, phis):
        xs = [r*np.sin(p)*np.cos(t) for r, t, p in zip(rs, np.deg2rad(thetas), np.deg2rad(phis) ) ]
        ys = [r*np.sin(p)*np.sin(t) for r, t, p in zip(rs, np.deg2rad(thetas), np.deg2rad(phis) ) ]
        zs = [r*np.cos(p) for r, t, p in zip(rs, np.deg2rad(thetas), np.deg2rad(phis) ) ]
        return np.array(xs), np.array(ys), np.array(zs)


class raypath3d:
    def __init__(self, evdp, evlo, evla, stlo, stla, phase, min_raypara=True, mod1d_nm = 'ak135'):
        self.mod1d = taup.TauPyModel(mod1d_nm)
        self.raypath = dict()
        self.raypath_mod1d(evdp, evlo, evla, stlo, stla, phase, min_raypara= min_raypara )
    def raypath_mod1d(self, evdp, evlo, evla, stlo, stla, phase, min_raypara=True):
        ###
        self.sta   = {'lon': stlo, 'lat': stla}
        self.event = {'lon': evlo, 'lat': evla, 'depth': evdp}
        ###
        arrs = self.mod1d.get_ray_paths_geo(evdp, evla, evlo, stla, stlo, [phase])
        if len(arrs) < 1:
            self.raypath = None
        elif len(arrs) > 1:
            idx = np.argmin( [it.ray_param for it in arrs] )
            self.raypath['lon'], self.raypath['lat'], self.raypath['depth'] = [arrs[idx].path[key] for key in ['lon', 'lat', 'depth'] ]
        else:
            self.raypath['lon'], self.raypath['lat'], self.raypath['depth'] = [arrs[0].path[key] for key in ['lon', 'lat', 'depth'] ]


if __name__ == "__main__":
    ###
    lons = np.arange(-2, 363, 2.0)
    lats = np.arange(92, -93, -2.0)
    depth = [-10, 0.0, 40.0, 410, 660, 2891.50,  5153.50, 6371.0, 6340.0]
    app = earthmod3d(thete_start= 150.0)
    app.init_grd(lons, lats, depth)
    ###
    I04 = raypath3d(36., 155.154007, 46.856998, 151.629303, -30.418301, 'PKIKP'*4 )
    I06 = raypath3d(36., 155.154007, 46.856998, 151.629303, -30.418301, 'PKIKP'*6 )
    I08 = raypath3d(36., 155.154007, 46.856998, 151.629303, -30.418301, 'PKIKP'*8 )
    app.add_single_raypath(I04, '')
    app.add_single_raypath(I06, '')
    #app.add_single_raypath(path8, '')
    ###
    ##app.plot()
