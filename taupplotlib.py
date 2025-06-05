#!/usr/bin/env python3

"""
This provides several auxiliary functions/classed for obspy.taup plot


Plot seismic arrivals for arbitrary geometry

--------------------------------
# obtain arrivals with obspy.taup
>>> import obspy.taup as taup
>>> mod = taup.TauPyModel('ak135')
>>> evlo, evdp = 100.0, 50.0
>>> stlo, stdp = 160.0, 0.0
>>> arr, arr_PcP, arr_ScP = mod.get_ray_paths(evdp, stlo-evlo, ['P', 'PcP', 'ScP'] )
>>> #
>>> # form `geo_arrival` object, and access information
>>> import sacpy.taupplotlib as taupplotlib
>>> geo_arr = taupplotlib.geo_arrival(evdp, evlo, stdp, stlo, arr, mod)
>>> print( geo_arr.ray_param_sec_km )
>>> print( geo_arr.ray_param_sec_degree )
>>> print(arr.ray_param)
>>> print(arr.ray_param_sec_degree)
>>> #
>>> # plot raypaths for P
>>> fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(10, 10) )
>>> lons, rs = geo_arr.get_raypath()
>>> ax.plot(lons, rs, 'k-', alpha=0.8, label='P')
>>> # add arrows to P path
>>> taupplotlib.add_arrow(ax, lons, rs, loc_ys=[5500], color='k' ) # for radius=5500
>>> #
>>> # plot raypaths for PcP
>>> geo_arr_PcP = geo_arrival(evdp, evlo, stdp, stlo, arr_PcP, mod)
>>> lons, rs = geo_arr_PcP.get_raypath()
>>> ax.plot(lons, rs, 'r--', label='PcP')
>>> # add arrows to PcP path
>>> taupplotlib.add_arrow(ax, lons, rs, loc_ratio=(0.3, 0.7), color='r' )
>>> #
>>> # plot raypath segments for ScP
>>> geo_arr_ScP = geo_arrival(evdp, evlo, stdp, stlo, arr_ScP, mod)
>>> for leg, (lons, rs) in geo_arr_ScP.get_split_raypath():
>>>     print(leg)
>>>     ax.plot(lons, rs, label=leg)
>>> #
>>> ax.legend(loc='lower left')
>>> # plot events, receivers and pretty earth
>>> plotEq(ax, mod, [evdp], [evlo], marker='*', markersize=30)
>>> plotStation(ax, mod, [stdp], [stlo], color='C4', markersize=30)
>>> plotPrettyEarth(ax, mod, distlabel=False, mode='vp')
>>> plt.show()

"""

import pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
import obspy.taup as taup
import numpy as np
from copy import deepcopy

def plotPrettyEarth(ax, model, distlabel=False, mode=None):
    """
    Plot Earth with discontinuities.

    ax: axes to plot;
    model: taup.model object;
    distlabel: False/True for degree label;
    mode: can be: None, 'core', 'vp', 'vs', 'density'.
    """
    ax.set_theta_zero_location('E')
    #ax.grid(False)
    radius = model.model.radius_of_planet
    if mode == None:
        discons = model.model.s_mod.v_mod.get_discontinuity_depths()
        ax.set_yticks(radius - discons)
        ax.yaxis.grid(True, color='k', linewidth= 0.5, alpha= 0.5)
        #ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plt.NullFormatter())
    elif mode == 'core':
        discons = radius - model.model.s_mod.v_mod.get_discontinuity_depths()[6:-1]
        r_cmb, r_icb = discons
        circle = pl.Circle((0, 0), radius, transform=ax.transData._b, color='white', linewidth=0, alpha=0.2, zorder=0)
        ax.add_artist(circle)
        circle = pl.Circle((0, 0), r_cmb, transform=ax.transData._b, color='gray', linewidth=0, alpha=0.2, zorder=0)
        ax.add_artist(circle)
        circle = pl.Circle((0, 0), r_icb, transform=ax.transData._b, color='white', linewidth=0, alpha=1, zorder=0)
        ax.add_artist(circle)
    elif mode in ('vp', 'vs', 'density', 'qp', 'qs'):
        layers = model.model.s_mod.v_mod.layers
        rs = model.model.radius_of_planet - layers['bot_depth']
        tmp = {'vp':'bot_p_velocity', 'vs':'bot_s_velocity', 'density':'bot_density', 'qp':'bot_qp', 'qs':'bot_qs'}
        vel = layers[tmp[mode]]
        thetas =  np.radians(np.linspace(0, 360,1440) )

        ys, xs = np.meshgrid(rs, thetas)
        values = np.zeros((thetas.size, rs.size) )
        for idx, v in enumerate(vel):
            values[:,idx] = v
        vmin, vmax = vel.min(), vel.max()
        dv = (vmax-vmin)*0.01
        vmin = vmin-(vmax-vmin)*0.6
        clev = np.arange(vmin, vmax, dv)

        ax.contourf(xs, ys, values, clev, cmap='gray', zorder=0)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_rmax(radius)
    ax.set_rmin(0.0)
    if distlabel:
        ax.set_xticks( np.deg2rad(list( range(0, 360, 30) ) ) )
    ax.xaxis.grid(False)
def plotStation(ax, model, stdp_list, stlo_list, color = '#E5E5E5', markersize=12, alpha= 0.8, zorder=200):
    """
    Plot stations.

    ax: axes to plot;
    model: taup.model object;
    stlo_list, stdp_list: list of longitude (degree) and depth for several stations;
    """
    stlo_list = np.deg2rad(stlo_list)
    radius = model.model.radius_of_planet
    station_radius_list = [ radius-stdp for stdp in stdp_list]
    #arrowprops = dict(arrowstyle='-|>,head_length=0.8,' 'head_width=0.5', color='#C95241', lw=1.5)
    scale = markersize / 12
    arrowstyle='-|>,head_length=%.1f,head_width=%.1f' % (scale, scale*0.6)
    arrowprops = dict(arrowstyle=arrowstyle, color=color, lw=1.2, alpha= alpha)
    for stlo, station_radius in zip(stlo_list, station_radius_list):
        ax.annotate('', xy=(stlo, station_radius), xycoords='data',
                    xytext=(stlo, station_radius + radius * 0.01), textcoords='data',
                    arrowprops=arrowprops, clip_on=False, zorder=zorder)
def plotEq(ax, model, eqdp_list, eqlo_list, marker='*', color ='#FEF200', markersize=12, alpha= 1.0, zorder=200, markeredgewidth=0.5, markeredgecolor='#aaaaaa', ):
    """
    Plot Earthquake.

    ax: axes to plot;
    model: taup.model object;
    eqdp_list, eqlo_list: list of longitude (degree) and depth for several earthquakes.
    """
    radius = model.model.radius_of_planet
    eq_radius = [radius-eqdp for eqdp in eqdp_list]
    eqlo_list = np.deg2rad(eqlo_list)
    ax.plot(eqlo_list, eq_radius,
            marker=marker, color=color, markersize=markersize, zorder=zorder, linestyle='None',
            markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor, alpha= alpha,
            clip_on=False)
def add_arrow(ax, xs, ys, loc_xs=None, loc_ys=None, loc_ratio=None,
                color='C0', headlength=5, headwidth=4, alpha=1.0,
                reverse_arrow=False, zorder=None):
    """
    Add arrows to a line indicated by `xs` and `ys` given the arrow locations
    indicated by either `loc_xs`, or `loc_ys`, or `loc_ratio`.

    The arrows can be adjusted by setting values for `color`, `headlength`,
    `headwidth`, `alpha`, `reverse_arrow`, `zorder`.
    """
    if loc_xs==None and loc_ys==None and loc_ratio==None:
        return
    idxs = list()
    if loc_ratio!=None:
        n = len(xs)
        idxs = [int(it*n) for it in loc_ratio]
    elif loc_xs!=None:
        tmp1 = xs[:-1]
        tmp2 = xs[1:]
        for it_idx, (it_start, it_end) in enumerate(zip(tmp1, tmp2)):
            for it_x in loc_xs:
                if (it_start-it_x)*(it_end-it_x)<=0:
                    idxs.append(it_idx)
    elif loc_ys!=None:
        tmp1 = ys[:-1]
        tmp2 = ys[1:]
        for it_idx, (it_start, it_end) in enumerate(zip(tmp1, tmp2)):
            for it_y in loc_ys:
                if (it_start-it_y)*(it_end-it_y)<=0:
                    idxs.append(it_idx)

    incre1, incre2 = (0, 1) if reverse_arrow else (1, 0)
    for idx in idxs:
        ax.annotate('', xy=(xs[idx+incre1], ys[idx+incre1]), xytext=(xs[idx+incre2], ys[idx+incre2]),
                        arrowprops=dict( headlength=headlength, headwidth=headwidth, color = color, alpha=alpha),
                        zorder=zorder )
def split_raypath(model, phase_name, depths, path):
    """
    Split `path` into segments, each of which correspond to 'P', 'S', 'K', 'I', 'J'.
    =======
    model:      The `TauPyModel` object for getting the raypath.
    phase_name: phase name.
    depths:     An array for depths of the ray path points.
    path:       The path. Can be a tuple of arrays.

    Return:     a list [(legname1, path1), (legname2, path2), ... ]
                Each `pathi` have the same form of the input `path`.
    Example:
        split_raypath(mod, 'PKIKP', deps, (time, distance, p, depth) )
        or,
        split_raypath(mod, 'PKIKP', deps, (lons, lats, deps) )
    """
    depth_cmb = model.model.cmb_depth
    depth_icb = model.model.iocb_depth

    deps = np.array(depths)
    idxs = [0, deps.size]
    idxs.extend(np.where(deps==depth_cmb)[0])
    idxs.extend(np.where(deps==depth_icb)[0])
    idxs.extend(np.where(deps==0.0      )[0])
    idxs = set(idxs)
    idxs = sorted(idxs)

    tmp = phase_name.replace('c', '').replace('i', '')
    segs = list()
    for i1, i2, ray_leg in zip(idxs[:-1], idxs[1:], tmp):
        segment = ray_leg, [dimen[i1:i2+1] for dimen in path]
        segs.append( segment )
    return segs
class geo_arrival:
    """
    geo_arrival
    ============

    This is for processing `taup.tau.Arrival` for specific geometry and specific phase.

    Event-station geometry (via longitude) is considered, while just in 2D.
    """
    def __init__(self, eqdp, eqlo, stdp, stlo, arrival, model):
        """
        eqdp, eqlo: depth(km) and longitude(degree) of earthquake.
        stdp, stlo: depth(km) and longitude(degree) of station.
        arrs: object of taup.tau.Arrivals, which is the return of get_ray_paths(...) in taup
        model: model used in taup.

        Note:
        (`eqdp`, `eqlo`) and (`stdp`, `stlo`) should match settings in `arrs`.
        """
        self.eqdp = eqdp
        self.eqlo = eqlo % 360
        self.eqlo_rad = np.deg2rad( eqlo )
        self.stdp = stdp
        self.stlo = stlo % 360
        self.stlo_rad = np.deg2rad( stlo )

        self.arrival = arrival
        self.model = model
        self.name = arrival.name
        self.distance = arrival.distance
        self.time = arrival.time
        self.ray_param_sec_degree = self.__get_rayp()
        self.ray_param_sec_km = self.__get_rayp(True)
    def get_raypath(self):
        """
        Return the ray paths (lons, rs) from earthquake to station.
        `lons` is in rad, and `rs` in km.
        """
        eqlo = self.eqlo % 360
        stlo = self.stlo % 360
        dist = (stlo - eqlo) % 360
        radius = self.model.model.radius_of_planet

        rs = radius - self.arrival.path['depth']
        lons = deepcopy(self.arrival.path['dist'] ) # don't touch the original `Arrival`
        d_path = np.rad2deg(lons[-1] - lons[0]) % 360
        if (dist > 180 and d_path < 180) or (dist < 180 and d_path > 180):
            lons *= -1.0
        lons += self.eqlo_rad
        return lons, rs
    def get_split_raypath(self):
        """
        Return a list of (ray_leg_name, lons, rs) from earthquake to station for the seismic wave.
        `lons` is in rad, and `rs` in km.
        """
        lons, rs = self.get_raypath()
        radius = self.model.model.radius_of_planet
        depths = radius-rs
        return split_raypath(self.model, self.name, depths, (lons, rs) )
    def __get_rayp(self, sec_km=False):
        """
        Return ray parameter/slowness, in Sec/Deg.
        """
        #wrong way:
        #rp = np.abs(self.arrival.ray_param_sec_degree)
        #if (self.stlo-self.eqlo)%360 > 180:
        #    return -rp
        #return rp
        #
        eqlo = self.eqlo % 360
        stlo = self.stlo % 360
        dist = (stlo - eqlo) % 360
        ###
        rp = -1.0*self.arrival.ray_param_sec_degree
        lons = self.arrival.path['dist']
        d_path = np.rad2deg(lons[-1] - lons[0]) % 360
        if (dist > 180 and d_path < 180) or (dist < 180 and d_path > 180):
            rp = -rp
        if sec_km:
            rp = rp / (np.pi/180.0*self.model.model.radius_of_planet )
        return -rp
    @classmethod
    def benchmark(cls):
        # obtain arrivals with obspy.taup
        import obspy.taup as taup
        mod = taup.TauPyModel('ak135')
        evlo, evdp = 100.0, 50.0
        stlo, stdp = 160.0, 0.0
        arr, arr_PcP = mod.get_ray_paths(evdp, stlo-evlo, ['P', 'PcP'] )[:2]
        # form `geo_arrival` object, and access information
        geo_arr = geo_arrival(evdp, evlo, stdp, stlo, arr, mod)
        print( geo_arr.ray_param_sec_km )
        print( geo_arr.ray_param_sec_degree )
        print(arr.ray_param)
        print(arr.ray_param_sec_degree)
        # plot raypaths for P
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        lons, rs = geo_arr.get_raypath()
        ax.plot(lons, rs, 'k-', alpha=0.8, label='P')
        # plot raypaths for PcP
        geo_arr_PcP = geo_arrival(evdp, evlo, stdp, stlo, arr_PcP, mod)
        lons, rs = geo_arr_PcP.get_raypath()
        ax.plot(lons, rs, 'r--', label='PcP')
        ax.legend(loc='upper right', fontsize=20)
        # plot events, receivers and pretty earth
        plotEq(ax, mod, [evdp], [evlo], marker='*', markersize=20)
        plotStation(ax, mod, [stdp], [stlo], color='C4', markersize=20)
        plotPrettyEarth(ax, mod, distlabel=False, mode='vp')
        plt.show()

if __name__ == '__main__':
    # obtain arrivals with obspy.taup
    import obspy.taup as taup
    mod = taup.TauPyModel('ak135')
    evlo, evdp = 100.0, 50.0
    stlo, stdp = 160.0, 0.0
    arr, arr_PcP, arr_ScP = mod.get_ray_paths(evdp, stlo-evlo, ['P', 'PcP', 'ScSSKiKP'] )
    #
    # form `geo_arrival` object, and access information
    import sacpy.taupplotlib as taupplotlib
    geo_arr = taupplotlib.geo_arrival(evdp, evlo, stdp, stlo, arr, mod)
    print( geo_arr.ray_param_sec_km )
    print( geo_arr.ray_param_sec_degree )
    print(arr.ray_param)
    print(arr.ray_param_sec_degree)
    #
    # plot raypaths for P
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(10, 10) )
    lons, rs = geo_arr.get_raypath()
    ax.plot(lons, rs, 'k-', alpha=0.8, label='P')
    # add arrows to P path
    taupplotlib.add_arrow(ax, lons, rs, loc_ys=[5500], color='k' ) # for radius=5500
    #
    # plot raypaths for PcP
    geo_arr_PcP = geo_arrival(evdp, evlo, stdp, stlo, arr_PcP, mod)
    lons, rs = geo_arr_PcP.get_raypath()
    ax.plot(lons, rs, 'r--', label='PcP')
    # add arrows to PcP path
    taupplotlib.add_arrow(ax, lons, rs, loc_ratio=(0.3, 0.7), color='r' )
    #
    # plot raypath segments for ScP
    geo_arr_ScP = geo_arrival(evdp, evlo, stdp, stlo, arr_ScP, mod)
    for leg, (lons, rs) in geo_arr_ScP.get_split_raypath():
        print(leg)
        ax.plot(lons, rs, label=leg)
    #
    ax.legend(loc='lower left')
    # plot events, receivers and pretty earth
    plotEq(ax, mod, [evdp], [evlo], marker='*', markersize=30)
    plotStation(ax, mod, [stdp], [stlo], color='C4', markersize=30)
    plotPrettyEarth(ax, mod, distlabel=False, mode='vp')
    plt.show()
    #