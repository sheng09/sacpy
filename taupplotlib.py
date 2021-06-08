#!/usr/bin/env python3

"""
This provides several auxiliary functions/classed for obspy.taup plot


Plot seismic arrivals for arbitrary geometry

--------------------------------

>>> # obtain arrivals with obspy.taup
>>> import obspy.taup as taup
>>> mod = taup.TauPyModel('ak135')
>>> evlo, evdp = 100.0, 50.0
>>> stlo, stdp = 160.0, 0.0
>>> arr, arr_PcP = mod.get_ray_paths(evdp, stlo-evlo, ['P', 'PcP'] )[:2]
>>> # form `geo_arrival` object, and access information
>>> geo_arr = geo_arrival(evdp, evlo, stdp, stlo, arr, mod)
>>> print( geo_arr.ray_param_sec_km )
>>> print( geo_arr.ray_param_sec_degree )
>>> print(arr.ray_param)
>>> print(arr.ray_param_sec_degree)
>>> # plot raypaths for P
>>> fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(10, 10) )
>>> lons, rs = geo_arr.get_raypath()
>>> ax.plot(lons, rs, 'k-', alpha=0.8, label='P')
>>> # plot raypaths for PcP
>>> geo_arr_PcP = geo_arrival(evdp, evlo, stdp, stlo, arr_PcP, mod)
>>> lons, rs = geo_arr_PcP.get_raypath()
>>> ax.plot(lons, rs, 'r--', label='PcP')
>>> ax.legend(loc='upper right', fontsize=20)
>>> # plot events, receivers and pretty earth
>>> plotEq(ax, mod, [evdp], [evlo], marker='*', markersize=30)
>>> plotStation(ax, mod, [stdp], [stlo], color='C4', markersize=30)
>>> plotPrettyEarth(ax, mod, distlabel=False, mode='vp')
>>> plt.show()
>>>

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
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_rmax(model.model.radius_of_planet)
    ax.set_rmin(0.0)
    if distlabel:
        ax.set_xticks( np.deg2rad(list( range(0, 360, 30) ) ) )
    ax.xaxis.grid(False)
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
        rs = model.model.radius_of_planet - layers['top_depth']
        tmp = {'vp':'top_p_velocity', 'vs':'top_s_velocity', 'density':'top_density', 'qp':'top_qp', 'qs':'top_qs'}
        vel = layers[tmp[mode]]
        thetas =  np.radians(np.linspace(0, 360,1440) )

        ys, xs = np.meshgrid(rs, thetas)
        values = np.zeros((thetas.size, rs.size) )
        for idx, v in enumerate(vel):
            values[:,idx] = v
        vmin, vmax = vel.min(), vel.max()
        dv = (vmax-vmin)*0.01
        print(vmin, vmax)
        vmin = vmin-(vmax-vmin)*0.6
        clev = np.arange(vmin, vmax, dv)

        ax.contourf(xs, ys, values, clev, cmap='gray', zorder=0)
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
def plotEq(ax, model, eqdp_list, eqlo_list, marker='*', color ='#FEF200', markersize=12, alpha= 1.0, zorder=200):
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
            markeredgewidth=0.5, markeredgecolor='0.3', alpha= alpha,
            clip_on=False)

class geo_arrival:
    """
    geo_arrival
    ============

    This is for processing `taup.tau.Arrival` for specific geometry and specific phase.

    Event-station geometry (via longitude) is considered, while just in 2D.

    Example:

    >>> # obtain arrivals with obspy.taup
    >>> import obspy.taup as taup
    >>> mod = taup.TauPyModel('ak135')
    >>> evlo, evdp = 100.0, 50.0
    >>> stlo, stdp = 160.0, 0.0
    >>> arr, arr_PcP = mod.get_ray_paths(evdp, stlo-evlo, ['P', 'PcP'] )[:2]
    >>> # form `geo_arrival` object, and access information
    >>> geo_arr = geo_arrival(evdp, evlo, stdp, stlo, arr, mod)
    >>> print( geo_arr.ray_param_sec_km )
    >>> print( geo_arr.ray_param_sec_degree )
    >>> print(arr.ray_param)
    >>> print(arr.ray_param_sec_degree)
    >>> # plot raypaths for P
    >>> fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    >>> lons, rs = geo_arr.get_raypath()
    >>> ax.plot(lons, rs, 'k-', alpha=0.8, label='P')
    >>> # plot raypaths for PcP
    >>> geo_arr_PcP = geo_arrival(evdp, evlo, stdp, stlo, arr_PcP, mod)
    >>> lons, rs = geo_arr_PcP.get_raypath()
    >>> ax.plot(lons, rs, 'r--', label='PcP')
    >>> ax.legend(loc='upper right', fontsize=20)
    >>> # plot events, receivers and pretty earth
    >>> plotEq(ax, mod, [evdp], [evlo], marker='*', markersize=20)
    >>> plotStation(ax, mod, [stdp], [stlo], color='C4', markersize=20)
    >>> plotPrettyEarth(ax, mod, distlabel=False, mode='vp')
    >>> plt.show()
    >>>

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
    arr, arr_PcP = mod.get_ray_paths(evdp, stlo-evlo, ['P', 'PcP'] )[:2]
    # form `geo_arrival` object, and access information
    geo_arr = geo_arrival(evdp, evlo, stdp, stlo, arr, mod)
    print( geo_arr.ray_param_sec_km )
    print( geo_arr.ray_param_sec_degree )
    print(arr.ray_param)
    print(arr.ray_param_sec_degree)
    # plot raypaths for P
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(10, 10) )
    lons, rs = geo_arr.get_raypath()
    ax.plot(lons, rs, 'k-', alpha=0.8, label='P')
    # plot raypaths for PcP
    geo_arr_PcP = geo_arrival(evdp, evlo, stdp, stlo, arr_PcP, mod)
    lons, rs = geo_arr_PcP.get_raypath()
    ax.plot(lons, rs, 'r--', label='PcP')
    ax.legend(loc='upper right', fontsize=20)
    # plot events, receivers and pretty earth
    plotEq(ax, mod, [evdp], [evlo], marker='*', color='#FF3322', markersize=30)
    plotStation(ax, mod, [stdp], [stlo], color='#0088FF', markersize=30)
    plotPrettyEarth(ax, mod, distlabel=False, mode='vp')
    plt.show()