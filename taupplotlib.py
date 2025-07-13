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
def split_pxt_legs(ps, xs, lst_of_lst=None):
    """
    Split the p-x-[l1-l-...] (slowness-distance-[l1-l2-...]) into several legs so that within each leg all x (distance) are monotonic.
    This is for taking care of the multi-pathing phenomenon (different waves travel different paths to the same receiver) (e.g., PKPab and PKPbc).
    #
    ps, xs:     1D arrays of slowness, distance and time. (Do not need to be sorted.)
    lst_of_lst: a list of lists, and this parameter is optional.
                e.g., [l1, l2, l3,...] each li has the same length as ps and xs and will be split into several segments according to the legs.
    #
    Return:     a list of legs: [leg1, leg2,... legi,...], and each legi is a tuple (ps, xs, lst_of_lst) or (ps, xs) if lst_of_lst is None.
    #
    E.g.,
    >>> # values below are meaningless, just for showing non-monotonic xs
    >>> ps = [100, 105, 110, 105, 100]
    >>> xs = [10,   20,  30,  20,  10] # !!! Note here the xs is not monotonic
    >>> ts = [200, 210, 220, 210, 200]
    >>> sth = [None, None, None, None, None] # some other data
    >>> legs = split_pxt_legs(ps, xs, [ts, sth])
    """
    ### check inputs
    if len(ps) != len(xs):
        raise ValueError("ps and xs should have the same length.")
    if lst_of_lst is not None:
        for it in lst_of_lst:
            if len(it) != len(ps):
                raise ValueError("Each list in lst_of_lst should have the same length as ps and xs!")
    ### sort with respect to ps
    idx_sort = np.argsort(ps)[::-1]
    ps = np.array(ps)[idx_sort]
    xs = np.array(xs)[idx_sort]
    if lst_of_lst is not None:
        lst_of_lst = [ [lst[it] for it in idx_sort] for lst in lst_of_lst]
    ### if there are just two data points (which means a single leg), then return a single leg
    if len(ps) == 2:
        if lst_of_lst is not None:
            return [(ps, xs, lst_of_lst)]
        else:
            return [(ps, xs)]
    ### split legs
    dx = np.diff(xs)
    for ind in range(1, len(dx)): # take care of the zeros in dx
        if dx[ind] == 0:
            dx[ind] = dx[ind-1]  # if two xs are the same, use the previous one
    tmp = dx[:-1] * dx[1:]
    idxs_minial_maximal = [0]
    idxs_minial_maximal.extend( np.where( tmp < 0 )[0]+1 )
    idxs_minial_maximal.append(len(xs)-1)
    legs = list()
    for i0, i1 in zip(idxs_minial_maximal[:-1], idxs_minial_maximal[1:]):
        leg_xs = xs[i0:i1+1]
        leg_ps = ps[i0:i1+1]
        if lst_of_lst is not None:
            leg_lst_of_lst = [lst[i0:i1+1] for lst in lst_of_lst]
            legs.append( (leg_ps, leg_xs, leg_lst_of_lst) )
        else:
            legs.append( (leg_ps, leg_xs) )
    return legs

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
    if True:
        mod = taup.TauPyModel('ak135')
        d = np.arange(0, 180.1, 1)
        tmp = [mod.get_ray_paths(0, it, ['PKP'] ) for it in d]
        tmp = [it for it in tmp if it is not None]
        tmp2 = tmp[0]
        for it in tmp[1:]:
            tmp2.extend(it)
        #
        xs = np.array([it.distance for it in tmp2])
        ts = np.array([it.time for it in tmp2])
        ps = np.array([it.ray_param for it in tmp2])
        #
        lst_of_legs = split_pxt_legs(ps, xs, [ts, tmp2])
        #
        fig = plt.figure(figsize=(16, 4))
        grd = fig.add_gridspec(1, 4)
        ax1 = fig.add_subplot(grd[0, 0], projection='polar')
        ax2 = fig.add_subplot(grd[0, 1])
        ax3 = fig.add_subplot(grd[0, 2])
        ax4 = fig.add_subplot(grd[0, 3])
        #
        for idx_leg, leg in enumerate(lst_of_legs[:]):
            clr = mpl.cm.tab10.colors[idx_leg % 10]
            leg_ps, leg_xs, (leg_ts, arrs) = leg
            geo_arrs = [geo_arrival(0, 0, 0, it.purist_distance, it, mod) for it in arrs]
            print(geo_arrs[0].name )
            for it in geo_arrs[::2]:
                lons, rs = it.get_raypath()
                ax1.plot(lons, rs, color=clr, alpha=0.8)
            ax2.plot(leg_xs, leg_ts, color=clr)
            ax3.plot(leg_xs, leg_ps, color=clr)
            ax4.plot(leg_ts, leg_ps, color=clr)
        plotPrettyEarth(ax1, mod, distlabel=False, mode='core')
        #
        plt.show()


    if False:
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