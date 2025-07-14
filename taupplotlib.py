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

import warnings
import pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import obspy.taup as taup
from obspy.taup.seismic_phase import SeismicPhase
import numpy as np
import scipy
from copy import deepcopy

def colored_line(x, y, c, ax, vmin=None, vmax=None, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    vmin, vmax : float, optional for color scaling.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.

    This function is from https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # update the norm is vmin and vmax are provided
    if vmin is not None or vmax is not None:
        lc_kwargs['norm'] = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    ax.plot(x, y, alpha=0.0) # just for adjust the xlim and ylim
    return ax.add_collection(lc)
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
                        arrowprops=dict( headlength=headlength, headwidth=headwidth, color = color, alpha=alpha, lw=0.0,),
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
def split_px_legs(ps, xs, lst_of_lst=None):
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
    >>> legs = split_px_legs(ps, xs, [ts, sth])
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
def plot_raypath_geo_arrivals(lst_of_geo_arrivals, c='C0', ax=None,
                              plot_arrow_at_radii=None, plot_pretty_earth=True,
                              vmin=None, vmax=None, cmap='viridis', radius_range=None, **kwargs):
    """
    Plot the ray path for a list of `geo_arrival` objects.
    #
    #Parameters:
    lst_of_geo_arrivals: a list of `geo_arrival` objects.
    c:                   a single value, or a list of colors, or list of floating number which specific
                         colors for each of the `geo_arrival` objects.
    ax:                  axes to plot, if None, create a new one with `projection=polar`.
    plot_arrow_at_radii: A single value or a list of radii to plot arrows at these radii.
                         Default is `None`, which means no arrows.
    plot_pretty_earth:   `True` or `False` to plot the pretty Earth with discontinuities.
    vmin, vmax:          for color scaling, if `c` is a list of floating numbers.
                         In default, will use the min and max of `c` as vmin and vmax.
    cmap:                colormap to use for the colors.
    radius_range:        a tuple of (min_radius, max_radius) to select the ray path within this range.
    kwargs:              other keyword arguments for `ax.plot()` or `taupplotlib.add_arrow(...)`.
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(6, 6) )
    #####
    c = np.array(c).flatten()
    if c.size == 1:
        lst_of_clrs = [c[0] for junk in lst_of_geo_arrivals]
    elif c.size == len(lst_of_geo_arrivals):
        if vmin is None or vmax is None:
            vmin, vmax = np.min(c), np.max(c)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap)
        lst_of_clrs = [cmap(norm(it)) for it in c]
    else:
        raise ValueError("c should be a single value or a list of the same length as lst_of_geo_arrivals.")
    #####
    for single_geo_arrival, clr in zip(lst_of_geo_arrivals, lst_of_clrs):
        single_geo_arrival.plot_raypath(ax=ax, color=clr, plot_arrow_at_radii=plot_arrow_at_radii,
                                        plot_pretty_earth=False, radius_range=radius_range, **kwargs)
    #####
    if plot_pretty_earth:
        plotPrettyEarth(ax, lst_of_geo_arrivals[0].taupy_model, distlabel=False, mode='vp')
    return ax

def generate_interp1d_funcs(data_dict, kind='linear', fill_value=np.nan, func_keys=None):
    """
    Generate many interpolation functions given many 1d data.
    #
    #Parameters:
    data_dict:  a dictionary of 1d data, where each key is a name and each value is a 1d array.
                all arrays should have the same length.
                E.g., {'a': array1, 'b': array2, 'c': array3, ...}
                Please don't use any key that contains the character `2`.
    kind:       the kind of interpolation, e.g., 'linear' (Default), 'cubic', etc.
    fill_value: the value to use for points outside the range of the data.
                Default is `np.nan`, which means no extrapolation.
    func_keys:  a list of keys to generate interpolation functions for.
                E.g., ['a2b', 'a2c', 'c2b'].
                If `None`, will generate all possible pairs of keys in `data_dict`.
    #
    #Return:    a dictionary object that store the interpolation functions.
                E.g., {'a2b': interp1d_func_a2b, 'b2c': interp1d_func_b2c, 'b2a': interp1d_func_b2a, ...}
    #
    !!!Note:    It is user's reponsibility to ensure that the data_dict['a'] is monotonic when
                generating the func['a2b'], while this function will loosely check for that.
    #
    #E.g.,
    >>> xs = np.linspace(0, 10, 100)
    >>> ys = xs**2
    >>> zs = xs+1
    >>> data_dict = {'x': xs, 'y': ys, 'z': zs}
    >>> interp_funcs = generate_interp1d_functions(data_dict, kind='linear', fill_value=np.nan)
    >>> y = interp_funcs['x2y'](3.1)
    >>> z = interp_funcs['x2y'](3.1)
    """
    ####
    keys = sorted(data_dict.keys() )
    for it in keys:
        if '2' in it:
            raise ValueError("Please avoid using keys that contain the character '2'.")
    ####
    if func_keys is None: # generate all possible pairs of keys
        func_keys =  [f"{c1}2{c2}" for c1 in keys for c2 in keys if c1 != c2]
    else:
        for single_func_key in func_keys: # check if all the input keys are valid
            # check if there is one '2' in the key
            if single_func_key.count('2') != 1:
                raise ValueError(f"func_keys should be a list of keys in the form of 'x2y', but got {single_func_key}.")
            c1, c2 = single_func_key.split('2')
            if (c1 not in keys) or (c2 not in keys):
                raise ValueError(f"Keys for {it} are not in data_dict.")
    ####
    func_dict = dict()
    for single_func_key in keys:
        c1, c2 = single_func_key.split('2')
        xs, ys = data_dict[c1], data_dict[c2]
        if len(xs) != len(ys):
            raise ValueError(f"Keys {c1} and {c2} should have the same length, but got {len(xs)} and {len(ys)}.")
        # check if xs is monotonic increasing or decreasing
        tmp1 = np.diff(xs)
        tmp2 = -tmp1
        if not(np.all(tmp1 >= 0) or np.all(tmp2 <= 0)):
            raise ValueError(f"Arrat for the key {c1} is not monotonic!")
        func_dict[single_func_key] = scipy.interpolate.interp1d(xs, ys, kind=kind, fill_value=fill_value, assume_sorted=False)
    return func_dict

class geo_arrival:
    """
    geo_arrival
    ============

    This is for processing `taup.tau.Arrival` for specific geometry and specific phase.

    Event-station geometry (via longitude) is considered, while just in 2D.

    Could run with generated `taup.tau.Arrival` object, or with specific phase name and ray parameter.

    E.g.,
    >>> import matplotlib.pyplot as plt
    >>> import obspy.taup as taup
    >>> from sacpy.taupplotlib import geo_arrival
    >>> model = taup.TauPyModel('ak135')
    >>> #
    >>> # Method 1 using pre-generated `taup.tau.Arrival` object
    >>> evdp, evlo = 100.0, 10.0
    >>> stdp, stlo = 0.0, 170.0
    >>> arr = model.get_ray_paths(evdp, stlo-evlo, ['PKP'])[0]
    >>> geo_arr = geo_arrival(evdp, evlo, stdp, stlo, arrival=arr, model=model)
    >>> rp = geo_arr.ray_param
    >>> #
    >>> # Method 2
    >>> geo_arr2 = geo_arrival(evdp, evlo, stdp, phase_name='PKP', ray_param=rp, model=model)
    >>> #
    >>> #compare the two geo_arrival objects
    >>> print(geo_arr.name, geo_arr2.name)
    >>> print(geo_arr.distance, geo_arr2.distance)
    >>> print(geo_arr.ray_param, geo_arr2.ray_param, arr.ray_param)
    >>> print(geo_arr.ray_param_sec_km, geo_arr2.ray_param_sec_km)
    >>> print(geo_arr.ray_param_sec_degree, geo_arr2.ray_param_sec_degree, arr.ray_param_sec_degree)
    >>> print(geo_arr.time, geo_arr2.time)
    >>> ax = geo_arr.plot_raypath(plot_arrow_at_radii=[5500], plot_pretty_earth=False, color='k', lw=3, alpha=0.8, label='method1')
    >>> ax = geo_arr2.plot_raypath(plot_arrow_at_radii=[5500], plot_pretty_earth=True, color='r', alpha=0.8, label='method2', ax=ax)
    >>> ax.legend()
    >>> plt.show()
    >>>
    """
    def __init__(self, eqdp, eqlo, stdp, stlo=None, arrival=None, model=taup.TauPyModel('ak135'),
                 phase_name=None, ray_param=None ):
        """
        #
        # Parameters:
        eqdp:    depth (km) of earthquake.
        eqlo:    longitude (degree) of earthquake.
        stdp:    depth (km) of station.
        stlo:    longitude (degree) of station.
                 Would be useless if `arrival` is not provided (which means the `phase_name` and
                 `ray_param` must be provided).
        arrival: an object of taup.tau.Arrivals, which is returned from get_ray_paths(...).
                 Could be `None` so that a new `Arrival` object will be generated using the provided
                 `phase_name` and `ray_param`.
        model:   model used in taup. Default is taup.TauPyModel('ak135').
        phase_name:             name of the seismic phase, e.g., 'P', 'S', 'PcP', 'ScP', etc.
                                Should be provided if `arrival` is not provided.
                                Will be useless if `arrival` is provided.
        ray_param:              ray parameter in sec/radian, which is used to generate an new `Arrival` object.
                                Should be provided if `arrival` is not provided.
                                Will be useless if `arrival` is provided.
        #
        #Note:
        (`eqdp`, `eqlo`) and (`stdp`, `stlo`) should match settings in `arrival` if the `arrival` is provided.
        """
        if arrival is not None:
            self.eqdp = eqdp
            self.eqlo = eqlo % 360
            self.eqlo_rad = np.deg2rad( eqlo )
            self.stdp = stdp
            self.stlo = stlo % 360
            self.stlo_rad = np.deg2rad( stlo )
            #
            self.arrival = arrival
            self.taupy_model = model
            self.ray_param = self.__get_rayp()
        elif phase_name is not None and ray_param is not None:
            evdp_km  = eqdp
            rcvdp_km = stdp
            phase    = phase_name
            #########
            tau_mod = model.model
            tau_mod_corr = tau_mod.depth_correct(evdp_km)                # correct/split the source and receiver depth
            tau_mod_corr = tau_mod_corr.split_branch(rcvdp_km)
            seismic_phase = SeismicPhase(phase, tau_mod_corr, rcvdp_km)  # generate a seismic phase object
            arr = seismic_phase.shoot_ray(0, ray_param)       # shoot the ray, please note, the first parameter is assigned to returned `Arrival.distance`!
            arr = seismic_phase.calc_path_from_arrival(arr)              # get the raypath
            arr.distance = arr.purist_distance % 360
            if arr.distance > 180:
                arr.distance = 360 - arr.distance
            #########
            self.__init__(eqdp, eqlo, stdp, arr.distance % 360, arrival=arr, model=model)
    def get_raypath(self):
        """
        Return the ray paths (lons, rs) from earthquake to station.
        `lons` is in rad, and `rs` in km.
        """
        eqlo = self.eqlo % 360
        stlo = self.stlo % 360
        dist = (stlo - eqlo) % 360
        radius = self.taupy_model.model.radius_of_planet

        rs = radius - self.arrival.path['depth']
        lons = deepcopy(self.arrival.path['dist'] ) # don't touch the original `Arrival`
        d_path = np.rad2deg(lons[-1] - lons[0]) % 360
        if (dist > 180 and d_path < 180) or (dist < 180 and d_path > 180):
            lons *= -1.0
        lons += self.eqlo_rad
        return lons, rs
    def plot_raypath(self, ax=None, plot_arrow_at_radii=None, plot_pretty_earth=False, split_ray_legs=False, radius_range=None, **kwargs):
        """
        Plot the ray path.
        #
        ax:                  axes to plot, if None, create a new one with `projection=polar`.
        plot_pretty_earth:   `True` or `False` to plot the pretty Earth.
        plot_arrow_at_radii: A single value or a list of radii to plot arrows at these radii.
                             Default is `None`, which means no arrows.
        split_ray_legs:      `True` or `False` (default) to split the ray path into segments according to
                             seismi rayleg names including 'P', 'S', 'K', 'I', 'J'.
        radius_range:        a tuple of (min_radius, max_radius) to select the ray path within this range.
        kwargs:              other keyword arguments for `ax.plot()` or `taupplotlib.add_arrow(...)` as above.
        """
        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(6, 6) )
        default_clr_lst =  mpl.cm.tab10.colors
        #### split ray legs or treat the entire ray path as a single leg
        ray_legs = self.get_split_raypath() if split_ray_legs else [(self.name, self.get_raypath())]
        #### plot arrows at specified radii
        if plot_arrow_at_radii is not None:
            for idx_leg, (junk_leg_name, (lons, rs)) in enumerate(ray_legs):
                default_clr = default_clr_lst[idx_leg % len(default_clr_lst)]
                plot_arrow_at_radii = np.array(plot_arrow_at_radii).flatten() # so that the input can be either a single value or an array-like list
                for radius in plot_arrow_at_radii:
                    if radius < 0 or radius > self.taupy_model.model.radius_of_planet:
                        raise ValueError("radius should be in [0, %d] km" % self.taupy_model.model.radius_of_planet)
                    add_arrow(  ax, lons, rs, loc_ys=[radius], color=kwargs.get('color', default_clr),
                                headlength=kwargs.get('headlength', 5),
                                headwidth=kwargs.get('headwidth', 4),
                                alpha=kwargs.get('alpha', 1.0),
                                reverse_arrow=kwargs.get('reverse_arrow', False),
                                zorder=kwargs.get('zorder', None) )
        #### remove arrow properties from kwargs when are not needed for plt.plot(...)
        for keys in ('headlength', 'headwidth', 'reverse_arrow'):
            kwargs.pop(keys, None)
        #### plot the ray path
        for idx_leg, (junk_leg_name, (lons, rs)) in enumerate(ray_legs):
            if radius_range is None:
                selected_rs   = rs
                selected_lons = lons
            else:
                selected_idx = np.where( (rs >= radius_range[0]) & (rs <= radius_range[1]) )[0]
                selected_rs   = rs[selected_idx]
                selected_lons = lons[selected_idx]
            if 'color' not in kwargs:
                default_clr = default_clr_lst[idx_leg % len(default_clr_lst)]
                ax.plot(selected_lons, selected_rs, color=default_clr, **kwargs)
            else:
                ax.plot(selected_lons, selected_rs, **kwargs)
        #### adjust
        ax.set_ylim((0, self.taupy_model.model.radius_of_planet) )
        if plot_pretty_earth:
            plotPrettyEarth(ax, self.taupy_model, distlabel=False, mode='vp')
        return ax
    def get_split_raypath(self):
        """
        Return a list of (ray_leg_name, lons, rs) from earthquake to station for the seismic wave.
        `lons` is in rad, and `rs` in km.
        """
        lons, rs = self.get_raypath()
        radius = self.taupy_model.model.radius_of_planet
        depths = radius-rs
        return split_raypath(self.taupy_model, self.name, depths, (lons, rs) )
    def __get_rayp(self):
        """
        Return ray parameter/slowness, in sec per radians.
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
        rp = -1.0*self.arrival.ray_param # in sec/radian
        lons = self.arrival.path['dist']
        d_path = np.rad2deg(lons[-1] - lons[0]) % 360
        if (dist > 180 and d_path < 180) or (dist < 180 and d_path > 180):
            rp = -rp
        return -rp
    @property
    def name(self):
        return self.arrival.name
    @property
    def distance(self):
        return self.arrival.distance
    @property
    def purist_distance(self):
        return self.arrival.purist_distance
    @property
    def time(self):
        return self.arrival.time
    @property
    def ray_param_sec_degree(self):
        return self.ray_param * (np.pi/180.0)
    @property
    def ray_param_sec_km(self):
        return self.ray_param /self.taupy_model.model.radius_of_planet
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
        import matplotlib.pyplot as plt
        import obspy.taup as taup
        from sacpy.taupplotlib import geo_arrival
        model = taup.TauPyModel('ak135')
        #
        # Method 1 using pre-generated `taup.tau.Arrival` object
        evdp, evlo = 100.0, 10.0
        stdp, stlo = 0.0, 173.2
        arr = model.get_ray_paths(evdp, stlo-evlo, ['PKP'])[0]
        geo_arr = geo_arrival(evdp, evlo, stdp, stlo, arrival=arr, model=model)
        rp = geo_arr.ray_param
        #
        # Method 2
        geo_arr2 = geo_arrival(evdp, evlo, stdp, phase_name='PKP', ray_param=rp, model=model)
        #
        #compare the two geo_arrival objects
        print(geo_arr.name, geo_arr2.name)
        print(geo_arr.distance, geo_arr2.distance)
        print(geo_arr.ray_param, geo_arr2.ray_param, arr.ray_param)
        print(geo_arr.ray_param_sec_km, geo_arr2.ray_param_sec_km)
        print(geo_arr.ray_param_sec_degree, geo_arr2.ray_param_sec_degree, arr.ray_param_sec_degree)
        print(geo_arr.time, geo_arr2.time)
        ax = geo_arr.plot_raypath(plot_arrow_at_radii=[4000], plot_pretty_earth=False, color='k', ls='--', alpha=0.8, label='method1')
        ax = geo_arr2.plot_raypath(plot_arrow_at_radii=[5500], plot_pretty_earth=True, color='r', alpha=0.8, label='method2', ax=ax)
        ax.legend()
        plt.show()

    if False:
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
        ps = np.array([it.ray_param_sec_degree for it in tmp2])
        #
        lst_of_legs = split_px_legs(ps, xs, [ts, tmp2])
        #
        fig = plt.figure(figsize=(16, 4))
        grd = fig.add_gridspec(1, 4)
        ax1 = fig.add_subplot(grd[0, 0], projection='polar')
        ax2 = fig.add_subplot(grd[0, 1])
        ax3 = fig.add_subplot(grd[0, 2])
        ax4 = fig.add_subplot(grd[0, 3])
        #
        for idx_leg, leg in enumerate(lst_of_legs[:]):
            #clr = mpl.cm.tab10.colors[idx_leg % 10]
            leg_ps, leg_xs, (leg_ts, arrs) = leg
            geo_arrs = [geo_arrival(0, 0, 0, it.purist_distance, it, mod) for it in arrs]
            print(geo_arrs[0].name )
            plot_raypath_geo_arrivals(geo_arrs[::2], ax=ax1, c=leg_ps[::2], plot_arrow_at_radii=[5000], plot_pretty_earth=False, split_ray_legs=False, 
                                      vmin=110, vmax=260, cmap='viridis', alpha=0.6)
            #for it in geo_arrs[::2]:
            #    it.plot(ax=ax1, alpha=0.9, linewidth=1.5, headwidth=5, headlength=10,
            #            plot_arrow_at_radii=[5000], zorder=100, split_ray_legs=True)
            colored_line(leg_xs, leg_ts, leg_ps, ax2, linewidth=3, cmap='viridis', alpha=0.8, vmin=110, vmax=260)
            colored_line(leg_xs, leg_ps, leg_ps, ax3, linewidth=3, cmap='viridis', alpha=0.8, vmin=110, vmax=260)
            colored_line(leg_ts, leg_ps, leg_ps, ax4, linewidth=3, cmap='viridis', alpha=0.8, vmin=110, vmax=260)
            ax2.plot(leg_xs, leg_ts, lw=0.6, color='w')
            ax3.plot(leg_xs, leg_ps, lw=0.6, color='w')
            ax4.plot(leg_ts, leg_ps, lw=0.6, color='w')
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