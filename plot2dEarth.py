#!/usr/bin/env python3

"""
This module provides functions to plot cross-correlation related data on a map (a 2D map).
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from .geomath import great_circle_plane_center_triple, great_circle_plane_center_triple_rad, haversine

__xy_data_ccrs = ccrs.PlateCarree()
__gc_data_ccrs = ccrs.Geodetic()
# Plot a single great circle path passing through two points (or using the third point in case the two points are at the same location).
def plot_great_circle_path_deg(lo1, la1, lo2, la2, ptlo, ptla, ax=None, color='k', linewidth=0.8, alpha=0.8, **kwargs):
    """
    Plot a great circle path passing through two points on a 2D map.
    #
    Parameters:
        lo1: a single value. The longitude of one point in degree.
        la1: a single value. The latitude of  one point in degree.
        ptlo: the longitude for the third point in degree in case that the two points provides are at
              the same location. Default is `None` to use the default value.
        ptla: the same as `ptlo` but for the latitude.
        lo2: a single value. The longitude of the other point in degree.
        la2: a single value. The latitude of the other point in degree.
        ax:  a matplotlib axis object. Default is `None` to create a new figure and axis.
        **kwargs: other keyword arguments for the pyplot.plot options.
    Return:
        ax: a matplotlib axis object used for plotting.
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson() } )
        ax.set_global()
        ax.add_feature(cfeature.LAND, color='#eeeeee')
    if np.abs(lo1-lo2)<1e-6 and np.abs(la1-la2)<1e-6:
        lo2, la2 = ptlo, ptla
    lo3, la3 = lo1+180, -la1
    lo4, la4 = lo2+180, -la2
    ax.plot([lo1, lo2, lo3, lo4, lo1], [la1, la2, la3, la4, la1], transform=__gc_data_ccrs,
            color=color, linewidth=linewidth, alpha=alpha, **kwargs)
    return ax
# Plot a single great circle path given its center point
def plot_great_circle_path_given_center_deg(clo_deg, cla_deg, ax=None, color='k', linewidth=0.8, alpha=0.8, **kwargs):
    """
    Plot a great circle path given its center point.
    #
    Parameters:
        clo_deg: a single value. The longitude of the center point in degree.
        cla_deg: a single value. The latitude of the center point in degree.
        ax: a matplotlib axis object. Default is `None` to create a new figure and axis.
        **kwargs: other keyword arguments used for plotting the great circle path line (the pyplot.plot options).
    """
    lo1, la1 = clo_deg+90, 0
    lo2 = clo_deg
    la2 = cla_deg+90 if cla_deg<0 else cla_deg-90
    return plot_great_circle_path_deg(lo1, la1, lo2, la2, 0, 0, ax=ax, color=color, linewidth=linewidth, alpha=alpha, **kwargs)
# Plot the center of great circle path passing through two points (or using the third point in case the two points are at the same location).
def plot_great_circle_centers_deg(lo1, la1, lo2, la2, ptlo, ptla,
                                  plot_northern_centers=True, plot_southern_centers=False,
                                  color_northern='C3', color_southern='C1',
                                  ax=None, s=20, marker='.', alpha=0.8, **kwargs):
    """
    Plot the center of great circle path passing through two points, or the centers of multiple
    great circle paths passing through multiple pairs of points on a 2D map.
    #
    Parameters:
        lo1: a single value or a list of values for the longitude of the first point(s) in degree.
        la1: a single value or a list of values for the latitude  of the first point(s) in degree.
        lo2: a single value or a list of values for the longitude of the second point(s) in degree.
        la2: a single value or a list of values for the latitude  of the second point(s) in degree.
        ptlo: the longitude for the third point in degree in case that the two points provides are at
              the same location. Default is `None` to use the default value.
        ptla: the same as `ptlo` but for the latitude.
        plot_northern_centers: a boolean. If `True`, plot the centers of the great circle paths in the northern hemisphere.
                               Default is `True`.
        plot_southern_centers: a boolean. If `True`, plot the centers of the great circle paths in the southern hemisphere.
                               Default is `False`.
        color_northern: a string. The color for the centers in the northern hemisphere. Default is `C3`.
        color_southern: a string. The color for the centers in the southern hemisphere. Default is `C1`.
        ax: a matplotlib axis object. Default is `None` to create a new figure and axis.
        **kwargs: other keyword arguments for the pyplot.plot options.
    Return:
        ax: a matplotlib axis object used for plotting.
    #
    Note:
    The input of `lo1, la1, lo2, la2` could be single values(v) or np.ndarray objects(a). They could be:
    1. plot_great_circle_centers(lo1=v1, la1=v2, lo2=v3, la2=v4, ...) where v1, v2, v3, v4 are single values.
       Then, a single center point will be plotted.
    2. plot_great_circle_centers(lo1=v1, la1=v2, lo2=a1, la2=a2, ...) where a1 and a2 have the same size.
       Then, multiple center points will be plotted for the pairs (v1, v2, a1[0], a2[0]), (v1, v2, a1[1], a2[1]), ...
    3. plot_great_circle_centers(lo1=a1, la1=a2, lo2=a3, la2=a4, ...) where all a1, a2, a3, a4 have the same size.
       Then, multiple center points will be plotted for the pairs (a1[0], a2[0], a3[0], a4[0]), (a1[1], a2[1], a3[1], a4[1]), ...
    """
    (clos, clas), (aclos, aclas) = great_circle_plane_center_triple(lo1, la1, lo2, la2, ptlo, ptla, critical_distance=0.01)
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson() } )
        ax.set_global()
        ax.add_feature(cfeature.LAND, color='#eeeeee')
    if 'color' in kwargs:
        kwargs = kwargs.copy()
        kwargs.pop('color')
    if plot_northern_centers:
        ax.scatter(clos, clas, s=s, c=color_northern, transform=__xy_data_ccrs, marker=marker, alpha=alpha, **kwargs)
    if plot_southern_centers:
        ax.scatter(aclos, aclas, s=s, c=color_southern, transform=__xy_data_ccrs, marker=marker, alpha=alpha, **kwargs)
    return ax
# Plot correlation pairs formed using a list of points (e.g., station locations), and a common point (an event location).
def plot_correlation_pairs_deg(los_deg, las_deg, ptlo, ptla, selection_mat=None, random_selection = None, ax=None,
                               plot_gc_path=True, plot_northern_centers=True, plot_southern_centers=False,
                               color_northern='C3', color_southern='C1',
                               **kwargs):
    """
    Plot correlation pairs on a 2D map.
    #
    Parameters:
        los_deg: a list of values or a 1D numpy array. The longitude of the points in degree.
        las_deg: a list of values or a 1D numpy array. The latitude of the points in degree.
        ptlo: the longitude for the third point in degree in case that the two points provides are at
              the same location. Default is `None` to use the default value.
        ptla: the same as `ptlo` but for the latitude.
        selection_mat: a 2D numpy array. The selection matrix.
                       The matrix has the shape of NxN, where N is the number of points (len(los_deg)).
                       The matrix is symmetric, and each element is either 0 or 1.
                       The ijth element (ith row and jth col) is 1 if the correlation pair formed
                       by the ith and jth points are selected.
                       Default is `None` to select all correlation pairs.
        random_selection: a float between 0 and 1 OR an integer to randomly select a portion of
                          correlation pairs. Default is `None` to select all correlation pairs.
        ax: a matplotlib axis object. Default is `None` to create a new figure and axis.
        plot_northern_centers: a boolean. If `True`, plot the centers of the great circle paths in the northern hemisphere.
                               Default is `True`.
        plot_southern_centers: a boolean. If `True`, plot the centers of the great circle paths in the southern hemisphere.
                               Default is `False`.
        color_northern: a string. The color for the centers in the northern hemisphere. Default is `C3`.
        color_southern: a string. The color for the centers in the southern hemisphere. Default is `C1`.
        **kwargs: other keyword arguments for the pyplot.plot options.
    Return:
        ax: a matplotlib axis object used for plotting.
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson() } )
    los_deg = np.array(los_deg)
    las_deg = np.array(las_deg)
    n = los_deg.size
    #### input selection
    selection_mat = np.ones((n, n), dtype=np.int8)   if   (selection_mat is None)   else   selection_mat
    n_valid = np.sum( np.where( np.triu(selection_mat) != 0, 1, 0) )
    #### random select a portion of correlation pairs
    if random_selection is None:
        n_selected = n_valid
    elif type(random_selection) is float:
        n_selected = int( n_valid*random_selection )
    elif type(random_selection) is int:
        n_selected = random_selection
    else:
        raise ValueError('Invalid random_selection value. must be None, float, or int.', random_selection)
    random_selection_arr = np.zeros(n_valid, dtype=np.int8)
    random_selection_arr[:n_selected] = 1
    np.random.shuffle(random_selection_arr) # in-place modification
    #### get random selection matrix
    random_selection_mat = np.zeros((n, n), dtype=np.int8)
    random_selection_mat[ np.where( np.triu(selection_mat) != 0) ] = random_selection_arr
    random_selection_mat |= random_selection_mat.T # make symmetric
    #### apply random selection results
    random_selection_mat &= selection_mat # here avoid modifying the input selection_mat
    ####
    for i1 in range(n):
        lo1, la1 = los_deg[i1], las_deg[i1]
        lo2, la2 = los_deg[i1:], las_deg[i1:]
        #### selection
        row_selection = random_selection_mat[i1, i1:]
        lo2, la2 = lo2[row_selection!=0], la2[row_selection!=0]
        ####
        if lo2.size == 0:
            continue
        ####
        if plot_northern_centers or plot_southern_centers:
            plot_great_circle_centers_deg(lo1, la1, lo2, la2, ptlo, ptla, ax=ax,
                                          plot_northern_centers=plot_northern_centers, plot_southern_centers=plot_southern_centers,
                                          color_northern=color_northern, color_southern=color_southern,
                                          **kwargs)
        if plot_gc_path:
            for _lo2, _la2 in zip(lo2, la2):
                plot_great_circle_path_deg(lo1, la1, _lo2, _la2, ptlo, ptla, ax=ax, **kwargs)
    return ax
# Get the centers of great circle paths passing through pairs of points (or using the third point in case the two points are at the same location).
def get_great_circle_centers_deg(los_deg, las_deg, ptlo, ptla, selection_mat=None):
    """
    Return a list of points representing the centers of great circle paths passing through pairs of points.
    #
    Parameters:
        los_deg: a list of values or a 1D numpy array. The longitude of the points in degree.
        las_deg: a list of values or a 1D numpy array. The latitude of the points in degree.
        ptlo: the longitude for the third point in degree in case that the two points provides are at
              the same location. Default is `None` to use the default value.
        ptla: the same as `ptlo` but for the latitude.
        selection_mat: a 2D numpy array. The selection matrix.
                       The matrix has the shape of NxN, where N is the number of points (len(los_deg)).
                       The matrix is symmetric, and each element is either 0 or 1.
                       The ijth element (ith row and jth col) is 1 if the correlation pair formed
                       by the ith and jth points are selected.
                       Default is `None` to select all correlation pairs.
    Return:
        (clo_n, cla_n), (clo_s, cla_s): two tuples of np.ndarray.
                                        The first tuple is for the center longitudes and latitudes in the northern hemisphere,
                                        and the second tuple is for the centers in the southern hemisphere.
    """
    los = np.deg2rad( np.array(los_deg) )
    las = np.deg2rad( np.array(las_deg) )
    ptlo, ptla = np.deg2rad(ptlo), np.deg2rad(ptla)
    n = los.size
    #### input selection
    selection_mat = np.ones((n, n), dtype=np.int8)   if   (selection_mat is None)   else   selection_mat
    ####
    clo_n, cla_n = list(), list()
    clo_s, cla_s = list(), list()
    ####
    for i1 in range(n):
        lo1, la1 = los[i1],  las[i1]
        lo2, la2 = los[i1:], las[i1:]
        #### selection
        row_selection = selection_mat[i1, i1:]
        lo2, la2 = lo2[row_selection!=0], la2[row_selection!=0]
        ####
        if lo2.size == 0:
            continue
        ####
        tmp = great_circle_plane_center_triple_rad(lo1, la1, lo2, la2, ptlo, ptla, critical_distance=0.00171)
        (clos, clas), (aclos, aclas)  = tmp # `cl` is for the northern hemisphere, and `acl` is for the southern hemisphere
        clo_n.extend(clos)
        cla_n.extend(clas)
        clo_s.extend(aclos)
        cla_s.extend(aclas)
    ####
    clo_n = np.rad2deg(clo_n)
    cla_n = np.rad2deg(cla_n)
    clo_s = np.rad2deg(clo_s)
    cla_s = np.rad2deg(cla_s)
    return (clo_n, cla_n), (clo_s, cla_s)

if __name__ == '__main__':
    plateCr = ccrs.Robinson()
    plateCr._threshold = plateCr._threshold/10.  #set finer threshold
    fig, ax = plt.subplots(subplot_kw={'projection': plateCr } )
    ax.set_global()
    ax.add_feature(cfeature.LAND, color='#eeeeee')
    #plot_great_circle_path(0, 0, 90, 30, ax=ax)
    #plot_great_circle_centers(0, 0, 90, 30, 10, 10, ax=ax)
    los_deg = np.random.rand(10)*10+130
    las_deg = np.random.rand(10)*10+30
    ptlo, ptla = 0, 0
    plot_correlation_pairs_deg(los_deg, las_deg, ptlo, ptla, ax=ax, random_selection_rate=None, zorder=0, color='#987654', linewidth=0.3,
                               plot_gc_path=True, plot_northern_centers=True, plot_southern_centers=True)
    ax.scatter(los_deg, las_deg, c='C0', s=30, marker='^', transform=__xy_data_ccrs, zorder=1)
    plt.show()