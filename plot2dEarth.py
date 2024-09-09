#!/usr/bin/env python3

"""
This module provides functions to plot cross-correlation related data on a map (a 2D map).
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from .geomath import great_circle_plane_center_triple, haversine

__xy_data_ccrs = ccrs.PlateCarree()
__gc_data_ccrs = ccrs.Geodetic()
def plot_great_circle_path(lo1, la1, lo2, la2, ptlo, ptla, ax=None, color='k', linewidth=0.8, alpha=0.8, **kwargs):
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
def plot_great_circle_centers(lo1, la1, lo2, la2, ptlo, ptla,
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
        ax: a matplotlib axis object. Default is `None` to create a new figure and axis.
        **kwargs: other keyword arguments for the pyplot.plot options.
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
    if plot_northern_centers:
        ax.scatter(clos, clas, s=s, c=color_northern, transform=__xy_data_ccrs, marker=marker, alpha=alpha, **kwargs)
    if plot_southern_centers:
        ax.scatter(aclos, aclas, s=s, c=color_southern, transform=__xy_data_ccrs, marker=marker, alpha=alpha, **kwargs)
    pass
def plot_correlation_pairs(los_deg, las_deg, ptlo, ptla, selection_mat=None, random_selection_rate=None, ax=None, **kwargs):
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
        random_selection_rate: a float between 0 and 1 to randomly select a portion of
                               correlation pairs. Default is `None` to select all correlation pairs.
        ax: a matplotlib axis object. Default is `None` to create a new figure and axis.
        **kwargs: other keyword arguments for the pyplot.plot options.
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson() } )
    los_deg = np.array(los_deg)
    las_deg = np.array(las_deg)
    n = los_deg.size
    #### input selection
    selection_mat = np.ones((n, n), dtype=np.int8) if selection_mat is None else selection_mat
    #### random select a portion of correlation pairs
    n_selected    = int( np.sum(np.triu(selection_mat) ) if (random_selection_rate is None) else random_selection_rate*np.sum(np.triu(selection_mat) ) )
    random_selection_arr = np.zeros(n*(n-1)//2+n, dtype=np.int8)
    random_selection_arr[:n_selected] = 1
    np.random.shuffle(random_selection_arr) # in-place modification
    ####
    j1, j2 = 0, n
    for i1 in range(n):
        lo1, la1 = los_deg[i1], las_deg[i1]
        lo2, la2 = los_deg[i1:], las_deg[i1:]
        #### selection
        geo_selection = selection_mat[i1, i1:] # input selection
        tmp = (random_selection_arr[j1:j2] & geo_selection) # random selection
        print(geo_selection.size, geo_selection.size)
        lo2, la2 = lo2[tmp!=0], la2[tmp!=0]
        ####
        j1=j2
        j2+=(n-i1-1)
        ####
        if lo2.size == 0:
            continue
        ####
        plot_great_circle_centers(lo1, la1, lo2, la2, ptlo, ptla, ax=ax, **kwargs)
        for _lo2, _la2 in zip(lo2, la2):
            plot_great_circle_path(lo1, la1, _lo2, _la2, ptlo, ptla, ax=ax, **kwargs)



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
    plot_correlation_pairs(los_deg, las_deg, ptlo, ptla, ax=ax, random_selection_rate=0.2, zorder=0)
    ax.scatter(los_deg, las_deg, c='C0', s=30, marker='^', transform=__xy_data_ccrs, zorder=1)
    plt.show()