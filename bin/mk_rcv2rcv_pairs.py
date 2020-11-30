#!/usr/bin/env python3
"""
Executable files for obtain sac hdr information for many files.
"""
from matplotlib.pyplot import figure, plot
from numpy import version
from sacpy.sac import c_rd_sachdr_wildcard, ffi
import h5py
import numpy as np
from sys import exit, argv
from getopt import getopt
from glob import glob
from os import getcwd
from numba import jit
from sacpy.geomath import haversine, great_circle_plane_center, point_distance_to_great_circle_plane, antipode

import cartopy.geodesic
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

geo_gcd = ccrs.Geodetic()
data_crs = ccrs.PlateCarree()

####################################################################################
def plot_basemap():
    """
    Plot the basemap
    """
    ###
    fig = plt.figure(figsize= (12, 8) )
    ax  = fig.add_axes([0.1, 0.35, 0.8, 0.6 ], projection= ccrs.PlateCarree(central_longitude=180.0))
    ax2 = fig.add_axes([0.1, 0.05, 0.8, 0.25] )
    #ax  = plt.subplot(211, projection= ccrs.PlateCarree(central_longitude=180.0) )
    #ax2 = plt.subplot(212)
    ###
    ax.set_global()
    ax.gridlines(linestyle='--', color='k', draw_labels=True)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    return fig, ax, ax2
def plot_single_gc(ax, x, y, **kwargs):
    """
    Plot the a single great-circle given its center (x, y)
    """
    lo1, la1 = x, y-90
    lo2, la2 = x+90, 0.0
    lo3, la3 = x+180, y+90
    lo4, la4 = x-90, 0.0
    if la1 < -90:
        la1 = -180-la1
    if la3 > 90:
        la3 = 180-la3
    ax.plot([lo1, lo2, lo3, lo4, lo1], [la1, la2, la3, la4, la1], transform=geo_gcd, **kwargs )
    pass
def plot_gc(ax, clo, cla, downsample_ratio=1, plot_gcpath=False, plot_gcc=False):
    """
    Plot many great-circles given a list of center longitude `clo` and latitude `cla`.
    """
    ### plot the great-cirles
    if plot_gcpath:
        for x, y in zip(clo[::downsample_ratio], cla[::downsample_ratio] ):
            plot_single_gc(ax, x, y, alpha= 0.6, color='white', linewidth=0.8 )
    ### plot the center
    if plot_gcc:
        ax.scatter(clo[::downsample_ratio], cla[::downsample_ratio], 10, color='gray', marker='.', alpha= 0.6, transform=data_crs )
def plot_dist(ax, dist, dist_range, dist_step=1.0):
    """
    """
    n, bins, patches = ax.hist(dist, np.arange(0.0, 180.1, dist_step), align= 'left', color='gray' )
    v = n[n!=0]
    v = np.sort(v)[v.size//2] * 3
    ax.set_xlim(dist_range)
    ax.set_ylim([0, v] )
    ax.set_xlabel('Inter-receiver distance ($\degree$)')
    ax.set_ylabel('Number of pairs')
def decipher_plot_options(plot_options):
    figname = 'junk.png'
    downsample = 10
    plot_gcc = False
    for it in plot_options.split(','):
        key, value = it.split('=')
        if key == 'fnm':
            figname = value
        elif key == 'downsample':
            downsample = int(value)
        elif key == 'gcp':
            plot_gcpath = eval(value)
        elif key == 'gcc':
            plot_gcc = eval(value)
    return figname, downsample, plot_gcpath, plot_gcc
####################################################################################


@jit(nopython=True, nogil=True)
def form_rcv2rcv_pairs(evlo, evla, stlo, stla, az, dist_range, ev_gcd_range, daz_range, 
                        gc_center_rect_lo1, gc_center_rect_lo2, 
                        gc_center_rect_la1, gc_center_rect_la2):
    """
    Given information for a list of station sharing a same event and many selectin criteria,
    return the formed receiver-to-receiver pairs.

    Return: (idx1, idx2, dist, clo, cla)

    idx1: a list of index of the first sac file in the pairs.
    idx2: a list of index of the second sac file in the pairs.
    dist: a list of distance between the two receivers in the pairs.
    clo:  a list of great-circle center longitude for the two receivers in the pairs.
    cla:  a list of great-circle center longitude for the two receivers in the pairs.
    """
    ###
    dist_min, dist_max = dist_range
    gcd_min, gcd_max = ev_gcd_range
    daz_min, daz_max = daz_range
    ###
    sz = stlo.size
    pairs = list()
    for idx1 in range(sz):
        lo1, la1, a1 = stlo[idx1], stla[idx1], az[idx1]
        for idx2 in range(idx1, sz):
            lo2, la2, a2 = stlo[idx2], stla[idx2], az[idx2]
            ###
            dist = haversine(lo1, la1, lo2, la2)
            if dist > dist_max or dist < dist_min:
                continue
            ###
            daz, gcd, clo, cla = -12345, -12345, -12345, -12345
            ### daz
            if daz_min > 0.0 or daz_max < 90.0:
                daz = abs(a1-a2) %360
                if daz > 180:
                    daz = 360-daz
                if daz > 90:
                    daz = 180-daz
                if daz > daz_max or daz < daz_min:
                    continue
            ### gcd
            if gcd_min > 0.0 or gcd_max < 90.0:
                gcd = abs( point_distance_to_great_circle_plane(evlo, evla, lo1, la1, lo2, la2) )
                if gcd > gcd_max or gcd < gcd_min:
                    continue
            ### gc_center_rect
            if gc_center_rect_lo1.size >0:
                clo, cla = 0.0, 0.0
                if abs(lo1-lo2)<1.0e-3 and abs(la1-la2)<1.0e-3:
                    (clo, cla), pt2 = great_circle_plane_center(evlo, evla, lo1, la1)
                    if cla < 0.0:
                        clo, cla = pt2
                else:
                    (clo, cla), pt2 = great_circle_plane_center(lo1, la1, lo2, la2)
                    if cla < 0.0:
                        clo, cla = pt2
                clo = clo % 360
                ###
                flag = 0
                for rlo1, rlo2, rla1, rla2 in zip(gc_center_rect_lo1, gc_center_rect_lo2, gc_center_rect_la1, gc_center_rect_la2):
                    if cla < rla1 or cla > rla2:
                        flag = 1
                        break
                    elif rlo1 < rlo2:
                        if clo < rlo1 or clo > rlo2:
                            flag = 1
                            break
                    else:
                        if clo > rlo1 or clo < rlo2:
                            flag = 1
                            break
                if flag == 1:
                    continue
            ###
            pairs.append( (idx1, idx2, dist, clo, cla) )
    return  (   np.array([it[0] for it in pairs ], dtype=np.int32),
                np.array([it[1] for it in pairs ], dtype=np.int32),
                np.array([it[2] for it in pairs ], dtype=np.float32),
                np.array([it[3] for it in pairs ], dtype=np.float32),
                np.array([it[4] for it in pairs ], dtype=np.float32) )

def run(h5fnm, dist_range, ev_gcd_range=None, daz_range=None,
                gc_center_rect_lo1=None, gc_center_rect_lo2=None, 
                gc_center_rect_la1=None, gc_center_rect_la2=None,
                verbose=False,
                plot_options=None  ):
    ###
    fid = h5py.File(h5fnm, 'r')
    grp_hdrtree = fid['/hdr_tree']
    ### for plotting
    cla_sum, clo_sum, npairs, all_dist = 0.0, 0.0, 0, list()
    fig, ax, ax2 = plot_basemap() if plot_options!=None else (None, None)
    figname, downsample, plot_gcpath, plot_gcc = decipher_plot_options(plot_options) if plot_options!=None else (None, None, None)
    ###
    for key in grp_hdrtree:
        grp = fid['/hdr_tree/%s' % key ]
        evlo = grp['evlo'][0]
        evla = grp['evla'][0]
        stlo = grp['stlo'][:]
        stla = grp['stla'][:]
        az   = grp['az'][:]
        ###
        i1, i2, dist, clo, cla = form_rcv2rcv_pairs(evlo, evla, stlo, stla, az, dist_range, ev_gcd_range, daz_range,
                            gc_center_rect_lo1, gc_center_rect_lo2, 
                            gc_center_rect_la1, gc_center_rect_la2 )
        npairs = npairs + clo.size
        ###
        if verbose:
            print(key, 'npairs:', dist.size )
        ### for plotting: plot all great-circles and their centers
        if plot_options != None:
            ###
            all_dist.extend(dist)
            ###
            clo_sum = clo_sum + np.sum(clo)
            cla_sum = cla_sum + np.sum(cla)
            plot_gc(ax, clo, cla, downsample, plot_gcpath, plot_gcc)
    ### for plotting: plot the great-circle average and the centers
    if npairs > 0 and plot_options != None:
        x, y = clo_sum/npairs, cla_sum/npairs
        if plot_gcpath:
            plot_single_gc(ax, x, y, alpha= 1.0, color='C1', linewidth=1.2 )
        if plot_gcc:
            ax.scatter([x], [y], 30, color='C1', marker='o', alpha= 0.6, transform=data_crs )
        plot_dist(ax2, all_dist, dist_range, 1.0)
        plt.savefig(figname, bbox_inches = 'tight', pad_inches = 0.2 )
        plt.close()

    #print(grp_hdrtree.keys() )
    ###
    fid.close()

if __name__ == "__main__":
    ####
    ev_gcd_range = (-0.1, 90.1)
    daz_range    = (-0.1, 90.1)
    dist_range   = (0.0, 180.0)
    gc_center_rect_lo1 = np.array([], dtype=np.float32 )
    gc_center_rect_lo2 = np.array([], dtype=np.float32 )
    gc_center_rect_la1 = np.array([], dtype=np.float32 )
    gc_center_rect_la2 = np.array([], dtype=np.float32 )
    verbose= False
    plot_options = None
    ####
    HMSG = """
    %s -I in.h5 [-D 0/180] [--daz -0.1/15.1] [--gcd_ev -0.1/20.1] [--gc_center_rect 120/160/0/30,170/190/0/10] 
    [--plot fnm=junk.png,downsample=10,gcp=True,gcc=True ] [-V]
    """ % argv[0]
    if len(argv) < 2:
        print(HMSG)
        exit(0)
    ####
    options, remainder = getopt(argv[1:], 'I:D:V', ['daz=', 'gcd_ev=', 'gc_center_rect=', 'plot='] )
    for opt, arg in options:
        if opt in ('-I'):
            h5_fnm = arg
        elif opt in ('-V'):
            verbose = True
        elif opt in ('-D'):
            dist_range = np.array( [float(it) for it in arg.split('/') ], dtype=np.float32 )
        elif opt in ('--daz'):
            daz_range = np.array( [float(it) for it in arg.split('/') ], dtype=np.float32 )
        elif opt in ('--gcd_ev'):
            ev_gcd_range = np.array( [float(it) for it in arg.split('/')] , dtype=np.float32 )
        elif opt in ('--gc_center_rect'):
            gc_center_rect_lo1 = list()
            gc_center_rect_lo2 = list()
            gc_center_rect_la1 = list()
            gc_center_rect_la2 = list()
            for rect in arg.split(','):
                tmp = [float(it) for it in rect.split('/')]
                gc_center_rect_lo1.append( tmp[0]%360 )
                gc_center_rect_lo2.append( tmp[1]%360 )
                gc_center_rect_la1.append( tmp[2] )
                gc_center_rect_la2.append( tmp[3] )
            gc_center_rect_lo1 = np.array(gc_center_rect_lo1, dtype=np.float32 )
            gc_center_rect_lo2 = np.array(gc_center_rect_lo2, dtype=np.float32 )
            gc_center_rect_la1 = np.array(gc_center_rect_la1, dtype=np.float32 )
            gc_center_rect_la2 = np.array(gc_center_rect_la2, dtype=np.float32 )
        elif opt in ('--plot'):
            plot_options = arg
        elif opt in ('-H', '-h', '-?'):
            print(HMSG)
            exit(0)
    ###
    run(h5_fnm, dist_range, ev_gcd_range, daz_range, gc_center_rect_lo1, gc_center_rect_lo2, gc_center_rect_la1, gc_center_rect_la2, verbose, plot_options )