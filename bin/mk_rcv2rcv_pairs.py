#!/usr/bin/env python3
"""
Executable files for obtain sac hdr information for many files.
"""
from sacpy.sac import c_rd_sachdr_wildcard, ffi
import h5py
import numpy as np
from sys import exit, argv
from getopt import getopt
from glob import glob
from os import getcwd
from numba import jit
from sacpy.geomath import haversine, great_circle_plane_center, point_distance_to_great_circle_plane

@jit(nopython=True, nogil=True)
def form_rcv2rcv_pairs(evlo, evla, stlo, stla, az, ev_gcd_range=None, daz_range=None, 
                        gc_center_rect_lo1=None, gc_center_rect_lo2=None, 
                        gc_center_rect_la1=None, gc_center_rect_la2=None):
    ###
    gcd_min, gcd_max  = -0.1, 90.1
    if ev_gcd_range!=None:
        gcd_min, gcd_max = ev_gcd_range
    daz_min, daz_max  = -0.1, 90.1
    if daz_range!= None:
        daz_min, daz_max = daz_range
    ###
    sz = stlo.size
    pairs = list()
    for idx1 in range(sz):
        lo1, la1, a1 = stlo[idx1], stla[idx1], az[idx1]
        for idx2 in range(idx1, sz):
            lo2, la2, a2 = stlo[idx2], stla[idx2], az[idx2]
            daz, gcd, clo, cla = -12345, -12345, -12345, -12345
            ### daz
            if daz_range != None:
                daz = abs(a1-a2) %360
                if daz > 180:
                    daz = 360-daz
                if daz > 90:
                    daz = 180-daz
                if daz > daz_max or daz < daz_min:
                    continue
            ### gcd
            if ev_gcd_range != None:
                gcd = abs( point_distance_to_great_circle_plane(evlo, evla, lo1, la1, lo2, la2) )
                if gcd > gcd_max or gcd < gcd_min:
                    continue
            ### gc_center_rect
            if gc_center_rect_lo1 != None:
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
            dist = haversine(lo1, la1, lo2, la2)
            ###
            pairs.append( (idx1, idx2, dist) )
    return  np.array([it[0] for it in pairs ], dtype=np.int32), np.array([it[1] for it in pairs ], dtype=np.int32), np.array([it[2] for it in pairs ], dtype=np.float32)
def run(h5fnm, ev_gcd_range=None, daz_range=None,
                gc_center_rect_lo1=None, gc_center_rect_lo2=None, 
                gc_center_rect_la1=None, gc_center_rect_la2=None  ):
    ###
    ###
    fid = h5py.File(h5fnm, 'r')
    grp_hdrtree = fid['/hdr_tree']
    for key in grp_hdrtree:
        grp = fid['/hdr_tree/%s' % key ]
        evlo = grp['evlo'][0]
        evla = grp['evla'][0]
        stlo = grp['stlo'][:]
        stla = grp['stla'][:]
        az   = grp['az'][:]
        ###
        i1, i2, dist = form_rcv2rcv_pairs(evlo, evla, stlo, stla, az, ev_gcd_range, daz_range,
                            gc_center_rect_lo1, gc_center_rect_lo2, 
                            gc_center_rect_la1, gc_center_rect_la2 )
    #print(grp_hdrtree.keys() )
    ###
    fid.close()

if __name__ == "__main__":
    ####
    ev_gcd_range = None
    daz_range = None
    gc_center_rect_lo1 = None
    gc_center_rect_lo2 = None
    gc_center_rect_la1 = None
    gc_center_rect_la2 = None
    ####
    HMSG = '%s -I in.h5 [--daz -0.1/15.1] [--gcd_ev -0.1/20.1] [--gc_center_rect 120/160/0/30,170/190/0/10]' % argv[0]
    if len(argv) < 2:
        print(HMSG)
        exit(0)
    ####
    options, remainder = getopt(argv[1:], 'I:', ['daz=', 'gcd_ev=', 'gc_center_rect='] )
    for opt, arg in options:
        if opt in ('-I'):
            h5_fnm = arg
            outfnm = arg
        elif opt in ('--daz'):
            daz_range = [float(it) for it in arg.split('/')]
        elif opt in ('--gcd_ev'):
            ev_gcd_range = [float(it) for it in arg.split('/')]
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
        elif opt in ('-H', '-h', '-?'):
            print(HMSG)
            exit(0)
    ###
    run(h5_fnm, ev_gcd_range, daz_range, gc_center_rect_lo1, gc_center_rect_lo2, gc_center_rect_la1, gc_center_rect_la2 )