#!/usr/bin/env python3

"""
This contain sets of geometric methods, especially for spherical coordinates.
"""

import shutil
import os
import numpy as np
from math import radians, cos, sin, asin, sqrt

### coordinates transformation and frame rotation
def xyz_to_rlola(x, y, z):
    """
    Given (x, y, z), return (r, lo, la)
    """
    r = np.sqrt( x*x+y*y+z*z )
    lo = np.rad2deg( np.arctan2(y, x) )
    la = np.rad2deg( np.arctan2(z, np.sqrt(x*x+y*y) ) )
    return r, lo, la
def rlola_to_xyz(lo, la, radius = 1.0):
    """
    Given (lo, la, radius=1.0), return (x,y,z) coordinates.
    """
    lam, phi = map(radians, [lo, la])
    x = np.cos(phi)*np.cos(lam)*radius
    y = np.cos(phi)*np.sin(lam)*radius
    z = np.sin(phi)*radius
    return x,y,z
def sphere_rotate_axis(lo1, la1, lo2, la2):
    """
    Get point cooridinates and right-hand rotation angle, with which after rotation the great circle formed by two fixed points
    (lo1, la1) and (lo2, la2) turns into equator.
    #
    Two antipodal points, each of which has a special rotation angle, will be returned.
    (lo1, la1, angle1), (lo2, la2, angle2)
    """
    lam1, phi1, lam2, phi2 = map(radians, [lo1, la1, lo2, la2])
    c_1 = np.sin(lam1)/np.tan(phi1) - np.sin(lam2)/np.tan(phi2)
    c_2 = np.cos(lam1)/np.tan(phi1) - np.cos(lam2)/np.tan(phi2)
    lo1 = np.rad2deg( np.arctan2(c_1, c_2) )
    lo2, junk = antipode(lo1, 0)
    return_value = []
    for lo in [lo1, lo2]:
        lam_tmp = np.deg2rad(lo + 90)
        c_1 = np.tan(phi1)/np.sin(lam1-lam_tmp) - np.tan(phi2)/np.sin(lam2-lam_tmp)
        c_2 = 1.0/np.tan(lam1-lam_tmp) - 1.0/np.tan(lam2-lam_tmp)
        rotate_angle = -np.rad2deg( np.arctan2(c_1, c_2) )
        return_value.append( (lo, 0, rotate_angle) )
    return return_value
def sphere_rotate(lo, la, axis_lo, axis_la, axis_angle_deg):
    """
    Rotate (lo, la) with given axis directin and rotation angle. 
    Return new coordinate (new_lo, new_la) after rotation.
    #
    v_r = v cos(a) + k x v sin(a) + k (k dot v) (1-cos (a) )
    """
    v = np.array( rlola_to_xyz(lo, la))
    k = np.array( rlola_to_xyz(axis_lo, axis_la) )
    a = np.deg2rad(axis_angle_deg)
    c, s = np.cos(a), np.sin(a)
    v_r = v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1.0-c)
    x, y, z = v_r
    junk, new_lo, new_la= xyz_to_rlola(x, y, z)    
    return new_lo, new_la
### spherical geometry
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = np.rad2deg( 2 * asin(sqrt(a))  )
    return c # degree
def azimuth(evlo, evla, stlo, stla):
    """
    Get the azimuth angle (degree) from (evlo, evla) to (stlo, stla)
    """
    phi1, phi2 = np.deg2rad(evla), np.deg2rad(stla)
    dlamda     = np.deg2rad(stlo - evlo)
    a = np.arctan2(np.cos(phi2) * np.sin(dlamda),  np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlamda) )
    return np.rad2deg(a) % 360
def point_distance_to_great_circle_plane(ptlon, ptlat, lon1, lat1, lon2, lat2):
    """
    Calculate the distance from a point (ptlon, ptlat) to the great circle plane formed by two points
    of (lon1, lat1) and (lon2, lat2).
    #
    dxt = asin( sin(d13) ⋅ sin(a13−a12) )
    where
        d13 is (angular) distance from start point to third point
        a13 is (initial) bearing from start point to third point
        a12 is (initial) bearing from start point to end point
    """
    d13 = np.deg2rad( haversine(lon1, lat1, ptlon, ptlat) )
    a13 = np.deg2rad( azimuth(lon1, lat1, ptlon, ptlat) )
    a12 = np.deg2rad( azimuth(lon1, lat1, lon2, lat2) )
    dis = np.arcsin(np.sin(d13) * np.sin(a13-a12) )
    return np.rad2deg(dis)
def antipode(lon, lat):
    """
    calculate antipode of a point.
    """
    new_lon = (lon + 180) % 360
    new_lat = -lat
    return new_lon, new_lat
def great_circle_plane_center(lon1, lat1, lon2, lat2):
    """
    Calculate the center point coordinate (lon, lat) for the great circle plane.
    Note, two points will be returned.
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    x1, y1, z1 = np.cos(lat1)*cos(lon1), np.cos(lat1)*np.sin(lon1), np.sin(lat1)
    x2, y2, z2 = np.cos(lat2)*cos(lon2), np.cos(lat2)*np.sin(lon2), np.sin(lat2)
    x3, y3, z3 = y1*z2-y2*z1, z1*x2-z2*x1, x1*y2-x2*y1
    lat = np.arctan2(z3, np.sqrt(x3*x3+y3*y3))
    lon = np.arctan2(y3, x3)
    #print(x3, y3, z3)
    lon, lat = np.rad2deg(lon), np.rad2deg(lat)
    return (lon, lat), antipode(lon, lat)

###
#  geometry for tranformating spherical points into equator
###

def trans2equator(d1, d2, az1, az2, inter_station_dist, daz):
    """
    For two stations and one event that are nearly on great circle plane, 
    given two distances, azimuths, and inter station distance,  and maxmium 
    azimuth difference, transform their geometry into equator.
    return s1, s2, eqlo on equator
    """
    s2 = inter_station_dist*0.5 # make two stations on equator
    s1 = -s2
    eqlo = 0.0
    if (az1 - az2) < daz:
        eqlo = (s1-d1 if d2 > d1 else s2+d2) 
    else:
        eqlo = s2 - d2
    return s1%360, s2%360, eqlo%360

###
#  geometry for same az difference in X-Y plane
###
def __internel_line_same_daz_xy(x1, x2, angle_deg):
    """
    Get list of points for which the difference of direction to two fixed points (x1, 0), (x2, 0) is a  constant angle.
    """
    npts = 128
    if np.abs(angle_deg) <= 1.0e-6:
        angle_deg = 10
        c_tan = np.tan( np.deg2rad(angle_deg) )
        c_l   = x2 - x1
        c_l_tan = c_l / c_tan
        x_min = (1 - np.sqrt(1.0+1.0/c_tan/c_tan) )* c_l * 0.5 * 0.999
        x_max = (1 + np.sqrt(1.0+1.0/c_tan/c_tan) )* c_l * 0.5 * 0.999
        xs = np.linspace(x_min, x_max, npts) + x1
        ys = xs * 0.0
        return (xs.tolist(), ys.tolist()), ([], [])
    c_tan = np.tan( np.deg2rad(angle_deg) )
    c_l   = x2 - x1
    c_l_tan = c_l / c_tan
    x_min = (1 - np.sqrt(1.0+1.0/c_tan/c_tan) )* c_l * 0.5 * 0.999
    x_max = (1 + np.sqrt(1.0+1.0/c_tan/c_tan) )* c_l * 0.5 * 0.999
    lst_x1, lst_y1 = [], []
    for x in np.linspace(x_min, x_max, npts):
        y = (c_l_tan + np.sqrt(c_l_tan*c_l_tan-4.0*(x*x-x*c_l))) * 0.5
        x = x + x1
        lst_x1.append(x)
        lst_y1.append(y)
    lst_x2, lst_y2 = [], []
    for x in np.linspace(x_min, x_max, npts):
        y = (c_l_tan - np.sqrt(c_l_tan*c_l_tan-4.0*(x*x-x*c_l))) * 0.5
        x = x + x1
        lst_x2.append(x)
        lst_y2.append(y)
    lst_x2.reverse(), lst_y2.reverse()
    return (lst_x1, lst_y1), (lst_x2, lst_y2 )
def line_same_daz_xy(x1, y1, x2, y2, angle_deg):
    """
    Get list of points for which the difference of direction to two fixed points (x1, y1), (x2, y2) is a  constant angle.
    """
    belta = np.arctan2(y2-y1, x2-x1)
    c, s = np.cos(belta), np.sin(belta)
    x1_prime =  x1*c+y1*s
    y1_prime = -x1*s+y1*c
    x2_prime =  x2*c+y2*s
    y2_prime = -x2*s+y2*c
    (xs1_prime, ys1_prime), (xs2_prime, ys2_prime) = __internel_line_same_daz(x1_prime, x2_prime, angle_deg)
    xs1, ys1 = [], []
    for x_prime, y_prime in zip(xs1_prime, ys1_prime):
        y_prime = y_prime + y2_prime
        x = x_prime*c - y_prime*s
        y = x_prime*s + y_prime*c
        xs1.append(x)
        ys1.append(y)
    xs2, ys2 = [], []
    for x_prime, y_prime in zip(xs2_prime, ys2_prime):
        y_prime = y_prime + y2_prime
        x = x_prime*c - y_prime*s
        y = x_prime*s + y_prime*c
        xs2.append(x)
        ys2.append(y)
    return (xs1, ys1), (xs2, ys2)

###
#  geometry for same az difference in sphere
###
def __internel_line_same_daz_sphere(lo1, lo2, angle_deg):
    """
    Get list of points for which the difference of azimuth to two fixed points (lo1, 0), (lo2, 0) is a  constant angle.
    """
    npts = 2048
    if np.abs(angle_deg) <= 1.0e-6:
        lo_lst = np.linspace(-180, 180, npts)
        la_lst = lo_lst * 0.0
        return lo_lst.tolist(), la_lst.tolist()
    c_tan = np.tan(np.deg2rad(angle_deg) )
    rad1, rad2 = np.deg2rad(lo1), np.deg2rad(lo2)
    lo_lst, la_lst = [], []
    for lo in np.linspace(-np.pi, np.pi, npts):
        tmp = np.rad2deg(lo)
        t1 = np.tan(rad1 - lo)
        t2 = np.tan(rad2 - lo)
        d = (t1-t2)*(t1-t2) - 4.0*c_tan*c_tan*t1*t2
        if d < 0:
            # no results
            continue
        sin_phi1 = (t2-t1 - np.sqrt(d)) / (2.0*c_tan)
        sin_phi2 = (t2-t1 + np.sqrt(d)) / (2.0*c_tan)
        for value in [sin_phi1, sin_phi2]:
            if np.abs(value ) <= 1:
                phi = np.arcsin(value)
                lo_lst.append( np.rad2deg(lo) )
                la_lst.append( np.rad2deg(phi) )
    return lo_lst, la_lst
def internel_line_same_daz_sphere(lo1, la1, lo2, la2, angle_deg):
    """
    Get list of points for which the difference of azimuth to two fixed points (lo1, la1), (lo2, la2) is a  constant angle.
    """
    # Find two points in equator
    (a_lo, a_la, a_ang), junk = sphere_rotate_axis(lo1, la1, lo2, la2)
    new_lo1, junk = sphere_rotate(lo1, la1, a_lo, a_la, a_ang)
    new_lo2, junk = sphere_rotate(lo2, la2, a_lo, a_la, a_ang)
    new_lo_lst, new_la_lst = __internel_line_same_daz_sphere(new_lo1, new_lo2, angle_deg)
    lo_lst, la_lst = [], []
    for it_lo, it_la in zip(new_lo_lst, new_la_lst):
        v_lo, v_la = sphere_rotate(it_lo, it_la, a_lo, a_la, -a_ang)
        lo_lst.append(v_lo)
        la_lst.append(v_la)
    return lo_lst, la_lst

