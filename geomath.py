#!/usr/bin/env python3

"""
This contain sets of geometric methods, especially for spherical coordinates.
"""

import numpy as np
from math import radians #, cos, sin, asin, sqrt
from numba import jit, float64, types

np_deg2rad = np.deg2rad
PI = np.pi
twoPI = np.pi * 2
##################################################################################################################
# coordinates transformation
##################################################################################################################
@jit(nopython=True, nogil=True)
def antipode(lon, lat):
    """
    Calculate antipode of a point or a list of points.
    Return (lon, lat) of the new point in degree.
    The (lon, lat) could two single values or two np.ndarray objects of the same size.
    #
    NOTE: it is users' responsibility to make sure the input are not list/tuple/etc but np.ndarray when necessary.
    """
    new_lon = (lon + 180) % 360
    new_lat = -lat
    return new_lon, new_lat
@jit(nopython=True, nogil=True)
def antipode_rad(lon, lat):
    """
    The same as antiode, but input and output are in radian.
    """
    new_lon = (lon + PI) % twoPI
    new_lat = -lat
    return new_lon, new_lat

@jit(nopython=True, nogil=True)
def xyz_to_rlola(x, y, z):
    """
    Given (x, y, z), return (r, lo, la) in degree.
    The (x, y, z) could be three single values or three np.ndarray objects of the same size.
    #
    NOTE: it is users' responsibility to make sure the input are not list/tuple/etc but np.ndarray when necessary.
    """
    r  = np.sqrt( x*x+y*y+z*z )
    lo = np.rad2deg( np.arctan2(y, x) )
    la = np.rad2deg( np.arctan2(z, np.sqrt(x*x+y*y) ) )
    return r, lo, la
@jit(nopython=True, nogil=True)
def xyz_to_rlola_rad(x, y, z):
    """
    The same as xyz_to_rlola, but return in radian.
    """
    r  = np.sqrt( x*x+y*y+z*z )
    lo = np.arctan2(y, x)
    la = np.arctan2(z, np.sqrt(x*x+y*y) )
    return r, lo, la

@jit(nopython=True, nogil=True)
def rlola_to_xyz(radius, lo, la):
    """
    Given (radius, lo, la), return (x,y,z) coordinates. The lo and la are in degree.
    The (radius, lo, la) could be three single values or three np.ndarray objects of the same size.
    #
    NOTE: it is users' responsibility to make sure the input are not list/tuple/etc but np.ndarray when necessary.
    """
    lam, phi = np.deg2rad(lo), np.deg2rad(la)
    x = np.cos(phi)*np.cos(lam)*radius
    y = np.cos(phi)*np.sin(lam)*radius
    z = np.sin(phi)*radius
    return x,y,z
@jit(nopython=True, nogil=True)
def rlola_to_xyz_rad(radius, lo, la):
    """
    The same as rlola_to_xyz, but input and output are in radian.
    """
    lam, phi = lo, la
    x = np.cos(phi)*np.cos(lam)*radius
    y = np.cos(phi)*np.sin(lam)*radius
    z = np.sin(phi)*radius
    return x,y,z

##################################################################################################################
# spherical geometry
##################################################################################################################
@jit(nopython=True, nogil=True)
def haversine(lon1, lat1, lon2, lat2):
    """
    Return the great circle distance (degree) between two points.
    #
    The input could be single values(v) or np.ndarray objects(a). They could be:
    1. haversine(v1, v2, v3, v4)
    2. haversine(v1, v2, a1, a2) where a1 and a2 have the same size.
    2. haversine(a1, a2, v1, v2) where a1 and a2 have the same size.
    3. haversine(a1, a2, a3, a4) where all a1, a2, a3, a4 have the same size.
    #
    NOTE: it is users' responsibility to make sure the input are not list/tuple/etc but np.ndarray when necessary.
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = np_deg2rad(lon1), np_deg2rad(lat1), np_deg2rad(lon2), np_deg2rad(lat2)
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    s1 = np.sin(dlat*0.5)
    s2 = np.sin(dlon*0.5)
    a =  s1*s1 + np.cos(lat1) * np.cos(lat2) * s2 * s2
    c = np.rad2deg( 2.0 * np.arcsin(np.sqrt(a))  )
    return c # degree
@jit(nopython=True, nogil=True)
def haversine_rad(lon1, lat1, lon2, lat2):
    """
    The same as haversine, but input and output are in radian.
    """
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    s1 = np.sin(dlat*0.5)
    s2 = np.sin(dlon*0.5)
    a =  s1*s1 + np.cos(lat1) * np.cos(lat2) * s2 * s2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return c # rad

@jit(nopython=True, nogil=True)
def azimuth(evlo, evla, stlo, stla):
    """
    Return the azimuth angle (degree) from (evlo, evla) to (stlo, stla).
    #
    The input could be single values(v) or np.ndarray objects(a). They could be:
    1. azimuth(v1, v2, v3, v4)
    2. azimuth(v1, v2, a1, a2) where a1 and a2 have the same size.
    2. azimuth(a1, a2, v1, v2) where a1 and a2 have the same size.
    3. azimuth(a1, a2, a3, a4) where all a1, a2, a3, a4 have the same size.
    #
    NOTE: it is users' responsibility to make sure the input are not list/tuple/etc but np.ndarray when necessary.
    """
    phi1, phi2 = np.deg2rad(evla), np.deg2rad(stla)
    dlamda     = np.deg2rad(stlo - evlo)
    a = np.arctan2(np.cos(phi2) * np.sin(dlamda),  np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlamda) )
    return np.rad2deg(a) % 360.0
@jit(nopython=True, nogil=True)
def azimuth_rad(evlo, evla, stlo, stla):
    """
    The same as azimuth, but input and output are in radian.
    """
    phi1, phi2 = evla, stla
    dlamda     = stlo - evlo
    a = np.arctan2(np.cos(phi2) * np.sin(dlamda),  np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlamda) )
    a = a % twoPI
    return a

@jit(nopython=True, nogil=True)
def inter_point_distance_mat(los, las):
    """
    Return an N by N matrix for inter spherical point distance.
    The matrix is symmetric.
    #
    NOTE: it is users' responsibility to make sure the input are not list/tuple/etc but np.ndarray when necessary.
    """
    n = len(los)
    dist_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        dist_mat[i] = haversine( los[i], las[i], los, las )
    return dist_mat
##################################################################################################################
# Complex geometry
##################################################################################################################
@jit(nopython=True, nogil=True)
def point_distance_to_great_circle_plane(ptlon, ptlat, lon1, lat1, lon2, lat2):
    """
    Return the distance (can be positive or negative) from a point (ptlon, ptlat) to the great circle plane formed by two points
    of (lon1, lat1) and (lon2, lat2). Input and output are in degree.

    The input could be single values(v) or np.ndarray objects(a). They could be:
    1. point_distance_to_great_circle_plane(v1, v2, v3, v4, v5, v6)

    2. point_distance_to_great_circle_plane(v1, v2, v3, v4, a1, a2) where a1 and a2 have the same size.
    3. point_distance_to_great_circle_plane(v1, v2, a1, a2, v3, v4) where a1 and a2 have the same size.
    4. point_distance_to_great_circle_plane(a1, a2, v1, v2, v3, v4) where a1 and a2 have the same size.

    5. point_distance_to_great_circle_plane(v1, v2, a1, a2, a3, a4) where all a1, a2, a3, a4 have the same size.
    6. point_distance_to_great_circle_plane(a1, a2, a3, a4, a5, a6) where all a1, a2, a3, a4, a5, a6 have the same size.
    No other input format options are allowed.
    #
    NOTE: it is users' responsibility to make sure the input are not list/tuple/etc but np.ndarray when necessary.
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
    ###
    loladif = np.abs(lon1-lon2) + np.abs(lat1-lat2)
    dis = np.where(loladif<1.0e-4, 0.0, dis)
    return np.rad2deg(dis)
@jit(nopython=True, nogil=True)
def point_distance_to_great_circle_plane_rad(ptlon, ptlat, lon1, lat1, lon2, lat2):
    """
    The same as point_distance_to_great_circle_plane, but input and output are in radian.
    """
    d13 = haversine_rad(lon1, lat1, ptlon, ptlat)
    a13 = azimuth_rad(lon1, lat1, ptlon, ptlat)
    a12 = azimuth_rad(lon1, lat1, lon2, lat2)
    dis = np.arcsin(np.sin(d13) * np.sin(a13-a12) )
    ###
    loladif = np.abs(lon1-lon2) + np.abs(lat1-lat2)
    dis = np.where(loladif<1.0e-6, 0.0, dis)
    return dis

@jit(nopython=True, nogil=True)
def great_circle_plane_center(lon1, lat1, lon2, lat2):
    """
    Return two center points coordinate (clon1, clat1), (clon2, clat2) for the great circle plane formed by
    the two input points (lon1, lat1) and (lon2, lat2). Input and output are in degree.
    #
    The input could be single values(v) or np.ndarray objects(a). They could be:
    1. great_circle_plane_center(v1, v2, v3, v4)
    2. great_circle_plane_center(v1, v2, a1, a2) where a1 and a2 have the same size.
    2. great_circle_plane_center(a1, a2, v1, v2) where a1 and a2 have the same size.
    3. great_circle_plane_center(a1, a2, a3, a4) where all a1, a2, a3, a4 have the same size.
    #
    NOTE: it is users' responsibility to make sure the input are not list/tuple/etc but np.ndarray when necessary.
    """
    x1, y1, z1 = rlola_to_xyz(1.0, lon1, lat1)
    x2, y2, z2 = rlola_to_xyz(1.0, lon2, lat2)
    x3, y3, z3 = y1*z2-y2*z1, z1*x2-z2*x1, x1*y2-x2*y1
    lat = np.arctan2(z3, np.sqrt(x3*x3+y3*y3))
    lon = np.arctan2(y3, x3)
    #print(x3, y3, z3)
    lon, lat = np.rad2deg(lon) % 360.0, np.rad2deg(lat)
    #### now make sure the center points are in the northern hemisphere
    alon, alat = antipode(lon, lat)
    lon = np.where(lat>0.0, lon, alon)
    lat = np.where(lat>0.0, lat, alat)
    ####
    alon, alat = antipode(lon, lat)
    return (lon, lat), (alon, alat)
@jit(nopython=True, nogil=True)
def great_circle_plane_center_rad(lon1, lat1, lon2, lat2):
    """
    The same as great_circle_plane_center, but input and output are in radian.
    """
    x1, y1, z1 = rlola_to_xyz_rad(1.0, lon1, lat1)
    x2, y2, z2 = rlola_to_xyz_rad(1.0, lon2, lat2)
    x3, y3, z3 = y1*z2-y2*z1, z1*x2-z2*x1, x1*y2-x2*y1
    lat = np.arctan2(z3, np.sqrt(x3*x3+y3*y3))
    lon = np.arctan2(y3, x3) % twoPI
    #print(x3, y3, z3)
    #### now make sure the center points are in the northern hemisphere
    alon, alat = antipode_rad(lon, lat)
    lon = np.where(lat>0.0, lon, alon)
    lat = np.where(lat>0.0, lat, alat)
    ####
    alon, alat = antipode_rad(lon, lat)
    return (lon, lat), (alon, alat)

@jit(nopython=True, nogil=True)
def great_circle_plane_center_triple(lon1, lat1, lon2, lat2, ptlo, ptla, critical_distance):
    """
    Return great_circle_plane_center_triple(lon1, lat1, lon2, lat2) is the two points #1 and #2 are far away (distance above `critical_distance`)
    or great_circle_plane_center_triple(lon1, lat1, ptlo, ptla) if the two points #1 and #2 are too close (distance below `critical_distance`).
    """
    dist = haversine(lon1, lat1, lon2, lat2)
    lon1 = np.where(dist>critical_distance, lon1, ptlo)
    lat1 = np.where(dist>critical_distance, lat1, ptla)
    return great_circle_plane_center(lon1, lat1, lon2, lat2)
@jit(nopython=True, nogil=True)
def great_circle_plane_center_triple_rad(lon1, lat1, lon2, lat2, ptlo, ptla, critical_distance):
    """
    The same as great_circle_plane_center_triple, but input and output are in radian.
    """
    dist = haversine_rad(lon1, lat1, lon2, lat2)
    lon1 = np.where(dist>critical_distance, lon1, ptlo)
    lat1 = np.where(dist>critical_distance, lat1, ptla)
    return great_circle_plane_center_rad(lon1, lat1, lon2, lat2)

##################################################################################################################
# Spherical declustering
##################################################################################################################
def decluster_spherical_pts(los_deg, las_deg, approximate_lo_dif=2, approximate_la_dif=2, key_type='float'):
    """
    Decluster a list of spherical points
    Return a dictionary
    (lo, la) -> a list of indexes
    where the (lo, la) is the grid center for a cluster of points.
    #
    los_deg, las_deg: the longitude and latitude of the points.
    approximate_lo_dif, approximate_la_dif: the approximate longitude and latitude step for clustering.
    key_type: the type of key, 'int' or 'float'. For 'float', the key will be
              meaningful (latitude, longitude) pair. For 'int', the key will be
              some integer values which is useful for further cluster operations.
    #
    NOTE: the `decluster_spherical_pts` has the following two methods:
          lalo2intkey(los_deg, las_deg, approximate_lo_dif, approximate_la_dif) --> [intkey1, intkey2, ...]
          intkey2lalo(intkeys, approximate_lo_dif, approximate_la_dif) --> las, los
    #
    The algorithm is like this:
    Lets say given a point lo, la, and step dlo, dla.
    First, we round the latitude, and obtain the int key for latitude.
        la2 = np.round(la / dla) * dla
        int_la = (la2 * 1000).astype(np.int64)
    Second, we adjust the longitude step, so that the 0->360 range can be evened dividied by dlo.
        dlo2 = dlo / np.cos( la2_in_rad )
        n = np.round(360 / dlo2)
        dlo3 = 360 / n
        lo2 = np.floor(lo / dlo3) # here we use the floor to make every longitude starts from 0 degree
        int_lo = (lo2 * 1000).astype(np.int64)
    Third, we have obtained the key (int_lo, int_la) for the point (lo, la), and we can use
    the key to cluster the points.
    """
    def lalo2intkey(los_deg, las_deg, approximate_lo_dif=2, approximate_la_dif=2):
        los = np.array(los_deg, dtype=np.float64) % 360
        las = np.array(las_deg, dtype=np.float64)
        ######
        dlo, dla = approximate_lo_dif, approximate_la_dif
        la2 = np.round(las / dla) * dla
        int_la = (la2 * 1000).astype(np.int64)
        #
        dlo2 = dlo / np.cos(np.deg2rad(la2))
        n = np.round(360 / dlo2)
        n = np.where(n == 0, 1, n) # replace the zeros within an array, n, to 1
        dlo3 = 360 / n
        lo2 = np.floor(los / dlo3)
        int_lo = (lo2 * 1000).astype(np.int64)
        ######
        intkey = [it for it in zip(int_la, int_lo)]
        return intkey
    def intkey2lalo(intkeys, approximate_lo_dif=2, approximate_la_dif=2):
        try:
            int_la = np.array([it[0] for it in intkeys] )
            int_lo = np.array([it[1] for it in intkeys] )
        except Exception as e:
            int_la, int_lo = intkeys
        ######
        dlo, dla = approximate_lo_dif, approximate_la_dif
        ######
        la2 = int_la * 0.001
        las = la2
        ######
        dlo2 = dlo / np.cos(np.deg2rad(la2))
        n = np.round(360 / dlo2)
        n = np.where(n == 0, 1, n) # replace the zeros within an array, n, to 1
        dlo3 = 360 / n
        #
        lo2 = int_lo * 0.001
        los = (lo2+0.5) * dlo3
        return las, los%360
    ####
    decluster_spherical_pts.lalo2intkey = lalo2intkey
    decluster_spherical_pts.intkey2lalo = intkey2lalo
    ####
    idxs    = np.arange( len(los_deg) )
    intkeys = lalo2intkey(los_deg, las_deg, approximate_lo_dif, approximate_la_dif)
    ####
    tmp_dict = {it: set() for it in intkeys}
    for intkey, idx in zip(intkeys, idxs):
        tmp_dict[intkey].add(idx)
    ####
    declustered_points = dict()
    if key_type == 'float':
        for intkey, set_idxs in tmp_dict.items():
            this_idxs = sorted(set_idxs)
            key = intkey2lalo(intkey, approximate_lo_dif, approximate_la_dif)
            declustered_points[key] = this_idxs
    elif key_type == 'int':
        for intkey, set_idxs in tmp_dict.items():
            this_idxs = sorted(set_idxs)
            declustered_points[intkey] = this_idxs
    else:
        raise ValueError(f"Unknown key_type: {key_type}")
    return declustered_points

##################################################################################################################
# Functions/methods below are for special purpose
##################################################################################################################
### frame rotation
@jit(nopython=True, nogil=True)
def sphere_rotate_axis(lo1, la1, lo2, la2):
    """
    Get point cooridinates and right-hand rotation angle, with which after rotation the great circle formed by two fixed points
    (lo1, la1) and (lo2, la2) turns into equator.
    #
    Two antipodal points, each of which has a special rotation angle, will be returned.
    (lo1, la1, angle1), (lo2, la2, angle2)
    """
    lam1, phi1, lam2, phi2 = radians(lo1), radians(la1), radians(lo2), radians(la2)
    c_1 = np.sin(lam1)/np.tan(phi1) - np.sin(lam2)/np.tan(phi2)
    c_2 = np.cos(lam1)/np.tan(phi1) - np.cos(lam2)/np.tan(phi2)
    lo1 = np.rad2deg( np.arctan2(c_1, c_2) )
    lo2, junk = antipode(lo1, 0.0)
    return_value = []
    for lo in [lo1, lo2]:
        lam_tmp = radians(lo + 90)
        c_1 = np.tan(phi1)/np.sin(lam1-lam_tmp) - np.tan(phi2)/np.sin(lam2-lam_tmp)
        c_2 = 1.0/np.tan(lam1-lam_tmp) - 1.0/np.tan(lam2-lam_tmp)
        rotate_angle = -np.rad2deg( np.arctan2(c_1, c_2) )
        return_value.append( (lo, 0, rotate_angle) )
    return return_value

@jit(nopython=True, nogil=True)
def sphere_rotate(lo, la, axis_lo, axis_la, axis_angle_deg):
    """
    Rotate (lo, la) with given axis directin and rotation angle. 
    Return new coordinate (new_lo, new_la) after rotation.
    #
    v_r = v cos(a) + k x v sin(a) + k (k dot v) (1-cos (a) )
    """
    v = np.array( rlola_to_xyz(1.0, lo, la))
    k = np.array( rlola_to_xyz(1.0, axis_lo, axis_la) )
    a = radians(axis_angle_deg)
    c, s = np.cos(a), np.sin(a)
    v_r = v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1.0-c)
    x, y, z = v_r
    junk, new_lo, new_la= xyz_to_rlola(x, y, z)    
    return new_lo, new_la


##################################################################################################################
#  geometry for tranformating spherical points into equator
##################################################################################################################
@jit(nopython=True, nogil=True)
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
@jit(nopython=True, nogil=True)
def __internel_line_same_daz_xy(x1, x2, angle_deg):
    """
    Get list of points for which the difference of direction to two fixed points (x1, 0), (x2, 0) is a  constant angle.
    """
    npts = 128
    if np.abs(angle_deg) <= 1.0e-6:
        angle_deg = 10
        c_tan = np.tan( radians(angle_deg) )
        c_l   = x2 - x1
        c_l_tan = c_l / c_tan
        x_min = (1 - np.sqrt(1.0+1.0/c_tan/c_tan) )* c_l * 0.5 * 0.999
        x_max = (1 + np.sqrt(1.0+1.0/c_tan/c_tan) )* c_l * 0.5 * 0.999
        xs = np.linspace(x_min, x_max, npts) + x1
        ys = xs * 0.0
        return (xs, ys), (np.array(0), np.array(0) )
    c_tan = np.tan( radians(angle_deg) )
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
@jit(nopython=True, nogil=True)
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
@jit(nopython=True, nogil=True)
def __internel_line_same_daz_sphere(lo1, lo2, angle_deg):
    """
    Get list of points for which the difference of azimuth to two fixed points (lo1, 0), (lo2, 0) is a  constant angle.
    """
    npts = 2048
    if np.abs(angle_deg) <= 1.0e-6:
        lo_lst = np.linspace(-180, 180, npts)
        la_lst = lo_lst * 0.0
        return lo_lst.tolist(), la_lst.tolist()
    c_tan = np.tan(radians(angle_deg) )
    rad1, rad2 = radians(lo1), radians(lo2)
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
#@jit( types.UniTuple(float64[:],2)(float64, float64, float64, float64, float64), nopython=True, nogil=True)
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
    return np.array(lo_lst), np.array(la_lst)

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    los = np.arange(0, 360.01, 0.5)
    las = np.arange(-90, 90.01, 0.5)
    mat = np.meshgrid(los, las)
    los = mat[0].flatten()
    las = mat[1].flatten()
    print(los.shape)
    clusters = decluster_spherical_points(los, las, approximate_lo_dif=20, approximate_la_dif=20)
    ax = plt.subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
    ax.set_global()
    ax.add_feature(cfeature.LAND, color='#cccccc')
    for (la, lo), idxs in sorted(clusters.items()):
        xs = los[idxs]
        ys = las[idxs]
        ls, = ax.plot(xs, ys, '.', markersize=0.6, transform=ccrs.PlateCarree() )
        ###
        ax.plot(lo, la, 's', markersize=10, color='k', transform=ccrs.PlateCarree() )
    plt.show()
    sys.exit()

    lo0, la0 = -1, -10
    lo1, la1 = 10, 60
    lo2, la2 = 30, 20
    lo3, la3 = np.array([0, 10, 20]), np.array([0, 0, 0])
    lo4, la4 = lo3 + 10, la3 + 10
    rlo0, rla0 = np.deg2rad(lo0), np.deg2rad(la0)
    rlo1, rla1 = np.deg2rad(lo1), np.deg2rad(la1)
    rlo2, rla2 = np.deg2rad(lo2), np.deg2rad(la2)
    rlo3, rla3 = np.deg2rad(lo3), np.deg2rad(la3)
    rlo4, rla4 = np.deg2rad(lo4), np.deg2rad(la4)

    x1, y1, z1 = 1, 1, 1
    x2, y2, z2 = np.array([1, 2]), np.array([1, 3]), np.array([1, 4])

    #print(antipode(lo1, la1) ); print( np.rad2deg(antipode_rad(rlo1, rla1)) )
    #print(antipode(lo3, la3) ); print( np.rad2deg(antipode_rad(rlo3, rla3)) )
    #
    #print( xyz_to_rlola(x1, y1, z1) ); print( np.rad2deg(xyz_to_rlola_rad(x1, y1, z1)) )
    #print( xyz_to_rlola(x2, y2, z2) ); print( np.rad2deg(xyz_to_rlola_rad(x2, y2, z2)) )
    #
    #print( rlola_to_xyz(10, lo1, la1) ); print( rlola_to_xyz_rad(10, rlo1, rla1))
    #print( rlola_to_xyz(10, lo3, la3) ); print( rlola_to_xyz_rad(10, rlo3, rla3))
    #print( rlola_to_xyz(10, lo1, la1) ); print( rlola_to_xyz_rad(10, rlo1, rla1))
    #
    #print(haversine(lo1, la1, lo2, la2) ); print( np.rad2deg(haversine_rad(rlo1, rla1, rlo2, rla2)) )
    #print(haversine(lo3, la3, lo4, la4) ); print( np.rad2deg(haversine_rad(rlo3, rla3, rlo4, rla4)) )
    #print(haversine(lo1, la1, lo4, la4) ); print( np.rad2deg(haversine_rad(rlo1, rla1, rlo4, rla4)) )
    #print(haversine(lo3, la3, lo2, la2) ); print( np.rad2deg(haversine_rad(rlo3, rla3, rlo2, rla2)) )
    #
    #print(azimuth(lo1, la1, lo2, la2) ); print( np.rad2deg(azimuth_rad(rlo1, rla1, rlo2, rla2)) )
    #print(azimuth(lo3, la3, lo4, la4) ); print( np.rad2deg(azimuth_rad(rlo3, rla3, rlo4, rla4)) )
    #print(azimuth(lo1, la1, lo4, la4) ); print( np.rad2deg(azimuth_rad(rlo1, rla1, rlo4, rla4)) )
    #print(azimuth(lo3, la3, lo2, la2) ); print( np.rad2deg(azimuth_rad(rlo3, rla3, rlo2, rla2)) )
    #
    #print(point_distance_to_great_circle_plane(lo0, la0, lo1, la1, lo2, la2) ); print(np.rad2deg(point_distance_to_great_circle_plane_rad(rlo0, rla0, rlo1, rla1, rlo2, rla2) ) )
    #print(point_distance_to_great_circle_plane(lo0, la0, lo1, la1, lo1, la1) ); print(np.rad2deg(point_distance_to_great_circle_plane_rad(rlo0, rla0, rlo1, rla1, rlo1, rla1) ) )
    #
    #print(point_distance_to_great_circle_plane(lo0, la0, lo1, la1, lo3, la3) ); print(np.rad2deg(point_distance_to_great_circle_plane_rad(rlo0, rla0, rlo1, rla1, rlo3, rla3) ) )
    #print(point_distance_to_great_circle_plane(lo0, la0, lo3, la3, lo3, la3) ); print(np.rad2deg(point_distance_to_great_circle_plane_rad(rlo0, rla0, rlo3, rla3, rlo3, rla3) ) )
    #print(point_distance_to_great_circle_plane(lo0, la0, lo3, la3, lo1, la1) ); print(np.rad2deg(point_distance_to_great_circle_plane_rad(rlo0, rla0, rlo3, rla3, rlo1, rla1) ) )
    #print(point_distance_to_great_circle_plane(lo0, la0, lo3, la3, lo4, la4) ); print(np.rad2deg(point_distance_to_great_circle_plane_rad(rlo0, rla0, rlo3, rla3, rlo4, rla4) ) )
    #
    #print(great_circle_plane_center(lo1, la1, lo2, la2) ); print( np.rad2deg(great_circle_plane_center_rad(rlo1, rla1, rlo2, rla2)) )
    #print(great_circle_plane_center(lo3, la3, lo4, la4) ); print( np.rad2deg(great_circle_plane_center_rad(rlo3, rla3, rlo4, rla4)) )
    #print(great_circle_plane_center(lo1, la1, lo4, la4) ); print( np.rad2deg(great_circle_plane_center_rad(rlo1, rla1, rlo4, rla4)) )
    print(great_circle_plane_center(lo3, la3, lo2, la2) ); print( np.rad2deg(great_circle_plane_center_rad(rlo3, rla3, rlo2, rla2)) )