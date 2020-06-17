#!/usr/bin/env python3
import numpy as np
from mayavi import mlab
import shapefile
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

###
#  Part to plot 3-D Earth basemap elements
###
basemap_color  = (0.7, 0.7, 0.7)
baseline_color = (0.9, 0.9, 0.9)
def geo2xyz(lon, lat, depth, R0=6371.0):
    r = R0-depth
    theta = np.deg2rad( np.array( deepcopy(lon) ) )
    phi   = np.deg2rad( 90.0 - np.array(lat) )
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)
    return x, y, z
def continent_boundary_segments(jump = 5, R0=6371.0):
    sf = shapefile.Reader("/home/catfly/Program_Sheng/sacpy/tomo/data/continent.shp")
    segments = []
    for it in sf.shapes():
        bounds = list(it.parts)
        bounds.append(-1)
        for i0, i1 in zip(bounds[0:-1], bounds[1:] ):
            lons = np.array( [pt[0] for pt in it.points[i0:i1:jump] ] )
            lats = np.array( [pt[1] for pt in it.points[i0:i1:jump] ] )
            depths   = np.zeros(lons.size)
            xs, ys, zs = geo2xyz(lons, lats, depths, R0)
            segments.append( {'lon':lons, 'lat': lats, 'x': xs, 'y': ys , 'z': zs } )
    return segments
def continent_boundary_points(jump = 5, R0=6371.0):
    sf = shapefile.Reader("/home/catfly/Program_Sheng/sacpy/tomo/data/continent.shp")
    lons, lats = [], []
    for it in sf.shapes():
        itlons = np.array( [pt[0] for pt in it.points[::jump] ] )
        itlats = np.array( [pt[1] for pt in it.points[::jump] ] )
        lons.extend(itlons)
        lats.extend(itlats)
    lons, lats = np.array(lons), np.array(lats)
    depths   = np.zeros(lons.size)
    xs, ys, zs = geo2xyz(lons, lats, depths, R0)
    return lons, lats, depths, xs, ys, zs
def plot_Earth_meridian(d_lon=30.0, min_lon= 0.0, max_lon=360.0, R0=6371.0 ):
    meridians = np.arange(min_lon, max_lon, d_lon)
    lats = np.arange(-90, 90, 0.5)
    depths = np.zeros(lats.size)
    for it in meridians:
        lons = np.zeros(lats.size) + it
        xs, ys, zs = geo2xyz(lons, lats, depths, R0)
        mlab.plot3d(xs, ys, zs, color=basemap_color, tube_radius=None, line_width= 1, opacity= 0.6 )
    pass
def plot_Earth_latitude(d_lat=30.0, min_lat=-90.0, max_lat=90.0, R0=6371.0 ):
    latitudes = np.arange(min_lat, max_lat, d_lat)
    lons = np.arange(0, 360, 0.5)
    depths = np.zeros(lons.size)
    for it in latitudes:
        lats = np.zeros(lons.size) + it
        xs, ys, zs = geo2xyz(lons, lats, depths, R0 )
        mlab.plot3d(xs, ys, zs, color=basemap_color, tube_radius=None, line_width= 1, opacity= 0.6 )
    pass
def plot_continent_boundary():
    lons, lats, depths, xs, ys, zs = continent_boundary_points(1)
    mlab.points3d(xs, ys, zs, color= baseline_color, mode='point', opacity= 0.7, scale_factor= 100000)
    #segments = continent_boundary_segments(20)
    #for it in segments:
    #    xs, ys, zs = it['x'], it['y'], it['z']
    #    mlab.plot3d(xs, ys, zs, color=(0.6, 0.6, 0.6), tube_radius=None, line_width= 2, opacity= 0.6 )
def plot_sphere(d_lon=2.0, d_lat=2.0, radius=3479.5, R0=6371.0, color= basemap_color):
    # CMB 2891.5 3479.5
    # ICB 5153.5 1217.5
    # print("Inside plot_sphere", radius)
    lons = []
    lats = []
    for lat in np.arange(-90.0, 90.0, d_lat):
        it_lons = np.arange(0.0, 360.0, d_lon)
        it_lats = np.zeros(it_lons.size) + lat
        lons.extend(it_lons)
        lats.extend(it_lats)
    lons, lats = np.array(lons), np.array(lats)
    #rs   = np.zeros(lons.size) + radius
    depth = np.zeros(lons.size) + R0 - radius
    xs, ys, zs = geo2xyz(lons, lats, depth, R0)
    mlab.points3d(xs, ys, zs, color= basemap_color, mode='point', opacity= 0.7, scale_factor= 100000)
def plot_Earth_pretty():
    plot_Earth_meridian()
    plot_Earth_latitude()
    plot_continent_boundary()
    # CMB
    #plot_Earth_meridian(R0=3479.5)
    #plot_Earth_latitude(R0=3479.5)
    print("here")
    plot_sphere(2.0, 2.0, 3479.5)
    # ICB
    #plot_Earth_meridian(R0=1217.5)
    #plot_Earth_latitude(R0=1217.5)
    plot_sphere(3.0, 3.0, 1217.5)


###
#  Part to plot 3-D grid points that represent P and S velocity structure
###
def open_mod3d(filename):
    vol = np.loadtxt(filename, comments='#', 
                dtype={ 'names':   ('index', 'lon', 'lat', 'depth', 'x', 'y', 'z', 'vp', 'vs', 'rho', 'dvp', 'dvs' ),
                        'formats': ('i4',    'f4',  'f4',  'f4',    'f4','f4','f4','f4', 'f4', 'f4',  'f4',  'f4' )  }   )
    vol['vp']  *= vol['dvp']
    vol['vs']  *= vol['dvs']
    vol['dvp'] -= 1.0
    vol['dvs'] -= 1.0
    return vol
def plot_mod3d(x, y, z, v, vmin=None, vmax=None, title='', colormap='RdYlBu', reverse_colorbar=False, dvel= 0.2):
    m1 = mlab.points3d(x, y, z, v, colormap=colormap, mode='point', opacity= 0.6, transparent=True, scale_factor= 5000 )
    if reverse_colorbar:
        lut = m1.module_manager.scalar_lut_manager.lut.table.to_array()
        ilut = lut[::-1]
        m1.module_manager.scalar_lut_manager.lut.table = ilut
        mlab.draw()
    #################
    nb_labels = int((vmax-vmin)/dvel + 1)
    #bar = mlab.colorbar(m1, orientation='horizontal' , title= 'dvs (percent )', nb_labels= 18)
    bar = mlab.colorbar(m1, orientation='horizontal' , title= title, nb_labels= nb_labels, label_fmt='%.3f' )
    bar.data_range = [vmin, vmax]
    #pass
    return m1, bar


###
#  Part to assemble everything
###
def plot_ray_path(lon, lat, dep, color=(1, 1, 1) ):
    xs, ys, zs = geo2xyz(lon, lat, dep)
    mlab.plot3d(xs, ys, zs, color=color, tube_radius=None, line_width= 2, opacity= 0.7 )
def plot_point_sphere(lon, lat, dep, color=(1, 1, 1) ):
    xs, ys, zs = geo2xyz(lon, lat, dep)
    mlab.points3d(xs, ys, zs, color=color, mode='sphere', opacity= 0.7, scale_factor= 100) 


def plot_absolute_earth_model(fnm, type='vp'):
    m = open_mod3d(fnm)
    x, y, z, dv = m['x'], m['y'], m['z'], m[type]
    vmin, vmax, dvel = 5.5, 14.0, 0.5
    if type == 'vs':
        vmin, vmax, dvel = 3.0, 7.5, 0.5
    plot_mod3d(x, y, z, dv, vmin= vmin, vmax= vmax, title= '%s (km/s)' % (type) , colormap='gist_heat', dvel= dvel )
    ###
    plot_Earth_pretty()
def plot_relative_earth_model(fnm, vmin, vmax, dvel, type='dvp'):
    m = open_mod3d(fnm)
    x, y, z, dv = m['x'], m['y'], m['z'], m[type]
    plot_mod3d(x, y, z, dv, vmin= vmin, vmax= vmax, title= '%s ' % (type) , colormap='RdYlBu', dvel= dvel )
    ###
    plot_Earth_pretty()

if __name__ == "__main__":
    #######
    plot_Earth_pretty()
    #######
    fnm = 'earth1d.txt'
    vmin, vmax = ()
    m = open_mod3d(fnm)
    x, y, z, dv = m['x'], m['y'], m['z'], m['dvs']
    plot_mod3d(x, y, z, dv, vmin= vmin, vmax= vmax, title= 'dvs (percent )', colormap='RdYlBu')
    #print(mod3d[1], mod3d[-3:])
    mlab.view(0.0, 60.0 )
    mlab.show()