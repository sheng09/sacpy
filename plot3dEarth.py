#!/usr/bin/env python3

# sphinx_gallery_thumbnail_number = 1
from glob import glob
from PIL.Image import radial_gradient
from matplotlib.colors import LightSource
from numpy.core.numeric import ones
import pyvista as pv
from pyvista import examples as pv_examples

import numpy as np
import pickle
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import obspy.taup as taup

from os.path import abspath as os_path_abspath
from os.path import exists as os_path_exists

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys


class globe3d:
    """
    Geometric data and methods for a 3D globe.
    """
    def __init__(self, radius = 6371.0, center=(0, 0, 0) ):
        """
        radius:
        center: (0, 0, 0) in default
        """
        self.center = center
        self.radius = radius
    def point_to_xyz(self, longitude, latitude, depth):
        """
        Return the (x, y, z) given a point (longitude, latitude, depth) relative to the globe.
        """
        cx, cy, cz = self.center
        r = self.radius-depth
        theta = np.deg2rad(longitude)
        phi = np.deg2rad(90.0-latitude)
        x = r * np.sin(phi) * np.cos(theta) + cx
        y = r * np.sin(phi) * np.sin(theta) + cy
        z = r * np.cos(phi) + cz
        return x, y, z
    def point_to_vec(self, longitude, latitude, depth):
        """
        Return a vector with the start point at (x, y, z) and the unit direction (vx, vy, vz).
        The vector is from the center of the globe to the point (longitude, latitude, depth) relative to the globe.
        """
        theta = np.deg2rad(longitude)
        phi = np.deg2rad(90.0-latitude)

        vx = np.sin(phi) * np.cos(theta)
        vy = np.sin(phi) * np.sin(theta)
        vz = np.cos(phi)
        x, y, z = self.point_to_xyz(longitude, latitude, depth)
        return (x, y, z), (vx, vy, vz)


def Earth_radial_model(mod='ak135', key='vp'):
    """
    key: any of 'vp', 'vs', 'density', 'qp', 'qs'.
    """
    model = taup.TauPyModel(mod)
    layers = model.model.s_mod.v_mod.layers
    rs = model.model.radius_of_planet - layers['bot_depth']
    tmp = {'vp':'bot_p_velocity', 'vs':'bot_s_velocity', 'density':'bot_density', 'qp':'bot_qp', 'qs':'bot_qs'}
    vel = layers[tmp[key]]
    rs[0] = 6800
    rs[-1] = -10
    return rs, vel

def get_global_texture(style='simple', coastline=False, land=None, ocean=None):
    """
    """
    figs = get_global_map(style, coastline, land, ocean)
    tex_pair = [None, None]
    for idx, it in enumerate(figs):
        #tex_file = it.replace('.png', '_texture.pkl')
        tex_pair[idx] = pv.read_texture(it)
    return tex_pair
def get_global_map(style='simple', coastline=False, land=None, ocean=None):
    """
    style:  styles for plotting the global map:
            'fancy1'
            'fancy2'
            'simple'
            'Mars'
            'Cat1'
    """
    loc = '%s/%s' % ('/'.join(os_path_abspath(__file__).split('/')[:-1] ), 'dataset/global_maps')

    if style == 'simple':
        figname1 = '%s/%s_coastline-%s_land-%s_ocean-%s_0-360.png' % (loc, style, coastline, land, ocean)
        figname2 = '%s/%s_coastline-%s_land-%s_ocean-%s_-180-180.png' % (loc, style, coastline, land, ocean)
        if (not os_path_exists(figname1)) or (not os_path_exists(figname2)):
            plot_global_map((figname1, figname2), style, coastline, land, ocean)
    elif style == 'fancy2':
        figname1 = '%s/%s_0-360.png' % (loc, style)
        figname2 = '%s/%s_-180-180.png' % (loc, style)
        if (not os_path_exists(figname1)) or (not os_path_exists(figname2)):
            plot_global_map((figname1, figname2), style, coastline, land, ocean)
    elif style == 'fancy1':
        figname1 = pv_examples.mapfile
        figname2 = pv_examples.mapfile
    elif style == 'Mars':
        figname1 = '%s/Mars.jpg' % loc
        figname2 = '%s/Mars.jpg' % loc
    elif style == 'Cat1':
        figname1 = '%s/Cat1.jpeg' % loc
        figname2 = '%s/Cat1.jpeg' % loc
    elif style == 'Mosaic':
        figname1 = '%s/Mosaic.jpeg' % loc
        figname2 = '%s/Mosaic.jpeg' % loc
        #plot_mosaic_map(figname1)
    else:
        print('Wrong style for get_global_map(...)', style)
        sys.exit(-1)
    return (figname1, figname2)
def plot_global_map(files, style='simple', coastline=False, land=None, ocean=None):
    """
    Plot two glolab map, the 1st has longitude range [0, 360] degree, and the 2nd [-180, 180] degree.
    style:  styles for plotting the global map:
            'fancy2'
            'simple'
    """
    prj1 = ccrs.PlateCarree(central_longitude=180)
    prj2 = ccrs.PlateCarree(central_longitude=0)
    for prj, figname in zip((prj1, prj2), files):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 1, 1, projection=prj)
        ax.set_global()
        if style == 'fancy2':
            ax.stock_img()
        if coastline:
            ax.coastlines(linewidth=0.8)
        if land != None:
            ax.add_feature(cfeature.LAND, color=land, alpha=1.0)
        if ocean !=None:
            ax.add_feature(cfeature.OCEAN, color=ocean, alpha=1.0)

        ax.axis('off')
        plt.savefig(figname, bbox_inches = 'tight', pad_inches = 0, dpi=300, transparent=True)
        plt.close()
def plot_mosaic_map(figname):
    mat = np.random.random((1000, 1000) )-0.5
    mat = gaussian_filter(mat, 3)
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.imshow(mat, cmap='gray') #, vmax=0.5, vmin=-1)
    ax.axis('off')
    plt.savefig(figname, bbox_inches = 'tight', pad_inches = 0, dpi=300, transparent=True)
    pass
#############################################################################################################################################################################################
# Functions for plotting globe and great-circle planes
#############################################################################################################################################################################################
def plot_globe3d(p, globe, style='simple', coastline=False, land=None, ocean=None, culling=None, alpha=1.0, color='gray', clip=None):
    """
    """
    ## first part of the sphere
    #sphere = pv.Sphere(radius=radius, center=(0, 0, 0), theta_resolution=60, phi_resolution=30, start_theta=0, end_theta=359.99999999, direction=(0, 0, -1) )
    #sphere.t_coords = np.zeros((sphere.points.shape[0], 2))
    #for i in range(sphere.points.shape[0]):
    #    sphere.t_coords[i] = [ (np.arctan2(sphere.points[i, 1], sphere.points[i, 0])*PI2_INV) % 1.0, 0.5+ np.arcsin(sphere.points[i, 2]/radius)*PI_INV]
    #p.add_mesh(sphere, texture=tex1, show_edges=False, opacity=1, smooth_shading=False, lighting=True, culling='back')

    ### 2nd part of the sphere
    radius, center = globe.radius, globe.center
    sphere = pv.Sphere(radius=radius, center=center, theta_resolution=60, phi_resolution=30, start_theta=0, end_theta=359.999999, direction=(0, 0, -1) )

    if type(clip) != type(None):
        if clip[0] == 'box': # ('box', ((xmin, xmax, ymin, ymax, zmin, zmax), invert)) )
            bounds, invert = clip[1]
            sphere = sphere.clip_box(bounds, invert)
        elif clip[0] == 'plane': # ('plane', (normal, origin, invert) )
            normal, origin, invert = clip[1]
            sphere.clip(normal, origin, invert, inplace= True)

    if style in ('simple', 'fancy1', 'fancy2', 'Mars', 'Cat1', 'Mosaic'):
        tex1, tex2 = get_global_texture(style, coastline, land, ocean)
        PI2_INV = 1.0/(2 * np.pi)
        PI_INV  = 2.0*PI2_INV
        sphere.t_coords = np.zeros((sphere.points.shape[0], 2))

        x, y, z = np.copy(sphere.points[:, 0]), np.copy(sphere.points[:, 1]), np.copy(sphere.points[:, 2])
        cx, cy, cz = center
        x -= cx
        y -= cy
        z -= cz
        sphere.t_coords[:,0] = 0.5+(np.arctan2(y, x)*PI2_INV) % 1.0
        sphere.t_coords[:,1] = 0.5+ np.arcsin(z/radius)*PI_INV

        p.add_mesh(sphere, texture=tex2, show_edges=False, opacity=alpha, smooth_shading=True, lighting=True, culling=culling)
    else:
        p.add_mesh(sphere, show_edges=False, opacity=alpha, color=color, smooth_shading=True, lighting=True, culling=culling)
    #sphere = pv.Sphere(radius=100, center=(radius, 0, 0), theta_resolution=180, phi_resolution=90)
    #p.add_mesh(sphere, show_edges=False)
def plot_splin_axis(p, globe):
    """
    """
    radius, center = globe.radius, globe.center
    spin_axis = pv.Cylinder(center=center, direction=(0.0, 0.0, 1), radius=10, height=2.3*radius, resolution=6)
    spin_axis = pv.Arrow(start=(0, 0, -radius), direction=(0, 0, 1),
                                tip_length=0.05, tip_radius=0.008, tip_resolution=20, scale=2.3*radius,
                                shaft_radius=0.003, shaft_resolution=20 )
    p.add_mesh(spin_axis, show_edges=False, opacity=1, lighting=True, color='gray', line_width=2, point_size=10, render_points_as_spheres=True, culling='back' )
def plot_grid(p, globe, lons=(0, 60, 120, 180, 240, 300), lats= (-60, -30, 0, 30, 60), color='k', culling=None, alpha=1.0, line_width=1 ):
    """
    """
    radius, center = globe.radius, globe.center
    for lo in np.deg2rad(lons):
        vx = np.sin(lo)
        vy = np.cos(lo)
        print(lo, vx, vy)
        disc = pv.Disc(center=center, inner=radius, outer=radius, normal=(vx, vy, 0), r_res=1, c_res=360)
        p.add_mesh(disc, edge_color=color, show_edges=True, culling=culling, opacity=alpha, line_width=line_width)
    for la in np.deg2rad(lats):
        z = radius*np.sin(la)
        r = radius*np.cos(la)
        disc = pv.Disc(center=center, inner=r, outer=r, normal=(0, 0, 1), r_res=1, c_res=360)
        p.add_mesh(disc, edge_color=color, show_edges=True, culling=culling, opacity=alpha, line_width=line_width)
def plot_great_circle_plane(p, globe, normal, r_range=None, alpha=1.0, culling=None, color_method=('uniform', 'k'), cmap='plasma'):
    """
    """
    radius, center = globe.radius, globe.center
    inner, outer = 0.0, radius
    if r_range:
        inner, outer = r_range
    if color_method[0] == 'uniform':
        circle_plane = pv.Disc(center=center, inner=inner, outer=outer, normal=normal, r_res=100, c_res=360)
        color=color_method[1]
        p.add_mesh(circle_plane, color=color, show_edges=False, opacity=alpha, smooth_shading=True, lighting=True, culling=culling, show_scalar_bar=False)
    elif color_method[0] == 'radial': # ('radial', (rs, values) )
        circle_plane = pv.Disc(center=center, inner=inner, outer=outer, normal=normal, r_res=100, c_res=360)

        rs, vs = color_method[1]
        func = interp1d(rs, vs,)
        pts = circle_plane.points
        x, y, z = np.copy(pts[:,0]), np.copy(pts[:,1]), np.copy(pts[:, 2])
        cx, cy, cz = center
        x -= cx
        y -= cy
        z -= cz
        scalars = func( np.sqrt(x*x + y*y + z*z) )
        p.add_mesh(circle_plane, scalars=scalars, show_edges=False, opacity=alpha, smooth_shading=True, lighting=True, culling=culling, show_scalar_bar=False, cmap=cmap)
    elif color_method[0] in ('vp', 'vs', 'density', 'qp', 'qs'):
        rs, vels = Earth_radial_model('ak135', color_method[0])
        plot_great_circle_plane(p, globe, normal=normal, alpha=1.0, culling=None, color_method=('radial', (rs, vels)), cmap=cmap)

def plot_vertical_half_globe3d(p, globe, style='simple', coastline=False, land=None, ocean=None, culling=None, alpha=1.0, color='gray', cut_longitude=100):
    """
    Plot a vertical half 3d globe. The half is cut at the boundary of `cut_longitude`.
    """
    a = np.deg2rad(cut_longitude)
    vx, vy = np.sin(a), np.cos(a)
    cut = ('plane', ((vx, vy, 0), (0, 0, 0), False) )
    plot_globe3d(p, globe, style, coastline, land, ocean, culling, alpha, color, cut)
def plot_horizontal_half_globe3d(p, globe, style='simple', coastline=False, land=None, ocean=None, culling=None, alpha=1.0, color='gray', lower=True):
    """
    Plot a vertical half 3d globe. The half is cut at the boundary of `cut_longitude`.
    """
    cut = ('plane', ((0, 0, 1), (0, 0, 0), lower) )
    plot_globe3d(p, globe, style, coastline, land, ocean, culling, alpha, color, cut)

#############################################################################################################################################################################################
# Functions for plotting objects in/on a globe
#############################################################################################################################################################################################
def plot_point(p, globe, lo, la, dp=0.0, size=300, symbol='cone', color='r', alpha=1.0, culling=None):
    """
    symbol: 'cone', 'sphere1', 'sphere2', 'cylinder'
    """
    if symbol == 'cone':
        dp -= size*0.5
        center, (vx, vy, vz) = globe.point_to_vec(lo, la, dp)
        sta = pv.Cone(center, (-vx, -vy, -vz), height=size, capping=True, angle=30, resolution=30 )
    elif symbol == 'sphere1':
        center, junk = globe.point_to_vec(lo, la, dp)
        sta = pv.Sphere(size, center)
    elif symbol == 'sphere2':
        dp -= size*0.5
        center, junk = globe.point_to_vec(lo, la, dp)
        sta = pv.Sphere(size*0.5, center)
    elif symbol == 'cylinder':
        dp -= size*0.5
        center, direction = globe.point_to_vec(lo, la, dp)
        sta = pv.Cylinder(center=center, direction=direction, radius=size*0.5, height=size, resolution=30, capping=True)

    p.add_mesh(sta, color=color, show_edges=False, opacity=alpha, smooth_shading=True, lighting=True, culling=culling)
def plot_line(p, globe, xs, ys, zs, color, show_edges=True, opacity=0.5, lighting=False, line_width=5):
    points = np.column_stack((xs, ys, zs) )
    poly = pv.Spline(points, xs.size)
    p.add_mesh(poly, show_edges=show_edges, opacity=opacity, color=color, lighting=lighting, line_width=line_width)


if __name__ == '__main__':
    p = pv.Plotter(notebook=0, shape=(1, 1), border=False, window_size=(1700, 1200) )
    p.set_background('white')
    globe = globe3d(radius=6371, center=(0, 0, 0) )
    plot_globe3d(p, globe, style='Mosaic', alpha=1.0, culling='back', clip=('plane', ((0, 0, 1), (0, 0, 0), True) )  ) #('plane', (normal, origin, invert) )
    #globe2 = globe3d(radius=6371, center=(0, 0, 5000) )
    #plot_globe3d(p, globe2, style='Mosaic', alpha=1.0, culling='back', clip=('plane', ((0, 0, 1), (0, 0, 5000), False) )  ) #('plane', (normal, origin, invert) )
    #plot_globe3d(p, globe, style='Mosaic', alpha=1.0, culling='back', clip=('plane', ((0, 1, 0), (0, 0, 0), True) )  ) #('plane', (normal, origin, invert) )
    #plot_globe3d(p, globe, style='Mosaic', alpha=1.0, culling='back', clip=('box', [[-10000, 0, -10000, 0, -10000, 10000], True]) ) #('box', ((xmin, xmax, ymin, ymax, zmin, zmax), invert)) )
    rs, vs = (0, 3500, 3500, 3500, 6371), (9, 8, 12, 13, 5)
    plot_great_circle_plane(p, globe, normal=(0, 0, 1), color_method=('radial', (rs, vs)), cmap='gray' )

    # Plot stations and events
    plot_point(p, globe, 0, 0, 0, size=300, symbol='sphere1', color='r', culling='back')



    #plot_great_circle_plane(p, globe, normal=(0, 1, 0), color_method=('vp', None), cmap='gray' )

    #plot_globe3d(p, globe, style='simple', coastline=True, culling='back')
    #plot_globe3d(p, globe, style='simple', coastline=False, land='#9B7E55', ocean=None, alpha=1, culling='back')
    #plot_globe3d(p, globe, style='simple', coastline=False, land='#895A17', ocean='#C4DCFF', alpha=0.5, culling='back')
    #plot_globe3d(p, globe, style='fancy1', alpha=1.0, culling='back')
    #plot_globe3d(p, globe, style='Cat1')

    #plot_globe3d(p, globe, style='fancy1', alpha=1.0, culling='back', clip=('box', ((0, 10000, -10000, 10000, -10000, 10000), False) )  )
    #plot_globe3d(p, globe, style='fancy1', alpha=1.0, culling='back', clip=('plane', ((0, 0, 1), (0, 0, 0), False) )  ) #('plane', (normal, origin, invert) )

    #plot_vertical_half_globe3d(p, globe, style='Mars', alpha=1.0, culling='back', cut_longitude=30.0 )
    #plot_horizontal_half_globe3d(p, globe, style='fancy1', alpha=1.0, culling='back')
    #plot_great_circle_plane(p, globe, normal=(0, 0, 1), color_method=('vp', None), cmap='copper' )
    #plot_point(p, globe, 0, 0, 300.0, size=600, symbol='cylinder')
    #plot_point(p, globe, 30, 0, 300.0, size=600, symbol='sphere1')
    #plot_point(p, globe, 60, 0, 300.0, size=600, symbol='cone')


    #globe2 = globe3d(radius=6371, center=(0, 0, 2000) )
    #plot_globe3d(p, globe2, style='Mars', alpha=1.0, culling='back', clip=('plane', ((0, 0, 1), (0, 0, 2000), False) )  ) #('plane', (normal, origin, invert) )
    #plot_great_circle_plane(p, globe2, normal=(0, 0, 1), color_method=('vp', None), cmap='copper' )

    #globe2 = globe3d(radius=6371, center=(0, 0, 000) )
    #plot_globe3d(p, globe2, style='Mars', alpha=1.0, culling='back', clip=('plane', ((0, 0, 1), (0, 0, 0), True) )  ) #('plane', (normal, origin, invert) )
    #plot_great_circle_plane(p, globe2, normal=(0, 0, 1), color_method=('vp', None), cmap='copper' )
    #plot_globe3d(p, radius=6371, style=None, coastline=True, culling='back', alpha=0.3, color='#AACCFF')
    #plot_globe3d(p, radius=3480, style=None, coastline=True, culling='back', alpha=0.3)
    #plot_globe3d(p, radius=1215, style=None, coastline=True, culling='back', alpha=0.6, color='r')

    #plot_splin_axis(p)
    #plot_grid(p, radius=6371, lons=np.arange(0, 360, 30), lats=np.arange(-70, 90, 20), color='#9B7E55', culling='back', alpha=0.6 )
    p.camera_position = [(18000, 18000, 18000), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    p.show()

    #junk = get_global_texture('simple', True)
    #print(junk)

    #junk = get_global_texture('simple', coastline=False, land='#895A17', ocean=None)
    #print(junk)

    #junk = get_global_texture('simple', coastline=False, land='#895A17', ocean='#0000AA')
    #print(junk)

    #junk = get_global_texture('fancy2', coastline=False, land='#895A17', ocean='#0000AA')
    #print(junk)

    #p = pv.Plotter(notebook=0, shape=(1, 1), border=False, window_size=(1700, 1200) )
    #plot_globe(p)
    #p.camera_position = [(18000, 18000, 18000), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    #p.show()
    pass