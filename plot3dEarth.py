#!/usr/bin/env python3

# sphinx_gallery_thumbnail_number = 1
from PIL.Image import radial_gradient
from matplotlib.colors import LightSource
import pyvista as pv
from pyvista import examples as pv_examples

import numpy as np
import pickle

from os.path import abspath as os_path_abspath
from os.path import exists as os_path_exists

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys

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


def plot_globe3d(p, radius_km=6371, style='simple', coastline=False, land=None, ocean=None, culling=None, alpha=1.0, color='gray'):
    """
    """
    ## first part of the sphere
    #sphere = pv.Sphere(radius=radius_km, center=(0, 0, 0), theta_resolution=60, phi_resolution=30, start_theta=0, end_theta=359.99999999, direction=(0, 0, -1) )
    #sphere.t_coords = np.zeros((sphere.points.shape[0], 2))
    #for i in range(sphere.points.shape[0]):
    #    sphere.t_coords[i] = [ (np.arctan2(sphere.points[i, 1], sphere.points[i, 0])*PI2_INV) % 1.0, 0.5+ np.arcsin(sphere.points[i, 2]/radius_km)*PI_INV]
    #p.add_mesh(sphere, texture=tex1, show_edges=False, opacity=1, smooth_shading=False, lighting=True, culling='back')

    ### 2nd part of the sphere
    sphere = pv.Sphere(radius=radius_km, center=(0, 0, 0), theta_resolution=60, phi_resolution=30, start_theta=0, end_theta=359.999999, direction=(0, 0, -1) )

    if style in ('simple', 'fancy1', 'fancy2', 'Mars', 'Cat1'):
        tex1, tex2 = get_global_texture(style, coastline, land, ocean)
        PI2_INV = 1.0/(2 * np.pi)
        PI_INV  = 2.0*PI2_INV
        sphere.t_coords = np.zeros((sphere.points.shape[0], 2))

        sphere.t_coords[:,0] = 0.5+(np.arctan2(sphere.points[:, 1], sphere.points[:, 0])*PI2_INV) % 1.0
        sphere.t_coords[:,1] = 0.5+ np.arcsin(sphere.points[:, 2]/radius_km)*PI_INV

        p.add_mesh(sphere, texture=tex2, show_edges=False, opacity=alpha, smooth_shading=True, lighting=True, culling=culling)
    else:
        p.add_mesh(sphere, show_edges=False, opacity=alpha, color=color, smooth_shading=True, lighting=True, culling=culling)
    #sphere = pv.Sphere(radius=100, center=(radius_km, 0, 0), theta_resolution=180, phi_resolution=90)
    #p.add_mesh(sphere, show_edges=False)
def plot_splin_axis(p):
    spin_axis  = pv.Cylinder(center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1), radius=10, height=14000, resolution=6)
    spin_axis = pv.Arrow(start=(0, 0, -6371), direction=(0, 0, 1),
                                tip_length=0.05, tip_radius=0.008, tip_resolution=20, scale=14000,
                                shaft_radius=0.003, shaft_resolution=20 )
    p.add_mesh(spin_axis, show_edges=False, opacity=1, lighting=True, color='gray', line_width=2, point_size=10, render_points_as_spheres=True, culling='back' )
def plot_grid(p, radius_km=6371, lons=(0, 60, 120, 180, 240, 300), lats= (-60, -30, 0, 30, 60), color='k', culling=None, alpha=1.0, line_width=1 ):
    """
    """
    for lo in np.deg2rad(lons):
        vx = np.sin(lo)
        vy = np.cos(lo)
        print(lo, vx, vy)
        disc = pv.Disc(center=(0.0, 0.0, 0.0), inner=radius_km, outer=radius_km, normal=(vx, vy, 0), r_res=1, c_res=360)
        p.add_mesh(disc, edge_color=color, show_edges=True, culling=culling, opacity=alpha, line_width=line_width)
    for la in np.deg2rad(lats):
        z = radius_km*np.sin(la)
        r = radius_km*np.cos(la)
        disc = pv.Disc(center=(0.0, 0.0, z), inner=r, outer=r, normal=(0, 0, 1), r_res=1, c_res=360)
        p.add_mesh(disc, edge_color=color, show_edges=True, culling=culling, opacity=alpha, line_width=line_width)

if __name__ == '__main__':
    p = pv.Plotter(notebook=0, shape=(1, 1), border=False, window_size=(1700, 1200) )
    p.set_background('white')
    #plot_globe3d(p, radius_km=6371, style='simple', coastline=True, culling='back')
    plot_globe3d(p, radius_km=6371, style='simple', coastline=False, land='#9B7E55', ocean=None, alpha=1, culling='back')
    #plot_globe3d(p, radius_km=6371, style='simple', coastline=False, land='#895A17', ocean='#C4DCFF', alpha=0.5, culling='back')
    #plot_globe3d(p, radius_km=6371, style='fancy2', alpha=0.5, culling='back')
    #plot_globe3d(p, radius_km=6371, style='Mars')


    plot_globe3d(p, radius_km=6371, style=None, coastline=True, culling='back', alpha=0.3, color='#AACCFF')
    plot_globe3d(p, radius_km=3480, style=None, coastline=True, culling='back', alpha=0.3)
    plot_globe3d(p, radius_km=1215, style=None, coastline=True, culling='back', alpha=0.6, color='r')
    plot_splin_axis(p)
    #plot_grid(p, radius_km=6371, lons=np.arange(0, 360, 30), lats=np.arange(-70, 90, 20), color='#9B7E55', culling='back', alpha=0.6 )
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