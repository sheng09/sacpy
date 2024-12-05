#!/usr/bin/env python3

# sphinx_gallery_thumbnail_number = 1
from glob import glob
from PIL.Image import radial_gradient
from matplotlib.colors import LightSource,  ListedColormap
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
import stripy

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
    def deplola2xyz(self, dep, lo, la):
        """
        Given (depth, lo, la), return (x, y, z) coordinates. The lo and la are in degree.
        The (depth, lo, la) could be three single values or three np.ndarray objects of the same size.
        """
        radius = self.radius-dep
        lam, phi = np.deg2rad(lo), np.deg2rad(la)
        x = np.cos(phi)*np.cos(lam)*radius + self.center[0]
        y = np.cos(phi)*np.sin(lam)*radius + self.center[1]
        z = np.sin(phi)*radius + self.center[2]
        return x,y,z
    def xyz2deplola(self, x, y, z):
        """
        Given (x, y, z) coordinates, return (depth, lo, la). The lo and la are in degree.
        The (x, y, z) could be three single values or three np.ndarray objects of the same size.
        """
        x -= self.center[0]
        y -= self.center[1]
        z -= self.center[2]
        radius = np.sqrt(x*x + y*y + z*z)
        dep = self.radius-radius
        lo = np.rad2deg(np.arctan2(y, x))
        la = np.rad2deg(np.arcsin(z/radius))
        return dep, lo, la
    def get_up_vector(self, lo, la):
        """
        Given a point (lo, la) in degree, return the unit vector (vx, vy, yz) of the Up direction at the point.
        The (lo, la) could be two single values or two np.ndarray objects of the same size.
        """
        vx, vy, vz = self.deplola2xyz(self.radius-1, lo, la)
        vx -= self.center[0]
        vy -= self.center[1]
        vz -= self.center[2]
        return vx, vy, vz
    def get_gc_tangential_vector(self, srclo, srcla, rcvlo, rcvla):
        """
        Calculate the unit vector of the tangential direction along the great circle path from
        a source point (srclo, srcla) to the receiver point (rcvlo, rcvla) on the globe surface.
        #
        Note, the tangential direction are the same at both the source and receiver points.
        """
        vup_src = self.get_up_vector(srclo, srcla)
        vup_rcv = self.get_up_vector(rcvlo, rcvla)
        vt = np.cross(vup_rcv, vup_src)
        return vt
    def get_gc_radial_vector(self, srclo, srcla, rcvlo, rcvla):
        """
        Calculate the unit vector of the radial direction at the receiver point along the great circle path from
        a source point (srclo, srcla) to the receiver point (rcvlo, rcvla) on the globe surface.
        #
        Note, the vector is at the receiver point, not the source point.
        """
        vup_rcv = self.get_up_vector(rcvlo, rcvla)
        vt      = self.get_gc_tangential_vector(srclo, srcla, rcvlo, rcvla)
        vr_rcv  = np.cross(vup_rcv, vt)
        return vr_rcv
    def get_gc_path(self, lo1, la1, lo2, la2, npts=100, arc='minor'):
        """
        Calculate a list of points (los, las) along the great circle path between two points (lo1, la1) and (lo2, la2) on the globe surface.

        Parameters:
            lo1, la1: longitude and latitude of the source point in degree.
            lo2, la2: longitude and latitude of the receiver point in degree.
            npts:     number of points along the great circle path. (this number only corresponds to the number of points on the minor arc)
            arc:  'full' or 'major'  or 'minor'(default) arc to plot.
        """
        up1 = self.get_up_vector(lo1, la1)
        up2 = self.get_up_vector(lo2, la2)
        dot_product = np.dot(up1, up2)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Angle in radians
        #
        if arc == 'full':
            angles = np.linspace(0, np.pi*2, npts) # [0, 2pi]
        elif arc == 'major':
            angles = np.linspace(angle, np.pi*2, npts) # [angle, 2pi]
        else:
            angles = np.linspace(0, angle, npts) # [0, angle]
        coef1 = np.sin(angle-angles)
        coef2 = np.sin(angles)
        vx = coef1*up1[0] + coef2*up2[0]
        vy = coef1*up1[1] + coef2*up2[1]
        vz = coef1*up1[2] + coef2*up2[2]
        #
        junk, los, las = self.xyz2deplola(vx+self.center[0], vy+self.center[1], vz+self.center[2])
        return los, las
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
    @staticmethod
    def benchmark():
        # here are some benchmarking tests and also examples of how to use the class to plot event-receiver and receiver-receiver geometries
        center = (10000000, -2000, 3000)
        R0 = 6371.0
        globe = globe3d(R0, center)
        rcv1_deplola = (-30, -100, 60)
        rcv2_deplola = (-50,  -80, 30)
        ev_deplola = (100, 169, -50)
        #
        p = pv.Plotter(notebook=0, shape=(1, 1), border=False, window_size=(1700, 1000) )
        p.set_background('white')
        # event->rcv
        for rcv in (rcv1_deplola, rcv2_deplola):
            #gc
            gc_los, gc_las = globe.get_gc_path(ev_deplola[1], ev_deplola[2], rcv[1], rcv[2], npts=100)
            gc_xs, gc_ys, gc_zs = globe.deplola2xyz(0, gc_los, gc_las)
            print(gc_xs[0], gc_ys[0], gc_zs[0])
            print(globe.deplola2xyz(0, ev_deplola[1], ev_deplola[2]))
            plot_line(p, globe, gc_xs, gc_ys, gc_zs, color='k', show_edges=True, opacity=0.5, lighting=False, line_width=3)
            #z
            x, y, z = globe.deplola2xyz(0, rcv[1], rcv[2])
            vx, vy, vz = globe.get_up_vector(rcv[1], rcv[2])
            arrow = pv.Arrow(start=(x, y, z), direction=(vx, vy, vz), tip_length=0.2, tip_radius=0.1, tip_resolution=20, scale=1500, shaft_radius=0.03, shaft_resolution=20 )
            p.add_mesh(arrow, show_edges=False,  opacity=1.0, color='k', smooth_shading=True, lighting=True, culling=False)
            #r
            vx, vy, vz = globe.get_gc_radial_vector(ev_deplola[1], ev_deplola[2], rcv[1], rcv[2])
            arrow = pv.Arrow(start=(x, y, z), direction=(vx, vy, vz), tip_length=0.2, tip_radius=0.1, tip_resolution=20, scale=1500, shaft_radius=0.03, shaft_resolution=20 )
            p.add_mesh(arrow, show_edges=False,  opacity=1.0, color='r', smooth_shading=True, lighting=True, culling=False)
            #t
            vx, vy, vz = globe.get_gc_tangential_vector(ev_deplola[1], ev_deplola[2], rcv[1], rcv[2])
            arrow = pv.Arrow(start=(x, y, z), direction=(vx, vy, vz), tip_length=0.2, tip_radius=0.1, tip_resolution=20, scale=1500, shaft_radius=0.03, shaft_resolution=20 )
            p.add_mesh(arrow, show_edges=False,  opacity=1.0, color='b', smooth_shading=True, lighting=True, culling=False)
        # rcv1<-->rcv2
        #gc
        gc_los, gc_las = globe.get_gc_path(rcv1_deplola[1], rcv1_deplola[2], rcv2_deplola[1], rcv2_deplola[2], npts=100)
        gc_xs, gc_ys, gc_zs = globe.deplola2xyz(0, gc_los, gc_las)
        plot_line(p, globe, gc_xs, gc_ys, gc_zs, color='k', show_edges=True, opacity=0.5, lighting=False, line_width=3)
        #r@rcv2
        vx, vy, vz = globe.get_gc_radial_vector(rcv1_deplola[1], rcv1_deplola[2], rcv2_deplola[1], rcv2_deplola[2])
        x, y, z = globe.deplola2xyz(0, rcv2_deplola[1], rcv2_deplola[2])
        arrow = pv.Arrow(start=(x, y, z), direction=(vx, vy, vz), tip_length=0.2, tip_radius=0.1, tip_resolution=20, scale=1500, shaft_radius=0.03, shaft_resolution=20 )
        p.add_mesh(arrow, show_edges=False,  opacity=1.0, color='c', smooth_shading=True, lighting=True, culling=False)
        #t@rcv2
        vx, vy, vz = globe.get_gc_tangential_vector(rcv1_deplola[1], rcv1_deplola[2], rcv2_deplola[1], rcv2_deplola[2])
        x, y, z = globe.deplola2xyz(0, rcv2_deplola[1], rcv2_deplola[2])
        arrow = pv.Arrow(start=(x, y, z), direction=(vx, vy, vz), tip_length=0.2, tip_radius=0.1, tip_resolution=20, scale=1500, shaft_radius=0.03, shaft_resolution=20 )
        p.add_mesh(arrow, show_edges=False,  opacity=1.0, color='#123456', smooth_shading=True, lighting=True, culling=False)
        #r@rcv1
        vx, vy, vz = globe.get_gc_radial_vector(rcv2_deplola[1], rcv2_deplola[2], rcv1_deplola[1], rcv1_deplola[2])
        x, y, z = globe.deplola2xyz(0, rcv1_deplola[1], rcv1_deplola[2])
        arrow = pv.Arrow(start=(x, y, z), direction=(vx, vy, vz), tip_length=0.2, tip_radius=0.1, tip_resolution=20, scale=1500, shaft_radius=0.03, shaft_resolution=20 )
        p.add_mesh(arrow, show_edges=False,  opacity=1.0, color='#234567', smooth_shading=True, lighting=True, culling=False)
        #t@rcv1
        vx, vy, vz = globe.get_gc_tangential_vector(rcv2_deplola[1], rcv2_deplola[2], rcv1_deplola[1], rcv1_deplola[2])
        x, y, z = globe.deplola2xyz(0, rcv1_deplola[1], rcv1_deplola[2])
        arrow = pv.Arrow(start=(x, y, z), direction=(vx, vy, vz), tip_length=0.2, tip_radius=0.1, tip_resolution=20, scale=1500, shaft_radius=0.03, shaft_resolution=20 )
        p.add_mesh(arrow, show_edges=False,  opacity=1.0, color='#345678', smooth_shading=True, lighting=True, culling=False)

        #
        plot_globe3d(p, globe, style='fancy2', alpha=1, land='#ee0000', ocean='#0000aa'  ) #('plane', (normal, origin, invert) )
        plot_point(p, globe, ev_deplola[1], ev_deplola[2], ev_deplola[0], size=300, symbol='sphere1', color='r', alpha=1.0, culling='back')
        plot_point(p, globe, rcv1_deplola[1], rcv1_deplola[2], rcv1_deplola[0], size=300, symbol='sphere1', color='g', alpha=1.0, culling='back')
        plot_point(p, globe, rcv2_deplola[1], rcv2_deplola[2], rcv2_deplola[0], size=300, symbol='sphere1', color='b', alpha=1.0, culling='back')
        p.show()


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
    elif style == 'Mosaic_copper':
        figname1 = '%s/Mosaic_copper.jpeg' % loc
        figname2 = '%s/Mosaic_copper.jpeg' % loc
        plot_mosaic_map(figname1, cmap='copper_r')
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
def plot_mosaic_map(figname, cmap='gray'):
    mat1 = np.random.random((500, 1000) )-0.5
    mat = np.concatenate([mat1, mat1[::-1,:]])
    mat = mat.transpose()
    mat = gaussian_filter(mat, 3)
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.imshow(mat, cmap=cmap) #, vmax=0.5, vmin=-1)
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
    sphere = pv.Sphere(radius=radius, center=center, theta_resolution=60, phi_resolution=30, start_theta=0, end_theta=359.9999, direction=(0, 0, -1) )

    if type(clip) != type(None):
        if clip[0] == 'box': # ('box', ((xmin, xmax, ymin, ymax, zmin, zmax), invert)) )
            bounds, invert = clip[1]
            sphere = sphere.clip_box(bounds, invert)
        elif clip[0] == 'plane': # ('plane', (normal, origin, invert) )
            normal, origin, invert = clip[1]
            sphere.clip(normal, origin, invert, inplace= True)

    if style in ('simple', 'fancy1', 'fancy2', 'Mars', 'Cat1', 'Mosaic', 'Mosaic_copper'):
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


###########################################################################################
# Plot 3D beachball
###########################################################################################
class beachball_3d:
    """
    Class to plot 3d beachball given focal mechanism.

    Example:
    >>> import pyvista as pv
    >>>
    >>> p = pv.Plotter(notebook=0, shape=(1, 1), border=False, window_size=(1700, 1000) )
    >>> p.set_background('white')
    >>> ######################################################
    >>> # plot surface
    >>> mesh = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=30, j_size=30, i_resolution=10, j_resolution=10)
    >>> mesh.point_arrays.clear()
    >>> p.add_mesh(mesh, show_edges=True, opacity=0.6, smooth_shading=True, lighting=False, culling=False)
    >>> ######################################################
    >>> # plot East, South, Down vectors
    >>> start = (-15, -15, 0)
    >>> for v in ( (1, 0, 0), (0, 1, 0), (0, 0, -1) ):
    >>>     mesh = pv.Arrow(start=start, direction=v, tip_length=0.3, shaft_resolution=30, shaft_radius=0.01, scale=3)
    >>>     p.add_mesh(mesh, show_edges=True,  opacity=1.0, color='k', smooth_shading=True, lighting=True, culling=False)
    >>> ######################################################
    >>> # plot beachball
    >>> mt = [0.91, -0.89, -0.02, 1.78, -1.55, 0.47]
    >>> bb = beachball_3d(mt)
    >>> cmap = ListedColormap(('#444444', '#eeeeee'))
    >>> bb.plot_3d(p,     center=(0,0,0), radius=10.0, hemisphere='None', cmap='RdBu_r', plot_nodal=True, show_scalar_bar=True)
    >>> bb.plot_3d_vec(p, center=(0,0,0), radius=10.0, hemisphere=None, alpha=1.0, lighting=False, culling=False, scale=3)
    >>> bb.plot_3d_vcone(p, apex=(0,0,0), cones=[(3.0, 15, '#CA3C33', 0.3), (3.1, 15, '#407AA2', 0.3)])
    >>> p.camera_position=( (0, -55, -30), (0, 0, 0), (0, 0, 1) )
    >>> p.show()
    """
    def __init__(self, mt):
        """
        mt: (M11, M22, M33, M12, M13, M23) - the six independent components of the moment tensor,
            where the coordinate system is 1,2,3 = Up,South,East which equals r,theta,phi.
            - Harvard/Global CMT convention, or (Mrr=M11, Mtt=M22, Mpp=M33, Mrt=M12, Mrp=M13, Mtp=M23).

            The relation to Aki and Richards x,y,z equals North,East,Down convention is as follows:
            (Mzz=M11, Mxx=M22, Myy=M33, Mxz=M12, Myz=-M13, Mxy=-M23).

            Here we use x,y,z equals East,North,Up, hence:
            (Mzz=M11, Myy=M22, Mxx=M33, Myz=-M12, Mxz=M13, Mxy=-M23)

            Input  GCMT     Aki      Here
            1(U)   r(U)=1   x(N)=-2  x(E)= 3
            2(S)   t(S)=2   y(E)= 3  y(N)=-2
            3(E)   p(E)=3   z(D)=-1  z(U)= 1
        """
        M11, M22, M33, M12, M13, M23 = mt
        mat = np.zeros((3, 3))
        mat[2,2], mat[1,1], mat[0,0] = M11, M22, M33
        mat[1,2], mat[2,1] = -M12, -M12
        mat[0,2], mat[2,0] = M13, M13
        mat[0,1], mat[1,0] = -M23, -M23
        self.d_mt = mt
        self.d_mat = mat
    def __radiation_p(self, thetas, phis, binarization=False):
        """
        Compute the P-wave radiations given a list of directions.

        thetas, phis: a list of theta and phi. theta is angle between
                      the direction and x axis, and phi the angle
                      between the direction and z axis. Angles should
                      be in radian other than in degree.
        binarization: Modify the radiation amplitude to -1, 0, 1 for
                      negative, zero, and positive amplitudes, respectively.

        Return: vecs, radiations
                vecs: a matrix each row of which is a unit direction vector.
                radiations: a list of radiation amplitudes for the directions.
        """
        sin, cos = np.sin, np.cos
        sp, cp = sin(phis), cos(phis)
        vecs = np.zeros((len(thetas), 3 ) )
        vecs[:,0] = sp*cos(thetas)
        vecs[:,1] = sp*sin(thetas)
        vecs[:,2] = cp
        radiations = np.zeros(len(thetas) )
        for idx, vt in enumerate(vecs):
            radiations[idx] = np.matmul(np.matmul(vt, self.d_mat), vt.T)
        if binarization:
            radiations = np.sign(radiations)
        return vecs, radiations
    def plot_3d_vcone(self, p, apex=(0,0,0), cones=[(2.8, 18, 'r', 0.3), (3.1, 18, 'b', 0.3)], 
                      culling=False, lighting=False):
        """
        Plot 3d vertical cones.

        p:     an instance of pyvista.Plotter
        apex:  location of cones' apex
        cones: a list of `(phi, height, color, alpha)` for each of the
               cones to plot. The `phi` is the angle between the cone's
               slope and Up direction. It takes the range between 0 and pi,
               where 0 means Up direction and pi Down direction.
               The `height` is the height of the cone.
        culling:  pyvista parameters.
        lighting: pyvista parameters.
        """
        tan, pi = np.tan, np.pi
        for phi, height, clr, alpha in cones:
            if phi>0.5*pi:
                cone_center = apex[0], apex[1], apex[2]-height*0.5
                disc_center = apex[0], apex[1], apex[2]-height
                direction = (0, 0, 1)
                radius = height*tan(pi-phi)
            else:
                cone_center = apex[0], apex[1], apex[2]+height*0.5
                disc_center = apex[0], apex[1], apex[2]+height
                direction = (0, 0, -1)
                radius = height*tan(phi)
            mesh = pv.Cone(center=cone_center, direction=direction, height=height, radius=radius, capping=True,
                    resolution=90)
            p.add_mesh(mesh, show_edges=False,  opacity=alpha, color=clr,
                        smooth_shading=True, lighting=lighting, culling=culling)
            mesh = pv.Disc(center=disc_center, inner=radius, outer=radius, normal=(0, 0, 1), c_res=90)
            p.add_mesh(mesh, show_edges=True,  opacity=alpha, color='k',
                        smooth_shading=True, lighting=lighting, culling=culling)

        pass
    def plot_3d_vec(self, p, center=(0,0,0), radius=10.0, hemisphere=None,
                    neg_color='#0f0396', pos_color='#db620c',
                    alpha=1.0, scale=3.0, density_level=3,
                    lighting=False, culling=False, ):
        """
        Plot 3d P-wave radiation vectors at the surface of the beachball.

        p:          an instance of pyvista.Plotter
        center:     the center of the beachball.
        radius:     the radius of the beachball.
        hemisphere: `lower` or `upper` to plot the lower or upper hemisphere.
                    Default is `None` to plot the whole sphere.
        neg_color, pos_color: color for the negative and positive amplitudes, respectively.
        alpha:      transparency.
        scale:      scale all the vectors. (Default is 1.0)
        density_level: the bigger, the more vector arrows.
        culling:    pyvista parameters.
        lighting:   pyvista parameters.
        """
        sin, cos, pi = np.sin, np.cos, np.pi
        phi_min, phi_max = 0.0, pi
        mesh = stripy.spherical_meshes.triangulated_cube_mesh(refinement_levels=density_level)
        thetas = mesh.lons
        phis   = pi*0.5 - mesh.lats
        if hemisphere=='upper':
            phi_max = 0.5*pi
        elif hemisphere=='lower':
            phi_min = 0.5*pi

        idxs = np.where(phi_min<=phis)
        thetas, phis = thetas[idxs], phis[idxs]
        idxs = np.where(phis<=phi_max)
        thetas, phis = thetas[idxs], phis[idxs]

        vecs, radiations = self.__radiation_p(thetas, phis)
        bb_center = np.array(center)
        for v, r in zip(vecs, radiations):
            amp = abs(r)*scale
            start = v*radius+bb_center
            direction = v
            clr = pos_color
            if r<0:
                start = v*(radius+amp)+bb_center
                direction=-v
                clr=neg_color
            #start = start + bb_center
            mesh = pv.Arrow(start=start, direction=direction, tip_length=0.3, shaft_radius=0.02, scale=amp )
            p.add_mesh(mesh, show_edges=False, opacity=alpha, color=clr,
                    smooth_shading=True, lighting=lighting, culling=culling)
    def plot_3d(self, p, center=(0,0,0), radius=10.0, hemisphere=None, plot_nodal=False,
                binarization=False, cmap='bwr', show_scalar_bar=True,
                alpha=1.0, culling=False, lighting=False):
        """
        Plot 3d beachball with varied P-wave radiation amplitudes at different directions.
        p:              an instance of pyvista.Plotter
        center:         the center of the beachball.
        radius:         the radius of the beachball.
        hemisphere:     `lower` or `upper` to plot the lower or upper hemisphere.
                        Default is `None` to plot the whole sphere.
        plot_nodal:     `True` or `False` (default) to plot nodal planes on the beachball surface.
        binarization:   Modify the P-wave radiation amplitude to -1, 0, 1 for
                        negative, zero, and positive amplitudes, respectively.
                        Default is `False`, and will plot absolute amplitudes.
        cmap:           colormap to plot the P-wave radiation amplitudes.
        show_scalar_bar:`True` or `False` to plot the colorbar.
        alpha:      transparency.
        culling:    pyvista parameters.
        lighting:   pyvista parameters.
        """
        # Create mesh data
        x = np.arange(0, 360.001, 1)
        if hemisphere=='upper':
            y = np.arange(0, 90.001, 1)
        elif hemisphere=='lower':
            y = np.arange(90, 180.001, 1)
        else:
            y = np.arange(0, 180.001, 1)
        xx, yy = np.meshgrid(x, y)

        #
        vecs, scalar = self.__radiation_p( np.deg2rad(xx.flatten()), np.deg2rad(yy.flatten()), binarization=binarization )
        scalar *= (1.0/scalar.max() )
        scalar = scalar.reshape(xx.shape)

        # Vertical levels
        levels = [radius * 1.]

        #Create a structured grid
        grid_scalar = pv.grid_from_sph_coords(x, y, levels)
        grid_scalar.translate(center) #

        # And fill its cell arrays with the scalar data
        grid_scalar.point_arrays["Normalized P-wave radiation amplitudes"] = np.array(scalar).swapaxes(-2, -1).ravel("C")

        # Make a plot
        vmax = scalar.max()
        sargs = dict(color='k', vertical=True, interactive=False, n_colors=128, title='', outline=False, 
                     position_x=0.88, position_y=0.05, width=0.05, height=0.8, n_labels=5,
                     label_font_size=39, fmt='%.2f' )
        p.add_mesh(grid_scalar, show_edges=False, clim=[-vmax, vmax], opacity=alpha, cmap=cmap,
                   smooth_shading=True, lighting=lighting, culling=culling,
                   scalar_bar_args=sargs, show_scalar_bar=show_scalar_bar)
        if plot_nodal:
            if scalar.min() < 0.0 < scalar.max():
                contours = grid_scalar.contour([0.0] )
                p.add_mesh(contours, show_edges=True, opacity=1.0, color='k')
if __name__ == '__main__':
    globe3d.benchmark()
    sys.exit(0)
    p = pv.Plotter(notebook=0, shape=(1, 1), border=False, window_size=(1700, 1000) )
    p.set_background('white')
    ######
    # plot surface
    mesh = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=30, j_size=30, i_resolution=10, j_resolution=10)
    mesh.point_arrays.clear()
    p.add_mesh(mesh, show_edges=True, opacity=0.6,
               smooth_shading=True, lighting=False, culling=False)
    ######
    # plot East, South, Down
    start = (-15, -15, 0)
    for v in ( (1, 0, 0), (0, 1, 0), (0, 0, -1) ):
        mesh = pv.Arrow(start=start, direction=v, tip_length=0.3, shaft_resolution=30, shaft_radius=0.01, scale=3)
        p.add_mesh(mesh, show_edges=True,  opacity=1.0, color='k',
                        smooth_shading=True, lighting=True, culling=False)
    ######
    # plot beachball
    #mt = [1, 0, -1, 0, 0, 0] # reverse
    #mt = [-1, 0, 1, 0, 0, 0] # normal
    #mt = [0, -1, 1, 0, 0, 0] # strike-slip
    mt = [0, 0, 0, 0, -1, 0] # dip-slip

    #mt = [1, -1, 0, 0, 0, 0] # normal
    #mt = [-1, 0, 1, 0, 0, 0] # strike-slip
    #mt = [0, 0, 0, 0, -1, 0] # dip-slip
    #mt = [0.91, -0.89, -0.02, 1.78, -1.55, 0.47]
    bb = beachball_3d(mt)
    cmap = ListedColormap(('#444444', '#eeeeee'))
    bb.plot_3d(p,     center=(0,0,0), radius=10.0, hemisphere='None', cmap='RdBu_r', plot_nodal=True, show_scalar_bar=True)
    bb.plot_3d_vec(p, center=(0,0,0), radius=10.0, hemisphere=None, alpha=1.0, lighting=True, culling=False, scale=3)
    bb.plot_3d_vcone(p, apex=(0,0,0), cones=[(3.0, 15, '#CA3C33', 0.3), (3.1, 15, '#407AA2', 0.3)])
    p.camera_position=( (0, -55, -30), (0, 0, 0), (0, 0, 1) )
    p.show()
    print(p.camera_position)
    sys.exit(0)
    #
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
