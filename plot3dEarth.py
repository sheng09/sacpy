#!/usr/bin/env python3

from matplotlib.colors import  ListedColormap, LinearSegmentedColormap
import pyvista as pv
from pyvista import examples as pv_examples

import numpy as np
import pickle
from numba import jit
from scipy.linalg import eig
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import obspy.taup as taup

from os.path import abspath as os_path_abspath
from os.path import exists as os_path_exists

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
import stripy

pv.global_theme.allow_empty_mesh = True
class Scene3D:
    """
    A scene holding multiple 3D objects.
    Note, here theta is like the longitude, and phi is like the colatitude on a sphere.
    """
    def __init__(self, pv_plotter=None, pv_plotter_kargs={'window_size':(1700, 1200)}, number_of_peels=4 ):
        """
        Initialize the 3D scene.

        :param pv_plotter:       An existing PyVista plotter instance (default is None).
        :param pv_plotter_kargs: Keyword arguments for creating the PyVista plotter if `pv_plotter` is not provided.
        """
        if pv_plotter is not None:
            self.pv_plotter = pv_plotter
        else:
            self.pv_plotter = pv.Plotter(**pv_plotter_kargs)
        if number_of_peels>0:
            self.pv_plotter.enable_depth_peeling(number_of_peels=number_of_peels) #  to handles multiple transparent surfaces correctly
        else:
            self.pv_plotter.disable_depth_peeling()  # disable depth peeling for efficiency
        self.disable_clip()
        self.enable_add_mesh_to_plotter()
    def enable_clip(self, clip_kw={'normal': (0, 0, 1), 'origin': (0, 0, 0)},
                          clip_box_kw={'bounds': (-1, 1, -1, 1, -1, 1)} ):
        """
        Enable clipping for the 3D scene. All subsequent added objects will be clipped.
        :param clip_kw:      Keyword arguments for `pyvista.DataObjectFilters.clip(...)`
                             (default is {'normal': (0, 0, 1), 'origin': (0, 0, 0)})
        :param clip_box_kw:  Keyword arguments for `pyvista.DataObjectFilters.clip_box(...)`
                             (default is {'bounds': (-1, 1, -1, 1, -1, 1)})
        """
        self.clip_kw = clip_kw
        self.clip_box_kw = clip_box_kw
    def disable_clip(self):
        self.clip_kw = None
        self.clip_box_kw = None
    def disable_add_mesh_to_plotter(self):
        """
        This is combined with `add_mesh(...)` function to disable adding mesh to the plotter.
        So that the mesh can be created (clipped, and returned), but not added to the plotter.
        """
        self.flag_add_mesh_to_plotter = False
    def enable_add_mesh_to_plotter(self):
        self.flag_add_mesh_to_plotter = True
    def __clip(self, mesh):
        if self.clip_kw is not None:
            mesh = mesh.clip(**self.clip_kw)
        if self.clip_box_kw is not None:
            mesh = mesh.clip_box(**self.clip_box_kw)
        return mesh
    def add_mesh(self, mesh, **kwargs):
        mesh = self.__clip(mesh)
        actor = None
        if self.flag_add_mesh_to_plotter:
            actor = self.pv_plotter.add_mesh(mesh, **kwargs)
        return mesh, actor
    def add_plane(self, center=(0.,0.,0.), direction=(0,0,1), x_direction=None, color='gray',
                    i_size=10, j_size=10, i_resolution=10, j_resolution=10,
                    opacity=0.6, lighting=True, **kwargs):
        """
        Add a plane.

        :param center:    The center of the plane.
        :param direction:   The orientation direction of the plane normal (or its z-axis).
        :param x_direction: The orientation direction of the plane's x-axis.
                            (Default is `None`. If set, need to be perpendicular to `direction`, the z-axis.)
        :param color:     The color of the plane.
        :param i_size:    The size of the plane in the i direction.
        :param j_size:    The size of the plane in the j direction.
        :param i_resolution: The resolution of the plane in the i direction.
        :param j_resolution: The resolution of the plane in the j direction.
        :param opacity:      The opacity of the plane.
        :param lighting:       Whether to use lighting.
        :param kwargs:         Additional keyword arguments for `pyvista.Plotter.add_mesh(...)`.

        :return: (mesh, actor)
        """
        if x_direction is None:
            mesh = pv.Plane(center=center, direction=direction, i_size=i_size, j_size=j_size, i_resolution=i_resolution, j_resolution=j_resolution)
        else:
            if np.abs(np.dot(x_direction, direction)) > 1e-10:
                raise ValueError("The input `x_direction` must be perpendicular to `direction`!", x_direction, direction)
            mesh = pv.Plane(center=(0,0,0), direction=(0,0,1), i_size=i_size, j_size=j_size, i_resolution=i_resolution, j_resolution=j_resolution)
            mesh.points = Scene3D.rotate_and_translate2(mesh.points, current_orientation_direction1=(0,0,1), current_orientation_direction2=(1,0,0), current_origin_xyz=(0,0,0),
                                                        new_orientation_direction1=direction, new_orientation_direction2=x_direction, new_origin_xyz=center )
        ######
        return self.add_mesh(mesh, color=color, opacity=opacity, lighting=lighting, **kwargs)
    def add_spline(self, xs, ys, zs,
                   scalars=None, color='k', cmap='viridis', line_width=5, label='spline',
                   lighting=True, **kwargs):
        """
        Add a spline to the scene.

        :param xs: The x-coordinates of the line.
        :param ys: The y-coordinates of the line.
        :param zs: The z-coordinates of the line.
        :param scalars: The scalars to color the line. (default is `None`)
        :param color: The color of the line if `scalars=None`.
        :param cmap: The colormap to use if `scalars` is not `None`.
        :param line_width: The width of the line (or the radius of the tube).
        :param label: The label for the scalars (if `scalars` is not `None`).
        :param lighting: Whether to use lighting.
        :param kwargs: Additional keyword arguments for `pyvista.Plotter.add_mesh(...)`.

        :return: (mesh, actor)
        """
        points = np.column_stack((xs, ys, zs))
        spline = pv.Spline(points)
        if scalars is not None:
            spline.point_data[label] = scalars
        ######
        if scalars is None:
            return self.add_mesh(spline, color=color, cmap=cmap, line_width=line_width, lighting=lighting, **kwargs)
        else:
            return self.add_mesh(spline, scalars=label, cmap=cmap, line_width=line_width, lighting=lighting, **kwargs)
    def add_point(self, x, y, z, direction=(0,0,1), size=2, shape='sphere', color='k',
                  shift_along_direction=0,
                  lighting=True, **kwargs):
        """
        Add a point to the scene.

        :param x: The x-coordinate of the point.
        :param y: The y-coordinate of the point.
        :param z: The z-coordinate of the point.
        :param direction: The direction of the point.
        :param size: The size of the point (which means the radius of a sphere, or the base of a cylinder or a cone).
        :param shape: The shape of the point (could be 'sphere', 'cone', or 'cylinder').
        :param color: The color of the point.
        :param shift_along_direction: The amount to shift the point along the direction.
                                      Usually, a `-size` or `size` can be used to move
                                      the point object so that its vertex or base is at (x, y, z)!
        :param lighting: Whether to use lighting.
        :param kwargs: Additional keyword arguments for `pyvista.Plotter.add_mesh(...)`.

        :return: (mesh, actor)
        """
        direction = np.array(direction, dtype=np.float64)
        direction *= (1.0/np.linalg.norm(direction))  # Normalize the direction vector
        dx, dy, dz = direction * shift_along_direction
        x, y, z = x + dx, y + dy, z + dz
        if shape == 'sphere':
            mesh = pv.Sphere(center=(x, y, z), direction=direction, radius=size, theta_resolution=180, phi_resolution=90)
        elif shape == 'cone':
            mesh = pv.Cone(center=(x, y, z), direction=direction, height=size*2, capping=True, angle=30, resolution=90)
        elif shape == 'cylinder':
            mesh = pv.Cylinder(center=(x, y, z), direction=direction, radius=size, height=size*2, resolution=90, capping=True)
        else:
            raise ValueError(f"Unknown shape: {shape}. Supported shapes are 'sphere', 'cone', and 'cylinder'.")
        return self.add_mesh(mesh, color=color, lighting=lighting, **kwargs)
    def add_arrow(self, loc, direction, scale=1, loc_is_end=False, color='k',
                  tip_length=0.3, tip_radius=0.1, tip_resolution=20, shaft_radius=0.01, shaft_resolution=20, 
                  lighting=True, **kwargs):
        """
        Add an arrow to the scene.

        :param loc: The location point of the arrow.
                    If `loc_is_end=False` (default), then this is the start point.
                    If `loc_is_end=True`, then this is the end point of the arrow (the cap side).
        :param direction: The direction of the arrow.
        :param scale: The scale factor for the arrow (equivalent to the length (and radius as well) of the arrow).
        :param loc_is_end: (default is `False`) so the `loc` is used as the start point of the arrow.
                           If set to true, then the `loc` is used as the end point (the cap side) of the arrow.
        :param color: The color of the arrow.
        :param tip_length: The length of the arrow tip.
        :param tip_radius: The radius of the arrow tip.
        :param tip_resolution: The resolution of the arrow tip.
        :param shaft_radius: The radius of the arrow shaft.
        :param shaft_resolution: The resolution of the arrow shaft.
        :param lighting: Whether to use lighting.
        :param kwargs: Additional keyword arguments for `pyvista.Plotter.add_mesh(...)`.

        :return: (mesh, actor)
        """
        if loc_is_end:
            loc = np.array(loc, dtype=np.float64)
            vec = np.array(direction, dtype=np.float64)
            vec /= np.linalg.norm(vec)
            loc -= (vec*scale)
        mesh = pv.Arrow(start=loc, direction=direction, scale=scale, tip_length=tip_length, tip_radius=tip_radius, tip_resolution=tip_resolution, shaft_resolution=shaft_resolution, shaft_radius=shaft_radius)
        return self.add_mesh(mesh, color=color, lighting=lighting, **kwargs)
    def add_disk(self, center=(0.,0.,0.), inner=0.0, outer=1.0, direction=(0,0,1), x_direction=None, color='k', r_res=1, c_res=180,
                 lighting=True, opacity=0.1, **kwargs):
        if x_direction is None:
            disc = pv.Disc(center=center, inner=inner, outer=outer, normal=direction, r_res=r_res, c_res=c_res)
        else:
            if np.abs(np.dot(x_direction, direction)) > 1e-10:
                raise ValueError("The input `x_direction` must be perpendicular to `direction`!", x_direction, direction)
            disc = pv.Disc(center=(0,0,0), inner=inner, outer=outer, normal=(0,0,1), r_res=r_res, c_res=c_res)
            disc.points = Scene3D.rotate_and_translate2(disc.points, current_orientation_direction1=(0,0,1), current_orientation_direction2=(1,0,0), current_origin_xyz=(0,0,0),
                                                        new_orientation_direction1=direction, new_orientation_direction2=x_direction, new_origin_xyz=center )
        return self.add_mesh(disc, color=color, lighting=lighting, opacity=opacity, **kwargs)
    def add_sphere(self, radius=1., center=(0.,0.,0.),
                   north_pole_direction=(0., 0., 1.), lo0la0_direction=(1., 0., 0.),
                   color='r', texture=None, texture_theta_range=(0,360), texture_phi_range=(180,0),
                   theta_resolution=60, start_theta=0, end_theta=360,
                   phi_resolution=30,   start_phi=0,   end_phi=180,
                   lighting=True, **kwargs):
        """
        Add a sphere (with solid color or texture) to the scene.

        :param radius: The radius of the sphere.
        :param center: The center of the sphere.
        :param north_pole_direction: The direction vector (vx,vy,vz) of the north pole.
        :param lo0la0_direction: The direction vector (vx,vy,vz) from the center to (longitude=0, latitude=0).
        :param color: The color of the sphere.
        :param texture: The texture of the sphere. Could the filename of a figure, or a 2d-array of (r,g,b), or (r,g,b,a), or grayscale.
        :param texture_theta_range: The theta range for the texture from its left to right (default is (0,360))
        :param texture_phi_range:   The phi range for the texture from its bottom to top (or from the last row to the first row for 2d-array) (default is (180,0)).
        :param theta_resolution: The resolution of the theta direction for plotting this sphere.
        :param start_theta:      The start angle for the theta direction.
        :param end_theta:        The end angle for the theta direction.
        :param phi_resolution: The resolution of the phi direction for plotting this sphere.
        :param start_phi:      The start angle for the phi direction (must be between 0 and 180).
        :param end_phi:        The end angle for the phi direction (must be between 0 and 180).
        :param lighting: Whether to use lighting.
        :param kwargs: Additional keyword arguments for `pyvista.Plotter.add_mesh(...)`.

        :return: (mesh, actor)
        """
        if start_phi>180 or start_phi<0 or end_phi>180 or end_phi<0:
            raise ValueError("Invalid phi range", start_phi, end_phi)
        start_theta, end_theta = start_theta % 360, end_theta % 360
        if start_theta >= end_theta:
            start_theta -= 360
        ####
        theta = np.linspace(start_theta, end_theta, theta_resolution)
        phi   = np.linspace(start_phi, end_phi, phi_resolution)
        sphere = pv.grid_from_sph_coords(theta, phi, [radius,] )
        sphere.points = Scene3D.rotate_and_translate2(sphere.points, current_orientation_direction1=(0,0,1), current_orientation_direction2=(1,0,0), current_origin_xyz=(0,0,0),
                                                      new_orientation_direction1=north_pole_direction, new_orientation_direction2=lo0la0_direction, new_origin_xyz=center )
        ####
        if texture is None:
            return self.add_mesh(sphere, color=color, lighting=lighting, **kwargs)
        else:
            pp, tt = np.meshgrid(phi, theta)
            t0, t1 = texture_theta_range
            p0, p1 = texture_phi_range
            v = (pp - p0) / (p1 - p0)
            u = (tt - t0) / (t1 - t0)
            sphere.active_texture_coordinates = np.column_stack((u.ravel(), v.ravel() ) )
            ###
            texture = pv.Texture(texture)
            texture.repeat = True
            sphere._texture = texture
            return self.add_mesh(sphere, texture=texture, lighting=lighting,  **kwargs)
    def add_sphere_grd(self, theta_range, phi_range, mat2d,
                       radius=1., center=(0.,0.,0.),
                       north_pole_direction=(0., 0., 1.), lo0la0_direction=(1., 0., 0.),
                       cmap='viridis', label='sphere_grd',
                       lighting=True, **kwargs):
        """
        Add a spherical grid to the scene, with scalar values to paint the sphere.

        :param theta_range: The theta range (in degrees) of the grid. (theta could be considered as longitude.)
        :param phi_range:   The phi range (in degrees) of the grid. (phi could be considered as colatitude.)
        :param mat2d:       The 2d-ndarray matrix to be mapped onto the grid.
                            Note, the `mesh_grd` should have the shape (nphi, ntheta) (different rows correspond to phi and different columns for theta).
        :param radius: The radius of the sphere.
        :param center: The center of the sphere.
        :param north_pole_direction: The direction of the north pole of the sphere. (default is (0,0,1), the vertical direction).
        :param lo0la0_direction: The direction vector (vx,vy,vz) from the center to (longitude=0, latitude=0).
        :param cmap: The colormap to use.
        :param label: The label for the scalar data (grd_values).
        :param lighting: Whether to use lighting.
        :param kwargs: Additional keyword arguments for `pyvista.Plotter.add_mesh(...)`.

        :return: (mesh, actor)
        """
        if np.abs(np.dot(north_pole_direction, lo0la0_direction)) > 1e-10:
            raise ValueError("The north pole direction and the (longitude=0, latitude=0) direction must be orthogonal.", north_pole_direction, lo0la0_direction )
        ####
        nphi, ntheta = mat2d.shape
        theta = np.linspace(theta_range[0], theta_range[1], ntheta)
        phi   = np.linspace(phi_range[0], phi_range[1], nphi)
        ####
        sphere = pv.grid_from_sph_coords(theta, phi, [radius,] )
        sphere.point_data[label] = np.array(mat2d).T.flatten() # note here! The input `grd_value` lat as 0th axis and lon as 1th axis
        sphere.points = Scene3D.rotate_and_translate2(sphere.points, current_orientation_direction1=(0,0,1), current_orientation_direction2=(1,0,0), current_origin_xyz=(0,0,0),
                                                      new_orientation_direction1=north_pole_direction, new_orientation_direction2=lo0la0_direction, new_origin_xyz=center )
        ####
        return self.add_mesh(sphere, scalars=label, cmap=cmap, lighting=lighting, **kwargs)
    def add_earth(self, land_color='#ccccccff', ocean_color='#00000000',
                  coastline_width=0.5, coastline_color='r', coastline_style='-', plot_stock_img=False, dpi=100,
                  radius=1, center=(0,0,0),
                  north_pole_direction=(0., 0., 1.), lo0la0_direction=(1., 0., 0.),
                  theta_resolution=180, start_theta=0, end_theta=360,
                  phi_resolution=90,   start_phi=0,   end_phi=180,
                  lighting=True, **kwargs):
        """
        Add an Earth sphere to the scene with map (as a texture) on it.

        :param land_color:      The color of the land (default is '#ccccccff').
                                Note: support transparency (e.g., through hex value '#RRGGBBAA',
                                where AA=00 for fully transparent, AA=80 for some kind of transparency,
                                and AA=FF for fully not transparent)
        :param ocean_color:     The color of the ocean (default is '#00000000').
        :param coastline_width: The width of the coastline lines (default is 0.5).
        :param coastline_color: The color of the coastline lines (default is 'r').
        :param coastline_style: The style of the coastline lines (default is '-').
        :param plot_stock_img:  Whether to plot the stock image of the Earth (default is `False`).
                                If set to `True`, the stock image will be used as the texture, and
                                `land_color` and `ocean_color` will be ignored, and `coastline_*` parameters
                                will still be used to plot coastlines if intended.
        :param dpi:             The dpi for the earth basemap used as the texture.
        :param radius: The radius of the Earth sphere (default is 1).
        :param center: The center position of the Earth sphere (default is (0, 0, 0)).
        :param north_pole_direction: The direction of the north pole (default is (0, 0, 1)).
        :param lo0la0_direction: The direction of the 0° longitude line (default is (1, 0, 0)).
        :param theta_resolution: The resolution of the sphere in the theta direction.
        :param start_theta: The starting angle of the sphere in the theta direction.
        :param end_theta: The ending angle of the sphere in the theta direction.
        :param phi_resolution: The resolution of the sphere in the phi direction.
        :param start_phi: The starting angle of the sphere in the phi direction.
        :param end_phi: The ending angle of the sphere in the phi direction.
        :param lighting: Whether to use lighting.
        :param kwargs: Additional keyword arguments for `pyvista.Plotter.add_mesh(...)`.

        :return: (mesh, actor)
        Note, the returned `mesh` has an attribute `._texture` that save the texture used when call this function (if the `texture` is not `None`).
        """
        rgba = Scene3D.plot_earth_basemap(land_color=land_color, ocean_color=ocean_color,
                                          coastline_width=coastline_width, coastline_color=coastline_color, coastline_style=coastline_style,
                                          plot_stock_img=plot_stock_img, dpi=dpi)
        return self.add_sphere(radius=radius, center=center,
                                     north_pole_direction=north_pole_direction, lo0la0_direction=lo0la0_direction,
                                     texture=rgba, texture_theta_range=(0,360), texture_phi_range=(180,0),
                                     theta_resolution=theta_resolution, start_theta=start_theta, end_theta=end_theta,
                                     phi_resolution=phi_resolution, start_phi=start_phi, end_phi=end_phi,
                                     lighting=lighting, **kwargs)
    def show(self, **kwargs):
        self.pv_plotter.show(**kwargs)
    @staticmethod
    def benchmark_SKS_ScS_diagram():
        from sacpy.taupplotlib import geo_arrival
        ###
        app = Scene3D(pv_plotter_kargs={'window_size':(3000, 2000)}, )
        ### add a light source
        light = pv.Light(light_type='headlight',position=(-100,-100,100), focal_point=(0,0,0))
        light.diffuse_color = 0.3, 0.3, 0.3
        app.pv_plotter.renderer.add_light(light)
        if True: # add_earth & stock_img & coastlines
            ### Earth map
            norm = (-1,-0.3,1)
            norm2 = (0,0,1)
            norm3 = (np.sqrt(3)*0.5,0.5,0)
            rotate_norm = np.cross(norm2, norm)
            rotate_angle= np.arccos( np.dot(norm2, norm) / (np.linalg.norm(norm2)*np.linalg.norm(norm)) )
            new_norm3 = Scene3D.rotate_about_axis(norm3, rotate_norm, rotate_angle )
            ##
            box = pv.Cube(center=(0, 0, 5000), x_length=20000, y_length=20000, z_length=10000)
            box2 = box.rotate_vector(vector=rotate_norm, angle=np.rad2deg(rotate_angle) )
            box3 = box2.rotate_vector(vector=norm, angle=30)
            #app.add_mesh(box, color='#dddddd', opacity=0.5, lighting=True)
            #app.add_mesh(box2, color='r', opacity=0.5, lighting=True)
            #app.add_mesh(box3, color='b', opacity=0.5, lighting=True)
            ##
            #app.enable_clip(clip_box_kw={'bounds': (-10000, 10000, -10000, 10000, 0, 10000)}, clip_kw=None )
            #app.enable_clip(clip_box_kw=None, clip_kw={'normal': norm, 'origin': (0, 0, 0)} )
            app.enable_clip(clip_box_kw={'bounds': box3}, clip_kw=None )
            app.add_earth(land_color='#999999ff', ocean_color='#00000000', 
                  coastline_width=0, coastline_color='k', coastline_style='-', dpi=100,
                  radius=6372, center=(0,0,0),
                  north_pole_direction=(0., 0., 1.), lo0la0_direction=(1., 0., 0.),
                  theta_resolution=720, start_theta=0, end_theta=360,
                  phi_resolution=360,   start_phi=0,   end_phi=180,
                  lighting=True)
            app.add_sphere(radius=6371, center=(0.,0.,0.), opacity=1,
                   north_pole_direction=(0., 0., 1.), lo0la0_direction=(1., 0., 0.),
                   color='#eeeeee', texture=None,
                   theta_resolution=720, start_theta=0, end_theta=360,
                   phi_resolution=360,   start_phi=0,   end_phi=180,
                   lighting=True)
            app.disable_clip()
            app.add_disk(color='#aaaaaa', center=(0,0,0), outer=6371, inner=3480, direction=norm, show_edges=False, opacity=1.0)
            #app.add_disk(color='#999999', center=(0,0,0), outer=6371, inner=3480, direction=new_norm3, show_edges=False, opacity=1.0)
            #### Inner core
            app.add_sphere(radius=1220, center=(0.,0.,0.), opacity=0.3,
                   north_pole_direction=(0., 0., 1.), lo0la0_direction=(1., 0., 0.),
                   color='k', texture=None,
                   theta_resolution=180, start_theta=0, end_theta=360,
                   phi_resolution=90,   start_phi=0,   end_phi=180,
                   lighting=True)
            #### Outer core
            app.enable_clip(clip_box_kw=None, clip_kw={'normal': norm, 'origin': (0, 0, 0)} )
            app.add_sphere(radius=3480, center=(0.,0.,0.), opacity=1,
                   north_pole_direction=(0., 0., 1.), lo0la0_direction=(1., 0., 0.),
                   color='#ffffff', texture=None,
                   theta_resolution=180, start_theta=0, end_theta=360,
                   phi_resolution=90,   start_phi=0,   end_phi=180,
                   lighting=True)
            app.disable_clip()
            app.add_disk(color='#ffffff', center=(0,0,0), inner=0, outer=3280, direction=norm, show_edges=False, opacity=1.0)
            app.add_disk(color='#ffaaaa', center=(0,0,0), inner=3280, outer=3480, direction=norm, show_edges=False, opacity=1.0)
            #app.add_disk(color='#ffffff', center=(0,0,0), outer=3480, direction=new_norm3, show_edges=False, opacity=1.0)
            #app.add_disk(color='#ffffff', center=(0,0,0), outer=3480, direction=(0,1,0), show_edges=False, opacity=1.0)
            ##### ray paths
            flag_source = False
            evlo = 190
            clr_k = '#C72326'
            for rp, clr_s in zip([], ['#999999']):
                for phase in ['SKSSKSScS', 'ScSScSSKS', ]: #'SKSSKS', 'SKSScS', 'ScSSKS', 'SKSSKSSKS', 'SKSSKSScS']:
                    geo_arr = geo_arrival(0.0, evlo, 0.0, 0.0, phase_name=phase, ray_param=rp, model='PREM')
                    if True: # src and rcv
                        lons, rs = geo_arr.get_raypath()
                        xs = rs * np.cos(lons)
                        ys = rs * np.sin(lons)
                        zs = np.zeros_like(xs)
                        xyz = np.array((xs, ys, zs)).T
                        xyz = Scene3D.rotate_about_axis(xyz, rotate_norm, rotate_angle)
                        xs, ys, zs = xyz.T
                        if not flag_source:
                            flag_source = True
                            app.add_point(xs[0], ys[0], zs[0], size=230, shape='sphere', color='#eeeeee')
                        app.add_point(xs[-1], ys[-1], zs[-1], direction=(-xs[-1], -ys[-1], -zs[-1]), size=230, shape='cone',
                                      color=clr_s,
                                      shift_along_direction=-200, lighting=True)
                    for (leg_name, (lons, rs)) in  geo_arr.get_split_raypath():
                        xs = rs * np.cos(lons)
                        ys = rs * np.sin(lons)
                        zs = np.zeros_like(xs)
                        xyz = np.array((xs, ys, zs)).T
                        xyz = Scene3D.rotate_about_axis(xyz, rotate_norm, rotate_angle)
                        xs, ys, zs = xyz.T
                        clr = clr_k if leg_name=='K' else clr_s
                        lw  = 10  if leg_name=='K' else 6
                        app.add_spline(xs, ys, zs, color=clr, render_lines_as_tubes=True, line_width=lw)
            for rp, clr_s in zip([-400], ['#777777']):
                for phase in ['ScSSKSScS', 'ScSScSScS', ]: #'SKSSKS', 'SKSScS', 'ScSSKS', 'SKSSKSSKS', 'SKSSKSScS']:
                    geo_arr = geo_arrival(0.0, evlo, 0.0, 0.0, phase_name=phase, ray_param=rp, model='PREM')
                    if True: # src and rcv
                        lons, rs = geo_arr.get_raypath()
                        xs = rs * np.cos(lons)
                        ys = rs * np.sin(lons)
                        zs = np.zeros_like(xs)
                        xyz = np.array((xs, ys, zs)).T
                        xyz = Scene3D.rotate_about_axis(xyz, rotate_norm, rotate_angle)
                        xs, ys, zs = xyz.T
                        if not flag_source:
                            flag_source = True
                            app.add_point(xs[0], ys[0], zs[0], size=230, shape='sphere', color='#eeeeee')
                        app.add_point(xs[-1], ys[-1], zs[-1], direction=(-xs[-1], -ys[-1], -zs[-1]), size=230, shape='cone',
                                      color=clr_s,
                                      shift_along_direction=-200, lighting=True)
                    for (leg_name, (lons, rs)) in  geo_arr.get_split_raypath():
                        xs = rs * np.cos(lons)
                        ys = rs * np.sin(lons)
                        zs = np.zeros_like(xs)
                        xyz = np.array((xs, ys, zs)).T
                        xyz = Scene3D.rotate_about_axis(xyz, rotate_norm, rotate_angle)
                        xs, ys, zs = xyz.T
                        clr = 'r' if leg_name=='K' else clr_s
                        lw  = 10  if leg_name=='K' else 6
                        app.add_spline(xs, ys, zs, color=clr, render_lines_as_tubes=True, line_width=lw)
            for rp, clr_s in zip([405,], ['#dddddd',]):
                for phase in ['ScSSKS', 'ScSScS', ]: #'SKSSKS', 'SKSScS', 'ScSSKS', 'SKSSKSSKS', 'SKSSKSScS']:
                    geo_arr = geo_arrival(0.0, evlo, 0.0, 0.0, phase_name=phase, ray_param=rp, model='PREM')
                    if True: # src and rcv
                        lons, rs = geo_arr.get_raypath()
                        xs = rs * np.cos(lons)
                        ys = rs * np.sin(lons)
                        zs = np.zeros_like(xs)
                        xyz = np.array((xs, ys, zs)).T
                        xyz = Scene3D.rotate_about_axis(xyz, rotate_norm, rotate_angle)
                        xs, ys, zs = xyz.T
                        if not flag_source:
                            flag_source = True
                            app.add_point(xs[0], ys[0], zs[0], size=230, shape='sphere', color='#eeeeee')
                        app.add_point(xs[-1], ys[-1], zs[-1], direction=(-xs[-1], -ys[-1], -zs[-1]), size=230, shape='cone',
                                      color=clr_s,
                                      shift_along_direction=-200, lighting=True)
                    for (leg_name, (lons, rs)) in  geo_arr.get_split_raypath():
                        xs = rs * np.cos(lons)
                        ys = rs * np.sin(lons)
                        zs = np.zeros_like(xs)
                        xyz = np.array((xs, ys, zs)).T
                        xyz = Scene3D.rotate_about_axis(xyz, rotate_norm, rotate_angle)
                        xs, ys, zs = xyz.T
                        clr = clr_k if leg_name=='K' else clr_s
                        lw  = 10  if leg_name=='K' else 6
                        app.add_spline(xs, ys, zs, color=clr, render_lines_as_tubes=True, line_width=lw)
            for rp, clr_s in zip([420,], ['#444444']):
                for phase in ['SKS', 'ScS', ]: #'SKSSKS', 'SKSScS', 'ScSSKS', 'SKSSKSSKS', 'SKSSKSScS']:
                    geo_arr = geo_arrival(0.0, evlo, 0.0, 0.0, phase_name=phase, ray_param=rp, model='PREM')
                    if True: # src and rcv
                        lons, rs = geo_arr.get_raypath()
                        xs = rs * np.cos(lons)
                        ys = rs * np.sin(lons)
                        zs = np.zeros_like(xs)
                        xyz = np.array((xs, ys, zs)).T
                        xyz = Scene3D.rotate_about_axis(xyz, rotate_norm, rotate_angle)
                        xs, ys, zs = xyz.T
                        if not flag_source:
                            flag_source = True
                            app.add_point(xs[0], ys[0], zs[0], size=230, shape='sphere', color='#eeeeee')
                        app.add_point(xs[-1], ys[-1], zs[-1], direction=(-xs[-1], -ys[-1], -zs[-1]), size=230, shape='cone',
                                      color=clr_s,
                                      shift_along_direction=-200, lighting=True)
                    for (leg_name, (lons, rs)) in  geo_arr.get_split_raypath():
                        xs = rs * np.cos(lons)
                        ys = rs * np.sin(lons)
                        zs = np.zeros_like(xs)
                        xyz = np.array((xs, ys, zs)).T
                        xyz = Scene3D.rotate_about_axis(xyz, rotate_norm, rotate_angle)
                        xs, ys, zs = xyz.T
                        clr = clr_k if leg_name=='K' else clr_s
                        lw  = 10  if leg_name=='K' else 6
                        app.add_spline(xs, ys, zs, color=clr, render_lines_as_tubes=True, line_width=lw)
        #app.pv_plotter.add_axes()
        app.pv_plotter.camera_position =  [ (-28290.974432366187, 5512.347382088301, -4120.525256893221),
                                            (0.490478515625, 0.0, 0.0),
                                            (-0.2026463078455576, -0.3590334581430495, 0.9110595204762457)]
        #app.show(screenshot='benchmark_SKS_ScS_diagram.png')
        #####
        # print camera position
        print("Camera position:", app.pv_plotter.camera_position)
        # plot legend
        app = Scene3D(pv_plotter_kargs={'window_size':(800, 600)}, )
        app.add_point(230,1700,0, size=230, shape='sphere', color='#eeeeee')
        app.add_point(230,1000,0, size=230, shape='cone', direction=(0,-1,-0.5), color='#eeeeee')
        app.add_spline([0, 500], [500,500], [0,0], color='r', render_lines_as_tubes=True, line_width=10)
        app.add_spline([0, 500], [100,100], [0,0], color='#aaaaaa', render_lines_as_tubes=True, line_width=10)
        app.pv_plotter.view_xy()
        app.show(screenshot='benchmark_SKS_ScS_diagram_legend.png')
    @staticmethod
    def benchmark_SKS_ScS_diagram_quarter():
        from sacpy.taupplotlib import geo_arrival
        ###
        app = Scene3D(pv_plotter_kargs={'window_size':(3000, 2000)})
        if True: # add_earth & stock_img & coastlines
            ### Earth map
            norm = (-1,-0.3,1)
            norm2 = (0,0,1)
            norm3 = (np.sqrt(3)*0.5,0.5,0)
            rotate_norm = np.cross(norm2, norm)
            rotate_angle= np.arccos( np.dot(norm2, norm) / (np.linalg.norm(norm2)*np.linalg.norm(norm)) )
            new_norm3 = Scene3D.rotate_about_axis(norm3, rotate_norm, rotate_angle )
            ##
            box = pv.Cube(center=(5000, 0, 5000), x_length=10000, y_length=20000, z_length=10000)
            box2 = box.rotate_vector(vector=rotate_norm, angle=np.rad2deg(rotate_angle) )
            box3 = box2.rotate_vector(vector=norm, angle=30)
            #app.add_mesh(box, color='#dddddd', opacity=0.5, lighting=True)
            #app.add_mesh(box2, color='r', opacity=0.5, lighting=True)
            #app.add_mesh(box3, color='b', opacity=0.5, lighting=True)
            ##
            #app.enable_clip(clip_box_kw={'bounds': (-10000, 10000, -10000, 10000, 0, 10000)}, clip_kw=None )
            #app.enable_clip(clip_box_kw=None, clip_kw={'normal': norm, 'origin': (0, 0, 0)} )
            app.enable_clip(clip_box_kw={'bounds': box3}, clip_kw=None )
            app.add_earth(land_color='#999999ff', ocean_color='#00000000', 
                  coastline_width=0.5, coastline_color='k', coastline_style='-', dpi=100,
                  radius=6371, center=(0,0,0),
                  north_pole_direction=(0., 0., 1.), lo0la0_direction=(1., 0., 0.),
                  theta_resolution=180, start_theta=0, end_theta=360,
                  phi_resolution=90,   start_phi=0,   end_phi=180,
                  lighting=True)
            app.add_sphere(radius=6371, center=(0.,0.,0.), opacity=1,
                   north_pole_direction=(0., 0., 1.), lo0la0_direction=(1., 0., 0.),
                   color='#eeeeee', texture=None,
                   theta_resolution=180, start_theta=0, end_theta=360,
                   phi_resolution=90,   start_phi=0,   end_phi=180,
                   lighting=True)
            app.disable_clip()
            app.add_disk(color='#999999', center=(0,0,0), outer=6371, inner=3480, direction=norm, show_edges=False, opacity=1.0)
            app.add_disk(color='#999999', center=(0,0,0), outer=6371, inner=3480, direction=new_norm3, show_edges=False, opacity=1.0)
            #### Inner core
            app.add_sphere(radius=1220, center=(0.,0.,0.), opacity=0.5,
                   north_pole_direction=(0., 0., 1.), lo0la0_direction=(1., 0., 0.),
                   color='k', texture=None,
                   theta_resolution=180, start_theta=0, end_theta=360,
                   phi_resolution=90,   start_phi=0,   end_phi=180,
                   lighting=True)
            #### Outer core
            app.enable_clip(clip_box_kw=None, clip_kw={'normal': norm, 'origin': (0, 0, 0)} )
            app.add_sphere(radius=3480, center=(0.,0.,0.), opacity=1,
                   north_pole_direction=(0., 0., 1.), lo0la0_direction=(1., 0., 0.),
                   color='#ffffff', texture=None,
                   theta_resolution=180, start_theta=0, end_theta=360,
                   phi_resolution=90,   start_phi=0,   end_phi=180,
                   lighting=True)
            app.disable_clip()
            app.add_disk(color='#ffffff', center=(0,0,0), outer=3480, direction=norm, show_edges=False, opacity=1.0)
            app.add_disk(color='#ffffff', center=(0,0,0), outer=3480, direction=new_norm3, show_edges=False, opacity=1.0)
            #app.add_disk(color='#ffffff', center=(0,0,0), outer=3480, direction=(0,1,0), show_edges=False, opacity=1.0)
            ##### ray paths
            for rp in np.linspace(120, 400, 10):
                #### SKS
                geo_arr = geo_arrival(0.0, 0.0, 0.0, phase_name='SKS', ray_param=rp, model='PREM')
                for (leg_name, (lons, rs)) in  geo_arr.get_split_raypath():
                    xs = rs * np.cos(1.57-lons)
                    ys = rs * np.sin(1.57-lons)
                    zs = np.zeros_like(xs)
                    xyz = np.array((xs, ys, zs)).T
                    xyz = Scene3D.rotate_about_axis(xyz, rotate_norm, rotate_angle)
                    xs, ys, zs = xyz.T
                    clr = 'r' if leg_name=='K' else 'k'
                    lw  = 10  if leg_name=='K' else 4
                    app.add_spline(xs, ys, zs, color=clr, line_width=lw)
                #### ScS
                geo_arr = geo_arrival(0.0, 0.0, 0.0, phase_name='ScS', ray_param=rp, model='PREM')
                lons, rs = geo_arr.get_raypath()
                xs = rs * np.cos(1.57-lons)
                ys = rs * np.sin(1.57-lons)
                zs = np.zeros_like(xs)
                xyz = np.array((xs, ys, zs)).T
                xyz = Scene3D.rotate_about_axis(xyz, rotate_norm, rotate_angle)
                xs, ys, zs = xyz.T
                app.add_spline(xs, ys, zs, color='k', line_width=4)
                #
        app.pv_plotter.add_axes()
        app.pv_plotter.camera_position =  [(-1656.9956662285072, 13577.474367935813, 25703.63045028977),
                                            (0.490478515625, 0.0, 0.0),
                                            (-0.7972962491881405, -0.5534216001319333, 0.2409216128493525)]
        app.show()
        # print camera position
        print("Camera position:", app.pv_plotter.camera_position)
    @staticmethod
    def rotate_and_translate2(xyz, current_origin_xyz, current_orientation_direction1, current_orientation_direction2,
                               new_origin_xyz, new_orientation_direction1, new_orientation_direction2 ):
        ####
        """
        We (1) rotate object defined by `xyz`, of shape (3, n_points) and with 3D orientation defined
        by `current_orientation_direction1` and `current_orientation_direction2`, so that  the resulted
        object will have new 3D orientation defined by `new_orientation_direction1`, `new_orientation_direction2`;
        and (2) translate the object so that its origin moves from `current_origin_xyz` to `new_origin_xyz`.

        The math:
        Current orientation orthogonal unit vectors: u1, u2, (and u3=u1xu2)
        New orientation orthogonal unit vectors: v1, v2, (and v3=v1xv2)
        (Note: we need angle(u1, u2) == angle(v1, v2)! )

        v1 = R u1
        v2 = R u2  or  [v1, v2, v3] = R [u1, u2, u3] ==> V = R U ==> V U^(-1) = R ==> R = V U^(-1)
        v3 = R u3

        Also, as u1, u2, u3 are orthogonal (ui.T u = delta_ij),
        so,   U.T U= [u1.T] [u1, u2, u3]  = [1 0 0] = I ==> U.T U = I ==>  U.T = U^(-1)
                     [u2.T]                 [0 1 0]
                     [u3.T]                 [0 0 1]

        So, R = V U.T.

        :param xyz: the xyz coordinates of points to be transformed. The shape of xyz could be either (N_points, 3) or (3, N_points).
        :param current_origin_xyz:             the current origin point for defining the object.
        :param current_orientation_direction1: the current orientation direction 1 for defining the object.
        :param current_orientation_direction2: the current orientation direction 2 for defining the object.
        :param new_origin_xyz:             the new origin point for defining the object.
        :param new_orientation_direction1: the new orientation direction 1 for defining the object.
        :param new_orientation_direction2: the new orientation direction 2 for defining the object.
        :return xyz: the new xyz coordinates after rotation and translation. It has the same shape as the input `xyz`.
        """
        current_orientation_direction1 = np.asarray(current_orientation_direction1, dtype=np.float64) / np.linalg.norm(current_orientation_direction1)
        current_orientation_direction2 = np.asarray(current_orientation_direction2, dtype=np.float64) / np.linalg.norm(current_orientation_direction2)
        new_orientation_direction1     = np.asarray(new_orientation_direction1,     dtype=np.float64) / np.linalg.norm(new_orientation_direction1)
        new_orientation_direction2     = np.asarray(new_orientation_direction2,     dtype=np.float64) / np.linalg.norm(new_orientation_direction2)
        ####
        u1 = current_orientation_direction1
        u2 = np.cross(u1, current_orientation_direction2)
        u2 /=np.linalg.norm(u2) ####
        u3 = np.cross(u1, u2)
        v1 = new_orientation_direction1
        v2 = np.cross(v1, new_orientation_direction2)
        v2 /=np.linalg.norm(v2) ####
        v3 = np.cross(v1, v2)
        UT = np.array([u1, u2, u3], dtype=np.float64)
        VT = np.array([v1, v2, v3], dtype=np.float64)
        R  = np.transpose(VT) @ UT
        RT = np.transpose(R)
        ####
        nrow, ncol = xyz.shape
        if nrow == 3:
            current_origin_xyz = np.asarray(current_origin_xyz, dtype=np.float64).reshape((3,1))
            new_origin_xyz     = np.asarray(new_origin_xyz,     dtype=np.float64).reshape((3,1))
            xyz = np.asarray(xyz, dtype=np.float64)
            xyz = xyz - current_origin_xyz
            xyz = R @ xyz
            xyz += new_origin_xyz
            pass
        elif ncol == 3:
            current_origin_xyz = np.asarray(current_origin_xyz, dtype=np.float64).reshape((1,3))
            new_origin_xyz     = np.asarray(new_origin_xyz,     dtype=np.float64).reshape((1,3))
            xyz = np.asarray(xyz, dtype=np.float64)
            xyz = xyz - current_origin_xyz
            xyz = xyz @ RT ############ xyz.T = R @ xyz.T
            xyz += new_origin_xyz
        else:
            raise ValueError("The input `xyz` must have one dimension as 3 (for x,y,z)! The shape of `xyz` is: ", xyz.shape)
        ####
        return xyz
    @staticmethod
    def rotate_and_translate(mesh, current_mesh_center, current_norm, rotation_about_current_norm_deg,
                             new_norm, new_mesh_center):
        """
        Update the points of a mesh object due to rotation and translation.
        Will do (1) rotate about current norm, (2) rotate so that the norm aligned with new_norm, and (3) translate.

        :param mesh: The mesh object to update.
        :param current_mesh_center: The current center of the mesh.
        :param current_norm: The current normal direction of the mesh.
        :param rotation_about_current_norm_deg: The rotation angle about the current normal direction.
        :param new_norm: The new normal direction of the mesh.
        :param new_mesh_center: The new center position of the mesh.
        """
        #### 1. rotate about the current norm
        if (rotation_about_current_norm_deg%360) > 1e-10:
            angle = np.deg2rad(rotation_about_current_norm_deg)
            mesh.points = Scene3D.rotate_about_axis(mesh.points, rotation_axis_direction=current_norm,
                                                    anticlockwise_rotation_angle_rad=angle, rotation_axis_start_point=current_mesh_center)
        #### 2. rotate to change the norm direction
        if True:
            v0  = np.asarray(current_norm, dtype=np.float64)
            v1  = np.asarray(new_norm,     dtype=np.float64)
            v0  = v0 / np.linalg.norm(v0)
            v1  = v1 / np.linalg.norm(v1)
            #          v0
            #      v1  |
            #        \ |
            #         \|
            #          o
            #         /
            #        /rotation_axis =  v0 x v1
            angle = np.arccos(np.clip(np.dot(v0, v1), -1.0, 1.0) )
            if angle > 1.e-6:
                rotation_axis = np.cross(v0, v1)
                rotation_axis /= np.linalg.norm(rotation_axis)
                mesh.points = Scene3D.rotate_about_axis(mesh.points, rotation_axis, angle, rotation_axis_start_point=current_mesh_center)
        #### 4. translate
        mesh.points += ( np.asanyarray(new_mesh_center) - np.asarray(current_mesh_center) )
    @staticmethod
    def rotate_about_axis(xyz, rotation_axis_direction=(0,0,1), anticlockwise_rotation_angle_rad=0, rotation_axis_start_point=(0,0,0) ):
        """
        Rotation points w.r.t. an axis.

        :param xyz: in the shape of a list of (x, y, z) coordinates to rotate (e.g., a ndarray matrix with shape (nrow, 3) ).
        :param rotation_axis_direction: the axis direction to rotate around, in the format of (axis_x, axis_y, axis_z).
        :param anticlockwise_rotation_angle_rad: the angle (in radian) to rotate about the rotation axis.
        :param rotation_axis_start_point: the start point of the rotation axis vector (default is (0,0,0) ).

        :return: xyz: The rotated coordinates.
        """
        anticlockwise_rotation_angle_rad = (anticlockwise_rotation_angle_rad) % (2*np.pi)
        if anticlockwise_rotation_angle_rad > 1e-10:
            #### move the origin to `rotation_axis_start_point`
            xyz = np.asarray(xyz, dtype=np.float64)
            rotation_axis_start_point = np.asarray(rotation_axis_start_point, dtype=np.float64)
            xyz = xyz - rotation_axis_start_point  # move the origin to the rotation axis start point
            ####
            raxis = np.array(rotation_axis_direction, dtype=np.float64)
            raxis /= np.linalg.norm(raxis)  # ensure unit
            R = Scene3D.rotation_matrix(raxis, anticlockwise_rotation_angle_rad)
            #
            # [ x']   [   ][ x]
            # [ y'] = [ R ][ y]  ==> xyz'.T  = R @ xyz.T  ==> xyz' = xyz @ R.T
            # [ z']   [   ][ z]
            #
            RT = R.T
            xyz = xyz @ RT
            #### move the origin back
            xyz += rotation_axis_start_point
        return xyz
    @staticmethod
    def rotation_matrix(axis, angle):
        """
        Get the rotation matrix for rotation about `axis` (an vector) by `angle` (radian).

        :param axis:  The rotation axis (3D vector).
        :param angle: The rotation angle (in radians).
        :return:      The rotation matrix (3x3 numpy array).
        """
        axis = np.array(axis, dtype=np.float64)
        axis /= np.linalg.norm(axis)  # ensure unit
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        C = 1 - c

        R = np.array([
            [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
            [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
        ])
        return R
    @staticmethod
    def plot_earth_basemap(land_color='#bbbbbbFF', ocean_color='#bbbbbb00',
                           coastline_width=0.5, coastline_color='r', coastline_style='-',
                           plot_stock_img=False, dpi=100):
        prj = ccrs.PlateCarree(central_longitude=180)
        fig = plt.figure(figsize=(20, 10), dpi=dpi)
        fig.patch.set_alpha(0.0) # set background to transparent
        #
        ax = fig.add_axes([0, 0, 1, 1], projection=prj, frame_on=False)  # (left, bottom, width, height) in figure coords
        ax.set_frame_on(False) # Remove the axis frame
        ax.margins(0,0)
        ax.patch.set_alpha(0.0)
        #
        ax.set_global()
        if plot_stock_img:
            ax.stock_img()
        else:
            ax.add_feature(cfeature.LAND, color=land_color,  linewidth=0 )
            ax.add_feature(cfeature.OCEAN, color=ocean_color, linewidth=0 )
        ####
        if coastline_width>0:
            ax.coastlines(linewidth=coastline_width, color=coastline_color, linestyle=coastline_style)
        ax.axis("off")
        ####
        fig.canvas.draw()
        rgba = np.array(fig.canvas.buffer_rgba())
        plt.close(fig)
        ####
        if plot_stock_img: # fix some tiny error transparency
            nrow, ncol = rgba.shape[:2]
            sum_alpha = 255*ncol
            for _ in range(nrow):
                if np.sum(rgba[0, :, 3]) != sum_alpha:
                    rgba = rgba[1:, :, :]
                else:
                    break
            nrow, ncol = rgba.shape[:2]
            for _ in range(nrow):
                if np.sum(rgba[-1, :, 3]) !=sum_alpha:
                    rgba = rgba[:-1, :, :]
                else:
                    break
            #nrow, ncol = rgba.shape[:2]
            #for irow in range(nrow):
            #    if np.sum(rgba[irow, :, 3]) == 0:
            #        rgba[irow:, :, 3] = 255
            #    else:
            #        break
            #for irow in range(-nrow, 0, -1):
            #    if np.sum(rgba[-1, :, 3]) == 0:
            #        rgba[:irow, :, 3] = 255
            #    else:
            #        break
        ####
        return rgba
    @staticmethod
    def figure2mat2d(filename, cmap=None):
        pixels = plt.imread(filename)  # shape (H, W, 3) or (H, W, 4)
        if pixels.shape[2] == 4:  # remove alpha if exists
            pixels = pixels[:, :, :3]
        #
        unique_colors = np.unique(pixels, axis=0)
        brightness    = unique_colors.mean(axis=1)
        sorted_idx    = np.argsort(brightness)
        sorted_colors = unique_colors[sorted_idx]*(1.0/255)
        # === Step 2: Make a colormap ===
        cmap = ListedColormap(sorted_colors )
        #
        # Build a lookup table from the cmap
        N = 64
        lut_vals = cmap(np.linspace(0, 1, N))[:, :3]  # RGB only
        # Flatten image for processing
        flat_img = pixels.reshape(-1, 3)
        # Compute nearest colormap index for each pixel
        indices = np.argmin(np.linalg.norm(flat_img[:, None] - lut_vals[None, :], axis=2), axis=1)
        # Convert indices to scalar values
        scalar_values = indices / (N - 1)  # normalized 0-1
        mat2d = scalar_values.reshape(pixels.shape[0], pixels.shape[1])
        return mat2d, cmap
    @staticmethod
    def benchmark():
        app = Scene3D()
        #app.pv_plotter.add_axes(xlabel='E', ylabel='N', zlabel='U')
        #
        app.add_plane(color='gray', direction=(0,0,1), show_edges=False, opacity=0.5)
        app.add_plane(color='gray', direction=(0,1,0), show_edges=False, opacity=0.5)
        app.add_plane(color='gray', direction=(1,0,0), show_edges=False, opacity=0.5)
        app.enable_clip(clip_kw={'normal':(1,0,0), 'invert': False, 'origin':(0.5,0,0)}, clip_box_kw=None)
        if False: #
            #app.add_disk(color='r', outer=5*1.414,direction=(1,0,-1), show_edges=False, opacity=0.5)#, rotation_about_norm_deg=0)
            #app.add_point(-1, 2, -1, color='r', size=0.2)
            #app.add_point(1, -2, 1, color='r', size=0.1, opacity=0.2,)
            app.add_disk(color='g', outer=5*1.414,direction=(1,-1,0), show_edges=False, opacity=0.5)#, rotation_about_norm_deg=45)
            app.add_point(1, 1, -2, color='g', size=0.2)
            app.add_point(-1, -1, 2, color='g', size=0.1, opacity=0.2)
            app.add_disk(color='b', outer=5*1.414,direction=(0,1,-1), show_edges=False, opacity=0.5)#, rotation_about_norm_deg=0)
            app.add_point(2, -1, -1, color='b', size=0.2)
            app.add_point(-2, 1, 1, color='b', size=0.1, opacity=0.2)
            app.add_disk(color='k', outer=5*1.414, direction=(1,1,1), show_edges=True, opacity=0.5)
            pass
        if False:
            app.add_plane(center=(0,0,0), direction=(1,1,1), x_direction=(1,-1,0), show_edges=True, opacity=0.5, color='b', )
            app.add_plane(center=(0,0,3), direction=(1,1,1), x_direction=(1,-1,0), show_edges=True, opacity=0.5, color='r', )
            pass
        if False:
            app.add_disk(center=(0,0,0), direction=(1,1,1), x_direction=(1,-1,0), outer=10, show_edges=True, opacity=0.5, color='b', )
            app.add_disk(center=(0,0,-3), direction=(1,1,1), x_direction=(1,-1,0), outer=10, show_edges=True, opacity=0.5, color='r', )
        if False:
            xs = np.linspace(-5, 5, 1000)
            yz, zs = np.cos(5*xs), np.sin(5*xs)-2
            app.add_spline(xs, yz, zs, scalars=zs, color='k', cmap='viridis', line_width=20, opacity=1) #, scalar_bar_args={'vertical':True} )
            app.add_sphere(radius=2, center=(0,0,0),
                           color='g', opacity=0.9)
            for shape in ('cone', 'cylinder', 'sphere'):
                loc = np.random.rand(3)
                loc *= (2/np.linalg.norm(loc))
                x, y, z = loc
                app.add_point(x, y, z, -loc, 0.2, shape=shape, color='r', shift_along_direction=-0.2)
        if False:
            app.add_arrow(loc=(0,0,0), direction=(1, 0, 0), scale=5)
            app.add_arrow(loc=(0,0,0), direction=(1, 0, 0), loc_is_end=True, color='gray', scale=5)
            app.add_arrow(loc=(0,0,0), direction=(0, 1, 0), scale=-5, color=(255, 0, 0))
            app.add_arrow(loc=(0,0,0), direction=(0, 1, 0), loc_is_end=True, color=(255, 150, 150), scale=-5)
        if False:
            app.add_sphere(radius=2, center=(0,0,0),
                           color='g', opacity=0.9,
                           theta_resolution=60, start_theta=30, end_theta=300,
                           phi_resolution=30,   start_phi=20,   end_phi=150,)
        if False: # add_sphere & figure texture
            mesh, actor = app.add_sphere(radius=2, center=(0,0,0),
                                         texture='/Users/sw/Programs_Sheng/sacpy/dataset/global_maps/fancy2_0-360.png',
                                         texture_theta_range=(0, 360), texture_phi_range=(180, 0),
                                         theta_resolution=60, start_theta=330, end_theta=30,
                                         phi_resolution=30,   start_phi=0,   end_phi=180, culling=False)
        if False: # add_sphere & figure texture & translate & rotation
            app.add_disk(color='r', center=(1,1,1), outer=3, direction=(1,-1,0), show_edges=True, opacity=0.1)
            app.add_disk(color='r', center=(1,1,1), outer=3, direction=(1,1,-2), show_edges=True, opacity=0.1)
            app.add_disk(color='r', center=(1,1,1), outer=3, direction=(1,1,1), show_edges=True, opacity=0.1)
            mesh, actor = app.add_sphere(radius=2, center=(1,1,1),
                                         texture='/Users/sw/Programs_Sheng/sacpy/dataset/global_maps/fancy2_0-360.png',
                                         texture_theta_range=(0, 360), texture_phi_range=(180, 0),
                                         theta_resolution=60, start_theta=315, end_theta=45,
                                         north_pole_direction=(1,1,1), lo0la0_direction=(-1,-1,2),
                                         phi_resolution=30,   start_phi=0,   end_phi=180, culling=False)
        if False: # add_sphere & rgba grd texture
            grd_rgba = np.array(
                [
                    [ 0,0,0,100], [255, 255, 255, 255],  [255, 0, 0, 255],
                    [255,255,255,255], [0,   0, 0, 100], [0, 255, 0, 255],
                    [0,0,255,255], [255, 255, 255, 255], [0, 0,   0, 100],
                ],
                dtype=np.uint8
            ).reshape((3, 3, 4), order='C')#[::-1,:,:]  # (H, W, 4)
            plt.imshow(grd_rgba, interpolation='nearest')
            plt.show()
            app.add_sphere(radius=2, center=(0,0,0),
                           texture=grd_rgba,
                           north_pole_direction=(1,1,1), lo0la0_direction=(1,-1,0),
                           culling=True)
            app.add_sphere(radius=1, color='k')
        if False: # add_sphere & single-value grd texture
            grd_rgba = np.array(
                [
                    [ 0,0,0,100], [255, 255, 255, 255],  [255, 0, 0, 255],
                    [255,255,255,255], [0,   0, 0, 100], [0, 255, 0, 255],
                    [0,0,255,255], [255, 255, 255, 255], [0, 0,   0, 100],
                ],
                dtype=np.uint8
            ).reshape((3, 3, 4), order='C')#[::-1,:,:]  # (H, W, 4)
            grd_rgba = grd_rgba[:,:,0]
            plt.imshow(grd_rgba, interpolation='nearest')
            plt.show()
            app.add_sphere(radius=2, center=(0,0,0), cmap='bwr',
                           texture=grd_rgba,
                           culling=True)
            app.add_sphere(radius=1, color='k')
        if False: # add_sphere_grd & 2d-ndarray grd
            app.add_disk(color='r', center=(1, 1, 1), outer=6, direction=(1,-1,0), show_edges=True, opacity=0.1)
            app.add_disk(color='r', center=(1, 1, 1), outer=6, direction=(1,1,-2), show_edges=True, opacity=0.1)
            app.add_disk(color='r', center=(1, 1, 1), outer=6, direction=(1,1,1), show_edges=True, opacity=0.1)
            #
            theta_range = (0, 360)
            phi_range = (0, 180)
            theta = np.linspace(theta_range[0], theta_range[1], 360)
            phi = np.linspace(phi_range[0], phi_range[1], 180)
            grd = np.zeros( (phi.size, theta.size, ) )
            grd[:90,:90] = 1
            grd[40:50, 90:180] = -1
            mesh, actor = app.add_sphere_grd(theta_range, phi_range, grd,
                                             radius=5, center=(1,1,1),
                                             north_pole_direction=(1,1,1), lo0la0_direction=(1,1,-2),
                                             cmap='bwr', label='test', opacity=0.0, show_edges=True,)
            app.pv_plotter.add_mesh(mesh, culling=True, cmap='bwr',)
        if False: # add_earth & land & transparent ocean & coastlines
            #app.add_disk(color='r', center=(1, 1, 1), outer=6, normal=(1,-1,0), show_edges=True, opacity=0.1)
            #app.add_disk(color='r', center=(1, 1, 1), outer=6, normal=(1,1,-2), show_edges=True, opacity=0.1)
            #app.add_disk(color='r', center=(1, 1, 1), outer=6, normal=(1,1,1), show_edges=True, opacity=0.1)
            app.add_earth( land_color='#ccccccff', ocean_color='#0000ff11',
                            coastline_color='r', coastline_style='-', coastline_width=0.6, #2.0,
                            radius=5, center=(1, 1, 1),
                            start_theta=-50, end_theta=50, start_phi=30, end_phi=170,
                            culling=True, show_edges=False)
            app.add_sphere(radius=3, center=(1, 1, 1), color='g')
        if True: # add_earth & stock_img & coastlines
            app.add_disk(color='r', center=(1, 1, 1), outer=6, direction=(1,-1,0), show_edges=True, opacity=0.1)
            app.add_disk(color='r', center=(1, 1, 1), outer=6, direction=(1,1,-2), show_edges=True, opacity=0.1)
            app.add_disk(color='r', center=(1, 1, 1), outer=6, direction=(1,1,1), show_edges=True, opacity=0.1)
            app.add_earth( plot_stock_img=True,
                            coastline_color='r', coastline_style='-', coastline_width=0.6, #2.0,
                            radius=5, center=(1, 1, 1),
                            north_pole_direction=(1,1,1), lo0la0_direction=(1, 1, -2),
                            #start_theta=-50, end_theta=50, start_phi=30, end_phi=170,
                            theta_resolution=30, phi_resolution=15,
                            culling=False, show_edges=False)
            app.add_sphere(radius=3, center=(1, 1, 1), color='b')
        app.pv_plotter.add_axes()
        app.show()
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
    @staticmethod
    def benchmark2():
        """
        Plot global inter-receiver and inter-source geometries.
        """
        p = pv.Plotter(notebook=0, shape=(1, 2), border=False, window_size=(3500, 2000) )
        p.set_background('white')
        #
        center = (0, 0, 0)
        R0 = 6371.0
        globe = globe3d(R0, center)
        # use obspy to download station longitude and latitudes of a network 'II'
        from obspy.clients.fdsn import Client
        client = Client("IRIS")
        inventory = client.get_stations(network='II,IU,G', level='station')
        lons, las = [], []
        for network in inventory:
            for station in network:
                lons.append(station.longitude)
                las.append(station.latitude)
        lons = np.array(lons)
        las = np.array(las)
        # plot 2d globe map
        n_points = 100
        ref_lo, ref_la = -76, 9
        # select lons and lats so that -130<lo<-30 and -30<la<30
        selected_idxs = (lons>-75)*(lons<-30)*(las>-20)*(las<15)
        selected_lons = lons[selected_idxs]
        selected_las  = las[selected_idxs]
        #n_gc = 30
        #i1i2 = np.random.randint(0, n_points, (n_gc, 2))
        #print(i1i2.shape)
        #class LowerThresholdGeodetic(ccrs.Geodetic):
        #    @property
        #    def threshold(self):
        #        return 1e1
        #class LowerThresholdPlateCarree(ccrs.Geodetic):
        #    @property
        #    def threshold(self):
        #        return 1e1
        prj1 = ccrs.PlateCarree(central_longitude=0)
        prj3 = ccrs.Geodetic()
        prj1.threshold = 0.001
        # plot 2d globe map inter-receiver and inter-source
        for marker, ref_marker, color, ref_color, fnm, sz in zip('*^', '^*', ('#fc7200', '#0d90e0'), ('#0d90e0', '#fc7200'), ('2.png', '1.png'), (12, 8)):
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(1, 1, 1, projection=prj1)
            ax.set_global()
            ax.add_feature(cfeature.LAND, color='#cccccc', alpha=1.0)
            ax.add_feature(cfeature.OCEAN, color='#ffffff', alpha=1.0)
            ax.plot(lons, las, marker, markerfacecolor=color, transform=prj3, markeredgewidth=0, markeredgecolor='k', markersize=sz)
            ax.plot(ref_lo, ref_la, ref_marker, markerfacecolor=ref_color, transform=prj3, markeredgewidth=0, markeredgecolor='k', markersize=(20-sz)*1.2)
            # plot great circle paths
            # generate a random list, each list element is a tuple of two integers, representing the indices of the two points
            for i1 in range(selected_lons.size):
                lo1, la1 = selected_lons[i1], selected_las[i1]
                for i2 in range(i1+1, selected_lons.size)[::2]:
                    lo2, la2 = selected_lons[i2], selected_las[i2]
                    ax.plot([lo1, lo2], [la1, la2], '--', color='k', transform=ccrs.Geodetic(), linewidth=0.6, zorder=0)
            ax.axis('off')
            plt.savefig(fnm, bbox_inches = 'tight', pad_inches = 0, dpi=300, transparent=True)
            plt.close()
        #
        p.subplot(0, 0)
        plot_globe3d(p, globe, style='1.png', alpha=1, land='#999999', ocean='#ffffff'  ) #('plane', (normal, origin, invert) )
        p.camera_position = [(7164.4456150185315, -32959.49111285338, -3463.2338387224036), (0.0, 0.0, 0.0), (0.011160759659876018, -0.10209381453235096, 0.9947121646376144)]
        p.subplot(0, 1)
        plot_globe3d(p, globe, style='2.png', alpha=1, land='#999999', ocean='#ffffff'  ) #('plane', (normal, origin, invert) )
        p.camera_position = [(7164.4456150185315, -32959.49111285338, -3463.2338387224036), (0.0, 0.0, 0.0), (0.011160759659876018, -0.10209381453235096, 0.9947121646376144)]
        #
        p.show(screenshot='correlation.png')
        print(p.camera_position)
        pass
    @staticmethod
    def benchmark3():
        """
        Plot global inter-receiver and inter-source geometries.
        """
        p = pv.Plotter(notebook=0, shape=(1, 2), border=False, window_size=(3500, 2000) )
        p.set_background('white')
        #
        center = (0, 0, 0)
        R0 = 6371.0
        globe = globe3d(R0, center)
        # use obspy to download station longitude and latitudes of a network 'II'
        from obspy.clients.fdsn import Client
        client = Client("IRIS")
        inventory = client.get_stations(network='II,IU,G', level='station')
        lons, las = [], []
        for network in inventory:
            for station in network:
                lons.append(station.longitude)
                las.append(station.latitude)
        lons = np.array(lons)
        las = np.array(las)
        cat = client.get_events(starttime="2010-01-01", endtime="2015-01-02", minmagnitude=6.0)
        evlons, evlas, evdeps = [], [], []
        for event in cat:
            evlons.append(event.origins[0].longitude)
            evlas.append(event.origins[0].latitude)
            evdeps.append(event.origins[0].depth)
        #
        # plot 2d globe map
        n_points = 100
        ref_lo, ref_la = -76, 9
        # select lons and lats so that -130<lo<-30 and -30<la<30
        selected_idxs = (lons>-75)*(lons<-30)*(las>-20)*(las<15)
        selected_lons = lons[selected_idxs]
        selected_las  = las[selected_idxs]
        #n_gc = 30
        #i1i2 = np.random.randint(0, n_points, (n_gc, 2))
        #print(i1i2.shape)
        #class LowerThresholdGeodetic(ccrs.Geodetic):
        #    @property
        #    def threshold(self):
        #        return 1e1
        #class LowerThresholdPlateCarree(ccrs.Geodetic):
        #    @property
        #    def threshold(self):
        #        return 1e1
        prj1 = ccrs.PlateCarree(central_longitude=0)
        prj3 = ccrs.Geodetic()
        prj1.threshold = 0.001
        # plot 2d globe map inter-receiver and inter-source
        for marker, ref_marker, color, ref_color, fnm, sz in zip('^^', '**', ('#0d90e0', '#0d90e0'), ('#fc7200', '#fc7200'), ('2.png', '1.png'), (12, 8)):
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(1, 1, 1, projection=prj1)
            ax.set_global()
            ax.add_feature(cfeature.LAND, color='#cccccc', alpha=1.0)
            ax.add_feature(cfeature.OCEAN, color='#ffffff', alpha=1.0)
            if sz == 12:
                ax.plot(lons, las, marker, markerfacecolor=color, transform=prj3, markeredgewidth=0, markeredgecolor='k', markersize=sz)
                ax.plot(evlons, evlas, ref_marker, markerfacecolor=ref_color, transform=prj3, markeredgewidth=0, markeredgecolor='k', markersize=15)
            ax.plot(ref_lo, ref_la, ref_marker, markerfacecolor=ref_color, transform=prj3, markeredgewidth=0, markeredgecolor='k', markersize=15)
            # plot great circle paths
            # generate a random list, each list element is a tuple of two integers, representing the indices of the two points
            flag = True
            for i1 in range(selected_lons.size):
                lo1, la1 = selected_lons[i1], selected_las[i1]
                for i2 in range(i1+1, selected_lons.size)[::2]:
                    lo2, la2 = selected_lons[i2], selected_las[i2]
                    if flag:
                        ax.plot([lo1, lo2], [la1, la2], '--', color='k', transform=ccrs.Geodetic(), linewidth=2, zorder=0)
                        ax.plot([lo1, lo2], [la1, la2], marker, markerfacecolor=color, transform=prj3, markeredgewidth=0, markeredgecolor='k', markersize=10)
                        if sz == 8:
                            flag = False
            ax.axis('off')
            plt.savefig(fnm, bbox_inches = 'tight', pad_inches = 0, dpi=300, transparent=True)
            plt.close()
        #
        p.subplot(0, 0)
        plot_globe3d(p, globe, style='1.png', alpha=1, land='#999999', ocean='#ffffff'  ) #('plane', (normal, origin, invert) )
        p.camera_position = [(7164.4456150185315, -32959.49111285338, -3463.2338387224036), (0.0, 0.0, 0.0), (0.011160759659876018, -0.10209381453235096, 0.9947121646376144)]
        p.subplot(0, 1)
        plot_globe3d(p, globe, style='2.png', alpha=1, land='#999999', ocean='#ffffff'  ) #('plane', (normal, origin, invert) )
        p.camera_position = [(7164.4456150185315, -32959.49111285338, -3463.2338387224036), (0.0, 0.0, 0.0), (0.011160759659876018, -0.10209381453235096, 0.9947121646376144)]
        #
        p.show(screenshot='correlation2.png')
        print(p.camera_position)
        pass
    @staticmethod
    def benchmark4():
        """
        Plot global inter-receiver and inter-source geometries.
        """
        p = pv.Plotter(notebook=0, shape=(1, 2), border=False, window_size=(3500, 2000) )
        p.set_background('white')
        #
        center = (0, 0, 0)
        R0 = 6371.0
        globe = globe3d(R0, center)
        # use obspy to download station longitude and latitudes of a network 'II'
        from obspy.clients.fdsn import Client
        client = Client("IRIS")
        inventory = client.get_stations(network='II,IU,G', level='station')
        lons, las = [], []
        for network in inventory:
            for station in network:
                lons.append(station.longitude)
                las.append(station.latitude)
        lons = np.array(lons)
        las = np.array(las)
        cat = client.get_events(starttime="2010-01-01", endtime="2015-01-02", minmagnitude=6.0)
        evlons, evlas, evdeps = [], [], []
        for event in cat:
            evlons.append(event.origins[0].longitude)
            evlas.append(event.origins[0].latitude)
            evdeps.append(event.origins[0].depth)
        #
        # plot 2d globe map
        n_points = 100
        ref_lo, ref_la = -76, 9
        # select lons and lats so that -130<lo<-30 and -30<la<30
        selected_idxs = (lons>-75)*(lons<-30)*(las>-20)*(las<15)
        selected_lons = lons[selected_idxs]
        selected_las  = las[selected_idxs]
        #n_gc = 30
        #i1i2 = np.random.randint(0, n_points, (n_gc, 2))
        #print(i1i2.shape)
        #class LowerThresholdGeodetic(ccrs.Geodetic):
        #    @property
        #    def threshold(self):
        #        return 1e1
        #class LowerThresholdPlateCarree(ccrs.Geodetic):
        #    @property
        #    def threshold(self):
        #        return 1e1
        prj1 = ccrs.PlateCarree(central_longitude=0)
        prj3 = ccrs.Geodetic()
        prj1.threshold = 0.001
        # plot 2d globe map inter-receiver and inter-source
        for marker, ref_marker, color, ref_color, fnm, sz in zip('**', '^^', ('#fc7200', '#fc7200'), ('#0d90e0', '#0d90e0'), ('2.png', '1.png'), (12, 8)):
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(1, 1, 1, projection=prj1)
            ax.set_global()
            ax.add_feature(cfeature.LAND, color='#cccccc', alpha=1.0)
            ax.add_feature(cfeature.OCEAN, color='#ffffff', alpha=1.0)
            if sz == 12:
                ax.plot(lons, las, marker, markerfacecolor=color, transform=prj3, markeredgewidth=0, markeredgecolor='k', markersize=15)
                ax.plot(evlons, evlas, ref_marker, markerfacecolor=ref_color, transform=prj3, markeredgewidth=0, markeredgecolor='k', markersize=10)
            ax.plot(ref_lo, ref_la, ref_marker, markerfacecolor=ref_color, transform=prj3, markeredgewidth=0, markeredgecolor='k', markersize=10)
            # plot great circle paths
            # generate a random list, each list element is a tuple of two integers, representing the indices of the two points
            flag = True
            for i1 in range(selected_lons.size):
                lo1, la1 = selected_lons[i1], selected_las[i1]
                for i2 in range(i1+1, selected_lons.size)[::2]:
                    lo2, la2 = selected_lons[i2], selected_las[i2]
                    if flag:
                        ax.plot([lo1, lo2], [la1, la2], '--', color='k', transform=ccrs.Geodetic(), linewidth=2, zorder=0)
                        ax.plot([lo1, lo2], [la1, la2], marker, markerfacecolor=color, transform=prj3, markeredgewidth=0, markeredgecolor='k', markersize=15)
                        if sz == 8:
                            flag = False
            ax.axis('off')
            plt.savefig(fnm, bbox_inches = 'tight', pad_inches = 0, dpi=300, transparent=True)
            plt.close()
        #
        p.subplot(0, 0)
        plot_globe3d(p, globe, style='1.png', alpha=1, land='#999999', ocean='#ffffff'  ) #('plane', (normal, origin, invert) )
        p.camera_position = [(7164.4456150185315, -32959.49111285338, -3463.2338387224036), (0.0, 0.0, 0.0), (0.011160759659876018, -0.10209381453235096, 0.9947121646376144)]
        p.subplot(0, 1)
        plot_globe3d(p, globe, style='2.png', alpha=1, land='#999999', ocean='#ffffff'  ) #('plane', (normal, origin, invert) )
        p.camera_position = [(7164.4456150185315, -32959.49111285338, -3463.2338387224036), (0.0, 0.0, 0.0), (0.011160759659876018, -0.10209381453235096, 0.9947121646376144)]
        #
        p.show(screenshot='correlation4.png')
        print(p.camera_position)
        pass
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
    style: 'simple', 'fancy1', 'fancy2', 'Mars', 'Cat1', 'Mosaic', 'Mosaic_copper', or '.png' filename.
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
    elif '.png' in style:
        tex = pv.read_texture(style)
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
        p.add_mesh(sphere, texture=tex, show_edges=False, opacity=alpha, smooth_shading=True, lighting=True, culling=culling)
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
class beachball3d:
    """
    Class to plot 3d beachball given focal mechanism.

    Examples are in `beachball3d.benchmark(...)`.
    Note, here theta is like the longitude, and phi is like the colatitude on a sphere.
    """
    ZERO_TOL = 1e-3
    ##########################################################################################
    def __init__(self, gcmt=None, matENU=None, normalize=False, strike_dip_slip=None):
        """
        Initialize the beachball3d class with optional gcmt(6-element array), or matENU(3x3 matrix).

        :param gcmt:  (M11, M22, M33, M12, M13, M23) in USE coordinate.
        :param matENU: a 3x3 moment tensor in ENU coordinate.
        :param strike_dip_slip: (strike, dip, slip) in degrees.
        :param normalize: whether to normalize the moment tensor so that its Frobenius norm is 1. (default is False)

        Note:  Coordinate system Up-South-East equals r-theta-phi (the Harvard/Global CMT convention ).
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
        if gcmt is not None:
            self.matENU = beachball3d.gcmt2matENU(gcmt)
        elif matENU is not None:
            self.matENU = matENU
        elif strike_dip_slip is not None:
            strike, dip, slip = strike_dip_slip
            self.matENU = beachball3d.strike_dip_slip2matENU(strike, dip, slip)
        if normalize:
            v = beachball3d.frobenius_norm(self.matENU)
            if v > 0.0:
                self.matENU /= v
    @property
    def gcmt(self):   # the gcmt array (M11, M22, M33, M12, M13, M23) in USE coordinate
        """
        Return the gcmt array in USE coordinate (also Harvard GCMT convention).
        """
        return beachball3d.matENU2gcmt(self.matENU)
    @property
    def matUSE(self):
        m11, m22, m33, m12, m13, m23 = self.gcmt
        matUSE = np.array((m11, m12, m13, m12, m22, m23, m13, m23, m33) ).reshape(3,3)
        return matUSE
    @property
    def matXYZ(self, ijk='USE'): # the 3x3 matrix for the moment tensor in ijk coordinate
        """
        Return the 3x3 matrix moment tensor in ijk (e.g., 'USE', 'NED',...) coordinate.
        """
        return beachball3d.xyz2uvw(self.matENU, xyz='ENU', uvw=ijk)
    @property
    def norm(self): # IS this Correct?
        return beachball3d.frobenius_norm(self.matENU)
    ##########################################################################################
    def is_pure_iso(self):
        a  = beachball3d.angle_dist2direction(self.matENU, (1., 1. ,1.) )
        a %= np.pi
        if np.abs(a) < beachball3d.ZERO_TOL:
            return True
        return False
    def is_pure_clvd(self):
        a  = beachball3d.angle_dist2direction(self.matENU, (1., 1., -2) )
        a %= np.pi
        b  = beachball3d.angle_dist2direction(self.matENU, (2., -1., -1) )
        b %= np.pi
        if (np.abs(a) < beachball3d.ZERO_TOL) or (np.abs(b) < beachball3d.ZERO_TOL):
            return True
        return False
    def is_pure_dc(self):
        a  = beachball3d.angle_dist2direction(self.matENU, (1. ,0, -1.) )
        a %= np.pi
        if np.abs(a) < beachball3d.ZERO_TOL:
            return True
        return False
    ##########################################################################################
    def get_iso(self):
        l = np.trace(self.matENU)/3.0
        return beachball3d(matENU= np.diag((l, l, l) ) )
    def get_dev(self):
        l = np.trace(self.matENU)/3.0
        dev_mat = self.matENU - np.diag((l, l, l) )
        return beachball3d(matENU= dev_mat )
    def get_dc(self):   # Note: force to remove a DC even if is already a pure CLVD
        l = np.trace(self.matENU)/3.0
        dev_mat = self.matENU - np.diag((l, l, l) )
        dc_mat = beachball3d.proj_diag(dev_mat, (1., 0., -1.) )
        return beachball3d(matENU= dc_mat )
    def get_clvd(self): # Note: force to remove a DC before distilling a CLVD
        l = np.trace(self.matENU)/3.0
        dev_mat = self.matENU - np.diag((l, l, l) )
        dc_mat = beachball3d.proj_diag(dev_mat, (1., 0., -1.) )
        return beachball3d(matENU= dev_mat - dc_mat )
    def get_iso_dc_clvd(self): # force to distill ISO, DC, and CLVD components, in sequence, from a moment tensor (even if it is already a pure clvd one).
        """
        Return beachball3d objects for ISO, DC, and CLVD components.
        """
        m = self.matENU
        m_iso = beachball3d.get_iso(m)
        m_dev = m - m_iso
        m_dc  = beachball3d.get_dc(m_dev)
        m_clvd= m_dev - m_dc
        return beachball3d(matENU= m_iso), beachball3d(matENU= m_dc), beachball3d(matENU= m_clvd)
    ##########################################################################################
    def getTBPND(self, check_pure_iso_clvd=True): # get unit vectors in T, B, P, N, D directions
        """
        Compute the (n, d, t, p, b unit vectors) for the DC part of `self.matENU`.

        :param check_pure_iso_clvd: `True` (default) or `False` to check (and raise error) if the input moment tensor is purely isotropic or clvd.
                                    Note: set `False` will force to compute strike, dip, and slip even if the input is pure ISO or CLVD.

        :return: (t, b, p), (n1, d1), (n2, d2): the (t, b, p) unit vectors and (n1, d1), (n2, d2) pairs.
        """
        tensor_mat_ENU = self.matENU
        if check_pure_iso_clvd:
            if self.is_pure_iso():
                raise ValueError('The input moment tensor purely isotropic, hence cannot get T,B,P,N,D as for DC!', tensor_mat_ENU)
            elif self.is_pure_clvd():
                print('Warning: the input moment tensor purely CLVD, hence cannot get T,B,P,N,D as for DC!', tensor_mat_ENU)
        ################################################################################################################
        mat_dc_ENU = self.get_dc().matENU
        _, U = beachball3d.sorted_right_handed_eig(mat_dc_ENU)
        t, b, p = U.T
        #
        n1 = (t+p)*(np.sqrt(2)*0.5)
        d1 = (t-p)*(np.sqrt(2)*0.5)
        n2 = d1
        d2 = n1
        return (t, b, p), (n1, d1), (n2, d2)
    def getSDS(self, check_pure_iso_clvd=True): # get (Strike, Dip, Slip)
        """
        Compute the strike, dip, and slip angles for the DC part of `self.matENU`.

        :param tensor_mat_ENU: a 3by3 matrix for moment tensor in ENU coordinates.
        :param check_pure_iso_clvd: `True` (default) or `False` to check (and raise error) if the input moment tensor is purely isotropic or clvd.
                                    Note: set `False` will force to compute strike, dip, and slip even if the input is pure ISO or CLVD.

        :return: (strike1, dip1, slip1), (strike2, dip2, slip2): the strike, dip, and slip angles (in degree) for the fault and auxiliary planes.
        """
        tensor_mat_ENU = self.matENU
        _, (n1, d1), (n2, d2) = self.getTBPND(check_pure_iso_clvd)
        ################################################################################################################
        # Now, compute the strike, dip and slip angles using the obtained vec_n and vec_d
        pair_strike_dip_slip = list()
        for vec_n, vec_d in [ (n1, d1), (n2, d2) ]:
            nx, ny, nz = vec_n
            phi        = np.arctan2(-ny, nx) # Assume sin(delta)>0, and always works for n1,d1 which has dip!=0
            # get delta
            XE, YN, ZU = np.array((1.,0.,0.)), np.array((0.,1.,0.)), np.array((0.,0.,1.))
            sin_phi, cos_phi  = np.sin(phi), np.cos(phi)
            x3 =  ZU
            x1 =  sin_phi * XE + cos_phi * YN
            x2 = -cos_phi * XE + sin_phi * YN
            delta = np.arctan2(np.dot(vec_n, -x2), np.dot(vec_n, x3) ) #based on: vec_n = cos_delta*x3 - sin_delta*x2! sin(delta)>0 is assumed above!
            # get lamda
            sin_delta, cos_delta = np.sin(delta), np.cos(delta)
            sin_lamda = np.dot(vec_d, cos_delta*x2 + sin_delta*x3) #based on: vec_d = cos_lamda*x1 + sin_lamda*(cos_delta*x2 + sin_delta*x3)
            cos_lamda = np.dot(vec_d, x1)
            lamda     = np.arctan2(sin_lamda, cos_lamda)
            #
            strike_deg = np.rad2deg(phi)   % 360
            dip_deg    = np.rad2deg(delta) % 180 # sin(delta)>0 is assumed above! so that 0<delta<90 degree
            slip_deg   = np.rad2deg(lamda) % 360
            if dip_deg > 90: # in fact this could be useless... but just in case...
                strike_deg = (180+strike_deg) % 360
                dip_deg    = 180 - dip_deg
                slip_deg   = (360-slip_deg)%360
            pair_strike_dip_slip.append( (strike_deg, dip_deg, slip_deg ) )
        pair_strike_dip_slip = np.array(pair_strike_dip_slip, dtype=np.float64)
        ################################################################################################################
        # Fix the problem for dip=0
            # The n1,d1 and n2,d2 correspond to two nodal planes (one fault plane and the other auxiliary plane), and
            # n1 and n2 are the normal vector of the planes.
            #
            # However, if one plane conincide with the free surface, which means dip=0, then for that plane, the norm
            # n is vertical and has nx,ny=0. Notably, for the plane with dip=0, the strike and slip angles cannot be
            # determined, as there are multiple combination of strike and slip angles that form the same slip vector
            # in the plane, as long as strike-slip is a constant (the direction angle of the slip vector)! Think about that!
            #
            # Still, the work plane works! The other plane, being perpendicular to the plane with dip=0, will be perpendicular
            # to the free surface. Obviously, that plane will intersect with the free surface at a line, and the line
            # is delineate the strike!
            #
            # So, let us use that strike for both planes if such case happen!
        for this_idx in (0, 1):
            other_idx = 1-this_idx
            this_strike, this_dip, this_slip = pair_strike_dip_slip[this_idx]
            other_strike, junk, junk = pair_strike_dip_slip[other_idx]
            if (this_dip<1e-9): # dip is zero!
                new_strike = other_strike
                new_slip = (new_strike + (this_slip-this_strike)) % 360
                pair_strike_dip_slip[this_idx] = (new_strike, 0.0, new_slip)
        return pair_strike_dip_slip
    ##########################################################################################
    @staticmethod
    @jit(nopython=True, nogil=True)
    def __fast_radiation_all(matENU, theta, phi, binarization=False):
        sin_phi, cos_phi     = np.sin(phi), np.cos(phi)
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        #
        P_pol      = np.zeros((theta.size, 3 ) )
        P_pol[:,0] = sin_phi * cos_theta
        P_pol[:,1] = sin_phi * sin_theta
        P_pol[:,2] = cos_phi
        #
        SV_pol = np.zeros((theta.size, 3 ) )
        SV_pol[:,0] = cos_phi * cos_theta
        SV_pol[:,1] = cos_phi * sin_theta
        SV_pol[:,2] = -sin_phi # Note, here SV_pol is always pointing downwards!
        #
        SH_pol = np.zeros((theta.size, 3 ) )
        SH_pol[:,0] = sin_theta # Note, here SH_pol is always pointing Eastwards (or clockwise from top view)
        SH_pol[:,1] = -cos_theta
        SH_pol[:,2] = 0.0
        ######
        P_amp = np.zeros(theta.size)
        SV_amp= np.zeros(theta.size)
        SH_amp= np.zeros(theta.size)
        for idx in range(theta.size):
            tmp = P_pol[idx] @ matENU
            P_amp[idx]  = tmp @ P_pol[idx].T
            SV_amp[idx] = tmp @ SV_pol[idx].T
            SH_amp[idx] = tmp @ SH_pol[idx].T
        if binarization:
            P_amp = np.sign(P_amp)
            SV_amp= np.sign(SV_amp)
            SH_amp= np.sign(SH_amp)
        return P_pol, P_pol, SV_pol, SH_pol, P_amp, SV_amp, SH_amp
    def radiation(self, thetas, phis, binarization=False):
        """
        Compute the P-, SV- and SH-wave radiations given a list of points on a unit sphere.

        thetas, phis: a list of theta and phi for the points on
                      a unit sphere. theta is angle (0->360) between
                      the direction and x+-z plane, and phi the angle
                      between the direction and z axis. Angles should
                      be in radian other than in degree.
        binarization: Modify the P-wave radiation amplitude to -1, 0, 1 for
                      negative, zero, and positive amplitudes, respectively.
                      Will not modify SV- and SH-wave amplitudes.

        Return: loc_xyz, P_pol, SV_pol, SH_pol, P_amp, SV_amp, SH_amp
                loc_xyz: a matrix each row of which specify a point's (x,y,z) on a unit sphere.
                P_pol:   a matrix each row of which specify the unit vector for the polarity direction of the P wave at the point (x,y,z).
                SV_pol:  a matrix each row of which specify the unit vector for the polarity direction of the SV wave at the point (x,y,z).
                SH_pol:  a matrix each row of which specify the unit vector for the polarity direction of the SH wave at the point (x,y,z).
                P_amp:   an array each element of which specify the radiation amplitude of the P wave along the unit vector polarity direction at the point (x,y,z).
                SV_amp:  an array each element of which specify the radiation amplitude of the SV wave along the unit vector polarity direction at the point (x,y,z).
                SH_amp:  an array each element of which specify the radiation amplitude of the SH wave along the unit vector polarity direction at the point (x,y,z).

                !!! Note: the unit vector for a polarity should be combined with its amplitude! Usually, a negative amplitude means
                the displacement is in the opposite direction of the unit vector, and a positive amplitude means the displacement is in
                the direction of the unit vector!

                In details,
                For P wave,
                positive radiation amplitude means particles are going away from the source along the ray path. (Recorded P wave is positive
                in both radial and up directions).
                negative radiation amplitude means particles are going toward the source along the ray path. (recorded P wave is negative
                in both radial and up directions).
                #
                For SV wave,
                positive radiation amplitude means particles are going downards along the ray path. (Recorded SV wave is positive in radial
                direction but negative in up direction).
                negative radiation amplitude means particles are going upwards along the ray path. (Recorded SV wave is negative in radial
                direction but positive in up direction).
                #
                For SH wave,
                positive radiation amplitude means particles are going clockwise (view from top) (or T direction) if we center at source. (Records SH
                is positive in T direction).
                negative radiation amplitude means particles are going anti-clockwise (view from top) (for -T direction) if we center at source.
                (Recorded SH is negative in T direction).
                #
                Above, the up is up (Z), radial is from source to receiver, and T is T=RxZ (RTZ is left-handed, or RZT right-handed)
        """
        if True: # use fast radiation computation
            return beachball3d.__fast_radiation_all(self.matENU, thetas, phis, binarization=binarization)
            #loc_xyz, p_pol, P_amp= beachball3d.__fast_radiation(self.matENU, thetas, phis, wave_type='P', binarization=binarization)
            #loc_xyz, sv_pol, SV_amp= beachball3d.__fast_radiation(self.matENU, thetas, phis, wave_type='SV', binarization=binarization)
            #loc_xyz, sh_pol, SH_amp= beachball3d.__fast_radiation(self.matENU, thetas, phis, wave_type='SH', binarization=binarization)
            #return loc_xyz, p_pol, sv_pol, sh_pol, P_amp, SV_amp, SH_amp
        sin_phi, cos_phi     = np.sin(phis), np.cos(phis)
        sin_theta, cos_theta = np.sin(thetas), np.cos(thetas)
        ####################
        # R,T,Z is a left-handed coordinate system, and hence R,T,-Z is right-handed
        # Here use P-polarity the one close to R direction,
        #          SV-...     ...     close to -Z directin,
        #          SH-...     ...     same to T direction.
        #
        # uR = cos(t) ux + sin(t) uy
        # SV = cos(p) uR - sin(p) uz = cos(p)cos(t) ux + cos(p)sin(t) uy - sin(p) uz
        # SH = sin(t) ux - cos(t) uy
        #
        #unit P wave polarity vectors from the center of the source
        P_pol      = np.zeros((len(thetas), 3 ) )
        P_pol[:,0] = sin_phi * cos_theta
        P_pol[:,1] = sin_phi * sin_theta
        P_pol[:,2] = cos_phi
        loc_xyz = P_pol.copy() # necessary in case of latter edits
        #the SV vectors that are perpendicular to the radial vectors and in vertical planes
        SV_pol      = np.zeros((len(thetas), 3 ) )
        SV_pol[:,0] = cos_phi * cos_theta
        SV_pol[:,1] = cos_phi * sin_theta
        SV_pol[:,2] = -sin_phi # Note, here SV_pol is always pointing downwards!
        #the SH vectors that are perpendicular to the radial vectors and in vertical planes
        SH_pol      = np.zeros((len(thetas), 3 ) )
        SH_pol[:,0] = sin_theta # Note, here SH_pol is always pointing Eastwards (or clockwise from top view)
        SH_pol[:,1] = -cos_theta
        SH_pol[:,2] = 0.0
        ####################
        P_amp, SV_amp, SH_amp = np.zeros(len(thetas) ), np.zeros(len(thetas) ), np.zeros(len(thetas) )
        for wave_pol, amp in zip((P_pol, SV_pol, SH_pol), (P_amp, SV_amp, SH_amp)):
            for idx, pol in enumerate(wave_pol):
                amp[idx] = np.matmul(np.matmul(pol, self.matENU), P_pol[idx].T) # based on Eq.4.97 in Aki&Richard (2002) Quantitative Seismology (2nd edition)
                # The amp could be negative! Which means the polarity direction should be -pol!
        if binarization:
            P_amp = np.sign(P_amp)
            SV_amp = np.sign(SV_amp)
            SH_amp = np.sign(SH_amp)
        return loc_xyz, P_pol, SV_pol, SH_pol, P_amp, SV_amp, SH_amp
    @staticmethod
    @jit(nopython=True, nogil=True)
    def __fast_radiation(matENU, theta, phi, wave_type='P', binarization=False):
        sin_phi, cos_phi     = np.sin(phi), np.cos(phi)
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        P_pol      = np.zeros((theta.size, 3 ) )
        P_pol[:,0] = sin_phi * cos_theta
        P_pol[:,1] = sin_phi * sin_theta
        P_pol[:,2] = cos_phi
        if wave_type == 'P':
            pol = P_pol
        else:
            pol = np.zeros((theta.size, 3 ) )
            if wave_type == 'SV':
                pol[:,0] = cos_phi * cos_theta
                pol[:,1] = cos_phi * sin_theta
                pol[:,2] = -sin_phi # Note, here SV_pol is always pointing downwards!
            elif wave_type == 'SH':
                pol[:,0] = sin_theta # Note, here SH_pol is always pointing Eastwards (or clockwise from top view)
                pol[:,1] = -cos_theta
                pol[:,2] = 0.0
        ######
        amp = np.zeros(theta.size)
        for idx in range(theta.size):
            amp[idx] = pol[idx] @ matENU @ P_pol[idx].T
        if binarization:
            amp = np.sign(amp)
        return P_pol, pol, amp
    def radiation_fast(self, theta, phi, wave_type='P', binarization=False): # return loc_xyz, pol, amp
        """
        theta: a single value or an array like object for many values.
               In radian. theta is measured from east and anti-clockwise. East is 0 and North in pi/2.
        phi:   ...
               In radian. phi is measured from top. Vertical up is phi=0, and vertical down is phi=pi.
        wave_type: 'P', or 'SV', or 'SH'
        binarization: True or False(default)
        """
        if hasattr(theta, '__len__'):
            return beachball3d.__fast_radiation(self.matENU, theta, phi, wave_type, binarization)
        else:
            theta, phi = np.asarray(theta), np.asarray(phi)
            P_pol, pol, amp = beachball3d.__fast_radiation(self.matENU, theta, phi, wave_type, binarization)
            return P_pol[0], pol[0], amp[0]
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
        return p
    def plot3d_pol(self, p=None, wave_type='P', hemisphere=None,
                    neg_color='#0f0396', pos_color='#db620c',
                    center=(0,0,0), radius=1.0, scale=1.0, density_level=3,
                    opacity=1, lighting=True, **kwargs):
        """
        Plot 3d P-wave radiation vectors at the surface of the beachball.

        p:          an instance of pyvista.Plotter
        wave_type:  'P', 'SV', 'SH', 'S', or 'all' to specify the type of wave.
        hemisphere: `lower` or `upper` to plot the lower or upper hemisphere.
                    Default is `None` to plot the whole sphere.
        neg_color, pos_color: color for the negative and positive amplitudes, respectively.
        center:     the center of the beachball.
        radius:     the radius of the beachball.
        scale:      scale all the vectors. (Default is 1.0)
        density_level: the bigger, the more vector arrows.
        opacity:    transparency.
        culling:    pyvista parameters.
        lighting:   pyvista parameters.
        **kwargs:   other parammeters for `pyvista.plotter.add_mesh(...)`
        """
        if wave_type not in ('P', 'SV', 'SH', 'S', 'all'):
            raise ValueError(f"Unknown wave_type: {wave_type}. Use 'P', 'SV', 'SH', 'S', or 'all'.")
        mesh = stripy.spherical_meshes.triangulated_cube_mesh(refinement_levels=density_level)
        thetas = mesh.lons
        phis   = np.pi*0.5 - mesh.lats
        phi_min = 0.0   if (hemisphere!='lower') else 0.5*np.pi
        phi_max = np.pi if (hemisphere!='upper') else 0.5*np.pi
        #########
        if wave_type != 'P': # remove north and south poles for which SV and SH are meaningless
            idx = np.where((0.001<=phis) & ((np.pi-0.001)<=phi_max) )
            thetas = thetas[idx]
            phis   = phis[idx]
        #########
        idxs = np.where((phi_min<=phis) & (phis<=phi_max))
        thetas, phis = thetas[idxs], phis[idxs]
        idxs = np.where(phis<=phi_max)
        thetas, phis = thetas[idxs], phis[idxs]
        #########
        loc_xyz, P_pol, SV_pol, SH_pol, P_amp, SV_amp, SH_amp = self.radiation(thetas, phis)
        dict_pol = {'P': P_pol, 'SV': SV_pol, 'SH': SH_pol}
        dict_amp = {'P': P_amp, 'SV': SV_amp, 'SH': SH_amp}
        #########
        # add for S and all if necessary
        if wave_type=='S':
            S_pol = np.zeros(SV_pol.shape, SV_pol.dtype)
            S_amp = np.zeros(SV_amp.shape, SV_amp.dtype)
            for idx, (sv, sh, asv, ash) in enumerate(zip(SV_pol, SH_pol, SV_amp, SH_amp)):
                pol = sv*asv + sh*ash
                amp = np.sqrt( np.sum(pol*pol) )
                #pol /= amp # no need to normalize in pyvista
                S_pol[idx] = pol
                S_amp[idx] = amp
            dict_pol['S'] = S_pol
            dict_amp['S'] = S_amp
        elif wave_type=='all': # In fact, this is meanless, as P and S have different factors as in Eq.4.96 in in Aki&Richard (2002) (2nd Edition)
            pol3d = np.zeros(SV_pol.shape, SV_pol.dtype)
            amp3d = np.zeros(SV_amp.shape, SV_amp.dtype)
            for idx, (_p, sv, sh, ap, asv, ash) in enumerate(zip(P_pol, SV_pol, SH_pol, P_amp, SV_amp, SH_amp)):
                pol = _p*ap + sv*asv + sh*ash
                amp = np.sqrt( np.sum(pol*pol) )
                #pol /= amp # no need to normalize in pyvista
                pol3d[idx] = pol
                amp3d[idx] = amp
            dict_pol['all'] = pol3d
            dict_amp['all'] = amp3d
        ######### plot
        scale = np.abs(scale)*radius*0.5
        app = Scene3D(pv_plotter=p, number_of_peels=-1)
        #p.disable_depth_peeling() # important for efficient rendering
        loc_xyz *= radius #+= center
        loc_xyz += center
        for loc, pol, amp in zip(loc_xyz, dict_pol[wave_type], dict_amp[wave_type]*scale ):
            loc_is_end = True if (amp<0 and wave_type=='P')  else False
            color      = neg_color if amp<0 else pos_color
            if np.abs(amp) > 1e-6:
                # Note amp could be negative, so that direction will be reversed.
                #amp, pol = (-amp, -pol) if amp < 0 else (amp, pol) # pyvista will take care of this
                app.add_arrow(loc, direction=pol, scale=amp, loc_is_end=loc_is_end, color=color, lighting=lighting, opacity=opacity, tip_resolution=10, shaft_resolution=10  )
        if wave_type == 'S':
            for wave_type in ('SV', 'SH'):
                for loc, pol, amp in zip(loc_xyz, dict_pol[wave_type], dict_amp[wave_type]*scale ):
                    if np.abs(amp) > 1e-6:
                        # Note amp could be negative, so that direction will be reversed.
                        #amp, pol = (-amp, -pol) if amp < 0 else (amp, pol) # pyvista will take care of this
                        app.add_arrow(loc, direction=pol, scale=amp, loc_is_end=False, color='gray', lighting=lighting, opacity=opacity*0.5, tip_resolution=10, shaft_resolution=10  )
        return app.pv_plotter
    def plot3d(self, p=None, wave_type='P', binarization=False, hemisphere=None,
                plot_zero_contour=True, color_zero_contour='k', plot_zero_contour_width=1,
                plot_tbp=True, plot_dc_nodal_planes=True,
                cmap='RdBu_r', diverging_clim=True, label='M', center=(0,0,0), radius=1.0, resolution_step_deg=2,
                lighting=True, **kwargs):
        """
        Plot 3d beachball with varied P-wave radiation amplitudes at different directions.
        :param p:                   an instance of pyvista.Plotter
        :param wave_type:           'P', 'SV', 'SH', 'S', or 'all' to specify the type of wave.
        :param binarization:        Modify the P-wave radiation amplitude to -1, 0, 1 for
                                    negative, zero, and positive amplitudes, respectively.
                                    Default is `False`, and will plot absolute amplitudes.
        :param hemisphere:          `lower` or `upper` to plot the lower or upper hemisphere.
                                    Default is `None` to plot the whole sphere.
        :param plot_zero_contour:   `True` or `False` (default) to plot zero contour lines on the beachball surface.
        :param color_zero_contour:  color of the zero contour lines.
        :param plot_zero_contour_width: width of the zero contour lines.
        :param plot_tbp:            `True` or `False` (default) to plot the TBP directions on the beachball surface.
                                    T, B, P are the minimal compressive, null, maximal compressive directions, respectively.
        :param plot_dc_nodal_planes:`True` or `False` (default) to plot the DC nodal planes on the beachball surface.
        :param cmap:                colormap to plot the P-wave radiation amplitudes.
        :param center:              the center of the beachball.
        :param radius:              the radius of the beachball.
        :param lighting:            pyvista parameters.
        :param **kwargs:            other parameters used by `pyvista.Plotter.add_mesh(...)`.
        """
        if self.norm <= 0:
            return
        if wave_type not in ('P', 'SV', 'SH', 'S', 'all'):
            raise ValueError(f"Unknown wave_type: {wave_type}. Use 'P', 'SV', 'SH', 'S', or 'all'.")
        ################################################################################################################################################################################
        #### get theta-phi-radiation data
        theta = np.arange(0, 360.001, resolution_step_deg)
        phi   = np.arange(0, 180.001, resolution_step_deg)
        if hemisphere=='upper':
            phi = np.arange(0, 180.001, resolution_step_deg)
        elif hemisphere=='lower':
            phi = np.arange(90, 180.001, resolution_step_deg)
        theta_mesh, phi_mesh = np.meshgrid(theta, phi)
        loc_xyz, P_pol, SV_pol, SH_pol, P_amp, SV_amp, SH_amp = self.radiation(np.deg2rad(theta_mesh.ravel() ), np.deg2rad(phi_mesh.ravel() ), binarization=binarization)
        P_amp  = P_amp.reshape(theta_mesh.shape)
        SV_amp = SV_amp.reshape(theta_mesh.shape)
        SH_amp = SH_amp.reshape(theta_mesh.shape)
        #
        #dict_radiation = {'P': P_amp, 'SV': np.abs(SV_amp), 'SH': np.abs(SH_amp), }
        # We keep negative and positive amplitude for SH, SV. Positive SV amplitude means it is positive amplitude in radial direction at receiver.
        # Negative SV amplitude means it is negative amplitude in radial direction  at receiver. Positive SH amplitude means it is positive amplitude in
        # tangential direction at receiver. Negative SH amplitude means it is negative amplitude in tangential direction at receiver.
        dict_radiation = {'P': P_amp, 'SV': SV_amp, 'SH': SH_amp, }
        if wave_type=='S':
            dict_radiation['S'] = np.sqrt(SV_amp*SV_amp + SH_amp*SH_amp)
        elif wave_type=='all':
            dict_radiation['all'] = np.sqrt(P_amp*P_amp + SV_amp*SV_amp + SH_amp*SH_amp)
        ################################################################################################################################################################################
        if diverging_clim:
            vmax, vmin = np.max(dict_radiation[wave_type]), np.min(dict_radiation[wave_type])
            cmax = max(vmax, -vmin)
            clim = kwargs.pop('clim', (-cmax, cmax) )
        else:
            clim = kwargs.pop('clim', None)
        label = '%s (%s)' % (wave_type, label)
        ################################################################################################################################################################################
        app = Scene3D(pv_plotter=p)
        if hemisphere =='upper':
            app.enable_clip(clip_kw={'normal': '-z', 'origin': (0, 0, 0) }, clip_box_kw=None )
        elif hemisphere =='lower':
            app.enable_clip(clip_kw={'normal': 'z', 'origin': (0, 0, 0)}, clip_box_kw=None )
        ################################################################################################################################################################################
        app.add_sphere_grd((0, 360), (phi[0], phi[-1]), dict_radiation[wave_type], label=label, radius=radius, center=center, cmap=cmap, lighting=lighting, clim=clim, **kwargs)
        if plot_zero_contour:
            contour_label = '%s_contour0' % wave_type
            app.disable_add_mesh_to_plotter()
            mesh, actor = app.add_sphere_grd((0, 360), (phi[0], phi[-1]), dict_radiation[wave_type], label=contour_label, radius=radius, center=center, opacity=0.0, show_scalar_bar=False)
            app.enable_add_mesh_to_plotter()
            if mesh.point_data[contour_label].min() < 0.0 < mesh.point_data[contour_label].max():
                contours = mesh.contour([0.0])
                app.add_mesh(contours, show_edges=True, opacity=1.0, color=color_zero_contour, line_width=plot_zero_contour_width)
        ################################################################################################################################################################################
        if (not self.is_pure_iso()) and (not self.is_pure_clvd() ):
            cmap = plt.get_cmap(cmap)
            color_p = cmap(0.0)
            color_t = cmap(1.0)
            self.plot3d_tbp_nodal_planes( app.pv_plotter, hemisphere=hemisphere, center=center, radius=radius,
                                            plot_t=plot_tbp, plot_b=plot_tbp, plot_p=plot_tbp, color_t=color_t, color_p=color_p,
                                            plot_dc_nodal_planes=plot_dc_nodal_planes,
                                            lighting=lighting, **kwargs)
        ################################################################################################################################################################################
        app.disable_clip()
        return app.pv_plotter
    def plot3d_tbp_nodal_planes(self, p=None, hemisphere=None, plot_t=True, plot_b=True, plot_p=True, plot_dc_nodal_planes=True,
                                 center=(0,0,0), radius=1.0, color_t='red', color_p='black', color_b='gray', scale_tbp=0.5, color_nodal_planes='k',
                                 lighting=False, **kwargs):
        """
        """
        if self.norm <= 0:
            return
        (vt, vb, vp), (n1, d1), (n2, d2) = self.getTBPND()
        ################################################################################################################################################################################
        app = Scene3D(pv_plotter=p)
        if hemisphere =='upper':
            app.enable_clip(clip_kw={'normal': '-z', 'origin': (0, 0, 0) }, clip_box_kw=None )
        elif hemisphere =='lower':
            app.enable_clip(clip_kw={'normal': 'z', 'origin': (0, 0, 0)}, clip_box_kw=None )
        ################################################################################################################################################################################
        #### plot t, b, p vectors
        lst_v = [(plot_t, vt, color_t,  1, False),
                 (plot_b, vb, color_b,  1, False),
                 (plot_p, vp, color_p, -1, True)]
        for plot_flag, vec, clr, sign, loc_is_end in lst_v:
            if not plot_flag:
                continue
            for v in (vec, -vec):
                loc = v*radius+center
                if hemisphere == 'upper' and loc[2] <=center[2]:
                    continue
                if hemisphere == 'lower' and loc[2] >=center[2]:
                    continue
                app.add_arrow(loc=loc, direction=sign*v, loc_is_end=loc_is_end, color=clr, lighting=lighting, scale=radius*scale_tbp, **kwargs)
        ################################################################################################################################################################################
        #### plot two nodal planes and the third plane
        if plot_dc_nodal_planes:
            n3 = np.cross(n1, n2)
            d3 = n1
            for n, d in ((n1, d1), (n2, d2), (n3, d3) ):
                flag = 'opacity' in kwargs
                opacity = kwargs.pop('opacity', 1.0)
                app.add_plane(center=center, direction=n, x_direction=d, color=color_nodal_planes, opacity=opacity*0.2,
                              i_size=radius*2.01, j_size=radius*2.01, i_resolution=10, j_resolution=10, **kwargs)
                if flag: # restore kwargs
                    kwargs['opacity'] = opacity
        ################################################################################################################################################################################
        app.disable_clip()
        return app.pv_plotter
    def plot2d(self, ax_polar=None, wave_type='P', binarization=False, hemisphere='lower', proj_method='Schmidt',
                plot_zero_contour=True, color_zero_contour='k', plot_zero_contour_width=0.5,
                plot_tbp=True, color_tbp='y', plot_dc_nodal_planes=True, color_nodal_planes='gray',
                cmap='RdBu_r', diverging_clim=True, radius=1.0, resolution_step_deg=1, markersize=5,
                figname=None, show=False, cax=None, **kwargs ):
        """
        """
        if self.norm <= 0 or hemisphere not in ('lower', 'upper'):
            return
        ####
        theta = np.linspace(0, 2*np.pi, 360)
        if hemisphere == 'lower':
            phi = np.deg2rad(np.arange(90, 180.001, resolution_step_deg))
        else:
            phi = np.deg2rad(np.arange(0, 90.001, resolution_step_deg))
        theta_mesh, phi_mesh = np.meshgrid(theta, phi)
        loc_xyz, P_pol, SV_pol, SH_pol, P_amp, SV_amp, SH_amp = self.radiation(theta_mesh.ravel(), phi_mesh.ravel(), binarization=binarization)
        #dict_radiation = {'P': P_amp, 'SV': np.abs(SV_amp), 'SH': np.abs(SH_amp) }
        dict_radiation = {'P': P_amp, 'SV': SV_amp, 'SH': SH_amp }
        if wave_type == 'S':
            dict_radiation['S'] =  np.sqrt(SV_amp**2 + SH_amp**2)
        elif wave_type == 'all':
            dict_radiation['all'] = np.sqrt(P_amp**2 + SV_amp**2 + SH_amp**2)
        ####
        if proj_method== 'Schmidt':
            proj_r = beachball3d.schmidt_phi2r(phi, radius, hemisphere=='lower' )
            rr, tt = np.meshgrid(proj_r, theta)
            values = dict_radiation[wave_type].reshape(theta_mesh.shape).T
            if ax_polar is None:
                fig, ax_polar = plt.subplots(subplot_kw={'projection': 'polar'})
            if diverging_clim:
                vmax, vmin = np.max(dict_radiation[wave_type]), np.min(dict_radiation[wave_type])
                #print('%s %.2f %.2f'% (wave_type, vmin, vmax))
                cmax = max(vmax, -vmin)
                clim = kwargs.pop('clim', (-cmax, cmax) )
            else:
                clim = kwargs.pop('clim', (None, None))
            ########## plot radiations
            im = ax_polar.pcolormesh(tt, rr, values, shading='gouraud', cmap=cmap, vmin=clim[0], vmax=clim[1] )
            if cax is not None:
                plt.colorbar(im, cax=cax, orientation='horizontal')
            if plot_zero_contour:
                if values.min() < 0.0 < values.max():
                    contours = ax_polar.contour(tt, rr, values, levels=[0.0], colors=color_zero_contour, linewidths=plot_zero_contour_width)
            ########## plot tbp
            if plot_tbp and (not self.is_pure_iso() ) and (not self.is_pure_clvd() ):
                color_p = color_tbp
                color_t = color_tbp
                color_b = color_tbp
                plot_t = True
                plot_p = True
                plot_b = True
                self.plot2d_tbp_nodal_planes(ax_polar=ax_polar, hemisphere=hemisphere,
                                              plot_t=plot_t, plot_b=plot_b, plot_p=plot_p,  color_t=color_t, color_p=color_p, color_b=color_b, markersize=5,
                                              plot_dc_nodal_planes=plot_dc_nodal_planes, color_nodal_planes=color_nodal_planes,
                                              radius=radius)
            ########## adjust
            ax_polar.set_ylim((0, radius) )
            ax_polar.grid(False)
            ax_polar.set_xticks([])
            ax_polar.set_yticks([])
            if figname is not None:
                plt.savefig(figname, bbox_inches='tight', dpi=300)
            if show:
                plt.show()
            return ax_polar
    def plot2d_tbp_nodal_planes(self, ax_polar=None, hemisphere='lower', plot_t=True, plot_b=True, plot_p=True, plot_dc_nodal_planes=True,
                                 radius=1.0, color_t='c', color_p='c', color_b='c', markersize=5,
                                 color_nodal_planes='gray', linestyle='-', linewidth=0.3,
                                 proj_method= 'Schmidt'):
        """
        """
        if self.norm <= 0 or hemisphere not in ('lower', 'upper'):
            return
        (vt, vb, vp), (n1, d1), (n2, d2) = self.getTBPND()
        ####
        if hemisphere == 'lower':
            marker_t='x' # going into the paper
            marker_p='o' # going out of the paper
            marker_b='s'
        else:
            marker_t='o'
            marker_p='X'
            marker_b='s'
        ####
        def xyz2tr(x, y, z, scale, hemisphere):
            x, y, z = np.array(x).ravel(), np.array(y).ravel(), np.array(z).ravel()
            theta = np.arctan2(y, x)
            phi   = np.arctan2(np.sqrt(x*x+y*y), z )
            if hemisphere == 'lower':
                idxs = np.where(z < 0)
                theta = theta[idxs]
                phi = phi[idxs]
                r = beachball3d.schmidt_phi2r(phi, scale, True)
            elif hemisphere == 'upper':
                idxs = np.where(z > 0)[0]
                theta = theta[idxs]
                phi = phi[idxs]
                r = beachball3d.schmidt_phi2r(phi, scale, False)
            return theta, r
        if proj_method== 'Schmidt':
            #### plot nodal planes
            if (not self.is_pure_iso()) and (not self.is_pure_clvd()) and plot_dc_nodal_planes:
                for n in (n1, d1):
                    n /= np.linalg.norm(n)
                    n = -n if n[2]<0 else n # make sure n is upward and unit
                    if np.abs(n[2]-1)<1e-9: # n is vertical, no need to plot
                        continue
                    ########
                    # We rotate from uvw to uqn about the axis u
                    # u = w x n
                    # Then, we need to find a half circle in x-y plane, so that rotating it about the u direction result in
                    # the nodal plane (the rotation that bring w to n).
                    # Apparently, the half circle in x-y plane starts from -u direction to u direction anti-clockwisely.
                    # So we need to find the theta angle for -u, and then the half circle in x-y plane has theta range is
                    # (angle-pi, angle) for the lower half and (angle, angle+pi) for the upper half.
                    # 
                    w = (0,0,1)
                    u = np.cross(w,n)
                    vcos = np.dot((1,0,0), u)
                    vsin = np.dot(np.cross((1,0,0), u), (0,0,1) )
                    start_angle = np.arctan2(vsin, vcos)
                    angle = np.linspace(start_angle, start_angle+np.pi, 100) if hemisphere=='upper' else np.linspace(start_angle-np.pi, start_angle, 100)
                    xs = np.cos(angle)
                    ys = np.sin(angle)
                    zs = np.zeros(xs.size)
                    #
                    xs, ys, zs= Scene3D.rotate_and_translate2(np.array([xs,ys,zs]), current_origin_xyz=(0,0,0),
                                                              current_orientation_direction1=u, current_orientation_direction2=w,
                                                              new_orientation_direction1=u, new_orientation_direction2=n, new_origin_xyz=(0,0,0) )
                    #
                    prj_theta, prj_r = xyz2tr(xs, ys, zs, radius, hemisphere)
                    if ax_polar is None:
                        fig, ax_polar = plt.subplots(subplot_kw={'projection': 'polar'})
                    ax_polar.plot(prj_theta, prj_r, color=color_nodal_planes, linewidth=linewidth, linestyle=linestyle)
            #### plot t, b, p vectors
            lst_v = [(plot_t, vt, color_t, marker_t),
                     (plot_b, vb, color_b, marker_b),
                     (plot_p, vp, color_p, marker_p)]
            for plot_flag, vec, clr, marker in lst_v:
                if not plot_flag:
                    continue
                for v in (vec, -vec):
                    x, y, z = v
                    prj_theta, prj_r = xyz2tr(x, y, z, radius, hemisphere)
                    if (hemisphere == 'upper' and z >0.0) or (hemisphere == 'lower' and z<0.0):
                        ax_polar.plot(prj_theta, prj_r, color=clr, linewidth=0.0, markersize=markersize, marker=marker)
            ########## adjust
            ax_polar.set_ylim((0, radius) )
            ax_polar.grid(False)
            ax_polar.set_xticks([])
            ax_polar.set_yticks([])
            return ax_polar
    ##########################################################################################
    @staticmethod
    @jit(nopython=True, nogil=True)
    def schmidt_phi2r(phi, radius=1.0, lower_hemisphere=True):
        if lower_hemisphere:
            proj_r = np.sqrt(2)*np.cos(0.5*phi)*radius
        else:
            proj_r = np.sqrt(2)*np.cos(0.5*(np.pi-phi) )*radius
        return proj_r
    @staticmethod
    def TP2tensor(t, p, norm=1.0):
        """
        Return the DC moment tensor matrix in the coordinate where `t`, `p` is measured.
        :param t: the minimal compressive direction.
        :param p: the maximal compressive direction.
        """
        tensor = np.outer(t, t) - np.outer(p, p) # the moment tensor matrix in ENU coordinates
        tensor *= (norm/beachball3d.frobenius_norm(tensor))
        return tensor
    @staticmethod
    def ND2tensor(n, d, norm=1.0):
        """
        Return the DC moment tensor matrix in the coordinate where `n` and `d` is measured.
        :param n: the normal direction of the fault plane.
        :param d: the slip direction on the fault plane.
        """
        t = n+d
        p = n-d
        return beachball3d.TP2tensor(t, p, norm=norm)
    @staticmethod
    def strike_dip_slip2ND(strike_deg, dip_deg, slip_deg):
        """
        Compute the N, D directions in ENU coordinates for given strike, dip, and slip angles for a purely DC source.

        :param strike_deg: strike measured clockwise from North (in degree)
        :param dip_deg:    dip angle (in degree, must be within 0-90 degree)
        :param slip_deg:   slip angle (in degree)
        """
        if dip_deg < 0 or dip_deg > 90:
            raise ValueError(f"Dip angle must be within 0-90 degrees, but got {dip_deg} degree.")
        phi, delta, lamda    = np.deg2rad(strike_deg), np.deg2rad(dip_deg), np.deg2rad(slip_deg)
        sin_phi, cos_phi     = np.sin(phi),   np.cos(phi)
        sin_delta, cos_delta = np.sin(delta), np.cos(delta)
        sin_lamda, cos_lamda = np.sin(lamda), np.cos(lamda)
        XE, YN, ZU = np.array((1.,0.,0.)), np.array((0.,1.,0.)), np.array((0.,0.,1.)) # the coordinate in all internal computation
        # Calculate the x1, x2, x3 show in Fig 4.2-2 in the ase XE,YN,ZU
        x3 =  ZU
        x1 =  sin_phi * XE + cos_phi * YN
        x2 = -cos_phi * XE + sin_phi * YN
        #
        vec_n = cos_delta*x3 - sin_delta*x2 # the normal vector, n,  of the fault plane
        vec_d = cos_lamda*x1 + sin_lamda*(cos_delta*x2 + sin_delta*x3) #the slip vector
        return vec_n, vec_d
    @staticmethod
    def strike_dip_slip2matENU(strike_deg, dip_deg, slip_deg):
        """
        Compute the moment tensor matrix in ENU coordinates for given strike, dip, and slip angles for a purely DC source.

        :param strike_deg: strike measured clockwise from North (in degree)
        :param dip_deg:    dip angle (in degree, must be within 0-90 degree)
        :param slip_deg:   slip angle (in degree)
        """
        n, d = beachball3d.strike_dip_slip2ND(strike_deg, dip_deg, slip_deg)
        return beachball3d.ND2tensor(n, d)
    ##########################################################################################
    # some math in lambda1-lambda2-lambda3 space
    @staticmethod
    def frobenius_norm(tensor_mat): # return the sqrt of squared summation of a matrix
        return np.linalg.norm(tensor_mat )
    @staticmethod
    def sorted_right_handed_eig(tensor_mat): # return sorted l=(l1,l2,l3) and U=(u1,u2,u3) l1>=l2>=l3, and u1xu2=u3
        """
        Return l, and U for M = U L, where L = diagonal(l) and l =(l1,l2,l3) with l1>=l2>=l3
        and U=[u1,u2,u3] with u1xu2=u3
        :param tensor_mat:
        :return l, U:
        """
        l, U = eig(tensor_mat)
        U = U.real
        l = l.real
        #
        idx = np.argsort(l)[::-1]
        l = l[idx]
        U = U[:, idx]
        u1, u2, u3 = U.T
        if np.dot(np.cross(u1,u2), u3) < 0:
            U[:,2] = -U[:,2]
        return l, U
    @staticmethod
    def proj_diag(tensor_mat, direction=(1,0,0) ): # proj the diagonal of a matrix to a target direction.
        """
        :param tensor_mat:
        :param direct: a single or a list of (vx, vy, vz) as directions.
        :return result: a single matrix or a list of matrix depending on the input `direction`
        """
        l, U = beachball3d.sorted_right_handed_eig(tensor_mat=tensor_mat) # M = U L or M = U L U.T
        vec = np.asarray(direction).reshape((-1, 3) )
        vec = [np.asarray(it) / np.linalg.norm(it) for it in vec] # normalize
        lnew= [np.dot(l, it)*it for it in vec]     # project eigvalues
        result = [U @ np.diag(it) @ U.T for it in lnew] # from eigvalues to matrix
        return result if len(result)>1 else result[0]
    @staticmethod
    def angle_dist2direction(tensor_mat, direction=(1,0,0)):
        """
        Return the angle distance in [0, pi] between the diagonal of a matrix and a given direction.
        :param tensor_mat:
        :param direction: a direction (vx, vy, vz).
        :return angle:
        """
        l, U = beachball3d.sorted_right_handed_eig(tensor_mat=tensor_mat) # M = U L or M = U L U.T
        l /= np.linalg.norm(l)
        direction = np.asarray(direction) / np.linalg.norm(direction)
        return np.arccos( np.clip(np.dot(l, direction), -1.0, 1.0) )
    ##########################################################################################
    # Conversion between gcmt (M11,M22,M33,M12,M13,M23), matUSE(3x3matrix), and matENU(3x3matrix)
    @staticmethod
    def gcmt2matENU(gcmt):
        """
        Convert GCMT moment tensor components to a 3 by 3 moment tensor in East-North-Up coordinates.

        :param gcmt: (M11, M22, M33, M12, M13, M23) - the six independent components of the moment tensor,
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

        :return mENU: a 3by3 matrix for the moment tensor in East-North-Up coordinates
        """
        M11, M22, M33, M12, M13, M23 = gcmt
        mat = np.array(((M33, -M23, M13), (-M23, M22, -M12), (M13, -M12, M11)), dtype=np.float64)
        return mat
    @staticmethod
    def matENU2gcmt(matENU):
        mee, men, meu = matENU[0]
        mne, mnn, mnu = matENU[1]
        mue, mun, muu = matENU[2]
        mus, mss, mse = -mue, mnn, -mne
        return np.array( (muu, mss, mee, mus, mue, mse), dtype=np.float64 )
    @staticmethod
    def xyz2uvw(matxyz, xyz='ENU', uvw='USE'):
        # 1. make sure both xyz and uvw are valid
        b2b = {'E':'W', 'N':'S', 'U':'D'}
        tmp1 = set([c if (c not in b2b) else b2b[c]  for c in xyz])
        tmp2 = set([c if (c not in b2b) else b2b[c]  for c in uvw])
        if len(tmp1)!=3 or len(tmp2)!=3:
            raise ValueError(f"Invalid xyz={xyz} or uvw={uvw} with repeated axis!")
        # 2. make all coordinate to be in UVW while unsorted
        matxyz = np.copy(matxyz).reshape((3,3) )
        for ic, c in enumerate(xyz):
            if c not in uvw:
                for j in range(3):
                    matxyz[ic, j] = -matxyz[ic, j]
                    matxyz[j, ic] = -matxyz[j, ic]
        # 3. sort to UVW order
        c2c = {'E':'W', 'W':'E', 'N':'S', 'S':'N', 'D':'U', 'U':'D'}
        c2i = {c:i for i, c in enumerate(uvw)}
        matuvw = np.zeros((3,3), dtype=np.float64 )
        for i, ci in enumerate(xyz):
            ci = ci if ci in uvw else c2c[ci]
            i_new = c2i[ci]
            for j, cj in enumerate(xyz):
                cj = cj if cj in uvw else c2c[cj]
                j_new = c2i[cj]
                matuvw[i_new, j_new] = matxyz[i,j]
        return matuvw
    ##########################################################################################
    @staticmethod
    def benchmark9(): # test radiation and radiation_fast
        app = Scene3D()
        bb = beachball3d(gcmt=(1,0.1,-1,0.2,0.3,0.1), normalize=True)
        cmap = plt.get_cmap('RdBu_r', 11)
        theta = np.deg2rad((10, 20, 30)).astype(np.float64)
        phi   = np.deg2rad((10, 20, 30)).astype(np.float64)
        print( bb.radiation_fast(theta, phi, wave_type='P', binarization=False) )
        print( bb.radiation_fast(theta[0], phi[0], wave_type='P', binarization=False) )
        pass
    @staticmethod
    def benchmark8(): # test radiation and radiation_fast
        app = Scene3D()
        bb = beachball3d(gcmt=(1,0.1,-1,0.2,0.3,0.1), normalize=True)
        cmap = plt.get_cmap('RdBu_r', 11)
        wave_type = 'S'
        #
        bb.use_fast = False
        bb.plot3d(p=app.pv_plotter,  wave_type=wave_type,  center=(0,0,0), cmap=cmap)
        bb.plot3d_pol(p=app.pv_plotter,  wave_type=wave_type,  center=(0,0,0), cmap=cmap)
        #
        bb.use_fast = True
        bb.plot3d(p=app.pv_plotter,  wave_type=wave_type,  center=(2,0,0), cmap=cmap)
        bb.plot3d_pol(p=app.pv_plotter,  wave_type=wave_type,  center=(2,0,0), cmap=cmap)
        app.pv_plotter.show()
        pass
    @staticmethod
    def benchmark7(): # test the P, SV, SH, and S polarity (vectors) for a DC source
        app = Scene3D()
        cmap = plt.get_cmap('RdBu_r', 11)
        gcmt = (1,0,-1,0,0,0)
        bb = beachball3d(gcmt=gcmt, normalize=True)#.get_dc()
        print(bb.norm)
        bb.plot3d(    p=app.pv_plotter, wave_type='P',  center=(  0,0,0), cmap=cmap )
        bb.plot3d_pol(p=app.pv_plotter, wave_type='P',  center=(  0,0,0), cmap=cmap )
        #
        bb.plot3d(    p=app.pv_plotter, wave_type='SV', center=( -3,0,0), cmap=cmap )
        bb.plot3d_pol(p=app.pv_plotter, wave_type='SV', center=( -3,0,0), cmap=cmap )
        #
        bb.plot3d(    p=app.pv_plotter, wave_type='SH', center=( -6,0,0), cmap=cmap )
        bb.plot3d_pol(p=app.pv_plotter, wave_type='SH', center=( -6,0,0), cmap=cmap )
        #
        bb.plot3d(    p=app.pv_plotter, wave_type='S',  center=( -9,0,0), cmap=cmap )
        bb.plot3d_pol(p=app.pv_plotter, wave_type='S',  center=( -9,0,0), cmap=cmap )
        app.pv_plotter.add_axes()
        app.pv_plotter.show()
    @staticmethod
    def benchmark6():  # plot DC 2d for all possible dip and slip angles
        ##### critical i angle for P and S wave from a source
        if True:
            prem_table =  [ (  0.0, 6371.0,  1.45, 0.00), # depth, radius, Vp (km/s), Vs (km/s)
                            (  3.0, 6368.0,  1.45, 0.00), # src: https://www.soest.hawaii.edu/GG/FACULTY/smithkonter/GG631/problemsets/PS14_PREM.pdf
                            (  3.0, 6368.0,  5.80, 3.20),
                            ( 15.0, 6356.0,  5.80, 3.20),
                            ( 15.0, 6356.0,  6.80, 3.90),
                            ( 24.4, 6346.6,  6.80, 3.90),
                            ( 24.4, 6346.6,  8.11, 4.49),
                            ( 71.0, 6300.0,  8.08, 4.47),
                            ( 80.0, 6291.0,  8.08, 4.47),
                            ( 80.0, 6291.0,  8.08, 4.47),
                            (171.0, 6200.0,  8.02, 4.44),
                            (220.0, 6151.0,  7.99, 4.42),
                            (220.0, 6151.0,  8.56, 4.64),
                            (271.0, 6100.0,  8.66, 4.68),
                            (371.0, 6000.0,  8.85, 4.75),
                            (400.0, 5971.0,  8.91, 4.77),
                            (400.0, 5971.0,  9.13, 4.93),
                            (471.0, 5900.0,  9.50, 5.14),
                            (571.0, 5800.0, 10.01, 5.43),
                            (600.0, 5771.0, 10.16, 5.52),
                            (600.0, 5771.0, 10.16, 5.52),
                            (670.0, 5701.0, 10.27, 5.57),
                            (670.0, 5701.0, 10.75, 5.95),
                            (771.0, 5600.0, 11.07, 6.24), ]
            prem_table = np.array(prem_table)
            evdp = np.arange(10, 700, 2)
            vp = np.interp(evdp, prem_table[:,0], prem_table[:,2])
            vs = np.interp(evdp, prem_table[:,0], prem_table[:,3])
            critical_slowness = 5 / (6371*np.pi/180) # 5 s/degree to s/km
            critical_ip = np.arcsin(critical_slowness * vp) # in radian
            critical_is = np.arcsin(critical_slowness * vs) # in radian
            # convert the i angle to radius in Schmidt projection
            critical_rp = np.sqrt(2)*np.sin(0.5*critical_ip)
            critical_rs = np.sqrt(2)*np.sin(0.5*critical_is)
            plt.semilogx(evdp, critical_rp, label='P-wave')
            plt.semilogx(evdp, critical_rs, label='S-wave')
            plt.grid(True)
            plt.legend()
            plt.savefig('critical_angles.png', bbox_inches='tight', dpi=300)
        ###### plot the beachball in 2d
        if False:
            dip  = np.arange(90, 0,  -5)
            slip = np.arange(0, 360,  5)
            fig, axmat = plt.subplots(dip.size, slip.size, subplot_kw={'projection': 'polar'}, figsize=(40, 40/slip.size*dip.size) )
            #cmap = plt.get_cmap('gray_r', 2)
            cmap = plt.get_cmap('RdBu_r', 9)
            for irow, d in enumerate(dip):
                for icol, s in enumerate(slip):
                    ax = axmat[irow, icol]
                    bb = beachball3d(strike_dip_slip=(90, d, s) )
                    bb.plot2d(ax_polar=ax, wave_type='P', hemisphere='lower', proj_method='Schmidt', radius=1.0, resolution_step_deg=1,
                            plot_zero_contour=False, cmap=cmap, diverging_clim=True,
                            color_nodal_planes='k', linewidth=10 )
                    theta = np.linspace(0, 2*np.pi, 100)
                    for r in (0.3, 0.4, 0.5):
                        ax.plot(theta, np.full(theta.shape, r), linewidth=0.9)
                    #ax.set_ylim((0, 0.5) )
            # plot the color bar for cmap
            # plt.colorbar(ax.collections[0], ax=axmat, orientation='horizontal', pad=0.1)
            for irow, d in enumerate(dip):
                ax = axmat[irow, 0]
                ax.set_xticks([np.pi])
                ax.set_xticklabels([d])
            for icol, s in enumerate(slip):
                ax = axmat[-1, icol]
                ax.set_xticks([1.5*np.pi])
                ax.set_xticklabels([s])
            plt.savefig('P_dip_slip.png', bbox_inches='tight', dpi=300)
    @staticmethod
    def benchmark5():
        ######################################################
        # plot beachball
        #gcmt = [0.91, -0.89, -0.02, 1.78, -1.55, 0.47]
        #gcmt = [0.91, -0.89, -0.02, 1.78, -1.55, 0.47]
        #gcmt = [0, 0, 0.2, 1, 0, 0]
        #gcmt = [2, 2, -4, 0, 0, 0]
        #gcmt = [-1, -0.5, 2, 1, 0.5, 0.6]
        gcmt = [2, -1, -1, 0, 0, 0]
        ######################################################
        #gcmt = [0, 0, 0, 0, 1, 0]
        bb = beachball3d(gcmt, normalize=False)
        bb.plot2d(wave_type='P', radius=1.0, binarization=False, hemisphere='upper', proj_method='Schmidt', resolution_step_deg=2, cmap='RdBu_r',
                   figname='junk2.png') #, clim=(-2, 2) )
        p = bb.plot3d(clim=(-2, 2) )#wave_type='P'
                        #  hemisphere='None', cmap=cmap, plot_zero_contour=True,
                        #  show_scalar_bar=False, scalar_bar_args={"vertical": False, "height":0.02,}, clim=(-2, 2) )
        p.add_axes()
        p.set_viewup([0, 1, 0])
        p.show()
    @staticmethod
    def benchmark4():
        gcmt = [1,2,3,4,5,6]
        bb = beachball3d(gcmt=gcmt)
        print(bb.matENU)
        matUSE = beachball3d.xyz2uvw(bb.matENU, xyz='ENU', uvw='USE')
        print(matUSE)
    def benchmark3():
        app = Scene3D()
        c1, c2, c3, c4 = (-15,0,0), (-5,0,0), (5,0,0), (15,0,0)
        for direc in ((1,0,0),(0,1,0), (0,0,1)):
            for center in (c1, c2, c3, c4):
                app.add_plane(center=center, direction=direc, opacity=0.05, i_size=8, j_size=8, i_resolution=8, j_resolution=8, show_edges=True, )
        app.pv_plotter.add_axes(xlabel='E', ylabel='N', zlabel='U')
        ######################################################
        # plot beachball
        #gcmt = [0.91, -0.89, -0.02, 1.78, -1.55, 0.47]
        #gcmt = [0.91, -0.89, -0.02, 1.78, -1.55, 0.47]
        gcmt = [0, 0, 0.2, 1, 0, 0]
        #gcmt = [2, 2, -4, 0, 0, 0]
        #gcmt = [-1, -1, 2, 0, 0, 0]
        ######################################################
        #gcmt = [0, 0, 0, 0, 1, 0]
        bb = beachball3d(gcmt, normalize=False)
        ######################################################
        p = app.pv_plotter
        p.disable_depth_peeling() # important for efficient rendering
        cmap = plt.get_cmap('PiYG_r', 13)
        #cmap = plt.get_cmap('RdGbBu_r', 13)
        np.set_printoptions(precision=2, suppress=True)
        for center, it_bb, label in zip((c1, c2, c3, c4), (bb, bb.get_iso(), bb.get_dc(), bb.get_clvd() ), ('M', 'ISO', 'DC', 'CLVD')):
            if it_bb.norm <= 0:
                continue
            #print(label, 'iso:', it_bb.is_pure_iso(), 'clvd:', it_bb.is_pure_clvd(), 'matENU:\n', it_bb.matENU )
            it_bb.plot3d(p, wave_type='P', label=label, center=center, radius=it_bb.norm,
                          hemisphere='None', cmap=cmap, plot_zero_contour=True,
                          show_scalar_bar=False, scalar_bar_args={"vertical": False, "height":0.02,}, clim=(-2, 2) )
        ######################################################
        app.show()
        #print(p.camera_position)
    @staticmethod
    def benchmark2():
        np.set_printoptions(formatter={'float': '{:5.1f}'.format})
        p = pv.Plotter()
        #####
        strike, dip, slip = 100, 30, 120
        print('Input:')
        print('strike, dip, slip:', strike, dip, slip)
        #
        bb = beachball3d(strike_dip_slip=(strike, dip, slip))
        bb.plot3d(p, center=(-1, 0, 0), radius=0.5, hemisphere='full')
        print('\nCHECK:')
        for idx, sds in enumerate(bb.getSDS()):
            print('strike, dip, slip:', sds)
            bb = beachball3d(strike_dip_slip=sds)
            bb.plot3d(p, center=(idx, 0, 0), radius=0.5, hemisphere='full')
        p.show()
    @staticmethod
    def benchmark():
        app = Scene3D()
        #app.add_plane(center=(20,0,0), direction=(1,0,0), opacity=0.1, i_size=30, j_size=30, i_resolution=30, j_resolution=30, show_edges=True, )
        #app.add_plane(center=(20,0,0), direction=(0,1,0), opacity=0.1, i_size=30, j_size=30, i_resolution=30, j_resolution=30, show_edges=True, )
        #app.add_plane(center=(20,0,0), direction=(0,0,1), opacity=0.1, i_size=30, j_size=30, i_resolution=30, j_resolution=30, show_edges=True, )
        #app.add_plane(center=(-20,0,0), direction=(1,0,0), opacity=0.1, i_size=30, j_size=30, i_resolution=30, j_resolution=30, show_edges=True, )
        #app.add_plane(center=(-20,0,0), direction=(0,1,0), opacity=0.1, i_size=30, j_size=30, i_resolution=30, j_resolution=30, show_edges=True, )
        #app.add_plane(center=(-20,0,0), direction=(0,0,1), opacity=0.1, i_size=30, j_size=30, i_resolution=30, j_resolution=30, show_edges=True, )
        app.pv_plotter.add_axes(xlabel='E', ylabel='N', zlabel='U')
        ######################################################
        # plot beachball
        #gcmt = [0.91, -0.89, -0.02, 1.78, -1.55, 0.47]
        #gcmt = [0, 0, -0.1, 1, 0, 0] # (Mrr=M11, Mtt=M22, Mpp=M33, Mrt=M12, Mrp=M13, Mtp=M23).
        #gcmt = [0.2, 0.2, 0.2, -0.2, -1, 0.2]
        gcmt = [-1, 0, 1, 0, 0, 0]
        #gcmt = [0, 0, 0, 0, 1, 0]
        bb = beachball3d(gcmt)
        cmap = ListedColormap(('#444444', '#eeeeee'))
        cmap = plt.get_cmap('gray_r', 2)
        cmap = plt.get_cmap('RdBu_r', 13)
        binarization = False
        p = app.pv_plotter
        app = Scene3D(pv_plotter=p)
        #
        bb.plot3d(p,     center=(-20,0,0), radius=6.0, hemisphere=None, cmap=cmap, plot_zero_contour=True, show_scalar_bar=True, binarization=binarization)
        bb.plot3d_pol(p, center=(-20,0,0), radius=6.0, hemisphere=None, alpha=1.0, lighting=False, culling=False, scale=3)
        #
        bb.plot3d(p,     center=(0,0,0), radius=6.0, hemisphere=None, cmap=cmap, plot_zero_contour=True, show_scalar_bar=True, wave_type='S', binarization=binarization)
        bb.plot3d_pol(p, center=(0,0,0), radius=6.0, hemisphere=None, alpha=1.0, lighting=False, culling=False, scale=3, wave_type='S')
        #
        app.add_disk(center=(0,0,0), direction=(1,1,0), opacity=0.8, outer=8, show_edges=False, color='c', culling=False)
        app.disable_add_mesh_to_plotter()
        mesh, actor = app.add_disk(center=(0,0,0), direction=(1,1,0), opacity=0.8, outer=8, show_edges=False, color='k', culling=False)
        app.enable_add_mesh_to_plotter()
        mesh.points = Scene3D.rotate_and_translate2(mesh.points, current_origin_xyz=(0,0,0), current_orientation_direction1=(0,0,1), current_orientation_direction2=(1,0,0),
                                                    new_origin_xyz=(0,0,0), new_orientation_direction1=(1,0,1), new_orientation_direction2=(1,0,-1) )
        p.add_mesh(mesh, opacity=0.8, show_edges=False, color='y', culling=False)
        #################################################################
        gcmt = [0, 0, 0, 0, 1, 0]
        bb2 = beachball3d(gcmt)
        bb2.plot3d(p,     center=(40,0,0), radius=6.0, hemisphere=None, cmap=cmap, plot_zero_contour=True, show_scalar_bar=True, wave_type='P', binarization=binarization)
        bb2.plot3d_pol(p, center=(40,0,0), radius=6.0, hemisphere=None, alpha=1.0, lighting=False, culling=False, scale=3, wave_type='P')
        #
        bb2.plot3d(p,     center=(20,0,0), radius=6.0, hemisphere=None, cmap=cmap, plot_zero_contour=True, show_scalar_bar=True, wave_type='S', binarization=binarization)
        bb2.plot3d_pol(p, center=(20,0,0), radius=6.0, hemisphere=None, alpha=1.0, lighting=False, culling=False, scale=3, wave_type='S')
        #
        app.add_disk(center=(20,0,0), direction=(1,1,0), opacity=0.8, outer=8, show_edges=False, color='y', culling=False)
        app.disable_add_mesh_to_plotter()
        mesh, actor = app.add_disk(center=(20,0,0), direction=(1,1,0), opacity=0.8, outer=8, show_edges=False, color='k', culling=False)
        app.enable_add_mesh_to_plotter()
        mesh.points = Scene3D.rotate_and_translate2(mesh.points, current_origin_xyz=(20,0,0), current_orientation_direction1=(0,0,1), current_orientation_direction2=(1,0,0),
                                                    new_origin_xyz=(20,0,0), new_orientation_direction1=(-1,0,1), new_orientation_direction2=(1,0,1) )
        p.add_mesh(mesh, opacity=0.8, show_edges=False, color='c', culling=False)
        #
        p.camera_position= [(-43.53802264331248, -82.34998370517927, 54.98109464231128),
                            (0.0, 0.0, 0.0),
                            (0.03490696877103766, 0.5420982707769743, 0.8395897619384318)]
        ######################################################
        #print(p.camera_position)
        p.export_gltf("DC_P_SV_SH.gltf")
        app.show()
if __name__ == '__main__':
    Scene3D.benchmark_SKS_ScS_diagram()
    sys.exit(0)
    beachball3d.benchmark9()
    sys.exit(0)
    #sys.exit(0)
    beachball3d.benchmark8()
    sys.exit(0)
    beachball3d.benchmark7()
    sys.exit(0)
    beachball3d.benchmark3()
    sys.exit(0)
    globe3d.benchmark4()
    sys.exit(0)
    globe3d.benchmark()
    sys.exit(0)
    p = pv.Plotter(notebook=0, shape=(1, 1), border=False, window_size=(1700, 1000) )
    p.set_background('white')
    ######
    # plot surface
    mesh = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=30, j_size=30, i_resolution=10, j_resolution=10)
    p.add_mesh(mesh, show_edges=True, opacity=0.6,
               smooth_shading=True, lighting=False, culling=False)
    ######
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
