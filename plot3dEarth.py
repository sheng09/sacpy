#!/usr/bin/env python3

from matplotlib.colors import  ListedColormap, LinearSegmentedColormap
import pyvista as pv
from pyvista import examples as pv_examples

import numpy as np
import pickle
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

class Scene3D:
    """
    A scene holding multiple 3D objects.
    """
    def __init__(self, pv_plotter=None, pv_plotter_kargs={'window_size':(1700, 1200)} ):
        """
        Initialize the 3D scene.

        :param pv_plotter:       An existing PyVista plotter instance (default is None).
        :param pv_plotter_kargs: Keyword arguments for creating the PyVista plotter if `pv_plotter` is not provided.
        """
        if pv_plotter is not None:
            self.pv_plotter = pv_plotter
        else:
            self.pv_plotter = pv.Plotter(**pv_plotter_kargs)
        self.pv_plotter.enable_depth_peeling() #  to handles multiple transparent surfaces correctly
    def add_plane(self, center=(0,0,0), direction=(0,0,1), color='gray',
                    i_size=10, j_size=10, i_resolution=10, j_resolution=10,
                    opacity=0.6, lighting=True, **kwargs):
        """
        Add a plane.

        :param center:    The center of the plane.
        :param direction: The direction of the plane normal.
        :param color:     The color of the plane.
        :param i_size:    The size of the plane in the i direction.
        :param j_size:    The size of the plane in the j direction.
        :param i_resolution: The resolution of the plane in the i direction.
        :param j_resolution: The resolution of the plane in the j direction.
        :param opacity:      The opacity of the plane.
        :param lighting:       Whether to use lighting.
        :param kwargs:         Additional keyword arguments for `pyvista.Plotter.add_mesh(...)`.
        """
        mesh = pv.Plane(center=center, direction=direction, i_size=i_size, j_size=j_size, i_resolution=i_resolution, j_resolution=j_resolution)
        self.pv_plotter.add_mesh(mesh, color=color, opacity=opacity, lighting=lighting, **kwargs)
        pass
    def add_spline(self, xs, ys, zs,
                 scalars=None, color='k', cmap='viridis', line_width=5, label='line',
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
        :param lighting: Whether to use lighting.
        :param kwargs: Additional keyword arguments for `pyvista.Plotter.add_mesh(...)`.
        """
        points = np.column_stack((xs, ys, zs))
        spline = pv.Spline(points)
        if scalars is None:
            self.pv_plotter.add_mesh(spline, color=color, cmap=cmap, line_width=line_width, lighting=lighting, **kwargs)
        else:
            spline.point_data[label] = scalars
            self.pv_plotter.add_mesh(spline, scalars=label, color=color, cmap=cmap, line_width=line_width, lighting=lighting, **kwargs)
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
        self.pv_plotter.add_mesh(mesh, color=color, lighting=lighting, **kwargs)
    def add_arrow(self, loc, direction, scale=1, loc_is_end=False, color='k',
                  tip_length=0.3, shaft_resolution=30, shaft_radius=0.01,
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
        :param shaft_resolution: The resolution of the arrow shaft.
        :param shaft_radius: The radius of the arrow shaft.
        :param lighting: Whether to use lighting.
        :param kwargs: Additional keyword arguments for `pyvista.Plotter.add_mesh(...)`.
        """
        if loc_is_end:
            loc = np.array(loc, dtype=np.float64)
            vec = np.array(direction, dtype=np.float64)
            vec /= np.linalg.norm(vec)
            loc -= (vec*scale)
        mesh = pv.Arrow(start=loc, direction=direction, scale=scale, tip_length=tip_length, shaft_resolution=shaft_resolution, shaft_radius=shaft_radius)
        self.pv_plotter.add_mesh(mesh, color=color, lighting=lighting, **kwargs)
    def add_disk(self, center=(0,0,0), inner=0.0, outer=1.0, normal=(0,0,1), color='k', r_res=1, c_res=180,
                 lighting=True, **kwargs):
        disc = pv.Disc(center=center, inner=inner, outer=outer, normal=normal, r_res=r_res, c_res=c_res)
        self.pv_plotter.add_mesh(disc, color=color, lighting=lighting, **kwargs)
    def add_sphere(self, radius=1, center=(0,0,0),
                   rotation_about_north_pole_deg=0.0, north_pole_direction=(0, 0, 1),
                   color='r', texture=None,
                   theta_resolution=60, start_theta=0, end_theta=360,
                   phi_resolution=30,   start_phi=0,   end_phi=180,
                   lighting=True, **kwargs):
        """
        Add a sphere to the scene.

        :param radius: The radius of the sphere.
        :param center: The center of the sphere.
        :param rotation_about_north_pole_deg: The rotation angle about the north pole in degrees. (default is 0 degree).
        :param north_pole_direction: The direction of the north pole of the sphere. (default is (0,0,1), the vertical direction).
        :param color: The color of the sphere.
        :param texture: The texture of the sphere (default is None and will use the color parameter for uniform color).
                        The texture could be a figure filename, or a grd of rgb values, where the grd should have a
                        meaningful shape: (nrow, ncol, 3) for RGB or (nrow, ncol, 4) for RGBA values. RGBA are in [0,255]!
                        The upper-left of the image (or the grd) should have the lon=0,lat=90; the upper-right lon=360,lat=90;
                        the lower-left lon=0,lat=-90; and the lower-right lon=360,lat=-90.
        :param theta_resolution: The resolution of the sphere in the theta direction.
        :param start_theta: The starting angle of the sphere in the theta direction.
        :param end_theta: The ending angle of the sphere in the theta direction.
        :param phi_resolution: The resolution of the sphere in the phi direction.
        :param start_phi: The starting angle of the sphere in the phi direction.
        :param end_phi: The ending angle of the sphere in the phi direction.
        :param lighting: Whether to use lighting.
        :param kwargs: Additional keyword arguments for `pyvista.Plotter.add_mesh(...)`.
        """
        if texture is not None:
            # fix the distortion at seam
            # ref: https://discourse.paraview.org/t/single-azimuthal-segment-texture-distortion-for-earth-texturemaptosphere/783/5
            start_theta += 0.001
            end_theta   -= 0.001
        sphere = pv.Sphere(radius=1, center=(0,0,0), direction=(0,0,1),
                           theta_resolution=theta_resolution, start_theta=start_theta, end_theta=end_theta,
                           phi_resolution=phi_resolution, start_phi=start_phi, end_phi=end_phi)
        #### 1. add the `active_texture_coordinates` to the mesh
        if texture is not None:
            ## Manually set the texture coordinates (u, v) for the sphere
            ## u corresponds to the horizontal direction (left to right) on the texture, here need to be 0 to 2pi in longitude
            ## v corresponds to the vertical direction (bottom to top) on the texture, here need to be -pi/2 to pi/2 in latitude
            ## ref: https://github.com/pyvista/pyvista-support/issues/257#issuecomment-705518157
            #x, y, z = sphere.points[:, 0], sphere.points[:, 1], sphere.points[:, 2]
            #lon = np.arctan2(y, x) % (np.pi*2) # lon: [0, 2pi]
            #u = lon*(0.5/np.pi)
            #lat = np.arcsin(z) # lat: [-pi/2, pi/2]
            #v = (lat+np.pi/2)*(1/np.pi)
            #sphere.active_texture_coordinates = np.column_stack((u, v))
            ########################################################################################################################
            ## Use internal texture mapping
            # Now, the u is 0->1 for theta (or longitude) 0->360
            #      the v is 0->1 for phi 0->180 (or latitude 90-> -90), so that we need to change it
            sphere.texture_map_to_sphere(inplace=True, prevent_seam=False)
            sphere.active_texture_coordinates[:,1] = 1-sphere.active_texture_coordinates[:,1] # flip the v directin to top->bot to bot->top
        #### 2. rotate about the north pole
        if (rotation_about_north_pole_deg%360) > 1e-6:
            x, y = sphere.points[:, 0], sphere.points[:, 1]
            rotation_about_north_pole = np.deg2rad(rotation_about_north_pole_deg)
            c, s = np.cos(rotation_about_north_pole), np.sin(rotation_about_north_pole)
            new_x =  x*c - y*s
            new_y =  x*s + y*c
            sphere.points[:, 0] = new_x
            sphere.points[:, 1] = new_y
        #### 3. rotate to make the unit sphere's north pole direction match the given direction
        if True:
            v0 = np.array((0.,0.,1.), dtype=np.float64)
            v1 = np.array(north_pole_direction, dtype=np.float64)
            v1 /= np.linalg.norm(v1)
            #          v0
            #      v1  |
            #        \ |
            #         \|
            #          o
            #         /
            #        /rotation_axis =  v0 x v1
            angle = np.arccos(np.clip(np.dot(v0, v1), -1.0, 1.0) )
            if angle > 1.e-6:
                rotation_axis = np.cross(v0, v1)  # rotation axis (not unit)
                R = Scene3D.rotation_matrix(rotation_axis, angle)
                xyz = sphere.points
                R = R.T
                # xyz.T = R @ xyz.T ==> xyz = xyz @ R.T
                xyz = xyz @ R
                sphere.points = xyz
        #### 4. scale to radius and translate to the center
        if True:
            sphere.points *= radius
            sphere.points += center
        ####
        if texture is None:
            self.pv_plotter.add_mesh(sphere, color=color, lighting=lighting, **kwargs)
        else:
            texture = pv.Texture(texture)
            texture.interpolate = False                  # avoid smearing a tiny 2x2
            texture.repeat = False                       # avoid repeating the texture
            self.pv_plotter.add_mesh(sphere, texture=texture, lighting=lighting,  **kwargs)
    def add_sphere_grd(self, grd_theta_deg, grd_phi_deg, grd_values,
                       radius=1, center=(0,0,0),
                       rotation_about_north_pole_deg=0.0, north_pole_direction=(0, 0, 1),
                       cmap='viridis', label='sphere_grd',
                       start_theta=0, end_theta=360,
                       start_phi=0,   end_phi=180,
                       lighting=True, **kwargs):
        """
        Add a spherical grid to the scene.

        :param grd_theta_deg: The theta coordinates (0->360) of the grid. (theta could be considered as longitude.)
        :param grd_phi_deg:   The phi coordinates (0->180) of the grid. (phi could be considered as colatitude.)
        :param grd_values:  The values to be mapped onto the grid.
                            Note, the `grd_values` should correspond to a matrix with the shape (len(grd_phi_deg), len(grd_theta_deg))!
        :param radius: The radius of the sphere.
        :param center: The center of the sphere.
        :param rotation_about_north_pole_deg: The rotation angle about the north pole in degrees. (default is 0 degree).
        :param north_pole_direction: The direction of the north pole of the sphere. (default is (0,0,1), the vertical direction).
        :param cmap: The colormap to use.
        :param label: The label for the scalar data (grd_values).
        :param start_theta: The starting angle of the sphere in the theta direction.
        :param end_theta: The ending angle of the sphere in the theta direction.
        :param start_phi: The starting angle of the sphere in the phi direction.
        :param end_phi: The ending angle of the sphere in the phi direction.
        :param lighting: Whether to use lighting.
        :param kwargs: Additional keyword arguments for `pyvista.Plotter.add_mesh(...)`.
        """
        x, y       = grd_theta_deg, grd_phi_deg
        grd_values = grd_values.reshape(len(grd_phi_deg), len(grd_theta_deg))
        # find idx in x for start_theta and end_theta, which might not exist in the x
        ix0 = np.searchsorted(x, start_theta, side='left')
        ix1 = np.searchsorted(x, end_theta,   side='right')
        iy0 = np.searchsorted(y, start_phi,   side='left')
        iy1 = np.searchsorted(y, end_phi,     side='right')
        x = x[ix0:ix1]
        y = y[iy0:iy1]
        grd_values = grd_values[iy0:iy1, ix0:ix1]
        #### 1. set a unit sphere centered at (0,0,0)
        sphere = pv.grid_from_sph_coords(x, y, [1.0,] )
        sphere.point_data[label] = np.array(grd_values).T.flatten() # note here! The input `grd_value` lat as 0th axis and lon as 1th axis
        #### 2. rotate about the north pole
        if (rotation_about_north_pole_deg%360) > 1e-6:
            x, y = sphere.points[:, 0], sphere.points[:, 1]
            rotation_about_north_pole = np.deg2rad(rotation_about_north_pole_deg)
            c, s = np.cos(rotation_about_north_pole), np.sin(rotation_about_north_pole)
            new_x =  x*c - y*s
            new_y =  x*s + y*c
            sphere.points[:, 0] = new_x
            sphere.points[:, 1] = new_y
        #### 3. rotate to make the unit sphere's north pole direction match the given direction
        if True:
            v0 = np.array((0.,0.,1.), dtype=np.float64)
            v1 = np.array(north_pole_direction, dtype=np.float64)
            v1 /= np.linalg.norm(v1)
            #          v0
            #      v1  |
            #        \ |
            #         \|
            #          o
            #         /
            #        /rotation_axis =  v0 x v1
            angle = np.arccos(np.clip(np.dot(v0, v1), -1.0, 1.0) )
            if angle > 1.e-6:
                rotation_axis = np.cross(v0, v1)  # rotation axis (not unit)
                R = Scene3D.rotation_matrix(rotation_axis, angle)
                xyz = sphere.points
                R = R.T
                # xyz.T = R @ xyz.T ==> xyz = xyz @ R.T
                xyz = xyz @ R
                sphere.points = xyz
        #### 4. scale to radius and translate to the center
        if True:
            sphere.points *= radius
            sphere.points += center
        self.pv_plotter.add_mesh(sphere, cmap=cmap, lighting=lighting, **kwargs)
    def add_earth(self, land_color='#ccccccff', ocean_color='#00000000',
                  coastline_width=0.5, coastline_color='r', coastline_style='-', plot_stock_img=False,
                  radius=1, center=(0,0,0),
                  rotation_about_north_pole_deg=0.0, north_pole_direction=(0, 0, 1),
                  theta_resolution=180, start_theta=0, end_theta=360,
                  phi_resolution=90,   start_phi=0,   end_phi=180,
                  lighting=True, **kwargs):
        ####
        rgba = Scene3D.plot_earth_basemap(land_color=land_color, ocean_color=ocean_color,
                                          coastline_width=coastline_width, coastline_color=coastline_color, coastline_style=coastline_style,
                                          plot_stock_img=plot_stock_img)
        self.add_sphere(radius=radius, center=center,
                        rotation_about_north_pole_deg=rotation_about_north_pole_deg, north_pole_direction=north_pole_direction,
                        texture=rgba, theta_resolution=theta_resolution, start_theta=start_theta, end_theta=end_theta,
                        phi_resolution=phi_resolution, start_phi=start_phi, end_phi=end_phi,
                        lighting=lighting, **kwargs)
        pass
    def add_planet(self, basemap_filename, basemap_lon_range=(0, 360), basemap_lat_range=(-90,90),
                  radius=1, center=(0,0,0),
                  rotation_about_north_pole_deg=0.0, north_pole_direction=(0, 0, 1),
                  theta_resolution=40, start_theta=0, end_theta=360,
                  phi_resolution=20,   start_phi=0,   end_phi=180,
                  lighting=True, **kwargs):
        """
        """
        pixels = plt.imread(basemap_filename)
        if pixels.dtype != np.uint8:
            pixels = (pixels * 255).astype(np.uint8)  # Ensure pixels are in [0, 255] range
        if (basemap_lon_range != (0, 360)) or (basemap_lat_range != (-90, 90)):
            if True:
                ny, nx = pixels.shape[:2]
                x0, x1 = basemap_lon_range
                y0, y1 = basemap_lat_range
                dx = (x1 - x0) / (nx - 1)
                dy = (y1 - y0) / (ny - 1)
                #
                u0, u1 =   0, 360
                v0, v1 = -90, 90
                nu = int(round((u1 - u0) / dx) )
                nv = int(round((v1 - v0) / dy) )
                u_start = int(round((x0-u0) / dx) )
                v_start = int(round((y0-v0) / dy) )
                u_end   = u_start + nx
                v_end   = v_start + ny
                #
                x_start, x_end = 0, nx
                y_start, y_end = 0, ny
                #
                if u_start<0:
                    x_start = -u_start
                    u_start = 0
                if u_end>nu:
                    x_end -= (u_end - nu)
                    u_end = nu
                if v_start<0:
                    y_start = -v_start
                    v_start = 0
                if v_end>nv:
                    y_end -= (v_end - nv)
                    v_end = nv
                #
                out = np.zeros((nv, nu, 4), dtype=np.uint8)
                out[v_start:v_end, u_start:u_end, :] = pixels[y_start:y_end, x_start:x_end, :]
                pixels = out
            else: # interpolation which is too slow
                if pixels.shape[2] == 3:  # add alpha if not exists
                    alpha_channel = np.ones((pixels.shape[0], pixels.shape[1], 1), dtype=pixels.dtype) * 255
                    pixels = np.concatenate((pixels, alpha_channel), axis=2)
                # adjust the basemap to make its range 0 to 360 for longitude and -90 to 90 for latitude
                ny, nx = pixels.shape[:2]
                x0, x1 = basemap_lon_range
                y0, y1 = basemap_lat_range
                # Original grid coordinates
                x = np.linspace(x0, x1, nx)
                y = np.linspace(y0, y1, ny)
                #
                dx = (x1 - x0) / (nx - 1)
                dy = (y1 - y0) / (ny - 1)
                u0, u1 =   0, 360
                v0, v1 = -90, 90
                nx_new = int(round(abs(u1 - u0) / abs(dx))) + 1
                ny_new = int(round(abs(v1 - v0) / abs(dy))) + 1
                #
                x_new = np.linspace(u0, u1, nx_new)
                y_new = np.linspace(v0, v1, ny_new)
                Xn, Yn = np.meshgrid(x_new, y_new)
                #
                new_pixels = np.zeros((ny_new, nx_new, 4), dtype=np.uint8)
                for depth in range(4):
                    interp = RegularGridInterpolator( (y, x), pixels[:,:,depth], method='linear', bounds_error=False, fill_value=0 )
                    pts = np.column_stack([Yn.ravel(), Xn.ravel()])  # (y, x) order
                    new_pixels[:,:,depth] = interp(pts).reshape(ny_new, nx_new).astype(np.uint8)
                pixels = new_pixels
        self.add_sphere(radius=radius, center=center,
                        rotation_about_north_pole_deg=rotation_about_north_pole_deg, north_pole_direction=north_pole_direction,
                        texture=pixels, theta_resolution=theta_resolution, start_theta=start_theta, end_theta=end_theta,
                        phi_resolution=phi_resolution, start_phi=start_phi, end_phi=end_phi,
                        lighting=lighting, **kwargs)
    def add_planet2(self, basemap_filename, basemap_lon_range=(0, 360), basemap_lat_range=(-90,90),
                    radius=1, center=(0,0,0),
                    rotation_about_north_pole_deg=0.0, north_pole_direction=(0, 0, 1),
                    lighting=True, **kwargs):
        """
        """
        pixels = plt.imread(basemap_filename)#[::10,::10,:]
        nrow, ncol = pixels.shape[:2]
        ratio = int(ncol / 720)+1
        pixels = pixels[::ratio, ::ratio, :]  # downsample the image
        r,g,b = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
        vs = np.arctan2(np.sqrt(3)*(g-b), 2*r-g-b) # hue
        #vs = (r+r+r+b+g+g+g+g)/8
        lons = np.linspace(basemap_lon_range[0], basemap_lon_range[1], pixels.shape[1])
        lats = 90+np.linspace(basemap_lat_range[0], basemap_lat_range[1], pixels.shape[0])
        self.add_sphere_grd(lons, lats, vs,
                            radius=radius, center=center,
                            rotation_about_north_pole_deg=rotation_about_north_pole_deg, north_pole_direction=north_pole_direction,
                            lighting=lighting, **kwargs)
    def show(self):
        self.pv_plotter.show()
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
                           plot_stock_img=False):
        prj = ccrs.PlateCarree(central_longitude=180)
        fig = plt.figure(figsize=(20, 10), dpi=200)
        ax = fig.add_axes([0, 0, 1, 1], projection=prj)  # (left, bottom, width, height) in figure coords
        fig.patch.set_alpha(0.0) # set background to transparent
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
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        rgba = np.array(fig.canvas.buffer_rgba())
        plt.close(fig)
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
        app.add_plane(color='gray', show_edges=True, opacity=0.1)
        app.add_plane(color='gray', direction=(0,1,0), show_edges=True, opacity=0.1)
        app.add_plane(color='gray', direction=(1,0,0), show_edges=True, opacity=0.1)
        if False:
            app.add_disk((0,0,1), 0.5, 1.0, (0, 0, 1), 'c', opacity=0.6)
            app.add_arrow((0,0,0), (0,0,1), 2)
        if False:
            xs = np.linspace(-5, 5, 1000)
            yz, zs = np.cos(5*xs), np.sin(5*xs)-2
            scalars = zs
            app.add_spline(xs, yz, zs, scalars=scalars, color='k', cmap='viridis', line_width=20, opacity=1) #, scalar_bar_args={'vertical':True} )
            app.add_sphere(radius=2, center=(0,0,0),
                           color='g', opacity=0.9)
            for shape in ('cone', 'cylinder', 'sphere'):
                loc = np.random.rand(3)
                loc *= (2/np.linalg.norm(loc))
                x, y, z = loc
                app.add_point(x, y, z, -loc, 0.2, shape=shape, color='r', shift_along_direction=-0.2)
        if True:
            app.add_arrow(loc=(0,0,0), direction=(1, 0, 0), scale=5)
            app.add_arrow(loc=(0,0,0), direction=(1, 0, 0), loc_is_end=True, color='gray', scale=5)
            app.add_arrow(loc=(0,0,0), direction=(0, 1, 0), scale=-5, color=(255, 0, 0))
            app.add_arrow(loc=(0,0,0), direction=(0, 1, 0), loc_is_end=True, color=(255, 150, 150), scale=-5)
        if False:
            app.add_sphere(radius=2, center=(0,0,0),
                           color='g', opacity=0.9,
                           theta_resolution=60, start_theta=30, end_theta=300,
                           phi_resolution=30,   start_phi=20,   end_phi=150,)
        if False:
            app.add_sphere(radius=2, center=(0,0,0),
                           texture='/Users/sw/Programs_Sheng/sacpy/dataset/global_maps/fancy2_0-360.png',
                           opacity=0.9,
                           theta_resolution=60, start_theta=30, end_theta=300,
                           phi_resolution=30,   start_phi=20,   end_phi=150, culling=True)
        if False:
            app.add_disk(color='r', outer=3, normal=(1,-1,0), show_edges=True, opacity=0.1)
            app.add_disk(color='r', outer=3, normal=(1,1,-2), show_edges=True, opacity=0.1)
            app.add_disk(color='r', outer=3, normal=(1,1,1), show_edges=True, opacity=0.1)
            app.add_sphere(radius=2, center=(0,0,0),
                           texture='/Users/sw/Programs_Sheng/sacpy/dataset/global_maps/fancy2_0-360.png',
                           opacity=0.9,
                           rotation_about_north_pole_deg=45, north_pole_direction=(1, 1, 1),
                           culling=True)
        if False:
            grd_rgba = np.array(
                [
                    [50,0,0,100], [100, 0, 0, 255], [255, 0, 0, 255],
                    [0,50,0,255], [0, 100, 0, 100], [0, 255, 0, 255],
                    [0,0,50,255], [0, 0, 100, 255], [0, 0, 255, 100],
                ],
                dtype=np.uint8
            ).reshape((3, 3, 4), order='C')#[::-1,:,:]  # (H, W, 4)
            app.add_sphere(radius=2, center=(0,0,0),
                           texture=grd_rgba,
                           rotation_about_north_pole_deg=0, north_pole_direction=(0, 0, 1),
                           culling=True)
            app.add_sphere(radius=1, color='k')
        if False:
            lons   = np.arange(0, 360.1, 1)
            colats = np.arange(0, 180.1, 1)
            grd = np.zeros( (colats.size, lons.size) )
            grd[:90,:90] = 1
            grd[40:50, 90:180] = -1
            app.add_sphere_grd(lons, colats, grd,
                               radius=5, center=(0,0,0),
                               cmap='bwr', label='test', opacity=1, show_edges=True,
                               start_theta=0, end_theta=350,
                               start_phi=10,   end_phi=170,)
        if False:
            lons   = np.arange(0, 360.1, 1)
            colats = np.arange(0, 180.1, 1)
            grd = np.zeros( (colats.size, lons.size) )
            grd[:90,:90] = 1
            grd[40:50, 90:180] = -1
            app.add_sphere_grd(lons, colats, grd,
                               radius=5, center=(0,0,0),
                               cmap='bwr', label='test', opacity=1, show_edges=True,
                               rotation_about_north_pole_deg=30, north_pole_direction=(-1, -1, 1) )
        if False:
            app.add_earth( land_color='#ccccccff', ocean_color='#0000ff11',
                            coastline_color='r', coastline_style='-', coastline_width=0.6, #2.0,
                            radius=10, center=(0, 0, 0),
                            #rotation_about_north_pole_deg=0, north_pole_direction=(1, 1, 3),
                            start_theta=0, end_theta=360, start_phi=0, end_phi=180,
                            culling=True, show_edges=False)
            app.add_sphere(radius=5, color='k')
        if False:
            app.add_earth( plot_stock_img=True,
                            coastline_color='r', coastline_style='-', coastline_width=0.6, #2.0,
                            radius=10, center=(0, 0, 0),
                            #rotation_about_north_pole_deg=0, north_pole_direction=(1, 1, 3),
                            start_theta=0, end_theta=360, start_phi=0, end_phi=180,
                            culling=True, show_edges=False)
            app.add_sphere(radius=5, color='k')
        if False:
            app.add_planet('/Users/sw/Programs_Sheng/sacpy/dataset/global_maps/fancy2_0-360.png', (0, 360), (-90, 90),
                        radius=3, center=(0, 0, 0),
                        rotation_about_north_pole_deg=45, north_pole_direction=(1, 1, 1),
                        start_theta=0, end_theta=360, start_phi=0, end_phi=180,
                        opacity=1, culling=True, show_edges=True)
        if False:
            app.add_planet2('/Users/sw/Programs_Sheng/sacpy/dataset/global_maps/fancy2_0-360.png', (0, 360), (-90, 90),
                        radius=5, center=(0, 0, 0),
                        rotation_about_north_pole_deg=45, north_pole_direction=(1, 1, 1),
                        start_theta=0, end_theta=360, start_phi=0, end_phi=180,
                        opacity=1, culling=True, show_edges=True)
        ##
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
    """
    def __init__(self, gcmt=None, mtUSE=None, mtENU=None, normalize=True):
        """
        Initialize the beachball3d class with optional gcmt(6-element array), or mtUSE(3x3 matrix), or mtENU(3x3 matrix).

        :param gcmt:  (M11, M22, M33, M12, M13, M23) in Up-South-East coordinate..
        :param mtUSE: a 3x3 moment tensor in Up-South-East coordinate.
        :param mtENU: a 3x3 moment tensor in East-North-Up coordinate.
        :param normalize: whether to normalize the moment tensor. (default is True)

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
            self.matENU = beachball3d.gcmt2mtENU(gcmt, normalize=normalize)
        elif mtUSE is not None:
            self.matENU = beachball3d.mtUSE2mtENU(mtUSE, normalize=normalize)
        elif mtENU is not None:
            self.matENU = mtENU
    @property
    def gcmt(self):   # the gcmt array (M11, M22, M33, M12, M13, M23) in USE coordinate
        return beachball3d.mtENU2gcmt(self.matENU, normalize=False)
    @property
    def matUSE(self): # the 3x3 matrix for the moment tensor in USE coordinate
        return beachball3d.mtENU2mtUSE(self.matENU, normalize=False)
    @property
    def iso(self):    # the isotropic part of the moment tensor (invariant for any coordinate system)
        (mtTBP, deviatoric_mtTBP, iso_mt), _ = beachball3d.mtENU2strike_dip_slip(self.matENU)
        return iso_mt
    @property
    def deviatoric(self): # the deviatoric part of the moment tensor in ENU coordinate
        return self.matENU - self.iso
    @property
    def deviatoricTBP(self): # the deviatoric part of the moment tensor in TBP coordinate (so the mat has only non-zero values at 11 and 33 positions)
        (mtTBP, deviatoric_mtTBP, iso_mt), _ = beachball3d.mtENU2strike_dip_slip(self.matENU)
        return deviatoric_mtTBP
    @property
    def mtTBP(self):  # the full moment tensor in TBP coordinate
        (mtTBP, deviatoric_mtTBP, iso_mt), _ = beachball3d.mtENU2strike_dip_slip(self.matENU)
        return mtTBP
    @property
    def two_strike_dip_slip(self): # a pair of (strike, dip, slip) for two planes as defined in Fig 4.2-2 in Stein and Wysession (2003).
        _, pair_strike_dip_slip = beachball3d.mtENU2strike_dip_slip(self.matENU)
        return pair_strike_dip_slip
    @property
    def tbp(self):    # T, B, P unit vectors (minimal compressive, null, maximal compressive directions). Note: (-T, -B, -P) also works
        (vec_t, vec_b, vec_p), _, _ = beachball3d.getTBP(self.matENU)
        return vec_t, vec_b, vec_p
    @property
    def two_nd(self): # a pair of (n, d) for two planes. Note: (-N, -D) also works.
        _, (vec_n1, vec_d1), (vec_n2, vec_d2) = beachball3d.getTBP(self.matENU)
        return (vec_n1, vec_d1), (vec_n2, vec_d2)
    def __radiation(self, thetas, phis, binarization=False):
        """
        Compute the P-, SV- and SH-wave radiations given a list of points on a unit sphere.

        thetas, phis: a list of theta and phi for the points on
                      a unit sphere. theta is angle between
                      the direction and x axis, and phi the angle
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
        """
        sin_phi, cos_phi     = np.sin(phis), np.cos(phis)
        sin_theta, cos_theta = np.sin(thetas), np.cos(thetas)
        ####################
        #unit P wave polarity vectors from the center of the source
        P_pol      = np.zeros((len(thetas), 3 ) )
        P_pol[:,0] = sin_phi * cos_theta
        P_pol[:,1] = sin_phi * sin_theta
        P_pol[:,2] = cos_phi
        loc_xyz = P_pol
        #the SV vectors that are perpendicular to the radial vectors and in vertical planes
        SV_pol      = np.zeros((len(thetas), 3 ) )
        SV_pol[:,0] = cos_phi * cos_theta
        SV_pol[:,1] = cos_phi * sin_theta
        SV_pol[:,2] = -sin_phi
        #the SH vectors that are perpendicular to the radial vectors and in vertical planes
        SH_pol      = np.zeros((len(thetas), 3 ) )
        SH_pol[:,0] = sin_theta
        SH_pol[:,1] = -cos_theta
        SH_pol[:,2] = 0.0
        ####################
        P_amp, SV_amp, SH_amp = np.zeros(len(thetas) ), np.zeros(len(thetas) ), np.zeros(len(thetas) )
        for wave_pol, amp in zip((P_pol, SV_pol, SH_pol), (P_amp, SV_amp, SH_amp)):
            for idx, pol in enumerate(wave_pol):
                amp[idx] = np.matmul(np.matmul(pol, self.matENU), P_pol[idx].T) # based on Eq.4.97 in Aki&Richard (2002) Quantitative Seismology (2nd edition)
        if binarization:
            P_amp = np.sign(P_amp)
        return loc_xyz, P_pol, SV_pol, SH_pol, P_amp, SV_amp, SH_amp
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
                    lighting=False, culling=False, wave_type='P'):
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
        wave_type:  'P', 'SV', 'SH', 'S', or 'all' to specify the type of wave.
        """
        if wave_type not in ('P', 'SV', 'SH', 'S', 'all'):
            raise ValueError(f"Unknown wave_type: {wave_type}. Use 'P', 'SV', 'SH', 'S', or 'all'.")
        mesh = stripy.spherical_meshes.triangulated_cube_mesh(refinement_levels=density_level)
        thetas = mesh.lons
        phis   = np.pi*0.5 - mesh.lats
        phi_min = 0.0   if (hemisphere!='lower') else 0.5*np.pi
        phi_max = np.pi if (hemisphere!='upper') else 0.5*np.pi
        #########
        idxs = np.where((phi_min<=phis) & (phis<=phi_max))
        thetas, phis = thetas[idxs], phis[idxs]
        idxs = np.where(phis<=phi_max)
        thetas, phis = thetas[idxs], phis[idxs]
        #########
        loc_xyz, P_pol, SV_pol, SH_pol, P_amp, SV_amp, SH_amp = self.__radiation(thetas, phis)
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
        elif wave_type=='all':
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
        bb_center = np.array(center)
        scale = np.abs(scale)
        for loc, pol, amp in zip(loc_xyz, dict_pol[wave_type], dict_amp[wave_type]*scale ):
            if np.abs(amp) > 0:
                start = loc*radius + bb_center
                clr = pos_color
                if wave_type == 'P':
                    clr = pos_color if amp>0 else neg_color
                    start = start if amp>0 else (start + pol*np.abs(amp))
                direction = pol * np.sign(amp)
                mesh = pv.Arrow(start=start, direction=direction, tip_length=0.3, shaft_radius=0.02, scale=np.abs(amp) )
                p.add_mesh(mesh, show_edges=False, opacity=alpha, color=clr,
                    smooth_shading=True, lighting=lighting, culling=culling)
    def plot_3d(self, p, center=(0,0,0), radius=10.0, hemisphere=None, plot_nodal=False,
                binarization=False, cmap='bwr', show_scalar_bar=True,
                alpha=1.0, culling=False, lighting=False, wave_type='P'):
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
        wave_type:  'P', 'SV', 'SH', 'S', or 'all' to specify the type of wave.
        """
        if wave_type not in ('P', 'SV', 'SH', 'S', 'all'):
            raise ValueError(f"Unknown wave_type: {wave_type}. Use 'P', 'SV', 'SH', 'S', or 'all'.")
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
        loc_xyz, P_pol, SV_pol, SH_pol, P_amp, SV_amp, SH_amp = self.__radiation(np.deg2rad(xx.flatten()), np.deg2rad(yy.flatten()), binarization=binarization)
        dict_amp = {'P': P_amp, 'SV': np.abs(SV_amp), 'SH': np.abs(SH_amp),
                    'S': np.sqrt(SV_amp**2 + SH_amp**2),
                    'all': np.sqrt(P_amp**2 + SV_amp**2 + SH_amp**2) }
        dict_grd = dict()
        for wave_type in set([wave_type, 'P']):
            scalar = dict_amp[wave_type]

            scalar *= (1.0/scalar.max() )
            scalar = scalar.reshape(xx.shape)

            # Vertical levels
            levels = [radius * 1.]

            #Create a structured grid
            grid_scalar = pv.grid_from_sph_coords(x, y, levels)
            grid_scalar.translate(center, inplace=True)

            # And fill its cell arrays with the scalar data
            grid_scalar.point_data[wave_type] = np.array(scalar).swapaxes(-2, -1).ravel("C")
            dict_grd[wave_type] = grid_scalar
        # Make a plot
        vmax = dict_grd[wave_type].point_data[wave_type].max()
        vmin = -vmax if (wave_type == 'P') else 0.0
        sargs = dict(color='k', vertical=True, interactive=False, n_colors=128, title=wave_type, outline=False, 
                     position_x=0.88, position_y=0.05, width=0.05, height=0.8, n_labels=5,
                     label_font_size=39, fmt='%.2f' )
        p.add_mesh(dict_grd[wave_type], show_edges=False, clim=[vmin, vmax], opacity=alpha, cmap=cmap,
                   smooth_shading=True, lighting=lighting, culling=culling,
                   scalar_bar_args=sargs, show_scalar_bar=show_scalar_bar)
        if plot_nodal:
            if dict_grd['P'].point_data['P'].min() < 0.0 < dict_grd['P'].point_data['P'].max():
                contours = dict_grd['P'].contour([0.0])
                p.add_mesh(contours, show_edges=True, opacity=1.0, color='k')
    #############################################################################################
    # Conversion between gcmt (M11,M22,M33,M12,M13,M23), mtUSE(3x3matrix), and mtENU(3x3matrix)
    @staticmethod
    def gcmt2mtENU(gcmt, normalize=True):
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
        : param normalize: whether to normalize the output moment tensor.

        :return mENU: a 3by3 matrix for the moment tensor in East-North-Up coordinates
        """
        M11, M22, M33, M12, M13, M23 = gcmt
        mat = np.array(((M33, -M23, M13), (-M23, M22, -M12), (M13, -M12, M11)), dtype=np.float64)
        if normalize:
            norm = np.sqrt(np.sum(mat*mat))
            if norm > 0:
                mat *= (1.0/norm)
        return mat
    @staticmethod
    def gcmt2mtUSE(gcmt, normalize=True):
        """
        Return a 3 by 3 matrix in USE coordinate for a GCMT array.
        :param gcmt: (M11, M22, M33, M12, M13, M23) - the six independent components of the moment tensor,
            where the coordinate system is 1,2,3 = Up,South,East which equals r,theta,phi.
            - Harvard/Global CMT convention, or (Mrr=M11, Mtt=M22, Mpp=M33, Mrt=M12, Mrp=M13, Mtp=M23).

            The relation to Aki and Richards x,y,z equals North,East,Down convention is as follows:
            (Mzz=M11, Mxx=M22, Myy=M33, Mxz=M12, Myz=-M13, Mxy=-M23).

        : param normalize: whether to normalize the output moment tensor matrix.

        :return mUSE: a 3by3 matrix for the moment tensor in East-North-Up coordinates
        """
        M11, M22, M33, M12, M13, M23 = gcmt
        mat = np.array(((M11, M12, M13), (M12, M22, M23), (M13, M23, M33)), dtype=np.float64)
        if normalize:
            norm = np.sqrt(np.sum(mat*mat))
            if norm > 0:
                mat *= (1.0/norm)
        return mat
    @staticmethod
    def mtUSE2gcmt(matUSE, normalize=True):
        if normalize:
            norm = np.sqrt(np.sum(matUSE*matUSE))
            if norm > 0:
                matUSE = matUSE * (1.0/norm)
        M11, M12, M13 = matUSE[0]
        M21, M22, M23 = matUSE[1]
        M31, M32, M33 = matUSE[2]
        return np.array( (M11, M22, M33, M12, M13, M23), dtype=np.float64)
    @staticmethod
    def mtUSE2mtENU(mtUSE, normalize=True):
        gcmt = beachball3d.mtUSE2gcmt(mtUSE, normalize=False)
        return beachball3d.gcmt2mtENU(gcmt, normalize=normalize)
    @staticmethod
    def mtENU2mtUSE(mtENU, normalize=True):
        if normalize:
            norm = np.sqrt(np.sum(mtENU*mtENU))
            if norm > 0:
                mtENU /= norm
        M33, M23, M13 = mtENU[0]; M23=-M23
        _,   M22, M12 = mtENU[1]; M12=-M12
        _,   _,   M11 = mtENU[2]
        matUSE = np.array(((M11, M12, M13), (M12, M22, M23), (M13, M23, M33)), dtype=np.float64)
        if normalize:
            norm = np.sqrt(np.sum(matUSE*matUSE))
            if norm > 0:
                matUSE *= (1.0/norm)
        return matUSE
    @staticmethod
    def mtENU2gcmt(mtENU, normalize=True):
        mtUSE = beachball3d.mtENU2mtUSE(mtENU, normalize=False)
        return beachball3d.mtUSE2gcmt(mtUSE, normalize=normalize)
    #############################################################################################
    # Convert between (strike, dip, slip) and mtENU(3x3 matrix)
    @staticmethod
    def strike_dip_slip2mtENU(strike_deg, dip_deg, slip_deg):
        """
        Compute moment tensor matrix in ENU coordinates (and n, d, t, p, b unit vectors) given strike, dip, and slip angles.

        The strike, dip, and slip angles are defined as in Fig 4.2-2 in Stein and Wysession (2003).
        Also, please note here we use x,y,z equals East,North,Up.

        :param strike_deg: strike measured clockwise from North (in degrees)
        :param dip_deg:    dip angle (in degrees, must be within 0-90 degree)
        :param slip_deg:   slip angle (in degrees)

        :return: mtENU,(n, d, t, b, p)
            mtENU:           a 3x3 matrix for moment tensor in ENU coordinates
            (n, d, t, b, p): unit vectors for normal, slip, T (minimal compressive),
                             B (null), and P (maximal compressive) directions in ENU coordinates
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
        vec_d = cos_lamda*x1 + sin_lamda*(cos_delta*x2 + sin_delta*x3) #the slip vector, d
        #
        vec_b = np.cross(vec_n, vec_d) # the null vector, b
        vec_p = (vec_n-vec_d)*(np.sqrt(2)*0.5) #the maximal compressive direction, P
        vec_t = (vec_n+vec_d)*(np.sqrt(2)*0.5) #the minimal compressive direction, T
        # Now, compute the moment tensor matrix
        moment_mat = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(i, 3):
                moment_mat[i,j] = vec_n[i]*vec_d[j] + vec_n[j]*vec_d[i]
                moment_mat[j,i] = moment_mat[i,j]
        moment_mat *= (0.5*np.sqrt(2.)) # to normalize
        return moment_mat, (vec_n, vec_d, vec_t, vec_b, vec_p)
    @staticmethod
    def mtENU2strike_dip_slip(mtENU):
        """
        Compute the strike, dip, and slip angles (and n, d, t, p, b unit vectors) given a moment tensor.

        :param moment_mat: a 3by3 matrix for moment tensor in ENU coordinates

        :return: (mtTBP, deviatoric_mtTBP, iso_mt), ((strike1, dip1, slip1), (strike2, dip2, slip2))
                mtTBP:              the rotated moment tensor in the t-b-p coordinate system
                deviatoric_mtTBP:   the diagonalized deviatoric moment tensor (with isotropic enegery removed) in the t-b-p coordinate system
                iso_mt:             the isotropic part of the moment tensor (regardless of whatever coordinates)
                ((strike1, dip1, slip1), (strike2, dip2, slip2)): the strike, dip, and slip angles corresponding to the two nodal planes.
        """
        # get the deviatoric part of the moment tensor
        iso_mt        = (np.trace(mtENU)/3.0) * np.eye(3)
        deviatoric_mt = mtENU - iso_mt
        # compute the t, b, p (minimal compressive, maximal compressive, null) vectors
        L, U = eig(deviatoric_mt)
        if np.sum(np.abs(np.imag(L))) > 0.0:
            raise ValueError("Err: complex eigenvalues found for moment tensor")
        L     = np.real(L)
        idxs  = np.argsort(L)
        L     = L[idxs]
        U     = U[:, idxs]
        vec_p = U[:, 0]
        vec_t = U[:, 2]
        vec_b = np.cross(vec_p, vec_t) # the null vector, b
        # U is also the rotation matrix, so that Ut M U = L, because M = U L Ut
        mtTBP            = U.T @ mtENU @ U
        deviatoric_mtTBP = np.diag(L) #U.T @ deviatoric_moment_mat @ U
        # Now, compute n, d. In fact, there are two solutions, (n1,d1) and (n2,d2) where n1=d2, and n2=d1
        vec_n1 = (vec_t+vec_p)*(np.sqrt(2)*0.5)
        vec_d1 = (vec_t-vec_p)*(np.sqrt(2)*0.5)
        vec_n2 = vec_d1
        vec_d2 = vec_n1
        # Now, compute the strike, dip and slip angles using the obtained vec_n and vec_d
        pair_strike_dip_slip = list()
        for vec_n, vec_d in [ (vec_n1, vec_d1), (vec_n2, vec_d2) ]:
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
            this_strike, this_dip, this_slip = pair_strik_dip_slip[this_idx]
            other_strike, junk, junk = pair_strik_dip_slip[other_idx]
            if (this_dip<1e-9): # dip is zero!
                new_strike = other_strike
                new_slip = (new_strike + (this_slip-this_strike)) % 360
                pair_strik_dip_slip[this_idx] = (new_strike, 0.0, new_slip)
        return (mtTBP, deviatoric_mtTBP, iso_mt), pair_strike_dip_slip
    @staticmethod
    def getTBP(mtENU):
        """
        Get the T, B, P unit vectors (and n, d) from a moment tensor centered at (0,0,0) in ENU coordinates.

        :param mtENU: Moment tensor in ENU coordinates
        :return:  (t, b, p), (n1, d1), (n2, d2)
                t, b, p: the maximal compressive, minimal compressive, and null vectors.
                         Note, (-t, -b, -p) also works.
                n1, d1: the normal and deviatoric vectors for the first plane
                        Note, (-n1, -d1) also works.
                n2, d2: the normal and deviatoric vectors for the second plane
                        Note, (-n2, -d2) also works.
        """
        # get the deviatoric part of the moment tensor
        iso_mt        = (np.trace(mtENU)/3.0) * np.eye(3)
        deviatoric_mt = mtENU - iso_mt
        # compute the t, b, p (minimal compressive, maximal compressive, null) vectors
        L, U = eig(deviatoric_mt)
        if np.sum(np.abs(np.imag(L))) > 0.0:
            raise ValueError("Err: complex eigenvalues found for moment tensor")
        L     = np.real(L)
        idxs  = np.argsort(L)
        L     = L[idxs]
        U     = U[:, idxs]
        vec_p = U[:, 0]
        vec_t = U[:, 2]
        vec_b = np.cross(vec_p, vec_t) # the null vector, b
        vec_n1 = (vec_t+vec_p)*(np.sqrt(2)*0.5)
        vec_d1 = (vec_t-vec_p)*(np.sqrt(2)*0.5)
        vec_n2 = vec_d1
        vec_d2 = vec_n1
        return (vec_t, vec_b, vec_p), (vec_n1, vec_d1), (vec_n2, vec_d2)
    @staticmethod
    def benchmark2():
        np.set_printoptions(formatter={'float': '{:5.1f}'.format})
        #####
        strike, dip, slip = 210, 0, 120
        print('Input:')
        print('strike, dip, slip:', strike, dip, slip)
        #
        mtENU,(n, d, t, b, p)= beachball3d.strike_dip_slip2mtENU(strike, dip, slip)
        print('n, d, t, b, p:', n, d, t, b, p)
        print('mtENU:\n', mtENU)
        #
        (mtTBP, deviatoric_mtTBP, iso_mt), pair_strik_dip_slip = beachball3d.mtENU2strike_dip_slip(mtENU)
        print('\nmtTBP:\n', mtTBP)
        print('deviatoric_mtTBP:\n', deviatoric_mtTBP)
        print('iso_mt:\n', iso_mt)
        print('\nCHECK:')
        for sds in pair_strik_dip_slip:
            print('strike, dip, slip:', sds)
            mtENU,(n, d, t, b, p)= beachball3d.strike_dip_slip2mtENU(*sds)
            print('n, d, t, b, p:', n, d, t, b, p)
            print('mtENU:\n', mtENU)
    @staticmethod
    def benchmark():
        p = pv.Plotter(notebook=0, shape=(1, 1), border=False, window_size=(1700, 1000) )
        p.set_background('white')
        ######################################################
        # plot surface
        mesh = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=90, j_size=30, i_resolution=90, j_resolution=30)
        mesh.clear_point_data()#not useful here
        p.add_mesh(mesh, show_edges=True, opacity=0.1, smooth_shading=True, lighting=False, culling=False)
        ######################################################
        # plot East, South, Down vectors
        start = np.array((-45, -15, 0))
        for v, label in zip(( (1, 0, 0), (0, 1, 0), (0, 0, 1) ), 'ENU'):
            v = np.array(v)
            mesh = pv.Arrow(start=start, direction=v, tip_length=0.3, shaft_resolution=30, shaft_radius=0.01, scale=3)
            p.add_mesh(mesh, show_edges=True,  opacity=1.0, color='k', smooth_shading=True, lighting=True, culling=False, )
            label = pv.Label(label, position=start+v*3)
            p.add_actor(label)
        ######################################################
        # plot beachball
        #mt = [0.91, -0.89, -0.02, 1.78, -1.55, 0.47]
        #mt = [0, 0, -0.1, 1, 0, 0] # (Mrr=M11, Mtt=M22, Mpp=M33, Mrt=M12, Mrp=M13, Mtp=M23).
        mt = [0, 0, 0, 0, -1, 0]
        #mt = [0, 0, 0, 0, 1, 1]
        bb = beachball3d(mt)
        cmap = ListedColormap(('#444444', '#eeeeee'))
        bb.plot_3d(p,     center=(0,0,0), radius=10.0, hemisphere='None', cmap='Blues', plot_nodal=True, show_scalar_bar=False, wave_type='SV')
        bb.plot_3d_vec(p, center=(0,0,0), radius=10.0, hemisphere=None, alpha=1.0, lighting=False, culling=False, scale=3, wave_type='SV')
        bb.plot_3d(p,     center=(30,0,0), radius=10.0, hemisphere='None', cmap='Blues', plot_nodal=True, show_scalar_bar=False, wave_type='S')
        bb.plot_3d_vec(p, center=(30,0,0), radius=10.0, hemisphere=None, alpha=1.0, lighting=False, culling=False, scale=3, wave_type='S')
        # plot P wave radiations
        bb.plot_3d(p,     center=(-30,0,0), radius=10.0, hemisphere='None', cmap='RdBu_r', plot_nodal=True, show_scalar_bar=True)
        bb.plot_3d_vec(p, center=(-30,0,0), radius=10.0, hemisphere=None, alpha=1.0, lighting=False, culling=False, scale=3)
        ##bb.plot_3d_vcone(p, apex=(0,0,0), cones=[(3.0, 15, '#CA3C33', 0.3), (3.1, 15, '#407AA2', 0.3)])
        p.camera_position=( (0, -90, 60), (0, 0, 0), (0, 0.5, 0.8) )
        p.show()
        print(p.camera_position)
if __name__ == '__main__':
    Scene3D.benchmark()
    sys.exit(0)
    beachball3d.benchmark()
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
    bb = beachball3d(mt)
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
