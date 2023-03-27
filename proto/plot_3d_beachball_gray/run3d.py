#!/usr/bin/env python3

from sacpy.plot3dEarth import beachball_3d
import pyvista as pv
from matplotlib.colors import  ListedColormap, Colormap
from matplotlib.cm import get_cmap
import numpy as np

#neg = get_cmap('Blues')
#pos = get_cmap('Oranges')
#cbr_neg = [neg(i) for i in np.linspace(0, 0.8, 100)][::-1]
#cbr_pos = [pos(i) for i in np.linspace(0, 0.8, 100)]
#cbr = ListedColormap(np.concatenate( (cbr_neg, cbr_pos) ))
cbr = ListedColormap(('#eeeeee', '#444444'))

mts = [ #mt                   name           Plot-coord  Plot_colormap   colormap  window_width
        ([1, 0, -1, 0, 0, 0], 'thrust',      True ,      False,          cbr,      1000-100),
        ([-1, 0, 1, 0, 0, 0], 'normal',      False,      False,          cbr,      1000-100),
        ([0, -1, 1, 0, 0, 0], 'strike-slip', False,      False,          cbr,      1000-100),
        ([0, 0, 0, 0, 1, 0], 'dip-slip',     False,      False,          cbr,      1000-100),
        ([1, 1, 1, 0, 0, 0],  'explosive',   False,      False ,          cbr,      1300-100),
        ]
for mt, name, flag_plot_coord, flag_plot_cmap, cmap, window_width in mts:
    figname = '%s_3d.png' % (name)
    p = pv.Plotter(notebook=0, shape=(1, 1), border=False, window_size=(window_width, 1000) )
    p.set_background('white')
    ######
    # plot surface
    mesh = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=23, j_size=30, i_resolution=9, j_resolution=11)
    mesh.point_arrays.clear()
    p.add_mesh(mesh, show_edges=True, opacity=0.6,
               smooth_shading=True, lighting=False, culling=False)
    ######
    if flag_plot_coord:
        # plot East, South, Down
        start = (-11.5, -15, 0)
        for v in ( (1, 0, 0), (0, 1, 0), (0, 0, -1) ):
            mesh = pv.Arrow(start=start, direction=v, tip_length=0.3, shaft_resolution=30, shaft_radius=0.01, scale=3)
            p.add_mesh(mesh, show_edges=True,  opacity=1.0, color='k',
                            smooth_shading=True, lighting=True, culling=False)
    ######
    # plot beachball
    bb = beachball_3d(mt)
    #cmap = ListedColormap(('#444444', '#eeeeee'))
    #cmap = 'RdBu_r'
    bb.plot_3d(p,     center=(0,0,0), radius=10.0, hemisphere='None', cmap=cmap, plot_nodal=True, show_scalar_bar=flag_plot_cmap)
    bb.plot_3d_vec(p, center=(0,0,0), radius=10.0, hemisphere=None, alpha=1.0, lighting=False, culling=False, scale=4)
    #bb.plot_3d_vcone(p, apex=(0,0,0), cones=[(3.0, 15, '#CA3C33', 0.3), (3.1, 15, '#407AA2', 0.3)])
    bb.plot_3d_vcone(p, apex=(0,0,0), cones=[(np.pi-0.297, 15, '#CA3C33', 0.3),] )
    p.camera_position=( (0, -55, -30), (0, 0, 0), (0, 0, 1) )
    p.show(screenshot=figname, interactive=False, auto_close=True)
    #print(p.camera_position)