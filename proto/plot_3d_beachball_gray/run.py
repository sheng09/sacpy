#!/usr/bin/env python3
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

from datetime import datetime
from glob import glob
import pickle
import sys
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import matplotlib.pyplot as plt
import obspy.imaging.beachball as beachball
import sacpy.geomath as geomath
import obspy.taup as taup

mod = taup.TauPyModel('ak135')
layers = mod.model.s_mod.v_mod.layers
def first_motion_angle(evdp, slowness_s_deg, verbose=True ):
    """
    Compute the P radiation angle given the slowness and the depth of the event.
    slownss = sin(theta) / Vp
    """
    slowness_s_km = slowness_s_deg/111.19492664455873
    ang_rad = 0.0
    for l in layers:
        if l[0] <= evdp < l[1]:
            v0, v1 = l[2], l[3]
            x0, x1 = l[0], l[1]
            vp = (x1-evdp)/(x1-x0)*v0 + (evdp-x0)/(x1-x0)*v1
            ang_rad = np.arcsin(slowness_s_km * vp)
            break
    return ang_rad
def plot_cmt(ax, cmt, facecolor='#444444', width=130):
    tmp = beachball.beach(cmt, facecolor=facecolor, width=width, linewidth=0.3, axes=ax, xy=(0, 0), alpha= 1.0 )
    ax.add_collection(tmp)
def plot_P_first_motion(ax, evdp, slowness_s_deg, beachball_width=130, fill=False, **kwargs):
    ang_rad = first_motion_angle(evdp, slowness_s_deg)
    #ang_rad = np.pi*0.25
    radius  = np.sin(ang_rad) * (57*beachball_width/130) # 57 and 130 were found by testing angle=45 degree
    junk = np.linspace(0, np.pi*2.0, 360)
    xs = np.cos(junk)*radius
    ys = np.sin(junk)*radius
    ax.plot(xs, ys, **kwargs, zorder=100)
    #label = r'$\leq\!$%dkm' % evdp
    label = 'Slowness $\leq$ 5 s/$\degree$'
    ax.fill(xs, ys, **kwargs, zorder=100, label= label)
    print("event_dp: %d slowness(s/deg): %.1f normalized_radius: %.3f angle: %.3f" % (evdp, slowness_s_deg, np.sin(ang_rad), ang_rad) )



fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(16, 5), gridspec_kw={'wspace': 0.1} )

ax_lst = ax1, ax2, ax3, ax4, ax5
ax = ax_lst[0]

cbr=None
mts = [ #mt                   name           Plot-coord  colormap
        ([1, 0, -1, 0, 0, 0], 'thrust',      True ,     cbr),
        ([-1, 0, 1, 0, 0, 0], 'normal',      False,     cbr),
        ([0, -1, 1, 0, 0, 0], 'strike-slip', False,     cbr),
        ([0, 0, 0, 0, 1, 0], 'dip-slip',    False,     cbr),
        ([1, 1, 1, 0, 0, 0],  'explosive',   False,     cbr),
        ]
cmts = [ it[0] for it in mts]

for cmt, ax in zip(cmts, ax_lst):
    print(cmt)
    plot_cmt(ax, cmt)

for ax in ax_lst:
    #plot_P_first_motion(ax, 600, 5, color='C0', linewidth=0.6, alpha=0.3)
    #plot_P_first_motion(ax, 200, 5, color='C1', linewidth=0.6, alpha=0.4)
    plot_P_first_motion(ax,  25, 5, color='C3', linewidth=0.6, alpha=0.5)
legend = ax1.legend(loc=(0.92, -0.1), fontsize=9, frameon=False)
plt.setp(legend.get_title(),fontsize=9)

if False:
    ax1.text(-75,   0, 'Dilatation',  color='#555555', horizontalalignment='center', verticalalignment='center', zorder=200, fontsize=9)
    ax1.text( 75,   0, 'Dilatation',  color='#555555', horizontalalignment='center', verticalalignment='center', zorder=200, fontsize=9)
    ax1.text(  0, -61, 'Compression', color='#000000', horizontalalignment='center', verticalalignment='center', zorder=200, fontsize=9)

    ax2.text(-74.5,   0, 'Compression', color='#000000', horizontalalignment='center', verticalalignment='center', zorder=200, fontsize=9)
    ax2.text( 74.5,   0, 'Compression', color='#000000', horizontalalignment='center', verticalalignment='center', zorder=200, fontsize=9)
    ax2.text(  0, -61, 'Dilatation',  color='#555555', horizontalalignment='center', verticalalignment='center', zorder=200, fontsize=9)


    ax3.text(-74,   0, 'Compression', color='#000000', horizontalalignment='center', verticalalignment='center', zorder=200, fontsize=9)
    ax3.text( 74,   0, 'Compression', color='#000000', horizontalalignment='center', verticalalignment='center', zorder=200, fontsize=9)
    ax3.text(  0, -80, 'Dilatation',  color='#555555', horizontalalignment='center', verticalalignment='center', zorder=200, fontsize=9)
    ax3.text(  0,  80, 'Dilatation',  color='#555555', horizontalalignment='center', verticalalignment='center', zorder=200, fontsize=9)


    ax4.text(-74,   0, 'Compression', color='#000000', horizontalalignment='center', verticalalignment='center', zorder=200, fontsize=9)
    ax4.text( 74,   0, 'Dilatation',  color='#555555', horizontalalignment='center', verticalalignment='center', zorder=200, fontsize=9)


for ax in ax_lst:
    ax.set_aspect('equal')
    ax.set_xlim([-103, 103])
    ax.set_ylim([-103, 103])
    #ax.grid(True)
    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

for idx, ax, title in zip('abcde', ax_lst, ('Thrust', 'Normal', 'Strike-slip', 'Dip-slip', 'Explosive')):
    ax.set_title(title, fontsize=16)
    ax.text(0.0, 1.03, '(%c)' % idx, transform=ax.transAxes, size=16 )


plt.savefig('first_motion.pdf', bbox_inches = 'tight', pad_inches = 0.01, dpi=200)
