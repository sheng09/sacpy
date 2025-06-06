#!/usr/bin/env python3
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
from matplotlib.pyplot import figure
from numpy.core.numeric import extend_all
from h5py import File as h5_File
import matplotlib.pyplot as plt
from getopt import getopt
from sys import exit, argv
import numpy as np
from sacpy.sac import c_wrt_sac, c_mk_sachdr_time
from sacpy.processing import iirfilter_f32
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter

plt.rc('font', size=12) #controls default text size

def slant_stack(mat, delta, dist, dist_ref=0, slowness_range= (-4, 0), nroot=1 ):
    """
    """
    slowness = np.arange(slowness_range[0], slowness_range[1]+delta, delta )
    taup_mat = np.zeros((slowness.size, (mat.shape[1])), dtype=np.float32 )
    ###
    if nroot > 1:
        for idx in range(mat.shape[0] ):
            mat[idx] = np.sign(mat[idx]) * ( np.abs(mat[idx]) ** (1.0/nroot) )
    ###
    for ip, p in enumerate(slowness):
        for idx, d in enumerate(dist):
            dt = -p*(d-dist_ref)
            idt = int( np.round(dt/delta) )
            tmp = np.roll(mat[idx], idt)
            if idt >=0:
                tmp[:idt] = 0.0
            else:
                tmp[idt:] = 0.0
            taup_mat[ip] += tmp
    ###
    if nroot >= 1:
        for ip, p in enumerate(slowness):
            taup_mat[ip] = np.sign(taup_mat[ip]) * (np.abs(taup_mat[ip]) ** (nroot) )
    ###
    return taup_mat.transpose()

def run(h5_filename, figname, dist_range=None, cc_time_range=None, slowness_range = (-4, 0), dist_ref=0, search_range= None, search_type='pos',
        filter_setting =(None, 0.02, 0.0666), nroot= 1,
        figsize= (3, 4), title='', interpolation= 'gaussian', ylabel= True, maxpoint=True, extent=None, contour=None, vmin_scale=1.0, vmax_scale=1.0, grid=True, color='#ffbb00',
        section=None, section_out=None ):
    """
    """
    fid = h5_File(h5_filename, 'r')
    cc_t0, cc_t1, delta = fid['ccstack'].attrs['cc_t0'], fid['ccstack'].attrs['cc_t1'], fid['ccstack'].attrs['delta']
    mat = fid['ccstack'][:]
    dist = fid['dist'][:]
    dist_step = dist[1] - dist[0]
    ### filter
    btype, f1, f2 = filter_setting
    if not (btype is None):
        for irow in range(dist.size):
            iirfilter_f32(mat[irow], delta, 0, 2, f1, f2, 2, 2)
    ### cut time
    mat[:, :1000] = 0.0
    for irow in range(dist.size):
        v = mat[irow].max()
        if v > 0.0:
            mat[irow] *= (1.0/v)
    if not(cc_time_range is None):
        i1 = int( np.round((cc_time_range[0]-cc_t0)/delta) )
        i2 = int( np.round((cc_time_range[1]-cc_t0)/delta) ) + 1
        mat = mat[:, i1:i2]
        cc_t0, cc_t1 = cc_time_range
    ### cut dist
    if not(dist_range is None):
        i1 = int(round((dist_range[0]-dist[0])/dist_step))
        i2 = int(round((dist_range[1]-dist[0])/dist_step)) + 1
        mat = mat[i1:i2, :]
        dist = dist[i1:i2]
    ### slant stacking
    taup_mat = slant_stack(mat, delta, dist, dist_ref, slowness_range, nroot )
    taup_mat *= (-1.0/taup_mat.min() )
    ###
    ax = None
    axh, axv = None, None
    if section != None:
        w, h = figsize
        w = w *1.2 if 'h' in section else w
        h = h *1.2 if 'v' in section else h
        fig, ((axh, junk), (ax, axv)) = plt.subplots(2, 2, figsize= (w, h), gridspec_kw={'width_ratios': [3.7, 1], 'height_ratios': [1, 6], 'wspace': 0.0 , 'hspace': 0.0  } )
        junk.axis('off')
        if 'h' not in section:
            axh.axis('off')
        if 'v' not in section:
            axv.axis('off')
    else:
        fig, ax = plt.subplots(1, 1, figsize= figsize)
    taup_mat_copy = deepcopy( taup_mat )
    ### set those outside the search range to ZERO
    if search_range != None:
        x1, x2, y1, y2 = search_range
        ix1, ix2 = int( (x1-slowness_range[0])/delta), int( (x2-slowness_range[0])/delta )
        iy1, iy2 = int( (y1-cc_t0)/delta ), int( (y2-cc_t0)/delta )
        taup_mat[:iy1,:] = 0.0
        taup_mat[iy2:,:]  = 0.0
        taup_mat[iy1:iy2,:ix1] = 0.0
        taup_mat[iy1:iy2,ix2:] = 0.0
        pass
    ### the optimal point
    irow, icol = 0, 0
    if search_type == 'neg':
        irow, icol = np.unravel_index(taup_mat.argmin(), taup_mat.shape)
    else:
        irow, icol = np.unravel_index(taup_mat.argmax(), taup_mat.shape)
    t = cc_t0 + irow*delta
    p = slowness_range[0] + delta*icol
    ##### plot sections passing through the section
    critical_value = 0.7
    if section != None:
        if 'h' in section:
            tmp_y = taup_mat_copy[irow, :]
            tmp_x = np.linspace(slowness_range[0], slowness_range[1], tmp_y.size)
            axh.fill_between(tmp_x, 0, tmp_y, color=color)
            axh.plot(tmp_x, tmp_y, color='k')
            axh.plot(tmp_x, tmp_y*0.0, ':', color='gray', linewidth=0.6)
        if 'v' in section:
            tmp_y = taup_mat_copy[:, icol]
            tmp_x = np.linspace(cc_t0, cc_t1, tmp_y.size)
            axv.plot(tmp_y, tmp_x, color='k')
            axv.plot(tmp_y*0.0, tmp_x, ':', color='gray', linewidth=0.6)
            if search_type == 'neg':
                junk = tmp_y<=-critical_value
                tmp_x2 = tmp_x[junk]
                tmp_y2 = tmp_y[junk]
                axv.fill_betweenx(tmp_x2, tmp_y2, -critical_value, color=color)
            else:
                junk = tmp_y>=critical_value
                tmp_x2 = tmp_x[junk]
                tmp_y2 = tmp_y[junk]
                axv.fill_betweenx(tmp_x2, tmp_y2, critical_value, color=color)

    if section_out != None:
        tmp_y = taup_mat_copy[:, icol]
        fnm = '%s_v.sac' % (section_out)
        hdr = c_mk_sachdr_time(cc_t0, delta, tmp_y.size)
        hdr.user7 = p
        hdr.user8 = t
        c_wrt_sac(fnm, tmp_y, hdr, False)
    #####
    vmin = vmin_scale * np.min(taup_mat)
    vmax = vmax_scale * np.max(taup_mat)
    ###
    ax.imshow(taup_mat_copy, extent=(slowness_range[0], slowness_range[1], cc_t0, cc_t1), origin='lower', aspect='auto', cmap='gray', interpolation=  interpolation, vmin=vmin, vmax=vmax)
    if contour != None:
        tmp = gaussian_filter(taup_mat, sigma=0.3)
        if contour < 0:
            ax.contour(tmp, [contour*abs(np.min(taup_mat)) ], colors=color, origin='lower', extent=(slowness_range[0], slowness_range[1], cc_t0, cc_t1))
        if contour > 0:
            ax.contour(tmp, [contour*abs(np.max(taup_mat)) ], colors=color, origin='lower', extent=(slowness_range[0], slowness_range[1], cc_t0, cc_t1))
    if maxpoint:
        ax.plot(slowness_range, [t, t], color=color, linewidth= 0.6, alpha= 0.8)
        ax.plot([p, p], [cc_t0, cc_t1], color=color, linewidth= 0.6, alpha= 0.8)
        ax.plot(p, t, 's', color=color, alpha= 0.8)
        ax.text(slowness_range[0], t+1, '%.1f s' % (t), color=color, fontsize=18 )
    if ylabel:
        ax.set_ylabel('Time (s)', fontsize=15)
    else:
        ax.set_yticklabels([])
    if grid:
        ax.grid(linestyle=':', color='k' )
    ax.set_xlabel('Slowness (s/$\degree$)', fontsize=15)
    if section != None:
        if 'h' in section:
            axh.set_title(title)
        else:
            ax.set_title(title)
    else:
        ax.set_title(title)
    if extent != None:
        ax.set_xlim((extent[0], extent[1]) )
        ax.set_ylim((extent[2], extent[3]) )
        if section != None:
            if 'h' in section:
                axh.set_xlim((extent[0], extent[1]) )
                axh.set_xticklabels([])
                axh.set_yticks([])
            if 'v' in section:
                axv.yaxis.tick_right()
                axv.set_ylim((extent[2], extent[3]) )
                axv.set_yticklabels([])
                axv.set_xticks([])
    ###
    plt.savefig(figname, bbox_inches = 'tight', pad_inches = 0.05, dpi=200)
    plt.close()


def plt_options(args):
    """
    """
    figsize = (6, 15)
    interpolation = None
    title = ''
    ylabel = True
    maxpoint = True
    extent = None
    contour = None
    vmin_scale, vmax_scale = 1.0, 1.0
    grid= False
    color = '#ffbb00'
    ###
    for it in args.split(','):
        opt, value = it.split('=')
        if opt == 'figsize':
            figsize = tuple( [float(it) for it in value.split('/') ] )
        elif opt == 'interpolation':
            interpolation = value
        elif opt == 'title':
            title = value
        elif opt == 'ylabel':
            ylabel = True if value == 'True' else False
        elif opt == 'maxpoint':
            maxpoint = True if value == 'True' else False
        elif opt == 'extent':
            extent = tuple( [float(it) for it in value.split('/') ] )
        elif opt == 'contour':
            contour = float(value)
        elif opt == 'vmin_scale':
            vmin_scale = abs( float(value) )
        elif opt == 'vmax_scale':
            vmax_scale = abs( float(value))
        elif opt == 'grid':
            grid = True if value == 'True' else False
        elif opt == 'color':
            color = value
    return figsize, interpolation, title, ylabel, maxpoint, extent, contour, vmin_scale, vmax_scale, grid, color

if __name__ == "__main__":
    #run(filename, figname, None)
    h5_fnm = None
    figname = None
    dist_range = None
    cc_time_range = None
    slowness_range = (-4, 0)
    dist_ref = 0
    search_range= None
    search_type= 'pos'
    nroot = 1
    #### pyplot options
    figsize = (3, 4)
    interpolation = None
    title = ''
    ylabel = True
    maxpoint = True
    extent = None
    contour = None
    vmin, vmax = None, None
    grid = True
    color = '#ffbb00'
    ####
    section = None
    section_out = None
    #### line along which to normalize
    norm_settings = (None, 'pos', (-10, 10) )
    #### lines to plot
    lines = None
    filter_setting = (None, 0.02, 0.0666)
    ####
    HMSG = """
    %s -I in.h5 -P img.png [-D 0/50] [-T 0/3000] [-S -4/-1] --Dref=10 [--search_range=-3/-2/100/250] [--search_type=pos]
        [--filter bandpass/0.02/0.0666] [--nroot 1] [--section=vh] [--section_out=prefix]
        [--plt figsize=3/4,interpolation=gaussian,title=all,maxpoint=False,extent=-4/-1/100/300,contour=-0.9,grid=False,color=#ffbb00] [-H]
    """ % argv[0]
    if len(argv) < 2:
        print(HMSG)
        exit(0)
    ####
    options, remainder = getopt(argv[1:], 'I:P:D:T:S:VHh?', ['Dref=', 'filter=', 'plt=', 'nroot=', 'section=', 'section_out=', 'search_range=', 'search_type='] )
    for opt, arg in options:
        if opt in ('-I'):
            h5_fnm = arg
        elif opt in ('-P'):
            figname = arg
        elif opt in ('-D'):
            dist_range = tuple([float(it) for it in arg.split('/') ] )
        elif opt in ('-T'):
            cc_time_range = tuple([float(it) for it in arg.split('/') ] )
        elif opt in ('-S'):
            slowness_range = tuple( [float(it) for it in arg.split('/') ] )
        elif opt in ('--Dref'):
            dist_ref = float(arg)
        elif opt in ('--filter'):
            filter_setting = arg.split('/')
            vol = {'LP':0, 'HP':1, 'BP': 2, 'BR': 3, 'lowpass':0, 'highpass':1, 'bandpass':2}
            filter_setting[0] = vol[filter_setting[0]]
            filter_setting[1] = float(filter_setting[1] )
            filter_setting[2] = float(filter_setting[2] )
            filter_setting = tuple(filter_setting)
        elif opt in ('--plt'):
            figsize, interpolation, title, ylabel, maxpoint, extent, contour, vmin_scale, vmax_scale, grid, color = plt_options(arg)
        elif opt in ('--nroot'):
            nroot = int(arg)
        elif opt in ('--section'):
            section = arg
        elif opt in ('--section_out'):
            section_out = arg
        elif opt in ('--search_range'):
            search_range = tuple( [float(it) for it in arg.split('/') ] )
        elif opt in ('--search_type'):
            search_type = arg
        else:
            print(HMSG)
            exit(0)
    ####
    ####
    run(h5_fnm, figname, dist_range, cc_time_range, slowness_range, dist_ref, search_range, search_type,
        filter_setting, nroot, figsize, title, interpolation, ylabel, maxpoint, extent, contour, vmin_scale, vmax_scale, grid, color, section, section_out)







