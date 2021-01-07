#!/usr/bin/env python3

from matplotlib.pyplot import figure
from numpy.core.numeric import extend_all
from h5py import File as h5_File
import matplotlib.pyplot as plt
from getopt import getopt
from sys import exit, argv
import numpy as np
from sacpy.processing import max_amplitude_timeseries, filter


def slant_stack(mat, delta, dist, slowness_range= (-4, 0), nroot=1 ):
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
            dt = -p*d
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

def run(h5_filename, figname, dist_range=None, cc_time_range=None, slowness_range = (-4, 0),
        filter_setting =(None, 0.02, 0.0666), nroot= 1,
        figsize= (3, 4), title='', interpolation= 'gaussian', ylabel= True, maxpoint=True, extent=None ):
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
            mat[irow] = filter(mat[irow], 1.0/delta, btype, (f1, f2), 2, 2 )
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
    taup_mat = slant_stack(mat, delta, dist, slowness_range, nroot )
    taup_mat *= (-1.0/taup_mat.min() )
    ### the optimal point
    irow, icol = np.unravel_index(taup_mat.argmin(), taup_mat.shape)
    t = cc_t0 + irow*delta
    p = slowness_range[0] + delta*icol
    ###
    fig, ax = plt.subplots(1, 1, figsize= figsize)
    ax.imshow(taup_mat, extent=(slowness_range[0], slowness_range[1], cc_t0, cc_t1), origin='lower', aspect='auto', cmap='gray', interpolation=  interpolation)
    ax.contour(taup_mat, [-0.9], colors='#ffbb00', origin='lower', extent=(slowness_range[0], slowness_range[1], cc_t0, cc_t1))
    if maxpoint:
        ax.plot(slowness_range, [t, t], '#ffbb00', linewidth= 0.6, alpha= 0.8)
        ax.plot([p, p], [cc_t0, cc_t1], '#ffbb00', linewidth= 0.6, alpha= 0.8)
        ax.plot(p, t, '.', color='#ffbb00', alpha= 0.8)
        ax.text(slowness_range[0], t, '%.1f s' % (t), color='#ffbb00', fontsize=18 )
    if ylabel:
        ax.set_ylabel('Time (s)')
    else:
        ax.set_yticklabels([])
    ax.set_xlabel('Slowness (s/$\degree$)')
    ax.set_title(title)
    if extent != None:
        ax.set_xlim((extent[0], extent[1]) )
        ax.set_ylim((extent[2], extent[3]) )
    ###
    plt.savefig(figname, bbox_inches = 'tight', pad_inches = 0.05)
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
    return figsize, interpolation, title, ylabel, maxpoint, extent

if __name__ == "__main__":
    #run(filename, figname, None)
    h5_fnm = None
    figname = None
    dist_range = None
    cc_time_range = None
    slowness_range = (-4, 0)
    nroot = 1
    #### pyplot options
    figsize = (3, 4)
    interpolation = None
    title = ''
    ylabel = True
    maxpoint = True
    extent = None
    #### line along which to normalize
    norm_settings = (None, 'pos', (-10, 10) )
    #### lines to plot
    lines = None
    filter_setting = (None, 0.02, 0.0666)
    ####
    HMSG = """
    %s -I in.h5 -P img.png [-D 0/50] [-T 0/3000] [-S -4/0] [--filter bandpass/0.02/0.0666] [--nroot 1]
        [--plt figsize=3/4,interpolation=gaussian,title=all,maxpoint=False,extent=-4/-1/100/300] [-H]
    """ % argv[0]
    if len(argv) < 2:
        print(HMSG)
        exit(0)
    ####
    options, remainder = getopt(argv[1:], 'I:P:D:T:S:VHh?', ['filter=', 'plt=', 'nroot='] )
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
        elif opt in ('--filter'):
            filter_setting = arg.split('/')
            filter_setting[1] = float(filter_setting[1] )
            filter_setting[2] = float(filter_setting[2] )
            filter_setting = tuple(filter_setting)
        elif opt in ('--plt'):
            figsize, interpolation, title, ylabel, maxpoint, extent = plt_options(arg)
        elif opt in ('--nroot'):
            nroot = int(arg)
        else:
            print(HMSG)
            exit(0)
    ####
    ####
    run(h5_fnm, figname, dist_range, cc_time_range, slowness_range, filter_setting, nroot, figsize, title, interpolation, ylabel, maxpoint, extent )






