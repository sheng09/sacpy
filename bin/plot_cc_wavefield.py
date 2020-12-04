#!/usr/bin/env python3

from matplotlib.pyplot import figure
from typing import Tuple
from h5py import File as h5_File
import matplotlib.pyplot as plt
from getopt import getopt
from sys import exit, argv
import numpy as np

def run(h5_filename, figname, dist_range=None, cc_time_range=None, lines= None, 
    figsize= (6, 15), interpolation= None, title='', vmax= 1.0, axhist=True, ylabel=True, grid=False ):
    """
    """
    fid = h5_File(h5_filename, 'r')
    cc_t0, cc_t1 = fid['ccstack'].attrs['cc_t0'], fid['ccstack'].attrs['cc_t1']
    mat = fid['ccstack'][:]
    dist = fid['dist'][:]
    stack_count = fid['stack_count'][:]
    ###
    delta = (cc_t1-cc_t0)/(mat.shape[1]-1)
    if cc_time_range != None:
        i1 = int( np.round((cc_time_range[0]-cc_t0)/delta) )
        i2 = int( np.round((cc_time_range[1]-cc_t0)/delta) )
        mat = mat[:, i1:i2]
        cc_t0, cc_t1 = cc_time_range
    for irow in range(dist.size):
        v = mat[irow].max()
        if v > 0.0:
            mat[irow] *= (1.0/v)
    mat = mat.transpose()
    ###
    ax1, ax2 = None, None
    if axhist:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize= figsize, gridspec_kw={'height_ratios': [5, 1]} )
    else:
        fig, ax1 = plt.subplots(1, 1, figsize= figsize )
    ###
    ax1.imshow(mat, extent=(dist[0], dist[-1], cc_t0, cc_t1 ), aspect='auto', cmap='gray', interpolation= interpolation,
            vmin=-vmax, vmax=vmax, origin='lower')
    ###
    if lines != None:
        for d, t in lines:
            ax1.plot(d, t, '.', color='C0', alpha= 0.8)
    ###
    dist_range = (dist[0], dist[-1] ) if dist_range == None else dist_range
    ax1.set_xlim(dist_range)
    ax1.set_xlabel('Inter-receiver distance ($\degree$)')
    if ylabel:
        ax1.set_ylabel('Correlation time (s)')
    else:
        ax1.set_yticklabels([])
    if grid:
        ax1.grid(linestyle=':', color='k' )
    if cc_time_range:
        ax1.set_ylim(cc_time_range)
    ax1.set_title(title)
    ###
    if axhist:
        tmp = stack_count[stack_count>0]
        ax2.bar(dist, stack_count, align='center', color='gray', width= dist[1]-dist[0] )
        ax2.set_xlim(dist_range)
        ax2.set_ylim((0, sorted(tmp)[-2] * 1.1) )
        ax2.set_xlabel('Inter-receiver distance ($\degree$)')
        if ylabel:
            ax2.set_ylabel('Number of receiver pairs')
        else:
            ax2.set_yticklabels([])
    plt.savefig(figname, bbox_inches = 'tight', pad_inches = 0.2)
    plt.close()

def plt_options(args):
    """
    """
    figsize = (6, 15)
    interpolation = None
    title = ''
    vmax = 1.0
    axhist = True
    ylabel = True
    grid   = False
    ###
    for it in args.split(','):
        opt, value = it.split('=')
        if opt == 'figsize':
            figsize = tuple( [float(it) for it in value.split('/') ] )
        elif opt == 'interpolation':
            interpolation = value
        elif opt == 'title':
            title = value
        elif opt == 'vmax':
            vmax = float(value)
        elif opt == 'axhist':
            axhist = True if value == 'True' else False
        elif opt == 'ylabel':
            ylabel = True if value == 'True' else False
        elif opt == 'grid':
            grid = True if value == 'True' else False
    return figsize, interpolation, title, vmax, axhist, ylabel, grid

def get_lines(fnms):
    """
    """
    lines = list()
    for it in fnms.split(','):
        tmp = np.loadtxt(it, comments='#')
        lines.append( (tmp[:,0], tmp[:,1]) )
    return lines

if __name__ == "__main__":

    #run(filename, figname, None)
    h5_fnm = None
    figname = None
    dist_range = None
    cc_time_range = None
    #### pyplot options
    figsize = (6, 15)
    interpolation = None
    title = ''
    vmax = 1.0
    axhist = True
    ylabel = True
    grid = False
    #### lines to plot
    lines = None
    ####
    HMSG = """
    %s -I in.h5 -P img.png [-D 0/50] [-T 0/3000] [--lines fnm1,fnm2,fnm3] [--plt figsize=6/12,interpolation=gaussian,title=all,vmax=1.0,axhist=False,ylabel=True,grid=False] [-H]
    """ % argv[0]
    if len(argv) < 2:
        print(HMSG)
        exit(0)
    ####
    options, remainder = getopt(argv[1:], 'I:P:D:T:VHh?', ['lines=', 'plt='] )
    for opt, arg in options:
        if opt in ('-I'):
            h5_fnm = arg
        elif opt in ('-P'):
            figname = arg
        elif opt in ('-D'):
            dist_range = tuple([float(it) for it in arg.split('/') ] )
        elif opt in ('-T'):
            cc_time_range = tuple([float(it) for it in arg.split('/') ] )
        elif opt in ('--lines'):
            lines = get_lines(arg)
        elif opt in ('--plt'):
            figsize, interpolation, title, vmax, axhist, ylabel, grid = plt_options(arg)
        else:
            print(HMSG)
            exit(0)
    ####
    run(h5_fnm, figname, dist_range, cc_time_range, lines, 
            figsize, interpolation, title, vmax, axhist, ylabel, grid)

