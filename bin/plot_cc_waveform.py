#!/usr/bin/env python3

from sacpy.processing import max_amplitude_timeseries, taper
from h5py import File as h5_File
import matplotlib.pyplot as plt
from getopt import getopt
from sys import exit, argv
import numpy as np

def run(h5_filename, figname, dist_range=None, cc_time_range=None, lines= None,
        norm_settings = (None, 'pos', (-20, 20) ),
        figsize= (6, 15), interpolation= None, title='', vmax= 1.0, axhist=True, ylabel=True, grid=False, baseline=False ):
    """
    """
    fid = h5_File(h5_filename, 'r')
    cc_t0, cc_t1, delta = fid['ccstack'].attrs['cc_t0'], fid['ccstack'].attrs['cc_t1'], fid['ccstack'].attrs['delta']
    mat = fid['ccstack'][:]
    dist = fid['dist'][:]
    stack_count = fid['stack_count'][:]
    dist_step = dist[1] - dist[0]
    vmax *= -dist_step
    ### Cut all the time series
    if cc_time_range != None:
        i1 = int( np.round((cc_time_range[0]-cc_t0)/delta) )
        i2 = int( np.round((cc_time_range[1]-cc_t0)/delta) )
        mat = mat[:, i1:i2]
        cc_t0, cc_t1 = cc_time_range
    for irow in range(dist.size):
        v = mat[irow].max()
        if v > 0.0:
            mat[irow] *= (1.0/v)
    ### Init the axis
    ax1, ax2 = None, None
    if axhist:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize= figsize, gridspec_kw={'height_ratios': [5, 1] } )
    else:
        fig, ax1 = plt.subplots(1, 1, figsize= figsize )
    ### normalize the waveform is necessary
    if not (norm_settings[0] is None):
        (xs, ts), method, search_window, outfnm = norm_settings
        tpos, tneg = np.zeros(dist.size), np.zeros(dist.size)
        if method in ('pos', 'both', 'all'):
            for irow in range(dist.size):
                tref = ts[irow]
                idx, t_phase, amp_phase= max_amplitude_timeseries(mat[irow], cc_t0, delta, tref, search_window, 1)
                ax1.plot(vmax*amp_phase+dist[irow], t_phase, '.', color='C1')
                tpos[irow] = t_phase
        if method in ('neg', 'both', 'all'):
            for irow in range(dist.size):
                tref = ts[irow]
                idx, t_phase, amp_phase= max_amplitude_timeseries(mat[irow], cc_t0, delta, tref, search_window, -1)
                ax1.plot(vmax*amp_phase+dist[irow], t_phase, '.', color='C2')
                tneg[irow] = t_phase
        with open(outfnm, 'w') as fid:
            for(d, tp, tn) in zip(dist, tpos, tneg):
                print('%.1f %.3f %.3f' % (d, tp, tn), file=fid)
    ### plot the baseline (optional)
    time_axis = np.arange(mat.shape[1] ) * delta + cc_t0
    if baseline:
        for it in dist:
            ax1.plot([it, it], [cc_t0, cc_t1], '--', color='gray', linewidth= 0.5 )
    ### plot the waveforms
    for irow in range(dist.size):
        ys = mat[irow]*vmax + dist[irow]
        ax1.plot(ys, time_axis, 'k', linewidth=0.7 )
    ### plot the x-t lines
    if lines != None:
        for d, t in lines:
            ax1.plot(d, t, '.', color='C0', alpha= 0.6)
    ### adjust ax1 for waveforms
    dist_range = (dist[0]+vmax, dist[-1]-vmax ) if dist_range == None else dist_range
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
    ### plot the histogram (optional)
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
    ### done
    plt.savefig(figname, bbox_inches = 'tight', pad_inches = 0.2)
    plt.close()

def plt_options(args):
    """
    """
    figsize = (12, 15)
    interpolation = None
    title = ''
    vmax = 0.5
    axhist = True
    ylabel = True
    grid   = False
    baseline = False
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
        elif opt == 'baseline':
            baseline = True if value == 'True' else False
    return figsize, interpolation, title, vmax, axhist, ylabel, grid, baseline

def get_lines(fnms):
    """
    """
    lines = list()
    for it in fnms.split(','):
        tmp = np.loadtxt(it, comments='#')
        lines.append( (tmp[:,0], tmp[:,1]) )
    return lines

def get_norm_methods(arg):
    """
    args: --norm fnm=test.txt,method=pos,window=-10/10
    """
    x_t = None
    method = 'pos'
    window = (-10, 10)
    outfnm = 'junk.txt'
    for it in  arg.split(','):
        opt, value = it.split('=')
        if opt == 'fnm':
            tmp = np.loadtxt(value, comments='#')
            x_t = (tmp[:,0], tmp[:,1])
        elif opt == 'method':
            method = value
        elif opt == 'window':
            window = tuple( [float(it) for it in value.split('/') ] )
        elif opt == 'outfnm':
            outfnm = value
    return x_t, method, window, outfnm

if __name__ == "__main__":

    #run(filename, figname, None)
    h5_fnm = None
    figname = None
    dist_range = None
    cc_time_range = None
    #### pyplot options
    figsize = (16, 15)
    interpolation = None
    title = ''
    vmax = 1.0
    axhist = True
    ylabel = True
    grid = False
    baseline = False
    #### line along which to normalize
    norm_settings = (None, 'pos', (-10, 10) )
    #### lines to plot
    lines = None
    ####
    HMSG = """
    %s -I in.h5 -P img.png [-D 0/50] [-T 0/3000] [--norm fnm=in.txt,method=pos,outfnm=o.txt] [--lines fnm1,fnm2,fnm3]
        [--plt figsize=16/12,title=all,vmax=1.0,axhist=False,ylabel=True,grid=False] [-H]
    """ % argv[0]
    if len(argv) < 2:
        print(HMSG)
        exit(0)
    ####
    options, remainder = getopt(argv[1:], 'I:P:D:T:VHh?', ['norm=', 'search=', 'lines=', 'plt='] )
    for opt, arg in options:
        if opt in ('-I'):
            h5_fnm = arg
        elif opt in ('-P'):
            figname = arg
        elif opt in ('-D'):
            dist_range = tuple([float(it) for it in arg.split('/') ] )
        elif opt in ('-T'):
            cc_time_range = tuple([float(it) for it in arg.split('/') ] )
        elif opt in ('--norm', '--search'):
            norm_settings = get_norm_methods(arg)
        elif opt in ('--lines'):
            lines = get_lines(arg)
        elif opt in ('--plt'):
            figsize, interpolation, title, vmax, axhist, ylabel, grid, baseline = plt_options(arg)
        else:
            print(HMSG)
            exit(0)
    ####
    run(h5_fnm, figname, dist_range, cc_time_range, lines, 
            norm_settings,
            figsize, interpolation, title, vmax, axhist, ylabel, grid, baseline)

