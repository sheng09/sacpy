#!/usr/bin/env python3
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
from matplotlib.pyplot import figure
from h5py import File as h5_File
import matplotlib.pyplot as plt
from getopt import getopt
from sys import exit, argv
import numpy as np
from sacpy.processing import max_amplitude_timeseries, filter
import os, os.path
import matplotlib.ticker as mtick

def run(h5_filename, figname, dist_range=None, cc_time_range=None, lines= None, 
        filter_setting =(None, 0.02, 0.0666), norm_settings = (None, 'pos', (-10, 10) ),
        figsize= (6, 15), interpolation= None, title='', vmax= 1.0, axhist=True, yticks='all', ylabel='all', grid=False ):
    """
    """
    fig_outdir = '/'.join( figname.split('/')[:-1] )
    if not os.path.exists(fig_outdir):
        os.makedirs(fig_outdir)
    ###
    fid = h5_File(h5_filename, 'r')
    cc_t0, cc_t1, delta = fid['ccstack'].attrs['cc_t0'], fid['ccstack'].attrs['cc_t1'], fid['ccstack'].attrs['delta']
    mat = fid['ccstack'][:]
    dist = fid['dist'][:]
    stack_count = fid['stack_count'][:]
    ### filter
    btype, f1, f2 = filter_setting
    if not (btype is None):
        for irow in range(dist.size):
            mat[irow] = filter(mat[irow], 1.0/delta, btype, (f1, f2), 2, 2 )
    ###
    junk_t = 50 if cc_time_range == None else cc_time_range[0]
    mat[:, :int(junk_t/delta)] = 0.0
    for irow in range(dist.size):
        v = mat[irow].max()
        if v > 0.0:
            mat[irow] *= (1.0/v)
    if cc_time_range != None:
        i1 = int( np.round((cc_time_range[0]-cc_t0)/delta) )
        i2 = int( np.round((cc_time_range[1]-cc_t0)/delta) )
        mat = mat[:, i1:i2]
        cc_t0, cc_t1 = cc_time_range
    ### normalize the waveform if necessary
    if not (norm_settings[0] is None):
        (xs, ts), method, search_window, outfnm = norm_settings
        tpos, tneg = np.zeros(dist.size), np.zeros(dist.size)
        if method in ('pos'):
            for irow in range(dist.size):
                tref = ts[irow]
                idx, t_phase, amp_phase= max_amplitude_timeseries(mat[irow], cc_t0, delta, tref, search_window, 1)
                mat[irow] *= (1.0/(amp_phase) )
                #ax1.plot(vmax*amp_phase+dist[irow], t_phase, '.', color='C1')
                #tpos[irow] = t_phase
        elif method in ('neg'):
            for irow in range(dist.size):
                tref = ts[irow]
                idx, t_phase, amp_phase= max_amplitude_timeseries(mat[irow], cc_t0, delta, tref, search_window, -1)
                mat[irow] *= (1.0/(amp_phase) )
                #ax1.plot(vmax*amp_phase+dist[irow], t_phase, '.', color='C2')
                #tneg[irow] = t_phase
        with open(outfnm, 'w') as fid:
            for(d, tp, tn) in zip(dist, tpos, tneg):
                print('%.1f %.3f %.3f' % (d, tp, tn), file=fid)
    ###
    mat = mat.transpose()
    ###
    ax1, ax2 = None, None
    if axhist:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize= figsize, gridspec_kw={'height_ratios': [4, 1]} )
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
    if not axhist:
        ax1.set_xlabel('Inter-receiver distance ($\degree$)')
    if ylabel == 'all' or ylabel == 'cc':
        ax1.set_ylabel('Time (s)')
    if yticks == None or yticks == 'hist':
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
        ax2.set_ylim(bottom=0 )
        ax2.set_xlabel('Inter-receiver distance ($\degree$)')

        fmt = '{x:,.0f}'
        tick = mtick.StrMethodFormatter(fmt)
        ax2.yaxis.set_major_formatter(tick)
        #ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0) )

        if ylabel == 'all' or ylabel == 'hist':
            ax2.set_ylabel('Number of\n receiver pairs')

        if yticks == None or yticks == 'cc':
            ax2.set_yticklabels([])
    #plt.tight_layout()
    plt.savefig(figname, bbox_inches = 'tight', pad_inches = 0.05)
    plt.close()

def plt_options(args):
    """
    """
    figsize = (6, 15)
    interpolation = None
    title = ''
    vmax = 1.0
    axhist = True
    yticks = None
    ylabel = None
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
        elif opt == 'yticks':
            yticks = value
            if yticks == 'True':
                yticks = 'all'
            elif yticks == 'False' or yticks == 'None':
                yticks = None
        elif opt == 'ylabel':
            ylabel = value
            if value == 'True':
                ylabel = 'all'
            elif value == 'False' or value == 'None':
                ylabel = None
        elif opt == 'grid':
            grid = True if value == 'True' else False
    return figsize, interpolation, title, vmax, axhist, yticks, ylabel, grid

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
    figsize = (6, 15)
    interpolation = None
    title = ''
    vmax = 1.0
    axhist = True
    yticks = 'all'
    ylabel = 'all'
    grid = False
    #### line along which to normalize
    norm_settings = (None, 'pos', (-10, 10) )
    #### lines to plot
    lines = None
    filter_setting = (None, 0.02, 0.0666)
    ####
    HMSG = """
Plot cc wavefield in hdf5 file generated by cc_stack_sac.py.
    %s -I in.h5 -P img.png [-D 0/50] [-T 0/3000] [--filter bandpass/0.02/0.0666]
    [--norm fnm=in.txt,method=pos,outfnm=o.txt] [--lines fnm1,fnm2,fnm3]
    [--plt figsize=6/12,interpolation=gaussian,title=all,vmax=1.0,axhist=False,yticks=all,ylabel=True,grid=False] [-H]

Args:
    --plt: plot options.
            figsize: width/height
            interpolation: interpolation methods of pyplot.imshow(...)
            title:
            vmax:
            axhist: True or False to turn on or off the histogram of number of receiver pairs.
            yticks: 'all', or 'cc', or 'hist'
            ylabel: 'all', or 'cc', or 'hist'
            grid;   True or False to turn on or off the grid lines.
    """ % argv[0]
    if len(argv) < 2:
        print(HMSG)
        exit(0)
    ####
    options, remainder = getopt(argv[1:], 'I:P:D:T:VHh?', ['filter=', 'norm=', 'search=', 'lines=', 'plt='] )
    for opt, arg in options:
        if opt in ('-I'):
            h5_fnm = arg
        elif opt in ('-P'):
            figname = arg
        elif opt in ('-D'):
            dist_range = tuple([float(it) for it in arg.split('/') ] )
        elif opt in ('-T'):
            cc_time_range = tuple([float(it) for it in arg.split('/') ] )
        elif opt in ('--filter'):
            filter_setting = arg.split('/')
            filter_setting[1] = float(filter_setting[1] )
            filter_setting[2] = float(filter_setting[2] )
            filter_setting = tuple(filter_setting)
        elif opt in ('--norm', '--search'):
            norm_settings = get_norm_methods(arg)
        elif opt in ('--lines'):
            lines = get_lines(arg)
        elif opt in ('--plt'):
            figsize, interpolation, title, vmax, axhist, yticks, ylabel, grid = plt_options(arg)
        else:
            print(HMSG)
            exit(0)
    ####
    run(h5_fnm, figname, dist_range, cc_time_range, lines, 
            filter_setting, norm_settings,
            figsize, interpolation, title, vmax, axhist, yticks, ylabel, grid)

