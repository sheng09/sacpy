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
from sacpy.processing import max_amplitude_timeseries, filter, taper_in_place
import os, os.path
import matplotlib.ticker as mtick

def run(h5_filename, figname, dist_range=None, cc_time_range=None, lines= None, 
        filter_setting =(None, 0.02, 0.0666), adjust_time_axis = None,
        taper_sec =0., search_amp= None, norm_settings = (None, 'pos', (-10, 10) ),
        figsize= (6, 15), interpolation= None, title='', vmax= 1.0, axhist=True, yticks='all', ylabel='all', grid=False, dpi=150):
    """
    adjust_time_axis: a tupel of (sc, xc). This will change {t-x} domain into {t-sc(x-xc),x} domain. 
                      This option help flat a steep phase.
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
    ### adjust time axis
    if adjust_time_axis != None:
        npts = mat.shape[1]
        sc, xc = adjust_time_axis
        for irow, x in enumerate(dist):
            dt = sc*(x-xc)
            ndt = int(np.round(dt/delta) )
            if 0 < ndt < npts:
                mat[irow, :-ndt] = mat[irow, ndt:]
                mat[irow, -ndt:] = 0
            elif -npts < ndt < 0:
                mat[irow, -ndt:] = mat[irow, :ndt]
                mat[irow, :-ndt] = 0
    ###
    junk_t = 50 if cc_time_range == None else cc_time_range[0]
    mat[:, :int(junk_t/delta)] = 0.0
    ###
    if cc_time_range != None:
        i1 = int( np.round((cc_time_range[0]-cc_t0)/delta) )
        i2 = int( np.round((cc_time_range[1]-cc_t0)/delta) )
        mat = mat[:, i1:i2]
        cc_t0, cc_t1 = cc_time_range
    ### taper
    if taper_sec > delta:
        taper_sz = int(taper_sec / delta)
        for irow in range(dist.size):
            taper_in_place(mat[irow], taper_sz)
    ### norm
    for irow in range(dist.size):
        v = mat[irow].max()
        if v > 0.0:
            mat[irow] *= (1.0/v)
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
            if adjust_time_axis != None:
                sc, xc = adjust_time_axis
                t = t - sc*(d-xc)
            ax1.plot(d, t, '.', color='C0', alpha= 0.6, markersize=3)
    ###
    dist_range = (dist[0], dist[-1] ) if dist_range == None else dist_range
    ### search for max/min points for each trace
    if search_amp != None:
        search_max_amplitude(ax1, mat, search_amp, cc_t0, delta, adjust_time_axis)
    ###
    ax1.set_xlim(dist_range)
    if not axhist:
        ax1.set_xlabel('Inter-receiver distance X ($\degree$)')
    else:
        ax1.set_xticklabels(())
    if ylabel == 'all' or ylabel == 'cc':
        if adjust_time_axis != None:
            sc, xc = adjust_time_axis
            if xc != 0:
                if sc > 0:
                    ax1.set_ylabel(r'$T-%.1f(X-%d^\degree)$ (s)' % (sc, xc) )
                else:
                    ax1.set_ylabel(r'$T+%.1f(X-%d^\degree)$ (s)' % (-sc, xc) )
            else:
                if sc > 0:
                    ax1.set_ylabel(r'$T-%.1fX$ (s)' % (sc) )
                else:
                    ax1.set_ylabel(r'$T+%.1fX$ (s)' % (-sc) )
        else:
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
        ax2.set_xlabel('Inter-receiver distance $X$ ($\degree$)')

        #fmt = '{x:,.0f}'
        #tick = mtick.StrMethodFormatter(fmt)
        #ax2.yaxis.set_major_formatter(tick)

        #ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0) )

        ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))

        if ylabel == 'all' or ylabel == 'hist':
            ax2.set_ylabel('Number of\n receiver pairs')

        if yticks == None or yticks == 'cc':
            ax2.set_yticklabels([])
    #plt.tight_layout()
    plt.savefig(figname, bbox_inches = 'tight', pad_inches = 0.05, dpi=dpi)
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
    dpi    = 150
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
        elif opt == 'dpi':
            dpi = float(value)
    return figsize, interpolation, title, vmax, axhist, yticks, ylabel, grid, dpi

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

def search_max_amplitude(ax, mat, search_amp, cc_t0, delta, adjust_time_axis):
    """
    """
    type, hw, fnm = search_amp
    nrow, ncol = mat.shape
    if fnm == 'None': # search for global max/min
        xs = list(range(ncol))
        if 'p' in type:
            ys = [np.argmax(mat[:,icol])*delta+cc_t0 for icol in xs ]
            ax.scatter(xs, ys, 5, marker='+', color='C3')
        if 'n' in type:
            ys = [np.argmin(mat[:,icol])*delta+cc_t0 for icol in xs ]
            ax.scatter(xs, ys, 5, marker='+', color='C2')
    else:  # search for max/min given reference time and a search time window
        junk = np.loadtxt(fnm)
        ref_d, ref_t= junk[:,0].astype(np.int), junk[:,1]
        if adjust_time_axis != None:
            sc, xc = adjust_time_axis
            ref_t -= sc*(ref_d-xc)
        for key, func, marker, clr in zip( ('p', 'n'), (np.argmax, np.argmin), ('+', '+'), ('C3', 'C2') ):
            if not key in type:
                continue
            pts = list()
            for icol in range(ncol):
                try:
                    t0 = ref_t[icol]
                    i1, i2 = [int( np.round((t0-cc_t0+it)/delta)) for it in (-hw, hw)]
                    i1 = 0 if i1 < 0 else i1
                    i2 = nrow if i2 > nrow else i2
                    if i2 <= i1:
                        continue
                    pts.append( (icol, (func(mat[i1:i2,icol])+i1)*delta+cc_t0 ) )
                except:
                    pass
            xs = [it[0] for it in pts]
            ys = [it[1] for it in pts]
            ax.scatter(xs, ys, 5, marker=marker, color=clr)

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
    dpi = 100
    #### line along which to normalize
    norm_settings = (None, 'pos', (-10, 10) )
    #### lines to plot
    lines = None
    filter_setting = (None, 0.02, 0.0666)
    adjust_time_axis = None
    taper_sec = 10
    #### find max/min amplitude for each trace
    search_amp = None #('pn', 10, None)
    ####
    HMSG = """
Plot cc wavefield in hdf5 file generated by cc_stack_sac.py.
    %s -I in.h5 -P img.png [-D 0/50] [-T 0/3000] [--filter bandpass/0.02/0.0666] [--adjust_time_axis 5/0]
    [--taper_sec 100 ]  [--search_amp p,10,None]
    [--norm fnm=in.txt,method=pos,outfnm=o.txt] [--lines fnm1,fnm2,fnm3]
    [--plt figsize=6/12,interpolation=gaussian,title=all,vmax=1.0,axhist=False,yticks=all,ylabel=True,grid=False,dpi=150] [-H]

Args:
    --search_amp type,half_window_length,fnm :
            the type can be 'p', 'n', 'pn',
            the half_window_length is in seconds,
            the fnm can be a string of filename or None.

    --plt: plot options.
            figsize: width/height
            interpolation: interpolation methods of pyplot.imshow(...)
            title:
            vmax:
            axhist: True or False to turn on or off the histogram of number of receiver pairs.
            yticks: 'all', or 'cc', or 'hist'
            ylabel: 'all', or 'cc', or 'hist'
            grid:   True or False to turn on or off the grid lines.
            dpi: dpi
    """ % argv[0]
    if len(argv) < 2:
        print(HMSG)
        exit(0)
    ####
    options, remainder = getopt(argv[1:], 'I:P:D:T:VHh?', ['filter=', 'adjust_time_axis=', 'taper_sec=', 'search_amp=', 'norm=', 'search=', 'lines=', 'plt='] )
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
        elif opt in ('--adjust_time_axis'):
            adjust_time_axis = [float(it) for it in arg.split('/')]
        elif opt in ('--taper_sec'):
            taper_sec = float(arg)
        elif opt in ('--search_amp'):
            search_amp = arg.split(',')
            search_amp[1] = float(search_amp[1] )
        elif opt in ('--norm'):
            norm_settings = get_norm_methods(arg)
        elif opt in ('--lines'):
            lines = get_lines(arg)
        elif opt in ('--plt'):
            figsize, interpolation, title, vmax, axhist, yticks, ylabel, grid, dpi = plt_options(arg)
        else:
            print(HMSG)
            exit(0)
    ####
    run(h5_fnm, figname, dist_range, cc_time_range, lines, 
            filter_setting, adjust_time_axis,
            taper_sec, search_amp, norm_settings,
            figsize, interpolation, title, vmax, axhist, yticks, ylabel, grid, dpi)

