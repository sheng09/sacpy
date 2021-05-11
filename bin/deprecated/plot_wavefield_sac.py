#!/usr/bin/env python3
"""
Plot wavefield given many sac files
"""
import sys
import getopt
import numpy as np
import h5py
import sacpy.sac as sac
import sacpy.processing as processing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal as signal
import glob
import copy
def rd_sac(filename_temaplate, tmark, t1, t2, f1, f2, smooth=0):
    fnms = sorted(glob.glob(filename_temaplate))
    sts = [sac.rd_sac(fnm) for fnm in fnms]
    ###
    if smooth > 0:
        tmp = copy.deepcopy(sts)
        sz = len(sts)
        for idx in range(sz):
            tmp[idx].dat *= 0.0  
            for isac in range(idx-smooth, idx+smooth+1):
                if isac < 0:
                    isac = 0
                elif isac >= sz:
                    isac = sz-1
                tmp[idx].dat += sts[isac].dat
        ####
        sts = tmp
    ###
    for irow, st in enumerate(sts):
        st.bandpass(f1, f2, 2, 2)
        st.truncate(tmark, t1, t2)
        st.norm()
    ###
    nrow, ncol = len(sts), sts[0]['npts']
    mat = np.zeros((nrow, ncol), dtype=np.float32 )
    vec = np.zeros(nrow, dtype=np.int32)
    for irow, st in enumerate(sts):
        #print(irow, st['npts'], ncol, mat.shape )
        mat[irow,:] = st.dat[:ncol]
        vec[irow] = st['user3']
    return mat[:,::2], vec
def plot(figname, mat, vec, t1, t2, d1, d2, curve=None, yrange=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [5, 1]} )
    ax1.imshow(mat, aspect='auto', vmin=-0.8, vmax=0.8, cmap='gray', origin='lower', extent = [t1, t2, d1, d2], interpolation='Gaussian' )
    ax1.set_xlim([t1, t2])
    ax1.set_ylim([d1, d2])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Distance ($\degree$)')
    ###
    if curve.any():
        xs = curve[:,0]
        ys = np.abs(curve[:,1])
        ax1.plot(ys, xs, '.', color='C1', alpha= 0.6, markersize=3)
    ###
    xs = np.linspace(d1, d2, len(vec) )
    ax2.barh(xs, vec, 1, color='gray', linewidth=0)
    ax2.set_ylim([d1, d2])
    ax2.set_xlim(left=0)
    if yrange != None:
        ax2.set_xlim(yrange)
    ###
    #plt.tight_layout()
    plt.savefig(figname, bbox_inches = 'tight', pad_inches = 0.2)
    plt.close()

def main():
###
    file_temaplte = None
    img_fnm = 'junk.png'
    tmark, t1, t2 = None, None, None
    d1, d2 = None, None
    f1, f2 = None, None
    curve = None
    yrange = None
    ###
    options, remainder = getopt.getopt(sys.argv[1:], 'I:O:T:D:c:Y:', ['bandpass='] )
    for opt, arg in options:
        if opt in ('-I'):
            file_temaplte = arg
        elif opt in ('-O'):
            img_fnm = arg
        elif opt in ('-T'):
            tmark, cut_t1, cut_t2 = arg.split('/')
            t1 = float(cut_t1)
            t2 = float(cut_t2)
        elif opt in ('-D'):
            d1, d2 = arg.split('/')
            d1 = float(d1)
            d2 = float(d2)
        elif opt in ('-c'):
            fnm_curve = arg
            curve = np.loadtxt(fnm_curve)
        elif opt in ('--bandpass'):
            tmp1, tmp2 = arg.split('/')
            f1 = float(tmp1)
            f2 = float(tmp2)
        elif opt in ('-Y'):
            tmp1, tmp2 = arg.split('/')
            y1 = float(tmp1)
            y2 = float(tmp2)
            yrange = (y1, y2)
        else:
            print('invalid options: %s' % (opt) )
            print(HMSG)
            sys.exit(0)
    ###
    mat, vec = rd_sac(file_temaplte, tmark, t1, t2, f1, f2)
    plot(img_fnm, mat, vec, t1, t2, d1, d2, curve, yrange )


if __name__ == '__main__':
    HMSG =  """
    %s -I filename_template -O img.png -T tmarker/t1/t2 -D d1/d2 [--bandpass f1/f2] [-c filename] [-Y y1/y2]

    Args:
        -I:
        -O:
        -T:
        -D: the distance extent for all sac files.
        --i:
        -Y:
    """ % (sys.argv[0] )
    if len(sys.argv) < 2:
        print(HMSG)
        sys.exit(0)
    main()
