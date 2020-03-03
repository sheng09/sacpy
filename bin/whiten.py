#!/usr/bin/env python3


"""
Executable files for whitening
"""

import scipy.signal as signal
import numpy as np
import pyfftw
from time import gmtime, strftime
import time
import sacpy.processing as processing
import sacpy.sac_hdf5 as sac_hdf5
import sacpy.geomath as geomath
import mpi4py
import h5py
import pyfftw
import numpy as np 
import sys
import getpass
import getopt


def whiten_run(inh5_fnm, cut_marker, cut_t1, cut_t2, twin, f1, f2, fwin, taper_ratio, out_fnm, username= 'Sheng'):
    """
    inh5_fnm: the hdf5 files that are output of `alignedSac2Hdf5`
    """
    # output file
    fid = h5py.File(out_fnm, 'a')
    t1h, t2h = int(cut_t1/3600), int(cut_t2/3600)
    key = 'whitened_%02dh-%02dh' % (t1h, t2h)
    if key in fid:
        fid.close()
        return # already ther
    # input
    h5 = h5py.File(inh5_fnm, 'r')
    mat = h5['raw_sac/data'][:]
    b   = h5['raw_sac/hdr/b'][:]
    t_ref=h5['raw_sac/hdr/%s' % (cut_marker) ][:]
    npts= h5['raw_sac/hdr/npts'][:]
    dt  = h5['raw_sac/hdr/delta'][0] # dt must be the same for all time-series
    nsac, ncol = mat.shape
    # whiten parameters
    new_npts = int( np.round((cut_t2-cut_t1)/dt) ) + 1
    sampling_rate = 1.0/dt
    df = 1.0/(new_npts*dt)
    twin_len = int(np.round(twin / dt) )
    fwin_len = int(np.round(fwin / df) )
    taper_length = int(taper_ratio * new_npts)
    #
    new_b = np.zeros(nsac, dtype=np.float32)
    new_mat = np.zeros((nsac, new_npts), dtype=np.float32 )
    for isac in range(nsac):
        i1 = int(np.round( (cut_t1 + t_ref[isac]-b[isac])/dt  ) )
        i2 = i1+new_npts
        new_b[isac] = b[isac] + i1*dt
        tmp = processing.temporal_normalization(mat[isac, i1:i2], sampling_rate, twin_len, f1, f2, taper_length = taper_length)
        tmp = processing.frequency_whiten(tmp, sampling_rate, fwin_len, taper_length = taper_length)
        new_mat[isac, :tmp.size] = tmp
    # create file
    g = fid.create_group(key)
    g.attrs['user'] = username
    g.attrs['timestamp'] = strftime("%Y-%m-%d %H:%M:%S", gmtime() )
    g.attrs['message'] = 'Whitened time-series'
    # whitening parameters
    grp_para = g.create_group('parameters')
    grp_para.attrs['twin'] = twin
    grp_para.attrs['dt']   = dt
    grp_para.attrs['twin_len'] = twin_len
    grp_para.attrs['f1'] = f1
    grp_para.attrs['f2'] = f2
    grp_para.attrs['fwin'] = fwin
    grp_para.attrs['df']   = df
    grp_para.attrs['sampling_rate'] = sampling_rate
    grp_para.attrs['fwin_len'] = fwin_len
    grp_para.attrs['taper_ratio']  = taper_ratio
    grp_para.attrs['taper_length'] = taper_length
    ## mk hdr
    grp_hdr = g.create_group('hdr')
    for key in h5['raw_sac/hdr']:
        #print(key)
        grp_hdr.create_dataset(key, data=h5['raw_sac/hdr/%s' % (key)][:] )
    grp_hdr['npts'][:] =  new_npts
    grp_hdr['b'][:] = new_b
    ## whitened data
    g.create_dataset('data', data= new_mat)
    # release
    fid.close()
    h5.close()
    del mat
    del b
    del t_ref
    del npts
    del new_b
    del new_mat

if __name__ == "__main__":
    #fnm = '/home/catfly/00-LARGE-IMPORTANT-PERMANENT-DATA/AU_dowload/01_resampled_bhz_to_h5/05_workspace_hdf5_bhz_1-9h/2000_008_16_47_20_+0000.h5'
    fnm = None
    cut_marker, cut_t1, cut_t2 = 'o', 1*3600, 9*3600
    output = None
    twin, f1, f2 = 128.0, 0.02, 0.0666666
    fwin = 0.02
    taper_ratio= 0.005
    ###
    HMSG = '%s -I in.h5 -T o/10800/32400 -O out.h5 -N 128/0.02/0.06666 -W 0.02 -P 0.005' % (sys.argv[0])
    options, remainder = getopt.getopt(sys.argv[1:], 'I:T:N:W:P:O:H')
    for opt, arg in options:
        if opt in ('-I'): # input
            fnm = arg
        elif opt in ('-T'): # time window
            cut_marker, cut_t1, cut_t2 = arg.split('/')
            cut_t1 = float(cut_t1)
            cut_t2 = float(cut_t2)
        elif opt in ('-N'): # temporal normalization
            twin, f1, f2 = [float(it) for it in arg.split('/')]
        elif opt in ('-W'): # spectral whitening
            fwin = float(arg)
        elif opt in ('-P'):# taper ratio
            taper_ratio = float(arg)
        elif opt in ('-O'):# output
            output = arg
        elif opt in ('-H'):# help
            print(HMSG)
            sys.exit(0)
        else:
            print('invalid options: %s' % (opt) )
            sys.exit(0)
    whiten_run(fnm, cut_marker, cut_t1, cut_t2, twin, f1, f2, fwin, taper_ratio, output)