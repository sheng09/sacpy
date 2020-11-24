#!/usr/bin/env python3
"""
Executable files for obtain sac hdr information for many files.
"""
from sacpy.sac import c_rd_sachdr_wildcard, ffi
import h5py
import numpy as np
from sys import exit, argv
from getopt import getopt
from glob import glob
from os import getcwd

def run(h5_fnm, fnm_wildcard, critical_time_window=None, info=None):
    fid = h5py.File(h5_fnm, 'w')
    if info!= None:
        fid.attrs['cmd'] = info
    ###
    buf = c_rd_sachdr_wildcard(fnm_wildcard, True, True, critical_time_window= critical_time_window)
    ###
    ev_set, st_set = set(), set()
    grp_hdrtree = fid.create_group('/hdr_tree')
    grp_hdrtree.attrs['size'] = len(buf)
    for wildcard, lst in buf:
        fnm  = [fnm.encode('utf8') for hdr, fnm in lst]
        evlo = np.array([hdr.evlo  for hdr, fnm in lst], dtype=np.float32 )
        evla = np.array([hdr.evla  for hdr, fnm in lst], dtype=np.float32 )
        evdp = np.array([hdr.evdp  for hdr, fnm in lst], dtype=np.float32 )
        mag  = np.array([hdr.mag   for hdr, fnm in lst], dtype=np.float32 )
        stlo = np.array([hdr.stlo  for hdr, fnm in lst], dtype=np.float32 )
        stla = np.array([hdr.stla  for hdr, fnm in lst], dtype=np.float32 )
        baz  = np.array([hdr.baz   for hdr, fnm in lst], dtype=np.float32 )
        az   = np.array([hdr.az    for hdr, fnm in lst], dtype=np.float32 )
        gcarc= np.array([hdr.gcarc for hdr, fnm in lst], dtype=np.float32 )
        kstnm= [ffi.string(hdr.kstnm).strip() for hdr, fnm in lst]
        knetwk= [ffi.string(hdr.knetwk).strip() for hdr, fnm in lst]
        ###
        grp = grp_hdrtree.create_group(wildcard.replace('/', '\\') )
        grp.attrs['size'] = stlo.size
        grp.create_dataset('fnm',  data=fnm )
        grp.create_dataset('evlo', data=evlo, dtype=np.float32 )
        grp.create_dataset('evla', data=evla, dtype=np.float32 )
        grp.create_dataset('evdp', data=evdp, dtype=np.float32 )
        grp.create_dataset('mag',  data=mag , dtype=np.float32 )
        grp.create_dataset('stlo', data=stlo, dtype=np.float32 )
        grp.create_dataset('stla', data=stla, dtype=np.float32 )
        grp.create_dataset('baz',  data=baz,  dtype=np.float32 )
        grp.create_dataset('az',   data=az,   dtype=np.float32 )
        grp.create_dataset('gcarc',data=gcarc,dtype=np.float32 )
        grp.create_dataset('kstnm', data=kstnm  )
        grp.create_dataset('knetwk', data=knetwk  )
        ###
        for pt in zip(evlo, evla, evdp, mag):
            ev_set.add(pt)
        for pt in zip(stlo, stla, kstnm, knetwk):
            st_set.add(pt)
        ###
    grp_ev = fid.create_group('/event')
    ev_set = sorted(ev_set)
    grp_ev.attrs['size'] = len(ev_set)
    grp_ev.create_dataset( 'evlo', data= np.array([it[0] for it in ev_set], dtype=np.float32) )
    grp_ev.create_dataset( 'evla', data= np.array([it[1] for it in ev_set], dtype=np.float32) )
    grp_ev.create_dataset( 'evdp', data= np.array([it[2] for it in ev_set], dtype=np.float32) )
    grp_ev.create_dataset( 'mag',  data= np.array([it[3] for it in ev_set], dtype=np.float32) )
    ###
    grp_st = fid.create_group('/receiver')
    st_set = sorted(st_set)
    grp_st.attrs['size'] = len(st_set )
    grp_st.create_dataset('stlo',   data= np.array([it[0] for it in st_set], dtype=np.float32) )
    grp_st.create_dataset('stla',   data= np.array([it[1] for it in st_set], dtype=np.float32) )
    grp_st.create_dataset('kstnm',  data= np.array([it[2] for it in st_set]) )
    grp_st.create_dataset('knetwk', data= np.array([it[3] for it in st_set]) )
    ###
    fid.close()

if __name__ == "__main__":
    critical_time_window = None
    ####
    outfnm= None
    fnm_wildcard = None
    HMSG = '%s -I "/path*wildcards*/fnm*wildcard*.sac"  -O junk.h5 [-T -10/30] ' % argv[0]
    ####
    if len(argv) < 3:
        print(HMSG)
        exit(0)
    ####
    options, remainder = getopt(argv[1:], 'I:O:T:hH?')
    for opt, arg in options:
        if opt in ('-I'):
            fnm_wildcard = arg
        elif opt in ('-O'):
            outfnm = arg
        elif opt in ('-T'):
            critical_time_window = [float(it) for it in arg.split('/')]
        elif opt in ('-H', '-h', '-?'):
            print(HMSG)
            exit(0)
    info = 'cd %s; ' % getcwd() + ' '.join(argv)
    run(outfnm, fnm_wildcard, critical_time_window, info)
