#!/usr/bin/env python3

import sacpy.sac as sac
import sacpy.geomath as geomath
import numpy as np
from glob import glob
import getopt
import os.path, os, sys

def get_meta_data(info_fnm):
    """
    Obtain delta, evlo, evla, evdp, stdp, and a list of (stlo, stla) from yspec.in file.
    """
    with open(info_fnm, 'r') as fid:
        tmp = fid.readlines()
        delta = float( tmp[37] )
        evdp = float(tmp[52] )
        evla = float(tmp[55] )
        evlo = float(tmp[58] )
        stdp = float(tmp[79] )
        tmp2 = [line.strip().split() for line in tmp[86:] ]
        stla = [float(it[0]) for it in tmp2]
        stlo = [float(it[1]) for it in tmp2]
        return delta, evlo, evla, evdp, stdp, stlo, stla

def run(infnm_wildcard, outfnm_prefnm, info_fnm, component, verbose=False):
    """
    """
    if '/' in outfnm_prefnm:
        dir_nm = '/'.join( outfnm_prefnm.split('/')[:-1] )
        if not os.path.exists(dir_nm):
            os.makedirs(dir_nm)

    delta, evlo, evla, evdp, stdp, stlo_lst, stla_lst = get_meta_data(info_fnm)
    fnmlst = sorted( glob(infnm_wildcard) )

    hdr = sac.c_mk_sachdr_time(0, delta, 0)
    hdr.evlo = evlo
    hdr.evla = evla
    hdr.evdp = evdp
    hdr.stdp = stdp
    for idx, (it_fnm, stlo, stla) in enumerate(zip(fnmlst, stlo_lst, stla_lst) ):
        mat = np.loadtxt(it_fnm)
        xz = mat[:, 1]
        xn = mat[:, 2]
        xe = mat[:, 3]
        hdr.npts = xz.size
        hdr.stlo = stlo
        hdr.stla = stla
        for it_xs, it_cmp in zip( (xz, xn, xe), 'ZNE' ):
            if it_cmp in component:
                ofnm = '%s_%04d.%s.sac' % (outfnm_prefnm, idx, it_cmp )
                if verbose:
                    print(it_fnm, '==>', ofnm)
                sac.c_wrt_sac(ofnm, it_xs, hdr, verbose, True)

if __name__ == "__main__":
    ###
    HMSG = '%s -I yspec.out.wildcard.*  -O out_prename --info yspec.in [--cmp [ZNE]] [-V] [-H] ' % (sys.argv[0])
    if len(sys.argv) <= 3:
        print(HMSG)
        sys.exit(0)
    ###
    input_wildcard = None
    output_prename = 'junk_'
    info_fnm = None
    component = 'nez'
    verbose = False
    ###
    options, remainder = getopt.getopt(sys.argv[1:], 'I:O:VHh', ['info=', 'cmp='] )
    for opt, arg in options:
        if opt in ('-I'): # input
            input_wildcard = arg
        elif opt in ('-O'):
            output_prename = arg
        elif opt in ('--info'):
            info_fnm = arg
        elif opt in ('--cmp'):
            component = arg
        elif opt in ('-V'):
            verbose = True
        else:
            print(HMSG)
    run(input_wildcard, output_prename, info_fnm, component, verbose)
