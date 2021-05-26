#!/usr/bin/env python3
"""
Executable files for group recordings from event-tree style to rcv-tree style.
Both the input and output are hdf5 format files.
"""
import numpy as np
from glob import glob
import sys
import getopt
import os.path, os
from h5py import File as h5_File
from h5py._hl.selections import select

def main(fnm_wildcard, out_prefix, networks=None, stations= None, merge_loc=False, info='', verbose=False):
    """
    """
    fnms = sorted(glob(fnm_wildcard) ) # in hdf5 formate
    ev_tree = get_ev_tree(fnms, networks, stations, merge_loc, verbose)
    #print(ev_tree)
    rcv_tree =evtree_to_rcvtree(ev_tree)
    for rcvnm, v in rcv_tree.items():
        run_single_rcv(rcvnm, v, out_prefix, info, verbose)

def get_ev_tree(fnms, networks=None, stations= None, merge_loc=False, verbose=False):
    """
    Get the event-receiver trees for the selected networks.
    """
    if verbose:
        print('Forming event-receiver trees...', flush=True)
    ev_tree = dict()
    for it in fnms:
        fid = h5_File(it, 'r')
        selected_idx = set(range(fid.attrs['nfile'] ) )

        if networks != None:
            tmp = [it.decode('utf8').strip() for it in fid['hdr/knetwk'][:] ] # select networks
            tmp_idx = [idx for idx, v in enumerate(tmp) if v in networks ]
            selected_idx.intersection_update(tmp_idx)

        if stations != None:
            tmp = [it.decode('utf8').strip() for it in fid['hdr/kstnm'][:] ] # select networks
            tmp_idx = [idx for idx, v in enumerate(tmp) if v in stations ]
            selected_idx.intersection_update(tmp_idx)

        selected_idx = sorted(selected_idx)
        knetwk = [it.decode('utf8').strip() for it in fid['hdr/knetwk'][selected_idx] ]
        kstnm  = [it.decode('utf8').strip() for it in fid['hdr/kstnm'][selected_idx] ]
        LL     = [it.decode('utf8').strip() for it in fid['LL'][selected_idx]]
        stlo   = fid['hdr/stlo'][selected_idx]
        stla   = fid['hdr/stla'][selected_idx]

        rcvnm = None
        tmp_dict = dict()
        for i1, i2, i3, i4, lo, la in zip(selected_idx, knetwk, kstnm, LL, stlo, stla):
            if merge_loc == False:
                rcvnm = '%s.%s.%s' % (i2, i3, i4)
            elif merge_loc == True:
                rcvnm = '%s.%s' % (i2, i3)
            elif merge_loc == i4: # force to select a location id
                rcvnm = '%s.%s.%s' % (i2, i3, i4)
            else:
                continue

            if rcvnm not in tmp_dict:
                tmp_dict[rcvnm] = (i1, lo, la)
            else:
                tmplo, tmpla = tmp_dict[rcvnm][1:]
                if abs(tmplo-lo) > 1.e-3 or abs(tmpla-la) > 1.0e-3:
                    print('Warning: more than one receivers have same knetwk and kstnm but are at different locations.', flush=True)
        ev_tree[it] = tmp_dict

        fid.close()
    return ev_tree
def evtree_to_rcvtree(ev_tree):
    """
    """
    if verbose:
        print('Converting to receiver-event trees...', flush=True)
    rcv_tree = dict()
    for evfnm, vol in ev_tree.items():
        for rcvnm, (idx, stlo, stla) in vol.items():
            if rcvnm not in rcv_tree:
                rcv_tree[rcvnm] = dict()
            tmp = rcv_tree[rcvnm]
            tmp[evfnm] = (idx, stlo, stla)
    return rcv_tree
def run_single_rcv(rcvnm, vol, out_prefix, info, verbose=False):
    fnms = sorted(vol.keys() )
    if len(fnms) < 1:
        return
    ofnm = '%s%s.h5' % (out_prefix, rcvnm)

    if verbose:
        print(rcvnm, ofnm)
        #print('\n'.join(['    %s %d' % (i1, i2) for i1, (i2, i3, i4) in vol.items()]) )

    ohdr_vol = dict() # the buf to store the selected data hdrs
    LL_vol  = list()
    fnm_vol = list()
    mat_buf = list()
    fid = h5_File(fnms[0], 'r')
    grp_hdr = fid['hdr']
    for key in grp_hdr:
        dtype = grp_hdr[key].dtype
        ohdr_vol[key] = (list(), dtype)
    fid.close()

    for fnm, (idx, stlo, stla) in vol.items():
        fid = h5_File(fnm, 'r')
        hdr = fid['hdr']
        for key in hdr:
            ohdr_vol[key][0].append(hdr[key][idx] )
        LL_vol.append(fid['LL'][idx] )
        fnm_vol.append(fid['filename'][idx])
        mat_buf.append(fid['dat'][idx][:])
        if verbose:
            print('    %s %d' % (fnm, idx) )
        fid.close()

    ofid = h5_File(ofnm, 'w')

    ohdr_grp = ofid.create_group('hdr')
    for key, (data, dtype) in ohdr_vol.items():
        ohdr_grp.create_dataset(key, data=data, dtype=dtype)
    ofid.create_dataset('LL',       data=LL_vol)
    ofid.create_dataset('filename', data=fnm_vol)

    nfile = len(mat_buf)
    npts = np.max([it.size for it in mat_buf] )
    ofid.create_dataset('dat', (nfile, npts), dtype=np.float32)
    mat = ofid['dat'] # the place to store time series
    for idx, v in enumerate(mat_buf):
        mat[idx, :v.size] = v

    ofid.attrs['nfile'] = nfile
    ofid.attrs['info'] = info

    ofid.close()


if __name__ == "__main__":
    fnm_wildcard = ''
    out_prefix = 'junk'
    networks = None   # None means everything
    stations = None   # None means everything
    merge_loc = False
    info = ''
    verbose = False
    ######################
    HMSG = """Regroup event-tree style recordings into receiver-tree style.

    %s  -I "in*/*.h5"  --networks II,IU --stations AAK,PAG -O out_filename_prefix  --merge_loc True -V

    -I  : filename wildcard for event-tree style hdf5 files.
          Usually, those files are downloaded using obspyDMT.
    --networks : selected networks, separated by comma.
    --stations:  selected stations, separated by comma.
    -O  : output directory.
    -V  : verbose.

E.g.,
    %s -I "in*/*.h5" --networks II,IU --stations AAK,PAG  -O out_filename_prefix [--info information] [-V]

    """

    if len(sys.argv) <= 1:
        print(HMSG % (sys.argv[0], sys.argv[0]), flush=True)
        sys.exit(0)
    ######################
    ######################
    options, remainder = getopt.getopt(sys.argv[1:], 'I:O:V', ['networks=', 'stations=', 'merge_loc=', 'info='])
    for opt, arg in options:
        if opt in ('-I'):
            fnm_wildcard = arg
        elif opt in ('--networks'):
            networks = tuple(arg.split(',') )
        elif opt in ('--stations'):
            stations = tuple(arg.split(',') )
        elif opt in ('--merge_loc'):
            if arg == 'True':
                merge_loc = True
            elif arg == 'False':
                merge_loc = False
            else:
                merge_loc = arg
        elif opt in ('--info'):
            info = arg
        elif opt in ('-O'):
            out_prefix = arg
        elif opt in ('-V'):
            verbose = True
    ######
    main(fnm_wildcard, out_prefix, networks, stations, merge_loc, info, verbose)


