#!/usr/bin/env python3
"""
Executable files for group sac files from event-tree style to rcv-tree style.
"""
from glob import glob
import sys
import getopt
import os.path, os

def main(fnm_wildcard, out_root_dir, ev_nm_pos=-3, verbose=False):
    """
    fnm_wildcard: wildcard for accessing all sac files. 
    """
    cwd = os.getcwdb().decode('utf8')
    if fnm_wildcard[0] != '/':
        fnm_wildcard = cwd + '/' + fnm_wildcard
    if out_root_dir[0] != '/':
        out_root_dir = cwd + '/' + out_root_dir
    #####
    directories   = sorted( glob('/'.join(fnm_wildcard.split('/')[:-1] ) ) )
    sac_wildcards = fnm_wildcard.split('/')[-1]
    for it in directories:
        run_single_event_obspyDMT(it + '/' + sac_wildcards, out_root_dir, ev_nm_pos, verbose )
def run_single_event_obspyDMT(fnm_wildcard, out_root_dir, ev_nm_pos=-3, verbose=False):
    """
    Obtain sacfile
    """
    ######
    event_nm = fnm_wildcard.split('/')[ev_nm_pos]
    fnms = sorted(glob(fnm_wildcard) )
    if verbose:
        print("Running %s (%d)" % (fnm_wildcard, len(fnms)) )
    ######
    for it in fnms:
        rcv_dir = out_root_dir + '/' + '.'.join( it.split('/')[-1].split('.')[:-1] )
        virtual_link = rcv_dir + '/' + event_nm + '.' + it.split('.')[-1]
        if not os.path.exists(rcv_dir):
            os.makedirs(rcv_dir)
        if os.path.exists(virtual_link):
            os.remove(virtual_link)
        #print(it, '=>', virtual_link)
        os.symlink(it, virtual_link)

if __name__ == "__main__":
    fnm_wildcard = ''
    out_root_dir = 'junk'
    ev_pos = -3
    verbose = False
    ######################
    HMSG = """Regroup event-tree style sac files into receiver-tree style.

    %s  -I "in*/*.sac" --ev_pos -3  -O out_root_dir  -V
    
    -I  : filename wildcard for event-tree style sac files. 
          Usually, those files are downloaded using obspyDMT.
    --ev_pos: position of the event number in the `filename wildcard` separated by `/`.
    -O  : output directory.
    -V  : verbose.

E.g.,
    %s -I "in*/*.sac"  --ev_pos -3 -O rcv_root 

    """

    if len(sys.argv) <= 1:
        print(HMSG % (sys.argv[0], sys.argv[0]), flush=True)
        sys.exit(0)
    ######################
    ######################
    options, remainder = getopt.getopt(sys.argv[1:], 'I:O:V', ['ev_pos='])
    for opt, arg in options:
        if opt in ('-I'):
            fnm_wildcard = arg
        elif opt in ('-O'):
            out_root_dir = arg
        elif opt in ('--ev_pos'):
            ev_pos = int(arg)
        elif opt in ('-V'):
            verbose = True
    ######
    main(fnm_wildcard, out_root_dir, ev_pos, verbose)


