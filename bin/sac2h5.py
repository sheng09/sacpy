#!/usr/bin/env python3
from sacpy.sac import sac2hdf5
from glob import glob
import sys
from getopt import getopt
from mpi4py import MPI
import os, os.path
#sac2hdf5(fnm_wildcard, hdf5_fnm, lcalda=False, verbose=False, info='', ignore_data=False):


def run(filename_wildcard, output_prefix, pos, verbose, hdr_only):
    """
    """
    mpi_comm = MPI.COMM_WORLD.Dup()
    mpi_rank = mpi_comm.Get_rank()
    mpi_ncpu = mpi_comm.Get_size()
    ####
    tmp = '/'.join(output_prefix.split('/')[:-1] )
    if len(tmp) > 1 and mpi_rank == 0:
        if not os.path.exists(tmp):
            os.makedirs(tmp)
    ####
    tmp = '/'.join(filename_wildcard.split('/')[:pos+1] )
    sub_folders   = sorted( glob(tmp) )
    remainder     = '/'.join(filename_wildcard.split('/')[pos+1:])
    keys          = [it.split('/')[-1] for it in sub_folders]
    sub_wildcards = ['%s/%s' % (it, remainder)  for it in sub_folders]
    ####
    local_keys = keys[mpi_rank::mpi_ncpu]
    local_sub_wildcards = sub_wildcards[mpi_rank::mpi_ncpu]
    for k, it in zip(local_keys, local_sub_wildcards):
        hdf5_fnm = '%s%s.h5' % (output_prefix, k)
        if verbose and mpi_rank == 0:
            print('[iproc:%02d] %s ==> %s' % (mpi_rank, it, hdf5_fnm) )
        sac2hdf5(sorted(glob(it)), hdf5_fnm, lcalda=True, verbose=verbose, info='', ignore_data=hdr_only)

HMSG = """Transform many sac files downloaded with obspyDMT into hdf5 files.

    mpirun -np 4 %s -I 'filename_wildcard' --pos -3 -O output_prefix [-V] [--hdr_only]

    -I  : filename wildcard for sac files.
    --pos: position of the tags in `filename_wildcard` separated by `/`
           to distinguish between different sub folders. Then, a single
           hdf5 file will be created for each sub folder.
    -O  : output filename prefix
    -V  : verbose.
    [--hdr_only]: just save sac headers to hdf5 files, and ignore time series.

"""
if __name__ == "__main__":
    filename_wildcard = None
    output_prefix = 'junk'
    verbose= False
    pos = -2
    hdr_only = False

    if len(sys.argv) <=1 :
        print(HMSG % (sys.argv[0]) )
        sys.exit(0)
    options, remainder = getopt(sys.argv[1:], 'I:O:HV', ['pos=', 'hdr_only'] )
    for opt, arg in options:
        if opt in ('-H'):
            print(HMSG % (sys.argv[0]) )
            sys.exit(0)
        elif opt in ('-I'):
            filename_wildcard = arg
        elif opt in ('-O'):
            output_prefix = arg
        elif opt in ('-V'):
            verbose = True
        elif opt in ('--pos'):
            pos = int(arg)
        elif opt in ('--hdr_only'):
            hdr_only = True
    run(filename_wildcard, output_prefix, pos, verbose, hdr_only)

