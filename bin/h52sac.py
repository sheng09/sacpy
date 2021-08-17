#!/usr/bin/env python3
from sacpy.sac import hdf52sac
from glob import glob
import sys
from getopt import getopt
from mpi4py import MPI
import os, os.path

def run(filename, output_template, verbose):
    """
    """
    hdf52sac(filename, output_template, verbose)
    pass
HMSG = """Transform a hdf5 file into many sac files.

    %s -I filename -O output_template [-V]

    -I  : filename for the hdf5 file.
    -O  : output filename template.
            example:  'junk/(knetwk).(stnm).(LL).BHZ' (or 'rcv')
                      'junk/(nzyear)-(nzjday)-(nzhour)-(nzmin)-(nzsec)-(nzmsec).BHZ' (or 'src')
    -V  : verbose.

"""
if __name__ == "__main__":
    filename = None

    output_template_type_rcv = 'junk/(knetwk).(stnm).(LL).BHZ'
    output_template_type_src = 'junk/(nzyear)-(nzjday)-(nzhour)-(nzmin)-(nzsec)-(nzmsec).BHZ'
    output_template = output_template_type_rcv
    verbose = False

    if len(sys.argv) <=1 :
        print(HMSG % (sys.argv[0]) )
        sys.exit(0)
    options, remainder = getopt(sys.argv[1:], 'I:O:HV', ['pos=', 'hdr_only', 'info='] )
    for opt, arg in options:
        if opt in ('-H'):
            print(HMSG % (sys.argv[0]) )
            sys.exit(0)
        elif opt in ('-I'):
            filename = arg
        elif opt in ('-O'):
            if arg == 'rcv':
                output_template = output_template_type_rcv
            elif arg == 'src':
                output_template = output_template_type_src
            else:
                output_template = arg
        elif opt in ('-V'):
            verbose = True
    run(filename, output_template, verbose)