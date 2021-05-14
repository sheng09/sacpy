#!/usr/bin/env python3

from cffi import FFI
ffibuilder = FFI()

tmp1 = '\n'.join([ it for it in open('sac_io.h', 'r') if it[0] != '#'  ] )
tmp2 = '\n'.join([ it for it in open('signal_proc.h', 'r') if it[0] != '#'  ] )
tmp = '\n'.join([tmp1, tmp2])
ffibuilder.cdef(tmp)

ffibuilder.set_source("_lib_sac",  # name of the output C extension
"""
    #include "sac_io.h"
    #include "signal_proc.h"
""",
    sources=['sac_io.c', 'signal_proc.c'],
    libraries=['m'])    # on Unix, link with the math library



if __name__ == "__main__":
    ffibuilder.compile(verbose=True)