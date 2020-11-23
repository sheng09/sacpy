#!/usr/bin/env python3

from cffi import FFI
ffibuilder = FFI()

ffibuilder.cdef(
    '\n'.join([ it for it in open('sac_io.h', 'r') if it[0] != '#'  ] )
)

ffibuilder.set_source("_sac_io",  # name of the output C extension
"""
    #include "sac_io.h"
""",
    sources=['sac_io.c'],   # includes pi.c as additional sources
    libraries=['m'])    # on Unix, link with the math library

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)