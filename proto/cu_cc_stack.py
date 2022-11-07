#!/usr/bin/env python3

import numpy as np
from numpy.random import rand
from pyfftw.interfaces.numpy_fft import rfft, irfft
from numpy import deg2rad
from h5py import File as h5_File

def haversine(lon1, lat1, lon2, lat2):
    """
    Return the great circle distance (degree) between two points.
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = deg2rad(lon1), deg2rad(lat1), deg2rad(lon2), deg2rad(lat2)
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    s1 = np.sin(dlat*0.5)
    s2 = np.sin(dlon*0.5)
    a =  s1*s1 + np.cos(lat1) * np.cos(lat2) * s2 * s2
    c = np.rad2deg( 2.0 * np.arcsin(np.sqrt(a))  )
    return c # degree

###
#  1. The input is multiple time series of the same length (npts)
#     They are stored in a matrix, each row of which is a single time series
###

ntrace = 16
#npts   = 500000 # half million
npts   = 1000
mat = np.ones((ntrace, npts), dtype=np.float32) # some numbers to make everything simple
longitude = rand(ntrace)*360
latitude  = (rand(ntrace)-0.5)*180

###
#  2. Prepare for computation of spectra
#     Allocate spectra buf
###
nfft = npts*2       # padding zeros
nspec = nfft//2+1   # length of a spectrum
spectra = np.zeros((ntrace, nspec), dtype=np.complex64)

###
#  3. Compute the spectra of the time series
###
for itrace in range(ntrace):
    spectra[itrace] = rfft(mat[itrace], nfft)

###
#  4. Prepare the cross correlation and stacking in frequency domain
#     Allocate stacking buf
###
nstack  = 180 # from 0 to 179 degree
stack_spectra = np.zeros((nstack, nspec), dtype=np.complex64)
stack_hist    = np.zeros(nstack, dtype=np.int32)

###
#  5. Compute the cross-correlation and stacking in frequency domain
###
for i1 in range(ntrace):
    lon1, lat1 = longitude[i1], latitude[i1]
    s1 = spectra[i1]
    for i2 in range(i1, ntrace):
        lon2, lat2 = longitude[i2], latitude[i2]
        s2 = spectra[i2]
        inter_distance = haversine(lon1, lat1, lon2, lat2)
        istack = int(round(inter_distance) )

        stack_spectra[istack] += s1*s2.conj()
        stack_hist[istack] = stack_hist[istack] + 1

###
#  6. Inverted FFT
###
cc_npts = nfft
stack = np.zeros((nstack, cc_npts), dtype=np.float32)
for irow in range(nstack):
    stack[irow] = irfft(stack_spectra[irow], cc_npts)


###
#  7. Post processing
###
rollsize = npts-1
for irow in range(nstack):
    stack[irow] = np.roll(stack[irow], rollsize)
stack = stack[:, :-1]
stack += stack[:, ::-1]
stack = stack[:,rollsize:]
stack *= 0.5

for irow in range(nstack):
    if stack_hist[irow] > 0:
        stack[irow] *= (1.0/stack_hist[irow])

###
#  8. output
###

h5_fnm = "out.h5"
f = h5_File(h5_fnm, 'w' )
dset1 = f.create_dataset('ccstack', data=stack )
dset2 = f.create_dataset('stack_count', data=stack_hist )
f.close()

###
#  9. simple check
###
#print(stack[0])