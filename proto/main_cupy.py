
import nvtx
import numpy as np
import cupy as cp
from numpy.random import rand
from cupyx.scipy.fft import rfft, irfft
#from pyfftw.interfaces.numpy_fft import rfft, irfft
import nvtx
import time 

from numpy import deg2rad
from h5py import File as h5_File


def haversine_cupy(lon1, lat1, lon2, lat2):
    """
    Return the great circle distance (degree) between two points.
    """
    # convert decimal degrees to radians
    import cupy as cp
    from cupy import deg2rad

    lon1, lat1, lon2, lat2 = deg2rad(lon1), deg2rad(lat1), deg2rad(lon2), deg2rad(lat2)
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    s1 = cp.sin(dlat*0.5)
    s2 = cp.sin(dlon*0.5)
    a =  s1*s1 + cp.cos(lat1) * cp.cos(lat2) * s2 * s2
    c = cp.rad2deg( 2.0 * cp.arcsin(cp.sqrt(a))  )
    return c # degree

###
#  1. The input is multiple time series of the same length (npts)
#     They are stored in a matrix, each row of which is a single time series
###


ntrace = 100
#npts   = 500000 # half million
npts   = 1500
mat = cp.ones((ntrace, npts), dtype=cp.float32) # some numbers to make everything simple
longitude = rand(ntrace)*360
latitude  = (rand(ntrace)-0.5)*180

###
#  2. Prepare for computation of spectra
#     Allocate spectra buf
###
nfft = npts*2       # padding zeros
nspec = nfft//2+1   # length of a spectrum

tic_init = time.time()
spectra = cp.zeros((ntrace, nspec), dtype=cp.complex64)
toc_init = time.time()


###
#  3. Compute the spectra of the time series
###

tic_fft = time.time()

with nvtx.annotate("fft", color="red"):
    for itrace in range(ntrace):
        spectra[itrace] = rfft(mat[itrace], nfft)

toc_fft = time.time()
dur_fft = toc_fft - tic_fft


print(dur_fft)
###
#  4. Prepare the cross correlation and stacking in frequency domain
#     Allocate stacking buf
###
nstack  = 181 # from 0 to 179 degree
stack_spectra = cp.zeros((nstack, nspec), dtype=cp.complex64)
stack_hist    = cp.zeros(nstack, dtype=cp.int32)

#  5. Compute the cross-correlation and stacking in frequency domain
###
with nvtx.annotate("crr", color="blue"):
    for i1 in range(ntrace):
        lon1, lat1 = longitude[i1], latitude[i1]
        s1 = spectra[i1]
        for i2 in range(i1, ntrace):
            lon2, lat2 = longitude[i2], latitude[i2]
            s2 = spectra[i2]
            inter_distance = haversine_cupy(lon1, lat1, lon2, lat2)
            istack = int(cp.round(inter_distance) )

            stack_spectra[istack] += s1*s2.conj()
            stack_hist[istack] = stack_hist[istack] + 1

###
#  6. Inverted FFT
###
cc_npts = nfft
stack = cp.zeros((nstack, cc_npts), dtype=cp.float32)

with nvtx.annotate("ifft", color="yellow"):
    for irow in range(nstack):
        stack[irow] = irfft(stack_spectra[irow], cc_npts)


###
#  7. Post processing
###
rollsize = npts-1
for irow in range(nstack):
    stack[irow] = cp.roll(stack[irow], rollsize)
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

h5_fnm = "out_gpu.h5"

stack_cpu =cp.asnumpy(stack)
stack_hist_cpu=cp.asnumpy(stack_hist)

f = h5_File(h5_fnm, 'w' )
dset1 = f.create_dataset('ccstack', data=stack_cpu )
dset2 = f.create_dataset('stack_count', data=stack_hist_cpu )
f.close()

###
#  9. simple check
###
#print(stack[0])

