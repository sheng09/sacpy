#!/usr/bin/env python3

"""
This is for 1D time series data processing.

Most of functions in this module are in place computation.
That means, after calling func(xs, ...), the xs is revised,
and no return.
"""

import numpy as np
#from pyfftw.interfaces.cache import enable as pyfftw_cache_enable
from pyfftw.interfaces.numpy_fft import rfft, irfft
from numba import jit, int64, int32, float64, float32, cffi_support, cgutils, types
from numba.extending import intrinsic

import sacpy.c_src._lib_sac as module_lib_sac
cffi_support.register_module(module_lib_sac)
libsac_xapiir   = module_lib_sac.lib.xapiir
libsac_design   = module_lib_sac.lib.design
libsac_apply    = module_lib_sac.lib.apply
libsac_moving_average = module_lib_sac.lib.moving_average
ffi = module_lib_sac.ffi # interesting here that we cannot use `ffi_from_buffer = module_lib_sac.ffi.from_buffer` ??
ffi_from_buffer = ffi.from_buffer


#############################################################################################################################
# JIT in place filter with fast C codes
# !!!Note: the input must be np.array with dtype=np.float32
#############################################################################################################################
@jit(nopython=True, nogil=True)
def iirfilter_f32(xs, delta, aproto, type, f1=0.0, f2=0.0, ord=2, npass=2, trbndw=0.0, a=0.0):
    """
    Inplace iirfitler for 1D trace `xs` of numpy.ndarray(dtype=np.float32).

    xs:
    delta:
    aproto: 0 : butterworth filter
            1 : bessel filter
            2 : chebyshev type i
            3 : chebyshev type ii
    type:
            0 : low pass
            1 : high pass
            2 : band pass
            3 : band reject
    f1, f2: low- and high- cutoff frequency
    ord:   order (do not exceed 10).
    npass: 1 or 2
    trbndw, a: parameters for chebyshev filter.
    """
    ptr = ffi.from_buffer( xs )
    libsac_xapiir(ptr, xs.size, aproto, trbndw, a, ord, type, f1, f2, delta, npass)
@jit(nopython=True, nogil=True)
def iirfilter2_f32(tuple_xs, delta, aproto, type, f1=0.0, f2=0.0, ord=2, npass=2, trbndw=0.0, a=0.0):
    """
    Inplace iirfitler for a tuple of 1D numpy.ndarray(dtype=np.float32).

    tuple_xs: a tuple of 1D trace. Each trace is of numpy.ndarray(dtype=np.float32).
    """
    sn    = np.zeros(30, dtype=np.float32)
    sd    = np.zeros(30, dtype=np.float32)
    nsect = np.zeros(2, dtype=np.int32)
    ptr_sn     = ffi.from_buffer(sn)
    ptr_sd     = ffi.from_buffer(sd)
    ptr_nsects = ffi.from_buffer(nsect)
    libsac_design(ord, type, aproto, a, trbndw, f1, f2, delta, ptr_sn, ptr_sd, ptr_nsects)
    flag = False if npass == 1 else True
    for it in tuple_xs:
        ptr = ffi.from_buffer( it )
        libsac_apply(ptr, it.size, flag, ptr_sn, ptr_sd, nsect[0])

#############################################################################################################################
# JIT in place taper
#############################################################################################################################
@jit(nopython=True, nogil=True)
def taper(xs, half_size, times=1):
    """
    Inplace taper a trace `xs` in place.
    The `xs` will be revised after running.

    0 <= half_size <= len(xs)/2
    times: apply the taper for many times (default is 1).
    """
    n = half_size
    xs[0] = 0.0
    xs[-1] = 0.0
    junk = np.pi/n
    for idx in range(0, n):
        c = 0.5*( 1+np.cos(junk*(idx-n) ) )
        c = c**times
        xs[idx] *= c
        xs[-1-idx] *= c
@jit(nopython=True, nogil=True)
def taper2(tuple_xs, half_size, times=1):
    """
    Inplace taper a tuple of trace of same size in place.
    The content of `tuple_tr` elements will be revised after running.

    0 <= half_size <= len(xs)/2
    times: apply the taper for many times (default is 1).
    """
    n = half_size
    junk = np.pi/n
    c = np.array( [0.5*( 1+np.cos(junk*(idx-n) ) ) for idx in range(0, n)] )
    if times > 1:
        c = c**times
    for xs in tuple_xs:
        xs[0] = 0.0
        xs[-1] = 0.0
        for idx in range(0, n):
            xs[idx] *= c
            xs[-1-idx] *= c

#############################################################################################################################
# JIT in place detrend
#############################################################################################################################
@jit(nopython=True, nogil=True)
def rmean(xs):
    """
    Inplace remove-mean for an 1D trace `xs` of numpy.ndarray(dtype=np.float32).
    """
    xs -= np.mean(xs)
@jit(nopython=True, nogil=True)
def detrend(xs):
    """
    Inplace detrend for an 1D trace `xs` of numpy.ndarray(dtype=np.float32).
    """
    xs -= np.mean(xs)
    len = xs.size
    ymean = np.mean(xs)
    xmean = (len-1)*len*0.5/len

    tmp = np.arange(len).astype(np.float64)
    s1 = np.sum(tmp*xs)
    s2 = np.sum(tmp*tmp)

    k = (s1-len*ymean*xmean) / (s2-len*xmean*xmean)
    b = ymean - k*xmean

    tmp = tmp*k+b
    xs -= tmp

#############################################################################################################################
# JIT cut
#############################################################################################################################
@jit(nopython=True, nogil=True)
def cut(xs, delta, t0, new_t0, new_t1):
    """
    Return cutted time series.

    Cut a time series `xs` with the time window `new_t0`, `new_t1`.
    `delta` and `t0` are for the input `xs`.
    """
    i0 = int(np.ceil((new_t0-t0)/delta))
    i1 = int(np.floor((new_t1-t0)/delta))+1
    new_size = i1-i0
    new_xs = np.zeros(new_size, dtype=xs.dtype)

    if i1 > xs.size:
        i1 = xs.size
    if i0<0:
        new_xs[-i0:i1-i0] = xs[:i1]
    else:
        new_xs[0:i1-i0] = xs[i0:i1]
    new_t0 = i0*delta + t0
    return new_xs, new_t0

#############################################################################################################################
#  whitening related
#############################################################################################################################
@jit(nopython=True, nogil=True)
def moving_average_f32(xs, wdn_sz=1, scale=False):
    """
    Return moving-average of an 1D trace `xs` of numpy.ndarray(dtype=np.float32).
    The input will not be revised.
    """
    ptr = ffi.from_buffer( xs )
    ys = np.empty(xs.size, dtype=np.float32)
    ptr2 = ffi.from_buffer(ys)
    libsac_moving_average(ptr, ptr2, xs.size, wdn_sz, 0, scale)
    return ys
@jit(nopython=True, nogil=True)
def moving_average_abs_f32(xs, wdn_sz=1, scale=False):
    """
    Return abs-moving-average of an 1D trace `xs` of numpy.ndarray(dtype=np.float32).
    """
    tmp = np.abs(xs).astype(np.float32)
    return moving_average_f32(tmp, wdn_sz, scale)
@jit(nopython=True, nogil=True)
def tnorm(xs, delta, winlen, f1, f2, water_level_ratio= 1.0e-5, taper_halfsize=0):
    """
    Inplace temporal normalization of input trace `xs` (a numpy.ndarray(dtype=np.float32) object).

    delta:             sampling ime interval in sec.
    winlen:            window size in sec.
    f1, f2:            frequency band in Hz that will be used to from the weight.
    water_level_ratio: default is 1.0e-5.
    taper length:      taper size (an int) in each end after the division.
    """
    wndsize = int(np.ceil(winlen/delta) )
    wndsize = (wndsize // 2)*2 +1

    weight = np.copy(xs)
    iirfilter_f32(weight, delta, 0, 2, f1, f2, 2, 2)
    weight = moving_average_abs_f32(weight, wndsize, False)
    weight += ( np.max(weight) * water_level_ratio )

    #if True in np.isnan(weight) or True in np.isinf(weight) or np.count_nonzero(weight) == 0 :
    #    xs[:] = 0.0
    #else :
    xs /= weight

    if taper_halfsize > 0:
        taper(xs, taper_halfsize)
def fwhiten(xs, delta, winlen, water_level_ratio= 1.0e-5, taper_halfsize=0, speedup_i1= -1, speedup_i2= -1):
    """
    Inplace frequency whitening of input trace `xs` (a numpy.ndarray(dtype=np.float32) object).

    delta:             sampling time interval in sec.
    winlen:            window size in Hz.
    water_level_ratio: default is 1.0e-5.
    taper_halfsize:    taper halfsize in each end after the division.
    """
    fftsize = xs.size
    if fftsize % 2 != 0: # EVET points make faster FFT than ODD points
        fftsize = fftsize + 1
    spec = rfft(xs, fftsize)

    if speedup_i2 >= 1: ## for acceleration purpose
        spec = spec[:speedup_i2]

    df =1.0/(fftsize*delta)
    wndsize = int(winlen/df)
    wndsize = (wndsize // 2)*2 +1

    weight = np.abs(spec).astype(np.float32)
    weight = moving_average_f32(weight, wndsize, False)
    weight += (weight.max() * water_level_ratio)
    #if True in np.isnan(weight) or True in np.isinf(weight) or np.count_nonzero(weight) == 0 :
    #    spec[:] = 0
    #else:
    spec /=  weight

    xs[:] = irfft(spec, fftsize )[:xs.size]

    if taper_halfsize > 0:
        taper(xs, taper_halfsize)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from copy import deepcopy
    from sacpy.sac import c_rd_sac
    st = c_rd_sac('junk/2010/20100103_223625.a/processed/II.NNA.00.BHZ')
    ts = st.get_time_axis()
    st.truncate(10800, 32400)
    #plt.plot(ts, st.dat, label='raw', alpha=0.3)

    st.rmean()
    st.detrend()
    st.taper(0.005)
    #plt.plot(ts, st.dat, label='preproc1', alpha=0.3)

    st.write('raw.sac')
    #plt.plot(ts, st.dat, label='truncate', alpha=0.3)

    st.tnorm(128, 0.02, 0.06666, 1.e-05, 1000)
    st.write('tnorm.sac')
    #plt.plot(ts, st.dat, label='tnorm', alpha=0.3)

    st.fwhiten(0.02, 1.e-5, 1000)
    st.write('fwhiten.sac')
    #plt.plot(ts, st.dat, label='fwhiten', alpha=0.3)

    #plt.legend()
    #plt.show()