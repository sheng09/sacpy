#!/usr/bin/env python3

"""
This is for 1D time series data processing.

Most of functions in this module are in place computation.
That means, after calling func(xs, ...), the xs is revised,
and no return.

Inplace IIR filter
-------------------------

>>> import numpy as np
>>> xs = np.random.random(10000)-0.5
>>> delta = 0.1 # sampling time interval
>>> btype, f1, f2, ord, npass = 2, 0.2, 2.0, 2, 2
>>> iirfilter_f32(xs, delta, 0, btype, f1, f2, ord, npass)
>>>
>>> ys = np.random.random(10000)-0.5
>>> zs = np.random.random(10000)-0.5
>>> iirfilter2_f32((xs, ys, zs), delta, 0, btype, f1, f2, ord, npass)
>>>

Inplace taper
-------------------------

>>> halfsize, times = 30, 2
>>> taper(xs, halfsize, times) # will apply the taper for 2 times
>>> taper((xs, ys, zs), halfsize, times)
>>>

Inplace detrend
-------------------------

>>> rmean(xs)
>>> detrend(xs)
>>>

Searching given time series
--------------------------

>>> import numpy as np
>>> t0, delta = 0.0, 0.1
>>> xs = np.random.rand(3000) - 0.5
>>> # manual cutting
>>> i1 = round_index(10.0, t0, delta)
>>> i2 = round_index(300.0, t0, delta) + 1
>>> new_xs = xs[i1:i2]
>>> # obtain max amplitude point
>>> tref, wnd = 50.0, -5.02, 300.0
>>> idx, t, amp = max_amp_index(xs, t0, delta, tref, wnd[0], wnd[1], polarity=1) # positive max
>>> idx, t, amp = max_amp_index(xs, t0, delta, tref, wnd[0], wnd[1], polarity=-1) # negative max
>>> idx, t, amp = max_amp_index(xs, t0, delta, tref, wnd[0], wnd[1], polarity=0) # absolute max
>>>


Not-Inplace cut
-------------------------

>>> t0 = 0.0
>>> w1, w2 = 99.5, 300
>>> new_xs, new_t0 = cut(xs, delta, t0, w1, w2)
>>>

Inplace whitening
-------------------------

>>> wtlen = 128.0 # sec
>>> f1, f2 = 0.02, 0.0667
>>> water_level_ratio, taper_halfsize = 1.0e-5, 30
>>> tnorm_f32(xs, delta, wtlen, f1, f2, water_level_ratio, taper_halfsize)
>>> wflen = 0.02
>>> fwhiten_f32(xs, delta, wflen, water_level_ratio, taper_halfsize)
>>>
"""



from matplotlib.pyplot import polar
import numpy as np
#from pyfftw.interfaces.cache import enable as pyfftw_cache_enable
from pyfftw.interfaces.numpy_fft import rfft, irfft
from numba import jit
from numba.core.typing import cffi_utils as cffi_support

from scipy.signal import correlate as scipy_correlate

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
            xs[idx] *= c[idx]
            xs[-1-idx] *= c[idx]

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

    tmp = np.arange(len, dtype=np.float64)
    s1 = np.sum(tmp*xs)
    s2 = np.sum(tmp*tmp)

    k = (s1-len*ymean*xmean) / (s2-len*xmean*xmean)
    b = ymean - k*xmean

    tmp = tmp*k+b
    xs -= tmp

#############################################################################################################################
#  search sth from time series
#############################################################################################################################
@jit(nopython=True, nogil=True)
def floor_index(t, t0, delta):
    """
    Return the floor index for time `t` given
    the start time `t0` and sampling time
    interval `delta`.
    """
    return int( np.floor((t-t0)/delta) )
@jit(nopython=True, nogil=True)
def ceil_index(t, t0, delta):
    """
    Return the ceil index for time `t` given
    the start time `t0` and sampling time
    interval `delta`.
    """
    return int( np.ceil((t-t0)/delta) )
@jit(nopython=True, nogil=True)
def round_index(t, t0, delta):
    """
    Return the round index for time `t` given
    the start time `t0` and sampling time
    interval `delta`.
    """
    return int( round((t-t0)/delta) )

@jit(nopython=True, nogil=True)
def max_amp_index(xs, t0, delta, tref, tmin, tmax, polarity=1):
    """
    Search for max amplitude, positive or negtive or absolute, for a time series.

    xs: the 1D time series. Should be an object of numpy.ndarray.
    t0: start time of the time series in second.
    delta: sampling time interval in second.
    tref, tmin, tmax: reference time and search time window (tmin, tmax).
    polarity: -1, 0, or 1 for negative, absolute, or positive amplitude maximum.

    Return: index, time, amplitude
    """
    i1 = round_index(tref+tmin, t0, delta)
    i2 = round_index(tref+tmax, t0, delta)+1

    i1 = i1 if i1>=0 else 0
    i2 = i2 if i2<=xs.size else xs.size

    idx = i1
    if polarity > 0: #positive
        idx += np.argmax(xs[i1:i2] )
    elif polarity < 0: # negative
        idx += np.argmin(xs[i1:i2] )
    else:
        idx1 = np.argmax(xs[i1:i2] )
        idx2 = np.argmin(xs[i1:i2] )
        v1, v2 = xs[i1+idx1], -xs[i1+idx2]
        idx += (idx1 if v1>=v2 else idx2)

    v = xs[idx]
    return idx, t0+idx*delta, v

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
    i0 = round_index(new_t0, t0, delta)
    i1 = round_index(new_t1, t0, delta) + 1
    new_size = i1-i0
    new_xs = np.zeros(new_size, dtype=xs.dtype)
    new_t0 = i0*delta + t0
    if i1<0 or i0>xs.size:
        return new_xs, new_t0
    if i1 > xs.size:
        i1 = xs.size
    if i0<0:
        new_xs[-i0:i1-i0] = xs[:i1]
    else:
        new_xs[0:i1-i0] = xs[i0:i1]
    return new_xs, new_t0

#############################################################################################################################
#
#############################################################################################################################
def cc_delay(x1, x2, sign='pos'):
    """
    Use cross-correlation method to find the waveform delay between two arrays.

    Return an int `n`. That means the `x1` and `x2` will have the maximum
    cross-correlation coefficient by shift the `x2` leftwards for `n` counts.

    For positive `n` that is `x1` and `x2[n:]`, or `np.concatenate((np.zeros(n), x1))`
    and `x2`. For negative `n` that is `x1` and `np.concatenate((np.zeros(-n), x2))`
    or `x1[-n:]` and `x2`.
    """
    cc = scipy_correlate(x2, x1, 'full')
    idx = np.argmax(cc) if sign =='pos' else np.argmin(cc)
    n = idx+(1-len(x1))
    return n

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
def tnorm_f32(xs, delta, winlen, f1, f2, water_level_ratio= 1.0e-5, taper_halfsize=0):
    """
    Inplace temporal normalization of input trace `xs` (a numpy.ndarray(dtype=np.float32) object).

    delta:             sampling ime interval in sec.
    winlen:            window size in sec.
    f1, f2:            frequency band in Hz that will be used to from the weight.
    water_level_ratio: default is 1.0e-5.
    taper length:      taper size (an int) in each end after the division.
    """
    wndsize = int(round(winlen/delta) )
    wndsize = (wndsize // 2)*2 +1

    weight = np.copy(xs) # obvious it is deep copy here as xs is a 1D array. weight will have same dtype as xs.
    iirfilter_f32(weight, delta, 0, 2, f1, f2, 2, 2)
    weight = moving_average_abs_f32(weight, wndsize, False)
    weight += ( np.max(weight) * water_level_ratio )

    #if True in np.isnan(weight) or True in np.isinf(weight) or np.count_nonzero(weight) == 0 :
    #    xs[:] = 0.0
    #else :
    xs /= weight

    if taper_halfsize > 0:
        taper(xs, taper_halfsize)
def fwhiten_f32(xs, delta, winlen, water_level_ratio= 1.0e-5, taper_halfsize=0, speedup_i1= -1, speedup_i2= -1):
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
    import sys
    x1 = [0, 1, 2, 1]
    x2 = [0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0]
    n = cc_delay(x1, x2, 'pos')
    sys.exit(0)
    import matplotlib.pyplot as plt
    from copy import deepcopy
    from sacpy.sac import c_rd_sac
    import numpy as np
    st = c_rd_sac('junk/2010/20100103_223625.a/processed/II.NNA.00.BHZ')
    delta = st.hdr.delta # sampling time interval
    st.truncate(10800, 32300)
    st.write('junk.sac')
    st.detrend()
    st.taper(0.005)
    ts = st.get_time_axis()
    xs = deepcopy(st.dat)
    ys = deepcopy(st.dat)
    zs = deepcopy(st.dat)

    plt.subplot(311); plt.plot(ts, xs)
    plt.subplot(312); plt.plot(ts, ys)
    plt.subplot(313); plt.plot(ts, zs)
    #Inplace IIR filter
    #-------------------------
    if False:
        #xs = np.random.random(10000)-0.5
        btype, f1, f2, ord, npass = 2, 0.02, 0.0666, 2, 2
        iirfilter_f32(xs, delta, 0, btype, f1, f2, ord, npass)

        #ys = np.random.random(10000)-0.5
        #zs = np.random.random(10000)-0.5
        iirfilter2_f32((xs, ys, zs), delta, 0, btype, f1, f2, ord, npass)
        plt.subplot(311); plt.plot(ts, xs)
        plt.subplot(312); plt.plot(ts, ys)
        plt.subplot(313); plt.plot(ts, zs)
        st.dat = ys
        #st.write('junk.sac')

    #Inplace taper
    #-------------------------
    if False:
        halfsize, times = 3000, 2
        taper(xs, halfsize, times) # will apply the taper for 2 times
        taper2((xs, ys, zs), halfsize, times)
        plt.subplot(311); plt.plot(ts, xs)
        plt.subplot(312); plt.plot(ts, ys)
        plt.subplot(313); plt.plot(ts, zs)
    #Inplace detrend
    #-------------------------
    if False:
        rmean(xs)
        detrend(xs)

        plt.subplot(311); plt.plot(ts, xs)
        plt.subplot(312); plt.plot(ts, ys)
        plt.subplot(313); plt.plot(ts, zs)

    #Not-Inplace cut
    #-------------------------
    if False:
        t0 = st.hdr.b
        print(t0)
        w1, w2 = 99.5, 300
        new_xs, new_t0 = cut(xs, delta, t0, 99.5, 300)
        ts_x = np.arange(new_xs.size)*delta + new_t0

        new_ys, new_t0 = cut(ys, delta, t0, 10900, 20000)
        ts_y = np.arange(new_ys.size)*delta + new_t0

        new_zs, new_t0 = cut(zs, delta, t0, 32000, 40000)
        ts_z = np.arange(new_zs.size)*delta + new_t0

        plt.subplot(311); plt.plot(ts_x, new_xs)
        plt.subplot(312); plt.plot(ts_y, new_ys)
        plt.subplot(313); plt.plot(ts_z, new_zs)


    #Inplace whitening
    #-------------------------
    if True:
        wtlen = 128.0 # sec
        f1, f2 = 0.02, 0.0667
        water_level_ratio, taper_halfsize = 1.0e-5, 30
        tnorm_f32(xs, delta, wtlen, f1, f2, water_level_ratio, taper_halfsize)
        wflen = 0.02
        fwhiten_f32(xs, delta, wflen, water_level_ratio, taper_halfsize)

        plt.subplot(311); plt.plot(ts, xs)
        plt.subplot(312); plt.plot(ts, ys)
        plt.subplot(313); plt.plot(ts, zs)
    plt.show()