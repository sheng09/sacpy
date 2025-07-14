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
import scipy
FLAG_PYFFTW_USED = False
try:
    from pyfftw.interfaces.cache import enable as pyfftw_cache_enable
    from pyfftw.interfaces.scipy_fft import rfft, irfft
    FLAG_PYFFTW_USED = True
except:
    FLAG_PYFFTW_USED = False
    from scipy.fft import rfft, irfft
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

def ceil_closest_amp_index(xs, t0, delta, tref, amplitude=0.0):
    """
    Search for the point that have the amplitude crossing to the `amplitude` and is the closest (ceil to) the `tref`.

    xs: the 1D time series. Should be an object of numpy.ndarray.
    t0: start time of the time series in second.
    delta: sampling time interval in second.
    tref: reference time to search from.
    amplitude:

    Return: index, time, amplitude
    """
    ys = np.array(xs)
    idx_tref = ceil_index(tref, t0, delta)
    if idx_tref<0:
        idx_tref = 0
    elif idx_tref >  ys.size:
        idx_tref = ys.size
    ys = ys[idx_tref:] - amplitude
    ###################################
    vmul = ys[:-1] * ys[1:]
    i0 = np.where(vmul<=0.0)[0][0]
    #print(i0+idx_tref, ys[i0], t0+delta*(i0+idx_tref) )
    return i0+idx_tref, t0+delta*(i0+idx_tref), ys[i0]

def floor_closest_amp_index(xs, t0, delta, tref, amplitude=0.0):
    """
    Search for the point that have the amplitude crossing to the `amplitude` and is the closest (floor to) the `tref`.

    xs: the 1D time series. Should be an object of numpy.ndarray.
    t0: start time of the time series in second.
    delta: sampling time interval in second.
    tref: reference time to search from.
    amplitude:

    Return: index, time, amplitude
    """
    ys = np.array(xs)
    idx_tref = floor_index(tref, t0, delta)
    if idx_tref<0:
        idx_tref = 0
    elif idx_tref >  ys.size:
        idx_tref = ys.size
    ys = ys[:idx_tref] - amplitude
    ###################################
    vmul = ys[:-1] * ys[1:]
    i0 = np.where(vmul<=0.0)[0][-1]
    return i0, t0+delta*i0, ys[i0]

#############################################################################################################################
# search sth for spectra processing
#############################################################################################################################
def get_rfft_spectra_bound(fftsize, delta, frequency_band, critical_level=0.01):
    """
    Return the index bound [i1, i2) for spectral computation, based on the
    fact that the spectra outside  [i1, i2), namely the given `frequency_band`,
    are below the `critical_level` and hence can be ignored.
    #
    Note: we consider the rfft for the transform between time and frequency domain.
    In other words, the `fftsize` means the spectra length is `fftsize//2+1`.
    #
    Parameters:
        fftsize:        the fft size.
        delta:          the sampling interval in seconds.
        frequency_band: the frequency band of interest. A tuple of (f1, f2) in Hz.
        critical_level: the critical level between 0.0 and 1.0 for ignoring the spectra.
                        `critical_level=0.01` means ignoring spectra below 1% of
                        the maximum amplitude
                        Default is 0.01.
    Return:
        i1, i2: the index bound for the frequency band.
    """
    df = 1.0/(fftsize*delta)
    fmin, fmax = df, 0.5/delta-df # safe value
    f1, f2 = frequency_band
    ##########################################################################################
    if f1 >= f2 or (f1<=fmin and f2>=fmax) or f1>=fmax or f2 <=fmin:
        return 0, fftsize//2+1
    ##########################################################################################
    i1, i2 = 0, fftsize//2+1
    x = np.zeros(fftsize, dtype=np.float32)
    x[fftsize//2] = 1.0 # Note! cannot use x[0] = 1.0 for setting the delta function
    if f1 <=fmin:    # a lowpass filter is used
        iirfilter_f32(x, delta, 0, 0, f1, f2, 2, 2)
        amp = np.abs( rfft(x, fftsize) )
        c = amp.max() * critical_level
        i2 = np.argmax(amp<c)
    elif f2 >= fmax: # a highpass filter is used
        iirfilter_f32(x, delta, 0, 1, f1, f2, 2, 2)
        amp = np.abs( rfft(x, fftsize) )
        c = amp.max() * critical_level
        i1 = np.argmax(amp>=c)
    else:             # a bandpass filter is used
        iirfilter_f32(x, delta, 0, 2, f1, f2, 2, 2)
        amp = np.abs( rfft(x, fftsize) )
        c = amp.max() * critical_level
        i1 = np.argmax(amp>=c)
        i2 = i1 + np.argmax(amp[i1:]<c)
    ##########################################################################################
    # in case of out of bound or invalid values
    if i1 < 0:
        i1 = 0
    if i2 > fftsize//2+1:
        i2 = fftsize//2+1
    if i2 <= i1:
        i1, i2 = 0, fftsize//2+1
    ##########################################################################################
    return i1, i2

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
def fwhiten_f32(xs, delta, winlen, water_level_ratio= 1.0e-5, taper_halfsize=0,
                speedup_i1= -1, speedup_i2= -1, fftsize=-1):
    """
    Inplace frequency whitening of input trace `xs` (a numpy.ndarray(dtype=np.float32) object).

    delta:             sampling time interval in sec.
    winlen:            window size in Hz.
    water_level_ratio: default is 1.0e-5.
    taper_halfsize:    taper halfsize in each end after the division.

    fftsize:    specific fftsize (default is -1 to use xs.size instead)
    speedup_i1: the low index range of the spectrum corresponding to fftsize. (default value is -1 to disable speedup)
    speedup_i2: the up ... ...                                                (default is -1 to disable speedup)
    """
    if fftsize < xs.size:
        fftsize = xs.size
        spec = rfft(xs, fftsize)
    else:
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


#############################################################################################################################
# JIT For angle computations
#############################################################################################################################
@jit(nopython=True, nogil=True)
def round_degree_360(xs):
    """
    Round degrees to be in [0, 360)
    xs: a single value or a list of values (e.g., an object ot np.array)
    """
    return xs % 360
def round_degree_180(deg):
    """
    First, round degrees to be in [0, 360), and then angles in [180, 360) will be rounded to 360-angles.
    For example, round_degree_180(170) = 170, round_degree_180(190) = 170...

    xs: a single value or a list of values (e.g., an object ot np.array)
    """
    x = round_degree_360(deg)
    return round_degree_360((x//180)*(-2*x) + x)

#############################################################################################################################
# JIT array processing
#############################################################################################################################
# Insert values into an array (maintaining its local monotonicity), and apply linear interpolation to other arrays for the inserted values.
def insert_values(xcs, xs, *args):
    """
    Insert many `xc` into an array `xs`, so that the result `new_xs` must not have two successive points 
    `new_xs[i]` and `new_xs[i+1]` across the value of `xc`. In other words, it is not allowed to have
    `new_xs[i] < xc < new_xs[i+1]` or new_xs[i+1] < xc < new_xs[i]`.
    Also, insertion are conducted for each array of the `*args` with respect to the insertion of `xc`
    into `xs`. Linear interpolation method is taken for the insertion.

    xcs:   a single number or a list of numbers.
    xs:    an 1D array.
    *args: more arrays that have the same size as `xs`. These arrays will be inserted with
           at the same index (and using the linear interpolation method) as inserting `xs`.

    Return:
           new_xs, arg1, arg2, arg3,...

    E.g.,
    >>> x1 = [0, 1, 2, 3, 2, 1]
    >>> y1 = x1+10
    >>> z1 = x1*x1
    >>> new_x1, new_y1, new_z1 = insert_values((1.5, 2.5), x1, y1, z1)
    >>> print(new_x1)
    >>> print(new_y1)
    >>> print(new_z1)
    """
    xcs = np.array(xcs).flatten()
    #############################################################
    xs   = list(xs)
    args = [list(it) for it in args]
    for xc in xcs:
        junk = np.array(xs)-xc
        junk2 = junk[:-1]*junk[1:]
        idx_cross_left  = np.where(junk2<0)[0]
        for il in idx_cross_left[::-1]:
            ir = il+1
            xl, xr = xs[il], xs[ir]
            xs.insert(ir, xc)
            for ys in args:
                yl, yr = ys[il], ys[ir]
                yc = yl*(xr-xc)/(xr-xl) + yr*(xc-xl)/(xr-xl)
                ys.insert(ir, yc)
    if len(args)==0:
        return np.array(xs)
    results = [np.array(xs)]
    results.extend( [np.array(ys) for ys in args] )
    return tuple(results)
# Split an array into many sub arrays given critical values, and split additional arrays at the same index.
def split_arrays(xcs, xs, *args, edge='i', **kwargs):
    """
    Split an 1D array `xs` (and optionally additional arrays) at the critical values `xcs`,
    and return a list of segments. Different methods will be used to take care of the
    critical values. Please see below for `edge` parameter.

    xcs:  a single or a list of critical values to split the array `xs`.
    xs:   an 1D array.
    *arg: more arrays that have the same size as `xs`. These arrays will
          be splitted at the same index (and using the same edge method) as splitting `xs`.
    edge: an argument for how to process the xs values across the a `xc`.
          '+': extend one more outside the selected range.
          '-': do not extend ...
          'i': use linear interpolation to compute the values at the x_c.
          's': split the array using the exact appearnce of `x_c` in the `xs`.

    Return:
          list_of_x_arrays, list_of_y_arrays, list_of_z_arrays, ...
          Each (e.g., `split_arrays`) is a list of arrays.

    e.g.,
           >>> xs = [0,  1,  2,  3,  4,  5,  6,  5,  4,  3,  2,  1,  0]
           >>> ys = [10,21, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
           >>> zs = [10,21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
           >>>
           >>> x_segments, y_segments, z_segments = split_arrays(4.1, xs, ys, zs, edge='i' )
           >>> # or
           >>> x_segments = split_arrays(4.1, xs, edge='i' )
           >>>
           >>> x_segments, y_segments, z_segments = split_arrays((4.1, 5.1), xs, ys, zs, edge='i' )
           >>> # or
           >>> x_segments = split_arrays((4.1, 5.1), xs, edge='i' )
    """
    xs = np.array(xs).astype(np.float64)
    try:
        junk = len(xcs)
    except Exception:
        xcs = (xcs, )
    #############################################
    if edge == 's':
        # find all indexs where elements in the xs equal to xc
        idxs = list()
        for x_c in xcs:
            difference = xs-x_c
            zero = np.abs( np.max(difference) )*1.0e-12 # 
            idxs.extend( np.where(np.abs(difference)<=zero)[0] )
        # add the 0 and n-1 to the idxs are they are natural boundaries
        idxs = sorted( set(idxs).union( (0, difference.size-1) ) )
        # segments = [x_segments, y_segments,...]
        segments = [list() for it in range(len(args)+1) ]
        x_segments = segments[0]
        for i1, i2 in zip( idxs[:-1], idxs[1:] ):
            x_segments.append( xs[i1:i2+1] )
            for ys_segments, ys in zip(segments[1:], args):
                ys_segments.append( ys[i1:i2+1] )
        if len(args) == 0:
            return x_segments
        return segments
    if edge == 'i':
        # Add necessary xc values into xs, and interpolated values to ys, zs,...
        # Then run split_arrays(... edge='s') to split with respect to all
        tmp = insert_values(xcs, xs, *args)
        if len(args)==0:
            xs, args = tmp, tuple()
        else:
            xs, args = tmp[0], tmp[1:]
        kwargs['edge'] = 's'
        return split_arrays(xcs, xs, *args, **kwargs)
    #############################################
    if edge in '-+':
        # Call split_array(..., edge='s') first
        kwargs['edge'] = 's'
        tmp = split_arrays(xcs, xs, *args, **kwargs)
        if len(args)==0:
            xs_segments, args_segments = tmp, tuple()
        else:
            xs_segments, args_segments = tmp[0], tmp[1:]
        # then split
        new_segments = [list() for it in range(len(args)+1) ]
        ihead, itail = (-1, -1) if edge == '-' else (0, -2)
        di0, di1 = (1, 1) if edge == '-'  else (0, 2)
        for iseg, xs in enumerate(xs_segments):
            idxs = [ihead, xs.size+itail]
            for x_c in xcs:
                junk = xs-x_c
                junk2 = junk[:-1]*junk[1:]
                idxs.extend(np.where(junk2<0.0)[0])
            idxs = sorted(idxs)
            for i0, i1 in zip( idxs[:-1], idxs[1:] ):
                i0, i1 = i0+di0, i1+di1
                new_segments[0].append( xs[i0:i1] )
                for ys_segments, new_ys_segments in zip(args_segments, new_segments[1:]):
                    ys = ys_segments[iseg]
                    new_ys_segments.append( ys[i0:i1] )
        if len(args)==0:
            return new_segments[0]
        return new_segments
def split_arrays_range(xmin, xmax, xs, *args, edge='i'):
    """
    e.g.,
        >>> x_segments = split_arrays_range(xmin=10, xmin=30, xs)
        >>> x_segments, y_segmemts, z_segments  = split_arrays_range(xmin=10, xmin=30, xs, ys, zs)
    """
    if xmin>=xmax:
        return [tuple() for it in range(len(args)+1) ]
    ########################################################
    tmp = split_arrays((xmin, xmax), xs, *args, edge=edge)
    if len(args)==0:
        x_segments, args_segments = tmp, tuple()
    else:
        x_segments, args_segments = tmp[0], tmp[1:]
    ########################################################
    segments = [list() for it in range(len(args)+1) ]
    for iseg, xs in enumerate(x_segments):
        mid_index = int((len(xs) - 1)/2)
        v = xs[mid_index]  # the mid index value
        if xmin <= v <= xmax:
            segments[0].append(xs)
            for qqq, www in zip(segments[1:], args_segments):
                qqq.append(www[iseg])
    ########################################################
    if len(args) == 0:
        segments = segments[0]
    return segments

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    if True:
        critical_level = 0.01 # 1 percent
        fftsize = 32401
        dt = 1.0
        band = (0.02, 0.06666)
        i1, i2 = get_rfft_spectra_bound(fftsize, dt, band, critical_level)
        ###
        xs = np.zeros(fftsize, dtype=np.float32)
        xs[fftsize//2] = 1.0
        ys = xs.copy()
        iirfilter_f32(ys, dt, 0, 2, band[0], band[1], 2, 2)
        xfs = rfft(xs, fftsize)
        yfs = rfft(ys, fftsize)
        fs = scipy.fft.rfftfreq(fftsize, d=dt)
        ###
        s1, s2 = fs[i1], fs[i2]
        ###
        plt.plot(fs, np.abs(xfs) )
        plt.plot(fs, np.abs(yfs) )
        plt.plot([s1, s1], [0, 1], 'r')
        plt.plot([s2, s2], [0, 1], 'r')
        plt.plot(fs, critical_level+fs*0)
        plt.ylim((-0.1, 2) )
        plt.title((i2-i1)/fs.size*100)
        plt.show()
    if False:
        xs = np.array([0, 1, 2, 4, 5, 6, 7, 5, 3, 2, 1, 0])
        ys = np.array(xs)+10
        zs = np.array(xs)+20
        xcs = 1.1, 5.0
        print('x', xs)
        print('xcs', xcs)
        vx = insert_values(xcs, xs)
        print('vx', vx)
        vx, vy, vz = insert_values(xcs, xs, ys, zs)
        print('vx', vx)
        print('vy', vy)
        print('vz', vz)
        print()
        for edge in 'si-+':
            vx = split_arrays(xcs, xs, edge=edge)
            print('xs', xs)
            print('xcs', xcs)
            print(edge, 'vx', vx)
            #print(v[1])
            vx, vy = split_arrays(xcs, xs, ys, edge=edge)
            print(edge, 'vx', vx)
            print(edge, 'vy', vy)
            print()
        
        for edge in 'i-+':
            xmin, xmax=1.1, 5.0
            vx = split_arrays_range(xmin, xmax, xs, edge=edge)
            print('xs', xs)
            print('xmin, xmax', xmin, xmax)
            print(edge, 'vx', vx)
            #print(v[1])
            vx, vy, vz = split_arrays_range(xmin, xmax, xs, ys, zs, edge=edge)
            print(edge, 'vx', vx)
            print(edge, 'vy', vy)
            print(edge, 'vy', vz)
            print()

        for edge in 'i-+':
            c = np.linspace(0, 10, 100)
            xs = np.sin(c)
            ys = xs+1
            zs = np.cos(xs)
            xmin, xmax= -0.2, 0.3
            vx, vy, vz = split_arrays_range(xmin, xmax, xs, ys, zs, edge=edge)
            plt.plot(xs, ys, 'o', color='C0')
            plt.plot(xs, zs, 'o', color='C4')
            for ivx, ivy, ivz in zip(vx, vy, vz):
                plt.plot(ivx, ivy, 'x', color='C1')
                plt.plot(ivx, ivz, 'x', color='C3')
                #plt.plot(c, ivy)
                #plt.plot(c, ivz)
            plt.show()
        print()

        #print(x)
        #print(y)
        #x, y = insert_x(3, x, y)
        #print(x)
        #print(y)
        #v = split_arrays(1.1, x, edge='-') #, edge='s') #, y, z, edge='-')
        #for it in v:
        #    print('final', it)
        #    pass
    if False:
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
    if False:
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