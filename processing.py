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


import copy
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
import numba
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
def taper2d(lst_of_xs, half_size, times=1):
    """
    Inplace taper a list of trace of same size in place.
    The content of `lst_of_xs` elements will be revised after running.

    0 <= half_size <= len(xs)/2
    times: apply the taper for many times (default is 1).
    """
    n = half_size
    junk = np.pi/n
    c = np.array( [0.5*( 1+np.cos(junk*(idx-n) ) ) for idx in range(0, n)] )
    if times > 1:
        c = c**times
    for xs in lst_of_xs:
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
    return int(np.floor((t-t0)/delta) )
@jit(nopython=True, nogil=True)
def ceil_index(t, t0, delta):
    """
    Return the ceil index for time `t` given
    the start time `t0` and sampling time
    interval `delta`.
    """
    return int(np.ceil((t-t0)/delta) )
@jit(nopython=True, nogil=True)
def round_index(t, t0, delta):
    """
    Return the round index for time `t` given
    the start time `t0` and sampling time
    interval `delta`.
    """
    return int(np.round((t-t0)/delta) )

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
    i1 = ceil_index(tref+tmin, t0, delta)
    i2 = floor_index(tref+tmax, t0, delta)+1

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
# JIT linear normalization
#############################################################################################################################
@jit(nopython=True, nogil=True)
def norm_array1d(xs, percentile=-1, scale=1.0):
    """
    Inplace linear normalization of an 1D array `xs` of numpy.ndarray.

    :param xs:         the 1D array to be normalized.
    :param percentile: a value. If -1, then the maximum value of the array will be used as the reference for normalization.
                       If the value is between 0 and 1, then the vth percentile of the absolute amplitude is used as the reference.
    :param scale:      a value to scale the normalized array. Default is 1.0.
    """
    vmax = np.percentile(np.abs(xs), percentile*100) if (0<percentile<=1) else np.max(np.abs(xs))
    vmax *= scale
    if vmax <= 0.0:
        vmax = 1.0
    xs *= (1.0/vmax)
@jit(nopython=True, nogil=True)
def norm_mat2d(mat, row_wise=True, percentile=-1, scale=1.0):
    """
    Inplace linear normalization of a 2D array `mat` of numpy.ndarray.

    :param mat:        the 2D array to be normalized.
    :param row_wise:   if True (default), then each row will be normalized separately. If False, all rows will be scaled together.
    :param percentile: a value. If -1, then the maximum value of the array will be used as the reference for normalization.
                       If the value is between 0 and 1, then the vth percentile of the absolute amplitude is used as the reference.
    :param scale:      a value to scale the normalized array. Default is 1.0.
    """
    if row_wise:
        nrow = mat.shape[0]
        for irow in range(nrow):
            norm_array1d(mat[irow], percentile, scale)
    else:
        vmax = np.percentile(np.abs(mat), percentile*100) if (0<percentile<=1) else np.max(np.abs(mat))
        vmax *= scale
        if vmax <= 0.0:
            vmax = 1.0
        mat *= (1.0/vmax)


#############################################################################################################################
# JIT cut
#############################################################################################################################
@jit(nopython=True, nogil=True)
def cut(xs, delta, t0, wnd_start, wnd_end):
    """
    Cut a time series `xs` within the time window [wnd_start, wnd_end]. If the window is bigger than the time range of input time
    series, then zeros will be filled in the returned new_xs for those data points that are not in the time series.

    No interpolation is applied, so the returned time series may not start at the time `wnd_start`!
    Instead, a valid value `new_t0` will be returned.
    If you want to have the first data point start exactly at `wnd_start`, then interpolation is needed, and please refer to
    the function `TSFunc.shift_and_cut_array1d(...)` for that purpose.

    :param xs:    the 1D time series.
    :param delta: sampling time interval in second.
    :param t0:    the start time of the time series.
    :param new_t0: the start of the time window to cut
    :param new_t1: the end of the time window to cut
    """
    i0 = ceil_index(wnd_start, t0, delta)
    i1 = floor_index(wnd_end,   t0, delta) + 1
    new_size = i1-i0
    new_xs = np.zeros(new_size, dtype=xs.dtype)
    new_t0 = i0*delta + t0
    if i1<0 or i0>xs.size:
        return new_xs, new_t0
    #####################
    if i1 > xs.size:
        i1 = xs.size
    #####################
    if i0<0:
        new_xs[-i0:i1-i0] = xs[:i1]
    else:
        new_xs[0:i1-i0] = xs[i0:i1]
    #####################
    return new_xs, new_t0
@jit(nopython=True, nogil=True)
def cut2d(mat, delta, t0, wnd_start, wnd_end, dx, x0, x_start, x_end):
    nrow, ncol = mat.shape
    # the index in the old mat
    irow0 = ceil_index(x_start, x0, dx)
    irow1 = floor_index(x_end,   x0, dx) + 1
    icol0 = ceil_index(wnd_start, t0, delta)
    icol1 = floor_index(wnd_end,   t0, delta) + 1
    new_x0 = irow0*dx + x0
    new_t0 = icol0*delta + t0
    ######################################################################################
    new_mat = np.zeros((irow1-irow0, icol1-icol0), dtype=mat.dtype)
    if nrow <= irow0 or irow1 <0 or ncol <= icol0 or icol1 < 0:
        return new_mat, new_t0, new_x0
    ######################################################################################
    # need to do: new_mat[0:, 0:] = mat[irow0:irow1, icol0:icol1], however those indexes may be out of bound
    if irow0 > 0:
        new_irow0 = 0
    else:
        new_irow0 = -irow0
        irow0 = 0
    if irow1 > nrow:
        irow1 = nrow
    new_irow1 = new_irow0+(irow1-irow0)
    #
    if icol0 > 0:
        new_icol0 = 0
    else:
        new_icol0 = -icol0
        icol0 = 0
    if icol1 > ncol:
        icol1 = ncol
    new_icol1 = new_icol0+(icol1-icol0)
    #
    new_mat[new_irow0:new_irow1, new_icol0:new_icol1] = mat[irow0:irow1, icol0:icol1]
    return new_mat, new_t0, new_x0


#############################################################################################################################
# Mask (not inplace)
#############################################################################################################################
def mask_time_window(xs, delta, t0, wnds, fill_value=0.0, taper_half_size=0, taper_order=1):
    """
    Mask (not inplace) a 1D array, keeping values that are within a time window.

    :param xs:         the 1D array to be masked.
    :param delta:      the sampling time interval in seconds.
    :param t0:         the start time of the time series.
    :param wnds:       a list of time windows, and each window is (start, end).
                       Do not worry if some winddows intersect, and they will be merged.
    :param fill_value: the value to fill for data points outside the time window.
    :param taper_half_size: an int that declares the number of data points of the tapered region at each edge.
                            If > 0, then a tukey taper will be applied to the cutted data points.
                            Default is 0, which means no tapering.
                            Please note, the taper will be applied to the data points that are within the time window.
    :param taper_order:     the order of the taper, which means to apply the tukey taper for how many times. (Default is 1)

    :return:           a new 1D array with the same size as `xs`, but only values within the time window are kept.
    """
    wnds = np.array(wnds).reshape((-1, 2)) # make sure wnds is a 2D array
    wnds = np.array( [it for it in wnds if it[0] < it[1] ]) # remove invalid windows
    if wnds.size  <= 0:
        raise ValueError("No valid time windows provided. Please check the input `wnds`.")
    ####
    wnds = wnds[np.argsort(wnds[:,0])]
    tmp = [wnds[0,0], wnds[0,1]] # start with the first window
    for v1, v2 in wnds[1:]:
        if v1 > tmp[-1]:
            tmp.append(v1)
            tmp.append(v2)
        else:
            tmp[-1] = v2
    wnds = np.array(tmp).reshape((-1, 2))
    ####
    idxs = np.zeros(wnds.shape, dtype=np.int64)
    idxs[:,0] = np.ceil(  (wnds[:,0]-t0)/delta ).astype(np.int64)
    idxs[:,1] = np.floor( (wnds[:,1]-t0)/delta ).astype(np.int64)+1
    idxs = np.where(idxs < 0, 0, idxs) # make sure idxs are not negative
    idxs = np.where(idxs > xs.size, xs.size, idxs) # make sure idxs are not larger than the size of xs
    #####
    new_xs = np.full(xs.shape, fill_value, dtype=xs.dtype)
    for i0, i1 in idxs:
        if i0 < i1:
            new_xs[i0:i1] = xs[i0:i1]
            if (taper_half_size>0) and ((i1-i0)>(2*taper_half_size)):
                taper(new_xs[i0:i1], taper_half_size, taper_order) # apply taper
    return new_xs
def mask2d_time_window(mat, delta, t0, xs, wnd_curves, fill_value=0.0, wnd_curve_extrapolate=False, wnd_curve_interp1d_kind='linear', taper_half_size=0, taper_order=1):
    """
    Mask (not inplace) a 2D array, keeping values that are within areas defined by curves.

    :param mat:        the 2D array to be masked.
    :param delta:      the sampling time interval in seconds.
    :param t0:         the start time of all the time series.
    :param xs:         the location of each time series in the spatial domain (the 0th dimension of `mat`).
    :param wnd_curves: a list of curves. Each curve is (curve_xs, curve_ts). So only data points between the 1st and the 2nd
                       curves, between the 3rd and the 4th curves, between the 5th and the 6th curves,..., will be kept.
    :param fill_value: the value to fill for data points outside the selected region as defined by the curves above.
    :param wnd_curve_extrapolate:   If False (default), then the curves will not be extrapolated, and zero will be used
                                    to the left and right ends of the `xs` array.
                                    If True, then the curves will be extrapolated to the left and right ends of the `xs` array.
    :param wnd_curve_interp1d_kind: The kind of interpolation for the curves.
                                    Default is 'linear'. Other options are 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'.
                                    Check `scipy.interpolate.interp1d(...)` for more details about these options.
    :param taper_half_size: an int that declares the number of data points of the tapered region at each edge.
                            If > 0, then a tukey taper will be applied to the cutted data points.
                            Default is 0, which means no tapering.
                            Please note, the taper will be applied to the data points that are within the time window.
    :param taper_order:     the order of the taper, which means to apply the tukey taper for how many times. (Default is 1)

    :return:           a new 2D array with the same shape as `mat`, but only values within areas defined by curves.
    """
    dict_interp1d_kind_min_size = {'zero': 1, 'slinear': 2, 'quadratic': 3, 'cubic': 4,  'linear': 2 }
    dict_min_size_interp1d_kind = {1: 'zero', 2: 'slinear', 3: 'quadratic', 4: 'cubic'}
    #######
    get_func = lambda _x, _y, _kind: scipy.interpolate.interp1d(_x, _y, kind=_kind, bounds_error=False, assume_sorted=False, fill_value=0.0) # Note, fill_value here is not the one in `mask_time_window(...)`!
    if wnd_curve_extrapolate:
        get_func = lambda _x, _y, _kind: scipy.interpolate.interp1d(_x, _y, kind=_kind, bounds_error=False, assume_sorted=False, fill_value='extrapolate')
    #######
    wnd_ts = np.zeros( (len(wnd_curves), xs.size), dtype=np.float64)
    for icurve, (curve_xs, curve_ts) in enumerate(wnd_curves):
        kind = wnd_curve_interp1d_kind
        if len(curve_xs) < dict_interp1d_kind_min_size[kind]:
            kind = dict_min_size_interp1d_kind[len(curve_xs)]
        func_interp = get_func(curve_xs, curve_ts, kind)
        wnd_ts[icurve] = func_interp(xs)
    wnd_ts = np.transpose(wnd_ts) # after transpose, each row of `wnd_ts` correspond to one row of mat (a time series)
    #######
    new_mat = np.full(mat.shape, fill_value, dtype=mat.dtype)
    for irow in range(mat.shape[0]):
        new_mat[irow] = mask_time_window(mat[irow], delta, t0, wnd_ts[irow], fill_value=fill_value, taper_half_size=taper_half_size, taper_order=taper_order)
    return new_mat


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
    xs   = copy.deepcopy(list(xs))
    args = [copy.deepcopy(list(it)) for it in args]
    for xc in xcs:
        # get the index_cross for xc[i] < xc < xc[i+1]
        diff = np.array(xs)-xc
        abs_diff = np.abs(diff)
        cross = diff[:-1]*diff[1:]
        idx_cross_left  =  np.where(cross<0)[0]
        idx_cross_right = idx_cross_left+1
        # get the index_on for xc[i] == xc
        zero = np.max(abs_diff)*1.0e-9 # a very small number
        idxs_on = set(np.where(abs_diff<=zero)[0] )
        # remove any index_on from idx_cross
        idx_cross_left = [ il for (il, ir) in zip(idx_cross_left, idx_cross_right) if ((il not in idxs_on) and (ir not in idxs_on)) ]
        idx_cross_left = sorted( set(idx_cross_left) ) # sort the idx_cross_left
        for il in idx_cross_left[::-1]:
            ir = il+1
            xl, xr = xs[il], xs[ir]
            xs.insert(ir, xc)
            for ys in args:
                yl, yr = ys[il], ys[ir]
                yc = yl*(xr-xc)/(xr-xl) + yr*(xc-xl)/(xr-xl)
                ys.insert(ir, yc)
    if len(args)==0:
        return xs
    results = [xs]
    results.extend( args )
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
          Note: could be non-number arrays if `edge` is 's' or '+' or '-'.
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
    ##############################################
    # The algorithm is as follows:
    #                              X(xc_on) 7 <-- idx_on   [...:7+1] 7 is considers as an idx_on
    #                             /
    #                            /
    #                           /
    #                          X 6
    #                         /
    #                        xc <-- xc_cross          for edge='-', apply [...:5+1], [5+1:...], in the format if [...:idx+1], [idx+1:...]
    #                       /                         for edge='+', apply [...:5+2], [5+0:...], in the format of [...:idx+2], [idx+0:...]
    #                      X 5 <-- idx_cross_left
    #                     /
    #                    /
    #                   /
    #                  X 4
    #                 /
    #                /
    #               /
    #              X 3
    #             /
    #            /
    #           /
    #          X(xc_on) 2 <-- idx_on           for whatever edge, apply [...:2+1], [2+0:...], in the format of [...:idx+1], [idx+0:...]
    #         /
    #        /
    #       X 1
    #      /
    #     /
    #    X(xc_on) 0 <-- idx_on               [0:...] 0 is considers as an idx_on
    #
    #    for each of idx_on, and whatever edge,    the splitting will be like [...:idx+1],  [idx+0:...]
    #    for each of idx_cross_left, and edge='-', the splitting will be like [...:idx+1],  [idx+1:...]
    #    for each of idx_cross_left, and edge='+', the splitting will be like [...:idx+2],  [idx+0:...]
    #
    #############################################
    if edge not in 'si-+':
        raise ValueError(f"Invalid edge value: {edge}. It should be one of 's', 'i', '-', '+'.")
    #############################################
    #### insert necessary edge values
    if edge == 'i':
        tmp = insert_values(xcs, xs, *args)
        if len(args) == 0:
            xs = tmp
        else:
            xs = tmp[0]
            args = tmp[1:]
        edge == 's'
    #############################################
    arr_xs = np.array(xs).astype(np.float64)
    xcs = np.array(xcs).flatten()
    #### get idxs_on
    idxs_on = {0, arr_xs.size-1}
    for x_c in xcs:
        abs_diff = np.abs(arr_xs-x_c)
        zero = np.max(abs_diff)*1.0e-9 # a very small number
        idxs_on.update( np.where(abs_diff<=zero)[0] )
    #### get idxs_cross_left
    idxs_cross_left, idxs_cross_right = list(), list()
    for x_c in xcs:
        diff = arr_xs-x_c
        cross = diff[:-1] * diff[1:]
        tmp = np.where(cross<0.0)[0]
        idxs_cross_left.extend(  tmp ) # add idxs_on to idxs_cross_left
        idxs_cross_right.extend( tmp+1)
    idxs_cross_left = [il for (il, ir) in zip(idxs_cross_left, idxs_cross_right) if (il not in idxs_on and ir not in idxs_on) ] # remove idxs_on from idxs_cross_left
    idxs_cross_left = sorted( set(idxs_cross_left) ) # sort the idx_cross_left
    #### form a list of triple (idx, dend, dstart) for each idx given the edge
    tri_idx_dend_dstart = list()
    if True: #edge in 's-+':
        tri_idx_dend_dstart.extend( [(it, 1, 0) for it in idxs_on] )
    if edge == '-':
        tri_idx_dend_dstart.extend( [(it, 1, 1)  for it in idxs_cross_left] )
    elif edge == '+':
        tri_idx_dend_dstart.extend( [(it, 2, 0)  for it in idxs_cross_left] )
    tri_idx_dend_dstart = sorted(tri_idx_dend_dstart, key=lambda it: it[0]) # sort w.r.t. the first element
    #### form a list of subarrays for each segment
    all_lst = [xs]
    all_lst.extend(args) # add args to all_lst
    all_lst_of_lst = [ [lst[idx1+dstart1 : idx2+dend2] for (idx1, dend1, dstart1), (idx2, dend2, dstart2) in zip(tri_idx_dend_dstart[:-1], tri_idx_dend_dstart[1:]) ] for lst in all_lst]
    if len(args) == 0:
        return all_lst_of_lst[0]
    return all_lst_of_lst
def select_subarrays(xmin, xmax, xs, *args, edge='i'):
    """
    e.g.,
        >>> x_segments = select_subarrays(xmin=10, xmin=30, xs)
        >>> x_segments, y_segmemts, z_segments  = select_subarrays(xmin=10, xmin=30, xs, ys, zs)
    """
    if xmin>=xmax:
        return [tuple() for it in range(len(args)+1) ]
    ########################################################
    lst_of_segments = split_arrays((xmin, xmax), xs, *args, edge=edge) # lst_of_segments = x_segments OR x_segments, y_segments, z_segments,...
    if len(args) == 0:
        lst_of_segments = [lst_of_segments]
    ########################################################
    is_inside = lambda arr: xmin <= np.mean(arr) <= xmax
    selected_idxs     = [idx_seg for (idx_seg, xseg) in enumerate(lst_of_segments[0]) if is_inside(xseg)] #
    selected_segments = [ [it_segments[idx] for idx in selected_idxs] for it_segments in lst_of_segments]
    if len(args) == 0:
        return selected_segments[0]
    return selected_segments


class RandFunc1DGenerator:
    """
    A class to generate a 1D random function.
    The function can be used to generate a 1D time series.
    """
    def __init__(self, t0, nt, dt=1.0, outside_value=0.0):
        """
        """
        self.tstart = t0
        self.nt = nt
        self.dt = dt
        self.time = np.arange(nt)*dt + t0
        self.data_seed = (np.random.rand(nt).astype(np.float32) - 0.5)*2.0 # keep the original noise free data # This will not be changed forever
        #####
        self.data_nf = np.copy(self.data_seed)    # noise free data, will be modified by taper and filter
        self.noise   = np.zeros(nt, dtype=np.float32) # noise data
        self.outside_value = outside_value
        self.vmax = np.max(np.abs(self.data_nf)) # update the max amplitude
        #####
        self.__update_func()
    def set_filter_noise_taper(self, freq_band=None, noise_level=0.0, taper_halfsize=0, taper_order=1):
        """
        Apply bandpass filter and a tapering window. This will modify `self.data_nf`, `self.noise`, and `self.data`.
        freq_band: a tuple of (f1, f2) in Hz.
        taper_halfsize: half size of the tapering window in samples.
        order: the order of the tapering window.
        """
        self.data_nf = np.copy(self.data_seed)  # reset
        if freq_band is not None:
            f1, f2 = freq_band
            iirfilter_f32(self.data_nf, self.dt, 0, 2, f1, f2, 2, 2)
        #auto_cc = np.sum(self.data_nf*self.data_nf)
        self.data_nf *= (1.0/np.max(np.abs(self.data_nf))) # normalize the noise free data)
        self.vmax = np.max(np.abs(self.data_nf)) # update the max amplitude
        #######
        self.noise = (np.random.rand(self.nt).astype(np.float32) - 0.5)*(2.0 * noise_level * self.vmax)
        iirfilter_f32(self.noise, self.dt, 0, 0, 0, self.fmax*0.5, 2, 2)
        if taper_halfsize > 0:
            taper(self.data_nf, taper_halfsize, times=taper_order)
            taper(self.noise,   taper_halfsize, times=taper_order)
        self.__update_func() # will modify `self.data`
    def __update_func(self):
        """
        Generate the interpolation function based on the current time and data. This will modify `self.data` and `self.func`.
        """
        self.data = self.data_nf + self.noise
        self.func = scipy.interpolate.interp1d(self.time, self.data, kind='linear', bounds_error=False, fill_value=self.outside_value)
    def __call__(self, t=None, more_noise_level=0.0, more_noise_freq_band=None, more_noise_taper_halfsize=0, more_noise_taper_order=1):
        if t is None:
            t = self.time
        vs = self.func(t)
        if more_noise_level > 0.0:
            more_noise = (np.random.rand(len(t)).astype(np.float32) - 0.5)
            if more_noise_freq_band is not None:
                f1, f2 = more_noise_freq_band
                iirfilter_f32(more_noise, self.dt, 0, 2, f1, f2, 2, 2)
            else:
                iirfilter_f32(more_noise, self.dt, 0, 0, 0, self.fmax*0.5, 2, 2) # filter the additional noise below half of the Nyquist frequency (close but not exactly)
            more_noise *= (more_noise_level*self.vmax/np.max( np.abs(more_noise) )) # normalize the noise
            #
            if more_noise_taper_halfsize > 0:
                taper(more_noise, more_noise_taper_halfsize, times=more_noise_taper_order)
            vs += more_noise
        return vs
    @property
    def df(self):
        return 1.0 / (self.nt*self.dt)
    @property
    def fmax(self):
        return 0.5/self.dt - self.df
    @staticmethod
    def benchmark():
        t0 = 10.0
        dt = 0.1
        nt = 500
        func = RandFunc1DGenerator(t0, nt, dt, outside_value=0.0)
        func.set_filter_noise_taper( (func.df*2, func.df*3 ), noise_level=0.1, taper_halfsize=int(nt*0.1), taper_order=2)
        #
        #plt.plot(func.time, func.data_seed, 'k--', label='seed')
        plt.plot(func.time, func.data, label='data')
        new_time = np.arange(-10, nt+10)*dt + t0# #np.random.randint(-100, nt+10, 20) * dt +10
        new_data = func(new_time, additional_noise_level=0.0)
        plt.plot(new_time, new_data, lw=4, label='interpolation', zorder=0)
        #
        plt.plot(func.time, func.noise, label='noise')
        plt.plot(func.time, func.data_nf, label='data_nf')
        plt.grid(True, which='both')
        plt.legend()
        plt.show()

##############################################################################################################################
#
##############################################################################################################################
class TSFuncs:
    ######### Purely linear-based functions
    @staticmethod # shift a 1D time series and then cut it to a specific time window.
    def shift_and_cut_array1d(data, tstart, dt, time_shift, new_tstart, new_size, method='fft', fill_value=0.0):
        """
        Shift a `time series` to a specific time window (with cutting).
        In other words, this will shift a `time_series` rightwards by a specific time, and then cut
        the shifted time series into a new time window.

        :param data:        a 1D numpy array starting from `tstart` with a time interval of `dt`.
        :param tstart:      the time of the first sample in the time series.
        :param dt:          the time interval between two samples in the time series.
        :param time_shift:  the time to shift the time series rightwards.
        :param new_tstart:  the start time of the window to cut the shifted time series.
        :param new_size:    the size used to cut the shifted time series.
        :param method:      the method to use for shifting and cutting. 'fft' (default) or 'linear'.
        :param fill_value:  the value to fill for the data points outside the time range defined by the original time series.
                            Only useful if `method`='fft'.  (default is 0.0)

        :return: new_data
            new_data:       a 1D numpy array of the new time series started at `new_tstart`.
        """
        if method == 'fft':
            # split the time_shift into two parts
            shift_part1 = new_tstart - tstart
            shift_part2 = time_shift - shift_part1
            # 1. for shift_part1, we can simply change the `tstart` to wnd_start`, and without need to change the `data`.
            # dummy action
            # 2. for the shift_part2, we use fft method shift the data
            zeropad = int(np.ceil( np.abs(shift_part2)/ dt)) # to avoid circular shift effects
            shifted = TSFuncs.rfft_time_shift(data, dt, shift_part2, zeropad=zeropad )
            if shifted.size >= new_size:
                return shifted[:new_size]
            else:
                new_data = np.full(new_size, fill_value, dtype=data.dtype) # initialize the new data
                new_data[:shifted.size] = shifted
                return new_data
        else:
            # 1. shift the time series precisely to the new tstart
            old_ts = tstart + np.arange(len(data))*dt + time_shift
            # 2. linear interpolate concerning the targeted time shift
            new_ts = np.arange(new_size)*dt + new_tstart
            new_data = np.interp(new_ts, old_ts, data, left=fill_value, right=fill_value)
            return new_data
    @staticmethod # shift a 2D mat (each row is a time series) and then cut it to a specific time window.
    def shift_and_cut_mat2d(mat, tstart, dt, time_shift, new_tstart, new_size,  method='fft', fill_value=0.0):
        """
        Shift each row of a matrix to a specific time window (with cutting).
        In other words, this will shift each row of a 2D numpy array rightwards by specific times, and then cut these
        shifted time series into the same new time window.

        :param mat:         a 2D numpy array, where each row is a time series.
        :param tstart:      the time of the first sample in the time series.
        :param dt:          the time interval between two samples in the time series.
        :param time_shift:  a single value, or a 1D numpy array of time shifts for each row in the
                            matrix `mat` for which the length of `time_shifts` should be equal to the number of rows in `mat`.
        :param new_tstart:  the start time of the window to cut the shifted time series.
        :param new_size:    the size used to cut the shifted time series.
        :param method:      the method to use for shifting and cutting. 'fft' (default) or 'linear'.
        :param fill_value:  the value to fill for the data points outside the time range defined by the original time series.
                            (default is 0.0)

        :return: new_mat
            new_mat:        a 2D numpy array with the same number of rows as `mat`, but with `new_size` columns.
                            Each row is a time series started at `new_tstart`.
        """
        time_shift = np.array(time_shift).flatten()
        if time_shift.size == 1:
            time_shift = np.full(mat.shape[0], time_shift[0])
        ####
        new_ncol = new_size
        new_mat = np.zeros((mat.shape[0], new_ncol), dtype=mat.dtype)
        for irow in range(mat.shape[0]):
            new_mat[irow]  = TSFuncs.shift_and_cut_array1d(mat[irow], tstart, dt, time_shift[irow], new_tstart, new_size, method=method, fill_value=fill_value)
        return new_mat
    @staticmethod
    def benchmark1():
        dt = 0.1
        nrow, nt = 10, 500
        tstart   = 10.0
        tend = tstart + (nt-1)*dt
        mat = np.zeros((nrow, nt), dtype=np.float32)
        ts  = np.arange(nt)*dt + tstart
        #
        func = RandFunc1DGenerator(tstart, nt, dt, outside_value=0.0)
        func.set_filter_noise_taper( (func.df*2, func.df*3 ), noise_level=0.2, taper_halfsize=int(nt*0.1), taper_order=2)
        for irow in range(nrow):
            mat[irow] = func(ts, more_noise_level=0.0)
        vmax = np.max(np.abs(mat))
        ##########
        new_tstart = -30
        shifts = np.random.randint(-200, -100, nrow) * dt + np.random.randint(30, 70, nrow)/100*dt
        new_mat1 = TSFuncs.shift_and_cut_mat2d(mat, tstart, dt, shifts, new_tstart, mat.shape[1], method='fft', fill_value=0.0)
        new_mat2 = TSFuncs.shift_and_cut_mat2d(mat, tstart, dt, shifts, new_tstart, mat.shape[1], method='linear', fill_value=0.0)

        new_tend = new_tstart + dt*(new_mat1.shape[1]-1)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10), )
        ax1.imshow(mat, aspect='auto', extent=(tstart, tend, -0.5, nrow-0.5), origin='lower', cmap='viridis', vmin=-vmax, vmax=vmax, interpolation='None')
        ax1.set_title('Original Matrix')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Row Index')

        ax2.imshow(new_mat1, aspect='auto', extent=(new_tstart, new_tend, -0.5, nrow-0.5), origin='lower', cmap='viridis', vmin=-vmax, vmax=vmax, interpolation='None')
        ax2.plot(shifts+tstart, np.arange(nrow), 'o', color='r')
        ax2.set_title('Shifted and Cut Matrix (FFT)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Row Index')

        ax3.imshow(new_mat2, aspect='auto', extent=(new_tstart, new_tend, -0.5, nrow-0.5), origin='lower', cmap='viridis', vmin=-vmax, vmax=vmax, interpolation='None')
        ax3.set_title('Shifted and Cut Matrix (Linear)')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Row Index')


        ax4.imshow(new_mat2-new_mat1, aspect='auto', extent=(new_tstart, new_tend, -0.5, nrow-0.5), origin='lower', cmap='viridis', vmin=-vmax*0.01, vmax=vmax*0.01, interpolation='None')
        ax4.set_title('Diff (FFT-Linear)')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Row Index')
        plt.show()
    @staticmethod
    def benchmark1_2():
        import sacpy.processing as processing
        from sacpy.processing import TSFuncs
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
        nx = 20 # + np.random.randint(10, 30)
        dt = 1
        func1  = processing.RandFunc1DGenerator(0.0, nx, dt, 0.0)
        func2  = processing.RandFunc1DGenerator(0.0, nx, dt, 0.0)
        func1.set_filter_noise_taper((0.05, 0.3), taper_halfsize=int(nx*0.1), taper_order=4 )
        func2.set_filter_noise_taper((0.05, 0.3), taper_halfsize=int(nx*0.1), taper_order=4 )
        x0  = func1.data + func2.data[::1]*1j
        x0_start = 20
        #TSFuncs.plot_time_series(x0, dt, ax1, ax3, start=x0_start, ls='-', lw=4, color='k', label='Original', zorder=0, alpha=0.5)
        denser_x0, denser_x0_dt = TSFuncs.rfft_interpolate(x0, dt, ny=100000)
        TSFuncs.plot_time_series(denser_x0, denser_x0_dt, ax1, ax3, start=x0_start, ls='-', lw=3, color='gray', alpha=1.0, label='Original')
        ############################################################################################################################
        remaining_time    = np.random.randint(30, 70)/100 * dt
        theoretical_shift = remaining_time - 60 # + (np.random.randint(0, 11)-5) * dt
        TSFuncs.plot_time_series(denser_x0, denser_x0_dt, ax1, ax3, start=x0_start+theoretical_shift, ls='-', lw=3, color='k', alpha=1.0, label='Precisely shifted')
        TSFuncs.plot_time_series(denser_x0, denser_x0_dt, ax2, ax4, start=x0_start+theoretical_shift, ls='-', lw=3, color='k', alpha=1.0, label='Precisely shifted')
        new_tstart = x0_start-60
        new_size = nx
        ############################################################################################################################
        y1 = TSFuncs.shift_and_cut_array1d(x0, x0_start, dt, theoretical_shift, new_tstart, new_size, method='linear')
        TSFuncs.plot_time_series(y1, dt, ax1, ax3, start=new_tstart, ls='-', lw=0.0, marker='o', color='b', alpha=1.0, label='linear shifted&cut')
        TSFuncs.plot_time_series(y1, dt, ax2, ax4, start=new_tstart, ls='-', lw=0.0, marker='o', color='b', markersize=6, label='linear shifted&cut', zorder=0, alpha=0.9)
        y2, dy = TSFuncs.rfft_interpolate(y1, dt, ny=100000)
        TSFuncs.plot_time_series(y2, dy, ax2, ax4, start=new_tstart, ls='-', lw=1, color='b', alpha=1.0)
        ############################################################################################################################
        y1 = TSFuncs.shift_and_cut_array1d(x0, x0_start, dt, theoretical_shift, new_tstart, new_size, method='fft')
        TSFuncs.plot_time_series(y1, dt, ax1, ax3, start=new_tstart, ls='-', lw=0.0, marker='o', color='r', alpha=1.0, label='fft shifted&cut')
        TSFuncs.plot_time_series(y1, dt, ax2, ax4, start=new_tstart, ls='-', lw=0.0, marker='o', color='r', markersize=6, label='fft shifted&cut', zorder=0, alpha=0.9)
        y2, dy = TSFuncs.rfft_interpolate(y1, dt, ny=100000)
        TSFuncs.plot_time_series(y2, dy, ax2, ax4, start=new_tstart, ls='-', lw=1, color='r', alpha=1.0)
        #############################
        #ax2.set_xlim([20, 30])
        #ax4.set_xlim([20, 30])
        #############################
        for ax in (ax1, ax2, ax3, ax4):
            ax.set_ylim((-1.2, 1.2) )
            ax.grid(True)
        ax1.set_title(f'Time Series (Real part)\nshifted rightwards by {theoretical_shift} s')
        ax3.set_title('Time Series (Imaginary part)')
        ax3.legend(loc=(0, 1.01), ncols=2)
        ax4.legend(loc=(0, 1.01), ncols=2)
        for ax in (ax3, ax4):
            ax.set_xlabel('Time (s)')
        plt.show()

    ######### Plot functions
    @staticmethod
    def plot_rfft_spec(spec, df, ax_real, ax_imag, **kwargs):
        """
        Plot the real and imaginary parts of a real FFT spectrum.
        :param spec:      The FFT spectrum, which is a complex numpy array.
        :param df:        The frequency interval. (df=1/(nt*dt))
        :param ax_real:   The axis for the real part of the spectrum.
        :param ax_imag:   The axis for the imaginary part of the spectrum.
        :param kwargs:    Additional keyword arguments for `pyplot.plot(...)`.
        """
        if 'markeredgewidth' not in kwargs:
            kwargs['markeredgewidth'] = 0.0
        n = spec.size
        fx = np.arange(n)*df
        ax_real.plot(fx, spec.real, **kwargs)
        ax_imag.plot(fx, spec.imag, **kwargs)
    @staticmethod
    def plot_fft_spec(spec, df, ax_real, ax_imag, **kwargs):
        """
        Plot the real and imaginary parts of a FFT spectrum.
        :param spec:      The FFT spectrum, which is a complex numpy array.
        :param df:        The frequency interval. (df=1/(nt*dt))
        :param ax_real:   The axis for the real part of the spectrum.
        :param ax_imag:   The axis for the imaginary part of the spectrum.
        :param kwargs:    Additional keyword arguments for `pyplot.plot(...)`.
        """
        if 'markeredgewidth' not in kwargs:
            kwargs['markeredgewidth'] = 0.0
        n  = spec.size
        fx = np.arange(n)*df
        fmax = df*n
        half_fmax = fmax / 2
        fx = np.where(fx <= half_fmax, fx, fx - fmax)  # Shift the frequency axis to center around zero
        #
        #fx   = np.roll(fx, n-n//2-1)
        #fx = fft_freqs(n, 1./(df*n), zero_to_fmax=True)
        #fx = np.roll(fx, n-n//2-1)  # Shift the frequency axis to center around zero
        #spec = np.roll(spec, n-n//2-1)
        fx, spec = TSFuncs.roll_fft_freqs(spec, 1./(df*n))  # Roll the frequencies and spectrum so that zero frequency is at the center
        idxs = fx >= 0
        kwargs['alpha'] = 1.0
        ax_real.plot(fx[idxs], spec.real[idxs], **kwargs)
        ax_imag.plot(fx[idxs], spec.imag[idxs], **kwargs)
        #idxs = fx >= 0
        kwargs.pop('label')
        kwargs['alpha'] = 0.3
        ax_real.plot(fx, spec.real, **kwargs)
        ax_imag.plot(fx, spec.imag, **kwargs)
    @staticmethod
    def plot_time_series(x, dt, ax_real, ax_imag=None, start=0.0, **kwargs):
        """
        Plot a time series in the real and imaginary parts.
        :param x:         The time series, which is a 1D numpy array.
        :param dt:        The time interval of the time series.
        :param ax_real:   The axis for the real part of the time series.
        :param ax_imag:   The axis for the imaginary part of the time series. (default is None, meaning no imaginary part will be plotted)
        :param start:     The start time of the time series. (default is 0.0)
        :param kwargs:    Additional keyword arguments for `pyplot.plot(...)`.
        """
        if 'markeredgewidth' not in kwargs:
            kwargs['markeredgewidth'] = 0.0
        tx = np.arange(x.size) * dt + start
        ax_real.plot(tx, np.real(x),  **kwargs)
        if ax_imag is not None:
            ax_imag.plot(tx, np.imag(x),  **kwargs)
    @staticmethod
    def plot_mat2d(mat, dt, dx, ax, tstart=0.0, xstart=0.0, orientation='horizontal', label_col=None, label_row=None, title=None, **kwargs):
        """
        Plot (imshow) a 2D matrix (each row is a time series).
        The 0th dimension of the matrix is the spatial dimension (X), and the 1st dimension is the time dimension (T).

        :param mat:         The 2D matrix, where each row is a time series.
        :param dt:          The time interval of the time series. (step in the 1st dimension of the matrix)
        :param dx:          The spatial interval of the time series. (step in the 0th dimension of the matrix)
        :param ax:          The axis for plotting.
        :param tstart:      The start of the time dimension (the 1st dimension). (default is 0.0)
        :param xstart:      The start position of the spatial dimension (the 0th dimension). (default is 0.0)
        :param orientation: Orientation of the time axis ('horizontal' or 'vertical'). (default is 'horizontal')
        :param label_col:   Label for the time axis (the 1st dimension). (default is None)
        :param label_row:   Label for the spatial axis (the 0th dimension). (default is None)
        :param title:       Title for the plot. (default is None)
        :param kwargs:      Additional keyword arguments for `pyplot.imshow(...)`.
        """
        if orientation == 'vertical':
            TSFuncs.plot_mat2d(mat.T, dx, dt, ax, tstart=xstart, xstart=tstart, label_col=label_row, label_row=label_col, title=title, **kwargs)
        elif orientation == 'horizontal':
            if 'aspect' not in kwargs:
                kwargs['aspect'] = 'auto'
            if 'origin' not in kwargs:
                kwargs['origin'] = 'lower'
            #########
            tend = tstart + (mat.shape[1]-1)*dt
            xend = xstart + (mat.shape[0]-1)*dx
            if kwargs['origin'] == 'lower':
                extent = (tstart-0.5*dt, tend+0.5*dt, xstart-0.5*dx, xend+0.5*dx) # pixel centers
            else:
                extent = (tstart-0.5*dt, tend+0.5*dt, xend+0.5*dx, xstart-0.5*dx) # pixel centers
            #########
            ax.imshow(mat, extent=extent, **kwargs)
            if tend>tstart:
                ax.set_xlim((tstart, tend) )
            if xend>xstart:
                ax.set_ylim((xstart, xend) )
            if label_col is not None:
                ax.set_xlabel(label_col)
            if label_row is not None:
                ax.set_ylabel(label_row)
            if title is not None:
                ax.set_title(title)
        else:
            raise ValueError(f"Invalid orientation: {orientation}. It should be 'horizontal' or 'vertical'.")
    @staticmethod
    def plot_mat2d_waveforms(mat, dt, ax, tstart=0.0, xs=None, scale=1.0, orientation='horizontal', invert_yaxis=False, label_col=None, label_row=None, title=None, **kwargs):
        """
        Plot each row of a 2D matrix (each row is a time series).
        The 0th dimension of the matrix is the spatial dimension (X), and the 1st dimension is the time dimension (T).

        :param mat:         The 2D matrix, where each row is a time series.
        :param dt:          The time interval of the time series. (step in the 1st dimension of the matrix)
        :param ax:          The axis for plotting.
        :param tstart:      The start of the time dimension (the 1st dimension). (default is 0.0)
        :param xs:          The spatial positions for each row of the matrix. If None, it will be set to the row index. (default is None)
        :param scale:       Scale factor for the spatial axis (to separate the waveforms). (default is 1.0)
        :param orientation: Orientation of the time axis ('horizontal' or 'vertical'). (default is 'horizontal')
        :param invert_yaxis:If True, invert the yaxis of the axis. (default is False)
        :param label_col:   Label for the time axis (the 1st dimension). (default is None)
        :param label_row:   Label for the spatial axis (the 0th dimension). (default is None)
        :param title:       Title for the plot. (default is None)
        :param kwargs:      Additional keyword arguments for `pyplot.imshow(...)`.
        """
        tend = tstart + (mat.shape[1]-1)*dt
        nx, nt = mat.shape
        xs = np.arange(nx) if xs is None else np.array(xs).flatten()
        ts = np.arange(nt)*dt + tstart
        dim0, dim1 = (0, 1) if (orientation == 'horizontal') else (1, 0) #
        for ix in range(nx):
            wv = mat[ix]*scale + xs[ix]  # scale the waveform to the spatial position
            pair = [ts, wv]
            ax.plot(pair[dim0], pair[dim1], **kwargs)
        #################################################################
        if orientation == 'horizontal':
            ax.set_xlim((tstart, tend) )
            ax.set_ylim((xs[0]-1, xs[-1]+1) )
        else:
            ax.set_xlim((xs[0]-1, xs[-1]+1) )
            ax.set_ylim((tstart, tend) )
            label_col, label_row = label_row, label_col  # swap labels for vertical orientation
        if invert_yaxis:
            ax.invert_yaxis()
        #################################################################
        if label_col is not None:
            ax.set_xlabel(label_col)
        if label_row is not None:
            ax.set_ylabel(label_row)
        if title is not None:
            ax.set_title(title)
    @staticmethod
    def benchmark2():
        nx = np.random.randint(1, 30)
        dt = np.random.randint(1, 10)/2.0
        func_gen = RandFunc1DGenerator(0.0, nx, dt, outside_value=0.0)
        func_gen.set_filter_noise_taper((0.001, 0.5*func_gen.fmax), 0.0, 0.5, 4)
        x = func_gen(more_noise_level=0.01)
        fft_sx = scipy.fft.fft(x, norm='forward')
        rfft_sx = scipy.fft.rfft(x, norm='forward')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        TSFuncs.plot_time_series(x, dt, ax3, ax4, marker='s', ls='--', markersize=9, lw=0.2, color='gray', label='time series')
        TSFuncs.plot_fft_spec(fft_sx, func_gen.df, ax1, ax2, marker='s', ls='-', markersize=15, lw=1.5, color='gray', label='fft spectrum')
        TSFuncs.plot_rfft_spec(rfft_sx, func_gen.df, ax1, ax2, marker='s', ls='-', markersize=9, lw=1.5, color='k', label='rfft spectrum')
        ax1.set_title('Spectrum (real part)')
        ax2.set_title('Spectrum (imaginary part)')
        ax3.set_title('Time Series (Real part)')
        ax4.set_title('Time Series (Imaginary part)')
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        plt.show()
        pass
    @staticmethod
    def benchmark2_1():
        t0, dt, nt = 0, 1, 1000
        fun_gen = RandFunc1DGenerator(t0, nt, dt, outside_value=0.0)
        fun_gen.set_filter_noise_taper((0.01, 0.5*fun_gen.fmax), noise_level=0.1, taper_halfsize=int(nt*0.1), taper_order=2)
        #
        x0, dx, nrow = 0, 1.0, 50
        mat = np.zeros((nrow, nt), dtype=np.float32)
        for irow in range(nrow):
            mat[irow] = fun_gen(more_noise_level=0.1)
        mat[10:,:] = 0.0
        mat[:,600:] = 0.0
        ####
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        TSFuncs.plot_mat2d(mat, dt, dx, ax1, tstart=t0, xstart=x0, cmap='viridis', vmin=-1.0, vmax=1.0, interpolation='None', aspect='auto', label_col='Time (s)', label_row='X', title='transpose=False', origin='lower')
        TSFuncs.plot_mat2d(mat, dt, dx, ax2, tstart=t0, xstart=x0, cmap='viridis', vmin=-1.0, vmax=1.0, interpolation='None', aspect='auto', label_col='Time (s)', label_row='X', title='transpose=True',  origin='lower', orientation='vertical', )
        ####
        TSFuncs.plot_mat2d_waveforms(mat, dt, ax3, tstart=t0, xs=np.arange(nrow), color='k', ls=':', lw=0.8, label_col='Time (s)', label_row='X', title='transpose=False', scale=0.5, invert_yaxis=True)
        TSFuncs.plot_mat2d_waveforms(mat, dt, ax4, tstart=t0, xs=np.arange(nrow), color='k', ls=':', lw=0.8, label_col='Time (s)', label_row='X', title='transpose=True',  orientation='vertical', scale=0.5, invert_yaxis=True)
        plt.show()

    ######### FFT constants calculation functions
    @staticmethod
    def fft_constants(n, dt):
        """
        Calculate the Fourier transform constants based on the time interval and number of points.
        :param n:  Number of points in the time series.
        :param dt: Time interval of the time series.

        :return:    (T, df, fc, fmax)
                    T:    Total time duration of the time series.
                    df:   Frequency interval, calculated as 1.0/T.
                    fc:   Nyquist frequency, which is half of the maximum frequency, half of the fmax.
                    fmax: Maximum frequency, calculated as df*n.
        """
        T    = n*dt
        df   = 1.0/T
        fmax = df*n
        fc   = 0.5 * fmax # Nyquist frequency. Half of the maximum frequency.
        return T, df, fc, fmax
    @staticmethod
    def fft_freqs(n, dt, zero_to_fmax=False):
        """
        Calculate the frequencys based on the time interval and number of points.
        :param n:            Number of points in the time series.
        :param dt:           Time interval of the time series.
        :param zero_to_fmax: If True, return frequencies from 0 to fmax, otherwise from -fmax/2 to fmax/2.

        :return: fs (an ndarray for the frequencys for fft.)
        """
        T, df, fc, fmax = TSFuncs.fft_constants(n, dt)
        fs_idxs = np.arange(n)
        if not zero_to_fmax:
            l = n//2 + 1
            fs_idxs[l:] -= n #
        fs = fs_idxs * df
        return fs
    @staticmethod
    def rfft_freqs(n, dt):
        """
        Calculate the frequencies for rfft based on the time interval and number of points.
        :param n:  Number of points in the time series.
        :param dt: Time interval of the time series.

        :return: fs (an ndarray for frequencies for rfft.)
        """
        T, df, fc, fmax = TSFuncs.fft_constants(n, dt)
        fs = np.arange(n//2 + 1) * df
        return fs
    @staticmethod
    def roll_fft_freqs(fft_spec, dt):
        """
        Return the rolled frequencies and spectrum so that the zero frequency is at the center.

        :param spec: The spectrum (generated using fft but not rfft!)
        :param dt:   The time interval of the time series.

        :return:    (rolled_frequencies, rolled_spectrum)
                    rolled_frequencies: The frequencies rolled so that the zero frequency is at the center.
                    rolled_spectrum:    The spectrum corresponding to the rolled frequencies.
        """
        n = fft_spec.size
        rollsize = n-(n//2+1)
        fs = TSFuncs.fft_freqs(n, dt, zero_to_fmax=False)  # Get the frequencies
        return np.roll(fs, rollsize), np.roll(fft_spec, rollsize)

    ######### FFT-based time shift
    """ Time Shift in Frequency Domain
        #######################################################################################################################################################
        0. Fourier Transform Basics
            Let's consider a function x(t) and its Fourier transform sx(f):
                sx(f) = integral{  x(t) * exp(-j*2pi*f*t) dt }
                x(t)  = integral{ sx(f) * exp( j*2pi*f*t) df }
            In the discrete case, we have:
                x[i] = x(i*dt), where dt is the time interval, and i=0,1,2,...,nx-1, and nx*dt = T, and
                sx[i] = sx(i*df), where df is the frequency interval, and i=0,1,2,...,nx-1, and df = 1/T.
            Some constants:
                T  = nx*dt,
                df = 1/T = 1/(nx*dt) <==> df*dx = 1/nx, or 1/(df*dx) = nx
            The Fourier transform sx(f) is periodic with period 1/dt, or 1/df, so that we can have:
                sx[i] = sx(i*df)
                    = integral{  x(t)    * exp(-j*2pi*(i*df)*t) dt }
                    = sum_l{     x(l*dt) * exp(-j*2pi*(i*df)*(l*dt)) }, where df*dt = 1/nx
                    = sum_l{     x[l]    * exp(-j*2pi*i*l/nx) }
                    = sum_l{     x[l]    * exp(-j*2pi*(l/nx))*i }
                x[i]  = x(i*dt)
                    = integral{ sx(f)    * exp( j*2pi*f*(i*dt)) df }
                    = sum_l{    sx(l*df) * exp( j*2pi*(l*df)*(i*dt)) }, where df*dt = 1/nx
                    = sum_i{    sx[i]    * exp( j*2pi*l*i/nx) }
                    = sum_l{    sx[i]    * exp( j*2pi*(l/nx)*i) }
        #######################################################################################################################################################
        1. Time shift.
        If we want to shift the function x(t) rightwards by a time interval a, that means to create a new function y(t) so that:
                y(t) = x(t-a)
        Then, let's consider the Fourier transform of y(t):
                sy(f) = integral{ y(t) * exp(-j*2pi*f*t) dt }
                    = integral{ x(t-a) * exp(-j*2pi*f*t) dt }
                    = integral{ x(t') * exp(-j*2pi*f*(t'+a)) dt' }  # where t' = t-a
                    = exp(-j*2pi*f*a) * integral{ x(t') * exp(-j*2pi*f*t') dt' }
                    = exp(-j*2pi*f*a) * sx(f)
        In discrete case, we have:
                sy[i] = sy(i*df) = exp(-j*2pi*(i*df)*a) * sx(i*df)

        Another method to derive sy[i]:
                y[i] = y(i*dt) = x(i*dt - a)
                            = integral{ sx(f)    * exp( j*2pi*f*(i*dt-a)) df }
                            = integral{ sx(f)    * exp(-j*2pi*f*a)      * exp( j*2pi*f*(i*dt))   df }
                            = integral{[sx(f)    * exp(-j*2pi*f*a)]     * exp( j*2pi*f*(i*dt))   df }
                            = sum_l{     [sx(l*df) * exp(-j*2pi*(l*df)*a)]* exp( j*2pi*(l*df)*(i*dt)) }
                            = sum_l{     [sx[l]    * exp(-j*2pi*(l*df)*a)]* exp( j*2pi*(l/nx)*i) }
        Also, we have:
                y[i]  =          sum_l{      sy[l]                           * exp( j*2pi*(l/nx)*i) }
        Through comparison, we can get:
                sy[l] = sx(l*df) * exp(-j*2pi*(l*df)*a)
    """
    @staticmethod
    def rfft_time_shift(x, dt, a, zeropad=0):
        """
        Shift the time series x by a time interval a, using FFT method.
        After shifting, the start of the new time series and its size will be the same as the input x.
        It is highly recommended to taper the input time series x before applying this shift to avoid edge effects!

        :param x:       time series to be shifted. (x can be real or complex)
        :param dt:      time interval of the time series x
        :param a:       time interval to shift the time series x. (a is in the same unit as dt)
        :param zeropad: number of zeros to pad the input signal (default is 0) before applying this shift.
                        (this is useful to avoid circular shifting effects due to fft method).
                        (please consider a safe zeropad is that greater than `ceil( abs(a)/dt )` ).

        :return:   the shifted time series y.
        """
        if np.iscomplexobj(x):
            y_real = TSFuncs.rfft_time_shift(np.real(x), dt, a, zeropad)
            y_imag = TSFuncs.rfft_time_shift(np.imag(x), dt, a, zeropad)
            y = y_real + 1j * y_imag
            return y
        if zeropad < 0:
            raise ValueError(f"Invalid zeropad value: {zeropad}. It should be a non-negative integer.")
        nx = len(x)
        if zeropad > 0:
            nx += zeropad
            nx += (512- nx % 512)  # Make sure nx is even for rfft
        #############
        T, df, fc, fmax = TSFuncs.fft_constants(nx, dt)
        sx = scipy.fft.rfft(x, n=nx, norm='forward')
        fs = np.arange(sx.size) * df
        coefs = np.exp(-1j * 2 * np.pi * fs * a)      # Coefficients for time shift
        sy = sx * coefs
        y  = scipy.fft.irfft(sy, n=nx, norm='forward')  # Inverse rfft to get the shifted time series
        y  = y[:len(x)]  # Remove additional elements outside the original x's time window
        return y
    @staticmethod
    def fft_time_shift(x, dt, a, zeropad=0):
        """
        Shift the time series x by a time interval a, using FFT method
        After shifting, the start of the new time series and its size will be the same as the input x.
        It is highly recommended to taper the input time series x before applying this shift to avoid edge effects!

        :param x:  time series to be shifted. (x can be real or complex)
        :param dt: time interval of the time series x
        :param a:  time interval to shift the time series x. (a is in the same unit as dt)
        :param zeropad: number of zeros to pad the input signal (default is 0) before applying this shift.
                        (this is useful to avoid circular shifting effects due to fft method).
                        (please consider a safe zeropad is that greater than `ceil( abs(a)/dt )` ).

        :return:   the shifted time series y
        """
        if zeropad < 0:
            raise ValueError(f"Invalid zeropad value: {zeropad}. It should be a non-negative integer.")
        nx = len(x)
        if zeropad > 0:
            nx += zeropad
            nx += (512- nx % 512)  # Make sure nx is even for rfft
        #############
        T, df, fc, fmax = TSFuncs.fft_constants(nx, dt)
        sx = scipy.fft.fft(x, n=nx, norm='forward')
        fs = TSFuncs.fft_freqs(nx, dt, False)     # will be 0 to fc and then -fc to 0!
        coefs = np.exp(-1j * 2 * np.pi * fs * a)  # Coefficients for time shift
        sy = sx * coefs
        y  = scipy.fft.ifft(sy, n=nx, norm='forward')
        y  = y[:len(x)]  # Remove additional elements outside the original x's time window
        if not np.iscomplexobj(x):
            y = y.real
        return y
    @staticmethod
    def benchmark4():
        import sacpy.processing as processing
        from sacpy.processing import TSFuncs
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        nx = 100 + np.random.randint(10, 30)
        dt = 1
        func1  = processing.RandFunc1DGenerator(0.0, nx, dt, 0.0)
        func2  = processing.RandFunc1DGenerator(0.0, nx, dt, 0.0)
        func1.set_filter_noise_taper((0.05, 0.3), taper_halfsize=int(nx*0.1), taper_order=4 )
        func2.set_filter_noise_taper((0.05, 0.3), taper_halfsize=int(nx*0.1), taper_order=4 )
        x0  = func1.data + func2.data[::1]*1j
        TSFuncs.plot_time_series(x0, dt, ax1, ax3, ls='-', lw=4, color='k', label='Original', zorder=0, alpha=0.5)
        TSFuncs.plot_time_series(x0, dt, ax2, ax4, ls='-', lw=0.0, marker='v', color='k', label='Original', zorder=0, alpha=0.9)
        y2, dy = TSFuncs.rfft_interpolate(x0, dt, ny=100000)
        TSFuncs.plot_time_series(y2, dy, ax2, ax4, ls='--', lw=0.5, color='k', alpha=1.0)
        ###########################
        remaining_time    = np.random.randint(30, 70)/100 * dt
        theoretical_shift = remaining_time # + (np.random.randint(0, 11)-5) * dt
        zeropad =int( np.ceil( abs(theoretical_shift)/dt ) )
        ########################### fft + zeropad=0
        y1 = TSFuncs.fft_time_shift(x0, dt, theoretical_shift, zeropad=0)
        TSFuncs.plot_time_series(y1, dt, ax1, ax3, ls='-', lw=4, color='b', alpha=1.0, label='rfft Shifted (pad=0)')
        TSFuncs.plot_time_series(y1, dt, ax2, ax4, ls='-', lw=0.0, marker='s', color='b', alpha=1.0, label='rfft Shifted (pad=0)')
        y2, dy = TSFuncs.rfft_interpolate(y1, dt, ny=100000)
        TSFuncs.plot_time_series(y2, dy, ax2, ax4, ls='--', lw=0.5, color='k', alpha=1.0)
        ###########################  fft + zeropad!=0
        y1 = TSFuncs.fft_time_shift(x0, dt, theoretical_shift, zeropad=zeropad)
        TSFuncs.plot_time_series(y1, dt, ax1, ax3, ls='-', lw=4, color='g', alpha=0.6, label=f'rfft Shifted (pad={zeropad})')
        TSFuncs.plot_time_series(y1, dt, ax2, ax4, ls='-', lw=0.0, marker='s', color='g', alpha=1, label=f'rfft Shifted (pad={zeropad})')
        y2, dy = TSFuncs.rfft_interpolate(y1, dt, ny=100000)
        TSFuncs.plot_time_series(y2, dy, ax2, ax4, ls='--', lw=0.5, color='k', alpha=1.0)
        ########################### rfft + zeropad=0
        y1 = TSFuncs.fft_time_shift(x0, dt, theoretical_shift, zeropad=0)
        TSFuncs.plot_time_series(y1, dt, ax1, ax3, ls='-', lw=1.5, color='C1', alpha=1, label='fft Shifted (pad=0)')
        TSFuncs.plot_time_series(y1, dt, ax2, ax4, ls='-', lw=0.0, marker='s', color='C1', alpha=1, label='fft Shifted (pad=0)')
        y2, dy = TSFuncs.rfft_interpolate(y1, dt, ny=100000)
        TSFuncs.plot_time_series(y2, dy, ax2, ax4, ls='--', lw=0.5, color='k', alpha=1.0)
        ###########################  fft + zeropad!=0
        y1 = TSFuncs.fft_time_shift(x0, dt, theoretical_shift, zeropad=zeropad)
        TSFuncs.plot_time_series(y1, dt, ax1, ax3, ls='--', lw=1.5, color='C5', alpha=1, label=f'fft Shifted (pad={zeropad})')
        TSFuncs.plot_time_series(y1, dt, ax2, ax4, ls='-',  lw=0.0, marker='s', color='C5', alpha=1, label=f'fft Shifted (pad={zeropad})')
        y2, dy = TSFuncs.rfft_interpolate(y1, dt, ny=100000)
        TSFuncs.plot_time_series(y2, dy, ax2, ax4, ls='--', lw=0.5, color='k', alpha=1.0)
        #############################
        ax2.set_xlim([20, 30])
        ax4.set_xlim([20, 30])
        #############################
        ax1.set_title(f'Time Series (Real part)\nshifted rightwards by {theoretical_shift} s')
        ax3.set_title('Time Series (Imaginary part)')
        ax3.legend(loc=(0, 1.01), ncols=2)
        ax4.legend(loc=(0, 1.01), ncols=2)
        for ax in (ax3, ax4):
            ax.set_xlabel('Time (s)')
        plt.show()
        pass

    ######### FFT-based interpolation functions
    """ Interpolation using FFT method
        ######################################################################################################################################################
        # The Algorithm for fft_interpolate(x, dx, ny) is as follows:
        #
        # 0. Some basics:
            #    For a time series x with size nx. Its spectrum (using fft) will has the size nx as well.
        # 0.1 Some critical parameters:
            #               Total duration (or implicit period):  T = nx*dx,    (Note: T is a constant for whatever interpolation to whatever new size ny)
            #               Spetrum's frequency interval         df = 1/T        = 1/(nx*dx)    <==> df*dx = 1/nx, or 1/(df*dx) = nx
            #                                                                   (Note: df is also a constant for whatever interpolation to whatever new size ny)
            #               Nyquist frequency:                   fc = 0.5/dx     = 1/2 *nx*df
            #
        # 0.2 Nyquist frequency
            #    Obviously, the sx will contain the Nyquist frequency if nx is even, or will not if nx is odd:
            #    (1) If nx = 2n, then fc = 1/2 *2n*df = n*df, which means the sx[n+1]. However, this is the only data point at positive Nyquist frequency,
            #        but sx does not have a data point at negative Nyquist frequency!
            #        In fft implementation standard, the sx[n+1] is the double of the value of the positive Nyquist frequency! (need to double-check!).
            #    (2) If nx = 2n+1, then fc = 1/2 *(2n+1)*df = (n+0.5)*df, which does not correspond to any data point in the spectrum sx.
            #        In this case, the sx[n] corresponds to the frequency n*df), and sx[n+1] the frequency (n+1)*df (or -n*df).
            #
            #    Also, each data point sx[ix] corresponds to the frequency ix*df (i=0,1,2,...,nx-1), so that the frequency range is 0 to (nx-1)*df.
            #    However, please note that fc (=1/2 *nx*df) is the Nyquist frequency, so that any sx's frequencies l*df larger than fc in fact denote
            #    negative frequencies:
            #         l*df --> 2*fc - l*df = nx*df - l*df = (n-l)*df, when l>n
            #    So we can have frequencies: [-n*df, (-n+1)*df, (-n+2)*df,..., -2*df, -df, 0, df, 2*df,..., (n-1)*df, n*df] for odd  nx=2n+1, or
            #                                       [(-n+1)*df, (-n+2)*df,..., -2*df, -df, 0, df, 2*df,..., (n-1)*df, n*df] for even nx=2n.
        # 0.3 rfft for real time series
            #    We can also consider the rrft spectrum of a real time series x, for which its spectrum will present conjugate symmetry w.r.t. zero
            #    frequency. So we only need the positive frequencies, and don't need the negative frequencies.
            #    It is obvious now that the rfft spectrum has the size nx//2+1, and the frequencies are:
            #                                [0, df, 2*df,..., (nx//2+1)*df] for both odd and even nx.
            #
        # 0.4 Aliasing effect
            #    Let us consider a sampling system with sampling rate fs = 2*fc, or the time interval dx = 0.5/fc = 1/fs.
            #    For an input signal at frequency f, we can denote it as exp{j*2pi*f*t}, then we can get discrete time series:
            #                      x[ix] = exp{ j*2pi * f * (ix * dx) }
            #     Obviously, frequencies are periodic modulo 2pi, so:
            #                      x[ix] = exp{ j*2pi * f * (ix * dx) + j*2pi*ix * k }
            #                            = exp{ j*2pi * ix * dx * f + j*2pi*ix*dx * (k/dx) }
            #                            = exp{ j*2pi * ix * dx *(f + k/dx ) }
            #                            = exp{ j*2pi * ix * dx *(f + k*fs ) }, where fc = 0.5/dx or fs=2fc=1/dx
            #     This denotes an equivalent frequency with at frequency (f+2k*fs).
            #     Then, we want to find an equivalent frequency fa in the range [0, fs), and this can be done by:
            #                         fa = f % fs,
            #     or, we want to find an equivalent frequency fa in the range [-fc, fc), and this can be done by:
            #                         fa = (f + fc) % (2*fc) - fc
            #     Here, we use a function g(x) = (x+c)%(2c) - c to convert values in range [0, 2c) to the range [-c, c).
            #                             g(x) = x+c -c     = x,    if 0<=x<c
            #                                  = x+c -2c -c = x-2c, if c<=x<2c
            #
            #     In summary, a sigal with frequency f will be aliased to a frequency fa in the range  [0, 2fc) by fa = f % (2*fc),
            #     or in the range [-fc, fc) by fa = (f + fc) % (2*fc) - fc.
            #
            #
        # 1.0 For ny>nx
            #     Now we need to lengthen the spectrum sx to sy through padding zeros.
        # 1.1 For ODD nx { nx = 2n + 1 or n = nx//2 }
        # 1.1.1 fft version
            #     The original spectrum sx: (We omit df in the frequency, as it is a constant. Or, we just take normalized df=1.0)
            #                       sx:  #---...---#---#---#---x===x===x===x===...===x
            #                frequency: -n   ...  -3  -2  -1   0   1   2   3   ...   n,  where nx=2n+1
            #     Please note the element sx[n] and sx[-n]. They are not at the Nyquist frequency (fc = 1/2 *nx = n+0.5)
            #     The physical layout of the original sx in the memory is:
            #                                                  x===x===x===x===...===x~~~#---...---#---#---#
            #                                       frequency: 0   1   2   3   ...   n  -n   ...  -3  -2  -1
            #                                     index in sx: 0   1   2   3   ...   n   n+1 ...  ...      nx-1
            #     By appending (ny-nx) zeros, we would have the spectrum sy for the expected time series y (size ny) after interpolation.
            #              sy: ...~~~0~~~x---...---#---#---#---x===x===x===x===...===x~~~0~~~0~~~0~~~...
            #       frequency:          -n   ...  -3  -2  -1   0   1   2   3   ...   n
            #     Its physical layout in the memory is:
            #                                              sy: x===x===x===x===...===x~~~0~~~0~~~... ...~~~0~~~0~~~#---...---#---#---#
            #                                     index in sx: 0   1   2   3   ...   n   |<--- ny-nx zeros  -->|   n+1 ...  ...      nx-1
            #     We can get the spectrum sy:
            #                                              sy =      sx[:n+1]          +      zeros(ny-nx)                 + sx[n+1:]
            #     So, the code should be:
            #
            #                                hn = nx//2+1
            #                                sx = fft(x)
            #                                sy = np.concatenate( [sx[:hn], np.zeros(ny-nx), sx[hn:] ], dtype=sx.dtype) )
            #                                y  = ifft(sy, n=ny)
            #                                                          or
            #                                sx = fft(x)
            #                                sy = np.zeros(ny, dtype=sx.dtype)
            #                                hn = nx//2+1
            #                                sy[:hn]         = sx[:hn]
            #                                sy[ny-(nx-hn):] = sx[hn:]
            #                                y  = ifft(sy)
            #
        # 1.1.2 rfft version for real time series x
            #     Besides, let's consider the rfft's spectrum, for which x is real and sx has conjugate symmetry w.r.t. zero frequency.
            #     So rfft spectrum rfft_x and rfft_sy have the layout in physica memory:
            #                                                  x===x===x===x===...===x
            #                                       frequency: 0   1   2   3   ...   n
            #                                index in rfft_sx: 0   1   2   3   ...   n
            #
            #                                              sy: x===x===x===x===...===x~~~0~~~0~~~... ... ... ... ...~~~0
            #                                index in rfft_sy: 0   1   2   3   ...   n   |<-- (ny//2+1) - nx zeros  -->|
            #
            #      So, code using rfft and irfft should be:
            #                                n  = nx//2
            #                                sx = rfft(x)
            #                                sy = np.zeros(ny//2+1)
            #                                sy[:n+1] = sx
            #                                sy = irfft(sy, n=ny) # need to explicitly tell n=ny
            #
        # 1.2 For EVEN nx { nx = 2n, or n=nx//2 }
        # 1.2.1 fft version
            #     The original spectrum sx: (We again omit df in the frequency, as it is a constant. Or, we just take normalized df=1.0)
            #                           sx:  #---...---#---#---x===x===x===x===...===x
            #                    frequency: -n+1 ...  -2  -1   0   1   2   3   ...   n,  where nx=2n
            #     Please note the element sx[n]. It is at sx's Nyquist frequency (fc = 1/2 *nx = n), and it does not have a correspondant
            #     at index -n (negative Nyquist frequency).
            #     So, when we lengthen the spectrum, we need to care that the new data point at frequency -n, sy(-n), should be same to the
            #     data point at frequency n, sy(n). It is obvious that sy(n) = sy(-n) = 0.5 * sx(n) = 0.5* sx[n].
            #
            #     The physical layout of the original sx in the memory is:
            #                                              sx: x===x===x===x===...===x~~~#---...---#---#
            #                                       frequency: 0   1   2   3   ...   n  -n+1 ...  -2  -1
            #                                     index in sx: 0   1   2   3   ...   n   n+1 ... ...   nx-1
            #     To append zeros, we need to:
            #     (1) add one point at frequency -n (the negative Nyquist frequency), and fix the sy at frequencies -n and n:
            #                     temp:  a---#---...---#---#---x===x===x===x===...===a
            #                frequency: -n  -n+1 ...  -2  -1   0   1   2   3   ...   n,  where nx=2n  and a = 0.5*sx[n]
            #     (2) append the remaining ny-nx-1 zeros.
            #              sy: ...~~~0~~~a---#---...---#---#---x===x===x===x===...===a~~~0~~~0~~~...~~~0
            #       frequency:          -n  -n+1 ...   -2  -1   0   1   2   3        n
            #     Its physical layout in the memory is:
            #                                                                           here we have ny-nx-1 zeros
            #                                              sy: x===x===x===x===...===a~~~0~~~0~~~... ...~~~0~~~0~~~a---x---...---#---#
            #                                       frequency: 0   1   2   3         n                            -n  -n+1      -2  -1
            #                                     index in sx: 0   1   2   3   ...   n   |<-- ny-nx-1 zeros -->|       n+1 ...  ...  nx-1
            #     We can get the spectrum sy:
            #                                              sy =     sx[:n]      +   [a] +    zeros(ny-nx-1)     + [a]   + sx[n+1:]
            #                                              or,
            #                                              sy =     sx'[:n+1]           +   zeros(ny-nx-1)      + sx'[n:],
            #                                                       where sx'[i] = sx[i], except sx'[n] = 0.5*sx[n]
            #     So, the code should be:
            #                                sx = fft(x)
            #                                n  = nx//2
            #                                a  = sx[n]*0.5
            #                                sy = np.concatenate( [sx[:n], [a], np.zeros(ny-nx-1), [a], sx[n+1:] ], dtype=sx.dtype) )
            #                                y  = ifft(sy, n=ny)
            #                                                          or
            #                                sx = fft(x)
            #                                n  = nx//2
            #                                sy = np.zeros(ny, dtype=sx.dtype)
            #                                sx[n]         *= 0.5
            #                                sy[:n+1]       = sx[:n+1]
            #                                sy[ny-(nx-n):] = sx[n:]
            #                                y = ifft(sy)
            #
        # 1.1.2 rfft version for real time series x
            #     Besides, let's consider the rfft's spectrum, for which x is real and sx has conjugate symmetry w.r.t. zero frequency.
            #     So rfft spectrum rfft_x and rfft_sy have the layout in physica memory:
            #                                                  x===x===x===x===...===x
            #                                       frequency: 0   1   2   3   ...   n
            #                                index in rfft_sx: 0   1   2   3   ...   n
            #
            #                                              sy: x===x===x===x===...===a~~~0~~~0~~~... ... ... ... ...~~~0
            #                                index in rfft_sy: 0   1   2   3   ...   n   |<-- (ny//2+1) - nx zeros  -->|
            #
            #      So, code using rfft and irfft should be:
            #                                n  = nx//2
            #                                sx = rfft(x)
            #                                sy = np.zeros(ny//2+1)
            #                                sy[:n+1] = sx
            #                                sy[n]   *= 0.5
            #                                sy       = irfft(sy, n=ny) # need to explicitly tell n=ny
        # 2.  For ny < nx
            #     Now we need to shorten the spectrum sx to sy. Obviously, there will be aliasing effect.
        # 2.1 For ODD nx { nx = 2n + 1 or n = nx//2 }
        # 2.1.1 fft version
            #     The original spectrum sx: (We omit df in the frequency, as it is a constant. Or, we just take normalized df=1.0)
            #                       sx:  #---...---#---#---#---x===x===x===x===...===x
            #                frequency: -n   ...  -3  -2  -1   0   1   2   3   ...   n,  where nx=2n+1
            #     Please note the element sx[n] and sx[-n]. They are not at the Nyquist frequency (fc = 1/2 *nx = n+0.5)
            #     The physical layout of the original sx in the memory is:
            #                                              sx: x===x===x===x===...===x~~~#---...---#---#---#
            #                                  sx's frequency: 0   1   2   3   ...   n  -n   ...  -3  -2  -1
            #                                     index in sx: 0   1   2   3   ...   n   n+1 ...  ...      nx-1
            #
            #     Now, when we resample x to y of size ny, the new Nyquist frequency for time series y will be fcy = 1/2 *ny.
            #     Then a sx[ix] data point at frequency ix will contribute to a sy data point at frequency iy:
            #                                  iy = ix % fsy
            #                                     = ix % (2*fcy), where fcy = 1/2 * ny
            #                                     = ix % ny
            #
            #     So, now each sx's data point will contribute to sy data points at frequencys:
            #                                     index in sx: 0   1   2   3   ...   n         n+1 ...  ...      nx-1
            #                                  sx's frequency: 0   1   2   3   ...   n        -n   ...  -3  -2  -1
            #                             ==>  sy's frequency: 0       2%ny    ...            (-n)%ny  ... (-2)%ny
            #                                                      1%ny    3%n ...   n%ny          ... (-3)%ny (-1)%ny
            #                             ==>  sy's frequency:[0:ny), [0:ny),  ...[0:n%ny]
            #                                                                                [(-n)%ny, ny-1]  ... [0:ny), [0:ny)
            #     So the code should be:
            #                           sx= fft(x)
            #                           sy= zeros(ny)
            #                           n = nx // 2
            #                           pos_part  = sx[:n+1]
            #                           remainder = pos_part % ny
            #                           tidy_size = pos_part.size - remainder
            #                           sy       += pos_part[:tidy_size].reshape((-1, ny)).sum(axis=0)
            #                           if remainder > 0:
            #                               sy[:remainder] += pos_part[tidy_size:]
            #
            #                           neg_part  = sx[n+1:]
            #                           remainder = neg_part % ny
            #                           sy       += neg_part[remainder:].reshape((-1, ny)).sum(axis=0)
            #                           if remainder > 0:
            #                               sy[ny-remainder] = neg_part[:remainder]
            #                           y = ifft(sy)
            #
            #
        # 2.1.2 rfft version for real time series x
            #     Besides, let's consider the rfft's spectrum, for which x is real and sx has conjugate symmetry w.r.t. zero frequency.
            #     So rfft spectrum rfft_x has the layout in physica memory:
            #                                              sx: x===x===x===x===...===x
            #                                index in rfft_sx: 0   1   2   3   ...   n
            #                              positive frequency: 0   1   2   3   ...   n
            #                              negative frequency:    -1  -2  -3   ...  -n
            #                    ==>  sy's positive frequency: 0       2%ny    ...
            #                                                      1%ny    3%n ...   n%ny
            #                    ==>  sy's positive frequency:[0:ny), [0:ny),  ...[0:n%ny]
            #
            #                    ==>  sy's positive frequency: 0 1 2 3 ... ny-1; 0 1 2 3 4 ny-1; 0 1 2 3...
            #
            #     This is only for the positive frequencies. For the negative frequencies, they should be conjugate to what we have formed.
            #                    ==>  sy's negative frequency:        -2%ny    ...
            #                                                     -1%ny   -3%n ...  -n%ny
            #                    ==>  sy's negative frequency:    (ny, 0],... (ny, 0], ...
            #
            #                    ==>  sy's negative frequency:    ny-1 ny-2 ny-3 ... 0; ny-1 ny-2 ny-3 ... 0; ny-1 ...
            #
            #     If ny is ODD (ny=2m+1, ny//2=m):
            #                                index in rfft_sx: 0   1   2    3    ... m-2 m-1 m   m+1 m+2 m+3 ....ny-1; ny ny+1 ny+2... ny+m ny+m+1 ...
            #                    ==>  sy's positive frequency: 0   1   2    3    ... m-2 m-1 m   m+1 m+2 m+3 ... ny-1; 0  1    2   ... m    m+1    ...
            #                    ==>  sy's negative frequency:    ny-1 ny-2 ny-3 ... m+3 m+2 m+1 m   m-1 m-2 ... 1     0; ny-1 ny-2... m+1  m      ...
            #     Only the first half [0, m+1) is useful:
            #                    ==>  sy's positive frequency: 0   1   2    3    ... m-2 m-1 m                       ; 0  1    2   ... m
            #                    ==>  sy's negative frequency:                                   m   m-1 m-2 ... 1     0;                   m...
            #
            #     If ny is EVEN(ny=2m, ny//2=m):
            #                                index in rfft_sx: 0   1   2    3    ... m-2 m-1 m   m+1 m+2 m+3 ....ny-1; ny ny+1 ny+2... ny+m ny+m+1 ...
            #                    ==>  sy's positive frequency: 0   1   2    3    ... m-2 m-1 m   m+1 m+2 m+3 ... ny-1; 0  1    2   ... m    m+1    ...
            #                    ==>  sy's negative frequency:    ny-1 ny-2 ny-3 ... m+2 m+1 m   m-1 m-2 m-3 ... 1     0; ny-1 ny-2... m    m-1    ...
            #     Only the first half m+1 is useful:
            #                    ==>  sy's positive frequency: 0   1   2    3    ... m-2 m-1 m                       ; 0  1    2   ... m
            #                    ==>  sy's negative frequency:                               m   m-1 m-2 m-3 ... 1     0;              m ...
            #
            #     Note that the sy has the length ny//2+1, so for each frequency segment [0:ny), only the first half [0:ny//2+1) is useful.
            #
            #     So, code using rfft and irfft should be:
            #
        # 2.2 For EVEN nx { nx = 2n or n = nx//2 }
        # 2.2.1 fft version
            #     The original spectrum sx: (We omit df in the frequency, as it is a constant. Or, we just take normalized df=1.0)
            #                           sx:  #---...---#---#---x===x===x===x===...===x
            #                    frequency: -n+1 ...  -2  -1   0   1   2   3   ...   n,  where nx=2n
            #     Please note the element sx[n]. It is at the Nyquist frequency (fc = 1/2 *nx = n)
            #     The physical layout of the original sx in the memory is:
            #                                              sx: x===x===x===x===...===x~~~#---...---#---#
            #                                  sx's frequency: 0   1   2   3   ...   n  -n+1 ...  -2  -1
            #                                     index in sx: 0   1   2   3   ...   n   n+1 ...       nx-1
            #     We need to, like what we did above in 1.2, fix the data point at negative Nyquist frequency (-n):
            #                                        fixed sx: x===x===x===x===...===a~~~a---#---...---#---#, where a = sx[n]*0.5
            #                            fixed sx's frequency: 0   1   2   3   ...   n  -n  -n+1 ...  -2  -1
            #                                     index in sx: 0   1   2   3   ...   n       n+1 ...       nx-1
            #
            #     Now, when we resample x to y of size ny, the new Nyquist frequency for time series y will be fcy = 1/2 *ny.
            #     Then a sx[ix] data point at frequency ix will contribute to a sy data point at frequency iy:
            #                                  iy = ix % fsy
            #                                     = ix % (2*fcy), where fcy = 1/2 * ny
            #                                     = ix % ny
            #
            #     So, now each sx's data point will contribute to sy data points at frequencys:
            #                                        fixed sx: x===x===x===x===...===a ~ ~ ~ a---#---...---#---#, where a = sx[n]*0.5
            #                                     index in sx: 0   1   2   3   ...   n           n+1 ...  nx-2 nx-1
            #                            fixed sx's frequency: 0   1   2   3   ...   n      -n  -n+1 ...  -2  -1
            #                             ==>  sy's frequency: 0       2%ny    ...         (-n)%ny   ... (-2)%ny
            #                                                      1%ny    3%n ...   n%ny      (-n+1)%ny     (-1)%ny
            #                             ==>  sy's frequency:[0:ny), [0:ny),  ...[0:n%ny]
            #                                                                             [(-n)%ny, ny),  ...[0:ny), [0:ny)
            #
            #     So the code should be:
            #                           sx= fft(x)
            #                           n = nx // 2
            #                           if nx % 2 == 0:
            #                               sx[n] *= 0.5
            #                           pos_part  = sx[:n+1]
            #                           remainder = pos_part % ny
            #                           tidy_size = pos_part.size - remainder
            #                           sy       += pos_part[:tidy_size].reshape((-1, ny)).sum(axis=0)
            #                           if remainder > 0:
            #                               sy[:remainder] += pos_part[tidy_size:]
            #
            #                           neg_part  = sx[n:] # Note this is different from the case where nx is ODD!
            #                           remainder = neg_part % ny
            #                           sy       += neg_part[remainder:].reshape((-1, ny)).sum(axis=0)
            #                           if remainder > 0:
            #                               sy[ny-remainder] = neg_part[:remainder]
            #                           y = ifft(sy)
            #
        # 2.2.2 rfft version for real time series x
            #     Besides, let's consider the rfft's spectrum, for which x is real and sx has conjugate symmetry w.r.t. zero frequency.
            #     So rfft spectrum rfft_x has the layout in physica memory:
            #                                        fixed sx: x===x===x===x===...===a, where a=sx[n]*0.5
            #                                index in rfft_sx: 0   1   2   3   ...   n
            #                              positive frequency: 0   1   2   3   ...   n
            #                              negative frequency:    -1  -2  -3   ...  -n
            #                    ==>  sy's positive frequency: 0       2%ny    ...
            #                                                      1%ny    3%n ...   n%ny
            #                    ==>  sy's positive frequency:[0:ny), [0:ny),  ...[0:n%ny]
            #
            #                    ==>  sy's positive frequency: 0 1 2 3 ... ny-1; 0 1 2 3 4 ny-1; 0 1 2 3...
            #
            #     This is only for the positive frequencies. For the negative frequencies, they should be conjugate to what we have formed.
            #                    ==>  sy's negative frequency:        -2%ny    ...
            #                                                     -1%ny   -3%n ...  -n%ny
            #                    ==>  sy's negative frequency:    (ny, 0],... (ny, 0], ...
            #
            #                    ==>  sy's negative frequency:    ny-1 ny-2 ny-3 ... 0; ny-1 ny-2 ny-3 ... 0; ny-1 ...
            #
            #     If ny is ODD (ny=2m+1, ny//2=m):
            #                                index in rfft_sx: 0   1   2    3    ... m-2 m-1 m   m+1 m+2 m+3 ....ny-1; ny ny+1 ny+2... ny+m ny+m+1 ...
            #                    ==>  sy's positive frequency: 0   1   2    3    ... m-2 m-1 m   m+1 m+2 m+3 ... ny-1; 0  1    2   ... m    m+1    ...
            #                    ==>  sy's negative frequency:    ny-1 ny-2 ny-3 ... m+3 m+2 m+1 m   m-1 m-2 ... 1     0; ny-1 ny-2... m+1  m      ...
            #     Only the first half [0, m+1) is useful:
            #                    ==>  sy's positive frequency: 0   1   2    3    ... m-2 m-1 m                       ; 0  1    2   ... m
            #                    ==>  sy's negative frequency:                                   m   m-1 m-2 ... 1     0;                   m...
            #
            #     If ny is EVEN(ny=2m, ny//2=m):
            #                                index in rfft_sx: 0   1   2    3    ... m-2 m-1 m   m+1 m+2 m+3 ....ny-1; ny ny+1 ny+2... ny+m ny+m+1 ...
            #                    ==>  sy's positive frequency: 0   1   2    3    ... m-2 m-1 m   m+1 m+2 m+3 ... ny-1; 0  1    2   ... m    m+1    ...
            #                    ==>  sy's negative frequency:    ny-1 ny-2 ny-3 ... m+2 m+1 m   m-1 m-2 m-3 ... 1     0; ny-1 ny-2... m    m-1    ...
            #     Only the first half m+1 is useful:
            #                    ==>  sy's positive frequency: 0   1   2    3    ... m-2 m-1 m                       ; 0  1    2   ... m
            #                    ==>  sy's negative frequency:                               m   m-1 m-2 m-3 ... 1     0;              m ...
            #
            #     Note that the sy has the length ny//2+1, so for each frequency segment [0:ny), only the first half [0:ny//2+1) is useful.
            #
            #     So, code using rfft and irfft should be:
        """
    @staticmethod
    def fft_interpolate(x, dx, ny):
        """
        Interpolate the time series x to a new time series y of size ny.
        It is highly recommended to taper the input time series x before applying this shift to avoid edge effects!

        :param x:  The input time series, a 1D numpy array.
        :param dx: The time step of the input time series x.
        :param ny: The size of the output time series y.

        :return: A tuple (y, dy, sy), where:
            y  : The interpolated time series of size ny.
            dy : The time step of the output time series y.
            sy : The Fourier spectrum of the output time series y, which has size ny.
        """
        nx = len(x)
        if ny == nx:
            return x, dx
        elif ny > nx:
            sx = scipy.fft.fft(x, n=nx, norm='forward')
            sy = np.zeros(ny, dtype=sx.dtype)
            if nx%2 != 0: # ODD nx, and sx does not have the Nyquist frequency
                hn              = nx//2+1
                sy[:hn]         = sx[:hn]
                sy[ny-(nx-hn):] = sx[hn:]
            else:         # EVEN nx, so that sx does have the Nyquist frequency
                n              = nx//2
                sx[n]         *= 0.5 # !!! Note, sx is modified here!
                sy[:n+1]       = sx[:n+1]
                sy[ny-(nx-n):] = sx[n:]
            y = scipy.fft.ifft(sy, norm='forward')
            ####
            if not np.iscomplexobj(x):
                y = y.real
            dy = (nx*dx)/ny
            return y, dy
        else:
            if   False:  # Method 1. Make the length of x and sx always to be ODD, (nx=2n+1)
                        #           so that we don't need to worry about the Nyquist frequency.
                        # This is just for showing the algorithm, not for practical use.
                if nx % 2 == 0:
                    nx = nx + 1
                    x, dx, sx = fft_interpolate(x, dx, nx)
                else:
                    sx = scipy.fft.fft(x, norm='forward')
                if False:   #### 1.2.1 Use the aliasing equation to interpolate.
                            #    The aliasing equation is:
                            #         fa = (f + fc) % (2*fc) - fc,
                            #    where fa is the aliasing frequency, f is the original frequency, and fc is the Nyquist frequency.
                            #    Let's assume df=1.0, so that fc_x = 0.5*nx, and fc_y = 0.5*ny, and
                            #    sx has frequencies fx: {ix} = {0, 1, 2,..., n, n+1,..., 2n} or {0, 1, 2,..., n, -n+1, -n+2,...,-1} where nx=2n+1
                            #
                            #         iy =  (fx + fc_y) % (2*fc_y) - fc_y
                            #         iy =  (ix + 0.5*ny) % ny     - 0.5*ny
                            #     pos iy = [(ix + 0.5*ny) % ny     - 0.5*ny ] % ny # make positive
                            #####
                    sy = np.zeros(ny, dtype=sx.dtype)
                    n = nx//2
                    fcy = 0.5*ny # Nyquist frequencies
                    #
                    ixs = np.arange(nx)
                    tmp = np.where(ixs<=n, ixs, ixs-nx)# take care of the negative frequencies
                    print(nx, 'ixs:', ixs)
                    print(nx, 'ixs:', tmp)
                    #
                    iys = (tmp+fcy)%(2*fcy) - fcy
                    iys = iys.astype(int)
                    iys = iys%ny
                    #
                    print(ny, 'iys:', iys)
                    i0 = 0
                    while i0<nx//2+1:
                        i1 = min(i0+ny, nx//2+1)
                        junk1 = np.arange(ny)[:i1-i0]
                        print(ixs[i0:i1], ',', iys[i0:i1], junk1)
                        i0 = i1
                    print(nx//2)
                    print()
                    i1 = nx
                    while i1>nx//2+1:
                        i0 = max(i1-ny, nx//2+1)
                        junk2 = np.arange(ny)[::-1][:i1-i0]
                        print(ixs[i0:i1][::-1], ',', iys[i0:i1][::-1], junk2)
                        i1 = i0
                    #####
                    overlap = int(nx//ny)+1
                    for ix, iy in zip(ixs, iys):
                        sy[iy] += sx[ix]
                else:       #### 1.2.2 Use the result of the aliasing equation to interpolate.
                            #     It is easy to find out that the ix and iy have the following relationship:
                            #         iy =  (fx + fc_y) % (2*fc_y) - fc_y
                            #         iy =  (ix + 0.5*ny) % ny     - 0.5*ny
                            #     pos iy = [(ix + 0.5*ny) % ny     - 0.5*ny ] % ny # make positive
                            #         ix:     0, 1, 2,..., ny-1, ny, ny+1, ny+2,..., 2ny-1,... nx//2 (also n), | ... n+1(also -n), n+2,... ...,nx-2ny,...,nx-ny-2, nx-ny-1, nx-ny,...,nx-3, nx-2, nx-1
                            #     pos iy:     0, 1, 2,..., ny-1,  0,    1,    2,...,  ny-1,...                 |                                   0,        ny-2,    ny-1,     0,...,ny-3, ny-2, ny-1
                            ####
                    sy = np.zeros(ny, dtype=sx.dtype)
                    n = nx//2
                    # for the positive frequencies
                    i0 = 0
                    while i0 < n+1:
                        i1 = min(i0+ny, n+1)
                        print(nx, n, np.arange(i0, i1))
                        sy[:i1-i0] += sx[i0:i1]
                        i0 = i1
                    print()
                    # for the negative frequencies
                    i1 = nx
                    while i1 > n+1:
                        i0 = max(i1-ny, n+1)
                        print(nx, n, np.arange(i0, i1))
                        sy[ny-(i1-i0):] += sx[i0:i1]
                        i1 = i0
                y = scipy.fft.ifft(sy, norm='forward')
            if   False:  # Method 2. Take care of the frequency at Nyquist frequency when necessary.
                        #           so that we don't need to extend the length of x and sx which is time consuming.
                        # Also, this is just for showing the algorithm, not for practical use.
                sx = scipy.fft.fft(x, norm='forward')
                if nx % 2 == 0:
                    sx[nx//2] *= 0.5 # !!! Note, sx is modified here! for the Nyquist frequency
                ######### Method 2.2 Use the result of the aliasing equation to interpolate.
                sy = np.zeros(ny, dtype=sx.dtype)
                n = nx//2
                # for the positive frequencies
                i0 = 0
                while i0 < n+1: # [0 to n] (n+1 points)  n=nx//2
                    i1 = min(i0+ny, n+1)
                    #print(nx, ny, n, np.arange(i0, i1), np.arange(i1-i0) )
                    sy[:i1-i0] += sx[i0:i1]
                    i0 = i1
                #print()
                # for the negative frequencies
                i1 = nx
                if nx%2==0:
                    while i1 > n: # [n, nx-1] (nx-n points for even nx); or [n+1, nx-1] (nx-n-1 points for odd nx)
                        i0 = max(i1-ny, n)
                        #print(nx, ny, n, np.arange(i0, i1), np.arange(ny-(i1-i0), ny) )
                        sy[ny-(i1-i0):] += sx[i0:i1]
                        i1 = i0
                else:
                    while i1 > n+1: # [n, nx-1] (nx-n points for even nx); or [n+1, nx-1] (nx-n-1 points for odd nx)
                        i0 = max(i1-ny, n+1)
                        #print(nx, ny, n+1, np.arange(i0, i1), np.arange(ny-(i1-i0), ny) )
                        sy[ny-(i1-i0):] += sx[i0:i1]
                        i1 = i0
                    pass
                y = scipy.fft.ifft(sy, norm='forward')
            if   True:   # Method 2. (accelerated) Take care of the frequency at Nyquist frequency when necessary
                sx = scipy.fft.fft(x, norm='forward')
                n = nx//2
                if nx%2 == 0:
                    sx[n] *= 0.5 # !!! Note, sx is modified here! for the Nyquist frequency
                ######### Method 2.2 Use the result of the aliasing equation to interpolate.
                sy = np.zeros(ny, dtype=sx.dtype)
                #### For the positive frequencies
                pos_part  = sx[:n+1]           # the positive part is sx[:nx//2+1] (including the Nyquist frequency if nx is even)
                                                # Note, sx does not have Nyquist frequency if nx is odd!
                remainder = pos_part.size % ny   # break the positive part into tidy pieces of size ny, and a remainder at the end
                tidy_size = pos_part.size - remainder
                sy       += pos_part[:tidy_size].reshape(-1, ny).sum(axis=0)
                #print(pos_part)
                #print(pos_part[:tidy_size] )
                #print(pos_part[:tidy_size].reshape(-1, ny))
                if remainder > 0:
                    sy[:remainder] += pos_part[tidy_size:]
                #### For the negative frequencies
                neg_part  = sx[n:] if nx%2==0 else sx[n+1:] # the negative part is sx[nx//2:] including the negative Nyquist frequency if nx is even,
                                                            # or sx[nx//2+1:] if nx is odd, for which sx does not have the Nyquist frequency).
                remainder = neg_part.size % ny                  # break the negative part into remainder at the start, and tidy pieces of size ny
                sy       += neg_part[remainder:].reshape(-1, ny).sum(axis=0)
                if remainder > 0:
                    sy[ny-remainder:] += neg_part[:remainder]
                y = scipy.fft.ifft(sy, norm='forward')
            #####
            if not np.iscomplexobj(x):
                y = y.real
            ####
            dy = (nx*dx)/ny
            return y, dy
    def rfft_interpolate(x, dx, ny):
        """
        Interpolate the time series x to a new time series y of size ny.
        It is highly recommended to taper the input time series x before applying this shift to avoid edge effects!

        :param x:  The input time series, a 1D numpy array.
        :param dx: The time step of the input time series x.
        :param ny: The size of the output time series y.
        :return: A tuple (y, dy, sy), where:
            y  : The interpolated time series of size ny.
            dy : The time step of the output time series y.
            sy : The Fourier spectrum of the output time series y, which has size ny//2+1.
        """
        if np.iscomplexobj(x):
            y_real, dy = TSFuncs.rfft_interpolate(x.real, dx, ny)
            y_imag, dy = TSFuncs.rfft_interpolate(x.imag, dx, ny)
            y  = y_real + 1j * y_imag
            return y, dy
        #######
        nx = len(x)
        #######
        if ny == nx:
            sx = scipy.fft.rfft(x, norm='forward')
            return x, dx
        elif ny >= nx:
            n  = nx//2
            sx = scipy.fft.rfft(x, norm='forward')
            sy = np.zeros(ny//2+1, dtype=sx.dtype)
            sy[:n+1] = sx
            if nx%2 == 0:
                sy[n] *= 0.5
            y = scipy.fft.irfft(sy, n=ny, norm='forward')
            dy = (nx*dx)/ny
            return y, dy
        else:
            m         = ny // 2
            n         = nx // 2
            sx        = scipy.fft.rfft(x, norm='forward')
            if nx%2 == 0:
                sx[n] *= 0.5
            ############
            pos_leg   = sx
            temp      = np.zeros(ny, dtype=sx.dtype)
            remainder = pos_leg.size % ny
            tidy_size = pos_leg.size - remainder
            temp     += pos_leg[:tidy_size].reshape((-1, ny)).sum(axis=0)
            if remainder > 0:
                temp[:remainder] += pos_leg[tidy_size:]
            #############
            neg_leg   = np.conj(pos_leg[1:])
            temp2     = np.zeros(ny, dtype=sx.dtype)
            remainder = neg_leg.size % ny
            tidy_size = neg_leg.size - remainder
            temp2     += neg_leg[:tidy_size].reshape((-1, ny)).sum(axis=0)
            if remainder > 0:
                temp2[:remainder] += neg_leg[tidy_size:]
            #############
            sy = temp + temp2[::-1]
            sy = sy[:m+1]
            y  = scipy.fft.irfft(sy, n=ny, norm='forward')
            dy = (nx*dx)/ny
            return y, dy
    @staticmethod
    def benchmark5():
        import sacpy.processing as processing
        from sacpy.processing import TSFuncs
        fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 10))
        nx = np.random.randint(10, 30)
        dx = 0.1
        func1  = processing.RandFunc1DGenerator(0.0, nx, dx, 0.0)
        func2  = processing.RandFunc1DGenerator(0.0, nx, dx, 0.0)
        func1.set_filter_noise_taper((0.001, 0.5*func1.fmax) )
        func2.set_filter_noise_taper((0.001, 0.5*func2.fmax) )
        x  = func1.data + func2.data[::1]*1j
        ###########################
        ny = 10000
        y, dy = TSFuncs.fft_interpolate(x, dx, ny)
        TSFuncs.plot_time_series(y, dy, ax3, ax4, lw=0.5, ls='-', color='k', zorder=100, label='true signal')
        ###########################
        TSFuncs.plot_time_series(x, dx, ax3, ax4, marker='s', markersize=12, lw=0.01, ls=':', color='k', zorder=100, label=f'Original samples {nx}')
        ###########################
        ny = np.random.randint(2, nx*5)
        y, dy = TSFuncs.fft_interpolate(x, dx, ny)
        TSFuncs.plot_time_series(y, dy, ax3, ax4, marker='o', markersize=9, lw=0.01, ls=':', color='C0', zorder=100, label=f'fft_interpolate {ny}')
        y, dy = TSFuncs.rfft_interpolate(x, dx, ny)
        TSFuncs.plot_time_series(y, dy, ax3, ax4, marker='o', markersize=5, lw=0.01, ls=':', color='r', zorder=100, label=f'rfft_interpolate {ny}')
        #############################
        ax3.set_title('Time Series (Real part)')
        ax4.set_title('Time Series (Imaginary part)')
        ax3.legend(loc=(0, 1.01))
        plt.show()

    ######### delay time picker between two time series
    """ Measure time difference between two time series using cross correlation method.
    """
    @staticmethod
    def time_diff_cc(ref, dat, dt, ref_start=0.0, dat_start=0.0, pre_normlized=False, denser_time_ratio=1,
                     diff_lim=None, std_ratio=0.95):
        """
        Measure time difference between two time series using cross correlation method.
        :param ref:         the reference time series.
        :param dat:         the time series to be aligned with the reference.
        :param dt:          time step of the time series
        :param ref_start:   (default is zero) start time of the reference time series
        :param dat_start:   (default is zero) start time of the data time series
        :param pre_normlized: (default is False) if True, the time series x and y are already normalized.
                              so that the summation of the autocorrelation of x or y is 1.0.
        :param denser_time_ratio: (default is 1) an int >=1, we will search for the time difference in a denser time grid so
                                  that the time step will be `dt/denser_time_ratio`.
        :param diff_lim:    a tuple (tmin, tmax) to limit the range of the time difference.
                            (default is None, which means no limit.)
        :param std_ratio:   (default is 0.95) the ratio of the maximum correlation value used to estimate the standard deviation
                            of the obtained time difference.


        :return: tshift
            tshift: time difference between reference `ref` and `dat`, and in the same unit as dt.
                    It means by shift the time series reference `ref` rightwards by tshift, we can align the two time series.
                    So positive tshift means `ref` is ahead of `dat`, and negative tshift means `ref` is behind `dat`.
        """
        if np.iscomplexobj(ref) or np.iscomplexobj(dat):
            raise ValueError("time_diff_cc(...) does not support complex time series.")
        ####
        corr = scipy.signal.correlate(dat, ref, mode='full')
        if not pre_normlized:
            corr /= np.sqrt( np.sum(ref*ref) * np.sum(dat*dat) )
        ####
        nref, ndat = len(ref), len(dat)
        if denser_time_ratio > 1:
            denser_time_ratio = int(denser_time_ratio)
            # Make corr denser by interpolating it. However, please note this is different from (1) make the time series denser and (2) correlation the denser time series.
            # Method below return a corr with          size1 = (ref.size + dat.size - 1) * denser_time_ratio
            # The other method will return a corr with size2 = ref.size* denser_time_ratio + dat.size*denser_time_ratio - 1
            # So, there will be a size difference of  size1-size2 = -denser_time_ratio+1.
            # It means, we lack (-denser_time_ratio+1) points at the beginning of the correlation.
            # Then, we solve this problem through cheating! We just roll the correlation array to the right by (denser_time_ratio-1) points.
            # This cheating is valid if the ref and dat are both tapered which means zero outside their valid time ranges.
            corr, dt = TSFuncs.rfft_interpolate(corr, dt, corr.size * denser_time_ratio)
            corr = np.roll(corr, denser_time_ratio-1)
            nref *= denser_time_ratio
            ndat *= denser_time_ratio
        ####
        if diff_lim is None:
            idx_max = np.argmax(corr)
        else:
            tdif_min, tdif_max = diff_lim
            i0 = int(np.ceil(  (tdif_min-(dat_start-ref_start))/dt)) - (1-nref)
            i1 = int(np.floor( (tdif_max-(dat_start-ref_start))/dt)) - (1-nref) + 1
            i0 = max(i0, 0)
            i1 = min(i1, len(corr))
            idx_max = np.argmax(corr[i0:i1]) + i0
        ####
        #corr_ts = np.arange(-nref + 1, ndat) * dt + dat_start - ref_start
        # t =  (idx -nref + 1) * dt + dat_start-ref_start
        # fuc_get_t = lambda idx: (idx -nref + 1) * dt + dat_start-ref_start
        #### find the maximum correlation value and its index
        t_max    =  (idx_max -nref + 1) * dt + dat_start-ref_start
        corr_max =  corr[idx_max]
        #### get the std at left and right
        if std_ratio<=0.0 or std_ratio>=1.0:
            raise ValueError("std_ratio should be in (0, 1).")
        lvl = corr_max * std_ratio
        # take care of the left of the maximum
        cc_left = corr[:idx_max+1]
        tmp = np.where(cc_left>lvl, 1, 0)
        index_err_left = np.where(tmp == 0)[0][-1] # get the last non-zero index from tmp
        # take care of the right of the maximum
        cc_right = corr[idx_max:]
        tmp = np.where(cc_right>lvl, 1, 0)
        index_err_right = np.where(tmp == 0)[0][0]+idx_max # get the first non-zero index from tmp
        #
        index_err_left  -= idx_max
        index_err_right -= idx_max
        err_left  = (index_err_left * dt)
        err_right = (index_err_right * dt)
        err_mean  = (-err_left + err_right)*0.5
        ####
        corr_tstart = (-nref + 1)* dt + dat_start - ref_start
        ####
        return t_max, (corr_max, err_left, err_right, err_mean, corr_tstart, dt, corr)
    @staticmethod
    def benchmark6():
        ##############
        x0, dt, nx = 0.0, 1, 100
        func_gen = RandFunc1DGenerator(x0, nx, dt, 0.0)
        func_gen.set_filter_noise_taper((0.001, 0.4*func_gen.fmax), 0.0, int(nx*0.5), 4 )
        x = func_gen(more_noise_level=0.0)
        ##############
        y = func_gen(more_noise_level=0.0)
        # 1. delay from time start
        shift_part1 = 3.3 #13.35134
        y0 = x0 + shift_part1
        # 2. delay from cut
        cut_i0, cut_i1 = 11, -13
        y0 += (cut_i0 * dt)
        y = y[cut_i0:cut_i1]
        # 3. delay from shift
        shift_part3 = 0.6234
        y = TSFuncs.shift_and_cut_array1d(y, tstart=y0, dt=dt, time_shift=shift_part3, new_tstart=y0, new_size=len(y))
        #
        theroretical_tdiff = shift_part1 + shift_part3
        ##############
        tdif, (corr_max, err_left, err_right, err_mean, corr_t0, corr_dt, corr) = TSFuncs.time_diff_cc(x, y, dt, ref_start=x0, dat_start=y0, pre_normlized=False, denser_time_ratio=10, diff_lim=(-10, 10), std_ratio=0.9)
        print(theroretical_tdiff, tdif, (err_left, err_right, err_mean), corr_max, )
        ##############
        check_start = y0
        check = TSFuncs.shift_and_cut_array1d(x, tstart=x0, dt=dt, time_shift=tdif, new_tstart=check_start, new_size=len(x), fill_value=np.nan)
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        TSFuncs.plot_time_series(x,     dt, start=x0,          ax_real=ax, marker='s', markersize=0, lw=3, ls='-', color='k',  zorder=100, label='x (ref)')
        TSFuncs.plot_time_series(y,     dt, start=y0,          ax_real=ax, marker='o', markersize=0, lw=3, ls='-', color='C0', zorder=100, label='y')
        TSFuncs.plot_time_series(check, dt, start=check_start, ax_real=ax, marker='o', markersize=0, lw=1, ls='--', color='r',  zorder=100, label='shifted x (ref)')
        ax.legend()
        ax.set_title(f"Time difference between x (ref) and y is m:{tdif:.6f} t:{theroretical_tdiff:.6f} seconds")
        ax.grid(True)
        ##############
        TSFuncs.plot_time_series(corr, corr_dt, start=corr_t0, ax_real=ax2, marker='o', markersize=0, lw=3, ls='-', color='k', zorder=100, label='corr')
        #
        denser_x, denser_dx = TSFuncs.rfft_interpolate(x, dt, len(x)*2)
        denser_y, denser_dy = TSFuncs.rfft_interpolate(y, dt, len(y)*2)
        corr_check = scipy.signal.correlate(denser_y, denser_x, mode='full')
        corr_check *= (1.0 / (np.sqrt(np.sum(denser_x**2)) * np.sqrt(np.sum(denser_y**2))))
        corr_check_tstart = y0-x0 + (1-denser_x.size)*denser_dx
        TSFuncs.plot_time_series(corr_check, denser_dx, start=corr_check_tstart, ax_real=ax2, marker='o', markersize=0, lw=1, ls='-', color='r', zorder=100, label='manul to check')
        print(corr.size, corr_check.size, corr_dt, denser_dx)

        ax2.legend()
        ax2.set_title(f"Cross-correlation between x and y")
        plt.show()

    @staticmethod
    def benchmark7():
        """test the mask function"""

        t0, dt, nt = 0, 1, 1000
        fun_gen = RandFunc1DGenerator(t0, nt, dt, outside_value=0.0)
        fun_gen.set_filter_noise_taper((0.01, 0.5*fun_gen.fmax), noise_level=0.1, taper_halfsize=int(nt*0.1), taper_order=2)
        #
        x0, dx, nrow = 0, 10, 100
        mat = np.zeros((nrow, nt), dtype=np.float32)
        for irow in range(nrow):
            mat[irow] = fun_gen(more_noise_level=0.5)
        #### mask using windows
        #mask_windows = [(-100, 150), (100, 200), (300, 400), (350, 800), (900, 1200) ]
        #for irow in range(nrow):
        #    mat[irow] = mask_time_window(mat[irow], dt, t0, mask_windows, fill_value=0.0)
        #### mask using curves
        curve0 = (0, 1000), (200, 200)
        curve1 = (0, 1000), (300, 400)
        curve2 = (0, 1000, 500), (500, 500, 600)
        curve4 = (0, 200, 300, 600, 700, 800), (600, 800, 900, 800, 780, 700)
        mask_curves = [ curve0, curve1, curve2, curve4 ]
        mat = mask2d_time_window(mat, dt, t0, np.arange(nrow)*dx, mask_curves, 0.0, wnd_curve_extrapolate=True, wnd_curve_interp1d_kind='slinear', taper_half_size=30, taper_order=2)
        ####
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        print(np.min(mat), np.max(mat) )
        TSFuncs.plot_mat2d(mat, dt, dx, ax1, tstart=t0, xstart=x0, cmap='viridis', interpolation='None', aspect='auto', label_col='Time (s)', label_row='X', title='transpose=False', origin='lower')
        TSFuncs.plot_mat2d(mat, dt, dx, ax2, tstart=t0, xstart=x0, cmap='viridis', interpolation='None', aspect='auto', label_col='Time (s)', label_row='X', title='transpose=True',  origin='lower', orientation='vertical', )
        ####
        TSFuncs.plot_mat2d_waveforms(mat, dt, ax3, tstart=t0, xs=np.arange(nrow)*dx, color='k', ls='-', lw=0.8, label_col='Time (s)', label_row='X', title='transpose=False', scale=5, invert_yaxis=False)
        TSFuncs.plot_mat2d_waveforms(mat, dt, ax4, tstart=t0, xs=np.arange(nrow)*dx, color='k', ls='-', lw=0.8, label_col='Time (s)', label_row='X', title='transpose=True',  orientation='vertical', scale=5, invert_yaxis=False)
        ####
        for curve_x, curve_t in mask_curves:
            ax1.plot(curve_t, curve_x, color='r', lw=2, ls='-', label='mask curve')
            ax3.plot(curve_t, curve_x, color='r', lw=2, ls='-', label='mask curve')
            #
            ax2.plot(curve_x, curve_t, color='r', lw=2, ls='-', label='mask curve')
            ax4.plot(curve_x, curve_t, color='r', lw=2, ls='-', label='mask curve')
        plt.show()
        pass



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    if True:
        pass
        #RandFunc1DGenerator.benchmark()
        #TSFuncs.benchmark1()
        #TSFuncs.benchmark1_2()
        #TSFuncs.benchmark2()
        #TSFuncs.benchmark2_1()
        #TSFuncs.benchmark4()
        #TSFuncs.benchmark5()
        TSFuncs.benchmark6()
        #TSFuncs.benchmark7()
        print( type(round_index(0.5, 0.0, 0.1)), round_index(0.5, 0.0, 0.1))
        #print( type(round_index(np.array([0.5, 1.3]), 0.0, 0.1)), round_index(np.array([0.5, 1.3]), 0.0, 0.1))
    if False:
        x1 = [0, 1, 2, 3]
        x2 = insert_values((1.5, 2.5), x1)
        print(x1)
        print(x2)
    if False:
        x1 = np.array([0, 1, 2, 3, 2, 1])
        y1 = x1+10
        z1 = x1*x1
        new_x1, new_y1, new_z1 = insert_values((1.5, 2.5), x1, y1, z1)
        print(new_x1)
        print(new_y1)
        print(new_z1)
    if False:
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
        xs = list( np.array([1, 3, 5, 7, 9, 7, 5, 3, 1])+10.0 )
        ys = list( np.array(xs)+10 )
        zs = list( np.array(xs)+20 )
        xcs = 17.0-(1e-3), 11.1, 13.0
        print('insert_values(xcs, xs) --------------------------------------------------------------------------')
        print('x', xs)
        print('xcs', xcs)
        vx = insert_values(xcs, xs)
        print('vx', vx)
        print('insert_values(xcs, xs, ys, zs)--------------------------------------------------------------------------')
        vx, vy, vz = insert_values(xcs, xs, ys, zs)
        print('vx', vx)
        print('vy', vy)
        print('vz', vz)
        print()
    if False:
        xs = list( np.array([1, 3, 5, 7, 9, 7, 5, 3, 1])+10.0 )
        ys = list( np.array(xs)+10 )
        zs = list( np.array(xs)+20 )
        xcs = xs[2]-1e-2
        for edge in 'si-+':
            print('split_arrays(xcs, xs, edge=edge)---------------------------------------------------------------------------')
            vx = split_arrays(xcs, xs, edge=edge)
            print('xs', xs)
            print('xcs', xcs)
            print(edge, 'vx', vx)
            if True:
                print('split_arrays(xcs, xs, ys, edge=edge)---------------------------------------------------------------------------')
                vx, vy, vz = split_arrays(xcs, xs, ys, zs, edge=edge)
                print(edge, 'vx', vx)
                print(edge, 'vy', vy)
                print(edge, 'vz', vz)
                print()
        print()
    if False:
        xs = list( np.array([0, 1, 2, 4, 5, 6, 7, 5, 3, 2, 1, 0]) )
        ys = list( np.array(xs)+10 )
        zs = list( np.array(xs)+20 )
        xcs = 1.1, 5.0
        for edge in 'i-+':
            xmin, xmax=1.1, 5.0
            print('select_subarrays(xmin, xmax, xs, edge=edge)---------------------------------------------------------------------------')
            vx = select_subarrays(xmin, xmax, xs, edge=edge)
            print('xs', xs)
            print('xmin, xmax', xmin, xmax)
            print(edge, 'vx', vx)
            #print(v[1])
            print('select_subarrays(xmin, xmax, xs, ys, zs, edge=edge)---------------------------------------------------------------------------')
            vx, vy, vz = select_subarrays(xmin, xmax, xs, ys, zs, edge=edge)
            print(edge, 'vx', vx)
            print(edge, 'vy', vy)
            print(edge, 'vy', vz)
            print()

        for edge in '':
            c = np.linspace(0, 10, 100)
            xs = np.sin(c)
            ys = xs+1
            zs = np.cos(xs)
            xmin, xmax= -0.2, 0.3
            vx, vy, vz = select_subarrays(xmin, xmax, xs, ys, zs, edge=edge)
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