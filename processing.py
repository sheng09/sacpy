#!/usr/bin/env python3

"""
This is for data processing.
"""

from scipy.signal import butter, lfilter, filtfilt, fftconvolve
import numpy as np
from pyfftw.interfaces.cache import enable as pyfftw_cache_enable
from pyfftw.interfaces.numpy_fft import rfft, irfft
from numba import jit, int64, float64
from copy import deepcopy
###
#  linear processing
###


###
#  smart filter methods which avoid repeated constructions of filters.
###
__dict_filter_proc__ = dict()
def filter(tr, sampling_rate, btype, frequency_band, order, npass= 2):
    """
    Filter a trace, and return filtered trace.
    tr:            the trace to be filtered
    sampling_rate: sampling rate of the trace;
    btype:         can be 'lowpass, 'highpass', 'bandpass';
    frequency_band:[f1, (f2)] critical frequency for filter;
    order:         order of the filter
    npass:         1 or 2 (default);
    """
    norm_frequency_band = (frequency_band[0]/sampling_rate * 2.0,  frequency_band[1]/sampling_rate * 2.0 )
    if norm_frequency_band not in __dict_filter_proc__:
        b, a = butter(order, norm_frequency_band, btype)
        __dict_filter_proc__[norm_frequency_band] = b, a
    b, a = __dict_filter_proc__[norm_frequency_band]
    #print(b, a)
    filter_methods = lfilter if npass ==1 else filtfilt
    return filter_methods(b, a, tr)
###
#  whitening
###

def moving_average_fft(xs, wdn_sz=1, average=False):
    """
    Moving average of a 1D trace `xs` with ODD window size `wdn_sz`.

    Use this function for small size trace.
    Use `moving_average` for large size trace and for multiple running.
    """
    wdn_sz = (wdn_sz//2)*2+1
    wdn = np.ones(wdn_sz)
    if average:
        wdn *= (1.0/wdn_sz)
    return fftconvolve(xs, wdn, 'same')

@jit(nopython=True, nogil=True)
def moving_average(xs, wdn_sz=1, average=False):
    """
    Moving average of a 1D trace `xs` with ODD window size `wdn_sz`.
    """
    N = len(xs)
    half_wdn_sz = wdn_sz // 2
    half_wdn_sz_plus_one = half_wdn_sz + 1
    minus_half_wdn_sz = N - half_wdn_sz
    if wdn_sz < 2:
        ys = xs
    #####
    ys = np.zeros(N)
    buf = np.sum( xs[:half_wdn_sz], dtype= np.float64 )
    for idx in range(half_wdn_sz_plus_one):
        buf += xs[idx+half_wdn_sz]
        ys[idx] = buf
    for idx in range(half_wdn_sz_plus_one, minus_half_wdn_sz):
        buf += (xs[idx+half_wdn_sz] - xs[idx-half_wdn_sz_plus_one] )
        ys[idx] = buf
    for idx in range(minus_half_wdn_sz, N):
        buf -= xs[idx-half_wdn_sz_plus_one]
        ys[idx] = buf
    #####
    if average:
        ys *= (1.0/(2*half_wdn_sz+1) )
    return ys

@jit(nopython=True, nogil=True)
def moving_average_abs(xs, wdn_sz=3, average=False):
    """
    Absolute moving average of a 1D trace `xs` with ODD window size `wdn_sz`.
    """
    N = len(xs)
    half_wdn_sz = wdn_sz // 2
    half_wdn_sz_plus_one = half_wdn_sz + 1
    minus_half_wdn_sz = N - half_wdn_sz
    if wdn_sz < 2:
        ys = xs
    #####
    ys = np.zeros(N)
    buf = np.sum( np.abs(xs[:half_wdn_sz] ), dtype= np.float64 )
    for idx in range(half_wdn_sz_plus_one):
        buf += np.abs(xs[idx+half_wdn_sz] )
        ys[idx] = buf
    for idx in range(half_wdn_sz_plus_one, minus_half_wdn_sz):
        buf += np.abs(xs[idx+half_wdn_sz]) - np.abs(xs[idx-half_wdn_sz_plus_one] )
        ys[idx] = buf
    for idx in range(minus_half_wdn_sz, N):
        buf -= np.abs( xs[idx-half_wdn_sz_plus_one] )
        ys[idx] = buf
    #####
    if average:
        ys *= (1.0/(2*half_wdn_sz+1) )
    return ys

def temporal_normalize(tr, sampling_rate, twin_len, f1, f2, water_level_ratio= 1.0e-5, taper_length=0):
    """
    Temporal normalization of input trace `tr` with `sampling rate`.

    twin_len:          an integer to delcare the  moving window size.
    f1, f2:            frequency band that will be used to from the weight.
    water_level_ratio: default is 1.0e-5.
    taper_length:      taper size in each end after the division.
    """
    weight = np.abs( filter(tr, sampling_rate, 'bandpass', [f1, f2], 2, 2) )
    weight = moving_average(weight, twin_len, False)
    weight += ( np.max(weight) * water_level_ratio )
    if True in np.isnan(weight) or True in np.isinf(weight) or np.count_nonzero(weight) == 0 :
        weight[:] = 0.0
    else :
        weight = tr / weight
    #####
    if  taper_length > 0:
        weight = taper(weight, taper_length)
    return weight

pyfftw_cache_enable
def frequency_whiten(tr, fwin_len, water_level_ratio= 1.0e-5, speedup_i1= None, speedup_i2= None, taper_length=0):
    """
    Return the whitened time series in time domain.

    fwin_len:          an integer to declare the moving window size.
    water_level_ratio: default is 1.0e-5.
    taper_length:      taper size in each end after the division.
    """
    fftsize = len(tr)
    if fftsize % 2 != 0: # EVET points make faster FFT than ODD points
        fftsize = fftsize + 1

    spec = rfft(tr, fftsize)
    ### for acceleration purpose
    if speedup_i2 != None:
        spec = spec[:speedup_i2]
    amp = np.abs(spec)
    weight = moving_average(amp, fwin_len, False)
    weight += (weight.max() * water_level_ratio)
    if True in np.isnan(weight) or True in np.isinf(weight) or np.count_nonzero(weight) == 0 :
        spec[:] = 0
    else:
        spec /=  weight

    ys = irfft(spec, fftsize )
    if len(tr) != fftsize:
        ys = ys[:-1]
    ys = taper(ys, taper_length)
    return ys

###
#  taper
###

@jit(float64[:](int64, float64), nopython=True, nogil=True)
def tukey_jit(window_length, ratio):
    """
    Return a tukey time window.

    window_length: total length of the window.
    double_ratio:  the ratio between 0 and 0.5.
    """
    # Normal case
    w = np.ones(window_length)
    if ratio <= 0.0:
        return w
    elif ratio >= 0.5:
        ratio = 0.5
    x = np.linspace(0, 1, window_length)

    alpha = ratio*2.0
    # first condition 0 <= x < alpha/2
    first_condition = x<ratio
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))

    # second condition already taken care of

    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x>=(1 -ratio)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2)))

    return w

__dict_taper_window_func = dict()
def taper(tr, n):
    """
    Taper a trace `tr`, given the size `n`.
    0 <= n <= len(tr)/2
    """
    npts = len(tr)
    w = None
    if (npts, n) not in __dict_taper_window_func:
        w = tukey_jit(npts, float(n)/npts)
        __dict_taper_window_func[(npts, n)] = w
    else:
        w = __dict_taper_window_func[(npts, n)]
    x = deepcopy(tr)
    x[:n] *= w[:n]
    x[-n:] *= w[-n:]
    return x

if __name__ == "__main__":
    import copy
    import sacpy.sac as sac
    import sys
    import matplotlib.pyplot as plt

    #sys.exit(0)

    tr1= sac.rd_sac_2('test_tmp/test.sac', '0', 10800, 32400)
    tr1.plot()
    tr1.dat = taper(tr1.dat, tr1.dat.size//100 )
    tr1.plot(color='C1')
    plt.show()
    sys.exit()
    #tr1= sac.rd_sac('test_tmp/test.sac') #, '0', 10800, 32400)
    tr1.detrend()
    tr1.bandpass(0.02, 0.066666, 2, 2)

    sampling_rate = 1.0/tr1['delta']
    delta = tr1['delta']
    df = 1.0/(tr1['npts']*tr1['delta'] )
    tr2 = copy.deepcopy(tr1)
    tr2.dat = temporal_normalize(tr1.dat, sampling_rate, int(128.0/delta), 0.02, 0.06666, 1.0e-5, int(100/delta) )
    tr2.dat = frequency_whiten(tr2.dat, int(0.02/df) )
    #tr1.bandpass(0.02, 0.066666, 2, 2)
    tr2.bandpass(0.02, 0.066666, 2, 2)
    ##########
    tr1.norm()
    tr2.norm()
    fig, (ax1, ax2) = plt.subplots(2, 1)
    tr1.plot_ax(ax1, linewidth= 2)
    tr2.plot_ax(ax1, color='C1', alpha=0.5)
    plt.show()
    sys.exit(0)
    #
    s1 = copy.deepcopy(s)
    s1.bandpass(0.1, 2.0, 2, 2)
    #
    ys2 = filter(s['dat'], 1.0/s['delta'], 'bandpass', [0.1, 2.0], 2, 2)
    #
    plt.subplot(211)
    plt.plot(s1.get_time_axis(), s1['dat'])
    plt.plot(s1.get_time_axis(), ys2, '--')
    plt.xlim([500, 1500] )
    plt.subplot(212)
    plt.plot(s1.get_time_axis(), ys2-s1['dat'] )
    plt.show()
    #
