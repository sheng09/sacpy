#!/usr/bin/env python3

"""
This is for data processing.
"""

import scipy.signal as signal
import numpy as np
import pyfftw
###
#  base class
###
class __atom_proc__:
    """
    Empty base class
    """
    def __init__(self):
        pass
    def proc(self, trace):
        pass
###
#  intelligent filter class
###
class __atom_filter__(__atom_proc__):
    """
    """
    filter_methods = [None, signal.lfilter, signal.filtfilt]
    def __init__(self, btype, fs, sampling_rate, order= 2):
        """
        construct a filter.
        btype: can be 'lowpass, 'highpass', 'bandpass';
        fs: [f1, (f2)] critical frequency for filter;
        sampling_rate: sampling rate;
        order: order of the filter
        """
        super().__init__()
        self.btype = btype
        self.fs = __atom_filter__.__normalize__fs__(fs, sampling_rate) 
        self.order = order
        self.b, self.a = signal.butter(self.order, self.fs, self.btype)
    def apply(self, tr, npass= 2):
        """
        Apply this filter to a trace.
        tr: 
        npass: can be 1 or 2(default).
        """
        return __atom_filter__.filter_methods[npass](self.b, self.a, tr)   
    @staticmethod
    def __normalize__fs__(fs, sampling_rate):
        return tuple( [it/sampling_rate*2.0 for it in fs] )
class __atom_bp__(__atom_filter__):
    def __init__(self, f1, f2, sampling_rate, order= 2):
        """
        Init an atomic bandpass filter;
        f1, f2: low and high frequency for bandpass fitler;
        sampling_rate: sampling rate;
        order: order of the filter;
        """
        super().__init__('bandpass', [f1, f2], sampling_rate, order)
class __atom_lp__(__atom_filter__):
    def __init__(self, f, sampling_rate, order= 2):
        """
        Init an atomic lowpass filter;
        f: critical frequency for lowpass fitler;
        sampling_rate: sampling rate;
        order: order of the filter;
        """
        super().__init__('lowpass', [f], sampling_rate, order)
class __atom_hp__(__atom_filter__):
    def __init__(self, btype, f, sampling_rate, order=2):
        """
        Init an atomic highpass filter;
        f: critical frequency for lowpass fitler;
        sampling_rate: sampling rate;
        order: order of the filter;
        """
        super().__init__('highpass', [f], sampling_rate, order)

###
#  linear processing
###


###
#  smart filter methods which avoid repeated constructions of filters.
###
__dict_filter_proc__ = dict()
def filter(tr, sampling_rate, btype, fs, order, npass= 2):
    """
    Filter a trace, and return filtered trace.
    tr:            the trace to be filtered
    sampling_rate: sampling rate of the trace;
    btype:         can be 'lowpass, 'highpass', 'bandpass';
    fs:            [f1, (f2)] critical frequency for filter;
    order:         order of the filter
    npass:         1 or 2 (default);
    """
    key_fs = __atom_filter__.__normalize__fs__(fs, sampling_rate)
    if key_fs not in __dict_filter_proc__:
        __dict_filter_proc__[key_fs] = __atom_filter__(btype, fs, sampling_rate, order)
    return __dict_filter_proc__[key_fs].apply(tr, npass)

###
#  whitening
###
__dict_one_array = dict()
def temporal_normalization(tr, fs, twin_len, f1, f2, water_level_ratio= 1.0e-6):
    tmp = signal.detrend(tr)
    tmp = signal.detrend(tmp)
    tmp = np.abs( filter(tmp, fs, 'bandpass', [f1, f2], 2, 2) )
    if twin_len not in __dict_one_array:
        __dict_one_array[twin_len] = np.ones(twin_len)
    weight = signal.fftconvolve(tmp, __dict_one_array[twin_len], 'same')
    c = np.max(weight) * water_level_ratio
    weight[weight<c] = c
    return tr/weight

def frequency_whiten_spec(tr, fs, fwin_len, nrfft, water_level_ratio= 1.0e-6):
    """
    Return the whitened spectrum other than the time series
    """
    spec = pyfftw.interfaces.numpy_fft.rfft(tr, nrfft)
    am = np.abs(spec)
    #ph = np.angle(spec)
    if fwin_len not in __dict_one_array:
        __dict_one_array[fwin_len] = np.ones(fwin_len)
    weight = signal.fftconvolve(am, __dict_one_array[fwin_len], 'same')
    c = np.max(weight) * water_level_ratio
    weight[weight<c] = c
    spec /= weight
    return spec

def frequency_whiten(tr, fs, fwin_len, nrfft, water_level_ratio= 1.0e-6):
    spec = frequency_whiten_spec(tr, fs, fwin_len, nrfft, water_level_ratio= water_level_ratio)
    return pyfftw.interfaces.numpy_fft.irfft( spec , tr.size)

###
#  taper
###
__dict_taper_window_func = dict()
def taper(tr, n):
    """
    Taper a trace `tr`, given the size `n`.
    0 <= n <= len(tr)/2
    """
    npts = len(tr)
    if (npts, n) not in __dict_taper_window_func:
        __dict_taper_window_func[(npts, n)] = signal.tukey(npts, float(n)/npts)
    return __dict_taper_window_func[(npts, n)] * tr
    pass

if __name__ == "__main__":
    import copy
    import sacpy.sac as sac
    import matplotlib.pyplot as plt
    s = sac.rd_sac('1.sac')
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
