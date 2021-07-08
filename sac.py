#!/usr/bin/env python3
"""
A python3.x module for manipulating SAC format and related data.


File IO dependent methods
-------------------------

>>> # IO, view,  basic processing
>>> s = c_rd_sac('1.sac')
>>> s.norm('neg') # norm with respect to max negative amplitude, and update s.hdr.scale.
>>> s.shift_time(100.0)
>>> s.truncate(100, 500)
>>> s.plot()
>>> s.detrend()
>>> s.taper(0.02) # ratio can be 0 ~ 0.5
>>> s.filter('BP', (0.2, 1.0), order=2, npass=2)
>>> s.write('1_new.sac')
>>>
>>> # re-sampling
>>> s.downsample(3)
>>> s.upsample(4)
>>> s.interpolate_npts(334)
>>> s.interpolate_delta(0.02, force_lanczos=True)
>>>
>>> # arbitrary plot
>>> ts = s.get_time_axis()
>>> plt.plot(ts, s['dat'], color='black') # ...
>>>
>>> # read given time range
>>> s = c_rd_sac('1.sac', -5, 100, 4000)
>>> # -5 here means 'b. The time window is (100+b, 4000+b)
>>> # -3 means 'o', and 0, 1,...9 for 't0', 't1',...,'t9'.
>>>
>>> # obtain index for maximum
>>> idx, time, amplitude = s.max_amplitude_time('neg', (100.3, 111.5) )
>>>

Arbitrary data writing, and processing methods
----------------------------------------------

>>> import numpy as np
>>> dat = np.random.random(1000).astype(np.float32) # we recommend np.float32
>>> delta, b = 0.05, 50.0
>>> # writing method 1
>>> c_wrt_sac2('junk.sac', dat, b, delta)
>>>
>>> # writing method 2
>>> s = c_mk_sac(dat, b, delta)
>>> s.hdr.stnm = 'SYN' # update sachdr...
>>> s.taper()
>>> s.bandpass(0.5, 2.0, order= 4, npass= 2)
>>> s.write('junk2.sac')
>>>
>>> # writing method 3 assuming a hdr object is created somewhere else
>>> c_wrt_sac('junk3.sac', data, hdr, lcalda=False, verbose=False)
>>>

Sac header update and revision
------------------------------

>>> s = c_rd_sac('1.sac')
>>> hdr = c_rd_sachdr('1.sac', lcalda=True)
>>> # duplicate a hdr given the hdr is C structure object
>>> new_hdr = c_dup_sachdr(s.hdr)
>>> # access hdr parameters
>>> new_hdr.t1 = 2.0 # meaningless value, just for example
>>> new_hdr.t2 = 4.0
>>> # make an empty hdr
>>> empty_hdr = c_mk_empty_sachdr()
>>>

Access and update time information
------------------------------
>>> s = c_rd_sac('1.sac')
>>> # Access reference time, start time, and end time. The return an object of `datetime`.
>>> tref = s.reference_time()
>>> tref = s.start_time()
>>> tref = s.end_time()
>>>
>>> # Update sac header by changing reference time and set it to the origin time.
>>> s.set_reference_time((2010, 01, 03, 14, 59, 30, 300), is_origin=True)
>>>
>>> # Shift the time by change 'b, e, o, a, f, t0, t1,...,t9' in the sac header.
>>> # This won't change reference time.
>>> s.shift_time(10.32) # delay time series for 10.32 sec
>>>

Operate massive sac files
------------------------------

>>> # read sachdr given filename wildcard, and restrict b<=3600 and e>=7200
>>> # The `vol` will be a list of (hdr, filename)
>>> vol = c_rd_sachdr_wildcard('II.*.00.sac', lcalda=True, critical_time_window= (3600, 7200) )
>>>
>>> # read many sac files with same cuttin method
>>> # Return a list `hdrs` and a matrix `mat` each line of which correspond to a sac time series
>>> # The calling will jump over wrong/broken sac files
>>> fnms = ('1.sac', '2.sac', '3.sac' )
>>> hdrs, mat = c_rd_sac_mat(fnms, -3, 1000, 2000, lcalda=True, scale=True, filter=('BP', 0.2, 1.0), verbose=False )
>>>


Whitening
------------------------------

>>> s = c_rd_sac('1.sac')
>>> winlen_sec, f1, f2 = 128.0, 0.02, 0.0667
>>> s.tnorm(winlen_sec, f1, f2, water_level_ratio= 1.0e-5, taper_halfsize=0)
>>>
>>> winlen_hz = 0.02
>>> s.fwhiten(winlen_hz, water_level_ratio= 1.0e-5, taper_halfsize=0)
>>>
"""
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from struct import unpack, pack
import sys
from glob import glob
import pickle

from pyfftw.interfaces.numpy_fft import irfft, rfft
from sacpy.geomath import haversine, azimuth
from sacpy.processing import iirfilter_f32, taper, detrend, cut, tnorm_f32, fwhiten_f32
from obspy.signal.interpolation import lanczos_interpolation
#import sacpy.processing as processing
from os.path import exists as os_path_exists
from os.path import abspath as os_path_abspath
from sacpy.c_src._lib_sac import lib as libsac
from sacpy.c_src._lib_sac import ffi as ffi
from h5py import File as h5_File
import ctypes
from datetime import datetime, timedelta
###
#  dependend methods
###
def deprecated_run(func):
    def wrapper_deprecated_run(*args, **kwargs):
        print('%s() will be deprecated soon!' % func.__name__)
        return func(*args, **kwargs)
    return wrapper_deprecated_run
@deprecated_run
def rd_sac(filename, lcalda=False):
    """
    Read sac given `filename`, and return an object ot sactrace.
    """
    tmp = sactrace()
    tmp.read(filename, lcalda=lcalda)
    return tmp
@deprecated_run
def rd_sac_2(filename, tmark, t1, t2, lcalda=False):
    """
    Read sac data given filename and time window, and return an object ot sactrace.
    tmakr: 'b', 'e', 'o', 'a', 't0', 't1', ... 't9';
    t1, t2: float
    """
    tmp = sactrace()
    tmp.read_2(filename, tmark, t1, t2, lcalda=lcalda)
    return tmp
@deprecated_run
def rd_sachdr(filename, lcalda=False):
    """
    Read sac header given `filename`, and return an object ot sachdr.
    """
    tmp = sachdr()
    tmp.read(filename, 'filename', lcalda=lcalda)
    return tmp
@deprecated_run
def rd_sac_mat(sacfnm_lst, tmark, t1, t2, norm_each='pos', bp_range=None, warning_msg=True ):
    """
    Read a list of sac files, and form 2D matrix for those time series.

    tmark, t1, t2:
    norm_each: 'pos' to normalize each time series with the max positive amplitude. (default)
               'neg' ...                                        negative ...
               'abs' ...                                        absolute ...

    bp_range:  None(default) or (f1, f2) for bandpass filter.
    warning_msg: True or False to output warning message due to non-existence of sac files.

    Return (hdr_lst, tr_lst, mat).

    Note, if any of sac file does not exist, the corresponding item in the `hdr_lst` and the `tr_lst` will be None,
    and the row in the `mat` will be filled with ZEROs.
    """
    tr_lst = list()
    hdr_lst = list()
    ncol = 1
    for sacfnm in sacfnm_lst:
        if os_path_exists(sacfnm):
            tr = rd_sac(sacfnm)
            if bp_range != None:
                tr.detrend()
                tr.taper()
                tr.bandpass(bp_range[0], bp_range[1], 2, 2)
            tr = truncate_sac(tr, tmark, t1, t2)
            ####
            if norm_each != None:
                tr.norm(norm_each)
            ####
            tr_lst.append( tr )
            hdr_lst.append( tr.hdr )
            ncol = max(tr.dat.size, ncol)
        else:
            if warning_msg:
                print('Warnning, file does not exist! ', sacfnm)
            tr_lst.append(None)
            hdr_lst.append(None)
    ######
    nrow = len(tr_lst)
    mat  = np.zeros((nrow, ncol), dtype=np.float32 )
    for irow in range(nrow):
        if tr_lst[irow] == None:
            continue
        else:
            npts = tr_lst[irow].dat.size
            mat[irow,:npts] = tr_lst[irow].dat
    return hdr_lst, tr_lst, mat
@deprecated_run
def make_sachdr(delta, npts, b, **kwargs):
    """
    Generate sac header, and return an object of `sachdr`.
    """
    tmp = sachdr()
    tmp.init(delta, npts, b, **kwargs)
    return tmp
@deprecated_run
def make_sactrace_hdr(dat, hdr):
    """
    Generate and return an object of sactrace.
    #
    hdr: an object of sachdr;
    dat: an object of np.ndarray
    """
    tmp = sactrace()
    tmp.init(dat, hdr)
    return tmp
@deprecated_run
def make_sactrace_v(dat, delta, b, **kwargs):
    """
    Generate and return an object of sactrace.
    #
    delta,, b, kwargs:
    dat: an object of np.ndarray
    """
    hdr = make_sachdr(delta, np.size(dat), b, **kwargs)
    tmp = sactrace()
    tmp.init(dat, hdr)
    return tmp
@deprecated_run
def wrt_sac(filename, dat, hdr):
    """
    Write sac file given dat (an object of numpy.ndarray) and hdr (an object of sachdr)
    """
    tmp = sactrace()
    tmp.init(dat, hdr, False)
    tmp.write(filename)
@deprecated_run
def wrt_sac_2(filename, dat, delta, b, **kwargs):
    """
    Write sac file given dat (an object of numpy.ndarray) and header settings.
    """
    tmp = sactrace()
    tmp.init(dat, make_sachdr(delta, np.size(dat), b, **kwargs), False)
    tmp.write(filename)
@deprecated_run
def truncate_sac(sac_trace, tmark, t1, t2, clean_sachdr=False):
    """
    Generate a new SAC_TRACE object from an existed SAC_TRACE 
    given reference tmark, and time window.
    tmark: '0', 'b', 'e', 'o', 'a', 't0', 't1', ... 't9';
        '0' means to use the built-in time axis according to 'b'.
    t1, t2: float;
    clean_sachdr: set -12345 for sachdr elements that exclude `b`, `e`, `delta`
    Return: a new SAC_TRACE object.
    """
    tmp = deepcopy(sac_trace)
    tmp.truncate(tmark, t1, t2)
    if clean_sachdr:
        return make_sactrace_v(tmp['dat'], tmp['delta'], tmp['b'])
    return tmp
@deprecated_run
def correlation_sac(sac_trace1, sac_trace2):
    """
    Compute the cross-correlation between two SAC_TRACE objects, and
    return a new SAC_TRACE object.
    The definition of cross-correlation is:
    
        cc(t) = \int st1(t+\tau) st2(\tau) d\tau
    
    Please note cc{st1, st2} = reversed cc{st2, st1}

    """
    cc = correlate(sac_trace1['dat'], sac_trace2['dat'], 'full', 'fft')
    cc_start = sac_trace1['b'] - sac_trace2['e']
    return make_sactrace_v(cc, sac_trace1['delta'], cc_start)
@deprecated_run
def stack_sac(sac_trace_lst, amp_norm=False):
    """
    Stack a list of `sactrace` and return an object of `sactrace`.
    The stacking use the maximal `b` and the minimal `e` in sachdr.

    amp_norm: True to apply amplitude normalizatin before stacking.
    """
    ### one sac
    if len(sac_trace_lst) == 1:
        st = make_sactrace_v(sac_trace_lst[0]['dat'], sac_trace_lst[0]['delta'], sac_trace_lst[0]['b'], nstack= 1, user0 = 1)
        return st
    ### many sac
    b = np.max([it['b'] for it in sac_trace_lst] )
    e = np.min([it['e'] for it in sac_trace_lst] )
    if b >= e:
        print('Err in `stack_sac(...)`. The maximal `b` is larger than the minimal `e`.' )
        raise Exception
    ### the basic sac trace
    st = truncate_sac(sac_trace_lst[0], '0', b, e, clean_sachdr=True)
    if amp_norm:
        st.norm()
    st['nstack'] = len(sac_trace_lst)
    st['user0']  = len(sac_trace_lst)
    ### stack
    npts = st['npts']
    for it in sac_trace_lst[1:]:
        tmp_sac = truncate_sac(it, '0', b, e)
        if amp_norm:
            tmp_sac.norm()
        tmp_npts = tmp_sac['npts']
        if npts != tmp_npts:
            sz = min(tmp_npts, npts)
            st['dat'][:sz] += tmp_sac['dat'][:sz]
        else:
            st['dat'] += tmp_sac['dat']
    return st
@deprecated_run
def time_shift_all_sac(sac_trace, t_shift_sec):
    st = deepcopy(sac_trace)
    st.shift_time_all(t_shift_sec)
    return st
@deprecated_run
def optimal_timeshift_cc_sac(st1, st2, min_timeshift=-1.e12, max_timeshift=1.e12, cc_amp= 'pos', search_time_window=None):
    """
    Use cross-correlation method to search the optimal 
    time shift between sac traces. 
    Return (t, coef, cc) where `t` is the optimal time 
    shift, `coef` the correlation coefficient, and `cc`
    the cross-correlation traces in SACTRACE.
    """
    cc = None
    if search_time_window:
        t1, t2 = search_time_window
        cc = correlation_sac(truncate_sac(st1, '0', t1, t2), truncate_sac(st2, '0', t1, t2) )
    else:
        cc = correlation_sac(st1, st2)
    cc.truncate('0', min_timeshift, max_timeshift)
    t, coef = cc.max_amplitude_time(cc_amp )
    return t, coef, cc
@deprecated_run
def plot_sac_lst(st_lst, ax=None):
    fig = None
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    for isac, it in enumerate(st_lst):
        junk = deepcopy(it)
        junk.norm()
        junk['dat'] = junk['dat']*0.4 + isac
        junk.plot_ax(ax, color='k', linewidth= 0.6)
    return ax
###
#  classes
###
class sachdr:
    """
    Sac header archive.
    """
    f_keys = [  'delta', 'depmin', 'depmax', 'scale', 'odelta',  
                'b', 'e', 'o', 'a', 'internal1',
                't0', 't1', 't2', 't3', 't4', 
                't5', 't6', 't7', 't8', 't9', 
                'f', 'resp0', 'resp1', 'resp2', 'resp3',
                'resp4', 'resp5', 'resp6', 'resp7', 'resp8',
                'resp9', 'stla', 'stlo', 'stel', 'stdp',
                'evla', 'evlo', 'evel', 'evdp', 'mag',
                'user0', 'user1', 'user2', 'user3', 'user4',
                'user5', 'user6', 'user7', 'user8', 'user9', 
                'dist',  'az',    'baz',   'gcarc', 'internal2',
                'internal3', 'depmen', 'cmpaz', 'cmpinc', 'xminimum',
                'xmaximum', 'yminimum', 'ymaximum', 'ADJTM', 'unused1',
                'unused2', 'unused3', 'unused4', 'unused5', 'unused6' ]
    i_keys = [  'nzyear', 'nzjday', 'nzhour', 'nzmin', 'nzsec',
                'nzmsec', 'nvhdr',  'norid',  'nevid', 'npts', 
                'internal4', 'nwfid','nxsize', 'nysize', 'unused7',
                'iftype', 'idep', 'iztype', 'unused8', 'iinst',
                'istreg', 'ievreg', 'ievtyp', 'iqual', 'isynth',
                'imagtyp', 'imagsrc', 'unused9', 'unused10', 'unused11',
                'unused12', 'unused13', 'unused14', 'unused15', 'unused16',
                'leven', 'lpspol', 'lovrok', 'lcalda', 'unused17' ]
    s_keys = [  'kstnm', 'kevnm',
                'khole', 'ko', 'ka',
                'kt0', 'kt1', 'kt2', 
                'kt3', 'kt4', 'kt5', 
                'kt6', 'kt7', 'kt8',
                'kt9', 'kf', 'kuser0',
                'kuser1', 'kuser2', 'kcmpnm',
                'knetwk', 'kdatrd', 'kinst' ]
    all_keys = ['delta', 'depmin', 'depmax', 'scale', 'odelta',  
                'b', 'e', 'o', 'a', 'internal1',
                't0', 't1', 't2', 't3', 't4', 
                't5', 't6', 't7', 't8', 't9', 
                'f', 'resp0', 'resp1', 'resp2', 'resp3',
                'resp4', 'resp5', 'resp6', 'resp7', 'resp8',
                'resp9', 'stla', 'stlo', 'stel', 'stdp',
                'evla', 'evlo', 'evel', 'evdp', 'mag',
                'user0', 'user1', 'user2', 'user3', 'user4',
                'user5', 'user6', 'user7', 'user8', 'user9', 
                'dist',  'az',    'baz',   'gcarc', 'internal2',
                'internal3', 'depmen', 'cmpaz', 'cmpinc', 'xminimum',
                'xmaximum', 'yminimum', 'ymaximum', 'unused1', 'unused2',
                'unused3', 'unused4', 'unused5', 'unused6', 'unused7',
                'nzyear', 'nzjday', 'nzhour', 'nzmin', 'nzsec',
                'nzmsec', 'nvhdr',  'norid',  'nevid', 'npts', 
                'internal4', 'nwfid','nxsize', 'nysize', 'unused8',
                'iftype', 'idep', 'iztype', 'unused9', 'iinst',
                'istreg', 'ievreg', 'ievtyp', 'iqual', 'isynth',
                'imagtyp', 'imagsrc', 'unused9', 'unused10', 'unused11',
                'unused12', 'unused13', 'unused14', 'unused15', 'unused16',
                'leven', 'lpspol', 'lovrok', 'lcalda', 'unused17',
                'kstnm', 'kevnm',
                'khole', 'ko', 'ka',
                'kt0', 'kt1', 'kt2', 
                'kt3', 'kt4', 'kt5', 
                'kt6', 'kt7', 'kt8',
                'kt9', 'kf', 'kuser0',
                'kuser1', 'kuser2', 'kcmpnm',
                'knetwk', 'kdatrd', 'kinst' ]
    little_endian_format  = '<70f40i192s' #'f' * 70 + 'i' * 40 + '192s'
    big_endian_format     = '>70f40i192s' #'f' * 70 + 'i' * 40 + '192s'
    ###
    def __init__(self):
        """
        Empty constructor
        """
        print('`sacpy.sac.sachdr` will be deprecated soon!')
        self.d_arch = dict()
        # default is time series
        self.d_arch['iftype'] = 1
        self.d_arch['nvhdr'] = 6
    ###
    def read(self, f, type='fid', lcalda=False):
        """
        Read sac header given `f` as string, or file object.
        f:
        type: filename
              fid (default)
        """
        if type == 'filename':
            self.d_arch['filename'] = deepcopy(f)
            f = open(f, 'rb')
        hdrvol = f.read(632)
        #print(sachdr.little_endian_format)
        info = unpack(sachdr.little_endian_format, hdrvol)
        info, small_endian_tag = (info, True) if 1< info[76] < 7 else (unpack(sachdr.big_endian_format, hdrvol), False)
        ###
        dict_f = dict()
        dict_i = dict()
        dict_s = dict()
        for k, value in zip(sachdr.f_keys, info[:70] ):
            dict_f[k] = value
        for k, value in zip(sachdr.i_keys, info[70:110]):
            dict_i[k] = value
        tmp = info[110].decode('utf8')
        dict_s['kstnm'] = tmp[:8]
        dict_s['kevnm'] = tmp[8:24]
        for idx, k in enumerate(sachdr.s_keys[2:] ) :
            dict_s[k] = tmp[idx*8+24: idx*8+32]
        ###
        self.d_arch.update( {**dict_f, **dict_i, **dict_s} )
        ###
        if lcalda and self.d_arch['lcalda'] == -12345:
            stlo, stla = self.d_arch['stlo'], self.d_arch['stla']
            evlo, evla = self.d_arch['evlo'], self.d_arch['evla']
            self.d_arch['lcalda'] = 1
            self.d_arch['gcarc'] = haversine(stlo, stla, evlo, evla)
            self.d_arch['baz'] = azimuth(stlo, stla, evlo, evla)
            self.d_arch['az'] = azimuth(evlo, evla, stlo, stla)
        return small_endian_tag
    def init(self, delta, npts, b, **kwargs):
        """
        Make a new hdr given delta, npts, b, and other settings.
        """
        # default unset
        for k in sachdr.f_keys:
            self.d_arch[k] = -12345.0
        for k in sachdr.i_keys:
            self.d_arch[k] = -12345
        for k in sachdr.s_keys:
            self.d_arch[k] = '-12345'
        # default is time series
        self.d_arch['iftype'] = 1
        self.d_arch['nvhdr'] = 6
        #
        self.d_arch['delta'] = delta
        self.d_arch['npts'] = npts
        self.d_arch['b'] = b
        self.d_arch['e'] = b + (npts-1)*delta
        for k, v in kwargs.items():
            self.d_arch[k] = v
    def pack(self):
        """
        Pack sachdr data into binary string for output.
        Return the generated packed binary string.
        """
        lst = [self.d_arch[k] for k in self.f_keys ]
        lst.extend( [self.d_arch[k] for k in self.i_keys ] )
        s = '{:\0<8.8}{:\0<16.16}'.format(self.d_arch['kstnm'], self.d_arch['kevnm'] )
        s += ''.join( ['{:\0<8.8}'.format(self.d_arch[k]) for k in self.s_keys[2:] ] )
        lst.append( bytes(s, 'utf8') )
        return pack('70f40i192s', *lst)
    ###
    def update(self, **kwargs):
        """
        Update sac hdr information.
        kwargs: dict of (key: value), eg: {'delta': 0.1, 'kstnm': 'HYD}
        """
        for k, v in kwargs.items():
            self.d_arch[k] = v
    ###
    def __update_npts__(self, npts):
        """
        update npts, and associated e.
        """
        self.d_arch['npts'] = npts
        self.d_arch['e'] = (npts-1) * self.d_arch['delta'] + self.d_arch['b']
    def __update_b__(self, b):
        """
        Update b, and dependent e.
        """
        self.d_arch['e'] = b - self.d_arch['b'] + self.d_arch['e']
        self.d_arch['b'] = b
    def __getitem__(self, key):
        """
        Accessing with specified key.
        """
        return self.d_arch[key]
    def __setitem__(self, key, value):
        """
        Set value with specified key.
        """
        self.d_arch[key] = value
    def __contains__(self, key):
        """
        Check whether a key in included.
        """
        return key in self.d_arch
    def __str__(self):
        keys = set(self.d_arch.keys())
        sub_keys = keys - set(sachdr.all_keys)
        s_lst = []
        s_lst.extend( ['internal {:9}: {:f}'.format(k, self.d_arch[k]) for k in sachdr.f_keys if self.d_arch[k] != -12345.0 ] )
        s_lst.extend( ['internal {:9}: {:d}'.format(k, self.d_arch[k]) for k in sachdr.i_keys  if self.d_arch[k] != -12345   ]  )
        s_lst.extend( ['internal {:9}: {:}'.format(k, self.d_arch[k]) for k in sachdr.s_keys  if self.d_arch[k][:6] != '-12345' ] )
        s_lst.extend( ['externel {:9}: {:}'.format(k, self.d_arch[k]) for k in sub_keys] )
        return '\n'.join(s_lst)
class sactrace:
    """
    Sac archive.
    """
    def __init__(self):
        print('`sacpy.sac.sactrace` will be deprecated soon!')
        self.hdr = sachdr()
        self.dat = None
    ### init
    def init(self, dat, hdr, deepcopy=True):
        if deepcopy:
            self.hdr = deepcopy(hdr)
            self.dat = deepcopy(dat)
        else:
            self.hdr = hdr
            self.dat = dat
        self['npts'] = np.size(dat)
    ### file io
    def read(self, filename, lcalda=False):
        """
        Read sac data given filename.
        """
        with open(filename, 'rb') as fid:
            small_endian_tag = self.hdr.read(fid, lcalda=lcalda )
            self.hdr['filename'] = filename
            self.dat = np.fromfile(fid, dtype=np.float32)
            if self.dat.size != self.hdr['npts']:
                print('Warning: mismatch NPTS (%d != %d) `%s`. NPTS in sachdr is updated with time series size ' % (
                            self.dat.size, self.hdr['npts'], filename ), file=sys.stderr, flush=True )
                self.hdr['npts'] = self.dat.size
                self.hdr['e'] = self.hdr['b'] + self.dat.size * self.hdr['delta']
            if not small_endian_tag:
                self.dat = self.dat.byteswap() #.newbyteorder()
            if self.is_nan_inf():
                print("Warning. Inf or Nan values in %s. All values set to ZEROs.", filename, flush=True )
                self.dat[:] = 0.0
    def write(self, filename):
        """
        Write data into specified file.
        """
        self['depmax'] = np.max(self['dat'])
        self['depmin'] = np.max(self['dat'])
        self['depmen'] = np.average(self['dat'] )
        self['e'] = self['b'] + (self['npts']-1) * self['delta']
        self.update_geometry()
        self.hdr.__update_npts__(np.size(self.dat) ) # update npts in case the dat length is revised.
        with open(filename, 'wb') as fid:
            fid.write(self.hdr.pack() )
            self.dat.astype(np.float32).tofile(fid)
    def read_2(self, filename, tmark, t1, t2, lcalda=False):
        """
        Read sac data given filename and time window.
        tmark: '0', 'b', 'e', 'o', 'a', 't0', 't1', ... 't9';
        t1, t2: float;
        """
        with open(filename, 'rb') as fid:
            small_endian_tag = self.hdr.read(fid, lcalda= lcalda)
            #####
            self.hdr['filename'] = filename
            i1 = self.__get_t_idx_absolute__(tmark, t1)
            i2 = self.__get_t_idx_absolute__(tmark, t2) + 1
            old_npts = self['npts']
            new_npts = i2-i1
            self.dat = np.zeros( new_npts, dtype=np.float32 )
            #####
            dj1 = 0
            rj1, rj2 = 0, old_npts
            if i1 > 0:
                rj1 = i1
            else:
                dj1 = -i1
            if i2 < old_npts:
                rj2 = i2
            #####
            if rj1 < rj2:
                if rj1 >0:
                    fid.read(rj1*4)
                self.dat[dj1:dj1+(rj2-rj1) ] = np.fromfile(fid, dtype=np.float32, count= (rj2-rj1)  )
                if not small_endian_tag:
                    self.dat = self.dat.byteswap() #.newbyteorder()
                if self.is_nan_inf():
                    print("Warning. Inf or Nan values in %s. All values set to ZEROs.", filename, flush=True )
                    self.dat[:] = 0.0
            #####
            self['npts'] = new_npts
            self['b'] = self['b'] + i1*self['delta']
            self['e'] = self['b'] + new_npts*self['delta']
    ### hdr methods
    def update_hdr(self, **kwargs):
        """
        Update sac header.
        kwargs: dict of (key: value), eg: {'delta': 0.1, 'kstnm': 'HYD}
        """
        self.hdr.update(**kwargs)
    def update_geometry(self):
        """
        update 'gcarc', 'baz', and 'az' using evlo, evla, stlo, and stla inside the header.
        """
        if self['evla'] != -12345.0 and self['evlo'] != -12345.0 and self['stla'] != -12345.0 and self['stlo'] != -12345.0:
            self['gcarc']= haversine(self['evlo'], self['evla'], self['stlo'], self['stla'])  
            self['az']   = azimuth(  self['evlo'], self['evla'], self['stlo'], self['stla']) 
            self['baz']  = azimuth(  self['stlo'], self['stla'], self['evlo'], self['evla']) 
    ### internel methods
    def __get_t_idx__(self, tmark, t):
        """
        !!! Deprecated methods !!!
        Given t, get the closest index.
        """
        print('Warnning `sactrace.__get_t_idx__` will be deprecated soon. Please use `sactrace.__get_t_idx_valid__` instead.', flush=True, file=sys.stderr)
        if tmark == '0':
            tmark = 'b'
            t = t - self['b']
        if tmark in self:
            t_ref = self[tmark]
            if t_ref == -12345.0:
                print('unset header for tmark: %s %f', tmark, t_ref)
                sys.exit(0)
            idx = int(np.round( (t_ref+t-self['b'])/self['delta'] ) )
            idx = min(max(idx, 0), self['npts']-1 )
            idx = max(0, idx)
            return idx
        else:
            print('Unrecognized tmark for sactrace.read_sac_2(...) ', tmark)
            sys.exit(0)
    def __get_t_idx_valid__(self, tmark, t):
        """
        Given t, get the closest and valid index [0, npts-1].
        """
        idx = self.__get_t_idx_absolute__(tmark, t)
        idx = min(max(idx, 0), self['npts']-1 )
        idx = max(0, idx)
        return idx
    def __get_t_idx_absolute__(self, tmark, t):
        """
        Given t, return the index in current time axis.
        The returned index can be out of [0, npts-1].
        """
        if tmark == '0':
            tmark = 'b'
            t = t - self['b']
        if tmark in self:
            t_ref = self[tmark]
            if t_ref == -12345.0:
                print('unset header for tmark: %s %f', tmark, t_ref)
                sys.exit(0)
            idx = int(np.round( (t_ref+t-self['b'])/self['delta'] ) )
            return idx
        else:
            print('Unrecognized tmark for sactrace.read_sac_2(...) ', tmark)
            sys.exit(0)
    def __getitem__(self, key):
        """
        Accessing with specified key.
        """
        if key == 'dat':
            return self.dat
        return self.hdr[key]
    def __setitem__(self, key, value):
        """
        Set value with specified key.
        """
        if key == 'dat':
            self.dat = value
            self.hdr['npts'] = np.size(value)
        self.hdr[key] = value
    def __contains__(self, key):
        """
        Check whether a key in included.
        """
        if key == 'dat' and self.dat:
            return True
        return key in self.hdr
    def __str__(self):
        s = self.hdr.__str__()
        s += '\ndata: (%d) ' % np.size(self.dat)
        s += self.dat.__str__()
        return s
    def __update_dat__(self, dat):
        """
        Update self.dat, and revise npts, e in hdr.
        """
        self.dat = dat
        self.hdr.__update_npts__(np.size(dat) )
    ### numerical methods
    def norm(self, method='abs'):
        """
        Norm max amplitude to 1
        norm: 'pos' to normalize the max positive amplitude.
              'neg' ...                  negative ...
              'abs' ...                  absolute ... (default)
        """
        max_pos = self.dat.max()
        max_neg = -self.dat.min()
        if method  == 'pos':
            self.dat *= (1.0/max_pos)
        elif method == 'neg':
            self.dat *= (1.0/max_neg)
        else:
            self.dat *= (1.0/ max(max_pos, max_neg) )
    def get_time_axis(self):
        """
        Get time axis.
        """
        return np.arange(0.0, self['npts'], 1.0) * self['delta'] + self['b']
    def truncate(self, tmark, t1, t2):
        """
        Truncate given reference tmark, and time window.
        tmark: '0', 'b', 'e', 'o', 'a', 't0', 't1', ... 't9';
            '0' means to use the built-in time axis according to 'b'.
        t1, t2: float;
        """
        if tmark == '0':
            tmark = 'b'
            t1 = t1 - self['b']
            t2 = t2 - self['b']
        i1 = self.__get_t_idx_absolute__(tmark, t1)
        i2 = self.__get_t_idx_absolute__(tmark, t2) + 1
        #
        # update data and header info
        if i1 >= 0 and i2 <= self.dat.size:
            self.dat= self.dat[i1:i2][:]
        else:
            #####
            old_i1   = i1 if i1>=0 else 0
            old_i2   = i2 if i2<=self.dat.size else self.dat.size
            #print(old_i1, old_i2)
            old_npts = old_i2 - old_i1
            #####
            new_npts = i2-i1
            new_dat = np.zeros(new_npts, dtype=np.float32 )
            new_i1 = 0 if i1 >= 0 else -i1
            new_i2 = new_i1+old_npts
            #####
            new_dat[new_i1: new_i2] = self.dat[old_i1:old_i2][:]
            self.dat = new_dat
        #########
        self['npts'] = i2-i1
        self['b'] = self['b'] + i1*self['delta']
        self['e'] = self['b'] + (self['npts']-1)*self['delta']
        #return self
    def taper(self, ratio= 0.05):
        """
        Taper using tukey window.
        """
        w = tukey(self['npts'], ratio)
        self.dat *= w
        #return self
    def rmean(self):
        """
        Remove mean value
        """
        self.dat -= np.average(self.dat)
        #return self
    def detrend(self):
        """
        Remove linear trend
        """
        self.dat = detrend(self.dat)
        #return self
    def bandpass(self, f1, f2, order= 2, npass= 1):
        """
        Bandpass
        """
        self.dat = processing_filter(self.dat, 1.0/self['delta'], 'bandpass', [f1, f2], order, npass )
    def lowpass(self, f, order= 2, npass= 1):
        """
        Lowpass
        """
        self.dat = processing_filter(self.dat, 1.0/self['delta'], 'lowpass', [f], order, npass )
    def highpass(self, f, order= 2, npass= 1):
        """
        High pass
        """
        self.dat = processing_filter(self.dat, 1.0/self['delta'], 'highpass', [f], order, npass )
    def resample(self, delta):
        """
        Resample the time-series using Fourier method.
        """
        if (delta % self['delta'] == 0.0):
            factor = int(delta // self['delta'])
            return self.decimate(factor)
        new_npts = int( round(delta/self['delta']* self['npts']) )
        self['dat'] = resample(self['dat'], new_npts)
        self['npts'] = new_npts
        self['delta'] = delta
        self['e'] = self['b'] + (new_npts-1)*delta
    def decimate(self, factor):
        """
        Downsample the time-series using scipy.signal.decimate.
        """
        self['dat'] = decimate(self['dat'], factor)
        self['npts'] = self['dat'].size
        self['delta'] = self['delta']*factor
        self['e'] = self['b'] + (self['npts']-1)*self['delta']
    def shift_time_all(self, tshift_sec):
        """
        Shift the time axis of the whole time-series. 
        This function changes all time related sachdr elements, that includes 'b, e, o, a, f, t0, t1,...,t9'

        shift_sec: t_shift in seconds.
        """
        self['b'] = self['b'] + tshift_sec
        self['e'] = self['e'] + tshift_sec
        for key in ['o', 'a', 'f', 't0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9']:
            if self[key] != -12345.0:
                self[key] = self[key] + tshift_sec 
    def shift_time_b_e(self, tshift_sec):
        """
        Shift the 'b' and 'e' of within the sachdr. All other time picks would not be changed.
        shift_sec: t_shift in seconds.
        """
        self['b'] = self['b'] + tshift_sec
        self['e'] = self['e'] + tshift_sec
    def max_amplitude_time_old(self, amp= 'abs'):
        """
        Obtain the (time, amplitude) for the max amplitude point. 
        amp:'abs' to search for the max absolute amplitude point.
            'pos' to search for the max positive amplitude point.
            'neg' to search for the max negative amplitude point.
            'neg' can be problematic if the whole time series is positive.
        Return: (time, amplitude)
        """
        imax = np.argmax(self['dat'])
        imin = np.argmin(self['dat'])
        if amp == 'pos':
            return imax*self['delta']+self['b'], self['dat'][imax]
        elif amp == 'neg':
            return imin*self['delta']+self['b'], self['dat'][imin]
        else:
            iabs = imax if self['dat'][imax] > -self['dat'][imin] else imin
            return iabs*self['delta']+self['b'], self['dat'][iabs]
    def max_amplitude_time(self, amp= 'abs', tmark=None, time_range = None):
        """
        Obtain the (time, amplitude) for the max amplitude point. 
        amp:'abs' to search for the max absolute amplitude point.
            'pos' to search for the max positive amplitude point.
            'neg' to search for the max negative amplitude point.
        Return: (time, amplitude)
        """
        i0, i1 = 0, self.dat.size
        if tmark != None and time_range != None:
            t0, t1 = time_range
            i0 = self.__get_t_idx_valid__(tmark, t0)
            i1 = self.__get_t_idx_valid__(tmark, t1) + 1
        ####
        if amp == 'pos':
            idx = np.argmax( self.dat[i0:i1] ) + i0
            t = idx*self['delta'] + self['b']
            return idx, t, self.dat[idx]
        elif amp == 'neg':
            idx = np.argmin( self.dat[i0:i1] ) + i0
            t = idx*self['delta'] + self['b']
            return idx, t, self.dat[idx]
        else:
            idx = np.argmax( np.abs(self.dat[i0:i1]) ) + i0
            t = idx*self['delta'] + self['b']
            return idx, t, self.dat[idx]

    def is_nan_inf(self):
        if (True in np.isnan(self.dat) or True in np.isinf(self.dat) ):
            return True
        return False
    def set_zero(self):
        self.dat[:] = 0.0
    ### plotf
    def plot_ax(self, ax, **kwargs):
        """
        Plot into specified axis, with **kwargs used by pyplot.plot(...).
        """
        ax.plot(self.get_time_axis(), self.dat, **kwargs)
        ax.set_xlim([self['b'], self['e'] ] )
        max_amp, min_amp = self.dat.max(), self.dat.min()
        for key in ['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9']:
            #print(key, self[key])
            if self[key] != -12345.0:
                ax.plot( [self[key], self[key]], [min_amp, max_amp], 'r' )
                ax.text(self[key], max_amp, self['k%s' % key ])
        #plt.show()
    def plot(self, **kwargs):
        """
        Plot and show, with **kwargs used by pyplot.plot(...).
        """
        plt.plot(self.get_time_axis(), self.dat, **kwargs)
        plt.xlim([self['b'], self['e'] ])
        #plt.close()
    ###
    def rfft(self, zeropad = 0):
        """
        condut rfft to sac.
        zeropad: 0: same length rfft;
                 1: pad zero to the next power of two ;
                 2: pad zero to the next next power of two;
        return (spectrum, df)
        """
        if zeropad == 0:
            return np.fft.rfft(self.dat), 1.0/(self['npts']*self['delta'])
        else:
            new_length = int(2**(np.ceil(np.log(self['npts'], 2))))
            if zeropad == 1:
                return np.fft.rfft(self.dat, new_length), 1.0/(new_length*self['delta'])
            elif zeropad == 2:
                return np.fft.rfft(self.dat, new_length*2), 0.5/(new_length*self['delta'])
            else:
                print('Unsupported argv(zeropad) for sactrace.rfft()')
    ###
    def fft(self, zeropad = 0):
        """
        condut fft to sac.
        zeropad: 0: same length rfft;
                 1: pad zero to the next power of two ;
                 2: pad zero to the next next power of two;
        return (spectrum, df)
        """
        if zeropad == 0:
            return np.fft.fft(self.dat), 1.0/(self['npts']*self['delta'])
        else:
            new_length = int(2**(np.ceil(np.log(self['npts'], 2))))
            if zeropad == 1:
                return np.fft.fft(self.dat, new_length), 1.0/(new_length*self['delta'])
            elif zeropad == 2:
                return np.fft.fft(self.dat, new_length*2), 0.5/(new_length*self['delta'])
            else:
                print('Unsupported argv(zeropad) for sactrace.fft()')
    ###
    @classmethod
    def benchmark(cls):
        pass
        #junk1 = cls()
        #junk1.read('1.sac')
        #h1 = junk1.hdr
        #h2 = make_sachdr(0.2, h1['npts'], 0.0, **{'kstnm': 'test'} )
        ##print(h1['delta'], h2['delta'])
        #print(h1, h2)
        #print(junk.hdr)
        #print(junk.get_time_axis() )
        #
        ###
        #
        ###



###
#  dependend methods based on C libraries
###
def c_rd_sachdr(filename, lcalda=False, verbose=False):
    """
    Read and return a sac header struct given the filename.

    The returned object is stored as a C Struct in the memory, and hence it doesn't support `deepcopy(...)`.
    You can use the methods `new_hdr = c_dup_sachdr(old_hdr)` to copy/duplicate and generate a new object.
    """
    hdr = ffi.new('SACHDR *')
    libsac.read_sachead(filename.encode('utf8'), hdr, verbose)
    if lcalda == True and hdr.lcalda == 0:
        hdr.lcalda = 1
        hdr.gcarc = haversine(hdr.stlo, hdr.stla, hdr.evlo, hdr.evla)
        hdr.baz   = azimuth(  hdr.stlo, hdr.stla, hdr.evlo, hdr.evla)
        hdr.az    = azimuth(  hdr.evlo, hdr.evla, hdr.stlo, hdr.stla)
    return hdr
def c_rd_sachdr_wildcard(fnm_wildcard=None, lcalda=False, tree=False, log_file=None, critical_time_window= None):
    """
    Read and return a list of tuple (hdr, filename) given the filename wildcard.

    The returned `hdr` object is stored as a C Struct in the memory, and hence it doesn't support `deepcopy(...)`.
    You can use the methods `new_hdr = c_dup_sachdr(old_hdr)` to copy/duplicate and generate a new object.
    """
    if not tree:
        buf = [ (c_rd_sachdr(it, lcalda), it) for it in  sorted(glob(fnm_wildcard)) ]
        if critical_time_window!= None:
            t0, t1 = critical_time_window
            buf = [(it, fnm) for (it, fnm) in buf if (it.b<=t0 and it.e>=t1) ]
        return buf
    else:
        buf = []
        dir_wildcard = '/'.join(fnm_wildcard.split('/')[:-1] )
        reminders = fnm_wildcard.split('/')[-1]
        for it in sorted(glob(dir_wildcard) ):
            wildcard = it + '/' + reminders
            if log_file != None:
                print('c_rd_sachdr_wildcard(%s)...' %  wildcard, file=log_file, flush=True )
            buf.append( (wildcard, c_rd_sachdr_wildcard(wildcard, lcalda, False, None, critical_time_window) ) )
        return buf
def c_mk_empty_sachdr():
    """
    Return an empty hdr.

    The returned object is stored as a C Struct in the memory, and hence it doesn't support `deepcopy(...)`.
    You can use the methods `new_hdr = c_dup_sachdr(old_hdr)` to copy/duplicate and generate a new object.
    """
    return c_dup_sachdr( ffi.addressof(libsac.sachdr_null) )
def c_mk_sachdr_time(b, delta, npts):
    """
    Make a new sac header object for time series given several time related parameters.

    The returned object is stored as a C Struct in the memory, and hence it doesn't support `deepcopy(...)`.
    You can use the methods `new_hdr = c_dup_sachdr(old_hdr)` to copy/duplicate and generate a new object.
    """
    ###
    hdr = c_mk_empty_sachdr()
    ###
    hdr.b = b
    hdr.delta = delta
    hdr.npts = npts
    ###
    hdr.e = b+(npts-1)*hdr.delta
    hdr.iztype = libsac.IO
    hdr.iftype = libsac.ITIME
    hdr.leven  = 1
    return hdr
def c_dup_sachdr(hdr):
    """
    Return a deepcopy of the existing hdr.

    Please use this method to copy an existing `hdr` object instead of `deepcopy(...)` that is not supported.
    """
    hdr2 = ffi.new('SACHDR *')
    libsac.copy_sachdr(hdr, hdr2)
    return hdr2

def c_rd_sac(filename, tmark=None, t1=None, t2=None, lcalda=False, scale=False, verbose=False):
    """
    Read sac given `filename`, and return an object ot sactrace.
    """
    tmp = c_sactrace(filename, tmark, t1, t2, lcalda, scale, verbose)
    if tmp.dat is None:
        return None
    return tmp
def c_rd_sac_mat(fnms, tmark, t1, t2, lcalda=False, scale=False, filter=None, verbose=False):
    """
    Return (hdrs, mat), where `hdrs` is a list of sachdr and mat is the matrix of data.
    If a sac file does not exist, then zeros will be used to fill the row in the matrix `mat`,
    and None will be used for the element in the `hdrs`.
    """
    buf = [c_sactrace(it, tmark, t1, t2, lcalda, scale, verbose) if os_path_exists(it) else None for it in fnms ]
    ###
    if filter:
        btype, f1, f2 = filter
        for it in buf:
            if it != None:
                it.rmean()
                it.detrend()
                it.filter(btype, (f1, f2), 2, 2)
    ###
    hdrs = [it.hdr if it !=None else None for it in buf  ]
    ###
    npts = np.max( [it.hdr.npts for it in buf if it !=None ] )
    mat = np.zeros( (len(fnms), npts), dtype=np.float32 )
    for irow, it in enumerate(buf):
        if it != None:
            mat[irow][:it.dat.size] = it.dat
    ###
    del buf
    return hdrs, mat
def c_wrt_sac(filename, xs, hdr, lcalda=False, verbose=False):
    """
    Write.
    """
    if lcalda:
        evlo, evla = hdr.evlo, hdr.evla
        stlo, stla = hdr.stlo, hdr.stla
        hdr.az = azimuth(evlo, evla, stlo, stla)
        hdr.baz = azimuth(stlo, stla, evlo, evla)
        hdr.gcarc = haversine(evlo, evla, stlo, stla)

    np_arr = np.array(xs, dtype= np.float32 )
    hdr.npts = np_arr.size
    hdr.e = hdr.b + hdr.delta * (np_arr.size - 1)
    ###
    cffi_arr = ffi.cast('float*', np_arr.ctypes.data )
    libsac.write_sac(filename.encode('utf8'), hdr, cffi_arr, verbose)
def c_wrt_sac2(filename, xs, b, delta, verbose=False):
    """
    Write.
    """
    np_arr = np.array(xs, dtype= np.float32 )
    ###
    cffi_arr = ffi.cast('float*', np_arr.ctypes.data )
    libsac.write_sac2(filename.encode('utf8'), np_arr.size, b, delta, cffi_arr, verbose )

def c_mk_sac(xs, b, delta):
    """
    Return a c_sactrace object.
    """
    st = c_sactrace()
    st.dat = np.array(xs, dtype=np.float32)
    st.hdr.b = b
    st.hdr.delta = delta
    st.hdr.e = b+delta*(xs.size-1)
    st.hdr.npts = xs.size
    return st
def c_truncate_sac(c_sactr, t1, t2):
    """
    Truncate an object of `c_sactrace` with the time window (t1, t2).
    Return a new object that is the truncated `c_sactrace`.
    """
    obj = c_sactr.duplicate()
    obj.truncate(t1, t2)
    return obj
###
#  functions to convert many sacs into a single hdf5 file
###
__hf_keys=( 'delta',     'depmin',    'depmax',    'scale',     'odelta',
            'b',         'e',         'o',         'a',         'internal1',
            't0',        't1',        't2',        't3',        't4',
            't5',        't6',        't7',        't8',        't9',
            'f',         'resp0',     'resp1',     'resp2',     'resp3',
            'resp4',     'resp5',     'resp6',     'resp7',     'resp8',
            'resp9',     'stla',      'stlo',      'stel',      'stdp',
            'evla',      'evlo',      'evel',      'evdp',      'mag',
            'user0',     'user1',     'user2',     'user3',     'user4',
            'user5',     'user6',     'user7',     'user8',     'user9',
            'dist',      'az',        'baz',       'gcarc',     'internal2',
            'internal3', 'depmen',    'cmpaz',     'cmpinc',    'unused2',
            'unused3',   'unused4',   'unused5',   'unused6',   'unused7',
            'unused8',   'unused9',   'unused10',  'unused11',  'unused12' )
__hi_keys=( 'nzyear',    'nzjday',    'nzhour',    'nzmin',     'nzsec',
            'nzmsec',    'nvhdr',     'internal5', 'internal6', 'npts',
            'internal7', 'internal8', 'unused13',  'unused14',  'unused15',
            'iftype',    'idep',      'iztype',    'unused16',  'iinst',
            'istreg',    'ievreg',    'ievtyp',    'iqual',     'isynth',
            'unused17',  'unused18',  'unused19',  'unused20',  'unused21',
            'unused22',  'unused23',  'unused24',  'unused25',  'unused26',
            'leven',     'lpspol',    'lovrok',    'lcalda',    'unused27'  )
__hs_keys=( 'kstnm',     'kevnm',
            'khole',     'ko',        'ka',
            'kt0',       'kt1',       'kt2',
            'kt3',       'kt4',       'kt5',
            'kt6',       'kt7',       'kt8',
            'kt9',       'kf',        'kuser0',
            'kuser1',    'kuser2',    'kcmpnm',
            'knetwk',    'kdatrd',    'kinst'  )
def sac2hdf5(fnms, hdf5_fnm, lcalda=False, info='', ignore_data=False, verbose=False):
    """
    Convert many sac files into a single hdf5 file.

    fnms:        a list of filenames for sac files.
    hdf5_fnm:    filename for output hdf5 file.
    lcalda:      (default is False).
    info:        An information string that output the hdf5.
    ignore_data: Ignore time series, and only save sachdr data in the output.
    verbose:     (default is False).

    Note: this function does not check validity of sac files. If some sac files
    are invalid (e.g., broken), then the related hdr data in hdf5 file will be
    meaningless and the related time series will be zeros.
    """
    fnmlst = fnms
    nfile = len(fnmlst)

    fid = h5_File(hdf5_fnm, 'w')
    fid.attrs['info'] = info
    fid.attrs['nfile'] = nfile
    fid.create_dataset('filename', data=[it.encode('ascii') for it in  fnmlst] )
    fid.create_dataset('LL', data=[it.split('.')[-2].encode('ascii') for it in fnmlst] )
    grp_hdr = fid.create_group('hdr')
    hdrs = [c_rd_sachdr(it, lcalda, verbose) for it in fnmlst]

    tmp_float = [ffi.cast('float*', it) for it in hdrs] ## floating hdr values
    for idx, nm in enumerate(__hf_keys):
        grp_hdr.create_dataset(nm, data=[it[idx] for it in tmp_float], dtype=np.float32  )

    tmp_int = [ffi.cast('int*', it) for it in hdrs] ## int hdr values
    for idx, nm in enumerate(__hi_keys):
        grp_hdr.create_dataset(nm, data=[it[idx+70] for it in tmp_int], dtype=np.int32  )

    tmp_char = [ffi.cast('char*', it) for it in hdrs] ## string hdr values
    ffi_string = ffi.string
    grp_hdr.create_dataset('kstnm', data=[ffi_string(it[440:448]) for it in tmp_char],  dtype='S8' )
    grp_hdr.create_dataset('kevnm', data=[ffi_string(it[448:464]) for it in tmp_char],  dtype='S8' )
    for idx, nm in enumerate(__hs_keys[2:]):
        i1, i2 = 464+idx*8, 472+idx*8
        grp_hdr.create_dataset(nm, data=[ffi_string(it[i1:i2]) for it in tmp_char],  dtype='S8' )

    if not ignore_data:
        shape = nfile, np.max(grp_hdr['npts'])
        mat = np.zeros(shape, dtype=np.float32 )
        for irow, it in enumerate(fnmlst):
            try:
                tmp = c_rd_sac(it).dat
                mat[irow,:tmp.size] = tmp
            except:
                if verbose:
                    print('Jump over error data reading %s' % (it), file=sys.stderr)
        fid.create_dataset('dat', data=mat, dtype=np.float32)
    fid.close()
    return
def hdf52sac(hdf5_fnm, output_prefix, verbose=False):
    """
    Convert a single hdf5 file into many sac files into.
    The hdf5 file is generated with sac2hdf5(...).
    """
    fid = h5_File(hdf5_fnm, 'r')
    nfile = fid.attrs['nfile']

    grp_hdr = fid['hdr']
    hdr_dict = dict()
    delta     = grp_hdr['delta'][:]
    depmin    = grp_hdr['depmin'][:]
    depmax    = grp_hdr['depmax'][:]
    scale     = grp_hdr['scale'][:]
    odelta    = grp_hdr['odelta'][:]
    b         = grp_hdr['b'][:]
    e         = grp_hdr['e'][:]
    o         = grp_hdr['o'][:]
    a         = grp_hdr['a'][:]
    internal1 = grp_hdr['internal1'][:]
    t0        = grp_hdr['t0'][:]
    t1        = grp_hdr['t1'][:]
    t2        = grp_hdr['t2'][:]
    t3        = grp_hdr['t3'][:]
    t4        = grp_hdr['t4'][:]
    t5        = grp_hdr['t5'][:]
    t6        = grp_hdr['t6'][:]
    t7        = grp_hdr['t7'][:]
    t8        = grp_hdr['t8'][:]
    t9        = grp_hdr['t9'][:]
    f         = grp_hdr['f'][:]
    resp0     = grp_hdr['resp0'][:]
    resp1     = grp_hdr['resp1'][:]
    resp2     = grp_hdr['resp2'][:]
    resp3     = grp_hdr['resp3'][:]
    resp4     = grp_hdr['resp4'][:]
    resp5     = grp_hdr['resp5'][:]
    resp6     = grp_hdr['resp6'][:]
    resp7     = grp_hdr['resp7'][:]
    resp8     = grp_hdr['resp8'][:]
    resp9     = grp_hdr['resp9'][:]
    stla      = grp_hdr['stla'][:]
    stlo      = grp_hdr['stlo'][:]
    stel      = grp_hdr['stel'][:]
    stdp      = grp_hdr['stdp'][:]
    evla      = grp_hdr['evla'][:]
    evlo      = grp_hdr['evlo'][:]
    evel      = grp_hdr['evel'][:]
    evdp      = grp_hdr['evdp'][:]
    mag       = grp_hdr['mag'][:]
    user0     = grp_hdr['user0'][:]
    user1     = grp_hdr['user1'][:]
    user2     = grp_hdr['user2'][:]
    user3     = grp_hdr['user3'][:]
    user4     = grp_hdr['user4'][:]
    user5     = grp_hdr['user5'][:]
    user6     = grp_hdr['user6'][:]
    user7     = grp_hdr['user7'][:]
    user8     = grp_hdr['user8'][:]
    user9     = grp_hdr['user9'][:]
    dist      = grp_hdr['dist'][:]
    az        = grp_hdr['az'][:]
    baz       = grp_hdr['baz'][:]
    gcarc     = grp_hdr['gcarc'][:]
    internal2 = grp_hdr['internal2'][:]
    internal3 = grp_hdr['internal3'][:]
    depmen    = grp_hdr['depmen'][:]
    cmpaz     = grp_hdr['cmpaz'][:]
    cmpinc    = grp_hdr['cmpinc'][:]
    unused2   = grp_hdr['unused2'][:]
    unused3   = grp_hdr['unused3'][:]
    unused4   = grp_hdr['unused4'][:]
    unused5   = grp_hdr['unused5'][:]
    unused6   = grp_hdr['unused6'][:]
    unused7   = grp_hdr['unused7'][:]
    unused8   = grp_hdr['unused8'][:]
    unused9   = grp_hdr['unused9'][:]
    unused10  = grp_hdr['unused10'][:]
    unused11  = grp_hdr['unused11'][:]
    unused12  = grp_hdr['unused12'][:]
    nzyear    = grp_hdr['nzyear'][:]
    nzjday    = grp_hdr['nzjday'][:]
    nzhour    = grp_hdr['nzhour'][:]
    nzmin     = grp_hdr['nzmin'][:]
    nzsec     = grp_hdr['nzsec'][:]
    nzmsec    = grp_hdr['nzmsec'][:]
    nvhdr     = grp_hdr['nvhdr'][:]
    internal5 = grp_hdr['internal5'][:]
    internal6 = grp_hdr['internal6'][:]
    npts      = grp_hdr['npts'][:]
    internal7 = grp_hdr['internal7'][:]
    internal8 = grp_hdr['internal8'][:]
    unused13  = grp_hdr['unused13'][:]
    unused14  = grp_hdr['unused14'][:]
    unused15  = grp_hdr['unused15'][:]
    iftype    = grp_hdr['iftype'][:]
    idep      = grp_hdr['idep'][:]
    iztype    = grp_hdr['iztype'][:]
    unused16  = grp_hdr['unused16'][:]
    iinst     = grp_hdr['iinst'][:]
    istreg    = grp_hdr['istreg'][:]
    ievreg    = grp_hdr['ievreg'][:]
    ievtyp    = grp_hdr['ievtyp'][:]
    iqual     = grp_hdr['iqual'][:]
    isynth    = grp_hdr['isynth'][:]
    unused17  = grp_hdr['unused17'][:]
    unused18  = grp_hdr['unused18'][:]
    unused19  = grp_hdr['unused19'][:]
    unused20  = grp_hdr['unused20'][:]
    unused21  = grp_hdr['unused21'][:]
    unused22  = grp_hdr['unused22'][:]
    unused23  = grp_hdr['unused23'][:]
    unused24  = grp_hdr['unused24'][:]
    unused25  = grp_hdr['unused25'][:]
    unused26  = grp_hdr['unused26'][:]
    leven     = grp_hdr['leven'][:]
    lpspol    = grp_hdr['lpspol'][:]
    lovrok    = grp_hdr['lovrok'][:]
    lcalda    = grp_hdr['lcalda'][:]
    unused27  = grp_hdr['unused27'][:]
    kstnm     = grp_hdr['kstnm'][:]
    kevnm     = grp_hdr['kevnm'][:]
    khole     = grp_hdr['khole'][:]
    ko        = grp_hdr['ko'][:]
    ka        = grp_hdr['ka'][:]
    kt0       = grp_hdr['kt0'][:]
    kt1       = grp_hdr['kt1'][:]
    kt2       = grp_hdr['kt2'][:]
    kt3       = grp_hdr['kt3'][:]
    kt4       = grp_hdr['kt4'][:]
    kt5       = grp_hdr['kt5'][:]
    kt6       = grp_hdr['kt6'][:]
    kt7       = grp_hdr['kt7'][:]
    kt8       = grp_hdr['kt8'][:]
    kt9       = grp_hdr['kt9'][:]
    kf        = grp_hdr['kf'][:]
    kuser0    = grp_hdr['kuser0'][:]
    kuser1    = grp_hdr['kuser1'][:]
    kuser2    = grp_hdr['kuser2'][:]
    kcmpnm    = grp_hdr['kcmpnm'][:]
    knetwk    = grp_hdr['knetwk'][:]
    kdatrd    = grp_hdr['kdatrd'][:]
    kinst     = grp_hdr['kinst'][:]
    mat = fid['dat']
    fnmlst = fid['filename']
    ###
    hdr = c_mk_empty_sachdr()
    for idx in range(nfile):
        hdr.delta     = delta[idx]
        hdr.depmin    = depmin[idx]
        hdr.depmax    = depmax[idx]
        hdr.scale     = scale[idx]
        hdr.odelta    = odelta[idx]
        hdr.b         = b[idx]
        hdr.e         = e[idx]
        hdr.o         = o[idx]
        hdr.a         = a[idx]
        hdr.internal1 = internal1[idx]
        hdr.t0        = t0[idx]
        hdr.t1        = t1[idx]
        hdr.t2        = t2[idx]
        hdr.t3        = t3[idx]
        hdr.t4        = t4[idx]
        hdr.t5        = t5[idx]
        hdr.t6        = t6[idx]
        hdr.t7        = t7[idx]
        hdr.t8        = t8[idx]
        hdr.t9        = t9[idx]
        hdr.f         = f[idx]
        hdr.resp0     = resp0[idx]
        hdr.resp1     = resp1[idx]
        hdr.resp2     = resp2[idx]
        hdr.resp3     = resp3[idx]
        hdr.resp4     = resp4[idx]
        hdr.resp5     = resp5[idx]
        hdr.resp6     = resp6[idx]
        hdr.resp7     = resp7[idx]
        hdr.resp8     = resp8[idx]
        hdr.resp9     = resp9[idx]
        hdr.stla      = stla[idx]
        hdr.stlo      = stlo[idx]
        hdr.stel      = stel[idx]
        hdr.stdp      = stdp[idx]
        hdr.evla      = evla[idx]
        hdr.evlo      = evlo[idx]
        hdr.evel      = evel[idx]
        hdr.evdp      = evdp[idx]
        hdr.mag       = mag[idx]
        hdr.user0     = user0[idx]
        hdr.user1     = user1[idx]
        hdr.user2     = user2[idx]
        hdr.user3     = user3[idx]
        hdr.user4     = user4[idx]
        hdr.user5     = user5[idx]
        hdr.user6     = user6[idx]
        hdr.user7     = user7[idx]
        hdr.user8     = user8[idx]
        hdr.user9     = user9[idx]
        hdr.dist      = dist[idx]
        hdr.az        = az[idx]
        hdr.baz       = baz[idx]
        hdr.gcarc     = gcarc[idx]
        hdr.internal2 = internal2[idx]
        hdr.internal3 = internal3[idx]
        hdr.depmen    = depmen[idx]
        hdr.cmpaz     = cmpaz[idx]
        hdr.cmpinc    = cmpinc[idx]
        hdr.unused2   = unused2[idx]
        hdr.unused3   = unused3[idx]
        hdr.unused4   = unused4[idx]
        hdr.unused5   = unused5[idx]
        hdr.unused6   = unused6[idx]
        hdr.unused7   = unused7[idx]
        hdr.unused8   = unused8[idx]
        hdr.unused9   = unused9[idx]
        hdr.unused10  = unused10[idx]
        hdr.unused11  = unused11[idx]
        hdr.unused12  = unused12[idx]
        hdr.nzyear    = nzyear[idx]
        hdr.nzjday    = nzjday[idx]
        hdr.nzhour    = nzhour[idx]
        hdr.nzmin     = nzmin[idx]
        hdr.nzsec     = nzsec[idx]
        hdr.nzmsec    = nzmsec[idx]
        hdr.nvhdr     = nvhdr[idx]
        hdr.internal5 = internal5[idx]
        hdr.internal6 = internal6[idx]
        hdr.npts      = npts[idx]
        hdr.internal7 = internal7[idx]
        hdr.internal8 = internal8[idx]
        hdr.unused13  = unused13[idx]
        hdr.unused14  = unused14[idx]
        hdr.unused15  = unused15[idx]
        hdr.iftype    = iftype[idx]
        hdr.idep      = idep[idx]
        hdr.iztype    = iztype[idx]
        hdr.unused16  = unused16[idx]
        hdr.iinst     = iinst[idx]
        hdr.istreg    = istreg[idx]
        hdr.ievreg    = ievreg[idx]
        hdr.ievtyp    = ievtyp[idx]
        hdr.iqual     = iqual[idx]
        hdr.isynth    = isynth[idx]
        hdr.unused17  = unused17[idx]
        hdr.unused18  = unused18[idx]
        hdr.unused19  = unused19[idx]
        hdr.unused20  = unused20[idx]
        hdr.unused21  = unused21[idx]
        hdr.unused22  = unused22[idx]
        hdr.unused23  = unused23[idx]
        hdr.unused24  = unused24[idx]
        hdr.unused25  = unused25[idx]
        hdr.unused26  = unused26[idx]
        hdr.leven     = leven[idx]
        hdr.lpspol    = lpspol[idx]
        hdr.lovrok    = lovrok[idx]
        hdr.lcalda    = lcalda[idx]
        hdr.unused27  = unused27[idx]
        hdr.kstnm     = kstnm[idx]
        hdr.kevnm     = kevnm[idx]
        hdr.khole     = khole[idx]
        hdr.ko        = ko[idx]
        hdr.ka        = ka[idx]
        hdr.kt0       = kt0[idx]
        hdr.kt1       = kt1[idx]
        hdr.kt2       = kt2[idx]
        hdr.kt3       = kt3[idx]
        hdr.kt4       = kt4[idx]
        hdr.kt5       = kt5[idx]
        hdr.kt6       = kt6[idx]
        hdr.kt7       = kt7[idx]
        hdr.kt8       = kt8[idx]
        hdr.kt9       = kt9[idx]
        hdr.kf        = kf[idx]
        hdr.kuser0    = kuser0[idx]
        hdr.kuser1    = kuser1[idx]
        hdr.kuser2    = kuser2[idx]
        hdr.kcmpnm    = kcmpnm[idx]
        hdr.knetwk    = knetwk[idx]
        hdr.kdatrd    = kdatrd[idx]
        hdr.kinst     = kinst[idx]
        ys = mat[idx,:hdr.npts]
        #print(fnmlst[idx] )
        fnm = '%s%s' % (output_prefix, fnmlst[idx].decode('ascii').split('/')[-1] )
        if verbose:
            print(fnm)
        c_wrt_sac(fnm, ys, hdr, False, verbose)
###
#  classes based on C libraries
###
ffi_from_buffer = ffi.from_buffer
ffi_cast = ffi.cast
class c_sactrace:
    """
    The class `c_sactrace` is based on C libraries implemented in `c_src/...`.

    An `c_sactrace` object has two elements: #1. `c_sactrace.hdr` and #2. `c_sactrace.dat`.
    The 1st, `c_sactrace.hdr`, is stored as a C Struct in the memory, and it does not support
    `deepcopy(...)`, and please use `new_hdr = c_dup_sachdr(old_hdr)` to copy.
    The 2st, `c_sactrace.dat`, is a numpy.ndarray(dtype=np.float32). We recommend `dtype=np.float32`
    to avoid possible failure.
    """
    def __init__(self, fnm=None, tmark=None, t1=None, t2=None, lcalda=False, scale=False, verbose=False):
        """
        Creat an object of c_sactrace via reading from a sac file.
        If `fnm` is not provided, a c_sactrace object with empty hdr will be created.

        fnm: the sac filename that is a string.
        tmark:  default is None. can be -5, -3, 0, 1,...9 for 'b', 'o', 't0', 't1',...,'t9'.
        t1, t2: defaults are None. the time window to cut when reading.
        lcalda: default is False, and set True to enable the lcalda when reading.
        scale:  normalize the time series and put the inverse of the scaling factor into hdr.scale
        """
        self.hdr = c_dup_sachdr( ffi.addressof(libsac.sachdr_null) )
        self.dat  = None
        if fnm:
            if tmark == None or t1 == None and t2 == None:
                self.__read(fnm, lcalda, scale, verbose)
            else:
                self.__read2(fnm, tmark, t1, t2, lcalda, scale, verbose)
    def duplicate(self):
        """
        Return a new object that is the duplication of this object.
        """
        obj = c_sactrace()
        obj.hdr = c_dup_sachdr(self.hdr)
        obj.dat = deepcopy(self.dat)
        return obj
    def __read(self, fnm, lcalda=False, scale=False, verbose=False ):
        """
        Internal function.
        Read from a file.
        """
        buf = libsac.read_sac(fnm.encode('utf8'), self.hdr, scale, verbose)
        if buf == ffi.NULL: ## NAN happend in the data
            return
        buf = ffi.gc(buf, libsac.free)
        self.dat = np.frombuffer(ffi.buffer(buf, 4*self.hdr.npts), dtype=np.float32 )
        ###
        hdr = self.hdr
        self.hdr.e = hdr.b + hdr.delta * (hdr.npts - 1)
        if lcalda and self.hdr.lcalda != 1:
            self.hdr.lcalda = 1
            self.hdr.gcarc = haversine(hdr.stlo, hdr.stla, hdr.evlo, hdr.evla)
            self.hdr.baz   = azimuth(  hdr.stlo, hdr.stla, hdr.evlo, hdr.evla)
            self.hdr.az    = azimuth(  hdr.evlo, hdr.evla, hdr.stlo, hdr.stla)
    def __read2(self, fnm, tmark, t1, t2, lcalda=False, scale=False, verbose=False ):
        """
        Internal function.
        Read from a file with cutting method.
        tmark can be -5, -3, 0, 1,...9 for 'b', 'o', 't0', 't1',...,'t9'.
        """
        buf = libsac.read_sac2(fnm.encode('utf8'), self.hdr, tmark, t1, t2, scale, verbose)
        if buf == ffi.NULL: ## NAN happend in the data
            return
        buf = ffi.gc(buf, libsac.free)
        self.dat = np.frombuffer(ffi.buffer(buf, 4*self.hdr.npts), dtype=np.float32 )
        ###
        hdr = self.hdr
        self.hdr.e = hdr.b + hdr.delta * (hdr.npts - 1)
        if lcalda and self.hdr.lcalda != 1:
            self.hdr.lcalda = 1
            self.hdr.gcarc = haversine(hdr.stlo, hdr.stla, hdr.evlo, hdr.evla)
            self.hdr.baz   = azimuth(  hdr.stlo, hdr.stla, hdr.evlo, hdr.evla)
            self.hdr.az    = azimuth(  hdr.evlo, hdr.evla, hdr.stlo, hdr.stla)
    def write(self, fnm, verbose=False):
        """
        Write to a file in sac formate.
        """
        hdr = self.hdr
        hdr.npts = self.dat.size
        hdr = self.hdr
        hdr.e = hdr.b + hdr.delta * (hdr.npts - 1)
        xs = self.dat
        if xs.dtype != np.float32:
            xs = xs.astype(np.float32)
        cffi_arr = ffi_cast('const float *', ffi_from_buffer(xs) ) #ffi.cast('const float*', xs.ctypes.data)
        libsac.write_sac(fnm.encode('utf8'), hdr, cffi_arr, verbose)
    def get_time_axis(self):
        """
        Return time axis, that is an object of numpy.ndarray.
        """
        return np.arange(self.dat.size, dtype=np.float32) * self.hdr.delta + self.hdr.b
    def plot(self, ax=None, show=True, **kwargs):
        if ax != None:
            ax.plot(self.get_time_axis(), self.dat, **kwargs )
        else:
            plt.plot(self.get_time_axis(), self.dat, **kwargs )

        if show:
            plt.show()
    def norm(self, method='abs'):
        """
        Norm max amplitude to 1. Calling this function will revise hdr.scale.
        norm: 'pos' to normalize the max positive amplitude.
              'neg' ...                  negative ...
              'abs' ...                  absolute ... (default)
        """
        max_pos = self.dat.max()
        max_neg = -self.dat.min()
        if method  == 'pos':
            if max_pos > 0.0:
                v = (1.0/max_pos)
                self.hdr.scale *= v
                self.dat *= v
        elif method == 'neg':
            if max_neg > 0.0:
                (1.0/max_neg)
                self.hdr.scale *= v
                self.dat *= v
        else:
            v = max(max_pos, max_neg)
            if v > 0.0:
                self.hdr.scale *= v
                self.dat *= (1.0/ v )
    def rmean(self):
        """
        Remove mean value
        """
        self.dat -= np.mean(self.dat)
    def detrend(self):
        """
        """
        detrend(self.dat)
    def taper(self, half_ratio):
        """
        tukey window is used for the tapering.
        """
        taper(self.dat, int(self.dat.size*half_ratio) )
    def filter(self, btype, fs, order=2, npass=2, aproto=0):
        """
        btype:
            0 : low pass
            1 : high pass
            2 : band pass
            3 : band reject
        fs : cutoff frequency (f_low, f_high) for the filter.
        aproto: 0 : butterworth filter
            1 : bessel filter
            2 : chebyshev type i
            3 : chebyshev type ii
        """
        vol = {'LP':0, 'HP':1, 'BP': 2, 'BR': 3, 'lowpass':0, 'highpass':1, 'bandpass':2, 0: 0, 1: 1, 2: 2, 3: 3}
        iirfilter_f32(self.dat, self.hdr.delta, aproto, vol[btype], fs[0], fs[1], order, npass )
        #self.dat = processing_filter(self.dat, 1.0/self.hdr.delta, btype, fs, order, npass)
    def truncate(self, t1, t2):
        """
        """
        self.dat, nb = cut(self.dat, self.hdr.delta, self.hdr.b, t1, t2)
        hdr = self.hdr
        hdr.npts = self.dat.size
        hdr.b = nb
        hdr.e = nb + hdr.delta*(hdr.npts-1)
    def shift_time(self, tshift_sec):
        """
        Shift the time axis of the whole time-series.
        This function changes all time related sachdr elements, that includes 'b, e, o, a, f, t0, t1,...,t9'

        shift_sec: t_shift in seconds. Positive value to delay the time series.
        """
        hdr = self.hdr
        hdr.b += tshift_sec
        hdr.e += tshift_sec
        hdr.o += tshift_sec
        hdr.a += tshift_sec
        hdr.t0 += tshift_sec if hdr.t0 != -12345.0 else -12345.0
        hdr.t1 += tshift_sec if hdr.t1 != -12345.0 else -12345.0
        hdr.t2 += tshift_sec if hdr.t2 != -12345.0 else -12345.0
        hdr.t3 += tshift_sec if hdr.t3 != -12345.0 else -12345.0
        hdr.t4 += tshift_sec if hdr.t4 != -12345.0 else -12345.0
        hdr.t5 += tshift_sec if hdr.t5 != -12345.0 else -12345.0
        hdr.t6 += tshift_sec if hdr.t6 != -12345.0 else -12345.0
        hdr.t7 += tshift_sec if hdr.t7 != -12345.0 else -12345.0
        hdr.t8 += tshift_sec if hdr.t8 != -12345.0 else -12345.0
        hdr.t9 += tshift_sec if hdr.t9 != -12345.0 else -12345.0
    def set_reference_time(self, reference_time, is_origin=False):
        """
        Change the reference time by providing a `reference_time`.
        This will change the values of `b`, `e`, `a`, `o`, `t?`
        and the absolute time in fact won't change.

        The `reference_time` can be 1) a tuple of int (year, month, day, hour, minute, second, millisec),
        2) a tuple of int (year, jday, hour, minute, second, millisec), or 3) an object of datetime.

        is_origin: True if the `reference_time` is the origin time. Default is False.
        """
        new_ref = None
        if type(reference_time) == tuple:
            if len(reference_time) == 7:
                yyyy, mon, dd, hh, mm, ss, msec = reference_time
                new_ref = datetime(yyyy, mon, dd, hh, mm, ss, msec*1000)
            elif len(reference_time) == 6:
                yyyy, jjj, hh, mm, ss, msec = reference_time
                new_ref = datetime(yyyy, 1, 1, hh, mm, ss, msec*1000) + timedelta(days=jjj-1)
            else:
                print('Err: wrong reference time arg', reference_time)
                sys.exit(0)
        else:
            new_ref = reference_time
        old_ref = self.reference_time()
        self.shift_time((old_ref - new_ref).total_seconds() )

        hdr = self.hdr
        hdr.nzjday = new_ref.year
        hdr.nzjday = new_ref.timetuple().tm_yday
        hdr.nzhour = new_ref.hour
        hdr.nzmin  = new_ref.minute
        hdr.nzsec  = new_ref.second
        hdr.nzmsec = new_ref.microsecond//1000
        if is_origin:
            hdr.o = 0.0
    def reference_time(self):
        """
        Return the an object of `datetime` for the reference time.
        """
        hdr = self.hdr
        return datetime(hdr.nzyear, 1, 1, hdr.nzhour, hdr.nzmin, hdr.nzsec, hdr.nzmsec*1000) + timedelta(days=hdr.nzjday-1)
    def start_time(self):
        """
        Return the an object of `datetime` for the start time, that correspond to the B in the sac header.
        """
        return self.reference_time() + timedelta(seconds=self.hdr.b)
    def end_time(self):
        """
        Return the an object of `datetime` for the end time, that correspond to the E in the sac header.
        """
        return self.reference_time() + timedelta(seconds=self.hdr.e)
    def downsample(self, n):
        """
        Downsample given a factor `n` (an integer).

        Note: users should lowpass filter the data before downsampling if necessary.
        """
        self.dat = self.dat[::n]
        hdr = self.hdr
        hdr.delta *= n
        hdr.npts = self.dat.size
        hdr.e = hdr.b + hdr.delta*(hdr.npts-1)
    def upsample(self, n):
        """
        Upsample given a factor `n` (an integer) with fft method.
        """
        self.interpolate_npts(self.hdr.npts*n)
    def interpolate_npts(self, new_npts):
        """
        Interpolate with a new npts using fft method.

        Note: users should lowpass filter the data before if the interpolation is related to downsampling.
        """
        hdr = self.hdr
        old_npts = hdr.npts
        old_delta = hdr.delta
        s = rfft(self.dat)
        self.dat = irfft(s, new_npts).astype(np.float32) * (new_npts/old_npts)
        T = old_npts*old_delta
        new_delta = T/new_npts
        hdr.delta = new_delta
        hdr.npts = new_npts
        hdr.e = hdr.b + new_delta*(new_npts-1)
    def interpolate_delta(self, new_delta, force_lanczos=False):
        """
        Interpolate with a new delta.
        The calling will automatically determine to use `downsample(...)` or `upsample(...)`
        if the new delta is proportional to the old delta. If not, lanczos method will be used.
        Also, if `force_lanczos==True`, then lanczos method will always be used.

        Note: users should lowpass filter the data before if the interpolation is related to downsampling
        """
        hdr = self.hdr
        old_npts = hdr.npts
        old_delta = hdr.delta
        T = old_npts*old_delta
        new_npts = int(T/new_delta)

        downsample_n = old_npts // new_npts
        upsample_n   = new_npts // old_npts
        if not force_lanczos and downsample_n*new_npts == old_npts:
            self.downsample(downsample_n)
        elif not force_lanczos and upsample_n*old_npts == new_npts:
            self.upsample(upsample_n)
        else:
            xs = lanczos_interpolation(self.dat, hdr.b, old_delta, hdr.b, new_delta, new_npts, 20)
            self.dat = xs.astype(np.float32)
        hdr.delta = new_delta
        hdr.npts = new_npts
        hdr.e = hdr.b + new_delta*(new_npts-1)
    def max_amplitude_time(self, amp, t_range=None):
        """
        Get the (idx, time, amplitude) for the max amplitude point.

        Set `t_range=(t1, t2)` to search within a time window. In default, `t_range=None`
        will search for the whole time range.
        amp: 'pos' for max positive amplitude, and 'neg' for max negative amplitude, and 'abs'
        for max absolute amplitude.
        """
        x, i1 = self.dat, 0
        if t_range != None:
            t1, t2 = t_range
            i1 = libsac.get_valid_time_index(t1, self.hdr.delta, self.hdr.b, self.dat.size)
            i2 = libsac.get_valid_time_index(t2, self.hdr.delta, self.hdr.b, self.dat.size) + 1
            x = self.dat[i1:i2]
        ###
        if amp == 'pos':
            imax = np.argmax(x)
            return imax+i1, (imax+i1)*self.hdr.delta+self.hdr.b, x[imax]
        elif amp == 'neg':
            imin = np.argmin(x)
            return imin+i1, (imin+i1)*self.hdr.delta+self.hdr.b, x[imin]
        else:
            imax = np.argmax(x)
            imin = np.argmin(x)
            iabs = imax if x[imax]>-x[imin] else imin
            return iabs+i1, (iabs+i1)*self.hdr.delta+self.hdr.b, x[iabs]

    def tnorm(self, winlen, f1, f2, water_level_ratio= 1.0e-5, taper_halfsize=0):
        tnorm_f32(self.dat, self.hdr.delta, winlen, f1, f2, water_level_ratio, taper_halfsize)
    def fwhiten(self, winlen, water_level_ratio= 1.0e-5, taper_halfsize=0, speedup_i1= -1, speedup_i2= -1):
        fwhiten_f32(self.dat, self.hdr.delta, winlen, water_level_ratio, taper_halfsize, speedup_i1, speedup_i2)
##################################################################################################################

class scardec_stf:
    """
    Search for Source Time Function from SCARDEC Database.

    Please goto http://scardec.projects.sismo.ipgp.fr/ for reference and citations.

    >>> import matplotlib.pyplot as plt
    >>> from sacpy.sac import scardec_stf
    >>> app = scardec_stf()
    >>> as, stf, metadat = app.search_stf('1992-07-13-15-34-04.000', (158.63, 51.17), True)
    >>> print(metadat)
    >>> plt.plot(ts, stf)
    >>> plt.show()
    >>>
    """
    def __init__(self):
        sacpy_dir = '/'.join(os_path_abspath(__file__).split('/')[:-1] )
        stfs_fnm = '%s/bin/dataset/STFs_SCARDEC/stfs_scardec.h5' % sacpy_dir
        self.fid = h5_File(stfs_fnm, 'r')

        otime_str = list(self.fid.keys() ) # 1992-02-13-01-29-13.000
        otime_int = list()
        for it in otime_str:
            year, month, day, hour, minute, sec = [int(v) for v in it.split('.')[0].split('-')]
            time_stamp = sec + minute*100+ hour*10000+day*1000000 + month*100000000 + year*10000000000
            otime_int.append(time_stamp)

        self.otime_str = otime_str
        self.otime_int = np.array(otime_int, dtype=np.int64)
    def __del__(self):
        #self.fid.close()
        pass
    def search_stf(self, datetime_str, lonlat=None, verbose=True):
        """
        Search for STF from SCARDEC Source Time Functions Database

        datetime_str: the string for origin time. the format should be 'YYYY-MM-DD-HH-mm-SS.SSS'.
        lonlat: optional epicenter coordinate, that is a tuple (longitude, latitude).

        Return (ts, stf, metadata), in which `metadata` is a Python Dictionary object.
        """
        idx, flag_exact_time, flag_lonlat = 0, True, True
        try:
            idx = self.otime_str.index(datetime_str)
        except:
            flag_exact_time = False
            year, month, day, hour, minute, sec =  [int(v) for v in datetime_str.split('.')[0].split('-')]
            time_stamp = sec + minute*100+ hour*10000+day*1000000 + month*100000000 + year*10000000000 # int64 suits here for year 2XXX.
            idx = np.argmin( abs(self.otime_int-time_stamp) )

        key = self.otime_str[idx]
        grp = self.fid[key]
        evlo, evla = grp.attrs['lon'], grp.attrs['lat']
        if lonlat != None:
            if abs(evlo-lonlat[0]) > 1.0e-2 or abs(evla-lonlat[1]) > 1.0e-2:
                flag_lonlat = False
        stf, ts = grp['stf'][:], grp['ts'][:]

        str1 = '==' if flag_exact_time else '!='
        str2 = '==' if flag_lonlat     else '!='
        if lonlat:
            if verbose or (not flag_exact_time) or (not flag_lonlat):
                print('Find STF for %s%s%s at (%.2f, %.2f)%s(%.2f, %.2f)' % (key, str1, datetime_str, evlo, evla, str2, lonlat[0], lonlat[1]) )
        else:
            if verbose or (not flag_exact_time) or (not flag_lonlat):
                print('Find STF for %s%s%s at (%.2f, %.2f)None' % (key, str1, datetime_str, evlo, evla) )

        return ts, stf, dict(grp.attrs)

if __name__ == "__main__":
    fnm = 'test_tmp/1.sac'
    if False:
        hdr = c_rd_sachdr(fnm)
        print(hdr, hdr.b, hdr.e, hdr.npts, hdr.stlo, hdr.stla, hdr.evlo, hdr.evla, hdr.kstnm)
    if False:
        hdrs = c_rd_sachdr_wildcard('test_tmp/[12].sac')
        print(hdrs)
    if False:
        st = c_rd_sac(fnm)
        plt.plot(st.get_time_axis(), st.dat)
        st.rmean()
        st.detrend()
        st.taper(0.02)
        plt.plot(st.get_time_axis(), st.dat)
        plt.show()
    if False:
        st = c_rd_sac(fnm)
        plt.plot(st.get_time_axis(), st.dat)
        st.filter('BP', (0.1, 1.0), 2, 2)
        st.write('junk_bp.sac')
        plt.plot(st.get_time_axis(), st.dat)
        #plt.show()
    if False:
        st = c_rd_sac(fnm)
        print(st.hdr.b, st.hdr.e, st.hdr.npts, st.hdr.delta)
        plt.plot(st.get_time_axis(), st.dat, marker='o')

        st.truncate(300, 1000)
        st.taper(0.01)
        print(st.hdr.b, st.hdr.e, st.hdr.npts, st.hdr.delta)
        plt.plot(st.get_time_axis(), st.dat, marker='x')

        st.interpolate_delta(0.05)
        print(st.hdr.b, st.hdr.e, st.hdr.npts, st.hdr.delta)
        plt.plot(st.get_time_axis(), st.dat, marker='.')

        idx, time, amplitude = st.max_amplitude_time('pos')
        print(idx, time, amplitude, st.dat[idx] )
        plt.plot((time,), (amplitude,), 'r+')
        plt.show()

    if True:
        st = c_rd_sac('test_tmp/1.sac')
        st.set_reference_time((1999, 11, 16, 1, 23, 20, 200))
        st.write('junk1.sac')
        print(st.start_time())

        st = c_rd_sac('test_tmp/1.sac')
        st.set_reference_time((1999, 320, 1, 23, 20, 200))
        st.write('junk2.sac')
        print(st.start_time())

        st = c_rd_sac('test_tmp/1.sac')
        st.set_reference_time(datetime(1999, 11, 16, 1, 23, 20, 200000) )
        st.write('junk3.sac')
        print(st.start_time())
        sys.exit(0)
    #sac2hdf5('/home/catfly/00-LARGE-IMPORTANT-PERMANENT-DATA/AU_dowload/01_resampled_bhz_to_h5/01_workspace_bhz_sac/2000_008_16_47_20_+0000/*SAC',
    #            'junk.h5')
    #hdf52sac('junk.h5', 'junk/sac_', True)
    app = scardec_stf()
    ts, stf, metadat = app.search_stf('1992-07-13-15-34-04.000', (158.63, 51.17), False)
    print(metadat)
    plt.plot(ts, stf)
    plt.show()
