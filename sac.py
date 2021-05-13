#!/usr/bin/env python3
"""
SAC implementation by python3.x.

SACPY, a memory-resided package, is definitely faster than disk- associated SAC
in data processing. And, hugely massive methods, functions, packages, etc 
provided by Python communities make this SACPY much more flexible, extensible,
and convenient in both data processing, and program development.


File IO dependent methods
-------------------------

>>> # io, view,  basic processing
>>> s = rd_sac('1.sac')
>>> s.plot()       
>>> s.detrend()    
>>> s.taper(0.02) # ratio can be 0 ~ 0.5
>>> s.bandpass(0.5, 2.0, order= 4, npass= 2)
>>> s.write('1_new.sac')
>>> 
>>> # arbitrary plot
>>> ts = s.get_time_axis()
>>> plt.plot(ts, s['dat'], color='black') # ...
>>>
>>> # cut and read
>>> s = rd_sac_2('1.sac', 'e', -50, -20)
>>> s.detrend()    
>>> # some other processing...
>>> s.write('1_truncated.sac')

Arbitrary data writing, and processing methods
----------------------------------------------

>>> import numpy as np
>>> dat = np.random.random(1000)
>>> delta, b = 0.05, 50.0
>>> # writing method 1
>>> wrt_sac_2('junk.sac', dat, delta, b, **{'kstnm': 'syn', 'stlo': 0.0, 'stla': 0.0} )
>>> 
>>> # writing method 2, with processings
>>> s = make_sactrace_v(dat, delta, b, **{'kstnm': 'syn', 'stlo': 0.0, 'stla': 0.0} )
>>> s.taper()
>>> s.bandpass(0.5, 2.0, order= 4, npass= 2)
>>> s.write('junk2.sac')
>>>

Sac header update and revision
------------------------------

>>> import copy
>>> s = rd_sac('1.sac')
>>> new_hdr = deepcopy(s.hdr)
>>> new_hdr['t1'] = 2.0 # meaningless value, just for example
>>> new_hdr['t2'] = 4.0
>>> new_hdr.update( **{'stlo': -10.0, 'stla': 10.0, 'delta': 0.2 } )
>>> s_new = make_sactrace_hdr(s['dat'], new_hdr)
>>> s_new.write('1_new.sac')


Massive data IO and processing
------------------------------

>>> # read, process, and write a bunch of sac files in 3 lines
>>> fnm_lst = ['1.sac', '2.sac', '3.sac']
>>> for fnm in fnm_lst:
...     s = rd_sac(fnm)
...     s.detrend()
...     s.taper()
...     s.lowpass(1.2, order= 2, npass= 2)
...     s.write(s['filename'].replace('.sac', '_proced.sac') )
...
>>> 


"""
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from scipy.signal import tukey, detrend, decimate, correlate, resample
from struct import unpack, pack
import sys
from glob import glob
import pickle
from sacpy.geomath import haversine, azimuth, point_distance_to_great_circle_plane
from sacpy.processing import filter as processing_filter
from sacpy.processing import taper as processing_taper
#import sacpy.processing as processing
from os.path import exists as os_path_exists
from sacpy.c_src._sac_io import lib as libsac
from sacpy.c_src._sac_io import ffi as ffi
from h5py import File as h5_File
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
        if tmark is '0':
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
def c_rd_sachdr(filename=None, lcalda=False, verbose=False):
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
def c_mk_empty_sachdr():
    """
    Return an empty hdr.

    The returned object is stored as a C Struct in the memory, and hence it doesn't support `deepcopy(...)`.
    You can use the methods `new_hdr = c_dup_sachdr(old_hdr)` to copy/duplicate and generate a new object.
    """
    return c_dup_sachdr( ffi.addressof(libsac.sachdr_null) )
def c_dup_sachdr(hdr):
    """
    Return a deepcopy of the existing hdr.

    Please use this method to copy an existing `hdr` object instead of `deepcopy(...)` that is not supported.
    """
    hdr2 = ffi.new('SACHDR *')
    libsac.copy_sachdr(hdr, hdr2)
    return hdr2
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


def c_rd_sac(filename, tmark=None, t1=None, t2=None, lcalda=False, scale=False, verbose=False):
    """
    Read sac given `filename`, and return an object ot sactrace.
    """
    tmp = c_sactrace(filename, tmark, t1, t2, lcalda, scale, verbose)
    if tmp.dat is None:
        return None
    return tmp
def c_rd_sac_mat(fnms, tmark, t1, t2, lcalda=False, norm=None, filter=None, scale=False, verbose=False):
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
    if norm != None:
        for it in buf:
            if it != None:
                it.norm(norm)
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
def c_wrt_sac(filename, xs, hdr, verbose, lcalda=False):
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
def c_wrt_sac2(filename, xs, b, delta, verbose):
    """
    Write.
    """
    np_arr = np.array(xs, dtype= np.float32 )
    ###
    cffi_arr = ffi.cast('float*', np_arr.ctypes.data )
    libsac.write_sac2(filename.encode('utf8'), np_arr.size, b, delta, cffi_arr, verbose )

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
def sac2hdf5(fnm_wildcard, hdf5_fnm, lcalda=False, verbose=False, info='', ignore_data=False):
    """
    Convert many sac files into a single hdf5 file.
    """
    fnmlst = sorted(glob(fnm_wildcard) )
    nfile = len(fnmlst)

    fid = h5_File(hdf5_fnm, 'w')
    fid.attrs['info'] = info
    fid.attrs['nfile'] = nfile
    fid.create_dataset('filename', data=[it.encode('ascii') for it in  fnmlst] )
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
        c_wrt_sac(fnm, ys, hdr, False, True)
###
#  classes based on C libraries
###
class c_sactrace:
    """
    The class `c_sactrace` is based on C libraries implemented in `c_src/...`.

    An `c_sactrace` object has two elements: #1. `c_sactrace.hdr` and #2. `c_sactrace.dat`.
    The 1st, `c_sactrace.hdr`, is stored as a C Struct in the memory, and it does not support
    `deepcopy(...)`, and please use `new_hdr = c_dup_sachdr(old_hdr)` to copy.
    The 2st, `c_sactrace.dat`, is a numpy.ndarray. You can manipulate it, and we suggest to 
    use `dtype=np.float32` when manipulating it.
    """
    def __init__(self, fnm=None, tmark=None, t1=None, t2=None, lcalda=False, scale=False, verbose=False):
        """
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
                self.read(fnm, lcalda, scale, verbose)
            else:
                self.read2(fnm, tmark, t1, t2, lcalda, scale, verbose)
    def duplicate(self):
        """
        Return a new object that is the duplication of this object.
        """
        obj = c_sactrace()
        obj.hdr = c_dup_sachdr(self.hdr)
        obj.dat = deepcopy(self.dat)
        return obj
    def read(self, fnm, lcalda=False, scale=False, verbose=False ):
        """
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
    def read2(self, fnm, tmark, t1, t2, lcalda=False, scale=False, verbose=False ):
        """
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
        hdr.e = hdr.b + hdr.delta * (hdr.npts - 1)
        cffi_arr = ffi.cast('float*', self.dat.astype(np.float32).ctypes.data )
        libsac.write_sac(fnm.encode('utf8'), self.hdr, cffi_arr, verbose)
    def get_time_axis(self):
        return np.arange(self.dat.size) * self.hdr.delta + self.hdr.b
    def plot(self, ax=None, show=True, **kwargs):
        if ax != None:
            ax.plot(self.get_time_axis(), self.dat, **kwargs )
        else:
            plt.plot(self.get_time_axis(), self.dat, **kwargs )

        if show:
            plt.show()
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
            if max_pos > 0.0:
                self.dat *= (1.0/max_pos)
        elif method == 'neg':
            if max_neg > 0.0:
                self.dat *= (1.0/max_neg)
        else:
            v = max(max_pos, max_neg)
            if v > 0.0:
                self.dat *= (1.0/ v )
    def rmean(self):
        """
        Remove mean value
        """
        self.dat -= np.average(self.dat)
    def detrend(self):
        """
        """
        self.dat = detrend(self.dat)
    def taper(self, ratio):
        """
        tukey window is used for the tapering.
        """
        self.dat = processing_taper(self.dat, int(self.dat.size*ratio) )
    def filter(self, btype, fs, order=2, npass=2):
        """
        """
        self.dat = processing_filter(self.dat, 1.0/self.hdr.delta, btype, fs, order, npass)
    def truncate(self, t1, t2):
        """
        """
        i1 = libsac.get_absolute_time_index(t1, self.hdr.delta, self.hdr.b)
        i2 = libsac.get_absolute_time_index(t2, self.hdr.delta, self.hdr.b) + 1
        new_dat = np.zeros(i2-i1, dtype= np.float32 )
        ###
        n1, o1 = 0, i1
        o2 = i2
        if i1<0:
            n1 = -i1
            o1 = 0
        if i2 > self.dat.size:
            o2 = self.dat.size
        n2 = n1+(o2-o1)
        new_dat[n1:n2] = self.dat[o1:o2]
        ###
        self.dat = new_dat
        hdr = self.hdr
        hdr.npts = self.dat.size
        hdr.b = hdr.b + hdr.delta*i1
        hdr.e = hdr.b + hdr.delta*(self.dat.size-1)
    def max_amplitude_time(self, amp, t_range):
        """
        Get the (idx, time, amplitude) for the max amplitude point in the time range `t_range`.

        amp: 'pos' for max positive amplitude, and 'neg' for max negative amplitude
        """
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
##################################################################################################################
# Classes/method below are usually useless
##################################################################################################################
###
#  sactrace container
###
class sactrace_list(list):
    """
    Is this necessary ?
    """
    ### 
    def __int__(self):
        """
        Empty constructor
        """
        pass
    ### IO
    def read_fnm_lst(self, fnm_lst):
        """
        Read a list of sac filenames for initialization.
        """
        self.extend( [rd_sac(fnm) for fnm in fnm_lst] )
    def read_fnm_lst_2(self, fnm_lst, tmark, t1, t2):
        """
        Read a list of sac filenames with given time window setting for initialization.
        tmark, t1, t2: time window setting for read.
        """
        self.extend( [rd_sac_2(fnm, tmark, t1, t2) for fnm in fnm_lst] )
    def write(self, append_str):
        """
        Write, with filename appended by `append_str`.
        append_str: string; (use '' for over write)
        """
        for it in self:
            it.write(self['filename']+append_str)
    ### inplace dat revision processing
    def truncate(self, tmark, t1, t2):
        """
        """
        for it in self:
            it.truncate(tmark, t1, 2)
    def taper(self, ratio= 0.05):
        for it in self:
            it.taper(ratio= ratio)
    def rmean(self):
        for it in self:
            it.rmean()
    def detrend(self):
        for it in self:
            it.rmean()
            it.detrend()
    def bandpass(self, f1, f2, order=2, npass= 1):
        for it in self:
            it.bandpass(f1, f2, order= order, npass= npass)
    def lowpass(self, f, order=2, npass= 1):
        for it in self:
            it.lowpass(f, order= order, npass= npass)
    def highpass(self, f, order=2, npass= 1):
        for it in self:
            it.highpass(f, order= order, npass= npass)
    ### sort
    def sort(self, sachdr_key, sequence= 'ascend'):
        """
        Sort given sac header key.
        sachdr_key: sac head key to used for sorting;
        sequence: 'descend' or 'ascend';
        """
        #sorted(self, key=)
        tmp = {'descend': -1, 'ascend': 1}
        #print(np.argsort([it[sachdr_key] for it in self ] )[::tmp[sequence] ])
        lst = [self[idx] for idx in  np.argsort([it[sachdr_key] for it in self ] )[::tmp[sequence] ] ]
        self.clear()
        self.extend(lst)
    ### Into hdf5
    def tohdf5(self, filename):
        """
        """
        pass
    ### plot
    def plot1(self):
        """
        Similar to p1 in SAC. (need dev??)
        """
        ax = plt.subplot()
        for idx, tr in enumerate(self):
            ax.plot(tr.get_time_axis(), sactrace_list.normed(tr.dat)+idx )
        plt.show()
        #plt.close()
    def plot2(self):
        """
        Similar to p2 in SAC.
        """
        ax = plt.subplot(111)
        for  tr in self:
            ax.plot(tr.get_time_axis(), tr.dat, label= tr['filename'] )
        ax.legend(loc = 'upper right')
        plt.show()
    @staticmethod
    def normed(trace):
        scale = 1.0 / max(np.max(trace), -np.min(trace) )
        return trace * scale
###
#  hdf5 containing list of time series in sac format
###
class sactrace_list_hdf5():
    """
    """
    def __init__(self):
        """
        """
        pass
    def from_sactrace_list(self, filename, lst):
        """
        """
        pass
        #self.h5file = h5py.File(filename, 'w')
        #self.h5file.cr
###
#  sachdr container
#  - sachdr_list: list of `sachdr`
#  - sachdr_dict: dict of `sachdr`
###
class sachdr_list(list):
    """
    A list object for storing sac header.
    #
    This is for processing sac header, and associated, informations without
    touching time series, like selecting, sorting, pairing, etc.
    """
    def __init__(self):
        """
        Empty constructor
        """
        pass
    def pickle_save(self, filename):
        """
        Use pickle package to save this archive.
        filename:
        """
        with open(filename, 'wb') as fid:
            pickle.dump(self, fid, pickle.HIGHEST_PROTOCOL )
    def pickle_load(self, filename, extend=False):
        """
        Load archive from file saved by pickle.
        filename:
        """
        if not extend:
            self.clear()
        with open(filename, 'rb') as fid:
            self.extend(pickle.load(fid) )
    def deepcopy(self):
        """
        Deepcopy and return a new object.
        """
        return deepcopy(self)
    ### IO
    def read_fnm_lst(self, fnm_lst):
        """
        Read headers from a list of sac filenames for initialization.
        """
        self.extend( [rd_sachdr(fnm) for fnm in fnm_lst] )
    def write_table(self, filename, key_lst, sep='\t'):
        """
        Write table into file given keys.
        filename: string of filename for output;
        key_lst: a list contain keys for the table. (eg. ['filename', 'stlo', 'stla', 'stnm'] )
        sep: column separator, default is tab '\t';
        """
        with open(filename, 'w') as fid:
            print('\n'.join( [sep.join(['{:}'.format(it[key]) for key in key_lst]) for it in self ] ) , file=fid )
    ### update and self-revision methods
    def select(self, key, range):
        """
        Select and update inplacely given range for specific key.
        key: key being used for selection;
        range: a range of [low, high] for selection.
        #
        eg. select('mag', (6.0, 7.0) ) for selecting data with magnitude in [6.0, 7.0).
        """
        tmp = [it for it in self if it[key] >= range[0] and it[key] < range[1] ]
        self.clear()
        self.extend(tmp)
    def select_return(self, key, range):
        """
        Select and return a new sachdr_list object
        given range for specific key.
        key: key being used for selection;
        range: a range of [low, high] for selection.
        #
        eg. select_return('mag', (6.0, 7.0) ) for selecting data with magnitude in [6.0, 7.0).
        """
        tmp = sachdr_list()
        tmp.extend( [it for it in self if range[0] <= it[key] < range[1] ] )
        return tmp
    def sort(self, key, sequence= 'ascend'):
        """
        Sort with respect to specified `key`.
        key: reference for sorting; (eg. 'mag' );
        sequence: 'descend' or 'ascend';
        """
        tmp = {'descend': -1, 'ascend': 1}
        lst = [self[idx] for idx in np.argsort([it[key] for it in self ] )[::tmp[sequence] ] ]
        self.clear()
        self.extend(lst)
    def rm_duplication(self):
        """
        Update by removing duplicated items for same `filename`.
        Note this function will destroy the sequence.
        """
        tmp = self.to_sachdr_dict(False).to_sachdr_list(False)
        self.clear()
        self.extend(tmp)
    def set_id(self, start_value= 0):
        """
        Set a increasing key `key` to `sachdr`, started from `start_value`.
        """
        if 'id' in self[0]:
            print('Err: some id already set, in start_value.set_id(...)' )
            sys.exit(0)
        id = start_value
        for it in self:
            it['id'] = id
            id = id + 1
    ###
    def to_sachdr_dict(self, deepcopy=True):
        """
        Generate and return an object of `sachdr_dict`, and duplicated filenames will be removed.
        deepcopy: True (default) or False for deepcopy
        """
        vol = sachdr_dict()
        if deepcopy:
            for it in self:
                fnm = it['filename']
                if fnm not in vol:
                    vol[fnm] = deepcopy(it)
        else:
            for it in self:
                fnm = it['filename']
                if fnm not in vol:
                    vol[fnm] = it
        return vol
    def to_sachdr_pair_ev_list(self):
        """
        Generate and return an object of `sachdr_pair_ev_list`,
        which contain pairs of sachdr for all possible sachdr.
        """
        tmp = sachdr_pair_ev_list()
        for it1 in self:
            tmp.extend([sachdr_pair_ev(it1, it2) for it2 in self] )
        return tmp
    def get_key_lst(self, key):
        """
        Get and return a list of sac header key values.
        key: specified key; (eg. 'mag')
        """
        return [it[key] for it in self ]
    def get_key_range(self, key):
        """
        Get and return (min, max) key value for all sac headers.
        key: specified key; (eg. 'mag')
        """
        tmp = self.get_key_lst(key)
        return  np.min(tmp), np.max(tmp)
    ###
class sachdr_dict(dict):
    """
    A dict object for storing sac headers. Key used here is filename.
    #
    This is for processing sac header, and associated, informations without
    touching time series, like selecting, sorting, pairing, etc.
    """
    def __init__(self):
        """
        Empty constructor
        """
        pass
    def deepcopy(self):
        """
        Deepcopy and return a new object.
        """
        return deepcopy(self)
    ### IO
    def read_fnm_lst(self, fnm_lst):
        """
        Read headers from a list of sac filenames for initialization.
        """
        for fnm in fnm_lst:
            if fnm not in self:
                self[fnm] = rd_sachdr(fnm)
    def write_table(self, filename, key_lst, sep='\t'):
        """
        Write table into file given keys. Note that line sequence is random.
        filename: string of filename for output;
        key_lst: a list contain keys for the table. (eg. ['filename', 'stlo', 'stla', 'stnm'] )
        sep: column separator, default is tab '\t';
        """
        with open(filename, 'w') as fid:
            print('\n'.join( [sep.join(['{:}'.format(it[key]) for key in key_lst]) for it in self.values() ] ) , file=fid )
    ### update and self-revision methods
    ###
    def to_sachdr_list(self, deepcopy=True):
        """
        Generate and return an object of `sachdr_list`, and its sequence is random.
        deepcopy: True (default) or False for deepcopy
        """
        tmp = sachdr_list()
        tmp.extend(list( self.values() ) )
        return deepcopy(tmp) if deepcopy else tmp
    ###

###
#  high-level `sachdr_list` container
#  - sachdr_ev_dict: dict with respect to event
#                    evkey -> `sachdr_list`
#              HOW ABOUT A NEW VARIABLE NAME ???
###
class sachdr_ev_dict(dict):
    """
    A dict object for storing sac headers grouped with respect
    to different events.
    Key used here are user- defined, eg. strings, and corresponding values
    are list of `sachdr` object for that event.
    #
    """
    def __init__(self):
        """
        Empty constructor
        """
        pass
    ### IO
    def init_dir_template(self, fnm_template, verbose= False):
        """
        Read all sac file headers based on `fnm_template` to build this archive.
        fnm_template: filenames template for header reading and grouping;
        verbose: (default is False)
        eg: fnm_template = '/dat/2010*/*BHZ.SAC'
            for which, all sac files are grouped with respect to event
            that each `/dat/2010*/` directory contains sac files having
            same event information.
            Then, all sac files match this template will be read, and each 
            directory path `/dat/2010*/` will be used as key, and files
            within will be used as corresponding dat.
            And data structure will be:

            <this archive> + ['/dat/2010_01'] ->    (sachdr of '/dat/2010_01/a.BHZ.SAC' ) 
                           |                        (sachdr of '/dat/2010_01/b.BHZ.SAC' )
                           |                         ...
                           |                        (sachdr of '/dat/2010_01/?.BHZ.SAC' )
                           |                        # list if sachdr having same event info
                           |
                           + ['/dat/2010_02'] ->    (sachdr of '/dat/2010_02/a.BHZ.SAC' )
                           |                        (sachdr of '/dat/2010_02/b.BHZ.SAC' )
                           |                         ...
                           ...


        """
        root_dir = '/'.join( fnm_template.split('/')[:-1] )
        final_template = '{:}/' + fnm_template.split('/')[-1]
        for sub_dir in sorted(glob.glob(root_dir) ):
            lst = sachdr_list()
            fnm_str =  final_template.format(sub_dir)
            lst.read_fnm_lst( sorted(glob.glob(fnm_str ) ) )
            if verbose:
                print('{:} : ({:d})'.format(fnm_str, len(lst) ) )
            self.add(sub_dir, lst)
    def pickle_save(self, filename):
        """
        Use pickle package to save this archive.
        filename:
        """
        with open(filename, 'wb') as fid:
            pickle.dump(self, fid, pickle.HIGHEST_PROTOCOL )
    def pickle_load(self, filename):
        """
        Load archive from file saved by pickle.
        filename:
        """
        with open(filename, 'rb') as fid:
            self.update(pickle.load(fid) )
    ### Inserting, and accessing, and updating
    def add(self, evkey, hdr_lst):
        """
        Add a list of `sachdr` with same event information.
        #
        evkey: user- defined key, can be anything, eg. string.
        hdr_lst: a list of `sachdr` objects;
        """
        if evkey not in self:
            self[evkey] = sachdr_list()
        self[evkey].extend(hdr_lst)
    def set_id(self, start_id= 0):
        """
        Set a increasing key `key` to all `sachdr`, started from `start_value`.
        """
        id = start_id
        for v in self.values():
            v.set_id(id)
            id = v[-1]['id'] + 1 # faster, but not good for maintain
    def get_ev_info(self, evkey=None, sachdr_key_lst=['mag']):
        """
        Get event information specified by `sachdr_key_lst` given `evkey`.
        evkey: `evkey` used to search for a single event. (None for all event)
        sachdr_key_lst: list of sac header used to acquire info. (eg. 'mag', 'evlo', ...)
        """
        if evkey:
             return [ self[evkey][0][sackey] for sackey in sachdr_key_lst]
        return [ [v[0][sackey] for sackey in sachdr_key_lst] for v in self.values() ]
    def get_volumn_size(self):
        """
        Return number of events, and number of all sac files.
        (nev, nsac)
        """
        nev = len(self)
        nsac = np.sum( [len(v) for v in self.values()] )
        return nev, nsac
    ###
    def select_4_ev_return(self, sachdr_event_key, range, deepcopy=False):
        """
        Select and return a new `sachdr_ev_dict` object given range 
        for specified `sachdr_event_key`;
        sachdr_event_key: sac header key used to select; (eg: 'mag')
        range: [x1, x2) the range for selecting.
        deepcopy: whether use deepcopy in constructing new object (default is False);
        """
        tmp = sachdr_ev_dict()
        for key, value in self.items():
            v = value[0][sachdr_event_key] #self.get_ev_info(key, [sachdr_event_key])[0]
            #print(v)
            if range[0] <= v < range[1]:
                tmp[key] = deepcopy(value) if deepcopy else value
        return tmp
    def select_4_st_return(self, sachdr_st_key, range, deepcopy=False):
        """
        Select and return a new `sachdr_ev_dict` object given range 
        for specified `sachdr_st_key`;
        sachdr_st_key: sac header key used to select; (eg: 'gcarc' )
        range: [x1, x2) the range for selecting.
        deepcopy: whether use deepcopy in constructing new object (default is False);
        """
        tmp = sachdr_ev_dict()
        for key, value in self.items():
            v = value.select_return(sachdr_st_key, range)
            tmp[key] = deepcopy(v) if deepcopy else v
        return tmp
    ### 
    def to_sachdr_pair_ev_list(self):
        """
        Generate and return an object of sachdr_pair_ev_list, which 
        contains all pair of sachdr with same event information.
        """
        tmp = sachdr_pair_ev_list()
        for v in self.values():
            tmp.extend(v.to_sachdr_pair_ev_list() )
        return tmp
    ###
    #def __mk_key(self, evlo, evla, evdp_km, mag, otime):
    #    """
    #    Make key given a set of event information.
    #    #
    #    evlo, evla, evdp_km, mag, otime: event information;
    #    """
    #    return (evlo, evla, evdp_km, mag, otime)
    #    #{'evlo': evlo, 'evla': evla, 'evdp': evdp_km, 'mag': mag, 'otime': otime}

###
#  sachdr_pair
#  - sachdr_pair:    base class
#  - sachdr_pair_ev: class for pair having same event
###
class sachdr_pair(dict):
    """
    A pair of two sac header. Note that the order of two sac header is important.
    
    Two headers can be access by:
    eg['hdr1'], eg['hdr2'].
    And, more keys can be added into this object.
    """
    def __init__(self, h1, h2):
        """
        Construtor.
        h1, h2: `sachdr` object;
        """
        self.set_hdr(h1, h2)
    def set_hdr(self, h1, h2):
        """
        Set two sac headers.
        h1, h2: `sachdr` object;
        """
        #self.h1 = h1
        #self.h2 = h2
        self['hdr1'] = h1
        self['hdr2'] = h2
    def __eq__(self, other):
        """
        Check whether two `sachdr_pair_ev` object are equal based on filenames.
        """
        return self.h1['filename'] == other.h1['filename'] and self.h2['filename'] == other.h2['filename']
    ### accessing
    def get_sachdr(self, sachdr_key_lst):
        """
        Return header variables for two headers, given specified key names.
        sachdr_key_lst: list of sac header variable names; (eg. ['kstnm', 'id', ...] )
        """
        return [self['hdr1'][key] for key in sachdr_key_lst], [self['hdr2'][key] for key in sachdr_key_lst ]
class sachdr_pair_ev(sachdr_pair):
    """
    A pair of two sac header with same event. Note that the order of two sac header is important.

    Critical key are:
        ['hdr1']      : 1st sachdr object
        ['hdr1']      : 2nd sachdr object
        ['st_dist']   : inter-station distance in degree;
        ['daz']       : azimuth difference;
        ['dbaz']      : back- azimuth difference;
        ['gc2ev']     : distance from event to great circle formed by two stations;
        ['gc2st1']    : distance from the 1st station to great circle formed by event and the 2nd station;
        ['gc2st2']    : distance from the 2nd station to great circle formed by event and the 1st station;
    """
    def __init__(self, h1, h2):
        """
        Construtor.
        h1, h2: `sachdr` object;
        """
        sachdr_pair.__init__(self, h1, h2)
        #print(self.h1, self.h2)
        evlo, evla  = h1['evlo'], h1['evla']
        stlo1, stla1 = h1['stlo'], h1['stla']
        stlo2, stla2 = h2['stlo'], h2['stla']
        h1['az'] = azimuth(evlo, evla, stlo1, stla1)
        h2['az'] = azimuth(evlo, evla, stlo2, stla2)
        h1['baz'] = azimuth(stlo1, stla1, evlo, evla)
        h2['baz'] = azimuth(stlo1, stla1, evlo, evla)
        h1['gcarc'] = haversine(evlo, evla, stlo1, stla1)
        h2['gcarc'] = haversine(evlo, evla, stlo2, stla2)
        self['ddist'  ] = np.abs( h1['gcarc'] - h2['gcarc']  )
        self['st_dist'] = haversine(stlo1, stla1, stlo2, stla2 )
        self['daz'       ] = np.abs(h1['az']  - h2['az']  )
        self['dbaz'      ] = np.abs(h1['baz'] - h2['baz'] )
        self['gc2ev'     ] = np.abs( point_distance_to_great_circle_plane(evlo, evla, stlo1, stla1, stlo2, stla2) )
        self['gc2st1'    ] = np.abs( point_distance_to_great_circle_plane(stlo1, stla1, evlo, evla, stlo2, stla2) )
        self['gc2st2'    ] = np.abs( point_distance_to_great_circle_plane(stlo2, stla2, evlo, evla, stlo1, stla1) )
    def __str__(self):
        s = sachdr_pair.__str__(self)
        #s += '\n'
        #s += '\n'.join( ['{:<9s}: {:f}'.format(key, self[key]) for key in ['st_dist', 'daz', 'dbaz', 'gc2ev', 'gc2st1', 'gc2st2'] ] )
        return s

###
#  sachdr_pair container
#  - sachdr_pair_ev_list: list of `sachdr_pair_ev`
###
class sachdr_pair_ev_list(list):
    """
    """
    def __init__(self):
        """
        A list object for storing `sachdr_pair_ev`.
        """
        pass
    ### IO
    def write_table(self, filename, sachdr_key_lst, pair_key_lst, sep='\t'):
        """
        Write table into file given keys.
        filename: string of filename for output;
        sachdr_key_lst: a list containing sachdr keys for the table. 
            (eg. ['filename', 'stlo', ...], or other user defined keys )
        pair_key_lst:   a list containing sachdr_pair_ev keys for the table.
            (eg. ['st_dist', 'daz', ...], or other user defined keys )
        sep: column separator, default is tab '\t';
        """
        with open(filename, 'w') as fid:
            s = []
            for it in self:
                tmp = []
                #print(it['hdr1'])
                tmp.extend([ '{:}'.format(it['hdr1'][key]) for key in sachdr_key_lst] )
                tmp.extend([ '{:}'.format(it['hdr2'][key]) for key in sachdr_key_lst] )
                tmp.extend([ '{:}'.format(it[key]        ) for key in pair_key_lst  ] )
                s.append(sep.join(tmp) )
                #s = '\n'.join( [sep.join([it.hd1[key] for key in sachdr_key_lst] ) + sep +
                #                sep.join([it.hd2[key] for key in sachdr_key_lst] ) + sep +
                #                sep.join([it[key] for key in pair_key_lst] ) 
                #                for it in self] )
            print('\n'.join(s), file= fid)
    def select(self, key, range):
        """
        Select and update given range for specific key.
        key: key being used for selection, can be 'st_dist', 'daz', 'dbaz', 'gc2ev', 'gc2st1', 
             'gc2st2', as defined in `sachdr_pair_ev`, or other user defined key;
        range: a range of [low, high] for selection.
        #
        eg. select('daz', (-5.0, 5.0) ) for selecting data with daz in (-5.0, 5.0).
        """
        tmp = [it for it in self if it[key] >= range[0] and it[key] < range[1] ]
        self.clear()
        self.extend(tmp)
    def select_4_return(self, key, range):
        """
        Select and return a new `sachdr_pair_ev_list` object given range for specific key.
        key: key being used for selection, can be 'st_dist', 'daz', 'dbaz', 'gc2ev', 'gc2st1', 
             'gc2st2', as defined in `sachdr_pair_ev`, or other user defined key;
        range: a range of [low, high] for selection.
        #
        eg. select('daz', (0.0, 5.0) ) for selecting data with daz in (0.0, 5.0).
        """
        tmp = sachdr_pair_ev_list()
        tmp.extend( [it for it in self if it[key] >= range[0] and it[key] < range[1] ] )
        return tmp
    def pickle_save(self, filename):
        """

        """
        """
        Use pickle package to save this archive.
        filename:
        """
        with open(filename, 'wb') as fid:
            pickle.dump(self, fid, pickle.HIGHEST_PROTOCOL )
    def pickle_load(self, filename):
        """
        Load archive from file saved by pickle.
        filename:
        """
        with open(filename, 'rb') as fid:
            self.clear()
            self.extend(pickle.load(fid) )
    ### accesing
    def get_sachdr_lst(self, sachdr_key_lst):
        """
        Return list of two sac header variables, given specified key names.
        sachdr_key_lst: list of sac header variable names; (eg. ['kstnm', 'id', ...] )
        """
        return [it.get_sachdr(sachdr_key_lst) for it in self ]
    def to_sachdr_dict(self, deepcopy=True ):
        """
        Generate and return an object of `sachdr_dict` for all sac files used.
        """
        tmp = sachdr_dict()
        if deepcopy:
            for it_pair in self:
                for it in [it_pair['hdr1'], it_pair['hdr2'] ]:
                    fnm = it['filename']
                    if fnm not in tmp:
                        tmp[fnm] = deepcopy(it)
        else:
            for it_pair in self:
                for it in [it_pair['hdr1'], it_pair['hdr2'] ]:
                    fnm = it['filename']
                    if fnm not in tmp:
                        tmp[fnm] = it
        return tmp
    def to_sachdr_list(self, deepcopy=True):
        """
        Generate and return an object of `sachdr_list` for all sac files used with duplication removed.
        """
        return self.to_sachdr_dict(deepcopy=False).to_sachdr_list(deepcopy=deepcopy)
    def write_id_pair(self, fid, stack_id = None):
        """
        Write two columns of ids into file opened as `fid`.
        fid: file object
        stack_id: used for third column. 
                  If set to None(default), third column will be empty.
        """
        if stack_id:
            print( '\n'.join(['{:d} {:d} {:d}'.format(it['hdr1']['id'], it['hdr2']['id'], stack_id) for it in self] ), file= fid )
        else:
            print( '\n'.join(['{:d} {:d}'.format(it['hdr1']['id'], it['hdr2']['id']) for it in self] ), file= fid )
###
#  high-level `sachdr_pair_ev_list` container
###
#class

##########################
##########################
##########################
if __name__ == "__main__":
    #sac2hdf5('/home/catfly/00-LARGE-IMPORTANT-PERMANENT-DATA/AU_dowload/01_resampled_bhz_to_h5/01_workspace_bhz_sac/2000_008_16_47_20_+0000/*SAC',
    #            'junk.h5')
    hdf52sac('junk.h5', 'junk/sac_', True)