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
>>> new_hdr = copy.deepcopy(s.hdr)
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
import copy
import numpy as np
import scipy.signal as signal
import struct
import sys
import glob
import pickle
import h5py
import sacpy.geomath as geomath
import sacpy.processing as processing
import copy
from os.path import exists as os_path_exists
###
#  dependend methods
###
def rd_sac(filename, lcalda=False):
    """
    Read sac given `filename`, and return an object ot sactrace.
    """
    tmp = sactrace()
    tmp.read(filename, lcalda=lcalda)
    return tmp
def rd_sac_2(filename, tmark, t1, t2, lcalda=False):
    """
    Read sac data given filename and time window, and return an object ot sactrace.
    tmakr: 'b', 'e', 'o', 'a', 't0', 't1', ... 't9';
    t1, t2: float
    """
    tmp = sactrace()
    tmp.read_2(filename, tmark, t1, t2, lcalda=lcalda)
    return tmp
def rd_sachdr(filename, lcalda=False):
    """
    Read sac header given `filename`, and return an object ot sachdr.
    """
    tmp = sachdr()
    tmp.read(filename, 'filename', lcalda=lcalda)
    return tmp
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
def make_sachdr(delta, npts, b, **kwargs):
    """
    Generate sac header, and return an object of `sachdr`.
    """
    tmp = sachdr()
    tmp.init(delta, npts, b, **kwargs)
    return tmp
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
def wrt_sac(filename, dat, hdr):
    """
    Write sac file given dat (an object of numpy.ndarray) and hdr (an object of sachdr)
    """
    tmp = sactrace()
    tmp.init(dat, hdr, False)
    tmp.write(filename)
def wrt_sac_2(filename, dat, delta, b, **kwargs):
    """
    Write sac file given dat (an object of numpy.ndarray) and header settings.
    """
    tmp = sactrace()
    tmp.init(dat, make_sachdr(delta, np.size(dat), b, **kwargs), False)
    tmp.write(filename)
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
    tmp = copy.deepcopy(sac_trace)
    tmp.truncate(tmark, t1, t2)
    if clean_sachdr:
        return make_sactrace_v(tmp['dat'], tmp['delta'], tmp['b'])
    return tmp
def correlation_sac(sac_trace1, sac_trace2):
    """
    Compute the cross-correlation between two SAC_TRACE objects, and
    return a new SAC_TRACE object.
    The definition of cross-correlation is:
    
        cc(t) = \int st1(t+\tau) st2(\tau) d\tau
    
    Please note cc{st1, st2} = reversed cc{st2, st1}

    """
    cc = signal.correlate(sac_trace1['dat'], sac_trace2['dat'], 'full', 'fft')
    cc_start = sac_trace1['b'] - sac_trace2['e']
    return make_sactrace_v(cc, sac_trace1['delta'], cc_start)
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
def time_shift_all_sac(sac_trace, t_shift_sec):
    st = copy.deepcopy(sac_trace)
    st.shift_time_all(t_shift_sec)
    return st
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
def plot_sac_lst(st_lst, ax=None):
    fig = None
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    for isac, it in enumerate(st_lst):
        junk = copy.deepcopy(it)
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
            self.d_arch['filename'] = copy.deepcopy(f)
            f = open(f, 'rb')
        hdrvol = f.read(632)
        #print(sachdr.little_endian_format)
        info = struct.unpack(sachdr.little_endian_format, hdrvol)
        info, small_endian_tag = (info, True) if 1< info[76] < 7 else (struct.unpack(sachdr.big_endian_format, hdrvol), False)
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
            self.d_arch['gcarc'] = geomath.haversine(stlo, stla, evlo, evla)
            self.d_arch['baz'] = geomath.azimuth(stlo, stla, evlo, evla)
            self.d_arch['az'] = geomath.azimuth(evlo, evla, stlo, stla)
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
        return struct.pack('70f40i192s', *lst)
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
        self.hdr = sachdr()
        self.dat = None
    ### init
    def init(self, dat, hdr, deepcopy=True):
        if deepcopy:
            self.hdr = copy.deepcopy(hdr)
            self.dat = copy.deepcopy(dat)
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
            self['gcarc']= geomath.haversine(self['evlo'], self['evla'], self['stlo'], self['stla'])  
            self['az']   = geomath.azimuth(  self['evlo'], self['evla'], self['stlo'], self['stla']) 
            self['baz']  = geomath.azimuth(  self['stlo'], self['stla'], self['evlo'], self['evla']) 
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
        w = signal.tukey(self['npts'], ratio)
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
        self.dat = signal.detrend(self.dat)
        #return self
    def bandpass(self, f1, f2, order= 2, npass= 1):
        """
        Bandpass
        """
        self.dat = processing.filter(self.dat, 1.0/self['delta'], 'bandpass', [f1, f2], order, npass )
    def lowpass(self, f, order= 2, npass= 1):
        """
        Lowpass
        """
        self.dat = processing.filter(self.dat, 1.0/self['delta'], 'lowpass', [f], order, npass )
    def highpass(self, f, order= 2, npass= 1):
        """
        High pass
        """
        self.dat = processing.filter(self.dat, 1.0/self['delta'], 'highpass', [f], order, npass )
    def resample(self, delta):
        """
        Resample the time-series using Fourier method.
        """
        if (delta % self['delta'] == 0.0):
            factor = int(delta // self['delta'])
            return self.decimate(factor)
        new_npts = int( round(delta/self['delta']* self['npts']) )
        self['dat'] = signal.resample(self['dat'], new_npts)
        self['npts'] = new_npts
        self['delta'] = delta
        self['e'] = self['b'] + (new_npts-1)*delta
    def decimate(self, factor):
        """
        Downsample the time-series using scipy.signal.decimate.
        """
        self['dat'] = signal.decimate(self['dat'], factor)
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
        return copy.deepcopy(self)
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
                    vol[fnm] = copy.deepcopy(it)
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
        return copy.deepcopy(self)
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
        return copy.deepcopy(tmp) if deepcopy else tmp
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
                tmp[key] = copy.deepcopy(value) if deepcopy else value
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
            tmp[key] = copy.deepcopy(v) if deepcopy else v
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
        h1['az'] = geomath.azimuth(evlo, evla, stlo1, stla1)
        h2['az'] = geomath.azimuth(evlo, evla, stlo2, stla2)
        h1['baz'] = geomath.azimuth(stlo1, stla1, evlo, evla)
        h2['baz'] = geomath.azimuth(stlo1, stla1, evlo, evla)
        h1['gcarc'] = geomath.haversine(evlo, evla, stlo1, stla1)
        h2['gcarc'] = geomath.haversine(evlo, evla, stlo2, stla2)
        self['ddist'  ] = np.abs( h1['gcarc'] - h2['gcarc']  )
        self['st_dist'] = geomath.haversine(stlo1, stla1, stlo2, stla2 )
        self['daz'       ] = np.abs(h1['az']  - h2['az']  )
        self['dbaz'      ] = np.abs(h1['baz'] - h2['baz'] )
        self['gc2ev'     ] = np.abs( geomath.point_distance_to_great_circle_plane(evlo, evla, stlo1, stla1, stlo2, stla2) )
        self['gc2st1'    ] = np.abs( geomath.point_distance_to_great_circle_plane(stlo1, stla1, evlo, evla, stlo2, stla2) )
        self['gc2st2'    ] = np.abs( geomath.point_distance_to_great_circle_plane(stlo2, stla2, evlo, evla, stlo1, stla1) )
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
                        tmp[fnm] = copy.deepcopy(it)
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
    #print(moving_average(xs, 5) )
    tr1 = rd_sac('test_tmp/test.sac')
    print(tr1)
    sys.exit(0)
    
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(211)
    tr1.plot_ax(ax2)
    #tr2.plot_ax(ax1)
    #tr3.plot_ax(ax1)
    plt.show()
    sys.exit(0)

    ax = plt.subplot(111)

    tr = rd_sac('test_tmp/2.sac')
    print(tr['b'], tr['e'], tr['npts'], tr.dat.size )
    tr.plot_ax(ax, alpha= 0.6, color='k', linewidth= 2)

    tr = rd_sac_2('test_tmp/2.sac', '0', 200, 900)
    print(tr['b'], tr['e'], tr['npts'], tr.dat.size )
    tr.plot_ax(ax, alpha= 0.6, color='C0')

    tr = rd_sac_2('test_tmp/2.sac', '0', -300, 900)
    print(tr['b'], tr['e'], tr['npts'], tr.dat.size )
    tr.plot_ax(ax, alpha= 0.6, color='C1')

    tr = rd_sac_2('test_tmp/2.sac', '0', -500, 1600)
    print(tr['b'], tr['e'], tr['npts'], tr.dat.size )
    tr.plot_ax(ax, alpha= 0.6, color='r')



    idx, t, y = tr.max_amplitude_time('neg', '0', (100, 400) )
    print(t, y)
    ax.plot(t, y, 'rx')

    plt.show()
    sys.exit(0)

    import glob
    vol = sachdr_ev_dict()
    id = 1
    for evdir in glob.glob('/mnt/Tdata/.ccPhysics/10_real_data/04/whitened_bp_data_1_12/2010*'):
        evkey = evdir.split('/')[-1]
        fnmlst = glob.glob(evdir+'/*BHZ.SAC')
        tmp = sachdr_list()
        tmp.read_fnm_lst(fnmlst)
        vol.add(evkey, tmp )
    vol.set_id(1)
    print(vol.keys() ) 
    pair_vol = vol.to_sachdr_pair_ev_list()
    pair_vol.select('daz', ( -5, 5) )
    print(len(pair_vol) )
    pair_vol.write_table('test.txt', ['id'], ['daz'] )
    sub_vol = pair_vol.to_sachdr_dict()
    sub_vol.write_table('test.2.txt', ['filename', 'id'] )
    print(len(sub_vol) )
    sys.exit(0)
    ####################
    ###
    vol = sachdr_list()
    vol.read_fnm_lst(glob.glob('/mnt/Tdata/.ccPhysics/10_real_data/04/whitened_bp_data_1_12/2010*/*.BHZ.SAC') )
    ### select
    vol.select('mag', (6.9, 9999) )
    ###
    sys.exit(0)
    ###
    vol = sachdr_list()
    vol.read_fnm_lst(['test2.sac', 'test.sac', 'test.sac'])
    vol.write_table('junk.txt', ['b', 'e', 'npts', 'delta', 'kstnm', 'filename'])
    v2 = vol.deepcopy()
    v2.clear()
    print(len(vol), len(v2) )
    print(vol, v2)
    d = vol.to_sachdr_dict()
    print(d)
    print(vol)
    vol.rm_duplication()
    print(vol)
    ###
    p = sachdr_pair_ev(vol[0], vol[1])
    p2 =sachdr_pair(vol[0], vol[1])
    print(p)
    print(p2)
    sys.exit(0)
    sys.exit()
    wrt_sac_2('test.sac', np.array([1,2,3,4,5,6,7] ), 1.0, 0.0, **{'kstnm': 'aaaaaaaaaaaaaaaaaa', 'kevnm': 'bbbbbbbbbbbbbbbbbbbb'} )
    s = rd_sac('test.sac')
    print(s)
    s['dat'] = np.array([1,2] )
    print(s)
    s.write('test2.sac')
    s2 = rd_sac('test2.sac')
    print(s)
    #print([s['kt0'] ] )
    ##
    
    vol  = sactrace_list()
    vol.read_fnm_lst(['1.sac', '2.sac', '3.sac'])
    vol.sort('mag', 'descend')
    vol.plot2()
    #sys.exit(0)
    #for it in vol:
    #    print( it['nzjday'])
    #sys.exit(0)
    # io and basic processing
    s = rd_sac('1.sac')
    s.plot()       
    s.detrend()    
    s.taper(0.02) # ratio can be 0 ~ 0.5
    s.bandpass(0.5, 2.0, order= 4, npass= 2)
    s.write('1_new.sac')
    # cut and read
    s = rd_sac_2('1.sac', 'e', -50, -20)
    s.detrend()    
    # some other processing...
    s.write('1_truncated.sac')
    ###
    #
    ###

    dat = np.random.random(1000)
    delta, b = 0.05, 50.0
    # writing method 1
    wrt_sac_2('junk.sac', dat, delta, b, **{'kstnm': 'syn', 'stlo': 0.0, 'stla': 0.0} )
    #
    # writing method 2, with processings
    s = make_sactrace_v(dat, delta, b, **{'kstnm': 'syn', 'stlo': 0.0, 'stla': 0.0} )
    s.taper()
    s.bandpass(0.5, 2.0, order= 4, npass= 2)
    s.write('junk2.sac')
    ###
    #
    ###
    import copy
    s = rd_sac('1.sac')
    new_hdr = copy.deepcopy(s.hdr)
    new_hdr['t1'] = 2.0
    new_hdr['t2'] = 4.0
    new_hdr.update( **{'stlo': -10.0, 'stla': 10.0, 'delta': 0.2 } )
    s_new = make_sactrace_hdr(s['dat'], new_hdr)
    s_new.write('1_new.sac')
    print(s_new)
    ###
    #
    ###
    # read, process, and write a bunch of sac files in 3 lines
    fnm_lst = ['1.sac', '2.sac', '3.sac']
    for fnm in fnm_lst:
        s = rd_sac(fnm)
        s.detrend()
        s.taper()
        s.lowpass(1.2, order= 2, npass= 2)
        s.write(s['filename'].replace('.sac', '_proced.sac') )
    #[ s.write(s['filename'].replace('.sac', '_proced.sac') ) for s in vol ]
    #import matplotlib.pyplot as plt
    #tr1 = np.array([0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0])
    #tr2 = np.array([0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0])
    #i = idx_shift_res(tr1, tr2, -4, 4)
    #print(i)
    #sactrace.benchmark()
    #s1 = rd_sac('1.sac')
    #s2 = rd_sac_2('1.sac', 'e', -5000, 0)
    #s3 = copy.deepcopy(s1)
    #s3.bandpass(0.02, 0.0666, 2, 2)
    #s2.write('2.sac')
    #ax = plt.subplot(111)
    #s1.plot_ax(ax, label='raw')
    #s3.plot_ax(ax, label='bp')
    #plt.legend()
    #plt.show()
