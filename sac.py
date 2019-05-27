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
import scipy.signal as signal
import matplotlib.pyplot as plt
import copy
import numpy as np
import scipy.signal as signal
import struct
import sys
import sacpy.geomath as geomath
###
#  dependend methods
###
def rd_sac(filename):
    """
    Read sac given `filename`, and return an object ot sactrace.
    """
    tmp = sactrace()
    tmp.read(filename)
    return tmp
def rd_sac_2(filename, tmark, t1, t2):
    """
    Read sac data given filename and time window, and return an object ot sactrace.
    tmakr: 'b', 'e', 'o', 'a', 't0', 't1', ... 't9';
    t1, t2: float
    """
    tmp = sactrace()
    tmp.read_2(filename, tmark, t1, t2)
    return tmp
def rd_sachdr(filename):
    """
    Read sac header given `filename`, and return an object ot sachdr.
    """
    tmp = sachdr()
    tmp.read(filename, 'filename')
    return tmp
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
                'xmaximum', 'yminimum', 'ymaximum', 'unused1', 'unused2',
                'unused3', 'unused4', 'unused5', 'unused6', 'unused7' ]
    i_keys = [  'nzyear', 'nzjday', 'nzhour', 'nzmin', 'nzsec',
                'nzmsec', 'nvhdr',  'norid',  'nevid', 'npts', 
                'internal4', 'nwfid','nxsize', 'nysize', 'unused8',
                'iftype', 'idep', 'iztype', 'unused9', 'iinst',
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
    def read(self, f, type='fid'):
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
        info = struct.unpack('70f40i192s', hdrvol)
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
        pass
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
        self.d_arch['e'] = (npts-1) * self.d_arch['delta'] * self.d_arch['delta']
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
    def read(self, filename):
        """
        Read sac data given filename.
        """
        with open(filename, 'rb') as fid:
            self.hdr.read(fid)
            self.hdr['filename'] = filename
            self.dat = np.fromfile(fid, dtype=np.float32, count=self.hdr['npts'])
    def write(self, filename):
        """
        Write data into specified file.
        """
        self.hdr.__update_npts__(np.size(self.dat) ) # update npts in case the dat length is revised.
        with open(filename, 'wb') as fid:
            fid.write(self.hdr.pack() )
            self.dat.astype(np.float32).tofile(fid)
    def read_2(self, filename, tmark, t1, t2):
        """
        Read sac data given filename and time window.
        tmark: 'b', 'e', 'o', 'a', 't0', 't1', ... 't9';
        t1, t2: float;
        """
        self.read(filename)
        self.truncate(tmark, t1, t2)
    ### hdr methods
    def update_hdr(self, **kwargs):
        """
        Update sac header.
        kwargs: dict of (key: value), eg: {'delta': 0.1, 'kstnm': 'HYD}
        """
        self.hdr.update(**kwargs)
    ### internel methods
    def __get_t_idx__(self, tmark, t):
        """
        Given t, get the closest index.
        """
        if tmark in self:
            t_ref = self[tmark]
            if t_ref == -12345.0:
                print('unset header for tmark: %s %f', tmark, t_ref)
                sys.exit(0)
            idx = int(np.round( (t_ref+t-self['b'])/self['delta'] ) )
            idx = min(max(idx, 0), self['npts']-1 )
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
    def get_time_axis(self):
        """
        Get time axis.
        """
        return np.arange(0.0, self['npts'], 1.0) * self['delta'] + self['b']
    def truncate(self, tmark, t1, t2):
        """
        Truncate given reference tmark, and time window.
        tmark: 'b', 'e', 'o', 'a', 't0', 't1', ... 't9';
        t1, t2: float;
        """
        i1 = self.__get_t_idx__(tmark, t1)
        i2 = self.__get_t_idx__(tmark, t2) + 1
        # update data and header info
        self.dat= self.dat[i1:i2]
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
        nyq = 0.5 / self['delta']
        low = f1 / nyq
        high = f2 / nyq
        b, a = signal.butter(order, [low, high], btype='bandpass')
        methods = [None, signal.lfilter, signal.filtfilt]
        self.dat = methods[npass](b, a, self.dat)
        #return self
    def lowpass(self, f, order= 2, npass= 1):
        """
        Lowpass
        """
        nyq = 0.5 / self['delta']
        fc = f/nyq
        b, a = signal.butter(order, [fc], btype='lowpass')
        methods = [None, signal.lfilter, signal.filtfilt]
        self.dat = methods[npass](b, a, self.dat)
        #return self
    def highpass(self, f, order= 2, npass= 1):
        """
        High pass
        """
        nyq = 0.5 / self['delta']
        fc = f/nyq
        b, a = signal.butter(order, [fc], btype='highpass')
        methods = [None, signal.lfilter, signal.filtfilt]
        self.dat = methods[npass](b, a, self.dat)
        #return self
    ### plot
    def plot_ax(self, ax, **kwargs):
        """
        Plot into specified axis, with **kwargs used by pyplot.plot(...).
        """
        ax.plot(self.get_time_axis(), self.dat, **kwargs)
    def plot(self, **kwargs):
        """
        Plot and show, with **kwargs used by pyplot.plot(...).
        """
        plt.plot(self.get_time_axis(), self.dat, **kwargs)
        plt.show()
        #plt.close()
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
        Select and update given range for specific key.
        key: key being used for selection;
        range: a range of [low, high] for selection.
        #
        eg. select('mag', (6.0, 7.0) ) for selecting data with magnitude in [6.0, 7.0).
        """
        tmp = [it for it in self if it[key] >= range[0] and it[key] < range[1] ]
        self.clear()
        self.extend(tmp)
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
        if deepcopy:
            return copy.deepcopy(list( self.values() ) )
        else:
            return list( self.values() ) 
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
            id = v.get_key_range('id')[1] + 1
    ###
    ### 
    def to_sachdr_pair_ev_list(self):
        """
        Generate and return an object of sachdr_pair_ev_list, which 
        contains all pair of sachdr with same event information.
        """
        tmp = sachdr_pair_ev_list()
        for v in self.values():
            for it1 in v:
                tmp.extend( [sachdr_pair_ev(it1, it2) for it2 in v] )
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
        self['st_dist'] = geomath.haversine(stlo1, stla1, stlo2, stla2 )
        self['daz'       ] = h1['az'] - h2['az']
        self['dbaz'      ] = h1['baz'] - h2['baz']
        self['gc2ev'     ] = geomath.point_distance_to_great_circle_plane(evlo, evla, stlo1, stla1, stlo2, stla2)
        self['gc2st1'    ] = geomath.point_distance_to_great_circle_plane(stlo1, stla1, evlo, evla, stlo2, stla2)
        self['gc2st2'    ] = geomath.point_distance_to_great_circle_plane(stlo2, stla2, evlo, evla, stlo1, stla1)
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
        eg. select('daz', (-5.0, 5.0) ) for selecting data with magnitude in [6.0, 7.0).
        """
        tmp = [it for it in self if it[key] >= range[0] and it[key] < range[1] ]
        self.clear()
        self.extend(tmp)
    ###
    def to_sachdr_dict(self):
        """
        Generate and return an object of `sachdr_dict` for all sac files used.
        """
        tmp = sachdr_dict()
        for it_pair in self:
            for it in [it_pair['hdr1'], it_pair['hdr2'] ]:
                fnm = it['filename']
                if fnm not in tmp:
                    tmp[fnm] = it
        return tmp

###
#  high-level `sachdr_pair_ev_list` container
###
#class 

##########################
##########################
##########################
if __name__ == "__main__":
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
