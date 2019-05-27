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
>>> write_sac_2('junk.sac', dat, delta, b, **{'kstnm': 'syn', 'stlo': 0.0, 'stla': 0.0} )
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
>>> s_new = make_sactrace_hdr(s.dat, new_hdr)
>>> s_new.write('1_new.sac')


Massive data IO and processing
------------------------------

>>> # read, process, and write a bunch of sac files in 3 lines
>>> fnm_lst = ['1.sac', '2.sac', '3.sac']
>>> vol = [rd_sac(fnm).detrend().taper().lowpass(1.2, order= 2, npass= 2) for fnm in fnm_lst]
>>> [ s.write(s['filename'].replace('.sac', '_proced.sac') ) for s in vol ]



"""
import scipy.signal as signal
import matplotlib.pyplot as plt
import copy
import numpy as np
import scipy.signal as signal
import struct
import sys

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
    tmakr: 'b', 'e', 'o', 'a', 't0', 't1', ... 't9'.
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
def write_sac(filename, dat, hdr):
    """
    Write sac file given dat (an object of numpy.ndarray) and hdr (an object of sachdr)
    """
    tmp = sactrace()
    tmp.init(dat, hdr, False)
    tmp.write(filename)
def write_sac_2(filename, dat, delta, b, **kwargs):
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
        lst.append( ''.join([self.d_arch[k] for k in self.s_keys ]).encode('unicode_escape') )
        return struct.pack('70f40i192s', *lst)
    ###
    def update(self, **kwargs):
        """
        Update sac hdr information.
        """
        for k, v in kwargs.items():
            self.d_arch[k] = v
    ###
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
        s_lst.extend( ['internal %-9s: %f' % (k, self.d_arch[k]) for k in sachdr.f_keys] )
        s_lst.extend( ['internal %-9s: %d' % (k, self.d_arch[k]) for k in sachdr.i_keys] )
        s_lst.extend( ['internal %-9s: %s' % (k, self.d_arch[k]) for k in sachdr.s_keys] )
        s_lst.extend( ['externel %-9s: %s' % (k, self.d_arch[k]) for k in sub_keys] )
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
        with open(filename, 'wb') as fid:
            fid.write(self.hdr.pack() )
            self.dat.astype(np.float32).tofile(fid)
    def read_2(self, filename, tmark, t1, t2):
        """
        Read sac data given filename and time window.
        tmakr: 'b', 'e', 'o', 'a', 't0', 't1', ... 't9'.
        """
        self.read(filename)
        if tmark in self:
            t_ref = self[tmark]
            if t_ref == -12345.0:
                print('unset header for tmark: %s %f', tmark, t_ref)
                sys.exit(0)
            i1 = max(int( np.round(t_ref - self['b'] + t1) / (self['delta'])  ), 0  )
            i2 = min(int( np.round(t_ref - self['b'] + t2) / (self['delta'])  ), self.dat.size  ) 
            # update data and header info
            self.dat= self.dat[i1:i2]
            self['npts'] = i2-i1
            self['b'] = t_ref + t1
            self['e'] = t_ref + t1 + (i2-i1) * self['delta']
        else:
            print('Unrecognized tmark for sactrace.read_sac_2(...) ', tmark)
            sys.exit(0)
        pass
    ### hdr methods
    def update_hdr(self, **kwargs):
        self.hdr.update(**kwargs)
    ### internel methods
    def __getitem__(self, key):
        """
        Accessing with specified key.
        """
        return self.hdr[key]
    def __setitem__(self, key, value):
        """
        Set value with specified key.
        """
        self.hdr[key] = value
    def __contains__(self, key):
        """
        Check whether a key in included.
        """
        return key in self.hdr
    ### numerical methods
    def get_time_axis(self):
        """
        Get time axis.
        """
        return np.arange(0.0, self['npts'], 1.0) * self['delta'] + self['b']
    def taper(self, ratio= 0.05):
        """
        Taper using tukey window.
        """
        w = signal.tukey(self['npts'], ratio)
        self.dat *= w
        return self
    def rmean(self):
        """
        Remove mean value
        """
        self.dat -= np.average(self.dat)
        return self
    def detrend(self):
        """
        Remove linear trend
        """
        self.dat = signal.detrend(self.dat)
        return self
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
        return self
    def lowpass(self, f, order= 2, npass= 1):
        """
        Lowpass
        """
        nyq = 0.5 / self['delta']
        fc = f/nyq
        b, a = signal.butter(order, [fc], btype='lowpass')
        methods = [None, signal.lfilter, signal.filtfilt]
        self.dat = methods[npass](b, a, self.dat)
        return self
    def highpass(self, f, order= 2, npass= 1):
        """
        High pass
        """
        nyq = 0.5 / self['delta']
        fc = f/nyq
        b, a = signal.butter(order, [fc], btype='highpass')
        methods = [None, signal.lfilter, signal.filtfilt]
        self.dat = methods[npass](b, a, self.dat)
        return self
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
if __name__ == "__main__":
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
    write_sac_2('junk.sac', dat, delta, b, **{'kstnm': 'syn', 'stlo': 0.0, 'stla': 0.0} )
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
    s_new = make_sactrace_hdr(s.dat, new_hdr)
    s_new.write('1_new.sac')
    ###
    #
    ###
    # read, process, and write a bunch of sac files in 3 lines
    fnm_lst = ['1.sac', '2.sac', '3.sac']
    vol = [rd_sac(fnm).detrend().taper().lowpass(1.2, order= 2, npass= 2) for fnm in fnm_lst]
    [ s.write(s['filename'].replace('.sac', '_proced.sac') ) for s in vol ]
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