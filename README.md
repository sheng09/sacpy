SACPY
=====

- sheng.wang(at)anu.edu.au

# 1. How to install

1. Clone [sacpy]() into your directory.

```
cd ~/<your_directory> 
git clone https://github.com/sheng09/sacpy.git
```

2. Configure your [PYTHONPATH](https://docs.python.org/2/using/cmdline.html#envvar-PYTHONPATH). Add below line into your `~/.bash_profile`, or `~/.bash_rc`, etc. 

```
export PATH="<your_directory>:${PYTHONPATH}" # for sh, bash
setenv PATH "<your_directory>:${PYTHONPATH}" # for csh
```

3. Keep updating

Update [sacpy]() to lastest version. Please keep updating often for less bugs, and new features, functions, etc.

```
cd ~/<your_directory> 
git pull origin master
```


# 2. Examples

```
File IO dependent methods
-------------------------

>>> from sacpy.sac import *
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
```
